"""Paired, held-out comparison of controller arms, and the publication gate.

Every (scenario, seed) is run once per arm with identical demand and events (``episodes.run_episode``), so differences are
paired. ``summarise`` is the RL branch's ``rl/evaluate.py:summarise`` (mean, spread, paired bootstrap 95 % CI and Wilcoxon
signed-rank against a baseline), generalised to any arm names.

Publication gate (``gate``): the trained agent is ``verified`` only if, against ``fixed_time`` on the held-out test seeds,

* it is better on waiting: the 95 % CI of its paired mean-wait difference is entirely below 0 (superiority), and
* it is not worse on anything else, by a non-inferiority test: the 95 % CI of the paired difference must stay inside a margin
  fixed here, before the evaluation is run (``MARGINS``):

  - time loss: upper bound at most +2 % of the baseline's mean;
  - throughput: lower bound at least -1 % of the baseline's mean;
  - teleports: upper bound at most +0.05 per episode;
  - vehicles never inserted: upper bound at most +2 % of the baseline's mean (or +1 vehicle per episode if that is larger), and
    insertion delay: upper bound at most +2 % of the baseline's mean (or +0.5 s);

* in no single scenario (the held-out one included) is it clearly and materially worse: a scenario fails if the 95 % CI of the paired
  wait or time-loss difference lies entirely above 0 *and* the mean difference exceeds 2 % of the baseline's mean in that scenario.
  Pooling must not hide a regression on one scenario.

A margin is needed because a controller that exactly ties the baseline has a mean difference above 0 half of the time; "mean
difference <= 0" would fail it at random. The margins are deliberately small.

Otherwise it is ``experimental`` and the Results page's data file is not built from it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from .episodes import HELDOUT_SCENARIO, TRAIN_SCENARIOS, model_dir, run_episode, split_seeds

METRICS = ["mean_wait_s", "mean_time_loss_s", "mean_wait_window_s", "mean_depart_delay_s", "unserved", "throughput_vph", "completed_fraction", "queue_mean_veh",
           "queue_max_veh", "pending_end", "teleports", "collisions"]
GATE_METRICS = ("mean_wait_s", "mean_time_loss_s", "teleports", "throughput_vph", "unserved", "mean_depart_delay_s")
# Non-inferiority margins of the gate: metric -> (% of the baseline mean, absolute floor). Fixed before the evaluation they are applied to.
MARGINS = {"mean_time_loss_s": (2.0, 0.0), "mean_wait_s": (2.0, 0.0), "throughput_vph": (1.0, 0.0), "teleports": (0.0, 0.05), "unserved": (2.0, 1.0),
           "mean_depart_delay_s": (2.0, 0.5)}
LOWER_IS_BETTER = {m: m not in ("throughput_vph", "completed_fraction") for m in METRICS}
GATE_BASELINE = "fixed_time"
AGENT = "rl_ppo"
DEFAULT_ARMS = ("fixed_time", AGENT)
EVAL_SCENARIOS = TRAIN_SCENARIOS + (HELDOUT_SCENARIO,)


def _worker(job: tuple[str, str, int, str | None]) -> dict[str, Any]:
    arm, scenario, seed, directory = job
    t0 = time.time()
    try:
        r = run_episode(arm, scenario, seed, directory=Path(directory) if directory else None)
    except Exception as exc:  # noqa: BLE001 - one failed episode must be reported, not hidden and not fatal to the batch
        return {"arm": arm, "scenario": scenario, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
    r["wall_s"] = round(time.time() - t0, 2)
    return r


def evaluate(arms: list[str], scenarios: list[str], seeds: list[int], *, workers: int = 4, directory: Path | None = None,
             partial: Path | None = None, stamp: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run every (arm, scenario, seed). With ``partial`` each finished episode is appended to that file, and episodes already in it
    are not run again if its first line equals ``stamp`` (same code commit and model files): an interrupted evaluation resumes."""
    rows: list[dict[str, Any]] = []
    if partial is not None:
        lines = partial.read_text(encoding="utf-8").splitlines() if partial.exists() else []
        if lines and json.loads(lines[0]) == stamp:
            rows = [r for r in map(json.loads, lines[1:]) if "error" not in r]
        else:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(stamp) + "\n", encoding="utf-8")
    have = {(r["arm"], r["scenario"], r["seed"]) for r in rows}
    rows = [r for r in rows if r["arm"] in arms and r["scenario"] in scenarios and r["seed"] in seeds]
    jobs = [(a, sc, s, str(directory) if directory else None) for sc in scenarios for s in seeds for a in arms if (a, sc, s) not in have]
    if have:
        print(f"[evaluate] resuming: {len(rows)} episodes kept, {len(jobs)} to run", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=12) as pool:  # recycle workers: long-lived SUMO processes grow
        for k, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            rows.append(r)
            if partial is not None:
                with partial.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            if k % 50 == 0 or k == len(jobs):
                print(f"[evaluate] {k}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["arm"]))
    return rows


def summarise(rows: list[dict[str, Any]], baseline: str = "fixed_time", rng_seed: int = 0) -> dict[str, Any]:
    """Per arm: mean / std / min / max of every metric, and paired differences against ``baseline`` (bootstrap CI, Wilcoxon).

    Only (scenario, seed) pairs that every arm completed are used, so all arms are scored on exactly the same episodes."""
    from scipy import stats

    ok = [r for r in rows if "error" not in r]
    by: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for r in ok:
        by.setdefault(r["arm"], {})[(r["scenario"], r["seed"])] = r
    keys = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    scenarios = sorted({k[0] for k in keys})
    out: dict[str, Any] = {"baseline": baseline, "arms": sorted(by), "scenarios": scenarios, "episodes": len(ok), "pairs": len(keys),
                           "failed": [{k: r[k] for k in ("arm", "scenario", "seed", "error")} for r in rows if "error" in r]}

    present = [m for m in METRICS if all(m in r for r in ok)]  # documents written before a measurement existed lack it

    def block(sel: list[tuple[str, int]]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        # one set of resamples per block, shared by every arm and metric: an interval does not depend on which other arms were evaluated
        tag = sum(map(ord, "".join(sorted({k[0] for k in sel}))))
        idx = np.random.default_rng([rng_seed, len(sel), tag]).integers(0, max(1, len(sel)), size=(4000, max(1, len(sel))))
        for arm, runs in by.items():
            res[arm] = {"n": len(sel), "metrics": {}}
            for m in present:
                v = np.array([runs[k][m] for k in sel], dtype=float)
                res[arm]["metrics"][m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}
                if arm != baseline and baseline in by:
                    b = np.array([by[baseline][k][m] for k in sel], dtype=float)
                    d = v - b
                    boot = d[idx].mean(axis=1) if len(d) > 1 else np.array([d.mean()])
                    try:
                        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 5 else None
                    except ValueError:
                        p = None
                    lower = LOWER_IS_BETTER[m]
                    res[arm]["metrics"][m]["vs_baseline"] = {
                        "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "pct_change": float(100.0 * d.mean() / b.mean()) if b.mean() else None, "wilcoxon_p": p,
                        "better_in": int(np.sum(d < 0) if lower else np.sum(d > 0)), "worse_in": int(np.sum(d > 0) if lower else np.sum(d < 0)), "paired": len(d)}
        return res

    if keys:
        out["overall"] = block(keys)
        out["per_scenario"] = {sc: block([k for k in keys if k[0] == sc]) for sc in scenarios}
    return out


def gate(summary: dict[str, Any], arm: str = AGENT) -> dict[str, Any]:
    """Apply the publication gate to ``arm`` in a summary whose baseline is ``fixed_time``."""
    if summary.get("baseline") != GATE_BASELINE:
        raise ValueError(f"the gate needs a summary against {GATE_BASELINE}")
    overall = summary.get("overall", {})
    m = overall.get(arm, {}).get("metrics")
    if not m or "vs_baseline" not in m["mean_wait_s"]:
        return {"arm": arm, "status": "experimental", "reason": "not evaluated against fixed-time", "checks": {}}
    missing = [k for k in GATE_METRICS if k not in m]
    if missing:
        return {"arm": arm, "status": "experimental", "reason": "the evaluation predates these measurements: " + ", ".join(missing), "checks": {}, "failed": ["evaluation_outdated"]}
    base = overall[GATE_BASELINE]["metrics"]

    def limit(k: str, ref: dict[str, Any] = base) -> float:
        pct, floor = MARGINS[k]
        return max(floor, pct / 100.0 * abs(ref[k]["mean"]))

    vb = {k: m[k]["vs_baseline"] for k in GATE_METRICS}
    wait, loss, tele, thr = vb["mean_wait_s"], vb["mean_time_loss_s"], vb["teleports"], vb["throughput_vph"]
    checks = {
        "wait_ci_below_zero": wait["ci95"][1] < 0,
        "time_loss_not_worse": loss["ci95"][1] <= limit("mean_time_loss_s"),
        "no_teleport_regression": tele["ci95"][1] <= limit("teleports"),
        "no_throughput_regression": thr["ci95"][0] >= -limit("throughput_vph"),
        "no_insertion_regression": vb["unserved"]["ci95"][1] <= limit("unserved") and vb["mean_depart_delay_s"]["ci95"][1] <= limit("mean_depart_delay_s"),
    }
    regressed = []
    for sc, blk in summary.get("per_scenario", {}).items():
        sm = blk.get(arm, {}).get("metrics", {})
        for k in ("mean_wait_s", "mean_time_loss_s"):
            d = sm.get(k, {}).get("vs_baseline")
            if d and d["paired"] > 1 and d["ci95"][0] > 0 and d["mean_diff"] > limit(k, blk[GATE_BASELINE]["metrics"]):
                regressed.append({"scenario": sc, "metric": k, "mean_diff": d["mean_diff"], "ci95": d["ci95"], "pct_change": d["pct_change"]})
    checks["no_scenario_regression"] = not regressed
    failed = [k for k, v in checks.items() if not v]
    return {"arm": arm, "scenario_regressions": regressed, "status": "verified" if not failed else "experimental", "baseline": GATE_BASELINE, "pairs": wait["paired"], "checks": checks,
            "failed": failed, "margins": {k: {"pct_of_baseline": MARGINS[k][0], "floor": MARGINS[k][1], "limit": limit(k)} for k in MARGINS},
            "mean_wait_diff_s": wait["mean_diff"], "mean_wait_ci95": wait["ci95"], "mean_wait_pct": wait["pct_change"],
            "time_loss_diff_s": loss["mean_diff"], "time_loss_ci95": loss["ci95"], "teleports_diff": tele["mean_diff"], "teleports_ci95": tele["ci95"],
            "throughput_pct": thr["pct_change"], "throughput_ci95": thr["ci95"], "unserved_diff": vb["unserved"]["mean_diff"], "unserved_ci95": vb["unserved"]["ci95"],
            "depart_delay_diff_s": vb["mean_depart_delay_s"]["mean_diff"], "depart_delay_ci95": vb["mean_depart_delay_s"]["ci95"]}


def run(arms: list[str] | None = None, n_seeds: int | None = None, *, split: str = "test", scenarios: list[str] | None = None, workers: int = 4,
        out: Path | None = None, directory: Path | None = None, reuse: Path | None = None) -> dict[str, Any]:
    """Evaluate ``arms`` on the split's seeds (default: the held-out test seeds, all six scenarios), summarise, gate, save.

    ``reuse`` names an earlier evaluation document of the same split: its episodes are kept for the (arm, scenario, seed) triples asked for
    here, and only the missing ones are run. Episodes are deterministic in (arm, scenario, seed, code, model), so only reuse a document
    produced by the same code and models."""
    arms = list(arms or DEFAULT_ARMS)
    if GATE_BASELINE not in arms:
        arms.insert(0, GATE_BASELINE)
    seeds = split_seeds(split, n_seeds)
    scenarios = list(scenarios or EVAL_SCENARIOS)
    t0 = time.time()
    kept: list[dict[str, Any]] = []
    if reuse is not None:
        old = json.loads(Path(reuse).read_text(encoding="utf-8"))
        if old["meta"]["split"] != split:
            raise ValueError("the reused evaluation is of a different split")
        prov = old["meta"].get("provenance")
        if prov is not None and prov.get("models") != provenance(directory)["models"]:
            raise ValueError("the reused evaluation was produced with different model files")
        want = {(a, sc, s) for a in arms for sc in scenarios for s in seeds}
        kept = [r for r in old["episodes"] if "error" not in r and (r["arm"], r["scenario"], r["seed"]) in want]
    have = {(r["arm"], r["scenario"], r["seed"]) for r in kept}
    todo = [a for a in arms if any((a, sc, s) not in have for sc in scenarios for s in seeds)]
    kept = [r for r in kept if r["arm"] not in todo]
    out = out or (directory or model_dir()) / ("evaluation.json" if split == "test" else f"evaluation_{split}.json")
    prov = provenance(directory)
    partial = out.with_suffix(".partial.jsonl")
    rows = kept + (evaluate(todo, scenarios, seeds, workers=workers, directory=directory, partial=partial, stamp={"split": split, **prov}) if todo else [])
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["arm"]))
    vs_fixed = summarise(rows, GATE_BASELINE)
    gates = {a: gate(vs_fixed, a) for a in arms if a != GATE_BASELINE}
    doc = {"meta": {"split": split, "seeds": [seeds[0], seeds[-1], len(seeds)], "scenarios": scenarios, "arms": arms, "heldout_scenario": HELDOUT_SCENARIO,
                    "wall_s": round(time.time() - t0, 1), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "provenance": prov,
                    "reused": {"from": str(reuse), "arms": sorted({r["arm"] for r in kept})} if reuse is not None else None,
                    "note": "simulation only: demand is a seeded model, not calibrated to Toronto counts (docs/TORONTO_DEMAND_STRATEGY.md)"},
           "gate": gates, "vs_fixed_time": vs_fixed,
           "episodes": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return doc


def provenance(directory: Path | None = None) -> dict[str, Any]:
    """Code commit and model hashes an evaluation was produced with (recorded in ``meta``; ``reuse`` refuses different models)."""
    import subprocess

    d = directory or model_dir()
    out: dict[str, Any] = {"models": {}}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        out["commit"] = None
    for name, man in ((AGENT, d / "rl" / "RL_MANIFEST.json"),):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            out["models"][name] = m.get("sha256") or m.get("artifact_sha256")
        except (OSError, ValueError):
            out["models"][name] = None
    return out


def regate(path: Path) -> dict[str, Any]:
    """Recompute summaries and gates of a stored evaluation from its episodes (no simulation), e.g. after the gate definition changed."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = doc["episodes"]
    doc["vs_fixed_time"] = summarise(rows, GATE_BASELINE)
    doc["gate"] = {a: gate(doc["vs_fixed_time"], a) for a in doc["meta"]["arms"] if a != GATE_BASELINE}
    doc["meta"]["regated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["meta"].setdefault("provenance", provenance())
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def table(doc: dict[str, Any]) -> str:
    """Plain-text summary of an evaluation document."""
    vh = doc["vs_fixed_time"]
    lines = [f"split={doc['meta']['split']} seeds={doc['meta']['seeds']} pairs={vh.get('pairs')} failed={len(vh.get('failed', []))}"]
    if "overall" not in vh:
        return lines[0] + "\n(no complete pairs)"
    lines.append(f"{'arm':18} {'wait s':>8} {'loss s':>8} {'thr vph':>8} {'tele':>6} | wait vs fixed-time (95% CI)     p      | loss vs fixed-time (95% CI)")
    for arm in doc["meta"]["arms"]:
        m = vh["overall"][arm]["metrics"]
        row = f"{arm:18} {m['mean_wait_s']['mean']:8.2f} {m['mean_time_loss_s']['mean']:8.2f} {m['throughput_vph']['mean']:8.0f} {m['teleports']['mean']:6.2f} | "
        vb, lb = m["mean_wait_s"].get("vs_baseline"), m["mean_time_loss_s"].get("vs_baseline")
        if vb:
            p = "n/a" if vb["wilcoxon_p"] is None else f"{vb['wilcoxon_p']:.3g}"
            row += f"{vb['mean_diff']:+7.2f} [{vb['ci95'][0]:+.2f}, {vb['ci95'][1]:+.2f}] {p:>8} | {lb['mean_diff']:+7.2f} [{lb['ci95'][0]:+.2f}, {lb['ci95'][1]:+.2f}]"
        else:
            row += "(baseline)"
        lines.append(row)
    for arm, g in doc["gate"].items():
        lines.append(f"gate {arm}: {g['status']}" + (f" (failed: {', '.join(g['failed'])})" if g.get("failed") else ""))
    return "\n".join(lines)
