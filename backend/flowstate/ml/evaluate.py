"""Paired, held-out comparison of controller arms, and the publication gate.

Every (scenario, seed) is run once per arm with identical demand and events (``episodes.run_episode``), so differences are
paired. ``summarise`` is the RL branch's ``rl/evaluate.py:summarise`` (mean, spread, paired bootstrap 95 % CI and Wilcoxon
signed-rank against a baseline), generalised to any arm names.

Publication gate (``gate``): an ML arm is ``verified`` only if, against ``queue_responsive`` on the held-out test seeds,

* the 95 % CI of its paired mean-wait difference is entirely below 0, and
* its time loss is not worse (paired mean difference <= 0), and
* it has no teleport regression (paired mean difference <= 0) and no throughput regression (the CI of the paired
  throughput difference is not entirely below 0, and the mean drop is under 1 %).

Otherwise it is ``experimental`` and must not be selected by default.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import HELDOUT_SCENARIO, TRAIN_SCENARIOS, split_seeds
from .episodes import model_dir, run_episode

METRICS = ["mean_wait_s", "mean_time_loss_s", "throughput_vph", "completed_fraction", "queue_mean_veh", "queue_max_veh", "pending_end", "teleports", "collisions"]
LOWER_IS_BETTER = {m: m not in ("throughput_vph", "completed_fraction") for m in METRICS}
GATE_BASELINE = "queue_responsive"
DEFAULT_ARMS = ("fixed_time", "queue_responsive", "safe_heuristic", "ml_forecast")
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


def evaluate(arms: list[str], scenarios: list[str], seeds: list[int], *, workers: int = 4, directory: Path | None = None) -> list[dict[str, Any]]:
    jobs = [(a, sc, s, str(directory) if directory else None) for sc in scenarios for s in seeds for a in arms]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            rows.append(r)
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
    rng = np.random.default_rng(rng_seed)
    out: dict[str, Any] = {"baseline": baseline, "arms": sorted(by), "scenarios": scenarios, "episodes": len(ok), "pairs": len(keys),
                           "failed": [{k: r[k] for k in ("arm", "scenario", "seed", "error")} for r in rows if "error" in r]}

    def block(sel: list[tuple[str, int]]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for arm, runs in by.items():
            res[arm] = {"n": len(sel), "metrics": {}}
            for m in METRICS:
                v = np.array([runs[k][m] for k in sel], dtype=float)
                res[arm]["metrics"][m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}
                if arm != baseline and baseline in by:
                    b = np.array([by[baseline][k][m] for k in sel], dtype=float)
                    d = v - b
                    boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(axis=1) if len(d) > 1 else np.array([d.mean()])
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


def gate(summary_vs_heuristic: dict[str, Any], arm: str) -> dict[str, Any]:
    """Apply the publication gate to ``arm`` in a summary whose baseline is ``queue_responsive``."""
    if summary_vs_heuristic.get("baseline") != GATE_BASELINE:
        raise ValueError(f"the gate needs a summary against {GATE_BASELINE}")
    m = summary_vs_heuristic.get("overall", {}).get(arm, {}).get("metrics")
    if not m or "vs_baseline" not in m["mean_wait_s"]:
        return {"arm": arm, "status": "experimental", "reason": "not evaluated against the heuristic", "checks": {}}
    wait, loss, tele, thr = (m[k]["vs_baseline"] for k in ("mean_wait_s", "mean_time_loss_s", "teleports", "throughput_vph"))
    checks = {
        "wait_ci_below_zero": wait["ci95"][1] < 0,
        "time_loss_not_worse": loss["mean_diff"] <= 0,
        "no_teleport_regression": tele["mean_diff"] <= 0,
        "no_throughput_regression": not (thr["ci95"][1] < 0) and (thr["pct_change"] is None or thr["pct_change"] > -1.0),
    }
    failed = [k for k, v in checks.items() if not v]
    return {"arm": arm, "status": "verified" if not failed else "experimental", "baseline": GATE_BASELINE, "pairs": wait["paired"], "checks": checks,
            "failed": failed, "mean_wait_diff_s": wait["mean_diff"], "mean_wait_ci95": wait["ci95"], "mean_wait_pct": wait["pct_change"],
            "time_loss_diff_s": loss["mean_diff"], "time_loss_ci95": loss["ci95"], "teleports_diff": tele["mean_diff"], "throughput_pct": thr["pct_change"]}


def run(arms: list[str] | None = None, n_seeds: int | None = None, *, split: str = "test", scenarios: list[str] | None = None, workers: int = 4,
        out: Path | None = None, directory: Path | None = None) -> dict[str, Any]:
    """Evaluate ``arms`` on the split's seeds (default: the held-out test seeds, all six scenarios), summarise, gate, save."""
    arms = list(arms or DEFAULT_ARMS)
    for needed in ("fixed_time", GATE_BASELINE):
        if needed not in arms:
            arms.insert(0, needed)
    seeds = split_seeds(split, n_seeds)
    scenarios = list(scenarios or EVAL_SCENARIOS)
    t0 = time.time()
    rows = evaluate(arms, scenarios, seeds, workers=workers, directory=directory)
    vs_fixed = summarise(rows, "fixed_time")
    vs_heur = summarise(rows, GATE_BASELINE)
    gates = {a: gate(vs_heur, a) for a in arms if a in ("ml_forecast", "rl_ppo", "safe_heuristic")}
    doc = {"meta": {"split": split, "seeds": [seeds[0], seeds[-1], len(seeds)], "scenarios": scenarios, "arms": arms, "heldout_scenario": HELDOUT_SCENARIO,
                    "wall_s": round(time.time() - t0, 1), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "note": "simulation only: demand is a seeded model, not calibrated to Toronto counts (docs/TORONTO_DEMAND_STRATEGY.md)"},
           "gate": gates, "vs_fixed_time": vs_fixed, "vs_queue_responsive": vs_heur,
           "episodes": rows}
    out = out or (directory or model_dir()) / ("evaluation.json" if split == "test" else f"evaluation_{split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def table(doc: dict[str, Any]) -> str:
    """Plain-text summary of an evaluation document."""
    vh = doc["vs_queue_responsive"]
    lines = [f"split={doc['meta']['split']} seeds={doc['meta']['seeds']} pairs={vh.get('pairs')} failed={len(vh.get('failed', []))}"]
    if "overall" not in vh:
        return lines[0] + "\n(no complete pairs)"
    lines.append(f"{'arm':18} {'wait s':>8} {'loss s':>8} {'thr vph':>8} {'tele':>6} | wait vs heuristic (95% CI)      p      | loss vs heuristic (95% CI)")
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
