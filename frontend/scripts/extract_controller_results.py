"""Build frontend/src/data/controllerResults.json from a local evaluation file (``backend/flowstate/ml/evaluate.py`` output).

Usage (repository root):
    python frontend/scripts/extract_controller_results.py [path-to-evaluation.json]     # default artifacts/ml/evaluation.json
Numbers are copied, never recomputed. Nothing is imported from the backend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = [
    "mean_wait_s", "mean_time_loss_s", "throughput_vph", "completed_fraction", "queue_mean_veh",
    "queue_max_veh", "pending_end", "teleports", "collisions",
]

# Titles follow backend/flowstate/scenarios/catalog.py; descriptions are the catalog taglines.
SCENARIO_TABLE = {
    "rush_hour": ("Rush Hour", "Directional commuter peak."),
    "accident": ("Road Accident", "Collision blocks lanes, responders attend."),
    "construction": ("Construction", "Lane or full closure with work zone."),
    "weather": ("Severe Weather", "Rain, snow or fog."),
    "transit": ("Transit Priority", "Scheduled buses with signal priority."),
    "stadium": ("Stadium Event", "Arrival or departure surge. Never used for training or tuning (held out)."),
    "ambulance": ("Ambulance Dispatch", "Calls from Mount Sinai Hospital."),
    "school": ("School Dismissal", "Pick-up queues and a school zone."),
    "disruption": ("Random Urban Disruption", "Seeded mix of incidents and surges."),
}

CONTROLLER_TABLE = {
    "fixed_time": {"title": "Fixed-time", "detail": "The pre-timed signal plans; ignores traffic.", "status": "baseline"},
    "queue_responsive": {"title": "FlowState rules", "detail": "Queue-responsive green durations on the fixed plans (gap-out, switch on demand, extend) plus ambulance green priority. Hand-written rules, no model.", "status": "default"},
    "safe_heuristic": {"title": "Safe rule reference", "detail": "The same queue rule evaluated per approach under the ML safety contract. No trained model; included to separate the safety contract from the model.", "status": "reference"},
    "ml_forecast": {"title": "Forecast-assisted rules", "detail": "The queue rule with a trained 30 s queue forecast (bounded demand shift, predicted spillback).", "status": "experimental"},
    "rl_ppo": {"title": "Learned policy (PPO)", "detail": "One shared network picks hold / end / extend for every signal every 5 s, restricted to safe timings.", "status": "experimental"},
}
ORDER = list(CONTROLLER_TABLE)


def slim_stats(block: dict) -> dict:
    out = {}
    for k in METRICS:
        m = block["metrics"].get(k)
        if m is None:
            continue
        row = {"mean": m["mean"], "std": m["std"], "min": m["min"], "max": m["max"]}
        vs = m.get("vs_baseline")
        if vs:
            row["vs"] = {"diff": vs["mean_diff"], "ci95": vs["ci95"], "pct": vs["pct_change"], "p": vs.get("wilcoxon_p"),
                         "better": vs["better_in"], "worse": vs["worse_in"], "paired": vs["paired"]}
        out[k] = row
    return {"n": block["n"], "metrics": out}


def slim_group(group: dict) -> dict:
    return {c: slim_stats(v) for c, v in group.items() if c in CONTROLLER_TABLE}


def slim_summary(s: dict) -> dict:
    return {"overall": slim_group(s["overall"]), "perScenario": {sc: slim_group(g) for sc, g in s["per_scenario"].items()}}


def main(src: Path) -> None:
    d = json.loads(src.read_text(encoding="utf-8"))
    meta = d["meta"]
    present = [c for c in ORDER if c in meta["arms"]]
    controllers = {}
    for c in present:
        entry = dict(CONTROLLER_TABLE[c])
        g = d.get("gate", {}).get(c)
        if g:
            if c in ("ml_forecast", "rl_ppo"):
                entry["status"] = g["status"]  # the gate decides for trained models, not the table
            entry["gate"] = {"status": g["status"], "failed": g.get("failed", []), "checks": g.get("checks", {}), "pairs": g.get("pairs"),
                             "meanWaitDiffS": g.get("mean_wait_diff_s"), "meanWaitCi95": g.get("mean_wait_ci95"),
                             "timeLossDiffS": g.get("time_loss_diff_s"), "timeLossCi95": g.get("time_loss_ci95"),
                             "teleportsDiff": g.get("teleports_diff"), "throughputPct": g.get("throughput_pct")}
        controllers[c] = entry
    heldout = meta.get("heldout_scenario")
    scenarios = [{"id": s, "title": SCENARIO_TABLE.get(s, (s, ""))[0], "description": SCENARIO_TABLE.get(s, (s, ""))[1], "heldOut": s == heldout} for s in meta["scenarios"]]
    fixed, rules = d["vs_fixed_time"], d["vs_queue_responsive"]
    out = {
        "schema": "flowstate.controller-results/2",
        "source": {"file": src.name, "builtAt": meta.get("built_at"), "split": meta.get("split"), "seeds": meta["seeds"], "note": meta.get("note")},
        "network": {"id": "district", "description": "Toronto King West district, 22 signals"},
        "baseline": "fixed_time",
        "flowstate": "queue_responsive",
        "controllers": controllers,
        "safety": {"minGreen": "8-12", "maxGreen": 60, "decision": 5, "note": "amber, all-red, pedestrian clearance and protected-left phases always run their programmed length"},
        "scenarios": scenarios,
        "results": {
            "seeds": meta["seeds"],
            "episodes": fixed["episodes"],
            "pairs": fixed["pairs"],
            "vsFixed": slim_summary(fixed),
            "vsRules": slim_summary(rules),
            "runs": [
                {"c": e["arm"], "s": e["scenario"], "seed": e["seed"], "wait": round(e["mean_wait_s"], 2), "loss": round(e["mean_time_loss_s"], 2),
                 "queueMax": e["queue_max_veh"], "teleports": e["teleports"]}
                for e in d["episodes"] if e["arm"] in controllers
            ],
        },
    }
    dest = Path(__file__).resolve().parents[1] / "src" / "data" / "controllerResults.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB) from {src}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "artifacts" / "ml" / "evaluation.json"))
