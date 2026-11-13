"""Which signal controllers exist on the district network, and whether each one may be offered.

Status comes from files, never from code: an ML controller is ``verified`` only if the publication gate in
``artifacts/ml/evaluation.json`` says so (``ml.evaluate.gate`` on held-out test seeds against ``queue_responsive``). Anything else is
``experimental``: it can be shown when someone asks for it by name, but it is never the default, and a controller whose model files are
missing or do not match their manifest is ``unavailable`` and falls back to the queue-responsive rules.

This module imports nothing heavy, so the API can serve it without the ML requirements installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONTROLLER = "queue_responsive"
SPECS: list[dict[str, Any]] = [
    {"id": "fixed_time", "title": "Fixed-time", "kind": "baseline", "detail": "The pre-timed signal plans; ignores traffic."},
    {"id": "queue_responsive", "title": "FlowState rules", "kind": "rule",
     "detail": "Queue-responsive green durations on the fixed plans (gap-out, switch on demand, extend) plus ambulance green priority. Hand-written rules."},
    {"id": "ml_forecast", "title": "Forecast-assisted rules", "kind": "ml",
     "detail": "The queue rule with a trained 30 s queue forecast (bounded demand shift, predicted spillback)."},
    {"id": "rl_ppo", "title": "Learned policy (PPO)", "kind": "ml",
     "detail": "One shared network picks hold / end / extend for every signal every 5 s, restricted to safe timings."},
]
_FILES = {"ml_forecast": ("forecaster", "MODEL_MANIFEST.json", "forecaster.pkl", ("sklearn",)), "rl_ppo": ("rl", "RL_MANIFEST.json", "policy.npz", ())}


def model_dir() -> Path:
    return Path(os.environ.get("FLOWSTATE_ML_DIR", str(Path(__file__).resolve().parents[3] / "artifacts" / "ml")))


def _artifact(cid: str, directory: Path) -> tuple[bool, str | None, dict[str, Any]]:
    sub, man_name, file_name, needs = _FILES[cid]
    man_path, f = directory / sub / man_name, directory / sub / file_name
    if not man_path.exists() or not f.exists():
        return False, "no trained model on this installation", {}
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except ValueError:
        return False, "unreadable model manifest", {}
    if hashlib.sha256(f.read_bytes()).hexdigest() != man.get("sha256"):
        return False, "model file does not match its manifest", man
    missing = [m for m in needs if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"needs {', '.join(missing)} (backend/requirements-ml.txt)", man
    return True, None, man


def controllers(directory: Path | None = None) -> dict[str, Any]:
    d = directory or model_dir()
    ev: dict[str, Any] = {}
    try:
        ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    gates = ev.get("gate", {})
    overall = ev.get("vs_queue_responsive", {}).get("overall", {})
    out = []
    for spec in SPECS:
        c: dict[str, Any] = {**spec, "available": True, "note": None, "gate": None}
        cid = spec["id"]
        if spec["kind"] == "baseline":
            c["status"] = "baseline"
        elif spec["kind"] == "rule":
            c["status"] = "default"
        else:
            ok, why, man = _artifact(cid, d)
            g = gates.get(cid)
            c["available"], c["note"], c["gate"] = ok, why, g
            c["status"] = "unavailable" if not ok else ("verified" if g and g.get("status") == "verified" else "experimental")
            c["networkId"] = man.get("network_id")
            if ok and not g:
                c["note"] = "not evaluated against the queue-responsive rules"
            elif ok and g.get("status") != "verified":
                c["note"] = "did not pass the publication gate (" + ", ".join(g.get("failed", [])) + "); shown for reference, never selected by default"
        m = overall.get(cid, {}).get("metrics")
        if m:
            c["heldOut"] = {k: m[k]["mean"] for k in ("mean_wait_s", "mean_time_loss_s", "throughput_vph", "teleports") if k in m}
        out.append(c)
    meta = ev.get("meta", {})
    return {"network": "district", "default_controller": DEFAULT_CONTROLLER, "controllers": out,
            "evaluation": {"split": meta.get("split"), "seeds": meta.get("seeds"), "scenarios": meta.get("scenarios"), "heldout_scenario": meta.get("heldout_scenario"),
                           "pairs": ev.get("vs_queue_responsive", {}).get("pairs"), "built_at": meta.get("built_at"), "baseline": "queue_responsive",
                           "note": meta.get("note")} if ev else None}


def resolve(requested: str | None, directory: Path | None = None) -> dict[str, Any]:
    """The controller to actually run for a request: an unavailable one falls back to the default, and the reason is reported."""
    info = {c["id"]: c for c in controllers(directory)["controllers"]}
    if not requested:
        return {"requested": None, "used": DEFAULT_CONTROLLER, "fallback": None, "status": info[DEFAULT_CONTROLLER]["status"]}
    c = info.get(requested)
    if c is None:
        raise ValueError(f"unknown controller '{requested}' (known: {', '.join(info)})")
    if not c["available"]:
        return {"requested": requested, "used": DEFAULT_CONTROLLER, "fallback": c["note"], "status": info[DEFAULT_CONTROLLER]["status"]}
    return {"requested": requested, "used": requested, "fallback": None, "status": c["status"]}
