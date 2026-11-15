"""The two signal controllers of the district network: the pre-timed plans, and the trained agent.

The agent's status comes from files, never from code: it is ``verified`` only if the publication gate in
``artifacts/ml/evaluation.json`` says so (``ml.evaluate.gate`` on held-out test seeds against fixed-time), ``experimental`` if it has
no gate result or failed it, and ``unavailable`` if its model file is missing or does not match its manifest. An unavailable agent is
never replaced by anything else: asking for it is an error (``ControllerUnavailable``).

This module imports nothing heavy, so the API can serve it without the ML requirements installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

BASELINE = "fixed_time"
AGENT = "rl_ppo"
DEFAULT_CONTROLLER = AGENT  # what a replay compares fixed-time with when nothing is asked for
SPECS: list[dict[str, Any]] = [
    {"id": BASELINE, "title": "Fixed-time", "kind": "baseline", "detail": "The pre-timed signal plans; ignores traffic."},
    {"id": AGENT, "title": "FlowState ML agent", "kind": "ml",
     "detail": "One shared network picks hold / end / extend for every signal every 5 s, restricted to safe timings."},
]
_FILES = ("rl", "RL_MANIFEST.json", "policy.npz")


class ControllerUnavailable(RuntimeError):
    """The trained agent cannot be run on this installation (the message says why)."""


def model_dir() -> Path:
    return Path(os.environ.get("FLOWSTATE_ML_DIR", str(Path(__file__).resolve().parents[3] / "artifacts" / "ml")))


def _artifact(directory: Path) -> tuple[bool, str | None, dict[str, Any]]:
    sub, man_name, file_name = _FILES
    man_path, f = directory / sub / man_name, directory / sub / file_name
    if not man_path.exists() or not f.exists():
        return False, "no trained model on this installation", {}
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except ValueError:
        return False, "unreadable model manifest", {}
    if hashlib.sha256(f.read_bytes()).hexdigest() != man.get("sha256"):
        return False, "model file does not match its manifest", man
    return True, None, man


def controllers(directory: Path | None = None) -> dict[str, Any]:
    d = directory or model_dir()
    ev: dict[str, Any] = {}
    try:
        ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    summary = ev.get("vs_fixed_time", {})
    overall = summary.get("overall", {})
    out = []
    for spec in SPECS:
        c: dict[str, Any] = {**spec, "available": True, "note": None, "gate": None}
        if spec["kind"] == "baseline":
            c["status"] = "baseline"
        else:
            ok, why, man = _artifact(d)
            g = ev.get("gate", {}).get(spec["id"])
            c["available"], c["note"], c["gate"] = ok, why, g
            c["status"] = "unavailable" if not ok else ("verified" if g and g.get("status") == "verified" else "experimental")
            c["networkId"] = man.get("network_id")
            if ok and not g:
                c["note"] = "not evaluated against fixed-time"
            elif ok and g.get("status") != "verified":
                c["note"] = "did not pass the publication gate (" + ", ".join(g.get("failed", [])) + ")"
        m = overall.get(spec["id"], {}).get("metrics")
        if m:
            c["heldOut"] = {k: m[k]["mean"] for k in ("mean_wait_s", "mean_time_loss_s", "throughput_vph", "teleports") if k in m}
        out.append(c)
    meta = ev.get("meta", {})
    return {"network": "district", "default_controller": DEFAULT_CONTROLLER, "controllers": out,
            "evaluation": {"split": meta.get("split"), "seeds": meta.get("seeds"), "scenarios": meta.get("scenarios"), "heldout_scenario": meta.get("heldout_scenario"),
                           "pairs": summary.get("pairs"), "built_at": meta.get("built_at"), "baseline": BASELINE, "note": meta.get("note")} if ev else None}


def resolve(requested: str | None, directory: Path | None = None) -> dict[str, Any]:
    """The controller to run for a request (default: the agent). Unknown ids raise ``ValueError``; an agent whose model is missing or
    does not match its manifest raises ``ControllerUnavailable``. Nothing is ever substituted for the controller asked for."""
    info = {c["id"]: c for c in controllers(directory)["controllers"]}
    cid = requested or DEFAULT_CONTROLLER
    c = info.get(cid)
    if c is None:
        raise ValueError(f"unknown controller '{requested}' (known: {', '.join(info)})")
    if not c["available"]:
        raise ControllerUnavailable(f"{c['title']} cannot run: {c['note']}")
    return {"requested": requested, "used": cid, "status": c["status"]}
