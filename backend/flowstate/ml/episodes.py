"""One seeded episode of a main scenario on the real network under a named controller arm, with the measurements every
comparison uses. The same code runs dataset generation and evaluation, so arms differ in nothing but their signal policies.

Arms (same network, same scenario, same seed, therefore the same demand and events):

* ``fixed_time``        main's pre-timed plans, no extra rule;
* ``queue_responsive``  the shipped "FlowState" rules exactly as ``replay/record.py`` runs them:
                        ``QueueResponsive`` + ``EmergencyPriority``;
* ``safe_heuristic``    the same queue rule evaluated per approach under the ML safety contract (no model);
* ``ml_forecast``       ``safe_heuristic`` + the trained queue forecast;
* ``rl_ppo``            the trained PPO policy (Phase 2);
* ``random_timing``     random safe green lengths (dataset generation only).

Measurement: waiting time and time loss are averaged over *every* motor vehicle that departed (completed trips and vehicles
still on the road at the end), so a controller is not flattered by trips it leaves unfinished. ``mean_wait_s`` is SUMO's
accumulated waiting time, which SUMO keeps over a sliding 100 s window; ``mean_time_loss_s`` has no such cap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("fixed_time", "queue_responsive", "safe_heuristic", "ml_forecast", "rl_ppo", "random_timing")
MOTOR = ("car", "truck", "bus")
_FORECASTERS: dict[str, Any] = {}


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_dir() -> Path:
    return Path(os.environ.get("FLOWSTATE_ML_DIR", str(root() / "artifacts" / "ml")))


def load_forecaster(net_path: Path, directory: Path | None = None) -> Any:
    """The registered forecaster for this network (cached per process). Raises if it is missing or does not match."""
    from .forecast import Forecaster
    from .topology import build_topology

    d = (directory or model_dir()) / "forecaster"
    key = f"{d}|{net_path}"
    if key not in _FORECASTERS:
        _FORECASTERS[key] = Forecaster.load(d, network_id=build_topology(net_path).network_id)
    return _FORECASTERS[key]


def make_policies(arm: str, seed: int, net_path: Path, directory: Path | None = None) -> list[Any]:
    from ..replay.heuristic import QueueResponsive
    from ..replay.priority import EmergencyPriority
    from .policies import ForecastPolicy, RandomTiming

    if arm == "fixed_time":
        return []
    if arm == "queue_responsive":
        return [QueueResponsive(), EmergencyPriority()]
    if arm == "safe_heuristic":
        return [EmergencyPriority(), ForecastPolicy()]
    if arm == "ml_forecast":
        return [EmergencyPriority(), ForecastPolicy(load_forecaster(net_path, directory))]
    if arm == "random_timing":
        return [RandomTiming(seed)]
    if arm == "rl_ppo":
        from .policies import LearnedPolicy

        return [EmergencyPriority(), LearnedPolicy.load((directory or model_dir()) / "rl", net_path)]
    raise ValueError(f"unknown arm {arm!r} (known: {', '.join(ARMS)})")


def run_episode(arm: str, scenario_id: str, seed: int, *, record: bool = False, params: dict[str, Any] | None = None,
                directory: Path | None = None, policies: list[Any] | None = None) -> dict[str, Any]:
    """Run one episode. With ``record`` the result carries ``X`` (samples x approaches x features) and ``t``.

    ``policies`` replaces the arm's signal policies (training rollouts pass their own policy object); ``arm`` is then only a label."""
    from ..api.service import NETWORKS_DIR, WORK_DIR
    from ..scenarios.catalog import create
    from ..sim.engine import Engine, RunConfig
    from .policies import RecorderPolicy

    sc = create(scenario_id, dict(params or {}))
    label = f"ml-{arm}-{scenario_id}-{seed}-{os.getpid()}"
    eng = Engine(sc, RunConfig(scenario_id, sc.params, seed, "fixed", label=label), NETWORKS_DIR, WORK_DIR / label)
    eng.start()
    try:
        if policies is None:
            policies = make_policies(arm, seed, eng.net_path, directory)
        rec = RecorderPolicy() if record else None
        eng.signals.policies.extend(policies + ([rec] if rec else []))
        while not eng.finished:
            eng.step()
        res = eng.results()
        m = eng.metrics
        wait = loss = 0.0
        n = done = 0
        for mode in MOTOR:
            s = m.modes.get(mode)
            if s:
                n += s.trips
                done += s.trips
                wait += s.waiting
                loss += s.loss
        for tr in m.trips.values():  # still on the road
            if tr.mode in MOTOR:
                n += 1
                wait += tr.waiting
                loss += tr.time_loss
        queued = [h["queued"] for h in m.history]
        duration = float(sc.duration or eng.t)
        out: dict[str, Any] = {
            "arm": arm, "scenario": scenario_id, "seed": seed, "network": eng.network_id, "duration_s": duration,
            "mean_wait_s": wait / n if n else 0.0, "mean_time_loss_s": loss / n if n else 0.0,
            "vehicles": n, "completed": done, "completed_fraction": done / n if n else 1.0, "throughput_vph": done * 3600.0 / duration,
            "queue_mean_veh": float(np.mean(queued)) if queued else 0.0, "queue_max_veh": float(max(queued)) if queued else 0.0,
            "pending_end": res["demandAccounting"]["pendingNow"], "teleports": res["teleports"], "collisions": res["collisions"],
            "tsp_extensions": res["signals"]["tspExtensions"],
            "rules": [p.summary() for p in policies if hasattr(p, "summary")],
        }
        if rec is not None:
            out["X"] = np.stack(rec.rows) if rec.rows else np.zeros((0, 0, 0), dtype=np.float32)
            out["t"] = np.asarray(rec.times)
            out["network_id"] = rec.topo.network_id if rec.topo else None
        return out
    finally:
        eng.close()
        _cleanup(WORK_DIR / label)


def _cleanup(work: Path) -> None:
    """Remove the per-episode scratch directory (thousands of episodes would otherwise leave one each)."""
    try:
        for f in work.iterdir():
            f.unlink()
        work.rmdir()
    except OSError:
        pass
