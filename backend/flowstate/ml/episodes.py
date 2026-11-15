"""One seeded episode of a main scenario on the real network under a named controller arm, with the measurements every
comparison uses. The same code runs training rollouts and evaluation, so arms differ in nothing but their signal policies.

Arms (same network, same scenario, same seed, therefore the same demand and events):

* ``fixed_time``  main's pre-timed plans, no extra rule;
* ``rl_ppo``      the trained PPO agent, with ambulance green priority beside it.

Leakage control: training, selection and test use *disjoint seed ranges* (``SEEDS``), and one scenario (``HELDOUT_SCENARIO``) is
never trained or selected on.

Measurement: waiting time and time loss are averaged over *every* motor vehicle that departed (completed trips and vehicles
still on the road at the end), so a controller is not flattered by trips it leaves unfinished. ``mean_wait_s`` is the whole time a
vehicle stood still (SUMO's waiting definition, every standstill spell summed); ``mean_wait_window_s`` is SUMO's own accumulated
waiting time, which only covers a sliding 100 s window and is kept for comparison with the app's figures. Vehicles that never got
onto the network are in no average, and the demand model itself generates fewer trips when the network is congested, so a
controller that jams the district would otherwise be scored on less traffic. Those trips are reported separately: ``unserved`` =
trips the demand model held back because of spillback (``throttled_trips``) + trips it gave up on at full gateways
(``refused_trips``) + vehicles asked for but not inserted by the end (``never_inserted``: still queued outside the network, or
dropped by SUMO after waiting 180 s); and ``mean_depart_delay_s`` (how long the inserted ones waited
to get in). The publication gate checks both.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("fixed_time", "rl_ppo")
MOTOR = ("car", "truck", "bus")
TRAIN_SCENARIOS = ("rush_hour", "construction", "accident", "weather", "transit")
HELDOUT_SCENARIO = "stadium"
SEEDS = {"train": (1000, 1199), "val": (2000, 2039), "test": (3000, 3079)}


def split_seeds(split: str, n: int | None = None) -> list[int]:
    """The split's seeds; ``n`` takes the first n (smoke runs), never crossing into another split's range."""
    lo, hi = SEEDS[split]
    seeds = list(range(lo, hi + 1))
    return seeds if n is None else seeds[: max(1, min(n, len(seeds)))]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_dir() -> Path:
    return Path(os.environ.get("FLOWSTATE_ML_DIR", str(root() / "artifacts" / "ml")))


def make_policies(arm: str, seed: int, net_path: Path, directory: Path | None = None) -> list[Any]:
    if arm == "fixed_time":
        return []
    if arm == "rl_ppo":
        from ..replay.priority import EmergencyPriority
        from .policies import LearnedPolicy

        return [EmergencyPriority(), LearnedPolicy.load((directory or model_dir()) / "rl", net_path)]
    raise ValueError(f"unknown arm {arm!r} (known: {', '.join(ARMS)})")


def run_episode(arm: str, scenario_id: str, seed: int, *, params: dict[str, Any] | None = None,
                directory: Path | None = None, policies: list[Any] | None = None) -> dict[str, Any]:
    """Run one episode.

    ``policies`` replaces the arm's signal policies (training rollouts pass their own policy object); ``arm`` is then only a label."""
    from ..api.service import NETWORKS_DIR, WORK_DIR
    from ..scenarios.catalog import create
    from ..sim.engine import Engine, RunConfig

    sc = create(scenario_id, dict(params or {}))
    label = f"ml-{arm}-{scenario_id}-{seed}-{os.getpid()}"
    eng = Engine(sc, RunConfig(scenario_id, sc.params, seed, "fixed", label=label), NETWORKS_DIR, WORK_DIR / label)
    eng.start()
    try:
        if policies is None:
            policies = make_policies(arm, seed, eng.net_path, directory)
        eng.signals.policies.extend(policies)
        while not eng.finished:
            eng.step()
        res = eng.results()
        m = eng.metrics
        wait = loss = window = delay = 0.0
        n = done = 0
        for mode in MOTOR:
            s = m.modes.get(mode)
            if s:
                n += s.trips
                done += s.trips
                wait += s.wait_total
                window += s.waiting
                loss += s.loss
                delay += s.depart_delay
        for tr in m.trips.values():  # still on the road
            if tr.mode in MOTOR:
                n += 1
                wait += tr.wait_total
                window += tr.waiting
                loss += tr.time_loss
                delay += tr.depart_delay
        queued = [h["queued"] for h in m.history]
        acc = res["demandAccounting"]
        duration = float(sc.duration or eng.t)
        out: dict[str, Any] = {
            "arm": arm, "scenario": scenario_id, "seed": seed, "network": eng.network_id, "duration_s": duration,
            "mean_wait_s": wait / n if n else 0.0, "mean_time_loss_s": loss / n if n else 0.0,
            "mean_wait_window_s": window / n if n else 0.0, "mean_depart_delay_s": delay / n if n else 0.0,
            "unserved": acc["neverInserted"] + acc["throttledTrips"] + acc["backgroundRouteFailures"], "never_inserted": acc["neverInserted"],
            "throttled_trips": acc["throttledTrips"], "refused_trips": acc["backgroundRouteFailures"],
            "vehicles": n, "completed": done, "completed_fraction": done / n if n else 1.0, "throughput_vph": done * 3600.0 / duration,
            "queue_mean_veh": float(np.mean(queued)) if queued else 0.0, "queue_max_veh": float(max(queued)) if queued else 0.0,
            "pending_end": res["demandAccounting"]["pendingNow"], "teleports": res["teleports"], "collisions": res["collisions"],
            "tsp_extensions": res["signals"]["tspExtensions"],
            "rules": [p.summary() for p in policies if hasattr(p, "summary")],
        }
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
