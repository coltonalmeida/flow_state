"""The publication gate and the measurements it rests on: uncapped waiting time, unserved demand, non-inferiority margins."""

from types import SimpleNamespace

import numpy as np
import pytest

from flowstate.ml import evaluate as E
from flowstate.replay.priority import EmergencyPriority
from flowstate.sim.metrics import Metrics
from flowstate.sim.signals import HARD_MAX_GREEN_S, SignalManager, TlsInfo

SCENARIOS = ("rush_hour", "transit", "stadium")
BASE = {"mean_wait_s": 40.0, "mean_time_loss_s": 90.0, "mean_wait_window_s": 22.0, "mean_depart_delay_s": 1.0, "unserved": 40.0, "throughput_vph": 2000.0,
        "completed_fraction": 0.9, "queue_mean_veh": 30.0, "queue_max_veh": 60.0, "pending_end": 0.0, "teleports": 0.0, "collisions": 0.0}


def _rows(shift=None, per_scenario=None, seeds=20, drop=()):
    """Baseline episodes plus a candidate arm whose metrics are the baseline's + ``shift`` (+ ``per_scenario[scenario]``) + small paired noise."""
    rng = np.random.default_rng(1)
    rows = []
    for sc in SCENARIOS:
        for seed in range(seeds):
            level = {k: v * (1 + 0.1 * rng.standard_normal()) if v else 0.0 for k, v in BASE.items()}
            rows.append({"arm": "queue_responsive", "scenario": sc, "seed": seed, **level})
            cand = dict(level)
            for k, d in {**(shift or {}), **(per_scenario or {}).get(sc, {})}.items():
                cand[k] += d
            for k in ("mean_wait_s", "mean_time_loss_s", "throughput_vph"):
                cand[k] += 0.002 * BASE[k] * rng.standard_normal()
            rows.append({"arm": "rl_ppo", "scenario": sc, "seed": seed, **{k: v for k, v in cand.items() if k not in drop}})
    return rows


def _gate(**kw):
    return E.gate(E.summarise(_rows(**kw), E.GATE_BASELINE), "rl_ppo")


def test_a_tie_fails_only_the_superiority_check():
    g = _gate()
    assert g["status"] == "experimental" and g["failed"] == ["wait_ci_below_zero"], g["failed"]


def test_better_wait_with_nothing_worse_is_verified():
    g = _gate(shift={"mean_wait_s": -2.0})
    assert g["status"] == "verified" and all(g["checks"].values())
    assert g["margins"]["mean_time_loss_s"]["limit"] == pytest.approx(0.02 * 90.0, rel=0.1)


@pytest.mark.parametrize("shift, check", [
    ({"mean_time_loss_s": 4.0}, "time_loss_not_worse"),
    ({"teleports": 0.2}, "no_teleport_regression"),
    ({"throughput_vph": -60.0}, "no_throughput_regression"),
    ({"unserved": 5.0}, "no_insertion_regression"),
    ({"mean_depart_delay_s": 2.0}, "no_insertion_regression"),
])
def test_each_regression_beyond_its_margin_blocks_verification(shift, check):
    g = _gate(shift={"mean_wait_s": -2.0, **shift})
    assert g["status"] == "experimental" and check in g["failed"]


def test_a_regression_in_one_scenario_is_not_hidden_by_the_pool():
    g = _gate(shift={"mean_wait_s": -3.0}, per_scenario={"stadium": {"mean_time_loss_s": 5.0}})
    assert g["status"] == "experimental" and "no_scenario_regression" in g["failed"]
    assert {(r["scenario"], r["metric"]) for r in g["scenario_regressions"]} == {("stadium", "mean_time_loss_s")}
    small = _gate(shift={"mean_wait_s": -3.0}, per_scenario={"stadium": {"mean_time_loss_s": 0.5}})  # clearly above 0 but inside the 2 % margin
    assert small["checks"]["no_scenario_regression"]


def test_an_evaluation_without_the_new_measurements_cannot_verify():
    g = E.gate(E.summarise(_rows(shift={"mean_wait_s": -2.0}, drop=("unserved", "mean_depart_delay_s")), E.GATE_BASELINE), "rl_ppo")
    assert g["status"] == "experimental" and g["failed"] == ["evaluation_outdated"]


def test_waiting_time_sums_every_standstill_spell_without_a_window():
    m = Metrics()
    m.depart("a", "car", "car", 0.0)
    for spell in (0, 1, 2, 3, 0, 0, 1, 2, 0):  # two spells: 3 s and 2 s
        m.update_vehicle("a", 0.0, 0.0, 0.0, float(spell))
    assert m.trips["a"].wait_total == 5.0
    for spell in range(1, 151):  # one 150 s standstill: SUMO's windowed figure would stop at 100
        m.update_vehicle("a", 0.0, 100.0, 0.0, float(spell))
    assert m.trips["a"].wait_total == 155.0 and m.trips["a"].waiting == 100.0
    m.arrive("a", 200.0)
    assert m.modes["car"].wait_total == 155.0


class _FakeT:
    def __init__(self):
        self.now, self.switch, self.state, self.set = 0.0, 0.0, "r", []
        self.simulation = SimpleNamespace(getTime=lambda: self.now)
        self.vehicle = SimpleNamespace(getNextTLS=lambda vid: [("j", 0, 50.0, self.state)], getSpeed=lambda vid: 10.0)
        self.trafficlight = SimpleNamespace(getNextSwitch=lambda tid: self.switch, setPhaseDuration=lambda tid, d: self.set.append(d))


def _manager():
    T = _FakeT()
    eng = SimpleNamespace(T=T, dt=0.25, emergency_active=lambda: ["amb"])
    mgr = SignalManager(eng, "fixed", tsp=False)
    mgr.tls["j"] = TlsInfo("j", phase=0, phase_start=0.0, retimable={0})
    return mgr, T


def test_ambulance_priority_cuts_the_conflicting_green_in_every_cycle():
    mgr, T = _manager()
    prio = EmergencyPriority()
    T.now, T.switch = 12.0, 40.0
    prio.step(mgr, 12.0)
    prio.step(mgr, 12.25)  # the same green is cut once
    assert prio.cuts == 1
    mgr.tls["j"].phase_start = 100.0  # next cycle: the same phase index comes round again
    T.now, T.switch = 112.0, 140.0
    prio.step(mgr, 112.0)
    assert prio.cuts == 2 and T.set == [0.0, 0.0]


def test_priority_cannot_hold_a_green_past_the_ceiling():
    mgr, T = _manager()
    prio = EmergencyPriority()
    T.state = "G"
    T.vehicle.getNextTLS = lambda vid: [("j", 0, 110.0, "G")]  # needs about 13 s
    T.now, T.switch = 70.0, 73.0  # 70 s elapsed, 3 s left: 80 s ceiling leaves under 7 s
    assert mgr.green_room("j", 70.0, 3.0) == pytest.approx(HARD_MAX_GREEN_S - 0.25 - 73.0)
    prio.step(mgr, 70.0)
    assert len(T.set) == 1 and 70.0 + T.set[0] <= HARD_MAX_GREEN_S
    T.now, T.switch = 79.5, 79.75
    prio.step(mgr, 79.5)
    assert len(T.set) == 1, "extended a green that is already at the ceiling"
    mgr.tls["j"].retimable = set()  # a clearance interval is not capped (lengthening it is always safe)
    assert mgr.green_room("j", 79.5, 0.25) == float("inf")
