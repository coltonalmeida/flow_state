"""ML policies: the safety contract of GreenActuator, checked on the phase runs SUMO actually executed.

Amber, all-red, pedestrian clearance and left phases are never shortened; a green never ends before the minimum green or runs past
the maximum; the phase sequence is the program's.
"""

import pytest

from flowstate.ml.policies import ForecastPolicy, RandomTiming
from flowstate.ml.topology import build_topology
from flowstate.replay.priority import EmergencyPriority
from helpers import make_engine, run_until

TOL = 0.25  # one simulation step


class PhaseMonitor:
    """Records every completed phase run per signal: (phase index, observed duration). The first run is skipped (its start is unobserved)."""

    def __init__(self):
        self.cur: dict[str, tuple[int, float]] = {}
        self.seen_first: set[str] = set()
        self.runs: dict[str, list[tuple[int, float]]] = {}

    def __call__(self, eng) -> None:
        for tid, info in eng.signals.tls.items():
            if info.phase < 0:
                continue
            cur = self.cur.get(tid)
            if cur is None:
                self.cur[tid] = (info.phase, info.phase_start)
            elif info.phase_start != cur[1]:
                if tid in self.seen_first:
                    self.runs.setdefault(tid, []).append((cur[0], info.phase_start - cur[1]))
                self.seen_first.add(tid)
                self.cur[tid] = (info.phase, info.phase_start)


class StubForecaster:
    def predict(self, X):
        return X[:, 0] + 5, X[:, 0] + 5


def _drive(scenario, params, policies, until=None, seed=7):
    eng = make_engine(scenario, seed, params, label="ml_policy")
    eng.start()
    eng.signals.policies.extend(policies)
    mon = PhaseMonitor()
    try:
        run_until(eng, until if until is not None else 1e9, mon)
    finally:
        eng.close()
    return mon


def _check_contract(mon, topo, check_greens=True):
    assert sum(len(r) for r in mon.runs.values()) > 100
    for tid, runs in mon.runs.items():
        sig = topo.signals[tid]
        n = len(sig.phases)
        for k, (idx, dur) in enumerate(runs):
            prog = sig.phases[idx].duration
            if idx not in sig.green_phases:
                assert dur >= prog - TOL, f"{tid} phase {idx} ({sig.phases[idx].role}) ran {dur}s < programmed {prog}s"
            elif check_greens:
                assert dur >= min(prog, sig.min_green) - TOL, f"{tid} green {idx} ran {dur}s < min {min(prog, sig.min_green)}s"
                assert dur <= sig.max_green + TOL, f"{tid} green {idx} ran {dur}s > max_green"
            if k:
                assert idx == (runs[k - 1][0] + 1) % n, f"{tid}: phase {runs[k - 1][0]} -> {idx}"


@pytest.mark.parametrize("arm", ["random", "heuristic", "forecast"])
def test_safety_contract_holds_on_rush_hour(arm):
    pol = {"random": lambda: RandomTiming(seed=11), "heuristic": lambda: ForecastPolicy(),
           "forecast": lambda: ForecastPolicy(forecaster=StubForecaster())}[arm]()
    mon = _drive("rush_hour", {"duration": 5}, [pol], until=300.0)
    topo = pol.topo
    _check_contract(mon, topo)
    s = pol.act.summary()
    assert s["greensEnded"] + s["extensions"] > 0, f"{arm} policy did nothing: {s}"
    if arm == "forecast":
        assert pol.decisions > 0


def test_forecast_policy_only_touches_retimable_greens_beside_emergency_priority():
    """With ambulance priority running first, every TraCI call the ML policy makes is on a re-timable green that priority has not re-timed."""
    pol = ForecastPolicy()
    eng = make_engine("ambulance", 3100, {"duration": 5, "calls": 2}, label="ml_policy_ev")
    eng.start()
    prio = EmergencyPriority()
    eng.signals.policies.extend([prio, pol])
    calls: list[tuple[str, int, str]] = []
    try:
        eng.step()
        act = pol.act
        for name in ("end_green", "extend"):
            def wrapped(mgr, tid, t, *a, _f=getattr(act, name), _n=name, **kw):
                info = mgr.tls[tid]
                owned = prio._mark.get(tid) == info.phase_start and (tid in prio._ext)
                ok = _f(mgr, tid, t, *a, **kw)
                if ok:
                    calls.append((tid, info.phase, _n))
                    assert not owned, f"{tid}: ML {_n} on a phase ambulance priority extended"
                return ok
            setattr(act, name, wrapped)
        run_until(eng, 1e9)
    finally:
        eng.close()
    assert calls, "the ML policy never acted"
    for tid, phase, name in calls:
        assert phase in pol.topo.signals[tid].green_phases, f"{tid}: ML {name} on phase {phase} ({pol.topo.signals[tid].phases[phase].role})"


@pytest.mark.xfail(strict=True, reason="pre-existing on main: replay/priority.py EmergencyPriority treats a pedestrian-clearance phase that still shows vehicle "
                                       "greens as a conflicting green and cuts it (e.g. n3_7 clearance 14 s -> 10.25 s); replay/heuristic.py does the same")
def test_emergency_priority_never_shortens_clearances():
    from flowstate.api.service import NETWORKS_DIR

    topo = build_topology(NETWORKS_DIR / "district.net.xml")
    mon = _drive("ambulance", {"duration": 5, "calls": 2}, [EmergencyPriority()], seed=3100)
    _check_contract(mon, topo, check_greens=False)  # emergency priority legitimately re-times greens


def test_actuator_leaves_transit_priority_phases_alone():
    eng = make_engine("transit", 5, {"duration": 10}, label="ml_tsp")
    eng.start()
    pol = ForecastPolicy()
    eng.signals.policies.extend([pol])
    mgr = eng.signals
    checked = 0
    try:
        while eng.t < 600 and not eng.finished:
            eng.step()
            if pol.act is None:
                continue
            for tid, info in mgr.tls.items():
                if info.extended > 0:
                    checked += 1
                    assert not pol.act.controllable(mgr, tid), f"{tid} controllable after a TSP extension"
        tsp = mgr.tsp_count
    finally:
        eng.close()
    if tsp == 0 or checked == 0:
        pytest.skip("no transit priority extension happened within 600 s of the transit scenario")
    assert checked > 0
