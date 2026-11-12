"""Signal policies of the ML layer. All of them act through :class:`GreenActuator`, the only code here that calls TraCI.

Safety contract (enforced by the actuator, whatever a model asks for):

* only the *re-timable green* phases of a junction are ever touched (``Topology.Signal.green_phases``). Amber, all-red,
  pedestrian clearance and protected-left intervals always run their programmed length, and the phase sequence is the plan's;
* a green is never ended before the junction's minimum green (8 / 10 / 12 s by intersection type);
* a green is never extended past ``max_green`` (60 s);
* a phase that anything else has re-timed (transit signal priority, ambulance priority, any other rule) is left alone: the
  actuator compares SUMO's next switch with the programmed end plus its own changes. These policies must therefore be
  appended *after* :class:`~flowstate.replay.priority.EmergencyPriority` (TSP always runs first, inside the manager).

The only TraCI call is ``trafficlight.setPhaseDuration`` (end the green now, or let it run a few seconds longer).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..sim.signals import SignalManager, SignalPolicy
from .features import SAMPLE_S, F, FeatureTracker
from .topology import Topology, build_topology

EXTEND_S = 4.0
SWITCH_RATIO = 1.5
SWITCH_MARGIN = 3.0


class GreenActuator:
    """Ends or extends greens, and refuses anything outside the safety contract. Refusals are counted, never executed."""

    TOLERANCE_S = 0.3  # one simulation step

    def __init__(self, topo: Topology):
        self.topo = topo
        self._green = {tid: set(s.green_phases) for tid, s in topo.signals.items()}
        self._ended: dict[str, float] = {}  # tls -> phase_start of the phase already ended
        self._phase: dict[str, list[Any]] = {}  # tls -> [phase_start, expected end (SUMO time) or None]
        self.ended = self.extended = self.refused = 0
        self.extended_s = 0.0

    # -- state ---------------------------------------------------------------------------------------------------------
    def _track(self, mgr: SignalManager, tid: str) -> tuple[float, bool]:
        """(expected end of the current phase in SUMO time, phase start was observed).

        The expected end is the programmed one plus this actuator's own changes. If SUMO's next switch differs, someone
        else (transit priority, ambulance priority, any other rule) has re-timed the phase."""
        info = mgr.tls[tid]
        rec = self._phase.get(tid)
        if rec is None:  # first phase seen after start: its true start (the plan's offset) was not observed
            rec = self._phase[tid] = [info.phase_start, None]
        elif rec[0] != info.phase_start:
            start = info.phase_start + mgr.engine.warmup
            rec[0], rec[1] = info.phase_start, start + self.topo.signals[tid].phases[info.phase].duration
        return rec[1], rec[1] is not None

    def priority_owned(self, mgr: SignalManager, tid: str) -> bool:
        """The current phase was re-timed by something other than this actuator (or its start was not observed)."""
        expected, known = self._track(mgr, tid)
        if not known or mgr.tls[tid].extended > 0:
            return True
        return abs(mgr.T.trafficlight.getNextSwitch(tid) - expected) > self.TOLERANCE_S

    def controllable(self, mgr: SignalManager, tid: str) -> bool:
        info = mgr.tls.get(tid)
        sig = self.topo.signals.get(tid)
        if info is None or sig is None or info.phase not in self._green[tid]:
            return False
        if info.state != sig.phases[info.phase].state:  # another program is running: do not guess
            return False
        return not self.priority_owned(mgr, tid)

    def elapsed(self, mgr: SignalManager, tid: str, t: float) -> float:
        return t - mgr.tls[tid].phase_start

    def remaining(self, mgr: SignalManager, tid: str) -> float:
        return mgr.T.trafficlight.getNextSwitch(tid) - mgr.T.simulation.getTime()

    def can_end(self, mgr: SignalManager, tid: str, t: float) -> bool:
        return (self.controllable(mgr, tid) and self.elapsed(mgr, tid, t) >= self.topo.signals[tid].min_green - 1e-6
                and self._ended.get(tid) != mgr.tls[tid].phase_start)

    # -- actions -------------------------------------------------------------------------------------------------------
    def end_green(self, mgr: SignalManager, tid: str, t: float) -> bool:
        if not self.can_end(mgr, tid, t):
            self.refused += 1
            return False
        if self.remaining(mgr, tid) <= 1.0:  # it ends by itself
            return False
        self._ended[tid] = mgr.tls[tid].phase_start
        self.ended += 1
        mgr.T.trafficlight.setPhaseDuration(tid, 0.0)
        return True

    def extend(self, mgr: SignalManager, tid: str, t: float, seconds: float = EXTEND_S) -> bool:
        if not self.controllable(mgr, tid) or self._ended.get(tid) == mgr.tls[tid].phase_start:
            self.refused += 1
            return False
        remaining = self.remaining(mgr, tid)
        room = self.topo.signals[tid].max_green - (self.elapsed(mgr, tid, t) + remaining)
        grant = min(seconds, room)
        if grant < 0.5:
            self.refused += 1
            return False
        self.extended += 1
        self.extended_s += grant
        self._phase[tid][1] += grant
        mgr.T.trafficlight.setPhaseDuration(tid, remaining + grant)
        return True

    def summary(self) -> dict[str, Any]:
        return {"greensEnded": self.ended, "extensions": self.extended, "extendedSeconds": round(self.extended_s, 1), "refused": self.refused}


class _TopologyPolicy(SignalPolicy):
    """Base: builds the topology of the engine's network on first use and samples features every ``SAMPLE_S``."""

    def __init__(self) -> None:
        self.topo: Topology | None = None
        self.tracker: FeatureTracker | None = None
        self.act: GreenActuator | None = None
        self._last = -1e9

    def _ensure(self, mgr: SignalManager) -> None:
        if self.topo is None:
            self.topo = build_topology(mgr.engine.net_path)
            self.tracker = FeatureTracker(self.topo)
            self.act = GreenActuator(self.topo)

    def _sample(self, mgr: SignalManager, t: float) -> bool:
        """True when a new sample was taken at this step."""
        self._ensure(mgr)
        assert self.tracker is not None
        if not self.tracker.due(t):
            return False
        self.tracker.sample(mgr, t)
        self.on_sample(mgr, t)
        return True

    def on_sample(self, mgr: SignalManager, t: float) -> None:
        pass


class RecorderPolicy(_TopologyPolicy):
    """Passive: records the feature matrix every ``SAMPLE_S`` of scenario time. Never touches a signal."""

    id = "ml_recorder"

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[np.ndarray] = []
        self.times: list[float] = []

    def step(self, mgr: SignalManager, t: float) -> None:
        self._sample(mgr, t)

    def on_sample(self, mgr: SignalManager, t: float) -> None:
        if t >= 0:  # warm-up samples only fill the history
            assert self.tracker is not None
            self.rows.append(self.tracker.X.copy())
            self.times.append(t)


class RandomTiming(_TopologyPolicy):
    """Dataset arm: random (safe) green lengths, so a forecaster does not only see one controller's timing."""

    id = "random_timing"

    def __init__(self, seed: int = 0, p_end: float = 0.12, p_extend: float = 0.5) -> None:
        super().__init__()
        self.rng = random.Random(seed * 9973 + 5)
        self.p_end, self.p_extend = p_end, p_extend

    def step(self, mgr: SignalManager, t: float) -> None:
        if not self._sample(mgr, t):
            return
        assert self.topo is not None and self.act is not None
        for tid in self.topo.tls_ids:
            if not self.act.controllable(mgr, tid):
                continue
            r = self.rng.random()
            if r < self.p_end and self.act.can_end(mgr, tid, t):
                self.act.end_green(mgr, tid, t)
            elif self.act.remaining(mgr, tid) <= SAMPLE_S + 0.5 and self.rng.random() < self.p_extend:
                self.act.extend(mgr, tid, t, SAMPLE_S)

    def summary(self) -> dict[str, Any]:
        return {"rule": "random safe green lengths (dataset generation only)", **(self.act.summary() if self.act else {})}


class ForecastPolicy(_TopologyPolicy):
    """Queue-responsive rule (gap-out, switch on demand, extend) with bounded adjustments from a trained queue forecast.

    The rule is ``replay/heuristic.py:QueueResponsive``'s, evaluated per approach instead of per lane and acted through
    :class:`GreenActuator`. The forecast enters in two bounded places, with the parameters of the lab's ``ForecastAware``:

    * **anticipated demand** - the demand of the green and of the red approaches is shifted by
      ``beta x sum(queue predicted in 30 s - queue now)``, clipped to +/- ``bound`` vehicles;
    * **predicted spillback** - if a block the green discharges into is forecast to be at least ``spill`` full, the green is
      treated as blocked: it is not extended, and it ends as soon as the minimum green allows if anyone waits on red.

    Without a forecaster (``forecaster=None``) this is the plain rule under the same safety contract; that arm is what the
    forecast must beat for the ML to be credited with anything.
    """

    id = "ml_forecast"

    def __init__(self, forecaster: Any | None = None, beta: float = 0.5, bound: float = 6.0, spill: float = 0.85) -> None:
        super().__init__()
        self.forecaster = forecaster
        self.beta, self.bound, self.spill = beta, bound, spill
        self.p30: np.ndarray | None = None
        self.gap_outs = self.switches = self.spill_ends = self.spill_holds = 0
        self.decisions = self.changed = 0
        self._step_last = -1e9
        if forecaster is None:
            self.id = "safe_heuristic"

    def on_sample(self, mgr: SignalManager, t: float) -> None:
        if self.forecaster is not None:
            assert self.tracker is not None
            self.p30 = self.forecaster.predict(self.tracker.X)[0]

    def _decide(self, tid: str, q: np.ndarray, c: np.ndarray, use_forecast: bool) -> tuple[str, bool]:
        """('gap' | 'switch' | 'spill' | 'none', may_extend) for one junction in a re-timable green, from the latest sample."""
        assert self.topo is not None and self.tracker is not None
        sig = self.topo.signals[tid]
        green = [i for i in sig.approaches if self.tracker.green[i]]
        red = [i for i in sig.approaches if not self.tracker.green[i]]
        green_q, red_q = float(sum(q[i] for i in green)), float(sum(q[i] for i in red))
        green_n = float(sum(c[i] for i in green))
        dem_g, dem_r, blocked = green_q, red_q, False
        if use_forecast and self.p30 is not None:
            cap = self.tracker.X[:, F["capacity"]]
            p = np.minimum(self.p30, cap)
            dem_g += max(-self.bound, min(self.bound, self.beta * float(sum(p[i] - q[i] for i in green))))
            dem_r += max(-self.bound, min(self.bound, self.beta * float(sum(p[i] - q[i] for i in red))))
            blocked = any(p[k] >= self.spill * cap[k] for i in green for k in self.topo.approaches[i].downstream)
        if green_n == 0 and red_q > 0:
            return "gap", False
        if dem_r > SWITCH_RATIO * dem_g + SWITCH_MARGIN:
            return "switch", False
        if blocked and red_q > 0:
            return "spill", False
        return "none", (dem_g > dem_r + 1 and not blocked)

    def step(self, mgr: SignalManager, t: float) -> None:
        self._sample(mgr, t)
        if t - self._step_last < 1.0 - 1e-6:
            return
        self._step_last = t
        assert self.topo is not None and self.tracker is not None and self.act is not None
        act = self.act
        # live queues (the engine polls edges once a second); the forecast is the one from the latest 5 s sample
        es = mgr.engine.edge_stats
        n = len(self.topo.approaches)
        q, c = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        for a in self.topo.approaches:
            for e in a.edges:
                s = es.get(e)
                if s:
                    c[a.index] += s[0]
                    q[a.index] += s[1]
            info = mgr.tls.get(a.tls)
            st = info.state if info is not None else ""
            self.tracker.green[a.index] = any(k < len(st) and st[k] in "Gg" for k in a.links)
        for tid in self.topo.tls_ids:
            if not act.controllable(mgr, tid):
                continue
            what, may_extend = self._decide(tid, q, c, True)
            if self.forecaster is not None:
                base = self._decide(tid, q, c, False)
                self.decisions += 1
                self.changed += 1 if base != (what, may_extend) else 0
            remaining = act.remaining(mgr, tid)
            if what != "none" and remaining > 1.0 and act.can_end(mgr, tid, t):
                if act.end_green(mgr, tid, t):
                    if what == "gap":
                        self.gap_outs += 1
                    elif what == "switch":
                        self.switches += 1
                    else:
                        self.spill_ends += 1
                continue
            if remaining <= 1.5 and may_extend:
                act.extend(mgr, tid, t, EXTEND_S)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rule": "queue-responsive green durations" + (" + 30 s queue forecast (bounded demand shift, predicted spillback)" if self.forecaster is not None else " (no forecast)"),
                               "gapOuts": self.gap_outs, "switches": self.switches, "spillbackEnds": self.spill_ends, **(self.act.summary() if self.act else {})}
        if self.forecaster is not None:
            out |= {"decisions": self.decisions, "decisionsChangedByForecast": self.changed, "beta": self.beta, "bound": self.bound, "spill": self.spill}
        return out
