"""Signal policies of the ML layer. All of them act through :class:`GreenActuator`, the only code here that calls TraCI.

Safety contract (enforced by the actuator, whatever a model asks for):

* only the *re-timable green* phases of a junction are ever touched (``Topology.Signal.green_phases``). Amber, all-red,
  pedestrian clearance and protected-left intervals always run their programmed length, and the phase sequence is the plan's;
* a green is never ended before the junction's minimum green (8 / 10 / 12 s by intersection type);
* a green is never extended past ``max_green`` (60 s) by this actuator. Transit and ambulance priority are separate rules with their
  own budgets (15 s / 20 s) and may lengthen a green afterwards, but the signal manager stops them at ``HARD_MAX_GREEN_S`` (80 s), so
  no combination of rules holds a green longer than that;
* the actuator only acts on the pre-timed (``fixed``) program: under SUMO's actuated logic it does nothing;
* a phase that anything else has re-timed (transit signal priority, ambulance priority, any other rule) is left alone: the
  actuator compares SUMO's next switch with the programmed end plus its own changes. These policies must therefore be
  appended *after* :class:`~flowstate.replay.priority.EmergencyPriority` (TSP always runs first, inside the manager).

The only TraCI call is ``trafficlight.setPhaseDuration`` (end the green now, or let it run a few seconds longer).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.signals import SignalManager, SignalPolicy
from .features import SAMPLE_S, F, FeatureTracker
from .topology import Topology, build_topology

EXTEND_S = 4.0


class ModelArtifactError(RuntimeError):
    """The trained policy is missing, or does not match its manifest, this code or this network."""


class GreenActuator:
    """Ends or extends greens, and refuses anything outside the safety contract. Refusals are counted, never executed."""

    TOLERANCE_S = 0.3  # one simulation step, when comparing SUMO's next switch with the expected one

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
        if info is None or sig is None or info.phase not in self._green[tid] or info.phase not in info.retimable or mgr.mode != "fixed":
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

    def _room(self, mgr: SignalManager, tid: str, t: float, remaining: float) -> float:
        """Seconds the green may still be extended. A phase start is observed up to one simulation step late, so one step is held back:
        the green then ends by ``max_green`` on the clock, not one step after it."""
        return self.topo.signals[tid].max_green - mgr.engine.dt - (self.elapsed(mgr, tid, t) + remaining)

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
        grant = min(seconds, self._room(mgr, tid, t, remaining))
        if grant < 0.5:
            self.refused += 1
            return False
        self.extended += 1
        self.extended_s += grant
        self._phase[tid][1] += grant
        mgr.T.trafficlight.setPhaseDuration(tid, remaining + grant)
        return True

    def can_extend(self, mgr: SignalManager, tid: str, t: float, window: float) -> bool:
        """An extension would be accepted now: the green is ours, not ended, within ``window`` s of its end and below the maximum."""
        if not self.controllable(mgr, tid) or self._ended.get(tid) == mgr.tls[tid].phase_start:
            return False
        remaining = self.remaining(mgr, tid)
        return remaining <= window and self._room(mgr, tid, t, remaining) >= 0.5

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


class LearnedPolicy(_TopologyPolicy):
    """The trained per-junction policy (``ml/train_rl.py``): one shared network picks hold / end / extend for every signal every 5 s.

    Every action goes through :class:`GreenActuator` and is drawn only from the actions the actuator says are valid, so the
    safety contract at the top of this module holds for any weights. Modes: ``greedy`` (deployment), ``sample`` (training
    rollouts), ``teacher`` (the queue-balancing starting rule of ``rl_env.observe``; behaviour-cloning data for the warm start). With ``record`` the trajectory is kept.
    """

    id = "rl_ppo"

    def __init__(self, weights: dict[str, np.ndarray] | None = None, mode: str = "greedy", seed: int = 0, eps: float = 0.0, record: bool = False) -> None:
        super().__init__()
        if mode != "teacher" and weights is None:
            raise ValueError("a learned policy needs weights")
        self.weights, self.mode, self.eps, self.record = weights, mode, eps, record
        self.rng = np.random.default_rng(seed)
        self.traj: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "mask", "act", "logp", "val", "teacher", "queue", "backlog")}
        self.counts = [0, 0, 0]
        self.meta: dict[str, Any] = {}

    @classmethod
    def load(cls, directory: Any, net_path: Any = None) -> "LearnedPolicy":
        """Load ``policy.npz`` after checking its manifest (hash, observation layout, feature definition, network)."""
        import hashlib
        import json
        from pathlib import Path

        from .features import feature_hash
        from .rl_env import OBS_DIM, OBS_VERSION

        d = Path(directory)
        man_path, w_path = d / "RL_MANIFEST.json", d / "policy.npz"
        if not man_path.exists() or not w_path.exists():
            raise ModelArtifactError(f"no trained policy in {d} (run: python -m flowstate.ml.cli train-rl)")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if hashlib.sha256(w_path.read_bytes()).hexdigest() != man.get("sha256"):
            raise ModelArtifactError("policy.npz does not match its manifest")
        if man.get("obs_dim") != OBS_DIM or man.get("obs_version") != OBS_VERSION or man.get("feature_hash") != feature_hash():
            raise ModelArtifactError("the policy was trained on a different observation or feature definition")
        if net_path is not None and man.get("network_id") != build_topology(net_path).network_id:
            raise ModelArtifactError(f"the policy was trained on network {man.get('network_id')}, not this one")
        with np.load(w_path) as z:
            pol = cls({k: z[k] for k in z.files})
        pol.meta = man
        return pol

    def step(self, mgr: SignalManager, t: float) -> None:
        self._sample(mgr, t)

    def on_sample(self, mgr: SignalManager, t: float) -> None:
        from . import rl_env as R

        assert self.topo is not None and self.tracker is not None and self.act is not None
        o = R.observe(self.topo, self.tracker, self.act, mgr, t)
        mask = o["mask"]
        n = len(mask)
        logp, val = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        if self.mode == "teacher":
            action = o["teacher"].copy()
        else:
            logits, val = R.forward(self.weights, o["obs"])  # type: ignore[arg-type]
            lp = R.masked_log_probs(logits, mask)
            if self.mode == "sample":
                u = self.rng.random((n, 1))
                action = np.minimum((np.cumsum(np.exp(lp), axis=1) < u).sum(axis=1), R.N_ACTIONS - 1)
                action = np.where(mask[np.arange(n), action], action, lp.argmax(axis=1))
            else:
                action = lp.argmax(axis=1)
            logp = lp[np.arange(n), action].astype(np.float32)
        if self.eps > 0:  # exploration around the teacher: a random valid action
            for j in np.nonzero(self.rng.random(n) < self.eps)[0]:
                action[j] = self.rng.choice(np.nonzero(mask[j])[0])
        for j, tid in enumerate(self.topo.tls_ids):
            a = int(action[j])
            self.counts[a] += 1
            if a == R.END:
                self.act.end_green(mgr, tid, t)
            elif a == R.EXTEND:
                self.act.extend(mgr, tid, t, SAMPLE_S)
        if self.record:
            for k, v in (("obs", o["obs"]), ("mask", mask), ("act", action.astype(np.int64)), ("logp", logp), ("val", val.astype(np.float32)),
                         ("teacher", o["teacher"]), ("queue", o["queue"]),
                         ("backlog", np.float32((mgr.engine.kpi.get("pending", 0) if mgr.engine.kpi else 0) + mgr.engine.demand.deferred))):
                self.traj[k].append(v)

    def trajectory(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.traj.items()} if self.traj["obs"] else {}

    def summary(self) -> dict[str, Any]:
        return {"rule": "learned per-junction policy (shared network; hold / end / extend every 5 s, masked to safe timings)" if self.mode != "teacher"
                else "warm-start teacher rule at 5 s decisions", "decisions": sum(self.counts), "holds": self.counts[0], "ends": self.counts[1],
                "extends": self.counts[2], **(self.act.summary() if self.act else {})}
