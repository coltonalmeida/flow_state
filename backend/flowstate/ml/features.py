"""Per-approach traffic features, sampled every ``SAMPLE_S`` seconds from a running :class:`~flowstate.sim.engine.Engine`.

The 19 columns were first defined for the Intelligence Lab's grid model (``lab/sim.py:FEATURES``), redefined so they need no compass
and no fixed approach count:

==================  ==========================================================================================
queue               halting vehicles on the approach (all edges of its block)
count               vehicles on the approach
moving              count - queue
mean_wait           mean current waiting time of the vehicles on the approach (s)
down_occ            mean occupancy (vehicles / capacity, capped at 1) of the blocks the approach discharges into
is_green            any of the approach's signal links shows green
in_transition       the junction is not in one of its re-timable green phases (amber, all-red, clearance, left)
elapsed             seconds the current phase has run, if the approach is green
to_max_green        seconds left before the maximum green, if the approach is green
up_count            vehicles on the edges that feed the approach
up_halting          halting vehicles on those edges
up_released         vehicles on upstream signal approaches that are green right now (a platoon on its way)
q_lag10/20/30       queue 10 / 20 / 30 s ago
count_lag30         count 30 s ago
capacity            storage of the approach (vehicles)
is_entry            the approach is not fed by a signal
stage_demand_other  queue on the junction's approaches that the current signal state does not serve
==================  ==========================================================================================

Everything is measured by SUMO (edge vehicle / halting counts, edge waiting time, signal state); nothing is estimated.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from .topology import Topology

if TYPE_CHECKING:  # pragma: no cover
    from ..sim.signals import SignalManager

FEATURES = ("queue", "count", "moving", "mean_wait", "down_occ", "is_green", "in_transition", "elapsed", "to_max_green",
            "up_count", "up_halting", "up_released", "q_lag10", "q_lag20", "q_lag30", "count_lag30", "capacity", "is_entry",
            "stage_demand_other")
F = {name: i for i, name in enumerate(FEATURES)}
SAMPLE_S = 5.0  # sampling / decision interval
HISTORY = 7  # samples kept per approach (35 s)
FEATURE_VERSION = 1  # bump when the meaning of a column changes


def feature_hash() -> str:
    """Identifies the feature definition a model was trained on (names, order, sampling grid, history, version)."""
    spec = {"features": list(FEATURES), "sample_s": SAMPLE_S, "history": HISTORY, "version": FEATURE_VERSION}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


class FeatureTracker:
    """Samples the feature matrix ``X`` (approaches x features) and keeps the short history the lag columns need."""

    def __init__(self, topo: Topology):
        self.topo = topo
        n = len(topo.approaches)
        self.q_hist = np.zeros((HISTORY, n), dtype=np.float32)
        self.c_hist = np.zeros((HISTORY, n), dtype=np.float32)
        self.X = np.zeros((n, len(FEATURES)), dtype=np.float32)
        self.X[:, F["capacity"]] = [a.capacity for a in topo.approaches]
        self.X[:, F["is_entry"]] = [1.0 if a.is_entry else 0.0 for a in topo.approaches]
        self.green = np.zeros(n, dtype=bool)  # approach is served by the current signal state
        self.samples = 0
        self.t = -1e9
        self._green_set = {tid: set(s.green_phases) for tid, s in topo.signals.items()}

    def due(self, t: float) -> bool:
        return t - self.t >= SAMPLE_S - 1e-6

    def sample(self, mgr: "SignalManager", t: float) -> np.ndarray:
        topo = self.topo
        eng = mgr.engine
        es: dict[str, tuple] = eng.edge_stats  # edge -> (vehicles, halting, mean speed), polled by the engine once a second
        T = mgr.T
        zero = (0, 0, 0.0)
        X = self.X
        n = len(topo.approaches)
        q = np.zeros(n, dtype=np.float32)
        c = np.zeros(n, dtype=np.float32)
        for a in topo.approaches:
            i = a.index
            cnt = halt = 0
            for e in a.edges:
                s = es.get(e, zero)
                cnt += s[0]
                halt += s[1]
            q[i], c[i] = halt, cnt
            info = mgr.tls.get(a.tls)
            state = info.state if info is not None else ""
            self.green[i] = any(k < len(state) and state[k] in "Gg" for k in a.links)
        self.q_hist = np.roll(self.q_hist, 1, axis=0)
        self.c_hist = np.roll(self.c_hist, 1, axis=0)
        self.q_hist[0], self.c_hist[0] = q, c
        for tid, sig in topo.signals.items():
            info = mgr.tls.get(tid)
            phase = info.phase if info is not None else -1
            elapsed = max(0.0, t - info.phase_start) if info is not None else 0.0
            transition = 0.0 if phase in self._green_set[tid] else 1.0
            unserved = float(sum(q[k] for k in sig.approaches if not self.green[k]))
            for i in sig.approaches:
                a = topo.approaches[i]
                wait = sum(T.edge.getWaitingTime(e) for e in a.edges) if q[i] else 0.0
                occ = [min(1.0, sum(es.get(e, zero)[0] for e in g) / cap) for g, cap in zip(a.down_groups, a.down_caps)]
                g = bool(self.green[i])
                row = X[i]
                row[0] = q[i]
                row[1] = c[i]
                row[2] = c[i] - q[i]
                row[3] = wait / c[i] if c[i] else 0.0
                row[4] = sum(occ) / len(occ) if occ else 0.0
                row[5] = 1.0 if g else 0.0
                row[6] = transition
                row[7] = elapsed if g else 0.0
                row[8] = max(0.0, sig.max_green - elapsed) if g else 0.0
                row[9] = sum(es.get(e, zero)[0] for e in a.up_edges)
                row[10] = sum(es.get(e, zero)[1] for e in a.up_edges)
                row[11] = sum(c[k] for k in a.upstream if self.green[k])
                row[18] = unserved - (0.0 if g else q[i])
        self.samples += 1
        k = min(self.samples - 1, HISTORY - 1)
        X[:, 12] = self.q_hist[min(2, k)]
        X[:, 13] = self.q_hist[min(4, k)]
        X[:, 14] = self.q_hist[min(6, k)]
        X[:, 15] = self.c_hist[min(6, k)]
        self.t = t
        return X

    def signal_rows(self, tid: str) -> np.ndarray:
        """The feature rows of one junction's approaches (variable length)."""
        return self.X[list(self.topo.signals[tid].approaches)]


def describe() -> dict[str, Any]:
    return {"features": list(FEATURES), "sampleS": SAMPLE_S, "history": HISTORY, "version": FEATURE_VERSION, "featureHash": feature_hash()}
