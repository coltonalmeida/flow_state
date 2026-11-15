"""Per-junction decision problem for the learned signal controller: observation, action mask, teacher rule, reward, and the
numpy forward pass of the shared policy network.

One set of weights controls every signal. Every ``SAMPLE_S`` (5 s) each junction gets its own observation and picks one of

* ``HOLD``    let the plan's green run as timed;
* ``END``     end the green now (valid only once the minimum green is served);
* ``EXTEND``  let the green run ``SAMPLE_S`` longer (valid only when it is about to end and the maximum green allows).

The mask comes from :class:`~flowstate.ml.policies.GreenActuator`, which also refuses anything invalid at execution time, so a
badly trained network can only choose among safe timings. Outside a re-timable green the only valid action is ``HOLD``.

The observation has no compass, no junction id and no fixed approach count: approaches are pooled into the group the signal is
serving ("green") and the group it is holding ("red"), so the same network reads a 3-leg and a 4-leg junction.

Reward (per junction, per decision): minus the halting vehicles 5 s later, blended between the junction's own approaches and
the network mean (so a junction is not paid for pushing its queue onto a neighbour). The network term also counts the vehicles
held outside the network (waiting to be inserted, or trips the demand model held back because of spillback), so a policy is not
paid for keeping traffic off the network either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .features import SAMPLE_S, F

if TYPE_CHECKING:  # pragma: no cover
    from ..sim.signals import SignalManager
    from .features import FeatureTracker
    from .policies import GreenActuator
    from .topology import Topology

HOLD, END, EXTEND = 0, 1, 2
N_ACTIONS = 3
GROUP_DIM = 10
OBS_DIM = 2 * GROUP_DIM + 6
OBS_VERSION = 1
SWITCH_RATIO, SWITCH_MARGIN = 1.5, 3.0  # the teacher ends a green when the red side queues more than ratio x green + margin


def _group(rows: np.ndarray) -> list[float]:
    if not len(rows):
        return [0.0] * GROUP_DIM
    q, cap = rows[:, F["queue"]], np.maximum(rows[:, F["capacity"]], 1.0)
    return [float(q.sum()) / 10.0, float((q / cap).max()), float(rows[:, F["count"]].sum()) / 20.0, float(rows[:, F["moving"]].sum()) / 20.0,
            min(2.0, float(rows[:, F["mean_wait"]].max()) / 60.0), float(rows[:, F["down_occ"]].max()), float(rows[:, F["up_count"]].sum()) / 20.0,
            float(rows[:, F["up_released"]].sum()) / 20.0, float(rows[:, F["q_lag30"]].sum()) / 10.0, len(rows) / 4.0]


def observe(topo: "Topology", tracker: "FeatureTracker", act: "GreenActuator", mgr: "SignalManager", t: float) -> dict[str, np.ndarray]:
    """Observation, action mask, teacher action and halting count of every junction, in ``topo.tls_ids`` order."""
    n = len(topo.tls_ids)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    mask = np.zeros((n, N_ACTIONS), dtype=bool)
    mask[:, HOLD] = True
    teacher = np.zeros(n, dtype=np.int64)
    queue = np.zeros(n, dtype=np.float32)
    X = tracker.X
    for j, tid in enumerate(topo.tls_ids):
        sig = topo.signals[tid]
        apps = list(sig.approaches)
        g = tracker.green[apps]
        rows = X[apps]
        green, red = rows[g], rows[~g]
        ctrl = act.controllable(mgr, tid)
        elapsed = act.elapsed(mgr, tid, t)
        remaining = act.remaining(mgr, tid)
        can_end = ctrl and remaining > 1.0 and act.can_end(mgr, tid, t)
        can_ext = ctrl and act.can_extend(mgr, tid, t, SAMPLE_S + 0.5)
        mask[j, END], mask[j, EXTEND] = can_end, can_ext
        obs[j, :GROUP_DIM] = _group(green)
        obs[j, GROUP_DIM:2 * GROUP_DIM] = _group(red)
        obs[j, 2 * GROUP_DIM:] = [1.0 if ctrl else 0.0, min(2.0, elapsed / 60.0), max(0.0, min(2.0, remaining / 60.0)), sig.min_green / 60.0,
                                  1.0 if can_end else 0.0, 1.0 if can_ext else 0.0]
        gq, rq = float(green[:, F["queue"]].sum()), float(red[:, F["queue"]].sum())
        gn = float(green[:, F["count"]].sum())
        queue[j] = gq + rq
        if can_end and ((gn == 0 and rq > 0) or rq > SWITCH_RATIO * gq + SWITCH_MARGIN):
            teacher[j] = END
        elif can_ext and gq > rq + 1:
            teacher[j] = EXTEND
    return {"obs": obs, "mask": mask, "teacher": teacher, "queue": queue}


def forward(w: dict[str, np.ndarray], obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(logits (n, 3), value (n,)) of the shared network: two tanh layers per head, weights as saved by ``train_rl``."""
    def head(name: str) -> np.ndarray:
        h = obs
        for i in (0, 2):
            h = np.tanh(h @ w[f"{name}.{i}.weight"].T + w[f"{name}.{i}.bias"])
        return h @ w[f"{name}.4.weight"].T + w[f"{name}.4.bias"]

    return head("pi"), head("v")[:, 0]


def masked_log_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -1e9)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def rewards(queue: np.ndarray, blend: float = 0.5, scale: float = 0.1, backlog: np.ndarray | None = None) -> np.ndarray:
    """Reward of the decision at sample k for every junction: (T-1, J) from the (T, J) halting counts and the (T,) insertion backlog."""
    nxt = queue[1:]
    net = nxt.mean(axis=1, keepdims=True)
    if backlog is not None:
        net = net + backlog[1:, None] / queue.shape[1]
    return (-scale * (blend * nxt + (1.0 - blend) * net)).astype(np.float32)


def describe() -> dict[str, Any]:
    return {"obsDim": OBS_DIM, "obsVersion": OBS_VERSION, "actions": ["hold", "end", "extend"], "decisionS": SAMPLE_S}
