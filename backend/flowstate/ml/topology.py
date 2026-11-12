"""Signal topology of any SUMO network, read from the ``.net.xml`` alone (no running simulation).

An *approach* is one incoming edge of a signalised junction, keyed by its edge id. There is no compass: T junctions simply have
three approaches, one-way streets have no opposing approach, and an arterial block that netconvert split into a main part and a
turn-bay part is one approach whose ``edges`` cover both parts.

For every approach the topology records

* the vehicle lanes and signal link indices at the stop line,
* the green phases of the junction's program that serve it,
* where it discharges (``to_edges``, and the approaches of the next signal those edges lead to),
* what feeds it (``up_edges``, and the approaches of the upstream signal if there is one),
* its storage capacity in vehicles (lane length x lanes / 7.5 m).

For every signal it records the program (duration, state, role per phase), which phases a controller may re-time
(``green_phases``: the vehicle greens; never amber, all-red, pedestrian clearance or the short protected-left phase) and the
minimum green of its intersection type.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..sim.signals import ALL_RED, CLEAR, GREEN, LEFT, LEFT_YELLOW_ROLE, TYPE_PLANS, YELLOW, infer_crossings, phase_roles, read_signal_designs

SLOT_M = 7.5  # road space of one queued vehicle (same constant as NetInfo lane capacity)
DEFAULT_MIN_GREEN_S = 10.0
MAX_GREEN_S = 60.0
# names written by ``sim.signals.retime_net`` -> role
NAME_ROLES = {"walk": GREEN, "green": GREEN, "ped-clear": CLEAR, "yellow": YELLOW, "left": LEFT, "left-yellow": LEFT_YELLOW_ROLE, "all-red": ALL_RED}
VEHICLE_CLASSES = ("passenger", "bus", "truck", "delivery", "emergency", "taxi")


@dataclass(frozen=True)
class Phase:
    index: int
    duration: float
    state: str
    role: str  # green | clear | yellow | left | left_yellow | all_red


@dataclass(frozen=True)
class Approach:
    index: int
    tls: str
    edge: str  # incoming edge at the stop line (the key)
    edges: tuple[str, ...]  # stop-line edge first, then the upstream parts of the same block
    lanes: tuple[str, ...]  # motor-vehicle lanes at the stop line
    links: tuple[int, ...]  # signal link indices of those lanes
    green_phases: tuple[int, ...]  # re-timable green phases that serve this approach
    to_edges: tuple[str, ...]  # edges it discharges into
    down_groups: tuple[tuple[str, ...], ...]  # per to-edge: the edges of that downstream block
    down_caps: tuple[float, ...]  # capacity of each downstream block (vehicles)
    downstream: tuple[int, ...]  # approaches (of other signals) its to-edges lead to
    up_edges: tuple[str, ...]  # edges feeding this approach
    upstream: tuple[int, ...]  # approaches of the upstream signal among ``up_edges``
    upstream_tls: str | None
    is_entry: bool  # not fed by a signal
    capacity: float  # vehicles


@dataclass(frozen=True)
class Signal:
    id: str
    kind: str  # intersection type (arterial_collector, ...) or "" when unknown
    phases: tuple[Phase, ...]
    green_phases: tuple[int, ...]  # phases a controller may shorten or extend
    min_green: float
    max_green: float
    approaches: tuple[int, ...]
    crossing_links: frozenset[int]


@dataclass(frozen=True)
class Topology:
    network_id: str  # "<file stem>:<hash of the signal structure>"
    signals: dict[str, Signal]
    approaches: tuple[Approach, ...]
    by_edge: dict[str, int]  # stop-line edge -> approach index
    edges: tuple[str, ...]  # every edge any feature reads

    @property
    def tls_ids(self) -> list[str]:
        return sorted(self.signals)

    @property
    def max_approaches(self) -> int:
        return max(len(s.approaches) for s in self.signals.values())

    def of(self, tls: str) -> list[Approach]:
        return [self.approaches[i] for i in self.signals[tls].approaches]

    def describe(self) -> dict[str, Any]:
        return {"networkId": self.network_id, "signals": len(self.signals), "approaches": len(self.approaches),
                "approachesPerSignal": {t: len(s.approaches) for t, s in sorted(self.signals.items())}}


def _is_vehicle_lane(lane: Any) -> bool:
    return any(lane.allows(c) for c in VEHICLE_CLASSES)


def _vehicle_lanes(edge: Any) -> list[Any]:
    return [l for l in edge.getLanes() if _is_vehicle_lane(l)]


def _capacity(edges: list[Any]) -> float:
    return max(1.0, sum(e.getLength() * max(1, len(_vehicle_lanes(e))) for e in edges) / SLOT_M)


def _pass_through(node: Any, signalised: set[str]) -> bool:
    """A node that only joins two parts of the same directed block (e.g. where a turn bay opens)."""
    if node.getID() in signalised or node.getType() == "dead_end" or len(node.getIncoming()) != 1 or len(node.getOutgoing()) != 1:
        return False
    a, b = node.getIncoming()[0].getShape(), node.getOutgoing()[0].getShape()
    if len(a) < 2 or len(b) < 2:
        return True
    # the block continues in the same direction; a fringe turn-around (exit feeding the entry beside it) reverses it
    return (a[-1][0] - a[-2][0]) * (b[1][0] - b[0][0]) + (a[-1][1] - a[-2][1]) * (b[1][1] - b[0][1]) > 0


def _reverse(a: Any, b: Any) -> bool:
    """``b`` is the opposite direction of ``a`` (a U-turn at the node between them)."""
    return a.getFromNode().getID() == b.getToNode().getID() and a.getToNode().getID() == b.getFromNode().getID()


def _chain_up(edge: Any, signalised: set[str]) -> list[Any]:
    out = [edge]
    while _pass_through(out[-1].getFromNode(), signalised) and len(out) < 8:
        prev = out[-1].getFromNode().getIncoming()[0]
        if _reverse(prev, out[-1]):  # fringe turn-around: the block ends here
            break
        out.append(prev)
    return out


def _chain_down(edge: Any, signalised: set[str]) -> list[Any]:
    out = [edge]
    while _pass_through(out[-1].getToNode(), signalised) and len(out) < 8:
        nxt = out[-1].getToNode().getOutgoing()[0]
        if _reverse(out[-1], nxt):
            break
        out.append(nxt)
    return out


def _phase_names(net_path: Path) -> dict[str, list[str | None]]:
    names: dict[str, list[str | None]] = {}
    for _ev, el in ET.iterparse(str(net_path), events=("end",)):
        if el.tag == "tlLogic":
            names.setdefault(el.get("id", ""), [p.get("name") for p in el.iter("phase")])
            el.clear()
    return names


def build_topology(net_path: Path | str) -> Topology:
    return _build(str(Path(net_path).resolve()), Path(net_path).stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _build(net_path: str, _mtime: int) -> Topology:
    import sumolib

    path = Path(net_path)
    net = sumolib.net.readNet(net_path, withPrograms=True, withInternal=False)
    names = _phase_names(path)
    try:
        designs = read_signal_designs(path)
    except Exception:  # noqa: BLE001 - a network without the district's edge metadata still gets a topology
        designs = {}
    tls_ids = sorted(t.getID() for t in net.getTrafficLights())
    signalised = set(tls_ids)

    raw: list[dict[str, Any]] = []
    signals_raw: dict[str, dict[str, Any]] = {}
    for tid in tls_ids:
        tls = net.getTLS(tid)
        prog = next(iter(tls.getPrograms().values()))
        states = [p.state for p in prog.getPhases()]
        durs = [float(p.duration) for p in prog.getPhases()]
        per_edge: dict[str, dict[str, Any]] = {}
        vehicle_links: set[int] = set()
        for in_lane, out_lane, idx in tls.getConnections():
            if not _is_vehicle_lane(in_lane):
                continue
            e = in_lane.getEdge()
            d = per_edge.setdefault(e.getID(), {"edge": e, "lanes": [], "links": [], "to": []})
            if in_lane.getID() not in d["lanes"]:
                d["lanes"].append(in_lane.getID())
            d["links"].append(int(idx))
            vehicle_links.add(int(idx))
            if out_lane.getEdge().getID() not in d["to"]:
                d["to"].append(out_lane.getEdge().getID())
        n_links = max((len(s) for s in states), default=0)
        crossings = frozenset(i for i in range(n_links) if i not in vehicle_links)  # pedestrian crossings and bicycle-only lanes
        nm = names.get(tid) or []
        if len(nm) == len(states) and all(n in NAME_ROLES for n in nm):
            roles = [NAME_ROLES[n] for n in nm]  # type: ignore[index]
        else:
            roles = phase_roles(states, infer_crossings(states) - vehicle_links)
        # a re-timable green: a vehicle green that is not a clearance, amber or protected-left interval
        green = tuple(i for i, (r, s) in enumerate(zip(roles, states)) if r == GREEN and any(s[k] in "Gg" for k in vehicle_links if k < len(s)))
        kind = designs[tid].kind if tid in designs and designs[tid].links else ""
        min_green = TYPE_PLANS[kind].min_green if kind in TYPE_PLANS else DEFAULT_MIN_GREEN_S
        signals_raw[tid] = {"kind": kind, "phases": tuple(Phase(i, durs[i], states[i], roles[i]) for i in range(len(states))), "green": green,
                            "min_green": float(min_green), "crossings": crossings, "approaches": []}
        for eid in sorted(per_edge):
            d = per_edge[eid]
            served = tuple(i for i in green if any(k < len(states[i]) and states[i][k] in "Gg" for k in d["links"]))
            raw.append({"tls": tid, "edge": d["edge"], "lanes": tuple(sorted(d["lanes"])), "links": tuple(sorted(set(d["links"]))), "green": served, "to": tuple(d["to"])})

    by_edge = {r["edge"].getID(): i for i, r in enumerate(raw)}
    approaches: list[Approach] = []
    used: set[str] = set()
    for i, r in enumerate(raw):
        chain = _chain_up(r["edge"], signalised)
        head = chain[-1]
        ups = [u for u in head.getIncoming() if not _reverse(u, head)]
        ups = [u for u in ups if u.getFromNode().getID() != r["tls"]]  # not the opposite direction of this very block
        from_id = head.getFromNode().getID()
        up_tls = from_id if from_id in signalised else None
        upstream = tuple(sorted(by_edge[u.getID()] for u in ups if u.getID() in by_edge and raw[by_edge[u.getID()]]["tls"] == up_tls))
        groups, caps, down = [], [], []
        for to in r["to"]:
            dc = _chain_down(net.getEdge(to), signalised)
            groups.append(tuple(e.getID() for e in dc))
            caps.append(_capacity(dc))
            if dc[-1].getID() in by_edge:
                down.append(by_edge[dc[-1].getID()])
        a = Approach(index=i, tls=r["tls"], edge=r["edge"].getID(), edges=tuple(e.getID() for e in chain), lanes=r["lanes"], links=r["links"],
                     green_phases=r["green"], to_edges=r["to"], down_groups=tuple(groups), down_caps=tuple(caps), downstream=tuple(sorted(set(down))),
                     up_edges=tuple(sorted(u.getID() for u in ups)), upstream=upstream, upstream_tls=up_tls, is_entry=up_tls is None,
                     capacity=_capacity(chain))
        approaches.append(a)
        signals_raw[r["tls"]]["approaches"].append(i)
        used.update(a.edges)
        used.update(a.up_edges)
        for g in a.down_groups:
            used.update(g)

    signals = {tid: Signal(tid, s["kind"], s["phases"], s["green"], s["min_green"], MAX_GREEN_S, tuple(s["approaches"]), s["crossings"]) for tid, s in signals_raw.items()}
    structure = [[tid, [(p.state, p.role) for p in s.phases], [(approaches[i].edge, approaches[i].edges, approaches[i].links, approaches[i].to_edges, approaches[i].up_edges) for i in s.approaches]] for tid, s in sorted(signals.items())]
    digest = hashlib.sha256(json.dumps(structure, separators=(",", ":")).encode()).hexdigest()[:16]
    stem = path.name.split(".")[0]
    return Topology(network_id=f"{stem}:{digest}", signals=signals, approaches=tuple(approaches), by_edge=by_edge, edges=tuple(sorted(used)))
