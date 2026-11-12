"""ML topology: signal and approach structure read from the district net file alone (no simulation)."""

import pytest

from flowstate.api.service import NETWORKS_DIR
from flowstate.ml import features as features_mod
from flowstate.ml import topology as topology_mod
from flowstate.ml.topology import build_topology

NET = NETWORKS_DIR / "district.net.xml"


@pytest.fixture(scope="module")
def topo():
    return build_topology(NET)


@pytest.fixture(scope="module")
def net():
    import sumolib

    return sumolib.net.readNet(str(NET), withPrograms=True, withInternal=False)


def test_signal_and_approach_counts(topo):
    assert len(topo.signals) == 22
    assert len(topo.signals["n4_3"].approaches) == 3
    assert len(topo.signals["n4_7"].approaches) == 3
    for tid, sig in topo.signals.items():
        assert 2 <= len(sig.approaches) <= 4, tid
        assert len(topo.of(tid)) == len(sig.approaches)


def test_every_approach_maps_to_a_green_phase(topo):
    for a in topo.approaches:
        sig = topo.signals[a.tls]
        assert len(a.green_phases) >= 1, a.edge
        assert set(a.green_phases) <= set(sig.green_phases), a.edge
    for tid, sig in topo.signals.items():
        assert sig.green_phases, tid
        links = {k for i in sig.approaches for k in topo.approaches[i].links}
        for p in sig.green_phases:
            ph = sig.phases[p]
            assert ph.role == "green", (tid, p)
            assert any(k < len(ph.state) and ph.state[k] in "Gg" for k in links), (tid, p)


def test_approaches_are_keyed_by_real_incoming_edges(topo, net):
    for edge, i in topo.by_edge.items():
        a = topo.approaches[i]
        assert a.edge == edge and a.index == i
        e = net.getEdge(edge)
        assert e.getToNode().getID() == a.tls  # the edge ends at its signal's node
        assert all(net.getEdge(x) for x in a.edges)
    assert len(topo.by_edge) == len(topo.approaches)


def test_no_compass_assumptions_in_source():
    for mod in (topology_mod, features_mod):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "OPPOSITE" not in src
        assert "DIRECTIONS" not in src


def test_min_green_and_network_id(topo):
    assert {s.min_green for s in topo.signals.values()} <= {8.0, 10.0, 12.0}
    assert all(s.max_green == 60.0 for s in topo.signals.values())
    assert topo.network_id.startswith("district:")
    other = build_topology(str(NET))
    assert other.network_id == topo.network_id


def test_structural_sanity(topo):
    n = len(topo.approaches)
    for a in topo.approaches:
        assert a.capacity > 0
        for k in a.downstream:
            assert 0 <= k < n
            assert topo.approaches[k].tls != a.tls, (a.edge, k)
        for k in a.upstream:
            assert 0 <= k < n
            assert topo.approaches[k].tls != a.tls, (a.edge, k)
        if a.is_entry:
            assert not a.upstream, a.edge
