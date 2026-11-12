"""ML features: the recorded matrices of a short real run, and the feature hash."""

import numpy as np

from flowstate.ml import features
from flowstate.ml.features import F, FEATURES, SAMPLE_S, feature_hash
from flowstate.ml.policies import RecorderPolicy
from helpers import make_engine, run_until


def _record(until: float = 80.0):
    eng = make_engine("rush_hour", 3, {"duration": 5}, label="ml_features")
    eng.start()
    rec = RecorderPolicy()
    eng.signals.policies.extend([rec])
    try:
        run_until(eng, until)
    finally:
        eng.close()
    return rec


def test_recorded_matrices_are_well_formed():
    rec = _record()
    topo = rec.topo
    assert len(FEATURES) == 19 and len(rec.rows) >= 12
    assert np.all(np.diff(rec.times) > SAMPLE_S - 0.3) and np.all(np.diff(rec.times) < SAMPLE_S + 0.3)
    cap = np.array([a.capacity for a in topo.approaches])
    entry = np.array([1.0 if a.is_entry else 0.0 for a in topo.approaches])
    for X in rec.rows:
        assert X.shape == (len(topo.approaches), 19)
        assert np.isfinite(X).all()
        assert (X[:, F["queue"]] <= X[:, F["count"]] + 1e-6).all()
        assert np.allclose(X[:, F["moving"]], X[:, F["count"]] - X[:, F["queue"]])
        assert np.allclose(X[:, F["capacity"]], cap)
        assert np.array_equal(X[:, F["is_entry"]], entry)
        assert set(np.unique(X[:, F["is_green"]])) <= {0.0, 1.0}
        assert ((X[:, F["down_occ"]] >= 0) & (X[:, F["down_occ"]] <= 1)).all()
    assert max(X[:, F["count"]].max() for X in rec.rows) > 0  # traffic was actually seen


def test_lag_columns_are_earlier_queues():
    rec = _record()
    q = [X[:, F["queue"]] for X in rec.rows]
    checked = 0
    for i in range(6, len(rec.rows)):
        for col, lag in (("q_lag10", 2), ("q_lag20", 4), ("q_lag30", 6)):
            assert np.array_equal(rec.rows[i][:, F[col]], q[i - lag]), (col, i)
            checked += 1
        assert np.array_equal(rec.rows[i][:, F["count_lag30"]], rec.rows[i - 6][:, F["count"]])
    assert checked >= 3


def test_feature_hash_is_stable_and_order_sensitive(monkeypatch):
    h = feature_hash()
    assert h == feature_hash()
    assert len(h) == 16 and int(h, 16) >= 0
    monkeypatch.setattr(features, "FEATURES", tuple(reversed(FEATURES)))
    assert features.feature_hash() != h
