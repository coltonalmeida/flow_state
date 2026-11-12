"""ML forecaster: training, artifact manifests and refusal to load a model for another network or feature set."""

import json
import shutil

import numpy as np
import pytest

forecast = pytest.importorskip("flowstate.ml.forecast")
pytest.importorskip("sklearn")

from flowstate.ml.features import F, FEATURES, feature_hash  # noqa: E402

A, T, LEAD30, LEAD60 = 7, 80, 6, 12
NETWORK_ID = "toy:0123456789abcdef"
SPLITS = {"train": 5, "val": 2, "test": 2}  # episodes per split


def _episode(rng):
    cap = rng.uniform(8, 30, size=A).astype(np.float32)
    q = np.zeros((T, A), dtype=np.float32)
    q[0] = rng.uniform(0, 5, size=A)
    for t in range(1, T):
        q[t] = np.clip(q[t - 1] + rng.normal(0, 1.5, size=A), 0, cap)
    X = np.zeros((T, A, len(FEATURES)), dtype=np.float32)
    X[:, :, F["queue"]] = q
    X[:, :, F["count"]] = q + rng.integers(0, 4, size=(T, A))
    X[:, :, F["moving"]] = X[:, :, F["count"]] - q
    X[:, :, F["capacity"]] = cap
    X[:, :, F["is_green"]] = rng.integers(0, 2, size=(T, A))
    for name in ("q_lag10", "q_lag20", "q_lag30"):
        X[:, :, F[name]] = np.roll(q, {"q_lag10": 2, "q_lag20": 4, "q_lag30": 6}[name], axis=0)
    return X


def _rows(X):
    feats = X[: T - LEAD60].reshape(-1, X.shape[2])
    q = X[:, :, F["queue"]]
    y = np.stack([q[LEAD30 : T - LEAD60 + LEAD30], q[LEAD60:]], axis=-1).reshape(-1, 2)
    a = np.tile(np.arange(A), T - LEAD60)
    return feats, y, a


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("forecast")
    rng = np.random.default_rng(0)
    arrays, episodes, ep = {}, [], 0
    for split, n in SPLITS.items():
        parts = []
        for _ in range(n):
            X, y, a = _rows(_episode(rng))
            parts.append((X, y, a, np.full(len(a), ep)))
            episodes.append({"split": split, "scenario": "rush_hour", "seed": 100 + ep, "arm": "random", "rows": len(a)})
            ep += 1
        for k, name in enumerate("Xyae"):
            arrays[f"{split}_{name}"] = np.concatenate([p[k] for p in parts]).astype(np.float32 if name in "Xy" else np.int64)
    ds = root / "dataset.npz"
    np.savez(ds, **arrays)
    (root / "dataset_manifest.json").write_text(json.dumps({"network_id": NETWORK_ID, "feature_hash": feature_hash(), "features": list(FEATURES),
                                                            "heldout_scenario": None, "episodes": episodes}), encoding="utf-8")
    out = root / "model"
    result = forecast.train(ds, out, max_iter=20)
    return out, result, arrays


def test_train_writes_all_artifacts(trained):
    out, result, _ = trained
    assert isinstance(result, dict)
    for name in ("forecaster.pkl", "MODEL_MANIFEST.json", "calibration.json", "forecast_eval.json"):
        assert (out / name).is_file(), name
    manifest = json.loads((out / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    assert NETWORK_ID in json.dumps(manifest)
    assert feature_hash() in json.dumps(manifest)


def test_load_predict_and_check(trained):
    out, _, arrays = trained
    fc = forecast.Forecaster.load(out, network_id=NETWORK_ID, feature_hash=feature_hash())
    X = arrays["test_X"]
    p30, p60 = fc.predict(X)
    cap = X[:, F["capacity"]]
    for p in (p30, p60):
        assert p.shape == (len(X),)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= cap + 1e-6).all()
    assert fc.check(X) < 1e-6
    forecast.Forecaster.load(out, network_id=NETWORK_ID)  # feature_hash is optional


def test_load_refuses_mismatches(trained, tmp_path):
    out, _, _ = trained
    with pytest.raises(forecast.ModelArtifactError):
        forecast.Forecaster.load(out, network_id="other:ffffffffffffffff")
    with pytest.raises(forecast.ModelArtifactError):
        forecast.Forecaster.load(out, network_id=NETWORK_ID, feature_hash="0" * 16)
    bad = tmp_path / "tampered"
    shutil.copytree(out, bad)
    with open(bad / "forecaster.pkl", "ab") as f:
        f.write(b"\x00tampered")
    with pytest.raises(forecast.ModelArtifactError):
        forecast.Forecaster.load(bad, network_id=NETWORK_ID, feature_hash=feature_hash())


def test_rows_from_pairs_features_with_future_queues():
    a_n, t_n = 5, 40
    X = np.zeros((t_n, a_n, len(FEATURES)), dtype=np.float32)
    for t in range(t_n):
        for a in range(a_n):
            X[t, a, F["queue"]] = t * 100 + a  # encodes (time, approach)
    X[:, :, F["capacity"]] = 1e6
    feats, y, idx = forecast.rows_from(X)
    assert feats.shape[1] == len(FEATURES) and y.shape == (len(feats), 2) and idx.shape == (len(feats),)
    assert len(feats) == (t_n - LEAD60) * a_n
    t = (feats[:, F["queue"]] // 100).astype(int)
    a = (feats[:, F["queue"]] % 100).astype(int)
    assert np.array_equal(idx, a)
    assert np.array_equal(y[:, 0], (t + LEAD30) * 100 + a)
    assert np.array_equal(y[:, 1], (t + LEAD60) * 100 + a)
