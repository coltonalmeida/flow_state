"""Queue forecasting: a lightweight CPU model (30 s and 60 s ahead, per approach), baselines, calibrated intervals and a registered artifact.

Target: the number of halting vehicles on an approach 30 s and 60 s after the observation, for any number of approaches.
Features: :data:`flowstate.ml.features.FEATURES`.

Leakage control is the dataset's job (train / val / test are disjoint seed ranges, one optional scenario is held out for test only);
:func:`train` consumes the splits as given and scores every baseline on exactly the same test rows as the model.

The artifact is ``forecaster.pkl`` plus ``MODEL_MANIFEST.json`` (sha256, network id, feature hash, library versions). :meth:`Forecaster.load`
verifies the manifest before unpickling and refuses a model for another network or feature definition.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import F, FEATURES, SAMPLE_S, feature_hash

# Small batches: OpenMP thread start-up costs far more than the trees themselves, and one thread bounds CPU per worker.
# Set before scikit-learn is imported (it is imported lazily below).
os.environ.setdefault("OMP_NUM_THREADS", "1")

H30, H60 = 6, 12  # horizons in samples (SAMPLE_S = 5 s each)
SPILL_FRAC = 0.5  # an approach queued over half its length is a bottleneck / spillback risk
LEVEL = 0.90  # nominal coverage of the calibrated interval
PRED_EDGES = (0.0, 0.10, 0.25, 0.50)  # bins of predicted occupancy (prediction / capacity) for the Mondrian calibration
MIN_BIN = 500  # a calibration bin with fewer validation rows than this falls back to the pooled residuals
CALIBRATION_SCHEMA = "flowstate.calibration/1"
MANIFEST_SCHEMA = "flowstate.model_manifest/1"
Q, CAP = F["queue"], F["capacity"]


class ModelArtifactError(RuntimeError):
    """The model artifact is missing, tampered with, or built for another network / feature definition."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _current_feature_hash() -> str:
    return feature_hash()


# ---------------------------------------------------------------------------------------------------------------- dataset rows
def rows_from(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, targets[30 s, 60 s], approach index) for every sample of one episode ``X`` (T, A, F) that has both future targets inside it."""
    n = X.shape[0] - H60
    if n <= 0:
        return np.zeros((0, X.shape[2])), np.zeros((0, 2)), np.zeros(0, dtype=int)
    feats = X[:n].reshape(-1, X.shape[2])
    y = np.stack([X[H30:H30 + n, :, Q].reshape(-1), X[H60:H60 + n, :, Q].reshape(-1)], axis=1)
    appr = np.tile(np.arange(X.shape[1]), n)
    return feats, y, appr


# ---------------------------------------------------------------------------------------------------------------- model
class FlatTrees:
    """A fitted HistGradientBoostingRegressor (squared error, numeric features, no missing values) as flat arrays.

    scikit-learn walks its trees one Python call per tree, which costs ~15 us x 300 trees for a small batch. Here every tree is
    descended at once with NumPy indexing (depth iterations), which is exact: ``Forecaster.check`` compares against scikit-learn.
    """

    def __init__(self, model: Any):
        trees = [p[0].nodes for p in model._predictors]
        sizes = np.array([len(t) for t in trees])
        off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        n = np.concatenate(trees)
        shift = np.repeat(off, sizes)
        leaf = n["is_leaf"].astype(bool)
        idx = np.arange(len(n), dtype=np.int64)
        # leaves point to themselves, so every row can take the same number of steps without branching
        self.feat = np.where(leaf, 0, n["feature_idx"]).astype(np.int64)
        self.thr = np.where(leaf, np.inf, n["num_threshold"]).astype(np.float64)
        self.left = np.where(leaf, idx, n["left"].astype(np.int64) + shift)
        self.right = np.where(leaf, idx, n["right"].astype(np.int64) + shift)
        self.value = n["value"].astype(np.float64)
        self.roots = off.astype(np.int64)
        self.depth = int(n["depth"].max())
        self.base = float(np.asarray(model._baseline_prediction).ravel()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        flat = X.ravel()
        base_idx = (np.arange(len(X)) * X.shape[1])[:, None]
        node = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        for _ in range(self.depth):
            node = np.where(flat[base_idx + self.feat[node]] <= self.thr[node], self.left[node], self.right[node])
        return self.base + self.value[node].sum(axis=1)


class Forecaster:
    """Two gradient-boosted trees (30 s, 60 s). Predictions are clipped to [0, approach capacity]."""

    manifest: dict[str, Any] | None = None
    sha256: str = ""
    calibration: dict[str, Any] | None = None

    def __init__(self, models: dict[str, Any], meta: dict[str, Any]):
        self.models, self.meta = models, meta
        self.fast = {k: FlatTrees(m) for k, m in models.items()}

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cap = X[:, CAP]
        p30 = np.clip(self.fast["p30"].predict(X), 0.0, cap)
        p60 = np.clip(self.fast["p60"].predict(X), 0.0, cap)
        return p30, p60

    def check(self, X: np.ndarray) -> float:
        """Largest absolute difference between the flat evaluation and scikit-learn's own predict."""
        return max(float(np.abs(self.fast[k].predict(X) - m.predict(X)).max()) for k, m in self.models.items())

    def predict_full(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """p30/p60 plus, when a matching calibration is attached, empirical lo/hi bounds per horizon."""
        p30, p60 = self.predict(X)
        out = {"p30": p30, "p60": p60}
        if self.calibration is not None:
            cap = X[:, CAP]
            for h, p in ((30, p30), (60, p60)):
                out[f"lo{h}"], out[f"hi{h}"] = apply_calibration(self.calibration, h, p, cap)
        return out

    def save(self, out_dir: Path, *, network_id: str, feature_hash: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write ``forecaster.pkl`` and ``MODEL_MANIFEST.json`` into ``out_dir``; returns the manifest."""
        import sklearn

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = pickle.dumps({"models": self.models, "meta": self.meta})  # FlatTrees are rebuilt on load
        (out_dir / "forecaster.pkl").write_bytes(raw)
        manifest: dict[str, Any] = {"schema": MANIFEST_SCHEMA, "artifact": "forecaster.pkl", "artifact_sha256": _sha256(raw), "network_id": network_id,
                                    "feature_hash": feature_hash, "features": list(FEATURES), "horizons_s": [30, 60], "sample_s": SAMPLE_S,
                                    "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
                                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
        (out_dir / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
        self.manifest, self.sha256 = manifest, manifest["artifact_sha256"]
        return manifest

    @classmethod
    def load(cls, model_dir: Path, *, network_id: str, feature_hash: str | None = None) -> "Forecaster":
        """Load a registered artifact, refusing one that is unregistered, altered, or built for another network or feature set."""
        model_dir = Path(model_dir)
        mf_path = model_dir / "MODEL_MANIFEST.json"
        if not mf_path.exists():
            raise ModelArtifactError(f"MODEL_MANIFEST.json not found in {model_dir}; refusing an unregistered model")
        try:
            manifest = json.loads(mf_path.read_text())
        except ValueError as exc:
            raise ModelArtifactError(f"MODEL_MANIFEST.json in {model_dir} is not valid JSON: {exc}") from exc
        art = model_dir / str(manifest.get("artifact", "forecaster.pkl"))
        if not art.exists():
            raise ModelArtifactError(f"model artifact {art.name} not found in {model_dir}")
        raw = art.read_bytes()
        sha = _sha256(raw)
        if sha != manifest.get("artifact_sha256"):  # never unpickle a file that is not the registered artifact
            raise ModelArtifactError(f"{art.name} sha256 {sha[:12]}... does not match MODEL_MANIFEST.json ({str(manifest.get('artifact_sha256'))[:12]}...)")
        if manifest.get("network_id") != network_id:
            raise ModelArtifactError(f"model was trained for network {manifest.get('network_id')!r}, not {network_id!r}")
        want = feature_hash if feature_hash is not None else _current_feature_hash()
        if manifest.get("feature_hash") != want:
            raise ModelArtifactError(f"model feature_hash {manifest.get('feature_hash')!r} does not match expected {want!r}")
        d = pickle.loads(raw)  # our own artifact, written by save() and hash-checked above
        fc = cls(d["models"], d["meta"])
        fc.manifest, fc.sha256, fc.calibration = manifest, sha, None
        cal_path = model_dir / "calibration.json"
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except ValueError:
                cal = {}
            if cal.get("model_sha256") == sha:  # a calibration for another artifact is ignored, never guessed
                fc.calibration = cal
        return fc


# ---------------------------------------------------------------------------------------------------------------- calibration
def _bin(occ: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges[1:]), occ, side="right")


def _f1_at(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = int((truth & pred).sum())
    prec, rec = tp / max(1, int(pred.sum())), tp / max(1, int(truth.sum()))
    return 2 * prec * rec / max(1e-9, prec + rec)


def fit_calibration(y_true: np.ndarray, preds: tuple[np.ndarray, np.ndarray], cap: np.ndarray, level: float = LEVEL, model_sha256: str | None = None,
                    fitted_on: str = "validation split") -> dict[str, Any]:
    """Mondrian split-conformal style intervals from validation rows.

    ``y_true`` is (n, 2) (30 s, 60 s), ``preds`` the (p30, p60) point forecasts, ``cap`` the approach capacity per row. For each horizon the
    signed residuals ``y - p`` are binned by predicted occupancy ``p / cap`` (edges :data:`PRED_EDGES`); in each bin the (1-level)/2 and
    1-(1-level)/2 empirical quantiles are the lo / hi offsets (a bin with fewer than :data:`MIN_BIN` rows uses the pooled quantiles). Also tunes
    an alert threshold tau (share of capacity, 0.20-0.60) maximising F1 for "queue >= SPILL_FRAC * capacity at t+h" on the same rows.
    """
    cap = np.maximum(np.asarray(cap, dtype=np.float64), 1e-9)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out: dict[str, Any] = {"schema": CALIBRATION_SCHEMA, "method": "Mondrian signed-residual quantiles on validation rows", "level": level,
                           "pred_occupancy_edges": list(PRED_EDGES), "fitted_on": fitted_on, "rows_val": int(len(y_true)), "model_sha256": model_sha256, "horizons": {}}
    for j, h in enumerate((30, 60)):
        p = preds[j]
        res = y_true[:, j] - p
        bins = _bin(p / cap, PRED_EDGES)
        pooled = (float(np.quantile(res, lo_q)), float(np.quantile(res, hi_q))) if len(res) else (0.0, 0.0)
        cells = []
        for k in range(len(PRED_EDGES)):
            r = res[bins == k]
            ok = len(r) >= MIN_BIN
            cells.append({"n": int(len(r)), "lo": float(np.quantile(r, lo_q)) if ok else pooled[0], "hi": float(np.quantile(r, hi_q)) if ok else pooled[1],
                          "pooled_fallback": not ok})
        truth = y_true[:, j] >= SPILL_FRAC * cap
        best = max((_f1_at(truth, p >= tau * cap), float(tau)) for tau in np.arange(0.20, 0.601, 0.025))
        out["horizons"][str(h)] = {"pooled": {"lo": pooled[0], "hi": pooled[1]}, "bins": cells,
                                   "alert": {"tau": round(best[1], 3), "val_f1": round(best[0], 3), "tuned_on": "validation split"}}
    return out


def apply_calibration(cal: dict[str, Any], h: int, pred: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) bounds for point forecasts ``pred`` at horizon ``h`` (30 or 60), clipped to [0, capacity]."""
    cells = cal["horizons"][str(h)]["bins"]
    k = _bin(pred / np.maximum(cap, 1e-9), tuple(cal["pred_occupancy_edges"]))
    lo = np.array([c["lo"] for c in cells])[k]
    hi = np.array([c["hi"] for c in cells])[k]
    return np.clip(pred + lo, 0.0, cap), np.clip(pred + hi, 0.0, cap)


# ---------------------------------------------------------------------------------------------------------------- training
def _baselines(X: np.ndarray, h_s: int, mean_by_appr: np.ndarray, appr: np.ndarray) -> dict[str, np.ndarray]:
    q = X[:, Q]
    lag30 = X[:, F["q_lag30"]]
    return {"persistence": q, "linear_trend": np.maximum(0.0, q + (q - lag30) * h_s / 30.0), "approach_mean": mean_by_appr[appr]}


def _scores(y: np.ndarray, p: np.ndarray, cap: np.ndarray) -> dict[str, float]:
    err = p - y
    ss = float(((y - y.mean()) ** 2).sum())
    hit_t, hit_p = y >= SPILL_FRAC * cap, p >= SPILL_FRAC * cap
    tp = int((hit_t & hit_p).sum())
    prec = tp / max(1, int(hit_p.sum()))
    rec = tp / max(1, int(hit_t.sum()))
    return {"mae": round(float(np.abs(err).mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "r2": round(1.0 - float((err ** 2).sum()) / ss, 3) if ss else 0.0,
            "spill_precision": round(prec, 3), "spill_recall": round(rec, 3), "spill_f1": round(2 * prec * rec / max(1e-9, prec + rec), 3)}


def _importance(model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.abs(model.predict(X) - y).mean()
    out = []
    for k in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        out.append(np.abs(model.predict(Xp) - y).mean() - base)
    return np.asarray(out)


def train(dataset_path: Path, out_dir: Path, *, max_iter: int = 300, grid: tuple = ((0.05, 31), (0.1, 31), (0.1, 63))) -> dict[str, Any]:
    """Fit the two models on the dataset's train split (grid chosen on validation MAE only), score on test, calibrate on validation, save to ``out_dir``."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    dataset_path, out_dir = Path(dataset_path), Path(out_dir)
    dm_path = dataset_path.with_name("dataset_manifest.json")
    if not dm_path.exists():
        raise ModelArtifactError(f"dataset manifest not found next to {dataset_path}")
    dm = json.loads(dm_path.read_text())
    if dm.get("feature_hash") != feature_hash():
        raise ModelArtifactError(f"dataset feature_hash {dm.get('feature_hash')!r} differs from the current feature definition {feature_hash()!r}; regenerate the dataset")
    d = np.load(dataset_path)
    Xtr, ytr, atr = d["train_X"], d["train_y"], d["train_a"]
    Xva, yva = d["val_X"], d["val_y"]
    Xte, yte, ate = d["test_X"], d["test_y"], d["test_a"]
    cap_te = Xte[:, CAP]
    n_appr = int(atr.max()) + 1
    glob_mean = float(ytr[:, 0].mean())
    mean_by_appr = np.array([ytr[atr == k, 0].mean() if (atr == k).any() else glob_mean for k in range(max(n_appr, int(ate.max()) + 1))])
    t0 = time.time()
    models, val, chosen = {}, {}, {}
    for j, name in enumerate(("p30", "p60")):
        best = None
        for lr, leaves in grid:  # small grid, selected on validation seeds only
            m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.1, random_state=0)
            m.fit(Xtr, ytr[:, j])
            mae = float(np.abs(m.predict(Xva) - yva[:, j]).mean())
            val[f"{name}:lr={lr},leaves={leaves}"] = round(mae, 4)
            if best is None or mae < best[0]:
                best = (mae, m, {"learning_rate": lr, "max_leaf_nodes": leaves, "iterations": int(m.n_iter_)})
        models[name], chosen[name] = best[1], best[2]
    train_s = time.time() - t0
    ridge = {name: Ridge(alpha=1.0).fit(Xtr, ytr[:, j]) for j, name in enumerate(("p30", "p60"))}
    fc = Forecaster(models, {})
    p30, p60 = fc.predict(Xte)
    results: dict[str, Any] = {}
    for j, (name, h, p) in enumerate((("p30", 30, p30), ("p60", 60, p60))):
        r = {"model": _scores(yte[:, j], p, cap_te), "ridge": _scores(yte[:, j], np.clip(ridge[name].predict(Xte), 0, cap_te), cap_te)}
        for bname, bp in _baselines(Xte, h, mean_by_appr, ate).items():
            r[bname] = _scores(yte[:, j], bp, cap_te)
        r["improvement_vs_persistence_mae_pct"] = round(100.0 * (1 - r["model"]["mae"] / r["persistence"]["mae"]), 1)
        results[name] = r
    # held-out scenario on its own (never seen in training)
    held = dm.get("heldout_scenario")
    if held:
        scen = np.array([ep["scenario"] == held for ep in dm["episodes"]])
        mask = scen[d["test_e"]]
        if mask.any():
            results["heldout_scenario_p30"] = {"scenario": held, "rows": int(mask.sum()), "model": _scores(yte[mask, 0], p30[mask], cap_te[mask]),
                                               "persistence": _scores(yte[mask, 0], Xte[mask, Q], cap_te[mask])}
    exact = fc.check(Xte[:20000])
    assert exact < 1e-6, f"flat tree evaluation differs from scikit-learn by {exact}"
    lat = []
    for k in range(200):  # one control decision = every approach x both horizons
        x = Xte[(k * n_appr) % max(1, len(Xte) - n_appr):][:n_appr]
        t1 = time.perf_counter()
        fc.predict(x)
        lat.append((time.perf_counter() - t1) * 1000)
    imp = {f: round(float(v), 4) for f, v in zip(FEATURES, _importance(models["p30"], Xte[:4000], yte[:4000, 0]))}
    meta = {"kind": "HistGradientBoostingRegressor x2 (30 s, 60 s)", "network_id": dm["network_id"], "features": list(FEATURES), "chosen": chosen, "validation_mae": val,
            "rows": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))}, "approaches": n_appr, "train_wall_s": round(train_s, 1),
            "test": results, "inference_ms_per_decision": {"mean": round(float(np.mean(lat)), 3), "p95": round(float(np.percentile(lat, 95)), 3)},
            "flat_vs_sklearn_max_abs_diff": exact, "permutation_importance_p30": imp, "heldout_scenario": held,
            "split": "by episode (disjoint seed ranges); the held-out scenario, if any, appears only in test",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    fc.meta = meta
    version = "1.0." + str(int(time.time()))
    manifest = fc.save(out_dir, network_id=dm["network_id"], feature_hash=dm["feature_hash"], extra={"version": version, "kind": meta["kind"], "rows": meta["rows"]})
    pv30, pv60 = fc.predict(Xva)
    cal = fit_calibration(yva, (pv30, pv60), Xva[:, CAP], model_sha256=manifest["artifact_sha256"], fitted_on="validation split (never used for the test numbers)")
    (out_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    fc.calibration = cal
    (out_dir / "forecast_eval.json").write_text(json.dumps(meta, indent=1))
    return meta
