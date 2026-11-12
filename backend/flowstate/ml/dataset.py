"""Forecasting dataset from real SUMO episodes on the district network.

Every episode is a main scenario run through :class:`~flowstate.sim.engine.Engine` with a passive
:class:`~flowstate.ml.policies.RecorderPolicy`. Training episodes are recorded under three controllers (fixed-time, the
queue-responsive rules, and random safe timing) so the model does not only learn one controller's timing.

Leakage control: train / validation / test use *disjoint seed ranges* (1000-1199, 2000-2039, 3000-3079), a row never leaves
its episode's split, and one scenario (``stadium``) appears only in test. By default the first 60 / 16 / 30 seeds of each range
are recorded (``DATASET_SEEDS``), and a fixed random fraction of each episode's rows is kept (drawn from a generator seeded by the
episode, so the dataset is reproducible).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from .episodes import root, run_episode
from .features import FEATURES, SAMPLE_S, feature_hash
from .forecast import rows_from

TRAIN_SCENARIOS = ("rush_hour", "construction", "accident", "weather", "transit")
HELDOUT_SCENARIO = "stadium"
SEEDS = {"train": (1000, 1199), "val": (2000, 2039), "test": (3000, 3079)}
SPLIT_ARMS = {"train": ("fixed_time", "queue_responsive", "random_timing"), "val": ("fixed_time", "queue_responsive"), "test": ("fixed_time", "queue_responsive")}
# Episodes recorded per split by default: the first N seeds of each range. A 15-minute episode takes 20-55 s of CPU, and the forecaster's
# validation error stops improving long before all 200 training seeds are used; the remaining seeds of each range stay unused by the
# dataset (controller evaluation uses the full test range). ``generate(n_seeds=...)`` overrides this for every split.
DATASET_SEEDS = {"train": 60, "val": 16, "test": 30}
ROW_FRACTION = {"train": 0.5, "val": 0.5, "test": 0.3}


def data_dir() -> Path:
    return root() / "data" / "ml"


def split_seeds(split: str, n: int | None = None) -> list[int]:
    """The split's seeds; ``n`` takes the first n (smoke runs), never crossing into another split's range."""
    lo, hi = SEEDS[split]
    seeds = list(range(lo, hi + 1))
    return seeds if n is None else seeds[: max(1, min(n, len(seeds)))]


def jobs_for(n_seeds: int | None = None) -> list[tuple[str, str, int, str]]:
    out = []
    for split in ("train", "val", "test"):
        scenarios = TRAIN_SCENARIOS + ((HELDOUT_SCENARIO,) if split == "test" else ())
        for sc in scenarios:
            for seed in split_seeds(split, n_seeds if n_seeds is not None else DATASET_SEEDS[split]):
                for arm in SPLIT_ARMS[split]:
                    out.append((split, sc, seed, arm))
    return out


def _episode(job: tuple[str, str, int, str]) -> dict[str, Any]:
    split, scenario, seed, arm = job
    ep = run_episode(arm, scenario, seed, record=True)
    f, y, a = rows_from(ep["X"])
    frac = ROW_FRACTION[split]
    if frac < 1.0 and len(y):
        rng = np.random.default_rng([seed, sum(map(ord, scenario + arm))])
        keep = np.sort(rng.choice(len(y), size=max(1, int(round(len(y) * frac))), replace=False))
        f, y, a = f[keep], y[keep], a[keep]
    return {"split": split, "scenario": scenario, "seed": seed, "arm": arm, "X": f.astype(np.float32), "y": y.astype(np.float32), "a": a.astype(np.int16),
            "network_id": ep["network_id"], "approaches": int(ep["X"].shape[1]) if ep["X"].ndim == 3 else 0,
            "metrics": {k: ep[k] for k in ("mean_wait_s", "mean_time_loss_s", "teleports", "vehicles", "completed")}}


def generate(n_seeds: int | None = None, workers: int = 4, out: Path | None = None) -> dict[str, Any]:
    out = out or data_dir() / "district_dataset.npz"
    jobs = jobs_for(n_seeds)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        eps = []
        for k, ep in enumerate(pool.imap(_episode, jobs, chunksize=1), 1):
            eps.append(ep)
            if k % 50 == 0 or k == len(jobs):
                print(f"[dataset] {k}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
    net_ids = {e["network_id"] for e in eps}
    if len(net_ids) != 1:
        raise RuntimeError(f"episodes ran on different networks: {net_ids}")
    arrays: dict[str, list[np.ndarray]] = {}
    episodes: list[dict[str, Any]] = []
    for ep in eps:
        e_id = np.full(len(ep["y"]), len(episodes), dtype=np.int32)
        for k, v in (("X", ep["X"]), ("y", ep["y"]), ("a", ep["a"]), ("e", e_id)):
            arrays.setdefault(f"{ep['split']}_{k}", []).append(v)
        episodes.append({"split": ep["split"], "scenario": ep["scenario"], "seed": ep["seed"], "arm": ep["arm"], "rows": int(len(ep["y"])), **ep["metrics"]})
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    seeds_used = {s: sorted({e["seed"] for e in episodes if e["split"] == s}) for s in SEEDS}
    assert not (set(seeds_used["train"]) & set(seeds_used["val"])) and not (set(seeds_used["train"]) & set(seeds_used["test"])) and not (set(seeds_used["val"]) & set(seeds_used["test"]))
    manifest = {"network_id": net_ids.pop(), "feature_hash": feature_hash(), "features": list(FEATURES), "sample_s": SAMPLE_S, "horizons_s": [30, 60],
                "approaches": eps[0]["approaches"], "heldout_scenario": HELDOUT_SCENARIO, "train_scenarios": list(TRAIN_SCENARIOS),
                "seeds": {s: [v[0], v[-1], len(v)] for s, v in seeds_used.items() if v}, "arms": {s: list(a) for s, a in SPLIT_ARMS.items()}, "row_fraction": ROW_FRACTION,
                "rows": {s: int(len(data[f"{s}_y"])) for s in SEEDS if f"{s}_y" in data}, "wall_s": round(time.time() - t0, 1),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "episodes": episodes}
    (out.parent / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
