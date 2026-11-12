"""ML command line (run from ``backend/``; needs ``requirements-ml.txt``).

    python -m flowstate.ml.cli topology
    python -m flowstate.ml.cli dataset  [--seeds N] [--workers W]     # data/ml/district_dataset.npz (+ dataset_manifest.json)
    python -m flowstate.ml.cli train    [--max-iter N]                # artifacts/ml/forecaster/
    python -m flowstate.ml.cli evaluate [--seeds N] [--workers W] [--arms a,b] [--split test|val]

``--seeds N`` uses the first N seeds of each split (smoke runs); without it the full split ranges are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flowstate.ml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("topology")
    d = sub.add_parser("dataset")
    d.add_argument("--seeds", type=int, default=None)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--out", default=None)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default=None)
    t.add_argument("--out", default=None)
    t.add_argument("--max-iter", type=int, default=300)
    e = sub.add_parser("evaluate")
    e.add_argument("--seeds", type=int, default=None)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--arms", default=None)
    e.add_argument("--split", default="test", choices=("test", "val"))
    e.add_argument("--scenarios", default=None)
    e.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "topology":
        from ..api.service import NETWORKS_DIR
        from .features import describe
        from .topology import build_topology

        print(json.dumps({**build_topology(NETWORKS_DIR / "district.net.xml").describe(), **describe()}, indent=1))
        return 0
    if a.cmd == "dataset":
        from .dataset import generate

        man = generate(a.seeds, a.workers, Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in man.items() if k != "episodes"} | {"episodes": len(man["episodes"])}, indent=1))
        return 0
    if a.cmd == "train":
        from .dataset import data_dir
        from .episodes import model_dir
        from .forecast import train

        meta = train(Path(a.dataset) if a.dataset else data_dir() / "district_dataset.npz", Path(a.out) if a.out else model_dir() / "forecaster", max_iter=a.max_iter)
        print(json.dumps({k: meta[k] for k in ("rows", "chosen", "test", "inference_ms_per_decision", "train_wall_s") if k in meta}, indent=1))
        return 0
    if a.cmd == "evaluate":
        from .evaluate import run, table

        doc = run(a.arms.split(",") if a.arms else None, a.seeds, split=a.split, scenarios=a.scenarios.split(",") if a.scenarios else None, workers=a.workers,
                  out=Path(a.out) if a.out else None)
        print(table(doc))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
