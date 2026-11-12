"""Network-agnostic machine-learning layer for signal control (topology, features, forecaster, policies).

Nothing in this package assumes a grid, compass directions or a fixed number of approaches: everything is derived from the SUMO
network file. Training-only dependencies (scikit-learn, torch, ...) are listed in ``backend/requirements-ml.txt`` and are imported
lazily, so the API process does not need them.
"""
