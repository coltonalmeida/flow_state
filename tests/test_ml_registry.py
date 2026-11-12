"""Controller registry: status comes from the gate and the model files, and nothing unverified becomes the default."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from flowstate.ml import registry


def _write_rl(d, sha_ok=True):
    (d / "rl").mkdir(parents=True)
    (d / "rl" / "policy.npz").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest() if sha_ok else "0" * 64
    (d / "rl" / "RL_MANIFEST.json").write_text(json.dumps({"sha256": sha, "network_id": "district:x"}))


def _by_id(doc):
    return {c["id"]: c for c in doc["controllers"]}


def test_missing_models_are_unavailable_and_fall_back(tmp_path):
    doc = registry.controllers(tmp_path)
    c = _by_id(doc)
    assert doc["default_controller"] == "queue_responsive" and c["queue_responsive"]["status"] == "default" and c["fixed_time"]["status"] == "baseline"
    assert c["rl_ppo"]["status"] == "unavailable" and c["ml_forecast"]["status"] == "unavailable"
    r = registry.resolve("rl_ppo", tmp_path)
    assert r["used"] == "queue_responsive" and r["fallback"]
    assert registry.resolve(None, tmp_path)["used"] == "queue_responsive"
    with pytest.raises(ValueError):
        registry.resolve("nope", tmp_path)


def test_status_follows_the_gate_and_the_manifest_hash(tmp_path):
    _write_rl(tmp_path)
    assert _by_id(registry.controllers(tmp_path))["rl_ppo"]["status"] == "experimental"  # a model without a gate result is never verified
    (tmp_path / "evaluation.json").write_text(json.dumps({"gate": {"rl_ppo": {"status": "experimental", "failed": ["no_teleport_regression"]}}, "meta": {}}))
    c = _by_id(registry.controllers(tmp_path))["rl_ppo"]
    assert c["status"] == "experimental" and "no_teleport_regression" in c["note"] and registry.resolve("rl_ppo", tmp_path)["used"] == "rl_ppo"
    (tmp_path / "evaluation.json").write_text(json.dumps({"gate": {"rl_ppo": {"status": "verified", "failed": []}}, "meta": {}}))
    assert _by_id(registry.controllers(tmp_path))["rl_ppo"]["status"] == "verified"
    assert registry.controllers(tmp_path)["default_controller"] == "queue_responsive"


def test_tampered_model_is_unavailable(tmp_path):
    _write_rl(tmp_path, sha_ok=False)
    c = _by_id(registry.controllers(tmp_path))["rl_ppo"]
    assert c["status"] == "unavailable" and not c["available"]


def test_api_lists_controllers():
    from flowstate.api.app import app

    doc = TestClient(app).get("/api/controllers").json()
    assert doc["default_controller"] == "queue_responsive"
    assert {"fixed_time", "queue_responsive", "ml_forecast", "rl_ppo"} <= set(_by_id(doc))
    assert all(c["status"] != "verified" or c["gate"]["status"] == "verified" for c in doc["controllers"] if c["kind"] == "ml")


def test_committed_models_are_recognised():
    """The repository's own model files must match their manifests (both manifest formats), or the UI silently falls back."""
    c = _by_id(registry.controllers())
    for cid in ("ml_forecast", "rl_ppo"):
        assert c[cid]["note"] != "model file does not match its manifest", cid
