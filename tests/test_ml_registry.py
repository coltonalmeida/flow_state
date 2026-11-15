"""Controller registry: two controllers, the agent's status comes from the gate and the model file, and nothing is substituted for it."""

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


def test_there_are_exactly_two_controllers_and_the_agent_is_the_default(tmp_path):
    doc = registry.controllers(tmp_path)
    c = _by_id(doc)
    assert list(c) == ["fixed_time", "rl_ppo"] and doc["default_controller"] == "rl_ppo"
    assert c["fixed_time"]["status"] == "baseline"


def test_a_missing_model_is_an_error_not_a_substitution(tmp_path):
    assert _by_id(registry.controllers(tmp_path))["rl_ppo"]["status"] == "unavailable"
    for asked in ("rl_ppo", None):
        with pytest.raises(registry.ControllerUnavailable, match="no trained model"):
            registry.resolve(asked, tmp_path)
    assert registry.resolve("fixed_time", tmp_path)["used"] == "fixed_time"
    with pytest.raises(ValueError):
        registry.resolve("nope", tmp_path)


def test_status_follows_the_gate_and_the_manifest_hash(tmp_path):
    _write_rl(tmp_path)
    assert _by_id(registry.controllers(tmp_path))["rl_ppo"]["status"] == "experimental"  # a model without a gate result is never verified
    (tmp_path / "evaluation.json").write_text(json.dumps({"gate": {"rl_ppo": {"status": "experimental", "failed": ["no_teleport_regression"]}}, "meta": {}}))
    c = _by_id(registry.controllers(tmp_path))["rl_ppo"]
    assert c["status"] == "experimental" and "no_teleport_regression" in c["note"]
    assert registry.resolve(None, tmp_path) == {"requested": None, "used": "rl_ppo", "status": "experimental"}
    (tmp_path / "evaluation.json").write_text(json.dumps({"gate": {"rl_ppo": {"status": "verified", "failed": []}}, "meta": {}}))
    assert _by_id(registry.controllers(tmp_path))["rl_ppo"]["status"] == "verified"
    assert registry.resolve("rl_ppo", tmp_path)["status"] == "verified"


def test_tampered_model_is_unavailable(tmp_path):
    _write_rl(tmp_path, sha_ok=False)
    c = _by_id(registry.controllers(tmp_path))["rl_ppo"]
    assert c["status"] == "unavailable" and not c["available"]
    with pytest.raises(registry.ControllerUnavailable, match="does not match its manifest"):
        registry.resolve("rl_ppo", tmp_path)


def test_api_lists_controllers():
    from flowstate.api.app import app

    doc = TestClient(app).get("/api/controllers").json()
    assert doc["default_controller"] == "rl_ppo" and set(_by_id(doc)) == {"fixed_time", "rl_ppo"}
    assert all(c["status"] != "verified" or c["gate"]["status"] == "verified" for c in doc["controllers"] if c["kind"] == "ml")


def test_the_committed_agent_is_recognised_and_verified_against_fixed_time():
    """The repository's own model file must match its manifest, and its stored gate result must be the one against fixed-time."""
    c = _by_id(registry.controllers())["rl_ppo"]
    assert c["available"], c["note"]
    assert c["status"] == "verified" and c["gate"]["baseline"] == "fixed_time" and not c["gate"]["failed"]
