"""Unit tests for the Results-page replay tooling that do not need SUMO."""

from flowstate.replay.priority import MAX_EXTENSION_S, MIN_GREEN_S, EmergencyPriority
from flowstate.replay.record import asset, paired


def _run(seed, ev, queue, waiting, frames=()):
    return {"seed": seed, "tlsIds": ["j1"], "frames": list(frames), "units": [], "focus": {"kind": "network"},
            "metrics": {"ev_travel_mean_s": ev, "ev_stopped_red_s": 0.0, "ev_stopped_queue_s": 0.0, "queue_total_mean_veh": queue, "throughput_vph": 1000.0,
                        "teleports": 0, "collisions": 0, "car": {"waiting": waiting, "speed": 20.0, "trips": 50}}}


def test_paired_signs_and_counts():
    pairs = [{"seed": s, "left": _run(s, 100 + s, 20.0, 25.0), "right": _run(s, 80 + s, 18.0, 24.0)} for s in range(6)]
    pairs.append({"seed": 9, "left": _run(9, 100, 20.0, 25.0), "right": _run(9, 130, 25.0, 30.0)})  # one run where the right side is worse
    agg = paired(pairs, ["waiting", "speed", "trips"])
    ev = agg["ev_travel_mean_s"]
    assert ev["pairs"] == 7 and ev["better"] == 6 and ev["worse"] == 1
    assert ev["mean_diff"] < 0 and ev["ci95"][0] <= ev["mean_diff"] <= ev["ci95"][1]
    assert agg["car_waiting"]["better"] == 6 and agg["car_waiting"]["worse"] == 1
    assert agg["car_speed"]["better"] == 0 and agg["car_speed"]["worse"] == 0  # unchanged speed is neither


def test_asset_lists_every_ambulance_and_picks_by_rule():
    def unit(call, iid, travel, red, queue, finished=True):
        return {"vid": f"amb{call}", "label": f"Ambulance {call}", "kind": "ambulance", "call": call, "id": iid, "travelS": travel, "stoppedRedS": red,
                "stoppedQueueS": queue, "finished": finished, "departS": 10.0 + call, "first": 10, "last": 10 + int(travel)}

    left = _run(1, 100.0, 20.0, 25.0, [{"t": 0, "v": [], "s": [""]}])
    right = _run(1, 80.0, 18.0, 24.0, [{"t": 0, "v": [], "s": [""]}])
    left["units"] = [unit(1, 5, 120.0, 10.0, 30.0), unit(2, 6, 150.0, 0.0, 5.0), unit(3, 7, 90.0, 0.0, 0.0, finished=False)]
    right["units"] = [unit(1, 50, 70.0, 0.0, 2.0), unit(2, 60, 60.0, 0.0, 0.0), unit(3, 70, 80.0, 0.0, 0.0)]
    left["focus"] = {"kind": "network"}
    doc = asset("ambulance", {"seed": 1, "left": left, "right": right}, {}, [1])
    assert [u["label"] for u in doc["units"]] == ["Ambulance 1", "Ambulance 2", "Ambulance 3"]
    assert doc["units"][2]["savedS"] is None  # unfinished on one side is reported, not dropped
    assert doc["example"] == 5  # call 1: queued >= 10 s on the left and held <= 3 s on the right; call 2 saved more but was not queue-blocked
    assert doc["units"][0]["right"]["id"] == 50 and doc["codes"] == "main" and doc["network"] == "district" and doc["kpis"][0]["key"] == "ev_travel_mean_s"


def test_queue_rule_summary():
    from flowstate.replay.heuristic import QueueResponsive

    q = QueueResponsive()
    assert q.summary()["gapOuts"] == 0 and "min 10 s" in q.summary()["rule"]


def test_priority_rule_constants_and_summary():
    p = EmergencyPriority()
    assert MIN_GREEN_S >= 7.0 and MAX_EXTENSION_S <= 30.0
    s = p.summary()
    assert s["extensions"] == 0 and s["earlyEnds"] == 0 and "no skipped amber" in s["rule"]


def test_run_rejects_unknown_controller_and_maps_aliases():
    import pytest

    from flowstate.replay import record

    with pytest.raises(ValueError):
        record.run(1, "nope", "rush_hour", False)
    assert record.ARM_ALIASES["flowstate"] == "queue_responsive" and record.ASSET_CONTROLLER["queue_responsive"] == "flowstate_rules"
    assert set(record.ARM_IDS) == {"fixed_time", "queue_responsive", "safe_heuristic", "ml_forecast", "rl_ppo"}


def test_right_includes_is_truthful_for_experimental_models():
    from flowstate.replay.record import right_includes

    assert "did not pass the held-out publication gate" in right_includes("rl_ppo", "experimental")
    assert "did not pass" in right_includes("ml_forecast", "experimental")
    assert "no learned model" in right_includes("queue_responsive", "default")
    assert "no trained model" in right_includes("safe_heuristic", "experimental")


def test_compare_one_falls_back_when_controller_unavailable(monkeypatch, tmp_path):
    from concurrent.futures import Future

    from flowstate.replay import record

    monkeypatch.setenv("FLOWSTATE_ML_DIR", str(tmp_path))
    calls = []

    class Inline:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, *args):
            f = Future()
            f.set_result(fn(*args))
            return f

    def fake_run(seed, arm, scenario_id="x", frames=True):
        calls.append(arm)
        r = _run(seed, 1.0, 1.0, 1.0)
        r.update({"network": "district", "controller": record.ASSET_CONTROLLER.get(arm, arm)})
        return r

    monkeypatch.setattr(record, "run", fake_run)
    monkeypatch.setattr(record, "ProcessPoolExecutor", Inline)
    doc = record.compare_one("rush_hour", 5, "rl_ppo")
    assert calls == ["fixed_time", "queue_responsive"]
    assert doc["schema"] == "flowstate.results-demo/4" and doc["rightController"] == "queue_responsive" and doc["rightStatus"] == "default"
    assert doc["fallback"] and doc["right"]["controller"] == "flowstate_rules"
    calls.clear()
    doc = record.compare_one("rush_hour", 5)
    assert calls == ["fixed_time", "queue_responsive"] and doc["fallback"] is None


def test_replay_endpoint_rejects_unknown_controller():
    from fastapi.testclient import TestClient

    from flowstate.api.app import app

    r = TestClient(app).post("/api/replay", json={"scenario": "rush_hour", "seed": 1, "controller": "bogus"})
    assert r.status_code == 400 and "unknown controller" in r.json()["detail"]
