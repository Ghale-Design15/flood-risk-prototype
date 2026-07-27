"""
Automated tests for the multi-model Flood Risk Prediction API.
Run from the backend/ folder:  pytest -q

Passes whether or not the model files are present (skips model-specific
assertions when a model is unavailable), so CI stays green on a fresh checkout.
"""

from fastapi.testclient import TestClient

import main
from main import app, FEATURE_ORDER, _risk_band, _features_from_series

client = TestClient(app)

SAMPLE = {"level_lag1": 0.73, "level_lag2": 0.69, "level_roll7": 0.68, "level_change3": 0.06}


def _first_available():
    for m in main.MODELS:
        if main._available(m):
            return m
    return None


def test_feature_order_matches_shared_base():
    # Invariant: must equal notebooks/common.py FEATURES.
    assert FEATURE_ORDER == ["level_lag1", "level_lag2", "level_roll7", "level_change3"]


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["features"] == FEATURE_ORDER
    assert "default_model" in body


def test_models_lists_registry():
    r = client.get("/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()]
    assert set(ids) == set(main.MODELS)
    assert sum(m["default"] for m in r.json()) == 1  # exactly one default


def test_predict_default_model():
    r = client.post("/predict", json=SAMPLE)
    if _first_available() is None:
        assert r.status_code == 503  # no model files on this checkout
        return
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["flood_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Moderate", "High"}
    assert body["model"] in main.MODELS


def test_predict_named_model():
    mid = _first_available()
    if mid is None:
        return
    r = client.post("/predict", json={**SAMPLE, "model": mid})
    assert r.status_code == 200
    assert r.json()["model"] == mid


def test_unknown_model_rejected():
    r = client.post("/predict", json={**SAMPLE, "model": "does_not_exist"})
    assert r.status_code == 404


def test_predict_series_derives_features():
    r = client.post("/predict_series", json={"levels": [0.72, 0.70, 0.67, 0.65, 0.69, 0.69, 0.73]})
    if _first_available() is None:
        assert r.status_code == 503
        return
    assert r.status_code == 200
    assert set(r.json()["features"]) == set(FEATURE_ORDER)
    assert r.json()["features"]["level_lag1"] == 0.73


def test_series_too_short_rejected():
    r = client.post("/predict_series", json={"levels": [0.7, 0.7]})
    assert r.status_code == 422


def test_feature_helper_math():
    f = _features_from_series([0.6, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72])
    assert f.level_lag1 == 0.72
    assert f.level_lag2 == 0.70
    assert round(f.level_change3, 2) == round(0.72 - 0.66, 2)


def test_risk_band_thresholds():
    assert _risk_band(0.10) == "Low"
    assert _risk_band(0.50) == "Moderate"
    assert _risk_band(0.90) == "High"
