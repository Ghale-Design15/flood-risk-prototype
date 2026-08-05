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


# ---- Ensemble (default model) ---------------------------------------------
def test_ensemble_is_registered_and_default():
    assert "ensemble" in main.MODELS
    assert main.DEFAULT_MODEL == "ensemble"
    assert main.MODELS["ensemble"]["kind"] == "ensemble"


def test_ensemble_available_when_members_are():
    # Available iff at least ENSEMBLE_MIN_MEMBERS tabular members are available.
    n_members = len(main._ensemble_members("ensemble"))
    assert main._available("ensemble") == (n_members >= main.ENSEMBLE_MIN_MEMBERS)


def test_ensemble_is_soft_vote_of_members():
    if not main._available("ensemble"):
        return  # no member files on this checkout
    feats = main.FloodFeatures(**SAMPLE)
    members = main._ensemble_members("ensemble")
    expected = sum(main._probability(m, feats) for m in members) / len(members)
    got = main._probability_ensemble("ensemble", feats)
    assert abs(got - expected) < 1e-9
    assert 0.0 <= got <= 1.0


def test_predict_uses_ensemble_by_default():
    if not main._available("ensemble"):
        return
    r = client.post("/predict", json=SAMPLE)  # no "model" -> default
    assert r.status_code == 200
    assert r.json()["model"] == "ensemble"


def test_predict_series_ensemble_returns_four_features():
    if not main._available("ensemble"):
        return
    r = client.post("/predict_series",
                    json={"levels": [0.72, 0.70, 0.67, 0.65, 0.69, 0.69, 0.73], "model": "ensemble"})
    assert r.status_code == 200
    assert set(r.json()["features"]) == set(FEATURE_ORDER)


# ---- Alert audit chain ----------------------------------------------------
import alerts

ALERT = {"station_id": "A4261", "station_name": "Murray Bridge", "risk_band": "High",
         "flood_probability": 0.81, "horizon": "48h", "operator": "op-7", "message": "High risk"}


def _chain(payloads):
    rows, prev = [], alerts.GENESIS_HASH
    for i, p in enumerate(payloads):
        r = alerts.build_record(p, prev, alert_id=f"A-{i:04d}", created_at=f"2026-07-31T0{i}:00:00+00:00")
        rows.append(r)
        prev = r["row_hash"]
    return rows


def test_chain_hash_is_deterministic():
    rows = _chain([ALERT])
    assert alerts.compute_hash(rows[0]) == rows[0]["row_hash"]


def test_intact_chain_verifies():
    rows = _chain([ALERT, {**ALERT, "risk_band": "Low"}, {**ALERT, "risk_band": "Moderate"}])
    assert alerts.verify_chain(rows)["ok"] is True


def test_tampered_value_is_detected():
    rows = _chain([ALERT, {**ALERT, "risk_band": "Low"}])
    rows[0]["flood_probability"] = 0.99  # edit a stored field without fixing the hash
    result = alerts.verify_chain(rows)
    assert result["ok"] is False and result["broken_at"] == 0


def test_removed_link_is_detected():
    rows = _chain([ALERT, {**ALERT, "risk_band": "Low"}, {**ALERT, "risk_band": "Moderate"}])
    del rows[1]  # break the prev_hash linkage
    assert alerts.verify_chain(rows)["ok"] is False


def test_recipients_are_in_the_chain():
    with_r = alerts.build_record({**ALERT, "recipients": ["a@x.com", "b@y.com"]},
                                 alerts.GENESIS_HASH, alert_id="A-0000", created_at="2026-07-31T00:00:00+00:00")
    without_r = alerts.build_record(ALERT, alerts.GENESIS_HASH,
                                    alert_id="A-0000", created_at="2026-07-31T00:00:00+00:00")
    # recipients participate in the hash, so the two differ; both still self-verify.
    assert with_r["recipients"] == "a@x.com, b@y.com"
    assert with_r["row_hash"] != without_r["row_hash"]
    assert alerts.verify_chain([with_r])["ok"] is True


def test_float_normalisation_is_stable():
    a = alerts.build_record(ALERT, alerts.GENESIS_HASH, alert_id="A-0000", created_at="2026-07-31T00:00:00+00:00")
    b = alerts.build_record({**ALERT, "flood_probability": 0.8100000001}, alerts.GENESIS_HASH,
                            alert_id="A-0000", created_at="2026-07-31T00:00:00+00:00")
    assert a["row_hash"] == b["row_hash"]


def test_alerts_endpoint_503_when_store_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    alerts._client_ready = False  # reset the lazy client cache
    r = client.post("/alerts", json=ALERT)
    assert r.status_code == 503


def test_alerts_endpoint_validates_payload():
    r = client.post("/alerts", json={"flood_probability": 0.5})  # missing risk_band + operator
    assert r.status_code == 422
