"""
Flood Risk Prediction API — multi-model
=======================================
FastAPI service for the flood-risk prototype (River Murray catchment).

Serves any of the team's trained models through one contract. Two kinds:

  * tabular models (Logistic Regression, Random Forest, XGBoost) score the four
    river-level features below via predict_proba, on /predict or /predict_series.
  * the sequence model (LSTM) scores a window of recent daily levels and is only
    available on /predict_series (it needs the raw series, not the flat features).

    level_lag1     water level 1 day ago (m)
    level_lag2     water level 2 days ago (m)
    level_roll7    mean of the 7 most recent prior days (m)
    level_change3  level_lag1 minus the level 4 days ago (m)

Endpoints
    GET  /health          service + which models are available
    GET  /models          list models (id, name, available, default, metrics)
    POST /predict         score the 4 features directly; optional "model"
    POST /predict_series  score from recent daily levels; optional "model"

Add a model in one place — the MODELS registry below. Drop its .joblib into
backend/models/ and, if you have them, its metrics into docs/metrics.json.
"""

from __future__ import annotations

import importlib.util
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import joblib
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    joblib = None
    np = None
    pd = None

# Feature order MUST match notebooks/common.py FEATURES. A test asserts this.
FEATURE_ORDER = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]

# Risk threshold used to label training data (0.80 quantile of the level).
TRAIN_RISK_THRESHOLD_M = 0.806

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR.parent / "docs" / "metrics.json"

# ---- Model registry -------------------------------------------------------
# Add a model here, drop its file in backend/models/, done.
#   kind "tabular"  -> sklearn-style estimator with predict_proba on the 4 features
#   kind "sequence" -> Keras model scored on a window of recent daily levels; needs
#                      an "aux" bundle (joblib dict with scaler, window, threshold)
MODELS: Dict[str, Dict[str, str]] = {
    "logistic_regression": {"name": "Logistic Regression", "file": "logistic_regression_real.joblib", "kind": "tabular"},
    "random_forest": {"name": "Random Forest", "file": "random_forest.joblib", "kind": "tabular"},
    "xgboost": {"name": "XGBoost", "file": "xgboost.joblib", "kind": "tabular"},
    "lstm": {"name": "LSTM", "file": "lstm.keras", "kind": "sequence", "aux": "lstm_scaler.joblib"},
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "random_forest")

_loaded: Dict[str, object] = {}   # id -> estimator (lazy cache)
_metrics: Dict[str, dict] = {}    # id -> metrics dict


def _model_path(model_id: str) -> Path:
    return MODEL_DIR / MODELS[model_id]["file"]


def _kind(model_id: str) -> str:
    return MODELS[model_id].get("kind", "tabular")


def _keras_available() -> bool:
    """True if TensorFlow/Keras is installed. Uses find_spec so we don't pay the
    (heavy) import cost just to answer /models or /health."""
    return importlib.util.find_spec("tensorflow") is not None


def _available(model_id: str) -> bool:
    meta = MODELS[model_id]
    if joblib is None or not _model_path(model_id).exists():
        return False
    if meta.get("kind") == "sequence":
        # Needs Keras at runtime plus its companion scaler/window bundle.
        return _keras_available() and (MODEL_DIR / meta["aux"]).exists()
    return True


def _sequence_window(model_id: str) -> int:
    """Read just the window length from the aux bundle (cheap) for error messages."""
    try:
        return int(joblib.load(MODEL_DIR / MODELS[model_id]["aux"])["window"])
    except Exception:
        return 14


def _resolve(model_id: Optional[str]) -> str:
    """Pick a valid, available model id or raise a clear 4xx."""
    if model_id is None:
        if _available(DEFAULT_MODEL):
            return DEFAULT_MODEL
        for mid in MODELS:
            if _available(mid):
                return mid
        raise HTTPException(status_code=503, detail="No model files are available on the server.")
    if model_id not in MODELS:
        raise HTTPException(status_code=404,
                            detail=f"Unknown model '{model_id}'. Options: {list(MODELS)}")
    if not _available(model_id):
        raise HTTPException(status_code=409,
                            detail=f"Model '{model_id}' is registered but its file is missing on the server.")
    return model_id


def _get_model(model_id: str):
    if model_id not in _loaded:
        meta = MODELS[model_id]
        if meta.get("kind") == "sequence":
            # Keras is imported lazily so the tabular models work even if TF is absent.
            from tensorflow.keras.models import load_model  # noqa: WPS433
            bundle = joblib.load(MODEL_DIR / meta["aux"])   # {scaler, window, threshold}
            bundle["model"] = load_model(_model_path(model_id))
            _loaded[model_id] = bundle
        else:
            _loaded[model_id] = joblib.load(_model_path(model_id))
    return _loaded[model_id]


def _load_metrics() -> None:
    global _metrics
    try:
        _metrics = json.loads(METRICS_PATH.read_text())
    except Exception:
        _metrics = {}


def _probability(model_id: str, feats: "FloodFeatures") -> float:
    model = _get_model(model_id)
    row = pd.DataFrame([[getattr(feats, f) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return float(model.predict_proba(row)[0][1])


def _probability_sequence(model_id: str, levels: List[float]) -> float:
    """Score a Keras sequence model on the most recent `window` daily levels,
    scaled with the model's saved StandardScaler (see FloodRiskPrediction_LSTM)."""
    bundle = _get_model(model_id)
    model, scaler, window = bundle["model"], bundle["scaler"], int(bundle["window"])
    if len(levels) < window:
        raise HTTPException(
            status_code=422,
            detail=f"The '{model_id}' model needs at least {window} recent daily levels; received {len(levels)}.",
        )
    seq = np.asarray(levels[-window:], dtype="float32").reshape(-1, 1)
    seq = scaler.transform(seq).reshape(1, window, 1)
    return float(model.predict(seq, verbose=0).ravel()[0])


def _risk_band(p: float) -> Literal["Low", "Moderate", "High"]:
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Moderate"
    return "Low"


def _features_from_series(levels: List[float]) -> "FloodFeatures":
    """Derive the 4 features from recent daily levels (most recent last),
    matching notebooks/common.build_features. Predicts the NEXT day."""
    if len(levels) < 4:
        raise HTTPException(status_code=422,
                            detail="Provide at least 4 recent daily levels (7+ preferred for level_roll7).")
    window = levels[-7:]
    return FloodFeatures(
        level_lag1=levels[-1], level_lag2=levels[-2],
        level_roll7=sum(window) / len(window), level_change3=levels[-1] - levels[-4],
    )


# ---- Schemas --------------------------------------------------------------
class FloodFeatures(BaseModel):
    level_lag1: float = Field(..., description="Water level 1 day ago (m)")
    level_lag2: float = Field(..., description="Water level 2 days ago (m)")
    level_roll7: float = Field(..., description="Mean of 7 most recent prior days (m)")
    level_change3: float = Field(..., description="level_lag1 minus level 4 days ago (m)")


class PredictRequest(FloodFeatures):
    model: Optional[str] = Field(None, description="Model id from GET /models. Omit for the default.")


class SeriesRequest(BaseModel):
    levels: List[float] = Field(..., min_length=4,
                                description="Recent daily water levels (m), oldest first, most recent last.",
                                examples=[[0.72, 0.70, 0.67, 0.65, 0.69, 0.69, 0.73]])
    model: Optional[str] = Field(None, description="Model id from GET /models. Omit for the default.")
    station_id: str = Field("A4261162", description="Gauging station id (Murray Bridge default).")


class PredictionResponse(BaseModel):
    model: str
    flood_probability: float
    risk_band: Literal["Low", "Moderate", "High"]
    features: dict


class ModelInfo(BaseModel):
    id: str
    name: str
    available: bool
    default: bool
    metrics: Optional[dict] = None


class AlertRequest(BaseModel):
    """Authorised alert from the dashboard's two-step confirm (human-in-the-loop)."""
    risk_band: Literal["Low", "Moderate", "High"]
    operator: str = Field(..., min_length=1, description="Operator ID who authorised the alert.")
    station_id: Optional[str] = None
    station_name: Optional[str] = None
    flood_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    horizon: Optional[str] = None
    message: Optional[str] = None


class AlertResponse(BaseModel):
    alert_id: str
    created_at: str
    prev_hash: str
    row_hash: str
    chained: bool
    email_status: str


# ---- App ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_metrics()
    yield


app = FastAPI(
    title="Flood Risk Prediction API",
    description="Multi-model next-day river-flood risk for the River Murray catchment (ITA602).",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "features": FEATURE_ORDER,
        "default_model": DEFAULT_MODEL,
        "available_models": [m for m in MODELS if _available(m)],
        "train_risk_threshold_m": TRAIN_RISK_THRESHOLD_M,
    }


@app.get("/models", response_model=List[ModelInfo])
def models() -> List[ModelInfo]:
    return [
        ModelInfo(id=mid, name=meta["name"], available=_available(mid),
                  default=(mid == DEFAULT_MODEL), metrics=_metrics.get(mid))
        for mid, meta in MODELS.items()
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest) -> PredictionResponse:
    model_id = _resolve(req.model)
    if _kind(model_id) == "sequence":
        raise HTTPException(
            status_code=422,
            detail=(f"Model '{model_id}' scores a time series, not flat features. "
                    f"Use /predict_series with at least {_sequence_window(model_id)} recent daily levels."),
        )
    feats = FloodFeatures(**{f: getattr(req, f) for f in FEATURE_ORDER})
    p = _probability(model_id, feats)
    return PredictionResponse(model=model_id, flood_probability=round(p, 4),
                              risk_band=_risk_band(p), features=feats.model_dump())


@app.post("/predict_series", response_model=PredictionResponse)
def predict_series(req: SeriesRequest) -> PredictionResponse:
    model_id = _resolve(req.model)
    if _kind(model_id) == "sequence":
        p = _probability_sequence(model_id, req.levels)
        window = _sequence_window(model_id)
        return PredictionResponse(model=model_id, flood_probability=round(p, 4),
                                  risk_band=_risk_band(p),
                                  features={"window": window, "levels_used": len(req.levels)})
    feats = _features_from_series(req.levels)
    p = _probability(model_id, feats)
    return PredictionResponse(model=model_id, flood_probability=round(p, 4),
                              risk_band=_risk_band(p),
                              features={k: round(v, 4) for k, v in feats.model_dump().items()})


@app.post("/alerts", response_model=AlertResponse)
def post_alert(req: AlertRequest) -> AlertResponse:
    """Record an authorised flood alert as a link in the tamper-evident audit
    chain, then email it (SendGrid) when configured. Returns the alert id and
    the chain hashes. Requires the Supabase audit store to be configured."""
    import alerts

    if not alerts.store_configured():
        raise HTTPException(
            status_code=503,
            detail="Audit store not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
        )
    try:
        result = alerts.record_alert(req.model_dump())
    except Exception as exc:  # surface a clean 502 rather than a stack trace
        raise HTTPException(status_code=502, detail=f"Failed to record alert: {exc}")
    return AlertResponse(**result)


@app.get("/alerts/verify")
def verify_alerts() -> dict:
    """Recompute the whole audit chain and report whether it is intact."""
    import alerts

    if not alerts.store_configured():
        raise HTTPException(
            status_code=503,
            detail="Audit store not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
        )
    try:
        rows = alerts.fetch_chain()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to read audit chain: {exc}")
    return alerts.verify_chain(rows)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
