"""
Flood Risk Prediction API — multi-model
=======================================
FastAPI service for the flood-risk prototype (River Murray catchment).

Serves any of the team's trained models through one contract. Every model is
trained on the shared base in notebooks/common.py, so they all take the same
four river-level features and expose predict_proba:

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
    import pandas as pd
except Exception:  # pragma: no cover
    joblib = None
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
MODELS: Dict[str, Dict[str, str]] = {
    "logistic_regression": {"name": "Logistic Regression", "file": "logistic_regression_real.joblib"},
    "random_forest": {"name": "Random Forest", "file": "random_forest.joblib"},
    "xgboost": {"name": "XGBoost", "file": "xgboost.joblib"},
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "random_forest")

_loaded: Dict[str, object] = {}   # id -> estimator (lazy cache)
_metrics: Dict[str, dict] = {}    # id -> metrics dict


def _model_path(model_id: str) -> Path:
    return MODEL_DIR / MODELS[model_id]["file"]


def _available(model_id: str) -> bool:
    return joblib is not None and _model_path(model_id).exists()


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
    feats = FloodFeatures(**{f: getattr(req, f) for f in FEATURE_ORDER})
    p = _probability(model_id, feats)
    return PredictionResponse(model=model_id, flood_probability=round(p, 4),
                              risk_band=_risk_band(p), features=feats.model_dump())


@app.post("/predict_series", response_model=PredictionResponse)
def predict_series(req: SeriesRequest) -> PredictionResponse:
    model_id = _resolve(req.model)
    feats = _features_from_series(req.levels)
    p = _probability(model_id, feats)
    return PredictionResponse(model=model_id, flood_probability=round(p, 4),
                              risk_band=_risk_band(p),
                              features={k: round(v, 4) for k, v in feats.model_dump().items()})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
