# API Contract — Flood Risk Prediction (v2, multi-model)

Reference for Julieth's Week 9 backend tasks: model parameter, `GET /models`,
Render deploy, and CI. This is a working draft — `main.py` and `test_main.py`
in this folder already implement everything below and pass 10 tests.

## The idea in one line

Every model (Logistic Regression, Random Forest, XGBoost) trains through
`notebooks/common.py`, so they all take the **same four features** and expose
`predict_proba`. That means one contract serves all of them; the caller just
names which model.

## Endpoints

### `GET /health`
```json
{ "status": "ok",
  "features": ["level_lag1","level_lag2","level_roll7","level_change3"],
  "default_model": "random_forest",
  "available_models": ["logistic_regression","random_forest"],
  "train_risk_threshold_m": 0.806 }
```

### `GET /models`
Drives the dashboard's model dropdown.
```json
[ { "id": "logistic_regression", "name": "Logistic Regression",
    "available": true, "default": false,
    "metrics": { "F1": 0.793, "MCC": 0.732, "RMSE": 0.304, "Brier": 0.092, "NSE": 0.492 } },
  { "id": "random_forest", "name": "Random Forest",
    "available": true, "default": true,
    "metrics": { "F1": 0.932, "MCC": 0.910, "RMSE": 0.174, "Brier": 0.030, "NSE": 0.833 } } ]
```

### `POST /predict`
Body — the four features, plus an optional `model`:
```json
{ "level_lag1": 0.73, "level_lag2": 0.69, "level_roll7": 0.68,
  "level_change3": 0.06, "model": "random_forest" }
```
Response:
```json
{ "model": "random_forest", "flood_probability": 0.069,
  "risk_band": "Low",
  "features": { "level_lag1":0.73, "level_lag2":0.69, "level_roll7":0.68, "level_change3":0.06 } }
```

### `POST /predict_series`
Body — recent daily levels (oldest first), optional `model`:
```json
{ "levels": [0.72,0.70,0.67,0.65,0.69,0.69,0.73], "model": "logistic_regression" }
```
The API derives the four features and returns the same response shape.

## Rules

- `model` omitted → the default (`DEFAULT_MODEL`, currently `random_forest`).
- Unknown model id → `404`. Registered but file missing on server → `409`.
- No model files at all → `503`.
- Fewer than 4 levels on `/predict_series` → `422`.
- The response always echoes which `model` actually served the call.

## Adding a model (one place)

1. Train it through `common.py` so it uses the same features.
2. Save the `.joblib` into `backend/models/`.
3. Add one line to the `MODELS` registry in `main.py`:
   `"xgboost": {"name": "XGBoost", "file": "xgboost.joblib"}`
4. Add its metrics to `docs/metrics.json`.

## Two things to settle first (coordination)

1. **Move `Random_Forest.joblib` into `backend/models/`.** It currently lives in
   `notebooks/` (16.5 MB). The API can't serve it until it's in `backend/models/`
   and committed. Fine for Render's free tier, but confirm the filename matches
   the registry.
2. **`docs/metrics.json`** should be written by the evaluation notebook, not
   hand-maintained, so `/models` never drifts from the real numbers. A sample is
   in `metrics.sample.json` — copy it to `docs/metrics.json` for now.

## Metrics note (chronological split from common.py)

| Model | F1 | MCC |
|---|---|---|
| Persistence baseline | 0.796 | 0.732 |
| Logistic Regression | 0.793 | 0.732 |
| Random Forest | 0.932 | 0.910 |

Random Forest is the model that beats the baseline, which is why it's the
default. (These replace the older random-split numbers.)

## Render deploy (unchanged from before)

`render.yaml` already sets root dir `backend`, start command
`uvicorn main:app --host 0.0.0.0 --port 10000`, health check `/health`, free plan.
Once the code is pushed: render.com → New Web Service → pick the repo → it reads
`render.yaml`. Add `DEFAULT_MODEL` as an env var if you want to override the default.

## CI (GitHub Actions)

A ready workflow is in `ci.yml` — put it at `.github/workflows/ci.yml`. It installs
`backend/requirements.txt` and runs `pytest` on every push and pull request. The
tests are written to stay green even if the large model files aren't in the
checkout, so CI won't break on a shallow clone.
