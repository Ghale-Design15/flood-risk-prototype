# API Contract — Flood Risk Prediction (v3, ensemble default)

**FROZEN for Week 11 demo — do not change the request/response shape after this.**
Any later internal tuning must keep this interface stable so the dashboard and the
demo rehearsals stay valid. `main.py` and `test_main.py` implement everything below
and pass 23 tests.

## The idea in one line

The tabular models (Logistic Regression, Random Forest, XGBoost) all train through
`notebooks/common.py`, so they take the **same four features** and expose
`predict_proba`. One contract serves all of them; the caller just names which model.
Two extras sit on top: the **ensemble** (the default) soft-votes over the tabular
models, and the **LSTM** scores a raw daily-level series on `/predict_series`.

## Endpoints

### `GET /health`
```json
{ "status": "ok",
  "features": ["level_lag1","level_lag2","level_roll7","level_change3"],
  "default_model": "ensemble",
  "available_models": ["ensemble","logistic_regression","random_forest","xgboost"],
  "train_risk_threshold_m": 0.806 }
```

### `GET /models`
Drives the dashboard's model dropdown. `metrics` come from `docs/metrics.json`;
entries still being evaluated report `null` until that file is updated.
```json
[ { "id": "ensemble", "name": "Ensemble (soft-vote)",
    "available": true, "default": true,
    "metrics": { "F1": 0.806, "MCC": 0.743, "RMSE": 0.271, "Brier": 0.074, "NSE": 0.596 } },
  { "id": "logistic_regression", "name": "Logistic Regression",
    "available": true, "default": false,
    "metrics": { "F1": 0.800, "MCC": 0.741, "RMSE": 0.300, "Brier": 0.090, "NSE": 0.505 } },
  { "id": "random_forest", "name": "Random Forest",
    "available": true, "default": false,
    "metrics": { "F1": 0.799, "MCC": 0.734, "RMSE": 0.264, "Brier": 0.070, "NSE": 0.618 } },
  { "id": "xgboost", "name": "XGBoost",
    "available": true, "default": false,
    "metrics": { "F1": 0.802, "MCC": 0.738, "RMSE": 0.276, "Brier": 0.076, "NSE": 0.582 } } ]
```

### `POST /predict`
Body — the four features. Omit `model` to use the default (the ensemble), or name
any tabular model:
```json
{ "level_lag1": 0.73, "level_lag2": 0.69, "level_roll7": 0.68,
  "level_change3": 0.06 }
```
Response (the `model` field echoes what actually served the call):
```json
{ "model": "ensemble", "flood_probability": 0.1118,
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

- `model` omitted → the default (`DEFAULT_MODEL`, currently `ensemble`; override
  with the `DEFAULT_MODEL` env var).
- Unknown model id → `404`. Registered but file missing on server → `409`.
- No model files at all → `503`.
- Fewer than 4 levels on `/predict_series` → `422`.
- The response always echoes which `model` actually served the call.

### `POST /predict` / `/predict_series` with the ensemble

The **ensemble** is the default model. It has no file of its own: it soft-votes by
averaging the `predict_proba` of its available tabular members
(`logistic_regression`, `random_forest`, `xgboost`) and returns the mean
probability. It is served on **both** `/predict` and `/predict_series` and returns
the same four-feature response shape as a single tabular model. It is offered only
while at least **2** members are available; otherwise it reports unavailable in
`/models` and `/health` (a single available model would not be a real vote).
Weighting is equal by default.

### `POST /predict_series` with the LSTM

The LSTM is a **sequence** model: it scores the last 14 daily levels, so it is
only served on `/predict_series` (not `/predict`). Send at least 14 levels:
```json
{ "levels": [ ... 14+ recent daily levels ... ], "model": "lstm" }
```
Fewer than the window (14) → `422`. Calling `/predict` with `model: "lstm"` → `422`
(needs a series, not the four flat features).

### `POST /alerts`
Records an authorised flood alert in a **tamper-evident hash-chained** audit log
(Supabase) and emails it via SendGrid when configured. Human-in-the-loop: the
dashboard only calls this after a two-step confirm.

Body (from the dashboard's alert card). `recipients` is chosen in the dashboard
(predefined organisations plus an optional custom address); it drives the email
`To` and is recorded in the audit chain. Omit it to fall back to `ALERT_EMAIL_TO`.
```json
{ "risk_band": "High", "operator": "op-7",
  "station_id": "A4261162", "station_name": "Murray Bridge",
  "flood_probability": 0.81, "horizon": "48h",
  "message": "High flood risk (81%) for Murray Bridge over the next 48h.",
  "recipients": ["ses@example.gov.au", "professor@university.edu"] }
```
Response:
```json
{ "alert_id": "A-3F9A2B10", "created_at": "2026-07-31T09:12:04+00:00",
  "recipients": "ses@example.gov.au, professor@university.edu",
  "prev_hash": "…", "row_hash": "…", "chained": true,
  "email_status": "sent (202)" }
```
Each row stores `prev_hash` (previous row's hash) and `row_hash`
(`sha256` of the row's canonical fields, `prev_hash` included). The table is
append-only at the DB level. `email_status` is `"skipped (…)"` when no SendGrid
key is set — the alert is still logged.

### `GET /alerts/verify`
Recomputes the whole chain and reports integrity:
```json
{ "ok": true, "checked": 42, "broken_at": null }
```
If a row was altered or removed, `ok` is `false` and `broken_at` is the index of
the first broken link.

**Setup:** run `db/supabase_audit.sql` in Supabase once, then set `SUPABASE_URL`,
`SUPABASE_SERVICE_KEY` (and optionally `SENDGRID_API_KEY`, `ALERT_EMAIL_FROM`,
`ALERT_EMAIL_TO`) — see `backend/.env.example`. Without the Supabase vars,
`/alerts` and `/alerts/verify` return `503`.

## Adding a model (one place)

1. Train it through `common.py` so it uses the same features.
2. Save the `.joblib` into `backend/models/`.
3. Add one line to the `MODELS` registry in `main.py`:
   `"xgboost": {"name": "XGBoost", "file": "xgboost.joblib", "kind": "tabular"}`
4. Add its metrics to `docs/metrics.json`.

## One thing to settle first (coordination)

1. **`docs/metrics.json`** should be written by the evaluation notebook, not
   hand-maintained, so `/models` never drifts from the real numbers. A sample is
   in `metrics.sample.json` — copy it to `docs/metrics.json` for now.

## Metrics note (chronological split from common.py)

Honest, time-ordered numbers from `docs/metrics.json`. The earlier Random Forest
F1 of 0.932 was **data leakage from a shuffled split** and has been removed.

Regenerate with `python notebooks/eval_metrics.py --write` (same 954-day test fold).

| Model | F1 | MCC | RMSE | Brier | NSE |
|---|---|---|---|---|---|
| Persistence baseline | 0.796 | 0.732 | – | – | – |
| Logistic Regression | 0.800 | 0.741 | 0.300 | 0.090 | 0.505 |
| Random Forest | 0.799 | 0.734 | 0.264 | 0.070 | 0.618 |
| XGBoost | 0.802 | 0.738 | 0.276 | 0.076 | 0.582 |
| **Ensemble (default)** | **0.806** | **0.743** | 0.271 | 0.074 | 0.596 |
| LSTM | *pending* | *pending* | *pending* | *pending* | *pending* |

The tabular models sit within one F1 point of the persistence baseline, which is
the honest headline of the project. The **ensemble** is the default because its
soft-vote gives the best F1 and MCC of any model with stable calibration, while
Random Forest keeps the best Brier/NSE. The LSTM row is pending (sequence model,
evaluated in its own notebook). Note: the Logistic Regression numbers moved from
the old 0.793 to 0.800 when regenerated against Manuela's week-10 scaled-LR model.

## Render deploy (unchanged from before)

`render.yaml` already sets root dir `backend`, start command
`uvicorn main:app --host 0.0.0.0 --port 10000`, health check `/health`, free plan.
Once the code is pushed: render.com → New Web Service → pick the repo → it reads
`render.yaml`. Add `DEFAULT_MODEL` as an env var if you want to override the default.

### Cold-start handling (so the demo never stalls)

The free instance sleeps after ~15 min idle; the first request then pays a cold
start (worse with the LSTM's TensorFlow load). Two safeguards:

- **Keep it warm:** `.github/workflows/keep-alive.yml` pings `/health` every 10 min
  and has a manual "Run workflow" button. Set repo variable `API_URL` to the Render
  URL. Cron timing is best-effort, so also —
- **Warm up before the demo:** run `python backend/warmup.py` (or pass the URL) a few
  minutes before presenting. It pings `/health` until the service answers in under 3s.

Front-ends should also send a `/health` ping on load and show a brief "warming up"
state rather than blocking on the first prediction.

## CI (GitHub Actions)

A ready workflow is in `ci.yml` — put it at `.github/workflows/ci.yml`. It installs
`backend/requirements.txt` and runs `pytest` on every push and pull request. The
tests are written to stay green even if the large model files aren't in the
checkout, so CI won't break on a shallow clone.
