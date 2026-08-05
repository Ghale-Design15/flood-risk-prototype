"""
Regenerate docs/metrics.json from the saved models — honest, reproducible.

Evaluates every served model on the SAME chronological test fold and label as
common.py, so the /models endpoint and the report's comparison table never drift
from the real numbers. The ensemble is the equal-weight soft-vote (mean of the
tabular members' predict_proba), scored exactly as the API serves it.

Run from the repo (needs data/ and backend/models/):
    python notebooks/eval_metrics.py            # print the table
    python notebooks/eval_metrics.py --write     # also update docs/metrics.json

The 0/1 call uses p >= 0.5 (sklearn's default), which reproduces the stored
random_forest and xgboost numbers exactly. The LSTM is a sequence model with its
own scaler and tuned threshold; evaluate it in FloodRiskPrediction_LSTM.ipynb and
paste its row in, or extend this script once TensorFlow is available here.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

REPO = HERE.parent
MODEL_DIR = REPO / "backend" / "models"
METRICS_PATH = REPO / "docs" / "metrics.json"

# Tabular members that make up the ensemble (must match the API's MODELS registry).
MEMBERS = {
    "logistic_regression": "logistic_regression_real.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}
THRESHOLD = 0.5  # 0/1 decision threshold for the class call


def _round(d: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def main(write: bool) -> None:
    df = common.load_data()
    X, y = common.build_features(df)
    _, X_test, _, y_test = common.chronological_split(X, y)
    print(f"test fold: {len(X_test)} days, {y_test.mean():.3f} positive, "
          f"threshold {common.risk_threshold(df):.3f} m")

    proba = {}
    for mid, fname in MEMBERS.items():
        model = joblib.load(MODEL_DIR / fname)
        proba[mid] = model.predict_proba(X_test)[:, 1]

    results = {}
    for mid, p in proba.items():
        r = common.evaluate(mid, y_test, (p >= THRESHOLD).astype(int), p)
        results[mid] = _round({k: r[k] for k in ("F1", "MCC", "RMSE", "Brier", "NSE")})

    # Ensemble = equal-weight soft-vote (mean probability), scored like the API.
    p_ens = np.mean([proba[m] for m in MEMBERS], axis=0)
    r_ens = common.evaluate("ensemble", y_test, (p_ens >= THRESHOLD).astype(int), p_ens)
    results["ensemble"] = _round({k: r_ens[k] for k in ("F1", "MCC", "RMSE", "Brier", "NSE")})

    base = common.persistence_baseline(df, y_test)

    print(common.comparison_table(
        [dict(model=m, **results[m]) for m in ("ensemble", *MEMBERS)]))
    print(f"\npersistence baseline: F1 {base['F1']:.3f}, MCC {base['MCC']:.3f}")

    if write:
        out = {
            "ensemble": results["ensemble"],
            "logistic_regression": results["logistic_regression"],
            "random_forest": results["random_forest"],
            "xgboost": results["xgboost"],
            "_baseline": {"name": "Persistence baseline",
                          "F1": round(base["F1"], 3), "MCC": round(base["MCC"], 3)},
            "_note": ("Chronological split (common.py), 0/1 call at p>=0.5. "
                      "Ensemble = equal-weight soft-vote over LR/RF/XGB on the same "
                      "test fold. Regenerate with notebooks/eval_metrics.py --write. "
                      "LSTM row pending (sequence model, evaluate in its notebook)."),
        }
        METRICS_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {METRICS_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update docs/metrics.json")
    main(ap.parse_args().write)
