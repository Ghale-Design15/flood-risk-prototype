"""
Export an XGBoost model on the shared common.py split.

Run from the repo (xgboost must be installed — it is in requirements.txt):

    python notebooks/export_xgboost.py

It trains an XGBClassifier on Manuela's chronological split, saves
backend/models/xgboost.joblib, and writes the honest metrics into
docs/metrics.json so GET /models reports them. Then commit both files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import common` works
import common  # noqa: E402
import joblib  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def main():
    repo = Path(__file__).resolve().parent.parent

    df = common.load_data()
    X, y = common.build_features(df)
    X_train, X_test, y_train, y_test = common.chronological_split(X, y)

    # Handle the rare high-water class with scale_pos_weight, matching the
    # class_weight="balanced" idea the other models use.
    pos = max(int(y_train.sum()), 1)
    neg = int((y_train == 0).sum())

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    out = repo / "backend" / "models" / "xgboost.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"saved {out}")

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    result = common.evaluate("XGBoost", y_test, pred, prob)
    metrics = {k: round(float(result[k]), 3) for k in ["F1", "MCC", "RMSE", "Brier", "NSE"]}
    base = common.persistence_baseline(df, y_test)
    print(f"XGBoost: {metrics}")
    print(f"persistence baseline: F1 {base['F1']:.3f}  MCC {base['MCC']:.3f}")

    mpath = repo / "docs" / "metrics.json"
    data = json.loads(mpath.read_text()) if mpath.exists() else {}
    data["xgboost"] = metrics
    mpath.write_text(json.dumps(data, indent=2))
    print(f"updated {mpath}")

    print("\nNext: commit backend/models/xgboost.joblib and docs/metrics.json, then push.")


if __name__ == "__main__":
    main()
