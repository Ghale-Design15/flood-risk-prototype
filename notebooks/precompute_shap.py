"""Precompute SHAP importances so the dashboard loads them instead of recomputing.

- TreeExplainer on the tree models (Random Forest, XGBoost) -> mean |SHAP| per feature.
- GradientExplainer on the LSTM -> mean |SHAP| per day in the 14-day window.

Saves one .npy per model in shap/, plus a small shap/index.json describing them.
Run:  python notebooks/precompute_shap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import joblib
import shap

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
import common  # noqa: E402

SHAP_DIR = REPO / "shap"
SHAP_DIR.mkdir(exist_ok=True)
MODELS = REPO / "backend" / "models"

np.random.seed(42)

df = common.load_data()
X, y = common.build_features(df)
X_train, X_test, y_train, y_test = common.chronological_split(X, y)
index = {}


def positive_class(vals):
    """Return a 2D (n_samples, n_features) array of positive-class SHAP values."""
    vals = np.asarray(vals)
    if vals.ndim == 3:          # (n, features, classes)
        return vals[:, :, -1]
    return vals                 # (n, features)


# ---- Tree models: TreeExplainer ------------------------------------------
for mid, fname in [("random_forest", "random_forest.joblib"), ("xgboost", "xgboost.joblib")]:
    path = MODELS / fname
    if not path.exists():
        print(f"skip {mid}: {fname} not found")
        continue
    model = joblib.load(path)
    explainer = shap.TreeExplainer(model)
    vals = positive_class(explainer.shap_values(X_test))
    importance = np.abs(vals).mean(axis=0)              # mean |SHAP| per feature
    np.save(SHAP_DIR / f"{mid}.npy", importance.astype("float32"))
    index[mid] = {"kind": "tree", "labels": common.FEATURES,
                  "importance": [round(float(v), 4) for v in importance]}
    print(mid, "->", dict(zip(common.FEATURES, importance.round(4))))


# ---- LSTM: GradientExplainer ---------------------------------------------
lstm_path = MODELS / "lstm.keras"
if lstm_path.exists():
    from tensorflow.keras.models import load_model
    bundle = joblib.load(MODELS / "lstm_scaler.joblib")
    scaler, N = bundle["scaler"], bundle["window"]
    lstm = load_model(lstm_path)

    levels = df["water_level_m"].to_numpy("float32")
    seqs = np.array([levels[t - N:t] for t in range(N, len(levels))])
    i_val, i_test = int(len(seqs) * 0.70), int(len(seqs) * 0.85)

    def prep(a):
        return scaler.transform(a.reshape(-1, 1)).reshape(a.shape[0], N, 1)

    X_tr, X_te = prep(seqs[:i_val]), prep(seqs[i_test:])
    background = X_tr[np.random.choice(len(X_tr), 100, replace=False)]
    ge = shap.GradientExplainer(lstm, background)
    vals = np.asarray(ge.shap_values(X_te[:200]))       # (n, N, 1) [maybe extra axis]
    importance = np.abs(vals).reshape(-1, N).mean(axis=0)  # mean |SHAP| per day
    np.save(SHAP_DIR / "lstm.npy", importance.astype("float32"))
    day_labels = [f"day -{N - i}" for i in range(N)]
    index["lstm"] = {"kind": "sequence", "labels": day_labels,
                     "importance": [round(float(v), 4) for v in importance]}
    print("lstm  -> per-day importance, top day:",
          day_labels[int(np.argmax(importance))], round(float(importance.max()), 4))
else:
    print("skip lstm: lstm.keras not found")

(SHAP_DIR / "index.json").write_text(json.dumps(index, indent=2))
print("\nSaved", len(index), "SHAP files to", SHAP_DIR)
