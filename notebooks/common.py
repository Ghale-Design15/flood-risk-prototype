"""
Shared setup for the flood risk models.

Import this in every model notebook so we all train on the same data, the same
label and the same split. Otherwise the comparison table is meaningless.

    import common
    df = common.load_data()
    X, y = common.build_features(df)
    X_train, X_test, y_train, y_test = common.chronological_split(X, y)

Manuela Munoz Ramirez
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, matthews_corrcoef,
                             mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit

# Don't change these without telling the group. If two of us train on
# different values the models are no longer comparable.
RISK_QUANTILE = 0.80
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
N_SPLITS = 5
FEATURES = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]


def _repo_root():
    here = Path.cwd()
    for folder in (here, here.parent, here.parent.parent):
        if (folder / "data").is_dir():
            return folder
    raise FileNotFoundError("Can't find the data/ folder. Run this from inside the repo.")


# Station -> data file in data/. Add a line when a new gauge's CSV lands.
STATIONS = {
    "murray_bridge": "murray_bridge_river_level_historical.csv",
    "morgan": "morgan_river_level.csv",
    "mannum": "Mannum.csv",
}


def _first_line(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.readline()


def _load_murray_bridge(csv_path):
    df = pd.read_csv(csv_path, skiprows=4,
                     names=["datetime", "water_level_m", "conductivity", "water_temp_c"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%H:%M:%S %d/%m/%Y",
                                    errors="coerce")
    return (df.dropna(subset=["datetime", "water_level_m"])
              .sort_values("datetime")
              .reset_index(drop=True))


def _load_water_data_sa(csv_path):
    """Water Data SA 'Bulk Export': 5 header rows, then Timestamp, Value (m), Grade.

    Collapsed to one reading per calendar day so the day-over-day lags in
    build_features stay meaningful (some exports carry sub-daily rows).
    """
    df = pd.read_csv(csv_path, skiprows=5, usecols=[0, 1],
                     names=["datetime", "water_level_m"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")
    df = df.dropna(subset=["datetime", "water_level_m"]).sort_values("datetime")
    df["datetime"] = df["datetime"].dt.normalize()
    return (df.groupby("datetime", as_index=False)["water_level_m"].last()
              .reset_index(drop=True))


def load_data(csv_path=None, station="murray_bridge"):
    """Load a daily river-level series as datetime + water_level_m.

    Reads two formats automatically: the Murray Bridge export (default) and the
    Water Data SA 'Bulk Export' files (Morgan, Mannum). Backwards compatible --
    load_data() with no arguments still returns Murray Bridge exactly as before.

        common.load_data()                  # Murray Bridge
        common.load_data(station="morgan")  # another gauge in data/
        common.load_data(csv_path=...)      # any file, format auto-detected

    Keep reading from the local data/ folder, never a raw GitHub URL from
    another repo -- that is what made the first notebook impossible to re-run.
    """
    if csv_path is None:
        csv_path = _repo_root() / "data" / STATIONS.get(station, station)
    csv_path = Path(csv_path)

    if _first_line(csv_path).lstrip().startswith("#Bulk Export"):
        return _load_water_data_sa(csv_path)
    return _load_murray_bridge(csv_path)


def risk_threshold(df):
    # Level in metres above which a day counts as high risk (~0.806 m).
    return float(df["water_level_m"].quantile(RISK_QUANTILE))


def build_features(df):
    """The four river level features plus the binary label.

    Same formulas the dashboard uses in features_from_levels(), so a model
    trained here can be served there without changes.
    """
    out = df.copy()
    out["level_lag1"] = out["water_level_m"].shift(1)
    out["level_lag2"] = out["water_level_m"].shift(2)
    out["level_roll7"] = out["water_level_m"].rolling(7).mean().shift(1)
    out["level_change3"] = out["water_level_m"].shift(1) - out["water_level_m"].shift(4)
    out["high_risk"] = (out["water_level_m"] >= risk_threshold(df)).astype(int)

    out = out.dropna(subset=FEATURES).reset_index(drop=True)
    return out[FEATURES], out["high_risk"]


def chronological_split(X, y, return_val=False):
    """Split by date, oldest first. Never shuffle.

    With lagged features a random split leaks: a test day sitting between two
    training days carries nearly the same information, so the model scores
    well on days it has effectively already seen.
    """
    n = len(X)
    i_test = int(n * (1 - TEST_FRACTION))
    i_val = int(n * (1 - TEST_FRACTION - VAL_FRACTION))

    if return_val:
        return (X.iloc[:i_val], X.iloc[i_val:i_test], X.iloc[i_test:],
                y.iloc[:i_val], y.iloc[i_val:i_test], y.iloc[i_test:])
    return X.iloc[:i_test], X.iloc[i_test:], y.iloc[:i_test], y.iloc[i_test:]


def cv_splitter():
    # TimeSeriesSplit, not StratifiedKFold. Stratifying over a time series
    # mixes future into past, which is the leak we are trying to avoid.
    # We explain this substitution in the report.
    return TimeSeriesSplit(n_splits=N_SPLITS)


def nse(y_true, y_prob):
    # Nash Sutcliffe against the predicted probability. Note in the report
    # that classic NSE is for continuous streamflow, not a binary classifier.
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return 1 - np.sum((y_true - y_prob) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def persistence_baseline(df, y_subset):
    """Predict each day as whatever yesterday was. No model at all.

    Scores about F1 0.796 on the test split, so it is the number our models
    have to beat. Pass the Series from chronological_split with its index
    intact, since the index is what lines each day up with the day before.
    """
    if not isinstance(y_subset, pd.Series):
        raise TypeError(
            f"Pass the Series from chronological_split, not a "
            f"{type(y_subset).__name__}. The index is needed to align days.")

    labels = (df["water_level_m"] >= risk_threshold(df)).astype(int).to_numpy()
    offset = len(df) - len(build_features(df)[0])  # rows dropped for missing history
    yesterday = [labels[i + offset - 1] for i in y_subset.index]

    return {"model": "Persistence baseline",
            "F1": f1_score(y_subset, yesterday),
            "MCC": matthews_corrcoef(y_subset, yesterday)}


def evaluate(name, y_true, y_pred, y_prob):
    """All five metrics from the plan. y_pred is the 0/1 call, y_prob the probability."""
    return {"model": name,
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_prob))),
            "Brier": brier_score_loss(y_true, y_prob),
            "NSE": nse(y_true, y_prob)}


def comparison_table(results):
    # results is a list of evaluate() dicts.
    return pd.DataFrame(results).set_index("model").round(3)


def _self_check():
    df = load_data()
    print(f"Rows: {len(df)}  ({df.datetime.min().date()} to {df.datetime.max().date()})")
    print(f"Risk threshold: {risk_threshold(df):.3f} m")

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"Train: {len(X_train)} rows, {y_train.mean():.3f} positive")
    print(f"Test:  {len(X_test)} rows, {y_test.mean():.3f} positive")

    base = persistence_baseline(df, y_test)
    print(f"\nBaseline to beat: F1 {base['F1']:.3f}, MCC {base['MCC']:.3f}")


if __name__ == "__main__":
    _self_check()
