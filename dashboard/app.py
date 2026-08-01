"""
Flood Risk Dashboard (Streamlit) — single-screen operations view
===============================================================
River Murray flood-risk prototype. Serves the team's trained Logistic
Regression and Random Forest models (next-day high-river-level risk for
Murray Bridge).

Layout (single screen, no scrolling):
    Header      title | Live | time | operator | horizon tabs
    Left card   catchment map (modelled + context stations)
    Right cards 1) selected station: gauge + uncertainty band
                2) why this score (feature contributions / importance)
                3) alert authorisation (human-in-the-loop + audit log)

Honesty notes:
- Only Murray Bridge (A4261162) has a trained model.
- Morgan (A4260554) has a real daily series (Water Data SA, 2009-2026) but its
  flood events are all historical, so it is shown as level monitoring (current
  level vs its own threshold), not a validated prediction.
- The remaining stations are context markers with no data yet.
- 24/48/72h horizons extrapolate the recent level trend, then score the model.
- Two models are available: Logistic Regression (Manuela) and Random Forest
  (Ghale). Card 2 adapts automatically: signed feature contributions for the
  Logistic Regression, feature importance for the Random Forest.
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

TRAIN_RISK_THRESHOLD_M = 0.806
FEATURES = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]
FEATURE_LABEL = {
    "level_lag1": "Level yesterday",
    "level_lag2": "Level 2 days ago",
    "level_roll7": "7-day average level",
    "level_change3": "3-day level change",
}
BAND_HEX = {"Low": "#12a150", "Moderate": "#e0982b", "High": "#d64545"}
BAND_COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}
# Colour-blind-safe band metadata: every band carries an icon + light tint + dark
# text so meaning never depends on colour alone.
BAND_META = {
    "Low":      {"hex": "#12a150", "bg": "#e6f5ec", "tx": "#0c6b39", "icon": "✓", "advice": "no action needed"},
    "Moderate": {"hex": "#e0982b", "bg": "#fdf2df", "tx": "#8a5a0b", "icon": "◆", "advice": "monitor closely"},
    "High":     {"hex": "#d64545", "bg": "#fbe9e9", "tx": "#a12b2b", "icon": "⚠", "advice": "operator review recommended"},
}

MODELLED = {"name": "Murray Bridge", "id": "A4261162", "lat": -35.12, "lon": 139.27}
OPERATOR = "Authorized Person"

# Predefined alert recipients (organisations). For the prototype these map to the
# team's own inboxes as mockups so alerts actually deliver during testing.
RECIPIENT_DIRECTORY = [
    {"label": "Department for Environment and Water (DEW)", "email": "manur9669@gmail.com"},
    {"label": "SA Police (SAPOL), Murray Bridge", "email": "julieth.sanjju@gmail.com"},
    {"label": "Mid Murray Council", "email": "alejandragn2323@gmail.com"},
    {"label": "SA Bureau of Meteorology", "email": "ghale.phurku@gmail.com"},
]
DEFAULT_RECIPIENTS = ["Department for Environment and Water (DEW)", "SA Police (SAPOL), Murray Bridge"]
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(addr: str) -> bool:
    return bool(_EMAIL_RE.match((addr or "").strip()))
CONTEXT_STATIONS = [
    {"name": "Renmark", "lat": -34.1746, "lon": 140.7461},
    {"name": "Berri", "lat": -34.2833, "lon": 140.60},
    {"name": "Loxton", "lat": -34.45, "lon": 140.57},
    {"name": "Waikerie", "lat": -34.18, "lon": 139.98},
    {"name": "Blanchetown", "lat": -34.35, "lon": 139.62},
    {"name": "Mannum", "lat": -34.91, "lon": 139.31},
]

# Stations with a real daily series but no validated model -> shown as level
# monitoring: current level vs the station's own 0.80-quantile threshold.
MONITORED_STATIONS = [
    {"name": "Morgan", "id": "A4260554", "lat": -34.03, "lon": 139.49,
     "file": "morgan_river_level.csv"},
]

FALLBACK_LEVELS = [0.725, 0.689, 0.731, 0.661, 0.653, 0.647, 0.616, 0.604, 0.654,
                   0.631, 0.592, 0.668, 0.644, 0.638, 0.667, 0.664, 0.686, 0.755,
                   0.733, 0.726, 0.713, 0.717, 0.718, 0.718, 0.7, 0.668, 0.648,
                   0.686, 0.694, 0.73]
FALLBACK_BASELINE = {"level_lag1": 0.693, "level_lag2": 0.693, "level_roll7": 0.698, "level_change3": -0.001}
FALLBACK_COEF = {"level_lag1": 13.095, "level_lag2": 3.991, "level_roll7": 9.148, "level_change3": 1.235}
FALLBACK_INTERCEPT = -20.165

# Where to look for each model, in order of preference. The app only offers
# the models it actually finds, so nothing breaks if a file is missing.
MODEL_FILES = {
    "Logistic Regression": [
        "backend/models/logistic_regression_real.joblib",
    ],
    "Random Forest": [
        "backend/models/random_forest.joblib",
    ],
    "XGBoost": [
        "backend/models/xgboost.joblib",
    ],
}
@st.cache_resource
def load_models():
    """Load every model we can find. Missing files are skipped, not fatal."""
    here = Path(__file__).resolve().parent
    repo = here.parent
    models = {}
    for name, candidates in MODEL_FILES.items():
        for rel in candidates:
            p = repo / rel
            if p.exists():
                try:
                    import joblib
                    models[name] = joblib.load(p)
                    break
                except Exception:
                    pass
    return models


SHAP_ID = {"Random Forest": "random_forest", "XGBoost": "xgboost", "LSTM": "lstm"}


@st.cache_data
def load_shap_importance(model_name):
    """Precomputed SHAP importance (shap/<id>.npy) as {feature: value}, or None.

    Ghale precomputes these with TreeExplainer / GradientExplainer in
    notebooks/precompute_shap.py so the dashboard shows SHAP without recomputing
    it on every reload.
    """
    sid = SHAP_ID.get(model_name)
    if not sid:
        return None
    p = Path(__file__).resolve().parent.parent / "shap" / f"{sid}.npy"
    if not p.exists():
        return None
    vals = np.load(p)
    if len(vals) == len(FEATURES):
        return dict(zip(FEATURES, [float(v) for v in vals]))
    return None


@st.cache_data
def load_history():
    here = Path(__file__).resolve().parent
    p = here.parent / "data" / "murray_bridge_river_level_historical.csv"
    try:
        df = pd.read_csv(p, skiprows=4, names=["datetime", "water_level_m", "conductivity", "water_temp_c"])
        df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")
        df = df.dropna(subset=["water_level_m"])
        df = df[(df["water_level_m"] > -1) & (df["water_level_m"] < 6)]
        levels = df["water_level_m"].tail(30).round(3).tolist()
        d = df.copy()
        d["level_lag1"] = d["water_level_m"].shift(1)
        d["level_lag2"] = d["water_level_m"].shift(2)
        d["level_roll7"] = d["water_level_m"].shift(1).rolling(7).mean()
        d["level_change3"] = d["water_level_m"].shift(1) - d["water_level_m"].shift(4)
        d = d.dropna(subset=FEATURES)
        base = {f: round(float(d[f].median()), 3) for f in FEATURES}
        if len(levels) >= 8:
            return levels, base
    except Exception:
        pass
    return FALLBACK_LEVELS, FALLBACK_BASELINE


@st.cache_data
def load_monitored():
    """Level status for stations with real data but no validated model.

    Reads the Water Data SA 'Bulk Export' format (Morgan), collapses to one
    reading per day, and bands the latest level against the station's own
    0.80-quantile threshold. No model, no prediction — just monitoring.
    """
    here = Path(__file__).resolve().parent
    out = []
    for s in MONITORED_STATIONS:
        p = here.parent / "data" / s["file"]
        try:
            df = pd.read_csv(p, skiprows=5, usecols=[0, 1], names=["datetime", "level"])
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna().sort_values("datetime")
            df["datetime"] = df["datetime"].dt.normalize()
            df = df.groupby("datetime", as_index=False)["level"].last()
            if len(df) < 8:
                continue
            latest = float(df["level"].iloc[-1])
            threshold = float(df["level"].quantile(0.80))
            band = "High" if latest >= threshold else "Moderate" if latest >= 0.9 * threshold else "Low"
            out.append({**s, "latest": latest, "threshold": threshold, "band": band,
                        "recent": df["level"].tail(30).round(3).tolist(),
                        "asof": df["datetime"].iloc[-1].date().isoformat()})
        except Exception:
            pass
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_api_models(api_url):
    """Models offered by the backend's GET /models as {id, name, available}
    dicts, or None if unavailable. Availability lets us list API-only models
    (e.g. the LSTM, which has no local joblib) and route their prediction
    through the API while still using local objects for models we have.
    """
    if not api_url:
        return None
    try:
        r = requests.get(api_url.rstrip("/") + "/models", timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("models", [])
        out = []
        for d in data:
            if isinstance(d, dict) and (d.get("name") or d.get("id")):
                out.append({"id": d.get("id"), "name": d.get("name", d.get("id")),
                            "available": d.get("available", True)})
        return out or None
    except Exception:
        return None


def final_estimator(model):
    """The estimator itself, unwrapping a Pipeline if there is one."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _pre_transform(model, frame):
    """Run the pipeline's pre-processing steps (e.g. StandardScaler) over a frame."""
    if hasattr(model, "steps"):
        for _, step in model.steps[:-1]:
            frame = step.transform(frame)
    return frame


def coef_map(model):
    est = final_estimator(model) if model is not None else None
    if est is not None and hasattr(est, "coef_"):
        return {f: float(est.coef_[0][i]) for i, f in enumerate(FEATURES)}
    return FALLBACK_COEF


def signed_contributions(model, feats, baseline):
    """Per-feature contribution for a linear model, or None if the model is not linear.

    The Logistic Regression is a Pipeline with a StandardScaler, so its coefficients
    live in scaled space. Both the current point and the baseline are pushed through
    the same scaler before the difference is taken, otherwise the contributions are
    off by the scaling factor.
    """
    est = final_estimator(model) if model is not None else None
    if est is None or not hasattr(est, "coef_"):
        return None
    x = pd.DataFrame([[feats[f] for f in FEATURES]], columns=FEATURES)
    b = pd.DataFrame([[baseline[f] for f in FEATURES]], columns=FEATURES)
    xs = np.asarray(_pre_transform(model, x)).ravel()
    bs = np.asarray(_pre_transform(model, b)).ravel()
    coefs = est.coef_[0]
    return {f: float(coefs[i] * (xs[i] - bs[i])) for i, f in enumerate(FEATURES)}


def features_from_levels(levels):
    return {"level_lag1": levels[-1], "level_lag2": levels[-2],
            "level_roll7": sum(levels[-7:]) / len(levels[-7:]),
            "level_change3": levels[-1] - levels[-4]}


def band_of(p):
    return "High" if p >= 0.66 else "Moderate" if p >= 0.33 else "Low"


def predict_local(model, feats):
    if model is not None:
        row = pd.DataFrame([[feats[f] for f in FEATURES]], columns=FEATURES)
        return float(model.predict_proba(row)[0][1])
    z = FALLBACK_INTERCEPT + sum(FALLBACK_COEF[f] * feats[f] for f in FEATURES)
    return 1.0 / (1.0 + math.exp(-z))


def predict(api_url, levels, model, model_id=None):
    feats = features_from_levels(levels)
    if api_url:
        try:
            body = {"levels": levels, "station_id": MODELLED["id"]}
            if model_id:
                body["model"] = model_id          # route to the chosen model (incl. LSTM)
            r = requests.post(api_url.rstrip("/") + "/predict_series", json=body, timeout=30)
            r.raise_for_status()
            b = r.json()
            return b["flood_probability"], b["risk_band"], feats, b.get("model", "api")
        except Exception as exc:
            if model is None:
                st.warning(f"'{model_id}' is served by the API and it's unreachable ({exc}); "
                           "showing a rough local estimate instead.")
            else:
                st.warning(f"API unreachable, using local model ({exc}).")
    p = predict_local(model, feats)
    return p, band_of(p), feats, "local model"


def dispatch_alert(api_url, payload):
    """POST an authorised alert to the backend /alerts endpoint (SendGrid email + audit log).

    Returns (ok, detail). The /alerts endpoint is built by the backend teammate; until it
    exists (or when no API URL is set) this returns (False, reason) and the UI falls back to
    a local confirmation so the demo still works.
    """
    if not api_url:
        return False, "no API URL set"
    try:
        r = requests.post(api_url.rstrip("/") + "/alerts", json=payload, timeout=30)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:
        return False, str(exc)


def apply_scenario(base_levels, scenario, offset):
    s = list(base_levels)
    n = len(s)
    if scenario == "Rising river":
        s = [v + (i / n) * 0.5 for i, v in enumerate(s)]
    elif scenario == "Flood watch":
        s = [v + (i / n) * 1.1 for i, v in enumerate(s)]
    return [round(v + offset, 3) for v in s]


def project(levels, days):
    """Extrapolate the recent trend forward `days` days."""
    if days <= 0:
        return list(levels)
    y = levels[-7:]
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    out = list(levels)
    for _ in range(days):
        out.append(round(out[-1] + slope, 3))
    return out


def gauge_svg(p, color):
    r, length = 72, math.pi * 72
    off = length * (1 - p)
    return f"""
    <svg width="100%" height="104" viewBox="0 0 184 104" preserveAspectRatio="xMidYMid meet">
      <path d="M20 96 A72 72 0 0 1 164 96" fill="none" stroke="#eef2f7" stroke-width="15" stroke-linecap="round"/>
      <path d="M20 96 A72 72 0 0 1 164 96" fill="none" stroke="{color}" stroke-width="15"
            stroke-linecap="round" stroke-dasharray="{length:.1f}" stroke-dashoffset="{off:.1f}"/>
      <text x="92" y="84" text-anchor="middle" font-size="30" font-weight="700" fill="#0f2438">{round(p*100)}%</text>
      <text x="92" y="99" text-anchor="middle" font-size="10" fill="#7a8aa0">flood probability</text>
    </svg>"""


def band_bar(lo, hi, p):
    return f"""
    <div style="position:relative;height:10px;border-radius:5px;background:#eef2f7;margin-top:4px">
      <div style="position:absolute;left:{lo*100:.0f}%;width:{max(hi-lo,0.01)*100:.0f}%;top:0;bottom:0;
                  background:#cfe1fb;border-radius:5px"></div>
      <div style="position:absolute;left:{p*100:.0f}%;top:-3px;width:3px;height:16px;background:#1f6feb;border-radius:2px"></div>
    </div>"""


def trend_of(levels):
    """Direction of the recent series: returns (label, key)."""
    if len(levels) < 4:
        return "▬ Steady", "steady"
    recent = levels[-7:]
    delta = recent[-1] - recent[0]
    if delta > 0.03:
        return "▲ Rising", "rising"
    if delta < -0.03:
        return "▼ Easing", "falling"
    return "▬ Steady", "steady"


def spark_svg(levels, color):
    """A tiny inline sparkline of the recent level series."""
    pts = list(levels[-14:]) or [0, 0]
    if len(pts) < 2:
        pts = pts * 2
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    n = len(pts)
    coords = " ".join(f"{i/(n-1)*120:.0f},{22 - (v-lo)/rng*20:.1f}" for i, v in enumerate(pts))
    return (f'<svg width="100%" height="26" viewBox="0 0 120 26" preserveAspectRatio="none">'
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/></svg>')


# --------------------------------------------------------------------------
st.set_page_config(page_title="River Murray Flood Risk", page_icon="🌊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 0.6rem; max-width: 1500px;}
  h1, h2, h3 {margin-bottom: .2rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.15rem;}
  [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
  div[data-testid="stHorizontalBlock"] {gap: .8rem;}
  .badge {display:inline-block;padding:3px 9px;border-radius:7px;font-size:12px;font-weight:600}
  .card-title {font-size:13px;font-weight:600;color:#0f2438;margin-bottom:2px}
  .muted {color:#7a8aa0;font-size:11px}
  .row {display:flex;justify-content:space-between;font-size:12px;color:#4a5b70;margin:2px 0}
  /* Responsive: stack columns on narrow screens (tablet/phone). */
  @media (max-width: 720px){
    div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;}
    div[data-testid="stHorizontalBlock"] > div{min-width:100% !important;}
    .block-container{padding-left:.6rem;padding-right:.6rem;}
  }
</style>
""", unsafe_allow_html=True)

models = load_models()
if not models:
    st.error("No trained model found in the repo. Run the notebooks to create "
             "backend/models/logistic_regression_real.joblib and/or backend/models/random_forest.joblib.")
    st.stop()

base_levels, baseline = load_history()
try:
    default_api = st.secrets.get("API_URL", os.getenv("API_URL", ""))
except Exception:
    default_api = os.getenv("API_URL", "")

with st.sidebar:
    st.header("Controls")
    api_url = st.text_input("API URL (optional)", value=default_api,
                            placeholder="https://flood-risk-api.onrender.com")

    # The dropdown is populated from GET /models when the backend is reachable,
    # and falls back to whatever joblib files are in the repo. Either way the
    # prediction below uses the locally loaded model object.
    api_models = fetch_api_models(api_url)
    if api_models:
        available = [m for m in api_models if m.get("available")]
        options = [m["name"] for m in available] or list(models.keys())
        name_to_id = {m["name"]: m.get("id") for m in api_models}
    else:
        options = list(models.keys())
        name_to_id = {}
    model_choice = st.selectbox("Model", options, index=0)
    model = models.get(model_choice)          # None for API-only models (e.g. LSTM)
    model_id = name_to_id.get(model_choice)   # sent to the API so it uses this model
    st.caption(f"Model list from API GET /models ({len(api_models)} offered)"
               if api_models else "Model list from the joblib files in the repo.")

    scenario = st.radio("River condition", ["Recent (actual)", "Rising river", "Flood watch"])
    offset = st.slider("Level offset (m)", -0.30, 1.50, 0.0, 0.05)
    with st.expander("About the model"):
        _thr = f"'Flood' = level at/above {TRAIN_RISK_THRESHOLD_M} m (0.80 quantile)."
        if model_choice == "Logistic Regression":
            st.write(f"Logistic Regression on Murray Bridge river levels. {_thr} "
                     "Horizons extrapolate the recent trend.")
        elif model_choice == "XGBoost":
            st.write(f"XGBoost (Optuna-tuned) on Murray Bridge river levels. {_thr} "
                     "Trained on the chronological split from common.py.")
        elif model_choice == "LSTM":
            st.write("LSTM on the last 14 daily levels (a sequence model, served by the API). "
                     f"{_thr} Trained on the chronological split; scored via /predict_series.")
        else:
            st.write(f"Random Forest (Optuna-tuned) on Murray Bridge river levels. {_thr} "
                     "Trained on the chronological split from common.py.")

# ---- Header ----
h1, h2 = st.columns([2.2, 1])
with h1:
    st.markdown("")
    st.markdown("### 🌊 River Murray Flood Risk")
    st.markdown("<span class='muted'>SES &amp; Local Council decision support · ITA602 prototype</span>",
                unsafe_allow_html=True)
with h2:
    st.markdown(
        f"<div style='text-align:right;margin-top:6px'>"
        f"<span class='badge' style='background:#e6f5ec;color:#0c6b39'>● Live</span> "
        f"<span class='badge' style='background:#f2f5f9;color:#4a5b70'>{datetime.now().strftime('%H:%M')}</span> "
        f"<span class='badge' style='background:#eaf3ff;color:#0c447c'>Operator: {OPERATOR}</span></div>",
        unsafe_allow_html=True)

horizon = st.radio("Forecast horizon", ["24h", "48h", "72h"], index=1,
                   horizontal=True, label_visibility="collapsed")
days = {"24h": 1, "48h": 2, "72h": 3}[horizon]

levels = project(apply_scenario(base_levels, scenario, offset), days)
prob, band, feats, source = predict(api_url, levels, model, model_id)
if model is not None:
    lo = predict_local(model, features_from_levels([v - 0.05 for v in levels]))
    hi = predict_local(model, features_from_levels([v + 0.05 for v in levels]))
else:
    lo = hi = prob   # API-only model (e.g. LSTM): no local ±5 cm band available
blo, bhi = min(lo, hi), max(lo, hi)

# ---- Main: map | right column ----
left, right = st.columns([1.55, 1], gap="medium")

with left:
    with st.container(border=True):
        t1, t2 = st.columns([1, 1])
        t1.markdown("<div class='card-title'>Catchment overview</div>", unsafe_allow_html=True)
        t2.markdown(
            "<div style='text-align:right;font-size:11px'>"
            "<span style='color:#12a150'>●</span> Low "
            "<span style='color:#e0982b'>●</span> Moderate "
            "<span style='color:#d64545'>●</span> High "
            "&nbsp;<b>◯</b> monitored "
            "<span style='color:#9aa6bd'>●</span> no data</div>", unsafe_allow_html=True)
        fmap = folium.Map(location=[-34.6, 139.9], zoom_start=7, tiles="CartoDB positron")

        # South Australia LGA boundaries, coloured by the current flood-risk band.
        # Look in a few likely spots (repo data/ first, where the simplified file lives)
        # and fail quietly if it isn't present, so the rest of the map keeps working.
        boundary_candidates = [
            Path(__file__).resolve().parent.parent / "data" / "lga_boundaries.geojson",
            Path(__file__).resolve().parent / "data" / "lga_boundaries.geojson",
            Path(__file__).resolve().parent / "lga_boundaries.geojson",
        ]
        boundary_path = next((p for p in boundary_candidates if p.exists() and p.stat().st_size > 0), None)
        if boundary_path:
            try:
                with open(boundary_path, "r", encoding="utf-8") as fh:
                    boundary_geo = json.load(fh)
                if boundary_geo.get("features"):
                    polygon_colour = BAND_COLOR[band]
                    folium.GeoJson(
                        boundary_geo,
                        name="LGA boundaries",
                        style_function=lambda feature: {
                            "fillColor": polygon_colour,
                            "color": polygon_colour,
                            "weight": 2,
                            "fillOpacity": 0.25,
                        },
                        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["LGA:"]),
                    ).add_to(fmap)
            except (json.JSONDecodeError, OSError):
                pass  # data missing or malformed — skip the layer, keep the map working

        for s in CONTEXT_STATIONS:
            folium.CircleMarker([s["lat"], s["lon"]], radius=6, color="#9aa6bd", fill=True,
                                fill_color="#9aa6bd", fill_opacity=0.8,
                                tooltip=f"{s['name']} — context only (no data yet)").add_to(fmap)
        monitored = load_monitored()
        for s in monitored:
            folium.CircleMarker([s["lat"], s["lon"]], radius=9, color="#ffffff", weight=2,
                                fill=True, fill_color=BAND_COLOR[s["band"]], fill_opacity=0.9,
                                tooltip=(f"{s['name']} (monitored): {s['band']} · "
                                         f"{s['latest']:.2f} m vs {s['threshold']:.2f} m threshold "
                                         f"· as of {s['asof']}")).add_to(fmap)
        folium.CircleMarker([MODELLED["lat"], MODELLED["lon"]], radius=12, color=BAND_COLOR[band],
                            fill=True, fill_color=BAND_COLOR[band], fill_opacity=0.9,
                            tooltip=f"{MODELLED['name']} (modelled): {band} ({prob*100:.0f}%)").add_to(fmap)
        st_folium(fmap, height=395, use_container_width=True, returned_objects=[])
        n_total = len(CONTEXT_STATIONS) + len(monitored) + 1
        mon_names = ", ".join(s["name"] for s in monitored) or "—"
        st.markdown(f"<div class='muted'>{n_total} stations · modelled: Murray Bridge · "
                    f"monitored: {mon_names} · sources: BoM, DEW, Water Data SA</div>",
                    unsafe_allow_html=True)
        if monitored:
            chips = " ".join(
                f"<span class='badge' style='background:{BAND_HEX[s['band']]};color:#fff'>"
                f"{s['name']} {s['latest']:.2f} m · {s['band']}</span>" for s in monitored)
            st.markdown("<div style='margin-top:6px;font-size:11px'>"
                        "<span class='muted'>Level monitoring (latest vs own threshold): </span>"
                        f"{chips}</div>", unsafe_allow_html=True)
        with st.expander("30-day level trend"):
            tdf = pd.DataFrame({"days ago": list(range(-len(levels) + 1, 1)), "level (m)": levels})
            line = alt.Chart(tdf).mark_line(point=False, color="#1f6feb").encode(
                x=alt.X("days ago:Q", title=None),
                y=alt.Y("level (m):Q", title="m", scale=alt.Scale(zero=False)))
            rule = alt.Chart(pd.DataFrame({"y": [TRAIN_RISK_THRESHOLD_M]})).mark_rule(
                color="#d64545", strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=150), use_container_width=True)

with right:
    # Decision-first verdict banner (the focal point), then supporting detail.
    bm = BAND_META[band]
    tlabel, tdir = trend_of(levels)
    tword = {"rising": "rising", "falling": "easing", "steady": "steady"}[tdir]
    st.markdown(
        f"<div style='border-radius:13px;padding:14px 16px;border-left:7px solid {bm['hex']};"
        f"background:{bm['bg']};color:{bm['tx']};display:flex;justify-content:space-between;"
        f"align-items:center;gap:12px;flex-wrap:wrap'>"
        f"<div><div style='font-size:22px;font-weight:700;display:flex;align-items:center;gap:8px'>"
        f"{bm['icon']} {band} risk</div>"
        f"<div style='font-size:12px;margin-top:2px;opacity:.85'>{MODELLED['name']} · {tword} · {bm['advice']}</div></div>"
        f"<div style='text-align:right'><div style='font-size:34px;font-weight:800;line-height:1'>{prob*100:.0f}%</div>"
        f"<div style='font-size:10px;opacity:.8'>flood probability · next {horizon}</div></div></div>",
        unsafe_allow_html=True)

    with st.container(border=True):
        d1, d2, d3 = st.columns([1, 1, 1])
        with d1:
            st.markdown(gauge_svg(prob, bm["hex"]), unsafe_allow_html=True)
        with d2:
            st.markdown(f"<div class='muted'>Trend</div>"
                        f"<div style='font-size:14px;font-weight:700'>{tlabel}</div>"
                        f"{spark_svg(levels, bm['hex'])}", unsafe_allow_html=True)
        with d3:
            st.markdown(f"<div class='muted'>Latest level</div>"
                        f"<div style='font-size:14px;font-weight:700'>{levels[-1]:.2f} m</div>"
                        f"<div class='muted'>{levels[-1]-TRAIN_RISK_THRESHOLD_M:+.2f} m vs threshold</div>",
                        unsafe_allow_html=True)
        st.markdown(f"<div class='row' style='margin-top:8px'><span>Uncertainty band (±5 cm gauge error)</span>"
                    f"<b>{blo*100:.0f}% – {bhi*100:.0f}%</b></div>{band_bar(blo, bhi, prob)}",
                    unsafe_allow_html=True)

    # Card 2 — explainability (adapts to the selected model)
    with st.container(border=True):
        contribs = signed_contributions(model, feats, baseline)
        estimator = final_estimator(model)
        shap_imp = load_shap_importance(model_choice)
        if contribs is not None:
            st.markdown("<div class='card-title'>Why this score</div>"
                        "<div class='muted'>per-feature contribution (exact for logistic regression)</div>",
                        unsafe_allow_html=True)
            contrib = pd.DataFrame([{"feature": ("▲ " if contribs[f] >= 0 else "▼ ") + FEATURE_LABEL[f],
                                     "contribution": round(contribs[f], 3)}
                                    for f in FEATURES])
            contrib["direction"] = contrib["contribution"].apply(
                lambda v: "Increases risk" if v >= 0 else "Decreases risk")
            bars = alt.Chart(contrib).mark_bar().encode(
                x=alt.X("contribution:Q", title=None),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("direction:N",
                                scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                                range=["#d64545", "#1f6feb"]),
                                legend=alt.Legend(orient="bottom", title=None, labelFontSize=10)),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("contribution:Q", format="+.3f")],
            ).properties(height=132)
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3ccd8").encode(x="x:Q")
            st.altair_chart(zero + bars, use_container_width=True)
        elif shap_imp is not None or hasattr(estimator, "feature_importances_"):
            if shap_imp is not None:
                sub = "SHAP feature importance (precomputed · mean |impact|)"
                imp_by_feat = shap_imp
            else:
                sub = "feature importance (magnitude only, not signed)"
                imp_by_feat = {f: float(estimator.feature_importances_[i]) for i, f in enumerate(FEATURES)}
            st.markdown(f"<div class='card-title'>Why this score</div>"
                        f"<div class='muted'>{sub}</div>", unsafe_allow_html=True)
            importances = pd.DataFrame([{"feature": FEATURE_LABEL[f],
                                         "importance": round(imp_by_feat[f], 3)}
                                        for f in FEATURES])
            bars = alt.Chart(importances).mark_bar(color="#1f6feb").encode(
                x=alt.X("importance:Q", title=None),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".3f")],
            ).properties(height=132)
            st.altair_chart(bars, use_container_width=True)
        else:
            st.caption("No explainability view available for this model.")

    # Card 3 — alert authorisation (deliberate two-step confirm)
    with st.container(border=True):
        st.markdown("<div class='card-title'>Alert authorisation</div>", unsafe_allow_html=True)
        if "alert_state" not in st.session_state:
            st.session_state.alert_state = "pending"
        state = st.session_state.alert_state

        if band == "Low":
            st.markdown("<div class='row' style='background:#f2f5f9;padding:6px 9px;border-radius:7px'>"
                        "No alert proposed at this level</div>"
                        "<div class='muted' style='margin-top:8px'>Audit log #A-2291 · tamper-evident · "
                        "retained per State Records Act 1997 (SA)</div>", unsafe_allow_html=True)
        elif state == "sent":
            res = st.session_state.get("alert_result", {"ok": False, "detail": "local"})
            chosen = ", ".join(st.session_state.get("alert_recipient_labels", [])) or "—"
            if res["ok"]:
                d = res["detail"] if isinstance(res["detail"], dict) else {}
                aid = d.get("alert_id", d.get("id", "A-2292"))
                email_status = d.get("email_status", "")
                st.markdown(f"<div class='row' style='background:#e6f5ec;color:#0c6b39;padding:6px 9px;"
                            f"border-radius:7px'>✓ Alert sent via backend · {datetime.now().strftime('%H:%M')} "
                            f"· authorised by {OPERATOR}</div>"
                            f"<div class='row'><span class='muted'>Delivered</span><span>{chosen}</span></div>"
                            + (f"<div class='row'><span class='muted'>Email</span><span>{email_status}</span></div>"
                               if email_status else ""),
                            unsafe_allow_html=True)
                st.markdown(f"<div class='muted'>Audit log #{aid} · signed by {OPERATOR} · "
                            "retained per State Records Act 1997 (SA)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='row' style='background:#fdf2df;color:#8a5a0b;padding:6px 9px;"
                            f"border-radius:7px'>⚠ Authorised locally · backend /alerts not connected</div>"
                            f"<div class='muted' style='margin-top:6px'>Reason: {res['detail']}. The button POSTs "
                            "to /alerts and will send real emails once that endpoint is live.</div>",
                            unsafe_allow_html=True)
            if st.button("Reset", use_container_width=True):
                st.session_state.alert_state = "pending"
                st.rerun()
        elif state == "confirm":
            to_line = ", ".join(st.session_state.get("alert_recipient_labels", [])) or "—"
            st.markdown(f"<div class='row' style='background:#fbe9e9;color:#a12b2b;padding:6px 9px;"
                        f"border-radius:7px'>Confirm dispatch of a {band} flood alert?</div>"
                        f"<div class='row'><span class='muted'>To</span><span>{to_line}</span></div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            if c1.button("Confirm dispatch", type="primary", use_container_width=True):
                payload = {
                    "station_id": MODELLED["id"],
                    "station_name": MODELLED["name"],
                    "risk_band": band,
                    "flood_probability": round(prob, 4),
                    "horizon": horizon,
                    "operator": OPERATOR,
                    "message": f"{band} flood risk ({prob*100:.0f}%) for {MODELLED['name']} over the next {horizon}.",
                    "recipients": st.session_state.get("alert_recipients", []),
                }
                ok, detail = dispatch_alert(api_url, payload)
                st.session_state.alert_result = {"ok": ok, "detail": detail}
                st.session_state.alert_state = "sent"
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.alert_state = "pending"
                st.rerun()
            st.markdown("<div class='muted'>This action is logged against your operator ID.</div>",
                        unsafe_allow_html=True)
        else:  # pending
            st.markdown("<div class='row' style='background:#fdf2df;color:#8a5a0b;padding:6px 9px;"
                        "border-radius:7px'>⚠ Awaiting human authorisation</div>", unsafe_allow_html=True)
            orgs = st.multiselect("Recipient organisations",
                                  options=[o["label"] for o in RECIPIENT_DIRECTORY],
                                  default=DEFAULT_RECIPIENTS, key="alert_orgs")
            custom = st.text_input("Additional email (optional)", key="alert_custom",
                                   placeholder="professor@university.edu")
            st.markdown("<div class='row'><span class='muted'>Channel</span>"
                        "<span>Email (SendGrid) · SMS descoped</span></div>", unsafe_allow_html=True)
            if st.button("Authorise & dispatch alert", type="primary", use_container_width=True):
                emails, labels = [], []
                for o in RECIPIENT_DIRECTORY:
                    if o["label"] in orgs:
                        emails.append(o["email"])
                        labels.append(o["label"])
                extra = (custom or "").strip()
                if extra and not is_valid_email(extra):
                    st.error("That additional email doesn't look valid.")
                elif not emails and not extra:
                    st.error("Select at least one recipient, or add a custom email.")
                else:
                    if extra:
                        emails.append(extra)
                        labels.append(extra)
                    st.session_state.alert_recipients = emails
                    st.session_state.alert_recipient_labels = labels
                    st.session_state.alert_state = "confirm"
                    st.rerun()
            st.markdown("<div class='muted'>Audit log · tamper-evident · "
                        "retained per State Records Act 1997 (SA)</div>", unsafe_allow_html=True)
