// Domain constants and helpers. The scenario and projection maths mirror
// dashboard/app.py so the React screen and the Streamlit prototype tell the
// same story during the demo.

export const TRAIN_RISK_THRESHOLD_M = 0.806;

export const MODELLED = { name: "Murray Bridge", id: "A4261162", lat: -35.12, lon: 139.27 };

export const OPERATOR = "Authorized Person";

export const FEATURES = ["level_lag1", "level_lag2", "level_roll7", "level_change3"];

export const FEATURE_LABEL = {
  level_lag1: "Level yesterday",
  level_lag2: "Level 2 days ago",
  level_roll7: "7-day average level",
  level_change3: "3-day level change",
};

export const BAND_META = {
  Low: { hex: "#12a150", bg: "#e6f5ec", tx: "#0c6b39", icon: "✓", advice: "no action needed" },
  Moderate: { hex: "#e0982b", bg: "#fdf2df", tx: "#8a5a0b", icon: "◆", advice: "monitor closely" },
  High: { hex: "#d64545", bg: "#fbe9e9", tx: "#a12b2b", icon: "⚠", advice: "operator review recommended" },
};

// Prototype recipients: the team's own inboxes, so alerts actually deliver in testing.
export const RECIPIENT_DIRECTORY = [
  { label: "Department for Environment and Water (DEW)", email: "manur9669@gmail.com" },
  { label: "SA Police (SAPOL), Murray Bridge", email: "julieth.sanjju@gmail.com" },
  { label: "Mid Murray Council", email: "alejandragn2323@gmail.com" },
  { label: "SA Bureau of Meteorology", email: "ghale.phurku@gmail.com" },
];

export const DEFAULT_RECIPIENTS = [
  "Department for Environment and Water (DEW)",
  "SA Police (SAPOL), Murray Bridge",
];

// Recent Murray Bridge daily levels (m), oldest first. Used as the starting series.
export const BASE_LEVELS = [
  0.725, 0.689, 0.731, 0.661, 0.653, 0.647, 0.616, 0.604, 0.654, 0.631, 0.592, 0.668,
  0.644, 0.638, 0.667, 0.664, 0.686, 0.755, 0.733, 0.726, 0.713, 0.717, 0.718, 0.718,
  0.7, 0.668, 0.648, 0.686, 0.694, 0.73,
];

export const SCENARIOS = ["Recent (actual)", "Rising river", "Flood watch"];
export const HORIZONS = { "24h": 1, "48h": 2, "72h": 3 };

const round3 = (v) => Math.round(v * 1000) / 1000;

export function applyScenario(base, scenario, offset) {
  const n = base.length;
  let s = base;
  if (scenario === "Rising river") s = base.map((v, i) => v + (i / n) * 0.5);
  else if (scenario === "Flood watch") s = base.map((v, i) => v + (i / n) * 1.1);
  return s.map((v) => round3(v + offset));
}

// Extrapolate the recent trend forward, matching project() in the Streamlit app.
export function project(levels, days) {
  if (days <= 0) return [...levels];
  const y = levels.slice(-7);
  const slope = (y[y.length - 1] - y[0]) / Math.max(y.length - 1, 1);
  const out = [...levels];
  for (let i = 0; i < days; i++) out.push(round3(out[out.length - 1] + slope));
  return out;
}

export function bandOf(p) {
  return p >= 0.66 ? "High" : p >= 0.33 ? "Moderate" : "Low";
}

export function trendOf(levels) {
  const y = levels.slice(-7);
  const delta = y[y.length - 1] - y[0];
  if (delta > 0.02) return { label: `Rising ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} m / 7d`, dir: "rising" };
  if (delta < -0.02) return { label: `Falling ${delta.toFixed(2)} m / 7d`, dir: "falling" };
  return { label: "Steady", dir: "steady" };
}

// Same four features the API derives, computed locally so the screen can show
// them next to the prediction without a second round trip.
export function featuresFromLevels(levels) {
  const w = levels.slice(-7);
  return {
    level_lag1: levels[levels.length - 1],
    level_lag2: levels[levels.length - 2],
    level_roll7: w.reduce((a, b) => a + b, 0) / w.length,
    level_change3: levels[levels.length - 1] - levels[levels.length - 4],
  };
}
