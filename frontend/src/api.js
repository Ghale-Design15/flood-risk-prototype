// Client for the flood risk API (docs/API_CONTRACT.md, frozen v3).
// The base URL comes from VITE_API_URL so the same build points at the Render
// service in production and a local uvicorn during development.

const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export const apiConfigured = Boolean(BASE);

async function request(path, options = {}, timeoutMs = 30000) {
  if (!BASE) throw new Error("VITE_API_URL is not set");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* keep the status line */
      }
      throw new Error(detail);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

// The free Render instance sleeps after ~15 min idle, so the first call pays a
// cold start. The screen pings /health on load and shows a warming-up state
// instead of blocking on the first prediction.
export const getHealth = () => request("/health", {}, 90000);

export const getModels = () => request("/models");

export const predictSeries = (levels, model, stationId) =>
  request("/predict_series", {
    method: "POST",
    body: JSON.stringify({ levels, ...(model ? { model } : {}), ...(stationId ? { station_id: stationId } : {}) }),
  });

export const postAlert = (payload) =>
  request("/alerts", { method: "POST", body: JSON.stringify(payload) });

export const verifyAlerts = () => request("/alerts/verify");
