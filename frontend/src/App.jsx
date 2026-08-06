// Single-screen operations view for the River Murray flood risk prototype.
// Everything on screen comes from the same API the Streamlit dashboard uses
// (docs/API_CONTRACT.md): the model list, the prediction and the alert dispatch.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import LgaMap from "./LgaMap";
import {
  apiConfigured, getHealth, getModels, predictSeries, postAlert,
} from "./api";
import {
  BAND_META, BASE_LEVELS, DEFAULT_RECIPIENTS, FEATURES, FEATURE_LABEL, HORIZONS,
  MODELLED, OPERATOR, RECIPIENT_DIRECTORY, SCENARIOS, TRAIN_RISK_THRESHOLD_M,
  applyScenario, bandOf, featuresFromLevels, project, trendOf,
} from "./domain";
import "./App.css";

const pct = (p) => `${Math.round(p * 100)}%`;

export default function App() {
  // Controls
  const [scenario, setScenario] = useState(SCENARIOS[0]);
  const [offset, setOffset] = useState(0);
  const [horizon, setHorizon] = useState("48h");
  const [modelId, setModelId] = useState("");

  // Data from the API
  const [models, setModels] = useState([]);
  const [status, setStatus] = useState(apiConfigured ? "warming" : "unconfigured");
  const [prediction, setPrediction] = useState(null);
  const [predError, setPredError] = useState("");

  // Alert flow: pending -> confirm -> sent
  const [alertState, setAlertState] = useState("pending");
  const [alertResult, setAlertResult] = useState(null);
  const [recipients, setRecipients] = useState(DEFAULT_RECIPIENTS);

  const levels = useMemo(
    () => project(applyScenario(BASE_LEVELS, scenario, offset), HORIZONS[horizon]),
    [scenario, offset, horizon],
  );
  const localFeats = useMemo(() => featuresFromLevels(levels), [levels]);
  const trend = useMemo(() => trendOf(levels), [levels]);

  // Wake the free Render instance and load the model list before predicting.
  useEffect(() => {
    if (!apiConfigured) return;
    let alive = true;
    (async () => {
      try {
        await getHealth();
        const list = await getModels();
        if (!alive) return;
        const available = list.filter((m) => m.available);
        setModels(available);
        setModelId((prev) => prev || available.find((m) => m.default)?.id || available[0]?.id || "");
        setStatus("ready");
      } catch (err) {
        if (alive) setStatus(`down:${err.message}`);
      }
    })();
    return () => { alive = false; };
  }, []);

  // Re-score whenever the series or the chosen model changes.
  const reqId = useRef(0);
  const runPrediction = useCallback(async () => {
    if (status !== "ready" || !modelId) return;
    const id = ++reqId.current;
    try {
      const body = await predictSeries(levels, modelId, MODELLED.id);
      if (id === reqId.current) { setPrediction(body); setPredError(""); }
    } catch (err) {
      if (id === reqId.current) { setPredError(err.message); setPrediction(null); }
    }
  }, [levels, modelId, status]);

  useEffect(() => { runPrediction(); }, [runPrediction]);

  // Changing the scenario invalidates any alert already authorised on screen.
  useEffect(() => { setAlertState("pending"); setAlertResult(null); }, [scenario, offset, horizon, modelId]);

  const prob = prediction?.flood_probability ?? null;
  const band = prediction?.risk_band ?? bandOf(prob ?? 0);
  const meta = BAND_META[band];
  const selected = models.find((m) => m.id === modelId);

  // The tabular models return the four features they scored. The LSTM is a
  // sequence model, so it reports its input window instead ({window, levels_used}).
  // Detect which shape came back rather than assuming the four keys exist.
  const apiFeats = prediction?.features;
  const isTabular = FEATURES.every((f) => Number.isFinite(Number(apiFeats?.[f])));
  const shownFeats = isTabular ? apiFeats : localFeats;

  const recipientEmails = RECIPIENT_DIRECTORY
    .filter((r) => recipients.includes(r.label))
    .map((r) => r.email);

  const toggleRecipient = (label) =>
    setRecipients((prev) => (prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label]));

  async function confirmDispatch() {
    const payload = {
      station_id: MODELLED.id,
      station_name: MODELLED.name,
      risk_band: band,
      flood_probability: Number((prob ?? 0).toFixed(4)),
      horizon,
      operator: OPERATOR,
      message: `${band} flood risk (${pct(prob ?? 0)}) for ${MODELLED.name} over the next ${horizon}.`,
      recipients: recipientEmails,
    };
    try {
      const body = await postAlert(payload);
      setAlertResult({ ok: true, body });
    } catch (err) {
      setAlertResult({ ok: false, detail: err.message });
    }
    setAlertState("sent");
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>🌊 River Murray Flood Risk</h1>
          <p className="muted">SES &amp; Local Council decision support · ITA602 prototype</p>
        </div>
        <div className="header-badges">
          <span className={`badge ${status === "ready" ? "badge-live" : "badge-wait"}`}>
            {status === "ready" ? "● Live" : status === "warming" ? "● Warming up" : "● API offline"}
          </span>
          <span className="badge badge-plain">Operator: {OPERATOR}</span>
        </div>
      </header>

      {status === "unconfigured" && (
        <p className="notice">VITE_API_URL is not set, so this build has no API to call.</p>
      )}
      {status === "warming" && (
        <p className="notice">Waking the API (free tier cold start), this can take up to a minute.</p>
      )}
      {status.startsWith("down") && (
        <p className="notice notice-bad">API unreachable: {status.slice(5)}</p>
      )}

      <div className="controls">
        <label>
          <span>Model</span>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={!models.length}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.name}{m.default ? " (default)" : ""}</option>
            ))}
          </select>
        </label>
        <label>
          <span>River condition</span>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {SCENARIOS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <label>
          <span>Level offset {offset.toFixed(2)} m</span>
          <input type="range" min="-0.3" max="1.5" step="0.05" value={offset}
                 onChange={(e) => setOffset(Number(e.target.value))} />
        </label>
        <div className="horizon">
          <span>Forecast horizon</span>
          <div className="segmented">
            {Object.keys(HORIZONS).map((h) => (
              <button key={h} className={h === horizon ? "on" : ""} onClick={() => setHorizon(h)}>{h}</button>
            ))}
          </div>
        </div>
      </div>

      <main className="grid">
        <section className="card">
          <h2 className="card-title">Catchment overview</h2>
          <LgaMap band={band} />
        </section>

        <div className="col">
          <section className="verdict" style={{ background: meta.bg, color: meta.tx, borderLeftColor: meta.hex }}>
            <div>
              <div className="verdict-band">{meta.icon} {band} risk</div>
              <div className="verdict-sub">
                {MODELLED.name} · {trend.dir === "rising" ? "rising" : trend.dir === "falling" ? "easing" : "steady"} · {meta.advice}
              </div>
            </div>
            <div className="verdict-num">
              <div className="verdict-pct">{prob === null ? "–" : pct(prob)}</div>
              <div className="verdict-cap">flood probability · next {horizon}</div>
            </div>
          </section>

          <section className="card">
            <div className="stat-row">
              <div><div className="muted">Trend</div><div className="stat">{trend.label}</div></div>
              <div><div className="muted">Latest level</div>
                <div className="stat">{levels[levels.length - 1].toFixed(2)} m</div>
                <div className="muted">
                  {(levels[levels.length - 1] - TRAIN_RISK_THRESHOLD_M >= 0 ? "+" : "")}
                  {(levels[levels.length - 1] - TRAIN_RISK_THRESHOLD_M).toFixed(2)} m vs threshold
                </div>
              </div>
              <div><div className="muted">Served by</div>
                <div className="stat">{prediction?.model ?? "–"}</div>
                <div className="muted">{predError ? "prediction failed" : "live from API"}</div>
              </div>
            </div>
            {predError && <p className="notice notice-bad">{predError}</p>}
          </section>

          <section className="card">
            <h2 className="card-title">Why this score</h2>
            <p className="muted">
              {isTabular || !prediction
                ? "Model inputs for this prediction. Feature importance and SHAP are reported in the models and explainability section."
                : `The ${selected?.name ?? "sequence"} model reads the raw daily level series, not the four summary features. SHAP for it is reported per day of the window.`}
            </p>
            <table className="feat">
              <tbody>
                {isTabular || !prediction ? (
                  FEATURES.map((f) => (
                    <tr key={f}>
                      <td>{FEATURE_LABEL[f]}</td>
                      <td className="num">{Number(shownFeats[f]).toFixed(3)}</td>
                    </tr>
                  ))
                ) : (
                  <>
                    <tr>
                      <td>Input window</td>
                      <td className="num">{apiFeats.window} days</td>
                    </tr>
                    <tr>
                      <td>Levels supplied</td>
                      <td className="num">{apiFeats.levels_used}</td>
                    </tr>
                    <tr>
                      <td>Most recent level</td>
                      <td className="num">{levels[levels.length - 1].toFixed(3)} m</td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
            {selected?.metrics && (
              <p className="muted metrics">
                {selected.name} test metrics · F1 {selected.metrics.F1} · MCC {selected.metrics.MCC} · Brier {selected.metrics.Brier}
              </p>
            )}
          </section>

          <section className="card">
            <h2 className="card-title">Alert authorisation</h2>

            {band === "Low" && alertState !== "sent" && (
              <p className="pill pill-neutral">No alert proposed at this level</p>
            )}

            {band !== "Low" && alertState === "pending" && (
              <>
                <p className="pill pill-warn">⚠ Awaiting human authorisation</p>
                <fieldset className="recips">
                  <legend className="muted">Recipients</legend>
                  {RECIPIENT_DIRECTORY.map((r) => (
                    <label key={r.label} className="recip">
                      <input type="checkbox" checked={recipients.includes(r.label)}
                             onChange={() => toggleRecipient(r.label)} />
                      <span>{r.label}</span>
                    </label>
                  ))}
                </fieldset>
                <button className="btn btn-primary" disabled={!recipients.length || prob === null}
                        onClick={() => setAlertState("confirm")}>
                  Authorise &amp; dispatch alert
                </button>
              </>
            )}

            {alertState === "confirm" && (
              <>
                <p className="pill pill-bad">Confirm dispatch of a {band} flood alert?</p>
                <p className="muted">To {recipients.length} recipient{recipients.length === 1 ? "" : "s"}. This action is logged against your operator ID.</p>
                <div className="btn-row">
                  <button className="btn btn-primary" onClick={confirmDispatch}>Confirm dispatch</button>
                  <button className="btn" onClick={() => setAlertState("pending")}>Cancel</button>
                </div>
              </>
            )}

            {alertState === "sent" && (
              <>
                {alertResult?.ok ? (
                  <>
                    <p className="pill pill-good">✓ Alert dispatched · authorised by {OPERATOR}</p>
                    <p className="muted">
                      Audit log #{alertResult.body.alert_id} · chained {String(alertResult.body.chained)} ·
                      email {alertResult.body.email_status} · retained per State Records Act 1997 (SA)
                    </p>
                  </>
                ) : (
                  <>
                    <p className="pill pill-warn">⚠ Authorised locally · backend /alerts not connected</p>
                    <p className="muted">Reason: {alertResult?.detail}</p>
                  </>
                )}
                <button className="btn" onClick={() => { setAlertState("pending"); setAlertResult(null); }}>Reset</button>
              </>
            )}
          </section>
        </div>
      </main>

      <footer className="muted foot">
        Prototype for academic assessment (ITA602). Not for operational flood-warning use.
        Only Murray Bridge has a trained model; other council areas are shown for context.
      </footer>
    </div>
  );
}
