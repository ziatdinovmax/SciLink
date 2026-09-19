import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type LiveConfig,
  type LiveEvent,
  type LiveInstrumentInfo,
  type LiveSnapshot,
} from "../api";
import { fmt, LiveChart } from "./LiveChart";

/** Live tab — run a measurement loop against a simulated experiment (or the
 * user's own `package.module:Class` instrument) and watch it: every frame is
 * analysed by a locked recipe with no model call, flagged when the recipe stops
 * fitting, re-anchored in the background, and answered with a recommendation
 * the operator may accept. The page only observes; it shows what the loop wrote
 * to its own log. */

const CUSTOM = "__custom__";
const REPLAY = "replay";

/** "name: definition" per line → {name: definition}. */
function parseOutputs(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0 && line.slice(i + 1).trim()) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}
const POLL_MS = 1500;

const FLAG_WORDS: Record<string, string> = {
  fit_failed: "fit failed",
  gate_poor: "poor fit quality",
  drift_suspected: "recipe no longer fits the data",
  out_of_reference_range: "outside the reference range",
  deadline_missed: "frame deadline missed",
  llm_used: "a model was called",
};

function describeEvent(e: LiveEvent): string {
  const g = (k: string) => e[k] as string | number | undefined;
  switch (e.event) {
    case "setup": {
      const port = e.portability as { portable?: boolean; summary?: string } | undefined;
      return `armed from ${g("source") ?? "reference"}` +
        (port && port.portable === false ? ` — ▲ ${port.summary}` : "");
    }
    case "escalation_started":
      return "escalation started — a new recipe is being built in the background";
    case "reanchor":
      return `re-anchored (${g("source")}, ${g("seconds")} s, ${g("llm_calls")} model call(s)); ` +
        `${g("frames_answered_meanwhile")} frames answered meanwhile`;
    case "escalation_failed":
      return `escalation failed: ${String(g("error") ?? "").slice(0, 160)}`;
    case "recommendation":
      return `recommendation from ${g("source") ?? "recommender"}` +
        (e.valid === false ? " — refused by the instrument schema" : "");
    case "amend":
      return `recipe amended: ${g("note") ?? ""}`;
    default:
      return e.event;
  }
}

export function LivePanel({
  sessionId,
  active,
  localFiles,
}: {
  sessionId: string;
  active: boolean;
  /** Importing a user's instrument class runs code on the server machine, so
   * it is offered only when that machine is the user's own. */
  localFiles: boolean;
}) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // setup form
  const [instrument, setInstrument] = useState("");
  const [custom, setCustom] = useState("");
  const [nFrames, setNFrames] = useState(60);
  const [interval, setIntervalS] = useState(2);
  const [apply, setApply] = useState<LiveConfig["apply"]>("never");
  const [recommender, setRecommender] = useState<LiveConfig["recommender"]>("none");
  const [objectiveKey, setObjectiveKey] = useState("");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [objective, setObjective] = useState("");
  const [every, setEvery] = useState(8);
  const [autoEscalate, setAutoEscalate] = useState(true);
  const [profile, setProfile] = useState("thorough");
  const [replayDir, setReplayDir] = useState("");
  const [technique, setTechnique] = useState("");
  const [sample, setSample] = useState("");
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState("");
  const [outputsText, setOutputsText] = useState("");
  const [targetsText, setTargetsText] = useState("");

  // running view
  const [traceKey, setTraceKey] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const timer = useRef<number | null>(null);

  const refresh = useCallback(() => {
    api.live(sessionId).then((s) => { setSnap(s); }).catch((e) => setError(String(e)));
  }, [sessionId]);

  const state = snap?.state ?? "idle";
  const live = state === "arming" || state === "running";

  useEffect(() => {
    if (!active) return;
    refresh();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(refresh, live ? POLL_MS : 6000);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [active, live, refresh]);

  const simulators = snap?.simulators ?? [];
  useEffect(() => {
    if (!instrument && simulators.length) setInstrument(simulators[0].name);
  }, [simulators, instrument]);

  const chosen: LiveInstrumentInfo | undefined =
    state === "idle" ? simulators.find((s) => s.name === instrument) : snap?.instrument;
  const outputKeys = useMemo(() => Object.keys(chosen?.outputs ?? {}), [chosen]);

  useEffect(() => {
    if (outputKeys.length && !outputKeys.includes(objectiveKey)) setObjectiveKey(outputKeys[0]);
    if (outputKeys.length && !outputKeys.includes(traceKey)) setTraceKey(outputKeys[0]);
  }, [outputKeys, objectiveKey, traceKey]);

  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    setBusy(true);
    try {
      await fn();
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const start = () =>
    act(() => api.liveStart(sessionId, {
      instrument: instrument === CUSTOM ? custom.trim() : instrument,
      n_frames: nFrames, interval_s: interval, apply,
      recommender: instrument === REPLAY ? "none" : recommender,
      ...(instrument === REPLAY ? {
        replay_dir: replayDir.trim(),
        system_info: { technique, sample, x_axis: xAxis, y_axis: yAxis },
        outputs: parseOutputs(outputsText),
        targets: targetsText.split(",").map((t) => t.trim()).filter(Boolean),
      } : {}),
      objective_key: recommender === "gp" ? objectiveKey : undefined,
      direction, objective: recommender === "llm" ? objective : undefined, every,
      auto_escalate: autoEscalate, reference_profile: profile,
    }));

  // ── idle: the setup form ─────────────────────────────────────
  if (state === "idle") {
    return (
      <div className="live-panel">
        <section className="tools-section">
          <h3>Live measurement loop</h3>
          <p className="caption">
            SciLink analyses one reference measurement thoroughly, locks the verified recipe, and then
            answers every new frame with it in about a second — no model call per frame. When the recipe
            stops fitting the data, a new one is built in the background while the stream keeps being
            answered. Try it on a simulated experiment, then swap in your own instrument.
          </p>
          {error && <div className="error-banner">{error}</div>}
          <div className="live-form">
            <label>
              <span>Experiment</span>
              <select value={instrument} onChange={(e) => setInstrument(e.target.value)}>
                {simulators.map((s) => (
                  <option key={s.name} value={s.name}>{s.technique ?? s.name} (simulated)</option>
                ))}
                {localFiles && <option value={REPLAY}>Replay recorded data from a folder…</option>}
                {localFiles && <option value={CUSTOM}>My own instrument…</option>}
              </select>
            </label>
            {instrument === CUSTOM && (
              <label>
                <span>Instrument class</span>
                <input
                  type="text" placeholder="package.module:ClassName" value={custom}
                  onChange={(e) => setCustom(e.target.value)}
                />
                <span className="caption">
                  A <code>scilink.live.Instrument</code> subclass importable by the server: it declares its
                  parameter schema and implements <code>acquire(params)</code>.
                </span>
              </label>
            )}
            {instrument === REPLAY && (
              <div className="live-about">
                <p>
                  Real measurements through the live loop, before there is a live instrument: every
                  two-column file in the folder (.csv .txt .xy .dat .tsv .npy) is served as one frame, in
                  file order. The first file is the reference. Recorded data cannot be steered, so there
                  is no recommender.
                </p>
                <div className="live-form">
                  <label><span>Folder on this machine</span>
                    <input type="text" placeholder="/path/to/recorded/series" value={replayDir}
                      onChange={(e) => setReplayDir(e.target.value)} />
                  </label>
                  <div className="live-row">
                    <label><span>Technique</span>
                      <input type="text" placeholder="e.g. Raman spectroscopy" value={technique}
                        onChange={(e) => setTechnique(e.target.value)} />
                    </label>
                    <label><span>Sample and what is being done to it</span>
                      <input type="text" placeholder="e.g. carbon film, annealed in situ" value={sample}
                        onChange={(e) => setSample(e.target.value)} />
                    </label>
                    <label><span>x axis</span>
                      <input type="text" placeholder="e.g. Raman shift (cm^-1)" value={xAxis}
                        onChange={(e) => setXAxis(e.target.value)} />
                    </label>
                    <label><span>y axis</span>
                      <input type="text" placeholder="e.g. intensity (counts)" value={yAxis}
                        onChange={(e) => setYAxis(e.target.value)} />
                    </label>
                  </div>
                  <label><span>Outputs to report for every frame — one per line, name: definition</span>
                    <textarea rows={3} value={outputsText} onChange={(e) => setOutputsText(e.target.value)}
                      placeholder={"g_position: position of the G band (the band near 1585 cm^-1)\nd_over_g: ratio of the D band height to the G band height"} />
                  </label>
                  <label><span>What matters, in plain words (comma separated)</span>
                    <input type="text" placeholder="G band position, D/G ratio" value={targetsText}
                      onChange={(e) => setTargetsText(e.target.value)} />
                  </label>
                </div>
              </div>
            )}
            {chosen && (
              <div className="live-about">
                <p>{chosen.about}</p>
                <div className="live-about-grid">
                  <div>
                    <h4>Reported for every frame</h4>
                    <ul>{Object.entries(chosen.outputs).map(([k, v]) => <li key={k}><code>{k}</code> — {v}</li>)}</ul>
                  </div>
                  <div>
                    <h4>Acquisition parameters</h4>
                    <ul>
                      {Object.entries(chosen.schema).map(([k, p]) => (
                        <li key={k}>
                          <code>{k}</code> {p.low !== undefined && `${p.low}–${p.high}`} {p.units ?? ""}
                          {p.description ? ` — ${p.description}` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4>What will happen</h4>
                    <ul>{chosen.events.map((e) => <li key={e.frame + e.what}>frame {e.frame}: {e.what}</li>)}</ul>
                  </div>
                </div>
              </div>
            )}
            <div className="live-row">
              <label><span>Frames</span>
                <input type="number" min={1} max={2000} value={nFrames} onChange={(e) => setNFrames(+e.target.value)} />
              </label>
              <label><span>Seconds between frames</span>
                <input type="number" min={0} step={0.5} value={interval} onChange={(e) => setIntervalS(+e.target.value)} />
              </label>
              <label><span>Reference analysis</span>
                <select value={profile} onChange={(e) => setProfile(e.target.value)}>
                  <option value="thorough">thorough (2–4 min)</option>
                  <option value="quick">quick</option>
                  <option value="extract">extract (fastest)</option>
                </select>
              </label>
              <label className="live-check">
                <input type="checkbox" checked={autoEscalate} onChange={(e) => setAutoEscalate(e.target.checked)} />
                <span>Re-anchor automatically when the recipe stops fitting</span>
              </label>
            </div>
            {instrument !== REPLAY && <div className="live-row">
              <label><span>Next-measurement recommender</span>
                <select value={recommender} onChange={(e) => setRecommender(e.target.value as LiveConfig["recommender"])}>
                  <option value="none">none — analysis only</option>
                  <option value="gp">Bayesian optimization (GP)</option>
                  <option value="llm">LLM (slow clock, parameters as JSON)</option>
                </select>
              </label>
              {recommender === "gp" && (
                <>
                  <label><span>Optimize</span>
                    <select value={objectiveKey} onChange={(e) => setObjectiveKey(e.target.value)}>
                      {outputKeys.map((k) => <option key={k}>{k}</option>)}
                    </select>
                  </label>
                  <label><span>Direction</span>
                    <select value={direction} onChange={(e) => setDirection(e.target.value as "maximize" | "minimize")}>
                      <option>maximize</option><option>minimize</option>
                    </select>
                  </label>
                </>
              )}
              {recommender === "llm" && (
                <label><span>Every N frames</span>
                  <input type="number" min={2} value={every} onChange={(e) => setEvery(+e.target.value)} />
                </label>
              )}
              {recommender !== "none" && (
                <label><span>Recommendations are</span>
                  <select value={apply} onChange={(e) => setApply(e.target.value as LiveConfig["apply"])}>
                    <option value="never">shown — I accept them by hand</option>
                    <option value="valid">applied automatically when valid</option>
                  </select>
                </label>
              )}
            </div>}
            {instrument !== REPLAY && recommender === "llm" && (
              <label>
                <span>Goal, in words</span>
                <textarea
                  rows={2} value={objective} onChange={(e) => setObjective(e.target.value)}
                  placeholder="e.g. measure the G band position to ±0.5 cm⁻¹ with the shortest acquisition; avoid laser heating"
                />
              </label>
            )}
            <div>
              <button
                className="primary" onClick={start}
                disabled={busy || (instrument === CUSTOM && !custom.includes(":")) ||
                  (instrument === REPLAY && !(replayDir.trim() && technique.trim())) ||
                  (instrument !== REPLAY && recommender === "llm" && !objective.trim())}
              >
                Start live run
              </button>
            </div>
          </div>
        </section>
      </div>
    );
  }

  // ── a run exists ─────────────────────────────────────────────
  const frames = snap?.frames ?? [];
  const st = snap?.status ?? {};
  const latest = frames[frames.length - 1];
  const rec = snap?.recommendation ?? null;
  const inst = snap?.instrument;
  const hasTruth = frames.some((f) => f.truth && traceKey in f.truth);
  const events = snap?.events ?? [];
  const rules = events
    .filter((e) => e.event === "reanchor" && typeof e.step === "number")
    .map((e) => ({ x: e.step as number, label: "re-anchor" }));
  const params = snap?.current_params ?? {};
  const recIsCurrent = !!rec?.params &&
    Object.entries(rec.params).every(([k, v]) => String(params[k]) === String(v) || Number(params[k]) === Number(v));
  const pending = Object.fromEntries(
    Object.entries(edits).filter(([k, v]) => v !== "" && String(params[k]) !== v).map(([k, v]) => [k, isNaN(+v) ? v : +v]),
  );

  return (
    <div className="live-panel">
      <div className="live-head">
        <h3>{inst?.technique ?? inst?.name}</h3>
        <span className={`live-state ${state}`}>
          {state === "arming" ? "◌ analysing the reference" : state === "running" ? "● running"
            : state === "done" ? "✓ finished" : state === "stopped" ? "■ stopped" : "✕ error"}
        </span>
        <span className="caption">{Math.round(snap?.elapsed_s ?? 0)} s</span>
        <span className="live-head-actions">
          {live
            ? <button onClick={() => act(() => api.liveStop(sessionId))} disabled={busy}>Stop</button>
            : <button className="primary small" onClick={() => act(() => api.liveClear(sessionId))} disabled={busy}>New run</button>}
        </span>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {snap?.error && <div className="error-banner">{snap.error}</div>}
      {state === "arming" && (
        <p className="caption">
          One reference measurement is being analysed in full ({String(snap?.config?.reference_profile ?? "thorough")});
          the verified script becomes the locked recipe and the outputs are pinned to fixed names. A thorough
          reference typically takes 2–4 minutes; frames start streaming right after.
        </p>
      )}

      {frames.length === 0 && snap?.latest && (
        <section className="live-card live-reference">
          <div className="live-card-head"><h4>Reference measurement</h4></div>
          <LiveChart
            series={snap.latest.x.map((x, i) => ({ x, y: snap.latest?.y[i] ?? null }))}
            seriesLabel={inst?.y_axis ?? "signal"} xLabel={inst?.x_axis} yLabel={inst?.y_axis}
          />
        </section>
      )}

      {frames.length > 0 && (
        <>
          <div className="live-stats">
            <div><b>{st.frames ?? frames.length}</b><span>of {snap?.n_frames_total} frames</span></div>
            <div><b>{st.clean_frames ?? "—"}</b><span>clean</span></div>
            <div><b>{(st.latency_s?.median ?? 0).toFixed(1)} s</b><span>median per frame (max {(st.latency_s?.max ?? 0).toFixed(1)} s)</span></div>
            <div><b>{st.llm_calls_in_frames ?? 0}</b><span>model calls in frames</span></div>
            <div><b>{st.reanchors ?? 0}</b><span>re-anchor(s){st.escalating ? " — one running" : ""}</span></div>
            <div><b>{(latest?.recipe_id ?? "").slice(0, 6) || "—"}</b><span>recipe</span></div>
          </div>
          {latest && latest.flags.length > 0 && (
            <div className="live-flags">
              ▲ frame {latest.step}: {latest.flags.map((f) => FLAG_WORDS[f] ?? f).join("; ")}
              {st.escalating ? " — a new recipe is being built; frames are still answered" : ""}
            </div>
          )}

          <div className="live-grid">
            <section className="live-card">
              <div className="live-card-head">
                <h4>Output over frames</h4>
                <select value={traceKey} onChange={(e) => setTraceKey(e.target.value)}>
                  {[...outputKeys, "fit_r_squared"].map((k) => <option key={k}>{k}</option>)}
                </select>
              </div>
              <LiveChart
                series={frames.map((f) => ({ x: f.step, y: f.features[traceKey] ?? null }))}
                seriesLabel={traceKey}
                reference={hasTruth ? frames.map((f) => ({ x: f.step, y: f.truth?.[traceKey] ?? null })) : undefined}
                referenceLabel="simulator truth"
                flagged={frames.filter((f) => f.flags.length).map((f) => ({
                  x: f.step, label: f.flags.map((x) => FLAG_WORDS[x] ?? x).join("; "),
                }))}
                rules={rules}
                xLabel="frame" xInteger
              />
            </section>
            <section className="live-card">
              <div className="live-card-head"><h4>Latest frame{snap?.latest ? ` (${snap.latest.step})` : ""}</h4></div>
              <LiveChart
                series={(snap?.latest?.x ?? []).map((x, i) => ({ x, y: snap?.latest?.y[i] ?? null }))}
                seriesLabel={inst?.y_axis ?? "signal"}
                xLabel={inst?.x_axis} yLabel={inst?.y_axis}
              />
            </section>
          </div>

          <div className="live-grid">
            <section className="live-card">
              <div className="live-card-head"><h4>Frame {latest?.step}: reported values</h4></div>
              <table className="live-table">
                <thead><tr><th>output</th><th>value</th>{hasTruth && <th>simulator truth</th>}</tr></thead>
                <tbody>
                  {[...outputKeys, "fit_r_squared"].map((k) => (
                    <tr key={k}>
                      <td><code>{k}</code></td>
                      <td>{fmt(latest?.features[k])}</td>
                      {hasTruth && <td>{fmt(latest?.truth?.[k])}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
            <section className="live-card">
              <div className="live-card-head"><h4>Acquisition parameters</h4></div>
              <div className="live-params">
                {Object.entries(inst?.schema ?? {}).map(([k, p]) => (
                  <label key={k} title={p.description}>
                    <span><code>{k}</code> {p.units ? `(${p.units})` : ""}
                      {p.low !== undefined && <em> {p.low}–{p.high}</em>}</span>
                    <input
                      type="text" value={edits[k] ?? String(params[k] ?? "")} disabled={state !== "running"}
                      onChange={(e) => setEdits((d) => ({ ...d, [k]: e.target.value }))}
                    />
                  </label>
                ))}
                <button
                  disabled={state !== "running" || busy || !Object.keys(pending).length}
                  onClick={() => act(async () => { await api.liveParams(sessionId, pending); setEdits({}); })}
                >
                  Use from the next frame
                </button>
              </div>
              {rec && (
                <div className={`live-rec ${rec.valid ? "" : "invalid"}`}>
                  <div className="live-rec-head">
                    <b>Recommendation</b>
                    <span className="caption">
                      {rec.source} · from frame {rec.based_on_step}
                      {rec.acquisition_skill ? ` · guided by the ${rec.acquisition_skill} acquisition skill` : ""}
                    </span>
                  </div>
                  {rec.params && (
                    <div className="tool-chips">
                      {Object.entries(rec.params).map(([k, v]) => (
                        <span key={k} className="live-chip"><code>{k}</code> = {typeof v === "number" ? fmt(v) : v}</span>
                      ))}
                    </div>
                  )}
                  {rec.kind === "hold" && <p><b>Keep the current parameters.</b></p>}
                  {rec.protocol && <pre className="live-protocol">{rec.protocol}</pre>}
                  {!rec.valid && (
                    <p className="caption warn">Refused — not mappable onto the instrument: {(rec.problems ?? []).join("; ")}</p>
                  )}
                  <p className="caption">{rec.rationale}</p>
                  {rec.valid && rec.params && state === "running" && recIsCurrent && (
                    <p className="caption">These are the parameters already in use.</p>
                  )}
                  {rec.valid && rec.params && state === "running" && !recIsCurrent && (
                    <button
                      className="success" disabled={busy}
                      onClick={() => act(() => api.liveParams(sessionId, rec.params as Record<string, number | string>))}
                    >
                      Accept
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>
        </>
      )}

      <section className="live-card">
        <div className="live-card-head"><h4>Events</h4><span className="caption" title={snap?.run_dir}>{(snap?.run_dir ?? "").split("/").slice(-3).join("/")}</span></div>
        {events.length === 0 && <p className="caption">Nothing yet.</p>}
        <ul className="live-events">
          {[...events].reverse().map((e, i) => (
            <li key={i}>
              <span className="caption">{typeof e.step === "number" ? `frame ${e.step}` : ""}</span>
              {describeEvent(e)}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
