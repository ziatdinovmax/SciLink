import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  api,
  type LiveConfig,
  type LiveEvent,
  type LiveInstrumentInfo,
  type LiveSnapshot,
} from "../api";
import { fmt, fmtShort, LiveChart } from "./LiveChart";

/** Live tab. A measurement loop runs in the session as a background job and
 * this page watches it: every frame analysed by a locked recipe with no model
 * call, flagged when the recipe stops fitting, re-anchored in the background.
 * The page observes and never drives. Copy is kept short on purpose. Anything
 * explanatory sits behind an info button. */

const CUSTOM = "__custom__";
const REPLAY = "replay";
const FIRST_FRAME = "__first_frame__";
const POLL_MS = 1500;

const FLAG_WORDS: Record<string, string> = {
  fit_failed: "fit failed",
  gate_poor: "poor fit",
  drift_suspected: "data changed",
  out_of_reference_range: "value out of range",
  deadline_missed: "slow frame",
  llm_used: "model called",
};

/** A small "i" button. The explanation opens on click and closes on blur, so
 * the page stays quiet for people who do not need it. */
function Info({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="live-info">
      <button
        type="button" aria-label="More information" aria-expanded={open}
        onClick={(e) => { e.preventDefault(); setOpen((o) => !o); }} onBlur={() => setOpen(false)}
      >
        i
      </button>
      {open && <span className="live-info-pop" role="note">{children}</span>}
    </span>
  );
}

/** "name: definition" per line to {name: definition}. */
function parseOutputs(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0 && line.slice(i + 1).trim()) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
    : `${m}:${String(s % 60).padStart(2, "0")}`;
}

function describeEvent(e: LiveEvent): string {
  const g = (k: string) => e[k] as string | number | undefined;
  switch (e.event) {
    case "setup": {
      const port = e.portability as
        { portable?: boolean; summary?: string; positions_summary?: string } | undefined;
      const refs = e.reference_frames as { n?: number; regimes?: number } | undefined;
      const how = g("source") === "anchor" ? "Armed from a past analysis."
        : refs?.n ? `Armed from the first ${refs.n} frames${(refs.regimes ?? 1) > 1 ? `, ${refs.regimes} regimes seen` : ""}.`
        : "Armed from the reference.";
      return [how, port && port.portable === false ? port.summary : "", port?.positions_summary ?? ""]
        .filter(Boolean).join(" ");
    }
    case "escalation_started":
      return "Rebuilding the recipe in the background.";
    case "reanchor": {
      const win = e.window as { n?: number } | undefined;
      return `New recipe adopted after ${g("seconds")} s${win?.n ? `, planned from the last ${win.n} frames` : ""}. ` +
        `${g("frames_answered_meanwhile")} frames were answered meanwhile.`;
    }
    case "drift_rebased":
      return "The stream settled under the new recipe. Change detection now compares with these frames.";
    case "escalation_failed":
      return `Rebuild failed. ${String(g("error") ?? "").slice(0, 160)}`;
    case "recommendation":
      return e.valid === false ? "Recommendation refused by the instrument limits." : "Recommendation.";
    case "amend":
      return `Recipe amended. ${g("note") ?? ""}`;
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
  /** Importing a user's instrument class or reading a folder happens on the
   * server machine, so both are offered only when that machine is the user's. */
  localFiles: boolean;
}) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // setup form
  const [instrument, setInstrument] = useState("");
  const [custom, setCustom] = useState("");
  const [reference, setReference] = useState(FIRST_FRAME);
  const [nFrames, setNFrames] = useState("");
  const [interval, setIntervalS] = useState("2");
  const [apply, setApply] = useState<LiveConfig["apply"]>("never");
  const [recommender, setRecommender] = useState<LiveConfig["recommender"]>("none");
  const [objectiveKey, setObjectiveKey] = useState("");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [objective, setObjective] = useState("");
  const [every, setEvery] = useState(8);
  const [autoEscalate, setAutoEscalate] = useState(true);
  const [profile, setProfile] = useState("thorough");
  const [refFrames, setRefFrames] = useState("1");
  const [replayDir, setReplayDir] = useState("");
  const [technique, setTechnique] = useState("");
  const [sample, setSample] = useState("");
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState("");
  const [outputsText, setOutputsText] = useState("");

  // running view
  const [edits, setEdits] = useState<Record<string, string>>({});
  const timer = useRef<number | null>(null);

  const refresh = useCallback(() => {
    api.live(sessionId).then(setSnap).catch((e) => setError(String(e)));
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
  const analyses = snap?.analyses ?? [];
  useEffect(() => {
    if (!instrument && simulators.length) setInstrument(simulators[0].name);
  }, [simulators, instrument]);

  const chosen: LiveInstrumentInfo | undefined =
    state === "idle" ? simulators.find((s) => s.name === instrument) : snap?.instrument;
  const formOutputs = useMemo(() => Object.keys(chosen?.outputs ?? {}), [chosen]);
  useEffect(() => {
    if (formOutputs.length && !formOutputs.includes(objectiveKey)) setObjectiveKey(formOutputs[0]);
  }, [formOutputs, objectiveKey]);

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

  const steerable = instrument !== REPLAY;
  const start = () =>
    act(() => api.liveStart(sessionId, {
      instrument: instrument === CUSTOM ? custom.trim() : instrument,
      n_frames: nFrames.trim() ? Math.max(1, parseInt(nFrames, 10)) : null,
      interval_s: parseFloat(interval) || 0,
      apply, recommender: steerable ? recommender : "none",
      reference_source: reference === FIRST_FRAME ? "first_frame" : "analysis",
      reference_analysis: reference === FIRST_FRAME ? undefined : reference,
      reference_frames: Math.max(1, Math.min(25, parseInt(refFrames, 10) || 1)),
      ...(instrument === REPLAY ? {
        replay_dir: replayDir.trim(),
        system_info: { technique, sample, x_axis: xAxis, y_axis: yAxis },
        outputs: parseOutputs(outputsText),
      } : {}),
      objective_key: recommender === "gp" ? objectiveKey : undefined,
      direction, objective: recommender === "llm" ? objective : undefined, every,
      auto_escalate: autoEscalate, reference_profile: profile,
    }));

  // ── idle: the setup form ─────────────────────────────────────
  if (state === "idle") {
    const ready = !busy &&
      !(instrument === CUSTOM && !custom.includes(":")) &&
      !(instrument === REPLAY && !(replayDir.trim() && technique.trim())) &&
      !(steerable && recommender === "llm" && !objective.trim());
    return (
      <div className="live-panel">
        <div className="live-head">
          <h3>Live</h3>
          <Info>
            One reference measurement is analysed in full and its verified script is locked as the recipe.
            Every new frame is then answered by that recipe in about a second, with no model call. When the
            recipe stops fitting, a new one is built in the background while frames keep being answered.
          </Info>
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="live-form">
          <div className="live-row">
            <label className="grow">
              <span>Data source
                {chosen && (
                  <Info>
                    {chosen.about}
                    {chosen.events.length > 0 && (
                      <ul>{chosen.events.map((e) => <li key={e.frame + e.what}>Frame {e.frame}: {e.what}</li>)}</ul>
                    )}
                  </Info>
                )}
              </span>
              <select value={instrument} onChange={(e) => setInstrument(e.target.value)}>
                {simulators.map((s) => (
                  <option key={s.name} value={s.name}>{s.technique ?? s.name} (simulated)</option>
                ))}
                {localFiles && <option value={REPLAY}>Recorded data in a folder</option>}
                {localFiles && <option value={CUSTOM}>My instrument</option>}
              </select>
            </label>
            <label className="grow">
              <span>Reference
                <Info>
                  The loop needs one analysed measurement to lock its recipe. Use the first frame, which is
                  analysed now and takes a few minutes, or a curve analysis already done in this session,
                  which is adopted in seconds without being redone.
                </Info>
              </span>
              <select value={reference} onChange={(e) => setReference(e.target.value)}>
                <option value={FIRST_FRAME}>{(parseInt(refFrames, 10) || 1) > 1 ? `First ${refFrames} frames` : "First frame"}</option>
                {analyses.map((a) => (
                  <option key={a.path} value={a.path} title={a.model}>
                    {a.from_live_run ? "Earlier live run" : "Analysis"}: {a.name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {instrument === CUSTOM && (
            <label>
              <span>Instrument class
                <Info>
                  A <code>scilink.live.Instrument</code> subclass the server can import. It declares the
                  acquisition parameters it accepts and implements <code>acquire(params)</code>.
                  See <code>examples/live_loop_demo.py</code>.
                </Info>
              </span>
              <input type="text" placeholder="package.module:ClassName" value={custom}
                onChange={(e) => setCustom(e.target.value)} />
            </label>
          )}

          {instrument === REPLAY && (
            <>
              <label>
                <span>Folder
                  <Info>
                    Every two-column file in the folder (.csv .txt .xy .dat .tsv .npy) is one frame, in file
                    order. Recorded data cannot be steered, so there is no recommender.
                  </Info>
                </span>
                <input type="text" placeholder="/path/to/recorded/series" value={replayDir}
                  onChange={(e) => setReplayDir(e.target.value)} />
              </label>
              <div className="live-row">
                <label className="grow"><span>Technique</span>
                  <input type="text" placeholder="Raman spectroscopy" value={technique}
                    onChange={(e) => setTechnique(e.target.value)} />
                </label>
                <label className="grow"><span>Sample</span>
                  <input type="text" placeholder="carbon film, annealed in situ" value={sample}
                    onChange={(e) => setSample(e.target.value)} />
                </label>
                <label><span>x axis</span>
                  <input type="text" placeholder="Raman shift (cm^-1)" value={xAxis}
                    onChange={(e) => setXAxis(e.target.value)} />
                </label>
                <label><span>y axis</span>
                  <input type="text" placeholder="intensity (counts)" value={yAxis}
                    onChange={(e) => setYAxis(e.target.value)} />
                </label>
              </div>
              <label>
                <span>Track
                  <Info>
                    One quantity per line as <code>name: definition</code>. Each is reported under that name
                    for every frame, whatever recipe is in use. Leave empty to get the recipe's own
                    quantities.
                  </Info>
                </span>
                <textarea rows={2} value={outputsText} onChange={(e) => setOutputsText(e.target.value)}
                  placeholder={"g_position: position of the G band\nd_over_g: ratio of the D band height to the G band height"} />
              </label>
            </>
          )}

          {chosen && formOutputs.length > 0 && (
            <div className="live-track">
              <span>Track
                <Info>
                  <ul>{Object.entries(chosen.outputs).map(([k, v]) => <li key={k}><code>{k}</code> {v}</li>)}</ul>
                </Info>
              </span>
              {formOutputs.map((k) => <span key={k} className="live-chip"><code>{k}</code></span>)}
            </div>
          )}

          {steerable && (
            <div className="live-row">
              <label>
                <span>Next measurement
                  <Info>
                    Optional. A recommender proposes the next acquisition parameters. Bayesian optimization
                    works on one tracked quantity. The language model takes a goal in words, which can name
                    several quantities and constraints, and answers every few frames without holding up the
                    stream. Nothing is applied unless you accept it or choose automatic.
                  </Info>
                </span>
                <select value={recommender} onChange={(e) => setRecommender(e.target.value as LiveConfig["recommender"])}>
                  <option value="none">No recommender</option>
                  <option value="gp">Bayesian optimization</option>
                  <option value="llm">Language model</option>
                </select>
              </label>
              {recommender === "gp" && (
                <>
                  <label><span>Quantity</span>
                    <select value={objectiveKey} onChange={(e) => setObjectiveKey(e.target.value)}>
                      {formOutputs.map((k) => <option key={k}>{k}</option>)}
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
                <label className="grow"><span>Goal</span>
                  <input type="text" value={objective} onChange={(e) => setObjective(e.target.value)}
                    placeholder="G position to ±0.5 cm⁻¹ with the shortest frames, no laser heating" />
                </label>
              )}
            </div>
          )}

          <details className="live-options">
            <summary>Options</summary>
            <div className="live-row">
              <label><span>Stop after
                  <Info>Number of frames. Leave empty to run until you press Stop.</Info>
                </span>
                <input type="number" min={1} placeholder="until stopped" value={nFrames}
                  onChange={(e) => setNFrames(e.target.value)} />
              </label>
              <label><span>Pause between frames (s)</span>
                <input type="number" min={0} step={0.5} value={interval} onChange={(e) => setIntervalS(e.target.value)} />
              </label>
              {reference === FIRST_FRAME && (
                <label><span>Reference frames
                    <Info>
                      How many frames the recipe is planned from. With several, the plan sees what moves,
                      appears or is only noise before the recipe is locked on the last of them. Each extra
                      frame adds a few seconds.
                    </Info>
                  </span>
                  <input type="number" min={1} max={25} value={refFrames}
                    onChange={(e) => setRefFrames(e.target.value)} />
                </label>
              )}
              {reference === FIRST_FRAME && (
                <label><span>Reference analysis
                    <Info>Thorough takes a few minutes and is verified. Quick and extract are faster and less checked.</Info>
                  </span>
                  <select value={profile} onChange={(e) => setProfile(e.target.value)}>
                    <option value="thorough">thorough</option>
                    <option value="quick">quick</option>
                    <option value="extract">extract</option>
                  </select>
                </label>
              )}
              {steerable && recommender !== "none" && (
                <label><span>Recommendations</span>
                  <select value={apply} onChange={(e) => setApply(e.target.value as LiveConfig["apply"])}>
                    <option value="never">I accept them by hand</option>
                    <option value="valid">apply automatically</option>
                  </select>
                </label>
              )}
              {steerable && recommender === "llm" && (
                <label><span>Every N frames</span>
                  <input type="number" min={2} value={every} onChange={(e) => setEvery(+e.target.value)} />
                </label>
              )}
              <label className="live-check">
                <input type="checkbox" checked={autoEscalate} onChange={(e) => setAutoEscalate(e.target.checked)} />
                <span>Rebuild the recipe automatically</span>
              </label>
            </div>
          </details>

          <div>
            <button className="primary" onClick={start} disabled={!ready}>Start</button>
          </div>
        </div>
      </div>
    );
  }

  // ── a run exists ─────────────────────────────────────────────
  const frames = snap?.frames ?? [];
  const st = snap?.status ?? {};
  const latest = frames[frames.length - 1];
  const rec = snap?.recommendation ?? null;
  const inst = snap?.instrument;
  const keys = snap?.output_keys ?? [];
  const events = snap?.events ?? [];
  const total = snap?.n_frames_total ?? null;
  const count = st.frames ?? frames.length;
  const flagCounts = Object.entries(st.flag_counts ?? {});
  const rules = events
    .filter((e) => e.event === "reanchor" && typeof e.step === "number")
    .map((e) => ({ x: e.step as number, label: "new recipe" }));
  const flagged = frames.filter((f) => f.flags.length).map((f) => ({
    x: f.step, label: f.flags.map((x) => FLAG_WORDS[x] ?? x).join(", "),
  }));
  const hasTruth = keys.some((k) => frames.some((f) => f.truth && k in f.truth));
  const params = snap?.current_params ?? {};
  const schema = Object.entries(inst?.schema ?? {});
  const pending = Object.fromEntries(
    Object.entries(edits).filter(([k, v]) => v !== "" && String(params[k]) !== v).map(([k, v]) => [k, isNaN(+v) ? v : +v]),
  );
  const recIsCurrent = !!rec?.params &&
    Object.entries(rec.params).every(([k, v]) => String(params[k]) === String(v) || Number(params[k]) === Number(v));
  const curve = snap?.latest;
  const r2 = latest?.features.fit_r_squared;

  return (
    <div className="live-panel">
      <div className="live-head">
        <h3>{inst?.technique ?? inst?.name}</h3>
        <span className={`live-state ${state}`}>
          {state === "arming" ? "◌ preparing" : state === "running" ? "● running"
            : state === "done" ? "✓ finished" : state === "stopped" ? "■ stopped" : "✕ error"}
        </span>
        {count > 0 && (
          <span className="live-meta">
            frame {count}{total ? ` of ${total}` : ""} · {(st.latency_s?.median ?? 0).toFixed(1)} s per frame
          </span>
        )}
        <span className="live-meta">{clock(snap?.elapsed_s ?? 0)}</span>
        <Info>
          <ul>
            <li>{st.clean_frames ?? 0} of {count} frames clean</li>
            {flagCounts.map(([k, n]) => <li key={k}>{n} flagged: {FLAG_WORDS[k] ?? k}</li>)}
            <li>{st.llm_calls_in_frames ?? 0} model calls while answering frames</li>
            <li>{st.reanchors ?? 0} recipe rebuilds</li>
            <li>Slowest frame {(st.latency_s?.max ?? 0).toFixed(1)} s</li>
            <li>Recipe {(st.recipe?.id ?? "").slice(0, 8) || "not locked yet"}</li>
            <li>{snap?.run_dir}</li>
          </ul>
        </Info>
        <span className="live-head-actions">
          {live
            ? <button onClick={() => act(() => api.liveStop(sessionId))} disabled={busy}>Stop</button>
            : <button className="primary small" onClick={() => act(() => api.liveClear(sessionId))} disabled={busy}>New run</button>}
        </span>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {snap?.error && <div className="error-banner">{snap.error}</div>}
      {snap?.note && <p className="caption">{snap.note}</p>}
      {state === "arming" && (
        <p className="caption">
          Locking the recipe.
          <Info>
            The reference is analysed in full, its verified script becomes the recipe, and the tracked
            quantities are fixed to their names. With a first-frame reference this takes a few minutes. A
            past analysis is adopted in seconds.
          </Info>
        </p>
      )}
      {latest && latest.flags.length > 0 && (
        <div className="live-flags">
          ▲ Frame {latest.step}: {latest.flags.map((f) => FLAG_WORDS[f] ?? f).join(", ")}.
          {st.escalating ? " Rebuilding the recipe. Frames are still answered." : ""}
        </div>
      )}

      <div className="live-grid">
        {frames.length > 0 && (
          <section className="live-card">
            <div className="live-card-head">
              <h4>Tracked</h4>
              <Info>
                One chart per tracked quantity against frame number. Triangles mark flagged frames and a
                dotted line marks a new recipe. Several quantities get several charts, never a shared axis.
              </Info>
              {hasTruth && (
                <span className="live-legend inline">
                  <span><i className="key series" /> measured</span>
                  <span><i className="key reference" /> simulated truth</span>
                </span>
              )}
            </div>
            {keys.length === 0 && <p className="caption">This recipe reports no numeric quantities.</p>}
            {keys.map((k, i) => {
              const err = latest?.features[`${k}_err`];
              return (
                <div key={k} className="live-multiple">
                  <div className="live-multiple-head">
                    <code title={inst?.outputs?.[k]}>{k}</code>
                    <b>{fmt(latest?.features[k])}</b>
                    {typeof err === "number" && <span className="caption">± {fmtShort(err)}</span>}
                  </div>
                  <LiveChart
                    compact legend={false} xInteger
                    series={frames.map((f) => ({ x: f.step, y: f.features[k] ?? null }))}
                    seriesLabel={k}
                    reference={hasTruth ? frames.map((f) => ({ x: f.step, y: f.truth?.[k] ?? null })) : undefined}
                    referenceLabel="simulated truth"
                    flagged={flagged} rules={rules}
                    xLabel={i === keys.length - 1 ? "frame" : undefined}
                    height={i === keys.length - 1 ? 116 : 104}
                  />
                </div>
              );
            })}
          </section>
        )}

        {curve && (
          <section className="live-card">
            <div className="live-card-head">
              <h4>{curve.step === 0 ? "Reference" : `Frame ${curve.step}`}</h4>
              {typeof r2 === "number" && curve.step > 0 && <span className="caption">R² {r2.toFixed(3)}</span>}
              <Info>
                The latest measurement with the recipe's fitted model drawn over it. This is the analysis
                result for the frame, produced without a model call.
              </Info>
            </div>
            <LiveChart
              series={curve.x.map((x, i) => ({ x, y: curve.y[i] ?? null }))}
              seriesLabel="data"
              overlay={curve.fit ? curve.x.map((x, i) => ({ x, y: curve.fit?.[i] ?? null })) : undefined}
              overlayLabel="fit"
              xLabel={inst?.x_axis} yLabel={inst?.y_axis} height={keys.length > 2 ? 300 : 240}
            />

            {schema.length > 0 && frames.length > 0 && (
              <div className="live-controls">
                <div className="live-card-head">
                  <h4>Acquisition</h4>
                  <Info>
                    The parameters in use. Change a value and press Use to apply it from the next frame. The
                    instrument's limits are checked first.
                    <ul>{schema.map(([k, p]) => (
                      <li key={k}><code>{k}</code> {p.low !== undefined ? `${p.low} to ${p.high}` : ""} {p.units ?? ""}. {p.description}</li>
                    ))}</ul>
                  </Info>
                </div>
                <div className="live-params">
                  {schema.map(([k, p]) => (
                    <label key={k}>
                      <span><code>{k}</code>{p.units ? ` (${p.units})` : ""}</span>
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
                    Use
                  </button>
                </div>
                {rec && (rec.params || rec.kind === "hold" || rec.protocol || !rec.valid) && (
                  <div className={`live-rec ${rec.valid ? "" : "invalid"}`}>
                    <div className="live-rec-head">
                      <b>{rec.kind === "hold" ? "Keep these settings" : "Recommended"}</b>
                      <span className="caption">from frame {rec.based_on_step}</span>
                      <Info>
                        {rec.rationale}
                        <ul>
                          <li>Source: {rec.source}</li>
                          {rec.acquisition_skill && <li>Guided by the {rec.acquisition_skill} acquisition skill</li>}
                        </ul>
                      </Info>
                    </div>
                    {rec.params && (
                      <div className="tool-chips">
                        {Object.entries(rec.params).map(([k, v]) => (
                          <span key={k} className="live-chip"><code>{k}</code> {typeof v === "number" ? fmt(v) : v}</span>
                        ))}
                      </div>
                    )}
                    {rec.protocol && <pre className="live-protocol">{rec.protocol}</pre>}
                    {!rec.valid && (
                      <p className="caption warn">Refused. {(rec.problems ?? []).join(". ")}</p>
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
              </div>
            )}
          </section>
        )}
      </div>

      {events.length > 0 && (
        <details className="live-options">
          <summary>Events ({events.length})</summary>
          <ul className="live-events">
            {[...events].reverse().map((e, i) => (
              <li key={i}>
                <span className="caption">{typeof e.step === "number" ? `frame ${e.step}` : ""}</span>
                {describeEvent(e)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
