import { useCallback, useContext, useEffect, useState } from "react";
import { api, type TelemetrySnapshot, type ToolCall, type WorkerAgent } from "../api";
import { UIContext } from "../UIContext";

/** The Telemetry tab's lower half — what the meta session's agents actually
 * did, from the read-only `/telemetry` snapshot (the port of Streamlit's
 * telemetry tab minus the graphviz graph, which the sidebar tree already
 * shows as context-flow edges):
 *
 *  - Tool sequence: every tool call each layer's LLM made, in order, with
 *    the input / output *shape* in the table and the actual JSON on click.
 *  - Worker agents: each sub-agent's action_history (curve fitting, BO,
 *    scalarizer, …) with outcomes; a row expands to its actions, an action
 *    to its input / result / rationale.
 *  - Analysis reports: each analysis's scientific claims and reasoning.
 *
 * Polled every 3 s while a turn runs — the tool sequence reads the agents'
 * live message lists, so it moves mid-turn — and refreshed on every ledger
 * change otherwise. */

const LAYER_LABELS: [string, string][] = [
  ["meta", "Meta-agent"],
  ["analysis", "Analysis specialist"],
  ["planning", "Planning specialist"],
];

const JSON_CAP = 4000;

function typeName(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return "bool";
  if (typeof v === "number") return Number.isInteger(v) ? "int" : "float";
  if (typeof v === "string") return "str";
  if (Array.isArray(v)) return `list[${v.length}]`;
  if (typeof v === "object") return `dict[${Object.keys(v as object).length}]`;
  return typeof v;
}

/** Type signature of a value: for a dict its keys mapped to value types,
 * otherwise the bare type. Shape without content, so the table stays narrow. */
function typeSummary(obj: unknown, limit = 160): string {
  if (obj === null || obj === undefined) return "—";
  let s: string;
  if (Array.isArray(obj)) s = `list[${obj.length}]`;
  else if (typeof obj === "object") {
    const entries = Object.entries(obj as Record<string, unknown>);
    s = entries.length ? `{${entries.map(([k, v]) => `${k}: ${typeName(v)}`).join(", ")}}` : "{}";
  } else s = typeName(obj);
  return s.length <= limit ? s : s.slice(0, limit - 1) + "…";
}

function asJson(v: unknown): string {
  let s: string;
  try {
    s = typeof v === "string" ? v : JSON.stringify(v, null, 2) ?? "null";
  } catch {
    s = String(v);
  }
  return s.length <= JSON_CAP ? s : s.slice(0, JSON_CAP) + `\n… (${s.length - JSON_CAP} more characters)`;
}

function shortTime(ts: string | null | undefined): string {
  if (!ts) return "";
  const t = ts.includes("T") ? ts.split("T", 2)[1] : ts;
  return t.slice(0, 8);
}

function relativeTo(root: string, path: string): string {
  const r = root.endsWith("/") ? root : root + "/";
  return path.startsWith(r) ? path.slice(r.length) : path;
}

export function TelemetryDetails({
  sessionId,
  active,
  running,
  ledgerVersion,
}: {
  sessionId: string;
  active: boolean;
  running: boolean;
  ledgerVersion: unknown; // any change (a delegations event) triggers a refresh
}) {
  const ui = useContext(UIContext);
  const [tel, setTel] = useState<TelemetrySnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.telemetry(sessionId).then((t) => { setTel(t); setError(null); })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [sessionId]);

  useEffect(() => {
    if (active) refresh();
  }, [active, refresh, ledgerVersion]);

  useEffect(() => {
    if (!active || !running) return;
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [active, running, refresh]);

  if (error) return <p className="caption warn">Telemetry unavailable: {error}</p>;
  if (!tel) return null;

  const seq = tel.tool_sequence ?? {};
  const layers = LAYER_LABELS.filter(([k]) => (seq[k]?.calls?.length ?? 0) > 0);
  const agents = tel.agents ?? [];
  const reports = tel.analysis_reports ?? [];
  const root = tel.meta?.session_dir ?? "";

  return (
    <div className="tel-details">
      <section className="tools-section">
        <h3>Tool sequence</h3>
        <p className="caption">
          Every tool call each agent's LLM made, in order. The table shows the shape of
          each input and output; click a row for the actual arguments and result.
        </p>
        {layers.length === 0 ? (
          <p className="caption">No tool calls recorded yet.</p>
        ) : (
          layers.map(([key, label]) => (
            <ToolSequence
              key={key}
              label={label}
              calls={seq[key].calls}
              source={seq[key].source ? relativeTo(root, seq[key].source) : null}
              onOpenSource={ui.openInFiles}
            />
          ))
        )}
      </section>

      <section className="tools-section">
        <h3>
          Worker agents <span className="caption">({agents.length})</span>
        </h3>
        <p className="caption">
          The sub-agents the specialists ran and each one's action history.
        </p>
        {agents.length === 0 ? (
          <p className="caption">No worker agent has recorded actions yet.</p>
        ) : (
          <table className="tel-table">
            <thead>
              <tr>
                <th>specialist</th><th>agent</th><th>status</th><th>actions</th>
                <th>outcomes</th><th>first</th><th>last</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((a, i) => <WorkerRow key={a.source_file ?? i} agent={a} />)}
            </tbody>
          </table>
        )}
      </section>

      {reports.length > 0 && (
        <section className="tools-section">
          <h3>
            Analysis reports <span className="caption">({reports.length})</span>
          </h3>
          {reports.map((r) => (
            <details key={r.report_file} className="tel-report">
              <summary>
                <code>{r.analysis_id}</code>
                <span className={`tel-status status-${r.status ?? ""}`}>{r.status ?? "—"}</span>
                <span className="caption">
                  {r.claims.length} claim{r.claims.length === 1 ? "" : "s"}
                </span>
              </summary>
              {r.claims.length > 0 && (
                <ul className="deleg-list">
                  {r.claims.map((c, i) => (
                    <li key={i}>
                      {c.claim}
                      {c.impact && <span className="caption"> — {c.impact}</span>}
                    </li>
                  ))}
                </ul>
              )}
              {r.detailed_analysis && <pre className="tel-json">{r.detailed_analysis}</pre>}
              <button
                type="button"
                className="file-chip-btn"
                title="Open in Files"
                onClick={() => ui.openInFiles(relativeTo(root, r.report_file))}
              >
                analysis_results.json
              </button>
            </details>
          ))}
        </section>
      )}
    </div>
  );
}

function ToolSequence({
  label,
  calls,
  source,
  onOpenSource,
}: {
  label: string;
  calls: ToolCall[];
  source: string | null;
  onOpenSource: (path: string) => void;
}) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="tel-layer">
      <div className="tel-layer-head">
        <strong>{label}</strong>
        <span className="caption">
          {calls.length} tool call{calls.length === 1 ? "" : "s"}
        </span>
        {source && (
          <button
            type="button"
            className="link-btn"
            title="The full chat history this sequence was read from"
            onClick={() => onOpenSource(source)}
          >
            full history
          </button>
        )}
      </div>
      <table className="tel-table">
        <thead>
          <tr><th>#</th><th>tool</th><th>input</th><th>output</th><th>status</th></tr>
        </thead>
        <tbody>
          {calls.map((c, i) => {
            const isOpen = open === i;
            return [
              <tr
                key={i}
                className={`tel-row${isOpen ? " open" : ""}`}
                onClick={() => setOpen(isOpen ? null : i)}
              >
                <td className="caption">{i + 1}</td>
                <td><code>{c.tool}</code></td>
                <td className="tel-shape">{typeSummary(c.args)}</td>
                <td className="tel-shape">{typeSummary(c.result)}</td>
                <td><span className={`tel-status status-${c.status}`}>{c.status}</span></td>
              </tr>,
              isOpen && (
                <tr key={`${i}-d`} className="tel-row-detail">
                  <td colSpan={5}>
                    <div className="deleg-field">
                      <span className="deleg-k">Arguments</span>
                      <pre className="tel-json">{asJson(c.args)}</pre>
                    </div>
                    <div className="deleg-field">
                      <span className="deleg-k">Result</span>
                      <pre className="tel-json">
                        {c.status === "pending" ? "(no result yet)" : asJson(c.result)}
                      </pre>
                    </div>
                  </td>
                </tr>
              ),
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}

function WorkerRow({ agent }: { agent: WorkerAgent }) {
  const [open, setOpen] = useState(false);
  const [openAction, setOpenAction] = useState<number | null>(null);
  const o = agent.outcomes;
  return (
    <>
      <tr className={`tel-row${open ? " open" : ""}`} onClick={() => setOpen(!open)}>
        <td>{agent.specialist}</td>
        <td><strong>{agent.name}</strong></td>
        <td><span className={`tel-status status-${agent.status ?? ""}`}>{agent.status ?? "—"}</span></td>
        <td>{agent.action_count}</td>
        <td className="tel-shape">
          {o.success} ✓{o.error ? ` · ${o.error} ✗` : ""}{o.other ? ` · ${o.other} other` : ""}
        </td>
        <td className="caption">{shortTime(agent.first_timestamp)}</td>
        <td className="caption">{shortTime(agent.last_timestamp)}</td>
      </tr>
      {open && (
        <tr className="tel-row-detail">
          <td colSpan={7}>
            <div className="caption tel-by-type">
              {Object.entries(agent.actions_by_type).map(([k, n]) => (
                <span key={k} className="deleg-chip">{k} ×{n}</span>
              ))}
            </div>
            <table className="tel-table tel-actions">
              <thead>
                <tr><th>#</th><th>time</th><th>action</th><th>status</th><th>rationale</th></tr>
              </thead>
              <tbody>
                {agent.actions.map((a, i) => {
                  const isOpen = openAction === i;
                  return [
                    <tr
                      key={i}
                      className={`tel-row${isOpen ? " open" : ""}`}
                      onClick={(e) => { e.stopPropagation(); setOpenAction(isOpen ? null : i); }}
                    >
                      <td className="caption">{i + 1}</td>
                      <td className="caption">{shortTime(a.timestamp)}</td>
                      <td><code>{a.action}</code></td>
                      <td><span className={`tel-status status-${a.status}`}>{a.status}</span></td>
                      <td className="tel-shape">{a.rationale ?? ""}</td>
                    </tr>,
                    isOpen && (
                      <tr key={`${i}-d`} className="tel-row-detail">
                        <td colSpan={5}>
                          <div className="deleg-field">
                            <span className="deleg-k">Input</span>
                            <pre className="tel-json">{asJson(a.input)}</pre>
                          </div>
                          <div className="deleg-field">
                            <span className="deleg-k">Result</span>
                            <pre className="tel-json">{asJson(a.result)}</pre>
                          </div>
                          {a.feedback != null && a.feedback !== "" && (
                            <div className="deleg-field">
                              <span className="deleg-k">Feedback</span>
                              <pre className="tel-json">{asJson(a.feedback)}</pre>
                            </div>
                          )}
                        </td>
                      </tr>
                    ),
                  ];
                })}
              </tbody>
            </table>
          </td>
        </tr>
      )}
    </>
  );
}
