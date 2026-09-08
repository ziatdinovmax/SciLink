import { useContext, useEffect, useMemo, useState } from "react";
import type { DelegationRow, DelegationView } from "../api";
import { UIContext } from "../UIContext";

/** Mission-control view of the meta session's delegation ledger — the web
 * twin of the Streamlit sidebar tree + telemetry graph. Grouped by
 * specialist, rows colored by status, context-flow edges ("← #1, #2") on
 * each row, and a click expands the delegation's task, summary, findings,
 * files (which open in the Files tab) and warnings. Live: the ledger
 * arrives with the session snapshot and is refreshed by `delegations` SSE
 * events while a turn runs. */

const SPECIALIST_ICON: Record<string, string> = {
  analysis: "🧪",
  planning: "📋",
  simulation: "🧬",
  fusion: "🔗",
};
const SPECIALIST_ORDER = ["analysis", "planning", "simulation", "fusion"];

const STATUS_GLYPH: Record<string, string> = {
  success: "✓",
  error: "✗",
  interrupted: "⏹",
  cancelled: "⏹",
  running: "⋯",
};

function shortTime(ts: string | null | undefined): string {
  if (!ts) return "";
  const t = ts.includes("T") ? ts.split("T", 2)[1] : ts;
  return t.slice(0, 8);
}

function elapsed(start?: string | null, end?: string | null): string {
  if (!start) return "";
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : Date.now();
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return "";
  const s = Math.round((b - a) / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function DelegationsPanel({
  view,
  running,
}: {
  view: DelegationView | null;
  running: boolean;
}) {
  const ui = useContext(UIContext);
  const [open, setOpen] = useState<number | null>(null);
  const rows = view?.delegations ?? [];
  const subAgents = view?.sub_agents ?? {};

  const groups = useMemo(() => {
    const by = new Map<string, DelegationRow[]>();
    for (const r of rows) by.set(r.mode, [...(by.get(r.mode) ?? []), r]);
    const keys = [...by.keys()].sort(
      (a, b) =>
        (SPECIALIST_ORDER.indexOf(a) + 1 || 99) - (SPECIALIST_ORDER.indexOf(b) + 1 || 99) ||
        a.localeCompare(b),
    );
    return keys.map((k) => [k, by.get(k)!] as const);
  }, [rows]);

  const nRunning = rows.filter((r) => r.status === "running").length;
  const byIndex = new Map(rows.map((r) => [r.index, r]));

  // The elapsed time of a running delegation is computed at render; tick
  // once a second while any row is running so it keeps counting between
  // ledger events (which only arrive when a delegation opens or closes).
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!nRunning) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [nRunning]);

  return (
    <div className="deleg-panel">
      <div className="deleg-head">
        <span className="deleg-title">🎛️ Mission control</span>
        <span className="caption">
          {rows.length} delegation{rows.length === 1 ? "" : "s"}
          {nRunning ? ` · ${nRunning} running` : ""}
          {running && !nRunning ? " · meta is reasoning" : ""}
        </span>
      </div>
      {rows.length === 0 ? (
        <p className="caption deleg-empty">
          No delegations yet — describe a goal and the meta routes it to a
          specialist. Delegations appear here as they start.
        </p>
      ) : (
        <div className="deleg-tree">
          {groups.map(([mode, list]) => (
            <div key={mode} className="deleg-group">
              <div className="deleg-group-head">
                <span>{SPECIALIST_ICON[mode] ?? "•"}</span>
                <span className="deleg-group-name">{mode}</span>
                <span className="caption">({list.length})</span>
                {subAgents[mode]?.length ? (
                  <span className="caption deleg-subagents">
                    ↳ {subAgents[mode].join(", ")}
                  </span>
                ) : null}
              </div>
              {list.map((r) => {
                const isOpen = open === r.index;
                const ctx = r.context_from
                  .map((i) => `#${i}`)
                  .join(", ");
                return (
                  <div key={r.index} className={`deleg-row status-${r.status}`}>
                    <button
                      type="button"
                      className="deleg-row-main"
                      onClick={() => setOpen(isOpen ? null : r.index)}
                      title={r.task}
                    >
                      <span className="deleg-glyph">
                        {STATUS_GLYPH[r.status] ?? "•"}
                      </span>
                      <span className="deleg-idx">#{r.index}</span>
                      <span className="deleg-label">{r.label}</span>
                      {r.fanout && <span className="deleg-tag">fan-out</span>}
                      {r.timed_out && <span className="deleg-tag warn">timed out</span>}
                      {r.resumed && <span className="deleg-tag">resumed</span>}
                      {ctx && (
                        <span className="deleg-ctx" title="context flowed from">
                          ← {ctx}
                        </span>
                      )}
                      <span className="deleg-meta caption">
                        {r.status}
                        {r.timestamp ? ` · ${shortTime(r.timestamp)}` : ""}
                        {elapsed(r.timestamp, r.completed_at)
                          ? ` · ${elapsed(r.timestamp, r.completed_at)}`
                          : ""}
                      </span>
                    </button>
                    {isOpen && (
                      <div className="deleg-detail">
                        <div className="deleg-field">
                          <span className="deleg-k">Task</span>
                          <span>{r.task}</span>
                        </div>
                        {r.context_from.length > 0 && (
                          <div className="deleg-field">
                            <span className="deleg-k">Context from</span>
                            <span>
                              {r.context_from.map((i) => {
                                const src = byIndex.get(i);
                                return (
                                  <span key={i} className="deleg-chip">
                                    #{i}
                                    {src ? ` ${src.label}` : ""}
                                  </span>
                                );
                              })}
                            </span>
                          </div>
                        )}
                        {r.labels.length > 0 && (
                          <div className="deleg-field">
                            <span className="deleg-k">Fused</span>
                            <span>{r.labels.join(", ")}</span>
                          </div>
                        )}
                        {r.summary && (
                          <div className="deleg-field">
                            <span className="deleg-k">Summary</span>
                            <span className="deleg-summary">{r.summary}</span>
                          </div>
                        )}
                        {r.key_findings.length > 0 && (
                          <div className="deleg-field">
                            <span className="deleg-k">Findings</span>
                            <ul className="deleg-list">
                              {r.key_findings.map((f, i) => (
                                <li key={i}>{f}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {r.files_produced.length > 0 && (
                          <div className="deleg-field">
                            <span className="deleg-k">Files</span>
                            <span className="deleg-files">
                              {r.files_produced.map((p) => (
                                <button
                                  key={p}
                                  type="button"
                                  className="file-chip-btn"
                                  title="Open in Files"
                                  onClick={() => ui.openInFiles(p)}
                                >
                                  {p.split("/").pop()}
                                </button>
                              ))}
                              {r.n_feature_tables > 0 && (
                                <span className="caption">
                                  {" "}· {r.n_feature_tables} feature table
                                  {r.n_feature_tables === 1 ? "" : "s"}
                                </span>
                              )}
                            </span>
                          </div>
                        )}
                        {r.warnings.length > 0 && (
                          <div className="deleg-field">
                            <span className="deleg-k">Warnings</span>
                            <ul className="deleg-list warn">
                              {r.warnings.map((w, i) => (
                                <li key={i}>{w}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {r.error && (
                          <div className="deleg-field">
                            <span className="deleg-k">Error</span>
                            <span className="deleg-error">{r.error}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
