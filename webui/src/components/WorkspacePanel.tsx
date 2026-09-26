import { useCallback, useEffect, useState } from "react";
import { api, type OpsStatus, type UsageSummary } from "../api";

function k(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function ago(s: number): string {
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function reason(r: string): string {
  const [kind, id] = r.split(":", 2);
  const short = id ? id.replace(/^[a-z_]+_session_/, "") : "";
  if (kind === "turn") return `a turn in ${short}`;
  if (kind === "live") return `a live run in ${short}`;
  if (kind === "memory_job") return `a memory job`;
  return r;
}

/** What the ops routes report about THIS workspace: whether the server is
 * busy or idle (and why), and the model usage of the current billing period
 * with its budget. Read on open and every 15 s while open; nothing here
 * changes anything. */
export function WorkspacePanel() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    Promise.all([api.opsStatus(), api.usage()])
      .then(([s, u]) => { setStatus(s); setUsage(u); setError(""); })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!open) return;
    load();
    const t = window.setInterval(load, 15000);
    return () => window.clearInterval(t);
  }, [open, load]);

  const sessions = usage ? Object.entries(usage.by_session).sort((a, b) => b[1].calls - a[1].calls) : [];
  const models = usage ? Object.entries(usage.by_model) : [];
  const budgetPct = usage && usage.budget_tokens ? Math.min(100, Math.round((usage.total_tokens / usage.budget_tokens) * 100)) : null;

  return (
    <div className="sidebar-section">
      <h3>
        <button type="button" className="link-btn" onClick={() => setOpen((o) => !o)}
                title="Server status and model usage of this workspace">
          {open ? "▾" : "▸"} Workspace{status?.workspace ? ` · ${status.workspace}` : ""}
        </button>
      </h3>
      {open && (
        <div style={{ fontSize: 12, lineHeight: 1.5 }}>
          {error && <div style={{ color: "var(--danger, #c33)" }}>{error}</div>}
          {status && (
            <div>
              <div>
                <strong>{status.state === "busy" ? "Busy" : status.state === "draining" ? "Draining" : "Idle"}</strong>
                {status.state === "idle" && ` for ${ago(status.idle_for_s)}`}
                {status.state === "draining" && " — finishing running work, taking no new turns"}
                {" · "}{status.sessions_live} live session{status.sessions_live === 1 ? "" : "s"}
                {" · up "}{ago(status.uptime_s)}
              </div>
              {status.busy.length > 0 && (
                <ul style={{ margin: "2px 0 4px 16px", padding: 0 }}>
                  {status.busy.map((r) => <li key={r}>{reason(r)}</li>)}
                </ul>
              )}
            </div>
          )}
          {usage && (
            <div style={{ marginTop: 6 }}>
              <div>
                <strong>{usage.calls}</strong> model call{usage.calls === 1 ? "" : "s"} this period ·{" "}
                {k(usage.prompt_tokens)} in / {k(usage.completion_tokens)} out
                {usage.llm_seconds > 0 && ` · ${ago(usage.llm_seconds)} in the model`}
              </div>
              {usage.budget_tokens !== null && (
                <div title={`${k(usage.total_tokens)} of ${k(usage.budget_tokens)} tokens`}>
                  Budget: {budgetPct}% used
                  {usage.over_budget ? " — spent; new turns wait for a new period" : ` (${k(usage.remaining_tokens ?? 0)} left)`}
                  <div style={{ height: 4, background: "var(--border)", borderRadius: 2, marginTop: 2 }}>
                    <div style={{ width: `${budgetPct}%`, height: "100%", borderRadius: 2,
                                  background: usage.over_budget ? "var(--danger, #c33)" : "var(--accent, #58a)" }} />
                  </div>
                </div>
              )}
              {models.length > 0 && (
                <div style={{ color: "var(--text-muted)", marginTop: 4 }}>
                  {models.map(([m, r]) => `${m.replace(/^[a-z_]+\//, "")}: ${r.calls}`).join(" · ")}
                </div>
              )}
              {sessions.length > 0 && (
                <table style={{ width: "100%", marginTop: 4, borderCollapse: "collapse" }}>
                  <tbody>
                    {sessions.slice(0, 8).map(([sid, r]) => (
                      <tr key={sid} title={sid}>
                        <td style={{ color: "var(--text-muted)", paddingRight: 6, whiteSpace: "nowrap" }}>
                          {sid === "unattributed" ? "unattributed" : sid.replace(/^[a-z_]+_session_/, "")}
                        </td>
                        <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{r.calls} · {k(r.prompt_tokens + r.completion_tokens)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
          {!status && !usage && !error && <div style={{ color: "var(--text-muted)" }}>Loading…</div>}
        </div>
      )}
    </div>
  );
}
