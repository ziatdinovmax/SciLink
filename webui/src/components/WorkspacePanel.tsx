import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api, type OpsStatus, type UsageSummary } from "../api";
import { sessionLabel, sessionLabels, modelLabels } from "../sessionLabel";
import { VOCAB } from "../vocabulary";

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
  const where = id ? ` in session ${sessionLabel(id)}` : "";
  if (kind === "turn") return `a turn${where}`;
  if (kind === "live") return `a live run${where}`;
  if (kind === "memory_job") return `a memory job`;
  return r;
}

// Sessions listed by name; the rest are one row, so the rows add up.
const MAX_SESSIONS = 8;

/** One fact per line: what it is on the left, its value on the right. */
function Row({ label, value, title, strong }: { label: string; value: ReactNode; title?: string; strong?: boolean }) {
  return (
    <div className="ws-row" title={title}>
      <span className="ws-label">{label}</span>
      <span className={strong ? "ws-value strong" : "ws-value"}>{value}</span>
    </div>
  );
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

  const byCalls = (a: [string, { calls: number }], b: [string, { calls: number }]) => b[1].calls - a[1].calls;
  const sessions = usage ? Object.entries(usage.by_session).sort(byCalls) : [];
  const models = usage ? Object.entries(usage.by_model).sort(byCalls) : [];
  const shown = sessions.slice(0, MAX_SESSIONS);
  const rest = sessions.slice(MAX_SESSIONS);
  const sessionNames = sessionLabels(shown.map(([sid]) => sid));
  const modelNames = modelLabels(models.map(([m]) => m));
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
        <div className="ws-panel">
          {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
          {status && (
            <div>
              <div className="ws-state">
                <strong>{status.state === "busy" ? "Busy" : status.state === "draining" ? "Draining" : "Idle"}</strong>
                {status.state === "idle" && <span className="ws-label"> for {ago(status.idle_for_s)}</span>}
              </div>
              {status.state === "draining" && (
                <div className="ws-label">Finishing running work, taking no new turns</div>
              )}
              {status.busy.length > 0 && (
                <ul className="ws-busy">
                  {status.busy.map((r) => <li key={r}>{reason(r)}</li>)}
                </ul>
              )}
              <Row label="Live sessions" value={status.sessions_live} />
              <Row label="Server up" value={ago(status.uptime_s)} />
            </div>
          )}
          {usage && (
            <div>
              <h4 className="ws-head">This period</h4>
              {/* The dollar estimate, worded by the server; absent when no
                  call of the period had a known price. What it is an
                  estimate of is on hover. */}
              {usage.cost_text && (
                <Row label="Spent" value={usage.cost_text} title={VOCAB.cost_words.note} strong />
              )}
              <Row label="Model calls" value={usage.calls} />
              <Row label="Tokens in" value={k(usage.prompt_tokens)} />
              <Row label="Tokens out" value={k(usage.completion_tokens)} />
              {usage.llm_seconds > 0 && <Row label="Time in the model" value={ago(usage.llm_seconds)} />}
              {usage.budget_tokens !== null && (
                <div title={`${k(usage.total_tokens)} of ${k(usage.budget_tokens)} tokens`}>
                  <Row label="Budget used"
                       value={usage.over_budget ? `${budgetPct}%, spent` : `${budgetPct}% (${k(usage.remaining_tokens ?? 0)} left)`} />
                  <div className="ws-bar">
                    <div style={{ width: `${budgetPct}%`, background: usage.over_budget ? "var(--danger)" : "var(--accent)" }} />
                  </div>
                  {usage.over_budget && <div className="ws-label">New turns wait for a new period</div>}
                </div>
              )}
              {models.length > 0 && (
                <>
                  <h4 className="ws-head">Calls by model</h4>
                  {models.map(([m, r]) => (
                    <Row key={m} label={modelNames.get(m) ?? m} value={r.calls} title={m} />
                  ))}
                </>
              )}
              {sessions.length > 0 && (
                <>
                  <h4 className="ws-head">Calls by session</h4>
                  {shown.map(([sid, r]) => (
                    <Row key={sid} label={sessionNames.get(sid) ?? sid} value={r.calls}
                         title={`${sid} · ${k(r.prompt_tokens + r.completion_tokens)} tokens`} />
                  ))}
                  {rest.length > 0 && (
                    <Row label={`${rest.length} more`} value={rest.reduce((n, [, r]) => n + r.calls, 0)}
                         title={`${rest.length} sessions with fewer calls, together`} />
                  )}
                </>
              )}
            </div>
          )}
          {!status && !usage && !error && <div className="ws-label">Loading…</div>}
        </div>
      )}
    </div>
  );
}
