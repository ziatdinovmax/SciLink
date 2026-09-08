import { useEffect, useMemo, useState } from "react";
import type { DelegationRow, DelegationView } from "../api";

/** Compact mission-control tree for the sidebar — the web twin of the
 * Streamlit `_meta_delegation_tree`: specialist branches, one line per
 * delegation (status glyph, index, short label, "←#1" context edges),
 * colored by status, in a bounded scroll box, live while a turn runs.
 * A click hands the delegation to the Telemetry tab for its detail. */

const ICON: Record<string, string> = {
  analysis: "🧪",
  planning: "📋",
  simulation: "🧬",
  fusion: "🔗",
};
const ORDER = ["analysis", "planning", "simulation", "fusion"];
const GLYPH: Record<string, string> = {
  success: "✓",
  error: "✗",
  interrupted: "⏹",
  cancelled: "⏹",
  running: "⋯",
};

function clip(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function elapsed(start?: string | null, end?: string | null): string {
  if (!start) return "";
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : Date.now();
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return "";
  const s = Math.round((b - a) / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h${m % 60}m`;
}

export function DelegationTree({
  view,
  running,
  onSelect,
}: {
  view: DelegationView | null;
  running: boolean;
  onSelect: (index: number) => void;
}) {
  const rows = view?.delegations ?? [];
  const groups = useMemo(() => {
    const by = new Map<string, DelegationRow[]>();
    for (const r of rows) by.set(r.mode, [...(by.get(r.mode) ?? []), r]);
    return [...by.keys()]
      .sort(
        (a, b) =>
          (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99) ||
          a.localeCompare(b),
      )
      .map((k) => [k, by.get(k)!] as const);
  }, [rows]);
  const nRunning = rows.filter((r) => r.status === "running").length;

  // Keep the elapsed time of running rows counting between ledger events.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!nRunning) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [nRunning]);

  return (
    <div className="dtree">
      <div className="dtree-root">
        🎛️ Mission control
        <span className="caption">
          {" "}
          · {rows.length}
          {nRunning ? ` · ${nRunning} running` : running ? " · reasoning" : ""}
        </span>
      </div>
      {rows.length === 0 ? (
        <p className="caption dtree-empty">
          No delegations yet — the meta routes your goal to a specialist.
        </p>
      ) : (
        <div className="dtree-scroll">
          {groups.map(([mode, list], gi) => {
            const lastGroup = gi === groups.length - 1;
            return (
              <div key={mode} className="dtree-group">
                <div className="dtree-branch">
                  <span className="dtree-line">{lastGroup ? "└─" : "├─"}</span>
                  <span>{ICON[mode] ?? "•"}</span>
                  <span className="dtree-mode">{mode}</span>
                  <span className="caption">({list.length})</span>
                </div>
                {list.map((r, ri) => {
                  const last = ri === list.length - 1;
                  const ctx = r.context_from.map((i) => `#${i}`).join(",");
                  return (
                    <button
                      type="button"
                      key={r.index}
                      className={`dtree-row status-${r.status}`}
                      title={`${r.label} — ${r.status}${ctx ? ` (context from ${ctx})` : ""}. Click for details.`}
                      onClick={() => onSelect(r.index)}
                    >
                      <span className="dtree-line">
                        {lastGroup ? "   " : "│  "}
                        {last ? "└─" : "├─"}
                      </span>
                      <span className="dtree-glyph">{GLYPH[r.status] ?? "•"}</span>
                      <span className="dtree-idx">#{r.index}</span>
                      <span className="dtree-label">{clip(r.label, 28)}</span>
                      {ctx && <span className="dtree-ctx">←{ctx}</span>}
                      {r.status === "running" && (
                        <span className="dtree-time caption">
                          {elapsed(r.timestamp, r.completed_at)}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
