import { useMemo, useState } from "react";
import type { DelegationRow } from "../api";
import { computeGraph, groupStatus, NODE_H, NODE_W } from "../delegationGraph";

/** Dependency graph of the delegation ledger — the web twin of the
 * Streamlit telemetry tab's graphviz chart, drawn as plain SVG so it needs
 * no renderer: a meta-agent root, one box per delegation colored by status
 * and annotated with the sub-agents its specialist used, grey dispatch
 * edges from the root, and blue "ctx" edges where one delegation's result
 * fed another as context. Layout and routing live in
 * `../delegationGraph.ts` (pure, checked by `scripts/graph_check.ts`);
 * this file only draws. Edge paths carry `data-from` / `data-to` and node
 * groups `data-node`, so the rendered geometry can be audited against the
 * ledger in a browser. */

const FILL: Record<string, string> = {
  success: "#3fb950",
  error: "#f85149",
  running: "#d29922",
  interrupted: "#8893a5",
  cancelled: "#8893a5",
};
const DEFAULT_FILL = "#8893a5";
const ROOT_FILL = "#30363d";
const LABEL_MAX_SOURCES = 3; // "ctx" labels only where they stay legible

function trunc(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

export function DelegationGraph({
  rows,
  subAgents,
  metaMode,
  selected,
  onSelect,
}: {
  rows: DelegationRow[];
  subAgents: Record<string, string[]>;
  metaMode?: string | null;
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  const g = useMemo(() => computeGraph(rows), [rows]);
  // A wide ledger scaled to the panel became unreadable; past this width
  // the graph keeps its natural size and scrolls, with a fit toggle.
  const [fit, setFit] = useState<boolean | null>(null);
  if (rows.length === 0) return null;
  const fitted = fit ?? g.width <= 1100;
  const { placed, root, width, height, edges, nodeOfIndex } = g;
  const selectedId = selected == null ? null : nodeOfIndex.get(selected) ?? null;
  const modeLabel = metaMode
    ? metaMode.charAt(0).toUpperCase() + metaMode.slice(1).toLowerCase()
    : "";
  const sourcesOf = new Map(placed.map((p) => [p.node.id, p.node.sources.length]));

  return (
    <div className="deleg-graph-wrap">
      {g.width > 1100 && (
        <button type="button" className="link-btn deleg-graph-fit" onClick={() => setFit(!fitted)}>
          {fitted ? "Actual size" : "Fit to width"}
        </button>
      )}
      <svg
        className={`deleg-graph${fitted ? " fit" : ""}`}
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label="Delegation dependency graph"
      >
        <defs>
          <marker id="dg-arrow-grey" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8893a5" />
          </marker>
          <marker id="dg-arrow-blue" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#58a6ff" />
          </marker>
        </defs>

        {/* dispatch edges: root -> every node (faint below the first row) */}
        {edges.filter((e) => e.kind === "dispatch").map((e) => {
          const p = placed.find((q) => q.node.id === e.to)!;
          return (
            <path
              key={`d${e.to}`}
              data-kind="dispatch"
              data-from="root"
              data-to={e.to}
              d={e.d}
              fill="none"
              stroke="#8893a5"
              strokeWidth={p.layer === 0 ? 1.2 : 0.8}
              opacity={p.layer === 0 ? 1 : 0.45}
              markerEnd="url(#dg-arrow-grey)"
            />
          );
        })}
        {/* context edges: source -> consumer */}
        {edges.filter((e) => e.kind === "context").map((e) => {
          const many = (sourcesOf.get(e.to) ?? 0) > LABEL_MAX_SOURCES;
          return (
            <g key={`c${e.from}-${e.to}`}>
              <path
                data-kind="context"
                data-from={e.from}
                data-to={e.to}
                d={e.d}
                fill="none"
                stroke="#58a6ff"
                strokeWidth={many ? 1.2 : 2}
                opacity={many ? 0.7 : 1}
                markerEnd="url(#dg-arrow-blue)"
              />
              {!many && (
                <g>
                  <rect x={e.mid.x + 3} y={e.mid.y - 12} width={22} height={13} rx={3}
                        className="deleg-graph-halo" />
                  <text x={e.mid.x + 5} y={e.mid.y - 2} fontSize={10} fill="#58a6ff">
                    ctx
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* root */}
        <g data-node="root">
          <rect x={root.x} y={root.y} width={NODE_W} height={NODE_H} rx={8}
                fill={ROOT_FILL} stroke="#30363d" />
          <text x={root.x + NODE_W / 2} y={root.y + NODE_H / 2 + 4}
                textAnchor="middle" fontSize={12} fill="#fff" fontWeight={600}>
            Meta-agent{modeLabel ? ` (${modeLabel})` : ""}
          </text>
        </g>

        {/* nodes — a fan-out group is one stacked box */}
        {placed.map((p) => {
          const n = p.node;
          const isSel = selectedId === n.id;
          if (n.group) {
            const rs = n.rows;
            const nOk = rs.filter((r) => r.status === "success").length;
            const nErr = rs.filter((r) => r.status === "error").length;
            const nRun = rs.filter((r) => r.status === "running").length;
            const first = rs.find((r) => r.status === "error") ?? rs[0];
            const lo = Math.min(...rs.map((r) => r.index));
            const hi = Math.max(...rs.map((r) => r.index));
            const subs = subAgents[rs[0].mode] ?? [];
            const counts = [`${nOk} ✓`, nErr ? `${nErr} ✗` : "", nRun ? `${nRun} ⋯` : ""]
              .filter(Boolean).join(" · ");
            const fill = FILL[groupStatus(rs)] ?? DEFAULT_FILL;
            return (
              <g
                key={n.id}
                data-node={n.id}
                className="deleg-graph-node"
                onClick={() => onSelect(first.index)}
                style={{ cursor: "pointer" }}
              >
                <title>{`fan-out #${lo}–#${hi} (${rs.length} branches) · ${counts}\n${rs.map((r) => r.label).join("\n")}`}</title>
                <rect x={p.x + 6} y={p.y - 6} width={NODE_W} height={NODE_H} rx={8}
                      fill={fill} opacity={0.45} stroke="#30363d" />
                <rect x={p.x + 3} y={p.y - 3} width={NODE_W} height={NODE_H} rx={8}
                      fill={fill} opacity={0.7} stroke="#30363d" />
                <rect data-box="1" x={p.x} y={p.y} width={NODE_W} height={NODE_H} rx={8}
                      fill={fill}
                      stroke={isSel ? "#fff" : "#30363d"} strokeWidth={isSel ? 2 : 1} />
                <text x={p.x + NODE_W / 2} y={p.y + 17} textAnchor="middle"
                      fontSize={11} fill="#fff" fontWeight={600}>
                  #{lo}–#{hi} · fan-out ×{rs.length}
                </text>
                <text x={p.x + NODE_W / 2} y={p.y + 32} textAnchor="middle"
                      fontSize={11} fill="#fff">
                  {counts}
                </text>
                {subs.length > 0 && (
                  <text x={p.x + NODE_W / 2} y={p.y + 46} textAnchor="middle"
                        fontSize={10} fill="#fff" opacity={0.85}>
                    ↳ {trunc(subs.join(", "), 26)}
                  </text>
                )}
              </g>
            );
          }
          const r = n.rows[0];
          const subs = subAgents[r.mode] ?? [];
          return (
            <g
              key={n.id}
              data-node={n.id}
              className="deleg-graph-node"
              onClick={() => onSelect(r.index)}
              style={{ cursor: "pointer" }}
            >
              <title>{`#${r.index} · ${r.mode} · ${r.status}\n${r.label}${subs.length ? `\n↳ ${subs.join(", ")}` : ""}`}</title>
              <rect data-box="1" x={p.x} y={p.y} width={NODE_W} height={NODE_H} rx={8}
                    fill={FILL[r.status] ?? DEFAULT_FILL}
                    stroke={isSel ? "#fff" : "#30363d"} strokeWidth={isSel ? 2 : 1} />
              <text x={p.x + NODE_W / 2} y={p.y + 17} textAnchor="middle"
                    fontSize={11} fill="#fff" fontWeight={600}>
                #{r.index} · {r.mode}
              </text>
              <text x={p.x + NODE_W / 2} y={p.y + 32} textAnchor="middle"
                    fontSize={11} fill="#fff">
                {trunc(r.label, 24)}
              </text>
              {subs.length > 0 && (
                <text x={p.x + NODE_W / 2} y={p.y + 46} textAnchor="middle"
                      fontSize={10} fill="#fff" opacity={0.85}>
                  ↳ {trunc(subs.join(", "), 26)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <p className="caption">
        Grey edge = the meta dispatched the delegation. Blue edge = a
        delegation's result fed the next as context. A stacked box is one
        fan-out (its parallel branches). Click a box to expand it below.
      </p>
    </div>
  );
}
