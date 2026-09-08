import { useMemo } from "react";
import type { DelegationRow } from "../api";

/** Dependency graph of the delegation ledger — the web twin of the
 * Streamlit telemetry tab's graphviz chart, drawn as plain SVG so it needs
 * no renderer: a meta-agent root, one box per delegation colored by status
 * and annotated with the sub-agents its specialist used, grey dispatch
 * edges from the root, and blue "ctx" edges where one delegation's result
 * fed another as context. Nodes are layered by context depth (a delegation
 * sits below everything it read from), so the flow reads top to bottom. */

const FILL: Record<string, string> = {
  success: "#3fb950",
  error: "#f85149",
  running: "#d29922",
  interrupted: "#8893a5",
  cancelled: "#8893a5",
};
const DEFAULT_FILL = "#8893a5";
const ROOT_FILL = "#30363d";

const NODE_W = 168;
const NODE_H = 54;
const GAP_X = 22;
const GAP_Y = 46;
const PAD = 12;

function trunc(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

interface Placed {
  row: DelegationRow;
  x: number;
  y: number;
  layer: number; // 0 = directly under the root
}

/** An edge that skips layers would run straight through the boxes in
 * between (nodes alone in a layer sit on the same centre line), so it is
 * bowed sideways by one step per skipped layer — right for context
 * edges, left for dispatch edges — the way a layout engine routes
 * around obstacles. */
function bowedPath(x1: number, y1: number, x2: number, y2: number, bow: number): string {
  if (!bow) return `M ${x1} ${y1} L ${x2} ${y2}`;
  const my = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1 + bow} ${my}, ${x2 + bow} ${my}, ${x2} ${y2}`;
}
const BOW = 70;

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
  const { placed, width, height, root } = useMemo(() => {
    const byIndex = new Map(rows.map((r) => [r.index, r]));
    const depth = new Map<number, number>();
    const depthOf = (r: DelegationRow, seen = new Set<number>()): number => {
      const cached = depth.get(r.index);
      if (cached !== undefined) return cached;
      if (seen.has(r.index)) return 0; // defensive: a cycle in context_from
      seen.add(r.index);
      let d = 0;
      for (const src of r.context_from) {
        const s = byIndex.get(src);
        if (s) d = Math.max(d, depthOf(s, seen) + 1);
      }
      depth.set(r.index, d);
      return d;
    };
    const layers: DelegationRow[][] = [];
    for (const r of [...rows].sort((a, b) => a.index - b.index)) {
      const d = depthOf(r);
      (layers[d] ??= []).push(r);
    }
    const widest = Math.max(1, ...layers.map((l) => l.length));
    const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X;
    const root = { x: width / 2 - NODE_W / 2, y: PAD };
    const placed: Placed[] = [];
    layers.forEach((layer, li) => {
      const rowW = layer.length * NODE_W + (layer.length - 1) * GAP_X;
      const x0 = (width - rowW) / 2;
      layer.forEach((row, i) => {
        placed.push({
          row,
          x: x0 + i * (NODE_W + GAP_X),
          y: PAD + (li + 1) * (NODE_H + GAP_Y),
          layer: li,
        });
      });
    });
    const height = PAD * 2 + (layers.length + 1) * NODE_H + layers.length * GAP_Y;
    const margin = Math.max(0, layers.length - 1) * BOW * 0.75;
    for (const p of placed) p.x += margin;
    root.x += margin;
    return { placed, width: width + 2 * margin, height, root };
  }, [rows]);

  if (rows.length === 0) return null;
  const at = new Map(placed.map((p) => [p.row.index, p]));
  const modeLabel = metaMode
    ? metaMode.charAt(0).toUpperCase() + metaMode.slice(1).toLowerCase()
    : "";

  return (
    <div className="deleg-graph-wrap">
      <svg
        className="deleg-graph"
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

        {/* dispatch edges: root -> every delegation */}
        {placed.map((p) => (
          <path
            key={`d${p.row.index}`}
            d={bowedPath(root.x + NODE_W / 2, root.y + NODE_H,
                         p.x + NODE_W / 2, p.y, -BOW * p.layer)}
            fill="none"
            stroke="#8893a5"
            strokeWidth={1.2}
            markerEnd="url(#dg-arrow-grey)"
          />
        ))}
        {/* context edges: source -> consumer */}
        {placed.flatMap((p) =>
          p.row.context_from.map((src) => {
            const s = at.get(src);
            if (!s) return null;
            const x1 = s.x + NODE_W / 2;
            const y1 = s.y + NODE_H;
            const x2 = p.x + NODE_W / 2;
            const y2 = p.y;
            const my = (y1 + y2) / 2;
            const bow = BOW * (p.layer - s.layer - 1);
            return (
              <g key={`c${src}-${p.row.index}`}>
                <path
                  d={bowedPath(x1, y1, x2, y2, bow)}
                  fill="none"
                  stroke="#58a6ff"
                  strokeWidth={2}
                  markerEnd="url(#dg-arrow-blue)"
                />
                <text x={(x1 + x2) / 2 + bow * 0.75 + 4} y={my - 2} fontSize={10} fill="#58a6ff">
                  ctx
                </text>
              </g>
            );
          }),
        )}

        {/* root */}
        <g>
          <rect x={root.x} y={root.y} width={NODE_W} height={NODE_H} rx={8}
                fill={ROOT_FILL} stroke="#30363d" />
          <text x={root.x + NODE_W / 2} y={root.y + NODE_H / 2 + 4}
                textAnchor="middle" fontSize={12} fill="#fff" fontWeight={600}>
            Meta-agent{modeLabel ? ` (${modeLabel})` : ""}
          </text>
        </g>

        {/* delegations */}
        {placed.map((p) => {
          const r = p.row;
          const subs = subAgents[r.mode] ?? [];
          const isSel = selected === r.index;
          return (
            <g
              key={r.index}
              className="deleg-graph-node"
              onClick={() => onSelect(r.index)}
              style={{ cursor: "pointer" }}
            >
              <title>{`#${r.index} · ${r.mode} · ${r.status}\n${r.label}${subs.length ? `\n↳ ${subs.join(", ")}` : ""}`}</title>
              <rect x={p.x} y={p.y} width={NODE_W} height={NODE_H} rx={8}
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
        delegation's result fed the next as context. Click a box to expand it below.
      </p>
    </div>
  );
}
