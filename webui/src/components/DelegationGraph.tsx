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

/** A graph node: one delegation, or a whole fan-out group (the N parallel
 * branches of one run_fanout call) collapsed into a single stacked box —
 * 30 branches as 30 boxes with 30 context edges in and 30 out is noise,
 * and the branches are siblings of one meta call anyway. */
interface GNode {
  id: string;
  rows: DelegationRow[];
  group: string | null; // fanout_group when collapsed
  sources: string[]; // node-level context_from
}

interface Placed {
  node: GNode;
  x: number;
  y: number;
  layer: number; // visual row; 0 = directly under the root
}

function buildNodes(rows: DelegationRow[]): GNode[] {
  const nodeOf = new Map<number, string>();
  const nodes: GNode[] = [];
  const groups = new Map<string, GNode>();
  for (const r of [...rows].sort((a, b) => a.index - b.index)) {
    if (r.fanout && r.fanout_group) {
      let g = groups.get(r.fanout_group);
      if (!g) {
        g = { id: `g:${r.fanout_group}`, rows: [], group: r.fanout_group, sources: [] };
        groups.set(r.fanout_group, g);
        nodes.push(g);
      }
      g.rows.push(r);
      nodeOf.set(r.index, g.id);
    } else {
      const n = { id: `d:${r.index}`, rows: [r], group: null, sources: [] };
      nodes.push(n);
      nodeOf.set(r.index, n.id);
    }
  }
  for (const n of nodes) {
    const seen = new Set<string>();
    for (const r of n.rows)
      for (const src of r.context_from) {
        const id = nodeOf.get(src);
        if (id && id !== n.id && !seen.has(id)) {
          seen.add(id);
          n.sources.push(id);
        }
      }
  }
  return nodes;
}

function groupStatus(rows: DelegationRow[]): string {
  if (rows.some((r) => r.status === "running")) return "running";
  if (rows.every((r) => r.status === "error")) return "error";
  return "success";
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
const MAX_PER_ROW = 6;   // a wider layer wraps into several rows
const LABEL_MAX_SOURCES = 3; // "ctx" labels only where they stay legible

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
    const nodes = buildNodes(rows);
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const depth = new Map<string, number>();
    const depthOf = (n: GNode, seen = new Set<string>()): number => {
      const cached = depth.get(n.id);
      if (cached !== undefined) return cached;
      if (seen.has(n.id)) return 0; // defensive: a cycle in context_from
      seen.add(n.id);
      let d = 0;
      for (const src of n.sources) {
        const s = byId.get(src);
        if (s) d = Math.max(d, depthOf(s, seen) + 1);
      }
      depth.set(n.id, d);
      return d;
    };
    const layers: GNode[][] = [];
    for (const n of nodes) {
      const d = depthOf(n);
      (layers[d] ??= []).push(n);
    }
    // A layer wider than MAX_PER_ROW wraps into several visual rows (a
    // 40-branch fan-out would otherwise be one 7000 px line). `layer` on
    // a placed node is its visual row, which is what edge bowing needs.
    const visualRows: GNode[][] = [];
    for (const layer of layers) {
      for (let i = 0; i < layer.length; i += MAX_PER_ROW) {
        visualRows.push(layer.slice(i, i + MAX_PER_ROW));
      }
    }
    const widest = Math.max(1, ...visualRows.map((l) => l.length));
    const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X;
    const root = { x: width / 2 - NODE_W / 2, y: PAD };
    const placed: Placed[] = [];
    visualRows.forEach((vr, li) => {
      const rowW = vr.length * NODE_W + (vr.length - 1) * GAP_X;
      const x0 = (width - rowW) / 2;
      vr.forEach((node, i) => {
        placed.push({
          node,
          x: x0 + i * (NODE_W + GAP_X),
          y: PAD + (li + 1) * (NODE_H + GAP_Y),
          layer: li,
        });
      });
    });
    const height = PAD * 2 + (visualRows.length + 1) * NODE_H + visualRows.length * GAP_Y;
    // Bows are capped (see below), so the side margin is bounded too.
    const margin = Math.min(2, Math.max(0, visualRows.length - 1)) * BOW;
    for (const p of placed) p.x += margin;
    root.x += margin;
    return { placed, width: width + 2 * margin, height, root };
  }, [rows]);

  if (rows.length === 0) return null;
  const at = new Map(placed.map((p) => [p.node.id, p]));
  const nodeOfIndex = new Map<number, string>();
  for (const p of placed) for (const r of p.node.rows) nodeOfIndex.set(r.index, p.node.id);
  const selectedId = selected == null ? null : nodeOfIndex.get(selected) ?? null;
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
            key={`d${p.node.id}`}
            d={bowedPath(root.x + NODE_W / 2, root.y + NODE_H,
                         p.x + NODE_W / 2, p.y, -Math.min(2, p.layer) * BOW)}
            fill="none"
            stroke="#8893a5"
            strokeWidth={p.layer === 0 ? 1.2 : 0.8}
            opacity={p.layer === 0 ? 1 : 0.45}
            markerEnd="url(#dg-arrow-grey)"
          />
        ))}
        {/* context edges: source -> consumer */}
        {placed.flatMap((p) =>
          p.node.sources.map((src) => {
            const s = at.get(src);
            if (!s) return null;
            const x1 = s.x + NODE_W / 2;
            const y1 = s.y + NODE_H;
            const x2 = p.x + NODE_W / 2;
            const y2 = p.y;
            const my = (y1 + y2) / 2;
            // Collinear nodes need the bow; a source off to the side does not.
            const skipped = p.layer - s.layer - 1;
            const bow = Math.abs(x1 - x2) < NODE_W / 2 ? BOW * Math.min(2, skipped) : 0;
            const many = p.node.sources.length > LABEL_MAX_SOURCES;
            return (
              <g key={`c${src}-${p.node.id}`}>
                <path
                  d={bowedPath(x1, y1, x2, y2, bow)}
                  fill="none"
                  stroke="#58a6ff"
                  strokeWidth={many ? 1.2 : 2}
                  opacity={many ? 0.7 : 1}
                  markerEnd="url(#dg-arrow-blue)"
                />
                {!many && (
                  <text x={(x1 + x2) / 2 + bow * 0.75 + 4} y={my - 2} fontSize={10} fill="#58a6ff">
                    ctx
                  </text>
                )}
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

        {/* delegations — a fan-out group is one stacked box */}
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
            return (
              <g
                key={n.id}
                className="deleg-graph-node"
                onClick={() => onSelect(first.index)}
                style={{ cursor: "pointer" }}
              >
                <title>{`fan-out #${lo}–#${hi} (${rs.length} branches) · ${counts}\n${rs.map((r) => r.label).join("\n")}`}</title>
                <rect x={p.x + 6} y={p.y - 6} width={NODE_W} height={NODE_H} rx={8}
                      fill={FILL[groupStatus(rs)] ?? DEFAULT_FILL} opacity={0.45} stroke="#30363d" />
                <rect x={p.x + 3} y={p.y - 3} width={NODE_W} height={NODE_H} rx={8}
                      fill={FILL[groupStatus(rs)] ?? DEFAULT_FILL} opacity={0.7} stroke="#30363d" />
                <rect x={p.x} y={p.y} width={NODE_W} height={NODE_H} rx={8}
                      fill={FILL[groupStatus(rs)] ?? DEFAULT_FILL}
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
        delegation's result fed the next as context. A stacked box is one
        fan-out (its parallel branches). Click a box to expand it below.
      </p>
    </div>
  );
}
