import type { DelegationRow } from "./api";

/** Pure layout + routing for the delegation dependency graph (no React, no
 * DOM), so it can be checked headlessly: `scripts/graph_check.ts` runs a
 * set of ledger scenarios through `computeGraph` and asserts that the
 * drawn edges are exactly the ledger's `context_from` relations, that
 * every edge starts and ends on the boxes it names, that no edge crosses
 * any other box, and that the flow runs top to bottom.
 *
 * Nodes: one per delegation, except the parallel branches of one
 * run_fanout call (same `fanout_group`), which collapse into a single
 * stacked node. Edges: one grey dispatch edge root → node, and one blue
 * context edge per (source node → consumer node) relation. */

export const NODE_W = 168;
export const NODE_H = 54;
const GAP_X = 22;
const GAP_Y = 46;
const PAD = 12;
export const MAX_PER_ROW = 6; // a wider layer wraps into several rows
const HIT_MARGIN = 6;

export interface GraphNode {
  id: string; // "d:<index>" or "g:<fanout_group>"
  rows: DelegationRow[];
  group: string | null; // fanout_group when collapsed
  sources: string[]; // node-level context_from, deduplicated, self excluded
}

export interface PlacedNode {
  node: GraphNode;
  x: number;
  y: number;
  layer: number; // visual row; 0 = directly under the root
}

export interface GraphEdge {
  kind: "dispatch" | "context";
  from: string; // "root" or a node id
  to: string; // a node id
  d: string; // SVG path
  mid: { x: number; y: number }; // where a label goes
  bow: number; // 0 = straight; else the channel's offset from the straight midline
}

export interface Graph {
  nodes: GraphNode[];
  placed: PlacedNode[];
  root: { x: number; y: number };
  width: number;
  height: number;
  edges: GraphEdge[];
  nodeOfIndex: Map<number, string>;
}

export interface Rect { x: number; y: number; w: number; h: number }

export function buildNodes(rows: DelegationRow[]): GraphNode[] {
  const nodeOf = new Map<number, string>();
  const nodes: GraphNode[] = [];
  const groups = new Map<string, GraphNode>();
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

/* ── routing ──────────────────────────────────────────────────────── */

/** Rows are separated by a band of GAP_Y that holds no box, and columns
 * by a gap of GAP_X. An edge is therefore routed as: a sweep from the
 * source's bottom centre into a vertical channel, made entirely inside
 * the free band below the source row; the channel straight down; a sweep
 * from the channel to the target's top centre, entirely inside the free
 * band above the target row. Only the channel can meet a box, so the
 * candidates are vertical lanes (see `lanes` in computeGraph): the column
 * gaps of every row, the strip beside each row, the canvas edges. A straight line is
 * tried first — between adjacent rows it can never cross a box, and a
 * straight diagonal that misses everything reads best. */
const EDGE_CHANNEL = 20; // lane just inside the canvas edge
const LANE_CLEAR = HIT_MARGIN + 8; // a lane just outside a row's first / last box
const SWEEP = GAP_Y - HIT_MARGIN - 4; // vertical room for a sweep (stays off the next row)
const SAMPLES = 40;

interface Curve { d: string; at: (t: number) => { x: number; y: number }; bow: number }

function bez(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

export function straight(x1: number, y1: number, x2: number, y2: number): Curve {
  return { d: `M ${x1} ${y1} L ${x2} ${y2}`, bow: 0,
           at: (t) => ({ x: x1 + (x2 - x1) * t, y: y1 + (y2 - y1) * t }) };
}

/** Sweep → channel at `xc` → sweep. */
export function channel(x1: number, y1: number, x2: number, y2: number, xc: number): Curve {
  const g = Math.min(SWEEP, (y2 - y1) / 2);
  const ya = y1 + g, yb = y2 - g; // channel from ya down to yb
  const d = `M ${x1} ${y1} C ${x1} ${ya}, ${xc} ${y1}, ${xc} ${ya}`
          + ` L ${xc} ${yb} C ${xc} ${y2}, ${x2} ${yb}, ${x2} ${y2}`;
  const at = (t: number) => {
    if (t < 1 / 3) {
      const u = t * 3;
      return { x: bez(x1, x1, xc, xc, u), y: bez(y1, ya, y1, ya, u) };
    }
    if (t < 2 / 3) {
      const u = (t - 1 / 3) * 3;
      return { x: xc, y: ya + (yb - ya) * u };
    }
    const u = (t - 2 / 3) * 3;
    return { x: bez(xc, xc, x2, x2, u), y: bez(yb, y2, yb, y2, u) };
  };
  return { d, at, bow: xc - (x1 + x2) / 2 };
}

function hits(curve: Curve, rects: Rect[]): boolean {
  for (let i = 1; i < SAMPLES; i++) {
    const { x, y } = curve.at(i / SAMPLES);
    for (const r of rects) {
      if (x >= r.x - HIT_MARGIN && x <= r.x + r.w + HIT_MARGIN &&
          y >= r.y - HIT_MARGIN && y <= r.y + r.h + HIT_MARGIN) return true;
    }
  }
  return false;
}

/** The first candidate that keeps the edge off every obstacle: straight,
 * then channels in the column gaps nearest the endpoints, then the canvas
 * edges. Falls back to straight (never happens with the edge channels
 * available, which are box-free by construction). */
function route(x1: number, y1: number, x2: number, y2: number, obstacles: Rect[],
               lanes: number[]) {
  const line = straight(x1, y1, x2, y2);
  if (!hits(line, obstacles)) return { d: line.d, mid: line.at(0.5), bow: 0 };
  const midX = (x1 + x2) / 2;
  const ordered = [...lanes].sort((a, b) => Math.abs(a - midX) - Math.abs(b - midX));
  for (const xc of ordered) {
    const c = channel(x1, y1, x2, y2, xc);
    if (!hits(c, obstacles)) return { d: c.d, mid: c.at(0.5), bow: c.bow };
  }
  return { d: line.d, mid: line.at(0.5), bow: 0 };
}

/* ── layout ───────────────────────────────────────────────────────── */

export function computeGraph(rows: DelegationRow[]): Graph {
  const nodes = buildNodes(rows);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const nodeOfIndex = new Map<number, string>();
  for (const n of nodes) for (const r of n.rows) nodeOfIndex.set(r.index, n.id);

  // Layer = context depth: a node sits below everything it read from. A
  // cycle (defensive; the ledger cannot produce one) breaks at the node
  // that closes it.
  const depth = new Map<string, number>();
  const depthOf = (n: GraphNode, seen = new Set<string>()): number => {
    const cached = depth.get(n.id);
    if (cached !== undefined) return cached;
    if (seen.has(n.id)) return 0;
    seen.add(n.id);
    let d = 0;
    for (const src of n.sources) {
      const s = byId.get(src);
      if (s) d = Math.max(d, depthOf(s, seen) + 1);
    }
    depth.set(n.id, d);
    return d;
  };
  const layers: GraphNode[][] = [];
  for (const n of nodes) (layers[depthOf(n)] ??= []).push(n);

  // A layer wider than MAX_PER_ROW wraps into rows; within a wrapped
  // layer the nodes that feed later layers go to its last row, so their
  // outgoing edges never cross a sibling row.
  const isSource = new Set(nodes.flatMap((n) => n.sources));
  const visualRows: GraphNode[][] = [];
  for (const layer of layers) {
    const ordered = layer.length > MAX_PER_ROW
      ? [...layer.filter((n) => !isSource.has(n.id)), ...layer.filter((n) => isSource.has(n.id))]
      : layer;
    for (let i = 0; i < ordered.length; i += MAX_PER_ROW)
      visualRows.push(ordered.slice(i, i + MAX_PER_ROW));
  }

  const widest = Math.max(1, ...visualRows.map((l) => l.length));
  const margin = visualRows.length > 1 ? 2 * EDGE_CHANNEL + 4 : 0; // room for edge channels
  const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X + 2 * margin;
  const root = { x: width / 2 - NODE_W / 2, y: PAD };
  const placed: PlacedNode[] = [];
  visualRows.forEach((vr, li) => {
    const rowW = vr.length * NODE_W + (vr.length - 1) * GAP_X;
    const x0 = (width - rowW) / 2;
    vr.forEach((node, i) => {
      placed.push({ node, x: x0 + i * (NODE_W + GAP_X),
                    y: PAD + (li + 1) * (NODE_H + GAP_Y), layer: li });
    });
  });
  const height = PAD * 2 + (visualRows.length + 1) * NODE_H + visualRows.length * GAP_Y;

  // Edges: dispatch root → every node; context source → consumer.
  const at = new Map(placed.map((p) => [p.node.id, p]));
  const rectOf = (p: PlacedNode): Rect => ({ x: p.x, y: p.y, w: NODE_W, h: NODE_H });
  const rootRect: Rect = { x: root.x, y: root.y, w: NODE_W, h: NODE_H };
  const obstaclesExcept = (...ids: string[]) =>
    placed.filter((q) => !ids.includes(q.node.id)).map(rectOf);
  // Candidate vertical lanes: the column gaps of every row, the strip just
  // outside each row's first and last box, and the canvas edges. An edge
  // that cannot go straight takes the nearest lane that clears every box.
  const lanes = new Set<number>([EDGE_CHANNEL, width - EDGE_CHANNEL]);
  for (const vr of visualRows) {
    const xs = placed.filter((p) => vr.includes(p.node)).map((p) => p.x).sort((a, b) => a - b);
    lanes.add(xs[0] - LANE_CLEAR);
    lanes.add(xs[xs.length - 1] + NODE_W + LANE_CLEAR);
    for (let i = 0; i + 1 < xs.length; i++) lanes.add((xs[i] + NODE_W + xs[i + 1]) / 2);
  }
  const laneList = [...lanes].filter((x) => x >= EDGE_CHANNEL - 1 && x <= width - EDGE_CHANNEL + 1);
  const edges: GraphEdge[] = [];
  for (const p of placed) {
    const r = route(root.x + NODE_W / 2, root.y + NODE_H, p.x + NODE_W / 2, p.y,
                    obstaclesExcept(p.node.id), laneList);
    edges.push({ kind: "dispatch", from: "root", to: p.node.id, ...r });
  }
  for (const p of placed) {
    for (const src of p.node.sources) {
      const s = at.get(src);
      if (!s) continue;
      const x1 = s.x + NODE_W / 2, y1 = s.y + NODE_H;
      const x2 = p.x + NODE_W / 2, y2 = p.y;
      const r = route(x1, y1, x2, y2, [rootRect, ...obstaclesExcept(s.node.id, p.node.id)],
                      laneList);
      edges.push({ kind: "context", from: s.node.id, to: p.node.id, ...r });
    }
  }
  return { nodes, placed, root, width, height, edges, nodeOfIndex };
}

export function groupStatus(rows: DelegationRow[]): string {
  if (rows.some((r) => r.status === "running")) return "running";
  if (rows.every((r) => r.status === "error")) return "error";
  return "success";
}
