/** Headless check of the delegation graph's layout and routing.
 *
 *   npm run check:graph        (node --experimental-strip-types)
 *
 * Runs ledger scenarios through `computeGraph` and asserts, independently
 * of the module's own routing helpers (the path string is re-parsed and
 * re-sampled here):
 *   1. the context edges are exactly the ledger's `context_from`
 *      relations mapped to nodes (fan-out branches → their group node),
 *      deduplicated, self-edges dropped, unknown sources dropped;
 *   2. exactly one dispatch edge per node;
 *   3. every edge starts at the bottom centre of its `from` box (or the
 *      root) and ends at the top centre of its `to` box;
 *   4. no edge passes through a box it does not connect;
 *   5. a consumer sits below every source (except inside a cycle);
 *   6. boxes never overlap and stay inside the canvas.
 */
import { computeGraph, NODE_H, NODE_W, type Graph } from "../src/delegationGraph.ts";
import type { DelegationRow } from "../src/api.ts";

function row(index: number, mode: string, context_from: number[] = [], extra: Partial<DelegationRow> = {}): DelegationRow {
  return {
    index, mode, label: `d${index}`, task: `task ${index}`, status: "success",
    context_from, informed_by: [], fanout: false, fanout_group: null, labels: [],
    timestamp: null, completed_at: null, summary: "", key_findings: [],
    files_produced: [], n_feature_tables: 0, warnings: [], error: null,
    timed_out: false, resumed: false, ...extra,
  };
}
const fan = (index: number, group: string, cf: number[]) =>
  row(index, "analysis", cf, { fanout: true, fanout_group: group });

// ── scenarios ────────────────────────────────────────────────────────
const scenarios: Record<string, DelegationRow[]> = {
  single: [row(1, "analysis")],
  chain: [row(1, "analysis"), row(2, "analysis", [1]), row(3, "simulation", [2]),
          row(4, "planning", [3]), row(5, "analysis", [4])],
  diamond: [row(1, "analysis"), row(2, "analysis", [1]), row(3, "planning", [1]),
            row(4, "fusion", [2, 3])],
  skip_edges: [row(1, "analysis"), row(2, "analysis", [1]), row(3, "analysis", [2]),
               row(4, "planning", [1]), row(5, "analysis", [1, 3]), row(6, "analysis", [1, 2, 3, 4, 5])],
  fanout_fusion: [row(1, "analysis"), ...[2, 3, 4, 5, 6, 7, 8, 9].map((i) => fan(i, "fanout_2", [1])),
                  row(10, "fusion", [2, 3, 4, 5, 6, 7, 8, 9]), row(11, "planning", [10]),
                  row(12, "analysis", [5]),           // one branch feeds a non-member
                  row(13, "analysis", [5, 10])],       // a branch AND the fusion
  two_fanouts: [row(1, "analysis"), ...[2, 3, 4].map((i) => fan(i, "fanout_2", [1])),
                row(5, "fusion", [2, 3, 4]), ...[6, 7, 8].map((i) => fan(i, "fanout_6", [5])),
                row(9, "fusion", [6, 7, 8]), row(10, "planning", [5, 9])],
  fanout_mixed_sources: [row(1, "analysis"), row(2, "analysis"),
                         fan(3, "fanout_3", [1]), fan(4, "fanout_3", [2]), fan(5, "fanout_3", []),
                         row(6, "fusion", [3, 4, 5])],
  wide_standalone: Array.from({ length: 14 }, (_, i) => row(i + 1, "analysis")),
  wide_with_sources: [...Array.from({ length: 13 }, (_, i) => row(i + 1, "analysis")),
                      row(14, "planning", [1]), row(15, "analysis", [7, 13]), row(16, "fusion", [14, 15])],
  out_of_order_ledger: [row(3, "analysis", [1]), row(1, "analysis"), row(2, "planning", [3, 1])],
  bad_sources: [row(1, "analysis", [99]),         // unknown source
                row(2, "analysis", [2]),          // self
                row(3, "analysis", [1, 1, 2]),    // duplicate
                row(4, "analysis", [3, 99, 4])],
  cycle: [row(1, "analysis"), row(2, "analysis", [1, 3]), row(3, "analysis", [2])],
  statuses: [row(1, "analysis", [], { status: "error" }), row(2, "analysis", [1], { status: "running" }),
             row(3, "planning", [1], { status: "interrupted" }), row(4, "analysis", [2, 3], { status: "cancelled" })],
  big45: (() => {
    const rs = [row(1, "analysis")];
    for (let i = 2; i <= 31; i++) rs.push(fan(i, "fanout_2", [1]));
    rs.push(row(32, "fusion", Array.from({ length: 30 }, (_, k) => k + 2)));
    rs.push(row(33, "planning", [32]));
    for (let i = 34; i <= 40; i++) rs.push(row(i, "analysis"));
    rs.push(row(41, "analysis"), row(42, "analysis", [41]), row(43, "simulation", [42]),
            row(44, "planning", [43]), row(45, "analysis", [44, 32], { status: "running" }));
    return rs;
  })(),
  deep_chain_40: Array.from({ length: 40 }, (_, i) => row(i + 1, "analysis", i ? [i] : [])),
  everyone_feeds_last: [...Array.from({ length: 20 }, (_, i) => row(i + 1, "analysis")),
                        row(21, "fusion", Array.from({ length: 20 }, (_, i) => i + 1))],
};

// ── independent geometry helpers ─────────────────────────────────────
type Pt = { x: number; y: number };
/** Re-parse an SVG path of M / L / C commands into a sampler — independent
 * of the module's own curve code. */
function parsePath(d: string): { start: Pt; end: Pt; at: (t: number) => Pt } {
  const tokens = d.match(/[MLC]|-?\d+(\.\d+)?/g)!;
  const segs: ((t: number) => Pt)[] = [];
  let cur: Pt = { x: 0, y: 0 };
  let start: Pt | null = null;
  for (let i = 0; i < tokens.length;) {
    const cmd = tokens[i++];
    const n = (): number => Number(tokens[i++]);
    if (cmd === "M") { cur = { x: n(), y: n() }; start ??= cur; }
    else if (cmd === "L") {
      const a = cur, b = { x: n(), y: n() };
      segs.push((t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }));
      cur = b;
    } else if (cmd === "C") {
      const a = cur, c1 = { x: n(), y: n() }, c2 = { x: n(), y: n() }, b = { x: n(), y: n() };
      segs.push((t) => {
        const u = 1 - t;
        return { x: u ** 3 * a.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t ** 3 * b.x,
                 y: u ** 3 * a.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t ** 3 * b.y };
      });
      cur = b;
    } else throw new Error(`unexpected path token ${cmd} in ${d}`);
  }
  return {
    start: start!, end: cur,
    at: (t) => {
      const k = Math.min(segs.length - 1, Math.floor(t * segs.length));
      return segs[k](t * segs.length - k);
    },
  };
}
const inRect = (p: Pt, r: { x: number; y: number }, m = 0) =>
  p.x >= r.x - m && p.x <= r.x + NODE_W + m && p.y >= r.y - m && p.y <= r.y + NODE_H + m;
const near = (a: Pt, b: Pt) => Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6;

// ── checks ───────────────────────────────────────────────────────────
let failures = 0;
function check(scn: string, cond: boolean, msg: string) {
  if (!cond) { failures++; console.log(`  FAIL [${scn}] ${msg}`); }
}

function expectedContextEdges(rows: DelegationRow[], g: Graph): Set<string> {
  const out = new Set<string>();
  for (const r of rows) {
    const to = g.nodeOfIndex.get(r.index)!;
    for (const src of r.context_from) {
      const from = g.nodeOfIndex.get(src);
      if (from && from !== to) out.add(`${from}->${to}`);
    }
  }
  return out;
}

function inCycle(g: Graph, a: string, b: string): boolean {
  // b reachable from a AND a reachable from b through `sources`
  const reach = (from: string, to: string) => {
    const seen = new Set<string>(); const stack = [from];
    while (stack.length) {
      const id = stack.pop()!;
      if (id === to) return true;
      if (seen.has(id)) continue; seen.add(id);
      const n = g.nodes.find((x) => x.id === id);
      for (const s of n?.sources ?? []) stack.push(s);
    }
    return false;
  };
  return reach(a, b) && reach(b, a);
}

for (const [name, rows] of Object.entries(scenarios)) {
  const g = computeGraph(rows);
  const boxOf = new Map(g.placed.map((p) => [p.node.id, p]));
  const rootBox = { x: g.root.x, y: g.root.y };

  // 1. context edges == ledger relations
  const drawn = new Set(g.edges.filter((e) => e.kind === "context").map((e) => `${e.from}->${e.to}`));
  const expected = expectedContextEdges(rows, g);
  for (const e of expected) check(name, drawn.has(e), `missing context edge ${e}`);
  for (const e of drawn) check(name, expected.has(e), `spurious context edge ${e}`);
  check(name, g.edges.filter((e) => e.kind === "context").length === drawn.size, "duplicate context edge");

  // 2. one dispatch edge per node
  const disp = g.edges.filter((e) => e.kind === "dispatch");
  check(name, disp.length === g.placed.length, `dispatch edges ${disp.length} != nodes ${g.placed.length}`);
  check(name, new Set(disp.map((e) => e.to)).size === disp.length, "duplicate dispatch edge");

  // every ledger row is in exactly one node
  for (const r of rows) check(name, boxOf.has(g.nodeOfIndex.get(r.index)!), `row #${r.index} not placed`);

  for (const e of g.edges) {
    const path = parsePath(e.d);
    const fromBox = e.from === "root" ? rootBox : boxOf.get(e.from)!;
    const toBox = boxOf.get(e.to)!;
    // 3. endpoints on the right boxes
    check(name, near(path.start, { x: fromBox.x + NODE_W / 2, y: fromBox.y + NODE_H }),
          `edge ${e.from}->${e.to} does not start at the bottom of ${e.from}`);
    check(name, near(path.end, { x: toBox.x + NODE_W / 2, y: toBox.y }),
          `edge ${e.from}->${e.to} does not end at the top of ${e.to}`);
    // 4. no crossing of a third box (100 samples, 2 px margin)
    const others = [...g.placed.filter((p) => p.node.id !== e.from && p.node.id !== e.to),
                    ...(e.from === "root" ? [] : [{ x: g.root.x, y: g.root.y }])];
    let crossed: string | null = null;
    for (let i = 1; i < 300 && !crossed; i++) {
      const pt = path.at(i / 300);
      for (const o of others) if (inRect(pt, o, 2)) { crossed = "node" in o ? (o as any).node.id : "root"; break; }
    }
    check(name, crossed === null, `edge ${e.from}->${e.to} crosses ${crossed}`);
    // 5. top-to-bottom flow
    if (e.kind === "context" && !inCycle(g, e.from, e.to))
      check(name, toBox.y > fromBox.y + NODE_H, `edge ${e.from}->${e.to} does not flow downward`);
  }

  // 6. no overlapping boxes; inside the canvas
  const all = [...g.placed.map((p) => ({ id: p.node.id, x: p.x, y: p.y })), { id: "root", ...rootBox }];
  for (let i = 0; i < all.length; i++) {
    check(name, all[i].x >= 0 && all[i].y >= 0 && all[i].x + NODE_W <= g.width && all[i].y + NODE_H <= g.height,
          `box ${all[i].id} outside canvas`);
    for (let j = i + 1; j < all.length; j++) {
      const a = all[i], b = all[j];
      const overlap = a.x < b.x + NODE_W && b.x < a.x + NODE_W && a.y < b.y + NODE_H && b.y < a.y + NODE_H;
      check(name, !overlap, `boxes ${a.id} and ${b.id} overlap`);
    }
  }
  console.log(`${failures ? "  " : "ok  "}${name}: ${g.placed.length} nodes, ${drawn.size} context edges, ${disp.length} dispatch edges, ${g.width}×${g.height}`);
}
console.log(failures ? `\n${failures} FAILURE(S)` : "\nall scenarios pass");
process.exit(failures ? 1 : 0);
