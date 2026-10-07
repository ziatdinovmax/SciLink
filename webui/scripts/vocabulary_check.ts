/** Headless check that the narration twins agree: every case in
 * tests/fixtures/narration_activity.json (the Python side's test data) gets
 * the same activity label from the TS currentActivity, and the generated
 * vocabulary carries every mode. Run: npm run check:vocabulary */
import { readFileSync } from "node:fs";
import { LineClassifier, currentActivity } from "../src/narration.ts";
import { VOCAB } from "../src/vocabulary.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  [${cond ? "PASS" : "FAIL"}] ${name}${detail && !cond ? " — " + detail : ""}`);
  if (!cond) failures++;
}

interface Case { name: string; log: string; expected: string | null }
const fixture = new URL("../../tests/fixtures/narration_activity.json", import.meta.url);
const cases = JSON.parse(readFileSync(fixture, "utf-8")) as Case[];

console.log("currentActivity fixture:");
for (const c of cases) {
  const got = currentActivity(c.log);
  check(c.name, got === c.expected, `got ${JSON.stringify(got)}, expected ${JSON.stringify(c.expected)}`);
}

console.log("line classifier:");
const cl = new LineClassifier();
check("tool call is visible", (() => { const l = cl.push("  🔧 Calling tool: x"); return l.kind === "tool_call" && !l.verbose; })());
check("waiting is verbose", (() => { const l = cl.push("  ⏳ Waiting for response ..."); return l.kind === "waiting" && l.verbose; })());
check("thought continuation", (() => { cl.push("  💭 first"); return cl.push("     second").kind === "thought"; })());
check("specialist answer", (() => { const l = cl.push(`🤖${VOCAB.thought_mark} Specialist:`); return l.kind === "answer_header" && l.specialist; })());
check("answer body follows", cl.push("the fit converged").kind === "answer_body");
check("simulation handoff", cl.push("  ⚛️ Delegating to simulation specialist: x").kind === "handoff");
check("worker tag read off the line", (() => { const l = cl.push("[XRD 300 K]   💭 first"); return l.kind === "thought" && l.worker === "XRD 300 K"; })());
check("continuation is per worker", (() => { cl.push("[A]   💭 a thinks"); const b = cl.push("[B]      indented but B's"); return b.kind === "plain" && b.worker === "B"; })());
check("coordinator line is visible", (() => { const l = cl.push("  🐝 swarm_x: 2 item(s), up to 2 at a time"); return l.kind === "fanout" && !l.verbose; })());
check("an agent's ⏱ line is not a coordinator line", cl.push("  ⏱ Nobody answered the review in time").kind === "plain");

console.log("vocabulary:");
check("all modes present", ["meta", "analyze", "plan", "simulate"].every((k) => k in VOCAB.modes));
check("enter hint has a slot", VOCAB.enter_accepts_hint.includes("{accept}"));

console.log(failures ? `\n${failures} failure(s)` : "\nall passed");
process.exit(failures ? 1 : 0);
