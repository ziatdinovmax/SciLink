/** Headless check of the Workspace panel's names (src/sessionLabel.ts).
 * Run: npm run check:labels */
import { sessionLabel, sessionLabels, modelLabels } from "../src/sessionLabel.ts";

let failures = 0;
function check(name: string, got: unknown, expected: unknown) {
  const ok = JSON.stringify(got) === JSON.stringify(expected);
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${ok ? "" : ` — got ${JSON.stringify(got)}, expected ${JSON.stringify(expected)}`}`);
  if (!ok) failures++;
}
// the labels follow the reader's locale; the checks are written for en-US
const en = (s: string) => s.replace(/ /g, " ");

console.log("sessionLabel:");
check("start time", en(sessionLabel("meta_session_20261123_101530")), "Nov 23, 10:15");
check("midnight is 00, not 24", en(sessionLabel("meta_session_20261123_000512")), "Nov 23, 00:05");
check("with seconds", en(sessionLabel("meta_session_20261123_101530", true)), "Nov 23, 10:15:30");
check("a taken name's suffix is read", en(sessionLabel("meta_session_20261123_101530_2")), "Nov 23, 10:15 (2)");
check("an hour no clock of the browser skips", en(sessionLabel("meta_session_20260308_023000")), "Mar 8, 02:30");
check("digits that are no date", sessionLabel("meta_session_20261340_101530"), "20261340_101530");
check("no time in the id", sessionLabel("analysis_session_my_run"), "my_run");
check("unattributed", sessionLabel("unattributed"), "not in a session");

console.log("distinct labels:");
const ids = ["meta_session_20261123_101530", "meta_session_20261123_101548", "meta_session_20261123_101530_2",
             "analysis_session_20261123_090000", "meta_session_20261123_090000", "meta_session_20261122_201500"];
const names = sessionLabels(ids);
check("no two rows share a name", new Set(names.values()).size, ids.length);
check("same minute: seconds added", en(names.get(ids[0])!), "Nov 23, 10:15:30");
check("same minute: seconds added (other)", en(names.get(ids[1])!), "Nov 23, 10:15:48");
check("suffix already tells it apart", en(names.get(ids[2])!), "Nov 23, 10:15 (2)");
check("same second, two modes: the ids", [names.get(ids[3]), names.get(ids[4])], [ids[3], ids[4]]);
check("alone in its minute: short", en(names.get(ids[5])!), "Nov 22, 20:15");
const models = modelLabels(["anthropic/claude-opus-5-5", "bedrock/claude-opus-5-5", "openai/gpt-6"]);
check("one model on two routes keeps its routes", [models.get("anthropic/claude-opus-5-5"), models.get("bedrock/claude-opus-5-5")],
      ["anthropic/claude-opus-5-5", "bedrock/claude-opus-5-5"]);
check("a model alone drops its prefix", models.get("openai/gpt-6"), "gpt-6");

if (failures) { console.error(`${failures} failed`); process.exit(1); }
console.log("all passed");
