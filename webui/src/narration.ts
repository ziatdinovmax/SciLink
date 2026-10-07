/** Reading the agents' narration — the TypeScript twin of
 * scilink/ui/narration.py.
 *
 * Two readings of the captured stdout: what kind of line this is (LogView
 * colours by it) and what the agent is doing now (the spinner label). The
 * words come from the shared vocabulary; the regexes live in both twins
 * because the dialects differ, and tests/fixtures/narration_activity.json
 * pins them to the same answers (`npm run check:vocabulary`, and pytest on
 * the Python side). */

import { VOCAB } from "./vocabulary.ts";

export const THOUGHT_MARK: string = VOCAB.thought_mark;
export const HANDOFF_PREFIXES: readonly string[] = VOCAB.handoff_prefixes;
const L = VOCAB.activity_labels;

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*m/g;

export function stripAnsi(text: string): string {
  return (text || "").replace(ANSI_RE, "");
}

/** Fill a vocabulary template: fill("Attempt {n} · {label}", {n: "2", label: "x"}). */
export function fill(template: string, values: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, k: string) => values[k] ?? "");
}

// ── Line classification ──────────────────────────────────────────

export type LineKind =
  | "tool_call"
  | "waiting"
  | "thought"
  | "answer_header"
  | "answer_body"
  | "warning"
  | "checkpoint"
  | "files"
  | "memory"
  | "handoff"
  | "fanout"
  | "candidate"
  | "bookkeeping"
  | "rule"
  | "blank"
  | "plain";

export interface NarrationLine {
  kind: LineKind;
  text: string; // ANSI- and mark-stripped, indentation and worker tag removed
  specialist: boolean; // printed by a meta-delegated specialist
  verbose: boolean; // hidden unless verbose output is shown
  worker: string | null; // the swarm item whose line this is ("[label] …")
}

const VISIBLE_KINDS = new Set<LineKind>([
  "tool_call", "thought", "answer_header", "answer_body", "handoff",
  "fanout", "warning", "checkpoint", "files", "memory",
]);
const RULE_RE = /^[-=_*—─═]+$/;
const CAND_RE = /^\[cand[_-]?0*(\d+)\]\s*(.*)$/;
// A swarm item's lines carry its label, "[XRD 300 K] …", inserted at the line
// start by the item's own stream or the process worker's relay; never a
// candidate tag, which has its own rule.
const WORKER_RE = /^\[(?!cand[_-]?\d)([^[\]\n]{1,48})\]\s?([\s\S]*)$/;
// The swarm and fan-out coordinators' own lines, by their wording (an agent's
// line opening with the same emoji — "⏱ Nobody answered…", "🔁 Diagram render
// error" — is not one).
const COORDINATOR_RE = new RegExp(
  "^(?:🐝 |" +
    "⏸\\s+holding (?:branch )?'|" +
    "🧯 (?:free memory is low|'[^']*' ran out of memory)|" +
    "🔁 (?:running (?:branch )?'[^']*' again|Resuming \\d+ fan-out branch)|" +
    "⛔ (?:not started:|reaction refused|not running '[^']*' again)|" +
    "⏱️?\\s+(?:swarm item|analysis branch|resumed branch|raw-instrument branch\\(es\\)) |" +
    "⏳ (?:\\d+ swarm item\\(s\\) still running|\\d+ of \\d+ parallel analyses still|" +
    "\\d+ resumed branch\\(es\\) still|waiting for the cancelled (?:worker|branch) to end)|" +
    "✅ (?:swarm item|analysis branch|resumed branch) finished)",
  "u",
);

/** ["XRD 300 K", "💭 …"] for a swarm item's tagged line, else [null, clean]. */
export function splitWorkerTag(clean: string): [string | null, string] {
  const m = WORKER_RE.exec(clean.replace(/^\s+/, ""));
  return m ? [m[1], m[2]] : [null, clean];
}
// The atom emoji may or may not carry its variation selector (U+FE0F).
const HANDOFF_RE = /^(?:\u{1F9EA}|\u{1F4CB}|⚛️?)\s+Delegating to|^\u{1F9EC} Fusing delegations/u;

/** What a 💭 or 🤖 line opened and has not closed — per writer: the meta's
 * stream and each swarm item's are interleaved, and an indented line of item
 * B must never read as the continuation of item A's thought. */
interface Continuation {
  inThought: boolean;
  thoughtSpecialist: boolean;
  inAnswer: boolean;
  answerSpecialist: boolean;
}

/** Stateful: a 💭 line opens a thought whose continuation lines are
 * indented five spaces; a 🤖 line opens an answer that runs until the next
 * recognisable kind. The state is kept per worker (the meta's own lines and
 * each swarm item's). */
export class LineClassifier {
  private byWorker = new Map<string | null, Continuation>();

  private state(worker: string | null): Continuation {
    let st = this.byWorker.get(worker);
    if (!st) {
      st = { inThought: false, thoughtSpecialist: false, inAnswer: false, answerSpecialist: false };
      this.byWorker.set(worker, st);
    }
    return st;
  }

  push(raw: string): NarrationLine {
    let clean = stripAnsi(raw.replace(/\n$/, ""));
    const specialist = clean.includes(THOUGHT_MARK);
    clean = clean.replaceAll(THOUGHT_MARK, "");
    const [worker, rest] = splitWorkerTag(clean);
    clean = rest;
    const s = clean.trim();
    const st = this.state(worker);
    const line = (kind: LineKind, text: string, spec = false, verbose = true) =>
      ({ kind, text, specialist: spec, verbose, worker });

    if (s.startsWith("🤖")) {
      st.inThought = false;
      st.inAnswer = true;
      st.answerSpecialist = specialist;
      return line("answer_header", s, specialist, false);
    }
    if (HANDOFF_RE.test(s)) {
      st.inThought = false;
      st.inAnswer = false;
      return line("handoff", s, false, false);
    }
    if (s.startsWith("💭")) {
      st.inThought = true;
      st.inAnswer = false;
      st.thoughtSpecialist = specialist;
      return line("thought", s, specialist, false);
    }
    if (st.inThought && clean.startsWith("     ")) {
      return line("thought", s, st.thoughtSpecialist, false);
    }
    st.inThought = false;

    let kind: LineKind | null = null;
    if (s.startsWith("🔧 Calling tool:")) kind = "tool_call";
    else if (s.startsWith("🔀") || COORDINATOR_RE.test(s)) kind = "fanout"; // the coordinators' own lines
    else if (s.startsWith("⏳")) kind = "waiting";
    else if (s.startsWith("⚠")) kind = "warning";
    else if (s.startsWith("💾")) kind = "checkpoint";
    else if (s.startsWith("📂") || s.startsWith("📁")) kind = "files";
    else if (s.startsWith("🧠 Memory:")) kind = "memory";
    else if (/^(?:🔄|✅|⚡|🙋|📊|🧠|📚|🖼|📄|🗑)/u.test(s)) kind = "bookkeeping";
    else if (s.startsWith("Human feedback enabled")) kind = "bookkeeping";
    else if (CAND_RE.test(s)) kind = "candidate";
    else if (RULE_RE.test(s)) kind = "rule";
    if (kind !== null) {
      st.inAnswer = false;
      return line(kind, s, false, !VISIBLE_KINDS.has(kind));
    }
    if (st.inAnswer) return line("answer_body", clean, st.answerSpecialist, false);
    if (!s) return line("blank", "");
    return line("plain", s);
  }
}

// ── Current activity ─────────────────────────────────────────────

const clip = (s: string, n = 110) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
// "ImagePlanningController" -> "Image Planning"
const pretty = (s: string) =>
  s.replace(/Controller$/, "").replace(/([a-z0-9])([A-Z])/g, "$1 $2").trim();

const ACTION_RE =
  /^(?:(?:LLM Step|Tool):\s*)?((?:Executing|Generating|Running|Loading|Refitting|Preparing|Searching|Creating|Initializing|Hiring|Connecting|Refining|Reasoning|Processing|Saving|Verifying|Updating|Building|Synthesizing|Retrieving|Querying|Inspecting|Reviewing|Drafting|Writing|Analyzing)\b.+)$/;

/** One line saying what the agent is doing now, from the narration tail —
 * the middle ground between the bare spinner and the full verbose log.
 * Backward scan: the most recent milestone line wins. Returns null when the
 * log has no recognizable signal yet (caller shows VOCAB.default_activity). */
export function currentActivity(log: string): string | null {
  const lines = stripAnsi(log.slice(-6000)).split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    let line = lines[i].replaceAll(THOUGHT_MARK, "").trim();
    if (!line) continue;
    // Bare rules ("-" * 60, "=" * 60) frame headers in the narration; the
    // "--- title ---" pattern below would read one as a title of "-".
    if (RULE_RE.test(line)) continue;
    // A swarm item's line carries its label, "[XRD 300 K] <milestone>": strip
    // it, classify the milestone, prefix the label back on — so the activity
    // says WHOSE stage this is when several items run at once.
    const [worker, rest] = splitWorkerTag(line);
    line = rest.trim();
    if (!line) continue;
    // Best-of-N candidates narrate as "[cand_NN] <milestone>": strip the
    // tag, classify the milestone as usual, prefix the candidate back on.
    let cand: string | null = null;
    const cm = CAND_RE.exec(line);
    if (cm) {
      cand = cm[1];
      line = cm[2].trim();
      if (!line) continue;
    }
    const tag = (s: string) => {
      const c = cand ? fill(L.candidate, { n: cand, label: s }) : s;
      return worker ? fill(L.worker, { worker, label: c }) : c;
    };

    // The coordinators' own milestones (a swarm's, a fan-out's).
    let m = /^🐝 .*?(\d+) item\(s\)(?:, up to (\d+) at a time)?/u.exec(line);
    if (m) return tag(fill(L.swarm_started, { n: m[1] }));
    m = /^⏳ (\d+) swarm item\(s\) still running/u.exec(line);
    if (m) return tag(fill(L.swarm_running, { n: m[1] }));
    m = /^⏸\s+holding (?:branch )?'([^']+)'/u.exec(line);
    if (m) return tag(clip(fill(L.swarm_holding, { label: m[1] })));
    m = /^🧯 .*?cancelling (?:branch )?'([^']+)'/u.exec(line);
    if (m) return tag(clip(fill(L.swarm_guard, { label: m[1] })));
    m = /^🔁 running (?:branch )?'([^']+)' again/u.exec(line);
    if (m) return tag(clip(fill(L.swarm_rerun, { label: m[1] })));
    m = /^✅ swarm item finished: (.+?) \((\w+)\)/u.exec(line);
    if (m) return tag(clip(fill(L.swarm_item_done, { label: m[1], status: m[2] })));

    if (/^Candidate\s+\d+\s+finished\s+\(\d+\/\d+\)/.test(line)) return tag(clip(line));
    m = /escalating to (\d+) candidates/i.exec(line);
    if (m) return tag(fill(L.escalating, { n: m[1] }));
    if (line.startsWith("🤖")) return tag(L.writing_response);
    m = /^⏳\s*Waiting for (.+?) response/.exec(line);
    if (m) return tag(fill(L.waiting_for, { who: m[1] }));
    if (line.startsWith("💭")) return tag(clip(line));
    if (HANDOFF_PREFIXES.some((p) => line.startsWith(p))) return tag(clip(line));
    m = /STEP\s+\d+:\s*(\S.*)$/.exec(line);
    if (m) return tag(pretty(m[1]));
    m = /^-{2,}\s*(.+?)\s*-{2,}$/.exec(line);
    if (m) return tag(clip(m[1]));
    m = /^(?:[^\w\s]\s*)?Analyzing:\s*(.+)$/u.exec(line);
    if (m) return tag(clip(fill(L.analyzing, { target: m[1] })));
    m = /^(?:[^\w\s]\s*)?Delegating to\s+(.+)$/u.exec(line);
    if (m) return tag(clip(fill(L.delegating, { target: m[1] })));
    // ── inside the analysis agents (shared codegen/QC loop narration) ──
    m = /\(Attempt\s+(\d+)\)\s+Asking LLM to write code/.exec(line);
    if (m) return tag(fill(L.writing_code_attempt, { n: m[1] }));
    if (/Asking LLM to write code/.test(line)) return tag(L.writing_code);
    if (/Executing generated code/.test(line)) return tag(L.executing_code);
    m = /Executing (?:Python )?script\s*\(attempt\s+(\d+)\)/.exec(line);
    if (m) return tag(fill(L.executing_script_attempt, { n: m[1] }));
    if (/Executing (?:Python )?script/.test(line)) return tag(L.executing_script);
    m = /Execution attempt\s+(\d+)/.exec(line);
    if (m) return tag(fill(L.executing_code_attempt, { n: m[1] }));
    m = /Performing Visual QC on\s+(.+?)\.*$/.exec(line);
    if (m) return tag(clip(fill(L.visual_qc, { target: m[1] })));
    m = /Combined review on\s+(.+?)\.*$/.exec(line);
    if (m) return tag(clip(fill(L.reviewing, { target: m[1] })));
    m = /^Verification\s+(\d+)\/(\d+)(?:\s*\(annealing level\s+(\d+)\))?/.exec(line);
    if (m)
      return tag(
        m[3] && m[3] !== "0"
          ? fill(L.verification_annealing, { i: m[1], n: m[2], level: m[3] })
          : fill(L.verification, { i: m[1], n: m[2] }),
      );
    m = /Attempting script correction \(attempt\s+(\d+)\)/.exec(line);
    if (m) return tag(fill(L.correcting, { n: m[1] }));
    if (/Applying user feedback to existing script/.test(line)) return tag(L.applying_feedback);
    if (/^Best-of-\d+/.test(line)) return tag(clip(line));
    // Outcome/warning milestones read well as transient status too.
    // ... but not a section heading ("✅ Quality Criteria:") whose content follows.
    if ((line.startsWith("✅") || line.startsWith("⚠️")) && !line.endsWith(":")) return tag(clip(line));
    // Generic fallbacks — catch the many phrasing variants of action lines
    // without a bespoke pattern per print statement. `bare` drops a leading
    // emoji/symbol ("🧠 LLM Step: …", "📄 Generating …").
    const bare = line.replace(/^[^\p{L}\p{N}]+\s*/u, "");
    // Human-in-the-loop pauses: the narration ends on the prompt banner.
    if (/^(REQUESTING FEEDBACK|SELECT A PLAN CANDIDATE|CODE REVIEW REQUIRED)\b/.test(bare))
      return tag(L.waiting_for_input);
    if (/^FILES PRODUCED THIS TURN\b/.test(bare)) return tag(L.wrapping_up);
    m = /^-{2,}\s*(.+?)\s*-{2,}$/.exec(bare);
    if (m) return tag(clip(m[1]));
    m = /^Attempt\s+(\d+(?:\/\d+)?):\s*(.+)$/.exec(bare);
    if (m) return tag(clip(fill(L.attempt, { n: m[1], label: m[2] })));
    m = ACTION_RE.exec(bare);
    if (m) return tag(clip(m[1]));
  }
  return null;
}
