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
  text: string; // ANSI- and mark-stripped, indentation removed
  specialist: boolean; // printed by a meta-delegated specialist
  verbose: boolean; // hidden unless verbose output is shown
}

const VISIBLE_KINDS = new Set<LineKind>([
  "tool_call", "thought", "answer_header", "answer_body", "handoff",
  "fanout", "warning", "checkpoint", "files", "memory",
]);
const RULE_RE = /^[-=_*—─═]+$/;
const CAND_RE = /^\[cand[_-]?0*(\d+)\]\s*(.*)$/;
// The atom emoji may or may not carry its variation selector (U+FE0F).
const HANDOFF_RE = /^(?:\u{1F9EA}|\u{1F4CB}|⚛️?)\s+Delegating to|^\u{1F9EC} Fusing delegations/u;

/** Stateful: a 💭 line opens a thought whose continuation lines are
 * indented five spaces; a 🤖 line opens an answer that runs until the next
 * recognisable kind. */
export class LineClassifier {
  private inThought = false;
  private thoughtSpecialist = false;
  private inAnswer = false;
  private answerSpecialist = false;

  push(raw: string): NarrationLine {
    let clean = stripAnsi(raw.replace(/\n$/, ""));
    const specialist = clean.includes(THOUGHT_MARK);
    clean = clean.replaceAll(THOUGHT_MARK, "");
    const s = clean.trim();
    const line = (kind: LineKind, text: string, spec = false, verbose = true) =>
      ({ kind, text, specialist: spec, verbose });

    if (s.startsWith("🤖")) {
      this.inThought = false;
      this.inAnswer = true;
      this.answerSpecialist = specialist;
      return line("answer_header", s, specialist, false);
    }
    if (HANDOFF_RE.test(s)) {
      this.inThought = false;
      this.inAnswer = false;
      return line("handoff", s, false, false);
    }
    if (s.startsWith("💭")) {
      this.inThought = true;
      this.inAnswer = false;
      this.thoughtSpecialist = specialist;
      return line("thought", s, specialist, false);
    }
    if (this.inThought && clean.startsWith("     ")) {
      return line("thought", s, this.thoughtSpecialist, false);
    }
    this.inThought = false;

    let kind: LineKind | null = null;
    if (s.startsWith("🔧 Calling tool:")) kind = "tool_call";
    else if (s.startsWith("⏳")) kind = "waiting";
    else if (s.startsWith("⚠")) kind = "warning";
    else if (s.startsWith("💾")) kind = "checkpoint";
    else if (s.startsWith("📂") || s.startsWith("📁")) kind = "files";
    else if (s.startsWith("🧠 Memory:")) kind = "memory";
    else if (s.startsWith("🔀")) kind = "fanout";
    else if (/^(?:🔄|✅|⚡|🙋|📊|🧠|📚|🖼|📄|🗑)/u.test(s)) kind = "bookkeeping";
    else if (s.startsWith("Human feedback enabled")) kind = "bookkeeping";
    else if (CAND_RE.test(s)) kind = "candidate";
    else if (RULE_RE.test(s)) kind = "rule";
    if (kind !== null) {
      this.inAnswer = false;
      return line(kind, s, false, !VISIBLE_KINDS.has(kind));
    }
    if (this.inAnswer) return line("answer_body", clean, this.answerSpecialist, false);
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
    // Best-of-N candidates narrate as "[cand_NN] <milestone>": strip the
    // tag, classify the milestone as usual, prefix the candidate back on.
    let cand: string | null = null;
    const cm = CAND_RE.exec(line);
    if (cm) {
      cand = cm[1];
      line = cm[2].trim();
      if (!line) continue;
    }
    const tag = (s: string) => (cand ? fill(L.candidate, { n: cand, label: s }) : s);

    if (/^Candidate\s+\d+\s+finished\s+\(\d+\/\d+\)/.test(line)) return tag(clip(line));
    let m = /escalating to (\d+) candidates/i.exec(line);
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
