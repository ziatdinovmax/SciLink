import { useEffect, useRef, type ReactNode } from "react";

/** Colorized narration pane — TS port of app.py:47 `_log_to_html`.
 * 💭 reasoning renders dim+italic (cyan for the meta, amber for a
 * meta-delegated specialist tagged with invisible U+2063), 🤖 answer headers
 * bold, and meta→specialist handoffs as gold banded rows. */

const THOUGHT_MARK = "⁣";
const HANDOFF_PREFIXES = [
  "🧪 Delegating to",
  "📋 Delegating to",
  "🧬 Fusing delegations",
];

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*m/g;

export function stripAnsi(text: string): string {
  return (text || "").replace(ANSI_RE, "");
}

export function colorizeLog(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let inThought = false;
  let thoughtClass = "thought-meta";
  const lines = stripAnsi(text).split("\n");
  lines.forEach((line, i) => {
    const stripped = line.trimStart();
    const key = `l${i}`;
    if (stripped.startsWith("🤖")) {
      inThought = false;
      const cls = line.includes(THOUGHT_MARK) ? "answer-spec" : "answer-meta";
      out.push(
        <span key={key} className={cls}>
          {line.replaceAll(THOUGHT_MARK, "")}
        </span>,
      );
      out.push("\n");
      return;
    }
    if (HANDOFF_PREFIXES.some((p) => stripped.startsWith(p))) {
      inThought = false;
      out.push(
        <span key={key} className="handoff">
          {stripped}
        </span>,
      );
      out.push("\n");
      return;
    }
    if (stripped.startsWith("💭")) {
      inThought = true;
      thoughtClass = line.includes(THOUGHT_MARK) ? "thought-spec" : "thought-meta";
    } else if (inThought && !line.startsWith("     ")) {
      inThought = false;
    }
    const clean = line.replaceAll(THOUGHT_MARK, "");
    if (inThought) {
      out.push(
        <span key={key} className={thoughtClass}>
          {clean}
        </span>,
      );
    } else {
      out.push(clean);
    }
    out.push("\n");
  });
  return out;
}

/** Derive a one-line "what the agent is doing now" from the narration tail —
 * the middle ground between the bare spinner and the full verbose log.
 * Backward scan: the most recent milestone line wins. Returns null when the
 * log has no recognizable signal yet (caller falls back to the static text). */
export function currentActivity(log: string): string | null {
  const clip = (s: string, n = 110) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  // "ImagePlanningController" -> "Image Planning"
  const pretty = (s: string) =>
    s.replace(/Controller$/, "").replace(/([a-z0-9])([A-Z])/g, "$1 $2").trim();

  const lines = stripAnsi(log.slice(-6000)).split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    let line = lines[i].replaceAll(THOUGHT_MARK, "").trim();
    if (!line) continue;
    // Best-of-N fan-out lines are tagged "[cand_NN] <milestone>": strip the
    // tag, classify the milestone as usual, prefix the candidate back on —
    // so parallel candidates narrate like "Candidate 2 · Verification 1/7…".
    let candTag: string | null = null;
    const cm = /^\[cand[_-]?0*(\d+)\]\s*(.*)$/.exec(line);
    if (cm) {
      candTag = `Candidate ${cm[1]}`;
      line = cm[2].trim();
      if (!line) continue;
    }
    const withTag = (s: string) => (candTag ? `${candTag} · ${s}` : s);
    // Fan-out progress milestones.
    let bm = /^Candidate\s+\d+\s+finished\s+\(\d+\/\d+\).*$/.exec(line);
    if (bm) return withTag(clip(line));
    bm = /escalating to (\d+) candidates/i.exec(line);
    if (bm) return withTag(`Escalating to ${bm[1]} parallel candidates…`);
    if (line.startsWith("🤖")) return withTag("Writing response…");
    if (line.startsWith("💭")) return withTag(clip(line));
    if (HANDOFF_PREFIXES.some((p) => line.startsWith(p))) return withTag(clip(line));
    // The step index is an internal pipeline position — show only the
    // stage name.
    let m = /STEP\s+\d+:\s*(\S.*)$/.exec(line);
    if (m) return withTag(pretty(m[1]));
    m = /^-{2,}\s*(.+?)\s*-{2,}$/.exec(line);
    if (m) return withTag(clip(m[1]));
    m = /^(?:[^\w\s]\s*)?Analyzing:\s*(.+)$/.exec(line);
    if (m) return withTag(clip(`Analyzing ${m[1]}`));
    m = /^(?:[^\w\s]\s*)?Delegating to\s+(.+)$/.exec(line);
    if (m) return withTag(clip(`Delegating to ${m[1]}`));
    // ── inside the analysis agents (shared codegen/QC loop narration) ──
    m = /\(Attempt\s+(\d+)\)\s+Asking LLM to write code/.exec(line);
    if (m) return withTag(`Writing analysis code (attempt ${m[1]})…`);
    if (/Asking LLM to write code/.test(line)) return withTag("Writing analysis code…");
    if (/Executing generated code/.test(line)) return withTag("Executing analysis code…");
    m = /Executing (?:Python )?script\s*\(attempt\s+(\d+)\)/.exec(line);
    if (m) return withTag(`Executing analysis script (attempt ${m[1]})…`);
    if (/Executing (?:Python )?script/.test(line)) return withTag("Executing analysis script…");
    m = /Execution attempt\s+(\d+)/.exec(line);
    if (m) return withTag(`Executing analysis code (attempt ${m[1]})…`);
    m = /Performing Visual QC on\s+(.+?)\.*$/.exec(line);
    if (m) return withTag(clip(`Visual QC · ${m[1]}`));
    m = /Combined review on\s+(.+?)\.*$/.exec(line);
    if (m) return withTag(clip(`Reviewing ${m[1]}`));
    m = /^Verification\s+(\d+)\/(\d+)(?:\s*\(annealing level\s+(\d+)\))?/.exec(line);
    if (m)
      return withTag(m[3] && m[3] !== "0"
        ? `Verification ${m[1]}/${m[2]} · annealing level ${m[3]}…`
        : `Verification ${m[1]}/${m[2]}…`);
    m = /Attempting script correction \(attempt\s+(\d+)\)/.exec(line);
    if (m) return withTag(`Correcting the script (attempt ${m[1]})…`);
    if (/Applying user feedback to existing script/.test(line))
      return withTag("Applying your feedback to the script…");
    if (/^Best-of-\d+/.test(line)) return withTag(clip(line));
    // Outcome/warning milestones read well as transient status too.
    if (line.startsWith("✅") || line.startsWith("⚠️")) return withTag(clip(line));
    // Generic fallbacks — catch the many phrasing variants of action lines
    // without a bespoke pattern per print statement. `bare` drops a leading
    // emoji/symbol ("🧠 LLM Step: …", "📄 Generating …").
    const bare = line.replace(/^[^\p{L}\p{N}]+\s*/u, "");
    m = /^-{2,}\s*(.+?)\s*-{2,}$/.exec(bare);
    if (m) return withTag(clip(m[1]));
    m = /^Attempt\s+(\d+(?:\/\d+)?):\s*(.+)$/.exec(bare);
    if (m) return withTag(clip(`Attempt ${m[1]} · ${m[2]}`));
    m = /^(?:LLM Step:\s*)?((?:Executing|Generating|Running|Loading|Refitting|Preparing|Searching|Creating|Initializing|Hiring|Connecting)\b.+)$/.exec(bare);
    if (m) return withTag(clip(m[1]));
  }
  return null;
}

export function LogView({ text, maxLines = 200 }: { text: string; maxLines?: number }) {
  const ref = useRef<HTMLPreElement>(null);
  const tail = text.split("\n").slice(-maxLines).join("\n");
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [tail]);
  return (
    <pre className="log-pane" ref={ref}>
      {colorizeLog(tail)}
    </pre>
  );
}
