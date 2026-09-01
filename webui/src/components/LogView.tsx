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
    const line = lines[i].replaceAll(THOUGHT_MARK, "").trim();
    if (!line) continue;
    if (line.startsWith("🤖")) return "Writing response…";
    if (line.startsWith("💭")) return clip(line);
    if (HANDOFF_PREFIXES.some((p) => line.startsWith(p))) return clip(line);
    // The step index is an internal pipeline position — show only the
    // stage name.
    let m = /STEP\s+\d+:\s*(\S.*)$/.exec(line);
    if (m) return pretty(m[1]);
    m = /^-{2,}\s*(.+?)\s*-{2,}$/.exec(line);
    if (m) return clip(m[1]);
    m = /^(?:[^\w\s]\s*)?Analyzing:\s*(.+)$/.exec(line);
    if (m) return clip(`Analyzing ${m[1]}`);
    m = /^(?:[^\w\s]\s*)?Delegating to\s+(.+)$/.exec(line);
    if (m) return clip(`Delegating to ${m[1]}`);
    // ── inside the analysis agents (shared codegen/QC loop narration) ──
    m = /\(Attempt\s+(\d+)\)\s+Asking LLM to write code/.exec(line);
    if (m) return `Writing analysis code (attempt ${m[1]})…`;
    if (/Asking LLM to write code/.test(line)) return "Writing analysis code…";
    if (/Executing generated code/.test(line)) return "Executing analysis code…";
    m = /Execution attempt\s+(\d+)/.exec(line);
    if (m) return `Executing analysis code (attempt ${m[1]})…`;
    m = /Performing Visual QC on\s+(.+?)\.*$/.exec(line);
    if (m) return clip(`Visual QC · ${m[1]}`);
    m = /Combined review on\s+(.+?)\.*$/.exec(line);
    if (m) return clip(`Reviewing ${m[1]}`);
    m = /Attempting script correction \(attempt\s+(\d+)\)/.exec(line);
    if (m) return `Correcting the script (attempt ${m[1]})…`;
    if (/Applying user feedback to existing script/.test(line))
      return "Applying your feedback to the script…";
    if (/^Best-of-\d+/.test(line)) return clip(line);
    // Outcome/warning milestones read well as transient status too.
    if (line.startsWith("✅") || line.startsWith("⚠️")) return clip(line);
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
