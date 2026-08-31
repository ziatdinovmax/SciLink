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
