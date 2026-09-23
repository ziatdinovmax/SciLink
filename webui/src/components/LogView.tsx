import { useEffect, useRef, type ReactNode } from "react";
import { LineClassifier, stripAnsi, THOUGHT_MARK } from "../narration";

/** Colorized narration pane — the web twin of the terminal shell's
 * renderer, both reading the shared line classifier (narration.ts):
 * 💭 reasoning dim+italic (cyan for the meta, amber for a meta-delegated
 * specialist tagged with invisible U+2063), 🤖 answer headers bold,
 * 🔧 tool calls muted, and meta→specialist handoffs as gold banded rows. */

export function colorizeLog(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const classifier = new LineClassifier();
  const lines = stripAnsi(text).split("\n");
  lines.forEach((line, i) => {
    const key = `l${i}`;
    const ln = classifier.push(line);
    const clean = line.replaceAll(THOUGHT_MARK, "");
    let cls: string | null = null;
    let shown = clean;
    switch (ln.kind) {
      case "answer_header":
        cls = ln.specialist ? "answer-spec" : "answer-meta";
        break;
      case "handoff":
        cls = "handoff";
        shown = ln.text;
        break;
      case "thought":
        cls = ln.specialist ? "thought-spec" : "thought-meta";
        break;
      case "tool_call":
        cls = "tool-call";
        break;
    }
    out.push(
      cls ? (
        <span key={key} className={cls}>
          {shown}
        </span>
      ) : (
        clean
      ),
    );
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
