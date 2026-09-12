import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

/** Escape tildes outside $...$ / $$...$$ so they don't render as
 * strikethrough — port of app.py:95 `_escape_tildes`. */
export function escapeTildes(text: string): string {
  const parts = text.split(/(\$\$[\s\S]*?\$\$|\$[^$]+?\$)/);
  return parts
    .map((part, i) => (i % 2 === 0 ? part.replaceAll("~", "\\~") : part))
    .join("");
}

/** Shift headings down two levels (H1→H3, capped at H6) for in-chat
 * previews — port of app.py:338 `_demote_md_headings`. */
export function demoteHeadings(text: string): string {
  const out: string[] = [];
  let inFence = false;
  for (const line of text.split("\n")) {
    let l = line;
    if (l.trimStart().startsWith("```")) inFence = !inFence;
    if (!inFence) {
      const m = /^(#{1,6})(\s)/.exec(l);
      if (m) l = "#".repeat(Math.min(m[1].length + 2, 6)) + l.slice(m[1].length);
    }
    out.push(l);
  }
  return out.join("\n");
}

import { isFileToken } from "../filelink";
import { HighlightedCode } from "./CodeBlock";

/** react-markdown puts ```lang on the code element as `language-lang`. */
function fenceLanguage(className?: string): string | null {
  const m = /language-([\w+-]+)/.exec(className ?? "");
  return m ? m[1] : null;
}

export function MarkdownBody({
  text,
  escapeTilde = false,
  transformImageUri,
  onFileClick,
}: {
  text: string;
  escapeTilde?: boolean;
  transformImageUri?: (src: string) => string;
  /** When set, inline-code tokens that look like file paths become
   * clickable (chat messages: open the file in the explorer). */
  onFileClick?: (token: string) => void;
}) {
  const content = escapeTilde ? escapeTildes(text) : text;
  const components: Record<string, unknown> = {};
  if (transformImageUri) {
    components.img = ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src ? transformImageUri(src) : undefined} alt={alt ?? ""} />
    );
  }
  components.code = (props: {
    className?: string;
    children?: React.ReactNode;
  }) => {
    const raw = String(props.children ?? "");
    const lang = fenceLanguage(props.className);
    // A fenced block (language class, or multi-line) is highlighted (#605).
    if (lang || raw.includes("\n")) {
      return <HighlightedCode code={raw.replace(/\n$/, "")} language={lang} />;
    }
    // Inline code that looks like a file path opens it in the explorer.
    if (onFileClick && isFileToken(raw)) {
      return (
        <code
          className="file-link"
          title="Open in Files"
          onClick={() => onFileClick(raw)}
        >
          {props.children}
        </code>
      );
    }
    return <code className={props.className}>{props.children}</code>;
  };
  return (
    <div className="md-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={Object.keys(components).length ? components : undefined}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
