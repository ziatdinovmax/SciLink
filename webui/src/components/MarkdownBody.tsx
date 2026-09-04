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
  if (onFileClick) {
    components.code = (props: {
      className?: string;
      children?: React.ReactNode;
    }) => {
      const raw = String(props.children ?? "");
      // Inline code only (fenced blocks carry a language class / newlines).
      if (!props.className && !raw.includes("\n") && isFileToken(raw)) {
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
  }
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
