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

export function MarkdownBody({
  text,
  escapeTilde = false,
  transformImageUri,
}: {
  text: string;
  escapeTilde?: boolean;
  transformImageUri?: (src: string) => string;
}) {
  const content = escapeTilde ? escapeTildes(text) : text;
  return (
    <div className="md-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={
          transformImageUri
            ? {
                img: ({ src, alt }) => (
                  <img src={src ? transformImageUri(src) : undefined} alt={alt ?? ""} />
                ),
              }
            : undefined
        }
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
