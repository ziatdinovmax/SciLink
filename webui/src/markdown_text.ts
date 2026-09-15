/** Markdown text preprocessing and the remark plugin set shared by every
 * MarkdownBody — kept free of JSX so a headless check can exercise the
 * exact pipeline the UI renders with (`npm run check:markdown`). */
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/** Only `$$…$$` is math. LLM-authored reports use `$` for currency
 * ("~$0.79/m³", "$300/t"), and remark-math's default single-dollar inline
 * math typeset everything between two prices as KaTeX — serif italics,
 * collapsed spaces, primes for apostrophes — and defeated the tilde escape
 * below, so "~" became strikethrough (#636). Nothing in the app writes
 * single-dollar inline LaTeX on purpose; display math keeps working. */
export const MATH_OPTIONS = { singleDollarTextMath: false } as const;

export function markdownRemarkPlugins(): unknown[] {
  return [remarkGfm, [remarkMath, MATH_OPTIONS]];
}

/** Escape tildes outside `$$…$$` so they don't render as strikethrough —
 * port of app.py `_escape_tildes`. Single `$` is text (see MATH_OPTIONS),
 * so a currency span is escaped like any other prose. */
export function escapeTildes(text: string): string {
  const parts = text.split(/(\$\$[\s\S]*?\$\$)/);
  return parts
    .map((part, i) => (i % 2 === 0 ? part.replaceAll("~", "\\~") : part))
    .join("");
}

/** Shift headings down two levels (H1→H3, capped at H6) for in-chat
 * previews — port of app.py `_demote_md_headings`. */
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
