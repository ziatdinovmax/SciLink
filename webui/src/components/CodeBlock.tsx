import { useMemo } from "react";
import { highlightCode } from "../highlight";

/** A source / code block: tokenized when the language is known, plain
 * otherwise. `className` lets callers keep their existing pane styling
 * (`text-preview` in the file explorer). */
export function CodeBlock({
  code,
  language,
  className = "text-preview",
}: {
  code: string;
  language: string | null | undefined;
  className?: string;
}) {
  const hl = useMemo(() => highlightCode(code, language), [code, language]);
  if (!hl.language) return <pre className={className}>{code}</pre>;
  return (
    <pre className={`${className} hljs`} data-language={hl.language}>
      <code className={`hljs language-${hl.language}`} dangerouslySetInnerHTML={{ __html: hl.html }} />
    </pre>
  );
}

/** The `<code>` inside a markdown fenced block (react-markdown renders the
 * surrounding `<pre>` itself). */
export function HighlightedCode({ code, language }: { code: string; language: string | null }) {
  const hl = useMemo(() => highlightCode(code, language), [code, language]);
  if (!hl.language) return <code>{code}</code>;
  return <code className={`hljs language-${hl.language}`} dangerouslySetInnerHTML={{ __html: hl.html }} />;
}
