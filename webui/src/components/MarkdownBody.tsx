import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";

import { demoteHeadings, escapeTildes, markdownRemarkPlugins } from "../markdown_text";
export { demoteHeadings, escapeTildes };

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
        remarkPlugins={markdownRemarkPlugins() as never}
        rehypePlugins={[rehypeKatex]}
        components={Object.keys(components).length ? components : undefined}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
