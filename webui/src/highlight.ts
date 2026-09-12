/** Syntax highlighting (#605): highlight.js core with a curated language
 * set, so generated scripts, skill snippets and session files render as
 * tokens instead of plain text. Colors come from CSS tokens (`--hl-*`), so
 * the palette follows the app theme. */
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import json from "highlight.js/lib/languages/json";
import bash from "highlight.js/lib/languages/bash";
import yaml from "highlight.js/lib/languages/yaml";
import ini from "highlight.js/lib/languages/ini";
import markdown from "highlight.js/lib/languages/markdown";
import xml from "highlight.js/lib/languages/xml";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import plaintext from "highlight.js/lib/languages/plaintext";

const LANGS: Record<string, unknown> = {
  python, javascript, typescript, json, bash, yaml, ini, markdown, xml, css, diff, plaintext,
};
for (const [name, def] of Object.entries(LANGS)) {
  hljs.registerLanguage(name, def as Parameters<typeof hljs.registerLanguage>[1]);
}
hljs.registerAliases(["py", "python3"], { languageName: "python" });
hljs.registerAliases(["js", "jsx", "mjs"], { languageName: "javascript" });
hljs.registerAliases(["ts", "tsx"], { languageName: "typescript" });
hljs.registerAliases(["sh", "shell", "zsh", "console"], { languageName: "bash" });
hljs.registerAliases(["yml"], { languageName: "yaml" });
hljs.registerAliases(["toml", "cfg", "conf"], { languageName: "ini" });
hljs.registerAliases(["md"], { languageName: "markdown" });
hljs.registerAliases(["html", "htm", "svg"], { languageName: "xml" });
hljs.registerAliases(["jsonl"], { languageName: "json" });
hljs.registerAliases(["txt", "text", "log"], { languageName: "plaintext" });

/** File extension → registered language name (null: leave plain). */
const EXT_LANG: Record<string, string> = {
  py: "python", js: "javascript", ts: "typescript", tsx: "typescript", jsx: "javascript",
  json: "json", jsonl: "json", sh: "bash", bash: "bash", zsh: "bash",
  yaml: "yaml", yml: "yaml", toml: "ini", ini: "ini", cfg: "ini",
  md: "markdown", html: "xml", htm: "xml", xml: "xml", svg: "xml", css: "css", diff: "diff", patch: "diff",
};

export function languageForExtension(ext: string): string | null {
  return EXT_LANG[ext.toLowerCase()] ?? null;
}

/** Fence tags and their aliases → the registered (canonical) language. */
const ALIASES: Record<string, string> = {
  py: "python", python3: "python", js: "javascript", jsx: "javascript", mjs: "javascript",
  ts: "typescript", tsx: "typescript", sh: "bash", shell: "bash", zsh: "bash", console: "bash",
  yml: "yaml", toml: "ini", cfg: "ini", conf: "ini", md: "markdown", html: "xml", htm: "xml", svg: "xml",
  jsonl: "json", txt: "plaintext", text: "plaintext", log: "plaintext",
};

/** Normalize a fenced-block language tag (```python, ```py, ```Python) to the
 * canonical registered language, or null when unknown / not registered. */
export function resolveLanguage(tag: string | null | undefined): string | null {
  if (!tag) return null;
  const name = tag.toLowerCase().trim();
  const canonical = ALIASES[name] ?? name;
  return canonical in LANGS ? canonical : null;
}

/** Highlighting is skipped above this size — a 200 000-character log would
 * take longer to tokenize than to read. */
export const HIGHLIGHT_MAX_CHARS = 120_000;

export type Highlighted = { html: string; language: string | null };

/** Highlighted HTML for `code`; `language` null or plaintext returns the
 * escaped text unchanged (as `html`) with `language: null`. Never throws. */
export function highlightCode(code: string, language: string | null | undefined): Highlighted {
  const lang = resolveLanguage(language);
  if (!lang || lang === "plaintext" || code.length > HIGHLIGHT_MAX_CHARS) {
    return { html: escapeHtml(code), language: null };
  }
  try {
    const res = hljs.highlight(code, { language: lang, ignoreIllegals: true });
    return { html: res.value, language: lang };
  } catch {
    return { html: escapeHtml(code), language: null };
  }
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
