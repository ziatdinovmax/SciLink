/** Headless check of the markdown pipeline (#636): currency dollars stay
 * text, display math still renders, tildes never become strikethrough.
 * Renders with the same plugin set MarkdownBody uses.
 * Run: npm run check:markdown */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import { demoteHeadings, escapeTildes, markdownRemarkPlugins } from "../src/markdown_text.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  [${cond ? "PASS" : "FAIL"}] ${name}${detail && !cond ? " — " + detail : ""}`);
  if (!cond) failures++;
}
function render(text: string, escape = true): string {
  return renderToStaticMarkup(
    createElement(ReactMarkdown as never, {
      remarkPlugins: markdownRemarkPlugins(),
      rehypePlugins: [rehypeKatex],
      children: escape ? escapeTildes(text) : text,
    } as never),
  );
}

const tea = "Brine yields 1.58 g/L (~$0.79/m³ gross). It is the strongest candidate; MgO is only about $300/t and Li's price is ~$12/kg.";
const html = render(tea);
check("currency dollars are not typeset as math", !html.includes("katex"), html.slice(0, 300));
check("dollar signs survive as text", (html.match(/\$/g) ?? []).length === 3, html);
check("no strikethrough from ~ in prose", !html.includes("<del>"), html);
check("tildes render literally", html.includes("~$0.79") && html.includes("~$12/kg"), html);
check("the superscript survives as plain text", html.includes("m³"), html);

const disp = "Fit model:\n\n$$ I(x) = A e^{-x^2 / 2\\sigma^2} $$\n\nwith ~5% error.";
const dhtml = render(disp);
check("display math still renders through KaTeX", dhtml.includes("katex"), dhtml.slice(0, 200));
check("prose around display math keeps its tilde", dhtml.includes("~5%") && !dhtml.includes("<del>"), dhtml);

check("escapeTildes leaves $$ blocks untouched", escapeTildes("a ~ b $$ x \\sim y ~ z $$ c ~ d") === "a \\~ b $$ x \\sim y ~ z $$ c \\~ d");
check("escapeTildes treats single-dollar spans as prose", escapeTildes("~$1 and $2~") === "\\~$1 and $2\\~");
check("demoteHeadings shifts by two, skips fences", demoteHeadings("# T\n```\n# code\n```\n##### f") === "### T\n```\n# code\n```\n###### f");
check("gfm strikethrough still works when written as ~~text~~ unescaped", render("~~gone~~", false).includes("<del>"));

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall markdown checks passed");
process.exit(failures ? 1 : 0);
