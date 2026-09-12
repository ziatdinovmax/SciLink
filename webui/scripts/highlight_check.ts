/** Headless check of the syntax highlighter (#605): language mapping,
 * tokenization of the common languages, escaping, and the size cutoff.
 * Run: npm run check:highlight */
import { highlightCode, languageForExtension, resolveLanguage, HIGHLIGHT_MAX_CHARS } from "../src/highlight.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  [${cond ? "PASS" : "FAIL"}] ${name}${detail && !cond ? " — " + detail : ""}`);
  if (!cond) failures++;
}

check("py → python", languageForExtension("py") === "python");
check("yml → yaml, toml → ini, jsonl → json, md → markdown", languageForExtension("yml") === "yaml" && languageForExtension("toml") === "ini" && languageForExtension("jsonl") === "json" && languageForExtension("md") === "markdown");
check("unknown extension → null", languageForExtension("cif") === null && languageForExtension("") === null);
check("fence tags resolve case-insensitively with aliases", resolveLanguage("Python") === "python" && resolveLanguage("sh") === "bash" && resolveLanguage("py") === "python");
check("unregistered fence tag → null", resolveLanguage("fortran") === null && resolveLanguage("") === null);

const py = 'import numpy as np\n\ndef fit(x, y):\n    """Fit."""  # comment\n    return np.polyfit(x, y, 2) * 3.5\n';
const hp = highlightCode(py, "python");
check("python: keyword / string / number / comment tokens", hp.language === "python" && /hljs-keyword/.test(hp.html) && /hljs-string/.test(hp.html) && /hljs-number/.test(hp.html) && /hljs-comment/.test(hp.html), hp.html.slice(0, 200));
check("python: function name is a title token", /hljs-title[^>]*>fit</.test(hp.html));
check("json tokens", /hljs-attr/.test(highlightCode('{"a": 1, "b": [true, "x"]}', "json").html));
check("bash tokens", /hljs-built_in|hljs-keyword/.test(highlightCode("export X=1\nfor f in *.py; do echo $f; done\n", "bash").html));
check("yaml tokens", /hljs-attr/.test(highlightCode("name: eels\nsections:\n  - planning\n", "yaml").html));
check("markdown tokens", /hljs-section/.test(highlightCode("# Title\n\nsome *text*\n", "markdown").html));

const plain = highlightCode("<b>x</b> & y", null);
check("no language → escaped plain text", plain.language === null && plain.html === "&lt;b&gt;x&lt;/b&gt; &amp; y");
check("plaintext language → escaped plain text", highlightCode("a < b", "plaintext").language === null);
const big = highlightCode("x = 1\n".repeat(HIGHLIGHT_MAX_CHARS / 6 + 10), "python");
check("oversized input skips highlighting", big.language === null && !big.html.includes("<span"));
check("html in code is escaped, never injected", !highlightCode('s = "<script>alert(1)</script>"', "python").html.includes("<script>"));

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall highlight checks passed");
process.exit(failures ? 1 : 0);
