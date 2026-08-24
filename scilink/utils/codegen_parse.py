"""Robust extraction of generated code from LLM codegen responses (#238).

Codegen prompts ask the model to return the whole script as a single JSON-escaped
string (``{"script": "...escaped python..."}``; field name varies — ``code`` for
hyperspectral, ``script_content`` for structure). For large/complex scripts that
contract is brittle: the model mis-escapes quotes/newlines/braces, so ``json.loads``
fails, and the generic balanced-brace parser in ``base_agent`` then latches onto an
embedded ``results = {...}`` literal and returns a ``script``-less dict.

``extract_script`` is script-first, not JSON-first: it pulls the code value
directly and **compile-checks** it (``ast.parse`` — syntax only, never executes).
The compile-check is the discriminator — a complete-but-mis-escaped script
compiles (escaping recovered); a truncated one does not, so the caller can treat a
``None`` return plus a length finish_reason as truncation rather than a parse bug.
"""

import ast
import json
import re
from typing import Optional


def _compiles(script: Optional[str]) -> bool:
    """True iff *script* is non-empty and parses as valid Python (syntax only)."""
    if not script or not script.strip():
        return False
    try:
        ast.parse(script)
        return True
    except (SyntaxError, ValueError):
        return False


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _escape_newlines_in_strings(text: str) -> str:
    """Escape raw newlines/tabs inside double-quoted strings (mirrors base_agent).

    Recovers the common case where the model emitted a real newline inside the
    JSON string value instead of ``\\n``.
    """
    def repl(match):
        content = match.group(1)
        content = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return '"' + content + '"'

    return re.sub(r'"((?:[^"\\]|\\.)*?)"', repl, text, flags=re.DOTALL)


def _from_strict_json(text: str, field: str) -> Optional[str]:
    """Fast path: the response is valid (or newline-fixable) JSON with *field*."""
    cleaned = _strip_fences(text)
    for candidate in (text, cleaned, _escape_newlines_in_strings(cleaned)):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            val = obj.get(field)
            if isinstance(val, str) and _compiles(val):
                return val
    return None


def _unescape(s: str) -> str:
    """Un-escape a JSON-string body, tolerating partially-broken escaping."""
    try:
        return json.loads('"' + s + '"')
    except (json.JSONDecodeError, ValueError):
        out = s.replace("\\\\", "\x00")          # protect escaped backslashes
        out = (out.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
                  .replace('\\"', '"').replace("\\'", "'").replace("\\/", "/"))
        return out.replace("\x00", "\\")


def _salvage_json_string(text: str, field: str) -> Optional[str]:
    """Extract the *field* string value from malformed JSON, script-first.

    Locates ``"<field>": "`` anywhere in the text, then tries each plausible
    closing quote (one followed by JSON structure ``,`` / ``}`` / ``]``, longest
    candidate first) and the last quote overall, accepting the first whose
    un-escaped body compiles. Never keys off braces, so an embedded ``{...}``
    literal in the script cannot derail it.
    """
    key = re.search(r'"' + re.escape(field) + r'"\s*:\s*"', text)
    if not key:
        return None
    body = text[key.end():]
    closes = {m.start() for m in re.finditer(r'"\s*[,}\]]', body)}
    last_q = body.rfind('"')
    if last_q >= 0:
        closes.add(last_q)
    for rel in sorted(closes, reverse=True):
        candidate = _unescape(body[:rel])
        if _compiles(candidate):
            return candidate
    return None


def _from_fence(text: str) -> Optional[str]:
    """A fenced ```python (or bare ```) block whose contents compile."""
    for pattern in (r'```python\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```'):
        for match in re.findall(pattern, text, re.DOTALL):
            script = match.strip()
            if _compiles(script):
                return script
    return None


def _from_bare(text: str) -> Optional[str]:
    """The whole response as bare code, if it compiles and isn't a JSON object."""
    script = text.strip()
    if script.startswith("{") and script.endswith("}"):
        return None  # looks like a (malformed) JSON object, not a bare script
    return script if _compiles(script) else None


def extract_script(raw_text: str, field: str = "script") -> Optional[str]:
    """Extract a compilable script from a codegen response, or ``None``.

    Order: strict JSON -> salvage malformed JSON -> fenced code block -> bare code.
    Returns the first candidate that parses as valid Python; ``None`` means nothing
    compiled (likely truncation or an unrecoverable response).
    """
    if not raw_text or not raw_text.strip():
        return None
    for value in (
        _from_strict_json(raw_text, field),
        _salvage_json_string(raw_text, field),
        _from_fence(raw_text),
        _from_bare(raw_text),
    ):
        if value is not None:
            return value
    return None


def _response_text(response) -> str:
    """Best-effort text from a wrapper response, preferring the UNCLEANED output.

    ``raw_text`` is the model's original output before the wrapper's ``_clean_text``
    runs; codegen needs it because ``_clean_text``'s embedded-JSON extraction can
    slice a raw script that contains ``{`` braces (f-strings, dict literals),
    dropping everything before the first brace (#238). Falls back to ``.text`` for
    responses that predate the raw_text field or come from other sources.
    """
    raw = getattr(response, "raw_text", None)
    if raw:
        return raw
    txt = getattr(response, "text", None)
    if txt:
        return txt
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None and getattr(msg, "content", None):
            return msg.content
    return str(response) if response is not None else ""


def _is_truncated(response) -> bool:
    """True iff the response was cut off at the output-token limit (finish_reason=length)."""
    cand = getattr(response, "candidates", None)
    if cand:
        try:
            return getattr(cand[0], "finish_reason", None) == 0
        except Exception:
            return False
    return False


def parse_codegen_response(response, field: str = "script", logger=None):
    """Parse a codegen response into ``({field: script, **side_fields}, None)``
    or ``(None, error_dict)``.

    Script-first and compile-checked via :func:`extract_script`, so an embedded
    ``{...}`` literal can't derail extraction. Detects truncation
    (``finish_reason == length``) and reports it as a distinct error so the retry
    loop asks for a more compact script instead of misreading it as a parse bug.
    Drop-in replacement for the codegen call sites' ``self._parse(response)`` —
    returns a dict keyed by *field* (plus any side fields recovered from
    well-formed JSON, e.g. ``diagnosis``/``summary``).
    """
    if _is_truncated(response):
        if logger:
            logger.error("Codegen output truncated (finish_reason=length).")
        return None, {
            "error": "LLM output truncated (finish_reason=length)",
            "details": ("The generated script hit the model's output-token limit and "
                        "was cut off before completing. Regenerate a more compact "
                        "script (reuse helper functions, fewer inline literals)."),
            "truncated": True,
        }

    raw = _response_text(response)
    if not raw or not raw.strip():
        return None, {"error": "Empty response from LLM", "details": "Response text was empty"}

    script = extract_script(raw, field=field)
    if script is None:
        if logger:
            snippet = raw[:500] + "..." if len(raw) > 500 else raw
            logger.error(f"Codegen extraction failed (field='{field}'). Snippet: {snippet}")
        return None, {
            "error": f"Failed to extract a valid '{field}' from LLM response",
            "details": "No compilable script found (malformed escaping or truncated output).",
            "raw_response": raw[:2000],
        }

    result = {field: script}
    # Best-effort side fields (diagnosis/summary/analysis_approach/...) from
    # well-formed JSON.
    cleaned = _strip_fences(raw)
    parsed_ok = False
    for candidate in (raw, cleaned, _escape_newlines_in_strings(cleaned)):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key != field and key not in result:
                    result[key] = value
            parsed_ok = True
            break
    if not parsed_ok:
        # The (large, escaped) script value is usually what breaks json.loads,
        # which would silently drop the side fields. By convention the script is
        # the LAST key, so the side fields sit in a well-formed prefix — parse
        # that to recover them (e.g. trend's analysis_approach / key_metrics,
        # correction's diagnosis) instead of losing them when the script string
        # is malformed. The prefix contains no script text, so no false matches.
        m = re.search(r',?\s*"' + re.escape(field) + r'"\s*:', cleaned)
        if m:
            prefix = cleaned[:m.start()].rstrip().rstrip(",").rstrip()
            if prefix.startswith("{"):
                try:
                    obj = json.loads(prefix + "}")
                except (json.JSONDecodeError, ValueError):
                    obj = None
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key != field and key not in result:
                            result[key] = value
    return result, None
