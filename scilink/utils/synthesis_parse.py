"""Tolerant salvage of synthesis-style structured LLM responses.

Sibling of :mod:`codegen_parse`: where that recovers a malformed JSON *script*
field, this recovers the narrative + claims of a synthesis / interpretation
response when strict JSON parsing fails. The synthesis step asks for an object
whose dominant field is a long free-text narrative (``detailed_analysis``);
some providers return it with unescaped quotes or newlines, so strict parsing
fails and the whole payload would otherwise be discarded — leaving a
``success`` result with empty findings (and, under the meta agent, a
re-delegation loop). See ``BaseAnalysisAgent._salvage_synthesis_fields`` for the
wiring.
"""

import json
import re
from typing import Optional

from .codegen_parse import _unescape

_MIN_NARRATIVE = 40
_NEXT_KEY = re.compile(r'"\s*,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:')  # value -> next top-level key
_FINAL_BRACE = re.compile(r'"\s*\}\s*$')                          # value -> closing brace


def _salvage_narrative(raw: str) -> Optional[str]:
    """Recover the ``detailed_analysis`` string up to the next top-level key or
    the closing brace, tolerating unescaped inner quotes (the dominant mode)."""
    m = re.search(r'"detailed_analysis"\s*:\s*"', raw)
    if not m:
        return None
    rest = raw[m.end():]
    cut = None
    for rx in (_NEXT_KEY, _FINAL_BRACE):
        mm = rx.search(rest)
        if mm:
            cut = mm.start() if cut is None else min(cut, mm.start())
    body = rest[:cut] if cut is not None else rest
    body = body.strip().rstrip('}').strip().rstrip('"')
    narrative = _unescape(body).strip()
    return narrative if len(narrative) >= _MIN_NARRATIVE else None


def _salvage_array(raw: str, key: str) -> Optional[list]:
    """Recover a JSON array field by bracket balancing; accept only if it parses
    cleanly (secondary — failure just omits the field)."""
    am = re.search(r'"' + key + r'"\s*:\s*\[', raw)
    if not am:
        return None
    i = raw.index('[', am.start())
    depth = 0
    j = i
    for j in range(i, len(raw)):
        if raw[j] == '[':
            depth += 1
        elif raw[j] == ']':
            depth -= 1
            if depth == 0:
                break
    try:
        arr = json.loads(raw[i:j + 1])
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def salvage_synthesis_fields(raw_text: str) -> Optional[dict]:
    """Best-effort recovery of synthesis fields from a malformed-JSON response.

    Returns a dict with ``detailed_analysis`` (always, when recovery succeeds)
    plus ``scientific_claims`` / ``candidate_identifications`` when their arrays
    parse cleanly. Returns ``None`` when no usable narrative can be recovered, so
    the caller keeps the original parse error rather than accepting garbage.
    """
    if not raw_text or not raw_text.strip():
        return None
    narrative = _salvage_narrative(raw_text)
    if not narrative:
        return None
    recovered = {"detailed_analysis": narrative}
    for key in ("scientific_claims", "candidate_identifications"):
        arr = _salvage_array(raw_text, key)
        if arr is not None:
            recovered[key] = arr
    return recovered


def salvage_synthesis_from_response(response) -> Optional[dict]:
    """Salvage synthesis fields directly from a raw model response object.

    Extracts the text (``.text`` / ``.content`` / ``.choices[0].message.content``)
    then delegates to :func:`salvage_synthesis_fields`. This mirrors
    ``BaseAnalysisAgent._salvage_synthesis_fields`` for use by the standalone
    synthesis CONTROLLERS, which do not inherit that method. Returns ``None`` on
    any extraction failure or unrecoverable narrative.
    """
    try:
        if hasattr(response, "text"):
            raw = response.text
        elif hasattr(response, "content"):
            raw = response.content
        elif getattr(response, "choices", None):
            raw = response.choices[0].message.content
        else:
            raw = str(response)
    except Exception:  # noqa: BLE001 - any odd response shape -> no salvage
        return None
    return salvage_synthesis_fields(raw or "")
