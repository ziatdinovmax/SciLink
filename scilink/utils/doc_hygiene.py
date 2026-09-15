"""Deterministic hygiene checks for authored documents (advisory).

Run after a technical document is written; the notes are returned to the
agent (and printed) — never applied. Four checks, each of which caught a
real defect in reviewed white papers:

- image links that do not resolve beside the document (a figure link that
  was missing through two review rounds);
- agent meta-language that should never ship in a deliverable ("as
  retrieved in campaign literature", "the specialist");
- design arithmetic — every "A × B" style product in the text or a table
  is compared with stated totals ("≈ 30–40 episodes"); a "5 policies × 6–8"
  claim beside a 30–40 total slipped through as prose;
- acronym fidelity — an "Expansion (ACRONYM)" whose expansion differs from
  the one the source documents use (a platform acronym was expanded to an
  invented phrase).

No LLM. Everything here is a heuristic with a low false-positive bias:
a check that cannot decide stays silent.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

META_PHRASES = (
    "as retrieved in", "retrieved in campaign literature", "carried through",
    "the specialist", "the planning specialist", "this delegation",
    "the delegation", "the orchestrator", "sub-agent", "subagent",
    "as instructed", "per the request", "the user asked",
)


def check_image_links(text: str, doc_dir: Path) -> List[Dict[str, Any]]:
    notes = []
    for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        p = Path(target)
        if not p.is_absolute():
            p = Path(doc_dir) / target
        if not p.exists():
            notes.append({"lens": "artifact", "severity": "critical",
                          "note": f"Image link does not resolve: {target}"})
    return notes


def find_meta_language(text: str) -> List[Dict[str, Any]]:
    low = text.lower()
    hits = [ph for ph in META_PHRASES if ph in low]
    return [{"lens": "hygiene", "severity": "minor",
             "note": f"Agent meta-language in the document: '{ph}'"} for ph in hits]


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "sixty": 60, "eighty": 80, "hundred": 100}
_NUMTOK = r"(?:\d+(?:\.\d+)?|" + "|".join(_WORDS) + r")"
_NUM = rf"~?≈?\s*({_NUMTOK})"
_RANGE = rf"{_NUM}(?:\s*(?:[–\-—]|to)\s*{_NUM})?"
# "5 policies × 6–8 trajectories", "five policies × six to eight trajectories",
# "3 arms × ~10–12 coupons", "3 × 12"
_PRODUCT = re.compile(rf"\b({_NUMTOK})\s*(?:[A-Za-z\-]+\s+){{0,2}}[×x✕]\s*{_RANGE}", re.I)
# "≈ 30–40 episodes", "~30-40 episodes", "30–40 Stage III episodes"
_TOTAL = re.compile(rf"{_RANGE}\s*(?:[A-Za-z\-]+\s+){{0,3}}episodes", re.I)


def _val(s: str) -> float:
    s = s.strip().lower()
    return float(_WORDS.get(s, s)) if not s.replace(".", "", 1).isdigit() else float(s)


def _rng(a: str, b: Optional[str]):
    lo = _val(a); hi = _val(b) if b else lo
    return (min(lo, hi), max(lo, hi))


def check_design_arithmetic(text: str) -> List[Dict[str, Any]]:
    """Compare stated 'A × B' products against stated episode totals.

    Silent unless BOTH a product and a total are present. A note is raised
    only when a product range and every stated total range are disjoint —
    overlap anywhere is treated as consistent (the text may quote several
    totals for different scopes).
    """
    products = []
    for m in _PRODUCT.finditer(text):
        n = _val(m.group(1)); lo, hi = _rng(m.group(2), m.group(3))
        products.append((m.group(0).strip(), (n * lo, n * hi)))
    totals = []
    for m in _TOTAL.finditer(text):
        totals.append((m.group(0).strip(), _rng(m.group(1), m.group(2))))
    if not products or not totals:
        return []
    notes = []
    for ptxt, (plo, phi) in products:
        if all(phi < tlo or plo > thi for _, (tlo, thi) in totals):
            tot = "; ".join(t for t, _ in totals[:3])
            notes.append({"lens": "design", "severity": "critical",
                          "note": (f"Arithmetic: '{ptxt}' gives {plo:g}–{phi:g}, "
                                   f"but the stated totals are '{tot}'.")})
    return notes


# "Chemical Dynamics Observation & Control Platform (CDOC)" — capitalized words
# with lowercase connectors (of, and, for, the, in, on) allowed in between.
_ACRO = re.compile(
    r"((?:[A-Z][A-Za-z&\-]+(?:\s+(?:of|and|for|the|in|on|to|&))*\s+){2,8})"
    r"\(([A-Z][A-Z0-9]{2,7})\)")


def _expansions(text: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for m in _ACRO.finditer(text):
        exp = " ".join(m.group(1).split()).strip().lower()
        out.setdefault(m.group(2), set()).add(exp)
    return out


def check_acronym_fidelity(text: str, source_texts: Iterable[str]) -> List[Dict[str, Any]]:
    """Flag 'Expansion (ACR)' in the document when the sources expand ACR
    differently. Only acronyms that the sources actually expand are judged."""
    doc = _expansions(text)
    if not doc:
        return []
    src: Dict[str, set] = {}
    for s in source_texts or []:
        for k, v in _expansions(s or "").items():
            src.setdefault(k, set()).update(v)
    notes = []
    for acr, exps in doc.items():
        if acr not in src:
            continue
        for e in exps:
            # accept if the document expansion matches any source expansion
            # after trimming a leading article/adjective drift
            if any(e == s or e.endswith(s) or s.endswith(e) for s in src[acr]):
                continue
            notes.append({"lens": "alignment", "severity": "critical",
                          "note": (f"'{acr}' is expanded as '{e}' but the source "
                                   f"documents expand it as "
                                   f"'{sorted(src[acr])[0]}'.")})
    return notes


def check_document_hygiene(text: str, doc_dir: Path,
                           source_texts: Optional[Iterable[str]] = None
                           ) -> List[Dict[str, Any]]:
    """All checks; critical first."""
    notes = (check_image_links(text, doc_dir) + find_meta_language(text)
             + check_design_arithmetic(text)
             + check_acronym_fidelity(text, source_texts or []))
    order = {"critical": 0, "minor": 1}
    return sorted(notes, key=lambda n: order.get(n.get("severity"), 1))
