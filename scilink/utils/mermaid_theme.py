"""Shared Mermaid palette for SciLink figures.

The house style from the concept figures: Material 100/200 fills with
matching 600/700 strokes and 900 text, neutral gray links. Authors (and
the LLM) tag nodes with SEMANTIC classes — ``:::decision``,
``:::outcome`` — and this module supplies the hex values, so colors are
consistent across figures and never invented per diagram.

Classes
    stage     ordinary process / state node (blue) — also the default
    decision  gate, branch point, criterion (teal)
    outcome   endpoint, target phase, deliverable (green)
    caution   unresolvable / mixture / failure branch (pink)
    inactive  no-control region, deprecated path (gray)
    accent    the one node worth highlighting (purple)
"""

import re
from typing import Dict, Tuple

# class -> (fill, stroke, text)
PALETTE: Dict[str, Tuple[str, str, str]] = {
    "stage":    ("#BBDEFB", "#1E88E5", "#0D47A1"),
    "decision": ("#B2DFDB", "#00897B", "#004D40"),
    "outcome":  ("#C8E6C9", "#2E7D32", "#1B5E20"),
    "caution":  ("#F8BBD0", "#D81B60", "#880E4F"),
    "inactive": ("#ECEFF1", "#607D8B", "#263238"),
    "accent":   ("#E1BEE7", "#8E24AA", "#4A148C"),
}

LINK_COLOR = "#555555"

# Semantic class names offered to diagram authors / the LLM.
CLASS_HELP = (
    "stage (ordinary step/state), decision (gate or criterion), "
    "outcome (endpoint/deliverable), caution (unresolvable, mixture or "
    "failure branch), inactive (no-control region), accent (the single "
    "most important node)"
)

_STRIP = re.compile(r"^\s*(classDef|style|linkStyle)\s", re.I)


def theme_block() -> str:
    """The ``classDef`` lines, including a default for untagged nodes."""
    lines = []
    for name, (fill, stroke, color) in PALETTE.items():
        lines.append(f"  classDef {name} fill:{fill},stroke:{stroke},"
                     f"stroke-width:1.4px,color:{color}")
    fill, stroke, color = PALETTE["stage"]
    lines.append(f"  classDef default fill:{fill},stroke:{stroke},"
                 f"stroke-width:1.4px,color:{color}")
    lines.append(f"  linkStyle default stroke:{LINK_COLOR},stroke-width:1.6px")
    return "\n".join(lines)


def apply_theme(code: str) -> str:
    """Replace any styling in ``code`` with the house theme.

    Author-supplied ``classDef`` / ``style`` / ``linkStyle`` lines are
    dropped: structure is the author's, palette is the package's, so
    figures stay consistent and hex values can't drift per diagram.
    """
    body = [ln for ln in (code or "").splitlines() if not _STRIP.match(ln)]
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body + [theme_block()])
