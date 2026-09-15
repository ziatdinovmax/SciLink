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

_ARROW = re.compile(r"-{2,}>|={2,}>|-\.-*->|-{3,}|={3,}")
# The label alternative tries a WHOLE quoted string first. Without that,
# science inside a label — "Mix KMnO4 (aq)", "[H2O2] = 0.1 M" — looks
# like a node shape, and the class tag gets spliced into the middle of
# the sentence (seen live: "Mix KMnO4 (aq):::stage with 30% H2O2").
_NODE = re.compile(
    r"(?P<id>\b[A-Za-z_]\w*)"
    r"(?P<open>\{\{|\[\[|\(\(|\[|\{|\()"
    r'(?P<label>"[^"]*"|[^"\]\}\)]*)'
    r"(?P<close>\}\}|\]\]|\)\)|\]|\}|\))"
    r"(?P<cls>:::\w+)?")


def _edge_sources(code: str) -> set:
    """Node ids that have at least one outgoing edge."""
    srcs = set()
    for line in (code or "").splitlines():
        # Strip quoted labels first, then whole bracketed spans: an
        # unquoted label's words would otherwise read as node ids and the
        # real source would be lost ("A[Unquoted label] --> B" made
        # "label" the source, so A looked terminal).
        bare = re.sub(r'"[^"]*"', "", line)
        bare = re.sub(r"\|[^|]*\|", " ", bare)
        bare = re.sub(r":::\w+", " ", bare)
        bare = re.sub(r"\[[^\]]*\]|\{[^}]*\}|\([^)]*\)", " ", bare)
        bare = re.sub(r"[\[\]{}()]", " ", bare)
        segs = _ARROW.split(bare)
        for seg in segs[:-1]:
            ids = re.findall(r"\b[A-Za-z_]\w*\b", seg)
            if ids:
                srcs.add(ids[-1])
    return srcs


def enforce_semantics(code: str) -> str:
    """Make the palette mean something: reconcile each node's class with
    what the graph says the node IS.

    Color that merely decorates is noise, so the invariants that are
    checkable from structure are enforced rather than requested:

    * a diamond is a decision, and a decision is a diamond;
    * a node with no outgoing edge is a terminal — an ``outcome``, unless
      it is already flagged ``caution``/``inactive`` (a dead end is a
      terminal too, just not a good one);
    * a node WITH outgoing edges is not an ``outcome`` (an outcome that
      continues is a contradiction) — it becomes a ``stage``;
    * ``accent`` means "the single most important node", so only the
      first survives.

    Anything the structure cannot adjudicate (stage vs caution mid-graph)
    is left to the author.
    """
    if not code:
        return code
    sources = _edge_sources(code)
    seen_accent = [False]

    def fix(m: "re.Match") -> str:
        nid, cls = m.group("id"), (m.group("cls") or "")[3:]
        is_diamond = m.group("open") in ("{", "{{")
        terminal = nid not in sources
        if is_diamond:
            cls = "decision"
        elif cls == "decision":
            cls = "stage"                      # not a diamond: not a gate
        if terminal and cls not in ("caution", "inactive"):
            cls = "outcome"
        elif not terminal and cls == "outcome":
            cls = "stage"
        if cls == "accent":
            if seen_accent[0]:
                cls = "stage"
            seen_accent[0] = True
        return (f'{nid}{m.group("open")}{m.group("label")}'
                f'{m.group("close")}:::{cls or "stage"}')

    return "\n".join(_NODE.sub(fix, ln) for ln in code.splitlines())


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
    return "\n".join([enforce_semantics("\n".join(body)), theme_block()])
