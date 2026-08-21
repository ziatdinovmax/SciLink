"""Deterministic geometry-consistency check for NWChem decks (issue #400).

The deck's inline ``geometry ... end`` block is transcribed from the source
structure by the LLM; a dropped atom or a wrong element yields a deck that runs
to completion on the *wrong* system — the run succeeds, the output parses, and
the answer is simply for a different molecule. This module parses that block and
compares it against the source structure file on atom count and per-element
composition, so a silent transcription error fails generation loudly.

No LLM, cheap, engine-specific: it lives in the NWChem skill bundle and the
agent resolves it through the registry (the agent names no parser).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec

# Non-atom lines that can appear inside a `geometry` block.
_GEOMETRY_DIRECTIVES = {
    "units", "symmetry", "zcoord", "zmatrix", "load", "system", "adjust",
    "autosym", "noautosym", "autoz", "noautoz", "center", "nocenter", "print",
}
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _parse_geometry_elements(deck: str) -> Optional[List[str]]:
    """Element symbols from the first ``geometry ... end`` block.

    Each atom line is ``<Element> <x> <y> <z>``; directive lines (``units``,
    ``symmetry``, ``zcoord``, …) inside the block are skipped. Returns ``None``
    when the deck has no parseable geometry block.
    """
    in_block = False
    elements: List[str] = []
    for raw in deck.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not in_block:
            if line.lower().startswith("geometry"):
                in_block = True
            continue
        low = line.lower()
        if low == "end" or low.startswith("end "):
            return elements
        if not line:
            continue
        first = line.split()[0]
        if first.lower().rstrip(":") in _GEOMETRY_DIRECTIVES:
            continue
        parts = line.split()
        if len(parts) < 4:            # an atom line needs an element + x y z
            continue
        m = _ELEMENT_RE.match(first)  # strip tags/charges: "O1" -> "O"
        if m:
            elements.append(m.group(1))
    return elements if in_block else None


def _source_elements(structure_file: str) -> Optional[List[str]]:
    try:
        from ase.io import read as ase_read
        atoms = ase_read(structure_file)
        return list(atoms.get_chemical_symbols())
    except Exception:
        return None


def check_geometry_consistency(*, input_files: Dict[str, str],
                               structure_file: str,
                               **_ignored: Any) -> Dict[str, Any]:
    """Compare a generated deck's geometry to the source structure.

    Returns a status dict:
      * ``ok``       — atom count and per-element composition match;
      * ``mismatch`` — they differ (fail loud; a wrong-system deck is
                       unrecoverable downstream);
      * ``skipped``  — no parseable geometry block, or the structure file can't
                       be read (never block on an inapplicable case).
    """
    deck_elems: Optional[List[str]] = None
    for content in (input_files or {}).values():
        if isinstance(content, str):
            parsed = _parse_geometry_elements(content)
            if parsed is not None:
                deck_elems = parsed
                break
    if deck_elems is None:
        return {"status": "skipped",
                "reason": "no parseable `geometry` block in the deck"}

    src_elems = _source_elements(structure_file)
    if src_elems is None:
        return {"status": "skipped",
                "reason": f"could not read source structure {structure_file!r}"}

    deck_comp, src_comp = Counter(deck_elems), Counter(src_elems)
    if deck_comp == src_comp:
        return {"status": "ok", "n_atoms": len(deck_elems),
                "composition": dict(sorted(deck_comp.items()))}

    reasons: List[str] = []
    if len(deck_elems) != len(src_elems):
        reasons.append(f"atom count {len(deck_elems)} (deck) vs "
                       f"{len(src_elems)} (source)")
    per_el = [f"{el}: {deck_comp.get(el, 0)} vs {src_comp.get(el, 0)}"
              for el in sorted(set(deck_comp) | set(src_comp))
              if deck_comp.get(el, 0) != src_comp.get(el, 0)]
    if per_el:
        reasons.append("per-element counts (deck vs source) — " + ", ".join(per_el))
    return {
        "status": "mismatch",
        "reason": "; ".join(reasons),
        "deck_composition": dict(sorted(deck_comp.items())),
        "source_composition": dict(sorted(src_comp.items())),
    }


TOOL_SPEC = ToolSpec(
    name="check_geometry_consistency",
    description=(
        "Deterministically verify that a generated NWChem deck's inline "
        "geometry block matches the source structure file on atom count and "
        "per-element composition, catching a silent LLM transcription error "
        "(a dropped atom or wrong element) that would otherwise run to "
        "completion on the wrong system."
    ),
    parameters={
        "input_files": {"type": "object",
                        "description": "{filename: contents} of the generated deck"},
        "structure_file": {"type": "string",
                           "description": "the source structure the deck was generated from"},
    },
    required=["input_files", "structure_file"],
    signature=("check_geometry_consistency(*, input_files, structure_file) -> "
               "dict(status='ok'|'mismatch'|'skipped', ...)"),
    import_line=("from scilink.skills.molecular_qc.nwchem.nwchem_geometry "
                 "import check_geometry_consistency"),
    agents=["simulation"],
    returns="status dict; a 'mismatch' fails generation loudly",
)
