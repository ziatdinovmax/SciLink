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

# Plain non-atom directive lines that can appear inside a `geometry` block.
_GEOMETRY_DIRECTIVES = {
    "units", "autosym", "noautosym", "autoz", "noautoz", "center", "nocenter",
    "print", "system", "adjust",
}
# Sub-directives whose presence means the block is NOT a plain Cartesian list we
# can count atom-for-atom (internal coords, external/loaded coords, nested
# blocks with their own `end`). We skip the check rather than risk a false fail.
_NONCARTESIAN = {"zmatrix", "zcoord", "load", "constraints"}
# NWChem placeholders that are not physical atoms (ghost / dummy centers).
_GHOST_DUMMY = {"x", "bq"}
# Leading 1-2 letters of an atom-line token. Case-insensitive (NWChem element
# tags are case-insensitive, so `o` is a legal atom line); normalized with
# `.capitalize()` and checked against the real element symbols below.
_ELEMENT_RE = re.compile(r"^([A-Za-z]{1,2})")

_SYMBOLS_CACHE = None


def _is_element(symbol: str) -> bool:
    """Whether ``symbol`` (already capitalized) is a real chemical element."""
    global _SYMBOLS_CACHE
    if _SYMBOLS_CACHE is None:
        try:
            from ase.data import chemical_symbols
            _SYMBOLS_CACHE = frozenset(chemical_symbols)
        except Exception:
            _SYMBOLS_CACHE = frozenset()
    return symbol in _SYMBOLS_CACHE


def _parse_geometry_block(deck: str) -> Optional[Dict[str, Any]]:
    """Parse the first ``geometry ... end`` block.

    Returns ``None`` when there is no geometry block, otherwise
    ``{"elements": [...], "reliable": bool, "note": str}``. ``reliable`` is
    ``False`` when the block uses a form we cannot safely count atom-for-atom (a
    Z-matrix, external/loaded coordinates, or a non-C1 symmetry group that may
    list only the asymmetric unit) — the caller then skips instead of failing.
    """
    in_block = False
    elements: List[str] = []
    for raw in deck.splitlines():
        line = raw.split("#", 1)[0].strip()
        toks = line.split()
        if not in_block:
            if toks and toks[0].lower() == "geometry":   # whole-word match
                in_block = True
            continue
        if not toks:
            continue
        head = toks[0].lower().rstrip(":")
        if head == "end":
            return {"elements": elements, "reliable": True, "note": ""}
        if head in _NONCARTESIAN:
            return {"elements": [], "reliable": False,
                    "note": f"geometry uses `{head}` (non-Cartesian / external "
                            "coordinates); atom-count check skipped"}
        if head == "symmetry":
            grp = toks[1].lower() if len(toks) > 1 else "c1"
            if grp not in ("c1", ""):
                return {"elements": [], "reliable": False,
                        "note": f"geometry declares symmetry `{grp}`; the block "
                                "may list only the asymmetric unit — check skipped"}
            continue
        if head in _GEOMETRY_DIRECTIVES:
            continue
        # An atom line: element symbol + coordinates. Element tags are
        # case-insensitive, so normalize; a ghost/dummy centre is excluded; and
        # a >=4-token line whose leading token is NOT a real element cannot be
        # counted — skip the whole check rather than silently drop the line (a
        # dropped line would fail a correct deck loudly, the very failure this
        # module promises to avoid).
        if len(toks) < 4:                 # an atom line needs an element + x y z
            continue
        m = _ELEMENT_RE.match(toks[0])    # tag/charge-tolerant: "O1"/"o" -> "O"
        sym = m.group(1).capitalize() if m else None
        if sym is not None and sym.lower() in _GHOST_DUMMY:
            continue                      # ghost/dummy: not a physical atom
        if sym is None or not _is_element(sym):
            return {"elements": [], "reliable": False,
                    "note": f"unrecognized atom line in geometry block: "
                            f"{line[:60]!r}"}
        elements.append(sym)
    # Block opened but no explicit `end` — use what we parsed.
    return ({"elements": elements, "reliable": True, "note": ""}
            if in_block else None)


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
    parsed: Optional[Dict[str, Any]] = None
    for content in (input_files or {}).values():
        if isinstance(content, str):
            block = _parse_geometry_block(content)
            if block is not None:
                parsed = block
                break
    if parsed is None:
        return {"status": "skipped",
                "reason": "no parseable `geometry` block in the deck"}
    if not parsed["reliable"]:
        return {"status": "skipped", "reason": parsed["note"]}
    deck_elems = parsed["elements"]

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
