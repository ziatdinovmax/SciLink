"""Deterministic geometry-consistency check for ORCA decks (issue #400).

The deck's inline coordinate block (``* xyz <charge> <mult> ... *``) is
transcribed from the source structure by the LLM; a dropped atom or a wrong
element yields a deck that runs to completion on the *wrong* system — the run
succeeds, the output parses, and the answer is simply for a different molecule.
This module parses that block and compares it against the source structure file
on atom count and per-element composition, so a silent transcription error fails
generation loudly.

No LLM, cheap, engine-specific: it lives in the ORCA skill bundle and the agent
resolves it through the registry (the agent names no parser). It is the ORCA
twin of ``molecular_qc/nwchem/nwchem_geometry.py``.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec

# Coordinate-block styles ORCA accepts after the leading ``*``. Only ``xyz`` is
# a plain Cartesian list we can count atom-for-atom; the rest read coordinates
# out-of-band or in internal coordinates, so we skip rather than risk a false
# fail (mirrors the NWChem check's Z-matrix / external-coordinate handling).
_CARTESIAN_MARK = "xyz"
_NONCARTESIAN_MARKS = {"xyzfile", "int", "internal", "gzmt"}
# ORCA placeholders that are not physical atoms: ``DA`` dummy centres and ``Q``
# point charges. Ghost atoms (basis without a nucleus) are written with a
# trailing colon on the element tag (``O:``) and are excluded the same way.
_GHOST_DUMMY = {"da", "q"}
# Leading 1-2 letters of an atom-line token. Case-insensitive (ORCA element
# tags are case-insensitive), normalized with ``.capitalize()`` and checked
# against the real element symbols below.
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
    """Parse the first ORCA ``* xyz <charge> <mult> ... *`` coordinate block.

    Returns ``None`` when there is no coordinate block, otherwise
    ``{"elements": [...], "reliable": bool, "note": str}``. ``reliable`` is
    ``False`` when the block reads coordinates out-of-band (``xyzfile``) or in
    internal coordinates (``int``/``gzmt``) — the caller then skips instead of
    failing.
    """
    in_block = False
    elements: List[str] = []
    # A block carrying ghost/dummy centres is a counterpoise/BSSE fragment (one
    # monomer in the dimer basis, the rest as ghosts): its physical-atom count is
    # a SUBSET of the source complex by design, so an atom-count comparison would
    # false-fail. Track it and skip the check rather than reject a correct deck.
    _GHOST_NOTE = ("coordinate block carries ghost/dummy centres "
                   "(counterpoise/BSSE fragment); atom-count check skipped")
    ghost_seen = False
    for raw in deck.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("*"):
            rest = line[1:].strip()
            if not in_block:
                # Opening delimiter: `* <mark> <charge> <mult> [file]`.
                mark = rest.split()[0].lower() if rest else ""
                if mark == _CARTESIAN_MARK:
                    in_block = True
                    continue
                if mark in _NONCARTESIAN_MARKS:
                    return {"elements": [], "reliable": False,
                            "note": f"coordinate block uses `{mark}` (external / "
                                    "internal coordinates); atom-count check skipped"}
                # Unknown coordinate style — be conservative.
                return {"elements": [], "reliable": False,
                        "note": f"unrecognized coordinate style `* {rest[:40]}`; "
                                "check skipped"}
            # A bare `*` (or `* ...`) while in the block closes it.
            if ghost_seen:
                return {"elements": [], "reliable": False, "note": _GHOST_NOTE}
            return {"elements": elements, "reliable": True, "note": ""}
        if not in_block:
            continue
        # An atom line inside the Cartesian block: element symbol + x y z. A
        # ghost (trailing colon) or dummy/point-charge centre is not a physical
        # atom; a >=4-token line whose leading token is NOT a real element
        # cannot be counted — skip the whole check rather than silently drop the
        # line (a dropped line would fail a correct deck loudly, the very
        # failure this module promises to avoid).
        toks = line.split()
        if len(toks) < 4:                 # an atom line needs an element + x y z
            continue
        tag = toks[0]
        if tag.endswith(":"):             # ORCA ghost atom (basis, no nucleus)
            ghost_seen = True
            continue
        m = _ELEMENT_RE.match(tag)        # tag/fragment-tolerant: "O1"/"C(1)" -> "O"/"C"
        sym = m.group(1).capitalize() if m else None
        if sym is not None and sym.lower() in _GHOST_DUMMY:
            ghost_seen = True
            continue                      # dummy / point charge: not a physical atom
        if sym is None or not _is_element(sym):
            return {"elements": [], "reliable": False,
                    "note": f"unrecognized atom line in coordinate block: "
                            f"{line[:60]!r}"}
        elements.append(sym)
    # Block opened but no explicit closing `*` — use what we parsed.
    if not in_block:
        return None
    if ghost_seen:
        return {"elements": [], "reliable": False, "note": _GHOST_NOTE}
    return {"elements": elements, "reliable": True, "note": ""}


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
    """Compare a generated ORCA deck's geometry to the source structure.

    Returns a status dict:
      * ``ok``       — atom count and per-element composition match;
      * ``mismatch`` — they differ (fail loud; a wrong-system deck is
                       unrecoverable downstream);
      * ``skipped``  — no parseable coordinate block, or the structure file
                       can't be read (never block on an inapplicable case).
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
                "reason": "no parseable `* xyz ... *` coordinate block in the deck"}
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
        "Deterministically verify that a generated ORCA deck's inline "
        "coordinate block (`* xyz <charge> <mult> ... *`) matches the source "
        "structure file on atom count and per-element composition, catching a "
        "silent LLM transcription error (a dropped atom or wrong element) that "
        "would otherwise run to completion on the wrong system."
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
    import_line=("from scilink.skills.molecular_qc.orca.orca_geometry "
                 "import check_geometry_consistency"),
    agents=["simulation"],
    returns="status dict; a 'mismatch' fails generation loudly",
)
