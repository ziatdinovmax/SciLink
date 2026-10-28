"""
Read which elements a structure file contains, including LAMMPS data files.

``ase.io.read`` cannot detect the LAMMPS data format from a ``.data`` /
``.lmp`` extension, and a LAMMPS data file only says which element each atom
type is through its ``Masses`` section. ``inspect_lammps_data`` reads such a
file once with each of two readers, using each for what it gets right:

* the LAMMPS skill's ``parse_data_file`` for the Masses section: a ``# El``
  label is used when its element's mass agrees with the declared mass, else
  the nearest atomic mass; declared types with no atoms are kept; the
  free-form title line is skipped;
* ASE's ``read_lammps_data`` for the atoms: it takes the atom style from the
  file's ``Atoms # <style>`` header, and when there is none and its own guess
  fails, the style ``parse_data_file`` infers is tried.

When the element of some atom type cannot be told, the result says why
instead of guessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LAMMPS_DATA_SUFFIXES = (".data", ".lammps", ".lmp")


def is_lammps_data(path: str) -> bool:
    return Path(path).suffix.lower() in LAMMPS_DATA_SUFFIXES


def _is_element(sym: str) -> bool:
    # atomic_numbers maps the placeholder "X" to 0: a coarse-grained bead,
    # not an element.
    from ase.data import atomic_numbers
    return atomic_numbers.get(sym or "", 0) > 0


def _atoms_rows(path: str) -> List[List[str]]:
    """Split data rows of the Atoms section (comments stripped)."""
    rows, inside = [], False
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = line.split("#", 1)[0].strip()
            if not inside:
                inside = s == "Atoms"
                continue
            if not s:
                if rows:
                    break
                continue
            if not s.split()[0].isdigit():
                break
            rows.append(s.split())
    return rows


def _fallback_style(path: str, inferred: Optional[str], has_bonds: bool) -> Optional[str]:
    """The style to pass ASE when it cannot guess, or None if ambiguous.

    ASE knows 5/8 (atomic) and 7/10 (full) fields. Six (or nine) fields are
    "charge" (id type q x y z) or "molecular" (id mol type x y z), and the
    column count cannot tell them apart. Accept "charge" only when column 3
    holds a fractional charge, and "molecular" only when the file has bonds.
    """
    if inferred == "molecular" and has_bonds:
        return "molecular"
    if inferred == "charge":
        rows = _atoms_rows(path)
        if any(len(r) > 2 and not r[2].lstrip("+-").isdigit() for r in rows):
            return "charge"
    return None


def _read_atoms(path: str, info: Dict[str, Any]):
    """``(atoms, style_passed)``; style_passed is None when ASE chose it."""
    from ase.io.lammpsdata import read_lammps_data
    try:
        return read_lammps_data(path, sort_by_id=True), None
    except ValueError:
        # No "# <style>" header and ASE could not guess from the column count.
        style = _fallback_style(path, info.get("atom_style"),
                                bool(info.get("has_bonds")))
        if style is None:
            raise ValueError(
                "the atom style cannot be told from the columns (e.g. charge "
                "vs molecular); add it to the header, e.g. 'Atoms # charge'")
        return read_lammps_data(path, sort_by_id=True, atom_style=style), style


def inspect_lammps_data(path: str) -> Dict[str, Any]:
    """Everything element-related about a LAMMPS data file, read once.

    Returns a dict with:
      ``type_elements``  ``{type: element}`` for every declared type, or None
      ``present``        sorted elements of the atoms, or None if unknown
      ``n_atoms``        atom count
      ``n_types``        declared atom types
      ``atom_style``     style to pass ASE explicitly, or None (header/guess)
      ``problem``        why something is unknown, or None
    """
    from ..molecular_dynamics.lammps.lammps import parse_data_file
    info = parse_data_file(path)
    out: Dict[str, Any] = {
        "type_elements": None, "present": None,
        "n_atoms": int(info.get("atom_count") or 0),
        "n_types": int(info.get("atom_types") or 0),
        "atom_style": None, "problem": None,
    }
    problems = []

    mass_map = info.get("mass_map") or {}
    conflicts = info.get("mass_label_conflicts") or {}
    mapping = {int(t): el for t, (_m, el) in mass_map.items()}
    n_types = out["n_types"] or len(mapping)
    missing = [t for t in range(1, n_types + 1) if t not in mapping]
    bad = {t: el for t, el in mapping.items() if not _is_element(el)}
    if not mass_map:
        problems.append("it has no Masses section")
    elif missing:
        problems.append(f"atom types {missing} have no Masses entry")
    elif conflicts:
        detail = ", ".join(
            f"type {t} " + (f"'# {lab}' " if lab else "") + f"at {m:g}"
            for t, (lab, m) in sorted(conflicts.items()))
        problems.append(f"some Masses entries match no element closely "
                        f"enough to tell ({detail})")
    elif bad:
        problems.append(f"masses match no element for types {sorted(bad)}")
    else:
        out["type_elements"] = mapping

    try:
        atoms, style = _read_atoms(path, info)
        out["n_atoms"] = len(atoms)
        out["atom_style"] = style
        if out["type_elements"] is not None:
            out["present"] = sorted({mapping[int(t)] for t in atoms.arrays["type"]})
    except Exception as e:
        problems.append(f"its Atoms section could not be read ({e})")

    if problems:
        out["problem"] = f"{path}: " + "; ".join(problems)
    return out


def lammps_type_elements(path: str) -> Optional[Dict[int, str]]:
    """``{atom type: element}`` for every declared type, or None if unknown."""
    return inspect_lammps_data(path)["type_elements"]


def read_structure_elements(path: str) -> Tuple[List[str], int, Optional[str]]:
    """``(sorted elements present, n_atoms, problem)`` of a structure file.

    Elements are ``[]`` when they cannot be determined, and ``problem`` then
    says why. Callers should treat ``[]`` as unknown, never as "no elements".
    """
    try:
        if is_lammps_data(path):
            r = inspect_lammps_data(path)
            if r["present"] is None:
                return [], r["n_atoms"], r["problem"]
            return r["present"], r["n_atoms"], None
        from ase.io import read
        atoms = read(path)
        return sorted(set(atoms.get_chemical_symbols())), len(atoms), None
    except Exception as e:
        return [], 0, f"{path} could not be read ({e})"


def structure_elements(path: str) -> Tuple[List[str], int]:
    """``(sorted elements present, n_atoms)``; ``[]`` means unknown."""
    elements, n_atoms, problem = read_structure_elements(path)
    if problem:
        logger.warning(f"Could not determine elements: {problem}")
    return elements, n_atoms
