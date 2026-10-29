"""
Read which elements a structure file contains, including LAMMPS data files.

``ase.io.read`` cannot detect the LAMMPS data format from a ``.data`` /
``.lmp`` extension, and a LAMMPS data file only says which element each atom
type is through its ``Masses`` section. ``inspect_lammps_data`` reads such a
file once, using:

* ``lammps_types`` for the Masses section: one record per atom type with its
  element and the basis for it, or the reason it cannot be told;
* ASE's ``read_lammps_data`` for the atoms: it takes the atom style from the
  file's ``Atoms # <style>`` header, and when there is none and its own guess
  fails, ``_fallback_style``'s answer is tried.

When the element of some atom type cannot be told, the result says why
instead of guessing. The caller can then say what the file cannot:
``type_elements`` (``{type: element}``) and ``atom_style`` are applied,
checked against the file, and recorded with their source.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import lammps_types
from .lammps_types import DEFAULT_ISOTOPE_WINDOW, DEFAULT_MASS_TOLERANCE

logger = logging.getLogger(__name__)

LAMMPS_DATA_SUFFIXES = (".data", ".lammps", ".lmp")

# (path, type) pairs already warned about as isotopes: one deploy reads the
# file two or three times.
_ISOTOPE_WARNED = set()


def is_lammps_data(path: str) -> bool:
    return Path(path).suffix.lower() in LAMMPS_DATA_SUFFIXES


def _atoms_rows(lines: List[str]) -> List[List[str]]:
    """Split data rows of the Atoms section (comments stripped)."""
    rows, inside = [], False
    for line in lines:
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


def _fallback_style(rows: List[List[str]], has_bonds: bool) -> Optional[str]:
    """The style to pass ASE when it cannot guess, or None if ambiguous.

    ASE knows 5/8 (atomic) and 7/10 (full) columns. Six (or nine, with image
    flags) are "charge" (id type q x y z) or "molecular" (id mol type x y z),
    and the column count cannot tell them apart. The default rule: "molecular"
    when the file has bonds, "charge" when column 3 holds a fractional value
    (a charge, not a type number), else ambiguous. A caller who knows the
    style passes ``atom_style`` and this rule is not used.
    """
    if not rows or len(rows[0]) not in (6, 9):
        return None
    if has_bonds:
        return "molecular"
    if any(not r[2].lstrip("+-").isdigit() for r in rows):
        return "charge"
    return None


def _read_atoms(path: str, rows, has_bonds: bool, atom_style: Optional[str]):
    """``(atoms, style_passed)``; style_passed is None when ASE chose it."""
    from ase.io.lammpsdata import read_lammps_data
    if atom_style:
        return read_lammps_data(path, sort_by_id=True, atom_style=atom_style), atom_style
    try:
        return read_lammps_data(path, sort_by_id=True), None
    except ValueError:
        # No "# <style>" header and ASE could not guess from the column count.
        style = _fallback_style(rows, has_bonds)
        if style is None:
            raise ValueError(
                "the atom style cannot be told from the columns (charge vs "
                "molecular); pass atom_style, or add it to the header "
                "('Atoms # charge')")
        return read_lammps_data(path, sort_by_id=True, atom_style=style), style


def inspect_lammps_data(
    path: str,
    type_elements: Optional[Dict[Any, str]] = None,
    atom_style: Optional[str] = None,
    mass_tolerance: float = DEFAULT_MASS_TOLERANCE,
    isotope_window: float = DEFAULT_ISOTOPE_WINDOW,
) -> Dict[str, Any]:
    """Everything element-related about a LAMMPS data file, read once.

    Args:
        type_elements: ``{type: element}`` the caller asserts, e.g. for a
            coarse-grained bead or a united-atom type the masses cannot tell.
            Applied on top of the file's readings (see
            ``lammps_types.apply_type_elements``).
        atom_style: the file's atom style, when it has no ``Atoms # <style>``
            header and the columns are ambiguous.
        mass_tolerance, isotope_window: see ``lammps_types``.

    Returns a dict with:
      ``readings``       ``{type: record}``: element, basis, source, conflict
      ``type_elements``  ``{type: element}`` for every declared type, or None
      ``present``        sorted elements of the atoms, or None if unknown
      ``n_atoms``        atom count
      ``n_types``        declared atom types
      ``atom_style``     style to pass ASE explicitly, or None (header/guess)
      ``atoms_error``    why the Atoms section could not be read, or None
      ``problem``        why something is unknown, or None
    Raises ValueError for an override that names no element or no type.
    """
    with open(path, errors="ignore") as fh:
        lines = fh.readlines()
    counts = lammps_types.header_counts(lines)
    readings = lammps_types.read_types(lines, mass_tolerance, isotope_window)
    n_types = counts["atom types"] or len(readings)
    readings = lammps_types.apply_type_elements(readings, type_elements, n_types)
    out: Dict[str, Any] = {
        "readings": readings, "type_elements": None, "present": None,
        "n_atoms": counts["atoms"], "n_types": n_types,
        "atom_style": None, "problem": None, "atoms_error": None,
    }
    for r in readings.values():
        if r["isotope"] and (path, r["type"]) not in _ISOTOPE_WARNED:
            _ISOTOPE_WARNED.add((path, r["type"]))
            logger.warning(f"{path}: type {r['type']} '# {r['label']}' at "
                           f"{r['mass']:g} read as an isotope of {r['element']}")
        if r["source"] == "override" and r["file_reading"]["element"] not in (
                None, r["element"]):
            logger.warning(f"{path}: type {r['type']} set to {r['element']} by "
                           f"type_elements; the file reads it as "
                           f"{r['file_reading']['element']}")

    problems = []
    missing = [t for t in range(1, n_types + 1) if t not in readings]
    conflicts = {t: r["conflict"] for t, r in sorted(readings.items()) if r["conflict"]}
    if not readings:
        problems.append("it has no Masses section")
    elif missing:
        problems.append(f"atom types {missing} have no Masses entry")
    if conflicts:
        problems.append("; ".join(f"type {t}: {c}" for t, c in conflicts.items()))
    if not problems:
        out["type_elements"] = {t: r["element"] for t, r in sorted(readings.items())}
    mapping = out["type_elements"]

    rows = _atoms_rows(lines)
    try:
        atoms, style = _read_atoms(path, rows, counts["bonds"] > 0, atom_style)
        out["n_atoms"] = len(atoms)
        out["atom_style"] = style
        if mapping is not None:
            out["present"] = sorted({mapping[int(t)] for t in atoms.arrays["type"]})
    except KeyError as e:
        # ASE meets an atom whose type has no Masses entry.
        out["atoms_error"] = f"atom type {e.args[0]} has no Masses entry"
        problems.append(f"its Atoms section could not be read ({out['atoms_error']})")
    except Exception as e:
        out["atoms_error"] = str(e)
        problems.append(f"its Atoms section could not be read ({e})")

    if problems:
        out["problem"] = f"{path}: " + "; ".join(problems)
        if mapping is None:
            out["problem"] += (". Label the masses ('1 22.99 # Na') or pass "
                               "type_elements={type: element}")
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
