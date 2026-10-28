"""
Read which elements a structure file contains, including LAMMPS data files.

``ase.io.read`` cannot detect the LAMMPS data format from a ``.data`` /
``.lmp`` extension, and a LAMMPS data file only says which element each atom
type is through its ``Masses`` section. For LAMMPS data these helpers use
the LAMMPS skill's own ``parse_data_file`` for the Masses section: it reads
``# El`` mass labels first and falls back to the nearest atomic mass, keeps
declared types that have no atoms, and skips the free-form title line. The
atoms themselves are read with ASE, which honours the file's atom style.
They report "unknown" rather than guess when the masses don't say.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LAMMPS_DATA_SUFFIXES = (".data", ".lammps", ".lmp")

_SYMBOL = re.compile(r"^[A-Z][a-z]?$")


def is_lammps_data(path: str) -> bool:
    return Path(path).suffix.lower() in LAMMPS_DATA_SUFFIXES


def _parse(path: str) -> Dict:
    from ..molecular_dynamics.lammps.lammps import parse_data_file
    return parse_data_file(path)


def _is_element(sym: str) -> bool:
    from ase.data import chemical_symbols
    return bool(_SYMBOL.match(sym or "")) and sym in chemical_symbols


def lammps_type_elements(path: str) -> Optional[Dict[int, str]]:
    """``{atom type: element}`` for every declared type of a LAMMPS data file.

    Returns None when that cannot be known: no ``Masses`` section, a type
    missing from it, or a mass that matches no element.
    """
    info = _parse(path)
    mass_map = info.get("mass_map") or {}
    if not mass_map:
        return None
    mapping = {int(t): el for t, (_mass, el) in mass_map.items()}
    if not all(_is_element(el) for el in mapping.values()):
        return None
    n_types = info.get("atom_types") or 0
    if n_types and set(mapping) != set(range(1, n_types + 1)):
        return None
    return mapping


def structure_elements(path: str) -> Tuple[List[str], int]:
    """``(sorted elements present, n_atoms)`` of a structure file.

    Elements are ``[]`` when they cannot be determined: an unreadable file,
    or a LAMMPS data file whose masses don't identify every type. Callers
    should treat ``[]`` as unknown, never as "no elements".
    """
    try:
        if is_lammps_data(path):
            # Masses -> elements from parse_data_file (style-independent);
            # which types the atoms use from ASE, which takes the atom style
            # from the "Atoms # <style>" header (parse_data_file guesses the
            # style from column counts and can misread a full-style file).
            from ase.io.lammpsdata import read_lammps_data
            atoms = read_lammps_data(path, sort_by_id=True)
            mapping = lammps_type_elements(path)
            if mapping is None:
                logger.warning(
                    f"{path}: the Masses section does not identify every atom "
                    f"type's element")
                return [], len(atoms)
            present = {mapping[int(t)] for t in atoms.arrays["type"]}
            return sorted(present), len(atoms)
        from ase.io import read
        atoms = read(path)
        return sorted(set(atoms.get_chemical_symbols())), len(atoms)
    except Exception as e:
        logger.warning(f"Could not read elements from {path}: {e}")
        return [], 0
