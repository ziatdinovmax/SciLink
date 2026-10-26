"""
Read which elements a structure file contains, including LAMMPS data files.

``ase.io.read`` cannot detect the LAMMPS data format from a ``.data`` /
``.lmp`` extension, and a LAMMPS data file only says which element each atom
type is through its ``Masses`` section. These helpers read LAMMPS data with
ASE's LAMMPS reader (which takes the atom style from the file's ``Atoms #
<style>`` header) and report "unknown" rather than guess when there are no
masses: without them ASE numbers the atom types as H, He, ...
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LAMMPS_DATA_SUFFIXES = (".data", ".lammps", ".lmp")


def is_lammps_data(path: str) -> bool:
    return Path(path).suffix.lower() in LAMMPS_DATA_SUFFIXES


def lammps_type_elements(path: str) -> Optional[Dict[int, str]]:
    """``{atom type: element}`` from a LAMMPS data file's masses.

    Returns None when the file has no ``Masses`` section, since the element
    of each type is then unknowable. Raises ValueError if a declared atom
    type has no atoms (its element cannot be read from the atoms either).
    """
    from ase.io.lammpsdata import read_lammps_data
    atoms = read_lammps_data(path, sort_by_id=True)
    if "masses" not in atoms.arrays or "type" not in atoms.arrays:
        return None
    mapping: Dict[int, str] = {}
    for t, sym in zip(atoms.arrays["type"], atoms.get_chemical_symbols()):
        mapping[int(t)] = sym
    n_types = _declared_atom_types(path)
    if n_types is not None:
        missing = [t for t in range(1, n_types + 1) if t not in mapping]
        if missing:
            raise ValueError(
                f"{path}: atom types {missing} have no atoms, so their "
                f"elements cannot be determined")
    return mapping


def _declared_atom_types(path: str) -> Optional[int]:
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[1:3] == ["atom", "types"]:
                return int(parts[0])
            if line.strip().startswith(("Masses", "Atoms")):
                break
    return None


def structure_elements(path: str) -> Tuple[List[str], int]:
    """``(sorted elements, n_atoms)`` of a structure file.

    Elements are ``[]`` when they cannot be determined: an unreadable file,
    or a LAMMPS data file without masses. Callers should treat ``[]`` as
    unknown, never as "no elements".
    """
    try:
        if is_lammps_data(path):
            mapping = lammps_type_elements(path)
            from ase.io.lammpsdata import read_lammps_data
            n_atoms = len(read_lammps_data(path, sort_by_id=True))
            if mapping is None:
                logger.warning(
                    f"{path} has no Masses section: cannot tell which element "
                    f"each atom type is")
                return [], n_atoms
            return sorted(set(mapping.values())), n_atoms
        from ase.io import read
        atoms = read(path)
        return sorted(set(atoms.get_chemical_symbols())), len(atoms)
    except Exception as e:
        logger.warning(f"Could not read elements from {path}: {e}")
        return [], 0
