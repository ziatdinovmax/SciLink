"""A searchable table over a folder of crystal-structure files (#804).

One row per file: formula, elements, cell, density, declared (and, with
pymatgen, computed) space group, and quality flags (partial occupancy,
overlapping atoms, carbon without hydrogen). Plan mode writes it as a CSV
that ``query_knowledge_data`` filters like any table; the XRD matching
backend keeps it as a parquet pre-filter (``structure_matching._backends.
_cif_index``).

Parsing is ASE's CIF tag reader, not ``ase.io.read``: a site's symmetry
copies are generated one site at a time, so a P1 cell costs nothing extra
and a mixed-occupancy site keeps all of its species. (``ase.io.read`` keeps
one species per mixed site, and its duplicate check across the whole cell
scales with the square of the atom count: ~20 s for a 4,000-atom P1 cell.)
POSCAR, xyz/extxyz and LAMMPS data files go through ``ase.io.read``.

A file that cannot be parsed still gets a row, with ``parse_error`` set, and
is listed by :func:`build_index`: nothing is dropped silently.

This module lives in ``scilink.utils`` because its worker processes import
it: ``scilink.knowledge`` loads the embedding stack (~14 s) on import.
"""
from __future__ import annotations

import concurrent.futures
import logging
import warnings
import math
import os
import re
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Iterable, Optional

_logger = logging.getLogger(__name__)

# Bump when columns or their meaning change: an older saved index is rebuilt.
INDEX_VERSION = 1

COLUMNS = [
    "path", "subfolder", "file_format", "data_block", "n_data_blocks", "database_id",
    "formula", "formula_source", "reduced_formula", "elements", "metals", "n_elements",
    "n_atoms", "n_sites",
    "a", "b", "c", "alpha", "beta", "gamma", "volume", "volume_per_atom", "density",
    "spacegroup_number", "spacegroup_symbol", "crystal_system",
    "computed_spacegroup_number", "computed_spacegroup_symbol", "spacegroup_mismatch",
    "computed_elements",
    "temperature_K", "mineral_name",
    "partial_occupancy", "min_distance", "overlapping_atoms", "carbon_without_hydrogen",
    "sites_match_formula", "parse_error", "mtime", "size", "index_version",
]

# Atoms that can coexist and sit closer than this (Å) are flagged as overlapping.
OVERLAP_DISTANCE = 0.5
# Neighbour search radius (Å) for the shortest distance; farther is reported as empty.
_DISTANCE_CUTOFF = 3.0

_CIF_SUFFIXES = {".cif"}
_ASE_SUFFIXES = {".vasp": "vasp", ".poscar": "vasp", ".xyz": "extxyz",
                 ".extxyz": "extxyz", ".lmp": "lammps-data"}
_ASE_PREFIXES = {"POSCAR": "vasp", "CONTCAR": "vasp"}

# Not metals: nonmetals, noble gases and the usual metalloids.
_NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
              "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn", "Ts", "Og"}

_SKIP_DIRS = {"__pycache__", "node_modules", ".git", "venv", "env"}


# ── which files ──────────────────────────────────────────────────────────

def structure_format(path) -> Optional[str]:
    """``"cif"``, an ASE format name, or None when ``path`` is not a structure file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _CIF_SUFFIXES:
        return "cif"
    if suffix in _ASE_SUFFIXES:
        return _ASE_SUFFIXES[suffix]
    for prefix, fmt in _ASE_PREFIXES.items():
        if p.name.startswith(prefix) and suffix in ("", ".vasp"):
            return fmt
    return None


def walk_structures(root, recursive: bool = True,
                    formats: Optional[set] = None) -> list[tuple[Path, os.stat_result]]:
    """``(path, stat)`` for every structure file under ``root``, sorted by path.
    Hidden files and folders (and caches, virtualenvs) are skipped.
    ``formats`` limits the result (e.g. ``{"cif"}``)."""
    root = Path(root)
    out: list[tuple[Path, os.stat_result]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in _SKIP_DIRS)
        for name in filenames:
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            fmt = structure_format(p)
            if fmt is None or (formats is not None and fmt not in formats):
                continue
            try:
                out.append((p, p.stat()))
            except OSError:
                continue
        if not recursive:
            break
    out.sort(key=lambda x: str(x[0]))
    return out


# ── small helpers ────────────────────────────────────────────────────────

def _missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "?", "."))


_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\(\d+\))?\s*$")
_FRACTION = re.compile(r"^\s*([-+]?\d+)\s*/\s*(\d+)\s*$")


def _num(v) -> Optional[float]:
    """A CIF number ("4.7606(3)", 4.76, "1/3") as a float; None when missing
    ("?") or not a number (the row then says so instead of using a guess)."""
    if _missing(v):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    m = _NUMBER.match(str(v))
    if m:
        return float(m.group(1))
    m = _FRACTION.match(str(v))
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    return None


def _text(v) -> Optional[str]:
    if _missing(v):
        return None
    return " ".join(str(v).split())


# Water and similar labels, as pymatgen's CIF parser reads them.
_SPECIAL_LABELS = (("Hw", "H"), ("Ow", "O"), ("Wat", "O"), ("wat", "O"))


def _symbol(raw) -> Optional[str]:
    """Element of one type symbol or label, as pymatgen's CIF parser reads
    it: water labels first, then the first two letters title-cased ("CL1" →
    Cl, "Fe3+" → Fe), then the first letter ("O2-", "ow1" → O)."""
    from ase.data import atomic_numbers
    sym = str(raw).strip()
    if not sym:
        return None
    for prefix, el in _SPECIAL_LABELS:
        if sym.startswith(prefix):
            return el
    two = sym[:2].title()
    if len(two) == 2 and two.isalpha() and two in atomic_numbers:
        return two
    one = sym[0].upper()
    if one in ("D", "T"):
        return "H"                              # deuterium / tritium count as hydrogen
    if one.isalpha() and one in atomic_numbers and one != "X":
        return one
    return None


_MIXTURE = re.compile(r"([0-9]*\.?[0-9]+)(?:\(\d+\))?\s*([A-Z][a-z]?)")


def _species(type_symbol, label) -> list:
    """``[(element, fraction)]`` for an atom site. A type symbol may name a
    mixture, as Pauling File CIFs do (``'0.648(4)Mn + 0.352(4)Fe'``)."""
    text = "" if _missing(type_symbol) else str(type_symbol)
    if "+" in text.strip("+-0123456789. "):            # "0.5Al + 0.5Si", not "Fe3+"
        parts = _MIXTURE.findall(text)
        species = [(_symbol(el), float(frac)) for frac, el in parts]
        if len(species) >= 2 and all(el is not None for el, _ in species):
            total = sum(f for _, f in species)
            if 0.98 <= total <= 1.02:                   # rounded fractions: one full site
                species = [(el, f / total) for el, f in species]
            return species
    el = _element(type_symbol, label)
    return [(el, 1.0)] if el is not None else []


def _element(type_symbol, label) -> Optional[str]:
    """Element of an atom site: from ``_atom_site_type_symbol`` or, without
    one, the label."""
    for raw in (type_symbol, label):
        if not _missing(raw):
            el = _symbol(raw)
            if el is not None:
                return el
    return None


# Pauling electronegativity (as pymatgen's ``Element.X``), for formula order.
_X = {"H": 2.2, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
      "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.9, "P": 2.19, "S": 2.58,
      "Cl": 3.16, "K": 0.82, "Ca": 1.0, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
      "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.9, "Zn": 1.65, "Ga": 1.81,
      "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.0, "Rb": 0.82, "Sr": 0.95,
      "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Tc": 1.9, "Ru": 2.2, "Rh": 2.28,
      "Pd": 2.2, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1,
      "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12, "Pr": 1.13,
      "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.2, "Gd": 1.2, "Tb": 1.1, "Dy": 1.22,
      "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3, "Ta": 1.5,
      "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28, "Au": 2.54, "Hg": 2.0,
      "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0, "At": 2.2, "Rn": 2.2, "Fr": 0.7,
      "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
      "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3,
      "No": 1.3, "Lr": 1.3}


def _formula(counts: dict) -> str:
    """A formula in the usual order: Hill (C, H, then alphabetical) for
    compounds with carbon and hydrogen, otherwise by electronegativity
    ("NaCl", "TiO2", "LiFePO4"), as pymatgen orders reduced formulas."""
    def fmt(el):
        n = counts[el]
        if abs(n - round(n)) < 1e-3:
            n = int(round(n))
            return el if n == 1 else f"{el}{n}"
        return f"{el}{n:.3g}"
    els = [e for e in counts if counts[e] > 1e-6]
    if "C" in els and "H" in els:
        order = ["C", "H"] + sorted(e for e in els if e not in ("C", "H"))
    else:
        order = sorted(els, key=lambda e: (_X.get(e, 99.0), e))
    return "".join(fmt(e) for e in order)


def _reduce(counts: dict) -> dict:
    """Divide by the common factor when every count is (close to) an integer."""
    ints = [round(n) for n in counts.values()]
    if not ints or any(abs(n - i) > 1e-3 for n, i in zip(counts.values(), ints)) or min(ints) < 1:
        return dict(counts)
    g = reduce(math.gcd, ints)
    return {el: n / g for el, n in counts.items()}


def _parse_formula(text: str) -> dict:
    """``"Al2 O3"`` / ``"C13 H18 O2"`` / ``"Ba14 Ir8 O33"`` → counts."""
    out: dict = {}
    for el, n in re.findall(r"([A-Z][a-z]?)\s*([0-9.]*)", text or ""):
        el = "H" if el in ("D", "T") else el
        out[el] = out.get(el, 0.0) + (float(n) if n not in ("", ".") else 1.0)
    return out


def _same_composition(a: dict, b: dict, tol: float = 0.02) -> bool:
    """Same elements in the same ratios (a formula unit vs a whole cell)."""
    a = {k: v for k, v in a.items() if v > 1e-6}
    b = {k: v for k, v in b.items() if v > 1e-6}
    if set(a) != set(b) or not a:
        return False
    ta, tb = sum(a.values()), sum(b.values())
    return all(abs(a[k] / ta - b[k] / tb) <= tol * max(a[k] / ta, 1e-9) + 1e-4 for k in a)


_SYSTEMS = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
            (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic")]


def _crystal_system(number: Optional[int]) -> Optional[str]:
    if not number or number < 1:
        return None
    return next(name for top, name in _SYSTEMS if number <= top)


def _empty_row(path: Path, root: Optional[Path], fmt: Optional[str]) -> dict:
    row = {c: None for c in COLUMNS}
    row["path"] = str(path)
    try:
        rel = path.parent.relative_to(root) if root is not None else Path(".")
        row["subfolder"] = "" if str(rel) == "." else str(rel)
    except ValueError:
        row["subfolder"] = ""
    row["file_format"] = fmt
    try:
        st = path.stat()
        row["mtime"], row["size"] = float(st.st_mtime), int(st.st_size)
    except OSError:
        pass
    row["index_version"] = INDEX_VERSION
    for flag in ("partial_occupancy", "overlapping_atoms", "carbon_without_hydrogen"):
        row[flag] = False
    return row


# ── geometry from sites ──────────────────────────────────────────────────

@dataclass
class _Sites:
    """Every atom of the cell: element, fractional position, occupancy."""
    cell: object                       # ase.cell.Cell
    symbols: list = field(default_factory=list)
    scaled: list = field(default_factory=list)
    occupancy: list = field(default_factory=list)
    n_sites: int = 0                   # asymmetric-unit sites read


def _fill_geometry(row: dict, sites: _Sites) -> None:
    """Composition, density and distance columns from the expanded sites."""
    import numpy as np
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers

    counts: dict = {}
    mass = 0.0
    for el, occ in zip(sites.symbols, sites.occupancy):
        counts[el] = counts.get(el, 0.0) + occ
        mass += occ * atomic_masses[atomic_numbers[el]]
    n_atoms = sum(counts.values())
    row["n_atoms"] = float(round(n_atoms, 4))
    row["n_sites"] = sites.n_sites
    row["partial_occupancy"] = bool(any(o < 0.999 for o in sites.occupancy))

    volume = float(sites.cell.volume) if sites.cell is not None and sites.cell.rank == 3 else None
    if volume:
        row["volume"] = float(round(volume, 4))
        row["density"] = float(round(mass * 1.66053906660 / volume, 4))       # g/cm^3
        if n_atoms > 0:
            row["volume_per_atom"] = float(round(volume / n_atoms, 4))

    if len(sites.symbols) > 1 and volume:
        from ase.neighborlist import neighbor_list
        atoms = Atoms(symbols=sites.symbols, scaled_positions=np.array(sites.scaled),
                      cell=sites.cell, pbc=True)
        i, j, d = neighbor_list("ijd", atoms, _DISTANCE_CUTOFF)
        if len(d):
            occ = np.array(sites.occupancy)
            # two sites whose occupancies add to 1 or less can be alternatives
            # (a mixed or split site); only atoms that can coexist count
            coexist = (occ[i] + occ[j]) > 1.0 + 1e-3
            if coexist.any():
                dmin = float(d[coexist].min())
                row["min_distance"] = round(dmin, 4)
                row["overlapping_atoms"] = bool(dmin < OVERLAP_DISTANCE)
    return counts


# Tag names, old style and DDL2 (``_space_group.it_number``), first found wins.
_SG_NUMBER_TAGS = ("_space_group_it_number", "_symmetry_int_tables_number",
                   "_space_group.it_number", "_symmetry.int_tables_number")
_SG_SYMBOL_TAGS = ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m",
                   "_space_group.name_h-m_alt", "_symmetry.space_group_name_h-m")
_RHOMBOHEDRAL = {146, 148, 155, 160, 161, 166, 167}


def _tag(block, *names):
    for name in names:
        v = block.get(name)
        if isinstance(v, list) and v:
            return v
        if not isinstance(v, list) and not _missing(v):
            return v
    return None


def _orbit(rotations, translations, pos, tol: float = 1e-3) -> list:
    """Symmetry copies of one fractional position, duplicates removed."""
    import numpy as np
    pts = (rotations @ np.asarray(pos, dtype=float) + translations) % 1.0
    kept: list = []
    for q in pts:
        if not any(np.abs((q - k) - np.round(q - k)).max() < tol for k in kept):
            kept.append(q)
    return [list(k) for k in kept]


def _expander(block, ops, number, cell):
    """A function: fractional position → its symmetry copies.

    Listed operations are applied as given (they are the file's own
    definition; a group name ASE does not recognise must not lose the
    sites). Without them the group comes from the number or symbol, and a
    rhombohedral group takes its axes from the cell, not from a crystal-
    system tag (ASE reads "trigonal" as rhombohedral axes)."""
    import numpy as np
    if ops:
        from ase.spacegroup.spacegroup import parse_sitesym
        rot, trans = parse_sitesym([str(o) for o in ops])
        rot, trans = np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)
        return lambda pos: _orbit(rot, trans, pos)
    from ase.spacegroup import Spacegroup
    symbol = _text(_tag(block, *_SG_SYMBOL_TAGS)) or ""
    if number:
        no = int(number)
    else:
        try:
            no = block.get_spacegroup(True).no
        except Exception:
            # a full monoclinic symbol ("P 1 21/c 1") ASE does not list: try
            # its short form ("P 21/c")
            short = " ".join(t for t in symbol.split() if t != "1")
            if not short or short == symbol:
                raise
            no = Spacegroup(short).no
    if no in _RHOMBOHEDRAL:
        # axes from the cell, not from a crystal-system tag or the symbol
        # (ASE reads "trigonal" as rhombohedral axes, a bare symbol as hexagonal)
        a, b, c, al, be, ga = cell.cellpar()
        hexagonal = abs(a - b) < 1e-3 * a and abs(ga - 120.0) < 0.1
        sg = Spacegroup(no, setting=1 if hexagonal else 2)
    elif re.search(r"(:\s*2|\s[SZ])\s*$", symbol):
        sg = Spacegroup(no, setting=2)                 # origin choice 2
    else:
        try:
            sg = block.get_spacegroup(True)        # the file's own setting, as before
        except Exception:
            sg = Spacegroup(no)
    return lambda pos: sg.equivalent_sites([pos], onduplicates="keep")[0]


def _sites_from_cif_block(block) -> _Sites:
    """Expand a CIF block's atom sites by its symmetry, one site at a time."""
    from ase.cell import Cell

    x = block.get("_atom_site_fract_x")
    y = block.get("_atom_site_fract_y")
    z = block.get("_atom_site_fract_z")
    if x is None or y is None or z is None:
        raise ValueError("no fractional atom sites (_atom_site_fract_x/y/z)")
    try:
        cell = Cell.fromcellpar(block.get_cellpar())
    except Exception:
        raise ValueError("no cell parameters (_cell_length_*, _cell_angle_*)") from None
    n = len(x)
    labels = block.get("_atom_site_label") or [None] * n
    types = block.get("_atom_site_type_symbol") or [None] * n
    occs = block.get("_atom_site_occupancy") or [1.0] * n

    ops = _tag(block, "_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz",
               "_space_group_symop.operation_xyz", "_symmetry_equiv.pos_as_xyz")
    if isinstance(ops, str):
        ops = [ops]                     # one operation written as a plain tag
    number = _num(_tag(block, *_SG_NUMBER_TAGS))
    p1 = (ops is not None and len(ops) <= 1) or (ops is None and number == 1)
    expand = None if p1 else _expander(block, ops, number, cell)
    sites = _Sites(cell=cell, n_sites=n)
    for k in range(n):
        species = _species(types[k], labels[k])
        if not species:
            raise ValueError(f"atom site {labels[k]!r}: no element")
        # ICSD writes "." for a zero coordinate (pymatgen reads it as 0)
        pos = [0.0 if str(v).strip() == "." else _num(v) for v in (x[k], y[k], z[k])]
        if any(v is None for v in pos):
            raise ValueError(f"atom site {labels[k]!r}: missing coordinates")
        occ = _num(occs[k])
        occ = 1.0 if occ is None else occ
        copies = [pos] if p1 else expand(pos)
        for c in copies:
            for el, frac in species:
                sites.symbols.append(el)
                sites.scaled.append(list(c))
                sites.occupancy.append(occ * frac)
    return sites


# ── one file ─────────────────────────────────────────────────────────────

def _cif_row(path: Path, row: dict) -> dict:
    from ase.io.cif import parse_cif

    blocks = list(parse_cif(str(path)))
    if not blocks:
        raise ValueError("no data blocks")
    row["n_data_blocks"] = len(blocks)
    block = next((b for b in blocks if b.has_structure()), blocks[0])
    row["data_block"] = block.name
    g = block.get

    for tag, prefix in (("_cod_database_code", "COD"), ("_database_code_icsd", "ICSD"),
                        ("_database_code_depnum_ccdc_archive", "CCDC"),
                        ("_database_code_amcsd", "AMCSD")):
        if not _missing(g(tag)):
            v = str(g(tag)).strip()
            row["database_id"] = v if v.upper().startswith(prefix) else f"{prefix}:{v}"
            break

    try:
        a, b, c, al, be, ga = block.get_cellpar()
        row.update(a=float(round(a, 5)), b=float(round(b, 5)), c=float(round(c, 5)),
                   alpha=float(round(al, 4)), beta=float(round(be, 4)), gamma=float(round(ga, 4)))
    except Exception:
        pass

    sg = _num(_tag(block, *_SG_NUMBER_TAGS))
    row["spacegroup_number"] = int(sg) if sg else None
    row["spacegroup_symbol"] = _text(_tag(block, *_SG_SYMBOL_TAGS))
    row["crystal_system"] = (_text(g("_space_group_crystal_system")
                                   or g("_symmetry_cell_setting")) or "").lower() or \
        _crystal_system(row["spacegroup_number"])
    row["temperature_K"] = _num(g("_diffrn_ambient_temperature")
                                or g("_cell_measurement_temperature"))
    row["mineral_name"] = _text(_tag(block, "_chemical_name_mineral", "_chemical_name.mineral"))

    declared = _text(_tag(block, "_chemical_formula_sum", "_chemical_formula.sum"))
    counts = None
    try:
        sites = _sites_from_cif_block(block)
        counts = _fill_geometry(row, sites)
    except Exception as e:
        # the tags above still describe the file; say why the sites didn't parse
        row["parse_error"] = f"sites: {type(e).__name__}: {e}"[:300]
    if declared:
        row["formula"], row["formula_source"] = declared, "declared"
        composition = _parse_formula(declared)
    elif counts:
        z = _num(g("_cell_formula_units_z"))
        per_unit = {el: n / z for el, n in counts.items()} if z else _reduce(counts)
        row["formula"], row["formula_source"] = _formula(per_unit), "counted"
        composition = counts
    else:
        composition = {}
    if counts and declared:
        row["sites_match_formula"] = _same_composition(_parse_formula(declared), counts)
    if counts:
        composition = counts          # elements present come from the sites when read
    return composition


def _ase_row(path: Path, fmt: str, row: dict) -> dict:
    import ase.io
    atoms = ase.io.read(str(path), format=fmt)
    row["n_data_blocks"] = 1
    cell = atoms.cell
    if cell.rank == 3:
        a, b, c, al, be, ga = cell.cellpar()
        row.update(a=float(round(a, 5)), b=float(round(b, 5)), c=float(round(c, 5)),
                   alpha=float(round(al, 4)), beta=float(round(be, 4)), gamma=float(round(ga, 4)))
    syms = ["H" if s in ("D", "T") else s for s in atoms.get_chemical_symbols()]
    sites = _Sites(cell=cell if cell.rank == 3 else None, symbols=syms,
                   scaled=(atoms.get_scaled_positions(wrap=False).tolist()
                           if cell.rank == 3 else []),
                   occupancy=[1.0] * len(atoms), n_sites=len(atoms))
    if sites.cell is not None:
        counts = _fill_geometry(row, sites)
    else:
        counts = {}
        for s in syms:
            counts[s] = counts.get(s, 0.0) + 1.0
        row.update(n_atoms=float(len(atoms)), n_sites=len(atoms))
    row["formula"], row["formula_source"] = _formula(_reduce(counts)), "counted"
    return counts


def _computed_symmetry(path: Path, row: dict) -> None:
    """pymatgen's space group, as the XRD backend has always computed it."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        return
    try:
        struct = Structure.from_file(str(path))
    except Exception:
        row["computed_spacegroup_number"] = -1
        return
    # pymatgen's element set: the XRD backend re-checks candidates with it,
    # so its pre-filter uses the same set (never drops what pymatgen accepts)
    row["computed_elements"] = "-".join(sorted({str(el) for el in struct.composition.elements}))
    try:
        sym, num = struct.get_space_group_info()
        row["computed_spacegroup_number"] = int(num)
        row["computed_spacegroup_symbol"] = str(sym)
    except Exception:
        row["computed_spacegroup_number"] = -1
        return
    if row["spacegroup_number"] and row["computed_spacegroup_number"] > 0:
        row["spacegroup_mismatch"] = row["spacegroup_number"] != row["computed_spacegroup_number"]


def pymatgen_available() -> bool:
    try:
        import pymatgen.core  # noqa: F401
        return True
    except ImportError:
        return False


def parse_structure_file(path, root=None, compute_symmetry: Optional[bool] = None) -> dict:
    """The index row for one file. Never raises: a failure is a row with
    ``parse_error`` set. ``compute_symmetry`` adds pymatgen's space group
    (default: when pymatgen is installed)."""
    path = Path(path)
    fmt = structure_format(path)
    row = _empty_row(path, Path(root) if root is not None else path.parent, fmt)
    if fmt is None:
        row["parse_error"] = "not a structure file"
        return row
    with warnings.catch_warnings():
        # ASE and pymatgen warn per file (settings, rounded coordinates); the
        # flags in the row carry what matters, and thousands of files flood a log
        warnings.simplefilter("ignore")
        try:
            composition = _cif_row(path, row) if fmt == "cif" else _ase_row(path, fmt, row)
        except Exception as e:
            row["parse_error"] = (f"{type(e).__name__}: {e}" if str(e)
                                  else f"unreadable {fmt} file ({type(e).__name__})")[:300]
            composition = None
        if fmt == "cif" and row["n_atoms"] is None:
            # ASE could not read the file or its sites: pymatgen may (a value
            # spread over two lines, a tag ASE's tokenizer rejects)
            filled = _pymatgen_fill(path, row)
            composition = filled if filled is not None else composition
        if composition is None:
            return row
        _finish_row(row, composition, path, compute_symmetry)
    return row


def _pymatgen_fill(path: Path, row: dict) -> Optional[dict]:
    """Cell, composition and density from pymatgen, for a CIF ASE could not
    read; ``parse_error`` keeps ASE's reason. None without pymatgen or when
    pymatgen cannot read it either."""
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(str(path))
    except Exception:
        return None
    lat = struct.lattice
    row.update(a=float(round(lat.a, 5)), b=float(round(lat.b, 5)), c=float(round(lat.c, 5)),
               alpha=float(round(lat.alpha, 4)), beta=float(round(lat.beta, 4)),
               gamma=float(round(lat.gamma, 4)))
    comp = struct.composition
    counts = {"H" if el in ("D", "T") else el: float(n)
              for el, n in comp.element_composition.get_el_amt_dict().items()}
    n_atoms = float(comp.num_atoms)
    row["n_atoms"] = float(round(n_atoms, 4))
    row["n_sites"] = len(struct)
    row["volume"] = float(round(lat.volume, 4))
    row["density"] = float(round(float(struct.density), 4))
    if n_atoms > 0:
        row["volume_per_atom"] = float(round(lat.volume / n_atoms, 4))
    row["partial_occupancy"] = not struct.is_ordered
    row["min_distance"] = None
    row["overlapping_atoms"] = None              # not checked on this path
    if row["formula"] is None:
        row["formula"], row["formula_source"] = _formula(_reduce(counts)), "counted"
    elif row["formula_source"] == "declared":
        row["sites_match_formula"] = _same_composition(_parse_formula(row["formula"]), counts)
    row["parse_error"] = f"sites read by pymatgen; ASE: {row['parse_error']}"[:300]
    return counts


def _finish_row(row: dict, composition: dict, path: Path,
                compute_symmetry: Optional[bool]) -> None:
    els = sorted(el for el, n in composition.items() if n > 1e-6)
    row["elements"] = "-".join(els) if els else None
    row["metals"] = "-".join(e for e in els if e not in _NONMETALS) or None
    row["n_elements"] = len(els)
    row["reduced_formula"] = _formula(_reduce(composition)) if composition else None
    row["carbon_without_hydrogen"] = "C" in els and "H" not in els
    if compute_symmetry is None:
        compute_symmetry = pymatgen_available()
    if compute_symmetry:
        _computed_symmetry(path, row)


def _parse_batch(payload: dict) -> list[dict]:
    """Worker entry point (``run_in_child``): rows for a list of paths."""
    root = payload.get("root")
    cs = payload.get("compute_symmetry")
    return [parse_structure_file(p, root, cs) for p in payload["paths"]]


# ── the index ────────────────────────────────────────────────────────────

# Below this many files to parse, workers cost more than they save.
PARALLEL_MIN_FILES = 400


@dataclass
class IndexResult:
    """What :func:`build_index` returns."""
    df: object                          # pandas.DataFrame, columns = COLUMNS
    n_parsed: int = 0
    n_reused: int = 0
    failures: list = field(default_factory=list)   # [(path, error)] for unreadable files
    out_path: Optional[Path] = None


def _read_saved(out_path: Path):
    import pandas as pd
    try:
        df = pd.read_parquet(out_path) if out_path.suffix == ".parquet" else pd.read_csv(out_path)
    except Exception as e:
        _logger.info("Rebuilding structure index %s (could not read it: %s)", out_path, e)
        return None
    if any(c not in df.columns for c in COLUMNS) or (
            len(df) and not (df["index_version"] == INDEX_VERSION).all()):
        _logger.info("Rebuilding structure index %s (older columns)", out_path)
        return None
    return df


def _write(df, out_path: Path) -> None:
    """Write atomically, through a temporary file unique to this writer (two
    processes may refresh the same index at once)."""
    import tempfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=out_path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        if out_path.suffix == ".parquet":
            df.to_parquet(tmp, engine="pyarrow", index=False)
        else:
            df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_many(paths: list[str], root: Path, compute_symmetry: Optional[bool],
                workers: Optional[int]) -> list[dict]:
    """Rows for ``paths``: in this process for a few, in fresh interpreters
    for many (parsing is CPU-bound; threads would share one GIL)."""
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    if len(paths) < PARALLEL_MIN_FILES or workers <= 1:
        return [parse_structure_file(p, root, compute_symmetry) for p in paths]

    from .child_process import ChildLost, run_in_child
    size = max(50, math.ceil(len(paths) / (workers * 4)))
    chunks = [paths[i:i + size] for i in range(0, len(paths), size)]
    payload = lambda ch: {"paths": ch, "root": str(root), "compute_symmetry": compute_symmetry}

    def run(ch):
        try:
            return run_in_child("scilink.utils.structure_index:_parse_batch", payload(ch))
        except ChildLost as e:
            _logger.warning("Structure-index worker lost (%s); parsing %d files here",
                            e.reason, len(ch))
            return _parse_batch(payload(ch))

    rows: list[dict] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_rows in pool.map(run, chunks):
            rows.extend(chunk_rows)
            done += len(chunk_rows)
            _logger.info("  ... indexed %d / %d structure files", done, len(paths))
    return rows


def build_index(root, out_path=None, *, recursive: bool = True, refresh: bool = False,
                formats: Optional[set] = None, compute_symmetry: Optional[bool] = None,
                workers: Optional[int] = None) -> IndexResult:
    """Index every structure file under ``root``; write it to ``out_path``
    (``.csv`` or ``.parquet``) when given.

    Incremental: with a saved index at ``out_path`` (and ``refresh`` False),
    rows of files whose mtime and size are unchanged are reused, files that
    are gone are dropped, and only new or changed files are parsed.
    """
    import pandas as pd

    root = Path(root)
    walked = walk_structures(root, recursive=recursive, formats=formats)
    out_path = Path(out_path) if out_path is not None else None
    # pymatgen's space group only when asked for (default: when installed)
    # and installed: a request without pymatgen must not mark every row stale
    compute_symmetry = pymatgen_available() if compute_symmetry is None else (
        bool(compute_symmetry) and pymatgen_available())

    saved = None if (refresh or out_path is None or not out_path.is_file()) else _read_saved(out_path)
    keep = None
    to_parse = [str(p) for p, _ in walked]
    if saved is not None and len(saved):
        current = pd.DataFrame({"path": to_parse,
                                "_mtime": [float(st.st_mtime) for _, st in walked],
                                "_size": [int(st.st_size) for _, st in walked]})
        merged = saved.assign(path=saved["path"].astype(str)).merge(current, on="path", how="inner")
        same = ((merged["_mtime"] - merged["mtime"].astype(float)).abs() < 1e-3) & \
               (merged["_size"] == pd.to_numeric(merged["size"], errors="coerce").fillna(-1))
        # a saved row without computed symmetry is reparsed once it is wanted
        if compute_symmetry:
            same &= merged["computed_spacegroup_number"].notna() | merged["parse_error"].notna()
        keep = merged[same].drop(columns=["_mtime", "_size"])[COLUMNS]
        kept = set(keep["path"])
        to_parse = [p for p in to_parse if p not in kept]

    new_rows = _parse_many(to_parse, root, compute_symmetry, workers) if to_parse else []
    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    parts = [d for d in (keep, new_df) if d is not None and len(d)]
    df = (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS))
    if len(df):
        df = df.sort_values("path", kind="stable").reset_index(drop=True)
    unchanged = saved is not None and not new_rows and keep is not None and len(keep) == len(saved)
    if out_path is not None and not unchanged:
        _write(df, out_path)

    failed = df[df["elements"].isna() & df["parse_error"].notna()] if len(df) else df
    failures = list(zip(failed["path"].astype(str), failed["parse_error"].astype(str))) if len(failed) else []
    return IndexResult(df=df, n_parsed=len(new_rows),
                       n_reused=0 if keep is None else len(keep),
                       failures=failures, out_path=out_path)


def summarize(result: IndexResult, sample: int = 5) -> dict:
    """A compact description of an index, for a tool result."""
    df = result.df
    if not len(df):
        return {"n_rows": 0}
    def count(col):
        return int(df[col].fillna(False).astype(bool).sum())
    show = ["path", "formula", "spacegroup_number", "a", "b", "c", "density",
            "partial_occupancy", "overlapping_atoms"]
    rows = df[df["parse_error"].isna()].head(sample)[show]
    return {
        "n_rows": int(len(df)),
        "n_parsed": result.n_parsed,
        "n_reused": result.n_reused,
        "n_failed": len(result.failures),
        "failures": [{"path": p, "error": e} for p, e in result.failures[:20]],
        "quality": {
            "partial_occupancy": count("partial_occupancy"),
            "overlapping_atoms": count("overlapping_atoms"),
            "carbon_without_hydrogen": count("carbon_without_hydrogen"),
            "spacegroup_mismatch": count("spacegroup_mismatch"),
            "sites_differ_from_formula": int((df["sites_match_formula"] == False).sum()),  # noqa: E712
            "sites_not_read": int((df["parse_error"].notna() & df["n_atoms"].isna()
                                   & df["elements"].notna()).sum()),
            "read_by_pymatgen": int(df["parse_error"].astype(str)
                                    .str.startswith("sites read by pymatgen").sum()),
        },
        "formats": {str(k): int(v) for k, v in df["file_format"].value_counts().items()},
        "columns": COLUMNS,
        "sample_rows": rows.astype(object).where(rows.notna(), None).to_dict(orient="records"),
    }
