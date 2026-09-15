"""
LAMMPS utilities for the MDSimulationAgent.

Provides:
  - Data file parsing and system analysis
  - Script validation (structural checks)
  - Script cleaning and fixing
  - Force field file integration
  - run_with_potential: engine-side integration of a deployed MLIP
    potential — the LAMMPS skill's own knowledge of which MLIP
    backends have a LAMMPS pair_style and how to write the input.
    The MD agent calls this generically via TOOL_REGISTRY; it never
    branches on the engine itself.

Called by MDSimulationAgent when the LAMMPS skill is active.
Decoupled from the skill so they can be tested independently.
"""

import os
import re
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import ase.data

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)


# ─── Mass-to-Element Lookup (from ASE) ───────────────────────────────

def _build_mass_lookup() -> List[Tuple[float, str]]:
    """
    Build a sorted list of (mass, symbol) from ASE's periodic table.
    Used for nearest-mass matching when data file comments are absent.
    """
    pairs = []
    for Z in range(1, len(ase.data.atomic_masses)):
        symbol = ase.data.chemical_symbols[Z]
        mass = ase.data.atomic_masses[Z]
        if mass > 0:
            pairs.append((mass, symbol))
    pairs.sort(key=lambda x: x[0])
    return pairs

_ASE_MASS_TABLE = _build_mass_lookup()


def element_from_mass(mass: float, tolerance: float = 1.5) -> Optional[str]:
    """
    Identify element symbol from atomic mass using ASE data.

    Args:
        mass: Atomic mass in amu.
        tolerance: Maximum allowed deviation from standard mass.

    Returns:
        Element symbol, or None if no match within tolerance.
    """
    best_symbol = None
    best_delta = tolerance
    for ref_mass, symbol in _ASE_MASS_TABLE:
        delta = abs(ref_mass - mass)
        if delta < best_delta:
            best_delta = delta
            best_symbol = symbol
    return best_symbol


# ─── System Classification Constants ─────────────────────────────────

_METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
}
_SEMICONDUCTORS = {"Si", "Ge", "Ga", "As", "In", "Sb", "P"}
_HALIDES = {"F", "Cl", "Br", "I"}

# Pair styles that must NOT have kspace
_NO_KSPACE_STYLES = {
    "eam", "eam/alloy", "eam/fs", "meam", "meam/c",
    "tersoff", "tersoff/zbl", "sw", "airebo", "airebo-m",
    "rebo", "lcbop", "bop", "snap", "comb3",
}
# Pair styles that REQUIRE kspace
_REQUIRES_KSPACE = {
    "lj/cut/coul/long", "lj/charmm/coul/long", "buck/coul/long",
    "lj/long/coul/long", "coul/long", "tip4p/long",
}


# ─── Tool Availability ───────────────────────────────────────────────

def check_lammps() -> Dict[str, Any]:
    """
    Check if LAMMPS is available and what packages are installed.

    Returns:
        {"available": bool, "path": str|None, "packages": [...]}
    """
    lmp_path = (
        shutil.which("lmp")
        or shutil.which("lmp_serial")
        or shutil.which("lmp_mpi")
    )
    result = {"available": lmp_path is not None, "path": lmp_path, "packages": []}
    if lmp_path:
        try:
            import subprocess
            proc = subprocess.run(
                [lmp_path, "-h"], capture_output=True, text=True, timeout=10,
            )
            in_packages = False
            for line in proc.stdout.split("\n"):
                if "Installed packages:" in line:
                    in_packages = True
                    continue
                if in_packages and line.strip():
                    result["packages"].extend(line.strip().split())
                elif in_packages:
                    break
        except Exception:
            pass
    return result


def default_run_command(script: str = "{script}") -> Optional[str]:
    """Conventional local LAMMPS run-command template for the on-PATH binary.

    Resolves the first available LAMMPS binary (lmp / lmp_serial / lmp_mpi) and
    returns ``"<binary> -in {script}"`` — the invocation the refinement loop
    fills with each phase's deck filename. The engine's own knowledge of how it
    is launched, so a one-shot workflow can execute LAMMPS without any engine
    name or command hardcoded in shared code.

    Returns:
        The run-command template, or ``None`` if no LAMMPS binary is on PATH —
        the caller should then fall back to a user-supplied ``run_command``.
    """
    info = check_lammps()
    if not info.get("available") or not info.get("path"):
        return None
    return f"{info['path']} -in {script}"


# ─── Data File Parsing ───────────────────────────────────────────────

def _detect_atom_style(atoms_lines: List[str], has_bonds: bool) -> str:
    """
    Detect atom_style from the column count of the first data line.

      atomic:    id type x y z              (5 cols min)
      charge:    id type q x y z            (6 cols min)
      molecular: id mol type x y z          (6 cols min)
      full:      id mol type q x y z        (7 cols min)

    Image flags may add 3 extra columns. Uses has_bonds to
    disambiguate charge (no bonds) from molecular (bonds).
    """
    for line in atoms_lines:
        parts = line.split()
        ncols = len(parts)
        # Core columns (excluding possible image flags)
        # Image flags: 3 trailing integers, so core = ncols or ncols-3
        if ncols >= 10 and has_bonds:
            return "full"       # 7 core + 3 image
        elif ncols >= 9 and not has_bonds:
            return "charge"     # 6 core + 3 image
        elif ncols >= 9 and has_bonds:
            return "molecular"  # 6 core + 3 image (or full without q)
        elif ncols >= 7 and has_bonds:
            return "full"
        elif ncols == 6:
            return "molecular" if has_bonds else "charge"
        elif ncols >= 5:
            return "atomic"
        break
    return "full"  # conservative default


def _type_column_index(atom_style: str) -> int:
    """Return the 0-based column index of atom type."""
    if atom_style in ("full", "molecular"):
        return 2  # id mol type ...
    return 1      # id type ... (atomic, charge)


def _detect_vacuum_gap(
    box_dims: List[float],
    atoms_lines: List[str],
    atom_style: str,
    min_gap_angstrom: float = 8.0,
) -> Dict[str, Any]:
    """
    Detect vacuum gaps indicating surface/slab geometry.

    A gap is flagged only if BOTH:
      - vacuum fraction > 25% of box dimension
      - absolute gap size > min_gap_angstrom (default 8 Å)

    This prevents false positives on small unit cells where atoms
    sit at lattice sites that don't span the full box.
    """
    result = {"has_vacuum": False, "vacuum_axis": None, "vacuum_fraction": 0.0}

    offsets = {"full": 4, "charge": 3, "molecular": 3, "atomic": 2}
    xyz_start = offsets.get(atom_style, 2)

    coords: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    for line in atoms_lines[:5000]:
        parts = line.split()
        try:
            for dim in range(3):
                coords[dim].append(float(parts[xyz_start + dim]))
        except (IndexError, ValueError):
            continue

    if not coords[0]:
        return result

    axis_names = ["x", "y", "z"]
    for dim in range(3):
        if not coords[dim] or box_dims[dim] <= 0:
            continue
        span = max(coords[dim]) - min(coords[dim])
        gap = box_dims[dim] - span
        vac_frac = gap / box_dims[dim]
        if vac_frac > 0.25 and gap > min_gap_angstrom and vac_frac > result["vacuum_fraction"]:
            result["has_vacuum"] = True
            result["vacuum_axis"] = axis_names[dim]
            result["vacuum_fraction"] = round(vac_frac, 3)

    return result

def _dihedral_style_from_coeffs(coeffs: List[str]) -> str:
    """Infer dihedral_style from one Dihedral Coeffs row's coefficient arity."""
    n = len(coeffs)
    if n == 3:
        return "harmonic"               # K d n
    if n >= 4:
        # fourier: m K1 n1 d1 [K2 n2 d2 ...] — leading integer term count, 1+3m total
        try:
            m = int(coeffs[0])
            if m >= 1 and n == 1 + 3 * m:
                return "fourier"
        except ValueError:
            pass
        if n == 4:
            return "opls"               # K1 K2 K3 K4
    return ""


def _detect_ff_styles(lines: List[str]) -> Dict[str, str]:
    """Infer the bond/angle/dihedral/improper styles a data file's Coeffs require.

    A LAMMPS data file carries Coeffs but not the *_style commands; the style is
    implied by the coefficient arity. Declaring a mismatched style makes
    read_data abort ("Incorrect args for <...> coefficients"). We read the first
    data row of each Coeffs section and map its arity to the common
    SMIRNOFF/AMBER/GAFF styles; an unrecognized arity is omitted (no claim)
    rather than guessed.
    """
    coeff_headers = {"Bond Coeffs", "Angle Coeffs",
                     "Dihedral Coeffs", "Improper Coeffs"}
    enders = {"Atoms", "Velocities", "Bonds", "Angles", "Dihedrals",
              "Impropers", "Masses", "Pair Coeffs"}
    styles: Dict[str, str] = {}
    section = None
    for raw in lines:
        s = raw.strip()
        if s in coeff_headers:
            section = s
            continue
        if section is None:
            continue
        if not s or s.startswith("#"):
            continue
        if s in enders:
            section = None
            continue
        coeffs = s.split("#", 1)[0].split()[1:]   # drop the type id
        n = len(coeffs)
        if section == "Bond Coeffs" and "bond" not in styles:
            styles["bond"] = "harmonic" if n == 2 else ""
        elif section == "Angle Coeffs" and "angle" not in styles:
            styles["angle"] = "harmonic" if n == 2 else ("charmm" if n == 4 else "")
        elif section == "Dihedral Coeffs" and "dihedral" not in styles:
            styles["dihedral"] = _dihedral_style_from_coeffs(coeffs)
        elif section == "Improper Coeffs" and "improper" not in styles:
            styles["improper"] = "cvff" if n == 3 else ""
    return {k: v for k, v in styles.items() if v}


def parse_data_file(data_file: str) -> Dict[str, Any]:
    """
    Parse a LAMMPS data file to extract system information.

    Handles atom_style atomic, charge, molecular, and full.
    Detects system category and vacuum gaps.
    """
    info: Dict[str, Any] = {
        "atom_count": 0,
        "bond_count": 0,
        "angle_count": 0,
        "atom_types": 0,
        "bond_types": 0,
        "angle_types": 0,
        "dihedral_types": 0,
        "improper_types": 0,
        "box_dimensions": [0.0, 0.0, 0.0],
        "has_pair_coeffs": False,
        "has_bond_coeffs": False,
        "required_styles": {},
        "atom_style": "unknown",
        "atom_type_labels": {},
        "mass_map": {},
        "elements": [],
        "element_counts": {},
        "has_bonds": False,
        "has_water": False,
        "has_ions": False,
        "has_organic": False,
        "has_metal": False,
        "has_semiconductor": False,
        "system_category": "unknown",
        "has_vacuum": False,
        "vacuum_axis": None,
    }

    try:
        with open(data_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Cannot read data file: {e}")
        return info

    # ── Parse header ──
    _HEADER_PATTERNS = {
        " atoms": "atom_count",
        " bonds": "bond_count",
        " angles": "angle_count",
        "atom types": "atom_types",
        "bond types": "bond_types",
        "angle types": "angle_types",
        "dihedral types": "dihedral_types",
        "improper types": "improper_types",
    }
    for line in lines[:40]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            n = int(parts[0])
        except (ValueError, IndexError):
            # Box bounds
            lower = stripped.lower()
            try:
                if "xlo" in lower and "xhi" in lower:
                    info["box_dimensions"][0] = float(parts[1]) - float(parts[0])
                elif "ylo" in lower and "yhi" in lower:
                    info["box_dimensions"][1] = float(parts[1]) - float(parts[0])
                elif "zlo" in lower and "zhi" in lower:
                    info["box_dimensions"][2] = float(parts[1]) - float(parts[0])
            except (ValueError, IndexError):
                pass
            continue

        lower = stripped.lower()
        for pattern, key in _HEADER_PATTERNS.items():
            if pattern in lower:
                info[key] = n
                break

    info["has_bonds"] = info["bond_count"] > 0

    # ── Detect coefficient sections ──
    section_names = {l.strip() for l in lines}
    info["has_pair_coeffs"] = "Pair Coeffs" in section_names
    info["has_bond_coeffs"] = "Bond Coeffs" in section_names
    # Styles implied by the Coeffs arity (the deck must declare these exactly).
    info["required_styles"] = _detect_ff_styles(lines)

    # ── Parse Masses ──
    in_masses = False
    _SECTION_HEADERS = {
        "Atoms", "Pair Coeffs", "Bond Coeffs", "Velocities",
        "Bonds", "Angles", "Dihedrals", "Impropers",
    }
    for line in lines:
        stripped = line.strip()
        if stripped == "Masses":
            in_masses = True
            continue
        if in_masses:
            # Section headers end the block
            if stripped in _SECTION_HEADERS or stripped.startswith("Atoms"):
                break
            # Skip blank lines and comments (LAMMPS always has a blank
            # line between the section header and the data)
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                type_id = int(parts[0])
                mass = float(parts[1])
            except (ValueError, IndexError):
                continue

            label = ""
            if "#" in stripped:
                label = stripped.split("#", 1)[1].strip()

            element = None
            if label:
                match = re.match(r'[A-Z][a-z]?', label)
                if match:
                    element = match.group()

            if not element:
                element = element_from_mass(mass)

            if not element:
                element = f"X{type_id}"

            info["mass_map"][type_id] = (mass, element)
            info["atom_type_labels"][type_id] = label or element

    # ── Extract Atoms section ──
    atoms_lines: List[str] = []
    in_atoms = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Atoms" or stripped.startswith("Atoms #"):
            in_atoms = True
            continue
        if in_atoms:
            if stripped in _SECTION_HEADERS or stripped in (
                "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
            ):
                break
            if stripped and not stripped.startswith("#"):
                atoms_lines.append(stripped)

    # ── Detect atom_style and count types ──
    info["atom_style"] = _detect_atom_style(atoms_lines, info["has_bonds"])
    type_col = _type_column_index(info["atom_style"])

    type_counts: Dict[int, int] = {}
    for line in atoms_lines:
        parts = line.split()
        try:
            atom_type = int(parts[type_col])
            type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
        except (IndexError, ValueError):
            pass

    for type_id, count in type_counts.items():
        element = info["mass_map"].get(type_id, (0.0, f"Type{type_id}"))[1]
        info["element_counts"][element] = (
            info["element_counts"].get(element, 0) + count
        )
    info["elements"] = sorted(info["element_counts"].keys())

    # ── Vacuum detection ──
    vacuum = _detect_vacuum_gap(info["box_dimensions"], atoms_lines, info["atom_style"])
    info.update({
        "has_vacuum": vacuum["has_vacuum"],
        "vacuum_axis": vacuum.get("vacuum_axis"),
    })

    # ── System classification ──
    ec = info["element_counts"]
    elems = set(info["elements"])

    info["has_water"] = (
        "O" in ec and "H" in ec
        and ec.get("H", 0) >= 2 * ec.get("O", 0)
        and info["has_bonds"]
    )
    # Monatomic cations common in electrolytes (alkali, alkaline-earth, and a few
    # multivalent cations incl. Zn2+) plus halide anions. NOTE: this does not see
    # polyatomic anions (triflate, sulfate, nitrate) — a halide-free box with such
    # an anion falls to "solution" rather than "electrolyte" (both non-biomolecular,
    # so nothing downstream breaks). Teaching the ion check about polyatomic anions
    # is tracked in #494.
    info["has_ions"] = bool(elems & _HALIDES) or bool(
        elems & {"Na", "K", "Ca", "Mg", "Li", "Rb", "Cs", "Zn", "Al", "Sr", "Ba"}
    )
    info["has_organic"] = "C" in ec and info["has_bonds"]
    info["has_metal"] = bool(elems & _METALS) and not info["has_bonds"]
    info["has_semiconductor"] = bool(elems & _SEMICONDUCTORS) and not info["has_bonds"]

    # Category (bonded molecular systems first, then non-bonded solids).
    # KNOWN LIMITATION (#494): "biomolecular" keys on nitrogen (proteins and
    # nucleic acids carry N in their amide/amine/base backbones; N-free organic
    # electrolyte/solvent species — triflate, sulfone, carbonates, glymes — do
    # not). N is a PROXY, not a biomolecule signature: it mislabels N-bearing
    # SMALL-molecule electrolytes and solvents — acetonitrile, imidazolium /
    # ammonium / pyridinium ionic liquids, amide solvents (DMF/DMAc/NMP),
    # organic-nitrate systems — as biomolecular, because this branch is checked
    # before the electrolyte/solution ones. Those are common chemistries, not
    # rare misses. The honest discriminator is size/topology (biopolymers are
    # large; MeCN, imidazolium, triflate are small), but atom/bond counts alone
    # don't separate them cleanly (alanine dipeptide ~22 atoms overlaps
    # imidazolium ~19). N is the interim heuristic pending #494; the
    # MeCN+NaCl+water test pins the current (wrong) behavior so it stays visible.
    if info["has_bonds"]:
        if "N" in ec and "C" in ec:
            info["system_category"] = "biomolecular"
        elif info["has_water"] and info["has_ions"]:
            info["system_category"] = "electrolyte"
        elif info["has_water"] and info["has_organic"]:
            info["system_category"] = "solution"
        elif info["has_water"]:
            info["system_category"] = "liquid"
        elif info["has_organic"]:
            info["system_category"] = "molecular_liquid"
        else:
            info["system_category"] = "unknown"
    elif elems <= _METALS:
        info["system_category"] = "metal"
    elif elems & _SEMICONDUCTORS and "O" not in ec:
        info["system_category"] = "semiconductor"
    elif "O" in ec and elems & _METALS:
        info["system_category"] = "oxide"
    elif info["has_ions"]:
        info["system_category"] = "ionic"
    else:
        info["system_category"] = "unknown"

    return info


def _atoms_section_rows(data_file: str) -> List[str]:
    """Return the atom rows of a data file's ``Atoms`` section.

    Strips the ``Atoms`` header (and any ``# comment`` on it), the blank line
    that follows, and stops at the next section header or trailing blank.
    """
    with open(data_file) as f:
        lines = f.readlines()
    rows: List[str] = []
    in_atoms = False
    for line in lines:
        s = line.strip()
        if not in_atoms:
            if s.split("#", 1)[0].strip() == "Atoms":
                in_atoms = True
            continue
        if not s:
            if rows:            # blank after rows collected → section ended
                break
            continue            # blank right after the "Atoms" header
        first = s[0]
        if not (first.isdigit() or first == "-"):   # a new section header
            break
        rows.append(s)
    return rows


def map_selections_to_types(
    data_file: str,
    components_json: str,
    selections: List[str],
) -> Dict[str, List[int]]:
    """Map engine-neutral atom selections to the LAMMPS atom types satisfying them.

    A selection is ``"<species>"`` (all atoms of that molecular species) or
    ``"<species>:<Element>"`` (that element within that species). Species resolve
    to molecule-ID ranges from the components manifest (molecules are packed in
    manifest order); elements from each type's mass. Two selections that return
    the *same* type list are indistinguishable to a type-based analysis such as
    ``compute rdf`` — the signal the contradiction framework's
    ``selection_realizable`` check reads.

    Args:
        data_file: LAMMPS data file (``atom_style full``/``molecular`` carry
            molecule-IDs; ``atomic``/``charge`` do not, so species selections
            cannot be resolved and return empty).
        components_json: The run's components manifest (species names + counts).
        selections: Selection strings to resolve.

    Returns:
        ``{selection: sorted list of matching atom-type ids}``; an empty list for
        a selection matching no type (unknown species/element, or a molecule-less
        atom style).
    """
    import json

    info = parse_data_file(data_file)
    atom_style = info.get("atom_style", "full")
    # parse_data_file's mass_map maps type -> (mass, element); tolerate a bare
    # mass too (fall back to mass→element lookup).
    mass_map = info.get("mass_map", {}) or {}
    type_element: Dict[int, Optional[str]] = {}
    for t, m in mass_map.items():
        try:
            if isinstance(m, (tuple, list)) and len(m) >= 2:
                type_element[int(t)] = m[1]
            else:
                type_element[int(t)] = element_from_mass(float(m))
        except (TypeError, ValueError):
            continue

    with open(components_json) as fh:
        comps = json.load(fh).get("components", [])
    mol_species: Dict[int, str] = {}
    mid = 1
    for c in comps:
        for _ in range(int(c.get("count", 0))):
            mol_species[mid] = c.get("name")
            mid += 1

    type_col = _type_column_index(atom_style)
    mol_col = 1 if atom_style in ("full", "molecular") else None
    type_species: Dict[int, set] = {}
    if mol_col is not None:
        for row in _atoms_section_rows(data_file):
            toks = row.split()
            if len(toks) <= max(type_col, mol_col):
                continue
            try:
                t, m = int(toks[type_col]), int(toks[mol_col])
            except ValueError:
                continue
            sp = mol_species.get(m)
            if sp is not None:
                type_species.setdefault(t, set()).add(sp)

    result: Dict[str, List[int]] = {}
    for sel in selections:
        if ":" in sel:
            species, element = sel.split(":", 1)
        else:
            species, element = sel, None
        matched = [
            t for t in sorted(type_element)
            if (element is None or type_element.get(t) == element)
            and species in type_species.get(t, ())
        ]
        result[sel] = matched
    return result


_DATA_SECTIONS = (
    "Masses", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs",
    "Dihedral Coeffs", "Improper Coeffs", "Atoms", "Velocities", "Bonds",
    "Angles", "Dihedrals", "Impropers",
)


def split_shared_types(
    data_file: str,
    components_json: str,
    collisions: List[List[str]],
) -> Dict[str, Any]:
    """Give colliding species distinct atom types so a type-based analysis can
    separate them, without changing the physics.

    For each collision group (selections that ``map_selections_to_types`` found
    share an atom type), the first selection keeps its type; each other selection
    gets a fresh atom type per shared type, with **identical** mass and Pair
    Coeffs (so the force field is unchanged), and its atoms (that type, within
    that species' molecules) are reassigned to the new type. Bonds/angles/etc.
    reference atom and bond-type IDs, not atom types, so they are untouched.

    Args:
        data_file: LAMMPS data file to transform.
        components_json: Components manifest (for species → molecule-ID ranges).
        collisions: Groups of selections sharing a type, as reported by the
            ``selection_realizable`` check.

    Returns:
        ``{"data_file_text": <new data file>, "type_map": {selection: [types]},
        "new_types": {new_type: source_type}}``. ``type_map`` gives each
        collided selection its now-distinct type list.
    """
    import json

    all_sels = [s for grp in collisions for s in grp]
    sel_types = map_selections_to_types(data_file, components_json, all_sels)

    comps = json.load(open(components_json)).get("components", [])
    species_mols: Dict[str, set] = {}
    mid = 1
    for c in comps:
        cnt = int(c.get("count", 0))
        species_mols.setdefault(c.get("name"), set()).update(range(mid, mid + cnt))
        mid += cnt

    info = parse_data_file(data_file)
    atom_style = info.get("atom_style", "full")
    type_col = _type_column_index(atom_style)
    mol_col = 1 if atom_style in ("full", "molecular") else None
    n_types = int(info.get("atom_types", 0))
    mass_map = info.get("mass_map", {}) or {}

    rules: List[Tuple[int, set, int]] = []   # (source_type, mol_set, new_type)
    new_src: Dict[int, int] = {}             # new_type -> source_type
    type_map: Dict[str, List[int]] = {}
    next_type = n_types
    for grp in collisions:
        for i, sel in enumerate(grp):
            src = sel_types.get(sel) or []
            if i == 0:
                type_map[sel] = list(src)     # first selection keeps its type(s)
                continue
            mols = species_mols.get(sel.split(":", 1)[0], set())
            new_for_sel = []
            for st in src:
                next_type += 1
                rules.append((st, mols, next_type))
                new_src[next_type] = st
                new_for_sel.append(next_type)
            type_map[sel] = new_for_sel

    if not new_src or mol_col is None:
        return {"data_file_text": open(data_file).read(),
                "type_map": type_map, "new_types": new_src}

    # Grab each source type's raw Pair Coeffs row (tokens after the index).
    lines = open(data_file).read().splitlines()
    pair_rows: Dict[int, List[str]] = {}
    sec = None
    for ln in lines:
        head = ln.split("#", 1)[0].strip()
        if head in _DATA_SECTIONS:
            sec = head
            continue
        s = ln.strip()
        if sec == "Pair Coeffs" and s and s[0].isdigit():
            toks = s.split()
            pair_rows[int(toks[0])] = toks[1:]

    out: List[str] = []
    sec = None
    seen_rows = False
    for ln in lines:
        s = ln.strip()
        head = s.split("#", 1)[0].strip()
        if head in _DATA_SECTIONS:
            sec, seen_rows = head, False
            out.append(ln)
            continue
        if s.endswith("atom types") and s.split()[0].isdigit():
            out.append(f"{n_types + len(new_src)} atom types")
            continue
        # Append the duplicated rows at the END of Masses / Pair Coeffs.
        if sec in ("Masses", "Pair Coeffs") and not s and seen_rows:
            for nt in sorted(new_src):
                if sec == "Masses":
                    mass = mass_map.get(new_src[nt], (0, ""))
                    mass = mass[0] if isinstance(mass, (tuple, list)) else mass
                    out.append(f"{nt}\t{mass}")
                else:
                    out.append(f"{nt}\t" + "\t".join(pair_rows.get(new_src[nt], [])))
            out.append(ln)
            sec, seen_rows = None, False
            continue
        if sec in ("Masses", "Pair Coeffs") and s and s[0].isdigit():
            seen_rows = True
            out.append(ln)
            continue
        if sec == "Atoms" and s and (s[0].isdigit() or s[0] == "-"):
            toks = s.split()
            try:
                t, m = int(toks[type_col]), int(toks[mol_col])
            except (ValueError, IndexError):
                out.append(ln)
                continue
            for st, mols, nt in rules:
                if t == st and m in mols:
                    toks[type_col] = str(nt)
                    break
            out.append("\t".join(toks))
            continue
        out.append(ln)

    return {"data_file_text": "\n".join(out) + "\n",
            "type_map": type_map, "new_types": new_src}


def format_type_info(data_file: str) -> str:
    """Format data file contents for LLM prompts."""
    info = parse_data_file(data_file)
    lines = [
        "DATA FILE ANALYSIS:",
        f"  Atoms: {info['atom_count']} ({info['atom_types']} types)",
        f"  Bonds: {info['bond_count']} ({info['bond_types']} types)",
        f"  Angles: {info['angle_count']} ({info['angle_types']} types)",
        f"  Dihedrals: {info['dihedral_types']} types, Impropers: {info['improper_types']} types",
        f"  Box: {[f'{d:.2f}' for d in info['box_dimensions']]}",
        f"  Detected atom_style: {info['atom_style']}",
        f"  Coefficients in data file: {'Yes' if info['has_pair_coeffs'] else 'No'}",
        f"  System category: {info['system_category']}",
    ]
    rs = info.get("required_styles") or {}
    if rs:
        lines.append(
            "  Required styles (the data file's Coeffs are in these formats — "
            "declare EXACTLY these, a mismatch aborts read_data): "
            + ", ".join(f"{k}_style {v}" for k, v in rs.items())
        )
    if info["has_vacuum"]:
        lines.append(f"  Vacuum gap: {info['vacuum_axis']} axis (surface/slab)")
    lines.append("")
    lines.append("MASS-ELEMENT MAPPING:")
    for tid in sorted(info["mass_map"]):
        mass, element = info["mass_map"][tid]
        label = info["atom_type_labels"].get(tid, "")
        lines.append(f"  type {tid} = {element} (mass {mass:.3f}) {label}")
    lines.append("")
    lines.append("ELEMENT COUNTS:")
    for el in sorted(info["element_counts"]):
        lines.append(f"  {el}: {info['element_counts'][el]}")
    return "\n".join(lines)


# ─── Dry-run twin (cheap setup-only validation) ──────────────────────

def prepare_dry_run(script: str) -> str:
    """Return a setup-only "dry-run" twin of a LAMMPS deck.

    The twin keeps every command LAMMPS validates at setup (units, atom_style,
    the force-field styles, read_data, kspace_style, fixes) but trims the
    dynamics so it costs ~one force evaluation: every ``run N`` becomes
    ``run 0``, ``minimize ...`` becomes a setup-only ``minimize 0.0 0.0 0 0``,
    and output commands (dump/restart/write_*) are dropped. Running it surfaces
    syntax/setup errors — style/coeff mismatches, command ordering, kspace vs.
    triclinic — in ~1 s, without doing real dynamics. If the deck has no
    run/minimize, a ``run 0`` is appended so setup actually executes.
    """
    drop = {"dump", "dump_modify", "undump", "restart",
            "write_dump", "write_restart", "write_data"}
    out: List[str] = []
    has_setup = False
    for line in script.splitlines():
        s = line.strip()
        kw = s.split()[0].lower() if s else ""
        if kw == "run":
            out.append("run 0")
            has_setup = True
        elif kw == "minimize":
            out.append("minimize 0.0 0.0 0 0")
            has_setup = True
        elif kw in drop:
            continue
        else:
            out.append(line)
    if not has_setup:
        out.append("run 0")
    return "\n".join(out) + "\n"


# ─── Script Validation ───────────────────────────────────────────────

def validate_script(
    script_path: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate a LAMMPS input script structurally.

    Checks: required commands, ordering, forbidden combinations,
    unit-aware parameter ranges, potential file existence.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "has_minimize": False,
        "has_run": False,
        "has_shake": False,
        "timestep": None,
        "units": None,
        "atom_style": None,
        "pair_style": None,
        "boundary": None,
    }

    try:
        content = Path(script_path).read_text()
    except Exception as e:
        return {**result, "valid": False, "errors": [f"Cannot read: {e}"]}

    lines = content.split("\n")
    commands_seen: set = set()
    command_order: List[str] = []

    # Track integrator fixes over their lifecycle (fix … / unfix …) so a
    # legitimate sequential deck — NPT equilibrate, unfix, then NVT produce on
    # the same group — is not flagged. `active_integrators` maps fix-ID ->
    # (group, family); a conflict is recorded only when an nvt and an npt/nph
    # fix are active on the same group at the same time.
    active_integrators: Dict[str, tuple] = {}
    integrator_conflict: set = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        keyword = parts[0].lower()
        commands_seen.add(keyword)
        command_order.append(keyword)

        if keyword == "units" and len(parts) >= 2:
            result["units"] = parts[1].lower()
        elif keyword == "atom_style" and len(parts) >= 2:
            result["atom_style"] = parts[1].lower()
        elif keyword == "pair_style" and len(parts) >= 2:
            result["pair_style"] = parts[1].lower()
        elif keyword == "boundary" and len(parts) >= 4:
            result["boundary"] = parts[1:4]
        elif keyword == "timestep" and len(parts) >= 2:
            try:
                result["timestep"] = float(parts[1])
            except ValueError:
                pass
        elif keyword == "minimize":
            result["has_minimize"] = True
        elif keyword == "run":
            result["has_run"] = True
        elif keyword == "fix" and len(parts) >= 4:
            fix_id = parts[1]
            group = parts[2]
            style = parts[3].lower()
            if "shake" in style or "rattle" in style:
                result["has_shake"] = True
            if style in ("npt", "nph", "nvt"):
                family = "nvt" if style == "nvt" else "npt"
                # Conflict only if a DIFFERENT-family integrator is already
                # active on this same group (i.e. never unfixed).
                for a_group, a_family in active_integrators.values():
                    if a_group == group and a_family != family:
                        integrator_conflict.add(group)
                active_integrators[fix_id] = (group, family)
        elif keyword == "unfix" and len(parts) >= 2:
            active_integrators.pop(parts[1], None)

    errors = result["errors"]
    warnings = result["warnings"]

    # ── Required commands ──
    if "units" not in commands_seen:
        errors.append("Missing 'units' command")
    if "atom_style" not in commands_seen:
        errors.append("Missing 'atom_style' command")
    # A system can be brought in from a data/restart file or built in-deck with
    # create_atoms (after create_box + lattice/region) — all three are valid.
    if not commands_seen & {"read_data", "read_restart", "create_atoms"}:
        errors.append(
            "Missing system definition — need 'read_data', 'read_restart', "
            "or 'create_atoms'"
        )
    if not result["has_run"] and not result["has_minimize"]:
        errors.append("No 'run' or 'minimize' — script does nothing")

    # ── Command ordering ──
    def _before(a: str, b: str, msg: str):
        if a in command_order and b in command_order:
            if command_order.index(a) > command_order.index(b):
                errors.append(msg)

    _before("units", "read_data", "'read_data' before 'units'")
    _before("pair_style", "pair_coeff", "'pair_coeff' before 'pair_style'")
    _before("pair_style", "pair_modify", "'pair_modify' before 'pair_style'")
    _before("bond_style", "bond_coeff", "'bond_coeff' before 'bond_style'")

    # A fully-typed force-field data file (e.g. from OpenFF Interchange) carries
    # inline Pair/Bond/Angle/Dihedral Coeffs. LAMMPS parses those sections AT
    # read_data time, so the matching *_style commands must be declared BEFORE
    # read_data — the reverse of the bare-data-file ordering. Omitting/ordering
    # them wrong aborts the run with e.g. "Must define pair_style before Pair
    # Coeffs". Only the styles the data file actually needs (by section/type
    # counts) are required.
    if (system_info and system_info.get("has_pair_coeffs")
            and "read_data" in command_order):
        rd = command_order.index("read_data")
        needed = ["pair_style"]
        if system_info.get("bond_types", 0):
            needed.append("bond_style")
        if system_info.get("angle_types", 0):
            needed.append("angle_style")
        if system_info.get("dihedral_types", 0):
            needed.append("dihedral_style")
        for style in needed:
            if style not in commands_seen or command_order.index(style) > rd:
                errors.append(
                    f"Data file has inline coefficients, so '{style}' must be "
                    f"declared BEFORE 'read_data' (LAMMPS parses the *Coeffs "
                    f"sections at read_data time, e.g. 'Must define pair_style "
                    f"before Pair Coeffs')."
                )

    # kspace_style does NOT parse data-file sections, so unlike the coeff styles
    # it must come AFTER read_data: a data file with an 'xy xz yz' line (any
    # OpenFF Interchange export) makes the box triclinic, and PPPM aborts with
    # "Must redefine kspace_style after changing to triclinic box" if declared
    # first. It never needs to precede read_data.
    if {"kspace_style", "read_data"} <= set(command_order):
        if command_order.index("kspace_style") < command_order.index("read_data"):
            errors.append(
                "'kspace_style' before 'read_data' — declare it AFTER read_data "
                "(PPPM must be set after the box is defined; a triclinic data "
                "file otherwise aborts with 'Must redefine kspace_style after "
                "changing to triclinic box')."
            )

    # Declared bond/angle/dihedral/improper styles must match the format the data
    # file's Coeffs are in (read_data aborts: "Incorrect args for <...>
    # coefficients"). required_styles is inferred from the data file by
    # parse_data_file; only flag a genuine mismatch (both sides known).
    required_styles = (system_info or {}).get("required_styles") or {}
    if required_styles:
        declared_styles: Dict[str, str] = {}
        for line in lines:
            p = line.strip().split("#", 1)[0].split()
            if len(p) >= 2 and p[0] in (
                "bond_style", "angle_style", "dihedral_style", "improper_style"
            ):
                declared_styles[p[0].split("_", 1)[0]] = p[1].lower()
        for kind, need in required_styles.items():
            have = declared_styles.get(kind)
            if have and have != need:
                errors.append(
                    f"{kind}_style is '{have}' but the data file's {kind} "
                    f"coefficients are in '{need}' format — declare "
                    f"'{kind}_style {need}' (read_data otherwise aborts with "
                    f"'Incorrect args for {kind} coefficients')."
                )

    # ── Forbidden combinations ──
    pair_style = result["pair_style"] or ""
    atom_style = result["atom_style"] or ""
    boundary = result["boundary"]
    units = result["units"]

    # kspace with many-body potential
    if pair_style in _NO_KSPACE_STYLES and "kspace_style" in commands_seen:
        errors.append(
            f"kspace_style with {pair_style} — "
            f"this potential has no Coulomb term"
        )

    # coul/long without kspace (check both direct and hybrid)
    if pair_style in _REQUIRES_KSPACE and "kspace_style" not in commands_seen:
        errors.append(f"{pair_style} requires kspace_style")
    elif "kspace_style" not in commands_seen:
        for line in lines:
            s = line.strip()
            if not s.startswith("#") and "pair_style" in s and "coul/long" in s:
                errors.append("pair_style uses coul/long but no kspace_style")
                break

    # bond_style / fix shake with atom_style atomic
    if atom_style == "atomic":
        if "bond_style" in commands_seen:
            errors.append("bond_style with atom_style atomic — no bonds")
        if result["has_shake"]:
            errors.append("fix shake with atom_style atomic — no bonds to constrain")

    # ReaxFF without qeq
    if "reaxff" in pair_style:
        has_qeq = any(
            "qeq" in l.lower()
            for l in lines
            if l.strip() and not l.strip().startswith("#")
        )
        if not has_qeq:
            errors.append("reaxff without fix qeq/reaxff — charge equilibration required")

    # NPT on non-periodic dimension
    if boundary:
        non_periodic = [i for i, b in enumerate(boundary) if b != "p"]
        if non_periodic:
            for line in lines:
                s = line.strip()
                if not s.startswith("fix") or s.startswith("#"):
                    continue
                lower = s.lower()
                if ("npt" in lower or "nph" in lower) and ("iso" in lower or "aniso" in lower):
                    dims = ["x", "y", "z"]
                    bad = [dims[i] for i in non_periodic]
                    warnings.append(
                        f"NPT iso/aniso with boundary {' '.join(boundary)} — "
                        f"barostat acts on non-periodic dim(s) {bad}"
                    )
                    break

    # nvt + npt on same group
    if integrator_conflict:
        errors.append(
            f"fix nvt and fix npt both active on group(s): {integrator_conflict}"
        )

    # ── Unit-aware parameter ranges ──
    ts = result["timestep"]
    _TS_RANGES = {
        "metal": (0.0001, 0.01,  "ps"),
        "real":  (0.1,    4.0,   "fs"),
        "lj":    (0.001,  0.02,  "τ"),
    }
    if ts is not None and units and units in _TS_RANGES:
        lo, hi, label = _TS_RANGES[units]
        if ts < lo or ts > hi:
            errors.append(
                f"Timestep {ts} outside sane range for 'units {units}' "
                f"({lo}–{hi} {label})"
            )

    if units == "real" and ts is not None and ts >= 2.0 and not result["has_shake"]:
        if atom_style in ("full", "molecular"):
            warnings.append(
                f"Timestep {ts} fs without SHAKE — may be unstable with flexible H bonds"
            )

    # Tdamp sanity
    _TDAMP_WARN = {
        "metal": (10.0, "too large for 'units metal' (expect 0.05–0.5 ps)"),
        "real":  (1.0,  "too small for 'units real' (expect 50–500 fs)"),  # threshold: <1 is suspicious
    }
    for line in lines:
        s = line.strip()
        if not s.startswith("fix") or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4 or parts[3].lower() not in ("nvt", "npt"):
            continue
        if "temp" in parts:
            try:
                idx = parts.index("temp")
                tdamp = float(parts[idx + 3])
                if units == "metal" and tdamp > _TDAMP_WARN["metal"][0]:
                    warnings.append(f"Tdamp={tdamp} {_TDAMP_WARN['metal'][1]}")
                elif units == "real" and tdamp < _TDAMP_WARN["real"][0]:
                    warnings.append(f"Tdamp={tdamp} {_TDAMP_WARN['real'][1]}")
            except (ValueError, IndexError):
                pass

    # ── Unresolved templates ──
    # Scan CODE only — comments legitimately mention ${var} (e.g. a note about a
    # variable) and must not be flagged. A ${name} whose `name` has a
    # `variable name …` definition in the deck is a valid LAMMPS variable
    # reference, not an unrendered template. A bare {name} is not LAMMPS syntax,
    # so it is always suspect (an unrendered template placeholder).
    code = "\n".join(ln.split("#", 1)[0] for ln in lines)
    defined_vars = set(re.findall(
        r"(?mi)^\s*variable\s+([A-Za-z_]\w*)\b", code))
    referenced = set(re.findall(r"\$\{([A-Za-z_]\w*)\}", code))
    braces = set(re.findall(r"(?<!\$)\{([A-Za-z_]\w*)\}", code))
    unresolved = sorted((referenced - defined_vars) | braces)
    if unresolved:
        errors.append(f"Unresolved template variables: {unresolved}")

    # ── Force field completeness ──
    if system_info:
        if (
            not system_info.get("has_pair_coeffs")
            and "pair_coeff" not in commands_seen
            and pair_style not in _NO_KSPACE_STYLES
        ):
            warnings.append("No Pair Coeffs in data file and no pair_coeff in script")

    # ── Potential file existence ──
    working_dir = Path(script_path).parent
    _POTENTIAL_EXTS = (
        ".eam", ".alloy", ".fs", ".meam", ".tersoff", ".sw",
        ".airebo", ".reax", ".comb", ".snap", ".table", ".poly",
    )
    for line in lines:
        s = line.strip()
        if s.startswith("#") or not s.startswith("pair_coeff"):
            continue
        parts = s.split()
        for part in parts[3:]:
            if "." in part and not part.replace(".", "").replace("-", "").replace("+", "").replace("e", "").isdigit():
                if any(ext in part.lower() for ext in _POTENTIAL_EXTS):
                    if not (working_dir / part).exists():
                        warnings.append(f"Potential file '{part}' not found in {working_dir}")

    result["valid"] = len(errors) == 0
    return result


# ─── Script Cleaning ─────────────────────────────────────────────────

def clean_script(text: str) -> str:
    """Remove markdown fences and LLM artifacts."""
    text = re.sub(r'```(?:lammps|bash|text)?', '', text)
    text = text.replace('```', '')
    return text.strip()


def substitute_variables(
    script: str,
    temperature: float = 300.0,
    pressure: float = 1.0,
    timestep: float = 2.0,
    data_filename: str = "system.data",
) -> str:
    """Replace common template placeholders with actual values."""
    subs = {
        "${temperature}": temperature, "{temperature}": temperature,
        "${temp}": temperature, "{temp}": temperature,
        "${t}": temperature, "${T}": temperature,
        "${pressure}": pressure, "{pressure}": pressure,
        "${press}": pressure, "{press}": pressure,
        "${p}": pressure, "${P}": pressure,
        "${timestep}": timestep, "{timestep}": timestep,
        "${dt}": timestep, "{dt}": timestep,
        "${data_file}": data_filename, "{data_file}": data_filename,
        "${data_filename}": data_filename, "{data_filename}": data_filename,
    }
    for pattern, value in subs.items():
        script = script.replace(pattern, str(value))
    return script


SWEEP_PLACEHOLDER = "__SWEEP__"


def sweep_member_name(var_name: str, value: Any) -> str:
    """Build a filesystem- and LAMMPS-safe member name for one sweep value.

    Example: ``("temperature", 300)`` → ``"temperature_300"``. Used to name a
    fan-out member's run directory and result record, so a temperature sweep or
    umbrella window set is self-describing.
    """
    var = re.sub(r"[^A-Za-z0-9]+", "", str(var_name)) or "sweep"
    val = re.sub(r"[^A-Za-z0-9.+-]+", "_", str(value)).strip("_") or "0"
    return f"{var}_{val}"


def expand_parameter_sweep(
    base_script: str,
    var_name: str,
    values: List[Any],
    placeholder: str = SWEEP_PLACEHOLDER,
) -> List[Dict[str, str]]:
    """Expand one parameterized base script into a fan-out of member scripts.

    The base script carries ``placeholder`` everywhere the swept quantity
    appears (e.g. a temperature in ``velocity create`` and ``fix nvt``, or a
    restraint center in ``fix spring``); each member is produced by replacing
    the placeholder with one value. This is the engine-idiomatic way to author
    a temperature sweep or the windows of an umbrella-sampling run: write the
    physics once, vary one number. Deterministic — no LLM call per member.

    Args:
        base_script: The parameterized script containing ``placeholder``.
        var_name: Name of the swept quantity (used to name members).
        values: The values to sweep over, one member each.
        placeholder: Token in ``base_script`` to replace with each value.

    Returns:
        A list of ``{"name", "script"}`` dicts, one per value, in order.
    """
    members: List[Dict[str, str]] = []
    for value in values:
        members.append({
            "name": sweep_member_name(var_name, value),
            "script": base_script.replace(placeholder, str(value)),
        })
    return members


# ─── Force Field Integration ─────────────────────────────────────────

_FF_STYLE_COMMANDS = {
    "units", "atom_style", "dimension", "boundary",
    "pair_style", "bond_style", "angle_style",
    "dihedral_style", "improper_style",
    "kspace_style", "special_bonds", "pair_modify",
}
_FF_COEFF_COMMANDS = {
    "pair_coeff", "bond_coeff", "angle_coeff",
    "dihedral_coeff", "improper_coeff", "set", "mass",
}


def integrate_force_field_files(
    script: str,
    force_field_files: Dict[str, str],
    working_dir: str,
) -> str:
    """
    Integrate force field files into a LAMMPS script.

    Style-only files → include BEFORE read_data.
    Coefficient files → inline AFTER read_data.
    """
    if not force_field_files:
        return script

    styles_files: List[str] = []
    coeff_lines: List[str] = []

    for name, ff_path in force_field_files.items():
        resolved = ff_path
        if not os.path.exists(resolved):
            local = os.path.join(working_dir, os.path.basename(resolved))
            if os.path.exists(local):
                resolved = local
            else:
                logger.warning(f"FF file not found: {ff_path}")
                continue

        try:
            ff_lines = Path(resolved).read_text().splitlines()
        except Exception as e:
            logger.warning(f"Cannot read {resolved}: {e}")
            continue

        has_styles = has_coeffs = False
        for line in ff_lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            kw = s.split()[0].lower()
            if kw in _FF_STYLE_COMMANDS:
                has_styles = True
            elif kw in _FF_COEFF_COMMANDS:
                has_coeffs = True

        if has_styles and not has_coeffs:
            dest = Path(working_dir) / os.path.basename(resolved)
            if str(Path(resolved).resolve()) != str(dest.resolve()):
                shutil.copy2(resolved, dest)
            styles_files.append(dest.name)
        elif has_coeffs:
            for line in ff_lines:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.split()[0].lower() in _FF_COEFF_COMMANDS:
                    coeff_lines.append(s)

    if not styles_files and not coeff_lines:
        return script

    lines = script.split("\n")

    if styles_files:
        lines = [
            l for l in lines
            if l.strip().startswith("#")
            or not l.strip()
            or l.strip().split()[0].lower() not in _FF_STYLE_COMMANDS
        ]

    read_data_pos = next(
        (i for i, l in enumerate(lines) if l.strip().startswith("read_data")),
        None,
    )
    if read_data_pos is None:
        return script

    new_lines: List[str] = []
    for i, line in enumerate(lines):
        if i == read_data_pos and styles_files:
            new_lines += ["", "# ── Force field styles ──"]
            new_lines += [f"include {f}" for f in styles_files]
            new_lines.append("")
        new_lines.append(line)
        if i == read_data_pos and coeff_lines:
            new_lines += ["", "# ── Force field parameters ──"]
            new_lines += coeff_lines
            new_lines.append("")

    return "\n".join(new_lines)


# ─── MLIP potential integration (engine-side) ──────────────────────
# How LAMMPS consumes a deployed MLIP. The MD agent calls
# run_with_potential() generically via TOOL_REGISTRY — this is where
# the *LAMMPS-specific* knowledge of MLIP pair_styles lives, so adding
# an MLIP backend that has a LAMMPS pair_style means one entry here,
# and an MLIP backend that doesn't (CHGNet) means nothing here at all.

def _mace_pair(model_file: str, el_str: str):
    return (
        "pair_style     mace no_domain_decomposition",
        f"pair_coeff     * * {model_file} {el_str}",
    )


def _nequip_pair(model_file: str, el_str: str):
    return (
        "pair_style     nequip",
        f"pair_coeff     * * {model_file} {el_str}",
    )


def _deepmd_pair(model_file: str, el_str: str):
    return (
        f"pair_style     deepmd {model_file}",
        f"pair_coeff     * * {el_str}",
    )


# backend keyword -> (pair_style, pair_coeff) builder. A backend absent
# from this map has no LAMMPS pair_style (e.g. CHGNet is ASE-only);
# run_with_potential raises NotImplementedError so the MD agent can
# fall back to its universal ASE runner.
_MLIP_PAIR_BUILDERS = {
    "mace":   _mace_pair,
    "nequip": _nequip_pair,
    "deepmd": _deepmd_pair,
}


def supported_mlip_backends() -> list:
    """MLIP backend keywords LAMMPS can run via a pair_style."""
    return sorted(_MLIP_PAIR_BUILDERS)


def run_with_potential(
    potential,
    structure_file: str,
    working_dir: str,
    task: str = "md",
    timestep: float = 0.5,
    temperature: float = 300.0,
    pressure=None,
    n_steps: int = 100000,
) -> str:
    """
    Generate a LAMMPS input file that runs a deployed MLIP potential.

    This is the LAMMPS engine's side of the potential/runner split:
    MDSimulationAgent hands over a ``DeployedPotential`` (duck-typed
    here — only ``.backend``, ``.model_file``, ``.elements`` are read)
    and this function emits the LAMMPS input. The MD agent calls it
    generically through ``TOOL_REGISTRY["lammps"].run_with_potential``;
    it does not know anything LAMMPS-specific.

    Parameters
    ----------
    potential:
        A DeployedPotential. Its ``.backend`` selects the pair_style.
    structure_file:
        Structure the generated input reads (``read_data``); the file
        basename is used in the input.
    task:
        ``"md"`` — minimize then NVT/NPT dynamics for ``n_steps``.
        ``"relax"`` — ``fix box/relax`` + ``minimize`` only (cell +
        geometry optimization, no dynamics).
    pressure:
        ``None`` → NVT for MD; a value → NPT. Ignored for ``relax``.

    Returns
    -------
    Absolute path to the written ``in.lammps``.

    Raises
    ------
    NotImplementedError
        If ``potential.backend`` has no LAMMPS pair_style (the caller
        should fall back to the universal ASE runner).
    ValueError
        For an unknown ``task`` or a potential with no model file.
    """
    if task not in ("md", "relax"):
        raise ValueError(f"task must be 'md' or 'relax', got {task!r}")

    backend = getattr(potential, "backend", None)
    builder = _MLIP_PAIR_BUILDERS.get(backend)
    if builder is None:
        raise NotImplementedError(
            f"LAMMPS has no pair_style for MLIP backend {backend!r}. "
            f"Supported: {supported_mlip_backends()}. Use the ASE runner."
        )

    model_file = getattr(potential, "model_file", "") or ""
    if not model_file:
        raise ValueError(
            f"backend {backend!r} has no on-disk model file — it cannot "
            f"run via LAMMPS. Use the ASE runner."
        )

    elements = list(getattr(potential, "elements", []) or [])
    el_str = " ".join(elements)
    pair_style, pair_coeff = builder(model_file, el_str)
    data_name = os.path.basename(structure_file)

    head = (
        f"# LAMMPS input -- {backend} MLIP potential ({task})\n"
        f"# Model: {os.path.basename(model_file)}\n"
        f"# Elements: {el_str}\n\n"
        "units          metal\n"
        "atom_style     atomic\n"
        "boundary       p p p\n\n"
        f"read_data      {data_name}\n\n"
        f"{pair_style}\n"
        f"{pair_coeff}\n\n"
        "neighbor       2.0 bin\n"
        "neigh_modify   every 1 delay 0 check yes\n\n"
        "thermo         100\n"
        "thermo_style   custom step temp press pe ke etotal vol density\n"
    )

    if task == "relax":
        body = (
            "\n# cell + geometry relaxation\n"
            "fix            1 all box/relax iso 0.0 vmax 0.001\n"
            "min_style      cg\n"
            "minimize       1.0e-8 1.0e-10 10000 100000\n"
            "write_data     relaxed.data\n"
        )
    else:  # md
        ensemble_fix = (
            f"fix            1 all npt temp {temperature} {temperature} 0.1 "
            f"iso {pressure} {pressure} 1.0"
            if pressure is not None
            else f"fix            1 all nvt temp {temperature} {temperature} 0.1"
        )
        body = (
            "\ndump           traj all custom 1000 traj.lammpstrj "
            "id type x y z fx fy fz\n\n"
            "min_style      cg\n"
            "minimize       1.0e-6 1.0e-8 1000 10000\n\n"
            f"velocity       all create {temperature} 12345 mom yes rot yes\n"
            f"timestep       {timestep}e-3\n\n"
            f"{ensemble_fix}\n"
            f"run            {n_steps}\n"
        )

    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, "in.lammps")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + body)
    return path


# ─── Tool specs (resolved via get_tool_function) ─────────────────────

# --- observable-coverage detection (engine realization of the contradiction
#     framework's signal_present / cadence check kinds) ------------------------

# Semantic signal name -> LAMMPS thermo keywords that provide it.
_THERMO_KEYWORDS: Dict[str, set] = {
    "energy": {"etotal", "pe", "ke", "epair", "emol", "evdwl", "ecoul", "elong"},
    "temperature": {"temp"},
    "pressure": {"press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"},
    "stress": {"press", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"},
    "volume": {"vol"},
    "density": {"density", "vol"},
}
# Signals recoverable from a saved trajectory (a dump).
_TRAJECTORY_SIGNALS = {"trajectory", "positions", "coordinates", "traj",
                       "rdf", "structure", "msd"}
# LAMMPS default thermo_style ('one') keyword set.
_THERMO_ONE = {"step", "temp", "epair", "emol", "etotal", "press"}


def _thermo_interval(deck: str) -> Optional[int]:
    hits = re.findall(r'(?mi)^\s*thermo\s+(\d+)\b', deck)
    return int(hits[-1]) if hits else None


def _thermo_keywords(deck: str) -> set:
    kws: set = set()
    styles = re.findall(r'(?mi)^\s*thermo_style\s+(.*)$', deck)
    for line in styles:
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() == "custom":
            kws |= {p.lower() for p in parts[1:]}
        elif parts[0].lower() in ("one", "multi"):
            kws |= _THERMO_ONE
    if not styles:  # no explicit style -> LAMMPS default is 'one'
        kws |= _THERMO_ONE
    return kws


def _has_dump(deck: str) -> bool:
    """Any dump command — presence of a saved trajectory, interval or not."""
    return bool(re.search(r'(?mi)^\s*dump\s+\S+\s+\S+\s+\S+\s+\S+', deck))


def _dump_interval(deck: str) -> Optional[int]:
    """Coarsest dump interval, or None when only variable (``${...}``) intervals
    are used (present but unparseable)."""
    ivs = [int(n) for n in
           re.findall(r'(?mi)^\s*dump\s+\S+\s+\S+\s+\S+\s+(\d+)\b', deck)]
    return min(ivs) if ivs else None


def _ave_output_interval(deck: str) -> Optional[int]:
    """Smallest sampling cadence across averaging fixes, or None if unparseable.

    For ``ave/time`` the raw write interval is Nfreq (group 3). For
    ``ave/correlate`` the governing cadence is **Nevery** (group 1) — the fix
    samples the quantity internally at that rate, which is what sets Green-Kubo
    adequacy — not the (typically large) correlation-output interval Nfreq.
    """
    intervals = []
    for m in re.finditer(
            r'(?mi)^\s*fix\s+\S+\s+\S+\s+ave/(time|correlate)\s+(\d+)\s+(\d+)\s+(\d+)\b',
            deck):
        kind, nevery, _nrepeat, nfreq = (m.group(1).lower(), int(m.group(2)),
                                         int(m.group(3)), int(m.group(4)))
        intervals.append(nevery if kind == "correlate" else nfreq)
    return min(intervals) if intervals else None


def _has_pressure_ave(deck: str) -> bool:
    """Pressure/stress accumulated during the run: a pressure or per-atom stress
    compute together with an averaging fix that writes it (interval or not)."""
    has_compute = bool(re.search(
        r'(?mi)^\s*compute\s+\S+\s+\S+\s+(?:pressure|stress/atom|centroid/stress/atom)\b',
        deck))
    has_ave = bool(re.search(
        r'(?mi)^\s*fix\s+\S+\s+\S+\s+ave/(?:time|correlate)\b', deck))
    return has_compute and has_ave


def detect_signal_logging(deck_text: str, signal: str) -> Dict[str, Any]:
    """Report whether a LAMMPS deck logs ``signal`` and at what interval.

    Engine realization of the ``signal_present`` / ``cadence`` contradiction
    checks. Maps a semantic signal name to LAMMPS logging constructs — a saved
    trajectory (``dump``) for geometry/structure signals, ``thermo`` keywords
    for thermodynamic ones, and ``fix ave/time`` / ``ave/correlate`` output for
    quantities accumulated during the run (e.g. stress for Green-Kubo
    viscosity).

    Args:
        deck_text: The full LAMMPS input deck.
        signal: Semantic signal name (e.g. ``"stress"``, ``"trajectory"``,
            ``"energy"``, ``"temperature"``, ``"density"``).

    Returns:
        ``{"present": bool, "interval_steps": int | None}``. ``interval_steps``
        is the coarsest cadence at which the signal is written (``None`` when
        not logged or when the interval cannot be determined).
    """
    signal = str(signal).lower().strip()

    # Presence is decided by the logging COMMAND existing, independent of whether
    # its interval is a literal integer — decks commonly use variables
    # (`thermo ${freq}`, `dump 1 all custom ${n} ...`), which log the signal but
    # have an unparseable cadence. interval_steps is then None (cadence defers).
    if signal in _TRAJECTORY_SIGNALS:
        if not _has_dump(deck_text):
            return {"present": False, "interval_steps": None}
        return {"present": True, "interval_steps": _dump_interval(deck_text)}

    thermo_kws = _thermo_keywords(deck_text)
    wanted = _THERMO_KEYWORDS.get(signal, {signal})

    present = False
    intervals: List[int] = []
    if wanted & thermo_kws:
        present = True
        ti = _thermo_interval(deck_text)
        if ti is not None:
            intervals.append(ti)
    # stress/pressure accumulated during the run via a compute + averaging fix.
    if signal in ("stress", "pressure") and _has_pressure_ave(deck_text):
        present = True
        ai = _ave_output_interval(deck_text)
        if ai is not None:
            intervals.append(ai)

    if present:
        return {"present": True,
                "interval_steps": min(intervals) if intervals else None}
    return {"present": False, "interval_steps": None}


TOOL_SPECS = [
    ToolSpec(
        name="detect_signal_logging",
        description=(
            "Report whether a LAMMPS deck logs a named signal (trajectory, "
            "energy, temperature, pressure/stress, volume, density) and at what "
            "step interval. Engine realization of the signal_present / cadence "
            "observable-coverage checks."
        ),
        parameters={
            "deck_text": "The full LAMMPS input deck (string).",
            "signal": (
                "Semantic signal name to look for, e.g. 'stress', 'trajectory', "
                "'energy', 'temperature', 'density'."
            ),
        },
        agents=["simulation"],
    ),
]


TOOL_SPEC = ToolSpec(
    name="default_run_command",
    description=(
        "Return the conventional local LAMMPS run-command template "
        "('<lmp> -in {script}') for the first LAMMPS binary on PATH, or None if "
        "none is found. Lets a one-shot workflow execute LAMMPS with no engine "
        "name or launch command hardcoded in shared code."
    ),
    parameters={},
    agents=["simulation"],
)
