"""
AmberTools wrappers for the ForceFieldAgent AMBER pipeline.

This module provides Python functions that wrap the CLI tools:
  - pdb4amber  (PDB cleaning)
  - antechamber (atom typing + charges)
  - parmchk2    (missing parameter estimation)
  - tleap       (topology building)
  - ParmEd      (AMBER → LAMMPS conversion)

These are called by ForceFieldAgent when the AMBER skill is active.
They are intentionally decoupled from the skill (which provides LLM
context) so that the tools can be tested and used independently.
"""

import os
import shutil
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Tool Availability ───────────────────────────────────────────────

REQUIRED_TOOLS = ("antechamber", "tleap", "parmchk2")
OPTIONAL_TOOLS = ("sqm", "pdb4amber", "reduce", "cpptraj")


def check_amber_tools() -> Dict[str, Any]:
    """
    Check availability of all AmberTools executables and ParmEd.

    Returns:
        {"available": bool, "missing": [...], "tools": {...}, "parmed": {...}}
    """
    result: Dict[str, Any] = {
        "available": False,
        "missing": [],
        "tools": {},
        "parmed": {"available": False, "version": None},
    }

    for tool in REQUIRED_TOOLS + OPTIONAL_TOOLS:
        path = shutil.which(tool)
        result["tools"][tool] = {"found": path is not None, "path": path}
        if path is None and tool in REQUIRED_TOOLS:
            result["missing"].append(tool)

    try:
        import parmed
        result["parmed"]["available"] = True
        result["parmed"]["version"] = getattr(parmed, "__version__", "unknown")
    except ImportError:
        result["missing"].append("parmed")

    result["available"] = len(result["missing"]) == 0
    return result


# ─── PDB Cleaning ────────────────────────────────────────────────────

def run_pdb4amber(
    pdb_file: str,
    working_dir: str,
    remove_water: bool = False,
    remove_hydrogens: bool = False,
    output_file: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """
    Clean a PDB file for AMBER processing.

    Returns:
        Path to the cleaned PDB file (original if pdb4amber not available).
    """
    if output_file is None:
        output_file = os.path.join(working_dir, "cleaned.pdb")

    if not shutil.which("pdb4amber"):
        logger.info("pdb4amber not found; skipping PDB cleanup")
        return pdb_file

    cmd = ["pdb4amber", "-i", pdb_file, "-o", output_file]
    if remove_water:
        cmd.append("--dry")
    if remove_hydrogens:
        cmd.append("--nohyd")

    logger.info(f"pdb4amber: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=working_dir, timeout=timeout,
        )
        if proc.returncode != 0:
            logger.warning(f"pdb4amber rc={proc.returncode}: {proc.stderr[:300]}")

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return output_file
        return pdb_file

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"pdb4amber failed: {e}; using original PDB")
        return pdb_file


# ─── Antechamber ─────────────────────────────────────────────────────

_FORMAT_MAP = {
    ".pdb": "pdb", ".mol2": "mol2", ".sdf": "sdf",
    ".mol": "mdl", ".cif": "cif",
}


def run_antechamber(
    input_file: str,
    working_dir: str,
    net_charge: int = 0,
    charge_method: str = "bcc",
    atom_type: str = "gaff2",
    multiplicity: int = 1,
    output_prefix: Optional[str] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Run antechamber for atom typing and charge assignment.

    Returns:
        {"mol2": <path>, "atom_type": ..., "charge_method": ..., "net_charge": ...}
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_file))[0]

    ext = os.path.splitext(input_file)[1].lower()
    input_fmt = _FORMAT_MAP.get(ext, "pdb")
    output_mol2 = os.path.join(working_dir, f"{output_prefix}.mol2")

    cmd = [
        "antechamber",
        "-i", input_file,   "-fi", input_fmt,
        "-o", output_mol2,  "-fo", "mol2",
        "-c", charge_method,
        "-s", "2",
        "-at", atom_type,
        "-nc", str(net_charge),
        "-m", str(multiplicity),
        "-pf", "y",
    ]

    logger.info(f"antechamber: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=working_dir, timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "antechamber not found. Install: conda install -c conda-forge ambertools"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"antechamber timed out ({timeout}s). Try charge_method='gas'."
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"antechamber failed (rc={proc.returncode}):\n{proc.stderr[-500:]}"
        )

    if not os.path.exists(output_mol2):
        raise FileNotFoundError(f"antechamber did not produce {output_mol2}")

    logger.info(f"antechamber → {output_mol2}")
    return {
        "mol2": output_mol2,
        "atom_type": atom_type,
        "charge_method": charge_method,
        "net_charge": net_charge,
    }


# ─── Parmchk2 ────────────────────────────────────────────────────────

def run_parmchk2(
    mol2_file: str,
    working_dir: str,
    atom_type: str = "gaff2",
    output_prefix: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Run parmchk2 to estimate missing parameters.

    Returns:
        Path to the .frcmod file.
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(mol2_file))[0]

    frcmod_file = os.path.join(working_dir, f"{output_prefix}.frcmod")
    s_flag = {"gaff": "1", "gaff2": "2"}.get(atom_type, "2")

    cmd = [
        "parmchk2",
        "-i", mol2_file, "-f", "mol2",
        "-o", frcmod_file, "-s", s_flag,
    ]

    logger.info(f"parmchk2: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=working_dir, timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "parmchk2 not found. Install: conda install -c conda-forge ambertools"
        )

    if proc.returncode != 0:
        raise RuntimeError(f"parmchk2 failed (rc={proc.returncode}): {proc.stderr[-300:]}")

    if not os.path.exists(frcmod_file):
        raise FileNotFoundError(f"parmchk2 did not produce {frcmod_file}")

    # Check for ATTN markers
    with open(frcmod_file) as f:
        content = f.read()
    n_attn = content.count("ATTN")
    if n_attn > 0:
        logger.warning(f"parmchk2: {n_attn} ATTN-marked parameters (estimated by analogy)")

    logger.info(f"parmchk2 → {frcmod_file}")
    return frcmod_file


# ─── tleap ────────────────────────────────────────────────────────────

# Lookup tables for leaprc resolution
PROTEIN_FF_LEAPRC = {
    "ff14sb": "leaprc.protein.ff14SB",
    "ff19sb": "leaprc.protein.ff19SB",
    "ff99sb": "leaprc.protein.ff99SB",
    "fb15":   "leaprc.protein.fb15",
}

WATER_LEAPRC = {
    "tip3p":   "leaprc.water.tip3p",
    "spc/e":   "leaprc.water.spce",
    "spce":    "leaprc.water.spce",
    "opc":     "leaprc.water.opc",
    "opc3":    "leaprc.water.opc3",
    "tip4pew": "leaprc.water.tip4pew",
}

WATER_BOX = {
    "tip3p":   "TIP3PBOX",
    "spc/e":   "SPCBOX",
    "spce":    "SPCBOX",
    "opc":     "OPCBOX",
    "opc3":    "OPC3BOX",
    "tip4pew": "TIP4PEWBOX",
}


def generate_tleap_script(
    pdb_file: str,
    working_dir: str,
    composition: Dict[str, bool],
    mol2_files: Optional[List[Dict[str, Any]]] = None,
    frcmod_files: Optional[List[str]] = None,
    protein_ff: str = "ff19SB",
    water_model: str = "tip3p",
    gaff_version: str = "gaff2",
    solvate: bool = False,
    box_buffer: float = 10.0,
    neutralize: bool = True,
    ion_concentration: float = 0.0,
    output_prefix: str = "system",
) -> str:
    """Generate a tleap input script. All paths are made absolute."""
    
    # ── Make all paths absolute ──
    working_dir = os.path.abspath(working_dir)
    pdb_file = os.path.abspath(pdb_file)
    
    lines = ["# tleap script — generated by ForceFieldAgent (AMBER skill)", ""]

    # ── Always source protein FF if proteins or caps are present ──
    # ACE, NME, NHE are capping groups that need the protein leaprc
    needs_protein_ff = composition.get("proteins", False)
    
    # Also check: if it's a PDB with standard residues, we need protein FF
    # even if composition detection missed caps-only systems
    if not needs_protein_ff:
        try:
            with open(pdb_file) as f:
                cap_residues = {'ACE', 'NME', 'NHE'}
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        resname = line[17:20].strip()
                        if resname in cap_residues:
                            needs_protein_ff = True
                            logger.info(f"Detected cap residue '{resname}' — enabling protein FF")
                            break
        except Exception:
            pass

    if composition.get("small_molecules") or mol2_files:
        lines.append(f"source leaprc.{gaff_version}")

    if needs_protein_ff:
        key = protein_ff.lower().replace("amber", "").replace(" ", "")
        lines.append(f"source {PROTEIN_FF_LEAPRC.get(key, f'leaprc.protein.{protein_ff}')}")

    if composition.get("nucleic_acids"):
        lines += ["source leaprc.DNA.OL15", "source leaprc.RNA.OL3"]

    if composition.get("lipids"):
        lines.append("source leaprc.lipid21")

    if composition.get("carbohydrates"):
        lines.append("source leaprc.GLYCAM_06j-1")

    wm_key = water_model.lower().replace(" ", "")
    lines.append(f"source {WATER_LEAPRC.get(wm_key, f'leaprc.water.{wm_key}')}")
    lines.append("")

    # Load small molecule params (absolute paths)
    if mol2_files:
        for info in mol2_files:
            name = info.get("name", "MOL")
            mol2_path = os.path.abspath(info["mol2"])
            lines.append(f"{name} = loadmol2 {mol2_path}")
    if frcmod_files:
        for frc in frcmod_files:
            lines.append(f"loadamberparams {os.path.abspath(frc)}")
    if mol2_files or frcmod_files:
        lines.append("")

    # Load structure (absolute path)
    lines += [f"SYS = loadpdb {pdb_file}", ""]

    # Solvation
    if solvate:
        box = WATER_BOX.get(wm_key, "TIP3PBOX")
        lines += [f"solvatebox SYS {box} {box_buffer:.1f}", ""]

    # Ions
    if neutralize:
        lines.append("addIonsRand SYS Na+ 0")
        lines.append("addIonsRand SYS Cl- 0")
        lines.append("")

    # Save (absolute paths)
    prmtop = os.path.join(working_dir, f"{output_prefix}.prmtop")
    inpcrd = os.path.join(working_dir, f"{output_prefix}.inpcrd")
    lines += [
        "check SYS",
        f"saveamberparm SYS {prmtop} {inpcrd}",
        f"savepdb SYS {os.path.join(working_dir, f'{output_prefix}_tleap.pdb')}",
        "quit",
    ]

    script_path = os.path.join(working_dir, f"{output_prefix}_tleap.in")
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"tleap script → {script_path}")
    return script_path

def run_tleap(
    script_file: str,
    working_dir: str,
    timeout: int = 300,
) -> Tuple[str, str]:
    """
    Execute tleap.

    Returns:
        (prmtop_path, inpcrd_path)
    """
    cmd = ["tleap", "-f", script_file]
    logger.info(f"tleap: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=working_dir, timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "tleap not found. Install: conda install -c conda-forge ambertools"
        )

    # Check log for FATAL errors
    log_path = os.path.join(working_dir, "leap.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_text = f.read()
        fatals = [l.strip() for l in log_text.splitlines() if "FATAL" in l]
        if fatals:
            raise RuntimeError(f"tleap FATAL: {fatals[0]}\nSee {log_path}")
        for w in [l.strip() for l in log_text.splitlines() if "WARNING" in l.upper()][:15]:
            logger.warning(f"tleap: {w}")

    # Extract output paths from script
    prmtop, inpcrd = None, None
    with open(script_file) as f:
        for line in f:
            if line.strip().lower().startswith("saveamberparm"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    prmtop, inpcrd = parts[2], parts[3]

    if prmtop and inpcrd and os.path.exists(prmtop) and os.path.exists(inpcrd):
        logger.info(f"tleap → {prmtop}, {inpcrd}")
        return prmtop, inpcrd

    raise FileNotFoundError(f"tleap output missing. Check {log_path}")


# ─── ParmEd Conversion ───────────────────────────────────────────────

def convert_amber_to_lammps(
    prmtop: str,
    inpcrd: str,
    output_data: Optional[str] = None,
) -> str:
    """
    Convert AMBER prmtop/inpcrd to LAMMPS data file.

    Uses ParmEd to read the AMBER files, then writes LAMMPS format directly
    since ParmEd's save() doesn't support LAMMPS output in all versions.

    Returns:
        Path to the LAMMPS data file.
    """
    if output_data is None:
        output_data = os.path.splitext(prmtop)[0] + ".data"

    try:
        import parmed as pmd
    except ImportError:
        raise ImportError(
            "ParmEd required for AMBER→LAMMPS conversion.\n"
            "Install: pip install parmed  OR  conda install -c conda-forge parmed"
        )

    logger.info(f"ParmEd: {prmtop} + {inpcrd} → {output_data}")
    system = pmd.load_file(prmtop, xyz=inpcrd)

    logger.info(
        f"ParmEd loaded: {len(system.atoms)} atoms, "
        f"{len(system.residues)} residues, {len(system.bonds)} bonds"
    )

    _write_lammps_data(system, output_data)

    if not os.path.exists(output_data):
        raise FileNotFoundError(f"ParmEd did not create {output_data}")

    size_kb = os.path.getsize(output_data) / 1024
    logger.info(f"LAMMPS data file written ({size_kb:.1f} kB): {output_data}")
    return output_data


def _write_lammps_data(system, output_path: str):
    """
    Write a ParmEd Structure to LAMMPS data file format (atom_style full).

    Writes all sections: header, masses, pair coeffs, bond/angle/dihedral/improper
    coeffs, atoms (with charges), bonds, angles, dihedrals, impropers.
    """
    import parmed as pmd
    import math

    atoms = system.atoms
    bonds = system.bonds
    angles = system.angles
    dihedrals = system.dihedrals
    impropers = system.impropers

    # ── Build type maps ──────────────────────────────────────────
    # Atom types
    atom_type_map = {}  # pmd atom type → integer index
    atom_type_list = []  # ordered list of unique atom types
    for atom in atoms:
        key = (atom.type, round(atom.mass, 4))
        if key not in atom_type_map:
            atom_type_map[key] = len(atom_type_list) + 1
            atom_type_list.append(atom)

    # Bond types
    bond_type_map = {}
    bond_type_list = []
    for bond in bonds:
        bt = bond.type
        if bt is None:
            continue
        key = id(bt)
        if key not in bond_type_map:
            bond_type_map[key] = len(bond_type_list) + 1
            bond_type_list.append(bt)

    # Angle types
    angle_type_map = {}
    angle_type_list = []
    for angle in angles:
        at = angle.type
        if at is None:
            continue
        key = id(at)
        if key not in angle_type_map:
            angle_type_map[key] = len(angle_type_list) + 1
            angle_type_list.append(at)

    # Dihedral types
    dihedral_type_map = {}
    dihedral_type_list = []
    for dih in dihedrals:
        dt = dih.type
        if dt is None:
            continue
        key = id(dt)
        if key not in dihedral_type_map:
            dihedral_type_map[key] = len(dihedral_type_list) + 1
            dihedral_type_list.append(dt)

    # Improper types
    improper_type_map = {}
    improper_type_list = []
    for imp in impropers:
        it = imp.type
        if it is None:
            continue
        key = id(it)
        if key not in improper_type_map:
            improper_type_map[key] = len(improper_type_list) + 1
            improper_type_list.append(it)

    # ── Assign molecule IDs ──────────────────────────────────────
    # Use residue index as molecule ID
    mol_ids = {}
    for i, atom in enumerate(atoms):
        mol_ids[i] = atom.residue.idx + 1 if atom.residue else 1

    # ── Box dimensions ───────────────────────────────────────────
    box = system.box
    if box is not None:
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi = box[0]
        yhi = box[1]
        zhi = box[2]
    else:
        # No box info — compute from coordinates with padding
        coords = system.coordinates
        if coords is not None and len(coords) > 0:
            xlo = coords[:, 0].min() - 5.0
            xhi = coords[:, 0].max() + 5.0
            ylo = coords[:, 1].min() - 5.0
            yhi = coords[:, 1].max() + 5.0
            zlo = coords[:, 2].min() - 5.0
            zhi = coords[:, 2].max() + 5.0
        else:
            xlo, ylo, zlo = 0.0, 0.0, 0.0
            xhi, yhi, zhi = 50.0, 50.0, 50.0

    # ── Filter valid bonds/angles/dihedrals/impropers ────────────
    valid_bonds = [b for b in bonds if b.type is not None]
    valid_angles = [a for a in angles if a.type is not None]
    valid_dihedrals = [d for d in dihedrals if d.type is not None]
    valid_impropers = [i for i in impropers if i.type is not None]

    # ── Write the file ───────────────────────────────────────────
    with open(output_path, "w") as f:
        # Header
        f.write("LAMMPS data file via ParmEd/SciLink\n\n")
        f.write(f"{len(atoms)} atoms\n")
        if valid_bonds:
            f.write(f"{len(valid_bonds)} bonds\n")
        if valid_angles:
            f.write(f"{len(valid_angles)} angles\n")
        if valid_dihedrals:
            f.write(f"{len(valid_dihedrals)} dihedrals\n")
        if valid_impropers:
            f.write(f"{len(valid_impropers)} impropers\n")
        f.write("\n")

        f.write(f"{len(atom_type_list)} atom types\n")
        if bond_type_list:
            f.write(f"{len(bond_type_list)} bond types\n")
        if angle_type_list:
            f.write(f"{len(angle_type_list)} angle types\n")
        if dihedral_type_list:
            f.write(f"{len(dihedral_type_list)} dihedral types\n")
        if improper_type_list:
            f.write(f"{len(improper_type_list)} improper types\n")
        f.write("\n")

        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")
        f.write("\n")

        # Masses
        f.write("Masses\n\n")
        for i, atom in enumerate(atom_type_list):
            f.write(f"{i + 1} {atom.mass:.4f} # {atom.type}\n")
        f.write("\n")

        # Pair Coeffs (LJ parameters)
        f.write("Pair Coeffs # lj/charmm/coul/long\n\n")
        for i, atom in enumerate(atom_type_list):
            atype = atom.atom_type
            if atype is not None:
                epsilon = atype.epsilon  # kcal/mol
                rmin = atype.rmin        # Angstroms (Rmin/2)
                # LAMMPS uses sigma, not Rmin/2
                # sigma = Rmin/2 * 2 / 2^(1/6)
                if rmin is not None and rmin > 0:
                    sigma = rmin * 2.0 / (2.0 ** (1.0 / 6.0))
                else:
                    sigma = 0.0
                if epsilon is None:
                    epsilon = 0.0
            else:
                epsilon = 0.0
                sigma = 0.0
            f.write(f"{i + 1} {epsilon:.6f} {sigma:.6f} # {atom.type}\n")
        f.write("\n")

        # Bond Coeffs
        if bond_type_list:
            f.write("Bond Coeffs # harmonic\n\n")
            for i, bt in enumerate(bond_type_list):
                k = bt.k    # kcal/mol/A^2
                req = bt.req  # Angstroms
                f.write(f"{i + 1} {k:.4f} {req:.4f}\n")
            f.write("\n")

        # Angle Coeffs
        if angle_type_list:
            f.write("Angle Coeffs # harmonic\n\n")
            for i, at in enumerate(angle_type_list):
                k = at.k           # kcal/mol/rad^2
                theteq = at.theteq  # degrees
                f.write(f"{i + 1} {k:.4f} {theteq:.4f}\n")
            f.write("\n")

        # Dihedral Coeffs
        if dihedral_type_list:
            f.write("Dihedral Coeffs # fourier\n\n")
            for i, dt in enumerate(dihedral_type_list):
                # AMBER dihedrals can be multi-term (DihedralTypeList)
                if isinstance(dt, pmd.DihedralType):
                    terms = [dt]
                elif hasattr(dt, '__iter__'):
                    terms = list(dt)
                else:
                    terms = [dt]

                # Fourier style: N K1 d1 n1 K2 d2 n2 ...
                n_terms = len(terms)
                parts = [str(n_terms)]
                for term in terms:
                    phi_k = term.phi_k   # kcal/mol
                    per = term.per       # periodicity
                    phase = term.phase   # degrees
                    # Fourier format: K d n
                    # d = 1 if phase == 0, d = -1 if phase == 180
                    d = 1 if abs(phase) < 90 else -1
                    parts.extend([f"{phi_k:.6f}", str(d), str(int(per))])
                f.write(f"{i + 1} {' '.join(parts)}\n")
            f.write("\n")

        # Improper Coeffs
        if improper_type_list:
            f.write("Improper Coeffs # cvff\n\n")
            for i, it in enumerate(improper_type_list):
                psi_k = it.psi_k   # kcal/mol
                psi_eq = it.psi_eq  # degrees
                # cvff style: K d n
                d = 1 if abs(psi_eq) < 90 else -1
                n = int(getattr(it, 'per', 2))
                f.write(f"{i + 1} {psi_k:.6f} {d} {n}\n")
            f.write("\n")

        # Atoms section
        f.write("Atoms # full\n\n")
        coords = system.coordinates
        for i, atom in enumerate(atoms):
            atom_id = i + 1
            mol_id = mol_ids[i]
            key = (atom.type, round(atom.mass, 4))
            type_id = atom_type_map[key]
            charge = atom.charge
            x, y, z = coords[i]
            f.write(
                f"{atom_id} {mol_id} {type_id} {charge:.6f} "
                f"{x:.6f} {y:.6f} {z:.6f} # {atom.type} {atom.name}\n"
            )
        f.write("\n")

        # Bonds
        if valid_bonds:
            f.write("Bonds\n\n")
            for i, bond in enumerate(valid_bonds):
                bond_id = i + 1
                type_id = bond_type_map[id(bond.type)]
                a1 = bond.atom1.idx + 1
                a2 = bond.atom2.idx + 1
                f.write(f"{bond_id} {type_id} {a1} {a2}\n")
            f.write("\n")

        # Angles
        if valid_angles:
            f.write("Angles\n\n")
            for i, angle in enumerate(valid_angles):
                angle_id = i + 1
                type_id = angle_type_map[id(angle.type)]
                a1 = angle.atom1.idx + 1
                a2 = angle.atom2.idx + 1
                a3 = angle.atom3.idx + 1
                f.write(f"{angle_id} {type_id} {a1} {a2} {a3}\n")
            f.write("\n")

        # Dihedrals
        if valid_dihedrals:
            f.write("Dihedrals\n\n")
            for i, dih in enumerate(valid_dihedrals):
                dih_id = i + 1
                type_id = dihedral_type_map[id(dih.type)]
                a1 = dih.atom1.idx + 1
                a2 = dih.atom2.idx + 1
                a3 = dih.atom3.idx + 1
                a4 = dih.atom4.idx + 1
                f.write(f"{dih_id} {type_id} {a1} {a2} {a3} {a4}\n")
            f.write("\n")

        # Impropers
        if valid_impropers:
            f.write("Impropers\n\n")
            for i, imp in enumerate(valid_impropers):
                imp_id = i + 1
                type_id = improper_type_map[id(imp.type)]
                a1 = imp.atom1.idx + 1
                a2 = imp.atom2.idx + 1
                a3 = imp.atom3.idx + 1
                a4 = imp.atom4.idx + 1
                f.write(f"{imp_id} {type_id} {a1} {a2} {a3} {a4}\n")
            f.write("\n")

    logger.info(
        f"Wrote LAMMPS data: {len(atoms)} atoms, {len(atom_type_list)} types, "
        f"{len(valid_bonds)} bonds, {len(valid_angles)} angles, "
        f"{len(valid_dihedrals)} dihedrals, {len(valid_impropers)} impropers"
    )

# ─── Validation ───────────────────────────────────────────────────────

def validate_amber_data_file(data_file: str) -> Dict[str, Any]:
    """
    Validate a LAMMPS data file produced by the AMBER pipeline.

    Returns:
        {"valid": bool, "errors": [...], "warnings": [...], ...}
    """
    validation: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "sections_found": [],
        "n_atoms": 0,
        "total_charge": 0.0,
    }

    required = {"Masses", "Atoms", "Pair Coeffs"}

    try:
        with open(data_file) as f:
            content = f.read()

        # Check sections
        expected = [
            "Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
            "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
            "Dihedral Coeffs", "Improper Coeffs",
        ]
        for section in expected:
            if section in content:
                validation["sections_found"].append(section)

        for req in required:
            if req not in validation["sections_found"]:
                validation["errors"].append(f"Missing required section: {req}")
                validation["valid"] = False

        # Sum charges
        in_atoms = False
        total_q = 0.0
        n_atoms = 0
        n_nonzero = 0

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if not stripped:
                    if n_atoms > 0:
                        break
                    continue
                if stripped[0].isalpha():
                    if n_atoms > 0:
                        break
                    continue
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        q = float(parts[3])
                        total_q += q
                        n_atoms += 1
                        if abs(q) > 1e-6:
                            n_nonzero += 1
                    except (ValueError, IndexError):
                        pass

        validation["n_atoms"] = n_atoms
        validation["total_charge"] = round(total_q, 4)

        if abs(total_q) > 0.01:
            validation["warnings"].append(
                f"Net charge {total_q:+.4f} e (expected ~0 for neutralized system)"
            )

        if n_atoms > 10 and n_nonzero == 0:
            validation["errors"].append(
                "All charges are 0.0 — conversion likely dropped charges"
            )
            validation["valid"] = False

    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Could not read {data_file}: {e}")

    return validation
