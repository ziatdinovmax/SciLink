"""AMBER backend for the uniform ``build_parameterized_system`` contract.

Produces an engine-neutral ``(prmtop, inpcrd)`` pair via the AmberTools
pipeline (pdb4amber -> antechamber/parmchk2 for non-standard residues -> tleap)
and returns it as the uniform payload dict. It writes NO engine input file
(the LAMMPS ``system.data`` conversion the legacy orchestrator did belongs to
``write_md_inputs``, which bridges the prmtop/inpcrd to any engine).

Requires AmberTools + ParmEd; raises an actionable error when they are absent.
"""
from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec


def _topology_stats(prmtop: str, inpcrd: str) -> Dict[str, Any]:
    """(n_atoms, total_charge) read back from the AMBER topology via ParmEd."""
    import parmed
    system = parmed.load_file(prmtop, xyz=inpcrd)
    n_atoms = len(system.atoms)
    total_charge = round(sum(float(a.charge) for a in system.atoms), 4)
    return {"n_atoms": n_atoms, "total_charge": total_charge}


def build_parameterized_system(
    *,
    pdb_file: Optional[str] = None,
    working_dir: str = ".",
    composition: Optional[Dict[str, bool]] = None,
    protein_ff: str = "ff19SB",
    water_model: str = "tip3p",
    gaff_version: str = "gaff2",
    charge_method: str = "bcc",
    small_molecule_info: Optional[List[Dict[str, Any]]] = None,
    solvate: bool = False,
    box_buffer: float = 10.0,
    neutralize: bool = True,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Parameterize a biomolecular system from a PDB into an engine-neutral
    ``ParameterizedSystem`` payload (``source_format="amber"``). Extra keyword
    arguments other backends consume (``components``, ``coordinates_file``, …)
    are ignored."""
    if not pdb_file:
        raise ValueError("the AMBER backend needs a `pdb_file`.")
    from . import amber as amber_tools  # lazy: keep module import dep-free
    status = amber_tools.check_amber_tools()
    if not status.get("available"):
        raise RuntimeError(
            "AMBER backend requires AmberTools + ParmEd, which are missing: "
            f"{status.get('missing')}. Install e.g. "
            "`conda install -c conda-forge ambertools parmed`, or route to a "
            "backend whose toolchain is present."
        )

    composition = composition or {}

    # Step 1: clean the PDB (best-effort).
    cleaned_pdb = pdb_file
    if shutil.which("pdb4amber"):
        try:
            cleaned_pdb = amber_tools.run_pdb4amber(pdb_file, working_dir)
        except Exception:
            cleaned_pdb = pdb_file

    # Step 2: parameterize any non-standard residues (small molecules).
    mol2_files: List[Dict[str, Any]] = []
    frcmod_files: List[str] = []
    for sm in (small_molecule_info or []):
        sm_file = sm.get("pdb") or sm.get("file")
        sm_name = sm.get("name", "LIG")
        ac = amber_tools.run_antechamber(
            input_file=sm_file, working_dir=working_dir,
            net_charge=int(sm.get("charge", 0)), charge_method=charge_method,
            atom_type=gaff_version, output_prefix=sm_name.lower(),
        )
        frcmod = amber_tools.run_parmchk2(
            mol2_file=ac["mol2"], working_dir=working_dir,
            atom_type=gaff_version, output_prefix=sm_name.lower(),
        )
        mol2_files.append({"mol2": ac["mol2"], "name": sm_name})
        frcmod_files.append(frcmod)

    # Step 3: tleap -> prmtop / inpcrd.
    script = amber_tools.generate_tleap_script(
        pdb_file=cleaned_pdb, working_dir=working_dir, composition=composition,
        mol2_files=mol2_files or None, frcmod_files=frcmod_files or None,
        protein_ff=protein_ff, water_model=water_model, gaff_version=gaff_version,
        solvate=solvate, box_buffer=box_buffer, neutralize=neutralize,
    )
    prmtop, inpcrd = amber_tools.run_tleap(script, working_dir)

    stats = _topology_stats(prmtop, inpcrd)
    return {
        "source_format": "amber",
        "backend": f"amber-{protein_ff}+{gaff_version}",
        "n_atoms": stats["n_atoms"],
        "total_charge": stats["total_charge"],
        "amber_files": [prmtop, inpcrd],
        "coordinates_file": cleaned_pdb,
    }


TOOL_SPEC = ToolSpec(
    name="build_parameterized_system",
    description=(
        "AMBER backend: parameterize a biomolecular system from a PDB "
        "(pdb4amber -> antechamber/parmchk2 -> tleap) into an engine-neutral "
        "ParameterizedSystem payload (a prmtop/inpcrd pair). The uniform "
        "backend contract behind ForceFieldAgent.parameterize."
    ),
    parameters={
        "pdb_file": {"type": "string", "description": "biomolecular structure (PDB)"},
        "working_dir": {"type": "string", "description": "where to write the payload"},
        "composition": {"type": "object",
                        "description": "{protein, nucleic, water, ...} flags for tleap"},
        "protein_ff": {"type": "string", "description": "protein force field (default ff19SB)"},
        "water_model": {"type": "string", "description": "water model (default tip3p)"},
        "small_molecule_info": {"type": "list",
                                "description": "[{pdb, name, charge}] non-standard residues"},
    },
    required=["pdb_file"],
    signature=("build_parameterized_system(*, pdb_file, working_dir='.', "
               "composition=None, protein_ff='ff19SB', water_model='tip3p', "
               "gaff_version='gaff2', charge_method='bcc', "
               "small_molecule_info=None, solvate=False, box_buffer=10.0, "
               "neutralize=True) -> dict"),
    import_line=("from scilink.skills.force_field.amber.build_parameterized_system "
                 "import build_parameterized_system"),
    agents=["simulation"],
    returns="uniform ParameterizedSystem payload dict (source_format='amber')",
)
