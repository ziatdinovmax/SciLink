"""Pure-component reference measurement on the engine-neutral rails.

The measurement half of the pre-run force-field validation: given ONE component
and a chosen reference PROPERTY, build a small pure-component reference, param-
eterize it through the SAME force-field backend the production run uses, run a
short simulation to measure that property, and read the value back. Property-
general — density, a lattice constant, a bond length are all just the ``property``
argument; the measuring run (a short generated simulation) produces whatever it
names, so a new property needs no new code here.

Backend/engine-neutral by construction. The operations that touch a specific
backend or engine — parameterization and the measuring run — are passed in as
callables (``parameterize_fn`` yields a ``ParameterizedSystem`` for any backend;
``run_measure_fn`` runs it on any engine and returns the value). Building the
pure-component reference is a local liquid-box packer for now; it is injectable
(``build_fn``) for other system types, and will be replaced by the shared
``build_box`` once that lands on main.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Avogadro's number; molar masses are looked up from the SMILES at call time.
_NA = 6.02214076e23


def _molar_mass_from_smiles(smiles: str) -> Optional[float]:
    """Molar mass (g/mol) of a SMILES via RDKit, or None if unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(Descriptors.MolWt(mol))
    except Exception as e:  # RDKit missing or a bad SMILES is not fatal
        _logger.debug(f"molar mass lookup failed for {smiles!r}: {e}")
        return None


def _pack_pure_box(smiles: str, n_molecules: int, init_density: float,
                   working_dir: str) -> Optional[str]:
    """Pack ``n_molecules`` copies of one molecule into a periodic cube.

    Deterministic (fixed packmol seed). Returns the path to an extxyz file with
    a periodic cell, or None if a dependency (RDKit / packmol) is unavailable or
    packing fails. STOPGAP: a pure single-component pack; swap for the shared
    ``build_box`` when it is on main.
    """
    mw = _molar_mass_from_smiles(smiles)
    if mw is None or shutil.which("packmol") is None:
        _logger.warning("pure-component packing unavailable (need RDKit + packmol)")
        return None
    try:
        from ase.io import read as _ase_read, write as _ase_write
        from rdkit import Chem
        from rdkit.Chem import AllChem

        work = Path(working_dir)
        work.mkdir(parents=True, exist_ok=True)

        # One MMFF-optimized conformer, written as xyz for packmol.
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        single = work / "pure_single.xyz"
        Chem.MolToXYZFile(mol, str(single))

        # Cube edge from n * MW / (rho * NA), packed a bit loose so NPT compresses.
        vol_A3 = (n_molecules * mw / _NA / init_density) * 1e24
        L = vol_A3 ** (1.0 / 3.0)
        packed = work / "pure_packed.xyz"
        inp = work / "pure_pack.inp"
        inp.write_text(
            f"tolerance 2.0\nseed 12345\nfiletype xyz\noutput {packed}\n"
            f"structure {single}\n  number {n_molecules}\n"
            f"  inside box 1. 1. 1. {L - 1:.3f} {L - 1:.3f} {L - 1:.3f}\n"
            "end structure\n"
        )
        with open(inp) as fh:
            subprocess.run(["packmol"], stdin=fh, cwd=str(work), check=True,
                           stdout=subprocess.DEVNULL)

        atoms = _ase_read(str(packed))
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)
        box = work / "pure_box.extxyz"
        _ase_write(str(box), atoms)
        return str(box)
    except Exception as e:
        _logger.warning(f"pure-component packing failed for {smiles!r}: {e}")
        return None


def measure_pure_component_property(
    component: Dict[str, Any],
    property: str,
    working_dir: str,
    *,
    parameterize_fn: Callable[[List[Dict[str, Any]], str, str], Any],
    run_measure_fn: Callable[[Any, str, str], Optional[Dict[str, Any]]],
    n_molecules: int = 200,
    init_density: float = 0.85,
    build_fn: Callable[..., Optional[str]] = _pack_pure_box,
) -> Optional[Dict[str, Any]]:
    """Measure one component's chosen reference PROPERTY with the production model.

    Property-general orchestration — nothing here is specific to density or to a
    single engine. Build a small pure-component reference structure, parameterize
    it through the production force-field backend, then hand it to
    ``run_measure_fn`` to run a short simulation for ``property`` and read the
    value back. Density, a lattice constant, a bond length: the property is just
    passed through, and ``run_measure_fn`` (a short generated run) produces
    whatever it names — so a new property needs no new code here.

    Backend/engine-neutral: parameterization and the measuring run are injected
    (``parameterize_fn`` via ParameterizedSystem, ``run_measure_fn`` via any
    engine). ``build_fn`` constructs the pure-component reference — a packed
    liquid box by default; injectable for a crystal cell or other system types.

    Args:
        component: ``{"name", "smiles", ...}`` for the single pure component.
        property: The reference property to measure (e.g. ``"density"``).
        working_dir: Scratch directory for this measurement.
        parameterize_fn: ``(components, coordinates_file, working_dir) ->
            ParameterizedSystem``.
        run_measure_fn: ``(parameterized_system, property, working_dir) ->
            {"value": float, "units": str} | None`` — runs a short simulation to
            measure ``property`` and reads it (None if the run/read failed).
        n_molecules, init_density: default liquid-box packing size / density.
        build_fn: pure-component reference builder (default the liquid packer).

    Returns:
        ``{"property", "value", "units", "n_molecules"}`` on success, or
        ``{"error": str}`` if any step could not complete. Never raises — a
        failed measurement is recorded as unmeasured upstream, not fatal.
    """
    smiles = component.get("smiles")
    name = component.get("name") or smiles
    if not smiles:
        return {"error": "component has no SMILES to build a pure structure from"}

    structure = build_fn(smiles, n_molecules, init_density, working_dir)
    if not structure:
        return {"error": "could not build a pure-component reference structure "
                         "(needs RDKit + packmol)"}
    try:
        psystem = parameterize_fn(
            [{"name": name, "smiles": smiles, "count": n_molecules}],
            structure, working_dir,
        )
    except Exception as e:
        return {"error": f"parameterization failed: {e}"}

    try:
        result = run_measure_fn(psystem, property, working_dir)
    except Exception as e:
        return {"error": f"measurement run failed: {e}"}
    if not result or result.get("value") is None:
        return {"error": (result or {}).get(
            "error", f"measurement run produced no {property}")}

    return {
        "property": property,
        "value": round(float(result["value"]), 4),
        "units": result.get("units"),
        "n_molecules": n_molecules,
    }
