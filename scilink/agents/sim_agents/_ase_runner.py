"""
ASE runner — emits runnable Python scripts that drive any
``DeployedPotential`` through ASE, for either molecular dynamics
(``task="md"``) or geometry + cell relaxation (``task="relax"``).

This is MD-orchestration code owned by MDSimulationAgent, not an
MLIP concern — which is why it lives next to the agent and the
``DeployedPotential`` contract rather than under ``skills/_shared/``.

One template covers every MLIP backend. The only backend-specific
parts of an ASE script are the calculator import line, the
calculator-construction expression, and the device env var — all
three are data on ``potential.ase_calculator`` (an
``ASECalculatorSpec``). Adding a backend is declaring that spec in
the backend's deploy function; it never means writing a new script
generator here.
"""

import logging
import os
from typing import Any, Dict, Optional

from ._potential import DeployedPotential
from ...skills._shared.lammps_types import DEFAULT_ISOTOPE_WINDOW, DEFAULT_MASS_TOLERANCE

logger = logging.getLogger(__name__)


def generate_ase_script(
    potential: DeployedPotential,
    working_dir: str,
    structure_file: str = "system.data",
    task: str = "md",
    timestep: float = 1.0,
    temperature: float = 300.0,
    pressure: Optional[float] = None,
    n_steps: int = 1000,
    output_interval: int = 50,
    device: str = "cuda",
    fmax: float = 0.02,
    type_elements: Optional[Dict[Any, str]] = None,
    atom_style: Optional[str] = None,
    mass_tolerance: float = DEFAULT_MASS_TOLERANCE,
    isotope_window: float = DEFAULT_ISOTOPE_WINDOW,
) -> str:
    """
    Write a runnable ASE driver script (``run_md.py`` for MD,
    ``run_relax.py`` for relaxation) into ``working_dir`` and return
    its path.

    Parameters
    ----------
    potential:
        The deployed potential to drive. Its ``ase_calculator`` spec
        supplies the import + construction; the rest of the script is
        backend-agnostic.
    structure_file:
        Path the generated script reads at run time. Accepts any
        ASE-readable format (CIF, XYZ, EXTXYZ, POSCAR, PDB, …) as
        well as LAMMPS data files (.data/.lammps/.lmp).
    task:
        ``"md"`` — MaxwellBoltzmann init + Langevin (NVT) or NPT
        dynamics, writing a trajectory + thermo log.
        ``"relax"`` — BFGS + FrechetCellFilter geometry+cell
        optimization, writing the relaxed structure + a final
        lattice-parameter readout. ``temperature`` / ``pressure`` /
        ``n_steps`` (as a step cap) still apply where meaningful;
        ``fmax`` is the force convergence threshold.
    type_elements:
        For a LAMMPS data file, ``{type: element}`` for atom types its
        masses cannot tell (a deuterated "HW", a coarse-grained bead);
        applied on top of the file's readings, checked against the file,
        and written into the script. See ``_shared/lammps_types``.
    atom_style, mass_tolerance, isotope_window:
        Passed to the data-file reader (``inspect_lammps_data``);
        ``atom_style`` is also what the script passes ASE.
    """
    if task not in ("md", "relax"):
        raise ValueError(f"task must be 'md' or 'relax', got {task!r}")

    spec = potential.ase_calculator
    el_repr = ", ".join(repr(e) for e in potential.elements)
    # The generated script reads the structure by basename (the file is staged
    # into the run dir next to the script); the caller may pass any path.
    struct_basename = os.path.basename(structure_file)
    # For LAMMPS data, resolve each atom type's element now, from the data
    # file's Masses section (a label is used only if its element's mass
    # agrees), and write the map into the script. ASE's own nearest-mass
    # guess can mislabel (a labelled "59.0 # Ni" reads as Co) and has no
    # answer without masses. Also pass ASE an atom style when the file has no
    # "# <style>" header and ASE cannot guess it.
    from ...skills._shared.structure_io import is_lammps_data, inspect_lammps_data
    if is_lammps_data(structure_file) and os.path.isfile(structure_file):
        info = inspect_lammps_data(structure_file, type_elements=type_elements,
                                   atom_style=atom_style,
                                   mass_tolerance=mass_tolerance,
                                   isotope_window=isotope_window)
        if info["type_elements"] is None and not (
                info["n_types"] == 1 and len(potential.elements) == 1):
            raise ValueError(
                f"Cannot tell which element each atom type is "
                f"({info['problem']}).")
        if info["atoms_error"]:
            # The script would die inside ASE with a generic error; say why now.
            raise ValueError(f"Cannot read the atoms ({info['problem']}).")
        type_elements, atom_style = info["type_elements"], info["atom_style"]
    elif is_lammps_data(structure_file):
        # Not readable now: only the caller's type_elements (written in as
        # given) can map types; without it the script refuses a multi-type file.
        from ...skills._shared.lammps_types import check_type_elements
        type_elements = check_type_elements(type_elements) or None
        if not type_elements:
            logger.warning(
                f"{structure_file} is not readable now, so its atom types "
                f"cannot be mapped to elements; the script will refuse a "
                f"multi-type file. Pass type_elements to map them.")
    else:
        type_elements = atom_style = None

    # ── shared scaffold: imports, device, structure read, calc attach ──
    header = f'''"""
ASE+{potential.backend} {task.upper()} script -- generated by SciLink.

Potential: {potential.model_name} (backend: {potential.backend})
Elements:  {potential.elements}
Task:      {task}

Run:
    python {{script}}                          # uses default device
    {spec.device_env_var}=cpu python {{script}}    # force CPU
"""
import os
import pathlib
import time

from ase import units
from ase.io import read as _ase_read
{spec.import_line}

ELEMENTS = [{el_repr}]
TYPE_ELEMENTS = {type_elements!r}   # LAMMPS atom type -> element, from the data file
ATOM_STYLE = {atom_style!r}   # None: from the file's "Atoms # <style>" header
DEVICE = os.environ.get({spec.device_env_var!r}, {device!r})

# LAMMPS data files are not auto-detected by extension; everything else
# (CIF, XYZ, EXTXYZ, POSCAR, PDB, …) is handled by ase.io.read.
_LAMMPS_EXTS = {{".data", ".lammps", ".lmp"}}


def _load_structure():
    # The structure is staged into the run dir under its basename alongside this
    # script (see _generate_mlip_inputs), and the script runs with cwd=run_dir.
    # Reference it by basename so it resolves regardless of cwd and whether the
    # executor is local or remote — an absolute/relative path would break on a
    # compute node where the caller's local path does not exist.
    _path = {struct_basename!r}
    if pathlib.Path(_path).suffix.lower() in _LAMMPS_EXTS:
        from ase.io.lammpsdata import read_lammps_data
        atoms = read_lammps_data(_path, sort_by_id=True, atom_style=ATOM_STYLE)
        if TYPE_ELEMENTS:
            atoms.set_chemical_symbols(
                [TYPE_ELEMENTS[int(t)] for t in atoms.arrays["type"]])
        elif len(set(atoms.arrays["type"])) == 1 and len(ELEMENTS) == 1:
            # One type, one element: unambiguous.
            atoms.set_chemical_symbols(ELEMENTS * len(atoms))
        else:
            # Never fall back to ASE's nearest-mass guess: it ignores mass
            # labels (a "59.0 # Ni" reads as Co) and numbers types without
            # masses as H, He, ...
            raise SystemExit(
                f"{{_path}}: the element of each atom type was not resolved "
                f"when this script was generated (the data file was not "
                f"readable then). Regenerate it with the file in place.")
        # The potential may cover more elements than this file (e.g. a
        # model fine-tuned on a larger system); every atom must be covered.
        uncovered = sorted(set(atoms.get_chemical_symbols()) - set(ELEMENTS))
        if ELEMENTS and uncovered:
            raise SystemExit(
                f"{{_path}} contains {{uncovered}}, which the potential "
                f"(deployed for {{sorted(set(ELEMENTS))}}) does not cover.")
    else:
        atoms = _ase_read(_path)
    return atoms
'''

    if task == "md":
        if pressure is not None:
            dynamics_block = (
                "    from ase.md.npt import NPT\n"
                "    dyn = NPT(\n"
                "        atoms,\n"
                f"        timestep={timestep} * units.fs,\n"
                f"        temperature_K={temperature},\n"
                f"        externalstress={pressure} * units.bar,\n"
                "        ttime=20.0 * units.fs,\n"
                "        pfactor=(2e6 * units.GPa) * (20.0 * units.fs)**2,\n"
                "    )"
            )
        else:
            dynamics_block = (
                "    from ase.md.langevin import Langevin\n"
                "    dyn = Langevin(\n"
                "        atoms,\n"
                f"        timestep={timestep} * units.fs,\n"
                f"        temperature_K={temperature},\n"
                "        friction=0.01,\n"
                "    )"
            )
        body = f'''

def main():
    from ase.io.trajectory import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = _load_structure()
    print(f"system:  {{len(atoms)}} atoms "
          f"({{sorted(set(atoms.get_chemical_symbols()))}})")
    print(f"device:  {{DEVICE}}")
    atoms.calc = {spec.construct_expr}

    MaxwellBoltzmannDistribution(atoms, temperature_K={temperature})

{dynamics_block}

    traj = Trajectory("traj.traj", "w", atoms)
    dyn.attach(traj.write, interval={output_interval})

    log_file = open("thermo.log", "w")
    log_file.write("# step time(ps) PE(eV) KE(eV) T(K)\\n")
    log_file.flush()

    def log_step():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        step = dyn.nsteps
        time_ps = step * {timestep} / 1000.0
        line = (f"{{step}} {{time_ps:.4f}} {{epot:.6f}} "
                f"{{ekin:.6f}} {{temp:.2f}}")
        print(line)
        log_file.write(line + "\\n")
        log_file.flush()

    dyn.attach(log_step, interval={output_interval})

    t0 = time.time()
    print(f"running {n_steps} steps...")
    dyn.run({n_steps})
    log_file.close()
    elapsed = time.time() - t0
    print(f"done -- {{elapsed:.1f}}s "
          f"({{1000 * elapsed / max({n_steps}, 1):.2f}} ms/step)")


if __name__ == "__main__":
    main()
'''
        script_name = "run_md.py"

    else:  # task == "relax"
        body = f'''

def main():
    import json
    import numpy as np
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    from ase.io import write

    atoms = _load_structure()
    print(f"system:  {{len(atoms)}} atoms "
          f"({{sorted(set(atoms.get_chemical_symbols()))}})")
    print(f"device:  {{DEVICE}}")
    atoms.calc = {spec.construct_expr}

    cell0 = atoms.cell.cellpar()
    print(f"initial cellpar: {{np.round(cell0, 4)}}")

    # FrechetCellFilter relaxes atomic positions AND the cell, so the
    # equilibrium lattice constant falls out of the optimization.
    opt = BFGS(FrechetCellFilter(atoms), logfile="relax.log")
    t0 = time.time()
    converged = opt.run(fmax={fmax}, steps={n_steps})
    elapsed = time.time() - t0

    cell1 = atoms.cell.cellpar()
    write("relaxed.xyz", atoms)

    result = {{
        "converged": bool(converged),
        "steps": int(opt.nsteps),
        "fmax_threshold": {fmax},
        "final_energy_eV": float(atoms.get_potential_energy()),
        "initial_cellpar": [float(x) for x in cell0],
        "final_cellpar": [float(x) for x in cell1],
        "lattice_constant_A": float(cell1[0]),
        "elapsed_s": round(elapsed, 2),
    }}
    with open("relax_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"final cellpar:   {{np.round(cell1, 4)}}")
    print(f"lattice constant: {{cell1[0]:.4f}} A")
    print(f"converged: {{converged}}  ({{opt.nsteps}} steps, {{elapsed:.1f}}s)")


if __name__ == "__main__":
    main()
'''
        script_name = "run_relax.py"

    content = (header + body).replace("{script}", script_name)
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, script_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
