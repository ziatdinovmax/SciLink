"""FEFF-specific tools for EXAFS simulation workflows.

Provides carve_out (local environment extraction), batch FEFF input
generation from MD trajectories, and chi(k) averaging across FEFF outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import build
from ase.data import chemical_symbols
from ase.geometry import wrap_positions
from ase.io import read, write

from scilink.skills._shared._spec import ToolSpec


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _perp_heights(atoms) -> np.ndarray:
    """Perpendicular distances between opposite faces for each cell direction.

    For cell vectors a0, a1, a2 the perpendicular height in direction i is
    h_i = |V| / |a_j × a_k|.  This equals the vector norm only for
    orthogonal cells; for skewed cells it is strictly smaller, so using
    vector norms would underestimate the number of supercell repeats needed
    to enclose a sphere of given radius.
    """
    cell = atoms.cell.array
    volume = abs(np.linalg.det(cell))
    return np.array([
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ])


def _check_distance(atoms, rmax: float) -> bool:
    """Return True if a sphere of radius rmax fits inside the (wrapped) cell.

    After wrapping positions to center=(0,0,0) the origin sits at the centre
    of the supercell parallelepiped, so the inscribed-sphere radius is
    min(perpendicular_height) / 2.  The old Cartesian bounding-box test was
    incorrect for non-orthogonal cells.
    """
    return float(_perp_heights(atoms).min()) / 2.0 >= rmax


def _supercell_repeats(atoms, rmax: float) -> list[int]:
    """Repeats along each cell direction needed to enclose a sphere of rmax.

    Uses perpendicular face-to-face heights rather than lattice-vector norms
    so that non-orthogonal cells (hexagonal, monoclinic, triclinic …) are
    handled correctly.
    """
    repeats = []
    for h in _perp_heights(atoms):
        repeat = max(3, int(np.ceil((2 * rmax) / h)))
        if repeat % 2 == 0:
            repeat += 1
        repeats.append(repeat)
    return repeats


def carve_out(
    atoms,
    target_atom: int,
    rmax: float = 8.5,
) -> tuple[str, str, Any]:
    """Carve a local atomic environment from a structure for FEFF input.

    Builds a supercell, centers the target atom at the origin, and extracts
    all neighbors within ``rmax``. Returns POTENTIALS and ATOMS block strings
    ready for feff.inp assembly plus the carved ASE Atoms cluster.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (unit cell or snapshot from trajectory).
    target_atom : int
        Index of the absorbing atom.
    rmax : float
        Cluster radius in Angstroms (default 8.5). Should exceed the FEFF
        RMAX card value by ~2.5 A for safety.

    Returns
    -------
    tuple[str, str, Atoms]
        (ipots_string, atoms_string, cluster_atoms)

    Raises
    ------
    ValueError
        If the target atom index is invalid or the supercell is too small.
    """
    if target_atom < 0 or target_atom >= len(atoms):
        raise ValueError(
            f"target_atom index {target_atom} out of range for structure "
            f"with {len(atoms)} atoms."
        )

    supercell = build.make_supercell(
        atoms, P=np.diag(_supercell_repeats(atoms, rmax)).astype("int")
    )

    supercell.translate(-1 * supercell.get_positions()[target_atom])
    supercell.set_positions(
        wrap_positions(supercell.get_positions(), cell=supercell.cell, center=(0, 0, 0))
    )

    if not _check_distance(supercell, rmax):
        raise ValueError(
            f"Expanded supercell not large enough for rmax={rmax} A. "
            "Try a larger input cell or reduce rmax."
        )

    tags = np.ones(len(supercell))
    tags[target_atom] = 0
    tags[np.where(supercell.get_atomic_numbers() == 1)[0]] = 0
    tags[
        np.where(
            supercell.get_distances(target_atom, range(len(supercell)), mic=True) > rmax
        )[0]
    ] = 0
    supercell.set_tags(tags.astype(int))

    atoms_to_keep = np.where(supercell.get_tags() == 1)[0]

    elems = supercell[atoms_to_keep].get_atomic_numbers()
    central_tag = "" if atoms[target_atom].number not in elems else "0"
    ipot_map = {}
    ipots_string = (
        f"{0: >7}{atoms[target_atom].number: >7}"
        f"{chemical_symbols[atoms[target_atom].number]: >7}{central_tag}\n"
    )
    for i, elem in enumerate(sorted(set(elems)), start=1):
        ipots_string += f"{i: >7}{elem: >7}{chemical_symbols[elem]: >7}\n"
        ipot_map[elem] = i

    if not np.isclose(supercell.get_positions()[target_atom].sum(), 0.0, atol=0.0001):
        raise ValueError("Target atom not at (0,0,0) after centering.")

    atoms_string = "     0.000000    0.000000    0.000000    0    0.000000\n"
    for neighbor in atoms_to_keep:
        x = f"{supercell[neighbor].position[0]:.6f}"
        y = f"{supercell[neighbor].position[1]:.6f}"
        z = f"{supercell[neighbor].position[2]:.6f}"
        ipot = ipot_map[supercell[neighbor].number]
        nn = f"{supercell.get_distance(target_atom, neighbor, mic=True):.6f}"
        atoms_string += f"   {x: >10}   {y: >9}   {z: >9}    {ipot: >1}   {nn: >3}\n"

    cluster = supercell[np.append(atoms_to_keep, target_atom)]
    return ipots_string, atoms_string, cluster


def _build_feff_inp(
    run_name: str,
    frame: int,
    target_atom: int,
    output_dir: Path,
    hole: int,
    s02: float,
    control: str,
    print_flags: str,
    rmax: float,
    scf: str,
    corrections: str | None,
    ipots_string: str,
    atoms_string: str,
) -> str:
    """Assemble a complete feff.inp file from card parameters."""
    lines = []
    lines.append(f"* label:{output_dir}/feff_run{frame:0>6}_{target_atom}.inp:label")
    lines.append(f"* data_dir:{output_dir}:data_dir")
    lines.append(f"* frame: {frame} :frame  aindex: {target_atom} :aindex")
    lines.append(f"TITLE {run_name} frame {frame}")
    lines.append("")
    lines.append(f"HOLE {hole} {s02:.6f}")
    lines.append(f"CONTROL {control}")
    lines.append(f"PRINT   {print_flags}")
    lines.append("")
    lines.append(f"RMAX {rmax:<8.4f}")
    lines.append(f"SCF {scf}")
    if corrections is not None:
        lines.append(f"CORRECTIONS {corrections}")
    lines.append("")
    lines.append("POTENTIALS")
    lines.append("*  IPOT     Z     tag")
    lines.append(ipots_string)
    lines.append("ATOMS")
    lines.append("*      X           Y           Z      IPOT    NN-DIST")
    lines.append(atoms_string)
    lines.append("END")
    return "\n".join(lines) + "\n"


def generate_feff_inputs_from_trajectory(
    trajectory_path: str,
    target_atom: int,
    hole: int,
    rmax: float,
    scf: str,
    s02: float = 1.0,
    control: str = "1 1 1 1 1 1",
    print_flags: str = "0 0 0 0 0 0",
    corrections: str | None = None,
    step_size: int = 250,
    sampling_start: int = 0,
) -> dict[str, Any]:
    """Generate batch FEFF input files from an MD trajectory.

    Samples frames at regular intervals, carves local environments around
    the absorber, and writes feff.inp for each frame.

    Parameters
    ----------
    trajectory_path : str
        Path to an ASE-readable trajectory file (extxyz, traj, etc.).
    target_atom : int
        Index of the absorbing atom in trajectory frames.
    hole : int
        HOLE card index: 1=K, 2=L1, 3=L2, 4=L3.
    rmax : float
        FEFF RMAX path cutoff in Angstroms.
    scf : str
        SCF card parameters as space-separated string.
    s02 : float
        S0² amplitude reduction factor (default 1.0).
    control : str
        CONTROL card values (default "1 1 1 1 1 1").
    print_flags : str
        PRINT card values (default "0 0 0 0 0 0").
    corrections : str or None
        CORRECTIONS card values, or None to omit.
    step_size : int
        Sample every N-th frame (default 250).
    sampling_start : int
        First frame index to start sampling from (default 0).

    Returns
    -------
    dict
        Keys: output_dir (str), n_inputs (int), frames (list[int]).
    """
    traj_path = Path(trajectory_path)
    trajectory = read(str(traj_path), ":")
    run_name = traj_path.stem
    traj_dir = traj_path.parent

    absorber_symbol = trajectory[0][target_atom].symbol
    vrcorr_label = corrections.split()[0] if corrections else "0.0"

    output_dir = traj_dir / (
        f"exafs_{absorber_symbol}"
        f"_hole{hole}"
        f"_de_{vrcorr_label}"
        f"_s02_{s02:.1f}"
        f"_rc_{rmax:.1f}"
    )
    output_dir.mkdir(exist_ok=True)

    n_frames = len(trajectory)
    frames = list(range(sampling_start, n_frames, step_size))
    cluster_rmax = rmax + 2.5

    carved_regions = []
    for frame in frames:
        struct = trajectory[frame]
        ipots_string, atoms_string, region = carve_out(
            struct, target_atom, rmax=cluster_rmax
        )
        region.pbc = False
        region.cell = None
        carved_regions.append(region)

        subdir = output_dir / f"{frame:0>6}_{target_atom}"
        subdir.mkdir(exist_ok=True)

        inp_content = _build_feff_inp(
            run_name=run_name,
            frame=frame,
            target_atom=target_atom,
            output_dir=output_dir,
            hole=hole,
            s02=s02,
            control=control,
            print_flags=print_flags,
            rmax=rmax,
            scf=scf,
            corrections=corrections,
            ipots_string=ipots_string,
            atoms_string=atoms_string,
        )
        (subdir / "feff.inp").write_text(inp_content)

    neighborhoods_path = output_dir / f"neighborhoods_{target_atom}.xyz"
    write(str(neighborhoods_path), carved_regions, format="extxyz")

    return {
        "output_dir": str(output_dir),
        "n_inputs": len(frames),
        "frames": frames,
    }


def average_chi(directory: str, savefile: str) -> dict[str, Any]:
    """Average chi.dat files from FEFF batch output.

    Scans all subdirectories of ``directory`` for chi.dat files, reads
    their k and chi columns, interpolates every spectrum onto a shared
    uniform k-grid, and computes the mean chi(k).

    The common grid spans the intersection of all per-file k ranges
    (k_min = max of all starting k; k_max = min of all ending k) with
    spacing equal to the median per-file dk.  Each spectrum is linearly
    interpolated onto this grid before averaging, so files with different
    starting points, step sizes, or lengths are all handled correctly.

    Parameters
    ----------
    directory : str
        Directory containing FEFF output subdirectories with chi.dat files.
    savefile : str
        Base path for the output file (writes ``<savefile>-chi_avg.dat``).

    Returns
    -------
    dict
        Keys: k (ndarray), chi_avg (ndarray), n_samples (int),
        skipped (int), output_file (str).
    """
    feff_dir = Path(directory)
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    for sample_dir in sorted(feff_dir.iterdir()):
        chi_path = sample_dir / "chi.dat"
        if not chi_path.is_file():
            skipped += 1
            continue

        with open(chi_path) as f:
            lines = f.readlines()

        header_line = "#       k          chi          mag           phase @#\n"
        if header_line not in lines:
            skipped += 1
            continue

        index = lines.index(header_line)
        split_lines = [
            [tok for tok in line.split(" ") if tok != ""]
            for line in lines[index + 1:]
        ]
        k = np.array([float(row[0]) for row in split_lines])
        chi = np.array([float(row[1]) for row in split_lines])

        if len(k) < 2:
            skipped += 1
            continue

        raw.append((k, chi))

    if not raw:
        return {
            "k": np.array([]),
            "chi_avg": np.array([]),
            "n_samples": 0,
            "skipped": skipped,
            "output_file": "",
        }

    # Build a common grid over the intersection of all k ranges.
    k_min = max(k[0] for k, _ in raw)
    k_max = min(k[-1] for k, _ in raw)
    dk = float(np.median([float(np.median(np.diff(k))) for k, _ in raw]))
    k_common = np.arange(k_min, k_max + dk / 2, dk)

    # Interpolate each spectrum onto the common grid and stack.
    chi_interp = np.array([
        np.interp(k_common, k, chi) for k, chi in raw
    ])
    chi_mean = np.mean(chi_interp, axis=0)

    paired = np.vstack((k_common, chi_mean)).T
    output_file = f"{savefile}-chi_avg.dat"
    with open(output_file, "w") as f:
        f.writelines([" ".join(row) + "\n" for row in paired.astype(str)])

    return {
        "k": k_common,
        "chi_avg": chi_mean,
        "n_samples": len(raw),
        "skipped": skipped,
        "output_file": output_file,
    }


# ---------------------------------------------------------------------------
# TOOL_SPECS
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    ToolSpec(
        name="carve_out",
        description=(
            "Carve a local atomic environment from a crystal structure for "
            "FEFF EXAFS input generation. Builds a supercell, centers the "
            "absorber at the origin, and extracts neighbors within rmax."
        ),
        parameters={
            "structure_path": {
                "type": "string",
                "description": "Path to an ASE-readable structure file.",
            },
            "target_atom": {
                "type": "integer",
                "description": "Index of the absorbing atom.",
            },
            "rmax": {
                "type": "number",
                "description": (
                    "Cluster radius in Angstroms (default: 8.5). Should exceed "
                    "the FEFF RMAX card by ~2.5 A."
                ),
            },
        },
        required=["structure_path", "target_atom"],
        import_line=(
            "from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
            "import carve_out"
        ),
        signature="carve_out(atoms, target_atom: int, rmax: float = 8.5) -> tuple[str, str, Atoms]",
        agents=["simulation"],
        when_to_use=(
            "When generating FEFF input files from a crystal structure or MD "
            "snapshot. Call once per frame per absorber."
        ),
        returns=(
            "Tuple of (ipots_string, atoms_string, cluster_atoms) ready for "
            "feff.inp assembly."
        ),
        example=(
            "from ase.io import read\n"
            "from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
            "import carve_out\n\n"
            "atoms = read('snapshot.xyz')\n"
            "ipots, atoms_block, cluster = carve_out(atoms, target_atom=0, rmax=8.5)"
        ),
    ),
    ToolSpec(
        name="generate_feff_inputs_from_trajectory",
        description=(
            "Generate batch FEFF input files from an MD trajectory for EXAFS "
            "spectrum computation. Samples frames and writes feff.inp for each."
        ),
        parameters={
            "trajectory_path": {
                "type": "string",
                "description": "Path to MD trajectory (extxyz, traj, etc.).",
            },
            "target_atom": {
                "type": "integer",
                "description": "Index of absorbing atom in trajectory frames.",
            },
            "hole": {
                "type": "integer",
                "description": "HOLE card index: 1=K, 2=L1, 3=L2, 4=L3.",
            },
            "rmax": {
                "type": "number",
                "description": "FEFF RMAX path cutoff in Angstroms.",
            },
            "scf": {
                "type": "string",
                "description": 'SCF card parameters, e.g. "6.0 0 30 0.2 1".',
            },
            "step_size": {
                "type": "integer",
                "description": "Sample every N-th frame (default: 250).",
            },
            "sampling_start": {
                "type": "integer",
                "description": "First frame index to sample (default: 0).",
            },
        },
        required=["trajectory_path", "target_atom", "hole", "rmax", "scf"],
        import_line=(
            "from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
            "import generate_feff_inputs_from_trajectory"
        ),
        signature=(
            "generate_feff_inputs_from_trajectory(trajectory_path, target_atom, "
            "hole, rmax, scf, ...) -> dict"
        ),
        agents=["simulation"],
        when_to_use=(
            "After MD trajectory is complete, to generate FEFF inputs for "
            "batch EXAFS spectrum computation."
        ),
        returns="Dict with output_dir path, n_inputs generated, and frame indices.",
    ),
    ToolSpec(
        name="average_chi",
        description=(
            "Average all chi.dat files from FEFF batch output to produce a "
            "converged EXAFS chi(k) spectrum."
        ),
        parameters={
            "directory": {
                "type": "string",
                "description": (
                    "Directory containing FEFF output subdirectories with "
                    "chi.dat files."
                ),
            },
            "savefile": {
                "type": "string",
                "description": "Base path for the output averaged chi file.",
            },
        },
        required=["directory", "savefile"],
        import_line=(
            "from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
            "import average_chi"
        ),
        signature="average_chi(directory: str, savefile: str) -> dict",
        agents=["simulation"],
        when_to_use=(
            "After all FEFF jobs complete, to average chi(k) spectra across "
            "MD snapshots for a converged EXAFS signal."
        ),
        returns=(
            "Dict with k array, chi_avg array, n_samples averaged, skipped "
            "count, and output file path."
        ),
    ),
]
