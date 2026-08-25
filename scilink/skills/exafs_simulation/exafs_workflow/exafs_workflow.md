---
description: "End-to-end EXAFS simulation from crystal structures: simulation-engine-agnostic cell relaxation, NVT molecular dynamics, FEFF input generation, FEFF execution, chi(k) averaging, and publication-ready plotting. Delegates MD to co-active machine_learning_potentials, molecular_dynamics, or periodic_dft skills."
detect:
  binaries: [feff, feff.x, feff9, feff8, feff8l, feff6l, feff7]
  env_vars: [FEFF_DIR, FEFF_BIN]
  python_modules: [ase]
  guidance: |
    Requires ASE for structure manipulation and trajectory conversion. The MD
    calculator is provided by the co-active simulation skill — at least one of
    machine_learning_potentials, molecular_dynamics/lammps, or periodic_dft
    must be co-loaded. Any version of FEFF (6-lite through 9) can be used for
    EXAFS; the lite versions are free. Detection checks $FEFF_BIN, $FEFF_DIR,
    $PATH (feff/feff.x/feff9/feff8/feff8l/feff6l/feff7), then a site default
    at /share/feff/feff90_binaries/feff.x. All versions produce chi.dat in the
    same format used by this skill's averaging pipeline.
---

## overview

EXAFS (Extended X-ray Absorption Fine Structure) simulation computes
theoretical chi(k) spectra from crystal structures by sampling thermal
disorder via molecular dynamics and computing scattering paths with FEFF.

The pipeline has six stages:

1. **Validate inputs** — Load structure, confirm MD engine choice, set temperature
2. **Cell relaxation** — Optimize cell and positions with the chosen engine
3. **Molecular dynamics** — NVT production run for thermal sampling
4. **Generate FEFF inputs** — Carve local environments from trajectory frames
5. **Run FEFF** — Batch FEFF9 calculations on carved clusters
6. **Average and plot** — Compute mean chi(k) and generate convergence plots

This skill owns the FEFF-specific stages (4–6) and the engine-neutral
relaxation and MD scaffolding (1–3). The calculator for stages 2–3 is
provided by the **co-active simulation skill** — load one from
`machine_learning_potentials/`, `molecular_dynamics/lammps`, or
`periodic_dft/` alongside this skill (see `SKILL.md` for the co-loading
table). The FEFF operations (carving, input generation, averaging) are
provided as registered TOOL_SPEC helpers in `feff_tools.py`.


## planning

### MD engine selection

Co-load ONE simulation skill alongside `exafs_simulation/exafs_workflow`.
The choice determines the calculator for relaxation and MD:

- **MLIP via ASE (recommended default):** `machine_learning_potentials/{mace,chgnet,orb,uma,deepmd}`
  Refer to `machine_learning_potentials/general` for backend selection:
  magnetic systems → `chgnet`; organic molecules → `mace` (mace-off23);
  default inorganic/mixed → `mace` (mace-omat-0) or `uma`.

- **MLIP via LAMMPS (large cells, MPI-scale MD):** `machine_learning_potentials/{mace,deepmd}` +
  `molecular_dynamics/lammps`. Only MACE and DeePMD have LAMMPS `pair_style` support.
  Trajectory is a LAMMPS dump file; convert to extxyz before Stage 4.

- **AIMD / DFT-quality sampling (small cells, benchmark):** `periodic_dft/{vasp,qe}`.
  10–100× more expensive than MLIP. Trajectory is VASP XDATCAR or QE output;
  convert to extxyz before Stage 4.

### MD parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Temperature | 300 K | Match experimental conditions |
| Timestep | ~0.24 fs (10 a.u.) | Conservative for stability |
| Steps | 83,500 | ~20 ps; decrease for fast dynamics |
| Thermostat | NVT Nose-Hoover chain (ASE) | Correct canonical sampling |
| Trajectory output | extxyz | Readable by FEFF tools |

### FEFF binary availability

Before running Stage 5, locate or install a FEFF binary. All versions of FEFF
(6-lite through 9) produce `chi.dat` in the same format used by this skill's
averaging pipeline, so any installed version works.

**Detection order (first match wins):**
1. `$FEFF_BIN` env var — e.g. `export FEFF_BIN=/opt/feff9/bin/feff.x`
2. `$FEFF_DIR/<binary>` — e.g. `export FEFF_DIR=/share/feff/feff90_binaries`
3. Any of `feff`, `feff.x`, `feff9`, `feff8`, `feff8l`, `feff6l`, `feff7` on `$PATH`
4. Site default: `/share/feff/feff90_binaries/feff.x`

If none of the above resolves, FEFF is not installed. All versions are
available from the FEFF Project download page:
<https://feff.phys.washington.edu/feffproject-feff-download.html>

**Version summary:**

| Version | License | EXAFS capable | Notes |
|---------|---------|---------------|-------|
| FEFF6-lite | Free | Yes | EXAFS-only; good default for exploratory work |
| FEFF8-lite | Free | Yes | EXAFS-only; more recent scattering code than 6-lite |
| FEFF7 | Purchase required | Yes | Full MS code; adds XANES capability over lite versions |
| FEFF8 | Purchase required | Yes | Full MS; improved self-consistent field potentials |
| FEFF9 | Purchase required | Yes | Current generation; best accuracy for quantitative work |

Recommendation: use **FEFF8-lite** if you do not have a license (free, EXAFS
output identical to full FEFF8), or **FEFF9** if you have access and need
XANES or highest-accuracy multiple-scattering paths.

**General build instructions (applies to all versions):**

The download page provides source code and pre-compiled binaries for
Windows, Linux, and macOS. If a pre-compiled binary is available for your
platform, download and set `$FEFF_BIN` directly. To compile from source:

```bash
# After downloading and extracting the tarball or zip:
tar xzf feff<version>.tar.gz   # or unzip feff<version>.zip
cd feff<version>

# Most FEFF versions ship a Makefile or CMake build:
make                            # try this first
# or: cmake -B build && cmake --build build

# Set the env var to the resulting binary (name varies by version):
export FEFF_BIN=$(find . -type f -name "feff*" -perm /111 | head -1)
```

Compiler requirements: a Fortran compiler (`gfortran` or Intel `ifort`) is
required. On most Linux systems: `sudo apt install gfortran` or
`conda install -c conda-forge gfortran`.

### FEFF parameters

See the `generate_feff_input` sibling skill for detailed card selection.
Quick defaults for a first pass:

- **EDGE**: K for 3d metals, L3 for 4d/5d metals
- **RMAX**: 6.0 Å (2–4 shells), 8.0 Å (extended range)
- **SCF**: `6.0 0 30 0.2 1` (standard); `5.5 0 30 0.05 10` (f-elements)
- **Sampling**: Every 250th frame from production trajectory


## implementation

### Step 1: Calculator setup

Follow the co-active simulation skill's `implementation` section for
calculator construction. The result is an ASE-compatible `calculator`
object attached to the `atoms` structure:

```python
# If using the DeployedPotential contract (mlip_tools.deploy() output):
from scilink.skills._shared.mlip_tools import deploy

deployed = deploy(
    backend="mace",          # or "chgnet", "deepmd", etc.
    model="mace-omat-0",     # foundation-model keyword or path to trained model
    elements=atoms.get_chemical_symbols(),
    working_dir="./mlip_deploy",
    device="cuda",
)
spec = deployed.ase_calculator

# Reconstruct the calculator in the run script:
# (the MLIP skill's implementation snippet is authoritative for each backend)
exec(spec.import_line)
DEVICE = "cuda"
calculator = eval(spec.construct_expr)
atoms.calc = calculator
```

For LAMMPS MD, follow the `molecular_dynamics/lammps` implementation section
and configure the MLIP pair_style from the co-active MLIP skill's LAMMPS
configuration block. Trajectory output will be a LAMMPS dump file; convert to
extxyz before Stage 4:

```bash
ase convert md_trajectory.dump md_trajectory.xyz
```

For VASP AIMD, follow the `periodic_dft/vasp` implementation section (set
`IBRION=0, MDALGO=2, NSW=<n_steps>, TEBEG/TEEND` in INCAR). Read the
XDATCAR trajectory with:

```python
from ase.io import read
traj = read("XDATCAR", ":", format="vasp-xdatcar")
```

### Step 2: Cell relaxation

```python
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

atoms = read("input_structure.cif")
atoms.calc = calculator  # from Step 1

ecf = FrechetCellFilter(atoms)
optimizer = FIRE(ecf)
optimizer.run(fmax=0.05)
```

### Step 3: Molecular dynamics (NVT)

#### ASE-NVT (default — Nose-Hoover chain)

Correct canonical ensemble sampling for quantitative EXAFS disorder.

```python
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

temperature = 300  # K
timestep_fs = 0.2419  # 10 a.u. converted to fs
n_steps = 11000

atoms.calc = calculator
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)

dyn = NoseHooverChainNVT(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature,
    tdamp=100 * timestep_fs * units.fs,
    traj_file="md_trajectory.traj",
)
dyn.run(n_steps)

# Convert to extxyz for Stage 4
from ase.io import read, write
traj = read("md_trajectory.traj", ":")
write("md_trajectory.xyz", traj, format="extxyz")
```

#### LAMMPS-NVT (large cells / MPI)

Write a LAMMPS input script following the `molecular_dynamics/lammps`
implementation section. Use the MLIP `pair_style` from the co-active MLIP
skill's LAMMPS configuration block (e.g. `pair_style mace
no_domain_decomposition` for MACE). Output a dump file every N steps:

```lammps
dump 1 all custom 1 md_trajectory.dump id type x y z
dump_modify 1 every 250 first yes
```

Convert the dump to extxyz before Stage 4 (see Step 1 conversion command).

#### VASP AIMD

Follow the `periodic_dft/vasp` implementation section. Set in INCAR:

```
IBRION = 0       # MD run
MDALGO = 2       # Nosé-Hoover thermostat
NSW    = 5000    # number of ionic steps
TEBEG  = 300     # initial temperature (K)
TEEND  = 300     # final temperature (K)
POTIM  = 2.0     # timestep in fs (2 fs is safe for most systems)
```

Read XDATCAR trajectory after the run (see Step 1 conversion snippet) and
convert to extxyz before Stage 4.

### Step 4: Generate FEFF inputs

Use the `generate_feff_inputs_from_trajectory` tool:

```python
from scilink.skills.exafs_simulation.exafs_workflow.feff_tools import (
    generate_feff_inputs_from_trajectory,
)

result = generate_feff_inputs_from_trajectory(
    trajectory_path="md_trajectory.xyz",
    target_atom=0,        # absorber index
    hole=1,               # K-edge
    rmax=6.0,             # FEFF path cutoff
    scf="6.0 0 30 0.2 1",
    step_size=250,        # sample every 250th frame
)
```

### Step 5: Run FEFF

First, locate the FEFF binary (following the detection order in the planning
section). Any installed version (6-lite through 9) works for EXAFS:

```python
import os, shutil

# Search order: prefer newer/full versions, accept lite versions for EXAFS.
_FEFF_NAMES = ("feff.x", "feff9", "feff8", "feff7", "feff8l", "feff6l", "feff")

def find_feff_binary() -> str:
    """Return path to any FEFF binary, or raise with install instructions."""
    # 1. Explicit env var
    explicit = os.environ.get("FEFF_BIN", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    # 2. FEFF_DIR env var
    feff_dir = os.environ.get("FEFF_DIR", "").strip()
    if feff_dir:
        for name in _FEFF_NAMES:
            candidate = os.path.join(feff_dir, name)
            if os.path.isfile(candidate):
                return candidate
    # 3. $PATH
    for name in _FEFF_NAMES:
        found = shutil.which(name)
        if found:
            return found
    # 4. Site default
    site_default = "/share/feff/feff90_binaries/feff.x"
    if os.path.isfile(site_default):
        return site_default
    raise FileNotFoundError(
        "No FEFF binary found. Set $FEFF_BIN to the binary path, or download "
        "a free version (FEFF6-lite or FEFF8-lite) from: "
        "https://feff.phys.washington.edu/feffproject-feff-download.html"
    )

feff_exe = find_feff_binary()
```

Then execute FEFF batch jobs (each subdirectory contains a feff.inp):

```python
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

feff_dirs = sorted(Path(result["output_dir"]).iterdir())

def run_feff(d):
    subprocess.run([feff_exe], cwd=d, stdout=open(d / "feff.out", "w"),
                   stderr=subprocess.STDOUT)

with ThreadPoolExecutor(max_workers=32) as pool:
    pool.map(run_feff, feff_dirs)
```

### Step 6: Average and plot

Use the `average_chi` tool:

```python
from scilink.skills.exafs_simulation.exafs_workflow.feff_tools import average_chi

result = average_chi(
    directory=result["output_dir"],
    savefile="exafs_output",
)
# result["chi_avg"] contains the averaged spectrum
# result["output_file"] is the path to the saved data
```


## interpretation

### k-space convergence

Plot k²chi(k) for increasing numbers of averaged frames. The spectrum
should stabilize (oscillation amplitudes stop changing) once enough
snapshots are included. Typically 20–40 frames suffice for first-shell
EXAFS; extended-range analysis may need 50+.

### Real-space features

The Fourier transform |chi(R)| shows peaks at coordination shell distances
(shifted ~0.3–0.5 Å from crystallographic distances due to phase shifts).
Compare peak positions and amplitudes to experimental data.

### Common failure modes

- **Noisy high-k region**: Insufficient MD sampling or too-short trajectory
- **Missing peaks in |chi(R)|**: RMAX too small; increase to capture longer paths
- **Amplitude mismatch**: S0² needs fitting against experimental data (typical: 0.8–1.0)
- **Peak position shift**: SCF convergence issue or wrong CORRECTIONS; try adjusting vrcorr
- **Non-physical oscillations**: Timestep too large causing MD instability


## validation

Pre-run checks:

- [ ] FEFF binary located (`find_feff_binary()` returns a path that exists and is executable)
- [ ] FEFF version confirmed — FEFF8 for exploratory work, FEFF9 for quantitative publication results
- [ ] Structure loads without errors and has correct composition
- [ ] Cell relaxation converged (fmax < 0.05 eV/Å)
- [ ] MD calculator attached successfully (test single-point energy)
- [ ] Temperature and timestep are physically reasonable
- [ ] Co-active simulation skill is loaded and its implementation section followed

Post-MD checks:

- [ ] Trajectory file exists and contains expected number of frames
- [ ] No atomic drift or explosion (max displacement reasonable)
- [ ] Temperature fluctuations within expected range (±20 K at 300 K)
- [ ] MLIP uncertainty below threshold (< 10% of frames flagged) — MLIP paths only

Post-FEFF checks:

- [ ] All FEFF jobs produced chi.dat output files
- [ ] chi(k) convergence achieved (spectrum stable with frame count)
- [ ] Peak positions in |chi(R)| match known coordination distances
