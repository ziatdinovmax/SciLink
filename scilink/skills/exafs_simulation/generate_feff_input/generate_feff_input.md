---
description: "Generate FEFF9 input files (feff.inp) for EXAFS calculations. Covers local environment extraction via carve_out(), FEFF card parameter selection (EDGE/HOLE, SCF, RMAX, CONTROL, CORRECTIONS), and batch generation from MD trajectories."
detect:
  binaries: []
  env_vars: []
  python_modules: [ase]
  guidance: |
    Requires ASE for structure manipulation. The carve_out and batch
    generation functions are provided as TOOL_SPEC helpers in the
    exafs_workflow skill bundle's feff_tools.py module.
---

## overview

FEFF9 computes X-ray absorption spectra (EXAFS, XANES) from atomic clusters.
Each calculation requires a `feff.inp` file specifying the absorbing atom,
its local environment (POTENTIALS and ATOMS lists), and computation
parameters (HOLE, SCF, RMAX, CONTROL, CORRECTIONS cards).

This skill provides guidance on:

- Extracting local atomic environments from crystal structures or MD snapshots
- Selecting appropriate FEFF card parameters for different systems
- Batch generation of feff.inp files from molecular dynamics trajectories

The `carve_out` function (in `feff_tools.py`) handles the geometric
operations: building a supercell, centering the absorber at the origin,
and identifying all neighbors within a cutoff radius.


## planning

### Edge and HOLE card selection

| Element type | HOLE index | Edge | Notes |
|-------------|-----------|------|-------|
| 3d metals (Ti-Zn) | 1 | K | Standard choice |
| 4d metals (Zr-Cd) | 4 | L3 | Higher cross-section than K |
| 5d metals (Hf-Hg) | 4 | L3 | K-edge too high for most beamlines |
| Lanthanides | 4 | L3 | K accessible but L3 preferred |
| Light elements (C-Si) | 1 | K | Only option |

Set S0² = 1.0 initially; refine by fitting to experimental data (typical
range: 0.8-1.0).

### CONTROL card by calculation goal

| Goal | CONTROL | Notes |
|------|---------|-------|
| Full calculation | `1 1 1 1 1 1` | All modules: potentials, phases, paths, FEFF, chi |
| EXAFS only (skip XANES) | `1 1 0 1 1 1` | Skip module 3 (paths enumeration for XANES) |
| Re-run chi only | `0 0 0 0 0 1` | Use existing potentials/phases (E0 fitting) |
| Potentials only | `1 0 0 0 0 0` | Check SCF convergence |

### SCF card parameters

| System type | SCF card | Notes |
|------------|----------|-------|
| Standard metals/oxides | `6.0 0 30 0.2 1` | 6 A radius, 30 iterations |
| f-electron systems | `5.5 0 30 0.05 10` | Tighter convergence for f-states |
| Molecular systems | `4.0 0 30 0.2 1` | Smaller cluster radius adequate |

### RMAX selection

| Analysis goal | RMAX (A) | Notes |
|--------------|----------|-------|
| First shell only | 4.5 | Fast, adequate for coordination number |
| Standard 2-4 shells | 6.0 | Good balance of information and speed |
| Extended with heavy scatterers | 8.0 | Needed for multiple-scattering paths |

### CORRECTIONS card

- Omit initially and fit vrcorr post-hoc against experimental E0
- With SCF: expect ±1 eV shift
- Without SCF: expect ±3 eV shift
- Format: `CORRECTIONS vrcorr vicorr` (e.g., `CORRECTIONS 0.0 0.0`)


## implementation

### Single structure — carve and write feff.inp

```python
from ase.io import read
from scilink.skills.exafs_simulation.exafs_workflow.feff_tools import carve_out

atoms = read('structure.cif')
target_atom = 0  # index of the absorber

# Carve local environment (rmax should exceed FEFF RMAX by ~2.5 A)
ipots_string, atoms_string, cluster = carve_out(
    atoms, target_atom=target_atom, rmax=8.5
)

# Assemble feff.inp
feff_inp = f"""TITLE structure analysis
HOLE 1 1.000000
CONTROL 1 1 1 1 1 1
PRINT   0 0 0 0 0 0

RMAX 6.0000
SCF 6.0 0 30 0.2 1

POTENTIALS
*  IPOT     Z     tag
{ipots_string}
ATOMS
*      X           Y           Z      IPOT    NN-DIST
{atoms_string}
END
"""

with open('feff.inp', 'w') as f:
    f.write(feff_inp)
```

### Batch generation from MD trajectory

Use the `generate_feff_inputs_from_trajectory` tool for production runs:

```python
from scilink.skills.exafs_simulation.exafs_workflow.feff_tools import (
    generate_feff_inputs_from_trajectory,
)

result = generate_feff_inputs_from_trajectory(
    trajectory_path='md_trajectory.xyz',
    target_atom=0,
    hole=1,                    # K-edge
    rmax=6.0,                  # FEFF path cutoff
    scf='6.0 0 30 0.2 1',     # SCF parameters
    s02=1.0,                   # amplitude factor
    control='1 1 1 1 1 1',    # full calculation
    step_size=250,             # sample every 250th frame
    sampling_start=1000,       # skip equilibration
)
# result['output_dir'] contains all feff.inp files
# result['n_inputs'] is the number of inputs generated
```

### Output directory layout

```
<trajectory_dir>/
  exafs_<element>_hole<N>_de_<vrcorr>_s02_<s02>_rc_<rmax>/
    000000_<atom>/feff.inp
    000250_<atom>/feff.inp
    ...
    neighborhoods_<atom>.xyz
```


## interpretation

### chi.dat format

FEFF outputs `chi.dat` in each run directory with columns:
- k (inverse Angstroms)
- chi(k) (dimensionless)
- |chi(k)| magnitude
- phase (radians)

The header line is:
```
#       k          chi          mag           phase @#
```

### Reading FEFF output

- `chi.dat` — main EXAFS signal for averaging
- `xmu.dat` — full absorption spectrum mu(E) if XANES modules enabled
- `feff.out` — log with path enumeration and convergence info
- `pot.bin` — binary potentials (reusable with CONTROL `0 0 0 0 0 1`)

### vrcorr fitting workflow

1. Run FEFF with `CORRECTIONS 0.0 0.0`
2. Compare chi(k) to experiment
3. Shift E0 by adjusting vrcorr (typically ±1-3 eV)
4. Re-run with `CONTROL 0 0 0 0 0 1` (chi-only, fast)
5. Iterate until alignment with experimental edge position


## validation

### Pre-generation checks

- [ ] Input structure is physically reasonable (no overlapping atoms)
- [ ] Target atom index exists in the structure
- [ ] Chosen rmax for carve_out exceeds FEFF RMAX by at least 2 A
- [ ] HOLE card matches the target element's appropriate edge

### Post-generation checks

- [ ] All feff.inp files contain non-empty POTENTIALS and ATOMS blocks
- [ ] Absorber is at the origin (0, 0, 0) with ipot 0
- [ ] Number of atoms in cluster is reasonable (typically 50-300)
- [ ] No hydrogen atoms in ATOMS list (excluded by carve_out)
- [ ] Directory contains expected number of inputs based on trajectory length and step_size
