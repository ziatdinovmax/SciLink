---
description: MACE — equivariant message-passing neural network potential with production-ready foundation models for inorganic and organic systems; supports both ASE and LAMMPS pair_style. Includes the MACE-POLAR-1 family with explicit long-range electrostatics for charged/polar systems and external fields.
detect:
  binaries: []
  env_vars: []
  python_modules: [mace]
  guidance: |
    MACE is installed via `pip install mace-torch`. A successful `import mace`
    confirms the Python/ASE backend is available. For LAMMPS-side production
    runs, also check for the `lammps-mace` plugin (`pip install lammps-mace`)
    or a LAMMPS build with the ML-IAP package linked against libtorch
    (`lmp -help | grep mace`).

    The MACE-POLAR-1 (PolarMACE) electrostatic models need `mace-torch>=0.3.16`
    PLUS the `graph_electrostatics` package, which provides the
    `graph_longrange` module required at runtime:
    `pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0`.
    A successful `from mace.calculators import mace_polar` plus `import
    graph_longrange` confirms the polar backend is usable.
---
# MACE (Multi-ACE)

## overview

MACE is an equivariant message-passing neural network potential. It is the
recommended default MLIP backend because it provides production-ready
foundation models covering the full periodic table and both inorganic and
organic chemistry, and it supports large-scale LAMMPS production runs via
`pair_style mace`.

| Model Name           | Elements Covered | Training Dataset      | Level of Theory               | Target System                              | Model Size           | GitHub Release | Notes                                                              |
|----------------------|------------------|-----------------------|-------------------------------|--------------------------------------------|----------------------|----------------|--------------------------------------------------------------------|
| MACE-MP-0a           | 89               | MPTrj                 | DFT (PBE+U)                   | Materials                                  | small, medium, large | >=v0.3.6       | Initial release of foundation model.                               |
| MACE-OFF23           | 10               | SPICE                 | DFT (wB97M-D3(BJ)/def2-TZVPP) | Organic (bio)molecular systems             | small, medium, large | >=v0.3.6       | Neutral, nonradical, and nonreactive systems only.                 |
| MACE-MP-0b           | 89               | MPTrj                 | DFT (PBE+U)                   | Materials                                  | models               | >=v0.3.10      | Improve pair repulsion and correct isolated atoms.                 |
| MACE-MP-0b2          | 89               | MPTrj                 | DFT (PBE+U)                   | Materials                                  | models               | >=v0.3.9       | Improve stability at high pressure.                                |
| MACE-MP-0b3          | 89               | MPTrj                 | DFT (PBE+U)                   | Materials                                  | models               | >=v0.3.9       | Fixed some phonons issues compared to b2.                          |
| MACE-MPA-0           | 89               | MPTrj + sAlex         | DFT (PBE+U)                   | Materials                                  | medium               | >=v0.3.10      | Improved accuracy for materials, improved high pressure stability. |
| MACE-OMAT-0          | 89               | OMAT                  | DFT (PBE+U) VASP 54           | Materials                                  | small, medium        | >=v0.3.10      | Excellent phonons.                                                 |
| MACE-MATPES-PBE-0    | 89               | MATPES-PBE            | DFT (PBE)                     | Materials                                  | medium               | >=v0.3.10      | No +U correction.                                                  |
| MACE-MATPES-r2SCAN-0 | 89               | MATPES-r2SCAN         | DFT (r2SCAN)                  | Materials                                  | medium               | >=v0.3.10      | Better functional for materials.                                   |
| MACE-MH-0/1          | 89               | OMAT/OMOL/OC20/MATPES | DFT (PBE/R2SCAN/wB97M-VV10)   | Inorganic crystals, molecules and surfaces | mh-0 mh-1            | >=v0.3.14      | Very good cross domain performance on surfaces/bulk/molecules.     |
| MACE-MDP             | 10               | SPICE                 | DFT (wB97M-D3(BJ)/def2-TZVPP) | Organic systems                            | model                | >=v0.3.16      | Dipoles & polarizabilities only; not for energies/forces.          |
| MACE-POLAR-1         | 83               | OMol25                | DFT (ωB97M-V hybrid)          | Charged/polar systems, external fields     | polar-1-s/m/l        | >=v0.3.16      | Explicit long-range electrostatics; arbitrary charge/spin; dipoles & partial charges. Needs `graph_electrostatics`. |

MACE-MPA-0, achieves state-of-the-art accuracy on the Matbench benchmarks and
significantly improves accuracy compared to the MACE-MP-0 models on material systems.

Second generation models are not guaranteed to be better than first generation
models in all cases, but they are expected to be more stable during MD simulations.

### MACE-POLAR-1 (Electrostatic / PolarMACE)

MACE-POLAR-1 is a foundation-model family that extends the MACE architecture
with explicit long-range electrostatics. It keeps the local MACE backbone
for short-range chemistry and adds a non-self-consistent polarisable-field
update on spin-resolved atomic multipoles: it learns atomic charge and spin
densities (multipole expansions in a Gaussian-type-orbital basis) directly from
energy and force labels, enforces total charge and spin through learnable Fukui
equilibration, and sums local, explicit-electrostatic, and learned non-local
terms for the final energy. This lets it handle arbitrary charge and spin
states, respond to external electric fields, and expose physically
interpretable per-atom charges/dipoles.

Trained on the OMol25 dataset (100M structures at the ωB97M-V hybrid-DFT level).
Two receptive-field sizes are documented — `polar-1-m` (medium, 12 Å receptive
field) and `polar-1-l` (large, 18 Å receptive field); a `polar-1-s` (small)
checkpoint is also available. Reported improvements over the standard backbone
include protein–ligand binding (×2), molecular crystals (×4), supramolecular
complexes (×2), and better redox potentials for transition-metal complexes.

When to reach for MACE-POLAR-1 (see Planning for the full selection rule):
charged/ionic systems, redox chemistry, response to an applied electric field,
long-range electrostatics (charge transfer across fragments), or when per-atom
charges/dipoles are the target observable.

## planning

Model selection:
- Elements are inorganic (metals, oxides, ceramics) → `mace-omat-0`
- Elements are organic (C, H, N, O, S, P, halogens) and energy/forces are needed → `mace-off23`
- Mixed inorganic/organic system → `mace-mh-0` (cross domain coverage)
- Speed over accuracy is required → `mace-mp-0b`
- System is charged/ionic, redox-active, in an applied electric field, or has
  long-range electrostatics (charge transfer, solvated ions) → `polar-1-m`
  (12 Å field) or `polar-1-l` (18 Å field, longer-range electrostatics) via the
  `mace_polar` loader
- Per-atom charges/dipoles or spin-resolved charge densities are the target
  observable → `polar-1-m` / `polar-1-l`

`polar-1-*` models require the extra `graph_electrostatics` dependency (see the
detect guidance) and use a dedicated `mace_polar` loader rather than `mace_mp`.
Choose `polar-1-l` over `polar-1-m` when the electrostatics extend beyond ~12 Å;
it is more expensive per step.

Deployment path:
- System < 10k atoms, Python workflow → ASE calculator path
- System > 10k atoms or long-timescale MPI run → LAMMPS `pair_style mace`
- `no_domain_decomposition` is required for all current MACE LAMMPS builds;
  this restricts the run to single-node MPI

Fine-tuning hyperparameters:
- `r_max`: 5.0 Å (default); increase to 6.0 Å for layered materials
- `forces_weight`: 100 (default); reduce to 10 if DFT forces are noisy
- `batch_size`: 4 (16 GB GPU), 8–16 (40+ GB GPU)
- `learning_rate`: 0.001 for fine-tuning (10× lower than from-scratch)
- `max_num_epochs`: 100 for fine-tuning (vs 200+ from scratch)

Fine-tuning from a MACE-POLAR-1 checkpoint uses `mace_run_train` with
`--model="PolarMACE"` and `--foundation_model="polar-1-m"` (or `polar-1-l`); see
Implementation for the full command.

## implementation

ASE calculator path:

```python
from mace.calculators import mace_mp
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

atoms = read("structure.xyz")
atoms.calc = mace_mp(model="medium", dispersion=False, device="cuda")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

LAMMPS configuration:

```lammps
pair_style  mace no_domain_decomposition
pair_coeff  * * /path/to/model.model El1 El2 El3
```

Requirements:
- LAMMPS with ML-IAP package compiled against libtorch
- OR: `pip install lammps-mace` (lammps-mace plugin)
- Verify: `lmp -help | grep mace`

MACE-POLAR-1 (electrostatic) ASE path — use the dedicated `mace_polar` loader,
which resolves the checkpoint by name. Set charge, spin, and any external field
in `atoms.info` before evaluating; the model handles arbitrary charge/spin
states and responds to the field:

```python
from mace.calculators import mace_polar

calc = mace_polar(
    model="polar-1-m",      # or "polar-1-s" / "polar-1-l"
    device="cpu",           # or "cuda"
    default_dtype="float64",  # use "float32" for faster MD
)

atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]   # applied E-field, V/Å
atoms.calc = calc

energy = atoms.get_potential_energy()   # populates calc.results
forces = atoms.get_forces()
stress = atoms.get_stress()
```

Reading dipoles and per-atom charge density (call a property first so
`calc.results` is populated):

```python
_ = atoms.get_potential_energy()

# Total dipole — only well-defined for NON-PERIODIC systems; ignore for PBC.
mu = calc.results["dipole"]                 # shape (3,)

# Atom-centred multipole coefficients, shape (n_atoms, 4)
p = calc.results["density_coefficients"]
atomic_charges = p[:, 0]                     # monopole (partial charge)
atomic_dipoles = p[:, [1, 2, 3]]             # cartesian (px, py, pz)

# Spin-resolved multipoles (spin-capable models), shape (n_atoms, 2, 4)
p_spin = calc.results["spin_charge_density"]
charges_up   = p_spin[:, 0, 0]
charges_down = p_spin[:, 1, 0]               # sum over axis 1 recovers `p`
```

Partial charges/dipoles are not uniquely defined; sums over clusters or
molecules are meaningful only for isolated fragments (no atom within ~6 Å of
another fragment).

## interpretation

Post-run checks:
- Energy drift in NVE < 0.1 meV/atom/ps → stable potential for this chemistry
- Energy drift > 1 meV/atom/ps → potential is near or outside the training
  distribution; verify with a short DFT single-point on representative frames
- Forces > 10 eV/Å on equilibrium-looking structures → out-of-distribution
  geometry; inspect the frame and consider fine-tuning
- Sudden energy spikes almost always indicate atoms too close together (geometry
  artifact), not model failure; check minimum interatomic distances

Trajectories from the ASE path are standard `.traj` files readable with
`ase.io.read` or `ase.io.iread`. The LAMMPS path writes standard dump/thermo
output processed with the usual LAMMPS tools (OVITO, MDAnalysis, etc.).

MACE-POLAR-1 electrostatic outputs:
- The total dipole (`calc.results["dipole"]`) is physical only for
  non-periodic systems; under PBC it is ill-defined and should be ignored.
- Partial charges/dipoles (`density_coefficients`) are interpretive, not
  uniquely defined — use them for trends and per-atom insight, and only sum
  them over fragments that are truly isolated (>~6 Å from any other atom).
- For an applied-field study, confirm the response is sensible (dipole aligns
  with the field, charge transfer scales with field strength) before trusting
  the magnitudes.

## validation

MACE-specific accuracy thresholds (per model card / benchmark literature):

- Energy MAE < 2 meV/atom: excellent
- Energy MAE 2–5 meV/atom: good (suitable for most applications)
- Energy MAE 5–10 meV/atom: acceptable (may affect barrier heights)
- Energy MAE > 10 meV/atom: poor — fine-tune or retrain

- Force MAE < 50 meV/Å: excellent
- Force MAE 50–100 meV/Å: good
- Force MAE 100–200 meV/Å: marginal
- Force MAE > 200 meV/Å: unacceptable

System-size limits:
- CPU: practical limit ~10k atoms (~1 ms/atom/step)
- GPU: practical limit ~50k atoms (~0.01 ms/atom/step on 24 GB card)
