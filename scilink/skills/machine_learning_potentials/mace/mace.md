---
description: MACE — equivariant message-passing neural network potential with production-ready foundation models for inorganic and organic systems; supports both ASE and LAMMPS pair_style.
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

MACE-MPA-0, achieves state-of-the-art accuracy on the Matbench benchmarks and
significantly improves accuracy compared to the MACE-MP-0 models on material systems.

Second generation models are not guaranteed to be better than first generation
models in all cases, but they are expected to be more stable during MD simulations.

## planning

Model selection:
- Elements are inorganic (metals, oxides, ceramics) → `mace-omat-0`
- Elements are organic (C, H, N, O, S, P, halogens) and energy/forces are needed → `mace-off23`
- Mixed inorganic/organic system → `mace-mh-0` (cross domain coverage)
- Speed over accuracy is required → `mace-mp-0b`

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
