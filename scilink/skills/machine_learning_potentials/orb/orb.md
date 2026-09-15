---
description: Orb (Orbital Materials) — fast universal ML interatomic potential with ASE support, optimized for high-throughput screening and production MD.
detect:
  binaries: []
  env_vars: []
  python_modules: [orb_models]
  guidance: |
    Orb is distributed via the `orb-models` package
    (`pip install orb-models`). Model weights are downloaded on first use.
    A successful `from orb_models.forcefield import pretrained` confirms the
    backend is usable.
---
# Orb (Orbital Materials)

## overview

Orb is a universal MLIP from Orbital Materials, designed for high inference
speed without sacrificing the broad chemistry coverage of foundation models.
Orb models are trained on diverse inorganic and molecular datasets and are
competitive with MACE-mp-0 on accuracy benchmarks while running significantly
faster on CPU and GPU.

Model variants:
- **Orb-v2**: Second-generation model; balanced accuracy and speed.
- **Orb-v3**: Third-generation model; improved accuracy on solids and surfaces
  while maintaining orb-v2 throughput.

Orb-v3 models follow the naming pattern orb-v3-X-Y-Z:
- X tells how forces are computed: conservative (via energy gradients) or direct (predicted independently)
- Y sets the neighbor limit: 20 (maximum 20 neighbors per atom) or inf (unlimited)
- Z indicates the training dataset: omat (OMat24 AIMD subset) or mpa (MPTraj + Alexandria)

Conservative models are recommended to ensure energy conservation and stable trajectories.

Preferred models:
 - **orb-v3-conservative-inf-omat**: Most performant orb-v3 model for inorganic materials; good accuracy on surfaces and defects.
 - **orb-v3-conservative-inf-mpa**: Less performant orb-v3 model trained on mptraj + alexandria datasets; better for molecular systems and some materials not well-covered by orb-v3-omat.

The registry of Orb models can be found in the `orb_models` package:
```python
from orb_models.forcefield import pretrained

model_name_dict = pretrained.ORB_PRETRAINED_MODELS
```

Orb supports the Python/ASE calculator path.

## planning

When to choose Orb:
- High-throughput screening over many structures where inference speed is
  the primary constraint
- Rapid equilibration before switching to a more accurate backend for
  production
- CPU-only environments where MACE throughput is insufficient
- Comparative benchmarking against other universal models

When to prefer another backend:
- Highest accuracy for energy barriers or elastic properties → UMA or MACE
- Magnetic systems → CHGNet (dedicated magnetic moment head)
- LAMMPS production at scale with mature plugin support → MACE or DeePMD
- Organic / drug-like molecules → MACE (`mace-off23`) or UMA (`omol`)

Deployment path:
- Python/ASE workflow → `ORBCalculator` (primary, well-tested)

## implementation

ASE calculator path:

```python
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

orbff = pretrained.orb_v3_conservative_inf_omat()          # downloads weights on first call
atoms = read("structure.xyz")
atoms.calc = ORBCalculator(orbff, device="cuda")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

For CPU-only runs, pass `device="cpu"` to `ORBCalculator`. The model is
loaded once and reused across all calculator calls within the session.

Structure relaxation:

```python
from ase.optimize import FIRE

opt = FIRE(atoms, trajectory="relax.traj")
opt.run(fmax=0.05)   # convergence threshold: 0.05 eV/Å
```

## interpretation

Post-run checks:
- Energy drift in NVE < 0.2 meV/atom/ps → stable trajectory
- Forces > 10 eV/Å on equilibrium-looking structures → out-of-distribution
  geometry; verify with a DFT single-point or switch to a heavier model
- Lattice parameters deviating > 2% from experiment → consider UMA or MACE
  for higher accuracy on this chemistry

Trajectories are standard ASE `.traj` files readable with `ase.io.read` or
`ase.io.iread`. For throughput-intensive screening pipelines, process frames
in batch with `ase.io.iread` rather than loading the full trajectory at once.

When Orb diverges from a reference calculation, check:
1. Whether the structure is close to the training distribution (inspect
   composition and bonding motifs)
2. Whether a newer Orb version (orb-v3 vs orb-v2) improves agreement
3. Whether UMA or MACE gives a better result for the same structure

## validation

Orb accuracy thresholds (per Orbital Materials benchmarks):

- Energy MAE: ~15–25 meV/atom (inorganic solids, in-distribution)
- Force MAE: ~50–80 meV/Å (in-distribution)

Note: Orb is optimized for speed; for quantitative property prediction
(barriers, phonons, elastic constants) consider cross-checking against MACE
or UMA before reporting results.

Screening-mode accept/reject criteria:
- Structures with forces < 0.1 eV/Å after relaxation → converged; proceed
- Structures with forces > 1 eV/Å after 200 steps → likely bad geometry or
  out-of-distribution; flag for DFT validation
- Energy more than 0.5 eV/atom above the convex hull → metastable or
  unphysical; deprioritize in screening
