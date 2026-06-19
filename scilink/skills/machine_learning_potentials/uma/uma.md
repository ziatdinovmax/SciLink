---
description: UMA (Universal Model for Atoms) — Meta/FAIR state-of-the-art universal MLIP supporting ASE and LAMMPS, with models trained on diverse multi-fidelity datasets.
detect:
  binaries: []
  env_vars: []
  python_modules: [fairchem.core]
  guidance: |
    UMA is distributed via the `fairchem-core` package
    (`pip install fairchem-core`). Model weights are downloaded on first use
    through the `fairchem` model registry. A successful
    `from fairchem.core import FAIRChemCalculator` confirms the backend is
    usable. LAMMPS support requires the `fairchem-lammps` plugin or a LAMMPS
    build with the ML-IAP package linked against the fairchem library.
---
# UMA (Universal Model for Atoms)

## overview

UMA is Meta/FAIR's universal neural network potential family, released in
2025 as part of the OpenCatalysis / FAIRChem initiative. UMA models are
trained on large multi-fidelity datasets spanning inorganic materials,
catalytic surfaces, and molecular systems, making them among the most
broadly applicable foundation potentials currently available.

Model variants:
- **uma-s-1p1** (Small): Early version of the UMA small model, lower
  latency, suitable for large-cell screening and rapid equilibration,
  ~10× faster than uma-m-1.1 (6.6M/150M active/total params).
- **uma-s-1p2** (Small): Latest version of the UMA small model, fastest
  of the UMA models while still SOTA on most benchmarks (6.6M/290M
  active/total params).
- **uma-m-1p1** (Medium): Best in class UMA model across all metrics,
  but slower and more memory intensive; recommended for quantitative
  results (50M/1.4B active/total params).
- **uma-s-1** has a known extensivity bug so is not recommended for use.

Model weights are downloaded automatically on first use. Set `HF_HUB_CACHE`
to control the download location. UMA is a gated model. If model weights are
not available at `HF_HUB_CACHE`, the user must request access from
[UMA registry](https://huggingface.co/facebook/UMA).
Once granted, the user can log in to the Hugging Face Hub with a personal access
token to download the model weights:

```python
from huggingface_hub import login

login(token=HF_API_KEY)
```

UMA supports both the Python/ASE calculator path and a LAMMPS plugin path
for large-scale production runs.

## planning

When to choose UMA:
- State-of-the-art accuracy is required and compute budget allows
- System spans multiple chemistry classes (inorganic + molecular + catalytic)
- Catalytic surfaces, adsorbates, or reaction pathways are the target
- Benchmarking a new system where the best available model is needed as a
  reference

When to prefer a lighter backend:
- Rapid screening where throughput dominates → CHGNet or Orb
- Magnetic property prediction → CHGNet (dedicated magnetic moment head)
- LAMMPS-only environment without fairchem plugin → MACE or DeePMD

Deployment path:
- Python/ASE workflow → `FAIRChemCalculator`
- LAMMPS production run → `fairchem-lammps` plugin (check availability for
  your LAMMPS build; the LAMMPS path is newer and less widely tested than
  the ASE path)

## implementation

ASE calculator path:

```python
from fairchem.core import FAIRChemCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

atoms = read("structure.xyz")
atoms.calc = FAIRChemCalculator(checkpoint_path="uma-m-1p1", task_name="omat")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

`task_name` selects the prediction head:
- `"oc20"` — use for catalysis
- `"oc22"` — use for oxide catalysis (-1p2 only)
- `"oc25"` — use for (electro)catalysis (-1p2 only)
- `"omat"` — use for inorganic materials
- `"omol"` — use for molecules and polymers
- `"odac"` — use for MOFs
- `"omc"` — use for molecular crystals

The `"omol"` head supports charged and open-shell species, making it suitable for molecular systems and catalytic adsorbates. Example: singlet vs. triplet CH2 energy difference:

```python
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")

#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

triplet.get_potential_energy() - singlet.get_potential_energy()
```

## interpretation

Post-run checks:
- Energy drift in NVE < 0.1 meV/atom/ps → stable trajectory for this chemistry
- Energy drift > 0.5 meV/atom/ps → system may be near the training distribution
  boundary; verify with DFT single-points on representative frames
- Forces > 10 eV/Å on equilibrium-looking structures → out-of-distribution
  geometry; inspect the frame and consider fine-tuning or falling back to DFT

Trajectory files from the ASE path are standard `.traj` files; use
`ase.io.read` or `ase.io.iread` with any standard ASE-compatible analysis
library (OVITO, MDAnalysis, etc.).

When comparing UMA results to lighter models (CHGNet, Orb), focus on:
- Energy barriers and transition-state geometries (where accuracy matters most)
- Equilibrium lattice parameters and bulk moduli
- Adsorption energies on catalytic surfaces

## validation

UMA accuracy thresholds (per FAIRChem benchmark / model card):

- Energy MAE: ~10–20 meV/atom (OC20/OMAT benchmarks, model-dependent)
- Force MAE: ~30–60 meV/Å (in-distribution)

Cross-check indicators:
- Lattice parameter within 1% of experimental value → well-converged structure
- Phonon frequencies positive at Γ-point → dynamically stable minimum
- Adsorption energy within 0.1 eV of DFT reference → reliable catalytic
  predictions; > 0.3 eV deviation warrants fine-tuning on system-specific data

If UMA results diverge substantially from MACE or CHGNet, run a short DFT
single-point on the divergent structure to identify which backend is closer
to the ground truth.