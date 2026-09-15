---
description: DeePMD (Deep Potential Molecular Dynamics) — mature MLIP framework with first-class LAMMPS pair_style support, large-scale DFT training pipelines, and universal foundation models (DPA-2, DPA-3).
detect:
  binaries: [dp]
  env_vars: []
  python_modules: [deepmd]
  guidance: |
    DeePMD-kit is installed via `pip install deepmd-kit` (CPU) or
    `pip install deepmd-kit[gpu]` (CUDA). The CLI is `dp`. LAMMPS support
    requires a LAMMPS build with the DeePMD-kit plugin; check with
    `lmp -help | grep deepmd`. Universal foundation models (DPA-2, DPA-3)
    are downloaded from the AIS-Square model repository.
---
# DeePMD (Deep Potential Molecular Dynamics)

## overview

DeePMD-kit is a mature, production-grade MLIP framework that provides
first-class LAMMPS integration via `pair_style deepmd`, a comprehensive
training pipeline (`dp train` / `dp freeze` / `dp test`), and universal
foundation models (DPA-2, DPA-3) trained on large multi-fidelity datasets.

DeePMD is the standard choice when:
- Large-scale LAMMPS production runs are required (MPI + GPU, millions of
  atoms)
- The workflow involves custom model training on system-specific DFT data
- Uncertainty quantification via Deep Ensemble is needed for active learning

Foundation models:
- **DPA-2**: Universal potential covering 72 elements; strong on inorganic
  crystals, alloys, and oxides.
- **DPA-3**: Next-generation multi-fidelity model with improved coverage of
  molecular and surface chemistries.

## planning

When to choose DeePMD:
- LAMMPS-side production at scale (> 50k atoms, multi-node MPI + GPU)
- Custom model training from DFT reference data on a specific chemistry
- Active learning loop with Deep Ensemble uncertainty estimates
- High-pressure, high-temperature, or extreme conditions where a foundation
  model fine-tune is needed for accuracy

When to prefer another backend:
- No LAMMPS environment and Python/ASE is sufficient → MACE, CHGNet, UMA, or Orb
- Organic/molecular systems without training data → MACE (`mace-off23`) or UMA (`omol`)
- Magnetic systems → CHGNet
- Rapid prototyping without a training workflow → any foundation model with
  zero-shot deployment

Deployment path:
1. **Zero-shot with foundation model**: download DPA-2 or DPA-3, run directly
   in LAMMPS or ASE — no training required
2. **Fine-tuned model**: fine-tune DPA-2/DPA-3 on system-specific DFT data
3. **Trained from scratch**: full `dp train` workflow on custom dataset

## implementation

LAMMPS configuration (frozen model):

```lammps
units       metal
atom_style  atomic

pair_style  deepmd /path/to/model.pb
pair_coeff  * *
```

For multi-GPU runs, DeePMD supports domain decomposition without the
`no_domain_decomposition` restriction that MACE requires:

```lammps
pair_style  deepmd /path/to/model.pb out_freq 100 out_file model_devi.out
pair_coeff  * *
```

The `out_freq` / `out_file` options write per-frame model deviation
(uncertainty) to `model_devi.out` — essential for active learning pipelines.

Training a custom model:

```bash
dp train input.json        # train; input.json specifies type_map, training data, network
dp freeze -o model.pb      # freeze the trained model
dp test -m model.pb -s test_set/ -n 100    # evaluate on test set
```

Fine-tuning DPA-2 on system-specific data:

```bash
dp train --finetune DPA-2.pt input_finetune.json
dp freeze -o model_finetuned.pb
```

ASE calculator path:

```python
from deepmd.calculator import DP
from ase.io import read
from ase.md.langevin import Langevin
from ase import units

atoms = read("structure.xyz")
atoms.calc = DP(model="/path/to/model.pb")

dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

## interpretation

Post-run checks:
- Model deviation (from `model_devi.out`) < 0.05 eV/Å per atom → low
  uncertainty, trajectory is within the training distribution
- Model deviation 0.05–0.20 eV/Å → candidate frames for active learning;
  collect and run DFT
- Model deviation > 0.20 eV/Å → high uncertainty; do not use frames for
  quantitative results without DFT validation

Energy drift in NVE:
- < 0.1 meV/atom/ps → stable potential
- > 0.5 meV/atom/ps → model unstable for this chemistry; expand training set

Trajectory analysis: standard LAMMPS dump files are compatible with OVITO,
VMD, and MDAnalysis. The `model_devi.out` uncertainty file can be postprocessed
with `dpdata` or loaded directly as a plain-text time series.

## validation

DeePMD accuracy thresholds (custom-trained models; tighter than foundation
model baselines):

- Energy MAE < 1 meV/atom: excellent (well-converged custom model)
- Energy MAE 1–5 meV/atom: good (suitable for most MD applications)
- Energy MAE 5–10 meV/atom: acceptable for qualitative studies
- Energy MAE > 10 meV/atom: insufficient — expand or diversify the training set

- Force MAE < 50 meV/Å: excellent
- Force MAE 50–100 meV/Å: good
- Force MAE > 200 meV/Å: unacceptable — training set is too small or
  under-diverse for this chemistry

Active learning convergence criterion:
- < 5% of production frames exceed the model deviation threshold of 0.05 eV/Å
- Two consecutive active learning rounds add < 10 new DFT frames each →
  training set has converged for this chemistry and temperature range
