---
description: General MLIP guidance — when to use machine learning interatomic potentials, pretrained-first strategy, active learning loop, and cross-backend acceptance thresholds.
detect:
  binaries: []
  env_vars: []
  python_modules: []
  guidance: |
    This is a domain-level skill covering MLIP strategy and is not tied to
    a specific Python package. It loads alongside backend-specific skills
    (mace, chgnet, deepmd, uma, orb) to provide overarching context on
    model selection, validation, and active learning.
---
# Machine Learning Interatomic Potentials — General

## overview

Machine learning interatomic potentials (MLIPs) learn the potential energy
surface from quantum mechanical reference data. They achieve near-DFT accuracy
at a fraction of the computational cost for systems up to ~10,000 atoms
(CPU) or ~100,000 atoms (GPU).

Key distinctions from classical force fields:
- No fixed functional form — the model learns energy/force relationships
- Accuracy depends on training data coverage, not parameter fitting
- Units in LAMMPS: **metal** (eV, Å, ps) — NOT real (kcal/mol, Å, fs)
- Require LAMMPS compiled with ML-IAP package or backend-specific plugin

## planning

When to recommend MLIPs over classical force fields:
- Reactive chemistry (bond breaking/formation)
- Novel materials without established force-field parameters
- High-accuracy energy barriers (catalysis, diffusion, transition states)
- Metal/oxide interfaces and surfaces
- Phase transitions
- Systems where classical FF accuracy is demonstrably insufficient

When to prefer classical force fields:
- Large-scale sampling (> 50k atoms) without GPU resources
- Well-parameterized biomolecular systems (proteins in water)
- Routine production runs where speed matters more than accuracy
- Systems with no available DFT reference data and no pretrained coverage

Pretrained-first strategy (apply before any fine-tuning decision):
1. Always try a foundation model first — zero deployment cost
2. Run a short simulation (10–50 ps) on a representative structure
3. Evaluate uncertainty metrics and compare observables to experiment/DFT
4. Fine-tune only if quality is insufficient
5. Train from scratch only as a last resort when no pretrained model covers
   the chemistry

Backend selection guide:
- Default / inorganic systems → MACE (`mace-omat-0`) or UMA (`uma-s-1p2`)
- Organic / drug-like molecules → MACE (`mace-off23`)
- Magnetic systems (Fe, Co, Ni, Mn, Cr) → CHGNet (predicts magnetic moments)
- Speed-critical screening over many structures → CHGNet or Orb
- LAMMPS production runs at scale → MACE or DeePMD (`pair_style deepmd`)
- Universal coverage with state-of-the-art accuracy → UMA (`uma-m-1p1`)

## implementation

Active learning loop for training data augmentation:
1. Run MD with the pretrained foundation model
2. Collect per-atom uncertainty estimates (force variance or descriptor distance)
3. Flag high-uncertainty frames (> 10% of trajectory flagged → model unreliable)
4. Run DFT on flagged frames (VASP, CP2K, Quantum ESPRESSO, etc.)
5. Augment the training set and fine-tune
6. Repeat until uncertainty stays below threshold throughout production run

DFT data requirements for fine-tuning:
- Minimum: ~50 diverse structures with energies + forces (+ stresses if
  pressure-dependent properties matter)
- Recommended: ~200 structures spanning the relevant PES region
- Must include: equilibrium, compressed, expanded, and high-temperature
  snapshots; surface or interface frames if modelling those

LAMMPS units reminder — always set `units metal` for all MLIP backends:

```lammps
units       metal
atom_style  atomic
```

## interpretation

Simulation quality signals to check before trusting production results:

1. MLIP uncertainty:
   - Per-atom force variance or descriptor distance (backend-dependent metric)
   - > 10% of trajectory frames flagged as extrapolation → unreliable model
   - Max force > 10 eV/Å on an equilibrium-looking structure → out-of-
     distribution geometry; verify with DFT and consider fine-tuning

2. Physical observables:
   - Density within 5% of experimental/DFT value → acceptable
   - Radial distribution function peaks at correct bond lengths → good
   - Energy drift < 1% over total simulation time → stable potential
   - Temperature fluctuations consistent with the chosen ensemble

3. Common failure signatures:
   - Sudden structure collapse or "explosion" → check minimum interatomic
     distances; almost always a geometry artifact, not a model bug
   - Slow drift away from target density in NPT → check barostat coupling or
     consider switching backends
   - Periodic energy spikes → too-large timestep; try 0.5 fs for stiff bonds

## validation

Cross-backend acceptance thresholds (apply to all MLIP backends):

- Energy MAE < 5 meV/atom (bulk), < 10 meV/atom (surfaces/interfaces)
- Force MAE < 100 meV/Å
- Stress MAE < 0.5 GPa (if elastic properties are the target)

Action thresholds:
- Force MAE > 200 meV/Å → insufficient training data or wrong backend; do not
  use for quantitative predictions
- > 10% of production frames flagged as high-uncertainty → active learning
  loop required before trusting results
- Energy drift > 1 meV/atom/ps in NVE → potential unstable for this chemistry
