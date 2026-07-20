---
name: machine_learning_potentials
description: >
  Router for MLIP skill bundles. When a task involves machine learning
  interatomic potentials, consult the appropriate subdirectory skill based
  on the backend in use or the system's chemistry. Start with general/ for
  strategy and backend selection, then load the matching backend skill.
---

# Machine Learning Potentials — Skill Directory

This directory contains skill bundles for machine learning interatomic
potential (MLIP) backends. Each subdirectory is a self-contained skill.

## Skill index

| Subdirectory | Skill file | When to use |
|---|---|---|
| `general/` | `general.md` | Strategy, backend selection, active learning, cross-backend validation thresholds. Load this alongside any backend skill. |
| `mace/` | `mace.md` | MACE equivariant MLIP — default for inorganic systems (`mace-omat-0`) and organic molecules (`mace-off23`). Supports ASE and LAMMPS `pair_style`. |
| `chgnet/` | `chgnet.md` | CHGNet — best choice for magnetic systems (Fe, Co, Ni, Mn, Cr); predicts magnetic moments. ASE only (no LAMMPS `pair_style`). |
| `deepmd/` | `deepmd.md` | DeePMD — mature framework with first-class LAMMPS support (`pair_style deepmd`) and universal foundation models (DPA-2, DPA-3). |
| `uma/` | `uma.md` | UMA (Meta/FAIR) — state-of-the-art universal MLIP with multi-fidelity training. Supports ASE and LAMMPS. |
| `orb/` | `orb.md` | Orb (Orbital Materials) — fast universal MLIP optimised for high-throughput screening. ASE only (no LAMMPS `pair_style`). |

## Selection guidance

1. **Always load `general/general.md`** alongside whichever backend skill
   is active — it supplies the pretrained-first strategy, active learning
   loop, and cross-backend acceptance thresholds that apply universally.

2. **Choose a backend skill** using this priority order:
   - Magnetic systems (Fe, Co, Ni, Mn, Cr) → `chgnet/`
   - Organic / drug-like molecules → `mace/` (`mace-off23`) or `uma/` (`omol`)
   - LAMMPS production runs at scale → `mace/` or `deepmd/`
   - Speed-critical high-throughput screening → `chgnet/` or `orb/`
   - State-of-the-art accuracy → `uma/`
   - Default inorganic / mixed systems → `mace/` (`mace-omat-0`) or `uma/` (`omat`)

3. If the backend is already determined by the environment (e.g., a
   `dp` binary is present → DeePMD; `mace` Python module is importable →
   MACE), go directly to the matching skill.
