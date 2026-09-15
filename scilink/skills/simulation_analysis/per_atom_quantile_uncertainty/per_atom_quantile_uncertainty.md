---
description: Per-atom epistemic uncertainty for an MLIP trajectory via a quantile gradient-boosting model trained on the potential's own per-atom embeddings (the UQ-MLIP method).
technique: quantile_gbm
computes: [per_atom_uncertainty]
requires: [trajectory]
---

## Overview

Per-atom uncertainty quantification for a machine-learning interatomic
potential (MLIP), following the **UQ-MLIP** method. The idea: the same per-atom
latent embeddings the MLIP already computes are a fingerprint of an atom's local
environment, and a quantile regressor trained to bracket the potential's own
per-atom energies from those embeddings yields a cheap, calibrated *epistemic*
uncertainty — wide brackets flag atoms whose environments are unlike the
training/validation distribution (extrapolation), narrow brackets flag
in-distribution atoms.

Two stages, both post-hoc — **no MD is run here** and the UQ-decorated
calculator is *not* used:

1. **Train** a quantile gradient-boosting model (GBM) once, on per-atom
   embeddings + per-atom energies extracted from a *validation set* with the
   same MLIP backend that produced the trajectory.
2. **Predict** per-atom uncertainty for every atom in every frame of a *saved
   trajectory* by extracting its embeddings with that same backend and running
   them through the trained GBM.

This needs the coordinate **trajectory** (any ASE-readable format — `.traj`,
`.xyz`, `.lammpstrj`, `.dcd`, …) and the identity of the MLIP backend used for
the run (`chgnet`, `mace`, or `uma`). The `uq_mlip` package supplies the whole
pipeline; this skill orchestrates it.

## Planning

**Install UQ-MLIP.** This skill depends on the `uq_mlip` package (the PNNL
[UQ-MLIP](https://github.com/pnnl/UQ-MLIP) repo). Install it with the extra that
matches the MLIP backend you will use — the extractor for a backend is only
importable once that backend's dependency stack is present:

```bash
pip install -e ".[chgnet]"   # CHGNet extraction (shares an env with either below)
pip install -e ".[mace]"     # MACE extraction
pip install -e ".[uma]"      # UMA extraction
```

Run from a clone of the UQ-MLIP repo (requires Python 3.11–3.13). The base
package pulls in `ase`, `numpy`, `pandas`, and `xgboost`; the extra adds the
selected potential. **MACE and UMA depend on incompatible `e3nn` versions** — if
you need both, use separate environments; CHGNet has no `e3nn` dependency and
can share an env with either. On macOS, XGBoost also needs the OpenMP runtime
(`brew install libomp`); for UMA where `~/.cache` is not writable, set
`FAIRCHEM_CACHE_DIR`. Verify the install with `python -c "import uq_mlip"` before
running the recipe.

Other prerequisites and choices before running:

- **Backend must match the potential that generated the trajectory.** UQ is
  read off the MLIP's own embeddings, so extracting with a *different* backend
  than the one used for the MD is meaningless. Determine the backend from the
  run metadata / research goal (`chgnet`, `mace`, `uma`).
- **A validation set is required to train the GBM.** Use the same reference set
  the potential was validated against (e.g. an MPtrj validation split for
  CHGNet). If a GBM has already been trained and saved for this backend, skip
  straight to prediction with `UQModel.from_dir`. Do **not** train on the
  trajectory being analyzed — that would calibrate against the very
  distribution you are trying to test.
- **Device.** Embedding extraction (a GNN forward pass) benefits from a GPU;
  pass `device="cuda"` to `get_extractor` when available, else `"cpu"`. The GBM
  itself trains/predicts fine on CPU.
- **Isolated atoms (CHGNet).** CHGNet's graph converter aborts on isolated
  atoms by default; the `chgnet` extractor defaults to `on_isolated_atoms="warn"`
  so extraction never crashes on sparse/gas-phase frames.

## Implementation

Use the `uq_mlip` package end to end. The extractor turns `ase.Atoms` into an
`EmbeddingData` bundle (per-atom features + per-atom energies); `UQModel` is the
quantile GBM. **Do not run MD and do not attach `UQCalculator`** — read frames
from the saved trajectory file only.

```python
import json
import numpy as np
from ase.io import read
from uq_mlip import get_extractor, UQModel

# --- inputs (resolve from DATA_FILES / research goal) --------------------
BACKEND      = "chgnet"           # MUST match the MLIP that generated the run
TRAJ_PATH    = DATA_FILES["..."]  # the saved trajectory to analyze
VAL_PATH     = "/path/to/validation.xyz"   # reference set for GBM training
MODEL_DIR    = "uq_model"         # where the trained GBM is cached
DEVICE       = "cuda"             # or "cpu"

try:
    # 1) Backend extractor — same MLIP used for the trajectory.
    extractor = get_extractor(BACKEND, device=DEVICE)

    # 2) Train the quantile GBM (5%/95% brackets) once from the validation
    #    set, OR load a previously trained one. Training reads per-atom
    #    embeddings + energies from the validation frames.
    model_pkl = f"{MODEL_DIR}/GBMRegressor_0.05-0.95.pkl"
    import os
    if os.path.isfile(model_pkl):
        uq = UQModel.from_dir(MODEL_DIR, device=DEVICE)
    else:
        val_atoms = read(VAL_PATH, index=":")
        val_emb   = extractor.extract(val_atoms)         # EmbeddingData
        uq = UQModel(MODEL_DIR, lower_alpha=0.05, upper_alpha=0.95,
                     device=DEVICE).fit(val_emb)          # saves the .pkl

    # 3) Predict per-atom uncertainty for every frame of the SAVED trajectory.
    frames = read(TRAJ_PATH, index=":")
    per_frame_max, per_frame_mean, all_u = [], [], []
    for atoms in frames:
        emb = extractor.extract_atoms(atoms)             # one frame
        u   = uq.uncertainty(emb.node_feats)             # (n_atoms,) half-interval
        all_u.append(u)
        per_frame_max.append(float(u.max()))
        per_frame_mean.append(float(u.mean()))

    flat = np.concatenate(all_u)
    # 4) Flag extrapolation. A common choice is the validation upper tail as a
    #    threshold; here report the trajectory's own distribution and the
    #    fraction of atoms above the 95th percentile of the validation
    #    uncertainties (compute val uncertainties once if you need an absolute
    #    cutoff). Frames with any atom well above the bulk are OOD candidates.
    thr = float(np.percentile(flat, 95))
    frac_flagged = float((flat > thr).mean())

    print(json.dumps({
        "status": "success",
        "value": float(flat.mean()),
        "units": "eV/atom",
        "backend": BACKEND,
        "n_frames": len(frames),
        "n_atoms_total": int(flat.size),
        "max_uncertainty": float(flat.max()),
        "mean_uncertainty": float(flat.mean()),
        "flag_threshold": thr,
        "fraction_flagged": frac_flagged,
        "per_frame_max_uncertainty": per_frame_max,
    }))
except Exception as exc:
    print(json.dumps({"status": "error", "message": str(exc)}))
```

Notes on the API used above:

- `get_extractor(backend, device=...)` returns the backend extractor
  (`CHGNetExtractor`, `MACEExtractor`, `UMAExtractor`). CHGNet also accepts
  `model="0.3.0"`, `batch_size=16`, `on_isolated_atoms="warn"`.
- `extractor.extract(atoms_list)` / `extract_atoms(atoms)` → `EmbeddingData`
  with `.node_feats` (per-atom embeddings) and `.node_energies` (per-atom
  energies used as the GBM target).
- `UQModel(...).fit(EmbeddingData)` trains and pickles the booster;
  `UQModel.from_dir(dir, device=...)` reloads it. `uncertainty(node_feats)`
  returns half the (upper − lower) quantile interval per atom;
  `predict_embeddings(emb)` returns `{"lower","upper","uncertainty"}`.
- For a large trajectory, batch frames through a single `extractor.extract(...)`
  call rather than one-frame-at-a-time, splitting the flat per-atom result back
  into frames with `EmbeddingData.num_atoms`.

## Validation

- **Calibration.** The GBM is well-calibrated when the empirical coverage of the
  [lower, upper] bracket on a held-out validation set ≈ the nominal 0.90
  (upper_alpha − lower_alpha). A coverage far from 0.90 means the brackets are
  mis-scaled — retrain or widen/narrow the alphas.
- **Backend consistency.** If the extraction backend does not match the
  potential that generated the trajectory, the uncertainties are meaningless —
  verify the backend before trusting any numbers.
- **Not on the analyzed trajectory.** The GBM must be trained on an independent
  validation set, never on the trajectory under test; otherwise every atom looks
  in-distribution by construction.
- **Sanity of magnitudes.** Per-atom uncertainties are in eV/atom on the scale
  of the potential's per-atom energy errors. A trajectory whose *bulk* atoms all
  read far above the validation distribution suggests a systematic
  extraction/backend mismatch rather than genuine per-atom extrapolation.

## Interpretation

`per_atom_uncertainty` is the potential's *epistemic* uncertainty for each
atom's local environment — how far that environment sits from what the MLIP was
validated on. High-uncertainty atoms/frames mark **extrapolation**: regions
(reactive events, defects, phase boundaries, novel chemistry) where the MLIP's
predictions are least trustworthy and where DFT reference data would most
improve the model (active-learning target selection). Low, uniform uncertainty
across the trajectory indicates the run stayed in-distribution and the MLIP
energies/forces can be trusted.

Report the per-frame maximum uncertainty as a time series so uncertainty spikes
can be aligned with events in the trajectory, and the fraction of flagged atoms
as a single-number reliability summary: a large flagged fraction (e.g. > 10% of
frames containing flagged atoms) means the potential is operating outside its
validated domain for this system and results should be cross-checked against DFT
or a different backend.
