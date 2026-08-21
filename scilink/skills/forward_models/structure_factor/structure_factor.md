---
description: Static structure factor S(q) — a forward-simulated scattering / diffraction observable computed directly from atomic positions (the X-ray / neutron diffraction pattern a configuration would produce).
computes: [structure_factor]
requires: [trajectory]
output: curve
technique: static structure factor (Debye / direct sum)
---
# Static structure factor S(q)

## overview

The static structure factor S(q) is the observable measured in X-ray and neutron
diffraction. As a forward model it turns a simulated configuration (a trajectory
frame or a relaxed structure) into the curve an experiment would record — no
scattering code needed, just a sum over atomic positions. It is a *curve*
observable: S as a function of scattering vector magnitude q.

## implementation

Compute S(q) for the configuration in `DATA_FILES` and write the curve into
`OUTPUT_DIR`:

- Read the configuration with ASE (`ase.io.read(path, index=0)` for the first
  frame) or MDAnalysis for a large trajectory; average over a few frames if
  cheap.
- Build a q-grid (e.g. 0.5–12 Å⁻¹). Compute the orientationally-averaged S(q)
  either via the Debye scattering equation over pairwise distances,
  `S(q) = 1 + (1/N) Σ_{i≠j} sinc(q r_ij)`, or from the radial distribution
  function g(r) by Fourier transform using the number density from the cell
  volume.
- Save the curve to `OUTPUT_DIR` as a two-column CSV or an `.npy` array of shape
  `(2, n_q)` (`[q, S]`), and report a `summary` with `n_points`, the q-range,
  the peak S value and the q at which it occurs, and a NaN count.

Return exactly one JSON object on the last stdout line:
`{"status": "success", "output_type": "curve", "artifact": {"path": <file under
OUTPUT_DIR>, "format": "csv"|"npy", "shape": [...]}, "summary": {...}}`.

## validation

A physical S(q) is non-negative, approaches ~1 at large q, and shows a finite
first sharp diffraction peak. NaNs, negative S, or a monotonic curve with no
peak indicate a bug rather than a real spectrum.
