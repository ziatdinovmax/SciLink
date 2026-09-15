---
description: Shear viscosity via the Green-Kubo stress-autocorrelation integral from a logged pressure-tensor time series.
technique: green_kubo
computes: [shear_viscosity]
requires: [thermo_log]
---

## Overview

Shear viscosity η from equilibrium MD via the Green-Kubo relation — the time
integral of the off-diagonal stress (pressure-tensor) autocorrelation. It needs
a densely and long-enough sampled time series of the pressure tensor from an
equilibrated NVT/NPT production run; it does not use the coordinate trajectory.
The estimate is statistically noisy and sensitive to run length and sampling
cadence, so convergence must be checked, not assumed.

## Implementation

Read the pressure-tensor time series and its timestep from the thermo/stress log,
then evaluate the Green-Kubo integral:

1. **Parse** the off-diagonal pressure components Pxy, Pxz, Pyz as a time series
   with the sampling interval dt. Identify the columns by header — they appear in
   a `thermo_style custom … pxy pxz pyz` log or a `fix ave/time` output file. Also
   read the volume V and temperature T (from the log or the run metadata).

2. **Autocorrelation.** For each independent stress component P_k(t), compute the
   autocorrelation C_k(τ) = ⟨P_k(t) P_k(t+τ)⟩ averaged over time origins. Use an
   FFT-based estimator and normalize by the number of origins at each lag (NOT by
   C(0) — the absolute magnitude carries the units).

3. **Average equivalent components** to cut noise: the three off-diagonal terms
   (Pxy, Pxz, Pyz) plus the two independent traceless diagonal combinations
   (Pxx−Pyy)/2 and (Pyy−Pzz)/2. All five are unbiased estimators of the same shear
   viscosity; average their autocorrelations.

4. **Green-Kubo integral:** η = (V / (k_B T)) ∫₀^∞ C(τ) dτ. Integrate with the
   trapezoidal rule and track the *running* integral; take η as the plateau value.
   If the running integral is still rising at the end of the series, it has not
   converged — report the best estimate and set plateau_reached=false.

5. **Units — do this explicitly and state the assumed unit system.** LAMMPS
   `real`: pressure in atm, time in fs, volume in Å³, T in K. LAMMPS `metal`:
   pressure in bar, time in ps, volume in Å³. Convert the final η to Pa·s, then
   report in mPa·s. A missing/incorrect unit conversion is the most common error
   and shows up as a value orders of magnitude off.

Print one JSON object as the last stdout line:
`{"status":"success","value":<η in mPa·s>,"units":"mPa·s","plateau_reached":<bool>,"n_origins":<int>}`.
On failure: `{"status":"error","message":<str>}`.

## Validation

Green-Kubo viscosity is noisy; guard against false precision:
- The autocorrelation must decay to ≈0 well before the integration cutoff.
- The running integral must show a plateau; a still-rising integral is not
  converged (plateau_reached=false).
- Typical liquid viscosities are ~0.1–10 mPa·s; a value orders of magnitude
  outside this range signals a unit error, not physics.

## Interpretation

η is the shear viscosity. Non-polarizable water models (notably TIP3P)
characteristically *underestimate* viscosity, so a low value can reflect the
force field rather than the analysis. Always report the value with its
convergence status so the downstream validation panel can judge it against
measured data rather than trusting it blind.
