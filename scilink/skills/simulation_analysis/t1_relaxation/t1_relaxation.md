---
description: ¹H NMR spin-lattice relaxation time T1 from the reorientational correlation of internuclear vectors via an intramolecular dipolar (BPP) model.
technique: reorientational_dipolar
computes: [t1_relaxation]
requires: [trajectory]
---

## Overview

¹H spin-lattice relaxation time T1 from equilibrium MD. For protons on the same
molecule (e.g. the two water protons), relaxation is dominated by the
*intramolecular* dipolar interaction, modulated by molecular reorientation. T1 is
obtained from the reorientational correlation time τc of the internuclear vector
plus a dipolar relaxation model. This needs the coordinate **trajectory**, sampled
finely enough (sub-ps) to resolve the reorientation.

## Implementation

1. **Load** the trajectory (MDAnalysis/ASE). Identify the target protons and the
   internuclear vector whose reorientation drives relaxation — for water ¹H, the
   intramolecular H–H vector of each water molecule. Record the sampling dt and
   the mean H–H distance r (~1.5–1.6 Å for water).

2. **Reorientational autocorrelation** of the second-order Legendre polynomial:
   C(t) = ⟨P₂( û(0)·û(t) )⟩,  û = unit internuclear vector,  P₂(x) = (3x²−1)/2,
   averaged over molecules and time origins. Normalize by C(0)=1.

3. **Correlation time** τc: the integral of the normalized C(t),
   τc = ∫₀^∞ C(t) dt, is a robust effective value (integrate to where C(t)
   plateaus near 0). If C(t) has not decayed by the end of the trajectory, the
   sampling is too short — flag it.

4. **Dipolar relaxation model** (two identical spin-½ ¹H, isotropic reorientation):
   1/T1 = (3/10) (μ₀/4π)² (γ_H⁴ ħ² / r⁶) · τc · [ 1/(1+ω₀²τc²) + 4/(1+4ω₀²τc²) ],
   with the ¹H Larmor frequency ω₀ = 2π·γ_H·B₀/(2π) at the measurement field
   (state the assumed spectrometer frequency). Use real physical constants
   (μ₀, γ_H, ħ). For water τc≈2 ps, so ω₀τc≪1 (extreme-narrowing): T1 is then
   effectively field-independent and 1/T1 ≈ (3/2)(μ₀/4π)²(γ_H⁴ħ²/r⁶)τc — compute
   this limit too and report whether extreme-narrowing holds.

5. Report T1 in seconds. Print one JSON object as the last stdout line:
   `{"status":"success","value":<T1 in s>,"units":"s","tau_c_ps":<float>,
   "extreme_narrowing":<bool>,"model":"intramolecular H-H dipolar (BPP)"}`.
   On failure: `{"status":"error","message":<str>}`.

## Validation

- C(t) must decay to ≈0 within the trajectory (τc ≪ trajectory length); a
  non-decaying C(t) means the run is too short — do not report a converged τc.
- The dump interval must resolve the reorientation (sub-ps); a coarse interval
  undersamples C(t) at short lag and biases τc.
- Sanity anchors for bulk water at ~298 K: τc ≈ 2 ps, ¹H T1 ≈ 2–5 s. Values
  orders of magnitude off signal a unit error, the wrong vector, or the wrong
  relaxation mechanism.

## Interpretation

T1 reports the reorientational dynamics of the proton-bearing molecule in its
local environment. Stronger solvation / more constrained motion → larger τc →
shorter T1. In the Zn(OTf)₂/water/EIS electrolyte this is the target observable
(the water-proton T1); report it with τc and its convergence status so the trend
across composition can be judged against the measured T1. This intramolecular
model omits the intermolecular dipolar contribution — a refinement, not a
substitute — so state it as a lower bound on 1/T1 when comparing absolute values.
