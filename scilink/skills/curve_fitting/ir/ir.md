---
description: Infrared (FTIR) absorption spectroscopy of solids — transmittance/absorbance handling, baseline restraint for broad bands, and resolution of diagnostic band multiplets.
technique: [IR, FTIR, infrared, "infrared spectroscopy", "IR absorption"]
---

# Infrared Spectroscopy (FTIR)

## Overview

Mid-IR spectra (~400–4000 cm⁻¹) plot absorbance or transmittance against
wavenumber. **Check the orientation first**: absorbance-style data has
bands pointing UP from a low baseline; transmittance-style data has bands
pointing DOWN from a high plateau (often near 100 %). Convert transmittance
to absorbance (A = 2 − log₁₀(%T), or A = −log₁₀(T) for fractional T)
before fitting, and say that the conversion was applied. Strong bands in
raw absorbance data may be saturated (flat-topped) — apex positions of
saturated bands are unreliable.

## Planning

1. Determine orientation (dips vs peaks) and convert if needed.
2. **Fit the full measured range.** Never restrict fitting to one region
   (e.g., only the OH-stretch or only the fingerprint region) unless the
   user asked for it — the fingerprint region below ~1300 cm⁻¹ and the
   stretch region above 2800 cm⁻¹ carry complementary information and both
   must be modeled. If any region is excluded, say so and why.
3. **Broad bands are signal, not background.** O–H/N–H stretch envelopes
   (2800–3700 cm⁻¹), hydrogen-bonded water (~1630 plus broad 3000–3600),
   and broad ν3 envelopes are analyte absorption. Anchor any baseline ONLY
   in genuinely band-free windows (typically ~1900–2200 and ~2500–2700
   cm⁻¹, away from CO₂) — a baseline free to bend inside band regions will
   eat real absorption and cannot be recovered by the peak components.
4. Exclude or down-weight the atmospheric CO₂ region (2300–2380 cm⁻¹) and
   note narrow water-vapor structure (1400–1800, 3500–3900) when present.
5. Plan components for the fine structure of diagnostic multiplets: a
   resolved doublet or shoulder (e.g., sulfate ν3 splitting) must get its
   own components — the splitting pattern carries the site-symmetry
   information.

## Implementation

Fit pseudo-Voigt components on the (converted, baseline-restrained)
spectrum. Baseline: linear or single low-order polynomial pinned to the
band-free anchor windows from Planning — do not use aggressive iterative
baselines (ALS with low stiffness) on IR data with broad OH envelopes.
Saturated strong bands: fit the flanks and report the apex as uncertain
rather than forcing a narrow component through the flat top.

**Output-space contract.** If the spectrum was transformed (transmittance
to absorbance) or a baseline was subtracted, keep every saved output in
ONE consistent space: the saved data array, the saved fit curve, and the
reported R² must all refer to the same signal (transformed/corrected data
with a matching-space fit, or raw data with a fit that includes the
baseline term). Mixed spaces make the fit appear offset from the data and
invalidate the reported R². **The reported R² must be computed from exactly
the arrays you save** — any stricter alternative metric is reported
separately, clearly labeled (a user-requested fit window is the exception:
report the windowed R², labeled as windowed).

**Asymmetric bands.** A dispersive (S-shaped) residual across a single
band that persists after a symmetric refit indicates real asymmetry
(particle-size/shape effects in powders, hydrogen-bonding distributions,
unresolved fine structure). Refit that band with a split pseudo-Voigt —
independent left/right FWHM, apex height preserved, asymmetry ratio
bounded to 0.3–3 (use the same functional form as in the Raman recipe:
width chosen per side of the center). Decision order: a KNOWN multiplet
(e.g., sulfate ν3 splitting) gets separate symmetric components first;
asymmetry is not a substitute for resolving real splitting. If all bands
show the same strong asymmetry, suspect baseline or saturation problems
instead.

Report per band: center, FWHM (left/right if asymmetric), relative
intensity, and (for multiplets) the number of resolved components and
their splitting.

## Interpretation

The table below is **non-exhaustive** — for unlisted phases reason from
group frequencies rather than forcing a match, and treat a candidate as
supported only when multiple bands match, not just the strongest one.

Diagnostic mid-IR bands (cm⁻¹, ±5 unless noted):

- **Carbonates**: ν3 asym stretch broad **1420–1450**, ν2 sharp 875
  (calcite) / 881 (dolomite) / 856 (witherite), ν4 712 / 728 / 693
  respectively; combination ~1795, ~2513 weak.
- **Sulfates**: ν3 **1050–1150** — watch for SPLITTING into 2–3 components
  (site symmetry); ν4 590–680 multiplet; hydrated sulfates add water bends
  ~1630 and OH stretches.
- **Phosphates/arsenates**: ν3 ~**1000–1100** (phosphate), ~800–850
  (arsenate); OH bands where hydrated/hydroxylated.
- **Silicates**: Si–O stretch envelope **900–1100** (structure-rich —
  resolve its components), Si–O bends 400–550; sheet silicates add sharp
  structural OH 3620–3680 (talc 3677); amphiboles ~3630–3675.
- **Borates**: B–O stretches 1300–1450 (trigonal) and 950–1100
  (tetrahedral); hydrated borates show rich OH structure 3200–3600.
- **Oxides**: Sb₂O₃/As₂O₃ few strong bands 650–800; broad metal-oxide
  lattice absorption below 700.
- Water vs hydroxyl: molecular H₂O shows the 1630 bend + broad 3000–3600;
  structural OH gives sharp isolated 3500–3700 stretches without the bend.

Element consistency: reject candidates whose essential elements are absent
from provided chemistry; pattern-only identifications should be flagged as
such.

## Validation

- Zoomed residuals over every strong band and every diagnostic multiplet,
  acting on the residual's SHAPE: bimodal = unresolved component (split
  and refit); dispersive S-shape on a single band = asymmetry (switch that
  band to the asymmetric profile). Do not accept while either signature
  remains above ~3× noise. R² alone must not accept.
- Verify the baseline did not absorb broad bands: overlay raw data,
  baseline, and corrected data; if the baseline tracks the OH envelope or
  any band's flank, re-anchor it and refit.
- Every band above ~5× noise gets a component; no component sits on noise.
- For saturated bands, confirm the reported center comes from the flanks
  and is flagged uncertain.
