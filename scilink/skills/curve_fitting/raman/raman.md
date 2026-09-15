---
description: Raman spectroscopy band fitting and phase identification — fluorescence background handling, full-range multi-band deconvolution, and band-to-phase assignment for common mineral and materials classes.
technique: [Raman, "Raman spectroscopy", "micro-Raman", "Raman microscopy"]
---

# Raman Spectroscopy

## Overview

Raman spectra plot intensity against Raman shift (cm⁻¹). Bands are phonon
or molecular vibrational modes: sharp bands (FWHM a few cm⁻¹) indicate a
well-ordered crystal; broad bands indicate disorder, glass, or overlapping
modes. Band *positions* carry the chemistry and structure; absolute
intensities are arbitrary (relative intensities within one spectrum are
meaningful). Two artifacts dominate real spectra: **fluorescence** — a
broad, smooth background that can be orders of magnitude stronger than the
Raman bands, especially with visible excitation on colored or defective
samples — and cosmic-ray spikes (single-pixel). True Raman bands appear at
the same shift regardless of excitation wavelength; features that move
between excitations are fluorescence or artifacts.

## Planning

1. **Assess the background FIRST.** Estimate a rolling-minimum baseline and
   its share of total spectral area. If the baseline accounts for more than
   ~40 % of the area, the spectrum is fluorescence-dominated: plan an
   iterative asymmetric baseline subtraction (ALS, recipe in
   Implementation) BEFORE any peak fitting. Never fit peak components to
   the raw curve in this regime — a smooth curve through the ramp gives
   near-perfect R² while ignoring every Raman band.
2. **Fit the full measured range.** Do not restrict fitting to the
   dominant band region unless the user asked for it; weak low-wavenumber
   and high-wavenumber bands (lattice modes, OH stretches, overtones) carry
   identification-critical information. If any region is excluded, say so
   explicitly and why.
3. **Plan one component per resolved band, plus shoulders.** Count
   candidate bands by prominence above the noise floor on the
   baseline-corrected signal. A visible shoulder or asymmetry on a strong
   band is a component, not a nuisance.
4. Multiple excitation wavelengths of the same sample, when available, are
   a cross-check: keep bands that replicate across excitations; discount
   features that move or appear in only one.

## Implementation

When the background assessment (Planning step 1) indicates fluorescence
dominance, subtract an ALS baseline first:

```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def als_baseline(y, lam=1e6, p=0.001, niter=15):
    """Asymmetric least squares baseline (Eilers & Boelens).
    lam: stiffness, 1e5-1e7. RAISE if the baseline bends up into broad
         bands (eating signal); LOWER if it fails to follow the
         fluorescence curvature.
    p:   asymmetry, 0.001-0.01. RAISE slightly if the baseline sits below
         the noise floor between bands.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

y_corr = y - als_baseline(y)
```

Then fit the corrected signal with a sum of pseudo-Voigt components plus at
most a constant offset (the structured background is already gone — a
sloped or polynomial residual baseline at this stage usually means the ALS
parameters need adjusting, not that more baseline freedom is needed).

**Output-space contract (critical when a baseline is subtracted).** All
saved outputs must live in ONE consistent space: either (a) include the
subtracted baseline in the saved fit curve so it overlays the RAW data, or
(b) save the baseline-corrected signal as the data array next to a
corrected-space fit. Never save a peaks-only fit against raw data — the
verification overlay then shows the fit offset below the data by the
entire continuum, every residual is background rather than misfit, and
reported R² disagrees with R² recomputed from the saved arrays.

**The reported R² MUST be computed from exactly the two arrays you save**
(the saved data array vs the saved fit array) — not from the corrected
signal, not from any intermediate. If you also want a stricter
corrected-space or peak-region metric, report it as a separate,
clearly-labeled number; the primary R² is the saved-array one. The one
exception: when the user explicitly requested a fit window, the windowed
R² is the meaningful primary metric — report it labeled as windowed.

**The model must explain the peaks, not the ramp.** After fitting, check
where the model's variance lives: if the fitted curve with all peak
components removed still tracks most of the data (i.e., the "peaks" are
reproducing background curvature), the fit has failed regardless of R².

**Asymmetric bands.** Real Raman bands are often asymmetric — from
instrumental response, phonon-coupling (Fano-type) interactions,
disorder, or unresolved fine structure. The signature is a dispersive
(S-shaped, sign-flipping) residual centered on a single band that
persists after a symmetric refit. Handle it with a split pseudo-Voigt
(independent left/right widths, apex-preserving):

```python
import numpy as np

def split_pseudo_voigt(x, amp, center, fwhm_l, fwhm_r, eta):
    """Pseudo-Voigt with different left/right FWHM; amp = apex height.
    Constrain 0<=eta<=1 and 0.3 <= fwhm_r/fwhm_l <= 3 (larger asymmetry
    usually means unresolved structure, not lineshape)."""
    fwhm = np.where(x < center, fwhm_l, fwhm_r)
    s = fwhm / 2.3548
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)
    return amp * (eta * l + (1 - eta) * g)
```

Decision order matters: if the band is a KNOWN doublet (e.g., a
crystallographic splitting listed in the tables), fit two symmetric
components — never use asymmetry to absorb real splitting. Reach for the
asymmetric profile only when a second component fails to converge or has
no physical meaning, and report the asymmetry ratio per band. If EVERY
band comes out strongly asymmetric in the same direction, suspect a
baseline or calibration problem instead of lineshape physics.

Report per band: center (cm⁻¹), FWHM (left/right if asymmetric), relative
intensity (strongest = 1), and lineshape mixing. Cosmic spikes
(single-pixel) are removed by median filtering, never fitted.

## Interpretation

Assign every fitted band before concluding. The tables below cover common
classes and are **non-exhaustive** — for phases not listed, reason from
group frequencies (anion internal modes, metal-oxygen lattice-mode ranges,
OH/CH/NH stretch regions) rather than forcing a match to a listed entry.
**Never snap to the nearest table entry**: a candidate is supported only
when several bands (positions AND relative intensities) match within a few
cm⁻¹; if only the strongest band matches, say the pattern is not fully
explained and rank the candidate accordingly.

Diagnostic bands for common classes (positions ±3 cm⁻¹ unless noted;
strongest band in bold):

- **Carbonates**: calcite **1086**, 712, 282, 156; aragonite **1085**,
  705+701 doublet, 206, 152 (the ν4 region and lattice modes separate the
  CaCO₃ polymorphs — do not report "calcite" for aragonite); dolomite
  **1097**, 725, 299.
- **Sulfates**: gypsum **1008**, 415/494, 620, ~1141 (w); anhydrite
  **1017**, 417 (w)/499, 628/675, 1128 (the ν1 shift 1008→1017 separates
  hydrated from anhydrous); barite **988**, 462, 617.
- **Phosphates**: apatite group **964** (ν1, dominant); weak families at
  430–450 (ν2), 580–610 (ν4), 1030–1060 (ν3); F vs OH endmember needs the
  OH-stretch region (~3575 hydroxylapatite).
- **Silica/silicates**: α-quartz **464** (sharp, Lorentzian), 128, 206
  (broad), 355/394/401 (weak); feldspars **505–515 + 475–480 doublet** (microcline
  ~513+475, albite ~507+479) plus 150–290 lattice modes — a sharp
  ~465–515 doublet with no carbonate/sulfate ν1 says tectosilicate;
  olivine **doublet 815–825 + 838–857** (forsterite 824+856; both members
  shift down with Fe) with weak 920/965 and weak 300–620 SiO₄ modes;
  pyroxene (diopside) **665–670** + 1010–1015 + 320–395; amphibole
  (tremolite) **670–675** + 928/1028–1060 + sharp OH 3670–3675;
  serpentine (antigorite) **230/375–390/685–692** + OH 3665–3700; zircon
  **1008** + 974 + 356/439; kyanite **485–490 + 300/325** multiplet.
- **TiO₂ and oxides**: anatase **144** (very strong) + 399/516/639 + 197
  (weak, often below detection); rutile **447/612** + broad ~235 (no 144);
  hematite **225/292** + 245/412/498/613 + broad ~1320 two-magnon;
  magnetite broad **660–670**; spinel/chromite broad 680–700 (needs Cr/Fe
  chemistry).
- **Sulfides**: cinnabar **254**, 288, 343; chalcopyrite **292** + ~320
  (weak, resonance-prone); pyrite **343/379**; orpiment 310/355/384;
  molecular As-S (realgar) rich 180–370 multiplet.
- **Carbon**: D ~1350 + G ~1580–1600 (both broad = disordered sp²);
  graphene oxide shows intense broad D+G with weak, broad 2D ~2700;
  well-ordered graphite: sharp G, weak D, strong 2D; diamond single sharp
  **1332**.
- **Molecular anions in salts/solutions**: TFSI⁻ **740–745** (S–N–S
  expansion, sharp, solvation-sensitive), 280–350 cluster, 1240s;
  uranyl UO₂²⁺ ν1 **830–870** single symmetric band.
- Sulfur (α-S₈) **153/219/473**.

**Trap disambiguation — single strong band near 820–880.** Olivine, uranyl,
tungstates/molybdates (scheelite ~911, wolframite ~880), and some
phosphates all put their strongest band here. Olivine is a resolved PAIR
(≈30 cm⁻¹ apart) with weak 920/965 companions; uranyl is a single
symmetric band; scheelite-type adds 320–400 bending modes and nothing at
920–965. Never assign this region from the strongest band alone — use the
full pattern.

If literature context is provided with the task, weigh it against the
*fitted* band positions and prefer literature-confirmed assignments over
the tables above — the tables are a compact bootstrap, not an authority;
published spectra of the specific candidate phases are. Whether or not
literature is provided, always report a compact, machine-readable band
summary (centers and relative intensities, strongest first) alongside the
interpretation, so downstream identification steps can search the
literature from the measured features.

**Element consistency is a hard constraint.** When the sample's elements
are provided (from EDX/XRF or metadata), reject any candidate whose
essential elements are absent from that list (e.g., do not propose a
Cr-spinel when Cr is not among the reported elements). State when a
candidate is rejected for this reason. If no chemistry is given, say the
identification is pattern-only and rank candidates accordingly.

## Validation

- Inspect zoomed residuals across every strong band and act on the
  residual's SHAPE: a bimodal residual (two humps) means an unresolved
  shoulder/doublet — add a component; a dispersive S-shaped residual on a
  single band means asymmetry — switch that band to the asymmetric
  profile (Implementation). Do not accept while either signature remains
  above ~3× noise. R² alone must not accept a fit: near-unity R² is
  achievable while missing every real band (by fitting the background),
  while merging resolved structure, or while leaving structured residuals
  that are small relative to the strongest band.
- Completeness check: every band visible above ~5× noise in the
  baseline-corrected data must correspond to a fitted component; every
  fitted component must correspond to visible intensity (no components
  parked on noise or reproducing background curvature).
- If fluorescence was subtracted, report both the corrected fit and the
  baseline fraction so the user can judge the correction.
- For identification tasks: the top candidate must explain the strongest
  three bands AND not conflict with provided chemistry; alternatives
  within the same family should be listed when band positions cannot
  separate them (state which additional measurement would).
