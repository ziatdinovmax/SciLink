---
description: Hyperspectral Raman mapping analysis — per-pixel band fitting, phase mapping, and fluorescence-aware decomposition on Raman spectral imaging datacubes.
---
# Hyperspectral Raman Mapping Skill

## overview

Hyperspectral Raman mapping produces a 3D datacube (spatial_y, spatial_x,
wavenumber) — a full Raman spectrum at each pixel of an XY scan. The
scientifically meaningful quantities are band parameters: peak position
(phase identity, stress, chemistry), FWHM (crystallinity, disorder),
intensity and area ratios (phase fractions, disorder metrics such as
I_D/I_G). Raw intensity alone is rarely the answer — it convolves focus,
topography, and fluorescence. The pervasive confound in Raman mapping is
photoluminescence/fluorescence: a broad, slowly varying background that can
exceed the Raman signal by orders of magnitude and varies spatially, so any
per-pixel comparison must separate sharp Raman bands from broad PL.

## planning

**Artifact triage before analysis:**
- Cosmic-ray spikes: isolated single-channel (1–3 bin) intensity outliers
  in individual pixels. Detect by comparison with spectral neighbors
  (median filter along the wavenumber axis) and remove before any
  decomposition — one spike can dominate a component.
- Detector edges: channels near the low-wavenumber cutoff (< ~150–250
  cm⁻¹) may contain laser-line/notch-filter roll-off; exclude them.
- Check whether the data is already background-subtracted (the metadata
  or notes may say so; negative baseline excursions are the telltale).
  Subtracted data can be legitimately signed — do not clip negatives, and
  do not subtract a baseline twice.

**Baseline / fluorescence handling:**
- If broad background remains, remove it per pixel before band analysis
  (asymmetric least squares or low-order polynomial through band-free
  windows). Fit ratios and positions on baseline-corrected spectra.
- The PL background itself can be a useful map (e.g., substrate exposure),
  but always present it as PL, never as a Raman phase.

**Decomposition:**
- NMF on non-negative, baseline-handled data is the standard first look.
  Component count = expected phases + one for background/PL; verify with
  reconstruction error vs. component count.
- Expect NMF to absorb baseline-slope and PL differences into components.
  A component whose basis spectrum is dominated by a broad continuum, or
  whose amplitude is several times the measured spectra with
  derivative-like excursions, encodes background — say so and do not
  assign it chemistry.
- Decomposition maps locate regions; band *fitting* on a high-SNR
  spectrum establishes what each region *is*. These are two questions —
  "where is the contrast" and "which phase" — and answering only the
  first (an abundance map with no verified band identity) leaves a
  phase-identification objective unmet. Plan the identity step
  explicitly (see the identity-first ordering under implementation).

**Spectral windows:** fit bands in restricted windows (± 100–200 cm⁻¹
around the band) with a local linear baseline, not over the full spectrum.
The fingerprint region and any high-wavenumber region of interest (e.g.,
carbon 2D band, O–H/C–H stretches) can be treated as separate problems.

## implementation

**Identity first, then map — fit the high-SNR representative spectrum
before the per-pixel map.** A single noisy pixel rarely supports a
confident band fit, so a per-pixel fit is the wrong place to *establish*
that a phase is present — it is only the right place to map a band already
known to be there. Establish identity on a high-SNR spectrum first: the
mean spectrum of each NMF component's high-abundance pixels (or of a
region drawn from the abundance map), which averages down the noise and
resolves bands that no single pixel can. Fit the diagnostic bands on that
representative spectrum, report their measured positions and widths as the
phase identification, and only then attempt the per-pixel maps for spatial
variation. When the per-pixel fit fails but the representative-spectrum
fit succeeded, the identity result still stands — report it, and describe
the spatial distribution from the abundance map rather than leaving the
phase unidentified. The endmember/mean spectrum is often where the
diagnostic line (e.g. a sharp crystalline peak on a disordered-band
shoulder) is actually resolvable.

**If a diagnostic band is buried under PL/background, perform the targeted
refit — do not merely recommend it.** Recommending "baseline-subtract and
refit the <1200 cm⁻¹ window" and then not doing it leaves the analysis
incomplete; run the baseline-subtracted fit in the diagnostic window
within the same analysis.

**Per-pixel band-fit recipe.** For each diagnostic band: slice a window
around the nominal position, fit Lorentzian (crystalline single lines,
e.g., diamond) or pseudo-Voigt (disordered bands: D, G) profiles plus a
linear local baseline. Initialize centers at nominal positions, bound
within ± 30–50 cm⁻¹; bound FWHM to physically reasonable ranges (sharp
crystalline lines: 2–20 cm⁻¹; disordered-carbon bands: 20–200 cm⁻¹).
Overlapping bands in one window (D+G, or diamond on a D-band shoulder)
must be fit simultaneously, never sequentially.

Gate every fitted pixel: minimum band amplitude above the local noise
(e.g., > 3σ of the fit residual), R² threshold on the window, and center
not pinned at a bound. Set failed pixels to NaN rather than letting
implausible values contaminate the maps. Report the fraction of pixels
that passed — a map from 5% of pixels is a different claim than one from
90%.

**Outputs per band:** position map, FWHM map, and amplitude or area map.
For multi-phase samples add ratio maps (e.g., I_D/I_G; diamond-to-G
intensity) and a categorical phase-classification map from the fitted
parameters — those maps, with a stated decision rule, are the deliverable
for "where is each phase" objectives.

**Answer every named target the objective enumerates — on the
representative spectra, not per-pixel.** When it names specific bands,
report each band's fitted position and width numerically — but do this on
the high-SNR per-region / per-component spectra (a handful of fits), not
by fitting every named band at every pixel. Identifying a band without
stating its measured position is incomplete. When it names specific
regions or ROIs, address each by name — a representative-spectrum fit per
named region, not a whole-cube per-pixel sweep per band. When it names
higher-order or secondary bands, account for those on the representative
spectra too.

**Keep per-pixel mapping economical — it is the expensive step.** Fitting
many bands at every pixel of a large cube (10⁴–10⁵ pixels × multiple
components) is where runtime explodes, and most of it is redundant with
the representative-spectrum identity. Reserve per-pixel fitting for the
one or two quantities that genuinely need a spatial map to answer the
objective (typically the phase discriminant — a diamond/G or I_D/I_G
ratio, a target band's area) and derive the rest from the representative
spectra plus the NMF abundance maps. On a large cube, bin or subsample for
the per-pixel pass rather than fitting a full multi-band model at native
resolution.

## interpretation

**Carbon materials** (values for 532 nm excitation; D and 2D are
dispersive — at 633 nm, D shifts down to ~1330 and 2D to ~2640 cm⁻¹):
- Diamond: single sharp line at 1332 cm⁻¹ (FWHM ~2–10 cm⁻¹). Downshift
  and broadening indicate stress, heating, or nanocrystallinity.
  Distinguish from the D band by width — a "1332" line with FWHM
  > 30 cm⁻¹ is disordered carbon, not diamond.
- D band: ~1350 cm⁻¹, defect-activated; G band: 1580–1600 cm⁻¹, graphitic
  sp²; D′: ~1620 (shoulder on G); 2D: ~2690; D+G: ~2940 cm⁻¹.
- I_D/I_G is the standard disorder metric; carbon nanowalls show strong
  D, G, D′ and 2D with high I_D/I_G at edges.
- Amorphous/DLC: D and G merge into broad overlapping humps.

**Tungsten oxides and polyoxometalates:**
- Crystalline WO₃: 807 and 715 cm⁻¹ (O–W–O stretches), ~270 cm⁻¹ (bend).
- Hydrated/amorphous WOₓ: broad ~950 cm⁻¹ (terminal W=O).
- Reduction (HₓWO₃ / W⁵⁺): weakening and broadening of the 807/715 pair,
  intensity growth in the 200–400 cm⁻¹ region, darkened film.
- Keggin PW₁₂O₄₀: ~1008 and ~990 cm⁻¹ (νs/νas W=Od terminal doublet —
  the fingerprint), ~890–910 (W–Ob–W), ~520 cm⁻¹. Degradation to WOₓ
  shows as loss of the sharp ~1000 doublet.

**Other common phases** (same literature status as above — a reference
table, not an expectation of what any given sample contains):
- Si: sharp 520.7 cm⁻¹; amorphous Si a broad hump near ~480 cm⁻¹.
- TiO₂: anatase 144 (very strong), 397, 516, 639 cm⁻¹; rutile 447 and
  612 cm⁻¹.
- MoS₂ and 2D dichalcogenides: E₂g ~383 and A₁g ~408 cm⁻¹ (peak
  separation grows with layer count).
- Carbonates: calcite ν₁ 1086 cm⁻¹; sulfates: gypsum ~1008 cm⁻¹.

**Substrates and media:** oxide substrates contribute their own sharp
phonon lines (e.g., LaAlO₃ ~123 and ~152 cm⁻¹; sapphire ~418 cm⁻¹) —
identify substrate lines before assigning film chemistry. Metal (Au/Ag)
substrates give no Raman bands but a broad PL background, and can
plasmonically enhance adjacent film signal. Ionic liquids and organic
media contribute C–H stretches ~2900–3200 cm⁻¹ and ring/skeletal modes
~600–1600 cm⁻¹ that can overlap thin-film signals.

**PL vs. Raman:** any feature broader than ~200 cm⁻¹ or that dominates a
multi-thousand cm⁻¹ span is electronic (PL/fluorescence), not
vibrational — map it, but do not assign it a Raman mode.

## validation

- Internal wavenumber-calibration check: a sharp line of known position —
  Si at 520.7 cm⁻¹ (the canonical Raman calibration reference) where a
  silicon substrate is visible, or diamond at 1332 cm⁻¹ where diamond is
  present — fitted more than ~3 cm⁻¹ off across the whole map means
  calibration offset (or real stress — distinguish by whether the offset
  is uniform).
- Sanity-check the spectral axis itself: a Stokes shift beyond what the
  stated excitation wavelength and detector can physically record
  indicates a wavelength→shift conversion error in the data pipeline, not
  chemistry. Flag it and treat absolute positions as provisional rather
  than fitting bands to an impossible axis.
- Position maps should vary smoothly within a phase; salt-and-pepper
  position maps at sub-cm⁻¹ scale are fit noise, not physics. Compare the
  position spread against the spectral sampling interval — structure much
  finer than one channel is not resolvable.
- FWHM sanity: values at the fit bounds, or narrower than ~2 spectral
  channels, indicate a failed or pinned fit, not a real width.
- Phase-classification maps must be spatially coherent where the sample
  geometry says the phase is contiguous; check the map orientation against
  the stated scan geometry before describing where things are.
- Cross-check decomposition and band fits: a region NMF calls one phase
  should show that phase's fitted band parameters. Where they disagree,
  the band fit (with its per-pixel quality gates) is the authority.
- State low-SNR honestly: short exposures or sparse coverage that fail the
  per-pixel gates should be reported as "not measurable at this SNR", with
  the passing-pixel fraction, rather than papered over with map-wide
  averages. But distinguish "the per-pixel *map* failed the quality gates"
  from "the phase is unidentified": if the representative-spectrum fit
  resolved the diagnostic bands, the identification stands and must be
  reported even when the pixel-wise map did not pass — a failed map is not
  a failure to identify the phase.
