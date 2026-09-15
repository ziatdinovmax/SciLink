---
description: 'p-XRD profile fitting — the quantitative follow-up to phase identification: per-peak pseudo-Voigt fits (position, intensity, FWHM), Scherrer crystallite size and Williamson-Hall microstrain, and a Rietveld tier (phase fractions / accurate lattice; engine pending) that refines against the identified-phase CIF.'
technique: [XRD, "X-ray diffraction", "powder diffraction", pXRD]
quality_gate:
  metric: peak_region_r2
  accept_threshold: 0.90
  hard_reject_threshold: 0.55
  direction: higher_is_better
---
# p-XRD Profile Fitting Skill

## overview

Powder X-ray diffraction profile fitting — **the quantitative follow-up to
phase identification.** You normally arrive here *after* `structure_matching/
xrd` has answered "what phase(s) is this?": the identified phase(s) and their
reference CIF(s) are the prior this skill builds on. (It also runs standalone
when the phase is already known.)

The skill is organized as three **depth tiers** — go only as deep as the
question demands:

1. **Per-peak profile fit (default).** A global multi-peak (split) pseudo-Voigt
   fit (`fit_pattern`) yields calibrated peak positions, intensities, and full
   widths at half maximum (FWHMs). The identified phase's reference peak
   positions are a strong prior for *which* reflections to expect.
2. **Microstructure — size & strain.** The fitted FWHMs feed two classical
   line-broadening analyses:
   - **Scherrer** — average crystallite (coherent domain) size from a single
     peak's broadening: `D = K · λ / (β · cos θ)`.
   - **Williamson-Hall** — separates size and strain from the 2θ dependence of
     broadening: `β · cos θ = K · λ / D + 4 · ε · sin θ`.
3. **Rietveld refinement (deepest; engine pending).** Whole-pattern refinement
   against the matched CIF for quantitative phase fractions (QPA), accurate
   lattice parameters, and site occupancies. True Rietveld needs a dedicated
   engine (GSAS-II) — see *planning* for the seam. Tiers 1–2 are always
   available; tier 3 is a documented escalation, not yet wired.

Frame the work as: **identify (upstream) → profile-fit (here) → escalate to
Rietveld only when phase fractions / accurate cell / occupancies are the actual
deliverable.** Most follow-ups stop at tier 1 or 2.

When both skills are co-activated (`skill=["xrd", "xrd_profile"]`) the LLM can
chain identification and profile fitting in one script; the sequential
"identify first, then this" framing above is the same pipeline, just split
across calls.

**Out of scope:** texture / preferred-orientation correction; whole-pattern
Le Bail fitting. (Rietveld and quantitative phase fractions are **tier 3
above**, not out of scope — the structural model comes from the upstream ID
match; only the refinement *engine* is a pending GSAS-II seam.)

## planning

**You usually arrive here with an identified phase — use it as a prior.** When
`structure_matching/xrd` ran first, its result shapes this fit: the matched
phase's reference peak positions tell you *which* reflections to expect (a
sanity check on auto-detect, and a seed for a one-off single-pattern refine),
and the matched CIF is the structural model a tier-3 Rietveld refine needs. Do
**not** freeze those reference positions into a series script, though — for a
series keep `fit_pattern` auto-detecting (see "In-situ / series"); the ID tells
you the *phase*, not frozen centers. If no ID was run and the phase is already
known, proceed standalone.

**Default mechanism: global full-pattern fit.** Plan for a single
`fit_pattern` call that detects and fits all significant peaks at once
(background subtraction included), not a per-peak `fit_profile` loop. The
points below shape *that* fit — how many peaks matter, when Williamson-Hall
is admissible, which background — rather than a peak-by-peak procedure.

**Exclude a corrupting low-angle upturn — without clipping a real reflection.**
Lab patterns often have a steep air-scatter / beamstop rise in the first few
degrees. It is *not* diffraction, and it breaks the fit two ways: SNIP
background overshoots on the steep edge, and the inflated intensity range
starves prominence-based detection — the tell is `fit_pattern` returning only
1–2 peaks with a deceptively high R² on a near-empty fit. When you see that
rise, pass `fit_range=(lo, hi)` to exclude only the climb; excluded channels are
dropped from background, detection, and the fit but returned as matched, so the
residual diagnostics still work.

The hazard is over-cropping: genuine low-angle Bragg peaks are common (layered
materials, clays, MOFs, surfactants, many molecular/pharmaceutical crystals with
large d-spacings) and live in the same first few degrees, so **do not blindly use
a fixed cutoff** — look at the low-angle data and set `lo` deliberately. The
decision is **where the steep climb flattens into ordinary background**, not just
"below the lowest bump":
- **Set `lo` where the climb has flattened.** A reflection that sits on **flat**
  background just past the climb — keep it (put `lo` below it). A feature sitting
  **on the steep part of the climb itself** — exclude it with the rest of the
  climb. Do *not* lower `lo` into the climb to rescue such a feature: a peak
  buried in air-scatter cannot be reliably fit there anyway, and — because
  `fit_range` sets a single lower bound — keeping any of the steep climb keeps the
  SNIP overshoot and the starved detection across the **whole** pattern, so you
  trade one ambiguous low-angle peak for a bad fit everywhere.
- **Read the result, not your intent.** If the fit comes back with only a handful
  of peaks and/or a large oscillatory residual in the first few degrees, `lo` is
  still **inside the climb** — raise it to where the background goes flat and
  re-fit. A clean fit with many reflections and a structureless low-angle residual
  means `lo` is right.
- **A dominant low-angle feature that suppresses the rest of the pattern is
  air-scatter — exclude it.** Prominence-based detection scales its floor to the
  tallest peak in the window, so one enormous low-angle feature (the air-scatter /
  beam-stop maximum, often *orders of magnitude* above every real reflection)
  starves detection of the weaker peaks. The tell is a fit that captures only the
  two or three tallest peaks while the rest of the pattern — which plainly shows
  reflections in the raw data — sits unmodeled at baseline. When you see that, the
  giant low-angle feature is air-scatter, not a reflection you must keep: raise
  `lo` above it and re-fit; recovering the full set of reflections confirms it.
  The trigger is **suppression of otherwise-visible peaks**, not size alone — a
  genuinely strong reflection leaves the rest of the pattern detectable and does
  not fire this rule (e.g. a 100×-dominant peak whose neighbours are still fit is
  fine; it is only a problem when the neighbours vanish from the fit).
- A genuine reflection truly buried on the climb is a **background problem**, not
  a `fit_range` problem: report that the low-angle region needs a dedicated
  background (or a measurement with a beam stop / variable slits) rather than
  forcing the single-bound crop to do both jobs.

The same knob (with the same caution) excludes a detector artifact or a known
contaminant region. For a series, set one window on the establishing frame and
reuse it on every frame.

**How many peaks to fit:**
- Scherrer alone: 3–5 strongest, well-isolated peaks is enough — report
  size per peak and the mean.
- W-H plot: at least 5 peaks spread across the 2θ range. Peaks clustered
  at similar 2θ collapse the regression's lever arm; the slope estimate
  becomes degenerate. Refuse W-H and fall back to peak-by-peak Scherrer
  when sin θ range < 0.1.

**Background: default to SNIP; do not switch to polynomial to "keep a
pedestal".** `fit_background(method='snip')` (also `fit_pattern`'s default) is
the right choice for almost every p-XRD pattern: it follows smooth amorphous
humps, broad diffracted pedestals, and fluorescence floors without imposing a
shape, and its iteration count (`snip_iterations='auto'` sweeps it) sets how much
of a broad pedestal it treats as background. So if a broad pedestal under a
cluster is being over-subtracted (crystalline apexes not reaching full data
height), **tune the SNIP iterations** (fewer iterations leaves more of the
pedestal) — do **not** change method. Reach for `method='polynomial'` only for a
specific, known smooth scattering on a genuinely flat baseline (e.g. capillary
scattering), **never** to preserve an amorphous pedestal: a Chebyshev/polynomial
background oscillates and goes **negative between peaks**, producing a
pathological model that dips below zero in the gaps — almost never what XRD data
needs.

**Line shape: pseudo-Voigt as default.** A pseudo-Voigt mixes Gaussian
and Lorentzian contributions through a single mixing parameter `eta` —
**eta = 0 is pure Gaussian, eta = 1 is pure Lorentzian** (lmfit
convention). Pseudo-Voigt captures the experimental and physical
broadenings of typical p-XRD patterns without the slower numerical
convolution of a true Voigt. Switch to pure Lorentzian (`model='lorentzian'`)
only when fitted eta consistently lands ≥ 0.9 across peaks (rare).

**Instrumental broadening subtraction.** Both Scherrer and Williamson-
Hall require the *sample* broadening, not the total. Subtract the
instrumental FWHM in quadrature:
`β_sample² = FWHM_total² − FWHM_instrumental²`. The instrumental FWHM
comes from a standard reference pattern (LaB₆, Si, Al₂O₃) measured on
the same instrument. Pass it as the `instrumental_fwhm_deg` argument to
`scherrer` and `williamson_hall`. When unknown, default to 0.0 and flag
the result as an upper-bound on broadening (lower-bound on size).

**Peak windowing.** Fit each peak over a 2θ window roughly 5–8 × the
expected FWHM, centered on the peak. Too narrow → background slope
biases the fit; too wide → neighboring peaks intrude. The `window_deg`
argument to `fit_profile` defaults to 1.0° (typical CuKa FWHM ~0.2°);
widen for nanocrystalline samples with FWHM > 0.5°.

**Overlapping peaks.** When two peaks lie within 1.5 × FWHM of each
other, fit them jointly (two pseudo-Voigt components in one
`fit_profile` window) rather than sequentially. Sequential fitting
double-counts the overlap region.

**Scherrer K constant.** Default `K = 0.9` (spherical crystallites,
FWHM-based). Use `K = 0.94` for cubic crystallites or when explicit in
the literature reference. The choice is a < 5% correction; do not
agonize over it.

**Pairing with `structure_matching/xrd`.** When both skills are active,
the recommended chaining is: `extract_peaks` (from `structure_matching/
xrd`) seeds peak centers → `fit_profile` per peak → use the refined
FWHMs as `exp_peaks={'positions': [...], 'amplitudes': [...],
'fwhms': [...]}` for `score_xrd_match_robust`. The score gets sharper
broadening per peak instead of the default uniform FWHM, which matters
for nanocrystalline patterns with peaks several times broader than the
0.15° default.

**Escalating to Rietveld (tier 3).** Reach for Rietveld only when the
deliverable is **quantitative phase fractions (QPA), accurate refined lattice
parameters, or site occupancies** — not for size/strain, which tiers 1–2 cover.
Rietveld refines the *whole pattern* against a structural model, so it requires
the matched CIF from the upstream identification as the starting model (another
reason ID comes first). True Rietveld is **not** in pymatgen; it needs a
dedicated engine — GSAS-II via `GSASIIscriptable`. That plugs in through the
same pluggable-engine pattern already established for simulation
(`structure_matching/xrd/simulate_xrd.py`'s `_ENGINES` registry): a
`refine_rietveld(cif, exp_data, ...)` tool behind a lazy import + optional
`gsas` extra, leaving downstream scoring/reporting unaffected. **Until that
engine is wired, do not attempt a whole-pattern refine with the kinematic
tools** — report tiers 1–2 and recommend Rietveld as the explicit next step,
naming the matched CIF as the model to refine.

*Refinement order (engine-pending — the protocol for when `refine_rietveld`
exists; do not apply it to the tier-1/2 empirical `fit_pattern`, which has no
such refinement groups).* A manual Rietveld runs in a set order, and that order
is most of the skill: the logic is identical in GSAS-II or TOPAS, only the
parameter names differ. **Nothing is freed all at once** — refine one group at a
time, holding everything else at its current value and carrying the converged
values forward; freeing everything together tends to diverge or settle in a
false minimum on overlap-heavy data. The order:
1. **Scale + background** first — just to get the calculated curve roughly onto
   the observed one.
2. **Sample/setup group** (the parameters with no "correct" value to look up —
   they are whatever the packing and alignment produced that day): zero-shift /
   sample displacement for peak positions, the profile terms for width and
   shape, and preferred orientation if the intensities are off.
3. **Structural group** (a separate bucket, anchored by the published CIF for a
   known phase): the cell refines only a little for a fixed phase at fixed
   conditions; atomic coordinates and occupancies stay fixed unless the data
   really justify touching them.

The step-2 corrections are exactly the ones that quietly absorb error from
elsewhere and still produce a clean Rwp — judge each by the sample-and-mounting
priors in **validation** (a large displacement or preferred-orientation or
lattice change is suspect unless the sample/mounting/conditions make it real),
not by the fit statistic.

## implementation

**Default path: one global fit with `fit_pattern`.** Prefer `fit_pattern`
over a per-peak `fit_profile` loop. It detects *all* significant peaks in
one pass and fits them **simultaneously** on a shared baseline, so the
reported R² and residual are **global** (over the whole pattern) — the same
quantity the verifier judges. A per-peak loop reports per-window R², which
hides every unmodelled reflection as a global-residual spike and triggers
avoidable refinement iterations. `fit_pattern` seeds each amplitude from the
measured apex (sharp peaks are never clipped) and scales parameters
internally, so a busy pattern fits in ~1 s.

**CRITICAL workflow:**

1. Load experimental 2θ + intensity arrays.
2. `fit_pattern` (handles background + detection + global fit in one call).
3. Per-peak Scherrer crystallite size via `scherrer` on the returned FWHMs.
4. If ≥ 5 peaks span a useful 2θ range, run `williamson_hall`.
5. Emit `FIT_RESULTS_JSON: {...}` carrying the **global** R² from
   `fit_pattern` (not a mean of per-window R²s) plus per-peak results.

**Complete full-pattern template:**

```python
import json
import numpy as np

from scilink.skills._shared.curve_fitting_tools import load_curve_data
from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
from scilink.skills.curve_fitting.xrd_profile.scherrer import scherrer
from scilink.skills.curve_fitting.xrd_profile.williamson_hall import williamson_hall

# ---- Step 1: Load ----
data = load_curve_data(DATA_PATH)  # ndarray with X in col 0, Y in col 1
two_theta = np.asarray(data[:, 0], dtype=float)
intensity = np.asarray(data[:, 1], dtype=float)

WAVELENGTH_ANGSTROM = 1.5406  # CuKa1; replace from metadata if available
INSTRUMENTAL_FWHM_DEG = 0.05  # from LaB6/Si standard; 0.0 if unknown

# ---- Step 2: One global multi-peak fit (background handled inside) ----
# Let fit_pattern DETECT the peaks (do not hardcode centers): the same call
# then generalises unchanged to every frame of a series. See "In-situ / series".
fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    background='snip',          # 'none' if data is already background-subtracted
)
peaks = fit['peaks']            # each: center, fwhm, amplitude, area, eta
r_squared = fit['r_squared']    # GLOBAL R²

# Save the fit overlay (fit.npy) on the RAW-intensity scale so it matches the raw
# data. fit['fit_curve'] is on the BACKGROUND-SUBTRACTED scale; saving THAT makes a
# good fit look collapsed (peaks at a fraction of height, gaps near zero) whenever
# the background is large — use fit['fit_curve_raw'] (= fit_curve + background).
import numpy as np
np.save('fit.npy', np.asarray(fit['fit_curve_raw'], dtype=float))

# ---- Step 3: Per-peak Scherrer size ----
sizes_nm = []
for p in peaks:
    s = scherrer(
        fwhm_deg=p['fwhm'],
        two_theta_deg=p['center'],
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )
    sizes_nm.append(s['size_nm'])
mean_size_nm = float(np.mean(sizes_nm)) if sizes_nm else None

# ---- Step 4: Williamson-Hall (optional) ----
wh_input = [{'two_theta': p['center'], 'fwhm': p['fwhm']} for p in peaks]
wh = williamson_hall(
    peaks=wh_input,
    wavelength_angstrom=WAVELENGTH_ANGSTROM,
    instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
) if len(wh_input) >= 5 else None

# ---- Step 5: Emit ----
print("FIT_RESULTS_JSON: " + json.dumps({
    "peaks": [
        {k: p[k] for k in ('center', 'fwhm', 'amplitude', 'area', 'eta')}
        for p in peaks
    ],
    "scherrer_mean_size_nm": mean_size_nm,
    "scherrer_per_peak_nm": sizes_nm,
    "williamson_hall": wh,
    "fit_quality": {
        # peak_region_r2 is the GATE METRIC: R² over the channels that carry
        # diffracted intensity, so a correct fit of a weak/noisy pattern is not
        # failed by the noise-only background channels. fit_pattern returns it.
        "peak_region_r2": fit['peak_region_r2'],
        "r_squared": r_squared,                       # GLOBAL (reported for context)
        "residual_rms_over_noise": fit['residual_rms_over_noise'],
        # Worst single LOCAL misfit in noise units. R²/peak_region_r2/RMS average
        # and hide a localized error (a mis-shaped sharp apex on a tall peak); this
        # exposes it. Read against residual_rms_over_noise (>> RMS = localized ->
        # look at the figure); its scale rises with dynamic range, so it is an
        # attention signal, not a thresholded verdict.
        "max_abs_residual_over_noise": fit['max_abs_residual_over_noise'],
        "n_peaks_fitted": fit['n_peaks'],
    },
}))
```

**A high `peak_region_r2` is necessary but not sufficient — use
`max_abs_residual_over_noise` to decide where to look.** The gate metric (like
global R² and RMS) averages over the whole pattern, so on a high-dynamic-range
pattern — one or two giant peaks dwarfing the rest — a localized error such as a
mis-shaped sharp apex is invisible to it (`peak_region_r2 ≈ 0.99` while that apex
is far off). `max_abs_residual_over_noise` is the **worst single local misfit**,
the complement of the averaged statistics. Read it *against*
`residual_rms_over_noise`: when it is much larger than the RMS, the misfit is
**localized** (concentrated at a few channels) rather than spread out — that is a
prompt to **look at the residual figure at that location**, not a verdict.
Do **not** attach an absolute threshold to it: its scale rises with dynamic range
(a tiny fractional residual on a giant peak is many σ), so a large value is
*expected* on very sharp, intense peaks and is not by itself a failure. The
figure decides: a single sharp apex is the split pseudo-Voigt's irreducible
profile limit (accept, and **report the residual honestly rather than claiming a
clean fit** — do not refine endlessly); a residual that is an unmodelled
reflection or a missed shoulder is a real miss (refine). The value's job is to
stop a high `peak_region_r2` from being rubber-stamped without that look.

The gate scores `peak_region_r2`. A large global-`r_squared` ↔ `peak_region_r2`
gap is **only** a benign low-SNR artifact when you have *verified* both that the
pattern is genuinely low-SNR (weak counts over a noisy background) **and** that
the residual in the gap regions is structureless (noise-only). Do not assume the
gap is benign: on a high-SNR pattern a low global `r_squared` should be presumed
a **real** problem (a missed reflection or a mis-shaped peak) until the residual
is checked and found featureless. Conversely, when `peak_region_r2` itself is low,
real reflections are mis- or un-modelled — that is **not** rescued, so refine
(lower `prominence_frac`/`min_distance_deg` and re-run the single global
`fit_pattern`). `peak_region_r2` equals the global R² on a high-SNR pattern, so
on clean data there is no gap to explain.

**In-situ / series use — lock the method, not the values.** `fit_pattern` is
one fast call per frame, so it drops straight into the agent's per-spectrum
series loop. The locked series script **must call `fit_pattern` with
`peak_centers=None` (auto-detect)** — lock the *recipe* (the `fit_pattern` call
and its settings), never a hardcoded list of peak centers. Auto-detect re-finds
the peaks on every frame, so the one script follows peak shifts (thermal
expansion), intensity changes (phase fraction), and appearance/disappearance,
and composes with the agent's regime-segmentation and adaptive-refit paths. A
**value-locked** script — hardcoded `SEED_CENTERS` / an explicit `peak_centers`
list frozen from frame 1 — does **not** generalise: centers drift out of their
windows even within one phase, and break entirely across a reaction or
transition (measured: a list locked from a post-transition frame scored
R²≈0.5–0.7 on pre-transition frames). Frame-to-frame **consistency comes from
the identical method plus aligning the detected peaks across frames in
interpretation, not from frozen centers.** Only pass an explicit `peak_centers`
for a one-off re-fit of a single known-stable pattern — never as the series
default.

**Phase identity and fractions across a series are the `xrd` skill's job.**
This skill answers peak shapes/positions/widths per frame; WHICH phases each
frame contains and HOW MUCH of each is answered by the structure-matching
`xrd` skill's series workflow — identify the series endpoints, then
`track_phase_series` (or the lock-file pattern) scores every frame against
that fixed endmember set, returning per-frame phase shares, the transition
window, and residual alerts. The two compose: its transition-window frame
indices tell this skill's regime segmentation where to expect a model
change, and its lattice-scale drift corroborates peak-shift trends fitted
here. To COUPLE the two over a series — attribute the peak-evolution trends
fitted here to the phases identified there, and cross-check the transition —
run both passes and join them with the `xrd` skill's
`reconcile_series_phases`. That is the recommended in-situ deliverable when
the phases might not all be in a database: profile fitting (here) keeps
producing trends where identification hits the "not in any database" wall, so
the reconciliation labels what it can and reports the rest as honestly
unidentified.

**The detection settings ARE part of the recipe — tune them, then freeze them.**
`prominence_frac` and `min_distance_deg` are *settings*, not *values*: tune them
on the establishing frame so every visible reflection is modelled, then carry the
chosen numbers into the series call. Doing so is locking the *method* (correct);
it is the peak *centers* that must never be frozen. Watch `prominence_frac` in
particular — its floor scales with the pattern's intensity range, so the weakest
(usually high-angle) reflections sit just above it and are the first to be
dropped. **A residual statistic will not warn you**: a weak peak on a noisy
high-angle floor reads only a few σ even when it is entirely unmodelled, so the
fit can miss real reflections while R²/peak_region_r2/RMS all look clean. Verify
coverage **by eye against the data across the full 2θ range** — a log-scale
data-vs-fit overlay — not by the residual number: if the raw data plainly shows
reflections the model leaves at baseline, lower `prominence_frac` (0.02 → 0.01 →
0.005) until they are caught, then freeze that value for the series.

For speed across a long series: the default `snip_iterations='auto'` sweeps
a few background widths per frame (~4× the fit cost). Once the establishing
frame reports its choice (in `background_method`, e.g. `snip(iterations=10)`),
pass that integer as `snip_iterations` on the remaining frames to skip the
sweep — back to ~1 s/frame with the same background treatment.

**Unresolved doublet? Tune detection, don't drill.** `fit_pattern` already
fits all peaks jointly with an asymmetric (split pseudo-Voigt) shape, so it
resolves overlaps and peak asymmetry in one consistent global model. If a tight
doublet is smeared because auto-detect merged it, fix it the **method-locked**
way: lower `min_distance_deg` (so closely-spaced maxima are detected separately)
and/or lower `prominence_frac` (so the weaker partner is found) — these stay
generalisable across a series because the recipe still auto-detects per frame.
Do **not** splice in a separate `fit_profile` window fit: `fit_profile` uses a
*symmetric* profile, so its result conflicts with the global split-PV fit and
reintroduces exactly the asymmetric residual the global model removed (this
drives avoidable verifier iterations). Reserve `fit_profile` only for one-off
inspection of a single peak
outside the main fit.

**NumPy compatibility.** Use `np.trapezoid` (not removed `np.trapz`).

## interpretation

**Crystallite size ranges from peak FWHM (CuKa, 2θ ≈ 30°):**

| FWHM (deg) | Size (nm) | Regime |
|------------|-----------|--------|
| < 0.10 | > 100 | Coarse / well-crystallized; instrumental-limited |
| 0.10–0.30 | 30–100 | Typical microcrystalline |
| 0.30–1.00 | 10–30 | Nanocrystalline |
| > 1.00 | < 10 | Strongly nano / poorly crystallized |

These are rough; the exact size depends on 2θ via the cos θ factor. For
peaks at higher 2θ, the same FWHM in degrees corresponds to a smaller
crystallite.

**Strain vs size from Williamson-Hall slope:**
- Slope ≈ 0 (flat W-H plot): broadening dominated by crystallite size;
  strain is negligible. Report size only.
- Positive slope: real microstrain. Slope value `m = 4ε` gives strain
  directly. Typical ε in [0.0005, 0.005] for metals and oxides;
  > 0.01 suggests defects, alloying, or measurement issues.
- Negative slope: usually unphysical — typically indicates instrumental
  miscalibration, bad background subtraction, or peak misassignment.
  Re-check fits and instrumental FWHM before reporting.

**Inconsistencies to flag:**
- Per-peak Scherrer sizes vary by > 3×: anisotropic broadening
  (anisotropic crystallite shape, hkl-dependent strain). Report the
  *range* of per-peak sizes, not just the mean, and note that an
  isotropic Scherrer mean is an oversimplification.
- W-H linearity R² < 0.7: the linear model doesn't apply. Possible
  causes: anisotropic broadening, mixed-phase sample (some peaks
  broadened by phase A, others by phase B), or fits with large
  uncertainty on FWHM.

**Quantitative confidence.** Scherrer gives an *average* coherent
domain size; for log-normal size distributions the Scherrer size is
closer to a volume-weighted mean than a number mean. Quote sizes to
2 significant figures; ±15% is the typical accuracy on a well-
calibrated instrument.

**Cross-skill follow-up.** If profile fitting succeeded but the
crystal phase has not yet been identified, recommend running
`structure_matching/xrd` next on the same pattern. The
refined FWHMs from this skill can be passed in to sharpen the scoring.

## validation

**Know when the fit is done.** `fit_pattern`'s default split (asymmetric)
pseudo-Voigt already captures sharp-peak asymmetry, so once the global R² ≥ ~0.99
and every visible reflection is modelled (no *unmodelled*-peak spikes in the
residual), small residual at the apex of the few sharpest, highest-count peaks
is the irreducible profile limit — accept it rather than spending refinement
iterations chasing it. Do **not** re-fit individual peaks with a symmetric
`fit_profile` window: it conflicts with the global split-PV model and
reintroduces asymmetric residual, *worsening* the fit and adding iterations. If a
peak is genuinely missing, lower `prominence_frac` (and/or `min_distance_deg`) so
auto-detect catches it and re-run the single global `fit_pattern` — keep the
recipe auto-detecting (so it still generalises frame-to-frame), not spliced
sub-fits or hardcoded centers.

**Gate on `peak_region_r2`, read the global R² as context.** The whole-pattern
`r_squared` is depressed by noise-only background channels on a weak/noisy
pattern, so a correct fit can score a misleadingly low global R² (e.g. a faint
powder pattern whose Bragg peaks are well fit can sit at global R² ≈ 0.5 with
`peak_region_r2` ≈ 0.9). Score acceptance on `peak_region_r2`:
- `peak_region_r2` ≥ 0.90 accepts; 0.55–0.90 marginal (the verifier decides on
  residual structure); below 0.55 reject.
- A large `r_squared` ↔ `peak_region_r2` gap with a **structureless (noise-only)
  residual** = low-SNR pattern, fit is good → accept. The same gap with
  *peaked* residual (unmodelled-reflection spikes) = real miss → refine: lower
  `prominence_frac`/`min_distance_deg` and re-run the single global `fit_pattern`.
  `peak_region_r2` does not rescue a fit that genuinely misses reflections, so a
  low `peak_region_r2` is a true reject, not a metric artifact.
- FWHM sanity: 0.05° (typical instrumental floor for a lab CuKa
  diffractometer) ≤ FWHM ≤ 2.0° (very small crystallites; peak overlap
  dominates above this).
- `eta` (Gaussian-Lorentzian mixing) must be in [0, 1]. A fit that
  returns eta exactly at 0 or 1 with high uncertainty usually
  indicates the data don't constrain the mixing — fall back to a
  pure Gaussian or pure Lorentzian fit and compare R².
- Amplitude must be positive. Negative fitted amplitudes mean the
  initial center was wrong or background subtraction overshot —
  re-extract peaks and re-window.

**Scherrer sanity:**
- Size > 1000 nm: the peak is instrument-limited, not crystallite-
  limited. Report as "size > 100 nm (resolution-limited)" rather than a
  number.
- Size < 1 nm: physically unreasonable for a crystalline solid. Indicates
  one of: instrumental FWHM not subtracted, severe strain mistaken for
  size broadening, or peak overlap in the fit window.

**Williamson-Hall sanity:**
- W-H R² ≥ 0.9 for confident size + strain decomposition.
- W-H R² in [0.7, 0.9] reports the values with a caveat that the
  decomposition is unstable.
- W-H R² < 0.7 do not report the decomposition; fall back to per-peak
  Scherrer and report the size range only.

**Instrumental-broadening subtraction sanity:**
- If `β_sample² ≤ 0` (instrumental FWHM ≥ measured FWHM), the peak is
  resolution-limited. Report "size below sensitivity limit
  (~ Scherrer with β = instrumental)" rather than NaN or a complex
  number.

**Sample-and-mounting priors — the fit statistic can't tell you these, the
sample does.** The corrections that absorb position/intensity error (sample
displacement, preferred orientation, and a refined lattice) will quietly soak up
error from elsewhere and still leave a clean R²/Rwp, so a good fit statistic does
*not* mean the correction is real. Judge each against what the sample is and how
it was mounted, not against the fit:
- **Sample displacement / zero-shift:** ~0 on a properly seated zero-background
  holder. A large fitted displacement points to a misaligned sample, not a
  correction worth trusting — flag it rather than reporting it as physical.
- **Preferred orientation / intensity mismatch:** on a loosely packed random
  powder there should be little texture, so a large PO correction usually means
  something *else* is wrong (it is absorbing error); on a flat plate of platy or
  needle crystals the texture is real and a correction along the right axis is
  legitimate. Use the known morphology/mounting to decide which case you are in.
- **Lattice change:** for a fixed phase at fixed conditions the cell should move
  only a little; a large shift is suspect — *except* in a variable-temperature /
  in-situ run, where the cell genuinely moves and tracking it is the point.
