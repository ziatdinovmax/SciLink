---
description: 1D NMR curve fitting (solution and solid-state MAS) for ¹H, ¹³C, ¹⁹F, ³¹P, ²³Na, ²⁷Al, ¹⁷O, ⁶⁷Zn and related nuclei — peak deconvolution with Voigt/pseudo-Voigt lineshapes, rolling-baseline correction, phase/sign handling, MAS spinning-sideband awareness, and second-order quadrupolar central-transition lineshape fitting (C_Q, η_Q) for solid half-integer nuclei.
technique: [NMR, "nuclear magnetic resonance", "solid-state NMR", "ssNMR", "MAS NMR", "magic-angle spinning"]
quality_gate:
  metric: peak_region_r2
  accept_threshold: 0.85
  hard_reject_threshold: 0.30
  direction: higher_is_better
---
# NMR Curve Fitting Skill (1D, solution + solid-state MAS)

## overview

One-dimensional NMR spectra, processed and Fourier-transformed (a real
spectrum vs. chemical shift in ppm, decreasing left→right). Covers both
**solution-state** (narrow, near-Lorentzian lines) and **solid-state MAS**
(broader lines, spinning sidebands, and — for half-integer quadrupolar nuclei —
second-order quadrupolar lineshapes). Nuclei seen here: ¹H, ¹³C, ¹⁹F, ³¹P
(spin-½) and ²³Na, ²⁷Al, ¹⁷O, ⁶⁷Zn (half-integer quadrupolar).

**The single most important triage is solution vs. solid**, because it
dictates the lineshape:

- **Solution-state** (incl. quadrupolar nuclei like ¹⁷O/²³Na/⁶⁷Zn in liquids):
  fast molecular tumbling averages anisotropic interactions to zero, so every
  resonance is a near-perfect **Lorentzian** (or Voigt with a small Gaussian
  shim contribution). Fit a sum of Voigts. Do NOT use the quadrupolar lineshape.
- **Solid-state MAS**: lines are broadened by residual anisotropy; spin-½ nuclei
  give Voigt-like centrebands flanked by **spinning sidebands**; half-integer
  quadrupolar nuclei show an asymmetric **second-order quadrupolar
  central-transition** lineshape that carries C_Q and η_Q.

**Resolved J-coupled multiplets** (a doublet/triplet/quartet from one chemical
environment, e.g. a CF₃ carbon split by fluorines) are fit as ONE constrained
manifold with `fit_jcoupled_multiplet`, not as independent Voigts — see the
routing in Implementation.

**Out of scope (v0):** 2D experiments; second-order / non-binomial multiplet
patterns (roofing, second-order strong coupling); satellite transitions, MQMAS,
CSA tensor extraction from sideband intensities; relaxation/T1–T2 series (a
separate "integral vs. delay" task, not a single-spectrum fit); raw-FID
reprocessing (this skill works on the processed spectrum).

## planning

**Hard prerequisite: the Larmor frequency of the observed nucleus (MHz).** It
sets the ppm↔Hz conversion (linewidths, sideband spacing) and the ν_Q²/ν_L
scaling of quadrupolar lineshapes. Take it from the spectrometer metadata
(`spectrometer_frequency_MHz`); never guess. The nucleus identity
(`nucleus`) and, for MAS data, the spinning rate (from the acquisition metadata
or the experiment title, in kHz) are the other key metadata.

**Step 1 — Check the phase / sign.** NMR spectra can come phased with a
negative-going peak (a 180° phase error, or the real part of an unphased
spectrum). If the dominant feature is a **dip** below baseline, fit a *signed*
amplitude (allow negative) or re-phase — never let a baseline-fitter "explain"
an inverted peak as background.

**Distinguish a real resonance from an instrumental artifact by WIDTH, and do
not discard a real one.** A feature that rises a few × the noise level and spans
**multiple points** (finite width in ppm) is a genuine resonance — fit it
(re-phasing or signed-amplitude if inverted), even at low SNR. Only a
**single-point** spike sitting exactly at the carrier (x ≈ 0 ppm) is a DC/centre
glitch to be masked; an edge spike at the spectrum boundary is a fold artifact.
A weak, broad, or inverted line (common for low-γ quadrupolar nuclei like ⁶⁷Zn,
²⁵Mg, ³³S near their detection limit) is still a resonance — recover it rather
than declaring "no signal" and fitting baseline only.

**But abstain when the data cannot support a fit — the counterweight to "recover
weak lines".** On a near-noise spectrum a flexible Voigt+baseline model will
"succeed" by fitting the noise, reporting a confident lineshape and a high
peak-region R² for a measurement that is not there. Before trusting a low-SNR
fit, run `assess_detection` (see Implementation): it returns a verdict —
**detected** (fit and report), **marginal** (report the shift tentatively / as an
upper bound, no precise linewidth or integral), or **absent** (report a
NON-DETECTION with an intensity upper bound; do not report a fitted peak). A
resonance must be either a resolved contiguous feature OR statistically justified
over a flat baseline (F-test); a high R² alone does not make it real.

**Step 2 — Correct a rolling baseline BEFORE fitting.** Processed spectra
(especially ¹H MAS) frequently carry a broad oscillating baseline distortion
(first-point / digital-filter / probe-background artifact — wide humps and deep
lobes far from any real resonance). This is *not* chemistry and a peak model
cannot absorb it. Identify signal-free regions (away from all peaks and
sidebands), fit a low-order polynomial **or** a smoothing spline through only
those regions, and subtract. Re-baselining is sample-dependent: some spectra are
flat and need none. (See Implementation for the recipe.)

**Step 3 — Choose the lineshape by regime:**

| Situation | Model |
|---|---|
| Solution-state, any nucleus | Sum of **Voigt** (Lorentzian-dominant); use a pure Lorentzian only if the Gaussian width refines to ~0 |
| Solid MAS, spin-½ (¹³C, ³¹P, ¹⁹F, ¹H) | Sum of **Voigt** centrebands; account for sidebands (Step 4) |
| Solid MAS, half-integer quadrupolar | **`fit_quad_ct`** — attempt it by *default* (C_Q/η_Q is the deliverable); do not pre-judge "looks symmetric → skip it". The tool returns `Cq_resolved`: if False, fall back to a **pseudo-Voigt** and report FWHM, not C_Q (Step 5). A broad, washed-out *disordered* line → **`fit_quad_czjzek`**. |

Default to **Voigt over pure Lorentzian** — real NMR lines carry a Gaussian
(shim/disorder) contribution, and a pure Lorentzian over-peaks the apex.

**Step 4 — Spinning sidebands (MAS only).** A regular comb of weak peaks spaced
by exactly the **MAS rate** (ν_r in Hz → ν_r/ν_L ppm) flanking each centreband
is spinning sidebands, *not* distinct chemical sites. Two valid strategies: (a)
crop to the centreband and fit only that, or (b) fit the centreband plus the
sidebands as satellites at ±k·ν_r sharing the centreband's shape. **Either way,
any integrated-intensity / quantification claim must include sideband intensity**
— excluding it undercounts. Verify the spacing equals the known MAS rate.

**Step 5 — Let the tool, not the eye, decide whether C_Q is determined.** For a
solid half-integer quadrupolar line, *run* `fit_quad_ct` rather than pre-judging
from the apparent symmetry — the second-order lineshape pins C_Q and η_Q only
when the quadrupolar broadening exceeds the instrumental/disorder broadening, and
the tool reports this for you via `Cq_resolved`. When `Cq_resolved=True`, report
C_Q/η_Q. When `Cq_resolved=False` (a narrow, near-symmetric line where C_Q and the
linewidth are **degenerate**), treat C_Q as an upper bound and report the
pseudo-Voigt FWHM as the primary descriptor, recommending MQMAS / satellite data
to pin C_Q. Either way you have *attempted* the physics and let the fit — not a
visual guess — set whether C_Q is reportable.

**Series fits (composition / temperature / treatment ladders).** Lock the *form*
— lineshape type and baseline strategy — across the series so shifts and widths
stay comparable, and let shift/width/amplitude (and C_Q/η_Q where resolved) float
per spectrum. **Do NOT lock the number of components.** The structure can change
along the series — a new phase or site appearing/disappearing, a line splitting
under slow exchange, a polymorph emerging — and that change is usually the
scientific signal, not noise to suppress. At each point let the data set the
component count under the same parsimony test you use for a single spectrum (add
a component only if it materially improves the peak-region fit; drop one that
refines to negligible amplitude). Report **two** things across the series: the
per-sample parameter column (e.g. δ_iso vs x, C_Q vs x, linewidth vs T) **and the
component count vs the variable** — the value(s) of the variable where the count
changes mark the transition / onset. Do not assume the number of components is
fixed (or that it is one); let it be whatever the data justify at each point.

## implementation

**CRITICAL ordering: metadata → phase/sign → baseline → lineshape fit.**

**Rolling-baseline correction recipe** (apply when Step 2 flags a distorted
baseline; skip for a flat one):

```python
import numpy as np
from scipy.interpolate import UnivariateSpline

# x = ppm (ascending), y = intensity. Identify signal-free points: those below a
# robust threshold over a median-smoothed |y|, EXCLUDING a window around every
# peak/sideband. A spline through only those points captures the broad roll.
ynorm = np.abs(y - np.median(y))
thr = np.median(ynorm) + 2.0 * 1.4826 * np.median(np.abs(ynorm - np.median(ynorm)))
free = ynorm < thr                      # signal-free mask (coarse)
# widen exclusion around contiguous signal regions if needed, then:
spl = UnivariateSpline(x[free], y[free], k=3, s=len(x[free]) * np.var(y[free]))
y_corr = y - spl(x)                      # baseline-subtracted spectrum
```

A low-order polynomial (`np.polyfit` over `x[free]`, degree 2–4) is the simpler
alternative when the roll is gentle. Validate by eye: the corrected baseline
should be flat and centred on zero away from peaks.

**Voigt multi-peak fitting** (solution, and solid spin-½ centrebands). When the
spectrum has two or more overlapping environments (a shoulder, a sharp+broad
pair, a multi-site cluster), do NOT guess the component count — let the data
choose it with the skill tool, which adds peaks at the largest residual and
keeps each only if it materially improves the peak-region R² and is physically
distinct:

```python
from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt
# pass mas_rate_ppm (= spinning_rate_Hz/ν_L) so sidebands aren't fit as sites;
# tune improve_thresh down to catch a faint shoulder, or min_amp_snr up if noisy.
res = fit_multipeak_voigt(x.tolist(), y.tolist(), baseline=baseline.tolist(),
                          mas_rate_ppm=MAS_PPM)   # see the tool's parameter docs
```

For a single, clearly isolated symmetric line a one-component Voigt is fine.
Allow signed amplitudes if Step 1 flagged an inverted peak. Convert fitted widths
to Hz with the Larmor frequency for reporting. (Edge case the tool does not
robustly separate: two peaks at the *same* shift differing only in width
— a sharp+broad co-centred pair; fit those manually if present.)

**J-coupled multiplet fitting** (a resolved doublet/triplet/quartet from ONE
chemical environment — equally-spaced lines with a binomial 1:1 / 1:2:1 / 1:3:3:1
intensity ratio). Do NOT fit these as independent Voigts (badly under-determined,
unstable, and it discards the coupling). Use the constrained manifold fitter,
which shares one shift, one J, one linewidth across all lines (6 parameters total)
and returns J directly:

```python
from scilink.skills.curve_fitting.nmr.multiplet import fit_jcoupled_multiplet
# crop x/y to the multiplet. Pass the line count you read off the spectrum as
# `multiplicity` (4 for a quartet); omit it to auto-detect. nu_L_MHz gives J in Hz.
res = fit_jcoupled_multiplet(xroi.tolist(), yroi.tolist(), baseline=base.tolist(),
                             multiplicity=4, nu_L_MHz=NU_L)
# res['J_hz'], res['center_ppm'] (coupling-free shift), res['fwhm_hz']
```

A multiplet is one *environment*: two coupled species at different shifts are two
calls (or `fit_multipeak_voigt` for the centroids). This handles first-order
binomial coupling only — for second-order / roofing patterns fall back to
`fit_multipeak_voigt`.

**Disordered (Czjzek) quadrupolar lineshape.** A *disordered* solid — amorphous,
glassy, a solid solution, or heavily defective — does not have one well-defined
(C_Q, η); every site sees a slightly different EFG, so the central transition is
the average over a *distribution* of (C_Q, η). The line is then broad and
**smoothly skewed with the sharp single-site horns washed out**, and neither a
pseudo-Voigt nor a single-site `fit_quad_ct` reproduces it (they leave systematic
residual — `residual_structured=True`). Fit it with the Czjzek model, which
returns the EFG-distribution width σ_Cz:

```python
from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_czjzek
res = fit_quad_czjzek(ppm_roi.tolist(), y_roi.tolist(), nu_L_MHz=NU_L, I=1.5)
# res['parameters']: sigma_cz_MHz (disorder width), mean_Cq_MHz, delta_iso_ppm
```

Decide single-site vs Czjzek by the lineshape and the residual: visible, sharp
horns → `fit_quad_ct`; washed-out, smoothly skewed, or `fit_quad_ct` leaves
`residual_structured=True` → `fit_quad_czjzek`. (Solution-state quadrupolar
nuclei tumble fast → plain Voigt, neither model.)

**Routing — which fitter:** one symmetric uncoupled line → single Voigt; a
resolved J-coupled multiplet (equal spacing, binomial ratio) → `fit_jcoupled_multiplet`;
several distinct chemical shifts → `fit_multipeak_voigt`; a single *asymmetric
quadrupolar* powder line with sharp horns → `fit_quad_ct`; a *broad, disordered*
quadrupolar line (washed-out horns) → `fit_quad_czjzek` (do NOT mimic a
quadrupolar lineshape with a stack of Voigts). For any low-SNR spectrum, gate the
result with `assess_detection` before reporting (see Planning Step 1).

**Let the residual drive escalation — don't stop at a single component while
structure remains.** After any fit, inspect the residual *within the peak
region* (not the noise-only wings). A systematic, above-noise residual there — a
shoulder, a one-sided lean, a secondary bump, a sharp cusp the broad model
undershoots — means the model is incomplete, even if the global R² looks high.
Escalate: for a peak whose residual shows an extra *chemical environment*, hand
it to `fit_multipeak_voigt`; for a half-integer quadrupolar line whose residual
shows the characteristic asymmetric foot, try `fit_quad_ct` (one quadrupolar
site) before adding Voigts. This is residual-driven, not nucleus-driven — apply
it whenever a single symmetric component leaves real structure behind.

**A gate-passing fit is not automatically done.** If `peak_region_r2` clears the
gate but `residual_structured` is True, the (pseudo-)Voigt is the wrong *model*
(it returns no C_Q / J), so re-fit with the matching physical model and compare.
Trigger on the `residual_structured` flag only — it is calibrated on the
signal-relative `apex_resid_frac` / `resid_rms_frac`; do not escalate on
`apex_resid_over_noise` or `frac_resid_gt_3sigma`, which stay large even for a
perfect fit of a high-SNR line. Keep whichever fit has the lower `resid_rms_frac`
and clears the flag — **never adopt a model that fits worse** — and make sure the
physical model represents every site/environment the line actually contains (a
multi-site line needs a multi-site model). Fall back to the (pseudo-)Voigt with a
stated limitation when no physical model improves it.

**Second-order quadrupolar central transition** (solid half-integer, resolved
asymmetric line): call the skill tool — it does the orientation-averaged powder
lineshape, a multi-start (C_Q, η) search to avoid local minima, and returns a
reliability flag:

```python
from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_ct
# crop to the central-transition region first; pass the nucleus Larmor freq.
res = fit_quad_ct(ppm_roi.tolist(), y_roi.tolist(), nu_L_MHz=NU_L, I=1.5, mas=True)
p, d = res["parameters"], res["derived"]
# Honor d["Cq_resolved"]: if False, report p["Cq_MHz"] as an upper bound only.
```

**Detection check for low-SNR spectra.** When the signal is weak or the nucleus
is insensitive (low-γ, dilute), screen the result before reporting a lineshape:

```python
from scilink.skills.curve_fitting.nmr.detection import assess_detection
# pass the fit and its free-parameter count to enable the peak-vs-baseline F-test.
det = assess_detection(x.tolist(), y.tolist(), y_fit=y_fit.tolist(),
                       n_model_params=n_params, baseline=baseline.tolist())
# det["verdict"] in {detected, marginal, absent}; honor det["recommendation"].
```

If the verdict is **absent**, report a non-detection with `det["upper_bound"]`,
not a fitted peak; if **marginal**, report the shift with a wide uncertainty and
no quantitative integral.

**Quality metric (MANDATORY — the gate scores by `peak_region_r2`).** NMR's
wide digitized window makes a whole-window R² meaningless for a narrow peak
(noise dominates the variance), so after fitting, compute the peak-region R²
over the signal region and report it — the run is hard-rejected if the metric is
missing:

```python
from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
# y is the (baseline-corrected or raw) data fit to; y_fit is the model on the
# same x; baseline is the fitted baseline array (or omit if already subtracted).
q = peak_region_r2(x.tolist(), y.tolist(), y_fit.tolist(), baseline=baseline.tolist())
# q["peak_region_r2"] is the gate metric; q["r_squared"] is the global value.
```

**A high `peak_region_r2` is necessary but not sufficient — also read the
residual-structure diagnostics the same call returns.** `q["residual_structured"]`
is True when the residual is both large relative to the peak height
(`q["apex_resid_frac"]`) and systematically correlated (`q["resid_autocorr_lag1"]`
near 1, `q["frac_resid_gt_3sigma"]` high) — the signature of a fit that misses
where the signal is (an unmodeled shoulder, a wrong lineshape, a quadrupolar foot
forced into a Voigt) even at R² ≈ 0.96. When it is True, escalate the model
(add an environment, switch to `fit_quad_ct` / `fit_jcoupled_multiplet`) rather
than accept the fit on R² alone.

Emit `FIT_RESULTS_JSON:` with `fit_quality` containing **`peak_region_r2`** (the
gate metric) and the global `r_squared`, plus per-peak δ (ppm) and width
(ppm + Hz), the lineshape/model used, and — for quadrupolar fits — C_Q, η_Q,
P_Q, and the `Cq_resolved` flag.

## interpretation

**Chemical-shift reference ranges** (general, vs the IUPAC primary references:
¹H/¹³C vs TMS; ¹⁹F vs CFCl₃; ³¹P vs 85% H₃PO₄; ²³Na vs 1 M NaCl(aq); ¹⁷O vs
H₂O; ⁶⁷Zn vs 1 M Zn(NO₃)₂). The "common features" are illustrative, not
assumptions — assign from the actual sample chemistry, not this list:

| Nucleus | Typical range | Common features (illustrative) |
|---|---|---|
| ¹H | 0–15 ppm | aliphatic 0–3, OH/NH 1–6, H₂O/alcohol 1–5, aromatic 6–9, acid/aldehyde 9–12 |
| ¹³C | 0–220 ppm | aliphatic 0–60, C–O 50–90, aromatic/alkene 100–150, carbonyl/carboxyl 160–210 |
| ¹⁹F | −60 to −230 ppm | CF₃ ≈ −60 to −80, CF₂/CF higher field, aromatic-F ≈ −100 to −170 |
| ³¹P | −200 to +250 ppm | phosphate/organophosphate near 0 ± 30; speciation often via small δ shifts |
| ²³Na | −20 to +15 ppm | aqueous Na⁺ ≈ 0 (reference); oxide/coordinated Na typically 0 to −15 |
| ²⁷Al | −20 to +120 ppm | octahedral Al ≈ 0–15, tetrahedral Al ≈ 55–80 (coordination diagnostic) |
| ¹⁷O | −50 to +1100 ppm | H₂O/D₂O 0; M–O–M / carbonyl / oxo much higher |
| ⁶⁷Zn | −20 to +300 ppm | aqueous Zn²⁺ ≈ 0; shift tracks coordination (tetrahedral vs octahedral) |

**Linewidth.** Report FWHM in both ppm and Hz (Hz = ppm × ν_L). In solution,
homogeneous T₂* ≈ 1/(π·FWHM_Hz); a Gaussian contribution (Voigt) signals shim or
unresolved-coupling broadening rather than relaxation. In solids, linewidth
reflects disorder / dipolar coupling / residual CSA, not T₂.

**Quadrupolar parameters (when `Cq_resolved=True`).** C_Q (MHz) measures the EFG
magnitude at the nucleus — larger = more distorted coordination; η_Q ∈ [0,1] is
the EFG asymmetry (0 = axial). The observed centre of gravity is shifted from
δ_iso(CS) by the quadrupole-induced shift δ_QIS (negative, ∝ 1/ν_L²) returned in
`derived` — quote δ_iso(CS), not the apparent peak, as the chemical shift. If
`Cq_resolved=False`, do not report C_Q as a measurement.

**Lineshape is a discriminator for half-integer quadrupolar nuclei — do not
assign on the isotropic shift alone.** For these nuclei the shift is a weak
fingerprint: its chemical-shift range is narrow and the apparent peak is offset
by the field-dependent δ_QIS, so very different environments can share an
apparent position. The **linewidth / asymmetry supplies the missing dimension**:
a *narrow, near-symmetric* line indicates a mobile or high-symmetry environment
(motional averaging and/or a small EFG — e.g. a solvated/solution-like or
cubic-site species), whereas a *broad, asymmetric (second-order quadrupolar)*
line indicates a static, low-symmetry crystallographic site (large EFG). So when
assigning a species or proposing candidate phases from a quadrupolar nucleus,
**condition the assignment on the linewidth/asymmetry, not the shift** — and when
the line is broad enough to be quadrupolar, the resolved C_Q/η_Q (from
`fit_quad_ct`) are the most discriminating identifiers, far more specific than
the centre of gravity.

**Sidebands → CSA / quadrupolar product.** When sidebands are fit, their
intensity envelope encodes the CSA (spin-½) or the satellite/CSA interplay
(quadrupolar); v0 reports the manifold but defers tensor extraction.

## validation

- **Quality is `peak_region_r2`, not the whole-window R².** NMR's enormous
  dynamic range and wide digitized axis make a whole-spectrum R² noise-dominated:
  a correct fit of a narrow peak scores near 0. The gate therefore scores the
  peak-region R² (signal region only) reported by `peak_region_r2`. A low *global*
  R² with a healthy `peak_region_r2` and flat, noise-level residuals across the
  peak is an *acceptable* fit, not a failure. A low `peak_region_r2` means the fit
  is genuinely wrong *where the signal is* (missed/extra peak, wrong shape) —
  that is a real failure.
- **Sideband spacing must equal the MAS rate.** If fitted "extra peaks" sit at
  ±k·(ν_r/ν_L) ppm, they are sidebands — relabel them and fold their intensity
  into any quantification. If the spacing does NOT match ν_r, they are real
  resonances.
- **Integrated intensity includes sidebands.** A quantification (site
  populations, relative concentrations) that sums only centrebands undercounts
  when sidebands carry appreciable intensity.
- **Phase sanity.** A fit that required a large negative baseline plus a positive
  peak (or vice versa) to match an inverted line indicates a phase problem;
  prefer a signed-amplitude fit and flag for re-phasing.
- **Quadrupolar reliability.** Honor `Cq_resolved` from `fit_quad_ct`: report C_Q
  as a measurement only when resolved; otherwise as an upper bound with an
  MQMAS/satellite recommendation. Watch for η railing to 0 or 1 (poorly
  determined) — reported in `reliability_flags`.
- **Cross-series consistency.** In a composition/temperature ladder the per-sample
  parameters and the component count usually vary smoothly. A discontinuity has
  two possible causes — distinguish them, don't assume it is an error: a
  *spurious* jump from a mis-assigned sideband, an uncorrected baseline, or a
  phase flip (re-inspect and fix), versus a *real* structural transition — a new
  phase/site appearing, a peak splitting — which is a genuine result to report
  with the variable value at which it occurs. A locked-count model will hide the
  latter, so verify the count is data-driven before calling a discontinuity an
  artifact.
