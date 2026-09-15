---
description: CW-EPR axial powder fitting for S=1/2 systems at X-band — extracts g_par, g_perp, A_par, and linewidth from the first-derivative spectrum, with an optional second isotropic component for radical/defect contamination near g≈2.
technique: [EPR, ESR, "electron paramagnetic resonance", "electron spin resonance"]
quality_gate:
  metric: r_squared
  accept_threshold: 0.90
  hard_reject_threshold: 0.75
  direction: higher_is_better
---
# CW-EPR Curve Fitting Skill (axial powder, X-band)

## overview

Continuous-wave EPR powder spectra at X-band (~9.4 GHz). Scope is the
axial S = 1/2 case: a single g-tensor with g_par, g_perp, and (when
present) parallel hyperfine A_par on a nucleus of spin I (default I = 3/2
for Cu(II)). Output is the first-derivative spectrum, dχ"/dB, the
standard observable of phase-sensitive detection on a Bruker EMXnano /
EMXmicro class instrument.

This skill is designed to fit the *whole* derivative spectrum — both the
broad transition-metal powder pattern and any narrow superimposed
isotropic feature near g≈2 (organic radicals, defect electrons) — in one
shot. That second component matters: many real Cu(II) / VO(II) / Mn(II)
samples carry a few-percent radical contribution that an
unconstrained sum-of-Lorentzians fit will absorb, distorting the metal
parameters. Keeping the radical as an explicit, parametrized component
makes the metal fit robust.

**Out of scope (deferred; not implemented in v0):**

- Rhombic g-tensor — full three-axis powder integration.
- Resolved perpendicular hyperfine (A_perp); quadrupolar coupling (P);
  more than one nuclear-spin coupling per spin center.
- Q-band / W-band, ENDOR, pulsed EPR, double-resonance.
- Saturation / power-dependent linewidth.
- Multi-species deconvolution beyond one axial Cu(II)-class + one
  isotropic narrow line.
- Quantitative spin counting (requires standard, beyond a curve-fit skill).

For any of the above, treat this skill as "model selection scaffolding"
and either fall back to manual fitting or wait for the planned
EasySpin-bridge extension (see "Suggested follow-ups" in the report
template below).

## planning

**Hard prerequisite: the microwave frequency.** g-values are pinned by
the resonance condition `g·μ_B·B = h·ν`. Without ν the fit can move
g and B in lockstep and recover the wrong absolute g. **Take ν from the
spectrometer log** (Bruker EMXnano routinely records 9.6–9.7 GHz at
X-band) and pass it as `nu_GHz`. Do not let it float.

**Always background-subtract first.** A linear baseline from both wings
is sufficient for CW-EPR — the spectrum is already a derivative, so
broad smooth backgrounds appear as slowly-varying offsets, not as
spurious peaks. Sample 2–5% of the points from each end, fit a line,
subtract. Do this BEFORE invoking `fit_axial_powder`. Quadratic
baselines indicate an instrumental issue; raise a warning rather than
fitting them.

**Choosing the model:**

- **Pure axial (`include_isotropic=False`)** — for a clean transition-
  metal spectrum with no narrow feature near g≈2. Five free parameters.
- **Axial + isotropic (`include_isotropic=True`)** — when the spectrum
  shows a sharp doublet near 3300–3450 G at X-band that is *narrower*
  than the broad powder pattern. Eight free parameters. Default for any
  glassy or solid-state sample, since trace organic radicals and defect
  electrons are nearly always present.

If unsure, fit BOTH and compare R². A real isotropic contribution
typically lifts R² by ≥ 0.05; a spurious isotropic component fitted on
clean data drives `iso_amplitude → 0` (then drop it).

**Initial guesses by metal class** (X-band, room-temperature literature
ranges; refine from spectrometer-frequency lab calibration if available):

| Class | g_par | g_perp | A_par (G) | Notes |
|---|---|---|---|---|
| Cu(II) — equatorial O/N coordination | 2.25–2.40 | 2.05–2.10 | 130–200 | I = 3/2; quartet on g_par |
| Cu(II) — strongly-tetragonal | 2.20–2.30 | 2.05–2.08 | 170–220 | Larger A_par tracks weaker equatorial field |
| VO(II) (vanadyl) | 1.93–1.95 | 1.97–1.99 | 170–200 | I = 7/2; **8 lines** — set `I_nuc=3.5` |
| Mn(II) | 2.00–2.01 | 2.00–2.01 | 80–95 | I = 5/2; six lines, near-isotropic — `I_nuc=2.5` and shrink g bounds |
| Organic radical only | n/a | n/a | 0 | `I_nuc=0`, narrow lw (5–20 G), or use the isotropic component alone |

**Linewidth guess.** Cu(II) in a glass / KBr matrix typically gives an
orientation-averaged Gaussian FWHM of 30–80 G; rigid lattice-site Cu in a
single-environment crystal can go down to 10–20 G; concentrated samples
(> 5%) broaden into the 80–150 G range from dipolar coupling. Start at
40 G when in doubt.

**Bounds — keep them narrow.** Wide bounds make the optimizer wander
into nonphysical g-values that compensate for a model mismatch. Default
Cu(II) bounds: g_par ∈ [2.10, 2.50], g_perp ∈ [2.00, 2.20], A_par ∈
[50, 250] G, lw ∈ [5, 200] G. For other metals, narrow them per the
table above before calling.

**Series fits (multiple samples, same metal class).** Lock the structure
(`I_nuc`, bounds, `include_isotropic`) across the series; let g, A, lw
float per sample. The line-broadening question (this run vs that run)
becomes a one-row-per-sample comparison of fitted `lw_G`. For paired
templated-vs-untemplated comparisons, also report Δ(g_par), Δ(A_par),
and the isotropic amplitude — radical content often differs between
host environments and is itself a useful comparative.

## implementation

**CRITICAL: per-spectrum EPR fitting workflow.**

1. Load the field+derivative pair with `load_curve_data` (X column = B in
   Gauss, Y column = dχ"/dB). Sort by ascending field.
2. Read the microwave frequency from metadata. Fail loudly if absent.
3. Detrend with a linear baseline from both wings.
4. Choose `include_isotropic` based on whether the spectrum shows a
   narrow feature near g ≈ 2 (see Planning).
5. Call `fit_axial_powder` with conservative bounds.
6. Optionally re-fit with the opposite `include_isotropic` choice and
   accept the higher-R² result.
7. Emit `FIT_RESULTS_JSON:` with parameters, R², derived g_avg, and
   `model_used`.

**Complete EPR fitting template:**

```python
import json
import numpy as np

from scilink.skills._shared.curve_fitting_tools import load_curve_data
from scilink.skills.curve_fitting.epr.axial_powder import fit_axial_powder

# ---- Step 1: Load (X column = B in Gauss, Y column = dχ"/dB) ----
data = load_curve_data(DATA_PATH)
B_raw, y_raw = np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)
sort_idx = np.argsort(B_raw)
B, y = B_raw[sort_idx], y_raw[sort_idx]

# ---- Step 2: Microwave frequency from metadata (REQUIRED) ----
# Bruker EMXnano X-band typically logs 9.6-9.7 GHz; use the actual
# spectrometer value, not a fixed default. If metadata['nu_GHz'] is
# missing, ABORT with an explanatory error rather than guessing.
nu_GHz = float(METADATA["nu_GHz"])  # e.g. 9.64

# ---- Step 3: Linear baseline from both wings ----
n = len(y); nlo, nhi = int(n*0.02), int(n*0.02)
Bs = np.concatenate([B[:nlo], B[-nhi:]])
ys = np.concatenate([y[:nlo], y[-nhi:]])
slope, intercept = np.polyfit(Bs, ys, 1)
y_corr = y - (slope * B + intercept)

# ---- Step 4: Class-specific guess (Cu(II) shown; adjust per Planning) ----
class_init = dict(
    I_nuc=1.5,                    # Cu
    g_par_init=2.30, g_perp_init=2.06,
    A_par_G_init=170.0, lw_G_init=40.0,
    bounds_g_par=(2.10, 2.50),
    bounds_g_perp=(2.00, 2.20),
    bounds_A_par_G=(50.0, 250.0),
    bounds_lw_G=(5.0, 200.0),
)

# ---- Step 5/6: Fit both with and without the isotropic component ----
fit_cu = fit_axial_powder(B.tolist(), y_corr.tolist(), nu_GHz=nu_GHz,
                          include_isotropic=False, **class_init)
fit_cu_rad = fit_axial_powder(B.tolist(), y_corr.tolist(), nu_GHz=nu_GHz,
                              include_isotropic=True, **class_init)
better = fit_cu_rad if fit_cu_rad["fit_quality"]["r_squared"] \
                      > fit_cu["fit_quality"]["r_squared"] + 0.01 else fit_cu

# ---- Step 7: Emit results ----
print("FIT_RESULTS_JSON: " + json.dumps({
    "model_type": better["model_used"],
    "parameters": better["parameters"],
    "fit_quality": better["fit_quality"],
    "derived": {
        **better["derived"],
        "linewidth_Gauss": better["parameters"]["main"]["lw_G"],
        "isotropic_radical_present":
            "isotropic" in better["parameters"]
            and better["parameters"]["isotropic"]["amplitude"] > 0.05,
    },
    "nu_GHz": nu_GHz,
    "I_nuc": better["I_nuc"],
    "deviation_note": "",
}))
```

**For a series comparison** (e.g. templated vs non-templated), wrap the
above in a loop over spectra and emit a `series_summary` block alongside
`FIT_RESULTS_JSON` with one row per sample. The line-broadening
comparison is then a single `lw_G` column across samples.

## interpretation

**g-values and what they say:**

- g_par ≈ 2.20–2.30 with g_perp ≈ 2.05–2.10 (g_par > g_perp): tetragonally
  elongated d⁹ Cu(II), unpaired e⁻ in d(x²-y²). Equatorial coordination
  controls A_par.
- g_par ≈ g_perp (within 0.02): nearly isotropic system. Either a true
  isotropic ion (Mn(II), high-spin Fe(III) in cubic field) or a
  motionally-averaged spectrum (rule out: drop temperature or check
  rotational correlation time).
- g_par < g_perp ("inverted axial"): unusual for Cu — often a sign of
  d(z²) ground state (axial elongation flipped to compression) or a
  fitting artifact. Inspect the spectrum manually.

**Hyperfine A_par interpretation (Cu(II) only):**

| A_par (10⁻⁴ cm⁻¹) | A_par (G, X-band) | Coordination class |
|---|---|---|
| > 175 | > 187 | CuO₄ / CuN₂O₂ (oxygen-rich) |
| 150–175 | 160–187 | CuN₄ (porphyrin-like) |
| 125–150 | 134–160 | Distorted / weakly coordinated |
| < 125 | < 134 | Square-planar S-coordinated, or geometry distortion |

Conversion: A(10⁻⁴ cm⁻¹) ≈ A(G) × g_par × 0.4668 / 10. The cm⁻¹ form is
the standard literature unit; quote both when reporting.

**Linewidth (the "line broadening" answer):**

`lw_G` is an *orientation-averaged Gaussian FWHM* — a single scalar that
captures the cumulative effect of:

- **g-strain** (a distribution of g-values from heterogeneous sites).
- **Unresolved A_perp** and any other unresolved couplings.
- **Dipolar broadening** at higher concentrations.
- **Instrumental** modulation broadening (when modulation amplitude ≳
  intrinsic ΔBpp; here 8.23 G modulation puts a 10 G floor on lw_G —
  fitted values close to that are modulation-limited and not informative
  about the sample).

Compare lw_G across samples in the same series to quantify line
broadening. A 10–30% difference is a genuine effect; < 5% is within
fit noise.

**Isotropic component (when fitted):**

- `iso_g` ∈ [2.002, 2.005] and small `iso_lw_G` (5–20 G): organic
  radical / paramagnetic defect contribution. Often unaffected by metal
  coordination chemistry — it's a separate background.
- `iso_g` > 2.005 with broader `iso_lw_G`: may indicate a second metal
  species mis-fit as an isotropic line. Re-examine.
- `iso_amplitude` ≪ 0.05 (vs main = 1.0): treat as negligible, fall back
  to the no-isotropic fit.

## validation

**Per-fit checks:**

- `r_squared ≥ 0.90` accept; `[0.75, 0.90]` marginal — usually means
  either the wrong `I_nuc` (count the resolved parallel-region lines:
  4 → Cu, 8 → VO, 6 → Mn, 1 → no hyperfine), or a multi-species
  spectrum that this v0 sketch cannot deconvolve. Report numbers with a
  caveat. Below 0.75, reject and recommend manual inspection.
- `lw_G` near its lower bound (≤ 5 G): the optimizer collapsed to a
  delta. Almost always a sign that the spectrum lacks the broad
  axial-powder shape — try `I_nuc=0` (radical only) or report the
  spectrum as not fittable by an axial model.
- `g_par - g_perp < 0.01`: the fit collapsed to a near-isotropic
  solution. Either the spectrum *is* isotropic (fine — report g_avg
  only), or the optimizer missed a deeper minimum. Re-try with a
  slightly larger `g_par_init`.
- `A_par_G` at its upper bound: real Cu(II) rarely exceeds 220 G at
  X-band; a hit at 250 indicates either a non-Cu nucleus (relax `I_nuc`)
  or the optimizer is over-stretching to fit a feature outside the
  model.

**Sanity vs total field range.**
At X-band the parallel-hyperfine quartet of Cu(II) spans
B_g_par ± 1.5 × A_par (Gauss). A typical Cu spectrum needs B sampled
from ~ B_g=2.6 to B_g=1.9, i.e. 2600–3600 G at 9.6 GHz. If the input
field grid does not cover both g_par and g_perp regions, refuse to fit
and recommend a wider scan.

**Cross-fit comparison.**
For paired templated/non-templated comparisons, lock everything except
`g_perp, A_par, lw_G` (the parameters most sensitive to coordination
environment) across the two samples and refit. If `lw_G` is the only
parameter that changes significantly, the line-broadening interpretation
("MOF anchoring introduces g-strain without restructuring the
coordination sphere") is supported. If `g_par` or `A_par` also shift, a
single-environment model is wrong and the two samples have genuinely
different Cu coordination.
