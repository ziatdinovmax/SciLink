---
description: Geometric Phase Analysis (GPA) strain mapping of atomic-resolution / lattice-fringe images — maps the in-plane strain tensor (exx, eyy, exy) and lattice rotation (wxy) RELATIVE TO AN UNDISTORTED REFERENCE REGION, localizes strain concentration at defects/interfaces/boundaries, and honestly disclaims when the input is unsuitable (Fourier-pre-filtered, too few valid pixels, ill-conditioned reflections). Also the right tool for lattice-displacement tasks phrased differently — precipitate interface coherency (coherent vs incoherent / misfit strain), Burgers-vector / dislocation displacement circuits, and twin-boundary displacement — all of which reduce to the GPA displacement-gradient field.
technique: STEM, HAADF-STEM, HRTEM, lattice-fringe TEM
---
# GPA Strain Mapping Skill

## overview

Use for **strain / lattice-distortion mapping** of a single crystalline,
zone-axis (or lattice-fringe) image: deformed twin boundaries, dislocations,
coherent/semi-coherent precipitate–matrix interfaces, epitaxial mismatch,
misfit. GPA recovers the 2-D displacement-gradient (distortion) tensor from the
geometric phase of **two non-collinear Bragg reflections**, and reports
`exx, eyy, exy` (strain) and `wxy` (rigid rotation) **relative to a chosen
undistorted reference region**.

> **MANDATORY TOOL.** This skill ships its own validated GPA implementation,
> `gpa_strain_map` (imported as shown in *analysis* below). You MUST use it for
> the reciprocal-space / strain step. Do **NOT** use `fourier_reflection_map`,
> `run_fft_nmf_analysis`, or a hand-written FFT/GPA — those lack the reference-
> region carrier refinement, conditioning check, validity masking, and
> pre-filter guard that make GPA trustworthy (and are exactly what produced
> degenerate, unreferenced strain maps before). The entire strain computation is
> a **single call** to `gpa_strain_map`; the rest of the script is only I/O,
> plotting, and reporting its returned fields.

This skill calls a **validated tool** — do NOT hand-roll GPA in the generated
script. Naive GPA fails in three classic ways that this tool fixes:

1. **No reference → garbage absolute strain.** Strain is only meaningful
   relative to an unstrained lattice. The tool refines the carrier g-vectors
   from an auto-selected (or user-given) undistorted region, so the reference
   strain is ~0 by construction and there is no global offset.
2. **Ill-conditioned reflections.** It picks a near-orthogonal, similar-|g|,
   high-power pair so neither strain axis is under-determined (reports the
   condition number).
3. **Trusting unsuitable inputs.** It flags **Fourier-pre-filtered** images
   (where any derived strain / Burgers vector is untrustworthy) and low
   valid-pixel fractions, and returns `answerable=False` instead of a
   confident-looking but degenerate map.

If the objective is a **Burgers vector** rather than a strain field, GPA on a
single (often Fourier-filtered) frame is fragile — prefer a real-space Burgers
circuit on detected atomic-column positions; use GPA only for the strain field
around the defect.

## planning

### foundational — one focused goal
Pick ONE: (a) full strain-tensor map across a defect/interface, or (b) peak-
strain localization + magnitude at a boundary. Both come from one
`gpa_strain_map(...)` call — the plan's reciprocal-space/strain step is literally
"call `gpa_strain_map`", not `fourier_reflection_map` or any built-in GPA.

### foundational — calibration
Read calibration from the provided metadata block (authoritative). GPA strain is
**dimensionless** (independent of pixel size); pixel size is only needed to
report the *physical location/extent* of strain features. The tool assumes
**square pixels** — if `dx != dy`, resample to square physical pixels first.
Resolve pixel size with the shared helper rather than hand-rolling it:
`from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm`;
`px = resolve_pixel_size_nm(metadata, image.shape)` → `{"x","y","source"}` nm/px,
or `None`. It divides `field_of_view` by the image **shape**; never divide by a
metadata pixel-count field (`n_cols`/`width`), which is usually absent and
silently yields `None`.

### foundational — reference region
The single most important choice. Default `reference_roi="auto"` selects the
most uniform, high-Bragg-amplitude patch. **Override it** when you can see an
obviously undistorted region far from the defect (pass `(x, y, w, h)`); a
reference placed *on* the defect makes the whole map wrong. The reference is
drawn (cyan box) on the overlay — verify it sits in good, undistorted lattice.

### foundational — whole-field ramp vs. localized strain (the `detrend` decision)
`gpa_strain_map` ALWAYS returns `res["affine"]`: per-component linear-ramp slopes,
`max_inplane_ramp_fraction` (fraction of in-plane strain variance explained by a
whole-field linear ramp), `rotation_ramp_fraction`, and a `dominated_by_ramp`
flag. A high value = a smooth whole-field gradient, **usually STEM scan
distortion / slow scan drift, but possibly real long-range strain or a deformed
twin's orientation gradient.** This is the single most common reason a strain map
looks "wrong" (an extended left↔right gradient instead of a localized ridge).

Decision (two-pass — YOU set `detrend`):
1. First call with `detrend=False` (default). Read `affine["dominated_by_ramp"]`.
2. If False → the field is already localized; report the strain map + peak.
3. If True → re-call with **`detrend=True`** to remove the affine ramp, then
   compare localization (peak |strain| / background) before vs after:
   - a localized concentration **survives** de-ramping → that is the real local
     strain (report BOTH: "a whole-field ramp of fraction X — likely scan
     distortion — overlies a localized strain at …");
   - **nothing localized survives** → the field was essentially all ramp; report
     "no localized strain; field dominated by a whole-field gradient (scan
     distortion / uniform tilt / long-range orientation change)".
   NEVER present a de-ramped map as the clean answer without stating the ramp was
   removed and what it most likely is — detrend can erase genuine long-range
   strain or a twin's orientation offset.

### foundational — answerability gate (respect it)
If the tool returns `answerable=False`, do NOT report quantitative strain.
Read `flags`:
- `prefiltered=True` → the micrograph was Fourier-filtered; report this as the
  finding ("strain/|b| not recoverable from a pre-filtered frame") rather than
  inventing numbers.
- `valid_fraction` low or `condition_number` high → re-try with a different
  reference / explicit reflections; if still low, report "insufficient valid
  lattice / ill-conditioned — not quantifiable," which is the correct answer.

## analysis

```python
import numpy as np
from scilink.skills._shared.strain import gpa_strain_map

img = np.load("data.npy")
if img.ndim == 3:                      # use luminance for an RGB/stack input
    img = img[..., :3].mean(-1) if img.shape[-1] in (3, 4) else img[0]
img = img.astype(float)

from scipy.ndimage import gaussian_filter

def localization(res):                 # peak |in-plane strain| / background
    m = res["valid_mask"]
    emag = np.where(m, np.hypot(res["exx"], res["eyy"]), np.nan)
    es = gaussian_filter(np.nan_to_num(emag), 12)
    py, px = np.unravel_index(np.nanargmax(np.where(m, es, 0)), es.shape)
    bg = np.nanmedian(emag); peak = np.nanpercentile(emag[m], 99)
    return (px, py), peak, bg, peak / (bg + 1e-9)

res = gpa_strain_map(img, reference_roi="auto")   # detrend=False (default)

if not res["answerable"]:
    # report the flags; do NOT fabricate strain numbers
    reason = ("pre-filtered micrograph" if res["flags"]["prefiltered"]
              else f"valid_fraction={res['valid_fraction']:.2f}, "
                   f"cond={res['condition_number']:.1f}")
    print("IMAGE_ANALYSIS_RESULTS_JSON:" + __import__("json").dumps(
        {"answerable": False, "reason": reason, "flags": res["flags"]}))
else:
    af = res["affine"]
    if af["dominated_by_ramp"]:
        # whole-field ramp dominates -> re-run with detrend=True and compare
        res_dt = gpa_strain_map(img, reference_roi="auto", detrend=True)
        (px, py), pk0, bg0, loc0 = localization(res)       # raw (with ramp)
        (dx, dy), pk1, bg1, loc1 = localization(res_dt)    # de-ramped
        ramp = af["max_inplane_ramp_fraction"]
        if loc1 > 1.3 * loc0:        # a localized feature emerged under the ramp
            report = (f"whole-field linear ramp (fraction {ramp:.2f}; likely scan "
                      f"distortion / long-range gradient) overlies a LOCALIZED "
                      f"strain at ({dx},{dy}); de-ramped localization {loc1:.1f} "
                      f"vs {loc0:.1f}")
            res, px, py = res_dt, dx, dy   # report the de-ramped map + its peak
        else:                        # nothing localized survives -> all ramp
            report = (f"NO localized strain: field dominated by a whole-field "
                      f"ramp (fraction {ramp:.2f}) — scan distortion / uniform "
                      f"tilt / long-range orientation change")
    else:
        (px, py), pk0, bg0, loc0 = localization(res)
        report = f"localized strain at ({px},{py}); localization {loc0:.1f}"
    # exx/eyy/exy/wxy are 2-D arrays (dimensionless; wxy rad); use valid_mask
    # for every statistic/plot. res["stats"] gives referenced robust summaries.
    m = res["valid_mask"]
```

Save a `visualization.png` with: the raw image + reference box (cyan); the four
component maps (`RdBu_r`, symmetric limits from the valid-pixel 95th percentile);
and `|in-plane strain|` with the peak marked. Print the referenced robust stats
(`res["stats"]`), the reflections/condition number, `valid_fraction`, and the
peak-strain location in nm.

### tool outputs — do not confuse these
- `res["reference_box"]` is ALWAYS a 4-tuple `(x, y, w, h)` (never null) — the
  undistorted region where strain is anchored to ~0. Draw it as a rectangle and
  report `res["stats"]`/`res["stats_raw"]` (reference-region strain is ~0 by
  construction; if you need the number, take the median over that box).
- `res["flags"]["peak_to_background"]` and `["diffuse_fraction"]` are
  **pre-filter (FFT) diagnostics only** — they decide `prefiltered`. They are
  NOT a localization or strain-quality metric; a value like 9e5 is normal for a
  clean image. Compute localization yourself as peak |strain| / background from
  the strain field (see `localization()` above) — never read it off a flag.

## interpretation

- **exx, eyy** = normal strains along image x/y; **exy** = shear; **wxy** = rigid
  lattice rotation (radians). All are RELATIVE to the reference region (~0 there).
- A **deformed boundary / dislocation** shows up as a localized concentration in
  `|in-plane strain|` and/or a step in `wxy` (rotation) tracing the boundary; the
  peak-strain pixel is the boundary location.
- Typical lattice strains are a **few percent**; |strain| > ~0.3 over an extended
  area means the analysis is wrong (bad reference / wrong reflections), not real.
- A clean **null** (undeformed crystal) is exx≈eyy≈exy≈wxy≈0 with no localized
  band — that is a correct, reportable result, not a failure.
- A smooth **left↔right (or top↔bottom) gradient** filling the whole field is a
  ramp (`affine["dominated_by_ramp"]`), not a localized defect strain — handle it
  with the `detrend` two-pass above; report the ramp *and* whatever local strain
  survives de-ramping (or that none does).

## validation

### foundational
- **Reference ≈ 0**: the median strain over the reference box must be ~0 (the
  tool enforces this; if not, the reference is bad).
- **Self-test available**: `strain.make_strained_lattice` + a known
  displacement field recovers imposed strain to within ~10% — run it if unsure
  the calibration/orientation is being handled correctly.
- **Physical range**: referenced strains within a few percent (precipitate/defect
  cores may reach ~10–20% locally); anything larger and extended ⇒ revisit the
  reference / reflections.
- **Honesty**: `answerable=False` (pre-filtered, low valid fraction, ill-
  conditioned) is a legitimate, final answer — report the reason, do not
  manufacture a strain field.
- **Illumination-envelope guard (HAADF)**: the tool reports `amp_intensity_corr`
  and `flags["amplitude_tracks_intensity"]`. When the Bragg amplitude tracks the
  raw-intensity envelope (|corr| ≥ 0.6 — detector / thickness / illumination
  gradient), the validity mask and strain follow the envelope, not the lattice;
  the tool sets `answerable=False`. This is the classic HAADF-GPA failure — do
  NOT report the strain field. The fix is to flatten the illumination first
  (divide the image by a strong low-pass / Gaussian background, or high-pass it)
  so the Bragg amplitude reflects lattice quality, then re-run; if the flag
  persists, the data cannot support GPA at this contrast and the honest answer is
  a null with that reason.
