---
description: General AFM image analysis (topography, KPFM, PFM, conductance) where intensity is a physical quantity with units - emphasizes row alignment, pixel-square resampling, and physical-unit preservation through the pipeline.
---
# AFM Imaging Skill

## overview
Atomic force microscopy (AFM) produces images where the pixel
intensity encodes a **physical quantity** — most commonly topography
(height, in nm or µm), but also contact potential (KPFM, in V),
piezoresponse amplitude/phase (PFM), stiffness, adhesion, or current.
Unlike STEM/TEM where intensity is an arbitrary detector count, AFM
intensity has units that matter for interpretation. Use this skill
for any AFM-derived image whose value carries a physical meaning.

## planning
### foundational
Before any feature extraction, decomposition, or segmentation, the
raw AFM image must be put into an analysis-ready state. Three issues
dominate and must be handled up front.

**1. Row alignment (line-by-line leveling).**
AFM is a raster technique: each scan line is acquired sequentially
and is subject to thermal drift, piezo creep, tip-sample offsets, and
feedback baseline shifts between lines. Raw data almost always shows
horizontal streaks / row-to-row offsets that dominate the image
contrast and will wreck any downstream analysis (FFT, NMF,
segmentation, statistics).

Default behavior: **align rows before doing anything else.** Typical
options, in increasing aggressiveness:
- subtract the median of each row (robust to features)
- subtract a low-order polynomial (1st–2nd order) fit per row
- subtract a plane fit globally, then per-row median, for tilted
  samples

Use the median (not the mean) when features occupy a non-negligible
fraction of the line — otherwise tall objects bias the baseline and
create dark halos. Do **not** row-align images where the feature of
interest is genuinely row-correlated (e.g. striped domains parallel
to the fast-scan axis); in that case use a global plane fit only and
document the choice.

**2. Non-square pixels → rescale to square pixels.**
AFM scans frequently use different numbers of pixels (or different
scan sizes) along x and y, so a pixel is physically rectangular.
Every metadata block we work with contains the field of view (FOV)
in physical units along both axes, so the true pixel size is known.

Default behavior: **resample the image to square pixels** using the
metadata FOV before downstream analysis. This prevents silent
distortion in:
- FFT (anisotropic frequency axes)
- morphological operations and blob/peak detection (anisotropic
  kernels)
- aspect-ratio-sensitive visualization
- any measurement of angles and orientations

Record the new pixel size (nm/px) in the analysis state; all later
length-scale measurements must use it. When displaying, preserve
true aspect ratio — do not stretch images to fill subplot shapes.

**Defects in a periodic self-assembly (honeycomb / lattice of
molecules, proteins, particles):** once the image is row-aligned,
leveled and on square pixels (steps 1–2 above), the registered tool
`fft_defect_map` (`scilink.skills._shared.fft_defect`) locates point
defects — both protrusions (bright/`excess`) and missing units
(`deficit` vacancies) — by reconstructing the perfect lattice and
mapping null-gated residual anomalies. It needs that preprocessing
done first: it does **no** leveling or resampling, and a tilted or
anisotropic input gives wrong geometry. Pass the square-pixel size in
nm. Read `pattern_period_nm` for the self-assembly lattice parameter.

**3. Intensity is physical — track the mapping.**
AFM intensity is not arbitrary. It is height (nm), voltage (V),
phase (deg), current (A), etc. Whenever the image is stored or
converted to a bounded dtype (uint8 0–255, uint16 0–65535, float
0–1), a linear mapping is imposed between physical units and dtype
units. **You must remember and carry this mapping through every
step of the pipeline.**

Required discipline:
- On load, read `data_range_minimum`, `data_range_maximum`, and `data_range_units` (or equivalent) and the unit from
  metadata. Store them alongside the array.
- If converting dtypes (e.g. for visualization, SAM input, or
  saving), record the mapping:
  `physical = dtype_value * (data_range_maximum - data_range_minimum) / dtype_max + data_range_minimum`
- Any operation that produces a physically meaningful number
  (step height, roughness Rq/Ra, domain contact potential, grain
  depth) must convert back to physical units before reporting.
- Never report "intensity = 137" for a height map. Report
  "height = 4.82 nm".
- Be careful with operations that break the mapping: histogram
  equalization, CLAHE, per-window normalization, and any nonlinear
  contrast stretch destroy the physical scale. If you must use them
  for a downstream task (e.g. feeding SAM), keep a parallel copy of
  the physically-scaled image and do measurements on that copy.

## validation
### foundational
Validate that the preprocessing was correct before validating the
science.

**Preprocessing checks:**
- After row alignment, the horizontal-stripe pattern in the raw
  image should be gone; a line-profile across fast-scan direction
  should not show a global row-to-row offset. If residual stripes
  remain over featureless areas, escalate from median subtraction
  to per-row polynomial.
- After pixel-square resampling, the aspect ratio of known objects
  (e.g. circular grains should look circular, not elliptical).
  Confirm the recorded nm/px matches `field_of_view_x / N_x_new`
  and `field_of_view_y / N_y_new` to within floating-point tolerance.
- The physical-unit mapping must round-trip: converting physical →
  dtype → physical should recover the original values within the
  quantization error of the dtype.

**Scientific checks:**
- Any reported length (grain size, step width, roughness correlation
  length) is in nm/µm, not pixels.
- Any reported intensity-derived quantity (step height, surface
  potential contrast, Rq, Ra) is in the correct physical unit, with
  the unit stated explicitly.
- Reported values are physically plausible for the technique:
  topographic steps on layered materials are typically 0.3–1 nm per
  layer; KPFM contrasts are typically tens of mV to a few hundred
  mV; PFM phase contrast between antiparallel domains is ~180°.
  Flag values that are orders of magnitude off — they usually
  indicate a broken unit mapping, not real physics.
- For FFT/NMF-based analyses on PFM/KPFM-style data: acceptable
  output has non-noise frequency content and spatially coherent
  abundance maps on the square-pixel image. Do not over-interpret
  components as "domains" without corroboration from the physical-
  unit image.

Do not penalize an analysis for having preserved the raw line-to-line
baseline only when the features are genuinely row-correlated and the
choice was documented. Do penalize any analysis that reports
dimensionless pixel counts or dtype values in place of physical
quantities — that is a bug, not a stylistic choice.
