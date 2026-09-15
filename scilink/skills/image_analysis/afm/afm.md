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

Estimate each row's baseline from that **row's own** substrate — a
robust low estimator per row (its lowest histogram mode or a low
percentile), not from a single global-threshold substrate mask. A
smooth plane fit cannot remove an abrupt baseline step partway down a
scan (a tip event or feedback jump), and that is exactly where a global
mask fails: the stepped band's substrate sits *above* the global
threshold, so it is dropped from the mask, its per-row offset is
interpolated (≈0) from neighbors instead of measured, and the band is
left elevated — where it is then misread as a raised terrace/layer.
Correct each row against its own local substrate so a genuine
row-baseline step is removed regardless of the global level.

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

**Deliverable when the goal is to characterize orientations of domains
or patches.** "Quantify *its* orientation" means every identified
domain/patch gets its own reported orientation angle — not just aggregate
class counts or a dominant-orientation summary. Deliver all three:
1. **A per-domain angle value.** `features.csv` (or the saved table) must
   have one row per domain with its orientation angle in degrees, keyed to
   the same domain id used in the label map — so each domain's orientation
   is a readable number, not only a color. Aggregate-only statistics
   (n_classes, dominant angle) do not satisfy "quantify each domain's
   orientation."
2. **Angle labels on the overlay.** Print each domain's angle (e.g.
   "143°") at its centroid on the spatially-resolved map, so the number
   and its location are legible together.
3. **A spatially-resolved, color-coded orientation map/overlay** — each
   domain/patch colored by its measured orientation (cyclic colormap),
   distinct from a domain-*identity* map (which colors domains to tell
   them apart, not by angle). It **must carry a cyclic color-wheel legend**
   mapping hue → angle (not a linear colorbar — orientation wraps at
   0°≡180°); a color map with no legend cannot be read as angles.
**Fold orientation to the lattice symmetry.** A 2D lattice's orientation is
only defined modulo its symmetry — square/p4 folds mod 90°, hexagonal/p6 mod
60°, oblique mod 180°. Establish the symmetry (from the FFT spot geometry — spot
count and angular spacing: 4 spots 90° apart = square/rectangular, 6 spots 60°
apart = hexagonal) before reporting angles; using the wrong modulus makes a
single grain's symmetry-equivalent lattice directions read as different
orientations, so the per-grain mean is meaningless and its within-grain spread
balloons.

**Determine symmetry from a single-orientation region, never the whole-field
FFT of a grain mosaic.** When the field is many grains at different
orientations, the global FFT superimposes each grain's reflections into a
smeared *ring* — the azimuthal spot count and spacing that distinguish 4-fold
from 6-fold are exactly what the smearing destroys, so the global FFT cannot be
trusted for the point group (it also readily mis-reads a square lattice's
diagonal (11) reflections as extra spots → a false hexagon). Instead FFT each
grain's *interior* (single orientation → discrete, countable spots), classify
its symmetry, and take the cross-grain consensus; the global FFT is for the
average *period* only, never the symmetry. **When there are no grains** — a
single-domain / single-crystal field, or segmentation returns one region — the
whole image is already single-orientation, so read the symmetry from its FFT
directly; the smearing problem is specific to multiple co-present orientations.
State the symmetry with the evidence (spot count/spacing) and, if the spots are
too smeared or sparse to decide, say so rather than committing to one.

**Report orientation with a per-grain confidence — do not fabricate, and do
not cull with an absolute threshold.** Give *every* grain an angle *and* a
continuous reliability measure (the circular resultant length R, or the
within-grain angular spread, of its per-window estimates), so an ill-defined
grain is self-evidently unreliable from its own number rather than either
reported as confident or silently dropped. Judge whether an angle is meaningful
by **consistency relative to noise, not an absolute signal cut**: a grain's
orientation is resolved when its per-window estimates agree far more than they
would for an isotropic / phase-randomized null of that same grain (high R vs the
null's R) — this self-calibrates to each image's noise floor. Do NOT gate on a
fixed absolute Bragg-SNR / "SNR-passing-window count"; that is right for one
dataset and wrong for the next (too permissive → confident garbage, too strict →
real grains vanish). Below the physics floor there is nothing to fold: if the
period is at/near the 2-px Nyquist limit the orientation is unresolvable for
every grain regardless of consistency. Mark a low-confidence grain
"unresolvable" (with its spread) rather than as `ok`; a large within-grain
spread, or the same exact angle repeated across grains, is the signature of a
forced measurement. State reliability honestly: grid-snapped or FFT-quantized
values are relative/class labels, not precise crystallographic angles.

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
- The leveled substrate must be flat across the **whole field**, not
  just where the substrate mask landed. Check for a residual step: a
  full-width band (typically at a scan edge) that reads as bare
  background yet sits elevated is a leveling failure, not a layer.
  The tell in the output is a large contiguous background-looking
  region assigned to a nonzero terrace/height class; substrate-only
  flatness metrics will look clean because that band was dropped from
  the substrate mask, so judge it from the assignment map's geometry.
  If found, re-level (per-row-local baseline) before trusting any
  per-layer coverage or thickness statistics.
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
- If the objective was to characterize domain/patch orientations,
  confirm the results deliver each domain's orientation as a value, not
  just aggregate classes: a per-domain row in the saved table carrying its
  orientation angle, angle labels printed on the map at each domain, and a
  color-coded orientation overlay that carries a cyclic color-wheel legend
  (hue → angle). A run that reports only class counts / a dominant
  orientation, colors domains by identity rather than angle, or shows an
  orientation map with no color→angle legend has NOT met "quantify each
  domain's orientation" — even if it segmented the domains correctly.
- But a *confident* per-grain angle is worse than an honest decline when the
  measurement is not resolvable. Confirm the run established the lattice
  symmetry and folded to its modulus, and that each grain carries a continuous
  confidence (R or within-grain spread) judged against a noise null rather than
  culled by a fixed absolute-SNR cut. Large within-grain spreads (tens of
  degrees), an exact angle repeated across grains, or angles reported with no
  symmetry established are signs of a forced measurement. Equally, resolving
  only a tiny fraction of clearly-lattice-bearing grains points to an
  over-strict absolute gate, not honest caution — the criterion should scale
  with the image's own noise, not a hard threshold.
- When the figure marks FFT reflections (to report a lattice parameter or the
  symmetry), the markers must sit on the **actual detected peaks** — locate each
  peak as the argmax within an angular window on the ring, not as an idealized
  rosette of equally-spaced points drawn at an assumed orientation. Finding the
  ring *radius* from the radial power profile fixes the period but not the peak
  *angles*; markers placed at assumed angles land on empty ring positions and
  misrepresent where the peaks are. Confirm each marker overlies a bright spot;
  a marker with no peak under it is a visualization bug even when the measured
  period is correct.

Do not penalize an analysis for having preserved the raw line-to-line
baseline only when the features are genuinely row-correlated and the
choice was documented. Do penalize any analysis that reports
dimensionless pixel counts or dtype values in place of physical
quantities — that is a bug, not a stylistic choice.
