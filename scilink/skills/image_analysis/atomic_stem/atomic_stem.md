---
description: Atomic-resolution STEM (HAADF, MAADF) image analysis — column detection, sublattice separation, lattice metrology, and defect identification on zone-axis crystalline materials.
---
# STEM Atomic Resolution Image Analysis Skill

## overview

Atomic-resolution STEM image analysis (HAADF, MAADF). Individual atomic
columns are resolved as bright spots on a dark background. Applicable
to any crystalline material viewed along a zone axis. Covers column
detection, sublattice separation, lattice characterization, defect
identification, and structural variation analysis.

## planning

### foundational
**Pick ONE focused goal for this step.** This skill describes a toolkit
that covers detection, sublattice separation, lattice characterization,
defect identification, and strain analysis — but a single planning call
should answer one of these, not all of them. Common one-step goals
(pick the one the user's objective implies; if none is given, default
to detection + count):

- detect atomic columns and report count + per-column statistics
- separate sublattices in a multi-component structure
- measure lattice parameters / identify zone axis
- identify vacancies / missing columns relative to an ideal lattice
- map displacements / strain relative to an ideal lattice

Each goal becomes its own focused pipeline. Follow-up goals — sublattice
separation built on already-detected positions, displacement maps built
on an already-fit lattice — are best expressed as a separate
`run_analysis` call with `prior_analysis_paths` pointing at this run's
output, not appended to this plan.

**Detection vs. pattern-level analysis (`run_fft_nmf_analysis`):**
inspect the image for pattern-level heterogeneity — visible textures
or phase-like regions, disorder or defects at a scale coarser than
individual atoms, or atomic detail that is noisy or low-contrast (where
peak finding would be unreliable). If any of these is present, or the
objective targets disorder / defects / phase separation,
`run_fft_nmf_analysis` with a window size tuned to the feature scale
is a complete pipeline by itself. Otherwise pick atom-resolved
detection.

**For atom-resolved detection — choose the detector:**

Default to `detect_atoms_dcnn` when the material is in its training set
**and** `fov_nm` is available from metadata; otherwise use the
classical `detect_atoms`.

- `detect_atoms_dcnn` (AtomNet3 DCNN ensemble): best for transition-
  metal oxides (perovskites, layered perovskites, cuprate
  superconductors) and graphene; needs `fov_nm` from metadata.
  **Pass the raw image without preprocessing** — no CLAHE, no contrast
  normalization, no background subtraction, no bandpass filtering. The
  model is trained on raw images and handles intensity gradients
  internally; added preprocessing degrades detection. If weak columns
  are missed, lower the `threshold` parameter; do not preprocess.
- `detect_atoms` (classical peak detection): more general-purpose
  baseline; use when material is outside the DCNN's training set, when
  `fov_nm` is unknown, or when DCNN results look poor. **Preprocessing
  applies here, not to the DCNN path** — background subtraction or
  bandpass filtering before detection helps with non-uniform
  illumination.

Refine detected positions with 2D Gaussian fitting (built into
`detect_atoms`, available via `refine_positions` after
`detect_atoms_dcnn`) for sub-pixel precision.

**Validate the choice with an expected count:** estimate the visible
atomic columns from image dimensions, pixel calibration (if available),
and known lattice parameters for the material — count all sublattices
visible in HAADF (number of unit cells × visible column types per
cell). Compare detected count against this estimate; significant
discrepancy (outside 0.90-1.10×) suggests detection issues or a
structurally interesting region worth reporting.

**Calibration awareness.** Absolute spacings derived from images
typically carry a few percent uncertainty in scale: pixel-size metadata
may be approximate, scan distortion can be anisotropic (different stretch
along fast vs. slow axis), and older datasets can be off by 5%. Account
for this when designing the step's `quality_criteria`:

- Do **not** write quality criteria as a hard absolute-value match
  against bulk literature values (e.g. *"measured a-axis must be within
  1% of 0.38 nm"*). A 3-5% offset between measurement and literature is
  consistent with calibration error, not an analysis failure, but a
  tight criterion will fail and trigger pointless retries.
- Prefer **internally consistent** criteria the data can actually
  satisfy: ratios (e.g. *"`b/a` within 5% of expected ratio"* — cancels
  scale), FFT self-consistency (the reciprocal-lattice peaks form a
  consistent grid), or fit residuals in the data's own units (lattice
  fit residual / lattice spacing — dimensionless).
- An absolute lattice-value match against literature is fine as an
  **informational** check ("measured 0.38 nm matches YBCO a-axis to
  ~3% — consistent within calibration"), not as a pass/fail.

### advanced
**Tool reference:** detection and refinement helpers live in
`scilink.skills.image_analysis.atomic_stem.atom_finding` (`detect_atoms`, `detect_atoms_dcnn`,
`refine_positions`, `find_zone_axes`, `find_missing_atoms`,
`subtract_atoms`). Detailed parameter docs and per-tool usage are in
the `analysis` section below — refer to it when the goal you picked
above needs a specific tool.

**Goal-specific guidance** — apply only the bullet that matches the
goal you picked above:

- *If goal is detection + count:* one detector call, refinement (if
  not built in), and a focused interpretation is the complete pipeline.
  Do not add FFT, zone-axis analysis, or sublattice clustering to the
  same step.
- *If goal is sublattice separation:* three approaches, choose based
  on the data — see the `analysis` section for code:
  (a) **iterative detect-subtract-detect** when sublattices have
  noticeably different intensities (the classical Z-contrast case);
  (b) **local-environment GMM** when intensity alone is ambiguous and
  the neighborhood arrangement disambiguates the species;
  (c) **intensity + positional clustering** when both intensity and
  fractional position within the unit cell carry signal.
  Intensity-based clustering alone is insufficient for complex
  structures. Verify stoichiometric ratios in all cases. Detected
  positions from a prior detection step should come from
  `prior_analysis_paths`, not be re-detected here.
- *If goal is lattice parameter / zone axis:* use FFT for periodicity
  (NN distance is not the lattice parameter for multi-sublattice
  structures — true unit cell may be 2× or more of the shortest column
  spacing) and `find_zone_axes` for lattice vectors.
- *If goal is vacancy / missing-column search:* requires an ideal
  lattice — use detected positions plus zone vectors from a prior step
  (load via `prior_analysis_paths`). Compare ideal sites to detected
  positions; restrict to image interior; verify candidates with forced
  Gaussian fits.
- *If goal is displacement / strain mapping:* requires an ideal
  lattice from a prior step. Map displacements spatially; report only
  distortions exceeding the position fit uncertainty. Distinguish
  fitted-lattice residuals (local disorder) from deviations against a
  known ideal lattice (true strain).

## analysis

### foundational
**Workflow shape depends on the detector chosen in planning** (see
`## planning` for the decision):

- **DCNN path** (`detect_atoms_dcnn`): pass the raw image as-is →
  built-in detection → `refine_positions` for sub-pixel coordinates and
  Gaussian parameters → measure basic statistics. **No preprocessing
  step.** The model handles intensity gradients internally; CLAHE,
  contrast normalization, background subtraction, and bandpass
  filtering on the input degrade detection.
- **Classical path** (`detect_atoms`): normalization → optional
  background subtraction or bandpass filtering for non-uniform
  illumination → built-in peak detection → 2D Gaussian refinement
  (built into `detect_atoms` via `refine=True`).

Both paths end with the same statistics: column count, intensity
distribution, nearest-neighbor distances, and lattice parameters from
FFT.

### advanced

**Atom finding tools — detailed usage:**

`detect_atoms(image, separation, threshold_rel=0.02, refine=True, percent_to_nn=0.4, subtract_background=False, normalize_intensity=True)`
finds atomic column positions with optional sub-pixel Gaussian refinement.
- **separation** (int): minimum atom spacing in **pixels**. Estimate from
  known lattice parameter / pixel size, or from FFT peak position:
  `separation ≈ image_width / (2 × FFT_peak_distance_from_center)`.
  If unsure, use 70-80% of the apparent nearest-neighbor distance.
- **threshold_rel** (float): peak sensitivity. Default 0.02. Raise to
  0.05-0.1 for noisy images; lower to 0.01 for faint columns.
- **percent_to_nn** (float): Gaussian fit mask as fraction of NN distance.
  0.4 default. Increase for sparse lattices, decrease for dense.
- **subtract_background** (bool): Gaussian-blur background subtraction
  before peak finding. Default False. Enable for images with strong
  intensity gradients.
- **normalize_intensity** (bool): Normalize image to 0-1 before peak
  finding. Default True.
- Returns dict: `"positions"` (N,2 as x,y where x=col y=row),
  `"sigma_x"`, `"sigma_y"`, `"amplitude"`, `"rotation"` (all N arrays).

`detect_atoms_dcnn(image, fov_nm, model_dir=None, target_pixel_size=0.25, threshold=0.8, refine=True)`
detects atom columns using an AtomNet3 DCNN ensemble.
- **fov_nm** (float): field of view in **nanometers** (from metadata).
- **target_pixel_size** (float): target pixel size in Angstroms for
  the model. Default 0.25. May need tuning for different materials.
- **threshold** (float): detection confidence, 0-1. Default 0.8.
- Returns dict: `"positions"` (N,2 as x,y in original image pixels),
  `"heatmap"` (2D probability map, pre-threshold),
  sigma/amplitude/rotation are None. Use `refine_positions` to obtain them.

`refine_positions(image, positions, percent_to_nn=0.4)`
fits 2D Gaussians at known atom positions to get sub-pixel coordinates
and per-atom sigma, amplitude, and rotation. Use after `detect_atoms_dcnn`
or any other source that lacks Gaussian parameters (needed for
`subtract_atoms`).
- **positions** (N,2): atom positions as (x, y) from any detection method.
- Returns dict with `"positions"`, `"sigma_x"`, `"sigma_y"`,
  `"amplitude"`, `"rotation"` — same format as `detect_atoms(refine=True)`.

`find_zone_axes(positions, n_neighbors=9, distance_tolerance=None)`
detects lattice translation vectors by clustering displacement vectors.
- **n_neighbors**: 9 for simple lattices, 15-25 for complex unit cells.
- Returns list of (dx, dy) tuples, shortest first. Square lattice → 2
  vectors, hexagonal → 3. The shortest vector is the NN distance; the
  **lattice parameter** may be 2×+ for multi-sublattice structures.

`find_missing_atoms(positions, zone_vector, fraction=0.5, min_distance=3.0)`
predicts positions at fractional lattice sites along a zone vector.
- **fraction**: 0.5 = midpoint (binary compounds), 0.33/0.67 (ternary).
- **min_distance**: discard predictions within this distance of existing
  atoms (set to ~separation/3).
- Returns (M,2) predictions. Verify that the image has intensity there.

`subtract_atoms(image, positions, sigma_x, sigma_y, amplitude, rotation=None)`
removes fitted Gaussians from the image. Requires per-atom Gaussian
parameters — use `detect_atoms(refine=True)` or `refine_positions()`
to obtain them. Returns residual (clipped >= 0) where subtracted regions
drop to background. Run detection on the residual with lower threshold
to find the next sublattice.

**Multi-sublattice workflow (classical):**
```
result1 = detect_atoms(image, separation, refine=True)
zone_vecs = find_zone_axes(result1["positions"])
predicted = find_missing_atoms(result1["positions"], zone_vecs[0], fraction=0.5)
residual = subtract_atoms(image, result1["positions"],
                          result1["sigma_x"], result1["sigma_y"],
                          result1["amplitude"], result1["rotation"])
result2 = detect_atoms(residual, separation, threshold_rel=0.01, refine=True)
```

**Multi-sublattice workflow (DCNN):**
```
dcnn1 = detect_atoms_dcnn(image, fov_nm)
result1 = refine_positions(image, dcnn1["positions"])
zone_vecs = find_zone_axes(result1["positions"])
residual = subtract_atoms(image, result1["positions"],
                          result1["sigma_x"], result1["sigma_y"],
                          result1["amplitude"], result1["rotation"])
result2 = detect_atoms(residual, separation, threshold_rel=0.01, refine=True)
# Save dcnn1["heatmap"] to visualize detection confidence
```
Stop when residual has no peaks above 3× noise std. Validate: check
stoichiometric ratios, heavier atoms should have higher amplitude,
each sublattice's NN distances should be consistent.

**Sublattice separation** — three approaches, choose based on the data:
1. **Iterative detect-subtract-detect:** detect and refine the brightest
   columns, subtract them, detect the next brightest on the residual,
   repeat until no peaks remain. Separates by geometry without clustering.
2. **Local environment GMM:** crop a small square window (side length
   approximately equal to the lattice parameter) centered on each
   detected column, flatten each crop into a 1D vector, stack them into
   an (N, window*window) matrix, and cluster with GMM. Each cluster
   centroid is an average local environment image. This captures the
   full neighborhood (neighboring column arrangement, not just peak
   intensity) — useful when intensity alone is ambiguous.
3. **Intensity + positional analysis:** cluster by raw column intensity
   combined with fractional position within the unit cell.

Use raw (unnormalized) column intensities for any intensity-based
analysis — local normalization removes the Z-contrast difference
between species. Verify that each sublattice has consistent intensity
and the expected stoichiometric ratio.

**Defect identification:** Compare detected positions to ideal lattice
sites. Restrict vacancy search to the interior of the detected region
to avoid edge false positives. Verify vacancy candidates with forced
Gaussian fit at expected position.

**Strain and displacement:** Map displacements from ideal lattice
positions spatially across the image. Only report distortions exceeding
the position fit uncertainty.

## interpretation

### foundational
**HAADF intensity:** Scales as ~Z^1.6-2. Brighter columns contain
heavier atoms.

**Lattice parameters:** Compare against known bulk values, treating a
few-percent absolute deviation as expected calibration uncertainty.
Report in both pixels and physical units when calibration is
available, and note the calibration-driven uncertainty band when
matching against literature.

### advanced
**Sublattice assignment:** Use chemical identity from intensity and
positional analysis to interpret which sublattice corresponds to which
atomic species.

**Strain:** Distinguish fitted-lattice residuals (local disorder) from
deviations against known ideal lattice (true strain). Least-squares
fitting absorbs mean strain — use known lattice constants when
available.

**Vacancy concentration:** In pristine crystals, typically 0.01-1%.
Above 5-10% usually indicates detection or fitting error, unless the
sample was intentionally modified (irradiation, beam damage, quenching).

## validation

### foundational
**Detection completeness:** Detected vs expected column count (from
image area and unit cell) should be within 0.90-1.10.

**NN distance consistency:** CV below 15% for well-ordered crystals.

**Unit cell sanity:** Measured lattice parameters should be in the
right ballpark of known bulk values, but absolute scale carries a
few-percent calibration uncertainty. Treat 3-5% deviations from
literature as informational (likely calibration), not a pass/fail
failure. Hard checks should be self-consistency: ratio of measured
spacings (b/a) matching the expected ratio, or FFT peaks forming a
consistent reciprocal lattice.

**DCNN preprocessing check:** if the step uses `detect_atoms_dcnn`,
flag any pipeline step that preprocesses the image before the DCNN
call (CLAHE, contrast normalization, background subtraction, bandpass
filtering, etc.) — these belong only on the classical
`detect_atoms` path. See the planning-section detector selection for
the rationale.

### advanced
The following only apply when the step explicitly targets the named
goal — not as additional checks for a basic detection step.

**Sublattice populations** (when the step assigns sublattices): should
match expected stoichiometry for the material. Heavy doping can shift
intensities between clusters.

**Lattice fit residual** (when the step fits an ideal lattice): below
0.3× the lattice spacing.

**Displacement field** (when the step maps displacements): mean
displacement from ideal lattice should be small (<0.3× lattice
spacing). Large systematic displacements indicate fitting errors, not
real strain.
