---
description: Atomic-resolution STEM/HRTEM image analysis — column detection, sublattice separation, lattice metrology, defect identification, and superlattice/satellite-reflection mapping (ordered-domain & second-phase localization) on crystalline zone-axis or lattice-fringe images.
---
# STEM Atomic Resolution Image Analysis Skill

## overview

Atomic-resolution STEM (HAADF, MAADF) and HRTEM lattice-fringe image
analysis. Individual atomic columns are resolved as bright spots on a
dark background (STEM), or the crystal shows resolved lattice fringes
(HRTEM). Applicable to any crystalline material viewed along a zone
axis. Covers column detection, sublattice separation, lattice
characterization, defect identification, structural variation analysis,
and Fourier (reciprocal-space) mapping of superlattice / satellite
reflections to localize ordered domains, second phases, and
superstructures.

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
- detect & spatially map a superlattice / satellite reflection
  (localize ordered domains, second phases, or superstructures)

Each goal becomes its own focused pipeline. Follow-up goals — sublattice
separation built on already-detected positions, displacement maps built
on an already-fit lattice — are best expressed as a separate
`run_analysis` call with `prior_analysis_paths` pointing at this run's
output, not appended to this plan.

**Pixel size / FOV (calibration):** whenever a step needs pixel size in nm
(e.g. `fourier_reflection_map`, spacing measurements), resolve it with the
shared helper, not inline arithmetic:
`from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm`;
`px = resolve_pixel_size_nm(metadata, image.shape)` → `{"x","y","source"}` nm/px,
or `None`. It divides `field_of_view` by the image **shape** — never divide by a
metadata pixel-count field like `n_cols`/`width`, which is usually absent and
silently leaves pixel size `None`. (`detect_atoms_dcnn` still takes `fov_nm`
directly from metadata.)

**Detection vs. pattern-level analysis:**
inspect the image for pattern-level heterogeneity — visible textures
or phase-like regions, disorder or defects at a scale coarser than
individual atoms, or atomic detail that is noisy or low-contrast (where
peak finding would be unreliable). If any of these is present, or the
objective targets disorder / defects / phase separation / "identify the
lattice domains/phases", do a **pattern-level** analysis (next
paragraph) rather than atom-resolved detection. Otherwise pick
atom-resolved detection.

**Pattern-level: detect reflections FIRST, then route.** For a
crystalline lattice image, the cheap, rigorous first move for almost any
"what domains/phases are here" question is `fourier_reflection_map`
(no `d_nm`) — a reflection census that returns each spacing with a
significance and flags satellites. It tells you which branch you are in:
- a **satellite / superstructure** reflection is present (an ordered
  domain, second phase, vacancy/charge ordering, moire) → **map that
  reflection** with `fourier_reflection_map(image, px, d_nm=...)`; the
  amplitude map localizes the ordered domain and `spot_snr_domain ≫
  spot_snr_bulk` confirms it. This is sharper and more interpretable
  than NMF and is the route the η-precipitate / vacancy-superstructure
  class of objectives needs.
- domains differ only in **orientation** at the *same* spacing (e.g.
  **twins** — reflections share |k|, differ in direction), or the
  heterogeneity is genuinely **unknown / exploratory** → use
  `run_fft_nmf_analysis` (window-FFT + NMF), the unsupervised baseline.
- nothing clears the significance floor → report no resolvable lattice.

The two tools are complementary, not interchangeable: `fourier_reflection_map`
is the sharp, interpretable route for a specific reflection / ordered
phase; `run_fft_nmf_analysis` is the exploratory baseline for unknown or
orientation-only heterogeneity.

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

**QC the detection with `detection_quality_panels`, not a count.** After
detecting columns, call the registered tool `detection_quality_panels`
(image, positions, pixel_size_nm, and the DCNN `heatmap` when available)
and **save its `figure_bytes` as the step visualization** — a full-frame
overlay of thousands of marks is an unjudgeable haze, so the tool adds
targeted zoom-in panels (placed at the most-suspect regions) plus the
nearest-neighbor distance histogram, which is what actually reveals
problems. Read its `metrics`: `short_pair_fraction` (a spike of
anomalously short NN distances → duplicate/split marks = over-detection),
`coverage_gap_fraction` (large unmarked regions = misses), and
`heatmap_hit_fraction` (detections off every DCNN peak = spurious). Do
NOT judge detection by an absolute count or an a-priori "expected count":
the resolved columns-per-cell depends on the zone axis and on which
columns are bright enough to resolve, which is unreliable to predict for
multi-sublattice / complex structures — so a moderate detected/expected
ratio (e.g. ~1.3×) is NOT over-detection. An expected count is at most an
order-of-magnitude sanity check (~0.5-2×).

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
`subtract_atoms`). For the superlattice / satellite-reflection-mapping
goal, use `fourier_reflection_map`
(`scilink.skills._shared.fourier_reflection`) — the registered,
reciprocal-space tool that detects reflections and maps a chosen one
(amplitude + GPA phase), null-gated. For the point-defect /
vacancy-search goal, the reciprocal-space route is `fft_defect_map`
(`scilink.skills._shared.fft_defect`) — perfect-lattice reconstruction
plus null-gated residual anomalies, no atom finding required. Detailed parameter docs and
per-tool usage are in the `analysis` section below — refer to it when
the goal you picked above needs a specific tool.

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
- *If goal is lattice parameter / lattice constant / nearest-neighbor
  distance:* use the registered tool **`measure_lattice_constant`** — the
  fast, deterministic default for this exact question. It detects the Bragg
  peaks, picks the reciprocal basis by translational support (so a {200}
  harmonic and the centered {110} a/√2 sub-cell can NOT be mis-reported as
  the fundamental), and returns the axis-resolved cell (`a1_nm`, `a2_nm`,
  `gamma_deg`), `lattice_constant_nm`, and `nn_distance_nm` with `nn_basis`
  stating the projection relation (NN = a, or a/√2 for a centered/perovskite
  sublattice). Crop to a single crystalline domain first (exclude
  substrate/vacuum) and pass the true square-pixel size. ALWAYS check the
  returned `multi_lattice` / `low_confidence` flags: if set, the field of
  view holds more than one lattice or is under-sampled — crop to one domain
  (ROI) and re-run rather than trusting the number. Do NOT hand-roll this
  from `fourier_reflection_map` + manual fundamental-picking (that is the
  exact failure mode — choosing the strongest σ reflection, often a harmonic);
  `measure_lattice_constant` exists to remove it. Use `find_zone_axes` (real
  space) instead only when you need per-site / spatially-resolved cell maps.
  A lattice vector is the translation that maps the crystal onto itself — one
  column onto the next column **of the same species** — so when the
  projection resolves more than one inequivalent column per cell (multiple
  sublattices / a basis), the NN distance is *shorter* than the lattice
  constant by a projection factor (√2, √3, 2×, …); the tool's `nn_basis`
  makes this explicit. Report the quantity the objective names: "lattice
  constant / parameter / vector" → the repeat period; "nearest-neighbor
  distance / bond length" → the NN. If the request offers them as
  alternatives or is ambiguous, report both, explicitly labeled, with their
  geometric relation — never silently substitute one for the other.
  **This applies even when the lattice constant is a BYPRODUCT of another
  objective** (e.g. reporting the cell while characterizing fringe spacing for
  a defect study): whenever a lattice constant, cell parameter, or inter-axis
  angle is reported, get it from `measure_lattice_constant`, NOT by hand-picking
  two spacings from a `fourier_reflection_map` census as "the axes". The census
  lists reflection *spacings*, not lattice vectors — pairing e.g. a {110}
  face-diagonal (≈ a/√2) with a {100} axial spot yields a spurious ~45°
  "oblique" cell. `measure_lattice_constant` resolves the true basis (and flags
  `low_confidence` when the lattice is layered or under-sampled, e.g. below
  ~12 px per cell), so route cell determination through it every time.
- *If goal is locating or excluding a film/substrate (or grain) interface:*
  between two lattice-matched phases (e.g. a perovskite film on a perovskite
  substrate) the boundary is CHEMICAL, not structural — the reflection set /
  lattice spacing is identical on both sides, so FFT/periodicity-based
  detection is physically blind to it. The discriminator is the per-column
  **Z-contrast step** (HAADF intensity ∝ Z², so a heavier-cation substrate is
  brighter) or an ordering **superstructure** that appears/disappears across
  the boundary. Therefore do NOT flat-field / background-divide before
  interface detection: that erases the very Z-contrast step that marks the
  interface along with any global shading. A *smooth* low-frequency brightness
  ramp is a shading/thickness artifact and is not the interface; a *sustained
  per-column intensity step* is — separate them with a per-row mean
  column-intensity profile (on the raw image) and a significance-tested
  superstructure-satellite map, not a row-mean of flat-fielded intensity. If
  neither a Z-contrast step nor a superstructure change is found, the interface
  is genuinely absent from the field of view (report that) — but do not
  conclude "no interface" merely because the FFT was uniform.
- *If goal is vacancy / missing-column search:* two complementary
  routes — pick by data quality, or run both as a cross-check.
  **Real-space route** (defect typing, needs reliable columns): requires
  an ideal lattice — use detected positions plus zone vectors from a
  prior step (load via `prior_analysis_paths`). Compare ideal sites to
  detected positions; restrict to image interior; verify candidates with
  forced Gaussian fits. **Reciprocal-space route** (no atom finding):
  the registered tool `fft_defect_map` (see `analysis`) reconstructs the
  perfect lattice from its significant reflections and maps localized
  deviations against a noise null — prefer it when columns are too noisy
  / low-dose to detect reliably, when the field of view is large, or as
  an independent confirmation of real-space candidates. It returns typed
  signatures (deficit vs excess, lattice-coherence dip) but candidates
  still need real-space confirmation crops for chemical interpretation.
- *If goal is planar-defect detection (stacking fault, intergrowth, anti-
  phase / twin boundary):* the defect is a *localized break in periodicity* —
  a fringe-spacing jump, inserted/missing plane, or lateral phase shift across
  a line. General caution (the same one as flat-fielding an interface): any
  preprocessing that suppresses a background/banding component — de-streaking,
  row/column-mean or background subtraction, high-pass — will also erase a
  defect that lives in that component, so search on data that preserves it.
  Localize and classify it with `lattice_discontinuity_map` (next bullet),
  which runs on the RAW image; separate a localized structural discontinuity
  (the defect) from a periodic full-width modulation (normal layering) or a
  uniform line (detector artifact).
- *If goal is displacement / strain mapping (incl. dislocations / Burgers
  vectors, precipitate-interface coherency, twin-boundary displacement):*
  the dedicated tool is the **`gpa_strain` skill** (Geometric Phase
  Analysis) — it returns the referenced in-plane strain tensor
  (εxx, εyy, εxy) and lattice rotation (ωxy) against an undistorted
  reference region, without first detecting atoms. Co-activate it; do NOT
  hand-roll strain from the *unreferenced* raw FFT phase. Its caveat: GPA
  assumes ONE dominant lattice, so it is for SMALL distortions / coherent
  regions — across a large misorientation use `lattice_discontinuity_map`
  (below) instead. (A real-space alternative when columns fit cleanly: map
  displacements of detected positions vs an ideal lattice from a prior
  step.)

- *If goal is locating / classifying a grain or twin boundary, a planar
  defect (stacking fault, intergrowth, antiphase band), or an incoherent
  interface:* use the registered tool **`lattice_discontinuity_map`** — a
  sliding-window local-FFT map whose neighbour spectral dissimilarity
  detects the boundary line and whose orientation/spacing jumps classify
  it (orientation→grain/twin, spacing→interface/second-phase, coherence
  drop only→stacking fault/disorder). Save its `figure_bytes` as the
  visualization and read `boundaries`. Run it on the RAW image — do not
  de-streak / background-subtract first (that erases a defect that lives
  in the banding). Pick among the reciprocal-space tools by question:
  `run_fft_nmf_analysis` is the EXPLORATORY decomposer (unknown
  heterogeneity, "how many domains"); `lattice_discontinuity_map` is the
  sharp LOCALIZER (where is the boundary + what type); `gpa_strain` gives
  the strain MAGNITUDE near it; `fourier_reflection_map` maps a specific
  superstructure reflection's domain. NOTE the scope: a COHERENT
  lattice-matched chemical interface (same orientation+spacing+coherence
  both sides) is invisible to this tool — detect that via the per-column
  Z-contrast step (see the film/substrate interface bullet above).
- *If the lattice-change tools above (`lattice_discontinuity_map`,
  `fourier_reflection_map`) come back silent/unconvincing BUT a sustained
  dark/bright trough or step crosses the field:* you are in the
  **intensity/thickness-step regime**, not the lattice-change regime. A
  boundary whose projected lattice is near-continuous — an INCLINED
  grain/phase boundary (seen as a Z-contrast/thickness trough) or a
  lattice-matched CHEMICAL interface — carries no orientation or spacing
  jump, so *no* lattice-orientation method (FFT, structure tensor, per-column
  angle map) can localize it: the signal is in the intensity field, not the
  lattice. Detect the step itself with the intensity-step recipe (see
  `## analysis`) — de-lattice the RAW image, isolate a sustained step from
  smooth shading, trace the connected gradient ridge (orientation-agnostic).
  Report it as an inclined/chemical boundary; if only smooth shading is
  present and no ridge forms, report no boundary rather than forcing one.

- *If goal is superlattice / satellite-reflection mapping:* use the
  registered tool `fourier_reflection_map` (see `analysis` for the call).
  It is a reciprocal-space, atom-detection-free pipeline (STEM or HRTEM
  lattice fringes) that already bakes in the steps and failure modes that
  make or break this analysis, so you do **not** hand-write them:
  detrended azimuthally-averaged radial-PSD detection with a σ
  significance (a per-tile peak/median is fooled by single noise spikes);
  a matched annular band-pass mapping on the **un-windowed** image
  (a spatial window tapers the edges and biases amplitude to the centre);
  a phase-randomized **null gate** (so noise and sharp interface edges
  are not flagged as ordered); and a **local-FFT spot-SNR** confirmation
  (domain ≫ bulk separates a real localized reflection from an edge
  artifact). Recommended two-step use:
  1. Call it once (no `d_nm`) to **detect** reflections — it returns each
     `d_nm` with a σ, an `integer_multiple_of` list (this reflection is
     N× a shorter significant one — a candidate **superstructure /
     satellite**: ordering, antiphase, second phase, moiré), and
     `strongest_satellite_d_nm`.
  2. To localize a superstructure, map **`strongest_satellite_d_nm`** —
     call again with `d_nm=that value`; inspect `amplitude_map`/
     `domain_mask` for *where* it lives and require `spot_snr_domain ≫
     spot_snr_bulk` before claiming it is real.
  **Do NOT try to identify "the fundamental" first, and do NOT hunt for a
  specific multiple** (e.g. "the N=2 satellite"). Which reflection is the
  true fundamental is ill-posed from a 1-D PSD (the strongest peak is
  often a harmonic), so anchoring on it and looking for "2× the
  fundamental" will MISS a genuinely strong superstructure at a different
  multiple. Trust σ: map the strongest satellite, whatever its N.

  Then report on the satellite's **origin honestly — three outcomes, and
  do NOT force a binary real/not-real verdict** (the origin frequently
  cannot be settled from one frame; over-claiming and over-dismissing are
  equally wrong):
  - `strongest_satellite_d_nm is None` → **no resolvable superstructure**
    (do not manufacture one).
  - mapped satellite localizes to a **compact domain** with
    `spot_snr_domain ≫ spot_snr_bulk` → report a **(likely real) ordered
    superstructure** — name it (vacancy-/charge-ordered, antiphase, moiré)
    as *consistent with* the data using domain context, with its spacing
    and where it lives.
  - mapped satellite is significant but its amplitude **traces a thin
    interface/edge line** (high aspect ratio, low phase coherence, or a
    `spot_snr_domain` that is not from a compact region) → report it as a
    **candidate of ambiguous origin**: state the satellite IS present and
    where it concentrates, that this is consistent with **either** a
    surface-/interface-nucleated ordered phase **OR** an interface edge
    artifact, and name the specific diagnostic that is concerning. Do
    **NOT** discard it as "just an edge artifact" (don't throw out a real
    reflection) and do **NOT** upgrade it to a confirmed superstructure.
    Say what would disambiguate — e.g. the same reflection measured in a
    region away from the interface, a tilt or defocus series, a dose
    series, or another movie frame.
  Never resolve this by silently switching to a hand-picked "fundamental":
  report the satellite you actually found, with calibrated uncertainty.

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

Both paths end the same way: QC the detection with
`detection_quality_panels` (save its `figure_bytes` as the visualization,
and judge over/under-detection from its metrics — see `## planning` and
`## validation`), then report column count, intensity distribution, and
nearest-neighbor distances. For the **lattice parameters**, use
`measure_lattice_constant` (the deterministic, harmonic-safe tool), not a
hand-rolled FFT-peak pick.

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

**FFT point-defect mapping** — use the registered tool `fft_defect_map`
(reciprocal-space, atom-detection-free) when columns are unreliable to
detect or as an independent cross-check. It bakes in the steps that
make or break a Bragg-filtered residual: a reflection significance gate
(else noise "peaks" fabricate defects), a phase-randomized null
threshold, period-scaled smoothing, mirror padding against border
artifacts, and a pattern-validity mask. The script calls it and
interprets; do not hand-write this pipeline.
```
from scilink.skills._shared.fft_defect import fft_defect_map

res = fft_defect_map(image, pixel_size_nm)
if res["periodic"]:
    # deficit + coherence_dip => vacancy-like; excess without dip =>
    # adatom / heavier dopant on an intact lattice
    for d in res["defects"]:
        crop = image[d["y"]-r:d["y"]+r, d["x"]-r:d["x"]+r]  # confirm in real space
```
Heed `res["warnings"]`: a single-cluster flag means the candidates trace
ONE extended feature (precipitate, boundary, contamination) — switch to
domain mapping (`fourier_reflection_map`) instead of counting sites. A
coherent twin boundary whose two variants both have significant
reflections is reconstructed as part of the pattern and will NOT appear.
If a clearly visible lattice returns `periodic=False`, the note reports
the best candidate snr — lower `min_peak_snr` only when that margin is
small; far below the gate means the image is not periodic.

**Intensity/thickness-step (Z-contrast) boundary detection** — for a
boundary that does NOT change the projected lattice and is therefore
invisible to every FFT/lattice-change tool: an INCLINED grain/phase
boundary (its projected lattice is near-continuous, so it shows only as a
Z-contrast/thickness trough) or a lattice-matched CHEMICAL interface. The
signal lives in the non-periodic intensity field, not the lattice. No
registered tool — generate the detector (it is standard image processing;
the choice of operator is yours). The non-obvious constraints, not the
exact code, are what matter: (1) work on the RAW image — flat-fielding
erases the very step you are after; (2) remove the periodic lattice
(unit-cell-scale local mean, or notch the Bragg peaks) to expose the
slowly-varying intensity field; (3) separate a *sustained, localized*
step/trough from a *smooth, global* shading ramp (e.g. against a
much-larger-scale background) — the ramp is thickness/shading, not a
boundary; (4) the boundary is an *extended line* (a connected ridge),
orientation-agnostic so it handles horizontal, vertical, and diagonal/V
loci alike, whereas an isolated compact dip is a point defect
(→ `fft_defect_map`). If no ridge forms and only smooth shading is present,
report no boundary rather than forcing one.

**Strain and displacement:** Map displacements from ideal lattice
positions spatially across the image. Only report distortions exceeding
the position fit uncertainty.

**Superlattice / satellite-reflection mapping** — use the registered
tool `fourier_reflection_map` (reciprocal-space, atom-detection-free;
works on STEM or HRTEM lattice fringes). It performs detection,
matched-band-pass amplitude/phase mapping, the phase-randomized null
gate, and the local-FFT confirmation internally, so the script only
calls it and interprets the result. Pass the **square-pixel** size in
nm (resample first if pixels are anisotropic).
```
from scilink.skills._shared.fourier_reflection import fourier_reflection_map

# 1) detect: reflections + the strongest satellite (the ordering to localize)
det = fourier_reflection_map(image, pixel_size_nm)
for r in det["reflections"]:
    print(r["d_nm"], r["sigma"], r["integer_multiple_of"])  # sigma = significance
sat = det.get("strongest_satellite_d_nm")  # highest-sigma satellite, or None
if not sat:
    print("no resolvable superstructure")  # -> report it; do not map a fundamental

# 2) localize the superstructure: map the strongest satellite (NOT a chosen N)
res = fourier_reflection_map(image, pixel_size_nm, d_nm=sat)
np.save("superlattice_amplitude.npy", res["amplitude_map"])   # where it lives
np.save("superlattice_domain.npy", res["domain_mask"])        # null-gated segmentation
real = res["spot_snr_domain"] > 3 * res["spot_snr_bulk"]      # confirm vs edge artifact
# res["phase_map"] is the GPA displacement/strain channel of the same reflection.
```
Default (no `d_nm`) maps the strongest reflection — deliberately NOT a
superstructure, so the tool does not presume one exists. Only call a
satellite "real" when `spot_snr_domain ≫ spot_snr_bulk`.

## interpretation

### foundational
**HAADF intensity:** Scales as ~Z^1.6-2. Brighter columns contain
heavier atoms.

**Lattice parameters:** Compare against known bulk values, treating a
few-percent absolute deviation as expected calibration uncertainty.
Report in both pixels and physical units when calibration is
available, and note the calibration-driven uncertainty band when
matching against literature.

**Nearest-neighbor distance vs lattice constant:** these coincide only
for a lattice with one column per projected cell. When the projection
resolves more than one column per cell (multiple sublattices / a basis),
the NN distance is a column-to-column spacing that is *not* a lattice
translation and is smaller than the lattice constant by a geometry-
dependent factor. Keep them as distinct quantities — never label an NN
distance as the lattice constant. To report a lattice constant, take it
from the periodicity (FFT fundamental / same-species repeat) or convert
the NN by the factor implied by the resolved geometry.

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
(`fft_defect_map` reports this directly as `defects_per_1000_sites`.)

**Superlattice / satellite reflections:** A reflection at a rational
multiple/fraction of the fundamental (≈2×, ½, ⅓ …) indicates a
**superstructure** — ordering (oxygen/cation-vacancy, charge/orbital),
an antiphase modulation, or a distinct second phase. Where its amplitude
map is localized tells you *where* the ordered domains are (e.g. confined
to a surface/interface band vs. bulk-wide). State the structural
mechanism (e.g. "≈2× fundamental ⇒ vacancy-ordered superstructure") as
**consistent with** the observation — the measurement proves a localized
reflection at d = N×fundamental exists, not the specific chemistry. The
phase channel of the same reflection is a displacement/strain field (GPA).

## validation

### foundational
**Detection completeness / over-detection:** judge from
`detection_quality_panels` — its targeted zoom-in overlays and metrics, not
a count ratio. Over-detection = high `short_pair_fraction` (a spike of
anomalously short NN distances from split/duplicate marks) and/or
detections off the DCNN peaks (low `heatmap_hit_fraction`); under-detection
= high `coverage_gap_fraction` and clearly-visible columns unmarked in the
gap zoom. A structure-based expected count is only an order-of-magnitude
sanity check (~0.5-2×): resolved columns-per-cell is unreliable to predict
for multi-sublattice / complex structures, so a moderate detected/expected
ratio is NOT over-detection.

**NN distance consistency:** CV below ~15% for well-ordered crystals — and a
clean unimodal NN distribution at the expected column spacing means the
detection is correct *regardless of the absolute count*.

**Unit cell sanity:** Measured lattice parameters should be in the
right ballpark of known bulk values, but absolute scale carries a
few-percent calibration uncertainty. Treat 3-5% deviations from
literature as informational (likely calibration), not a pass/fail
failure. Hard checks should be self-consistency: ratio of measured
spacings (b/a) matching the expected ratio, or FFT peaks forming a
consistent reciprocal lattice.

**Reported quantity matches the request, and ballparks compare like with
like:** confirm the headline number is the quantity the objective named,
and that any expected-value check compares the same quantity — a lattice-
constant expectation must not be applied to a nearest-neighbor measurement,
or vice versa (they differ by a projection-dependent factor). A value that
misses a ballpark *only* because the wrong quantity was compared is a
labeling error to fix in reporting, not a bad measurement to reject.

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

**Point-defect mapping** (when the step targets vacancies / point
defects): if both routes ran, every `fft_defect_map` deficit candidate
should correspond to a real-space missing/weak column and vice versa;
disagreement at a site usually means contamination or a surface step
(FFT-only) or an atom-finding miss (real-space-only). When only the FFT
route ran, validate a sample of candidates with real-space crops before
reporting counts. A defect count must respect the tool's own flags: do
not report site statistics from a run whose warnings say the candidates
form one connected cluster (extended feature) or that the anomaly
fraction is dense-disorder territory.

**Superlattice mapping** (when the step targets a satellite reflection):
the reflection must clear the significance floor — a detrended radial-PSD
peak well above the residual baseline AND a local-FFT spot SNR in the
candidate feature clearly above the bulk reference. The amplitude map
must not simply trace image edges/the interface line (edge artifact);
confirm against the phase-randomized null.

**Check the conclusion against `fourier_reflection_map`'s OWN output, and
allow calibrated uncertainty.** A satellite's *origin* (genuine ordering
vs. interface/edge artifact) often cannot be settled from a single frame,
so an honest **"candidate of ambiguous origin"** report is a CORRECT
outcome — not a hedge to mark down. Score honest, calibrated reporting at
the top; penalize over-reach in EITHER direction, judged against the
returned `reflections` / `strongest_satellite_d_nm` / `spot_snr_*` /
amplitude-shape values rather than how well-argued the prose is:
- **False null** — a "no resolvable superstructure" claim is valid ONLY IF
  `strongest_satellite_d_nm is None`. If the tool returned a satellite but
  the claim is absence, that **fails** (do not accept a null that
  explained-away a flagged satellite, nor one reached by hunting "the N=2
  of a hand-picked fundamental").
- **Over-confident positive** — a *confirmed* "ordered superstructure"
  claim needs `spot_snr_domain ≫ spot_snr_bulk` AND a **compact, non-edge**
  localization. A satellite whose amplitude traces a thin interface line
  must NOT be reported as a confirmed superstructure.
- **Over-confident dismissal** — equally, a significant satellite must NOT
  be thrown out as "just an edge artifact / nothing here." When its origin
  is uncertain (interface-tracing amplitude, low phase coherence), the
  required output is the **ambiguous report**: the satellite is present +
  where it concentrates + both hypotheses (interface-nucleated ordering vs.
  edge artifact) + what further data would disambiguate. Accept that as a
  high-quality result.
