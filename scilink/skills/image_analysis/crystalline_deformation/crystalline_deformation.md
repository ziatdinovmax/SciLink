---
description: Atomic-resolution STEM deformation and defect analysis — per-atom 2D-PTM structural classification, Center-of-Symmetry distortion mapping, twin boundary / stacking fault / grain boundary identification, GPA validity gating, and Burgers vector extraction on zone-axis crystalline materials.
---
# Crystalline Deformation Mechanisms — Atomic-Resolution Skill

## overview

Identification and characterization of deformation mechanisms in
crystalline materials from **atomic-resolution STEM images where
individual atomic columns are resolved**. Do NOT apply to SEM, EBSD,
or diffraction-contrast TEM. Applies to **cubic systems** (FCC, BCC)
and **hexagonal** (HCP) structures viewed along low-index zone axes.
Covers twin boundaries, stacking faults, dislocations, grain
boundaries, and strain mapping at atomic scale using per-atom
structural classification and local symmetry analysis.


**Relationship to the general atomic-resolution STEM skill:** this skill
may be selected alongside the general atomic-resolution skill on the same
image. For objectives concerning defects, boundaries, strain, or
deformation, the detection sequence in THIS skill governs: detector
choice, the PTM unidentified-fraction gate, and the fallback order. The
general skill's guidance on sublattice separation, lattice metrology, and
calibration-aware quality criteria remains complementary and applicable.
Where the two skills differ on detector selection for a defect-analysis
objective, follow this skill.

## planning

### foundational
The key insight for atomic-resolution defect analysis is that
**structural classification of individual atom columns directly reveals
defects** without needing orientation mapping, FFT filtering, or domain
segmentation as intermediate steps.

**2D Polyhedral Template Matching (2D-PTM)** classifies each atom's
local neighbor environment against ideal crystal structure templates
by Kabsch-aligned RMSD matching. For each atom: find 6 nearest
neighbors via Delaunay triangulation, sort angularly, align against
rotated ideal templates (FCC, BCC, HCP projections), assign the
best-matching structure if RMSD is below threshold. This gives a
per-atom label: FCC, BCC, HCP, or unidentified.

**Center of Symmetry (CoS)** quantifies how much each atom's local
environment deviates from centrosymmetry. For each atom with N bond
vectors to neighbors, find the best anti-parallel partner for each
bond: D[j] = min_i(||bond_i + bond_j||). The metric:
M = sum(D²/2) / (2 × sum(||bond||²)). M=0 is perfectly
centrosymmetric (bulk); M>0 indicates broken symmetry (defect,
surface, boundary).

Always run both PTM and CoS — they are complementary. PTM gives
discrete classification; CoS gives a continuous distortion measure.

### advanced
**Deformation mechanisms in FCC — the Shockley partial progression:**

In FCC metals viewed along [110], deformation proceeds by passage of
Shockley partial dislocations (b = a/6<112>) on {111} planes. Each
partial shifts the stacking sequence by one position. The accumulated
effect is directly visible in 2D-PTM classification:

1. **One Shockley partial → Intrinsic Stacking Fault (ISF):**
   ABCABC becomes ABC|BCA. The fault plane comprises **2 adjacent
   atomic layers that classify as HCP** amid the FCC matrix. This is
   because locally the stacking is ...ABA... which is the HCP motif.

2. **Second Shockley partial on adjacent plane → Extrinsic Stacking
   Fault (ESF):** Creates **2 HCP layers sandwiching 1 FCC layer**
   (HCP-FCC-HCP). The enclosed FCC layer has the "wrong" registry
   relative to the surrounding matrix.

3. **Third Shockley partial → Twin Boundary:** Creates a coherent
   Σ3{111} twin. In PTM output: **1 HCP layer** at the boundary
   plane, with the FCC matrix on either side showing a mirror
   relationship. The twin misorientation is 70.53° about <110> (in
   3D). In [110] projection, the two domains show mirrored lattice
   vector angles.

**Counting HCP layers in PTM output directly identifies the defect:**
- 2 adjacent HCP layers → ISF
- 2 HCP layers separated by 1 FCC layer → ESF
- 1 HCP layer with mirror-related FCC on both sides → twin boundary
- Multiple isolated HCP layers → multiple stacking faults or
  deformation band

**Classification priority — the layer count is the primary classifier.**
The HCP layer-count signature is topological: it requires no plane
identification, no per-domain lattice-vector fitting, and no angular
measurement, and it is reliable whenever the classification is clean
(low unidentified fraction, correct zone axis). Angular corroborations
(mirror relationship, misorientation, {111} trace alignment) depend on
correctly identifying a specific plane and fitting lattice vectors per
domain — measurements that are fragile in exactly the mirror-symmetric
configurations they are meant to confirm. Therefore: angular checks may
STRENGTHEN a layer-count classification but must never overturn it. If
an angular check returns a physically impossible value (exactly 0° or
exactly 90° misorientation), the measurement itself has failed —
re-measure with a different method, or report the classification from
the layer count alone and note the corroboration as unavailable. Do
NOT downgrade a clean single-HCP-layer twin signature to "unresolved"
or "artifact" because an angular measurement failed to confirm it.

**Grain boundaries from PTM:**
- **Low-angle grain boundaries (LAGB):** Array of dislocation cores.
  Each core shows a cluster of unidentified atoms (no template match)
  with surrounding distorted FCC. The dislocation spacing gives the
  misorientation via Frank's formula: θ = b/d.
- **High-angle grain boundaries (HAGB):** Continuous band of
  unidentified atoms — the boundary core is too disordered for any
  template to match. Width is typically 2-5 atomic layers.
- **CSL boundaries (Σ3, Σ5, Σ7, etc.):** Σ3 twins as described above.
  Higher-Σ boundaries show periodic structural units with
  characteristic repeat distances.

**Unidentified atoms in PTM are diagnostic:**
- Isolated unidentified atoms in bulk → point defects, beam damage,
  or noise in atom positions
- Linear arrangement → dislocation line or planar defect edge-on
- Continuous band → grain boundary core
- Clustered region → amorphous pocket, radiation damage, or
  precipitate interface

**CoS map interpretation:**
- Bulk crystal: M ≈ 0 (uniform blue in viridis colormap)
- Stacking fault: slight elevation, M ≈ 0.001-0.005
- Twin boundary: moderate elevation at boundary plane, M ≈ 0.005-0.02
  for disordered or incoherent segments; a COHERENT twin boundary in
  2D projection can show negligible elevation (M < 0.001, near bulk
  noise) because the projected 6-NN environment of the boundary layer
  remains nearly mirror-symmetric — absent CoS elevation does NOT
  invalidate a coherent single-HCP-layer twin
- Dislocation core: strong peak, M ≈ 0.02-0.10
- Grain boundary: elevated band, M ≈ 0.01-0.05
- Surface/edge: elevated (fewer neighbors), exclude from analysis
- Threshold M > 0.05 typically indicates fitting artifacts, not real
  distortion — clamp or exclude these values

**Zone axis identification:**
Different zone axes produce different projected atom patterns. For FCC:
- [110]: dumbbells visible (two sublattices), hexagonal projected
  pattern, most common for defect analysis
- [100]: square pattern, single sublattice visible
- [111]: hexagonal pattern, three sublattices overlapping
- [112]: rectangular pattern, complex sublattice structure

The zone axis determines which defect features are visible. {111}
stacking faults and twins are edge-on in [110] and [112], but
inclined in [100] and [111].

## analysis

### foundational
**Required pipeline for defect characterization:**

1. Detect atom column positions (DCNN preferred with GPU, classical
   peak detection as fallback)
2. Run 2D-PTM classification on all detected positions
3. Compute CoS metric for all detected positions
4. Generate classification map (FCC=green, HCP=red, BCC=blue,
   unidentified=black/gray)
5. Generate CoS heatmap via griddata interpolation
6. Identify defect types from PTM pattern (see planning section)
7. Measure defect geometry: trace angle, spacing, width

**Detection quality gate — PTM unidentified fraction:**
After step 2, compute the fraction of atoms classified as "unidentified"
by PTM. If >15% of atoms are unidentified AND classical peak detection
(`detect_atoms`) was used in step 1, this indicates the detector missed
or mislocated atom columns (poor peak finding in noisy/low-contrast
regions). In this case:
- Retry atom detection using DCNN (`detect_atoms_dcnn`). If `fov_nm`
  is not provided in metadata, compute it as:
  `fov_nm = estimated_pixel_size_nm * image_width_px` (estimate pixel
  size from FFT lattice spacing and known bulk lattice parameter).
- Re-run PTM classification on the new DCNN positions
- Do NOT fall back to adjusting classical detection parameters — if
  classical detection failed to produce <15% unidentified, it is
  unlikely to succeed with parameter tweaks alone. DCNN is the
  required retry path.
- Do NOT proceed to defect interpretation with >15% unidentified atoms
  unless the image genuinely contains that level of disorder (e.g.,
  heavily irradiated material or amorphous regions)

### advanced
**Per-atom analysis workflow:**

Given N atom positions as (x, y) arrays from Tier 1 detection:

Step 1 — Build neighbor lists: For each atom, find 6 nearest
neighbors. Use Delaunay triangulation or KDTree with distance cutoff
at ~1.5× expected NN distance. Exclude atoms with fewer than 4
neighbors (edge atoms).

Step 2 — PTM classification: For each atom, extract the 6 NN
positions relative to the central atom, sort angularly. Compare
against pre-computed rotated templates for each crystal structure.
Use Kabsch algorithm for optimal alignment, compute RMSD. Assign
structure with lowest RMSD if below threshold (typically 0.1-0.15
in normalized units). Record: structure label, RMSD, rotation angle,
scaling factor.

Step 3 — CoS calculation: For each atom's bond vectors to its
neighbors, compute the anti-parallel partner distances D[j] and the
normalized metric M. This requires only the neighbor list from Step 1.

Step 4 — Boundary extraction from PTM map: Identify connected
regions of same-type atoms. Boundaries are where the classification
changes. For twin boundaries specifically: find contiguous lines of
HCP-classified atoms separating FCC regions. The boundary trace is
the best-fit line through the HCP atom positions.

Step 5 — Misorientation from lattice vectors: Within each
PTM-identified domain (connected FCC region), fit lattice vectors
from the NN displacement vector histogram. Compare lattice vector
angles between adjacent domains. For Σ3 twin in [110]: expect
mirrored a2 vectors (equal magnitude, opposite sign angles relative
to the boundary normal).

Robustness warning — this step is SUPPORTING evidence only,
subordinate to the layer count (see planning, classification
priority). Comparing a single "first" or dominant lattice vector per
side is degenerate for mirror twins: the mirror maps the NN vector
family onto itself, so a canonical-sort pick returns near-identical
angles (misorientation ≈ 0°) even for a genuine twin. If attempted,
compare the FULL set of NN vector angles per domain reflected about
the boundary normal, or compare the {111} plane-trace angles on each
side (for an edge-on Σ3 twin in [110], symmetric at about ±35° to
the boundary). A 0° result means the measurement is degenerate —
discard it, not the twin.

Step 6 — Technique decision after classification: After PTM
classifies all atoms and domains are identified, decide whether
downstream techniques are physically valid for this image:

- **Multiple grain orientations detected** (>1 distinct FCC domain
  with different lattice vector angles): Do NOT apply GPA. GPA
  computes strain relative to a single reference lattice — when
  multiple orientations are present, the phase field is discontinuous
  at every grain boundary, producing physically meaningless strain
  values (often hundreds of percent) at and near boundaries. The
  correct approach is per-atom PTM + CoS, which handles multiple
  orientations natively because each atom is classified independently
  against templates.
- **Multiple phases detected** (PTM finds domains of different
  structure types, e.g., FCC + BCC, or FCC + HCP beyond a stacking
  fault): Do NOT apply GPA. GPA tracks displacement of a periodic
  lattice by locking onto specific Bragg reflections (g-vectors).
  When the lattice structure changes, the periodicity changes — the
  reference g-vectors no longer exist in the second phase, so the
  phase field becomes undefined and the computed "strain" reflects
  the lattice mismatch between phases, not real elastic strain.
  Example: an FCC/BCC interface has different d-spacings and
  symmetry; GPA across the interface produces strain artifacts of
  10–50% that are crystallographic misfit, not deformation. Use PTM
  to classify each phase independently, then measure interfacial
  relationships (orientation, habit plane) from lattice vectors.
- **Single grain, no boundary**: GPA is valid and complementary to
  PTM — it maps continuous strain at every pixel, not just at atom
  positions. Use GPA for dislocation strain fields, compositional
  strain, or subtle elastic distortion below PTM sensitivity.
- **Image contains a twin boundary or grain boundary (bicrystal)**:
  Do NOT apply GPA — even "within individual domains." When the
  boundary is the region of scientific interest, GPA cannot
  characterize it because the phase field is undefined at the boundary
  and unreliable within ~20–50 px of it. Even within one domain of a
  bicrystal, the field of view is typically too small for meaningful
  GPA statistics, and boundary-proximity artifacts bleed inward. The
  correct approach is:
  1. Detect atomic columns (DCNN + refinement, or classical peak
     detection as fallback)
  2. Run PTM to classify the boundary structure (HCP layer count,
     misorientation)
  3. Compute Center of Symmetry (CoS) maps as a **proxy for strain**
     — CoS measures local lattice distortion without requiring a
     single reference lattice, works natively across boundaries, and
     directly highlights deformation concentration at the boundary
  4. If the user or objective explicitly asks for "strain maps" or
     "GPA," do NOT apply GPA anyway. Report CoS maps as "local
     distortion maps" and explain why GPA is not applicable to
     bicrystals/twin boundaries. The objective may be written by
     someone unfamiliar with GPA's limitations — always prioritize
     physical correctness over literal compliance with the request.
- **Single grain with stacking faults (no boundary traversing the
  full image)**: GPA is valid only within the continuous domain. The
  fault itself produces a phase discontinuity — mask fault regions
  (±5 px from PTM boundary atoms) before interpreting GPA strain
  values. This case applies only when the stacking fault is a minor
  feature within a large single-crystal field of view, NOT when the
  image is fundamentally a bicrystal.

If the user or objective requests a technique that is not physically
valid for the image (e.g., GPA on a multi-grain image), do NOT
silently apply it. Instead, explain why the technique is unsuitable
and what the correct alternative is. For example: "GPA cannot be
applied to this image because it contains two grain orientations
with different lattice vectors. GPA assumes a single reference
lattice and would produce artifact strain values at the grain
boundary. Per-atom PTM classification already identifies the
boundary structure and defect type without this limitation."

**Physical constraints for validation:**
- Σ3 twin misorientation: 70.53° about <110> (3D), projected angle
  depends on zone axis
- ISF: exactly 2 HCP layers, not 1, not 3
- ESF: exactly HCP-FCC-HCP sandwich (3 layers)
- Twin: exactly 1 HCP layer at coherent boundary
- NN distance should be consistent across the image (CV < 10%)
- Lattice vectors in both domains should have same magnitude (twins
  preserve lattice parameters)
- CoS should be elevated along the same trace as PTM boundary atoms

**What NOT to do (with physics reasons):**
- Do not apply GPA to images with multiple grain orientations or
  multiple phases. GPA measures displacement relative to one reference
  lattice; a second orientation produces a linearly-growing phase ramp
  that wraps repeatedly, creating artificial strain of 10–1000%. A
  different crystal structure has different g-vectors entirely, making
  the GPA phase field undefined in the second phase. PTM handles both
  cases because it classifies each atom independently against all
  templates.
- Do not use GPA as the primary defect identification tool when
  atomic columns are resolved. PTM directly classifies defect
  structure (twin vs SF vs GB) from atom positions; GPA only shows
  that "something is strained" without identifying what. Use GPA as
  a complement for continuous strain mapping within a single domain,
  not as a substitute for structural classification.
- Do not use orientation clustering (GMM, k-means) as the primary
  domain segmentation method — it is fragile against scan distortion,
  intensity gradients, and sample tilt. Use PTM classification
  instead.
- Do not use FFT Bragg filtering for twin identification — it is
  complex to implement correctly and unnecessary when PTM gives the
  answer directly from atom positions.
- Do not report misorientation angles that are physically impossible:
  -90° is never a crystallographic misorientation, 70.53° ± 5° is
  expected for Σ3, 36.87° for Σ5, 38.21° for Σ7.
- Do not classify a boundary as "low_angle" if misorientation > 15°.
  Low-angle: < 15°. High-angle: > 15°. CSL: specific angles.

## interpretation

### foundational
**Reading a PTM classification map:**

A well-classified image of defect-free FCC crystal is uniformly
green (all atoms match FCC template). Deviations indicate structural
features:

- Red line (HCP atoms) across green (FCC) background → stacking
  fault or twin boundary. Count the red layers to distinguish ISF
  (2 layers) from twin (1 layer).
- Black/gray spots (unidentified) in a line → dislocation array or
  grain boundary.
- Large black region → amorphous, heavily damaged, or wrong zone axis
  (templates don't match the projection).
- Uniform red (all HCP) → the material is HCP, or viewing along a
  zone axis where FCC projects as HCP-like pattern. Check zone axis.

### advanced
**Quantitative defect characterization from PTM + CoS:**

- **Stacking fault energy (qualitative):** Materials with many ISFs
  and ESFs have low stacking fault energy (e.g., austenitic stainless
  steel, CoCrNi). Materials with only twins and no isolated ISFs have
  moderate SFE. Materials with no planar defects likely have high SFE.

- **Deformation history:** ISFs form first (one partial), then ESFs
  (two partials), then twins (three partials). Observing only twins
  with no ISFs suggests significant accumulated strain. Observing
  ISFs without twins suggests early-stage deformation.

- **Boundary character from CoS profile:** The width of the CoS
  elevation across a boundary indicates its core width. Coherent
  twins: sharp (1-2 atom layers). Incoherent segments: broader
  (3-5 layers). HAGB: broadest (5+ layers). Use the full-width at
  half-maximum of the CoS peak perpendicular to the boundary trace.

- **Rigid body translation at boundary:** Compare the lattice
  registries on both sides. For a perfect Σ3 twin, there should be
  zero rigid body translation parallel to the boundary. Non-zero
  translation indicates a displacement shift complete (DSC) vector,
  which means the boundary contains disconnections.

## validation

### foundational
**PTM quality checks:**
- Fraction of unidentified atoms in bulk (away from boundaries)
  should be < 5%. Higher values indicate the RMSD threshold is too
  tight, poor atom positions, wrong templates, or wrong zone axis.
  If >5%, use `ptm_classify_sweep()` to auto-select a threshold.
- RMSD distribution: bulk atoms should cluster at low RMSD (< 0.1).
  A bimodal distribution with a second peak at 0.1-0.15 suggests
  a second structure type (e.g., HCP at stacking faults).
- All atoms in a single domain should have the same PTM label. Mixed
  FCC/HCP in bulk (not at a boundary) indicates noisy positions.

**CoS quality checks:**
- Bulk CoS should be near zero and spatially uniform. Large-scale
  gradients in bulk CoS indicate systematic position errors (scan
  distortion, drift) rather than real structural features.
- CoS > 0.05 is almost always artifact — clamp to zero or exclude.
- Edge atoms (< 6 neighbors) will have artificially high CoS —
  apply a border cutout before analysis (trim 2-3 NN distances
  from all edges).

### advanced
**Cross-validation between PTM and CoS:**
- A disordered PTM boundary (unidentified-atom band, dislocation
  array) should show elevated CoS; if it does not, that boundary is
  likely a classification artifact. This test does NOT apply to a
  coherent single-HCP-layer twin line, which can legitimately show
  negligible CoS elevation in 2D projection (see CoS map
  interpretation) — never reject a coherent twin on low CoS alone.
- CoS peaks without corresponding PTM transitions indicate subtle
  distortion (elastic strain, composition gradient) that doesn't
  change the structure type.
- The boundary trace from PTM atom positions and the ridge line of
  CoS elevation should coincide within 1 NN distance.
- For twins: CoS peak height at the boundary should be consistent
  along its length. Variations indicate boundary steps or
  disconnections.

**Physically impossible results that indicate analysis errors** (these
mean the MEASUREMENT failed — fix or omit that measurement; they are
never evidence that the defect is absent, and they never override the
layer-count classification):
- Misorientation of exactly 0° or exactly 90° between domains
- Twin boundary with 0 or 3+ HCP layers (should be exactly 1)
- ISF with 1 or 3+ HCP layers (should be exactly 2)
- Lattice parameter ratio between domains ≠ 1.0 for a twin
  (twins preserve the lattice)
- Negative CoS values (mathematically impossible with the standard
  metric)
- Boundary trace angle that doesn't correspond to any {111} trace
  for the identified zone axis

## implementation

### foundational
**Tool priority for defect analysis:**

1. Always start with atom detection — see detection strategy below
2. Run PTM classification immediately after detection — it requires
   only (x, y) positions and pre-computed templates
3. Run CoS calculation in parallel with PTM — same input, independent
   computation
4. Use PTM map as the primary defect identification, CoS as
   confirmation and continuous distortion measure
5. Extract boundary geometry from PTM-classified atom positions, not
   from orientation maps or FFT

**Atom detection strategy — critical for downstream PTM/CoS:**

Detection completeness is the single most important upstream step.
Missing atoms corrupt Delaunay neighbor-finding, causing PTM to
classify most atoms as "unidentified" and CoS to show false patterns.
A 1540×1540 atomic-resolution STEM image typically contains 1500–2500
atom columns.

Preferred approach: `detect_atoms_dcnn` with GPU. However, DCNN can
fail when the image has strong intensity gradients (e.g., thickness
fringes, detector nonuniformity) — it will only detect atoms in
bright regions and miss the rest.

**Always validate detection count.** If DCNN returns fewer than ~1500
atoms for a full-field atomic-resolution image, fall back to classical
detection:
```python
from scilink.skills.image_analysis.atomic_stem.atom_finding import detect_atoms
result = detect_atoms(image, separation=<NN_from_FFT>,
                      threshold_rel=0.02, refine=True)
```
Classical `detect_atoms` with a low `threshold_rel` (0.01–0.03) is
more robust to intensity gradients than DCNN. Estimate `separation`
from the FFT lattice peak spacing.

If the image has a visible intensity gradient, apply background
subtraction (large-sigma Gaussian blur, subtract) before detection.
Do NOT apply heavy filtering or CLAHE before DCNN — the model expects
raw or near-raw input.

### advanced
**PTM and CoS tools are available.** Call them directly — do NOT
reimplement template matching or symmetry calculations from scratch.

```python
from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import ptm_classify, ptm_classify_sweep, compute_cos

# After atom detection gives x, y arrays (pixel coordinates):
result = ptm_classify(x, y, threshold=0.05, structures=["FCC", "HCP"],
                      edge_cutout_nn=3)

labels = result["labels"]       # per-atom: "FCC", "HCP", "unidentified", "edge"
rmsd   = result["rmsd"]         # per-atom RMSD (NaN for unidentified/edge)
cos    = result["cos"]          # per-atom Center of Symmetry metric
codes  = result["structure_code"]  # 1=FCC, 2=BCC, 3=HCP, 0=other, -1=edge
```

**RMSD threshold sweep — use when default threshold gives too many
unidentified atoms (>5% of interior atoms).** Runs Kabsch alignment
once, then reclassifies at each candidate threshold in O(N) time:

```python
sweep = ptm_classify_sweep(x, y, structures=["FCC", "HCP"],
                           target_unidentified=0.05)
print(f"Best threshold: {sweep['best_threshold']}")
for s in sweep['sweep']:
    print(f"  t={s['threshold']:.3f}  unid={s['frac_unidentified']:.1%}  {s['per_structure']}")

# Use the auto-selected result
result = sweep['best_result']
labels = result['labels']
```

These tools encode pre-validated crystallographic templates (FCC
[110], BCC [111], HCP [2-1-10]) and correct Kabsch alignment,
neighbor-finding, and CoS normalization. Reimplementing them wastes
API calls and introduces errors in template construction.

**GPA strain mapping and Burgers vector tools are available.** Use
them when the objective asks for strain fields, Burgers vectors, or
dislocation character (edge/screw/mixed).

```python
from scilink.skills._shared.strain import gpa_strain_map
from scilink.skills.image_analysis.crystalline_deformation.gpa_tools import burgers_from_gpa

# Step 1: GPA strain mapping — use the SHARED, validated tool.
gpa = gpa_strain_map(image)          # auto reflections + auto reference
e_xx = gpa['exx']                    # normal strain X
e_yy = gpa['eyy']                    # normal strain Y
e_xy = gpa['exy']                    # shear strain (epsilon_xy)
w_z  = gpa['wxy']                    # rigid rotation omega_z — use + sign as-is
valid = gpa['valid_mask']            # exclude invalid pixels from interpretation

# Step 2: Burgers vectors from strain maps
# Option A — per-dislocation (if you know core positions from PTM/CoS):
bv = burgers_from_gpa(e_xx, e_yy, e_xy=e_xy, w_z=w_z,
                      pixel_size=pixel_size_nm,
                      dislocation_positions=[(row, col), ...],
                      circuit_half_width=10)
for d in bv['burgers_vectors']:
    print(d['b_vector'], d['b_magnitude'], d['b_direction'])

# Option B — full-field (auto-detect dislocations):
bv = burgers_from_gpa(e_xx, e_yy, e_xy=e_xy, w_z=w_z,
                      pixel_size=pixel_size_nm,
                      compute_field=True, field_half_width=5,
                      field_cutoff=0.01)
cores = bv['cores']       # (N, 2) dislocation positions
b_field = bv['b_field']   # (Ny, Nx, 2) vector field
b_mag = bv['b_mag']       # (Ny, Nx) magnitude map
```

Sign convention: `w_z` is `gpa_strain_map`'s `wxy` passed with the +
sign (omega_z = 0.5*(duy/dx − dux/dy)). Do not negate it.

**Dislocation character from Burgers vector:**
Once you have the Burgers vector **b** and the dislocation line
direction **t** (from the PTM boundary trace or CoS ridge):
- Edge: b perpendicular to t (dot product ≈ 0)
- Screw: b parallel to t (cross product ≈ 0)
- Mixed: b has components both parallel and perpendicular to t
- Character angle: α = arccos(|b·t| / |b||t|). α=90° → pure edge,
  α=0° → pure screw.

**GPA workflow notes:**
- GPA needs an undistorted reference region. `gpa_strain_map`'s
  `reference_roi="auto"` finds one; for images with a defect at center,
  pass `reference_roi=(x, y, w, h)` in a defect-free area.
- Reflections are auto-selected from the FFT; if the wrong pair is
  picked, pass `reflections=[(gx1, gy1), (gx2, gy2)]` explicitly.
- Respect `valid_mask` / `valid_fraction` in the returned dict — cores,
  vacuum, and edges are marked invalid and must not be interpreted.
- The image does NOT need to be square — rectangular images work.
- GPA produces strain at every pixel, not just at atom positions.
  It complements PTM (which gives per-atom classification).

Do NOT reimplement GPA (FFT masking, phase unwrapping, strain
differentiation) or Burgers vector line integrals from scratch —
call these tools directly.
