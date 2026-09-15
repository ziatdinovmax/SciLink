---
description: "EBSD IPF-colour-MAP analysis, and the EBSD-specific extras for grain maps: per-grain crystallographic TEXTURE (cubic IPF poles) and the straight-grain-boundary (annealing-twin / Sigma3) fraction. PRIMARY use is an RGB IPF orientation map. For PLAIN grain segmentation + size distribution on a GRAYSCALE / optical / channeling micrograph, prefer overlapping_objects (it segments space-filling grains well); only use this skill on grayscale when the objective specifically asks for twin-boundary character or texture that overlapping_objects cannot express. NOT for discrete particles/precipitates/indents/loops (overlapping_objects), single-feature geometry (region_morphometry), or atomic-lattice images (atomic_stem)."
technique: "EBSD (IPF colour maps); grain-boundary-character / texture extras for grain maps"
---
# Grain / EBSD Microstructure Skill

## overview

Use for **polycrystalline grain analysis**: count and size grains, quantify the
annealing-twin (straight Sigma3 boundary) fraction, count twin segments per
grain, and — for IPF colour maps — report the crystallographic texture. Two
input types:
- **IPF colour map** (RGB; grains separated by orientation colour) → `mode="ipf"`;
- **grayscale channeling / BSE** image (grains by intensity contrast) →
  `mode="gray"`. (`mode="auto"` chooses by channel variance.)

> **PREFERRED TOOL.** Use `grain_analysis` (imported below) for the
> segmentation. `res["label_map"]` is the FINAL grain map — it is hole-free and
> already assigns intra-grain etch/channeling **speckle and texture to their
> grain**. Do NOT re-segment it, do NOT treat bright/dark intra-grain speckle as
> holes or separate objects, and do NOT run your own boundary/Hough/twin
> detection — `twin_boundary_fraction` / `straight_segments` from the tool are
> already correct (0 / empty when twins are not resolvable). You ARE free to
> compute **derived metrics on the tool's outputs** (e.g. ASTM grain-size number
> G from `res["grain_diameters"]`/areas) and to tune `boundary_sensitivity`.
> Reach for a custom pipeline only if the tool genuinely fails at runtime.

## planning

### foundational — calibration
Read the pixel size from metadata (authoritative) and pass it as `pixel_size`
with `pixel_unit` (EBSD/SEM grain maps are usually µm/px). Note: a `step_size` /
`pixel_size` in metadata may refer to the ORIGINAL un-downscaled raster — if the
supplied image is smaller, compute pixel size from FOV / image-width so it
matches the pixels you actually have.

Use the shared resolver rather than hand-rolling this:
`from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm`;
`px = resolve_pixel_size_nm(metadata, image.shape)` returns `{"x","y","source"}`
in nm/px (convert to µm if reporting µm), or `None`. It divides `field_of_view`
by the image **shape** — never divide by a metadata pixel-count field like
`n_cols`/`width` (usually absent → silently `None`). If it returns `None`,
proceed in pixels and report sizes as uncalibrated rather than halting.

### foundational — pick the mode and deliverables
The objective usually asks for several of: grain size distribution, twin
(straight-boundary) fraction, twin count per grain, texture, EBSD-vs-channeling
detectability. `grain_analysis` returns all of them in one call — report each.

### foundational — tune `boundary_sensitivity` from the overlay (with a physical anchor)
Grain segmentation is image-dependent, **especially grayscale channeling
contrast** (faint, sub-grain-textured boundaries). Run once at the default
(`boundary_sensitivity=0.5`), overlay the grain boundaries on the image, and
adjust: raise it if obvious grains are merged, lower it if single grains are
split into sub-grains. IPF colour maps are far more reliable than channeling.

**Do NOT over-tune (the common failure).** Raising `boundary_sensitivity` until
intra-grain channeling texture fragments grains into sub-grains is wrong. Anchor
the choice physically:
- **Channeling/BSE UNDER-detects boundaries vs EBSD** — many true boundaries are
  weak/invisible in channeling, and coherent twins are merged. So a channeling
  grain COUNT should be an **underestimate** (LOWER), and median grain size
  **larger or equal**, relative to a co-registered EBSD/IPF map of the same
  region. A channeling count that EXCEEDS the EBSD count, or a median grain size
  far BELOW it, means you over-tuned into sub-grain fragments — **lower**
  `boundary_sensitivity`.
- If the objective references a co-registered EBSD map, use ITS grain count /
  median size as an upper-bound anchor; otherwise sanity-check the median grain
  size against physical expectation (annealed grains are typically many µm, not
  a flood of ~1-resolution fragments).
- When unsure, prefer **slight under-segmentation** to fragmentation, and report
  the count as a lower bound with the detectability caveat — that is the correct,
  honest answer for channeling contrast.

## analysis

```python
import numpy as np
from scilink.skills._shared.grain_analysis import grain_analysis, POLES

img = np.load("data.npy")          # RGB IPF map (H,W,3) or grayscale (H,W)
res = grain_analysis(
    img,
    pixel_size=0.4, pixel_unit="um",   # from metadata
    mode="auto",                        # "ipf" / "gray" / "auto"
    boundary_sensitivity=0.5,           # tune from the overlay (see above)
    contamination="auto",               # mask dark/bright debris (gray mode)
)

n          = res["n_grains"]
diam_um    = res["grain_diameters"]               # per-grain equiv diameter
twin_frac  = res["twin_boundary_fraction"]        # ALREADY 0 when the proxy is
                                                  # unreliable (all-straight network)
straight_frac = res["straight_boundary_fraction"] # raw straight-boundary fraction
twin_reliable = res["twin_proxy_reliable"]        # False -> no curved baseline,
                                                  # report twins ~0 (see interp.)
twin_per   = res["mean_twin_segments_per_grain"]  # also 0 when unreliable
if res["mode"] == "ipf":
    texture = res["texture"]                      # per-pole count/area fractions
    dominant = res["dominant_texture"]
```

Overlay the grain boundaries (`res["label_map"]` contours) and the detected
straight segments (`res["straight_segments"]`) on the image, and plot the
`grain_diameters` histogram. Report: n grains, equivalent-diameter stats,
twin-boundary fraction + twin segments/grain, and (IPF) the texture pole
fractions + dominant component.

**Use the tool's twin outputs directly — do NOT run your own Hough/Canny twin
detection.** `twin_boundary_fraction`, `mean_twin_segments_per_grain` and
`straight_segments` are already set to 0 / empty when `twin_proxy_reliable` is
False, so following them gives the right answer (twins ≈ 0, no straight lines
drawn) without any extra code. If you want the raw value, report
`straight_boundary_fraction` separately and clearly labelled as such.

The reliability gate is the SIZE of the curved-boundary baseline:
`twin_proxy_reliable` is True only when `curved_boundary_fraction` (= 1 −
straight fraction) is at least `min_curved_fraction` (default 0.30). A
predominantly straight network — polygonal/faceted/columnar grains, OR annealed
grains with naturally straight boundaries — has little curved baseline and is
gated off. (A straight fraction of ~0.83, e.g. a typical EBSD IPF map, is NOT a
0.83 twin fraction: curved fraction 0.17 < 0.30 → proxy unreliable → reported as
≈0.) Tune `min_curved_fraction` only if you can SEE the curved baseline and
disagree with the gate.

**Overlay coordinate order:** `straight_segments` are `((x0, y0), (x1, y1))`
pairs in (column, row) order. Plot them as `(x, y)` (e.g.
`plt.plot([x0, x1], [y0, y1])`), NOT as `(row, col)`; transposing them mislocates
segments and can place them outside a non-square frame.

## interpretation

- **Twin-boundary fraction is a STRAIGHT-boundary fraction, and only a valid twin
  proxy when the *regular* grain boundaries are CURVED** (the usual recrystallized
  case: curved general boundaries + atomically-flat straight Σ3 twins → the
  straight fraction ≈ twin fraction). **It false-positives whenever the grain
  boundaries are themselves straight** — idealized / synthetic polygonal
  tessellations, columnar or strongly faceted grains — where it can read ~0.9
  with NO twins present. So, before reporting it as a twin fraction, CHECK the
  microstructure:
  - If the boundary network is mostly straight polygon edges, OR there are no
    visible straight *intragranular* lamellae / paired parallel twin lines,
    **report twin fraction ≈ 0 with the explicit caveat** "no twins resolvable
    by straight-boundary morphology here" — do NOT report the raw straight
    fraction as a twin fraction.
  - Only report a high twin fraction when you can SEE straight twins against a
    curved-boundary background. It is a morphological proxy, never a
    misorientation measurement.
- **IPF texture**: `<001>`=red, `<101>`=green, `<111>`=blue (Z direction). A
  dominant pole with high area fraction indicates a preferred orientation /
  texture component; near-equal fractions indicate a random texture.
- **EBSD (IPF) vs channeling**: IPF colour segmentation is reliable. Grayscale
  channeling/BSE segmentation is **inherently less reliable** — many true grain
  boundaries are weak or invisible in channeling contrast, and coherent twins
  (which share orientation colour in IPF too) are easily merged into the parent
  grain. State this when comparing; the channeling grain count is typically an
  underestimate and the twin fraction less certain.

## validation

### foundational
- **Grain count vs visual**: overlay boundaries; the count should match a visual
  estimate within ~±25 %. Far too few ⇒ raise `boundary_sensitivity`; a field of
  tiny fragments ⇒ lower it (or stronger pre-smoothing).
- **Size distribution**: physically plausible, right-skewed for annealed
  microstructures; a spike of sub-resolution fragments ⇒ over-segmentation.
- **Twin fraction sanity**: the straight-segment detector reads ~1 for straight
  boundaries and ~0 for curved ones BY DESIGN — so a high value only means
  "boundaries are straight," NOT "twins exist." A high straight fraction in a
  polygonal/idealized microstructure, or with no visible intragranular twin
  lines, is a FALSE POSITIVE → report ~0 with the caveat (see interpretation).
  Cross-check against what you actually see in the image before reporting twins.
- **Derived metrics** (e.g. ASTM G): compute from the tool's per-grain
  diameters/areas; for ASTM E112, convert mean grain area to **mm²** first
  (G = -6.6438·log2(mean_area_mm²) - 3.293); sanity-bound G to ~0-14.
- **Texture**: pole fractions sum to ~1; a claimed dominant component should be
  visible as a colour cluster in the IPF map.
