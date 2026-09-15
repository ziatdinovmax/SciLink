---
description: "Geometry of ONE isolated feature with an apex/axis or an enclosed cavity — APT needle tip (apex, axis, apex-to-oxide distance, GB-plane angle), a single blister dome + delamination void cross-section, a needle/wedge taper, or an interface-plane angle. Measures principal axis, apex (pointed end), length-along-axis, perpendicular width profile, and the axial/lateral position and plane angle of a sub-feature. NOT for counting/sizing populations of many objects (use overlapping_objects), grain maps (use grain_ebsd), atomic-lattice analysis (use atomic_stem), or layered/multilayer structures."
technique: "STEM/TEM, SEM; single-feature geometry, APT-correlative needles, blister/void cross-sections"
---
# Region / Interface Morphometry Skill

## overview

Use when the objective asks for the **geometry** of a segmented feature: an
apex/tip and axis, a length and width (or width *profile*), an aspect ratio, the
distance from an apex to a sub-feature resolved into axial/lateral components, or
the angle of an interface/boundary plane relative to a reference axis. Examples:
APT-needle apex → grain-boundary-oxide distance + GB plane angle (18/19); blister
dome lateral diameter/height/aspect + delamination-void extent/thickness (08);
precipitate full extent; needle/wedge taper.

> **MANDATORY TOOL.** Use `region_geometry` / `feature_vs_axis` (imported below).
> Do NOT read geometry off a bounding box or `regionprops` major/minor axis —
> those misreport tip position, true length-along-axis, perpendicular width, and
> plane angles for tilted / tapered / non-elliptical shapes. YOUR job is to
> produce the binary mask(s); the tool does all the geometry.

## planning

### foundational — segment, then measure (two clear steps)
1. **Segment** the feature(s) into binary mask(s) with whatever is appropriate
   (threshold / fill / largest-component; for a sub-feature inside a bright body,
   remove the slow intensity/thickness gradient first and take the dark/bright
   anomaly). This is the only domain-specific part.
2. **Measure** with the tool — one call per region; never hand-compute geometry.

### foundational — calibration
Pass `pixel_size` + `pixel_unit` from metadata; all lengths come back in that
unit. Assumes square pixels. Resolve the value with the shared helper rather than
hand-rolling it:
`from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm`;
`px = resolve_pixel_size_nm(metadata, image.shape)` → `{"x","y","source"}` nm/px,
or `None`. It divides `field_of_view` by the image **shape**; never divide by a
metadata pixel-count field (`n_cols`/`width`) — that field is usually absent and
silently yields `None`. If `None`, report sizes in pixels (uncalibrated).

## analysis

```python
import numpy as np
from scilink.skills._shared.region_morphometry import region_geometry, feature_vs_axis, width_profile

# --- main region (e.g. APT needle, blister dome, precipitate) ---
geo = region_geometry(main_mask, pixel_size=0.207, pixel_unit="nm")
apex   = geo["apex"]            # (x,y) pointed/narrow end (e.g. needle tip)
axis   = geo["axis_vec"]        # unit vector apex -> body
length = geo["length"]          # extent along the principal axis
width  = geo["max_width"]       # max perpendicular width
# geo also has: mean_width, aspect_ratio, axis_angle_deg, base,
#               apex_is_defined (False if ~symmetric), width_profile

# --- a sub-feature measured against the main axis (e.g. oxide in the needle) ---
f = feature_vs_axis(sub_mask, axis, apex, pixel_size=0.207, pixel_unit="nm")
f["distance_along_axis"]        # apex -> sub-feature centroid, along the axis
f["perpendicular_offset"]       # lateral offset from the axis
f["axial_extent"], f["max_perpendicular_width"]
f["farthest_axial"], f["farthest_lateral"]   # apex -> farthest point, resolved
f["plane_angle_to_axis_deg"]    # sub-feature's own axis vs the main axis

# --- width along the axis (dome height vs base, void thickness, taper) ---
pos, wid = width_profile(main_mask, axis, geo["centroid"], pixel_size=0.207)
```

Overlay the apex (star), the axis (line from apex into the body), and the
sub-feature mask on the raw image; report the requested measurements with units.

## interpretation

- **apex** is the pointed/narrow end; `apex_is_defined=False` for near-symmetric
  shapes (a plain ellipse/disc has no meaningful apex — say so rather than
  picking an arbitrary end).
- For a **dome/cap** wider than tall: lateral diameter = the larger of
  length/`max_width`, height = the smaller; aspect = height/diameter.
- For a **void/slab**: lateral extent = `length`, thickness = `mean_width`
  (typical) and `max_width` (peak), or use `width_profile` for thickness vs
  position.
- **plane_angle_to_axis_deg** (0–90°) is the acute angle between a sub-feature's
  elongation (e.g. a grain-boundary oxide wedge) and the main axis (e.g. the
  needle/tip axis).
- 18 vs 19 are the **same image** with different wording — the geometric
  primitives are identical; resolve them per the asked quantities (18: apex→oxide
  distance + GB angle; 19: oxide-wedge axial/lateral components from
  `farthest_axial`/`farthest_lateral` + max perpendicular width).

## validation

### foundational
- **Sanity vs the overlay**: the apex must sit on the visible tip; the axis must
  run down the feature's long direction; the sub-feature mask must cover the
  intended region. If not, fix the SEGMENTATION (the tool geometry is exact for a
  correct mask — validated on synthetic ellipse/wedge/cone and offset sub-blobs).
- **Units & magnitudes**: lengths in the stated unit and physically plausible;
  aspect ratios > 1 for elongated features.
- Report `apex_is_defined=False` honestly for symmetric features instead of
  forcing a tip.
