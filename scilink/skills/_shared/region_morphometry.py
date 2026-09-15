"""Region & interface morphometry for microscopy features.

Self-contained (numpy + scipy). Given a binary region mask, returns robust
geometric descriptors that segmentation-only tools don't: a PCA principal axis,
the APEX (pointed/narrow end), length-along-axis, the width profile and maximum
perpendicular width, aspect ratio, and the plane angle of a sub-feature relative
to the main axis. Also measures the position of a sub-feature (e.g. an oxide
wedge inside an APT needle) resolved into axial / lateral components from the
apex.

Built for: APT needle apex + oxide geometry (18/19), blister dome + delamination
void geometry (08), precipitate/feature extent & interface-plane angles — the
"region geometry" objectives that have no home tool in scilink.

Main entry points:
  region_geometry(mask, ...)            -> dict of geometric descriptors
  feature_vs_axis(feature, axis, apex)  -> sub-feature position/angle vs an axis
  width_profile(mask, axis, center)     -> width(s) along the axis (dome height,
                                           void thickness, taper, ...)
Synthetic shape generators (for validation): make_ellipse, make_wedge,
make_cone, place_blob.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = ["region_geometry", "feature_vs_axis", "width_profile",
           "angle_between_deg", "make_ellipse", "make_wedge", "make_cone",
           "place_blob"]


def _coords(mask):
    ys, xs = np.where(mask)
    return np.column_stack([xs, ys]).astype(float)   # (N,2) as (x, y)


def _principal_axes(pts):
    """PCA: returns center, major unit vec, minor unit vec (major = larger var)."""
    center = pts.mean(0)
    cov = np.cov((pts - center).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return center, evecs[:, order[0]], evecs[:, order[1]]


def angle_between_deg(v1, v2):
    """Unsigned acute angle between two directions (axes), 0..90 deg."""
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    c = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def width_profile(mask, axis_vec, center, nbins=40, pixel_size=1.0):
    """Perpendicular width of `mask` sampled along `axis_vec`.
    Returns (axial_positions, widths) in physical units (apex-independent;
    positions are measured from `center` along axis_vec)."""
    pts = _coords(mask)
    perp = np.array([-axis_vec[1], axis_vec[0]])
    a = (pts - center) @ axis_vec
    p = (pts - center) @ perp
    edges = np.linspace(a.min(), a.max(), nbins + 1)
    pos, wid = [], []
    for i in range(nbins):
        m = (a >= edges[i]) & (a < edges[i + 1])
        if m.sum() >= 2:
            pos.append(0.5 * (edges[i] + edges[i + 1]) * pixel_size)
            wid.append((p[m].max() - p[m].min()) * pixel_size)
    return np.array(pos), np.array(wid)


def region_geometry(mask, pixel_size=1.0, pixel_unit="px"):
    """Geometric descriptors of a single binary region.

    Returns dict with: area, equivalent_diameter, centroid, axis_angle_deg
    (major-axis orientation), axis_vec, length (along major axis), max_width,
    mean_width, aspect_ratio (length/max_width), apex (the narrow/pointed end,
    as (x,y) px), base (the opposite end), apex_is_defined (False for ~symmetric
    shapes), and the width profile. All lengths in `pixel_unit`.
    """
    mask = np.asarray(mask, bool)
    pts = _coords(mask)
    if len(pts) < 8:
        return {"area": float(len(pts)), "valid": False}
    center, major, minor = _principal_axes(pts)
    a = (pts - center) @ major
    p = (pts - center) @ minor
    # robust extent (avoid single-pixel spikes) but keep true tips for apex
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    length = (hi - lo) * pixel_size
    max_width = (np.percentile(p, 99.5) - np.percentile(p, 0.5)) * pixel_size
    pos, wid = width_profile(mask, major, center, pixel_size=pixel_size)
    mean_width = float(np.mean(wid)) if len(wid) else max_width
    # apex = the end whose local width is smaller (the pointed/narrow tip)
    end_hi = center + major * hi
    end_lo = center + major * lo
    band = 0.15 * (hi - lo)
    w_hi = (np.ptp(p[a > hi - band]) if (a > hi - band).sum() > 2 else np.inf)
    w_lo = (np.ptp(p[a < lo + band]) if (a < lo + band).sum() > 2 else np.inf)
    apex_defined = abs(w_hi - w_lo) > 0.15 * max(w_hi, w_lo, 1e-9) / pixel_size
    if w_hi <= w_lo:
        apex, base = end_hi, end_lo
        axis_vec = major if (end_lo - end_hi) @ major < 0 else -major
    else:
        apex, base = end_lo, end_hi
        axis_vec = major if (end_hi - end_lo) @ major < 0 else -major
    # axis_vec points from apex toward base (into the body)
    axis_vec = (base - apex) / (np.linalg.norm(base - apex) + 1e-12)
    area = float(mask.sum()) * pixel_size ** 2
    return dict(
        valid=True, area=area,
        equivalent_diameter=2 * np.sqrt(area / np.pi),
        centroid=(float(center[0]), float(center[1])),
        axis_angle_deg=float(np.degrees(np.arctan2(major[1], major[0])) % 180),
        axis_vec=(float(axis_vec[0]), float(axis_vec[1])),
        length=float(length), max_width=float(max_width),
        mean_width=float(mean_width),
        aspect_ratio=float(length / (max_width + 1e-9)),
        apex=(float(apex[0]), float(apex[1])),
        base=(float(base[0]), float(base[1])),
        apex_is_defined=bool(apex_defined),
        width_profile=dict(axial_pos=pos.tolist(), width=wid.tolist()),
        pixel_unit=pixel_unit,
    )


def feature_vs_axis(feature_mask, axis_vec, apex_xy, pixel_size=1.0,
                    pixel_unit="px"):
    """Position & orientation of a sub-feature (e.g. an oxide region) relative to
    a main axis anchored at `apex_xy` (e.g. a needle tip).

    Returns: distance_along_axis (axial), perpendicular_offset (lateral),
    axial_extent, lateral_extent, farthest-point distance from apex resolved into
    (axial, lateral), the feature's own principal-axis angle vs the main axis
    (plane angle), and the feature centroid.
    """
    feature_mask = np.asarray(feature_mask, bool)
    pts = _coords(feature_mask)
    if len(pts) < 4:
        return {"valid": False}
    apex = np.asarray(apex_xy, float)
    ax = np.asarray(axis_vec, float); ax = ax / (np.linalg.norm(ax) + 1e-12)
    perp = np.array([-ax[1], ax[0]])
    rel = pts - apex
    axial = rel @ ax
    lateral = rel @ perp
    centroid = pts.mean(0)
    crel = centroid - apex
    # farthest feature point from apex (wedge tip)
    far = pts[np.argmax(np.linalg.norm(rel, axis=1))]
    fr = far - apex
    # feature's own principal axis (its elongation direction) vs the main axis
    _, fmaj, _ = _principal_axes(pts)
    return dict(
        valid=True,
        distance_along_axis=float((crel @ ax) * pixel_size),
        perpendicular_offset=float((crel @ perp) * pixel_size),
        axial_extent=float((axial.max() - axial.min()) * pixel_size),
        lateral_extent=float((lateral.max() - lateral.min()) * pixel_size),
        max_perpendicular_width=float((lateral.max() - lateral.min()) * pixel_size),
        farthest_point=(float(far[0]), float(far[1])),
        farthest_axial=float((fr @ ax) * pixel_size),
        farthest_lateral=float((fr @ perp) * pixel_size),
        farthest_distance=float(np.linalg.norm(fr) * pixel_size),
        plane_angle_to_axis_deg=angle_between_deg(fmaj, ax),
        centroid=(float(centroid[0]), float(centroid[1])),
        pixel_unit=pixel_unit,
    )


# --------------------------------------------------------------------------- #
#  Synthetic shape generators (for validation)
# --------------------------------------------------------------------------- #
def make_ellipse(shape, center, a, b, angle_deg):
    """Filled rotated ellipse. a=semi-major, b=semi-minor (px). angle from +x."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cx, cy = center
    th = np.radians(angle_deg)
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def make_wedge(shape, apex, axis_deg, length, half_angle_deg):
    """Filled isoceles triangle: tip at `apex`, pointing along axis_deg, given
    length and half-angle (so base width = 2*length*tan(half_angle))."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ax = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])
    perp = np.array([-ax[1], ax[0]])
    rel = np.stack([xx - apex[0], yy - apex[1]], -1)
    a = rel @ ax
    p = np.abs(rel @ perp)
    return (a >= 0) & (a <= length) & (p <= a * np.tan(np.radians(half_angle_deg)))


def make_cone(shape, apex, axis_deg, length, base_halfwidth):
    """Cone/needle: tip at apex, linearly widening to base_halfwidth at length."""
    return make_wedge(shape, apex, axis_deg, length,
                      np.degrees(np.arctan2(base_halfwidth, length)))


def place_blob(shape, center, radius):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
