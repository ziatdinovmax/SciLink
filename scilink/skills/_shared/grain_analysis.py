"""Grain / EBSD-IPF microstructure analysis.

Self-contained (numpy + scipy + scikit-image). Segments grains from either an
**IPF colour map** (RGB; grains separated by crystal-orientation colour) or a
**grayscale** image (electron-channeling / BSE contrast), and reports:

  * grain count + equivalent-diameter / area distributions,
  * straight-grain-boundary fraction (an annealing-twin / Sigma3 proxy) via the
    probabilistic Hough transform on the boundary skeleton,
  * IPF texture (fraction of grains / area near each cubic pole) for colour maps,
  * a labelled grain map + boundary skeleton + the detected straight segments.

Why a dedicated tool: a generic "discrete object" detector cannot express
grain-boundary topology, twin-boundary straightness, or IPF texture — the things
grain/EBSD objectives actually ask for. This packages a vetted pipeline so the
agent does not re-derive grain segmentation each time.

Main entry point: grain_analysis(image, ...) -> dict.
Synthetic generator: make_grain_map(...) for validation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, transform

__all__ = ["grain_analysis", "straight_boundary_fraction", "make_grain_map"]

POLES = ["<001> (red)", "<101> (green)", "<111> (blue)"]


# --------------------------------------------------------------------------- #
def _to_gray(image):
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., :3].astype(float)
        return 0.2125 * a[..., 0] + 0.7154 * a[..., 1] + 0.0721 * a[..., 2]
    return a.astype(float)


def _ipf_pole(mean_rgb):
    """Dominant-channel IPF pole. red=<001>, green=<101>, blue=<111>."""
    frac = np.asarray(mean_rgb, float) / (np.sum(mean_rgb) + 1e-9)
    return POLES[int(np.argmax(frac))]


def straight_boundary_fraction(boundary_bool, min_len_px=25, line_gap=2,
                               threshold=10):
    """Fraction of boundary skeleton length on straight segments (Hough).

    A curved boundary is locally straight over short spans, so a small line_gap
    and a min_len that is long relative to grain-boundary curvature are needed to
    keep curved boundaries from registering as straight (twin) segments.
    Returns (fraction, total_len_px, straight_len_px, segments)."""
    total = float(boundary_bool.sum())
    if total < 1:
        return float("nan"), 0.0, 0.0, []
    lines = transform.probabilistic_hough_line(
        boundary_bool, threshold=threshold, line_length=min_len_px,
        line_gap=line_gap)
    straight = sum(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) for p0, p1 in lines)
    return min(straight / total, 1.0), total, float(straight), lines


def _describe(arr, scale=1.0):
    a = np.asarray(arr, float) * scale
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75))}


# --------------------------------------------------------------------------- #
def _segment_ipf(rgb, boundary_sensitivity, min_area_px):
    rgbf = np.asarray(rgb, float)
    grad = sum(filters.sobel(filters.gaussian(rgbf[..., k], 1)) for k in range(3))
    grad = grad / (grad.max() + 1e-9)
    thr = np.percentile(grad, 100 * (1 - 0.25 * boundary_sensitivity)) * 0.0 + \
        0.04 + 0.12 * (1 - boundary_sensitivity)
    bnd = grad > thr
    interior = ~morphology.binary_dilation(bnd, morphology.disk(1))
    mk = measure.label(interior)
    ws = segmentation.watershed(grad, mk)
    return ws, None


def _segment_gray(gray, boundary_sensitivity, min_area_px, contamination):
    gs = filters.gaussian(gray, 2.0)
    contam = np.zeros(gs.shape, bool)
    if contamination in ("auto", "dark", True):
        contam |= gs < np.percentile(gs, 2)
    if contamination in ("auto", "bright"):
        contam |= gs > np.percentile(gs, 99.5)
    contam = morphology.binary_dilation(
        morphology.remove_small_objects(contam, 8), morphology.disk(3))
    grad = filters.sobel(gs)
    # normalise by a high PERCENTILE (not max): a few sharp contamination/edge
    # gradients would otherwise crush faint channeling-contrast grain boundaries.
    grad = grad / (np.percentile(grad, 99) + 1e-9)
    # h-minima depth sets marker count; calibrated so the default (0.5) gives a
    # plausible grain count on faint channeling contrast. Lower depth (higher
    # sensitivity) = more grains. TUNE from the overlay — channeling-contrast
    # grain segmentation is inherently image-dependent.
    h = 0.04 + 0.16 * (1 - boundary_sensitivity)
    hm = morphology.h_minima(filters.gaussian(grad, 2), h)
    mk = measure.label(hm)
    ws = segmentation.watershed(grad, mk, mask=~contam)
    return ws, contam


def grain_analysis(image, pixel_size=1.0, pixel_unit="px", mode="auto",
                   min_grain_diameter=None, boundary_sensitivity=0.5,
                   contamination="auto", min_len_frac=0.03,
                   min_curved_fraction=0.30):
    """Segment grains and quantify size, twin-boundary fraction, and texture.

    Parameters
    ----------
    image : 2-D grayscale or 3-D RGB (IPF map).
    pixel_size, pixel_unit : physical calibration (e.g. 0.4, "um"); sizes are
        reported in pixel_unit (and pixel_unit^2 for area).
    mode : "auto" | "ipf" (colour) | "gray" (channeling/BSE).
    min_grain_diameter : grains smaller than this (in pixel_unit) are dropped;
        default ~1.5 % of the frame.
    boundary_sensitivity : 0..1, higher = more boundaries (finer segmentation).
    contamination : "auto"/"dark"/"bright"/False — mask spurious particles (gray).
    min_len_frac : straight-segment minimum length as a fraction of min(H,W).

    Returns dict (see module docstring); key fields: n_grains, grain_diameter,
    grain_area, twin_boundary_fraction, texture, dominant_texture, label_map,
    boundary_skeleton, straight_segments.
    """
    a = np.asarray(image)
    if mode == "auto":
        mode = "ipf" if (a.ndim == 3 and a.shape[-1] >= 3 and
                         float(np.mean(np.std(a[..., :3].astype(float), axis=-1))) > 3) \
            else "gray"
    H, W = a.shape[:2]
    px2 = pixel_size ** 2
    min_d = (min_grain_diameter if min_grain_diameter is not None
             else 0.015 * (H + W) / 2 * pixel_size)
    min_area_px = np.pi * (min_d / pixel_size / 2) ** 2

    if mode == "ipf":
        rgb = a[..., :3]
        ws, contam = _segment_ipf(rgb, boundary_sensitivity, min_area_px)
    else:
        gray = _to_gray(a)
        ws, contam = _segment_gray(gray, boundary_sensitivity, min_area_px,
                                   contamination)

    rgbf = a[..., :3].astype(float) if mode == "ipf" else None
    grains, diam_px, area_px = [], [], []
    pole_count = {p: 0 for p in POLES}
    pole_area = {p: 0.0 for p in POLES}
    for r in measure.regionprops(ws):
        d_phys = 2 * np.sqrt(r.area / np.pi) * pixel_size
        if d_phys < min_d:
            continue
        diam_px.append(2 * np.sqrt(r.area / np.pi))
        area_px.append(r.area)
        rec = dict(label=r.label, area_px=r.area, eqd_phys=d_phys,
                   centroid=(float(r.centroid[1]), float(r.centroid[0])))
        if mode == "ipf":
            mean_rgb = rgbf[ws == r.label].mean(axis=0)
            pole = _ipf_pole(mean_rgb)
            rec["pole"] = pole
            pole_count[pole] += 1
            pole_area[pole] += r.area
        grains.append(rec)

    # straight (twin) boundary fraction on the grain-boundary skeleton
    skel = morphology.skeletonize(segmentation.find_boundaries(ws, mode="thick"))
    if contam is not None:
        skel &= ~contam
    frac, tot_len, str_len, segments = straight_boundary_fraction(
        skel, min_len_px=max(10, int(min_len_frac * min(H, W))))
    # The straight fraction is only a TWIN proxy when there is a curved-boundary
    # baseline to contrast straight twins against. The reliability test is the
    # SIZE of that baseline, not just whether `frac` clears some high bar: real
    # recrystallized twinned microstructures keep a substantial curved general-
    # boundary population (straight Σ3 twins are a MINORITY of total boundary
    # length), so the curved fraction (1 - frac) stays well above zero. When the
    # network is predominantly straight (idealized / polygonal / faceted /
    # columnar grains, or annealed grains with naturally straight boundaries),
    # the curved baseline collapses and the straightness says nothing about
    # twins — flag it so callers report ~0 rather than the raw fraction.
    #
    # Gating on the curved baseline (default >= 0.30) rather than the old
    # `frac < 0.9` is what stops a 0.83-straight EBSD network from being reported
    # as an 0.83 twin fraction (a real benchmark false-positive). NOTE: this is
    # still a morphological proxy that cannot separate a straight GRAIN boundary
    # from an intragranular twin lamella; a misorientation (Σ3) measurement is
    # the only definitive twin test when orientation data is available.
    curved_fraction = float("nan") if not np.isfinite(frac) else 1.0 - frac
    twin_proxy_reliable = bool(
        np.isfinite(curved_fraction) and curved_fraction >= min_curved_fraction)

    n = len(grains)
    out = dict(
        mode=mode, n_grains=n,
        pixel_size=pixel_size, pixel_unit=pixel_unit,
        grain_diameter=_describe(diam_px, pixel_size),
        grain_diameters=np.array(diam_px) * pixel_size,
        grain_area=_describe(area_px, px2),
        straight_boundary_fraction=frac,
        curved_boundary_fraction=curved_fraction,
        min_curved_fraction=float(min_curved_fraction),
        twin_boundary_fraction=(frac if twin_proxy_reliable else 0.0),
        twin_proxy_reliable=twin_proxy_reliable,
        boundary_length_px=tot_len, straight_length_px=str_len,
        n_straight_segments=len(segments),
        mean_twin_segments_per_grain=(len(segments) / max(n, 1)
                                      if twin_proxy_reliable else 0.0),
        label_map=ws, boundary_skeleton=skel,
        straight_segments=(segments if twin_proxy_reliable else []),
    )
    if mode == "ipf":
        area_tot = sum(pole_area.values()) + 1e-9
        out["texture"] = {p: dict(
            count=pole_count[p],
            count_fraction=pole_count[p] / max(n, 1),
            area_fraction=pole_area[p] / area_tot) for p in POLES}
        out["dominant_texture"] = max(POLES, key=lambda p: pole_area[p])
    return out


# --------------------------------------------------------------------------- #
#  Synthetic grain-map generator (for validation)
# --------------------------------------------------------------------------- #
def make_grain_map(shape=(400, 600), n_seeds=40, kind="ipf",
                   pole_fractions=(0.34, 0.33, 0.33), boundary_darkness=0.4,
                   noise=0.0, seed=0):
    """Voronoi grain map with KNOWN grain count, areas and (for IPF) pole mix.

    Returns (image, truth) where truth has n_grains, areas_px, label_map, and
    (ipf) pole_count/pole_area_fraction.
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.integers(0, H, n_seeds),
                           rng.integers(0, W, n_seeds)])
    yy, xx = np.mgrid[0:H, 0:W]
    # nearest-seed Voronoi label via chunked argmin (memory-safe)
    lab = np.zeros((H, W), int)
    best = np.full((H, W), np.inf)
    for i, (py, px) in enumerate(pts):
        d = (yy - py) ** 2 + (xx - px) ** 2
        m = d < best
        best[m] = d[m]; lab[m] = i
    truth = dict(n_grains=n_seeds,
                 areas_px=np.array([(lab == i).sum() for i in range(n_seeds)]),
                 label_map=lab)
    # assign poles / colours
    poles = rng.choice([0, 1, 2], size=n_seeds, p=np.array(pole_fractions) /
                       np.sum(pole_fractions))
    if kind == "ipf":
        base = np.array([[220, 40, 40], [40, 200, 60], [50, 70, 220]], float)
        img = np.zeros((H, W, 3), float)
        for i in range(n_seeds):
            jitter = rng.normal(0, 18, 3)
            img[lab == i] = np.clip(base[poles[i]] + jitter, 0, 255)
        truth["pole_count"] = {POLES[k]: int(np.sum(poles == k)) for k in range(3)}
        ta = {POLES[k]: float(truth["areas_px"][poles == k].sum()) for k in range(3)}
        s = sum(ta.values()) + 1e-9
        truth["pole_area_fraction"] = {p: ta[p] / s for p in POLES}
    else:
        levels = rng.uniform(60, 200, n_seeds)
        img = np.zeros((H, W), float)
        for i in range(n_seeds):
            img[lab == i] = levels[i]
    # draw boundaries (darken) — grayscale channeling look
    bnd = segmentation.find_boundaries(lab, mode="outer")
    if kind != "ipf":
        img[bnd] *= (1 - boundary_darkness)
    if noise > 0:
        img = img + rng.normal(0, noise * (255 if kind == "ipf" else 1), img.shape)
    return (np.clip(img, 0, 255).astype(np.uint8) if kind == "ipf"
            else img), truth
