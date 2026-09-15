"""Scale-matched detection, counting and sizing of discrete particles/objects.

The robust counterpart to ad-hoc global Otsu / watershed / SAM segmentation,
which is *scale-blind* and so over-detects on speckled/noisy frames and
under-detects low-contrast objects. Given the known object size (from the
calibration / objective) it:

  1. masks burned-in annotations (scale bar / text) by their geometry, so they
     are not detected as objects;
  2. detects with a SCALE-MATCHED band-pass (a difference-of-Gaussians sized
     from the object diameter) — this passes the object scale and rejects BOTH
     finer noise/speckle (too small) AND large-scale background gradients (too
     big), which is what fixes over- and under-detection at once;
  3. auto-selects intensity POLARITY (objects darker OR brighter than the
     background) instead of assuming "objects are bright";
  4. gates candidates on a background-relative contrast SNR (not an absolute
     threshold), so it transfers across images with different contrast;
  5. MEASURES each object's diameter on the flattened ORIGINAL image (the
     band-pass distorts large sizes), keeping detection and measurement
     separate.

It returns object positions, calibrated diameters, and per-object crop boxes
(for downstream per-object analysis, e.g. an FFT orientation/d-spacing step).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu, difference_of_gaussians
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _norm(g, p=(0.5, 99.5)):
    lo, hi = np.percentile(g, p)
    return np.clip((g - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _mask_annotation(g):
    """Exclude the scale BAR (the single most-elongated saturated component)
    and its label-text region via a bounding box — geometry, not shape rules
    that would wrongly catch a bright object's grainy saturated interior."""
    n = _norm(g, (0.2, 99.8))
    lbl = label(n > 0.93)
    best, best_aspect, best_box = None, 0.0, None
    for r in regionprops(lbl):
        aspect = r.major_axis_length / (r.minor_axis_length + 1e-6)
        if r.area > 400 and aspect > 4.0 and aspect > best_aspect:
            best, best_aspect, best_box = r.label, aspect, r.bbox
    excl = np.zeros(g.shape, bool)
    if best_box is not None:
        minr, minc, maxr, maxc = best_box
        pad = int(0.5 * (maxc - minc))           # scale text margin to bar length
        y0, y1 = max(0, minr - pad - 35), min(g.shape[0], maxr + 40)
        x0, x1 = max(0, minc - 40), min(g.shape[1], maxc + 40)
        excl[y0:y1, x0:x1] = True
    return excl


def _measure_diameter(flat, cy, cx, rough_r_px, px, nominal_r_px=None):
    """Diameter (nm) via the azimuthally-averaged radial HALF-MAX (FWHM) profile.

    Scale- and contrast-robust: the half-max radius is defined relative to this
    object's own (peak - local background) contrast, so it transfers across a
    4 um grain and a 14 nm nanoparticle alike. A windowed Otsu threshold, by
    contrast, floats with whatever else is in the crop and systematically over-
    or under-sizes when the object's contrast or scale differs from the global.
    """
    cy, cx = int(cy), int(cx)
    # Anchor the search scale to the larger of the detected size and the nominal
    # object size: the band-pass `rough_r` under-estimates large soft particles,
    # so a window anchored only to it cuts off before the true edge.
    srch_r = max(rough_r_px, nominal_r_px or 0)
    R = int(max(srch_r * 2.2, 12))
    y0, y1 = max(0, cy - R), min(flat.shape[0], cy + R)
    x0, x1 = max(0, cx - R), min(flat.shape[1], cx + R)
    win = gaussian(flat[y0:y1, x0:x1], max(1.0, rough_r_px * 0.06))
    cyl, cxl = cy - y0, cx - x0
    yy, xx = np.ogrid[:win.shape[0], :win.shape[1]]
    rr = np.sqrt((yy - cyl) ** 2 + (xx - cxl) ** 2).astype(int)
    nr = int(rr.max()) + 1
    if nr < 4:
        return None
    prof = ndi.mean(win, labels=rr, index=np.arange(nr))   # radial profile
    core = max(1, int(rough_r_px * 0.3))
    peak = prof[:core].mean()
    outer = prof[min(nr - 1, int(rough_r_px * 1.3)):]
    bg = np.median(outer) if outer.size else prof[-3:].mean()
    if peak - bg < 1e-9:
        return None
    # Edge = radius of STEEPEST radial descent (max-gradient inflection), not a
    # fixed-fraction threshold: this finds the true boundary independent of edge
    # sharpness — a half-max level under-sizes soft-edged particles and an Otsu
    # threshold floats with the crop. Object is bright in `flat`, so the profile
    # decreases outward and the edge is the most-negative gradient.
    prof_s = ndi.gaussian_filter1d(prof, max(1.0, srch_r * 0.08))
    lo = max(1, int(srch_r * 0.25))
    hi = min(nr - 1, int(srch_r * 2.0))
    if hi - lo < 3:
        return None
    grad = np.diff(prof_s[lo:hi + 1])
    r_edge = lo + int(np.argmin(grad))
    return 2.0 * float(r_edge) * px


def _detect_polarity(n, excl, radius_px, p):
    """Core detection for one polarity (n already oriented so objects bright)."""
    bg = gaussian(n, radius_px * 1.6)
    flat = n - bg                                  # illumination-flattened
    low = max(1.0, radius_px * p["low_sigma_frac"])
    high = radius_px * p["high_sigma_frac"]
    bp = np.clip(difference_of_gaussians(n, low, high), 0, None)
    ref = bp[(~excl) & (bp < np.percentile(bp[~excl], 80))]
    bg_mean, bg_std = ref.mean(), ref.std() + 1e-12
    thr = bg_mean + p["k_thresh"] * bg_std
    fg = (bp > thr) & ~excl
    fg = binary_opening(fg, disk(max(2, int(radius_px * 0.08))))
    fg = ndi.binary_fill_holes(fg)
    fg = remove_small_objects(fg, int((radius_px * p["min_frac"]) ** 2 * np.pi))
    dist = gaussian(ndi.distance_transform_edt(fg), max(2, int(radius_px * 0.25)))
    pk = peak_local_max(dist, min_distance=int(radius_px * 0.9), labels=fg)
    mk = np.zeros(fg.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, mk, mask=fg)
    return lab, flat, bp, bg_mean, bg_std


def _collect(lab, flat, bp, bg_mean, bg_std, radius_px, px, p, H, W):
    # Precision gate on the SCALE-MATCHED BAND-PASS response, not on intensity
    # contrast: the band-pass already suppresses both finer noise/speckle and
    # large-scale background, so a region's mean band-pass strength (in units of
    # the band-pass background sigma) cleanly separates a real object (strong,
    # focused response) from a speckle clump that barely crossed threshold —
    # and it is robust to a textured background (low-contrast crystalline
    # particles) that defeats an intensity-vs-ring SNR.
    parts, snrs = [], []
    for r in regionprops(lab):
        d_eq_nm = r.equivalent_diameter * px
        if d_eq_nm < p["d_min_nm"] or d_eq_nm > p["d_max_nm"]:
            continue
        if r.solidity < p["solidity_min"]:
            continue
        m = lab == r.label
        bp_snr = (bp[m].mean() - bg_mean) / bg_std
        if bp_snr < p["snr_min"]:
            continue
        cy, cx = r.centroid
        d_meas = _measure_diameter(flat, cy, cx, r.equivalent_diameter / 2, px,
                                   nominal_r_px=radius_px)
        d_nm = d_meas if d_meas else d_eq_nm
        border = cx < radius_px or cy < radius_px or cx > W - radius_px or cy > H - radius_px
        R = int(r.equivalent_diameter / 2)
        parts.append(dict(cy=float(cy), cx=float(cx), diameter_nm=round(float(d_nm), 3),
                          bandpass_snr=round(float(bp_snr), 1),
                          solidity=round(float(r.solidity), 3), border=bool(border),
                          bbox=(max(0, int(cy) - R), max(0, int(cx) - R),
                                min(H, int(cy) + R), min(W, int(cx) + R))))
        snrs.append(bp_snr)
    score = len(parts) * (np.median(snrs) if snrs else 0.0)
    return parts, score


def scale_matched_blob_detect(image_array, object_diameter_nm, pixel_size_nm,
                              polarity="auto", params=None):
    g = np.asarray(image_array, float)
    if g.ndim == 3:
        g = g[..., :3].mean(-1) if g.shape[2] in (3, 4) else g[..., 0]
    H, W = g.shape
    px = float(pixel_size_nm)
    radius_px = (float(object_diameter_nm) / px) / 2.0
    p = dict(low_sigma_frac=0.14, high_sigma_frac=1.3, k_thresh=3.0,
             min_frac=0.45, solidity_min=0.8, snr_min=5.0,
             d_min_nm=0.4 * object_diameter_nm, d_max_nm=4.0 * object_diameter_nm)
    p.update(params or {})

    excl = _mask_annotation(g)
    n = _norm(g)
    cands = {"bright": n, "dark": 1.0 - n}
    if polarity in ("bright", "dark"):
        order = [polarity]
    else:
        order = ["bright", "dark"]
    best = None
    for pol in order:
        lab, flat, bp, bg_mean, bg_std = _detect_polarity(cands[pol], excl, radius_px, p)
        parts, score = _collect(lab, flat, bp, bg_mean, bg_std, radius_px, px, p, H, W)
        if best is None or score > best[0]:
            best = (score, pol, parts)
    score, pol_used, parts = best

    if not parts:
        return dict(n_detected=0, polarity_used=pol_used,
                    annotation_masked=bool(excl.any()), objects=[],
                    note=("No objects matched the object scale; check "
                          "object_diameter_nm / pixel_size_nm or that objects exist."))
    diam = [q["diameter_nm"] for q in parts]
    return dict(
        n_detected=len(parts),
        n_interior=sum(not q["border"] for q in parts),
        polarity_used=pol_used,
        annotation_masked=bool(excl.any()),
        diameters_nm=diam,
        diameter_median_nm=round(float(np.median(diam)), 3),
        diameter_mean_nm=round(float(np.mean(diam)), 3),
        diameter_std_nm=round(float(np.std(diam)), 3),
        objects=parts,
    )


from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="scale_matched_blob_detect",
    description=(
        "Scale-matched detection, counting and sizing of discrete particles/"
        "objects. Robust alternative to global Otsu/watershed/SAM (which are "
        "scale-blind and over- or under-detect): a band-pass sized from the "
        "known object diameter, auto intensity polarity, an SNR contrast gate, "
        "annotation (scale-bar) masking, and size measured on the original."
    ),
    import_line="from scilink.skills._shared.blob_detect import scale_matched_blob_detect",
    signature=("scale_matched_blob_detect(image_array, object_diameter_nm, "
               "pixel_size_nm, polarity='auto', params=None) -> dict"),
    agents=["image_analysis"],
    when_to_use=(
        "Use to DETECT / COUNT / SIZE discrete particles that are SPARSE / "
        "WELL-SEPARATED on a noisy or SPECKLED background (nanoparticles, droplets, "
        "pores, protein assemblies) — its band-pass + band-pass-SNR gate give strong "
        "SPECKLE REJECTION, fixing the over-detection that global thresholding/"
        "watershed/SAM produce on grainy frames. The object diameter is known from "
        "the objective/calibration. Pair with a per-object step (fourier_reflection_"
        "map or a local FFT on each returned bbox) for a per-object property. "
        "Choose the RIGHT detector: for DENSELY-PACKED or FAINT small cores (e.g. "
        "a crowded field of small cores) this merges/under-counts — use log_blob_detect "
        "(scale-space LoG) instead. For SPACE-FILLING grains (polycrystalline, "
        "tessellated cells) use boundary segmentation. For genuinely touching "
        "objects needing instance masks, use run_sam_analysis."
    ),
    parameters={
        "object_diameter_nm": {"type": "number", "description":
            "Approximate object diameter in nm (sets the detection scale). For a "
            "broad size range pass a typical value — detection has bandwidth and "
            "the size filter defaults to [0.4x, 4x]."},
        "pixel_size_nm": {"type": "number", "description": "nm per pixel (calibration)."},
        "polarity": {"type": "string", "description":
            "'auto' (default) tries dark- and bright-on-background and keeps the "
            "better; or force 'dark' / 'bright'."},
        "params": {"type": "object", "description":
            "Tunable knobs — ADJUST from the result vs the image; defaults are a "
            "starting point: * k_thresh (band-pass threshold sigma, default 3.0) and "
            "* snr_min (per-object band-pass-SNR gate, default 5.0): RAISE either if "
            "speckle/noise is over-detected; LOWER if real objects are missed. "
            "* solidity_min (default 0.8): lower for irregular objects being dropped. "
            "* d_min_nm / d_max_nm (size filter, default 0.4x/4x the object size): "
            "tighten to a known band. Re-run with adjusted values until the overlay "
            "matches the image."},
    },
    required=["image_array", "object_diameter_nm", "pixel_size_nm"],
    returns=(
        "dict with: 'n_detected', 'n_interior' (excluding border-truncated), "
        "'polarity_used', 'annotation_masked' (bool — was a scale bar masked), "
        "'diameters_nm' and 'diameter_median/mean/std_nm', and 'objects' (list of "
        "{cy, cx, diameter_nm, contrast_snr, solidity, border, bbox=(y0,x0,y1,x1)}). "
        "Use each object's 'bbox' to crop for a per-object property step. If nothing "
        "matches: 'note' explains why (scale/calibration or no objects)."
    ),
    example=(
        "res = scale_matched_blob_detect(image, object_diameter_nm=20.0, pixel_size_nm=0.234)\n"
        "print(res['n_detected'], 'median d =', res.get('diameter_median_nm'))\n"
        "for o in res['objects']:\n"
        "    y0, x0, y1, x1 = o['bbox']; crop = image[y0:y1, x0:x1]\n"
        "    # e.g. local FFT on `crop` for this object's lattice orientation"
    ),
)
