"""Scale-space (LoG) detection of DENSE, faint, small particles.

The companion to ``scale_matched_blob_detect``: where the band-pass + watershed
detector excels at sparse/medium particles on a noisy background, it *merges*
densely-packed touching cores (a distance transform of a dense cluster is one
plateau) and its precision gates reject faint cores — it caps out well below the
true count on a dense, low-contrast field of small cores.

This tool instead uses ``skimage.feature.blob_log`` — a multi-scale Laplacian-of-
Gaussian that finds EACH particle as its own scale-space maximum, so densely-
packed and faint cores are detected individually. It is sized from the known
object diameter, selects intensity polarity (set explicitly, or 'auto'
fallback), masks burned-in annotations,
and measures each diameter on the original image (shared with the band-pass tool).
"""
import numpy as np
from skimage.feature import blob_log
from skimage.filters import gaussian
from .blob_detect import _norm, _mask_annotation, _measure_diameter


def log_blob_detect(image_array, object_diameter_nm=None, pixel_size_nm=None,
                    polarity="auto", *, min_sigma=None, max_sigma=None,
                    num_sigma=10, threshold=None, threshold_rel=None, overlap=0.5,
                    log_scale=False, exclude_border=False, mask_annotations=True,
                    d_min_nm=None, d_max_nm=None):
    """Faithful superset of skimage.feature.blob_log.

    Every native `blob_log` argument (min_sigma, max_sigma, num_sigma, threshold,
    threshold_rel, overlap, log_scale, exclude_border) is passed STRAIGHT THROUGH
    — with no annotation present and no extra args, this returns exactly what
    `blob_log` would (the defaults here mirror blob_log's own). It only layers on
    OPT-IN conveniences: derive the sigma range from a known object_diameter_nm
    when min/max_sigma are not given; select intensity polarity (blob_log finds
    only BRIGHT blobs, so dark objects need the image inverted — polarity='dark'
    does that for you); mask burned-in scale bars/annotations (no-op if none);
    report a calibrated nm diameter per blob when pixel_size_nm is given; and an
    optional nm size filter (off by default). It adds NO contrast/SNR gate."""
    g = np.asarray(image_array, float)
    if g.ndim == 3:
        g = g[..., :3].mean(-1) if g.shape[2] in (3, 4) else g[..., 0]
    H, W = g.shape
    px = float(pixel_size_nm) if pixel_size_nm else None
    radius_px = ((float(object_diameter_nm) / px) / 2.0
                 if (object_diameter_nm and px) else None)
    # sigma range: explicit pixel values win; else derive from the object size;
    # else fall back to blob_log's own defaults (1, 50).
    if min_sigma is None or max_sigma is None:
        if radius_px:
            sig = radius_px / np.sqrt(2.0)
            min_sigma = max(1.0, sig * 0.6) if min_sigma is None else min_sigma
            max_sigma = sig * 1.6 if max_sigma is None else max_sigma
        else:
            min_sigma = 1.0 if min_sigma is None else min_sigma
            max_sigma = 50.0 if max_sigma is None else max_sigma
    if threshold is None and threshold_rel is None:
        threshold = 0.2                              # blob_log's own default

    n = _norm(g)
    excl = _mask_annotation(g) if mask_annotations else np.zeros(g.shape, bool)
    order = [polarity] if polarity in ("bright", "dark") else ["bright", "dark"]

    best = None
    for pol in order:
        nn = n if pol == "bright" else 1.0 - n     # blob_log finds bright maxima
        img = nn.copy()
        if excl.any():
            img[excl] = np.median(nn[~excl])        # blank annotation region
        flat = gaussian(nn, max(1.0, min_sigma))
        flat = nn - gaussian(nn, max_sigma * 1.6)   # for calibrated sizing / polarity
        locbg = np.median(flat[~excl]) if (~excl).any() else float(np.median(flat))
        blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold,
                         threshold_rel=threshold_rel, overlap=overlap,
                         log_scale=log_scale, exclude_border=exclude_border)
        objs, contrasts = [], []
        for y, x, s in blobs:
            y, x = int(y), int(x)
            if excl[y, x]:
                continue
            r_px = s * np.sqrt(2.0)
            if px:
                d_meas = _measure_diameter(flat, y, x, r_px, px, nominal_r_px=radius_px)
                d_nm = d_meas if d_meas else 2.0 * r_px * px
            else:
                d_nm = 2.0 * r_px                    # pixels if uncalibrated
            if (d_min_nm is not None and d_nm < d_min_nm) or \
               (d_max_nm is not None and d_nm > d_max_nm):
                continue
            edge = radius_px or r_px
            border = x < edge or y < edge or x > W - edge or y > H - edge
            R = int(r_px)
            objs.append(dict(cy=float(y), cx=float(x), diameter_nm=round(float(d_nm), 3),
                             border=bool(border),
                             bbox=(max(0, y - R), max(0, x - R), min(H, y + R), min(W, x + R))))
            contrasts.append(float(flat[y, x] - locbg))
        # Polarity selection (NOT a per-blob gate): the CORRECT polarity has the
        # objects as genuinely high-contrast bright maxima, so its MEDIAN per-blob
        # contrast is higher. Median (not a sum) is robust to the wrong polarity
        # inflating its score by detecting many weak speckle blobs.
        pol_score = float(np.median(contrasts)) if contrasts else -1e9
        if best is None or pol_score > best[0]:
            best = (pol_score, pol, objs)
    _, pol_used, objs = best

    if not objs:
        return dict(n_detected=0, polarity_used=pol_used,
                    annotation_masked=bool(excl.any()), objects=[],
                    note="No blobs at the object scale; check object_diameter_nm / pixel_size_nm.")
    diam = [o["diameter_nm"] for o in objs]
    return dict(
        n_detected=len(objs),
        n_interior=sum(not o["border"] for o in objs),
        polarity_used=pol_used,
        annotation_masked=bool(excl.any()),
        diameters_nm=diam,
        diameter_median_nm=round(float(np.median(diam)), 3),
        diameter_mean_nm=round(float(np.mean(diam)), 3),
        diameter_std_nm=round(float(np.std(diam)), 3),
        objects=objs,
    )


from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="log_blob_detect",
    description=(
        "A faithful SUPERSET of skimage.feature.blob_log (scale-space LoG blob "
        "detection): every native blob_log argument is passed straight through, so "
        "with no annotation present it returns exactly what blob_log would. It adds "
        "only opt-in conveniences: sigma range derived from a known object size, "
        "polarity handling (blob_log finds only BRIGHT blobs; polarity='dark' "
        "inverts for dark objects), scale-bar/annotation masking, and a calibrated "
        "diameter per blob. Good for dense, faint, small cores where region-growing "
        "detectors merge touching particles."
    ),
    import_line="from scilink.skills._shared.log_blob import log_blob_detect",
    signature=("log_blob_detect(image_array, object_diameter_nm=None, "
               "pixel_size_nm=None, polarity='auto', *, min_sigma=None, "
               "max_sigma=None, num_sigma=10, threshold=None, threshold_rel=None, "
               "overlap=0.5, log_scale=False, exclude_border=False, "
               "mask_annotations=True, d_min_nm=None, d_max_nm=None) -> dict"),
    agents=["image_analysis"],
    when_to_use=(
        "Use to DETECT / COUNT / SIZE small particles that are DENSELY PACKED "
        "and/or FAINT (low contrast) — closely-spaced small cores on a crowded "
        "field — where a region-growing detector "
        "(scale_matched_blob_detect, watershed, SAM) merges touching cores and "
        "under-counts. Reach for scale_matched_blob_detect instead when particles "
        "are SPARSE/well-separated on a speckled background (it adds stronger "
        "speckle rejection); reach here when the field is crowded with many small "
        "low-contrast cores. Returns per-object bbox for a downstream property step."
    ),
    parameters={
        "image_array": {"type": "object", "description": "2D image (or RGB)."},
        "object_diameter_nm": {"type": "number", "description":
            "Optional — approximate particle diameter in nm. If given (with "
            "pixel_size_nm) it sets the sigma range; otherwise pass min_sigma/"
            "max_sigma directly, or let it fall back to blob_log's defaults."},
        "pixel_size_nm": {"type": "number", "description":
            "Optional — nm/pixel. Needed only for calibrated nm diameters; without "
            "it, diameters are returned in pixels."},
        "polarity": {"type": "string", "description":
            "SET from the image: 'dark' or 'bright' objects. blob_log finds only "
            "bright blobs, so 'dark' inverts the image for you. 'auto' is a fallback "
            "that can mis-pick on dense fields — prefer setting it."},
        "min_sigma": {"type": "number", "description": "blob_log arg (px) — passthrough; overrides object_diameter_nm."},
        "max_sigma": {"type": "number", "description": "blob_log arg (px) — passthrough."},
        "num_sigma": {"type": "number", "description": "blob_log arg (default 10)."},
        "threshold": {"type": "number", "description": "blob_log absolute LoG threshold (default 0.2 if neither threshold nor threshold_rel set)."},
        "threshold_rel": {"type": "number", "description":
            "blob_log relative threshold (0-1). The main sensitivity knob: LOWER if "
            "faint cores are missed, RAISE if noise is detected. Tune from the overlay."},
        "overlap": {"type": "number", "description": "blob_log arg (default 0.5); lower to split touching blobs."},
        "log_scale": {"type": "boolean", "description": "blob_log arg; helps very wide size ranges."},
        "exclude_border": {"type": "boolean", "description": "blob_log arg."},
        "mask_annotations": {"type": "boolean", "description":
            "Default true — exclude a detected burned-in scale bar; no-op if none. "
            "Set false for exact blob_log behavior."},
        "d_min_nm": {"type": "number", "description": "Optional nm size filter (off by default)."},
        "d_max_nm": {"type": "number", "description": "Optional nm size filter (off by default)."},
    },
    required=["image_array"],
    returns=(
        "dict with 'n_detected', 'n_interior', 'polarity_used', 'annotation_masked', "
        "'diameters_nm' (px if uncalibrated) and 'diameter_median/mean/std_nm', and "
        "'objects' (list of {cy, cx, diameter_nm, border, bbox=(y0,x0,y1,x1)}). If none: 'note'."
    ),
    example=(
        "# sized from the known particle diameter, dark cores:\n"
        "res = log_blob_detect(image, object_diameter_nm=7.0, pixel_size_nm=0.11,\n"
        "                      polarity='dark', threshold_rel=0.3)\n"
        "print(res['n_detected'], res.get('diameter_median_nm'))\n"
        "# or pure passthrough (== raw blob_log): \n"
        "# res = log_blob_detect(image, min_sigma=3, max_sigma=8, threshold=0.05)"
    ),
)
