"""Quality-control panels + prior-free metrics for dense atom-column detection.

Verifying that a detector found the right columns in an atomic-resolution image
is NOT a counting problem: a full-frame overlay of thousands of marks is an
unjudgeable red haze, and an a-priori "expected count" needs the zone axis and
per-column visibility (hard to predict for multi-sublattice materials), so a
count ratio false-flags correct detections as over-detection.

This tool verifies the way a microscopist actually does — spot-check zoomed
regions + check the spacing statistics:

  * TARGETED zoom-in overlays (dots drawn on the image crop) placed where
    problems would be — the region with the most short nearest-neighbor pairs
    (likely duplicates / over-detection) and the largest coverage gap (likely
    misses) — not random, so localized errors are not sampled away.
  * PRIOR-FREE metrics: the nearest-neighbor distance distribution (a spike of
    anomalously short distances = split/duplicate marks; a clean unimodal peak =
    correct regardless of the absolute count), coverage gaps, and — when a DCNN
    heatmap is available — the fraction of detections sitting on a heatmap peak.

The absolute count survives only as an order-of-magnitude sanity check.
"""

import io

import numpy as np


def _to_gray(a):
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a


def _nn_distances(pos):
    """Nearest-neighbor distance per point (pixels). Returns (dists, nn_index)."""
    from scipy.spatial import cKDTree
    if len(pos) < 2:
        return np.array([]), np.array([], int)
    tree = cKDTree(pos)
    d, i = tree.query(pos, k=2)            # k=1 is self
    return d[:, 1], i[:, 1]


def detection_quality_panels(image, positions, pixel_size_nm=None,
                             heatmap=None, params=None):
    """QC a dense atom-column detection with targeted zoom panels + metrics.

    Args:
        image: 2D grayscale (or HxWx3) detector image the positions came from.
        positions: (N, 2) array of detected column coordinates as (x, y) in
            image pixels (the convention detect_atoms / detect_atoms_dcnn return).
        pixel_size_nm: nm per pixel (square). If None, distances are in pixels
            and zoom sizes are taken from `zoom_px` instead of `zoom_nm`.
        heatmap: optional 2D DCNN probability map (same HxW as image). When
            given, `heatmap_hit_fraction` reports how many detections sit on a
            heatmap peak — detections off every peak are likely spurious.
        params: optional tunable dict (robust defaults):
            n_zoom (default 4): number of zoom-in panels.
            zoom_nm (4.0) / zoom_px (200): side length of each zoom crop.
            short_frac (0.5): a pair is "anomalously short" (duplicate/split
                suspect) if its NN distance < short_frac x median NN. LOWER to
                only flag near-coincident marks, RAISE to be stricter about
                over-detection.
            gap_grid (8): coarse NxN grid for the coverage-gap metric.
            duplicate_flag_frac (0.04): short_pair_fraction above this sets the
                `duplicate_suspect` flag.
            heatmap_hit_radius_nm (0.08) / _px (4): a detection counts as "on a
                peak" within this radius of a heatmap local maximum.
            a_nm: optional lattice constant for the ORDER-OF-MAGNITUDE count
                sanity only (never a pass/fail).

    Returns: dict with 'figure_bytes' (PNG of the composite QC figure),
        'metrics' (dict, see module docstring), and 'flags' (list of strings).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = params or {}
    n_zoom = int(p.get("n_zoom", 4))
    short_frac = float(p.get("short_frac", 0.5))
    gap_grid = int(p.get("gap_grid", 8))
    dup_flag_frac = float(p.get("duplicate_flag_frac", 0.04))

    img = _to_gray(image)
    H, W = img.shape
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)

    in_nm = pixel_size_nm is not None and pixel_size_nm > 0
    unit = "nm" if in_nm else "px"
    scale = float(pixel_size_nm) if in_nm else 1.0          # nm per px (or 1)
    zoom_px = int(round((p.get("zoom_nm", 4.0) / scale) if in_nm
                        else p.get("zoom_px", 200)))
    zoom_px = max(24, min(zoom_px, min(H, W)))

    # ----- nearest-neighbor distribution (prior-free) ------------------------
    nn_px, _ = _nn_distances(pos)
    nn = nn_px * scale
    metrics = {"n_detected": int(n), "units": unit}
    flags = []
    if n >= 2:
        med = float(np.median(nn))
        metrics["nn_median"] = round(med, 4)
        metrics["nn_cv"] = round(float(np.std(nn) / (np.mean(nn) + 1e-12)), 3)
        short_cut = short_frac * med
        short_mask = nn_px < (short_frac * np.median(nn_px))
        metrics["short_pair_fraction"] = round(float(short_mask.mean()), 4)
        metrics["short_cut"] = round(short_cut, 4)
        if metrics["short_pair_fraction"] > dup_flag_frac:
            flags.append("duplicate_suspect")
    else:
        med = float(min(H, W)) * scale
        short_mask = np.zeros(n, bool)

    # ----- coverage-gap metric on a coarse grid ------------------------------
    gx = np.clip((pos[:, 0] / W * gap_grid).astype(int), 0, gap_grid - 1)
    gy = np.clip((pos[:, 1] / H * gap_grid).astype(int), 0, gap_grid - 1)
    occ = np.zeros((gap_grid, gap_grid), int)
    for x, y in zip(gx, gy):
        occ[y, x] += 1
    gap_frac = float((occ == 0).mean())
    metrics["coverage_gap_fraction"] = round(gap_frac, 3)
    if gap_frac > 0.06:
        flags.append("coverage_gap_suspect")

    # ----- heatmap-hit fraction (DCNN only) ----------------------------------
    hm_max = None
    if heatmap is not None and n > 0:
        hm = _to_gray(heatmap)
        if hm.shape == (H, W):
            from skimage.feature import peak_local_max
            r = int(round((p.get("heatmap_hit_radius_nm", 0.08) / scale) if in_nm
                          else p.get("heatmap_hit_radius_px", 4)))
            hm_max = peak_local_max(hm, min_distance=max(2, r),
                                    threshold_rel=0.2)
            if len(hm_max):
                from scipy.spatial import cKDTree
                d, _ = cKDTree(hm_max[:, ::-1]).query(pos)   # hm_max is (row,col)
                metrics["heatmap_hit_fraction"] = round(float((d <= max(2, r)).mean()), 3)

    # ----- order-of-magnitude count sanity (labeled, never pass/fail) --------
    a_nm = p.get("a_nm")
    if a_nm and in_nm:
        area = (H * scale) * (W * scale)
        oom = n / (area / (float(a_nm) ** 2) + 1e-9)
        metrics["count_sanity_ratio"] = round(float(oom), 2)
        metrics["count_sanity_note"] = ("order-of-magnitude only (resolved "
            "columns/cell unknown a priori); ~0.5-2x is consistent, NOT "
            "over-detection")

    # ----- choose TARGETED zoom centers --------------------------------------
    centers = []
    if n:
        # 1) densest short-pair cell -> likely duplicates / over-detection
        if short_mask.any():
            sc = np.zeros((gap_grid, gap_grid), int)
            for x, y, s in zip(gx, gy, short_mask):
                if s:
                    sc[y, x] += 1
            yy, xx = np.unravel_index(int(sc.argmax()), sc.shape)
            centers.append(((xx + 0.5) / gap_grid * W, (yy + 0.5) / gap_grid * H,
                            "most short-NN pairs (over-detection check)"))
        # 2) largest coverage gap -> likely misses
        if (occ == 0).any():
            # pick the empty cell whose neighbourhood is otherwise dense
            empties = np.argwhere(occ == 0)
            yy, xx = empties[len(empties) // 2]
            centers.append(((xx + 0.5) / gap_grid * W, (yy + 0.5) / gap_grid * H,
                            "coverage gap (missed-column check)"))
        # 3) fill the rest evenly across the field
        k = max(0, n_zoom - len(centers))
        for j in range(k):
            fx = (j + 0.5) / max(k, 1)
            centers.append((fx * W, (0.5 if j % 2 == 0 else 0.78) * H, "field sample"))
    centers = centers[:n_zoom]

    # ----- compose the figure ------------------------------------------------
    nz = len(centers)
    ncols = 3
    nrows = 1 + int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    vlo, vhi = np.percentile(img, [1, 99])
    # full-frame overlay
    ax = axes[0, 0]; ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi)
    if n:
        ax.scatter(pos[:, 0], pos[:, 1], s=2, c="r", alpha=0.5, linewidths=0)
    ax.set_title(f"Full frame overlay (n={n})\nglobal coverage only — see zooms"); ax.axis("off")
    # heatmap or blank
    ax = axes[0, 1]
    if heatmap is not None:
        ax.imshow(_to_gray(heatmap), cmap="inferno"); ax.set_title("DCNN heatmap")
    else:
        ax.set_title("(no heatmap)")
    ax.axis("off")
    # NN histogram
    ax = axes[0, 2]; ax.axis("on")
    if n >= 2:
        spread = float(np.ptp(nn))
        if spread > 1e-9:
            ax.hist(nn, bins=min(40, max(5, n // 20)), range=(0, max(nn) * 1.05),
                    color="steelblue", edgecolor="k", linewidth=0.3)
        else:                                  # perfectly periodic (synthetic)
            ax.axvspan(med * 0.99, med * 1.01, color="steelblue", alpha=0.6)
        ax.axvline(med, color="k", ls="--", lw=1, label=f"median {med:.3f}")
        ax.axvline(short_frac * med, color="r", ls=":", lw=1.2,
                   label=f"short cut ({metrics.get('short_pair_fraction',0)*100:.1f}% below)")
        ax.set_xlabel(f"NN distance ({unit})"); ax.set_ylabel("count")
        ax.legend(fontsize=7); ax.set_title("NN-distance distribution")
    else:
        ax.set_title("NN distribution (n<2)")

    # zoom panels
    for idx, (cx, cy, label) in enumerate(centers):
        ax = axes[1 + idx // ncols, idx % ncols]; ax.axis("off")
        x0 = int(np.clip(cx - zoom_px / 2, 0, W - zoom_px))
        y0 = int(np.clip(cy - zoom_px / 2, 0, H - zoom_px))
        crop = img[y0:y0 + zoom_px, x0:x0 + zoom_px]
        clo, chi = np.percentile(crop, [1, 99])
        ax.imshow(crop, cmap="gray", vmin=clo, vmax=chi)
        m = ((pos[:, 0] >= x0) & (pos[:, 0] < x0 + zoom_px) &
             (pos[:, 1] >= y0) & (pos[:, 1] < y0 + zoom_px))
        if m.any():
            ax.scatter(pos[m, 0] - x0, pos[m, 1] - y0, s=28,
                       facecolors="none", edgecolors="r", linewidths=0.9)
        side = zoom_px * scale
        ax.set_title(f"zoom: {label}\n{side:.1f} {unit}, {int(m.sum())} marks", fontsize=8)
        ax.axis("off")

    metrics["flags"] = flags
    metrics["zoom_centers"] = [(round(c[0], 1), round(c[1], 1)) for c in centers]
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130); plt.close(fig)
    return {"figure_bytes": buf.getvalue(), "metrics": metrics, "flags": flags}


from scilink.skills._shared._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="detection_quality_panels",
    description=(
        "Quality-control a DENSE atom-column detection without counting: builds "
        "a composite figure (full-frame overlay for global coverage + TARGETED "
        "zoom-in overlays at the most-suspect regions + the nearest-neighbor "
        "distance histogram) and returns prior-free metrics. Use this to judge "
        "over/under-detection instead of an absolute count or an a-priori "
        "'expected count' (which is unreliable for multi-sublattice materials). "
        "A full-frame overlay of thousands of marks is unjudgeable; the zoom "
        "panels and the NN distribution are what actually reveal duplicates and "
        "misses."
    ),
    import_line=("from scilink.skills.image_analysis.atomic_stem.detection_qc "
                 "import detection_quality_panels"),
    signature=("detection_quality_panels(image, positions, pixel_size_nm=None, "
               "heatmap=None, params=None) -> dict"),
    agents=["image_analysis"],
    when_to_use=(
        "After ANY atom-column detection on a dense lattice (detect_atoms / "
        "detect_atoms_dcnn), call this to produce the QC visualization and the "
        "detection-quality metrics, and SAVE its `figure_bytes` as the step's "
        "visualization so the verifier sees the zoom panels (not a thousand-dot "
        "haze). Pass the DCNN `heatmap` when available for the heatmap-hit "
        "check. Judge over-detection from `short_pair_fraction` (a spike of "
        "anomalously short NN distances = duplicate/split marks) and the zoom "
        "panels, and under-detection from `coverage_gap_fraction` and the gap "
        "zoom — NOT from the absolute count or a count-vs-expected ratio. A "
        "clean unimodal NN distribution with correct zoom panels is a good "
        "detection regardless of how many columns were found."
    ),
    parameters={
        "image": {"type": "ndarray", "description": "2D grayscale detector image the positions came from."},
        "positions": {"type": "ndarray", "description": "(N,2) detected coordinates as (x, y) in image pixels."},
        "pixel_size_nm": {"type": "float", "description": "nm/px (square). If omitted, metrics/zooms are in pixels."},
        "heatmap": {"type": "ndarray", "description": "Optional DCNN probability map (HxW) for the heatmap-hit metric."},
        "params": {"type": "dict", "description": (
            "Optional knobs: n_zoom (4); zoom_nm (4.0) / zoom_px (200) — zoom "
            "crop side; short_frac (0.5) — NN below short_frac x median is a "
            "duplicate suspect, LOWER to flag only near-coincident marks, RAISE "
            "to be stricter; duplicate_flag_frac (0.04) — short_pair_fraction "
            "above this sets duplicate_suspect; gap_grid (8); "
            "heatmap_hit_radius_nm (0.08); a_nm — lattice constant for an "
            "order-of-magnitude count sanity only.")},
    },
    required=["image", "positions"],
    returns=(
        "dict with 'figure_bytes' (PNG of the QC composite — SAVE as the step "
        "visualization), 'metrics' (n_detected, nn_median, nn_cv, "
        "short_pair_fraction [over-detection signal], coverage_gap_fraction "
        "[under-detection signal], heatmap_hit_fraction [if heatmap given], "
        "count_sanity_ratio [order-of-magnitude only], zoom_centers), and "
        "'flags' (e.g. 'duplicate_suspect', 'coverage_gap_suspect'; empty = no "
        "detection-quality problem found)."
    ),
    example=(
        "det = detect_atoms_dcnn(image, fov_nm=18.0)\n"
        "qc = detection_quality_panels(image, det['positions'], "
        "pixel_size_nm=0.035, heatmap=det['heatmap'])\n"
        "open('visualization.png','wb').write(qc['figure_bytes'])\n"
        "print(qc['metrics'])   # judge from short_pair_fraction / gaps, not the count"
    ),
)
