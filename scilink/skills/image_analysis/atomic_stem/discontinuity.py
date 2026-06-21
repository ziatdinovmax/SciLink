"""Localize and classify STRUCTURAL lattice discontinuities in an
atomic-resolution image — the single front door for planar defects, grain /
twin boundaries, and incoherent interfaces.

All three are the same thing physically: a line across which the local lattice
changes its orientation, spacing, or coherence. A sliding-window local FFT
turns the image into per-region maps of dominant orientation, spacing, and
peak amplitude; a boundary is where those maps jump.

  * AMPLITUDE / coherence drop is the UNIVERSAL detector — any boundary
    disrupts the local periodicity, so the local FFT peak weakens right at it.
  * a jump in ORIENTATION classifies it as a grain / twin boundary;
  * a jump in SPACING classifies it as an interface / second phase / intergrowth;
  * amplitude drop with neither → a stacking fault / antiphase band / amorphous
    region.

Relationship to the other tools:
  * run_fft_nmf_analysis is the EXPLORATORY decomposer ("how many domains exist")
    — this tool is the sharp LOCALIZER that returns the boundary line plus a
    physical readout (delta-orientation in deg, delta-spacing in %). Same
    explore-vs-localize split as FFT-NMF vs fourier_reflection_map.
  * gpa_strain measures the continuous strain tensor against ONE reference
    lattice — ideal for small distortions and coherent interfaces, but it
    breaks at large misorientation (two grains = two reciprocal lattices). This
    tool handles large misorientation (each window finds its own orientation).

SCOPE: structural discontinuities only. A COHERENT, lattice-matched chemical
interface (e.g. a perovskite film on a perovskite substrate) has the same
orientation, spacing, and coherence on both sides — it is invisible here; that
is a Z-contrast boundary (see the atomic_stem interface guidance).
"""

import io

import numpy as np


def _to_gray(a):
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a


def lattice_discontinuity_map(image, pixel_size_nm=None, params=None):
    """Map and classify structural lattice discontinuities (boundaries).

    Args:
        image: 2D grayscale (or HxWx3) atomic-resolution / lattice-fringe image.
        pixel_size_nm: nm per pixel (square). If None, spacings are in pixels.
        params: optional tunable dict (robust defaults):
            window_px (64) / window_nm: sliding-window side; must span a few
                lattice periods. SMALLER localizes a boundary more tightly but
                resolves spacing/orientation worse; LARGER is the reverse.
            overlap (0.5): fractional window overlap (grid step).
            dissim_floor (0.05): absolute spectral-dissimilarity a boundary cell
                must clear above the uniform-crystal baseline. LOWER to catch a
                faint/subtle boundary, RAISE if noise is flagged.
            dissim_sigma (4.0): robustness multiplier (median + sigma*MAD) for
                the statistical part of the boundary threshold.
            orient_jump_deg (8.0) / spacing_jump_frac (0.05): CLASSIFICATION
                thresholds — orientation change above the former → grain/twin,
                spacing change above the latter → interface/second-phase.
            min_boundary_frac (0.03): below this boundary-cell fraction the field
                is reported as a single crystal.
            azimuthal_bins (36): angular resolution of the orientation profile.
            smooth (0.6): Gaussian smoothing of the dissimilarity map.

    Returns: dict with 'figure_bytes' (PNG: orientation map, spacing map,
        boundary overlay), 'metrics', and 'flags'. Key metrics: boundary_fraction,
        n_boundaries, dominant_type, boundaries (list of {type, centroid_px,
        orient_change_deg, spacing_change_pct, dissimilarity}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = params or {}
    overlap = float(p.get("overlap", 0.5))
    orient_jump = float(p.get("orient_jump_deg", 8.0))
    spacing_jump = float(p.get("spacing_jump_frac", 0.05))
    min_bf = float(p.get("min_boundary_frac", 0.03))
    nb = int(p.get("azimuthal_bins", 36))
    smooth = float(p.get("smooth", 0.6))
    dissim_sigma = float(p.get("dissim_sigma", 4.0))
    abs_floor = float(p.get("dissim_floor", 0.05))

    img = _to_gray(image)
    H, W = img.shape
    in_nm = pixel_size_nm is not None and pixel_size_nm > 0
    scale = float(pixel_size_nm) if in_nm else 1.0
    unit = "nm" if in_nm else "px"
    # Window side: a physical ~window_nm (default 2 nm) spans several periods of
    # a typical atomic lattice AND keeps the fundamental inside the analysis
    # band — more robust than a fixed pixel count, which spans too few periods
    # at fine pixel sizes. Clamped so the grid stays reasonably fine. For hard
    # cases (subtle twin, very fine/coarse px) tune window_nm / window_px.
    if "window_px" in p:
        win = int(p["window_px"])
    elif in_nm:
        win = int(round(float(p.get("window_nm", 2.0)) / scale))
    else:
        win = int(p.get("window_px", 64))
    win = int(np.clip(win, 40, min(H, W) // 6))
    step = max(6, int(win * (1 - overlap)))

    ys = list(range(0, H - win + 1, step))
    xs = list(range(0, W - win + 1, step))
    gh, gw = len(ys), len(xs)

    # Per-window POWER SPECTRUM (translation-invariant): within one crystal,
    # neighbouring windows have near-identical spectra; ANY boundary
    # (orientation, spacing, or structural change) makes them differ. This is
    # robust where peak-picking is not (the strongest reflection flips between
    # symmetry-equivalent / different families window-to-window).
    cy0, cx0 = win // 2, win // 2
    yy, xx = np.mgrid[0:win, 0:win]
    rr = np.hypot(yy - cy0, xx - cx0)
    ang = (np.arctan2(yy - cy0, xx - cx0) % np.pi)     # mod pi
    band = (rr > max(2, 0.04 * win)) & (rr < 0.47 * win)   # lattice-reflection band (incl. fundamental)
    abin = np.clip((ang / np.pi * nb).astype(int), 0, nb - 1)
    rbin = np.clip(rr.astype(int), 0, win)
    hann = np.outer(np.hanning(win), np.hanning(win))

    nr = win // 2
    specvec = {}                                        # (iy,ix) -> band power vector
    azim = np.zeros((gh, gw, nb))                       # azimuthal profile (orientation)
    radp = np.zeros((gh, gw, nr + 1))                  # radial profile (spacing)
    theta = np.full((gh, gw), np.nan)                   # display orientation (deg)
    dmap = np.full((gh, gw), np.nan)                    # display spacing (px)
    bidx = np.where(band.ravel())[0]
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            w = img[y0:y0 + win, x0:x0 + win]
            if w.std() < 1e-6:
                continue
            P = np.abs(np.fft.fftshift(np.fft.fft2((w - w.mean()) * hann))) ** 2
            v = P.ravel()[bidx]
            specvec[(iy, ix)] = v / (v.sum() + 1e-12)
            az = np.bincount(abin[band], weights=P[band], minlength=nb)
            azim[iy, ix] = az
            theta[iy, ix] = (np.argmax(az) + 0.5) / nb * 180.0
            rad = np.bincount(rbin[band], weights=P[band], minlength=win + 1)[:nr + 1]
            radp[iy, ix] = rad
            rpk = np.argmax(rad[1:]) + 1
            dmap[iy, ix] = win / max(rpk, 1)

    valid = np.array([[(iy, ix) in specvec for ix in range(gw)] for iy in range(gh)])
    if valid.sum() < 4:
        return {"metrics": {"note": "too few resolvable windows; not a clear "
                            "lattice (or window too small)", "boundary_fraction": 0.0},
                "flags": ["unresolved"], "figure_bytes": b""}

    # --- neighbour spectral dissimilarity = boundary detection signal -------
    from scipy.ndimage import gaussian_filter
    from scipy import ndimage as ndi
    diss = np.zeros((gh, gw))
    for iy in range(gh):
        for ix in range(gw):
            if (iy, ix) not in specvec:
                diss[iy, ix] = 1.0; continue
            v = specvec[(iy, ix)]; ds = []
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb_ = (iy + dy, ix + dx)
                if nb_ in specvec:
                    u = specvec[nb_]
                    ds.append(1.0 - float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
            diss[iy, ix] = np.mean(ds) if ds else 1.0
    diss = gaussian_filter(diss, smooth)

    # boundary cell: spectral dissimilarity clearly above the uniform-crystal
    # baseline. Robust statistical cut AND an absolute floor (a uniform crystal
    # has tiny, uniform dissimilarity → nothing clears the floor → bf~0).
    med = float(np.median(diss)); mad = float(np.median(np.abs(diss - med))) + 1e-9
    thr = max(med + dissim_sigma * 1.4826 * mad, abs_floor)
    bmask = diss > thr
    boundary_fraction = float(bmask.mean())
    score = diss

    # --- classify each boundary region by what changes across it ------------
    lbl, nlab = ndi.label(bmask)
    boundaries = []
    for k in range(1, nlab + 1):
        cell = lbl == k
        if cell.sum() < max(2, gh * gw * 0.01):
            continue
        # classify by comparing the FULL azimuthal/radial profiles on the two
        # sides of the boundary (robust; per-cell argmax flips between symmetry-
        # equivalent reflections and would fake an orientation change).
        rows = np.where(cell.any(1))[0]; cols = np.where(cell.any(0))[0]
        vertical = rows.size >= cols.size                 # tall region = vertical boundary
        cc = ndi.center_of_mass(cell)
        if vertical:
            a = valid & (np.arange(gw)[None, :] < cc[1] - 0.5)
            b = valid & (np.arange(gw)[None, :] > cc[1] + 0.5)
        else:
            a = valid & (np.arange(gh)[:, None] < cc[0] - 0.5)
            b = valid & (np.arange(gh)[:, None] > cc[0] + 0.5)
        if a.sum() < 2 or b.sum() < 2:
            a, b = cell, cell
        azA = azim[a].mean(0); azB = azim[b].mean(0)
        azA /= azA.sum() + 1e-12; azB /= azB.sum() + 1e-12
        # circular cross-correlation peak shift = orientation change (deg, mod 180)
        xcorr = np.array([np.dot(azA, np.roll(azB, k)) for k in range(nb)])
        shift = int(np.argmax(xcorr)); shift = min(shift, nb - shift)
        odeg = float(shift / nb * 180.0)
        rA = radp[a].mean(0); rB = radp[b].mean(0)
        pkA = np.argmax(rA[1:]) + 1; pkB = np.argmax(rB[1:]) + 1
        dA, dB = win / max(pkA, 1), win / max(pkB, 1)
        sfrac = float(abs(dA - dB) / ((dA + dB) / 2 + 1e-9))
        if odeg >= orient_jump and odeg >= sfrac * 100:
            btype = "grain/twin boundary (orientation change)"
        elif sfrac >= spacing_jump:
            btype = "interface / second phase (spacing change)"
        else:
            btype = "planar fault / disorder band (coherence drop)"
        cyc, cxc = ndi.center_of_mass(cell)
        boundaries.append({
            "type": btype,
            "centroid_px": (round(float(cxc) * step + win / 2, 1),
                            round(float(cyc) * step + win / 2, 1)),
            "orient_change_deg": round(odeg, 2),
            "spacing_change_pct": round(sfrac * 100, 2),
            "dissimilarity": round(float(diss[cell].mean()), 3),
            "n_cells": int(cell.sum()),
        })
    boundaries.sort(key=lambda b: -b["n_cells"])

    has_boundary = boundary_fraction >= min_bf and boundaries
    dominant = boundaries[0]["type"] if has_boundary else None
    flags = ["boundary_detected"] if has_boundary else ["single_crystal"]

    note = (f"{len(boundaries)} boundary region(s); dominant: {dominant}"
            if has_boundary else
            "No structural discontinuity resolved — single crystal / uniform "
            "lattice (a coherent lattice-matched chemical interface would be "
            "invisible here; use the Z-contrast interface route).")

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    vlo, vhi = np.percentile(img, [1, 99])
    ext = [0, W, H, 0]
    ax[0, 0].imshow(img, cmap="gray", vmin=vlo, vmax=vhi)
    ax[0, 0].set_title("Raw image"); ax[0, 0].axis("off")
    om = ax[0, 1].imshow(np.where(valid, theta, np.nan), cmap="hsv",
                         extent=ext, vmin=0, vmax=180)
    ax[0, 1].set_title("Local orientation (deg, mod 180)"); ax[0, 1].axis("off")
    plt.colorbar(om, ax=ax[0, 1], fraction=0.046)
    dm = ax[1, 0].imshow(np.where(valid, dmap, np.nan) * scale, cmap="viridis", extent=ext)
    ax[1, 0].set_title(f"Local spacing ({unit})"); ax[1, 0].axis("off")
    plt.colorbar(dm, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].imshow(img, cmap="gray", vmin=vlo, vmax=vhi)
    ax[1, 1].imshow(np.ma.masked_where(~bmask, score), cmap="autumn",
                    alpha=0.6, extent=ext)
    for b in boundaries:
        ax[1, 1].plot(*b["centroid_px"], "c+", ms=14, mew=2)
    ax[1, 1].set_title(f"Boundary overlay (frac={boundary_fraction:.2f})\n{note[:60]}")
    ax[1, 1].axis("off")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=120); plt.close(fig)

    metrics = {
        "boundary_fraction": round(boundary_fraction, 3),
        "n_boundaries": len(boundaries),
        "dominant_type": dominant,
        "boundaries": boundaries,
        "window_px": win, "grid_shape": (gh, gw),
        "median_spacing": round(float(np.nanmedian(dmap) * scale), 4), "units": unit,
        "note": note,
    }
    return {"figure_bytes": buf.getvalue(), "metrics": metrics, "flags": flags}


from scilink.skills._shared._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="lattice_discontinuity_map",
    description=(
        "Localize and CLASSIFY structural lattice discontinuities — planar "
        "defects (stacking faults, intergrowths, antiphase bands), grain / twin "
        "boundaries, and incoherent interfaces — in one pass. A sliding-window "
        "local FFT maps dominant orientation, spacing, and peak coherence; a "
        "boundary is where they jump (coherence drop detects it; an orientation "
        "jump = grain/twin, a spacing jump = interface/second phase). Returns a "
        "boundary overlay + the change magnitudes."
    ),
    import_line=("from scilink.skills.image_analysis.atomic_stem.discontinuity "
                 "import lattice_discontinuity_map"),
    signature="lattice_discontinuity_map(image, pixel_size_nm=None, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Use to LOCATE and TYPE a boundary / planar defect / interface in a "
        "crystalline atomic-resolution or lattice-fringe image — objectives like "
        "'find the grain/twin boundary', 'locate the stacking fault / "
        "intergrowth', 'where is the interface'. Save its `figure_bytes` as the "
        "visualization and read `boundaries` (each typed with its "
        "orient_change_deg / spacing_change_pct / dissimilarity).\n"
        "\n"
        "Pick among the reciprocal-space tools: `run_fft_nmf_analysis` is the "
        "EXPLORATORY decomposer (unknown heterogeneity, 'how many domains') — "
        "this tool is the sharp LOCALIZER that returns the boundary line + a "
        "physical readout. `gpa_strain` measures the continuous strain tensor "
        "against one reference lattice (small distortions, coherent interfaces, "
        "fine same-orientation faults) but breaks at large misorientation; this "
        "tool handles large misorientation, so use GPA for strain MAGNITUDE and "
        "this tool to LOCATE/TYPE the boundary. `fourier_reflection_map` maps a "
        "specific superstructure/satellite reflection's domain.\n"
        "\n"
        "SCOPE: structural discontinuities only. A COHERENT lattice-matched "
        "chemical interface (perovskite film on perovskite substrate) has the "
        "same orientation/spacing/coherence on both sides and is INVISIBLE here "
        "— detect that via the per-column Z-contrast step instead (it is a "
        "chemical, not structural, boundary). Likewise a perfectly COHERENT twin "
        "whose mirror leaves the power spectrum unchanged shows no spectral "
        "change — find it via a real-space displacement map / gpa_strain. The "
        "tool detects boundaries that change local orientation, spacing, or "
        "coherence; for very large or subtle-misorientation images tune "
        "window_nm and dissim_floor."
    ),
    parameters={
        "image": {"type": "ndarray", "description": "2D grayscale atomic-resolution / lattice-fringe image."},
        "pixel_size_nm": {"type": "float", "description": "nm/px (square). If omitted, spacings are in pixels."},
        "params": {"type": "dict", "description": (
            "Optional knobs (robust defaults; tune them): window_px (64) / "
            "window_nm — window side (must span a few periods; smaller = tighter "
            "localization, worse orientation/spacing resolution); overlap (0.5); "
            "dissim_floor (0.05) and dissim_sigma (4.0) — DETECTION threshold on "
            "spectral dissimilarity, LOWER dissim_floor for a subtle boundary; "
            "orient_jump_deg (8.0) and spacing_jump_frac (0.05) — CLASSIFICATION "
            "thresholds; min_boundary_frac (0.03) — below this it is a single "
            "crystal; azimuthal_bins (36); smooth (0.6).")},
    },
    required=["image"],
    returns=(
        "dict with 'figure_bytes' (PNG: raw, orientation map, spacing map, "
        "boundary overlay — SAVE as the visualization), 'metrics' "
        "(boundary_fraction; n_boundaries; dominant_type; 'boundaries' = list of "
        "{type [grain/twin | interface/second-phase | planar-fault/disorder], "
        "centroid_px, orient_change_deg, spacing_change_pct, dissimilarity, n_cells}; "
        "median_spacing; note), and 'flags' ('boundary_detected' or "
        "'single_crystal'; 'unresolved' if no clear lattice)."
    ),
    example=(
        "res = lattice_discontinuity_map(image, pixel_size_nm=0.035)\n"
        "open('visualization.png','wb').write(res['figure_bytes'])\n"
        "for b in res['metrics']['boundaries']:\n"
        "    print(b['type'], b['centroid_px'], b['orient_change_deg'], b['spacing_change_pct'])"
    ),
)
