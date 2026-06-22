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
  * amplitude/coherence drop with neither → check the real-space LATERAL SHIFT
    across the band: a fractional-period translational offset = a stacking fault
    / antiphase boundary (the power spectrum is translation-invariant, so this is
    the only signal that separates it from a non-translational disorder band or a
    scan/contrast artifact, which show ~zero shift).

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


def _lateral_shift(gray, cy, cx, horizontal, win):
    """Translational (lattice-phase) offset across a boundary, in REAL space.

    The window power spectrum is translation-invariant, so a stacking fault /
    antiphase boundary — which shifts the lattice laterally WITHOUT changing its
    orientation or spacing — is invisible to the spectral dissimilarity. This
    recovers exactly that: it cross-correlates a strip of the raw image on each
    side of the boundary (offset past the disordered core) ALONG the boundary
    direction, within +/- half a lattice period (so the offset is unambiguous),
    and returns the shift as a fraction of the local period. ~0 = no shift (a
    scan/contrast artifact or pure amorphization); a fractional-period shift =
    a genuine translational planar fault. Returns None if not measurable.
    """
    H, W = gray.shape
    t = max(6, int(round(win * 0.45)))      # strip thickness (~a couple of periods)
    g = max(3, int(round(win * 0.25)))      # gap skipping the disordered band core
    if horizontal:
        y = int(round(cy))
        a0, a1, b0, b1 = max(0, y - g - t), max(0, y - g), min(H, y + g), min(H, y + g + t)
        if a1 - a0 < 4 or b1 - b0 < 4:
            return None
        pa, pb = gray[a0:a1].mean(0), gray[b0:b1].mean(0)
    else:
        x = int(round(cx))
        a0, a1, b0, b1 = max(0, x - g - t), max(0, x - g), min(W, x + g), min(W, x + g + t)
        if a1 - a0 < 4 or b1 - b0 < 4:
            return None
        pa, pb = gray[:, a0:a1].mean(1), gray[:, b0:b1].mean(1)
    pa = pa - pa.mean(); pb = pb - pb.mean()
    n = len(pa)
    if n < 16 or pa.std() < 1e-6 or pb.std() < 1e-6:
        return None
    ac = np.correlate(pa, pa, "full")[n - 1:]          # local period from autocorr
    lo = 3
    if len(ac) <= lo + 2:
        return None
    period = int(np.argmax(ac[lo:]) + lo)
    if period < 4:
        return None
    half = max(1, period // 2)
    cc = np.correlate(pa, pb, "full"); mid = n - 1
    seg = cc[mid - half: mid + half + 1]
    lag = int(np.arange(-half, half + 1)[int(np.argmax(seg))])
    return {"lateral_shift_px": float(abs(lag)),
            "lateral_shift_frac": round(abs(lag) / period, 3),
            "parallel_period_px": float(period)}


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
            dissim_sigma (2.0): robustness multiplier (median + sigma*MAD) for
                the statistical part of the boundary threshold. LOWER (e.g. 1.5)
                to recover a subtle/coherent twin the default misses, RAISE
                (e.g. 4.0) if noise is being flagged as a boundary.
            orient_jump_deg (8.0) / spacing_jump_frac (0.05): CLASSIFICATION
                thresholds — orientation change above the former → grain/twin,
                spacing change above the latter → interface/second-phase.
            min_boundary_frac (0.03): below this boundary-cell fraction the field
                is reported as a single crystal.
            min_region_frac (0.01): minimum connected-component size (as a
                fraction of the window grid) a boundary locus must reach to
                count — speckle below this is dropped as false-positive
                contrast. RAISE to suppress scattered patches on a noisy image,
                LOWER to keep a short/faint boundary segment.
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
    min_region_frac = float(p.get("min_region_frac", 0.01))
    nb = int(p.get("azimuthal_bins", 36))
    smooth = float(p.get("smooth", 0.6))
    dissim_sigma = float(p.get("dissim_sigma", 2.0))
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
            # DISPLAY orientation: circular MEAN of the azimuthal profile (mod
            # 180), not argmax — argmax flips between symmetry-equivalent /
            # different reflection families window-to-window and renders the map
            # as a meaningless checkerboard.
            phi = (np.arange(nb) + 0.5) / nb * np.pi
            theta[iy, ix] = (np.degrees(0.5 * np.arctan2((az * np.sin(2 * phi)).sum(),
                                                         (az * np.cos(2 * phi)).sum())) % 180.0)
            rad = np.bincount(rbin[band], weights=P[band], minlength=win + 1)[:nr + 1]
            radp[iy, ix] = rad
            # DISPLAY spacing: power-weighted radial CENTROID (stable), not the
            # argmax bin (which also flips between {100}/{110}-type rings).
            ri = np.arange(1, nr + 1)
            dmap[iy, ix] = win / max((ri * rad[1:]).sum() / (rad[1:].sum() + 1e-12), 1.0)

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

    # A real boundary is a connected locus; isolated speckle that clears the
    # threshold on a noisy image is false-positive contrast, not a boundary.
    # Drop sub-size connected components BEFORE measuring/displaying so the
    # fraction and the overlay map reflect clean contours, not scattered patches.
    min_region = max(2, int(gh * gw * min_region_frac))
    lbl0, n0 = ndi.label(bmask)
    if n0:
        sizes = ndi.sum(np.ones_like(lbl0, float), lbl0, np.arange(1, n0 + 1))
        keep = np.where(sizes >= min_region)[0] + 1
        bmask = np.isin(lbl0, keep) if keep.size else np.zeros_like(bmask)
    boundary_fraction = float(bmask.mean())
    score = diss

    # --- classify each boundary region by what changes across it ------------
    lbl, nlab = ndi.label(bmask)
    boundaries = []
    for k in range(1, nlab + 1):
        cell = lbl == k
        if cell.sum() < min_region:
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
        cyc, cxc = ndi.center_of_mass(cell)
        cyp = float(cyc) * step + win / 2
        cxp = float(cxc) * step + win / 2
        ls = None
        if odeg >= orient_jump and odeg >= sfrac * 100:
            btype = "grain/twin boundary (orientation change)"
        elif sfrac >= spacing_jump:
            btype = "interface / second phase (spacing change)"
        else:
            # No orientation or spacing change: a coherence drop. This is the
            # ambiguous class — a real translational planar fault (stacking
            # fault / antiphase boundary) looks identical to a scan/contrast
            # artifact or amorphization in the translation-invariant spectrum.
            # Disambiguate in REAL space: a lateral lattice-phase shift across
            # the band is the signature of a translational fault (an artifact
            # shifts nothing). The boundary runs perpendicular to its long axis,
            # so a "vertical" region (tall) is a vertical boundary -> measure the
            # shift across x; a horizontal band -> across... measure along the
            # boundary direction, which is the long axis (horizontal here).
            ls = _lateral_shift(img, cyp, cxp, horizontal=not vertical, win=win)
            lat = ls["lateral_shift_frac"] if ls else None
            if lat is not None and lat >= 0.12:
                btype = ("stacking fault / antiphase boundary "
                         "(lateral lattice shift, no orientation/spacing change)")
            else:
                btype = "planar fault / disorder band (coherence drop)"
        bd = {
            "type": btype,
            "centroid_px": (round(cxp, 1), round(cyp, 1)),
            "orient_change_deg": round(odeg, 2),
            "spacing_change_pct": round(sfrac * 100, 2),
            "dissimilarity": round(float(diss[cell].mean()), 3),
            "n_cells": int(cell.sum()),
        }
        if ls:
            bd["lateral_shift_frac"] = ls["lateral_shift_frac"]
            bd["lateral_shift_px"] = round(ls["lateral_shift_px"], 1)
        boundaries.append(bd)
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
        "chemical, not structural, boundary). A layer-parallel planar fault "
        "(stacking fault / antiphase boundary / intergrowth) shifts the lattice "
        "TRANSLATIONALLY with no orientation or spacing change — invisible to the "
        "(translation-invariant) power spectrum on its own, but the tool catches "
        "it via a real-space lateral-shift check on every coherence-drop band and "
        "reports lateral_shift_frac (so a horizontal/layer-parallel coherence "
        "band with a fractional-period shift is a real stacking fault, NOT a scan "
        "artifact — do not dismiss it as one). The tool detects boundaries that "
        "change local orientation, spacing, or coherence; for very large or "
        "subtle-misorientation images tune window_nm and dissim_floor."
    ),
    parameters={
        "image": {"type": "ndarray", "description": "2D grayscale atomic-resolution / lattice-fringe image."},
        "pixel_size_nm": {"type": "float", "description": "nm/px (square). If omitted, spacings are in pixels."},
        "params": {"type": "dict", "description": (
            "Optional knobs (robust defaults; tune them): window_px (64) / "
            "window_nm — window side (must span a few periods; smaller = tighter "
            "localization, worse orientation/spacing resolution); overlap (0.5); "
            "dissim_floor (0.05) and dissim_sigma (2.0) — DETECTION threshold on "
            "spectral dissimilarity, LOWER either (dissim_sigma→1.5) to recover a "
            "subtle/coherent twin, RAISE if noise is flagged; "
            "orient_jump_deg (8.0) and spacing_jump_frac (0.05) — CLASSIFICATION "
            "thresholds; min_boundary_frac (0.03) — below this it is a single "
            "crystal; min_region_frac (0.01) — min connected boundary size, "
            "RAISE to suppress scattered false-positive patches on a noisy "
            "image; azimuthal_bins (36); smooth (0.6).")},
    },
    required=["image"],
    returns=(
        "dict with 'figure_bytes' (PNG: raw, orientation map, spacing map, "
        "boundary overlay — SAVE as the visualization), 'metrics' "
        "(boundary_fraction; n_boundaries; dominant_type; 'boundaries' = list of "
        "{type [grain/twin | interface/second-phase | stacking-fault/antiphase | "
        "planar-fault/disorder], centroid_px, orient_change_deg, "
        "spacing_change_pct, dissimilarity, n_cells, and — for coherence-drop "
        "bands — lateral_shift_frac (translational lattice-phase offset across "
        "the band as a fraction of the period: ~0 = a non-translational disorder "
        "band / scan-or-contrast artifact; a fractional-period shift, e.g. ~0.5, "
        "= a genuine stacking fault / antiphase boundary)}; "
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
