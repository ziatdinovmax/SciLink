"""Robust Geometric Phase Analysis (GPA) for atomic-resolution / lattice-fringe
images.

Self-contained (numpy + scipy only). Computes the in-plane strain tensor
(exx, eyy, exy) and rigid rotation (wxy) of a crystal lattice relative to an
UNDISTORTED REFERENCE REGION, from two non-collinear Bragg reflections.

Design goals (these are the things that make naive GPA give nonsense):
  * Strain is meaningless without a reference — the carrier g-vectors are
    refined from an auto-selected undistorted region so the reference strain is
    ~0 by construction (no global offset).
  * The two reflections are chosen to be a well-conditioned (near-orthogonal,
    similar-|g|) pair, so neither strain axis is ill-determined.
  * The Bragg mask radius is tied to |g| (sets spatial resolution); too wide
    leaks neighbouring spots / Fourier-filter striping, too narrow over-smooths.
  * A validity mask removes low-amplitude pixels (cores, vacuum, edges); we
    report valid_fraction and never average strain over invalid pixels.
  * Artifact guards flag (a) inputs that are already Fourier-filtered (where any
    derived strain/|b| is untrustworthy) and (b) strong directional striping.
  * A small repair ladder re-tries reference / reflections / mask when the valid
    fraction is too low, instead of silently returning a degenerate map.

Reference:
  M.J. Hytch, E. Snoeck, R. Kilaas, Ultramicroscopy 74 (1998) 131.

Main entry point: gpa_strain_map(image, ...) -> dict.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ._spec import ToolSpec

__all__ = ["gpa_strain_map", "find_bragg_peaks", "make_strained_lattice"]


# --------------------------------------------------------------------------- #
#  FFT / peak helpers
# --------------------------------------------------------------------------- #
def _fft(image):
    """Windowed FFT (for peak finding) and raw FFT (for filtering)."""
    H, W = image.shape
    img = image - image.mean()
    win = np.outer(np.hanning(H), np.hanning(W))
    Fwin = np.fft.fftshift(np.fft.fft2(img * win))
    Fraw = np.fft.fftshift(np.fft.fft2(img))
    return Fwin, Fraw


def find_bragg_peaks(image, n=6, rmin_frac=0.02, exclude_frac=0.45):
    """Find up to n strongest non-Friedel Bragg peaks.

    Returns list of (gx, gy) in pixels measured from the FFT centre (i.e. the
    spatial frequency in cycles-over-the-whole-image along x=cols, y=rows).
    rmin_frac : ignore peaks closer than rmin_frac*min(H,W) to DC (low-freq /
                tilt / illumination).
    exclude_frac : ignore peaks beyond exclude_frac*min(H,W) from DC (noise).
    """
    H, W = image.shape
    Fwin, _ = _fft(image)
    P = np.abs(Fwin)
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    rmin = max(3.0, rmin_frac * min(H, W))
    rmax = exclude_frac * min(H, W)
    Pm = P.copy()
    Pm[(r < rmin) | (r > rmax)] = 0
    peaks = []
    sup = max(3, int(rmin))                       # suppression radius around a found peak
    for _ in range(n * 4):
        idx = np.argmax(Pm)
        if Pm.flat[idx] <= 0:
            break
        py, px = np.unravel_index(idx, Pm.shape)
        gx, gy = px - cx, py - cy
        peaks.append((float(gx), float(gy), float(P[py, px])))
        # suppress this peak AND its Friedel mate
        Pm[(xx - px) ** 2 + (yy - py) ** 2 < sup ** 2] = 0
        Pm[(xx - (cx - gx)) ** 2 + (yy - (cy - gy)) ** 2 < sup ** 2] = 0
        if len(peaks) >= n:
            break
    return peaks  # (gx, gy, power)


def _select_pair(peaks):
    """From candidate peaks pick the best-conditioned strong pair: prefer
    near-orthogonal, similar-|g|, high-power. Returns ((gx,gy),(gx,gy))."""
    if len(peaks) < 2:
        raise ValueError("need >=2 Bragg peaks for GPA")
    mags = np.array([np.hypot(p[0], p[1]) for p in peaks])
    med = np.median(mags)
    # keep peaks near the dominant lattice magnitude (drop spurious low-freq)
    keep = [p for p, m in zip(peaks, mags) if 0.5 * med < m < 1.7 * med]
    if len(keep) < 2:
        keep = peaks
    best, best_score = None, -np.inf
    pw = np.array([p[2] for p in keep])
    pwn = pw / (pw.max() + 1e-9)
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            a, b = keep[i], keep[j]
            cross = abs(a[0] * b[1] - a[1] * b[0]) / (
                np.hypot(a[0], a[1]) * np.hypot(b[0], b[1]) + 1e-9)  # |sin| 0..1
            magbal = min(np.hypot(*a[:2]), np.hypot(*b[:2])) / (
                max(np.hypot(*a[:2]), np.hypot(*b[:2])) + 1e-9)
            power = 0.5 * (pwn[i] + pwn[j])
            score = cross * (0.5 + 0.5 * magbal) * (0.4 + 0.6 * power)
            if score > best_score:
                best_score, best = score, (a[:2], b[:2])
    return best


# --------------------------------------------------------------------------- #
#  Bragg-filtered complex image and its phase gradient
# --------------------------------------------------------------------------- #
def _bragg_filtered(Fraw, g, mask_sigma):
    """Inverse-FFT of the raw spectrum masked by a Gaussian around +g.
    g = (gx, gy) in pixels from centre. Returns complex H_g(r)."""
    H, W = Fraw.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.exp(-(((xx - (cx + g[0])) ** 2 + (yy - (cy + g[1])) ** 2)
                    / (2.0 * mask_sigma ** 2)))
    return np.fft.ifft2(np.fft.ifftshift(Fraw * mask)), np.abs


def _phase_gradient(Hc):
    """Per-pixel gradient of the (wrapped) phase of complex field Hc, computed
    wrapping-safe as Im(conj(Hc) dHc) / |Hc|^2. Returns (dP/dy, dP/dx)."""
    gy, gx = np.gradient(Hc)
    amp2 = (Hc.real ** 2 + Hc.imag ** 2) + 1e-12
    dpx = np.imag(np.conj(Hc) * gx) / amp2
    dpy = np.imag(np.conj(Hc) * gy) / amp2
    return dpy, dpx


# --------------------------------------------------------------------------- #
#  Reference-region selection
# --------------------------------------------------------------------------- #
def _auto_reference(amp_min, wavevec_var, frac=0.18):
    """Pick the reference window: high Bragg amplitude AND low local-wavevector
    variance (i.e. a uniform, well-resolved patch). Returns a boolean mask."""
    H, W = amp_min.shape
    win = max(8, int(frac * min(H, W)))
    # box-mean amplitude and box-variance of the wavevector-magnitude proxy
    amp_box = ndi.uniform_filter(amp_min, win)
    var_box = ndi.uniform_filter(wavevec_var, win)
    # normalise and score: want high amp, low var; avoid the outer border
    a = (amp_box - amp_box.min()) / (np.ptp(amp_box) + 1e-9)
    v = (var_box - var_box.min()) / (np.ptp(var_box) + 1e-9)
    score = a - 1.5 * v
    b = win // 2 + 1
    score[:b] = score[-b:] = -np.inf
    score[:, :b] = score[:, -b:] = -np.inf
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    ref = np.zeros((H, W), bool)
    h = win // 2
    ref[max(0, cy - h):cy + h, max(0, cx - h):cx + h] = True
    return ref, (int(cx - h), int(cy - h), int(2 * h), int(2 * h))  # mask, (x,y,w,h)


# --------------------------------------------------------------------------- #
#  Artifact guards
# --------------------------------------------------------------------------- #
def _artifact_flags(image, peaks):
    """Heuristics: (prefiltered, stripe_axis_deg_or_None, peak_to_bg)."""
    H, W = image.shape
    Fwin, _ = _fft(image)
    P = np.abs(Fwin) ** 2
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    rmin = max(3.0, 0.02 * min(H, W))
    annulus = (r > rmin) & (r < 0.45 * min(H, W))
    bg = np.median(P[annulus]) + 1e-12
    peak_pw = np.mean([p[2] ** 2 for p in peaks[:2]]) if peaks else 0.0
    peak_to_bg = peak_pw / bg
    # Fourier-filtered images have near-delta peaks on a strongly suppressed
    # diffuse background -> very high peak/background AND low "diffuse fraction".
    diffuse = P[annulus]
    # fraction of annulus power NOT in the brightest 1% bins (true diffuse halo)
    thr = np.percentile(diffuse, 99)
    diffuse_frac = diffuse[diffuse < thr].sum() / (diffuse.sum() + 1e-12)
    # Fourier-filtered images: near-delta Bragg peaks (peak/bg >~1e7) on a
    # strongly suppressed diffuse halo (diffuse_frac <~0.06). Real atomic images
    # sit far from BOTH (peak/bg <~1e6, diffuse_frac >~0.1), so require both.
    prefiltered = (peak_to_bg > 1e7) and (diffuse_frac < 0.06)
    # directional striping: compare power along the two cardinal high-freq wedges
    return dict(prefiltered=bool(prefiltered),
                peak_to_background=float(peak_to_bg),
                diffuse_fraction=float(diffuse_frac))


# --------------------------------------------------------------------------- #
#  Main GPA
# --------------------------------------------------------------------------- #
def _fit_plane(field, valid):
    """Least-squares affine (a + b*x + c*y) fit of `field` over `valid` pixels.
    Returns (slope_x, slope_y, ramp_fraction R^2, full-field plane array)."""
    H, W = field.shape
    ys, xs = np.where(valid)
    if xs.size < 20:
        return 0.0, 0.0, 0.0, np.zeros((H, W))
    z = field[valid]
    A = np.column_stack([np.ones(xs.size), xs.astype(float), ys.astype(float)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2)) + 1e-12
    r2 = max(0.0, min(1.0, 1 - ss_res / ss_tot))
    yy, xx = np.mgrid[0:H, 0:W]
    plane = coef[0] + coef[1] * xx + coef[2] * yy
    return float(coef[1]), float(coef[2]), float(r2), plane


def gpa_strain_map(image, reflections=None, reference_roi="auto",
                   mask_frac=0.25, amp_percentile=25.0, smooth=None,
                   pixel_size_nm=None, target_valid_fraction=0.5,
                   detrend=False, _repair_depth=0):
    """Geometric Phase Analysis strain mapping.

    Parameters
    ----------
    image : 2-D array (atomic-resolution / lattice-fringe; bright lattice).
    reflections : optional list of two (gx, gy) in pixels-from-FFT-centre to use
        as the lattice reflections. Default: auto-select a well-conditioned pair.
    reference_roi : "auto", or (x, y, w, h) box of an undistorted region whose
        strain is defined as zero. The carrier g-vectors are refined here.
    mask_frac : Bragg mask sigma as a fraction of |g| (spatial resolution knob).
    amp_percentile : pixels below this percentile of min(|H_g1|,|H_g2|) are
        marked invalid (cores / vacuum / edges).
    smooth : Gaussian sigma (px) applied to the phase-gradient fields. Default
        ~ 1.5 lattice spacings.
    pixel_size_nm : optional, only used to annotate physical extent.
    target_valid_fraction : if the valid fraction is below this, the repair
        ladder re-tries (smaller mask, then auto reference).
    detrend : if True, subtract a fitted whole-field affine (linear) ramp from
        each strain/rotation component — removes uniform STEM scan distortion /
        slow scan drift and reveals any LOCAL strain underneath. Default False.
        The affine ramp is ALWAYS detected and reported in res["affine"]
        regardless of this flag. WARNING: a whole-field ramp can also be real
        long-range physics (a continuous lattice bend, a misfit gradient, or a
        twin variant's orientation offset); removing it then erases real signal.
        So: inspect res["affine"]["dominated_by_ramp"] first, only set
        detrend=True to test whether a localized feature survives, and always
        report that the ramp was removed.

    Returns
    -------
    dict with exx, eyy, exy, wxy (2-D float arrays, dimensionless / radians for
    wxy), valid_mask, valid_fraction, g1, g2, condition_number, reference_box,
    flags, stats (referenced robust per-component summaries), and answerable.
    """
    image = np.asarray(image, float)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    H, W = image.shape

    peaks = find_bragg_peaks(image, n=6)
    flags = _artifact_flags(image, peaks)

    if reflections is not None and len(reflections) >= 2:
        g1, g2 = (tuple(map(float, reflections[0])),
                  tuple(map(float, reflections[1])))
    else:
        g1, g2 = _select_pair(peaks)
        g1, g2 = tuple(map(float, g1)), tuple(map(float, g2))

    # condition number of the g-matrix (cycles/pixel)
    Gpix = np.array([[g1[0] / W, g1[1] / H], [g2[0] / W, g2[1] / H]])
    cond = float(np.linalg.cond(Gpix))

    _, Fraw = _fft(image)
    sig1 = mask_frac * np.hypot(*g1)
    sig2 = mask_frac * np.hypot(*g2)
    H1, _ = _bragg_filtered(Fraw, g1, sig1)
    H2, _ = _bragg_filtered(Fraw, g2, sig2)
    A1, A2 = np.abs(H1), np.abs(H2)

    if smooth is None:
        smooth = max(2.0, 1.5 * min(W / abs(g1[0] if g1[0] else 1e9),
                                    H / abs(g1[1] if g1[1] else 1e9)))
        smooth = float(np.clip(smooth, 2.0, 0.03 * min(H, W)))

    # raw phase gradients (per pixel): these are 2*pi*(local wavevector)
    dP1y, dP1x = _phase_gradient(H1)
    dP2y, dP2x = _phase_gradient(H2)
    for arr in (dP1y, dP1x, dP2y, dP2x):
        arr[:] = ndi.gaussian_filter(arr, smooth)

    # --- reference: refine carrier g from the reference region ---
    # local wavevector (cycles/pixel) = dP/(2pi)
    k1 = np.stack([dP1x, dP1y]) / (2 * np.pi)   # (2,H,W): kx,ky for g1
    k2 = np.stack([dP2x, dP2y]) / (2 * np.pi)
    wavevec_var = (ndi.gaussian_filter(k1[0] ** 2 + k1[1] ** 2, smooth)
                   - ndi.gaussian_filter(k1[0], smooth) ** 2
                   - ndi.gaussian_filter(k1[1], smooth) ** 2)
    amp_min = np.minimum(A1, A2)
    if reference_roi == "auto":
        ref_mask, ref_box = _auto_reference(amp_min, np.abs(wavevec_var))
    else:
        x, y, w, h = reference_roi
        ref_mask = np.zeros((H, W), bool)
        ref_mask[y:y + h, x:x + w] = True
        ref_box = tuple(reference_roi)

    g1_ref = np.array([k1[0][ref_mask].mean(), k1[1][ref_mask].mean()])  # cyc/px
    g2_ref = np.array([k2[0][ref_mask].mean(), k2[1][ref_mask].mean()])

    # geometric-phase gradients (per pixel) = raw - carrier(2*pi*g_ref)
    dP1x_g = dP1x - 2 * np.pi * g1_ref[0]
    dP1y_g = dP1y - 2 * np.pi * g1_ref[1]
    dP2x_g = dP2x - 2 * np.pi * g2_ref[0]
    dP2y_g = dP2y - 2 * np.pi * g2_ref[1]

    # distortion D[a,b] = du_a/dx_b, solved per derivative direction with the
    # refined reference reciprocal vectors
    G = 2 * np.pi * np.array([[g1_ref[0], g1_ref[1]],
                              [g2_ref[0], g2_ref[1]]])
    Ginv = np.linalg.inv(G)
    dPx = np.stack([dP1x_g, dP2x_g])            # d/dx of (P1,P2)
    dPy = np.stack([dP1y_g, dP2y_g])            # d/dy
    Dx = -np.einsum("ab,bhw->ahw", Ginv, dPx)   # (du_x/dx, du_y/dx)
    Dy = -np.einsum("ab,bhw->ahw", Ginv, dPy)   # (du_x/dy, du_y/dy)
    exx, uyx = Dx[0], Dx[1]
    uxy, eyy = Dy[0], Dy[1]
    exy = 0.5 * (uxy + uyx)
    wxy = 0.5 * (uyx - uxy)

    # validity mask: enough Bragg amplitude in BOTH reflections, judged BOTH
    # globally (percentile) AND relative to the reference region's lattice
    # quality — a region much dimmer than the good reference gives unreliable
    # phase (the classic GPA edge/low-dose streak artefact).
    ref_amp = np.median(amp_min[ref_mask]) if ref_mask.any() else np.median(amp_min)
    a_thr = max(np.percentile(amp_min, amp_percentile), 0.3 * ref_amp)
    valid = amp_min > a_thr
    b = max(2, int(2 * smooth))
    valid[:b] = valid[-b:] = False              # drop FFT-edge artefacts
    valid[:, :b] = valid[:, -b:] = False
    # light cleanup: open to drop speckle, erode 1px to shave border fringes
    # (the reference-tied amplitude floor already removed the dim/unreliable
    # regions, so heavy erosion here would needlessly discard good lattice).
    valid = ndi.binary_opening(valid, iterations=1)
    valid = ndi.binary_erosion(valid, iterations=1)
    valid_fraction = float(valid.mean())

    # belt-and-suspenders: zero the median strain over the (valid) reference
    rv = ref_mask & valid
    if rv.sum() > 10:
        for f in (exx, eyy, exy):
            f -= np.median(f[rv])
        wxy -= np.median(wxy[rv])

    # repair ladder if too few valid pixels and we still have budget
    if valid_fraction < target_valid_fraction and _repair_depth < 2:
        return gpa_strain_map(
            image, reflections=[g1, g2], reference_roi=ref_box,
            mask_frac=mask_frac * 0.7, amp_percentile=max(5.0, amp_percentile - 10),
            smooth=smooth, pixel_size_nm=pixel_size_nm,
            target_valid_fraction=target_valid_fraction, detrend=detrend,
            _repair_depth=_repair_depth + 1)

    # --- affine (whole-field linear ramp) detection: ALWAYS reported ---
    # A strong ramp = smooth whole-field gradient: usually STEM scan distortion,
    # but possibly real long-range strain / a twin variant's rotation offset.
    affine = {}
    planes = {}
    for nm, f in (("exx", exx), ("eyy", eyy), ("exy", exy), ("wxy", wxy)):
        sx, sy, r2, plane = _fit_plane(f, valid)
        affine[nm] = dict(slope_x=sx, slope_y=sy, ramp_fraction=r2)
        planes[nm] = plane
    max_ramp = max(affine[c]["ramp_fraction"] for c in ("exx", "eyy", "exy"))
    affine["max_inplane_ramp_fraction"] = float(max_ramp)
    affine["rotation_ramp_fraction"] = float(affine["wxy"]["ramp_fraction"])
    # "dominated" = a whole-field linear ramp explains most of the IN-PLANE
    # STRAIN variance -> the result is a smooth gradient, not a localized
    # concentration. (Rotation alone is reported separately but not used for the
    # flag: rotation fields have long-range character even around localized
    # defects, so it would over-flag.)
    affine["dominated_by_ramp"] = bool(max_ramp > 0.5)
    affine["removed"] = False

    def rstats(f):
        v = f[valid]
        if v.size < 10:
            return dict(n=int(v.size))
        return dict(n=int(v.size), median=float(np.median(v)),
                    mean=float(np.mean(v)), std=float(np.std(v)),
                    p1=float(np.percentile(v, 1)), p99=float(np.percentile(v, 99)),
                    p99_abs=float(np.percentile(np.abs(v), 99)))

    stats_raw = dict(exx=rstats(exx), eyy=rstats(eyy),
                     exy=rstats(exy), wxy=rstats(wxy))   # before any detrend

    if detrend:
        for nm, f in (("exx", exx), ("eyy", eyy), ("exy", exy), ("wxy", wxy)):
            f -= planes[nm]                              # remove the linear ramp
        rv = ref_mask & valid                            # re-zero the reference
        if rv.sum() > 10:
            for f in (exx, eyy, exy, wxy):
                f -= np.median(f[rv])
        affine["removed"] = True

    # illumination-envelope guard: the Bragg amplitude should reflect LATTICE
    # quality, not the raw-intensity envelope (HAADF detector / thickness /
    # illumination gradient). If a low-pass of |amp| tracks a low-pass of the raw
    # image, the validity mask and strain are following the envelope rather than
    # the lattice — the classic HAADF-GPA false result. Flag it and withhold
    # `answerable` so callers don't trust an envelope-driven strain field.
    env_sigma = max(8.0, 0.06 * min(H, W))
    img_env = ndi.gaussian_filter(image.astype(float), env_sigma)
    amp_env = ndi.gaussian_filter(amp_min.astype(float), env_sigma)
    _m = valid if valid.sum() > 100 else np.ones((H, W), bool)
    _a, _b = amp_env[_m].ravel(), img_env[_m].ravel()
    if _a.std() > 1e-12 and _b.std() > 1e-12:
        amp_intensity_corr = float(np.corrcoef(_a, _b)[0, 1])
    else:
        amp_intensity_corr = 0.0
    flags["amplitude_tracks_intensity"] = bool(abs(amp_intensity_corr) >= 0.6)

    answerable = (valid_fraction >= 0.25 and cond < 50
                  and not flags["prefiltered"]
                  and not flags["amplitude_tracks_intensity"])

    return dict(
        exx=exx, eyy=eyy, exy=exy, wxy=wxy,
        valid_mask=valid, valid_fraction=valid_fraction,
        g1=g1, g2=g2, g1_ref_cyc_per_px=g1_ref.tolist(),
        g2_ref_cyc_per_px=g2_ref.tolist(), condition_number=cond,
        reference_box=ref_box, mask_frac=mask_frac, smooth=float(smooth),
        flags=flags, answerable=bool(answerable), detrended=bool(detrend),
        amp_intensity_corr=amp_intensity_corr,
        affine=affine, repair_depth=_repair_depth,
        stats=dict(exx=rstats(exx), eyy=rstats(eyy),
                   exy=rstats(exy), wxy=rstats(wxy)),
        stats_raw=stats_raw,
        pixel_size_nm=pixel_size_nm,
    )


# --------------------------------------------------------------------------- #
#  Synthetic lattice generator (for validation / self-test)
# --------------------------------------------------------------------------- #
def make_strained_lattice(shape, a_px=10.0, displacement=None, kind="hex",
                          noise=0.0, seed=0):
    """Synthetic atomic lattice with a known displacement field u(r).

    shape : (H, W). a_px : lattice spacing in pixels.
    displacement : callable (yy, xx) -> (uy, ux) in pixels, or None (perfect).
    kind : 'hex' (3 reflections) or 'square' (2 reflections).
    Returns (image, g_list_pixels) where g_list are the true (gx,gy) per image.
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    if displacement is not None:
        uy, ux = displacement(yy, xx)
    else:
        uy = ux = 0.0
    X, Y = xx - ux, yy - uy
    k = 1.0 / a_px                                # cycles/pixel
    if kind == "hex":
        angles = np.deg2rad([0, 60, 120])
    else:
        angles = np.deg2rad([0, 90])
    img = np.zeros((H, W))
    glist = []
    for th in angles:
        gx, gy = k * np.cos(th), k * np.sin(th)
        img += np.cos(2 * np.pi * (gx * X + gy * Y))
        glist.append((gx * W, gy * H))           # pixels-from-centre convention
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0, noise, img.shape)
    return img, glist


TOOL_SPEC = ToolSpec(
    name="gpa_strain_map",
    description=(
        "Geometric Phase Analysis (GPA): quantitative in-plane strain tensor "
        "(exx, eyy, exy) and rigid lattice rotation (wxy) of an atomic-"
        "resolution / lattice-fringe image, measured RELATIVE TO an undistorted "
        "reference region from two non-collinear Bragg reflections. This is the "
        "tool for a QUANTITATIVE referenced strain field — the sharp counterpart "
        "to fourier_reflection_map (which gives an UNREFERENCED Bragg-phase view, "
        "not a strain tensor)."
    ),
    import_line="from scilink.skills._shared.strain import gpa_strain_map",
    signature=(
        "gpa_strain_map(image, reflections=None, reference_roi='auto', "
        "mask_frac=0.25, amp_percentile=25.0, smooth=None, pixel_size_nm=None, "
        "target_valid_fraction=0.5, detrend=False) -> dict"),
    agents=["image_analysis"],
    when_to_use=(
        "Use for a QUANTITATIVE strain / lattice-rotation map on a single "
        "crystalline domain of an atomic-resolution (S)TEM / HRTEM lattice-"
        "fringe image: misfit strain at a film/substrate or hetero-interface, "
        "strain around a dislocation/defect, tetragonality or shear of a "
        "ferroic/perovskite cell. Pairs with map_polarization (which gives the "
        "polar displacement field) for the strain half of the same cell.\n"
        "\n"
        "ASSUMPTIONS / when NOT to trust it (read 'flags' and 'answerable'): "
        "(1) SMALL-DISTORTION, COHERENT lattice: GPA linearizes the phase, so it "
        "is valid for the smooth strain of a continuous lattice; it is NOT valid "
        "across a true discontinuity (a grain boundary / large-angle twin / phase "
        "boundary unwraps the phase and gives garbage there) — for a "
        "discontinuity use lattice_discontinuity_map, not GPA. (2) Needs a "
        "genuinely UNDISTORTED reference region in view (auto-selected, or pass "
        "reference_roi); strain is meaningless without one and is reported "
        "relative to it. (3) Needs two well-conditioned non-collinear reflections "
        "(condition_number < 50). (4) Do NOT run on an already Fourier-filtered "
        "micrograph (flagged 'prefiltered') — derived strain is then untrustworthy. "
        "(5) HAADF illumination/thickness envelope can masquerade as a strain "
        "gradient: the 'amplitude_tracks_intensity' flag and a withheld "
        "'answerable' mean the field is envelope-driven, not lattice strain. "
        "(6) A whole-field linear ramp (res['affine']['dominated_by_ramp']) is "
        "often STEM scan distortion but can be real long-range physics — inspect "
        "before setting detrend=True, and report when a ramp was removed."
    ),
    parameters={
        "image": {"type": "ndarray", "description": "2D grayscale atomic-resolution / lattice-fringe image (bright lattice). Pass RAW; do NOT Fourier-filter first."},
        "reflections": {"type": "list | None", "description": "Optional two (gx, gy) reflections in pixels-from-FFT-centre. Default None auto-selects a well-conditioned near-orthogonal, similar-|g| pair."},
        "reference_roi": {"type": "str | tuple", "description": "'auto' (default — picks a high-amplitude, low-wavevector-variance undistorted patch) or an explicit (x, y, w, h) box whose strain is defined as zero. Set explicitly when the auto reference lands on a distorted region."},
        "mask_frac": {"type": "float", "description": "Bragg mask sigma as a fraction of |g| — the spatial-resolution knob (default 0.25). LOWER for sharper localization of a small feature; RAISE if the phase is noisy / valid_fraction is low (the repair ladder also lowers it automatically)."},
        "amp_percentile": {"type": "float", "description": "Pixels below this percentile of the min Bragg amplitude are marked invalid (cores/vacuum/edges) (default 25). LOWER to keep more (dimmer) lattice when valid_fraction is too low; RAISE to restrict to the strongest lattice."},
        "smooth": {"type": "float | None", "description": "Gaussian sigma (px) on the phase-gradient fields (default None = ~1.5 lattice spacings). RAISE to suppress phase noise (coarser strain map); LOWER for finer detail at the cost of noise."},
        "pixel_size_nm": {"type": "float", "description": "nm/px; used only to annotate physical extent (strain is dimensionless)."},
        "target_valid_fraction": {"type": "float", "description": "If valid_fraction is below this, the repair ladder re-tries with a smaller mask then auto reference (default 0.5). LOWER to accept a sparser map without repair."},
        "detrend": {"type": "bool", "description": "Subtract the fitted whole-field affine ramp from each component to reveal LOCAL strain under a global gradient (default False). The ramp is ALWAYS detected/reported in res['affine'] regardless. WARNING: a ramp can be real long-range physics (continuous bend, misfit gradient, twin offset) — check res['affine']['dominated_by_ramp'] first; only enable to test whether a localized feature survives, and report that the ramp was removed."},
    },
    required=["image"],
    returns=(
        "dict with: 'exx','eyy','exy','wxy' (2D float arrays — in-plane normal/"
        "shear strain, dimensionless; wxy is rigid rotation in radians; all "
        "referenced so the reference region is ~0); 'valid_mask' (2D bool — "
        "where the lattice is well-resolved; strain is meaningless outside it) "
        "and 'valid_fraction'; 'g1','g2' (the reflections used) and "
        "'condition_number' (<50 = well-conditioned); 'reference_box' (the (x,y,"
        "w,h) reference used); 'answerable' (bool — FALSE = do NOT report strain "
        "numbers: too few valid pixels, ill-conditioned, prefiltered input, or "
        "envelope-driven [amplitude_tracks_intensity]); 'flags' ('prefiltered', "
        "'amplitude_tracks_intensity', peak/background diagnostics); 'affine' "
        "(whole-field linear-ramp fractions per component + 'dominated_by_ramp' + "
        "'removed'); 'stats'/'stats_raw' (robust per-component median/mean/std/"
        "percentiles over valid pixels, after/before detrend); 'amp_intensity_"
        "corr'. Report the strain magnitude from 'stats' (e.g. p99_abs) over the "
        "valid mask, never a raw whole-array mean."
    ),
    example=(
        "res = gpa_strain_map(img, reference_roi='auto')\n"
        "if res['answerable']:\n"
        "    print('exx p99_abs =', res['stats']['exx']['p99_abs'])\n"
        "    print('ramp-dominated?', res['affine']['dominated_by_ramp'])\n"
        "else:\n"
        "    print('not quantifiable:', res['flags'])"
    ),
)
