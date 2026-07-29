"""Deterministic lattice-constant measurement from an atomic-resolution image.

The companion to ``fourier_reflection_map``. That tool deliberately stops at a
reflection *census* and leaves "which reflection is the fundamental lattice" to
the caller, because the distinction is ill-posed from a **1-D** radial power
spectrum (the strongest peak is often a harmonic). This tool answers that exact
question by working in **2-D** reciprocal space, where it *is* well-posed.

Pipeline:
  1. Detect 2-D Bragg peaks (circular-windowed FFT, radial-median background,
     robust significance) — robust to lattice orientation and clean lattices.
  2. Pick the reciprocal basis by TRANSLATIONAL SUPPORT: the shortest reciprocal
     vectors that map the reflection set onto itself. This single size-invariant
     test rejects {200} harmonics (longer |g|), the centered {110} sub-cell (the
     a/sqrt2 flip — also longer |g| than {100}), and low-frequency artifacts /
     weak superlattices (short |g| but low support) at once. A hexagonal lattice
     is handled by its 60deg inner-ring signature; a centered cell with a weak
     {100} is recovered by an explicit conventional-cell promotion.
  3. Refine the basis by least squares over every indexed peak, then invert to
     the real-space cell.

Output is the axis-resolved lattice constant (a1, a2, included angle), the
nearest-neighbor column distance with the projection relationship made explicit
(NN = a for a primitive projection, a/sqrt2 when a centered sublattice is
resolved), a harmonic-labeled census, and a multi_lattice / explained_fraction
CONFIDENCE flag — the tool measures ONE domain, so a field of view holding
several lattices (film+substrate, precipitate, twin) is flagged rather than
silently mis-measured.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _to_gray(a):
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a


def _bragg_peaks(img, pixel_size_nm, d_min, d_max, min_sigma, max_peaks):
    """Detect discrete 2-D Bragg peaks above a radial-background significance floor.

    Returns a list of {fx, fy, d_nm, sigma} in the half-plane (fy > 0, or
    fy == 0 and fx > 0) so each +/- Friedel pair is counted once.
    """
    H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    # A CIRCULAR (radially symmetric) apodization, not the separable
    # outer-product Hann. A separable window leaks spectral power along the
    # kx/ky axes (the "FFT cross"), which swamps Bragg peaks of an
    # axis-aligned lattice; a circular window's leakage is an azimuthally
    # symmetric ring, so on-axis reflections survive.
    rr0 = np.hypot(yy - cy, xx - cx) / min(cy, cx)
    win = np.where(rr0 < 1, 0.5 * (1 + np.cos(np.pi * np.clip(rr0, 0, 1))), 0.0)
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    logP = np.log(np.abs(F) ** 2 + 1e-12)
    fy = (yy - cy) / H                       # cycles / px (per axis)
    fx = (xx - cx) / W
    fr = np.hypot(fy, fx)
    with np.errstate(divide="ignore"):
        d = np.where(fr > 0, pixel_size_nm / np.maximum(fr, 1e-12), np.inf)

    # Remove the smooth radial background (azimuthal MEDIAN per radius — robust
    # to the bright Bragg spots sitting in each shell) so discrete peaks stand
    # out; significance is measured on the residual.
    r_int = np.hypot(yy - cy, xx - cx).astype(int)
    order = np.argsort(r_int.ravel())
    rs = r_int.ravel()[order]
    ls = logP.ravel()[order]
    bnd = np.searchsorted(rs, np.arange(rs[-1] + 2))
    bg_radial = np.zeros(rs[-1] + 1)
    for i in range(rs[-1] + 1):
        seg = ls[bnd[i]:bnd[i + 1]]
        bg_radial[i] = np.median(seg) if seg.size else 0.0
    resid = gaussian_filter(logP - bg_radial[r_int], 0.8)

    band = (d > d_min) & (d < d_max) & np.isfinite(d)
    if not band.any():
        return []
    # Noise floor from the BULK of the residual, excluding the bright-peak tail
    # (top decile): a few hundred strong harmonics in a clean lattice would
    # otherwise inflate a full-distribution MAD and suppress every detection
    # (the n=0-on-clean failure).
    vals = resid[band]
    hi_cut = np.percentile(vals, 90.0)
    noise = vals[vals < hi_cut]
    med = float(np.median(noise))
    robust_std = 1.4826 * float(np.median(np.abs(noise - med))) + 1e-9
    sigma = (resid - med) / robust_std

    # peak_local_max: robust to flat ridges and broad peaks (few-period / small
    # FOV), where a strict ``resid == maximum_filter`` comparison drops peaks.
    from skimage.feature import peak_local_max
    coords = peak_local_max(np.where(band, resid, -np.inf),
                            min_distance=3, threshold_abs=med + min_sigma * robust_std,
                            exclude_border=False)

    peaks = []
    for y, x in coords:
        if fy[y, x] < 0 or (fy[y, x] == 0 and fx[y, x] <= 0):
            continue                          # keep one of each Friedel pair
        peaks.append({"fx": float(fx[y, x]), "fy": float(fy[y, x]),
                      "fr": float(fr[y, x]), "d_nm": float(d[y, x]),
                      "sigma": float(sigma[y, x])})
    peaks.sort(key=lambda r: -r["sigma"])
    return peaks[:max_peaks]


def _fit_basis(peaks, collinear_tol, strong_frac=0.33, supp_frac=0.45, match_tol=0.18):
    """Choose the reciprocal basis (g1, g2) by TRANSLATIONAL SUPPORT.

    A reciprocal-lattice basis vector ``v`` is, by definition, a translation
    that maps the whole reflection set onto itself: for most peaks ``p`` there
    is another peak at ``p + v``. This single, size-invariant test resolves all
    the confusable cases at once, without tuning order penalties or coverage
    (which are biased by cell size):

      * a {200}-type HARMONIC has support but is LONGER |g| than {100};
      * a centered {110} SUB-cell vector has support but is also longer |g|
        than {100} — so the shortest-|g| supported vector is {100}, never the
        a/sqrt2 sub-cell (this is what kills the flip);
      * a doubled SUPERLATTICE / low-frequency artifact is SHORTER |g| but has
        LOW support (translating by it does not map the main reflections onto
        peaks), so it is rejected despite being shortest.

    The basis is therefore the two shortest-|g| non-collinear peaks whose
    translational support clears ``supp_frac`` of the (sigma-weighted) peak set.

    Returns (g1, g2) dicts, or (g1, None) if only one direction is supported.
    """
    from scipy.spatial import cKDTree

    def _acute(a, b):
        ang = abs(np.degrees(np.arctan2(a["fy"], a["fx"])
                             - np.arctan2(b["fy"], b["fx"]))) % 180.0
        return min(ang, 180.0 - ang)

    smax = max(p["sigma"] for p in peaks)
    strong = [p for p in peaks if p["sigma"] >= strong_frac * smax]
    if len(strong) < 2:
        strong = peaks[:]
    # MERGE near-duplicate reflections (same spot detected twice, or a weak
    # satellite hugging a strong reflection), keeping the STRONGEST of each
    # cluster. Without this a weak longer-d satellite can sort ahead of the real
    # fundamental and become g1 (the La5BO3 0.44 vs 0.405 nm case).
    merge_tol = 0.12 * min(p["fr"] for p in strong)
    strong.sort(key=lambda r: -r["sigma"])
    merged = []
    for p in strong:
        if all(np.hypot(p["fx"] - q["fx"], p["fy"] - q["fy"]) > merge_tol
               for q in merged):
            merged.append(p)
    strong = merged
    # Full reflection set including Friedel (-g) partners, sigma-weighted.
    pts = np.array([[p["fx"], p["fy"]] for p in strong]
                   + [[-p["fx"], -p["fy"]] for p in strong])
    wts = np.array([p["sigma"] for p in strong] * 2)
    tree = cKDTree(pts)
    tot = float(wts.sum()) + 1e-9

    def support(v):
        # weighted fraction of peaks p with a peak at p + v
        dist, idx = tree.query(pts + np.asarray(v))
        ok = dist < match_tol * np.hypot(*v)
        return float(wts[ok].sum()) / tot

    # Artifact floor: a basis vector cannot be much longer-period than the
    # strongest reflection — removes very-low-frequency junk (finite-size beats,
    # 2x/4x doubling) that would otherwise be chosen as the longest-d g1.
    fr_strong = max(strong, key=lambda r: r["sigma"])["fr"]
    cand = sorted([r for r in strong if r["fr"] >= 0.55 * fr_strong],
                  key=lambda r: r["fr"])      # shortest |g| (longest d) first
    if len(cand) < 2:
        cand = sorted(strong, key=lambda r: r["fr"])
    # HEXAGONAL pre-check: a hex lattice's innermost ring has THREE directions
    # 60deg apart (vs two at 90deg for square/rect). The general support picker
    # might pair a 120deg-separated couple and give a wrong cell, so detect hex
    # and return an adjacent (60deg) inner-ring pair directly.
    # Decide hex vs square by the angle between the STRONGEST inner-ring
    # reflection and its STRONGEST non-collinear partner: ~60deg => hexagonal
    # (return that pair as the basis), ~90deg => square/rect (fall through).
    # Using the strongest partner is robust to weak spurious ring peaks that
    # would otherwise inject a false 90deg (or 60deg) pair.
    g_str = max(cand, key=lambda r: r["sigma"])
    ring = sorted([r for r in cand if abs(r["fr"] - g_str["fr"]) < 0.18 * g_str["fr"]],
                  key=lambda r: -r["sigma"])
    partner = next((r for r in ring if _acute(g_str, r) >= collinear_tol), None)
    if partner is not None and 52.0 <= _acute(g_str, partner) <= 68.0:
        return g_str, partner

    supp = {id(g): support([g["fx"], g["fy"]]) for g in cand}
    smax_s = max(g["sigma"] for g in cand)
    # A basis vector must be (a) STRONG — a main reflection, not noise or a weak
    # superstructure spot — and (b) translationally SUPPORTED above a low floor
    # (it maps the reflection set onto itself). Strength keeps the longest-d g1
    # on the conventional {100}, not the {110} sub-cell, even when vacancies or
    # blur degrade {100}'s outer harmonics and lower its support.
    supported = [g for g in cand
                 if supp[id(g)] >= 0.30 and g["sigma"] >= 0.45 * smax_s]
    if not supported:
        supported = cand

    g1 = supported[0]
    # g2 = the SHORTEST-|g| non-collinear supported partner (the second primary
    # axis). Not the best-supported one — a {110}-type diagonal also has high
    # translational support but would give a spurious 45deg cell.
    for g2 in supported[1:]:
        if _acute(g1, g2) >= collinear_tol:
            return g1, g2
    return g1, None


def measure_lattice_constant(image_array, pixel_size_nm, params=None):
    """Measure the lattice constant deterministically from FFT Bragg geometry.

    Args:
        image_array: 2D grayscale (or HxWx3) atomic-resolution image. Crop out
            substrate / vacuum BEFORE calling if the objective asks to exclude
            a region (pass the cropped array); the measurement itself is robust
            to a modest amount of a second crystalline region but cleanest on a
            single domain.
        pixel_size_nm: physical pixel size in nm (square pixels; resample first
            if anisotropic).
        params: optional dict of tunable knobs (all have robust defaults):
            d_range: (min, max) nm spacing window for accepted reflections,
                default (0.10, 1.5). RAISE the min to ignore high-order harmonics
                you don't care about; WIDEN if the lattice is very fine/coarse.
            min_sigma: significance floor (robust sigma over the residual
                spectrum) for a peak to count, default 4.0. LOWER to recover a
                weak lattice in a noisy image; RAISE if noise peaks leak in.
            collinear_tol_deg: a candidate for the 2nd basis vector must be at
                least this far from the 1st direction, default 15.0. RAISE if a
                near-collinear noise peak is being chosen as the 2nd axis.
            centering_sigma_frac: a {110}-type reflection (at g1+g2) counts as a
                resolved centered sublattice if its sigma >= this fraction of the
                fundamental's, default 0.25. This sets NN = a/sqrt2 vs NN = a.
            max_peaks: cap on detected reflections considered, default 60.

    Returns: dict (see TOOL_SPEC.returns). Key fields: lattice_constant_nm,
        a1_nm, a2_nm, gamma_deg, nn_distance_nm, nn_basis, reflections, note.
    """
    p = params or {}
    d_min, d_max = p.get("d_range", (0.10, 1.5))
    min_sigma = float(p.get("min_sigma", 4.0))
    collinear_tol = float(p.get("collinear_tol_deg", 15.0))
    strong_frac = float(p.get("strong_frac", 0.33))
    cen_frac = float(p.get("centering_sigma_frac", 0.25))
    max_peaks = int(p.get("max_peaks", 60))

    img = _to_gray(image_array)
    H, W = img.shape
    peaks = _bragg_peaks(img, pixel_size_nm, d_min, d_max, min_sigma, max_peaks)

    if len(peaks) < 1:
        return {"note": ("No Bragg reflection above the significance floor "
                         f"(min_sigma={min_sigma}). Lattice not resolved — try "
                         "lowering min_sigma or widening d_range."),
                "reflections": [], "lattice_constant_nm": None}

    # --- pick the reciprocal basis by translational support ------------------
    g1, g2 = _fit_basis(peaks, collinear_tol, strong_frac)
    one_d = g2 is None
    if one_d:                                  # only one lattice direction resolved
        g2 = {"fx": -g1["fy"], "fy": g1["fx"],  # synthesise an orthogonal partner
              "fr": g1["fr"], "d_nm": g1["d_nm"], "sigma": 0.0}

    # CONVENTIONAL-CELL PROMOTION: if the chosen cell is a centered sub-cell
    # (e.g. the {110} bright-column cell when {100} is weak from vacancies/
    # channeling), the larger {100}-type reflections sit at (g1+-g2)/2. When
    # both are present as real peaks, promote the basis to them so the reported
    # lattice constant is the conventional cell, not the a/sqrt2 sub-cell. This
    # fires ONLY when the larger reflections actually exist, so a true primitive
    # cell (no half-vector peaks) is never wrongly enlarged.
    if not one_d:
        from scipy.spatial import cKDTree
        ppos = np.array([[p["fx"], p["fy"]] for p in peaks])
        psig = np.array([p["sigma"] for p in peaks])
        ptree = cKDTree(ppos)
        smax_p = float(psig.max())

        def _find(vec):
            tol = 0.18 * np.hypot(*vec)
            for v in (vec, -vec):
                dst, i = ptree.query(v)
                if dst < tol and psig[i] >= 0.4 * smax_p:
                    return ppos[i] if (v is vec) else -ppos[i]
            return None

        def _ortho(a, b):                    # acute angle between two vectors (deg)
            ang = abs(np.degrees(np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0]))) % 180.0
            return min(ang, 180.0 - ang)

        for _ in range(2):
            a = np.array([g1["fx"], g1["fy"]]); b = np.array([g2["fx"], g2["fy"]])
            # Centering (a/sqrt2) only applies to a near-orthogonal cell. A
            # hexagonal/oblique cell's (g1+-g2)/2 may land on a real reflection
            # too, but promoting there is wrong — so require the current AND the
            # promoted cell to be ~rectangular.
            if not (78.0 <= _ortho(a, b) <= 102.0):
                break
            c1, c2 = _find((a + b) / 2), _find((a - b) / 2)
            if c1 is None or c2 is None or not (78.0 <= _ortho(c1, c2) <= 102.0):
                break
            g1 = {**g1, "fx": float(c1[0]), "fy": float(c1[1]), "fr": float(np.hypot(*c1))}
            g2 = {**g2, "fx": float(c2[0]), "fy": float(c2[1]), "fr": float(np.hypot(*c2))}

    # reciprocal basis in cycles/px; REFINE by least squares over every peak
    # that indexes cleanly onto it — averaging out per-spot localization noise
    # (a single noisy basis spot otherwise biases the constant by several %).
    Bpx = np.array([[g1["fx"], g1["fy"]], [g2["fx"], g2["fy"]]])
    if not one_d:
        try:
            Binv = np.linalg.inv(Bpx)
            Pall = np.array([[p["fx"], p["fy"]] for p in peaks])
            idx = np.round(Pall @ Binv)            # index = G . B^-1 (row convention)
            res = np.hypot(*(Pall - idx @ Bpx).T)
            keep = (res < 0.18 * np.hypot(*Bpx[0])) & (np.abs(idx).max(1) <= 4) \
                & (np.abs(idx).sum(1) > 0)
            if keep.sum() >= 3:
                Bpx, *_ = np.linalg.lstsq(idx[keep], Pall[keep], rcond=None)
                g1 = {**g1, "fx": Bpx[0, 0], "fy": Bpx[0, 1], "fr": float(np.hypot(*Bpx[0]))}
                g2 = {**g2, "fx": Bpx[1, 0], "fy": Bpx[1, 1], "fr": float(np.hypot(*Bpx[1]))}
        except np.linalg.LinAlgError:
            pass

    # reciprocal basis in 1/nm, real-space basis by inverse-transpose
    B = Bpx / pixel_size_nm
    try:
        A = np.linalg.inv(B).T
    except np.linalg.LinAlgError:
        return {"note": "Degenerate reciprocal basis; lattice not resolvable.",
                "reflections": peaks, "lattice_constant_nm": None}
    a1, a2 = A[0], A[1]
    a1_nm, a2_nm = float(np.hypot(*a1)), float(np.hypot(*a2))
    cosg = float(np.dot(a1, a2) / (a1_nm * a2_nm + 1e-12))
    gamma_deg = float(np.degrees(np.arccos(np.clip(cosg, -1, 1))))

    d1, d2 = 1.0 / (np.hypot(*B[0]) + 1e-12), 1.0 / (np.hypot(*B[1]) + 1e-12)
    lattice_const = float(np.mean([a1_nm, a2_nm]) if not one_d else a1_nm)
    # uncertainty ~ one FFT frequency bin propagated to real space
    df = 0.5 * (1.0 / W + 1.0 / H)
    lat_std = float(lattice_const ** 2 * df / pixel_size_nm)

    # --- harmonic-labeled census: order (n,m) of every peak on the basis -----
    Binv = np.linalg.inv(B * pixel_size_nm)    # back to cycles/px basis
    census = []
    for r in peaks:
        nm = np.array([r["fx"], r["fy"]]) @ Binv     # index = G . B^-1
        order = (int(round(nm[0])), int(round(nm[1])))
        resid_ord = float(np.hypot(*(nm - np.round(nm))))
        census.append({"d_nm": round(r["d_nm"], 4), "sigma": round(r["sigma"], 1),
                       "order": order, "on_lattice": resid_ord < 0.18,
                       "is_fundamental": order in [(1, 0), (0, 1), (-1, 0), (0, -1)]})

    # --- MULTI-LATTICE detection --------------------------------------------
    # A single global FFT superimposes the reciprocal lattices of every
    # crystalline domain in the field of view (a film+substrate of different a,
    # a precipitate, a rotated grain / twin). The primary cell only explains its
    # own reflections; if a large sigma-fraction of strong peaks is left
    # UNEXPLAINED, a second lattice is present and the single global constant is
    # not meaningful — the caller should crop to one domain (ROI) or use the
    # real-space mapping tools. We report the second cell and a warning.
    strong_all = [p for p in peaks if p["sigma"] >= strong_frac * max(q["sigma"] for q in peaks)]
    tot_sig = sum(p["sigma"] for p in strong_all) + 1e-9
    on_sig = 0.0
    off_peaks = []
    for p in strong_all:
        nm = np.array([p["fx"], p["fy"]]) @ Binv     # index = G . B^-1
        if np.hypot(*(nm - np.round(nm))) < 0.18 and abs(np.round(nm)).max() <= 10:
            on_sig += p["sigma"]
        else:
            off_peaks.append(p)
    explained_frac = on_sig / tot_sig
    secondary = None                          # best-effort second-lattice cell
    if explained_frac < 0.55 and len(off_peaks) >= 6:
        try:
            h1, h2 = _fit_basis(off_peaks, collinear_tol, strong_frac)
            if h2 is not None:
                A2 = np.linalg.inv(np.array([[h1["fx"], h1["fy"]],
                                             [h2["fx"], h2["fy"]]]) / pixel_size_nm).T
                s1, s2 = float(np.hypot(*A2[0])), float(np.hypot(*A2[1]))
                cg = float(np.dot(A2[0], A2[1]) / (s1 * s2 + 1e-12))
                secondary = {"a1_nm": round(s1, 4), "a2_nm": round(s2, 4),
                             "gamma_deg": round(float(np.degrees(np.arccos(np.clip(cg, -1, 1)))), 1),
                             "lattice_constant_nm": round((s1 + s2) / 2, 4)}
        except Exception:
            secondary = None

    # shortest real-space lattice translation (the generic NN of the cell)
    lat_nn = float(min(a1_nm, a2_nm, np.hypot(*(a1 + a2)), np.hypot(*(a1 - a2))))

    # --- centered-sublattice (a/sqrt2) check — only meaningful for ~90deg cells
    # On a near-rectangular cell a strong {110}-type reflection at g1+g2 means an
    # extra column sits at the cell centre (the perovskite A/B projection), so
    # the bright-column NN is the intra-cell a/sqrt2, NOT a lattice translation.
    # On a non-90deg cell (hexagonal, oblique) g1+g2 is just another lattice
    # point, so this test does not apply and NN is the shortest translation.
    g_sum = np.array([g1["fx"] + g2["fx"], g1["fy"] + g2["fy"]])
    centered = False
    if not one_d and abs(gamma_deg - 90.0) < 8.0:
        for r in peaks:
            if np.hypot(*(np.array([r["fx"], r["fy"]]) - g_sum)) < 0.3 * g1["fr"]:
                if r["sigma"] >= cen_frac * g1["sigma"]:
                    centered = True
                break
    if centered:
        nn_nm = float(1.0 / (np.hypot(*(B[0] + B[1])) + 1e-12))
        nn_basis = ("centered projection: a {110}-type reflection is resolved, "
                    "so the bright-column NN = lattice_constant / sqrt2")
    else:
        nn_nm = lat_nn
        nn_basis = ("primitive projection: NN is the shortest lattice translation "
                    "(no centered sublattice resolved)")

    note = (f"Fundamental basis from FFT: a1={a1_nm:.4f} nm, a2={a2_nm:.4f} nm, "
            f"gamma={gamma_deg:.1f} deg. Lattice constant {lattice_const:.4f} +/- "
            f"{lat_std:.4f} nm. NN column distance {nn_nm:.4f} nm ({nn_basis}).")
    if one_d:
        note += (" Only ONE lattice direction was resolved; a2 is a synthesised "
                 "orthogonal partner — treat as a 1-D spacing, not a 2-D cell.")
    multi = explained_frac < 0.55
    # Too few reflections (under-sampling, tiny FOV, severe scan distortion):
    # the cell can index its handful of spots trivially (explained ~1.0) yet be
    # wrong — so flag insufficiency separately from the multi-lattice case.
    too_few = len(strong_all) < 6
    low_confidence = multi or too_few or one_d
    if multi:
        note += (f" LOW-CONFIDENCE: the cell explains only {explained_frac*100:.0f}% "
                 "of the strong reflections. Either MULTIPLE LATTICES are present "
                 "(film+substrate, precipitate, rotated grain/twin) or the pattern "
                 "is complex/poorly sampled — the single global lattice constant "
                 "is not reliable. Crop to one domain (ROI) and re-run, or use "
                 "real-space mapping.")
        if secondary:
            note += (f" A second lattice may be present (a~"
                     f"{secondary['lattice_constant_nm']} nm, gamma~{secondary['gamma_deg']} deg).")
    if too_few:
        note += (f" LOW-CONFIDENCE: only {len(strong_all)} strong reflections were "
                 "resolved — too few to determine the cell robustly (under-sampled "
                 "lattice, tiny field of view, or severe scan distortion). Use a "
                 "larger / better-sampled region.")

    return {
        "lattice_constant_nm": round(lattice_const, 4),
        "lattice_constant_std_nm": round(lat_std, 4),
        "a1_nm": round(a1_nm, 4), "a2_nm": round(a2_nm, 4),
        "gamma_deg": round(gamma_deg, 1),
        "fundamental_d_nm": [round(float(d1), 4), round(float(d2), 4)],
        "nn_distance_nm": round(nn_nm, 4), "nn_basis": nn_basis,
        "centered_sublattice": centered, "one_direction_only": one_d,
        "multi_lattice": multi, "explained_fraction": round(explained_frac, 2),
        "low_confidence": low_confidence, "secondary_lattice": secondary,
        "n_reflections": len(peaks), "reflections": census, "note": note,
    }


from scilink.skills._shared._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="measure_lattice_constant",
    description=(
        "Deterministically measure the lattice constant of a SINGLE crystalline "
        "domain in an atomic-resolution (S)TEM/HRTEM image from FFT Bragg "
        "geometry. Picks the reciprocal basis by translational support (the "
        "shortest reciprocal vectors that map the reflection set onto itself), "
        "so a {200} HARMONIC, a centered {110} SUB-cell (the a/sqrt2 flip), and "
        "low-frequency artifacts are all rejected; refines by least squares over "
        "all indexed peaks. Returns the axis-resolved cell (a1, a2, angle), the "
        "nearest-neighbor column distance with the projection relationship "
        "(NN = a, or a/sqrt2 for a centered/perovskite sublattice), and a "
        "multi_lattice / explained_fraction confidence flag. Works for square, "
        "rectangular, centered/pseudocubic, and hexagonal lattices."
    ),
    import_line=(
        "from scilink.skills.image_analysis.atomic_stem.lattice import "
        "measure_lattice_constant"),
    signature="measure_lattice_constant(image_array, pixel_size_nm, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Use for a 'measure the lattice constant / lattice parameter / nearest-"
        "neighbor distance' objective on a crystalline atomic-resolution image. "
        "This is the FAST, deterministic default for that question — prefer it "
        "over real-space atom-column detection (detect_atoms / detect_atoms_dcnn "
        "+ find_zone_axes), which is for per-site work (strain maps, defect "
        "census, superlattice localization), not a single global lattice "
        "constant. Calibrate first: pass the true square-pixel size (nm/px); "
        "resample to square pixels if anisotropic. If the objective excludes a "
        "substrate/vacuum band, crop to the crystalline region of interest and "
        "pass that sub-array. The tool resolves fundamental-vs-harmonic itself, "
        "so do NOT pre-select 'the strongest reflection' as the lattice "
        "constant. To localize a SUPERSTRUCTURE/satellite (not measure the cell) "
        "use fourier_reflection_map instead.\n"
        "\n"
        "SCOPE / when NOT to trust it: this measures ONE domain. If the field of "
        "view holds MULTIPLE lattices (film+substrate of different a, a "
        "precipitate, a rotated grain or twin), a single global FFT superimposes "
        "their reciprocal lattices and the result is unreliable — check the "
        "returned `multi_lattice` flag and `explained_fraction`; if flagged, crop "
        "to one domain (ROI) and re-run, or use the real-space tools. Also needs "
        "a reasonably sampled lattice (roughly >=12 px per period) and an "
        "adequate field of view (several unit cells / >=~6 reflections); it is "
        "not reliable on heavily under-sampled lattices, tiny crops, or images "
        "with severe scan distortion."
    ),
    parameters={
        "image_array": {"type": "ndarray", "description": "2D grayscale (or HxWx3) atomic-resolution image (crop out substrate/vacuum first if excluded by the objective)."},
        "pixel_size_nm": {"type": "float", "description": "Physical pixel size in nm (square pixels)."},
        "params": {"type": "dict", "description": (
            "Optional tunable knobs (robust defaults): "
            "d_range (nm tuple, default (0.10, 1.5)) — RAISE min to ignore "
            "high-order harmonics, WIDEN for very fine/coarse lattices; "
            "min_sigma (default 4.0) — LOWER to recover a weak lattice in a "
            "noisy image, RAISE if noise peaks leak in; "
            "collinear_tol_deg (default 15.0) — minimum angle for the 2nd basis "
            "vector, RAISE if a near-collinear noise peak is picked as axis 2; "
            "strong_frac (default 0.33) — a reflection must reach this fraction "
            "of the strongest peak's significance to be a basis candidate; LOWER "
            "if the true fundamental is weak and gets skipped for a stronger "
            "harmonic, RAISE to ignore weak low-frequency artifacts; "
            "centering_sigma_frac (default 0.25) — {110} significance fraction "
            "above which a centered sublattice is declared (sets NN = a/sqrt2); "
            "max_peaks (default 60).")},
    },
    required=["image_array", "pixel_size_nm"],
    returns=(
        "dict with: 'lattice_constant_nm' (float — the fundamental repeat, mean "
        "of a1/a2; None if no lattice resolved) and 'lattice_constant_std_nm' "
        "(~one-FFT-bin uncertainty); 'a1_nm','a2_nm','gamma_deg' (axis-resolved "
        "cell); 'fundamental_d_nm' (the two basis reflection d-spacings); "
        "'nn_distance_nm' and 'nn_basis' (NN column distance and whether it "
        "equals a or a/sqrt2 from a resolved centered sublattice); "
        "'centered_sublattice' (bool); 'one_direction_only' (bool — True when "
        "only a 1-D spacing was resolvable); "
        "'multi_lattice' (bool — True = LOW CONFIDENCE: the cell explains <55% of "
        "the strong reflections, so multiple lattices are likely present or the "
        "pattern is complex/under-sampled; do NOT trust the single constant, crop "
        "to one domain); 'low_confidence' (bool — True if multi_lattice, too few "
        "reflections were resolved (under-sampled / tiny FOV / severe distortion), "
        "or only one axis was found; treat the constant as unreliable and read "
        "'note'); 'explained_fraction' (0-1, sigma-weighted fraction of "
        "strong reflections the cell accounts for — a fit-quality/confidence "
        "score); 'secondary_lattice' (best-effort {a1_nm,a2_nm,gamma_deg,"
        "lattice_constant_nm} of a second lattice when multi_lattice, else None); "
        "'n_reflections'; 'reflections' (census: each {d_nm, sigma, order=(n,m) "
        "on the fitted basis, on_lattice, is_fundamental} — harmonics are order "
        "(2,0) etc.); 'note' (human-readable summary incl. any warning). On "
        "failure: 'note' explains why and 'lattice_constant_nm' is None."
    ),
    example=(
        "res = measure_lattice_constant(film_crop, pixel_size_nm=0.02039)\n"
        "print(res['note'])\n"
        "a = res['lattice_constant_nm']; nn = res['nn_distance_nm']\n"
        "# harmonics are visible in the census but never returned as 'a':\n"
        "for r in res['reflections']:\n"
        "    print(r['d_nm'], r['order'], 'fundamental' if r['is_fundamental'] else '')"
    ),
)
