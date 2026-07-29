"""FFT-based point-defect / periodicity-anomaly mapping for periodic images.

Self-contained (numpy + scipy only). Finds the dominant reflections of a
periodic pattern, reconstructs the "perfect" pattern by masked inverse FFT,
and maps where the image deviates from it. A point defect (vacancy, dopant
column, adatom/interstitial, missing particle in a self-assembled array)
breaks local periodicity and appears as a localized residual anomaly.

The method is pattern-agnostic: it never asks what the repeating unit is, so
it applies equally to atomic-resolution STEM/HRTEM/STM lattices, moire
superlattices, self-assembled nanoparticle arrays, and any other image with
several repeat periods in the field of view.

Design notes (the things that make a naive Bragg-filter residual lie):

  * Reflections must clear a significance gate (peak amplitude vs the median
    amplitude on its own frequency annulus) — otherwise pure noise yields
    "peaks" and the residual map is meaningless. Below the gate the tool
    reports periodic=False instead of inventing defects.
  * The anomaly threshold is calibrated on a PHASE-RANDOMIZED NULL — an image
    with the identical power spectrum but no spatial organization — processed
    through the same pipeline, so "anomaly" means "stronger than anything the
    noise floor produces", not an arbitrary cutoff.
  * All smoothing scales are tied to the measured pattern period, not fixed
    pixel counts, so behaviour transfers across magnifications.
  * Anomalies are only meaningful where the periodic pattern actually exists:
    a validity mask from the summed Bragg-filtered amplitude excludes vacuum,
    amorphous regions and pattern-free background, and an FFT-wraparound
    border is excluded.
  * The residual flags ANY localized aperiodicity — contamination and surface
    steps included. Each candidate therefore carries a `coherence_dip` flag
    (is the lattice amplitude itself locally depressed?) separating defects
    that disrupt the pattern from intensity anomalies sitting on top of an
    intact pattern; final typing (vacancy vs dopant vs dirt) is for the
    caller, ideally against a real-space crop.

Main entry point: fft_defect_map(image, ...) -> dict.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .strain import find_bragg_peaks

__all__ = ["fft_defect_map", "make_defective_lattice"]


def _to_gray(a):
    g = np.asarray(a, float)
    if g.ndim == 3:
        g = g[..., :3].mean(-1) if g.shape[2] in (3, 4) else g[..., 0]
    return g


def _peak_significance(image, peaks):
    """Peak amplitude over the median amplitude of its own frequency annulus.

    Computed on the Hann-windowed spectrum (sidelobe suppression); the annulus
    median is a robust noise-floor estimate at that |g| that real reflections
    barely shift. Returns one ratio per peak.
    """
    H, W = image.shape
    img = image - image.mean()
    win = np.outer(np.hanning(H), np.hanning(W))
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * win)))
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    out = []
    for gx, gy, _ in peaks:
        r0 = np.hypot(gx, gy)
        ann = P[(r > 0.88 * r0) & (r < 1.12 * r0)]
        floor = np.median(ann) if ann.size else 1e-9
        out.append(float(P[int(cy + gy), int(cx + gx)] / (floor + 1e-12)))
    return out


def _gauss_mask(shape, centre_xy, sigma):
    H, W = shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    gx, gy = centre_xy
    return np.exp(-(((xx - (cx + gx)) ** 2 + (yy - (cy + gy)) ** 2)
                    / (2.0 * sigma ** 2)))


def _residual_and_amplitude(img, peaks, mask_frac, dc_frac, smooth_px):
    """Smoothed perfect-pattern residual and summed Bragg amplitude.

    residual = img - IFFT(F * mask) computed as IFFT(F * (1 - mask)), where
    mask covers DC plus a Gaussian at every reflection and its Friedel mate.
    """
    H, W = img.shape
    Fraw = np.fft.fftshift(np.fft.fft2(img))
    g_mags = [np.hypot(gx, gy) for gx, gy, _ in peaks]
    mask = _gauss_mask((H, W), (0, 0), max(2.0, dc_frac * min(g_mags)))
    amp = np.zeros((H, W))
    for (gx, gy, _), gm in zip(peaks, g_mags):
        sig = max(2.0, mask_frac * gm)
        m = _gauss_mask((H, W), (gx, gy), sig)
        mask = np.maximum(mask, m)
        mask = np.maximum(mask, _gauss_mask((H, W), (-gx, -gy), sig))
        amp += np.abs(np.fft.ifft2(np.fft.ifftshift(Fraw * m)))
    resid = np.real(np.fft.ifft2(np.fft.ifftshift(Fraw * (1.0 - mask))))
    resid_s = ndi.gaussian_filter(resid, smooth_px)
    amp_s = ndi.gaussian_filter(amp, smooth_px)
    return resid_s, amp_s, mask


def fft_defect_map(image_array, pixel_size_nm=None, params=None):
    """Map point defects / localized periodicity anomalies in a periodic image.

    Args:
        image_array: 2D grayscale (or HxWx3) array containing a periodic
            pattern (atomic lattice, particle array, moire, ...) with at
            least several repeat periods in the field of view.
        pixel_size_nm: optional square-pixel calibration; adds nm-valued
            fields. The image must already be on SQUARE pixels — resample
            anisotropic frames first (the FFT geometry is otherwise wrong).
            For scan-probe data this, and tilt leveling, are the upstream
            skill's preprocessing job, not this tool's.
        params: optional dict of knobs (see TOOL_SPEC.parameters).

    Returns: dict (see TOOL_SPEC.returns).
    """
    p = dict(n_reflections=12, min_peak_snr=8.0, mask_frac=0.15, dc_frac=0.5,
             smooth_frac=0.30, null_percentile=99.9, k_min=4.0,
             amp_valid_frac=0.35, dip_frac=0.85, border_periods=2.0,
             min_area_frac=0.04, merge_frac=0.75, max_defects=200,
             edge_pad=True)
    p.update(params or {})

    img = _to_gray(image_array)
    H, W = img.shape

    # --- 1. significant reflections ---
    raw_peaks = find_bragg_peaks(img, n=int(p["n_reflections"]))
    snrs = _peak_significance(img, raw_peaks) if raw_peaks else []
    peaks = [pk for pk, s in zip(raw_peaks, snrs) if s >= p["min_peak_snr"]]
    # ALL candidates are reported (gated ones flagged) so a caller can see
    # how close a faint lattice came to the gate and retune min_peak_snr
    reflections = [
        {"gx_px": pk[0], "gy_px": pk[1],
         "period_px": round(1.0 / max(np.hypot(pk[0] / W, pk[1] / H), 1e-9), 2),
         "snr": round(s, 1), "significant": bool(s >= p["min_peak_snr"])}
        for pk, s in zip(raw_peaks, snrs)]
    out = {"periodic": bool(len(peaks) >= 2), "reflections": reflections}
    if len(peaks) < 2:
        best = max(snrs) if snrs else 0.0
        out["note"] = (
            "Fewer than 2 reflections clear the significance gate "
            f"(min_peak_snr={p['min_peak_snr']}; best candidate snr="
            f"{best:.1f}) — no resolvable periodic pattern, so a "
            "periodicity-residual defect map is undefined. A best snr just "
            "below the gate may be a faint lattice (lower min_peak_snr and "
            "re-run); a best snr far below it usually means the image is "
            "not periodic (rings in the FFT = short-range order only) and "
            "a different tool fits better.")
        return out

    # Pattern period := the longest-period reflection AMONG THE STRONGEST.
    # Restricting to the top SNR tier first is what makes this robust both
    # ways: higher-order harmonics (shorter period) are weaker and excluded,
    # and a low-frequency artifact that grazes the significance gate (a slow
    # height/illumination modulation, a terrace) is also weaker than the true
    # first-order lattice spots, so it cannot hijack the period and inflate
    # every downstream scale. Plain max over all significant reflections is
    # fooled by that artifact; the strongest spot alone is fooled by a strong
    # harmonic — the strong-tier max avoids both.
    sig_refl = [r for r in reflections if r["significant"]]
    snr_cut = 0.5 * max(r["snr"] for r in sig_refl)
    period = float(max(r["period_px"] for r in sig_refl if r["snr"] >= snr_cut))
    out["pattern_period_px"] = round(period, 2)
    if pixel_size_nm:
        out["pattern_period_nm"] = round(period * float(pixel_size_nm), 4)

    # --- 2. perfect-pattern residual + lattice amplitude (period-scaled) ---
    smooth_px = max(1.0, p["smooth_frac"] * period)
    # The FFT assumes the image tiles, so the border discontinuity would
    # contaminate the reconstruction over the real-space extent of the
    # NARROWEST Fourier mask (sigma_x = N / (2*pi*sigma_f)). Mirror-padding
    # by that extent removes the value jump at the seam; only a much weaker
    # lattice-phase crease remains, so the border exclusion can stay small
    # and most of the frame stays usable.
    g_mags = [np.hypot(gx, gy) for gx, gy, _ in peaks]
    sig_f_min = max(2.0, p["mask_frac"] * min(g_mags))
    kernel_x = max(H, W) / (2.0 * np.pi * sig_f_min)
    if p["edge_pad"]:
        pad = int(round(min(2.5 * kernel_x + 3.0 * smooth_px,
                            min(H, W) // 2)))
        imgp = np.pad(img, pad, mode="reflect")
        b = int(round(p["border_periods"] * period + 3.0 * smooth_px))
    else:
        pad = 0
        imgp = img
        b = int(round(max(p["border_periods"] * period,
                          2.5 * kernel_x + 3.0 * smooth_px)))
    Hp, Wp = imgp.shape
    # peak coordinates are in pixels-from-centre, which scale with the frame
    peaks_p = [(gx * Wp / W, gy * Hp / H, pw) for gx, gy, pw in peaks]
    resid_s, amp_s, mask = _residual_and_amplitude(
        imgp, peaks_p, p["mask_frac"], p["dc_frac"], smooth_px)
    resid_s = resid_s[pad:pad + H, pad:pad + W]
    amp_s = amp_s[pad:pad + H, pad:pad + W]

    # --- 3. validity: pattern must exist locally; exclude seam-affected border ---
    valid = amp_s > p["amp_valid_frac"] * np.median(amp_s)
    if b > 0:
        valid[:b] = valid[-b:] = valid[:, :b] = valid[:, -b:] = False
    out["valid_fraction"] = round(float(valid.mean()), 3)
    if not valid.any():
        out["note"] = "No region passes the pattern-validity mask."
        return out

    # --- 4. phase-randomized null -> threshold ---
    rng = np.random.default_rng(0)
    Fraw = np.fft.fftshift(np.fft.fft2(imgp))
    phase = np.angle(np.fft.fftshift(np.fft.fft2(
        rng.standard_normal((Hp, Wp)))))
    Fn = np.abs(Fraw) * np.exp(1j * phase)
    resid_n = ndi.gaussian_filter(
        np.real(np.fft.ifft2(np.fft.ifftshift(Fn * (1.0 - mask)))), smooth_px)
    resid_n = resid_n[pad:pad + H, pad:pad + W]
    nv = resid_n[valid]
    null_sigma = 1.4826 * float(np.median(np.abs(nv - np.median(nv)))) + 1e-12
    thr = max(float(np.percentile(np.abs(nv), p["null_percentile"])),
              p["k_min"] * null_sigma)

    # --- 5. candidate extraction ---
    hot = (np.abs(resid_s) > thr) & valid
    # a real (period-smoothed) defect is an extended blob; correlated-noise
    # specks that graze the threshold are smaller than a fraction of a cell
    min_area = max(4, int(round(p["min_area_frac"] * period ** 2)))
    lab, nlab = ndi.label(hot)
    if nlab:
        areas = ndi.sum_labels(np.ones_like(lab), lab, np.arange(1, nlab + 1))
    cands = []
    amp_med = float(np.median(amp_s[valid]))
    for i in range(1, nlab + 1):
        if areas[i - 1] < min_area:
            continue
        m = lab == i
        a = np.where(m, np.abs(resid_s), 0)
        py, px = np.unravel_index(np.argmax(a), a.shape)
        val = float(resid_s[py, px])
        cands.append(dict(
            y=int(py), x=int(px),
            sign="excess" if val > 0 else "deficit",
            strength_sigma=round(abs(val) / null_sigma, 1),
            area_px=int(areas[i - 1]),
            coherence_dip=bool(amp_s[py, px] < p["dip_frac"] * amp_med),
        ))
    # merge candidates closer than merge_frac * period (one defect, split blob)
    cands.sort(key=lambda c: -c["strength_sigma"])
    merged = []
    for c in cands:
        if all(np.hypot(c["y"] - m["y"], c["x"] - m["x"])
               > p["merge_frac"] * period for m in merged):
            merged.append(c)
    truncated = len(merged) > int(p["max_defects"])
    merged = merged[: int(p["max_defects"])]
    if pixel_size_nm:
        for c in merged:
            c["y_nm"] = round(c["y"] * float(pixel_size_nm), 3)
            c["x_nm"] = round(c["x"] * float(pixel_size_nm), 3)

    # --- 6. summary stats + honesty flags ---
    n_sites = float(valid.sum()) / max(period ** 2, 1.0)   # ~1 site / cell
    anomaly_frac = float((hot & valid).sum()) / float(valid.sum())
    warnings = []
    if truncated:
        warnings.append(f"More than max_defects={p['max_defects']} candidates; "
                        "list truncated to the strongest.")
    if anomaly_frac > 0.05:
        warnings.append(
            f"Anomalous area fraction {anomaly_frac:.1%} — this is dense "
            "disorder or a second phase, not sparse point defects; treat "
            "per-site candidates with caution and consider domain mapping "
            "(fourier_reflection_map) instead.")
    if out["valid_fraction"] < 0.3:
        warnings.append("Pattern occupies <30% of the frame; defect statistics "
                        "cover only the valid region.")
    if len(merged) >= 5:
        # Clark-Evans: observed mean nearest-neighbour distance over the one
        # expected for the same count scattered randomly in the valid area.
        # Sparse point defects score ~1; a second phase / extended defect /
        # interface decorated with candidates scores far below.
        pts = np.array([[c["y"], c["x"]] for c in merged], float)
        d2 = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                      pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d2, np.inf)
        nn = d2.min(axis=1).mean()
        A, n = float(valid.sum()), len(merged)
        out["clark_evans_index"] = round(float(nn / (0.5 * np.sqrt(A / n))), 2)
        # single-linkage clustering at a couple of periods: independent point
        # defects scatter into many small clusters; one extended feature
        # (precipitate, boundary, contamination patch) decorated with anomaly
        # points chains most candidates into a single cluster. This is robust
        # where Clark-Evans is not: the merge step enforces a minimum spacing
        # (spatial inhibition) that inflates the CE index inside a cluster.
        link = 2.5 * period
        comp = list(range(n))
        def root(i):
            while comp[i] != i:
                comp[i] = comp[comp[i]]
                i = comp[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                if d2[i, j] < link:
                    comp[root(i)] = root(j)
        sizes = np.bincount([root(i) for i in range(n)])
        big = int(sizes.max())
        out["largest_cluster_fraction"] = round(big / n, 2)
        if n >= 10 and big / n > 0.5:
            warnings.append(
                f"{big} of {n} candidates form one connected cluster — this "
                "is one extended feature (second phase, precipitate, boundary, "
                "contamination patch) decorated with anomaly points, not a "
                "population of independent point defects; map it as a domain "
                "(fourier_reflection_map) instead of counting sites.")
    out.update(
        n_defects=len(merged),
        defects=merged,
        defects_per_1000_sites=round(1000.0 * len(merged) / max(n_sites, 1.0), 2),
        n_sites_estimate=int(n_sites),
        anomaly_area_fraction=round(anomaly_frac, 4),
        null_sigma=float(null_sigma),
        threshold=float(thr),
        residual_sigma_map=resid_s / null_sigma,
        lattice_amplitude_map=amp_s,
        valid_mask=valid,
        warnings=warnings,
    )
    return out


# --------------------------------------------------------------------------- #
#  Synthetic generator (testing / agent self-verification)
# --------------------------------------------------------------------------- #
def make_defective_lattice(shape=(512, 512), a_px=14.0, kind="hex",
                           n_vacancies=6, n_dopants=0, dopant_contrast=1.9,
                           n_interstitials=0, atom_sigma_frac=0.22,
                           noise=0.05, seed=0):
    """Gaussian-atom lattice with planted point defects.

    Returns (image, truth) where truth maps defect type -> list of (y, x).
    Defect sites are interior (>= 3 periods from the border) and mutually
    separated by >= 3 periods, so planted positions are unambiguous.
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    if kind == "hex":
        a1 = np.array([a_px, 0.0])
        a2 = np.array([a_px / 2.0, a_px * np.sqrt(3) / 2.0])
    else:
        a1 = np.array([a_px, 0.0])
        a2 = np.array([0.0, a_px])
    nmax = int(max(H, W) / a_px) + 4
    sites = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            x, y = i * a1 + j * a2
            if -a_px <= x < W + a_px and -a_px <= y < H + a_px:
                sites.append((y, x))
    sites = np.array(sites)

    margin = 3 * a_px
    interior = sites[(sites[:, 0] > margin) & (sites[:, 0] < H - margin)
                     & (sites[:, 1] > margin) & (sites[:, 1] < W - margin)]
    total = n_vacancies + n_dopants + n_interstitials
    chosen = []
    for k in rng.permutation(len(interior)):
        if len(chosen) >= total:
            break
        c = interior[k]
        if all(np.hypot(*(c - d)) > 3 * a_px for d in chosen):
            chosen.append(c)
    vac = chosen[:n_vacancies]
    dop = chosen[n_vacancies:n_vacancies + n_dopants]
    inter = [c + np.array([a_px / 2.0, a_px / 2.0])
             for c in chosen[n_vacancies + n_dopants:]]

    sigma = atom_sigma_frac * a_px
    img = np.zeros((H, W))
    R = int(np.ceil(3.5 * sigma))

    def stamp(y, x, w=1.0):
        y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
        x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += w * np.exp(
            -((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))

    vac_set = {tuple(np.round(v, 3)) for v in vac}
    dop_set = {tuple(np.round(d, 3)) for d in dop}
    for s in sites:
        key = tuple(np.round(s, 3))
        if key in vac_set:
            continue
        stamp(*s, w=dopant_contrast if key in dop_set else 1.0)
    for c in inter:
        stamp(*c)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    truth = {"vacancies": [tuple(map(float, v)) for v in vac],
             "dopants": [tuple(map(float, d)) for d in dop],
             "interstitials": [tuple(map(float, c)) for c in inter]}
    return img, truth


from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="fft_defect_map",
    description=(
        "FFT-based point-defect / periodicity-anomaly mapping for ANY periodic "
        "image (atomic lattice, self-assembled particle array, moire, ...). "
        "Reconstructs the perfect pattern from its significant reflections by "
        "masked inverse FFT and maps localized deviations against a "
        "phase-randomized noise null — vacancies / missing units appear as "
        "'deficit' anomalies, dopants / adatoms / extra units as 'excess'. "
        "Needs no atom finding, so it works at noise levels and fields of view "
        "where per-column detection is unreliable. Expects a clean input: "
        "SQUARE pixels and (for scan-probe data) a leveled background — both "
        "are upstream preprocessing, NOT done here."
    ),
    import_line="from scilink.skills._shared.fft_defect import fft_defect_map",
    signature="fft_defect_map(image_array, pixel_size_nm=None, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Use to FIND SPARSE POINT DEFECTS / localized anomalies in a PERIODIC "
        "pattern: vacancies or dopant columns in an atomic-resolution (S)TEM / "
        "HRTEM / STM lattice, missing or extra particles in a self-assembled "
        "array, localized faults in any repeating pattern with several periods "
        "in view. Complements real-space defect search (atom finding vs ideal "
        "lattice): prefer this route when columns are hard to detect reliably "
        "(noisy / low-dose / large field of view) or as an independent "
        "cross-check. NOT for dense disorder, extended defects or domains — "
        "for ordered-domain / second-phase LOCALIZATION use "
        "fourier_reflection_map; for quantitative strain use gpa_strain_map. "
        "Candidates are periodicity anomalies, not typed defects: confirm type "
        "(vacancy vs dopant vs contamination) on a real-space crop around each "
        "returned (y, x), using each candidate's sign and coherence_dip flag."
    ),
    parameters={
        "image_array": {"type": "ndarray",
                        "description": "2D grayscale (or HxWx3) array on SQUARE "
                        "pixels. Resample anisotropic frames (e.g. 256x512 of a "
                        "square scan) to square pixels first, and for AFM/STM/"
                        "cAFM apply the skill's row-alignment + leveling first — "
                        "this tool does no leveling or resampling."},
        "pixel_size_nm": {"type": "number",
                          "description": "Optional nm/pixel calibration; adds "
                                         "nm-valued coordinate fields."},
        "params": {"type": "object", "description": (
            "Tunable knobs — every default is a starting point; adjust from "
            "the result vs the image and re-run. Detection gates: "
            "* min_peak_snr (8.0): reflection significance gate; LOWER toward "
            "4-6 if a clearly visible lattice comes back periodic=False (the "
            "note reports the best candidate snr so you can see the margin); "
            "RAISE if spurious low-frequency 'reflections' pass. "
            "* n_reflections (12): max reflections searched; raise for "
            "patterns with many strong families (large unit cell, moire). "
            "Anomaly threshold: * null_percentile (99.9) and k_min (4.0) — "
            "threshold = max(percentile of the phase-randomized null, k_min * "
            "null sigma); RAISE either to keep only the strongest candidates, "
            "LOWER (e.g. k_min=3) if planted/known defects are missed. "
            "Scales (fractions of the measured pattern period, so they "
            "transfer across magnification): * smooth_frac (0.30): residual "
            "integration scale; raise toward 0.5 for blob-like defects spread "
            "over a whole cell, lower for sub-cell sharpness. * min_area_frac "
            "(0.04): minimum anomaly area as a fraction of period^2; raise to "
            "reject speckle. * merge_frac (0.75): candidates closer than this "
            "x period merge into one. * border_periods (2.0): excluded frame "
            "edge. Reconstruction: * mask_frac (0.15): Bragg mask width as a "
            "fraction of |g| — WIDEN (0.2-0.25) if smooth strain/bending "
            "lights up as false anomalies (wider mask tracks slow lattice "
            "variation), NARROW if real defects get absorbed into the "
            "reconstruction. * dc_frac (0.5): low-pass lobe vs the shortest "
            "|g|; raise if slow thickness/illumination gradients leak into "
            "the residual. Masks/limits: * amp_valid_frac (0.35): "
            "pattern-validity gate — lower if a weak-but-real lattice region "
            "is excluded (check valid_fraction), raise to confine the search "
            "to the strongest-pattern area. * dip_frac (0.85): coherence-dip "
            "flag level. * max_defects (200). * edge_pad (True): mirror-pad "
            "before the FFT to suppress border artifacts; disable only for "
            "speed on very large images.")},
    },
    required=["image_array"],
    returns=(
        "dict with: 'periodic' (bool — False means <2 significant reflections; "
        "'note' then reports the best candidate snr vs the gate so you can "
        "decide whether to lower min_peak_snr or conclude the image is not "
        "periodic), 'reflections' (ALL candidates: period_px, snr, "
        "significant), 'pattern_period_px' (and _nm), 'n_defects', 'defects' "
        "(list of {y, x, sign='deficit'|'excess', strength_sigma, area_px, "
        "coherence_dip} — deficit + coherence_dip is the vacancy / missing-"
        "unit signature; excess without dip suggests an adatom/dopant on an "
        "intact lattice), 'defects_per_1000_sites', 'anomaly_area_fraction', "
        "'valid_fraction', 'clark_evans_index' and 'largest_cluster_fraction' "
        "(candidate spatial statistics), 'residual_sigma_map' / "
        "'lattice_amplitude_map' / 'valid_mask' (arrays for overlay plots), "
        "'threshold', 'warnings' — HEED THESE: a dense-disorder flag "
        "(anomaly fraction too high for sparse defects) or a single-cluster "
        "flag (candidates trace ONE extended feature — precipitate, boundary, "
        "contamination — not independent point defects). Note one structural "
        "blind spot: a coherent boundary between twin variants whose "
        "reflections are BOTH significant is part of the reconstructed "
        "pattern and will NOT appear as an anomaly."
    ),
    example=(
        "res = fft_defect_map(image, pixel_size_nm=0.02)\n"
        "if res['periodic']:\n"
        "    print(res['n_defects'], 'candidates;',\n"
        "          res['defects_per_1000_sites'], 'per 1000 sites')\n"
        "    for d in res['defects'][:10]:\n"
        "        print(d['y'], d['x'], d['sign'], d['strength_sigma'],\n"
        "              'dip' if d['coherence_dip'] else '')\n"
        "    # confirm type on a real-space crop around each (y, x)"
    ),
)
