"""Targeted reciprocal-space (Fourier) mapping of a single lattice reflection.

The sophisticated counterpart to the exploratory ``run_fft_nmf_analysis``:
given a crystalline (S)TEM / HRTEM image, it (1) detects reflections from a
detrended, azimuthally-averaged radial power spectrum with a significance
(sigma), (2) classifies satellite / superstructure reflections by their ratio
to the fundamental, (3) builds a matched annular band-pass at a chosen
reflection and maps WHERE it lives (amplitude) plus a raw displacement-phase
view (the Bragg phase — UNREFERENCED, an exploratory displacement view, NOT a
quantitative strain tensor; use the ``gpa_strain`` tool for referenced strain),
(4) gates the map against a phase-randomized null, and
(5) confirms a candidate domain with a local windowed-FFT spot SNR vs. bulk.

These steps encode the failure modes that make ad-hoc versions wrong:
  * detection uses the azimuthally-AVERAGED, DETRENDED spectrum (a per-tile
    peak/median is fooled by single noise spikes);
  * the amplitude/phase MAP uses the un-windowed image (a spatial Hann window
    tapers the edges and biases amplitude toward the image centre);
  * a phase-randomized null sets the threshold so noise and sharp edges (an
    interface injects broadband power) are not flagged as ordered.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


def _to_gray(a):
    a = np.asarray(a, float)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a


def _radial_psd(img):
    """Azimuthally-averaged power spectrum. Returns (freq_cycles_per_px, power)."""
    H, W = img.shape
    win = np.outer(np.hanning(H), np.hanning(W))      # window: detection ONLY
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    ps = np.abs(F) ** 2
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy).astype(int)
    radial = np.bincount(r.ravel(), ps.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    freq = np.arange(len(radial)) / W                 # cycles / px
    return freq, radial


def _spot_snr(crop, target_d_nm, pixel_size_nm):
    n = min(crop.shape)
    if n < 32:
        return float("nan")
    crop = crop[:n, :n].astype(float)
    hn = np.outer(np.hanning(n), np.hanning(n))
    Fm = np.abs(np.fft.fftshift(np.fft.fft2((crop - crop.mean()) * hn)))
    c0 = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c0, yy - c0)
    rk = pixel_size_nm / target_d_nm * n
    band = (rr > 0.85 * rk) & (rr < 1.15 * rk)
    if not band.any():
        return float("nan")
    return float(Fm[band].max() / (np.median(Fm[rr > 10]) + 1e-12))


def fourier_reflection_map(image_array, pixel_size_nm, d_nm=None, params=None):
    """Detect and spatially map a lattice reflection (satellite / superstructure).

    Args:
        image_array: 2D grayscale (or HxWx3) array. If pixels are anisotropic,
            resample to square physical pixels BEFORE calling.
        pixel_size_nm: physical pixel size in nm (per axis; square pixels).
        d_nm: optional reflection to map (nm). If None, the strongest
            significant SUPERSTRUCTURE is mapped, else the strongest reflection.
        params: optional dict — bandwidth_frac (0.10), smooth_nm (1.2),
            null_percentile (99.5), d_range ((0.15, 2.0) nm), min_sigma (3.0).

    Returns: dict (see TOOL_SPEC.returns).
    """
    p = params or {}
    bw = float(p.get("bandwidth_frac", 0.10))
    smooth_nm = float(p.get("smooth_nm", 1.2))
    null_pct = float(p.get("null_percentile", 99.5))
    d_min, d_max = p.get("d_range", (0.15, 2.0))
    min_sigma = float(p.get("min_sigma", 3.0))

    img = _to_gray(image_array)
    H, W = img.shape

    # --- 1. detect reflections (detrended, azimuthally-averaged radial PSD) ---
    freq, radial = _radial_psd(img)
    with np.errstate(divide="ignore"):
        d = np.where(freq > 0, pixel_size_nm / np.maximum(freq, 1e-12), np.inf)
    logp = np.log(radial + 1e-12)
    resid = logp - gaussian_filter1d(logp, 15)
    valid = (d > d_min) & (d < d_max) & np.isfinite(d)
    base = resid[(d > 0.6) & (d < min(1.5, d_max)) & np.isfinite(d)]
    base_std = float(base.std()) if base.size > 5 else float(resid[valid].std() + 1e-9)
    masked = np.where(valid, resid, -np.inf)
    pk, _ = find_peaks(masked, prominence=max(0.08, 1.5 * base_std), distance=4)
    reflections = [{"d_nm": float(d[k]), "freq_cyc_px": float(freq[k]),
                    "sigma": float(resid[k] / (base_std + 1e-12))} for k in pk]
    reflections.sort(key=lambda r: -r["sigma"])

    # Materials-AGNOSTIC harmonic relationships: report, for each reflection,
    # whether it is an integer multiple (N>=2) of a shorter, also-significant
    # reflection. This is a *factual* spacing relationship, NOT a claim about
    # which reflection is "the fundamental lattice" — that distinction needs
    # physics (a 1-D radial PSD cannot tell a true cell from its harmonics),
    # so the tool does not assert it. A reflection that is N x a shorter one is
    # a candidate superstructure/satellite (ordering, antiphase, 2nd phase,
    # moire); confirm with spot_snr and name the structure with domain context.
    mult_tol = float(p.get("multiple_tol", 0.12))
    # The BASE of an integer-multiple relation only needs to be a detected peak
    # (a real, if weak, reflection) — NOT itself highly significant. Requiring the
    # base to clear the full significance floor makes the satellite flag fragile:
    # under heavy downsampling or in small crops the base drops out and a genuinely
    # strong superstructure reflection silently loses its flag (false negative).
    # Significance is enforced on the SATELLITE (and on strongest_satellite below).
    base_min_sigma = float(p.get("base_min_sigma", 1.5))
    for r in reflections:
        rel = []
        for r2 in reflections:
            if r2["d_nm"] < r["d_nm"] * 0.97 and r2["sigma"] >= base_min_sigma:
                m = r["d_nm"] / r2["d_nm"]
                if m >= 1.5 and abs(m - round(m)) < mult_tol:
                    rel.append({"base_d_nm": round(r2["d_nm"], 4), "multiple": int(round(m))})
        r["integer_multiple_of"] = rel
        r["is_satellite_candidate"] = bool(rel)

    # Strongest satellite (by sigma) — the unambiguous, physics-light pointer to
    # the ordering/superstructure to localize. Avoids having to identify "the
    # fundamental" (ill-posed from a 1-D PSD) or assume a specific multiple N.
    sats = [r for r in reflections
            if r["is_satellite_candidate"] and r["sigma"] >= min_sigma]
    strongest_sat = sats[0]["d_nm"] if sats else None   # reflections sorted by sigma

    out = {"reflections": reflections, "strongest_satellite_d_nm": strongest_sat,
           # Guard against a common misuse: these are reflection SPACINGS, not
           # real-space lattice vectors. Pairing two of them as cell axes (e.g.
           # a {110} face-diagonal at ~a/sqrt2 with a {100} axial spot) gives a
           # spurious ~45deg "oblique" cell. For the unit cell / lattice
           # constant / inter-axis angle, use measure_lattice_constant instead.
           "cell_note": ("reflection spacings, NOT lattice vectors — do not pair "
                         "them as cell axes; use measure_lattice_constant for the "
                         "unit cell / lattice constant / angle")}

    # --- choose which reflection to map ---
    # Default = STRONGEST significant reflection (general, defensible: the main
    # lattice signal). It is deliberately NOT "the longest period / a
    # superstructure" — that would bake in an expectation. To localize a
    # superstructure/satellite, read it off `reflections` (flagged
    # is_satellite_candidate) and call again with that d_nm. If nothing clears
    # the significance floor, map nothing and say so (honest negative).
    sig = [r for r in reflections if r["sigma"] >= min_sigma]
    if d_nm is not None:
        target = float(d_nm)
    elif sig:
        target = sig[0]["d_nm"]            # reflections sorted by sigma desc
    else:
        target = None
    out["mapped_d_nm"] = target
    if target is None:
        out["note"] = ("No reflection above the significance floor (min_sigma="
                       f"{min_sigma}) — no resolvable lattice/superstructure.")
        return out

    # --- 3. matched band-pass amplitude + phase map (NO spatial window) ---
    f0 = pixel_size_nm / target
    r0 = f0 * W
    dr = bw * r0
    Fr = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    rpx = np.hypot(xx - cx, yy - cy)
    mask = np.exp(-0.5 * ((rpx - r0) / max(dr, 1e-6)) ** 2)
    comp = np.fft.ifft2(np.fft.ifftshift(Fr * mask))
    sm = max(3, int(round(smooth_nm / pixel_size_nm)))
    amp = np.sqrt(gaussian_filter(np.real(comp) ** 2, sm))
    phase = np.angle(comp)

    # --- 4. phase-randomized null -> threshold ---
    rng = np.random.default_rng(0)
    rand_phase = np.angle(np.fft.fftshift(np.fft.fft2(rng.standard_normal((H, W)))))
    Fn = np.abs(Fr) * np.exp(1j * rand_phase)
    ampn = np.sqrt(gaussian_filter(np.real(np.fft.ifft2(np.fft.ifftshift(Fn * mask))) ** 2, sm))
    thr = float(np.percentile(ampn, null_pct))
    domain = amp > thr

    # --- 5. local-FFT confirmation: spot SNR in domain vs bulk ---
    win = int(min(256, H // 4, W // 4))
    b = win
    def crop_at(y, x):
        y = int(np.clip(y, b, H - b)); x = int(np.clip(x, b, W - b))
        return img[y - win // 2:y + win // 2, x - win // 2:x + win // 2]
    ai = amp.copy(); ai[:b] = 0; ai[-b:] = 0; ai[:, :b] = 0; ai[:, -b:] = 0
    dy, dx = np.unravel_index(np.argmax(ai), ai.shape)
    al = amp.copy(); al[:b] = np.inf; al[-b:] = np.inf; al[:, :b] = np.inf; al[:, -b:] = np.inf
    by, bx = np.unravel_index(np.argmin(al), al.shape)
    snr_dom = _spot_snr(crop_at(dy, dx), target, pixel_size_nm)
    snr_bulk = _spot_snr(crop_at(by, bx), target, pixel_size_nm)

    mapped_rel = next((r["integer_multiple_of"] for r in reflections
                       if abs(r["d_nm"] - target) < 1e-6), [])
    out.update(
        amplitude_map=amp, phase_map=phase, domain_mask=domain,
        null_threshold=thr, domain_fraction=float(domain.mean()),
        spot_snr_domain=snr_dom, spot_snr_bulk=snr_bulk,
        mapped_is_satellite_candidate=bool(mapped_rel),
        mapped_integer_multiple_of=mapped_rel,
    )
    return out


from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="fourier_reflection_map",
    description=(
        "Targeted reciprocal-space mapping of ONE lattice reflection. Detects "
        "reflections from a detrended radial power spectrum, flags satellite/"
        "superstructure reflections, and maps where a chosen reflection lives "
        "(amplitude, null-gated) plus a raw displacement-phase view. For a "
        "QUANTITATIVE referenced strain tensor (exx/eyy/exy/wxy), use the "
        "gpa_strain tool instead — this tool's phase is unreferenced."
    ),
    import_line="from scilink.skills._shared.fourier_reflection import fourier_reflection_map",
    signature="fourier_reflection_map(image_array, pixel_size_nm, d_nm=None, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Use for a SPECIFIC periodicity question on a crystalline (S)TEM / HRTEM "
        "lattice image: localizing a superlattice / satellite reflection, an "
        "ordered domain (oxygen/cation-vacancy, charge/orbital ordering, "
        "antiphase), or a second phase with a distinct spacing. (For a "
        "quantitative referenced strain field, use gpa_strain, not this tool's "
        "raw phase.) This is the sharp, interpretable counterpart to the "
        "exploratory run_fft_nmf_analysis: reach for FFT-NMF when the "
        "heterogeneity is unknown ('what domains exist?'); reach for "
        "fourier_reflection_map when you can name or first detect the reflection "
        "of interest and want to map WHERE it is.\n"
        "\n"
        "NOT for the unit cell: the returned `reflections` are spacings, not "
        "lattice vectors. Do NOT pick two of them as cell axes — a {110} "
        "face-diagonal (~a/sqrt2) paired with a {100} axial spot yields a "
        "spurious ~45deg oblique cell. For the lattice constant / cell "
        "parameters / inter-axis angle (even as a byproduct), use "
        "measure_lattice_constant.\n"
        "\n"
        "Calibrate first: pass the true square-pixel size (nm/px). If pixels are "
        "anisotropic, resample to square physical pixels before calling.\n"
        "\n"
        "To localize ordering / a superstructure, map "
        "`strongest_satellite_d_nm` — the highest-sigma satellite (a reflection "
        "that is an integer multiple of some shorter one). Do NOT first try to "
        "identify 'the fundamental' or assume a specific multiple N (e.g. "
        "'must be the N=2 satellite') — that is ill-posed from a 1-D PSD and a "
        "real strong satellite (high sigma) will be missed if you hunt for a "
        "particular N. CONFIRM the mapped domain by requiring spot_snr_domain >> "
        "spot_snr_bulk (a discrete local-FFT spot, not an interface-edge "
        "artifact). If 'strongest_satellite_d_nm' is None (no satellite clears "
        "the floor), report 'no resolvable superstructure' rather than "
        "thresholding noise. "
        "State the structural mechanism (e.g. 'vacancy-ordered superstructure') "
        "as CONSISTENT WITH the data — the tool proves a localized reflection at "
        "d = N x fundamental exists, not the specific chemistry."
    ),
    parameters={
        "image_array": {"type": "ndarray", "description": "2D grayscale (or HxWx3) array."},
        "pixel_size_nm": {"type": "float", "description": "Physical pixel size in nm (square pixels)."},
        "d_nm": {"type": "float", "description": (
            "Optional reflection to map, in nm. If omitted, the strongest "
            "significant superstructure is mapped (else the strongest reflection).")},
        "params": {"type": "dict", "description": (
            "Optional. Keys: bandwidth_frac (annulus width as fraction of radius, "
            "default 0.10), smooth_nm (amplitude smoothing length, default 1.2), "
            "null_percentile (default 99.5), d_range (nm tuple, default (0.15, 2.0)), "
            "min_sigma (significance floor to auto-map a satellite, default 3.0).")},
    },
    required=["image_array", "pixel_size_nm"],
    returns=(
        "dict with: "
        "'reflections' (list of {d_nm, freq_cyc_px, sigma, integer_multiple_of, "
        "is_satellite_candidate}, sorted by sigma; `integer_multiple_of` lists "
        "{base_d_nm, multiple} where this reflection is an N x a shorter "
        "significant reflection — a *factual spacing relation*, NOT a claim that "
        "the shorter one is the true fundamental lattice; that needs physics); "
        "'strongest_satellite_d_nm' (float or None — the highest-sigma "
        "satellite candidate; THIS is what to map to localize ordering / a "
        "superstructure: it needs no fundamental identification and no assumed "
        "multiple N); "
        "'mapped_d_nm' (float — the reflection that was mapped; default is the "
        "strongest reflection); "
        "'mapped_is_satellite_candidate' (bool); "
        "'amplitude_map' (2D ndarray — where the mapped reflection lives); "
        "'phase_map' (2D ndarray — raw Bragg phase, an UNREFERENCED "
        "displacement view; not a quantitative strain tensor — use gpa_strain); "
        "'domain_mask' (2D bool — null-gated ordered-domain segmentation); "
        "'domain_fraction' (float); 'null_threshold' (float); "
        "'spot_snr_domain' / 'spot_snr_bulk' (local-FFT spot SNR in the strongest "
        "domain vs. a bulk reference — domain >> bulk confirms a real reflection); "
        "'is_mapped_superstructure' (bool). "
        "If nothing clears the floor: 'note' explains no resolvable reflection "
        "and the map fields are absent."
    ),
    example=(
        "res = fourier_reflection_map(image, pixel_size_nm=0.061)\n"
        "print(res['reflections'][:3], 'mapped', res['mapped_d_nm'])\n"
        "if res.get('domain_mask') is not None:\n"
        "    frac = res['domain_fraction']; ok = res['spot_snr_domain'] > 3*res['spot_snr_bulk']\n"
        "    np.save('superstructure_amplitude.npy', res['amplitude_map'])\n"
        "    # ok==True confirms a real localized superstructure (not an edge artifact)."
    ),
)
