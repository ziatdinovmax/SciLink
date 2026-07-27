# Auto-generated Script
# Task: At each pixel, fit the dominant low-energy plasmon (LSPR) feature in the 0.2-1.13 eV EELS spectrum with a single Lorentzian (or Voigt) peak model using lmfit's LorentzianModel/VoigtModel over the ~0.3-0.75 eV window where Components 1 and 2 place their maxima. Extract the peak center energy (LSPR resonance energy), the FWHM (linewidth, inversely related to plasmon damping / carrier mobility), and the peak amplitude/integrated area. Produce spatial maps of each so the continuous plasmon-energy shift responsible for the split between PCA Components 1 and 2 can be visualized directly across the nanocrystal array, and gate fits on adequate signal so low-flux background pixels are masked rather than fit to noise.

def analyze_feature(data, axis, reconstruction=None):
    import scipy.integrate
    import scipy.ndimage

    axis = np.asarray(axis, dtype=float)
    H, W, E = data.shape
    n_pix = H * W

    raw = np.asarray(data, dtype=float).reshape(n_pix, E)
    if reconstruction is not None:
        recon = np.asarray(reconstruction, dtype=float).reshape(n_pix, E)
    else:
        recon = None

    # ---- fitting window ~0.3-0.75 eV where LSPR maxima sit ----
    lo, hi = 0.30, 0.75
    win = (axis >= lo) & (axis <= hi)
    if win.sum() < 5:
        # fall back to full range
        win = np.ones_like(axis, dtype=bool)
    x = axis[win]

    # ---- signal gate: use raw integrated intensity in window ----
    raw_win = raw[:, win]
    # light smoothing of raw for gating stability
    raw_area = scipy.integrate.trapezoid(np.clip(raw_win, 0, None), x, axis=1)
    finite_area = raw_area[np.isfinite(raw_area)]
    if finite_area.size:
        thr = np.nanpercentile(finite_area, 20)
    else:
        thr = 0.0
    # also require decent SNR: peak vs noise estimate
    noise = np.nanstd(np.diff(raw_win, axis=1), axis=1) / np.sqrt(2.0)
    peakval = np.nanmax(raw_win, axis=1)
    snr = np.where(noise > 0, peakval / noise, 0.0)
    good_mask = (raw_area > thr) & (snr > 2.0) & np.isfinite(raw_area)

    # ---- choose source for SHAPE fitting: reconstruction if available ----
    if recon is not None:
        shape_src = recon[:, win]
    else:
        shape_src = raw_win

    # per-pixel light smoothing along energy for convergence
    def smooth_row(row):
        return scipy.ndimage.gaussian_filter1d(row, sigma=1.5, mode='nearest')

    # ---- Lorentzian model for curve_fit (fast) ----
    def lorentzian(xx, amp, cen, sigma):
        # amp is integrated area; sigma is HWHM
        return (amp / np.pi) * (sigma / ((xx - cen) ** 2 + sigma ** 2))

    center_map = np.full(n_pix, np.nan)
    fwhm_map = np.full(n_pix, np.nan)
    amp_map = np.full(n_pix, np.nan)

    dx = np.mean(np.diff(x))
    x_lo, x_hi = x[0], x[-1]

    for p in range(n_pix):
        if not good_mask[p]:
            continue
        y = shape_src[p].astype(float)
        if not np.all(np.isfinite(y)):
            continue
        ys = smooth_row(y)
        # baseline: subtract linear background from window edges
        b0 = np.mean(ys[:3])
        b1 = np.mean(ys[-3:])
        base = b0 + (b1 - b0) * (x - x_lo) / (x_hi - x_lo)
        yc = ys - base
        # ensure positive-ish peak
        pk = np.nanmax(yc)
        if not np.isfinite(pk) or pk <= 0:
            continue
        cen0 = x[int(np.argmax(yc))]
        sigma0 = 0.08
        amp0 = pk * np.pi * sigma0

        try:
            popt, _ = curve_fit(
                lorentzian, x, yc,
                p0=[amp0, cen0, sigma0],
                bounds=([0.0, x_lo, 0.005],
                        [np.inf, x_hi, 0.5]),
                maxfev=5000,
            )
        except Exception:
            continue

        amp_fit, cen_fit, sigma_fit = popt
        fwhm_fit = 2.0 * sigma_fit

        # reject rail-gazing / bad fits
        if cen_fit <= x_lo + 0.5 * dx or cen_fit >= x_hi - 0.5 * dx:
            continue
        if fwhm_fit <= 0.011 or fwhm_fit >= 0.99:
            continue

        center_map[p] = cen_fit
        fwhm_map[p] = fwhm_fit

        # ---- amplitude / integrated area from RAW data (quantification) ----
        yraw = raw[p, win].astype(float)
        # baseline-subtract raw with same linear scheme
        rb0 = np.mean(yraw[:3])
        rb1 = np.mean(yraw[-3:])
        rbase = rb0 + (rb1 - rb0) * (x - x_lo) / (x_hi - x_lo)
        yraw_c = np.clip(yraw - rbase, 0, None)
        area = scipy.integrate.trapezoid(yraw_c, x)
        amp_map[p] = area

    # reshape
    center_map = center_map.reshape(H, W)
    fwhm_map = fwhm_map.reshape(H, W)
    amp_map = amp_map.reshape(H, W)

    # light spatial median cleanup of salt-and-pepper (preserve NaN mask)
    def clean_map(m):
        out = m.copy()
        finite = np.isfinite(m)
        if finite.sum() < 5:
            return out
        filled = np.where(finite, m, np.nanmedian(m[finite]))
        med = scipy.ndimage.median_filter(filled, size=3)
        # replace extreme outliers only
        resid = np.abs(filled - med)
        sc = np.nanstd(resid[finite]) if finite.sum() > 1 else 0.0
        if sc > 0:
            bad = (resid > 4.0 * sc) & finite
            out[bad] = med[bad]
        return out

    center_map = clean_map(center_map)
    fwhm_map = clean_map(fwhm_map)
    amp_map = clean_map(amp_map)

    maps = {
        "Peak_Center": center_map,
        "FWHM": fwhm_map,
        "Peak_Amplitude": amp_map,
    }
    units = {
        "Peak_Center": "eV",
        "FWHM": "eV",
        "Peak_Amplitude": "a.u.",
    }
    description = (
        "Per-pixel single-Lorentzian fit of the low-energy LSPR plasmon in the "
        "0.30-0.75 eV EELS window. Shape parameters (center = LSPR resonance "
        "energy, FWHM = 2*HWHM ~ plasmon damping/inverse carrier mobility) are "
        "fit on the denoised reconstruction when available for convergence on "
        "the low-SNR (~3.6) data, while the integrated peak area (amplitude) is "
        "measured on the RAW data for quantitative fidelity. Pixels with "
        "insufficient integrated signal or SNR<2 are gated out (NaN) rather than "
        "fit to noise. The continuous spatial shift in Peak_Center visualizes "
        "the plasmon-energy variation underlying the PCA Component 1/2 split."
    )

    return {"maps": maps, "units": units, "description": description}