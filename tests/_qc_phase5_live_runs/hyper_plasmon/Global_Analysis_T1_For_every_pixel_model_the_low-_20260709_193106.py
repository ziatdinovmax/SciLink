# Auto-generated Script
# Task: For every pixel, model the low-loss EELS spectrum (0.2-1.13 eV) as a single broad LSPR resonance on a smooth background. Use lmfit with a Lorentzian (or Gaussian if better constrained) peak model plus a low-order polynomial/power-law background to capture the tail, fitting per pixel. Extract (1) the resonance center energy in eV as the LSPR peak energy map, and (2) the fitted peak amplitude/integrated area as the plasmon intensity map. Gate fits on sufficient signal and flag pixels where the peak is not measurable. Report the peak center and intensity distributions so the spatial structure across the nanocrystal array can be assessed against the histogram (expect a bell-shaped, physically reasonable distribution, not a spike at zero or the window edge).

def analyze_feature(data, axis):
    import scipy.integrate

    ny, nx, ne = data.shape
    npix = ny * nx
    X = np.asarray(axis, dtype=float)
    spec = data.reshape(npix, ne).astype(float)

    # --- Basic cleaning: replace non-finite, clip negatives lightly ---
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

    # Light spectral smoothing per pixel to stabilize fits (preserve broad LSPR)
    # Use gaussian_filter along energy axis only.
    spec_sm = gaussian_filter(spec, sigma=(0, 2.0))

    # Estimate per-pixel noise from high-frequency residual
    noise = np.std(spec - spec_sm, axis=1) + 1e-9

    # Signal metric: peak-to-baseline amplitude
    smin = np.min(spec_sm, axis=1)
    smax = np.max(spec_sm, axis=1)
    sig_amp = smax - smin
    snr_pix = sig_amp / noise

    # Output maps
    peak_energy = np.full(npix, np.nan)
    plasmon_int = np.full(npix, np.nan)
    peak_width = np.full(npix, np.nan)
    measurable = np.zeros(npix, dtype=bool)

    x0 = X[0]
    x1 = X[-1]
    span = x1 - x0

    # Model: Lorentzian peak + linear background
    # y = bg0 + bg1*(x-x0) + A * (sigma^2 / ((x-center)^2 + sigma^2))
    def model(x, bg0, bg1, amp, center, sigma):
        return bg0 + bg1 * (x - x0) + amp * (sigma * sigma) / ((x - center) ** 2 + sigma * sigma)

    # Gate threshold on SNR
    snr_gate = 2.5

    for i in range(npix):
        if snr_pix[i] < snr_gate:
            continue
        y = spec_sm[i]

        # Initial guesses
        # Background from edges
        bg0_g = y[0]
        bg1_g = (y[-1] - y[0]) / span
        base_line = bg0_g + bg1_g * (X - x0)
        resid = y - base_line
        # Peak position from argmax of residual
        pk_idx = int(np.argmax(resid))
        center_g = X[pk_idx]
        amp_g = resid[pk_idx]
        if amp_g <= 0:
            amp_g = sig_amp[i]
            center_g = X[int(np.argmax(y))]
        sigma_g = 0.15  # broad LSPR guess in eV

        p0 = [bg0_g, bg1_g, max(amp_g, 1e-6), center_g, sigma_g]

        # Bounds: center within window (slightly padded), sigma physical, amp positive
        lb = [-np.inf, -np.inf, 0.0, x0, 0.01]
        ub = [np.inf, np.inf, np.inf, x1, span]

        try:
            popt, pcov = curve_fit(model, X, y, p0=p0, bounds=(lb, ub), maxfev=4000)
        except Exception:
            continue

        bg0_f, bg1_f, amp_f, center_f, sigma_f = popt

        # Quality checks: reject rail-gazing at center bounds & degenerate fits
        edge_tol = 0.02 * span
        if center_f <= x0 + edge_tol or center_f >= x1 - edge_tol:
            # peak not well localized inside window
            continue
        if amp_f <= 0 or not np.isfinite(amp_f):
            continue
        if sigma_f <= 0.011 or sigma_f >= 0.99 * span:
            continue

        # Fit residual quality
        yfit = model(X, *popt)
        ss_res = np.sum((y - yfit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        if r2 < 0.5:
            continue

        # Integrated area of Lorentzian peak component = pi * amp * sigma
        peak_only = amp_f * (sigma_f * sigma_f) / ((X - center_f) ** 2 + sigma_f * sigma_f)
        area = scipy.integrate.trapezoid(peak_only, X)

        peak_energy[i] = center_f
        plasmon_int[i] = area
        peak_width[i] = 2.0 * sigma_f  # FWHM = 2*sigma for this Lorentzian param
        measurable[i] = True

    LSPR_Peak_Energy = peak_energy.reshape(ny, nx)
    Plasmon_Intensity = plasmon_int.reshape(ny, nx)
    Peak_FWHM = peak_width.reshape(ny, nx)
    Measurable = measurable.reshape(ny, nx)

    return {
        "maps": {
            "LSPR_Peak_Energy": LSPR_Peak_Energy,
            "Plasmon_Intensity": Plasmon_Intensity,
            "Peak_FWHM": Peak_FWHM,
            "Measurable": Measurable.astype(float),
        },
        "units": {
            "LSPR_Peak_Energy": "eV",
            "Plasmon_Intensity": "a.u. (integrated area)",
            "Peak_FWHM": "eV",
            "Measurable": "bool",
        },
        "description": "Per-pixel low-loss EELS LSPR modeled as a single Lorentzian resonance on a linear background via scipy curve_fit. Light Gaussian smoothing (sigma=2 channels) stabilizes noisy (SNR~3.6) spectra without erasing the broad plasmon. Initial center guessed from the argmax of the background-subtracted spectrum; bounds keep the center inside the 0.20-1.13 eV window and sigma physical. LSPR_Peak_Energy = fitted Lorentzian center (eV); Plasmon_Intensity = integrated peak area (pi*amp*sigma). Pixels are gated on SNR>2.5 and rejected if the fit rails at the window edge, has non-positive amplitude, degenerate width, or R^2<0.5 (flagged NaN / Measurable=False), yielding a physically reasonable bell-shaped distribution rather than an edge spike."
    }