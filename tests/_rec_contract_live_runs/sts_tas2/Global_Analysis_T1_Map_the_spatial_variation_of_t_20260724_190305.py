# Auto-generated Script
# Task: Map the spatial variation of the dominant CDW LDOS peak using per-pixel fitting of the dI/dV spectra via `fit_per_pixel`. Fit the prominent occupied-state peak (the broad feature in the negative-bias region seen in Component 1) with a single Gaussian/Lorentzian to extract its center bias (Peak_Position) at every pixel, quantifying the peak shift that the PCA derivative-like components (2-5) are indirectly encoding. The spatially averaged mean spectrum can seed the initial peak-center guess. As secondary best-effort diagnostics, also return the peak amplitude and a Fermi-level (0 V) dI/dV value as a gap-depth proxy through the ordinary maps return. Focus attention on whether Peak_Position changes across the diagonal domain boundary highlighted in Component 3/4 abundance maps.

def analyze_feature(data, axis, reconstruction=None):
    from scilink.skills._shared.parallel_pixel_fit import fit_per_pixel
    import scipy.ndimage as ndi

    d = np.asarray(data, dtype=float)
    ax = np.asarray(axis, dtype=float)
    H, W, E = d.shape
    npix = H * W

    # Ensure ascending axis
    if ax[0] > ax[-1]:
        order = np.argsort(ax)
        ax = ax[order]
        d = d[:, :, order]
        if reconstruction is not None:
            reconstruction = np.asarray(reconstruction, float)[:, :, order]

    flat = d.reshape(npix, E)

    # ---- Light spectral smoothing for stability (keep feature) ----
    sm = ndi.gaussian_filter1d(flat, sigma=1.5, axis=1)

    # ---- Mean spectrum & noise estimate ----
    mean_spec = sm.mean(axis=0)
    resid = flat - sm
    sigma_pix = np.median(np.std(resid, axis=1))
    sigma_mean = sigma_pix / np.sqrt(npix)

    # ---- Occupied-state (negative bias) window ----
    neg_mask = ax < -0.05
    neg_ax = ax[neg_mask]
    neg_mean = mean_spec[neg_mask]

    # ---- Detrend the negative-bias ramp to expose local HUMPS ----
    # The occupied-state DOS is a monotonic decay toward E_F; the CDW peak
    # is a hump ON TOP of that ramp. Remove a smooth (quadratic) background
    # so a single-Gaussian localizes the actual local maximum, not the ramp.
    # Fit quadratic to the negative-window mean, find residual peak.
    if len(neg_ax) >= 5:
        coef = np.polyfit(neg_ax, neg_mean, 2)
        base = np.polyval(coef, neg_ax)
        detr = neg_mean - base
        # smooth detrended lightly then find peaks
        detr_s = ndi.gaussian_filter1d(detr, 2.0)
        peak_idx = int(np.argmax(detr_s))
        prominence = float(detr_s[peak_idx])
        seed_center = float(neg_ax[peak_idx])
    else:
        prominence = 0.0
        seed_center = -0.6

    # Guard: if the detected hump is at the very edge, fall back to the
    # physically-expected CDW hump region near -0.6 V
    if not (neg_ax.min() + 0.05 < seed_center < -0.10):
        # search restricted interior region for a hump
        interior = (neg_ax > neg_ax.min() + 0.05) & (neg_ax < -0.10)
        if np.any(interior):
            detr_i = (neg_mean - base) if len(neg_ax) >= 5 else neg_mean
            detr_i = detr_i.copy()
            detr_i[~interior] = -np.inf
            pk = int(np.argmax(detr_i))
            seed_center = float(neg_ax[pk])
            prominence = float(max(prominence, (neg_mean - base)[pk] if len(neg_ax) >= 5 else 0.0))
        else:
            seed_center = -0.6

    # ---- MEASURABILITY GATE ----
    snr_mean = prominence / (sigma_mean + 1e-30)

    neg_integ = sm[:, neg_mask].sum(axis=1)
    order_bright = np.argsort(neg_integ)[::-1]
    bright_snr_best = 0.0
    for frac in (0.05, 0.01, 0.002):
        nb = max(200, int(frac * npix))
        sel = order_bright[:nb]
        bspec = sm[sel][:, neg_mask].mean(axis=0)
        if len(neg_ax) >= 5:
            bc = np.polyfit(neg_ax, bspec, 2)
            bl = np.polyval(bc, neg_ax)
        else:
            bl = np.linspace(bspec[0], bspec[-1], len(neg_ax))
        bprom = np.max(bspec - bl)
        bsnr = bprom / (sigma_pix / np.sqrt(nb) + 1e-30)
        bright_snr_best = max(bright_snr_best, bsnr)

    if (snr_mean < 5.0) and (bright_snr_best < 5.0) and (prominence <= 3 * sigma_mean):
        return {
            "maps": {},
            "not_measurable": {
                "feature": "Occupied-state CDW LDOS peak position (Peak_Position) via per-pixel Gaussian fit of negative-bias dI/dV",
                "evidence": (f"Detrended negative-bias hump prominence={prominence:.4g} a.u.; "
                             f"sigma_pixel={sigma_pix:.4g}, sigma_mean(N={npix})={sigma_mean:.4g}; "
                             f"mean-spectrum SNR={snr_mean:.2f}, best bright-region SNR={bright_snr_best:.2f}. "
                             f"Feature does not clear ~5-sigma in field mean nor at any bright fraction tested (5%/1%/0.2%)."),
                "description": "Occupied-state dI/dV peak is statistically flat vs noise; not mappable."
            }
        }

    # ---- Per-pixel SNR / resolution check ----
    per_pix_snr = prominence / (sigma_pix + 1e-30)
    fit_source = d
    bin_note = ""
    if reconstruction is not None and per_pix_snr < 3.0:
        fit_source = reconstruction  # denoised shape source for position
        bin_note = "Used reconstruction (denoised) for peak-position shape due to low per-pixel SNR."

    # ---- Fit mask: fit all pixels ----
    fit_mask = np.ones((H, W), dtype=bool)

    # ---- TIGHT window bracketing the actual hump (fixes edge-railing) ----
    # Bracket the seed hump with a narrow window so the Gaussian center
    # cannot rail to -1.0 V or to the E_F clamp. Use a quadratic background
    # so the monotonic DOS decay is absorbed by the background, NOT the peak.
    half = 0.18  # +/- window around the detected hump center
    lo = max(float(neg_ax.min()), seed_center - half)
    hi = min(-0.06, seed_center + half)
    if hi - lo < 0.12:  # ensure enough span
        lo = max(float(neg_ax.min()), seed_center - 0.15)
        hi = min(-0.06, seed_center + 0.15)
    # Tight center bounds around the seed so it stays on the hump
    cen_lo = max(lo, seed_center - 0.12)
    cen_hi = min(hi, seed_center + 0.12)

    seed_sigma = 0.07

    model = [
        {'type': 'gaussian',
         'window': (cen_lo, cen_hi),
         'center': seed_center,
         'sigma': seed_sigma,
         'sigma_min': 0.02, 'sigma_max': 0.18},
        'quadratic'
    ]

    r = fit_per_pixel(
        fit_source, ax, model,
        mask=fit_mask,
        init='auto',
        bounds={'center': (cen_lo, cen_hi), 'sigma': (0.02, 0.18)},
        time_budget_s=240
    )

    maps_in = r['maps']
    err_in = r.get('err_maps', {})
    notes = r.get('notes', '')

    def find_key(mp, name):
        for k in mp:
            if k.endswith(name) and ('gaus' in k or 'p1' in k or name == k):
                return k
        for k in mp:
            if k.endswith(name):
                return k
        return None

    kc = find_key(maps_in, 'center')
    ka = find_key(maps_in, 'amplitude')
    kh = find_key(maps_in, 'height')

    peak_pos = maps_in[kc].astype(float) if kc else np.full((H, W), np.nan)
    peak_pos_err = err_in.get(kc, np.full((H, W), np.nan)).astype(float) if kc else np.full((H, W), np.nan)
    R2 = maps_in.get('R2', np.full((H, W), np.nan)).astype(float)

    if kh and kh in maps_in:
        peak_amp = maps_in[kh].astype(float)
    elif ka:
        peak_amp = maps_in[ka].astype(float)
    else:
        peak_amp = np.full((H, W), np.nan)

    # ---- Fermi-level (0 V) dI/dV gap-depth proxy (RAW smoothed) ----
    ef_idx = int(np.argmin(np.abs(ax)))
    ef_lo = max(0, ef_idx - 2)
    ef_hi = min(E, ef_idx + 3)
    fermi_ldos = sm[:, ef_lo:ef_hi].mean(axis=1).reshape(H, W)

    # ---- Rail-gaze + prominence/curvature gate ----
    # (a) center railed to either bound of the (now tight) window
    rail = (np.abs(peak_pos - cen_lo) < 5e-3) | (np.abs(peak_pos - cen_hi) < 5e-3)
    # (b) bad fit
    badfit = (R2 < 0.3) | ~np.isfinite(peak_pos)
    # (c) per-pixel curvature/prominence gate: require a genuine local maximum
    #     inside the window in the (smoothed) per-pixel spectrum after removing
    #     a quadratic background. Reject pixels with no real hump.
    win_mask = (ax >= lo) & (ax <= hi)
    wax = ax[win_mask]
    src_flat = fit_source.reshape(npix, E)
    src_win = ndi.gaussian_filter1d(src_flat[:, win_mask], sigma=1.5, axis=1)
    # per-pixel quadratic detrend (vectorized via lstsq)
    Vd = np.vander(wax, 3)  # columns: wax^2, wax, 1
    coef_all, _, _, _ = np.linalg.lstsq(Vd, src_win.T, rcond=None)  # (3, npix)
    base_all = (Vd @ coef_all).T  # (npix, len(wax))
    detr_all = src_win - base_all
    hump_prom = np.nanmax(detr_all, axis=1)  # per-pixel prominence above quad bg
    # noise scale of prominence: sigma_pix (raw) as conservative floor
    prom_gate = hump_prom.reshape(H, W) > (1.5 * sigma_pix)

    peak_pos_clean = peak_pos.copy()
    peak_pos_clean[rail | badfit | ~prom_gate] = np.nan

    # ---- Scalar: mean Peak_Position across valid pixels ----
    valid = np.isfinite(peak_pos_clean)
    mean_pos = float(np.nanmean(peak_pos_clean)) if valid.any() else float('nan')
    frac_valid = float(valid.mean())

    # ---- fit examples: extremes + median-quality ----
    fit_examples = []
    try:
        cen_map = maps_in[kc] if kc else None
        sig_key = find_key(maps_in, 'sigma')
        amp_key = ka
        # rebuild model curve per example pixel
        def eval_model(y, x):
            c = float(peak_pos[y, x])
            s = float(maps_in[sig_key][y, x]) if sig_key else seed_sigma
            A = float(maps_in[amp_key][y, x]) if amp_key else 0.0
            # quadratic bg coeffs
            bc = {}
            for nm in ('c0', 'c1', 'c2', 'a', 'b', 'c'):
                kk = find_key(maps_in, nm)
                if kk:
                    bc[nm] = float(maps_in[kk][y, x])
            g = (A / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((ax - c) / s) ** 2)
            # approximate bg from quadratic fit of raw window if coeffs absent
            idxp = y * W + x
            bg_full = np.polyval(coef_all[:, idxp], ax)
            return g + bg_full

        # candidate pixels
        vy, vx = np.where(valid)
        if len(vy) > 0:
            pvals = peak_pos_clean[vy, vx]
            r2vals = R2[vy, vx]
            picks = []
            picks.append((vy[np.argmin(pvals)], vx[np.argmin(pvals)], 'pos min'))
            picks.append((vy[np.argmax(pvals)], vx[np.argmax(pvals)], 'pos max'))
            picks.append((vy[np.argmax(r2vals)], vx[np.argmax(r2vals)], 'best R2'))
            picks.append((vy[np.argmin(r2vals)], vx[np.argmin(r2vals)], 'worst R2'))
            med_i = np.argsort(pvals)[len(pvals) // 2]
            picks.append((vy[med_i], vx[med_i], 'median pos'))
            # spatial spread
            for (yy, xx) in [(H // 4, W // 4), (3 * H // 4, 3 * W // 4)]:
                if valid[yy, xx]:
                    picks.append((yy, xx, 'spatial'))
            seen = set()
            for (yy, xx, lab) in picks:
                key = (int(yy), int(xx))
                if key in seen:
                    continue
                seen.add(key)
                fit_examples.append({
                    'pixel': [int(yy), int(xx)],
                    'fitted': eval_model(int(yy), int(xx)),
                    'axis': ax.copy(),
                    'label': lab
                })
    except Exception:
        fit_examples = []

    desc = (
        "Per-pixel single-Gaussian fit of the occupied-state CDW LDOS hump in the "
        "negative-bias dI/dV, over a QUADRATIC background so the monotonic DOS decay "
        "is absorbed by the background and the Gaussian localizes the true local maximum. "
        f"Search window tightly brackets the mean-spectrum detrended hump at {seed_center:+.3f} V "
        f"([{lo:+.3f},{hi:+.3f}] V, center bounds [{cen_lo:+.3f},{cen_hi:+.3f}] V). "
        "Pixels are rejected (NaN) if the center rails to a window bound, R2<0.3, or the "
        "per-pixel quadratic-detrended spectrum shows no genuine hump (prominence<1.5 sigma_pix). "
        "Peak_Amplitude and Fermi_LDOS_GapProxy retained from prior passing estimators. "
        + bin_note + (" " + str(notes) if notes else "")
    )

    return {
        "maps": {
            "Peak_Position": peak_pos_clean,
            "Peak_Position_err": peak_pos_err,
            "Peak_Amplitude": peak_amp,
            "Fermi_LDOS_GapProxy": fermi_ldos,
            "R2": R2,
        },
        "units": {
            "Peak_Position": "V",
            "Peak_Position_err": "V",
            "Peak_Amplitude": "a.u.",
            "Fermi_LDOS_GapProxy": "a.u.",
            "R2": "",
            "Mean_Peak_Position": "V",
            "Valid_Fraction": "",
        },
        "scalars": {
            "Mean_Peak_Position": mean_pos,
            "Valid_Fraction": frac_valid,
        },
        "fit_examples": fit_examples,
        "description": desc,
    }