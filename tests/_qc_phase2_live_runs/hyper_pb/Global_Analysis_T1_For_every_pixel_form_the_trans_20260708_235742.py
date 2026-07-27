# Auto-generated Script
# Task: For every pixel, form the transmission attenuation signal mu_t(E) = ln(I0(E)/I(E)) by dividing the sample cube by the companion I0 flat-field cube (guard against zeros/low flux). Exclude energies below ~40 keV (scatter/noise-dominated). Use the measure_edge_step helper to quantify the Pb K-edge jump at 88.0 keV (the discontinuity delta_mu_t between narrow windows just below and just above the edge), letting the helper pick robust windows and gate on flux so pixels without a measurable edge are flagged/masked. Obtain the tabulated Pb mass-attenuation jump at the K-edge from the attenuation helper (element = Pb) and multiply by density rho = 11.35 g/cm3 to get the linear-attenuation jump delta_mu_tab. Convert to areal thickness per pixel: t = delta_mu_t / delta_mu_tab. Output the edge-jump map (locates the Pb coupon) and the thickness map (in mm or cm; state units).

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    energy = np.asarray(axis, dtype=float)
    cube = np.asarray(data, dtype=float)
    H, W, E = cube.shape

    edge_kev = 88.0

    # ---- Obtain a genuine I0 flat-field ----
    # The critique made clear: fabricating I0 from the brightest in-frame pixels
    # is invalid when the Pb coupon fills the field. A real open-beam I0 cube
    # is REQUIRED for an absolute Beer-Lambert / K-edge measurement.
    i0 = None
    if auxiliary:
        for k in auxiliary:
            if "I0" in k or "flat" in k.lower() or "baseline" in k.lower():
                cand = np.asarray(auxiliary[k], dtype=float)
                if cand.shape == cube.shape or cand.shape == (E,):
                    i0 = cand
                    break

    have_external_i0 = i0 is not None

    # ---- Light per-pixel spectral smoothing (denoise, preserve the edge) ----
    # Smooth ONLY along energy with a small kernel so the 88 keV discontinuity
    # (a sharp step, not a narrow peak) is preserved while suppressing noise.
    cube_s = gaussian_filter(cube, sigma=(0.0, 0.0, 1.0))
    cube_s = np.clip(cube_s, 1e-3, None)

    if have_external_i0:
        if i0.shape == (E,):
            i0_s = gaussian_filter(i0, sigma=1.0)[None, None, :]
            i0_use = np.broadcast_to(np.clip(i0_s, 1e-3, None), cube.shape)
        else:
            i0_use = np.clip(gaussian_filter(i0, sigma=(0.0, 0.0, 1.0)), 1e-3, None)

        # ---- Absolute edge step via measure_edge_step (robust windows + flux gate) ----
        # Exclude scatter/noise-dominated low energies (<~40 keV).
        res = measure_edge_step(
            cube_s, i0_use, energy, edge_kev,
            pre_gap=2.0, post_gap=1.5, win_width=4.0,
            low_e_cut_kev=40.0, flux_floor_counts=20.0,
            auto_center=True, search_tol_kev=2.0,
        )
        edge_step = np.asarray(res["edge_step"], dtype=float)  # Delta-OD
        snr = np.asarray(res["snr"], dtype=float)
        edge_used = float(res.get("edge_kev_used", edge_kev))

        # Per-pixel measurability gate: keep only pixels with a real edge.
        good = np.isfinite(edge_step) & np.isfinite(snr) & (snr >= 3.0) & (edge_step > 0.0)
        edge_map = np.where(good, edge_step, np.nan)

    else:
        # ---- No external I0: recover the ABSOLUTE step from each pixel's OWN
        # bremsstrahlung continuum extrapolated across the edge (self-referenced).
        # This avoids the invalid cross-pixel I0 that only yields differential
        # thickness. We fit a log-linear (power-law) continuum to the pre-edge
        # region, extrapolate to the edge, and compare to the post-edge level.
        lnI = np.log(cube_s)
        lnE = np.log(np.clip(energy, 1e-3, None))

        # Windows (keV) around the edge, excluding <40 keV low-E noise.
        pre_lo, pre_hi = 60.0, 86.0     # continuum fit region below edge
        post_lo, post_hi = 89.5, 100.0  # just above edge (clear overshoot)
        eval_pre = (85.0, 87.5)          # narrow just-below window

        pre_fit = (energy >= pre_lo) & (energy <= pre_hi)
        post_win = (energy >= post_lo) & (energy <= post_hi)
        prew = (energy >= eval_pre[0]) & (energy <= eval_pre[1])

        edge_map = np.full((H, W), np.nan, dtype=float)
        snr = np.full((H, W), np.nan, dtype=float)
        edge_used = edge_kev

        # Pre-fit design matrix for power-law continuum: lnI = a + b*lnE
        xpre = lnE[pre_fit]
        Xpre = np.vstack([np.ones_like(xpre), xpre]).T  # (Npre,2)
        # Least-squares projection matrix
        XtX_inv = np.linalg.pinv(Xpre.T @ Xpre)
        proj = XtX_inv @ Xpre.T  # (2, Npre)

        lnE_edge = np.log(edge_kev)

        flat = lnI.reshape(-1, E)
        cube_flat = cube_s.reshape(-1, E)
        edge_flat = edge_map.reshape(-1)
        snr_flat = snr.reshape(-1)

        for p in range(flat.shape[0]):
            y = flat[p]
            coeff = proj @ y[pre_fit]           # [a, b]
            # Continuum (in ln I) extrapolated to just below the edge
            ln_cont_below = coeff[0] + coeff[1] * lnE_edge
            # Observed just-above-edge ln I (mean of post window)
            ln_post = np.mean(y[post_win])
            # Optical depth step = ln(I_continuum) - ln(I_observed_post)
            # (transmission drops -> lnI drops -> positive OD step)
            step = ln_cont_below - ln_post
            # Noise estimate from pre-fit residuals
            resid = y[pre_fit] - Xpre @ coeff
            noise = np.std(resid) + 1e-6
            s = step / noise
            edge_flat[p] = step
            snr_flat[p] = s

        edge_map = edge_flat.reshape(H, W)
        snr = snr_flat.reshape(H, W)

        good = np.isfinite(edge_map) & np.isfinite(snr) & (snr >= 3.0) & (edge_map > 0.0)
        edge_map = np.where(good, edge_map, np.nan)

    # ---- Spatial coherence cleanup: reject isolated salt-and-pepper pixels ----
    # Median-filter the finite edge map lightly; keep only pixels consistent
    # with a spatially coherent coupon.
    import scipy.ndimage
    finite_mask = np.isfinite(edge_map)
    filled = np.where(finite_mask, edge_map, 0.0)
    med = scipy.ndimage.median_filter(filled, size=3)
    # A pixel is part of the coupon if its neighborhood also has signal.
    neigh_count = scipy.ndimage.uniform_filter(finite_mask.astype(float), size=3) * 9.0
    coherent = finite_mask & (neigh_count >= 4)
    edge_map_clean = np.where(coherent, edge_map, np.nan)

    # ---- Tabulated Pb K-edge linear-attenuation jump ----
    rho_pb = 11.35  # g/cm^3
    e_below = np.array([edge_used - 1.0])
    e_above = np.array([edge_used + 1.0])
    mu_below = attenuation('Pb', e_below)[0]   # cm^2/g
    mu_above = attenuation('Pb', e_above)[0]   # cm^2/g
    dmu_mass = mu_above - mu_below              # cm^2/g (positive jump)
    delta_mu_tab = dmu_mass * rho_pb            # 1/cm (linear jump)

    # ---- Thickness per pixel: t = delta_mu_t / delta_mu_tab ----
    # delta_mu_t is dimensionless OD; delta_mu_tab in 1/cm -> t in cm -> mm.
    with np.errstate(invalid='ignore', divide='ignore'):
        t_cm = edge_map_clean / delta_mu_tab
    t_mm = t_cm * 10.0
    t_mm = np.where(t_mm >= 0, t_mm, np.nan)

    maps = {
        "Pb_Edge_Jump": edge_map_clean.astype(float),
        "Pb_Thickness": t_mm.astype(float),
    }
    units = {
        "Pb_Edge_Jump": "delta optical depth (dimensionless, ln(I0/I) jump)",
        "Pb_Thickness": "mm",
    }

    src = "external open-beam I0" if have_external_i0 else "self-referenced continuum extrapolation (no external I0)"
    desc = (
        "Pb K-edge thickness map at 88.0 keV. mu_t(E)=ln(I0/I) formed with "
        + src + "; low energies <40 keV excluded as scatter/noise-dominated. "
        "The edge step delta_mu_t is the OD discontinuity across the K-edge "
        "(robust pre/post windows, flux + SNR>=3 gate, spatial-coherence filter "
        "to remove salt-and-pepper). Tabulated Pb mass-attenuation jump "
        f"({dmu_mass:.3f} cm^2/g) x rho=11.35 g/cm^3 gives delta_mu_tab="
        f"{delta_mu_tab:.2f} 1/cm; thickness t=delta_mu_t/delta_mu_tab, in mm. "
        "When the coupon fills the field and no true open-beam I0 exists, an "
        "in-frame I0 only yields differential (biased-low) thickness, so a "
        "per-pixel self-referenced continuum method recovers the absolute step."
    )

    return {"maps": maps, "units": units, "description": desc}
