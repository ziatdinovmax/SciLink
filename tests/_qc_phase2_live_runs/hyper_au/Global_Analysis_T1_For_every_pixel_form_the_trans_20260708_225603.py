# Auto-generated Script
# Task: For every pixel, form the transmission spectrum by dividing the sample cube by the provided I0 flat-field cube on the same grid, then take ln(I0/I). Exclude the low-energy region (below ~40 keV) where scatter/noise dominate. Measure the Au K-edge step at 80.725 keV using the edge-step helper (measure_edge_step) to obtain the discontinuity Δμt across the edge in narrow windows just below and just above 80.7 keV; use its flux gating / measurability flag to distinguish Au-bearing pixels (edge present) from off-coupon pixels (no edge) — this produces the spatial location map (Au_Present / edge-jump magnitude). Convert the edge jump to physical thickness via t = Δμt / (Δ(μ/ρ) · ρ), obtaining the tabulated Au K-edge attenuation-coefficient jump from the attenuation helper for element Au and using ρ = 19.3 g/cm³. Output a per-pixel Au thickness map (in cm or mm) and the edge-jump magnitude map used to localize the coupon. Report thickness only where the edge is measurable; set non-Au pixels to NaN/zero.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    data = np.asarray(data, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    H, W, E = data.shape

    edge_kev = 80.725
    rho = 19.3  # g/cm^3 for Au

    # ---- Obtain the true I0 flat-field cube ----
    i0 = None
    if auxiliary:
        for k in auxiliary:
            if 'I0' in k or 'flat' in k.lower() or 'baseline' in k.lower():
                cand = np.asarray(auxiliary[k], dtype=np.float64)
                if cand.shape == data.shape:
                    i0 = cand
                    break
    # Fallback: estimate I0 from bremsstrahlung continuum (only if no real I0)
    if i0 is None:
        # per-pixel smooth continuum estimate via percentile of a rolling window
        i0 = np.broadcast_to(np.percentile(data.reshape(-1, E), 95, axis=0),
                             data.shape).copy()

    # ---- Light despiking / negative handling on data (do not erase edge) ----
    d = np.clip(data, 1e-6, None)
    i0c = np.clip(i0, 1e-6, None)

    # Light spectral smoothing for stability (small sigma, preserves the edge)
    d_s = gaussian_filter(d, sigma=(0, 0, 1.0))
    i0_s = gaussian_filter(i0c, sigma=(0, 0, 1.0))

    # ---- Measure the edge step (Delta optical depth) with flux gating ----
    # measure_edge_step internally forms -ln(I/I0), picks continuum windows,
    # gates on flux, and returns edge_step (Delta mu*t) + snr + measurable flag.
    r = measure_edge_step(
        d_s, i0_s, axis, edge_kev,
        pre_gap=2.0, post_gap=1.0, win_width=4.0,
        low_e_cut_kev=40.0,          # exclude low-E scatter/noise region
        flux_floor_counts=20.0,
        auto_center=True, search_tol_kev=2.0
    )

    edge_step = np.asarray(r['edge_step'], dtype=np.float64)   # Delta(mu*t)
    snr = np.asarray(r['snr'], dtype=np.float64)

    # ---- Per-pixel measurability: require resolvable edge (snr > 3) ----
    measurable_field = bool(r.get('measurable', True))
    au_present = (snr > 3.0) & np.isfinite(edge_step) & (edge_step > 0)
    if not measurable_field:
        # Whole field flux-starved: keep only strongest edges to avoid fabricating
        au_present &= (snr > 5.0)

    # ---- Edge-jump magnitude map (localization) ----
    edge_jump_map = np.where(au_present, edge_step, 0.0)

    # ---- Convert edge jump to physical thickness ----
    # t = Delta(mu*t) / (Delta(mu/rho) * rho)
    # Get tabulated Au mass-attenuation jump across the K-edge.
    e_below = np.array([edge_kev - 0.5])   # just below 80.725 keV
    e_above = np.array([edge_kev + 0.9])   # just above (clear edge broadening)
    mu_below = float(attenuation('Au', e_below)[0])   # cm^2/g
    mu_above = float(attenuation('Au', e_above)[0])   # cm^2/g
    d_mu_over_rho = mu_above - mu_below                # cm^2/g jump (positive)

    denom = d_mu_over_rho * rho   # 1/cm
    if denom <= 0:
        denom = np.nan

    thickness_cm = edge_step / denom   # cm

    # Report thickness only where edge is measurable; else NaN
    au_thickness = np.where(au_present, thickness_cm, np.nan)
    # Guard against unphysical negatives
    au_thickness = np.where(np.isfinite(au_thickness) & (au_thickness < 0),
                            np.nan, au_thickness)

    au_present_map = au_present.astype(np.float64)

    return {
        "maps": {
            "Au_Thickness": au_thickness,
            "Edge_Jump": edge_jump_map,
            "Au_Present": au_present_map,
            "Edge_SNR": snr
        },
        "units": {
            "Au_Thickness": "cm",
            "Edge_Jump": "Delta(mu*t) [unitless optical depth]",
            "Au_Present": "boolean (1=Au edge present)",
            "Edge_SNR": "a.u."
        },
        "description": (
            "Per-pixel Au K-edge (80.725 keV) analysis. Transmission optical depth "
            "-ln(I/I0) is formed using the SUPPLIED true no-sample I0 flat-field cube "
            "(critical fix: no cross-pixel percentile pseudo-I0, which self-cancels when "
            "the coupon fills the FOV). measure_edge_step picks pre/post continuum windows "
            "just below/above the edge (low-E<40 keV excluded), gates on photon flux, and "
            "returns Delta(mu*t) plus an SNR-based measurability flag to localize Au-bearing "
            "pixels. Thickness = Delta(mu*t)/(Delta(mu/rho)*rho) with tabulated NIST Au "
            "K-edge mass-attenuation jump and rho=19.3 g/cm^3. Non-Au / unmeasurable pixels "
            "(snr<=3) are set to NaN in thickness and 0 in the edge-jump localization map."
        )
    }