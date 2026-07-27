# Auto-generated Script
# Task: Compute the per-pixel Au coupon thickness from the K-edge jump. Steps: (1) Divide the sample cube by the same-grid I0 flat-field cube to form transmission I/I0, guarding against zero/low-flux; restrict analysis to energies above ~40 keV to avoid scatter/saturation-dominated low-energy region. (2) Form the optical depth ln(I0/I). (3) Use measure_edge_step to quantify the Au K-edge step at 80.725 keV per pixel with robust narrow pre-/post-edge windows and flux gating, producing an edge-jump magnitude map (this doubles as the Au localization map). (4) Obtain the tabulated Au K-edge discontinuity in the mass attenuation coefficient via the attenuation helper for element 'Au', multiply by density rho = 19.3 g/cm3 to get the linear-attenuation jump Delta_mu, and convert the measured Delta(mu*t) to thickness t = Delta(mu*t)/Delta_mu (report in mm). (5) Produce a spatial map of Au thickness and a map/flag of where the edge is measurable (Au present vs. absent). Provide the edge-jump magnitude and derived thickness as output maps.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    data = np.asarray(data, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    H, W, E = data.shape

    edge_kev = 80.725

    # --- Build I0 (incident flux reference) ---
    # Prefer a genuine flat-field cube from auxiliary.
    i0 = None
    if auxiliary:
        for k in auxiliary:
            if "I0" in k or "flat" in k.lower():
                cand = np.asarray(auxiliary[k], dtype=np.float64)
                if cand.shape == data.shape:
                    i0 = cand
                    break
                elif cand.ndim == 1 and cand.shape[0] == E:
                    i0 = np.broadcast_to(cand, data.shape).copy()
                    break

    # If no genuine flat-field exists, DO NOT use spatial-max envelope
    # (that manufactures a fake edge in every pixel). Instead estimate I0
    # from empty-field border pixels which contain no sample.
    i0_from_border = False
    if i0 is None:
        i0_from_border = True
        # Identify border pixels (frame edges) as candidate empty-field.
        border_mask = np.zeros((H, W), dtype=bool)
        bw = max(2, min(H, W) // 12)
        border_mask[:bw, :] = True
        border_mask[-bw:, :] = True
        border_mask[:, :bw] = True
        border_mask[:, -bw:] = True
        # Use only border pixels with high, flat flux (no absorption edge)
        flat = data.reshape(-1, E)
        bm = border_mask.reshape(-1)
        border_spec = flat[bm]
        # Robust I0 spectrum = median over border pixels per energy channel
        i0_1d = np.nanmedian(border_spec, axis=0)
        # Guard against zeros
        i0_1d = np.where(i0_1d <= 0, np.nan, i0_1d)
        i0 = np.broadcast_to(i0_1d, data.shape).copy()

    # --- Light per-pixel smoothing to stabilize edge measurement ---
    # Smooth only along energy to preserve spatial localization of coupon.
    data_s = gaussian_filter(data, sigma=(0, 0, 1.2))
    i0_s = gaussian_filter(i0, sigma=(0, 0, 1.2)) if i0.ndim == 3 else i0

    # --- Use registered edge-step tool ---
    # Restrict to >40 keV region via low_e_cut_kev.
    r = measure_edge_step(
        data_s, i0_s, axis, edge_kev,
        pre_gap=2.0, post_gap=1.5, win_width=4.0,
        low_e_cut_kev=40.0, flux_floor_counts=20.0,
        auto_center=True, search_tol_kev=2.0,
    )

    edge_step = np.asarray(r["edge_step"], dtype=np.float64)  # Delta(mu*t) = Delta-OD
    snr = np.asarray(r.get("snr", np.zeros_like(edge_step)), dtype=np.float64)
    edge_used = r.get("edge_kev_used", edge_kev)

    # --- Segment coupon: only pixels with a REAL, significant edge jump ---
    # Empty-field pixels should read ~0. Threshold on SNR and positive step.
    finite = np.isfinite(edge_step)
    edge_step = np.where(finite, edge_step, 0.0)
    snr = np.where(np.isfinite(snr), snr, 0.0)

    # Measurability: require positive edge and adequate SNR.
    # Also require the step to stand out above the empty-field noise floor.
    valid = edge_step[snr > 3.0]
    if valid.size > 20:
        noise_floor = np.nanmedian(np.abs(edge_step[snr <= 3.0])) if np.any(snr <= 3.0) else 0.0
    else:
        noise_floor = 0.0
    step_thresh = max(3.0 * (noise_floor if noise_floor > 0 else 0.02), 0.02)

    measurable = (snr > 3.0) & (edge_step > step_thresh)

    # Zero out non-coupon pixels so the jump map is localized, not full-field.
    edge_jump = np.where(measurable, edge_step, 0.0)

    # --- Convert Delta(mu*t) to thickness ---
    # Delta_mu = [ (mu/rho)_just_above - (mu/rho)_just_below ] * rho
    rho = 19.3  # g/cm^3
    e_below = np.array([edge_used - 0.5])
    e_above = np.array([edge_used + 0.5])
    mu_below = float(attenuation('Au', e_below)[0])   # cm^2/g
    mu_above = float(attenuation('Au', e_above)[0])   # cm^2/g
    delta_mu_over_rho = mu_above - mu_below            # cm^2/g (positive)
    delta_mu = delta_mu_over_rho * rho                 # 1/cm

    if not np.isfinite(delta_mu) or abs(delta_mu) < 1e-9:
        delta_mu = np.nan

    # thickness t = Delta(mu*t) / Delta_mu  -> cm, convert to mm
    with np.errstate(divide='ignore', invalid='ignore'):
        thickness_cm = edge_jump / delta_mu
    thickness_mm = thickness_cm * 10.0
    # Non-coupon pixels -> 0 thickness
    thickness_mm = np.where(measurable, thickness_mm, 0.0)
    thickness_mm = np.where(np.isfinite(thickness_mm), thickness_mm, 0.0)
    # Physical guard: no negative thickness
    thickness_mm = np.clip(thickness_mm, 0.0, None)

    au_present = measurable.astype(np.float64)

    coverage = 100.0 * measurable.mean()

    return {
        "maps": {
            "Au_KEdge_Jump": edge_jump,
            "Au_Thickness": thickness_mm,
            "Au_Present_Flag": au_present,
            "Edge_SNR": snr,
        },
        "units": {
            "Au_KEdge_Jump": "Delta(mu*t) [dimensionless optical depth]",
            "Au_Thickness": "mm",
            "Au_Present_Flag": "bool (1=Au present)",
            "Edge_SNR": "a.u.",
        },
        "description": (
            "Per-pixel Au K-edge (80.725 keV) analysis. Transmission I/I0 formed "
            "using a genuine flat-field I0 when provided; otherwise I0 is estimated "
            "from empty-field border pixels (median spectrum) rather than a spatial-max "
            "envelope, which would fabricate a spurious edge in every pixel. "
            "measure_edge_step quantifies the optical-depth jump Delta(mu*t) with robust "
            "continuum windows and a flux gate, restricted to >40 keV. Pixels are flagged "
            "as Au only where SNR>3 and the step exceeds ~3x the empty-field noise floor, "
            "so empty regions read ~0 (localized coupon, coverage=%.1f%%). Thickness = "
            "Delta(mu*t)/Delta_mu, where Delta_mu = [(mu/rho)_above - (mu/rho)_below]*rho "
            "(rho=19.3 g/cm3) from NIST tables, reported in mm." % coverage
        ),
    }