# Auto-generated Script
# Task: Compute per-pixel transmittance by dividing the sample cube by the provided I0 flat-field baseline cube on the same grid, then take -ln(I/I0) to obtain optical density (absorbance) vs energy, restricting to energies above 5 keV. For each of the three target elements (In, Pb, Sn), use the measure_edge_step helper to quantify the K-edge absorption jump at each pixel: In K-edge near 27.9 keV, Sn K-edge near 29.2 keV, and Pb K-edge near 88.0 keV. Because the In and Sn K-edges are only ~1.3 keV apart (near the 0.2 keV energy resolution / 0.1 keV bin width), measure their edge steps in their respective narrow windows and treat overlapping-signal pixels as mixed. Produce a spatial edge-step-height map for each element indicating where that element is present, plus a measurable/flux gate so weak-signal pixels are masked. Then convert each element's edge-step height to material thickness using the attenuation helper to obtain mu(E) across the K-edge for that element (derived from the element name), giving thickness = edge_step / (delta_mu). Report both presence (edge-step) maps and thickness maps for In, Pb, and Sn.

def analyze_feature(data, axis, reconstruction=None, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation
    import scipy.optimize

    H, W, E = data.shape
    N = H * W
    energy = np.asarray(axis, dtype=float)

    # ---- I0 baseline ----
    if auxiliary and "I0 flat-field baseline cube (no sample) on the same grid" in auxiliary:
        i0 = np.asarray(auxiliary["I0 flat-field baseline cube (no sample) on the same grid"], dtype=float)
    else:
        i0 = np.full_like(data, np.nanmedian(data), dtype=float)

    raw = np.asarray(data, dtype=float)
    # denoised for shape-based OD if available
    src = raw
    if reconstruction is not None:
        rec = np.asarray(reconstruction, dtype=float)
        if rec.shape == raw.shape:
            src = rec

    # ---- Optical density (absorbance) ----
    eps = 1.0
    T = np.clip(src, 1e-3, None) / np.clip(i0, eps, None)
    T = np.clip(T, 1e-4, None)
    OD = -np.log(T)  # (H,W,E)

    # restrict to > 5 keV validity via mask on energy
    e_valid = energy > 5.0

    # =====================================================================
    # PART A: Pb K-edge at 88 keV -- isolated, use helper directly on RAW
    # =====================================================================
    pb_edge = 88.0
    pb_res = measure_edge_step(raw, i0, energy, pb_edge,
                               pre_gap=2.0, post_gap=1.0, win_width=4.0,
                               low_e_cut_kev=5.0, flux_floor_counts=20.0,
                               auto_center=True, search_tol_kev=2.0)
    Pb_step = np.asarray(pb_res['edge_step'], dtype=float)
    Pb_snr = np.asarray(pb_res.get('snr', np.zeros_like(Pb_step)), dtype=float)
    Pb_gate = Pb_snr > 3.0
    Pb_step_map = np.where(Pb_gate, Pb_step, 0.0)
    Pb_step_map = np.clip(Pb_step_map, 0.0, None)

    # =====================================================================
    # PART B: JOINT TWO-EDGE FIT for In (27.94) and Sn (29.20) in 26-31 keV
    #   Model over narrow window w:
    #     OD(E) = c0 + c1*(E-Ec)
    #             + a_In * H(E - E_In)
    #             + a_Sn * H(E - E_Sn)
    #   where H is a smoothed step (erf) with fixed edge width sigma.
    #   a_In, a_Sn >= 0 are the edge-step heights. Linear least squares
    #   with fixed step shapes -> per-pixel NNLS. This resolves two
    #   closely-spaced edges that a single differential window cannot.
    # =====================================================================
    E_In = 27.94
    E_Sn = 29.20
    win_lo, win_hi = 26.0, 31.0
    wmask = (energy >= win_lo) & (energy <= win_hi) & e_valid
    Ew = energy[wmask]
    ODw = OD.reshape(N, E)[:, wmask]  # (N, k)

    from scipy.special import erf
    sigma = 0.15  # keV edge smoothing (~resolution/binning)
    Ec = 0.5 * (win_lo + win_hi)

    def step(Evec, E0):
        return 0.5 * (1.0 + erf((Evec - E0) / (np.sqrt(2.0) * sigma)))

    # Design matrix columns: [1, (E-Ec), step_In, step_Sn]
    col_const = np.ones_like(Ew)
    col_lin = (Ew - Ec)
    col_In = step(Ew, E_In)
    col_Sn = step(Ew, E_Sn)
    Aall = np.column_stack([col_const, col_lin, col_In, col_Sn])  # (k,4)

    # For per-pixel positivity on the two step amplitudes only, we split:
    # first remove the linear continuum via ordinary LSQ but constrain
    # step amps >=0. Use bounded lstsq per pixel is slow (6400 px). Instead:
    # Solve unconstrained LSQ vectorized, then clamp step amps to >=0 and
    # refit continuum. This is stable and fast.
    # Unconstrained pseudo-inverse:
    pinv = np.linalg.pinv(Aall)  # (4,k)
    coef = ODw @ pinv.T          # (N,4)
    a_In = coef[:, 2]
    a_Sn = coef[:, 3]

    # Refine: clamp negatives, and reduce In/Sn cross-talk by a single
    # joint NNLS on the two step columns after removing fitted continuum
    # for pixels flagged as candidate (speeds up + improves separation).
    cont = coef[:, 0:1] * col_const[None, :] + coef[:, 1:2] * col_lin[None, :]
    resid = ODw - cont  # (N,k) continuum-removed OD

    # Two-column design for NNLS
    B = np.column_stack([col_In, col_Sn])  # (k,2)
    # Normal equations for 2x2 NNLS solved analytically w/ nonneg clamp:
    BtB = B.T @ B                # (2,2)
    Btr = resid @ B              # (N,2)
    # Solve BtB x = Btr for all pixels
    try:
        invBtB = np.linalg.inv(BtB)
        xsol = Btr @ invBtB.T    # (N,2)
    except np.linalg.LinAlgError:
        xsol = np.column_stack([a_In, a_Sn])

    a_In = xsol[:, 0]
    a_Sn = xsol[:, 1]

    # Nonnegativity: if either negative, redo constrained (drop neg col)
    neg_In = a_In < 0
    neg_Sn = a_Sn < 0
    # If In negative -> set 0, refit Sn alone
    if np.any(neg_In):
        sn_only = (resid[neg_In] @ col_Sn) / (col_Sn @ col_Sn)
        a_In[neg_In] = 0.0
        a_Sn[neg_In] = sn_only
    if np.any(neg_Sn):
        in_only = (resid[neg_Sn] @ col_In) / (col_In @ col_In)
        a_Sn[neg_Sn] = 0.0
        a_In[neg_Sn] = in_only
    a_In = np.clip(a_In, 0.0, None)
    a_Sn = np.clip(a_Sn, 0.0, None)

    # ---- Flux / measurable gate for In & Sn ----
    # Median post-edge I0 flux around 30 keV must exceed floor
    post_e = (energy >= 29.5) & (energy <= 31.5)
    flux_post = np.median(i0.reshape(N, E)[:, post_e], axis=1)
    flux_gate = flux_post > 20.0

    # Residual-based SNR: signal amplitude vs fit residual RMS
    model = cont + a_In[:, None] * col_In[None, :] + a_Sn[:, None] * col_Sn[None, :]
    rms = np.sqrt(np.mean((ODw - model) ** 2, axis=1)) + 1e-6
    snr_In = a_In / rms
    snr_Sn = a_Sn / rms

    In_gate = (snr_In > 3.0) & flux_gate
    Sn_gate = (snr_Sn > 3.0) & flux_gate

    In_step_map = np.where(In_gate, a_In, 0.0).reshape(H, W)
    Sn_step_map = np.where(Sn_gate, a_Sn, 0.0).reshape(H, W)

    # =====================================================================
    # PART C: Thickness = edge_step / delta_mu   (delta_mu across edge)
    # =====================================================================
    def delta_mu(elem, edge_kev):
        e_lo = np.array([edge_kev - 0.5])
        e_hi = np.array([edge_kev + 0.5])
        mu_lo = attenuation(elem, e_lo)[0]
        mu_hi = attenuation(elem, e_hi)[0]
        return abs(mu_hi - mu_lo)

    dmu_In = delta_mu('In', E_In)
    dmu_Sn = delta_mu('Sn', E_Sn)
    dmu_Pb = delta_mu('Pb', pb_edge)

    def safe_div(step_map, dmu):
        if dmu is None or not np.isfinite(dmu) or dmu <= 0:
            return np.zeros_like(step_map)
        return step_map / dmu

    # thickness in cm (mu/rho in cm^2/g * rho). Since attenuation returns
    # mass mu without density, thickness = step/(delta(mu/rho)) has units
    # of areal density (g/cm^2). Report as areal density proxy (a.u./thickness).
    In_thick = safe_div(In_step_map, dmu_In)
    Sn_thick = safe_div(Sn_step_map, dmu_Sn)
    Pb_thick = safe_div(Pb_step_map, dmu_Pb)

    # clean NaNs
    for arr in (In_step_map, Sn_step_map, Pb_step_map, In_thick, Sn_thick, Pb_thick):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    maps = {
        "In_Edge_Step": In_step_map,
        "Sn_Edge_Step": Sn_step_map,
        "Pb_Edge_Step": Pb_step_map,
        "In_Thickness": In_thick,
        "Sn_Thickness": Sn_thick,
        "Pb_Thickness": Pb_thick,
        "In_SNR": snr_In.reshape(H, W),
        "Sn_SNR": snr_Sn.reshape(H, W),
    }
    units = {
        "In_Edge_Step": "Delta-OD",
        "Sn_Edge_Step": "Delta-OD",
        "Pb_Edge_Step": "Delta-OD",
        "In_Thickness": "g/cm^2 (areal density)",
        "Sn_Thickness": "g/cm^2 (areal density)",
        "Pb_Thickness": "g/cm^2 (areal density)",
        "In_SNR": "a.u.",
        "Sn_SNR": "a.u.",
    }
    desc = ("Per-pixel OD = -ln(I/I0). In(27.94) and Sn(29.20) K-edges, only 1.26 keV "
            "apart, are resolved by a JOINT two-edge fit over 26-31 keV: linear continuum "
            "plus two fixed-width (sigma=0.15 keV) erf step functions at the two edge energies, "
            "solved per pixel with nonnegativity so the Sn pre-window no longer straddles the In "
            "edge (kills the cross-talk that filled 96% of the field). Amplitudes a_In,a_Sn are the "
            "edge-step heights, gated by residual SNR>3 and I0 flux>20. Pb K-edge (88 keV, isolated) "
            "uses measure_edge_step directly. Thickness = edge_step / delta(mu/rho) with mu from NIST "
            "tables at edge +/-0.5 keV (areal density).")
    return {"maps": maps, "units": units, "description": desc}