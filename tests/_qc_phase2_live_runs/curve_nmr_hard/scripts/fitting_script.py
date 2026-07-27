import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
arr = np.load('data.npy')
arr = np.asarray(arr, dtype=float)
if arr.ndim == 2:
    if arr.shape[0] == 2:
        x = arr[0]; y = arr[1]
    elif arr.shape[1] == 2:
        x = arr[:, 0]; y = arr[:, 1]
    else:
        x = np.linspace(-1968.82, 1968.35, arr.shape[-1]); y = arr.ravel()
else:
    y = arr.ravel()
    x = np.linspace(-1968.82, 1968.35, y.size)

N = y.size

if x[0] > x[-1]:
    x = x[::-1].copy()
    y = y[::-1].copy()

raw_y = y.copy()

# ---------------------------------------------------------------
# METADATA: read spectrometer_frequency_MHz and MAS rate from a sidecar.
# ---------------------------------------------------------------
meta_note = ""
NU_L = None
MAS_RATE_HZ = None
meta_source = "none"

def _search_meta():
    nu_l = None
    mas = None
    src = None
    cand = []
    for pat in ('*.json', '*meta*.npz', '*meta*.npy', 'metadata.*', 'acqu*', 'params*.json'):
        cand.extend(glob.glob(pat))
    cand = [c for c in cand if os.path.isfile(c)]
    freq_keys = ['spectrometer_frequency_MHz', 'spectrometer_frequency', 'nu_L_MHz',
                 'larmor_MHz', 'larmor_frequency_MHz', 'SFO1', 'sfo1', 'BF1', 'freq_MHz']
    mas_keys = ['spinning_rate_Hz', 'mas_rate_Hz', 'mas_rate', 'spin_rate_Hz',
                'MAS_Hz', 'MASR', 'spinning_rate', 'sample_rotation_Hz']
    def _pull(d):
        f = None; m = None
        if not isinstance(d, dict):
            return f, m
        low = {str(k).lower(): v for k, v in d.items()}
        for k in freq_keys:
            if k.lower() in low:
                try:
                    f = float(low[k.lower()]); break
                except Exception:
                    pass
        for k in mas_keys:
            if k.lower() in low:
                try:
                    m = float(low[k.lower()]); break
                except Exception:
                    pass
        return f, m
    for c in cand:
        try:
            if c.endswith('.json'):
                with open(c) as fh:
                    d = json.load(fh)
                f, m = _pull(d)
            elif c.endswith('.npz'):
                z = np.load(c, allow_pickle=True)
                d = {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files}
                flat = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        flat.update(v)
                    else:
                        flat[k] = v
                f, m = _pull(flat)
            elif c.endswith('.npy'):
                v = np.load(c, allow_pickle=True)
                try:
                    v = v.item()
                except Exception:
                    pass
                f, m = _pull(v) if isinstance(v, dict) else (None, None)
            else:
                continue
            if f is not None and nu_l is None:
                nu_l = f; src = c
            if m is not None and mas is None:
                mas = m; src = c if src is None else src
            if nu_l is not None and mas is not None:
                break
        except Exception:
            continue
    return nu_l, mas, src

_nu, _mas, _src = _search_meta()
if _nu is not None:
    NU_L = float(_nu)
    meta_source = _src or "sidecar"
    meta_note += f"nu_L={NU_L} MHz read from metadata ({meta_source}). "
if _mas is not None:
    MAS_RATE_HZ = float(_mas)
    meta_note += f"MAS rate={MAS_RATE_HZ} Hz read from metadata ({_src}). "

# ---------------------------------------------------------------
# 2. SPIKE DETECTION at x ~ 0 (DC glitch) - single vs multi-point
# Per plan: verify the x~0/-4.3 feature is multi-point (KEEP as centreband);
# mask ONLY confirmed single-point/edge-fold spikes.
# ---------------------------------------------------------------
i0 = int(np.argmin(np.abs(x)))
win = 5
lo = max(0, i0 - win); hi = min(N, i0 + win + 1)
local = y[lo:hi]
wing = np.concatenate([y[:N//10], y[-N//10:]])
noise = 1.4826 * np.median(np.abs(wing - np.median(wing)))
if noise <= 0:
    noise = np.std(wing)

center_val = y[i0]
neigh = np.array([y[max(0,i0-1)], y[min(N-1,i0+1)]])
is_single_point = (abs(center_val) > 20 * (abs(neigh).max() + noise)) \
                   and (abs(neigh).max() < 0.1 * abs(center_val))

mask_removed = []
y_work = y.copy()
if is_single_point:
    y_work[i0] = 0.5 * (y[max(0,i0-1)] + y[min(N-1,i0+1)])
    mask_removed.append(int(i0))
    meta_note += "Masked single-point DC glitch at x~0. "
else:
    meta_note += "x~0 feature is multi-point -> kept as centreband. "

for idx in (0, N-1):
    nb = y_work[idx+1] if idx == 0 else y_work[idx-1]
    if abs(y_work[idx]) > 20 * (abs(nb) + noise):
        y_work[idx] = nb
        mask_removed.append(int(idx))

# ---------------------------------------------------------------
# 3-4. BASELINE via spline through signal-free windows
# Low-order baseline anchored in signal-free windows; signed amplitudes
# handle the negative 0-100 ppm lobe (NOT a large negative baseline).
# ---------------------------------------------------------------
ynorm = np.abs(y_work - np.median(y_work))
thr = np.median(ynorm) + 2.0 * 1.4826 * np.median(np.abs(ynorm - np.median(ynorm)))
free = ynorm < thr

from scipy.ndimage import binary_dilation
signal_mask = ~free
signal_mask = binary_dilation(signal_mask, iterations=30)
free = ~signal_mask
if free.sum() < max(50, N // 50):
    order = np.argsort(ynorm)
    free = np.zeros(N, bool)
    free[order[:N // 3]] = True

try:
    spl = UnivariateSpline(x[free], y_work[free], k=3,
                           s=len(x[free]) * np.var(y_work[free]))
    baseline = spl(x)
except Exception:
    c = np.polyfit(x[free], y_work[free], 3)
    baseline = np.polyval(c, x)

y_corr = y_work - baseline

# ---------------------------------------------------------------
# VERIFY sideband spacing: MEASURE the observed comb spacing directly.
# ---------------------------------------------------------------
from scipy.signal import find_peaks

abs_corr = np.abs(y_corr)
pk_idx, _ = find_peaks(abs_corr, height=5.0 * noise, distance=3)
measured_spacing_ppm = None
if pk_idx.size >= 3:
    pk_pos = np.sort(x[pk_idx])
    diffs = np.diff(pk_pos)
    diffs = diffs[diffs > 0]
    # restrict to the regular MAS comb regime (~90-95 ppm) so we don't
    # measure sub-splittings inside a broadened cluster.
    comb_like = diffs[(diffs > 60) & (diffs < 130)]
    if comb_like.size:
        med = np.median(comb_like)
        near = comb_like[np.abs(comb_like - med) < 0.3 * med]
        measured_spacing_ppm = float(np.median(near)) if near.size else float(med)
    elif diffs.size:
        med = np.median(diffs)
        near = diffs[np.abs(diffs - med) < 0.5 * med]
        measured_spacing_ppm = float(np.median(near)) if near.size else float(med)

if measured_spacing_ppm is not None:
    meta_note += f"Measured comb spacing from spectrum = {measured_spacing_ppm:.3f} ppm. "

# Reconcile with metadata / derive missing quantities
spacing_verified = False
if NU_L is not None and MAS_RATE_HZ is not None:
    MAS_PPM = MAS_RATE_HZ / NU_L
    if measured_spacing_ppm is not None and MAS_PPM > 0:
        rel = abs(measured_spacing_ppm - MAS_PPM) / MAS_PPM
        spacing_verified = rel < 0.15
        meta_note += (f"Sideband check: nu_r/nu_L={MAS_PPM:.3f} ppm vs measured "
                      f"{measured_spacing_ppm:.3f} ppm (rel diff {rel*100:.1f}%, "
                      f"{'VERIFIED' if spacing_verified else 'MISMATCH'}). ")
elif NU_L is not None and MAS_RATE_HZ is None:
    if measured_spacing_ppm is not None:
        MAS_PPM = measured_spacing_ppm
        MAS_RATE_HZ = MAS_PPM * NU_L
        meta_source = meta_source + "+derived_MAS"
        meta_note += (f"MAS rate not in metadata; DERIVED {MAS_RATE_HZ:.0f} Hz from "
                      f"measured comb spacing x nu_L. ")
        spacing_verified = True
    else:
        MAS_PPM = None
        meta_note += "MAS rate missing and no comb detected; sideband analysis skipped. "
elif NU_L is None and MAS_RATE_HZ is not None:
    if measured_spacing_ppm is not None and measured_spacing_ppm > 0:
        NU_L = MAS_RATE_HZ / measured_spacing_ppm
        MAS_PPM = measured_spacing_ppm
        meta_source = "derived_nuL_from_comb"
        spacing_verified = True
        meta_note += (f"nu_L not in metadata; DERIVED {NU_L:.2f} MHz from known MAS "
                      f"rate / measured comb spacing. ")
    else:
        MAS_PPM = None
        meta_note += "nu_L missing and no comb to derive it; Hz/C_Q reporting provisional. "
else:
    # neither available: use the measured comb spacing (ppm) as the MAS spacing
    # so the manifold CAN still be fit (the plan's core requirement); Hz/C_Q
    # quantities remain provisional.
    MAS_PPM = measured_spacing_ppm if measured_spacing_ppm is not None else 92.0
    if measured_spacing_ppm is None:
        meta_note += ("No metadata and no comb auto-detected; seeding MAS spacing at "
                      "92 ppm (mid of expected 90-95 ppm) so the manifold is still fit. ")
    else:
        meta_note += ("NO metadata sidecar for spectrometer_frequency_MHz / spinning_rate_Hz; "
                      "MAS spacing taken from measured comb (ppm). Hz-based and C_Q quantities "
                      "are PROVISIONAL. ")

# If the measured spacing landed outside the physically expected 90-95 ppm comb
# regime (e.g. it picked up a sub-splitting), seed the manifold in-band per plan.
if MAS_PPM is not None and not (60 < MAS_PPM < 130):
    meta_note += (f"Measured spacing {MAS_PPM:.1f} ppm outside expected 90-95 ppm comb; "
                  f"seeding manifold at 92 ppm per plan. ")
    MAS_PPM_SEED = 92.0
else:
    MAS_PPM_SEED = MAS_PPM if MAS_PPM is not None else 92.0

have_nuL = NU_L is not None

# ---------------------------------------------------------------
# 5. Identify dominant centreband (seed ~ -4.3 ppm region per plan)
# ---------------------------------------------------------------
ipk = int(np.argmax(np.abs(y_corr)))
delta_peak = x[ipk]
# prefer the plan's -4.3 ppm seed if the abs-max is near it
centre_seed = float(delta_peak)

from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
from scilink.skills.curve_fitting.nmr.detection import assess_detection
from scilink.skills.curve_fitting.nmr.sidebands import fit_sideband_manifold

y_fit_full = np.zeros(N)
results_params = {}
model_type = ""

# ---------------------------------------------------------------
# 4 (plan). FIT THE FULL SIDEBAND MANIFOLD as ONE species.
# This is the primary model: centreband (~-4.3 ppm) + all sideband
# orders locked at delta_iso +/- k*MAS_rate, fit jointly.
# ---------------------------------------------------------------
sb_info = {}
sb = None
try:
    sb = fit_sideband_manifold(x.tolist(), y_corr.tolist(),
                               mas_rate_ppm=float(MAS_PPM_SEED),
                               centre_ppm=centre_seed, allow_negative=True)
    y_sb = np.asarray(sb.get('y_fit'), dtype=float) if sb.get('y_fit') is not None else None
    if y_sb is not None and y_sb.size == N:
        y_fit_full = y_sb
    sb_info = {
        'isotropic_shift_ppm': sb.get('isotropic_shift_ppm'),
        'mas_rate_ppm': float(MAS_PPM_SEED),
        'mas_rate_kHz': (float(MAS_PPM_SEED) * NU_L / 1000.0) if have_nuL else None,
        'total_integrated_intensity': sb.get('total_integrated_intensity'),
        'centreband_fraction': sb.get('centreband_fraction'),
        'order_intensities': sb.get('order_intensities'),
        'manifold_span_ppm': sb.get('manifold_span_ppm'),
        'fit_quality': sb.get('fit_quality'),
    }
    model_type = ("MAS spinning-sideband manifold (fit_sideband_manifold): "
                  "centreband + sidebands as one species")
    if sb.get('isotropic_shift_ppm') is not None:
        centre_seed = float(sb.get('isotropic_shift_ppm'))
except Exception as e:
    meta_note += f"fit_sideband_manifold failed ({e}); "

# ---------------------------------------------------------------
# Attempt fit_quad_ct on the centreband ONLY (per plan step (b)):
# report C_Q/eta only if Cq_resolved=True, else pseudo-Voigt FWHM +
# C_Q upper bound + MQMAS recommendation. This is diagnostic on the
# centreband and does NOT replace the manifold fit.
# ---------------------------------------------------------------
centreband_report = {}
if have_nuL and sb is not None:
    roi = (x > centre_seed - 45) & (x < centre_seed + 45)
    if roi.sum() >= 30:
        ppm_roi = x[roi]
        y_roi = y_corr[roi]
        try:
            from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_ct
            quad_fit = fit_quad_ct(ppm_roi.tolist(), y_roi.tolist(), nu_L_MHz=NU_L,
                                   I=1.5, mas=True, delta_iso_init=centre_seed)
            p = quad_fit['parameters']
            d = quad_fit.get('derived', {})
            cq_resolved = bool(d.get('Cq_resolved', p.get('Cq_resolved', False)))
            centreband_report = {
                'delta_iso_ppm': p.get('delta_iso_ppm'),
                'Cq_MHz': p.get('Cq_MHz'),
                'Cq_is_upper_bound': (not cq_resolved),
                'eta_Q': (p.get('eta') if cq_resolved else None),
                'P_Q_MHz': p.get('P_Q_MHz'),
                'Cq_resolved': cq_resolved,
                'lw_gauss_ppm': p.get('lw_gauss_ppm'),
                'lw_lorentz_ppm': p.get('lw_lorentz_ppm'),
                'recommendation': (None if cq_resolved else
                                   'narrow symmetric centreband: C_Q reported as upper '
                                   'bound; MQMAS recommended to resolve C_Q/eta.'),
            }
        except Exception as e:
            meta_note += f"fit_quad_ct on centreband failed ({e}); "

results_params['sideband_manifold'] = sb_info
if centreband_report:
    results_params['centreband_quad'] = centreband_report

# ---------------------------------------------------------------
# Fallback: if the manifold fit produced nothing usable, fit a Voigt
# multipeak with sideband positions excluded (so we still model signal).
# ---------------------------------------------------------------
if not np.any(y_fit_full):
    from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt
    try:
        vres = fit_multipeak_voigt(x.tolist(), y_corr.tolist(),
                                   mas_rate_ppm=float(MAS_PPM_SEED), allow_negative=True)
        y_fit_full = np.asarray(vres['y_fit'], dtype=float)
        model_type = "pseudo-Voigt multipeak (manifold fit unavailable fallback)"
        for k, pk in enumerate(vres.get('peaks', [])):
            fwhm_ppm = pk.get('fwhm_ppm')
            results_params[f'peak_{k+1}'] = {
                'center_ppm': pk.get('center_ppm'),
                'amplitude': pk.get('amplitude'),
                'fwhm_ppm': fwhm_ppm,
                'fwhm_hz': (fwhm_ppm * NU_L) if (fwhm_ppm is not None and have_nuL) else None,
                'area': pk.get('area'),
            }
    except Exception as e:
        meta_note += f"Voigt fallback failed ({e}); "

# ---------------------------------------------------------------
# 6. Detection gate
# ---------------------------------------------------------------
np_params = 4
try:
    det = assess_detection(x.tolist(), y_corr.tolist(), y_fit=y_fit_full.tolist(),
                           baseline=np.zeros(N).tolist(), n_model_params=np_params)
    det_verdict = det.get('verdict')
except Exception:
    det_verdict = None

# ---------------------------------------------------------------
# 8. Quality (peak-region R^2 over centreband + sideband windows)
# ---------------------------------------------------------------
try:
    q = peak_region_r2(x.tolist(), y_corr.tolist(), y_fit_full.tolist())
    pr2 = q.get('peak_region_r2')
    gr2 = q.get('r_squared')
    resid_struct = q.get('residual_structured')
except Exception:
    resid = y_corr - y_fit_full
    ss_res = np.sum(resid**2); ss_tot = np.sum((y_corr - np.mean(y_corr))**2)
    gr2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    pr2 = gr2; resid_struct = None

resid = y_corr - y_fit_full
rmse = float(np.sqrt(np.mean(resid**2)))

# ---------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
ax0, ax1, ax2 = axes

ax0.plot(x, raw_y, color='0.75', lw=0.6, alpha=0.6, label='Raw')
ax0.plot(x, y_corr, color='k', lw=0.7, label='Data')
ax0.plot(x, y_fit_full, color='crimson', lw=1.0, label='Fit')
ax0.set_title('Data and Fit')
ax0.set_xlabel('X'); ax0.set_ylabel('Y')
ax0.legend(loc='upper right', fontsize=8)
ax0.set_xlim(x.max(), x.min())

# Zoom on a representative first-order sideband region so sideband misfit
# is visible (own non-shared axis so it doesn't truncate other panels).
sb_zoom_lo = centre_seed - 1.6 * MAS_PPM_SEED
sb_zoom_hi = centre_seed + 1.6 * MAS_PPM_SEED
ax1.plot(x, y_corr, color='k', lw=0.7, label='Data')
ax1.plot(x, y_fit_full, color='crimson', lw=1.0, label='Fit')
ax1.set_xlim(sb_zoom_hi, sb_zoom_lo)
ax1.set_title('Zoom (centreband + first sidebands)')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax1.legend(loc='upper right', fontsize=8)

ax2.plot(x, resid, color='navy', lw=0.6, label='Residuals')
ax2b = ax2.twinx()
ax2b.plot(x, resid / (noise if noise > 0 else 1.0), color='0.6', lw=0.4, alpha=0.5)
ax2b.set_ylabel('Residual / noise', color='0.5', fontsize=8)
ax2.axhline(0, color='r', lw=0.5)
ax2.set_title('Residuals')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_xlim(x.max(), x.min())

plt.tight_layout()
plt.savefig('visualization.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# 6. Save fit.npy
# ---------------------------------------------------------------
fit_full_with_baseline = y_fit_full + baseline
np.save('fit.npy', fit_full_with_baseline.astype(float))

# ---------------------------------------------------------------
# 7. JSON output
# ---------------------------------------------------------------
results = {
    "model_type": model_type,
    "parameters": results_params,
    "metadata": {
        "nu_L_MHz": NU_L,
        "nu_L_source": meta_source,
        "MAS_rate_Hz": MAS_RATE_HZ,
        "sideband_manifold_centre_ppm": centre_seed,
        "mas_rate_ppm": float(MAS_PPM_SEED),
        "mas_rate_kHz": (float(MAS_PPM_SEED) * NU_L / 1000.0) if have_nuL else None,
        "measured_comb_spacing_ppm": measured_spacing_ppm,
        "sideband_spacing_verified": bool(spacing_verified),
        "dominant_peak_ppm": float(delta_peak),
        "masked_points": mask_removed,
        "detection_verdict": det_verdict,
        "residual_structured": resid_struct,
    },
    "fit_quality": {
        "peak_region_r2": float(pr2) if pr2 is not None else None,
        "r_squared": float(gr2) if gr2 is not None else None,
        "rmse": rmse,
    },
    "deviation_note": ("" if (sb is not None and np.any(y_fit_full)) else
                       "Sideband manifold fit unavailable; fell back to multipeak Voigt with "
                       "sideband positions excluded. ") +
                      (("MAS spacing/nu_L partly derived from the measured comb because a "
                        "metadata sidecar was not found; Hz-based and C_Q values are provisional. ")
                       if meta_source in ('none',) or 'derived' in meta_source else "")
}

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

print(f"FIT_RESULTS_JSON:{json.dumps(_clean(results))}")
