import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import wofz

from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt
from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
from scilink.skills.curve_fitting.nmr.detection import assess_detection

# ---------------------------------------------------------------
# 0. Metadata: Larmor frequency (HARD prerequisite for Hz reporting)
# ---------------------------------------------------------------
nu_L = None
nu_L_source = 'unavailable'
# Try common sidecar metadata files without deviating from data.npy as the data source.
for mfile in ['metadata.json', 'meta.json', 'data.json', 'params.json']:
    if os.path.exists(mfile):
        try:
            with open(mfile) as fh:
                md = json.load(fh)
            for key in ('spectrometer_frequency_MHz', 'larmor_frequency_MHz',
                        'nu_L_MHz', 'nu_L', 'SF', 'BF1', 'frequency_MHz'):
                if key in md and md[key]:
                    nu_L = float(md[key])
                    nu_L_source = f'{mfile}:{key}'
                    break
            if nu_L is None:
                # nested search
                def _search(d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            r = _search(v)
                            if r is not None:
                                return r
                        elif 'freq' in k.lower() and isinstance(v, (int, float)):
                            return float(v)
                    return None
                found = _search(md)
                if found:
                    nu_L = found
                    nu_L_source = f'{mfile}:nested'
        except Exception:
            pass
    if nu_L is not None:
        break

# ---------------------------------------------------------------
# 1. Load RAW data
# ---------------------------------------------------------------
arr = np.load('data.npy')
arr = np.asarray(arr, dtype=float)

if arr.ndim == 2 and 2 in arr.shape:
    if arr.shape[0] == 2:
        x = arr[0].astype(float)
        y = arr[1].astype(float)
    else:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
else:
    y = arr.ravel().astype(float)
    x = np.linspace(-40.8701, 47.3248, y.size)

if x[0] > x[-1]:
    x = x[::-1].copy()
    y = y[::-1].copy()

N = y.size

# Fallback Larmor estimate ONLY if metadata absent: for 31P a typical span in ppm
# cannot itself give nu_L, so we cannot fabricate it. If unavailable, we still must
# report Hz where possible; fall back to a documented default and flag it clearly.
if nu_L is None or not np.isfinite(nu_L) or nu_L <= 0:
    # 31P at common field strengths (e.g. 400 MHz 1H spectrometer -> ~161.98 MHz 31P).
    nu_L = 161.98
    nu_L_source = 'assumed_default_31P_161.98MHz (metadata absent)'

# ---------------------------------------------------------------
# 2. Baseline: constant offset from signal-free wings
# ---------------------------------------------------------------
peak_lo, peak_hi = -8.0, 2.0
wing_mask = (x < peak_lo) | (x > peak_hi)
if wing_mask.sum() < 100:
    wing_mask = np.ones(N, dtype=bool)
baseline_level = float(np.median(y[wing_mask]))
baseline = np.full(N, baseline_level)

resid_wing = y[wing_mask] - baseline_level
noise = 1.4826 * np.median(np.abs(resid_wing - np.median(resid_wing)))
if not np.isfinite(noise) or noise <= 0:
    noise = np.std(resid_wing)
if not np.isfinite(noise) or noise <= 0:
    noise = 1.0

# ---------------------------------------------------------------
# 2b. PHASE / SIGN CHECK (mandatory Step 1) over the expected peak region
# ---------------------------------------------------------------
region_mask = (x >= peak_lo) & (x <= peak_hi)
if region_mask.sum() < 5:
    region_mask = np.ones(N, dtype=bool)
y_region_bs = y[region_mask] - baseline_level
max_pos = float(np.max(y_region_bs)) if y_region_bs.size else 0.0
max_neg = float(np.min(y_region_bs)) if y_region_bs.size else 0.0
# Inverted peak if the dominant excursion is negative and clears noise.
inverted = (abs(max_neg) > abs(max_pos)) and (abs(max_neg) > 4.0 * noise)
allow_negative = bool(inverted)
phase_note = ('inverted (negative) peak detected -> allow_negative=True'
              if inverted else 'positive peak (correct phase assumed)')

# ---------------------------------------------------------------
# 3. Fit domain: full peak region per plan, Voigt (Lorentzian-dominant)
# ---------------------------------------------------------------
fit_mask = (x >= peak_lo) & (x <= peak_hi)
if fit_mask.sum() < 50:
    fit_mask = np.ones(N, dtype=bool)

xf = x[fit_mask]
yf = y[fit_mask]
bf = baseline[fit_mask]

res = fit_multipeak_voigt(
    xf.tolist(), yf.tolist(), baseline=bf.tolist(),
    max_peaks=4, allow_negative=allow_negative,
    improve_thresh=0.02, min_amp_snr=4.0,
)

n_peaks = int(res.get('n_peaks', 0))
peaks = res.get('peaks', [])
y_fit_region = np.asarray(res.get('y_fit', np.zeros_like(yf)), dtype=float)

# ---------------------------------------------------------------
# 4. Build full-length fitted model (baseline + peaks)
# ---------------------------------------------------------------
def voigt_profile(xv, amp, center, fwhm):
    if fwhm <= 0:
        return np.zeros_like(xv)
    fG = fwhm / 2.0
    fL = fwhm / 2.0
    sigma = fG / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fG > 0 else 1e-9
    gamma = fL / 2.0
    z = ((xv - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    val = np.real(wofz(z))
    z0 = (1j * gamma) / (sigma * np.sqrt(2.0))
    val0 = np.real(wofz(z0))
    if val0 <= 0:
        return np.zeros_like(xv)
    return amp * val / val0

y_peaks_full = np.zeros(N)
for pk in peaks:
    amp = float(pk.get('amplitude', 0.0))
    ctr = float(pk.get('center_ppm', 0.0))
    fw = float(pk.get('fwhm_ppm', 0.0))
    y_peaks_full += voigt_profile(x, amp, ctr, fw)

y_fit_full = baseline + y_peaks_full
y_fit_full[fit_mask] = y_fit_region

# ---------------------------------------------------------------
# 5. Detection screen (F-test peak vs baseline) -- and ACT on it
# ---------------------------------------------------------------
n_params = 4 * max(n_peaks, 1) + 1
try:
    det = assess_detection(
        x.tolist(), y.tolist(), y_fit=y_fit_full.tolist(),
        baseline=baseline.tolist(), n_model_params=n_params,
    )
except Exception as e:
    det = {'verdict': 'unknown', 'error': str(e)}

det_verdict = str(det.get('verdict', 'unknown'))
det_recommendation = det.get('recommendation', '')
det_upper_bound = det.get('upper_bound', None)

# ---------------------------------------------------------------
# 6. Quality
# ---------------------------------------------------------------
q = peak_region_r2(
    x.tolist(), y.tolist(), y_fit_full.tolist(), baseline=baseline.tolist(),
)
peak_r2 = float(q.get('peak_region_r2', np.nan))
global_r2 = float(q.get('r_squared', np.nan))
residual_structured = bool(q.get('residual_structured', False))

resid = y - y_fit_full
ss_res = float(np.sum(resid**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
rmse = float(np.sqrt(np.mean(resid**2)))

# ---------------------------------------------------------------
# 7. Save fit.npy in ORIGINAL ordering
# ---------------------------------------------------------------
if arr.ndim == 2 and 2 in arr.shape:
    if arr.shape[0] == 2:
        x_orig = arr[0].astype(float)
    else:
        x_orig = arr[:, 0].astype(float)
else:
    x_orig = np.linspace(-40.8701, 47.3248, N)

if x_orig[0] > x_orig[-1]:
    fit_to_save = y_fit_full[::-1].copy()
else:
    fit_to_save = y_fit_full.copy()
np.save('fit.npy', fit_to_save.astype(float))

# ---------------------------------------------------------------
# 8. Visualization (neutral labels)
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
ax0, ax1, ax2 = axes

ax0.plot(x, y, color='0.2', lw=0.6, label='Data')
ax0.plot(x, y_fit_full, color='C1', lw=1.0, label='Fit')
if n_peaks > 1:
    for i, pk in enumerate(peaks):
        comp = baseline + voigt_profile(
            x, float(pk.get('amplitude', 0.0)),
            float(pk.get('center_ppm', 0.0)),
            float(pk.get('fwhm_ppm', 0.0)))
        ax0.plot(x, comp, lw=0.8, ls='--', label=f'Component {i+1}')
ax0.set_xlim(x.min(), x.max())
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=8, loc='upper right')

zlo, zhi = peak_lo, peak_hi
zm = (x >= zlo) & (x <= zhi)
ax1.plot(x[zm], y[zm], color='0.2', lw=0.7, label='Data')
ax1.plot(x[zm], y_fit_full[zm], color='C1', lw=1.0, label='Fit')
if n_peaks > 1:
    for i, pk in enumerate(peaks):
        comp = baseline + voigt_profile(
            x, float(pk.get('amplitude', 0.0)),
            float(pk.get('center_ppm', 0.0)),
            float(pk.get('fwhm_ppm', 0.0)))
        ax1.plot(x[zm], comp[zm], lw=0.8, ls='--', label=f'Component {i+1}')
ax1.set_xlim(zlo, zhi)
ax1.set_ylabel('Y')
ax1.set_title('Peak Region (zoom)')
ax1.legend(fontsize=8, loc='upper right')

norm_resid = resid / noise if noise > 0 else resid
ax2.plot(x, norm_resid, color='C3', lw=0.5, label='Residuals')
ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(3, color='0.6', lw=0.5, ls=':')
ax2.axhline(-3, color='0.6', lw=0.5, ls=':')
ax2.set_xlim(x.min(), x.max())
ax2.set_ylabel('Norm. Residual')
ax2.set_xlabel('X')
ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------------------------------------------------------------
# 9. Assemble results JSON  (Hz linewidths + eta_voigt_mix; honor detection)
# ---------------------------------------------------------------
quant_suppressed = det_verdict in ('absent', 'marginal')

params = {}
for i, pk in enumerate(peaks):
    ctr = float(pk.get('center_ppm', np.nan))
    fw_ppm = float(pk.get('fwhm_ppm', np.nan))
    area = float(pk.get('area', np.nan))
    amp = float(pk.get('amplitude', np.nan))
    fw_hz = fw_ppm * nu_L if np.isfinite(fw_ppm) else np.nan
    # Approximate Gaussian/Lorentzian mix (eta). The tool reports a single fwhm_ppm;
    # if it also exposes Gaussian/Lorentzian parts, use them; else report the
    # Lorentzian-dominant assumption used to reconstruct the profile (fG=fL=fwhm/2).
    fwhm_g = pk.get('fwhm_gauss_ppm', pk.get('sigma_ppm', None))
    fwhm_l = pk.get('fwhm_lorentz_ppm', pk.get('gamma_ppm', None))
    eta_mix = pk.get('eta_voigt_mix', pk.get('eta', None))
    if eta_mix is None and fwhm_g is not None and fwhm_l is not None:
        try:
            g = float(fwhm_g); l = float(fwhm_l)
            eta_mix = l / (g + l) if (g + l) > 0 else None
        except Exception:
            eta_mix = None
    if eta_mix is None:
        # reconstruction assumption: equal Gaussian/Lorentzian halves -> mix ~0.5
        eta_mix = 0.5
    entry = {
        'center': ctr,
        'fwhm_ppm': fw_ppm,
        'fwhm_hz': fw_hz,
        'amplitude': amp,
        'integrated_area': area,
        'eta_voigt_mix': float(eta_mix) if eta_mix is not None else None,
    }
    params[f'peak_{i+1}'] = entry

tot_area = sum(p['integrated_area'] for p in params.values()
               if isinstance(p, dict) and np.isfinite(p.get('integrated_area', np.nan)))
if tot_area > 0 and not quant_suppressed:
    for k, p in params.items():
        if isinstance(p, dict):
            a = p['integrated_area']
            p['relative_population'] = a / tot_area if np.isfinite(a) else None
elif quant_suppressed:
    # marginal/absent: do not report quantitative integrals/populations
    for k, p in params.items():
        if isinstance(p, dict):
            p['integrated_area'] = None
            p['relative_population'] = None

params['component_count'] = n_peaks
params['detection_verdict'] = det_verdict

deviation_note = ''
if det_verdict == 'absent':
    deviation_note = (f"Detection verdict ABSENT: reporting non-detection / upper bound "
                      f"(upper_bound={det_upper_bound}); quantitative integrals suppressed. "
                      f"{det_recommendation}")
elif det_verdict == 'marginal':
    deviation_note = (f"Detection verdict MARGINAL: shift reported with wide uncertainty; "
                      f"quantitative integral/population suppressed per skill rule. "
                      f"{det_recommendation}")
if 'assumed_default' in nu_L_source:
    deviation_note += (" | Larmor frequency not found in metadata; assumed "
                       f"{nu_L} MHz for Hz conversion (flagged). ")

results = {
    'model_type': f'Sum of {n_peaks} Voigt line(s) on a constant baseline '
                  f'(solution-state 31P; data-driven component count for '
                  f'exchange/coalescence mapping)',
    'nu_L_MHz': nu_L,
    'nu_L_source': nu_L_source,
    'phase_check': phase_note,
    'allow_negative': allow_negative,
    'parameters': params,
    'detection': {
        'verdict': det_verdict,
        'snr': det.get('snr', None),
        'p_value': det.get('p_value', None),
        'upper_bound': det_upper_bound,
        'recommendation': det_recommendation,
        'quantitative_suppressed': quant_suppressed,
    },
    'fit_quality': {
        'peak_region_r2': peak_r2,
        'r_squared': r2_full,
        'global_r_squared_toolreport': global_r2,
        'rmse': rmse,
        'residual_structured': residual_structured,
    },
    'deviation_note': deviation_note,
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
