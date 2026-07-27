import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern
from scilink.skills.curve_fitting.xrd_profile.scherrer import scherrer
from scilink.skills.curve_fitting.xrd_profile.williamson_hall import williamson_hall

# ---- Step 1: Load RAW data ----
data = np.load('data.npy')
data = np.asarray(data, dtype=float)
if data.ndim == 2 and data.shape[0] < data.shape[1]:
    data = data.T
two_theta = np.asarray(data[:, 0], dtype=float)
intensity = np.asarray(data[:, 1], dtype=float)

WAVELENGTH_ANGSTROM = 1.5406  # CuKa1
INSTRUMENTAL_FWHM_DEG = 0.0   # no instrumental standard provided -> sizes are lower bounds

# ---- Step 2: One global multi-peak fit over full 10-45 deg range ----
# Auto-detect (peak_centers=None) so the recipe generalises across the in-situ series.
# Split pseudo-Voigt line shape, SNIP background with auto iterations.
# Refinement: raise max_peaks above 30 (->40) and lower prominence_frac to 0.01 so
# auto-detect captures the previously-missed broad ~28.7 deg reflection, the resolved
# ~43.4 deg peak, and the weak ~15.5 deg shoulder between the ~14.75 and ~16.4 peaks.
# min_distance_deg kept modest (0.12) so the closely-spaced low-angle cluster and the
# 15.5 deg shoulder can still be resolved separately.
fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    peak_centers=None,
    background='snip',
    snip_iterations='auto',
    prominence_frac=0.01,
    min_distance_deg=0.12,
    max_peaks=40,
    peak_shape='split_pseudo_voigt',
    fit_range=None,
)

peaks = fit['peaks']
r_squared = float(fit['r_squared'])
fit_curve_raw = np.asarray(fit['fit_curve_raw'], dtype=float)
fit_curve_corr = np.asarray(fit['fit_curve'], dtype=float)
intensity_corr = np.asarray(fit['intensity_corrected'], dtype=float)
background = fit_curve_raw - fit_curve_corr

# ---- Save fit overlay on RAW scale (length N) ----
np.save('fit.npy', fit_curve_raw)

# ---- R2 and RMSE over full domain (raw scale) ----
resid_raw = intensity - fit_curve_raw
ss_res = float(np.sum(resid_raw ** 2))
ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = float(np.sqrt(np.mean(resid_raw ** 2)))

# ---- Step 3: Per-peak Scherrer crystallite size (lower bounds, inst FWHM=0) ----
sizes_nm = []
for p in peaks:
    s = scherrer(
        fwhm_deg=float(p['fwhm']),
        two_theta_deg=float(p['center']),
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        K=0.9,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )
    sizes_nm.append(s['size_nm'])
valid_sizes = [v for v in sizes_nm if v is not None and np.isfinite(v)]
mean_size_nm = float(np.mean(valid_sizes)) if valid_sizes else None

# ---- Step 4: Williamson-Hall (>=5 peaks span wide sin-theta range) ----
wh_input = [{'two_theta': float(p['center']), 'fwhm': float(p['fwhm'])} for p in peaks]
wh = None
if len(wh_input) >= 5:
    wh = williamson_hall(
        peaks=wh_input,
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        K=0.9,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )

# ---- Step 5: Visualization ----
noise = np.std(resid_raw) if np.std(resid_raw) > 0 else 1.0
norm_resid = resid_raw / noise

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

ax0 = fig.add_subplot(gs[0])
ax0.plot(two_theta, intensity, color='0.2', lw=0.8, label='Data')
ax0.plot(two_theta, fit_curve_raw, color='crimson', lw=1.0, label='Fit')
ax0.plot(two_theta, background, color='steelblue', lw=0.8, ls='--', label='Background')
for i, p in enumerate(peaks):
    ax0.axvline(float(p['center']), color='green', lw=0.4, alpha=0.25)
ax0.set_yscale('log')
ax0.set_ylim(bottom=max(1.0, np.min(intensity[intensity > 0]) * 0.5))
ax0.set_xlim(two_theta.min(), two_theta.max())
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.plot(two_theta, resid_raw, color='0.3', lw=0.6, label='Residuals')
ax1.axhline(0, color='k', lw=0.5)
ax1.set_ylabel('Residual')
ax1.legend(loc='upper right', fontsize=8)

ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax2.plot(two_theta, norm_resid, color='purple', lw=0.6, label='Residuals / noise')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Norm. resid.')
ax2.legend(loc='upper right', fontsize=8)

ax0.set_xlim(two_theta.min(), two_theta.max())
fig.savefig('visualization.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ---- Step 6: Results JSON ----
params = {}
for i, p in enumerate(peaks, start=1):
    params[f'peak_{i}'] = {
        'center': float(p['center']),
        'fwhm': float(p['fwhm']),
        'amplitude': float(p['amplitude']),
        'area': float(p['area']),
        'eta': float(p['eta']),
        'scherrer_size_nm_lower_bound': (float(sizes_nm[i-1]) if sizes_nm[i-1] is not None and np.isfinite(sizes_nm[i-1]) else None),
    }

results = {
    'model_type': 'Sum of split (asymmetric) pseudo-Voigt peaks on SNIP background (global fit_pattern, auto-detected peaks, 10-45 deg)',
    'parameters': params,
    'scherrer': {
        'mean_size_nm_lower_bound': mean_size_nm,
        'note': 'instrumental_fwhm_deg=0 (no standard); sizes are lower bounds',
    },
    'williamson_hall': (None if wh is None else {
        'size_nm': wh.get('size_nm'),
        'strain': wh.get('strain'),
        'r_squared': wh.get('r_squared'),
        'slope': wh.get('slope'),
        'intercept': wh.get('intercept'),
        'n_peaks_used': wh.get('n_peaks_used'),
    }),
    'fit_quality': {
        'r_squared': r2_full,
        'rmse': rmse,
        'peak_region_r2': float(fit['peak_region_r2']),
        'global_r_squared_from_fit': r_squared,
        'residual_rms_over_noise': float(fit['residual_rms_over_noise']),
        'max_abs_residual_over_noise': float(fit['max_abs_residual_over_noise']),
        'n_peaks_fitted': int(fit['n_peaks']),
        'background_method': fit.get('background_method'),
    },
    'deviation_note': '',
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
