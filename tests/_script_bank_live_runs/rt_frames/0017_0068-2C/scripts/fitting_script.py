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

# Wavelength unknown from metadata; use CuKa1 as placeholder for Scherrer/W-H
# (flag sizes as approximate). No instrumental standard -> instrumental FWHM 0.0
# (sizes are lower bound on domain size / broadening).
WAVELENGTH_ANGSTROM = 1.5406
INSTRUMENTAL_FWHM_DEG = 0.0

# ---- Step 2: One global multi-peak fit (background handled inside) ----
# Per the plan: exclude the low-angle air-scatter/beamstop climb (~3-4 deg,
# 50-500x the real reflections) which decays into flat background by ~5 deg.
# The first genuine reflection sits at ~11 deg on flat background, so set
# lo = 5.0 deg. Fit the full range 5-45 deg. Auto-detection is kept (no
# hardcoded centers) so the recipe generalizes across the in-situ series.
# Split (asymmetric) pseudo-Voigt line shape on SNIP background, eta free per
# peak. prominence_frac / min_distance_deg lowered so the ~19.2 shoulder and
# weak high-angle features are caught.
fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    background='snip',
    snip_iterations='auto',
    peak_centers=None,                 # auto-detect -> generalizes across series
    peak_shape='split_pseudo_voigt',   # asymmetric split pseudo-Voigt per plan
    prominence_frac=0.01,
    min_distance_deg=0.10,
    center_leeway_deg=0.3,
    max_fwhm_deg=2.0,                  # FWHM upper bound per plan constraint
    fit_range=[5.0, 45.0],
)

peaks = fit['peaks']
r_squared = float(fit['r_squared'])

fit_curve_raw = np.asarray(fit['fit_curve_raw'], dtype=float)
fit_curve = np.asarray(fit['fit_curve'], dtype=float)
background = fit_curve_raw - fit_curve

# ---- fit.npy on RAW scale (matches raw data), length N ----
np.save('fit.npy', fit_curve_raw)

# ---- R^2 and RMSE over full modelled domain ----
resid = intensity - fit_curve_raw
rmse = float(np.sqrt(np.mean(resid ** 2)))

# ---- Step 3: Per-peak Scherrer size (K=0.9) ----
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

# ---- Step 4: Williamson-Hall (only if >= 5 peaks) ----
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
finite_resid = resid[np.isfinite(resid)]
noise = np.std(finite_resid) if finite_resid.size else 1.0
if not np.isfinite(noise) or noise == 0:
    noise = 1.0
norm_resid = resid / noise

# individual split pseudo-Voigt components reconstructed for display
def split_pv(x, center, fwhm_l, fwhm_r, amp, eta):
    y = np.zeros_like(x, dtype=float)
    left = x < center
    for mask, fw in ((left, fwhm_l), (~left, fwhm_r)):
        if fw <= 0:
            continue
        sigma = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        g = np.exp(-(x[mask] - center) ** 2 / (2.0 * sigma ** 2))
        gamma = fw / 2.0
        l = gamma ** 2 / ((x[mask] - center) ** 2 + gamma ** 2)
        y[mask] = amp * (eta * l + (1.0 - eta) * g)
    return y

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1, 1]})
ax0, ax1, ax2 = axes

ax0.plot(two_theta, intensity, color='0.35', lw=0.8, label='Data')
ax0.plot(two_theta, fit_curve_raw, color='C3', lw=1.0, label='Fit')
ax0.plot(two_theta, background, color='C0', lw=0.8, ls='--', alpha=0.7, label='Background')

for i, p in enumerate(peaks):
    fw = float(p['fwhm'])
    fwl = float(p.get('fwhm_left', fw))
    fwr = float(p.get('fwhm_right', fw))
    comp = split_pv(two_theta, float(p['center']), fwl, fwr,
                    float(p['amplitude']), float(p['eta'])) + background
    ax0.plot(two_theta, comp, lw=0.6, alpha=0.5,
             label='Component 1' if i == 0 else None)

ax0.set_yscale('log')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8, ncol=2)
ax0.set_xlim(two_theta.min(), two_theta.max())

ax1.plot(two_theta, resid, color='0.3', lw=0.7)
ax1.axhline(0, color='C3', lw=0.6)
ax1.set_ylabel('Residual')

ax2.plot(two_theta, norm_resid, color='0.3', lw=0.7)
ax2.axhline(0, color='C3', lw=0.6)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')

ax0.set_xlim(two_theta.min(), two_theta.max())
fig.tight_layout()
fig.savefig('visualization.png', dpi=140)
plt.close(fig)

# ---- Step 6: Results JSON ----
params = {}
for i, p in enumerate(peaks, start=1):
    entry = {
        'center': float(p['center']),
        'fwhm': float(p['fwhm']),
        'amplitude': float(p['amplitude']),
        'area': float(p['area']),
        'eta': float(p['eta']),
    }
    if 'fwhm_left' in p:
        entry['fwhm_left'] = float(p['fwhm_left'])
    if 'fwhm_right' in p:
        entry['fwhm_right'] = float(p['fwhm_right'])
    if i - 1 < len(sizes_nm) and sizes_nm[i - 1] is not None:
        sv = sizes_nm[i - 1]
        entry['scherrer_size_nm'] = float(sv) if np.isfinite(sv) else None
    params[f'peak_{i}'] = entry

results = {
    'model_type': 'Sum of split (asymmetric) pseudo-Voigt peaks on SNIP background (snip_iterations=auto), single global fit_pattern with auto peak detection (prominence_frac=0.01, min_distance_deg=0.10, eta free per peak, FWHM in [0.05,2.0]); fit_range=(5.0,45.0) to exclude the low-angle air-scatter/beamstop climb',
    'parameters': params,
    'fit_quality': {
        'r_squared': r_squared,
        'rmse': rmse,
        'peak_region_r2': float(fit['peak_region_r2']),
        'residual_rms_over_noise': float(fit['residual_rms_over_noise']),
        'max_abs_residual_over_noise': float(fit['max_abs_residual_over_noise']),
        'n_peaks_fitted': int(fit['n_peaks']),
        'scherrer_mean_size_nm': mean_size_nm,
        'scherrer_per_peak_nm': [float(v) if (v is not None and np.isfinite(v)) else None for v in sizes_nm],
        'williamson_hall': wh,
        'background_method': fit.get('background_method'),
    },
    'deviation_note': ''
}
print('FIT_RESULTS_JSON:' + json.dumps(results))
