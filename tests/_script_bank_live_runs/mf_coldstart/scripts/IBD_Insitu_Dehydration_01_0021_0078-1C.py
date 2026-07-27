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

# Wavelength unknown from metadata; use CuKa1 as placeholder for Scherrer/W-H.
# No instrumental standard available -> instrumental FWHM = 0.0, so the Scherrer
# sizes are a LOWER BOUND on crystallite size / UPPER BOUND on broadening.
WAVELENGTH_ANGSTROM = 1.5406
INSTRUMENTAL_FWHM_DEG = 0.0

# ---- Inspect the low-angle region to set lo deliberately ----
# The plan: the giant ~3.9 deg feature is real Bragg but dominates the pattern
# ~50-500x. Place lo ABOVE the dominant low-angle feature at the point where it
# falls back to ordinary (flat) background, so detection is not starved. The
# confirmed reflections to model (~11.9, 17.5-17.9, ~20.0, faint 25-45) all sit
# above ~5 deg on flat background. Set lo = 5.0 deg initially and fit to 45 deg.
# Keep AUTO-DETECTION (no frozen centers) so the recipe reuses per frame; do NOT
# fight the giant peak by lowering prominence -- excluding it lets default/low
# prominence recover the weak reflections. prominence_frac lowered to 0.01 so
# weak high-angle reflections and the 17.55/17.85 pair are captured.

LO = 5.0
HI = 45.0

def run_fit(lo, hi, prom, mindist):
    return fit_pattern(
        two_theta.tolist(), intensity.tolist(),
        background='snip',
        snip_iterations='auto',
        peak_centers=None,                 # auto-detect -> generalizes across series
        peak_shape='split_pseudo_voigt',   # asymmetric split pseudo-Voigt per plan
        prominence_frac=prom,
        min_distance_deg=mindist,
        center_leeway_deg=0.3,
        max_fwhm_deg=2.0,                  # FWHM upper bound per plan constraint
        fit_range=[lo, hi],
    )

# ---- Step 2: Single global multi-peak fit ----
fit = run_fit(LO, HI, prom=0.01, mindist=0.10)

# Verify coverage: if too few peaks recovered or peak_region_r2 low, lower
# prominence/min_distance and re-run the single global fit (never splice).
if fit['n_peaks'] < 3 or float(fit['peak_region_r2']) < 0.90:
    fit2 = run_fit(LO, HI, prom=0.005, mindist=0.08)
    if float(fit2['peak_region_r2']) >= float(fit['peak_region_r2']):
        fit = fit2

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

fig, axes = plt.subplots(3, 1, figsize=(11, 10),
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
ax0.set_xlim(float(two_theta.min()), float(two_theta.max()))

ax1.plot(two_theta, resid, color='0.3', lw=0.7)
ax1.axhline(0, color='C3', lw=0.6)
ax1.set_ylabel('Residual')
ax1.set_xlim(float(two_theta.min()), float(two_theta.max()))

ax2.plot(two_theta, norm_resid, color='0.3', lw=0.7)
ax2.axhline(0, color='C3', lw=0.6)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')
ax2.set_xlim(float(two_theta.min()), float(two_theta.max()))

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
    'model_type': 'Sum of split (asymmetric) pseudo-Voigt peaks on SNIP background (snip_iterations=auto), single global fit_pattern with auto peak detection (prominence_frac=0.01, min_distance_deg=0.10, eta free per peak, FWHM in [0.05,2.0]); fit_range lo raised above the dominant ~3.9 deg low-angle Bragg feature (lo=5.0, hi=45.0). Scherrer K=0.9, instrumental_fwhm=0.0 -> sizes are lower-bound (broadening upper-bound).',
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
        'fit_range': fit.get('fit_range', [LO, HI]),
    },
    'deviation_note': ''
}
print('FIT_RESULTS_JSON:' + json.dumps(results))
