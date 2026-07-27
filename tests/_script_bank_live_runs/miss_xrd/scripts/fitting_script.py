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

# Synchrotron framework/hydrate material. Wavelength unknown from metadata; use
# CuKa1 as a placeholder for Scherrer/W-H (flag size as approximate). No
# instrumental standard available -> instrumental FWHM 0.0 (sizes = lower bound).
WAVELENGTH_ANGSTROM = 1.5406
INSTRUMENTAL_FWHM_DEG = 0.0

# ---- Step 2: One global multi-peak fit ----
# fit_range excludes the giant low-angle feature at ~3.3 deg that dominates the
# intensity range and starves prominence-based detection of the weaker genuine
# reflections. Keep everything from 4.5 to 45 deg on flat background.
#
# Refinement changes vs prior attempt (all method-locked, model itself unchanged):
#   - prominence_frac lowered 0.02 -> 0.01 to catch the weak high-angle shoulder
#     near 26.25 and the low-angle shoulder near 29.3 flagged by the verifier.
#   - min_distance_deg lowered 0.15 -> 0.10 so the closely-spaced shoulder maxima
#     (26.0/26.25 and 29.3/29.79) are detected as separate components.
#   - Explicit seeded components at x~26.25 and x~29.3 are appended to the
#     auto-detected centers so the two genuine but weak shoulders are modelled
#     even if detection under-triggers; the 22.2 apex is left to auto-detect with
#     the split pseudo-Voigt shape (independent left/right FWHM + free eta)
#     capturing the apex/wing asymmetry that produced the 65sigma residual.
#
# NOTE ON DEVIATION FROM PURE AUTO-DETECT: the skill rules recommend
# peak_centers=None for series generalisation. Per the LOCKED PLAN, two explicit
# shoulder seeds (26.25, 29.3) are added on top of auto-detection so previously
# merged weak reflections are recovered; the plan states to keep auto-detection
# PLUS these two seeds in the recipe for reuse across the VT series (centers
# allowed to migrate). We therefore first auto-detect to get the full center
# list, then union it with the two seeds and pass the combined list once.

# First pass: pure auto-detect (lowered sensitivity) to recover all reflections.
auto = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    background='snip',
    snip_iterations='auto',
    peak_centers=None,
    peak_shape='split_pseudo_voigt',
    prominence_frac=0.01,
    min_distance_deg=0.10,
    fit_range=[4.5, 45.0],
)

auto_centers = [float(c) for c in auto.get('peak_centers', [])]

# Union auto-detected centers with the two seeded shoulders. Only add a seed if
# no auto-detected center already sits within min_distance_deg of it (otherwise
# it would be a duplicate the optimizer collapses).
SEED_SHOULDERS = [26.25, 29.3]
MIN_SEP = 0.10
centers = list(auto_centers)
for seed in SEED_SHOULDERS:
    if not any(abs(seed - c) < MIN_SEP for c in centers):
        centers.append(seed)
centers = sorted(centers)

# Second pass: single global fit over the union center list. Still one global
# fit_pattern call (no spliced sub-fits); model is the locked split pseudo-Voigt
# sum on SNIP background. center_leeway lets the 22.2 apex and the shoulders
# migrate to their true positions.
fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    background='snip',
    snip_iterations='auto',
    peak_centers=centers,
    peak_shape='split_pseudo_voigt',
    prominence_frac=0.01,
    min_distance_deg=0.10,
    center_leeway_deg=0.3,
    fit_range=[4.5, 45.0],
)

peaks = fit['peaks']
r_squared = float(fit['r_squared'])

fit_curve_raw = np.asarray(fit['fit_curve_raw'], dtype=float)
fit_curve = np.asarray(fit['fit_curve'], dtype=float)
intensity_corrected = np.asarray(fit['intensity_corrected'], dtype=float)
background = fit_curve_raw - fit_curve

# ---- fit.npy on RAW scale (matches raw data), length N ----
np.save('fit.npy', fit_curve_raw)

# ---- RMSE over full modelled domain ----
resid = intensity - fit_curve_raw
rmse = float(np.sqrt(np.mean(resid ** 2)))

# ---- Step 3: Per-peak Scherrer size ----
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

# ---- Step 4: Williamson-Hall (optional) ----
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
noise = np.std(resid[np.isfinite(resid)]) if resid.size else 1.0
if not np.isfinite(noise) or noise == 0:
    noise = 1.0
norm_resid = resid / noise

# individual components (split pseudo-Voigt) reconstructed for display
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
    params[f'peak_{i}'] = {
        'center': float(p['center']),
        'fwhm': float(p['fwhm']),
        'amplitude': float(p['amplitude']),
        'area': float(p['area']),
        'eta': float(p['eta']),
    }
    if 'fwhm_left' in p:
        params[f'peak_{i}']['fwhm_left'] = float(p['fwhm_left'])
    if 'fwhm_right' in p:
        params[f'peak_{i}']['fwhm_right'] = float(p['fwhm_right'])
    if i - 1 < len(sizes_nm) and sizes_nm[i - 1] is not None:
        sv = sizes_nm[i - 1]
        params[f'peak_{i}']['scherrer_size_nm'] = float(sv) if np.isfinite(sv) else None

results = {
    'model_type': 'Sum of split pseudo-Voigt peaks (auto-detect prominence_frac=0.01, min_distance=0.10, plus seeded shoulders at 26.25 and 29.3) on SNIP background; fit_range=(4.5,45.0) to exclude giant ~3.3 deg low-angle feature',
    'parameters': params,
    'fit_quality': {
        'r_squared': r_squared,
        'rmse': rmse,
        'peak_region_r2': float(fit['peak_region_r2']),
        'residual_rms_over_noise': float(fit['residual_rms_over_noise']),
        'max_abs_residual_over_noise': float(fit['max_abs_residual_over_noise']),
        'n_peaks_fitted': int(fit['n_peaks']),
        'scherrer_mean_size_nm': mean_size_nm,
        'williamson_hall': wh,
        'background_method': fit.get('background_method'),
    },
    'deviation_note': 'Per the locked plan, ran auto-detect (prominence_frac=0.01, min_distance_deg=0.10) then unioned the detected centers with two explicit shoulder seeds (26.25, 29.3) and passed the combined list to a single global fit_pattern. This adds explicit peak_centers rather than pure None-auto-detect on the final call; justified because the plan mandates keeping the two seeded shoulders in the recipe so previously merged weak reflections are recovered. Still one global split-PV fit on SNIP background (model unchanged, no spliced sub-fits).'
}
print('FIT_RESULTS_JSON:' + json.dumps(results))
