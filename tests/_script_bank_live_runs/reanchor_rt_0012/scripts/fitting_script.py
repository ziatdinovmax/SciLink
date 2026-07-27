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
# (sizes are lower bound on domain size / upper bound on broadening).
WAVELENGTH_ANGSTROM = 1.5406
INSTRUMENTAL_FWHM_DEG = 0.0

# ---- Step 2: One global multi-peak fit (background handled inside) ----
# Per the plan: exclude the low-angle air-scatter/beamstop climb (~3-4 deg),
# which decays into flat background by ~5 deg. The first genuine reflection sits
# at ~12 deg on flat background, so set lo = 5.0 deg. Fit the full range 5-45 deg.
#
# Refinement (verifier flagged under-modelled structure at 11.5-12.0, 17.5-18.0,
# and 20.7-21.0): lower prominence_frac toward 0.005 and min_distance_deg toward
# 0.05 so the closely-spaced pairs separate, AND pass explicit seed centers at
# 11.6, 17.65, and 20.9 alongside auto-detected peaks to guarantee the three
# flagged features get their own component. Per the skill's 'unresolved doublet?
# tune detection, don't drill' rule, this stays inside the single global
# fit_pattern (no fit_profile splice). Seeds only bootstrap the global fit; the
# fit still refines every center within center_leeway_deg.
#
# NOTE: peak_centers here bootstrap detection of the three verifier-flagged
# features within a single global fit; auto-detection still finds all other
# peaks (the plan's 'seeds only bootstrap' clause). For the in-situ SERIES,
# the locked recipe must use peak_centers=None so it generalizes across frames
# (thermal drift / phase change); these seeds are for this establishing frame.
SEED_CENTERS = [11.6, 17.65, 20.9]

fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    background='snip',
    snip_iterations='auto',
    peak_centers=None,                 # auto-detect main reflections
    peak_shape='split_pseudo_voigt',   # asymmetric split pseudo-Voigt per plan
    prominence_frac=0.005,             # lowered to resolve weak flank/shoulder features
    min_distance_deg=0.05,             # lowered so tight pairs separate
    center_leeway_deg=0.3,
    max_fwhm_deg=2.0,                  # FWHM upper bound per plan constraint
    fit_range=[5.0, 45.0],
)

peaks = fit['peaks']

# ---- Ensure the three verifier-flagged features are seeded if auto-detect
# ---- merged them: check whether a detected center sits near each seed; if not,
# ---- re-run the single global fit with an explicit combined center list
# ---- (auto-detected + missing seeds). Still one global fit_pattern call.
detected_centers = [float(p['center']) for p in peaks]
missing_seeds = [c for c in SEED_CENTERS
                 if not any(abs(c - dc) < 0.15 for dc in detected_centers)]
if missing_seeds:
    combined = sorted(detected_centers + missing_seeds)
    fit = fit_pattern(
        two_theta.tolist(), intensity.tolist(),
        background='snip',
        snip_iterations='auto',
        peak_centers=combined,         # bootstrap flagged features into global fit
        peak_shape='split_pseudo_voigt',
        prominence_frac=0.005,
        min_distance_deg=0.05,
        center_leeway_deg=0.3,
        max_fwhm_deg=2.0,
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

fig, axes = plt.subplots(4, 1, figsize=(11, 12),
                         gridspec_kw={'height_ratios': [3, 1, 1, 1.4]})
ax0, ax1, ax2, ax3 = axes
for ax in (ax0, ax1, ax2):
    ax.get_shared_x_axes()

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
ax1.set_xlim(two_theta.min(), two_theta.max())

ax2.plot(two_theta, norm_resid, color='0.3', lw=0.7)
ax2.axhline(0, color='C3', lw=0.6)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')
ax2.set_xlim(two_theta.min(), two_theta.max())

# ---- Zoom panel over the three previously-flagged windows (own x-axis) ----
# Show normalized residual across 11-22 deg so the reviewer can confirm the
# 11.5-12.0, 17.5-18.0, and 20.7-21.0 windows are now structureless.
zoom_lo, zoom_hi = 11.0, 22.0
zmask = (two_theta >= zoom_lo) & (two_theta <= zoom_hi)
ax3.plot(two_theta[zmask], norm_resid[zmask], color='0.3', lw=0.8)
ax3.axhline(0, color='C3', lw=0.6)
for w in (11.75, 17.75, 20.85):
    ax3.axvline(w, color='C0', lw=0.5, ls=':', alpha=0.6)
ax3.set_ylabel('Residual / noise (zoom)')
ax3.set_xlabel('X')
ax3.set_xlim(zoom_lo, zoom_hi)

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
    'model_type': 'Sum of split (asymmetric) pseudo-Voigt peaks on SNIP background (snip_iterations=auto), single global fit_pattern; auto peak detection (prominence_frac=0.005, min_distance_deg=0.05) with explicit seed centers 11.6/17.65/20.9 bootstrapping the three verifier-flagged under-modelled features into the single global fit; eta free per peak, FWHM in [0.05,2.0]; fit_range=(5.0,45.0) to exclude the low-angle air-scatter/beamstop climb',
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
    'deviation_note': 'Establishing-frame re-fit: passed explicit seed centers 11.6/17.65/20.9 to bootstrap the three verifier-flagged under-modelled features into the single global fit_pattern (plan permits seeds to bootstrap; still one global fit, no fit_profile splice). For the in-situ series the locked recipe must use peak_centers=None to generalize across frames.'
}
print('FIT_RESULTS_JSON:' + json.dumps(results))
