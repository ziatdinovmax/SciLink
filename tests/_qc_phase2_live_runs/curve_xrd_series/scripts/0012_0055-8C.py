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
INSTRUMENTAL_FWHM_DEG = 0.0   # per plan: report Scherrer sizes as lower bounds

# ---- Step 2: One global multi-peak split pseudo-Voigt fit on SNIP background ----
# Plan: single global fit_pattern per frame, SNIP background, snip_iterations auto,
# split pseudo-Voigt, auto peak detection ON (peak_centers=None), fit_range=(6.0,45.0).
# Refinement: lower prominence_frac to 0.01 and min_distance_deg so the auto-detector
# resolves (1) the ~29.3 shoulder on the 29.8 peak, (2) the ~26.3 second component
# under the 26.0 peak, and (3) the weak ~27.45 reflection.
fit = fit_pattern(
    two_theta.tolist(), intensity.tolist(),
    peak_centers=None,               # auto-detect ON (do NOT freeze centers)
    background='snip',
    snip_iterations='auto',
    prominence_frac=0.01,            # lowered 0.02 -> 0.01 per plan to catch weak/shoulder features
    min_distance_deg=0.08,           # lowered so 26.0/26.3 and 29.8/29.3 pairs resolve separately
    peak_shape='split_pseudo_voigt',
    fit_range=(6.0, 45.0),
)

peaks = fit['peaks']
detected_centers = [float(p['center']) for p in peaks]

# ---- Fallback: if auto-detection at prominence_frac=0.01 still misses any of the
# three flagged real features, explicitly seed them (plan-sanctioned) and re-run
# the SINGLE global split-PV fit_pattern with the detected + seeded centers. This
# stays within one global fit (no per-peak fit_profile splice).
REQUIRED_SEEDS = [29.3, 26.3, 27.45]
TOL = 0.15  # deg: consider a required feature 'present' if a detected peak is this close


def _has_center(centers, target, tol):
    return any(abs(c - target) <= tol for c in centers)


missing = [t for t in REQUIRED_SEEDS if not _has_center(detected_centers, t, TOL)]
if missing:
    seeded_centers = sorted(detected_centers + missing)
    fit = fit_pattern(
        two_theta.tolist(), intensity.tolist(),
        peak_centers=seeded_centers,   # detected + explicit seeds; one global fit
        background='snip',
        snip_iterations='auto',
        prominence_frac=0.01,
        min_distance_deg=0.08,
        peak_shape='split_pseudo_voigt',
        fit_range=(6.0, 45.0),
    )
    peaks = fit['peaks']

r_squared = fit['r_squared']
intensity_corrected = np.asarray(fit['intensity_corrected'], dtype=float)
fit_curve = np.asarray(fit['fit_curve'], dtype=float)
fit_curve_raw = np.asarray(fit['fit_curve_raw'], dtype=float)
background = fit_curve_raw - fit_curve

# ---- Save fit overlay on RAW scale ----
np.save('fit.npy', np.asarray(fit_curve_raw, dtype=float))

# ---- Step 4: R^2 and RMSE over full modelled domain ----
resid = intensity - fit_curve_raw
ss_res = float(np.sum(resid ** 2))
ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = float(np.sqrt(np.mean(resid ** 2)))

# ---- Step 3: Per-peak Scherrer size ----
sizes_nm = []
for p in peaks:
    s = scherrer(
        fwhm_deg=p['fwhm'],
        two_theta_deg=p['center'],
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
    )
    sizes_nm.append(s['size_nm'])
mean_size_nm = float(np.nanmean(sizes_nm)) if sizes_nm else None

# ---- Williamson-Hall (optional, >=5 peaks) ----
wh_input = [{'two_theta': p['center'], 'fwhm': p['fwhm']} for p in peaks]
wh = williamson_hall(
    peaks=wh_input,
    wavelength_angstrom=WAVELENGTH_ANGSTROM,
    instrumental_fwhm_deg=INSTRUMENTAL_FWHM_DEG,
) if len(wh_input) >= 5 else None

# ---- Step 5: Visualization ----
noise = np.std(resid) if np.std(resid) > 0 else 1.0
norm_resid = resid / noise

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1, 1]})
ax0, ax1, ax2 = axes

ax0.plot(two_theta, intensity, color='0.25', lw=0.8, label='Data')
ax0.plot(two_theta, fit_curve_raw, color='crimson', lw=1.1, label='Fit')
ax0.plot(two_theta, background, color='steelblue', lw=0.8, ls='--', alpha=0.7,
         label='Background')
# individual components on raw scale = component (subtracted) + background
cmap = plt.get_cmap('viridis')
for i, p in enumerate(peaks):
    c = p['center']; fwhm = max(p['fwhm'], 1e-3); amp = p['amplitude']
    eta = p.get('eta', 0.5)
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((two_theta - c) / sig) ** 2)
    gam = fwhm / 2.0
    lor = gam ** 2 / ((two_theta - c) ** 2 + gam ** 2)
    comp = amp * (eta * lor + (1.0 - eta) * gauss)
    ax0.plot(two_theta, comp + background, lw=0.6, alpha=0.5,
             color=cmap(i / max(len(peaks), 1)))
ax0.set_yscale('log')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8, ncol=2)
ax0.set_xlim(two_theta.min(), two_theta.max())

ax1.plot(two_theta, resid, color='0.3', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.set_ylabel('Residuals')
ax1.set_xlim(two_theta.min(), two_theta.max())

ax2.plot(two_theta, norm_resid, color='darkorange', lw=0.7, label='Residuals / noise')
ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(3, color='r', lw=0.4, ls=':')
ax2.axhline(-3, color='r', lw=0.4, ls=':')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('X')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_xlim(two_theta.min(), two_theta.max())

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---- Step 6/7: Results JSON ----
params = {}
for i, p in enumerate(peaks, start=1):
    params[f"peak_{i}"] = {
        "center": float(p['center']),
        "fwhm": float(p['fwhm']),
        "fwhm_left": float(p.get('fwhm_left', np.nan)),
        "fwhm_right": float(p.get('fwhm_right', np.nan)),
        "asymmetry": float(p.get('asymmetry', np.nan)),
        "amplitude": float(p['amplitude']),
        "area": float(p['area']),
        "eta": float(p.get('eta', np.nan)),
        "scherrer_size_nm": float(sizes_nm[i - 1]) if i - 1 < len(sizes_nm) else None,
    }

results = {
    "model_type": "Sum of split pseudo-Voigt peaks on SNIP background (global fit_pattern, auto peak detection prominence_frac=0.01, min_distance_deg=0.08, fit_range 6-45 deg)",
    "parameters": params,
    "fit_quality": {
        "r_squared": float(r2_full),
        "rmse": float(rmse),
        "peak_region_r2": float(fit['peak_region_r2']),
        "global_r_squared": float(r_squared),
        "residual_rms_over_noise": float(fit['residual_rms_over_noise']),
        "max_abs_residual_over_noise": float(fit['max_abs_residual_over_noise']),
        "n_peaks_fitted": int(fit['n_peaks']),
        "number_of_reflections": int(fit['n_peaks']),
        "background_method": fit.get('background_method', 'snip'),
        "scherrer_mean_size_nm": mean_size_nm,
        "williamson_hall": wh,
    },
    "deviation_note": ""
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
