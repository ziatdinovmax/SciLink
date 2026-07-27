import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import wofz
from scipy.optimize import least_squares

from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
from scilink.skills.curve_fitting.nmr.detection import assess_detection

# ------------------------------------------------------------------
# 1. Load RAW data
# ------------------------------------------------------------------
arr = np.load('data.npy')
arr = np.asarray(arr, dtype=float)

# Expect a 2xN or Nx2 array (x, y). Handle both.
if arr.ndim == 2:
    if arr.shape[0] == 2:
        x = arr[0].astype(float)
        y = arr[1].astype(float)
    elif arr.shape[1] == 2:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
    else:
        x = arr[0].astype(float)
        y = arr[1].astype(float)
else:
    # 1-D: reconstruct x from known metadata range
    y = arr.astype(float)
    x = np.linspace(-40.8701, 47.3248, y.size)

# Ensure ascending x (skill recipe expects ascending ppm)
if x[0] > x[-1]:
    x = x[::-1].copy()
    y = y[::-1].copy()

N = y.size

# ------------------------------------------------------------------
# 2. Baseline: linear baseline from signal-free wings (plan).
#    Peaks live near -6.0, -5.6 and -3.5 ppm; wings are far away.
# ------------------------------------------------------------------
peak_lo, peak_hi = -12.0, 4.0
wing_mask = (x < peak_lo) | (x > peak_hi)

if np.count_nonzero(wing_mask) > 10:
    coeffs = np.polyfit(x[wing_mask], y[wing_mask], 1)
else:
    coeffs = np.polyfit(x, y, 1)
baseline = np.polyval(coeffs, x)
baseline_slope = float(coeffs[0])
baseline_intercept = float(coeffs[1])

# ------------------------------------------------------------------
# 3. Model: JOINT 3-Voigt fit on the (held) linear baseline.
#
#    NOTE ON APPROACH (deviation from the generic fit_multipeak_voigt call):
#    The locked plan requires EXACTLY three components with a physically
#    constrained inner-shoulder center (site2 between site1 and site3),
#    warm-started from the previous sharp+broad solution, with free,
#    independent Voigt widths per site. fit_multipeak_voigt chooses its
#    own component count and cannot pin the inner center between the two
#    horns, so per the plan I implement the joint 3-Voigt refit explicitly
#    with least_squares. This IS the plan's model class (Voigt profiles on a
#    linear baseline); only the fitter is a direct implementation to honor
#    the ordering/warm-start/constraint requirements.
# ------------------------------------------------------------------
# Work in baseline-subtracted space for the fit
yc = y - baseline

# Restrict fit domain per plan (-7.8 to 3.2 ppm); evaluate model on full grid later.
fit_lo, fit_hi = -7.8, 3.2
fmask = (x >= fit_lo) & (x <= fit_hi)
xf = x[fmask]
yf = yc[fmask]

# Each Voigt: amplitude (area-like scale), center, sigma (Gaussian), gamma (Lorentzian)
# Voigt profile normalized to unit area, scaled by amp.
def voigt_area(xx, center, amp, sigma, gamma):
    sigma = max(sigma, 1e-6)
    gamma = max(gamma, 0.0)
    z = ((xx - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

def voigt_fwhm(sigma, gamma):
    # Olivero & Longbothum approximation
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    return 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)

def model3(params, xx):
    m = np.zeros_like(xx)
    for k in range(3):
        c, a, s, g = params[4*k:4*k+4]
        m += voigt_area(xx, c, a, s, g)
    return m

def resid_fn(params, xx, yy):
    return model3(params, xx) - yy

# Warm-start seeds: site1 sharp horn ~-6.0, site2 inner shoulder ~-5.6,
# site3 broad Mg-coordinated peak ~-3.5.
peak_amp = float(np.max(yf))
# amplitude here is area scale; seed roughly amp*width
seed = [
    -6.0, peak_amp * 0.6, 0.15, 0.15,   # site1 sharp/narrow
    -5.6, peak_amp * 0.5, 0.4, 0.4,      # site2 inner shoulder
    -3.5, peak_amp * 2.5, 1.2, 1.2,      # site3 broad
]

# Bounds: positive amplitudes; site2 center constrained between site1 & site3.
# centers: site1 in [-7.0,-5.7], site2 in [-5.7,-4.5], site3 in [-4.5,-2.0]
lb = [
    -7.0, 0.0, 1e-3, 0.0,
    -5.7, 0.0, 1e-3, 0.0,
    -4.5, 0.0, 1e-3, 0.0,
]
ub = [
    -5.7, np.inf, 3.0, 3.0,
    -4.5, np.inf, 3.0, 3.0,
    -2.0, np.inf, 6.0, 6.0,
]
# clip seeds into bounds
seed = [min(max(s, lo), hi) for s, lo, hi in zip(seed, lb, ub)]

sol = least_squares(resid_fn, seed, args=(xf, yf), bounds=(lb, ub),
                    method='trf', max_nfev=20000, x_scale='jac')
popt = sol.x

# Parameter uncertainties from Jacobian
try:
    J = sol.jac
    dof = max(len(yf) - len(popt), 1)
    resid_fit = sol.fun
    s2 = float(np.sum(resid_fit**2) / dof)
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * s2
    perr = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
except Exception:
    perr = np.full(len(popt), np.nan)

# ------------------------------------------------------------------
# 4. Build full-grid model = baseline + 3 Voigts
# ------------------------------------------------------------------
y_peaks = model3(popt, x)
y_model = baseline + y_peaks

# Per-component (on baseline) for plotting and reporting
components = []
for k in range(3):
    c, a, s, g = popt[4*k:4*k+4]
    comp = voigt_area(x, c, a, s, g)
    fwhm = voigt_fwhm(s, g)
    area = float(trapezoid(comp, x))
    components.append({
        'center': float(c),
        'center_err': float(perr[4*k]) if np.isfinite(perr[4*k]) else None,
        'amplitude': float(a),
        'sigma': float(s),
        'gamma': float(g),
        'fwhm_ppm': float(fwhm),
        'lorentzian_gaussian_ratio': float(g / s) if s > 0 else None,
        'area': area,
        'comp_curve': comp,
    })

# sort by center ascending (site1 most negative)
components.sort(key=lambda d: d['center'])

# ------------------------------------------------------------------
# 5. Quality: peak_region_r2 (gate metric) + global R2 / RMSE
# ------------------------------------------------------------------
q = peak_region_r2(x.tolist(), y.tolist(), y_model.tolist(), baseline=baseline.tolist())
peak_r2 = q.get('peak_region_r2')
global_r2 = q.get('r_squared')

resid = y - y_model
rmse = float(np.sqrt(np.mean(resid**2)))
ss_res = float(np.sum(resid**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2_manual = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
if global_r2 is None:
    global_r2 = r2_manual

# ------------------------------------------------------------------
# 6. Detection gate (F-test for record)
# ------------------------------------------------------------------
n_params = 4 * 3 + 2  # 4/Voigt + 2 baseline
try:
    det = assess_detection(
        x.tolist(), y.tolist(), y_fit=y_model.tolist(),
        baseline=baseline.tolist(), n_model_params=n_params,
    )
    det_verdict = det.get('verdict')
    det_snr = det.get('snr')
except Exception:
    det_verdict = None
    det_snr = None

# ------------------------------------------------------------------
# 7. Per-site parameters + populations
# ------------------------------------------------------------------
nu_L = None  # Larmor frequency not provided in metadata

def fwhm_hz(fwhm_ppm):
    if nu_L is None or fwhm_ppm is None:
        return None
    return float(fwhm_ppm * nu_L)

total_area = sum(abs(c['area']) for c in components) or 1.0

site_labels = ['peak_1', 'peak_2', 'peak_3']
param_out = {}
for i, c in enumerate(components):
    label = site_labels[i]
    param_out[label] = {
        'center': c['center'],
        'center_err': c['center_err'],
        'fwhm_ppm': c['fwhm_ppm'],
        'fwhm_hz': fwhm_hz(c['fwhm_ppm']),
        'amplitude': c['amplitude'],
        'lorentzian_gaussian_ratio': c['lorentzian_gaussian_ratio'],
        'integrated_intensity': c['area'],
        'population_fraction': float(abs(c['area']) / total_area),
    }

# three-way population ratio (site1 : site2 : site3), normalized to site1
a1 = abs(components[0]['area'])
a2 = abs(components[1]['area'])
a3 = abs(components[2]['area'])
if a1 > 0:
    population_ratio = [1.0, float(a2 / a1), float(a3 / a1)]
else:
    population_ratio = [float(a1 / total_area), float(a2 / total_area), float(a3 / total_area)]

# ------------------------------------------------------------------
# 8. Visualization
# ------------------------------------------------------------------
noise = np.std(y[wing_mask]) if np.count_nonzero(wing_mask) > 10 else np.std(y)
if noise <= 0:
    noise = 1.0

fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                         gridspec_kw={'height_ratios': [3, 1.2, 1.6]})
ax_main, ax_res, ax_zoom = axes[0], axes[1], axes[2]

# Main panel: full domain
ax_main.plot(x, y, color='0.3', lw=0.8, label='Data')
ax_main.plot(x, y_model, color='C1', lw=1.3, label='Fit')
ax_main.plot(x, baseline, color='C2', lw=0.8, ls='--', label='Baseline')
for i, c in enumerate(components):
    ax_main.plot(x, baseline + c['comp_curve'], lw=0.9, ls=':', label=f'Component {i+1}')
ax_main.set_xlim(x.min(), x.max())
ax_main.invert_xaxis()
ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.legend(loc='upper right', fontsize=8)

# Residual panel: raw + normalized
ax_res.plot(x, resid, color='0.3', lw=0.7, label='Residuals')
ax_res.axhline(0, color='k', lw=0.5)
ax_res.set_xlim(x.min(), x.max())
ax_res.invert_xaxis()
ax_res.set_ylabel('Residual')
ax_res_n = ax_res.twinx()
ax_res_n.plot(x, resid / noise, color='C3', lw=0.5, alpha=0.5)
ax_res_n.set_ylabel('Resid / noise', color='C3')
ax_res.legend(loc='upper right', fontsize=8)

# Zoom panel over peak region (own non-shared x-axis)
zmask = (x > fit_lo) & (x < fit_hi)
if np.count_nonzero(zmask) > 5:
    ax_zoom.plot(x[zmask], y[zmask], color='0.3', lw=0.9, label='Data')
    ax_zoom.plot(x[zmask], y_model[zmask], color='C1', lw=1.3, label='Fit')
    for i, c in enumerate(components):
        ax_zoom.plot(x[zmask], baseline[zmask] + c['comp_curve'][zmask], lw=0.8, ls=':', label=f'Component {i+1}')
    ax_zoom.set_xlim(x[zmask].min(), x[zmask].max())
    ax_zoom.invert_xaxis()
ax_zoom.set_xlabel('X')
ax_zoom.set_ylabel('Y')
ax_zoom.set_title('Peak-region zoom')
ax_zoom.legend(loc='upper right', fontsize=8)

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ------------------------------------------------------------------
# 9. Save fitted model at all N x-points
# ------------------------------------------------------------------
np.save('fit.npy', y_model.astype(float))

# ------------------------------------------------------------------
# 10. Emit results
# ------------------------------------------------------------------
results = {
    'model_type': 'Sum of 3 Voigt profiles on a linear baseline (joint refit; site2 inner shoulder constrained between sharp horn and broad peak)',
    'parameters': param_out,
    'derived': {
        'n_components': 3,
        'population_ratio_site1_site2_site3': population_ratio,
        'baseline_slope': baseline_slope,
        'baseline_intercept': baseline_intercept,
        'detection_verdict': det_verdict,
        'detection_snr': (float(det_snr) if det_snr is not None else None),
    },
    'fit_quality': {
        'r_squared': float(global_r2) if global_r2 is not None else None,
        'rmse': rmse,
        'peak_region_r2': float(peak_r2) if peak_r2 is not None else None,
        'residual_structured': bool(q.get('residual_structured')) if q.get('residual_structured') is not None else None,
        'apex_resid_frac': q.get('apex_resid_frac'),
        'resid_rms_frac': q.get('resid_rms_frac'),
    },
    'deviation_note': "Implemented the joint 3-Voigt refit directly with scipy.least_squares (not fit_multipeak_voigt) to honor the plan's fixed 3-component count, physical center ordering, inner-shoulder-between-horns constraint, and warm-start requirement, which the auto-count tool cannot enforce. Model class (Voigt on linear baseline) unchanged. Larmor frequency not in metadata; FWHM reported in ppm only (Hz null)."
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
