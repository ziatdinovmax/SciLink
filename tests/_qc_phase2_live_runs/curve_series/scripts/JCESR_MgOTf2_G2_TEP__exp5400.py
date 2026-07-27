import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.special import wofz

from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
from scilink.skills.curve_fitting.nmr.detection import assess_detection

# ------------------------------------------------------------------
# 1. Load RAW data
# ------------------------------------------------------------------
arr = np.load('data.npy')
arr = np.asarray(arr, dtype=float)
if arr.ndim == 2:
    if arr.shape[0] == 2:
        x, y = arr[0], arr[1]
    elif arr.shape[1] == 2:
        x, y = arr[:, 0], arr[:, 1]
    else:
        x, y = arr[0], arr[1]
else:
    y = arr
    x = np.linspace(-40.8701, 47.3248, y.size)

# ensure ascending ppm
if x[0] > x[-1]:
    x = x[::-1].copy()
    y = y[::-1].copy()

N = y.size

# ------------------------------------------------------------------
# 2. Baseline: constant from |x|>15 ppm wings (per plan)
# ------------------------------------------------------------------
wing_mask = np.abs(x) > 15.0
if wing_mask.sum() < 50:
    wing_mask = np.abs(x) > 10.0
baseline_const = float(np.median(y[wing_mask]))
baseline = np.full_like(y, baseline_const)
noise = float(np.std(y[wing_mask]))
if noise <= 0:
    noise = 1.0

yc = y - baseline_const

# ------------------------------------------------------------------
# 3. Explicit 3-component Voigt model with full Voigt freedom
#    (per updated plan): sharp near -6.0, intermediate shoulder near
#    -5.6, broad near -3.5. Each Voigt uses independent Gaussian
#    (sigma) and Lorentzian (gamma) widths -> full Voigt shape.
#    Fit performed manually (least_squares) because the plan locks a
#    3-component sum-of-Voigt model with co-located sharp+shoulder
#    and specific full-Voigt freedom that the auto multipeak tool
#    (parsimony-driven, single width class) does not express.
# ------------------------------------------------------------------

def voigt_area(xv, center, sigma, gamma, area):
    """Area-normalized Voigt profile scaled to integrated `area`.
    sigma = Gaussian std (ppm), gamma = Lorentzian HWHM (ppm)."""
    sigma = max(sigma, 1e-6)
    z = ((xv - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    v = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return area * v


def fwhm_from_sigma_gamma(sigma, gamma):
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)


# Fit domain: signal region containing the peaks (per plan the model
# lives in the -12..+5 ppm window). We STILL evaluate the model on
# all N x-points for output; the fit is weighted on the peak region.
fit_lo, fit_hi = -12.0, 5.0
dom = (x >= fit_lo) & (x <= fit_hi)
xd = x[dom]
yd = yc[dom]

# Emphasis weighting on the flagged residual windows:
#   inner shoulder/valley x in [-6.0, -5.5]
#   broad-top / downfield x in [-2.5, +1.0]
w = np.ones_like(xd)
w[(xd >= -6.0) & (xd <= -5.5)] = 3.0
w[(xd >= -2.5) & (xd <= 1.0)] = 2.0

# Amplitude scale for seeding
ymax = float(np.max(yd))

# Parameter vector (11 peak params + baseline delta):
# [c_s, sig_s, gam_s, A_s,  c_sh, sig_sh, gam_sh, A_sh,  c_b, sig_b, gam_b, A_b]
# Seed areas from rough peak-height * width estimates.
p0 = [
    -6.0, 0.15, 0.15, ymax * 0.5,      # sharp
    -5.6, 0.30, 0.30, ymax * 0.05,     # intermediate shoulder (minor)
    -3.5, 1.0, 1.0, ymax * 2.0,        # broad
]

lb = [
    -6.5, 1e-3, 1e-3, 0.0,
    -5.9, 1e-3, 1e-3, 0.0,
    -5.0, 1e-3, 1e-3, 0.0,
]
ub = [
    -5.5, 2.0, 2.0, np.inf,
    -5.3, 1.5, 1.5, ymax * 1.0,   # keep shoulder minor / bounded
    -1.5, 6.0, 6.0, np.inf,
]


def model_on(xv, p):
    c_s, sig_s, gam_s, A_s = p[0:4]
    c_sh, sig_sh, gam_sh, A_sh = p[4:8]
    c_b, sig_b, gam_b, A_b = p[8:12]
    return (voigt_area(xv, c_s, sig_s, gam_s, A_s)
            + voigt_area(xv, c_sh, sig_sh, gam_sh, A_sh)
            + voigt_area(xv, c_b, sig_b, gam_b, A_b))


def resid_fun(p):
    return (model_on(xd, p) - yd) * w


# Stage 1: fit two-component (sharp + broad) first, shoulder ~0.
p0_2 = list(p0)
p0_2[7] = 0.0
sol2 = least_squares(resid_fun, p0_2, bounds=(lb, ub),
                     method='trf', max_nfev=20000)

# Stage 2: release intermediate shoulder from the 2-comp solution.
p_start = sol2.x.copy()
if p_start[7] < ymax * 0.02:
    p_start[7] = ymax * 0.05
sol = least_squares(resid_fun, p_start, bounds=(lb, ub),
                    method='trf', max_nfev=40000)

pfit = sol.x

# ------------------------------------------------------------------
# Parameter uncertainties from the Jacobian (covariance matrix)
# ------------------------------------------------------------------
def param_errors(sol_obj, npar):
    try:
        J = sol_obj.jac
        dof = max(J.shape[0] - J.shape[1], 1)
        rss = float(np.sum(sol_obj.fun ** 2))
        s2 = rss / dof
        JTJ = J.T @ J
        cov = np.linalg.pinv(JTJ) * s2
        perr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except Exception:
        perr = np.full(npar, np.nan)
    return perr

perr = param_errors(sol, len(pfit))

# ------------------------------------------------------------------
# 4. Evaluate model on ALL N x-points; add baseline back
# ------------------------------------------------------------------
y_model_peaks = model_on(x, pfit)
y_fit = y_model_peaks + baseline_const

# component curves (peak-only, no baseline) for plotting
comp_curves = []
for k in range(3):
    c, sig, gam, A = pfit[4 * k:4 * k + 4]
    comp_curves.append(voigt_area(x, c, sig, gam, A))

# ------------------------------------------------------------------
# 5. Quality: peak-region R2 (gate metric) + global R2/RMSE
# ------------------------------------------------------------------
q = peak_region_r2(x.tolist(), y.tolist(), y_fit.tolist(), baseline=baseline.tolist())
peak_r2 = float(q['peak_region_r2'])
global_r2 = float(q['r_squared'])
resid = y - y_fit
rmse = float(np.sqrt(np.mean(resid ** 2)))
resid_structured = bool(q.get('residual_structured', False))
autocorr = float(q.get('resid_autocorr_lag1', np.nan))

# ------------------------------------------------------------------
# F-test: 3-component vs 2-component on peak_region_r2 (guard overfit)
# ------------------------------------------------------------------
y_fit2 = model_on(x, sol2.x) + baseline_const
q2 = peak_region_r2(x.tolist(), y.tolist(), y_fit2.tolist(), baseline=baseline.tolist())
# F-test on the fit-domain residual sums of squares
rss3 = float(np.sum((model_on(xd, pfit) - yd) ** 2))
rss2 = float(np.sum((model_on(xd, sol2.x) - yd) ** 2))
n_obs = xd.size
p3, p2 = 12, 8
try:
    from scipy.stats import f as f_dist
    F_stat = ((rss2 - rss3) / (p3 - p2)) / (rss3 / (n_obs - p3))
    F_pvalue = float(1.0 - f_dist.cdf(F_stat, p3 - p2, n_obs - p3))
except Exception:
    F_stat, F_pvalue = np.nan, np.nan

# Detection screen with F-test (peak vs baseline-only)
n_params = len(pfit) + 1
det = assess_detection(
    x.tolist(), y.tolist(), y_fit=y_fit.tolist(),
    baseline=baseline.tolist(), n_model_params=n_params,
)

# ------------------------------------------------------------------
# 6. Relative populations from integrated Voigt areas
# ------------------------------------------------------------------
names = ['sharp', 'shoulder', 'broad']
areas = [abs(pfit[4 * k + 3]) for k in range(3)]
total_area = sum(areas)
if total_area <= 0:
    total_area = 1.0

params = {}
for k, nm in enumerate(names):
    c, sig, gam, A = pfit[4 * k:4 * k + 4]
    ec, esig, egam, eA = perr[4 * k:4 * k + 4]
    fwhm = fwhm_from_sigma_gamma(sig, gam)
    params[f"peak_{k+1}_{nm}"] = {
        "center_ppm": float(c),
        "center_err": float(ec),
        "sigma_gauss_ppm": float(sig),
        "gamma_lorentz_ppm": float(gam),
        "fwhm_ppm": float(fwhm),
        "amplitude": float(np.max(comp_curves[k])),
        "area": float(abs(A)),
        "area_err": float(eA),
        "relative_population": float(abs(A) / total_area),
    }

# Group sharp+shoulder (same physical environment) vs broad site
grouped_pop = {
    "sharp_plus_shoulder": float((areas[0] + areas[1]) / total_area),
    "broad": float(areas[2] / total_area),
}
area_ratio = float((areas[0] + areas[1]) / areas[2]) if areas[2] > 0 else None

# ------------------------------------------------------------------
# 7. Visualization
# ------------------------------------------------------------------
sig_idx = np.where((y - baseline_const) > 5.0 * noise)[0]
if sig_idx.size > 0:
    lo = x[max(sig_idx.min() - 200, 0)]
    hi = x[min(sig_idx.max() + 200, N - 1)]
    if lo > hi:
        lo, hi = hi, lo
else:
    lo, hi = -12.0, 2.0

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
ax0, ax1, ax2 = axes

colors = ['C2', 'C3', 'C4']

# main panel: full domain
ax0.plot(x, y, color='0.3', lw=0.6, label='Data')
ax0.plot(x, y_fit, color='C1', lw=1.0, label='Fit')
for k in range(3):
    ax0.plot(x, comp_curves[k] + baseline_const, color=colors[k], lw=0.8,
             label=f'Component {k+1}')
ax0.plot(x, baseline, color='0.7', lw=0.8, ls=':', label='Baseline')
ax0.set_xlim(x.min(), x.max())
ax0.set_ylabel('Y')
ax0.set_xlabel('X')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8)

# zoom panel over signal region (own x-axis, not shared)
ax1.plot(x, y, color='0.3', lw=0.7, label='Data')
ax1.plot(x, y_fit, color='C1', lw=1.0, label='Fit')
for k in range(3):
    ax1.plot(x, comp_curves[k] + baseline_const, color=colors[k], lw=0.8,
             label=f'Component {k+1}')
ax1.set_xlim(lo, hi)
ax1.set_ylabel('Y')
ax1.set_xlabel('X (zoom on signal region)')
ax1.set_title('Fit (zoom)')
ax1.legend(loc='upper right', fontsize=8)

# residual panel: normalized residual
ax2.plot(x, resid / noise, color='C0', lw=0.6, label='Residuals / noise')
ax2.axhline(0, color='0.5', lw=0.6)
ax2.axhline(3, color='0.7', lw=0.5, ls=':')
ax2.axhline(-3, color='0.7', lw=0.5, ls=':')
ax2.set_xlim(lo, hi)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X (zoom on signal region)')
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('visualization.png', dpi=130)
plt.close(fig)

# ------------------------------------------------------------------
# 8. Save fit array at all N x-points
# ------------------------------------------------------------------
np.save('fit.npy', y_fit.astype(float))

# ------------------------------------------------------------------
# 9. Emit results JSON
# ------------------------------------------------------------------
results = {
    "model_type": "Sum of 3 full-Voigt line(s) (independent Gaussian+Lorentzian widths) on a constant baseline: sharp ~-6.0, intermediate shoulder ~-5.6, broad ~-3.5",
    "parameters": params,
    "grouped_relative_populations": grouped_pop,
    "component_count": 3,
    "area_ratio_sharpshoulder_over_broad": area_ratio,
    "baseline_constant": baseline_const,
    "model_comparison": {
        "peak_region_r2_2comp": float(q2['peak_region_r2']),
        "peak_region_r2_3comp": peak_r2,
        "F_stat_3v2": float(F_stat) if F_stat == F_stat else None,
        "F_pvalue_3v2": float(F_pvalue) if F_pvalue == F_pvalue else None,
    },
    "detection": {
        "verdict": det.get('verdict'),
        "snr": det.get('snr'),
        "p_value": det.get('p_value'),
        "significant": det.get('significant'),
    },
    "fit_quality": {
        "peak_region_r2": peak_r2,
        "r_squared": global_r2,
        "rmse": rmse,
        "residual_structured": resid_structured,
        "resid_autocorr_lag1": autocorr,
    },
    "deviation_note": "Fit performed with an explicit 3-component full-Voigt least_squares model (independent Gaussian sigma + Lorentzian gamma per component) instead of the auto fit_multipeak_voigt tool, because the locked plan specifies a fixed 3-component sum-of-Voigt with a co-located sharp+shoulder pair near -6/-5.6 ppm and per-component free Gaussian fractions plus emphasis weighting on named residual windows and a 3-vs-2 F-test \u2014 constraints the parsimony-driven single-width auto tool does not express."
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
