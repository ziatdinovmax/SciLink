import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import wofz
from scipy.optimize import least_squares

from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt
from scilink.skills.curve_fitting.nmr.detection import assess_detection
from scilink.skills.curve_fitting.nmr.quality import peak_region_r2

# ----------------------------------------------------------------------
# 1. Load RAW data
# ----------------------------------------------------------------------
arr = np.load('data.npy')
arr = np.asarray(arr, dtype=float)

if arr.ndim == 2:
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        x = arr[0].astype(float)
        y = arr[1].astype(float)
    elif arr.shape[1] == 2:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
    else:
        x = arr[0].astype(float)
        y = arr[1].astype(float)
else:
    y = arr.astype(float)
    x = np.linspace(-40.8701, 47.3248, y.size)

N = y.size

# Ensure ascending ppm for the baseline/fit tools
if x[0] > x[-1]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    resort = True
else:
    order = np.arange(N)
    resort = False

# ----------------------------------------------------------------------
# 2. Preprocessing: metadata -> phase/sign -> baseline -> fit
# ----------------------------------------------------------------------
if abs(np.max(y)) < abs(np.min(y)):
    y_used = -y
    sign_flip = True
else:
    y_used = y.copy()
    sign_flip = False

# Signal-free mask for a linear baseline (plan: linear baseline as parameter)
ynorm = np.abs(y_used - np.median(y_used))
med = np.median(ynorm)
mad = np.median(np.abs(ynorm - med))
thr = med + 2.0 * 1.4826 * mad
free = ynorm < thr

# Also EXCLUDE the known signal window (-7.8 .. 3.2) from the baseline anchor
# per plan Step 3: anchor the linear baseline strictly on the peak-free wings
# so it does not absorb genuine flank curvature (source of the S-shaped residual).
signal_window = (x >= -7.8) & (x <= 3.2)
free_wings = free & (~signal_window)
if free_wings.sum() > 10:
    free = free_wings

if free.sum() > 10:
    coeffs = np.polyfit(x[free], y_used[free], 1)
    baseline = np.polyval(coeffs, x)
else:
    baseline = np.full(N, np.median(y_used))
    coeffs = np.array([0.0, np.median(y_used)])

baseline_slope = float(coeffs[0])
baseline_offset = float(coeffs[1])

noise_sigma = 1.4826 * mad if mad > 0 else (np.std(y_used[free]) if free.sum() > 5 else np.std(y_used))
if not np.isfinite(noise_sigma) or noise_sigma <= 0:
    noise_sigma = np.std(y_used)

# ----------------------------------------------------------------------
# 3. Reconcile the component count: FIRST fit a SINGLE Voigt with center,
#    FWHM, AND the L/G mixing fraction ALL free (previous run pinned the shape),
#    over the plan fit domain (-7.8 .. 3.2 signal region, evaluated on full x).
#    A pseudo-Voigt (eta = L/G mixing) is used so eta is an explicit free param
#    per the plan (peak_1_LG_mixing_fraction).
# ----------------------------------------------------------------------
yb = y_used - baseline  # baseline-subtracted for the peak fit

# fit domain per plan
fit_mask = (x >= -7.8) & (x <= 3.2)
xf = x[fit_mask]
yf = yb[fit_mask]


def pseudo_voigt(xx, amp, cen, fwhm, eta):
    """eta*Lorentzian + (1-eta)*Gaussian, peak-normalized to amp."""
    fwhm = max(abs(fwhm), 1e-6)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    G = np.exp(-((xx - cen) ** 2) / (2.0 * sigma ** 2))
    L = 1.0 / (1.0 + ((xx - cen) / gamma) ** 2)
    return amp * (eta * L + (1.0 - eta) * G)


# Seed center from apex within the signal window
apex_idx = np.argmax(yf)
cen0 = float(xf[apex_idx])
amp0 = float(yf[apex_idx])
# rough FWHM from half-max crossing
half = amp0 / 2.0
above = xf[yf >= half]
if above.size >= 2:
    fwhm0 = float(above.max() - above.min())
else:
    fwhm0 = 2.0
fwhm0 = max(fwhm0, 0.5)


def resid1(p):
    amp, cen, fwhm, eta = p
    return pseudo_voigt(xf, amp, cen, fwhm, eta) - yf


p0 = [amp0, cen0, fwhm0, 0.5]
lb = [0.0, cen0 - 3.0, 0.1, 0.0]
ub = [amp0 * 5.0, cen0 + 3.0, 30.0, 1.0]
sol1 = least_squares(resid1, p0, bounds=(lb, ub), max_nfev=20000)
p1 = sol1.x
y1_full = pseudo_voigt(x, *p1)  # single-Voigt model on full x (peak part)
rss1 = float(np.sum((pseudo_voigt(xf, *p1) - yf) ** 2))
n1 = 4

# ----------------------------------------------------------------------
# 4. Re-inspect residual; run F-test for a SECOND Voigt seeded on the
#    residual-negative-lobe side (just past center) per plan Step 4.
# ----------------------------------------------------------------------
resid_lobe = yf - pseudo_voigt(xf, *p1)
# seed 2nd center at largest |residual| just past center
cen2_seed = float(xf[np.argmax(np.abs(resid_lobe))])
amp2_seed = float(resid_lobe[np.argmax(np.abs(resid_lobe))])


def resid2(p):
    a1, c1, w1, e1, a2, c2, w2, e2 = p
    m = pseudo_voigt(xf, a1, c1, w1, e1) + pseudo_voigt(xf, a2, c2, w2, e2)
    return m - yf


p0_2 = [p1[0], p1[1], p1[2], p1[3],
        abs(amp2_seed) if amp2_seed != 0 else amp0 * 0.1, cen2_seed, fwhm0, 0.5]
lb2 = [0.0, cen0 - 3.0, 0.1, 0.0, 0.0, cen0 - 4.0, 0.1, 0.0]
ub2 = [amp0 * 5.0, cen0 + 3.0, 30.0, 1.0, amp0 * 5.0, cen0 + 4.0, 30.0, 1.0]
sol2 = least_squares(resid2, p0_2, bounds=(lb2, ub2), max_nfev=30000)
p2 = sol2.x
rss2 = float(np.sum((resid2(p2)) ** 2))
n2 = 8

# F-test: is the 2nd component justified?
m = xf.size
df1 = m - n1
df2 = m - n2
if rss2 > 0 and df2 > 0 and (n2 - n1) > 0:
    F = ((rss1 - rss2) / (n2 - n1)) / (rss2 / df2)
    from scipy.stats import f as fdist
    p_value = float(1.0 - fdist.cdf(F, n2 - n1, df2))
else:
    F = 0.0
    p_value = 1.0
F = float(F)

# peak-region R2 of each candidate on the signal region for the improvement gate


def peak_r2_of(model_full):
    q = peak_region_r2(x.tolist(), y_used.tolist(), (model_full + baseline).tolist(),
                       baseline=baseline.tolist())
    return q


q1 = peak_r2_of(y1_full)
y2_full = pseudo_voigt(x, *p2[:4]) + pseudo_voigt(x, *p2[4:])
q2 = peak_r2_of(y2_full)

r2_gain = float(q2['peak_region_r2'] - q1['peak_region_r2'])

# Gate: keep 2nd component only if F-test significant (p<0.01) AND peak-region
# R2 improves materially (>=0.005) in the -7.8..-2.3/signal window.
accept_second = (p_value < 0.01) and (r2_gain >= 0.005)

if accept_second:
    n_peaks = 2
    pbest = p2
    y_peak_full = y2_full
    comp_params = [p2[:4], p2[4:]]
else:
    n_peaks = 1
    pbest = p1
    y_peak_full = y1_full
    comp_params = [p1]

y_fit_used = y_peak_full + baseline

# ----------------------------------------------------------------------
# 5. Detection gate (F-test peak-vs-baseline)
# ----------------------------------------------------------------------
n_params = 4 * n_peaks + 2
det = assess_detection(
    x.tolist(), y_used.tolist(), y_fit=y_fit_used.tolist(),
    baseline=baseline.tolist(), n_model_params=n_params
)

# ----------------------------------------------------------------------
# 6. Quality metrics
# ----------------------------------------------------------------------
q = peak_region_r2(
    x.tolist(), y_used.tolist(), y_fit_used.tolist(), baseline=baseline.tolist()
)
peak_r2 = q['peak_region_r2']
global_r2 = q['r_squared']
resid_used = y_used - y_fit_used
rmse = float(np.sqrt(np.mean(resid_used ** 2)))

# residual sign-change count within the -7.8..-2.3 window (S-shape diagnostic)
diag_mask = (x >= -7.8) & (x <= -2.3)
rd = resid_used[diag_mask]
sgn = np.sign(rd)
sgn = sgn[sgn != 0]
sign_changes = int(np.sum(np.diff(sgn) != 0)) if sgn.size > 1 else 0
resid_rms_window = float(np.sqrt(np.mean(rd ** 2))) if rd.size else float('nan')

# ----------------------------------------------------------------------
# 7. Build fitted model in ORIGINAL sign / order for output
# ----------------------------------------------------------------------
y_fit_orig_sign = -y_fit_used if sign_flip else y_fit_used
if resort:
    inv = np.argsort(order)
    fit_out = y_fit_orig_sign[inv]
else:
    fit_out = y_fit_orig_sign
np.save('fit.npy', fit_out.astype(float))

# ----------------------------------------------------------------------
# 8. Per-peak parameters (integrated intensity via numeric integration)
# ----------------------------------------------------------------------
parameters = {}
for i, cp in enumerate(comp_params, start=1):
    amp, cen, fwhm, eta = cp
    comp_curve = pseudo_voigt(x, amp, cen, fwhm, eta)
    area = float(trapezoid(comp_curve, x))
    parameters[f"peak_{i}"] = {
        "center": float(cen),
        "fwhm_ppm": float(abs(fwhm)),
        "LG_mixing_fraction": float(eta),
        "amplitude": float(amp),
        "integrated_intensity": abs(area),
    }
parameters["component_count"] = int(n_peaks)
parameters["baseline_slope"] = baseline_slope
parameters["baseline_offset"] = baseline_offset
parameters["F_test"] = {"F_statistic": F, "p_value": p_value,
                        "peak_region_r2_gain": r2_gain,
                        "second_component_accepted": bool(accept_second)}
parameters["residual_window_-7.8_-2.3"] = {
    "rms": resid_rms_window, "sign_changes": sign_changes}

# ----------------------------------------------------------------------
# 9. Visualization
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
ax0, ax1, ax2 = axes

ax0.plot(x, y_used, color='0.25', lw=0.7, label='Data')
ax0.plot(x, baseline, color='green', lw=0.8, ls='--', alpha=0.7, label='Baseline')
ax0.plot(x, y_fit_used, color='red', lw=1.2, label='Fit')
for i, cp in enumerate(comp_params, start=1):
    comp_curve = pseudo_voigt(x, *cp) + baseline
    ax0.plot(x, comp_curve, color='C%d' % (i % 10), lw=0.9, ls='-',
             alpha=0.8, label='Component %d' % i)
ax0.set_xlim(x.min(), x.max())
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8)

ax1.plot(x, resid_used, color='0.3', lw=0.7)
ax1.axhline(0, color='red', lw=0.6)
ax1.set_xlim(x.min(), x.max())
ax1.set_xlabel('X')
ax1.set_ylabel('Residual')
ax1.set_title('Residuals')

# normalized residual over the diagnostic window (own non-shared x-axis)
if noise_sigma and noise_sigma > 0:
    ax2.plot(x, resid_used / noise_sigma, color='navy', lw=0.7)
else:
    ax2.plot(x, resid_used, color='navy', lw=0.7)
ax2.axhline(0, color='red', lw=0.6)
for s in (-3, 3):
    ax2.axhline(s, color='orange', lw=0.5, ls='--')
ax2.set_xlim(-7.8, -2.3)  # zoom on S-shape window; own axis, does not affect ax0
ax2.set_xlabel('X (zoom -7.8 to -2.3)')
ax2.set_ylabel('Residual / noise')
ax2.set_title('Normalized residuals (S-shape window)')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# 10. Emit JSON
# ----------------------------------------------------------------------
results = {
    "model_type": ("Sum of %d pseudo-Voigt profile(s) (center, FWHM, and L/G "
                   "mixing fraction eta all free) on a linear baseline anchored "
                   "on peak-free wings; 2nd component F-test + peak-region-R2 "
                   "gated (1=fast exchange, 2=slow exchange)" % n_peaks),
    "parameters": parameters,
    "fit_quality": {
        "peak_region_r2": float(peak_r2),
        "r_squared": float(global_r2),
        "rmse": rmse,
        "residual_structured": bool(q.get('residual_structured', False)),
        "apex_resid_frac": float(q.get('apex_resid_frac', float('nan'))),
        "detection_verdict": det.get('verdict'),
        "detection_snr": det.get('snr'),
    },
    "deviation_note": ("Implemented the single-Voigt lineshape as an explicit "
                       "pseudo-Voigt with a FREE L/G mixing fraction (eta) and "
                       "a custom least_squares fit rather than the "
                       "fit_multipeak_voigt tool, because the plan requires the "
                       "L/G mixing fraction to be an explicit free parameter and "
                       "the S-shaped residual fix requires re-seeding center/FWHM "
                       "and freeing eta; the tool does not expose eta. Second "
                       "component still gated by an explicit F-test + peak-region-R2 "
                       "improvement per the plan.")
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
