import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from lmfit.models import VoigtModel, LinearModel

# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data, dtype=float)

# Expect a 2xN or Nx2 array (ppm, intensity)
if data.ndim == 2:
    if data.shape[0] == 2 and data.shape[1] != 2:
        x = data[0].astype(float)
        y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
    else:
        # fallback: first row/col is x
        x = data[0].astype(float)
        y = data[1].astype(float)
else:
    # 1-D intensity only; build ppm axis from known range
    y = data.astype(float)
    x = np.linspace(-40.8701, 47.3248, y.size)

# Ensure ascending x (spline / fitting convenience)
if x[0] > x[-1]:
    x = x[::-1].copy()
    y = y[::-1].copy()

N = x.size

# ------------------------------------------------------------------
# 2. Preprocessing: data is clean, high-SNR intensity spectrum.
#    No smoothing (would distort widths). Baseline treated as a
#    fitted linear model parameter per the plan (linear baseline
#    from the wings). No clipping of negatives beyond leaving them
#    for the baseline model.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 3. Model: single Voigt on a linear baseline, seeded near -2.5 ppm,
#    positive amplitude, free Gaussian + Lorentzian widths.
#    (Plan: single Voigt, high-T fast-exchange averaged regime.)
# ------------------------------------------------------------------
voigt = VoigtModel(prefix='v_')
lin = LinearModel(prefix='b_')
model = voigt + lin

# --- Seed baseline from the wings (points far from the resonance) ---
seed_center = -2.5
wing_mask = np.abs(x - seed_center) > 15.0  # outer wings, far from peak
if wing_mask.sum() < 20:
    wing_mask = (x < np.percentile(x, 10)) | (x > np.percentile(x, 90))
wslope, wintercept = np.polyfit(x[wing_mask], y[wing_mask], 1)

# --- Seed the Voigt near -2.5 ppm ---
# find local maximum of baseline-corrected signal near seed
base_seed = wslope * x + wintercept
ycorr = y - base_seed
search = np.abs(x - seed_center) < 20.0
if search.sum() > 0:
    idx_local = np.argmax(ycorr[search])
    center_seed = x[search][idx_local]
    amp_height = ycorr[search][idx_local]
else:
    center_seed = seed_center
    amp_height = np.max(ycorr)
if amp_height <= 0:
    amp_height = np.max(np.abs(ycorr))
    center_seed = seed_center

# rough width seed (ppm) from data spacing
dx = np.abs(np.median(np.diff(x)))
sigma_seed = max(0.1, 5 * dx)
# Voigt area amplitude ~ height * width * factor
amp_seed = amp_height * sigma_seed * 3.0

params = model.make_params()
params['b_slope'].set(value=wslope)
params['b_intercept'].set(value=wintercept)
params['v_amplitude'].set(value=amp_seed, min=0.0)   # positive amplitude
params['v_center'].set(value=center_seed, min=center_seed - 10, max=center_seed + 10)
params['v_sigma'].set(value=sigma_seed, min=1e-4)
# free Lorentzian width (gamma) -- by default lmfit ties gamma=sigma; free it
params['v_gamma'].set(value=sigma_seed, min=1e-4, vary=True, expr='')

result = model.fit(y, params, x=x)

y_fit = result.best_fit
baseline = result.eval_components(x=x)['b_']
y_peak = y_fit - baseline

# ------------------------------------------------------------------
# 4. R^2 and RMSE over the full modelled domain
# ------------------------------------------------------------------
resid = y - y_fit
ss_res = float(np.sum(resid**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = float(np.sqrt(np.mean(resid**2)))

# ------------------------------------------------------------------
# Extract fitted parameters
# ------------------------------------------------------------------
def pval(name):
    p = result.params.get(name)
    return (float(p.value) if p is not None and p.value is not None else None)

def perr(name):
    p = result.params.get(name)
    return (float(p.stderr) if p is not None and p.stderr is not None else None)

delta_iso = pval('v_center')
delta_iso_err = perr('v_center')
fwhm = pval('v_fwhm')          # lmfit provides Voigt FWHM (ppm)
fwhm_err = perr('v_fwhm')
sigma = pval('v_sigma')
gamma = pval('v_gamma')

# Integrated intensity (area) of the Voigt component = v_amplitude,
# also verified by numerical integration of the peak.
integrated_intensity = pval('v_amplitude')
integrated_intensity_err = perr('v_amplitude')
integral_numeric = float(trapezoid(np.clip(y_peak, 0, None), x))

# FWHM in Hz would need the Larmor frequency (not provided in metadata);
# report FWHM in ppm only.

# ------------------------------------------------------------------
# 5. Visualization: data + fit + residuals
# ------------------------------------------------------------------
noise_mask = np.abs(x - (delta_iso if delta_iso is not None else seed_center)) > 15.0
if noise_mask.sum() > 10:
    noise = float(np.std((y - y_fit)[noise_mask]))
else:
    noise = float(np.std(y - y_fit))
if noise <= 0:
    noise = 1.0

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.35)

ax0 = fig.add_subplot(gs[0])
ax0.plot(x, y, color='0.3', lw=0.8, label='Data')
ax0.plot(x, y_fit, color='C1', lw=1.2, label='Fit')
ax0.plot(x, baseline, color='C2', lw=1.0, ls='--', label='Baseline')
ax0.plot(x, y_peak + baseline, color='C0', lw=0.9, alpha=0.7, label='Component 1')
ax0.set_xlim(x.min(), x.max())
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(loc='upper right', fontsize=8)

# raw residual
ax1 = fig.add_subplot(gs[1])
ax1.plot(x, resid, color='0.3', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlim(x.min(), x.max())
ax1.set_ylabel('Residuals')

# normalized residual (residual / noise) -- own x-axis, not shared
ax2 = fig.add_subplot(gs[2])
ax2.plot(x, resid / noise, color='C3', lw=0.7)
ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(3, color='0.6', lw=0.5, ls=':')
ax2.axhline(-3, color='0.6', lw=0.5, ls=':')
ax2.set_xlim(x.min(), x.max())
ax2.set_ylabel('Residuals / noise')
ax2.set_xlabel('X')

fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# 6. Save fitted model evaluated at same x-points (length N)
# ------------------------------------------------------------------
np.save('fit.npy', np.asarray(y_fit, dtype=float))

# ------------------------------------------------------------------
# Peak-region R2 (gate metric) via skill tool
# ------------------------------------------------------------------
peak_region_r2_val = None
global_r2_tool = None
residual_structured = None
try:
    from scilink.skills.curve_fitting.nmr.quality import peak_region_r2
    q = peak_region_r2(x.tolist(), y.tolist(), y_fit.tolist(),
                       baseline=baseline.tolist())
    peak_region_r2_val = q.get('peak_region_r2')
    global_r2_tool = q.get('r_squared')
    residual_structured = q.get('residual_structured')
except Exception as e:
    peak_region_r2_val = None

# ------------------------------------------------------------------
# 7. Results JSON
# ------------------------------------------------------------------
results = {
    "model_type": "Single Voigt profile on a linear baseline (fast-exchange averaged regime); positive amplitude, free Gaussian+Lorentzian widths",
    "parameters": {
        "peak_1": {
            "delta_iso_averaged": delta_iso,
            "delta_iso_averaged_err": delta_iso_err,
            "FWHM_ppm": fwhm,
            "FWHM_ppm_err": fwhm_err,
            "sigma_ppm": sigma,
            "gamma_ppm": gamma,
            "integrated_intensity": integrated_intensity,
            "integrated_intensity_err": integrated_intensity_err,
            "integrated_intensity_numeric": integral_numeric
        },
        "baseline": {
            "slope": pval('b_slope'),
            "intercept": pval('b_intercept')
        }
    },
    "fit_quality": {
        "r_squared": r_squared,
        "rmse": rmse,
        "peak_region_r2": peak_region_r2_val,
        "residual_structured": residual_structured
    },
    "deviation_note": ""
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
