import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# ---------------------------------------------------------------
# 1. Load RAW data
# ---------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2 and data.shape[1] != 2:
        x = data[0].astype(float)
        y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
    else:
        x = data[0].astype(float)
        y = data[1].astype(float)
else:
    raise ValueError('Unexpected data shape')

# ensure ascending x
if x[0] > x[-1]:
    x = x[::-1]
    y = y[::-1]

y_raw = y.copy()

# ---------------------------------------------------------------
# 2. Preprocess: intensity spectrum -> clip negatives (noise).
#    Baseline is negligible per plan (RRUFF bg-subtracted, ~0 across range)
#    so NO ALS; a flat offset c is fit as a model parameter.
# ---------------------------------------------------------------
y = np.clip(y, 0, None)

# noise estimate from a signal-free region (high wavenumber tail, above 3000)
noise_mask = x > 3000
if np.count_nonzero(noise_mask) > 10:
    noise_level = np.std(y[noise_mask])
else:
    noise_level = np.std(np.diff(y)) / np.sqrt(2)
if noise_level <= 0:
    noise_level = 1.0

# ---------------------------------------------------------------
# 3. Model: sum of 3 pseudo-Voigt components + constant offset
# ---------------------------------------------------------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

def model(x, a1, c1, w1, e1, a2, c2, w2, e2, a3, c3, w3, e3, c):
    return (pseudo_voigt(x, a1, c1, w1, e1)
            + pseudo_voigt(x, a2, c2, w2, e2)
            + pseudo_voigt(x, a3, c3, w3, e3)
            + c)

# ---------------------------------------------------------------
# 4. Initial guesses and bounds
# ---------------------------------------------------------------
# main band ~1336
p0 = [
    35000.0, 1336.0, 8.0, 0.5,   # component 1: sharp diamond band
    150.0,   2465.0, 45.0, 0.5,  # component 2: broad second-order band
    100.0,   830.0, 60.0, 0.5,   # component 3: optional broad ~830 band
    0.0                          # constant offset
]

lower = [
    0.0, 1320.0, 1.0, 0.0,
    0.0, 2440.0, 5.0, 0.0,
    0.0, 780.0, 5.0, 0.0,
    -1000.0
]
upper = [
    1e7, 1345.0, 100.0, 1.0,
    1e6, 2490.0, 300.0, 1.0,
    1e6, 900.0, 400.0, 1.0,
    1000.0
]

popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lower, upper),
                       maxfev=200000)
perr = np.sqrt(np.diag(pcov))

fit_full = model(x, *popt)

# ---------------------------------------------------------------
# 5. Fit quality (over full modelled domain, saved-array space)
# ---------------------------------------------------------------
residuals = y - fit_full
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = float(np.sqrt(np.mean(residuals ** 2)))

# ---------------------------------------------------------------
# 6. Per-component params, areas, absence check for ~830 band
# ---------------------------------------------------------------
def comp_area(amp, fwhm, eta):
    # analytic areas: Lorentzian = amp*pi*fwhm/2 ; Gaussian = amp*s*sqrt(2pi)
    s = fwhm / 2.3548200450309493
    area_l = amp * np.pi * fwhm / 2.0
    area_g = amp * s * np.sqrt(2 * np.pi)
    return eta * area_l + (1 - eta) * area_g

names = ['peak_1', 'peak_2', 'peak_3']
parameters = {}

# amplitudes for relative intensity
amps = [popt[0], popt[4], popt[8]]
max_amp = max(amps) if max(amps) > 0 else 1.0

for i, name in enumerate(names):
    base = i * 4
    amp = popt[base]
    center = popt[base + 1]
    fwhm = popt[base + 2]
    eta = popt[base + 3]
    amp_e = perr[base]
    center_e = perr[base + 1]
    fwhm_e = perr[base + 2]
    eta_e = perr[base + 3]
    area = comp_area(amp, fwhm, eta)

    # absence test for the optional ~830 band (component 3)
    absent = False
    if i == 2 and amp < 5.0 * noise_level:
        absent = True

    if absent:
        parameters[name] = {
            'center': None,
            'center_err': None,
            'fwhm': None,
            'fwhm_err': None,
            'amplitude': float(amp),
            'amplitude_err': float(amp_e),
            'area': float(area),
            'eta': None,
            'relative_intensity': float(amp / max_amp),
            'peak_3_absent': True
        }
    else:
        parameters[name] = {
            'center': float(center),
            'center_err': float(center_e),
            'fwhm': float(fwhm),
            'fwhm_err': float(fwhm_e),
            'amplitude': float(amp),
            'amplitude_err': float(amp_e),
            'area': float(area),
            'eta': float(eta),
            'eta_err': float(eta_e),
            'relative_intensity': float(amp / max_amp)
        }

parameters['baseline_offset'] = {
    'value': float(popt[12]),
    'value_err': float(perr[12])
}

# ---------------------------------------------------------------
# 7. Save fit.npy (evaluated at all N x-points, in data order)
# ---------------------------------------------------------------
# fit.npy must match original data ordering. If we reversed x, reverse back.
fit_to_save = fit_full.copy()
if np.load('data.npy').ndim == 2:
    orig = np.load('data.npy')
    if orig.shape[0] == 2 and orig.shape[1] != 2:
        orig_x = orig[0].astype(float)
    elif orig.shape[1] == 2:
        orig_x = orig[:, 0].astype(float)
    else:
        orig_x = orig[0].astype(float)
    if orig_x[0] > orig_x[-1]:
        fit_to_save = fit_full[::-1]
np.save('fit.npy', fit_to_save.astype(float))

# ---------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------
c1 = pseudo_voigt(x, popt[0], popt[1], popt[2], popt[3]) + popt[12]
c2 = pseudo_voigt(x, popt[4], popt[5], popt[6], popt[7]) + popt[12]
c3 = pseudo_voigt(x, popt[8], popt[9], popt[10], popt[11]) + popt[12]

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.2])

ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(x, y_raw, color='lightgrey', alpha=0.6, lw=0.8, label='Raw')
ax_main.plot(x, y, color='black', lw=0.9, label='Data')
ax_main.plot(x, fit_full, color='red', lw=1.2, label='Fit')
ax_main.plot(x, c1, '--', color='tab:blue', lw=0.9, label='Component 1')
ax_main.plot(x, c2, '--', color='tab:green', lw=0.9, label='Component 2')
ax_main.plot(x, c3, '--', color='tab:orange', lw=0.9, label='Component 3')
ax_main.set_yscale('symlog', linthresh=max(10.0, noise_level))
ax_main.set_xlim(x.min(), x.max())
ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.legend(fontsize=8, ncol=3)

# normalized residual panel (full domain)
ax_res = fig.add_subplot(gs[1, :])
ax_res.plot(x, residuals / noise_level, color='purple', lw=0.7)
ax_res.axhline(0, color='k', lw=0.5)
ax_res.axhline(3, color='grey', ls=':', lw=0.7)
ax_res.axhline(-3, color='grey', ls=':', lw=0.7)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_xlabel('X')
ax_res.set_ylabel('Resid / noise')
ax_res.set_title('Residuals (normalized)')

# zoom on main band (own non-shared x-axis)
ax_z1 = fig.add_subplot(gs[2, 0])
m1 = (x > popt[1] - 60) & (x < popt[1] + 60)
ax_z1.plot(x[m1], residuals[m1] / noise_level, color='tab:blue', lw=0.8)
ax_z1.axhline(0, color='k', lw=0.5)
ax_z1.axhline(3, color='grey', ls=':', lw=0.7)
ax_z1.axhline(-3, color='grey', ls=':', lw=0.7)
ax_z1.set_xlim(popt[1] - 60, popt[1] + 60)
ax_z1.set_xlabel('X')
ax_z1.set_ylabel('Resid / noise')
ax_z1.set_title('Residual zoom (band 1)')

# zoom on second-order band region
ax_z2 = fig.add_subplot(gs[2, 1])
m2 = (x > popt[5] - 150) & (x < popt[5] + 150)
ax_z2.plot(x[m2], residuals[m2] / noise_level, color='tab:green', lw=0.8)
ax_z2.axhline(0, color='k', lw=0.5)
ax_z2.axhline(3, color='grey', ls=':', lw=0.7)
ax_z2.axhline(-3, color='grey', ls=':', lw=0.7)
ax_z2.set_xlim(popt[5] - 150, popt[5] + 150)
ax_z2.set_xlabel('X')
ax_z2.set_ylabel('Resid / noise')
ax_z2.set_title('Residual zoom (band 2)')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------------------------------------------------------------
# 9. Results JSON
# ---------------------------------------------------------------
results = {
    'model_type': 'Sum of 3 pseudo-Voigt components + constant baseline offset (no ALS; RRUFF background-subtracted data sits near zero)',
    'parameters': parameters,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
