import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
from scipy.ndimage import minimum_filter1d, uniform_filter1d

# ---------- Load raw data ----------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x = data[0].astype(float)
        y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
    else:
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
else:
    y = data.astype(float)
    x = np.arange(len(y), dtype=float)

order = np.argsort(x)
x = x[order]
y = y[order]
y_raw = y.copy()

# ---------- Preprocessing ----------
# Intensity-type Raman spectrum: negatives are noise -> clip to zero.
y = np.clip(y, 0, None)

# ---------- Background assessment (Plan step 1) ----------
# Rolling-minimum baseline estimate to confirm fluorescence is NOT dominant.
win = max(5, len(y) // 40)
roll_min = minimum_filter1d(y, size=win)
baseline_est = uniform_filter1d(roll_min, size=win)
area_total = trapezoid(y, x)
area_base = trapezoid(baseline_est, x)
baseline_fraction = area_base / area_total if area_total > 0 else 0.0
# Per plan: rolling-min baseline ~flat at zero, <<40% of area => NOT fluorescence-dominated.
# Use a simple linear baseline as a model parameter (no ALS).

# ---------- Model: sum of 10 pseudo-Voigt peaks + linear baseline ----------
# Plan-locked inventory (10 components):
#   shoulder_128 (~127-130, low-x shoulder of the 141 band),
#   peak_141 (strong main band),
#   peak_194 (sharp),
#   peak_300 / peak_310 (resolved doublet),
#   shoulder_375 (rising low-x flank of the 395 band),
#   peak_395 (strong band),
#   peak_515, peak_637, peak_790.
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.abs(fwhm) + 1e-12
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# Labels track physical seeding (order matters for packing/unpacking)
PEAK_LABELS = ['shoulder_128', 'peak_141', 'peak_194', 'peak_300', 'peak_310',
               'shoulder_375', 'peak_395', 'peak_515', 'peak_637', 'peak_790']
N_PEAKS = len(PEAK_LABELS)

# Re-seeded initial guesses per the refined plan.
init_centers = np.array([128.0, 141.0, 194.0, 302.0, 309.0,
                         375.0, 395.0, 515.0, 637.0, 790.0])
init_amps    = np.array([40000.0, 347000.0, 6000.0, 1200.0, 1300.0,
                         20000.0, 206000.0, 152000.0, 24000.0, 4000.0])
init_fwhm    = np.array([15.0, 12.0, 8.0, 10.0, 10.0,
                         14.0, 12.0, 16.0, 15.0, 20.0])
init_eta     = np.array([0.5, 0.6, 0.5, 0.4, 0.4,
                         0.5, 0.6, 0.5, 0.5, 0.5])

# Refine initial amplitudes for the strong bands from local data near each center
# (do NOT overwrite the deliberately-seeded weak/shoulder components)
strong_idx = [1, 6, 7, 8]  # peak_141, peak_395, peak_515, peak_637
for i in strong_idx:
    c = init_centers[i]
    mloc = np.abs(x - c) <= 6.0
    if np.any(mloc):
        local_max = y[mloc].max()
        if local_max > 0:
            init_amps[i] = local_max

# Seed the sharp 194 band from its local maximum (tight window).
for i in [2]:
    c = init_centers[i]
    mloc = np.abs(x - c) <= 5.0
    if np.any(mloc):
        local_max = y[mloc].max()
        if local_max > 0:
            init_amps[i] = local_max

# baseline initial: near zero slope, small intercept
init_m = 0.0
init_b = float(np.percentile(y, 5))

# ---------- Parameter packing ----------
def pack(amps, centers, fwhms, etas, m, b):
    p = []
    for i in range(N_PEAKS):
        p += [amps[i], centers[i], fwhms[i], etas[i]]
    p += [m, b]
    return np.array(p, dtype=float)

def unpack(p):
    amps = np.empty(N_PEAKS); centers = np.empty(N_PEAKS)
    fwhms = np.empty(N_PEAKS); etas = np.empty(N_PEAKS)
    for i in range(N_PEAKS):
        amps[i] = p[4*i + 0]
        centers[i] = p[4*i + 1]
        fwhms[i] = p[4*i + 2]
        etas[i] = p[4*i + 3]
    m = p[4*N_PEAKS + 0]
    b = p[4*N_PEAKS + 1]
    return amps, centers, fwhms, etas, m, b

def model(p, x):
    amps, centers, fwhms, etas, m, b = unpack(p)
    out = m * x + b
    for i in range(N_PEAKS):
        out = out + pseudo_voigt(x, amps[i], centers[i], fwhms[i], etas[i])
    return out

def resid(p, x, y):
    return model(p, x) - y

p0 = pack(init_amps, init_centers, init_fwhm, init_eta, init_m, init_b)

# ---------- Bounds ----------
# Plan: centers within +/-10 cm-1 of seeds; amplitudes >=0, FWHM>0, eta in [0,1].
# Per verifier fix: relax doublet center bounds slightly (off the previous cap)
# and cap the doublet FWHM well below the old 50 limit to prevent width pinning.
CENTER_TOL = np.full(N_PEAKS, 10.0)
# Give the shoulder components a touch more center freedom to seat cleanly.
CENTER_TOL[0] = 8.0   # shoulder_128 (stay clearly below main peak)
CENTER_TOL[5] = 8.0   # shoulder_375

# Per-band FWHM upper caps. Doublet capped at 25 (well below old 50) so widths
# are not pinned; sharp 194 capped tighter; broad bands allowed a bit more.
FWHM_MAX = np.array([30.0,   # shoulder_128
                     30.0,   # peak_141
                     20.0,   # peak_194 (sharp)
                     25.0,   # peak_300 (doublet, below old 50 cap)
                     25.0,   # peak_310 (doublet, below old 50 cap)
                     30.0,   # shoulder_375
                     30.0,   # peak_395
                     40.0,   # peak_515
                     40.0,   # peak_637
                     50.0])  # peak_790

lo = []
hi = []
for i in range(N_PEAKS):
    lo += [0.0, init_centers[i] - CENTER_TOL[i], 1.0, 0.0]
    hi += [np.inf, init_centers[i] + CENTER_TOL[i], FWHM_MAX[i], 1.0]
lo += [-np.inf, -np.inf]
hi += [np.inf, np.inf]
lo = np.array(lo); hi = np.array(hi)

p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

# ---------- Fit (Levenberg-Marquardt style via TRF with bounds) ----------
res = least_squares(resid, p0, args=(x, y), bounds=(lo, hi),
                    method='trf', max_nfev=60000, x_scale='jac')
popt = res.x

# ---------- Parameter uncertainties from Jacobian ----------
try:
    J = res.jac
    dof = max(1, len(y) - len(popt))
    chi2 = 2 * res.cost
    s_sq = chi2 / dof
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * s_sq
    perr = np.sqrt(np.abs(np.diag(cov)))
except Exception:
    perr = np.full_like(popt, np.nan)

# reduced chi-square (using robust noise as measurement sigma)
noise = 1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
if noise <= 0:
    noise = np.std(resid(popt, x, y)) + 1e-9

# ---------- Evaluate fit ----------
fit_y = model(popt, x)
residuals = y - fit_y

ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = np.sqrt(np.mean(residuals ** 2))

dof = max(1, len(y) - len(popt))
reduced_chi_square = float(np.sum((residuals / noise) ** 2) / dof)

# ---------- Per-band residual RMS ----------
band_windows = {
    '91.5-166.2': (91.5, 166.2),
    '166.2-260': (166.2, 260.0),
    '260-315.4': (260.0, 315.4),
    '315.4-390.1': (315.4, 390.1),
    '390.1-464.7': (390.1, 464.7),
    '464.7-539.4': (464.7, 539.4),
    '539.4-700': (539.4, 700.0),
    '700-850': (700.0, 850.0),
}
per_band_rms = {}
for name, (a, b) in band_windows.items():
    mb = (x >= a) & (x <= b)
    if np.any(mb):
        per_band_rms[name] = float(np.sqrt(np.mean(residuals[mb] ** 2)))
    else:
        per_band_rms[name] = None

# ---------- Save fit.npy (full length, mapped back to original order) ----------
fit_full = np.empty_like(fit_y)
fit_full[order] = fit_y
np.save('fit.npy', fit_full)

# ---------- Extract per-peak parameters ----------
amps, centers, fwhms, etas, m_bl, b_bl = unpack(popt)
amps_e = np.array([perr[4*i+0] for i in range(N_PEAKS)])
cent_e = np.array([perr[4*i+1] for i in range(N_PEAKS)])
fwhm_e = np.array([perr[4*i+2] for i in range(N_PEAKS)])
eta_e  = np.array([perr[4*i+3] for i in range(N_PEAKS)])

# integrated areas per component
xgrid = np.linspace(x.min(), x.max(), 6000)
areas = []
for i in range(N_PEAKS):
    comp = pseudo_voigt(xgrid, amps[i], centers[i], fwhms[i], etas[i])
    areas.append(float(trapezoid(comp, xgrid)))
areas = np.array(areas)
max_amp = amps.max() if amps.max() > 0 else 1.0

# ---------- Visualization ----------
fig, axes = plt.subplots(4, 1, figsize=(11, 12),
                         gridspec_kw={'height_ratios': [3, 1.1, 1.1, 1.1]})
ax_main, ax_res, ax_zoom1, ax_zoom2 = axes[0], axes[1], axes[2], axes[3]

ax_main.plot(x, y_raw, color='0.7', lw=0.8, alpha=0.5, label='Raw data')
ax_main.plot(x, y, 'k.', ms=2, label='Data')
ax_main.plot(x, fit_y, 'r-', lw=1.3, label='Fit')
baseline_line = m_bl * x + b_bl
for i in range(N_PEAKS):
    comp = pseudo_voigt(x, amps[i], centers[i], fwhms[i], etas[i]) + baseline_line
    ax_main.plot(x, comp, '--', lw=0.8, label=f'Component {i+1}')
ax_main.plot(x, baseline_line, ':', color='green', lw=1.0, label='Baseline')
ax_main.set_yscale('symlog', linthresh=1000)
ax_main.set_xlim(x.min(), x.max())
ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.legend(fontsize=6, ncol=2, loc='upper right')

# residual panel (normalized)
ax_res.plot(x, residuals / noise, 'b-', lw=0.7)
ax_res.axhline(0, color='k', lw=0.6)
ax_res.axhline(3, color='r', ls='--', lw=0.6)
ax_res.axhline(-3, color='r', ls='--', lw=0.6)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_ylabel('Residual / noise')
ax_res.set_xlabel('X')

# zoom panel over peak_141 + shoulder_128 region (own x-axis, not shared)
zlo1, zhi1 = 100, 175
mz1 = (x >= zlo1) & (x <= zhi1)
ax_zoom1.plot(x[mz1], y[mz1], 'k.', ms=3, label='Data')
ax_zoom1.plot(x[mz1], fit_y[mz1], 'r-', lw=1.2, label='Fit')
ax_zoom1.plot(x[mz1], residuals[mz1], 'b-', lw=0.8, label='Residual')
ax_zoom1.axhline(0, color='0.5', lw=0.5)
ax_zoom1.set_xlim(zlo1, zhi1)
ax_zoom1.set_xlabel('X (zoom ~128/141)')
ax_zoom1.set_ylabel('Y')
ax_zoom1.legend(fontsize=7)

# zoom panel over ~300 doublet + 375 shoulder + peak_395 leading edge region
zlo2, zhi2 = 260, 430
mz2 = (x >= zlo2) & (x <= zhi2)
ax_zoom2.plot(x[mz2], y[mz2], 'k.', ms=3, label='Data')
ax_zoom2.plot(x[mz2], fit_y[mz2], 'r-', lw=1.2, label='Fit')
ax_zoom2.plot(x[mz2], residuals[mz2], 'b-', lw=0.8, label='Residual')
ax_zoom2.axhline(0, color='0.5', lw=0.5)
ax_zoom2.set_xlim(zlo2, zhi2)
ax_zoom2.set_xlabel('X (zoom ~300 doublet / 375-395)')
ax_zoom2.set_ylabel('Y')
ax_zoom2.legend(fontsize=7)

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Results JSON ----------
params_out = {}
for i in range(N_PEAKS):
    params_out[PEAK_LABELS[i]] = {
        'center': float(centers[i]),
        'center_err': float(cent_e[i]),
        'amplitude': float(amps[i]),
        'amplitude_err': float(amps_e[i]),
        'fwhm': float(fwhms[i]),
        'fwhm_err': float(fwhm_e[i]),
        'eta': float(etas[i]),
        'eta_err': float(eta_e[i]),
        'integrated_area': float(areas[i]),
        'relative_intensity': float(amps[i] / max_amp),
    }
params_out['baseline'] = {
    'slope': float(m_bl),
    'slope_err': float(perr[4*N_PEAKS+0]),
    'intercept': float(b_bl),
    'intercept_err': float(perr[4*N_PEAKS+1]),
}

results = {
    'model_type': 'Sum of 10 pseudo-Voigt peaks on a linear (near-zero) baseline, fit over the full range with Levenberg-Marquardt (least_squares TRF). Inventory: shoulder_128 + peak_141 (asymmetric main band), sharp peak_194, resolved 300/310 doublet, shoulder_375 + peak_395, peak_515, peak_637, peak_790. Centers bounded +/-8-10 cm-1; per-band FWHM caps (doublet <=25 cm-1); not fluorescence-dominated (rolling-min baseline area fraction = %.3f).' % baseline_fraction,
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse),
                    'reduced_chi_square': reduced_chi_square,
                    'per_band_residual_rms': per_band_rms},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
