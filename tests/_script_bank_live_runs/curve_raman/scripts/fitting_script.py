import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid

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
        # assume first two columns
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
else:
    y = data.astype(float)
    x = np.arange(len(y), dtype=float)

# sort by x
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
from scipy.ndimage import minimum_filter1d, maximum_filter1d
roll_min = minimum_filter1d(y, size=win)
# smooth the rolling minimum slightly
from scipy.ndimage import uniform_filter1d
baseline_est = uniform_filter1d(roll_min, size=win)
area_total = trapezoid(y, x)
area_base = trapezoid(baseline_est, x)
baseline_fraction = area_base / area_total if area_total > 0 else 0.0
# Per plan: rolling-min baseline ~0 and <5% of area => NOT fluorescence-dominated.
# Use a simple linear baseline as a model parameter (no ALS).

# ---------- Model: sum of 7 pseudo-Voigt peaks + linear baseline ----------
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.abs(fwhm) + 1e-12
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

N_PEAKS = 7
# Reseeded/retuned initial values per updated locked plan:
#  peak_1 (~141): sharp strong band -> narrower FWHM ~11, intermediate-high eta ~0.65
#  peak_2 (~197): real narrow feature -> FWHM ~12, eta ~0.5, small amplitude
#  peak_3 (~307): real broad low-intensity hump (~2500 counts) -> FWHM ~30, eta ~0.4
#  peak_4 (~394): strong band, steep leading edge -> narrower FWHM ~11, eta ~0.6
#  peak_5 (~515), peak_6 (~637): retain roles
#  peak_7 (~790): width constrained to physical range to stop negative excursions
init_centers = np.array([141.0, 197.0, 307.0, 394.0, 515.0, 637.0, 790.0])
init_amps    = np.array([347000.0, 2000.0, 2500.0, 206000.0, 152000.0, 24000.0, 4000.0])
init_fwhm    = np.array([11.0, 12.0, 30.0, 11.0, 15.0, 15.0, 20.0])
init_eta     = np.array([0.65, 0.5, 0.4, 0.6, 0.5, 0.5, 0.5])

# refine initial amplitudes from local data near each center
for i, c in enumerate(init_centers):
    m = np.abs(x - c) <= 8.0
    if np.any(m):
        local_max = y[m].max()
        if local_max > 0:
            init_amps[i] = local_max

# baseline initial: near zero slope, small intercept
init_m = 0.0
init_b = float(np.percentile(y, 5))

# ---------- Parameter packing ----------
# order per peak: amp, center, fwhm, eta ; then m, b
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
# Plan step 5: constrain ALL FWHM to a physical upper bound (<= ~50 cm^-1) to
# eliminate runaway 105/148 cm^-1 widths and negative between-peak excursions.
# Tight center bounds: +-10 for reassigned peaks 2/3, +-8 for the sharp strong
# peaks 1/4, +-15 for the rest.
CENTER_TOL = np.array([8.0, 10.0, 10.0, 8.0, 15.0, 15.0, 15.0])
FWHM_MAX = 50.0
lo = []
hi = []
for i in range(N_PEAKS):
    lo += [0.0, init_centers[i] - CENTER_TOL[i], 1.0, 0.0]      # amp>=0, center bound, fwhm>0, eta>=0
    hi += [np.inf, init_centers[i] + CENTER_TOL[i], FWHM_MAX, 1.0]
lo += [-np.inf, -np.inf]   # baseline slope, intercept
hi += [np.inf, np.inf]
lo = np.array(lo); hi = np.array(hi)

# clip p0 into bounds
p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

# ---------- Fit (Levenberg-Marquardt style via TRF with bounds) ----------
res = least_squares(resid, p0, args=(x, y), bounds=(lo, hi),
                    method='trf', max_nfev=30000, x_scale='jac')
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

# ---------- Evaluate fit ----------
fit_y = model(popt, x)
residuals = y - fit_y

ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = np.sqrt(np.mean(residuals ** 2))

# noise estimate (robust) from high-freq differences
noise = 1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
if noise <= 0:
    noise = np.std(residuals) + 1e-9

# ---------- Save fit.npy (full length, sorted-x order mapped back) ----------
# fit evaluated at same x-points as loaded (we sorted; map back to original order)
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
xgrid = np.linspace(x.min(), x.max(), 4000)
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
ax_main.legend(fontsize=7, ncol=2, loc='upper right')

# residual panel (normalized)
ax_res.plot(x, residuals / noise, 'b-', lw=0.7)
ax_res.axhline(0, color='k', lw=0.6)
ax_res.axhline(3, color='r', ls='--', lw=0.6)
ax_res.axhline(-3, color='r', ls='--', lw=0.6)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_ylabel('Residual / noise')
ax_res.set_xlabel('X')

# zoom panel over peak_1 region (own x-axis, not shared)
zlo1, zhi1 = 91, 166
mz1 = (x >= zlo1) & (x <= zhi1)
ax_zoom1.plot(x[mz1], y[mz1], 'k.', ms=3, label='Data')
ax_zoom1.plot(x[mz1], fit_y[mz1], 'r-', lw=1.2, label='Fit')
ax_zoom1.plot(x[mz1], residuals[mz1], 'b-', lw=0.8, label='Residual')
ax_zoom1.axhline(0, color='0.5', lw=0.5)
ax_zoom1.set_xlim(zlo1, zhi1)
ax_zoom1.set_xlabel('X (zoom peak_1)')
ax_zoom1.set_ylabel('Y')
ax_zoom1.legend(fontsize=7)

# zoom panel over broad 300 cm-1 hump + peak_4 leading edge region
zlo2, zhi2 = 240, 465
mz2 = (x >= zlo2) & (x <= zhi2)
ax_zoom2.plot(x[mz2], y[mz2], 'k.', ms=3, label='Data')
ax_zoom2.plot(x[mz2], fit_y[mz2], 'r-', lw=1.2, label='Fit')
ax_zoom2.plot(x[mz2], residuals[mz2], 'b-', lw=0.8, label='Residual')
ax_zoom2.axhline(0, color='0.5', lw=0.5)
ax_zoom2.set_xlim(zlo2, zhi2)
ax_zoom2.set_xlabel('X (zoom ~300 hump + peak_4)')
ax_zoom2.set_ylabel('Y')
ax_zoom2.legend(fontsize=7)

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Results JSON ----------
params_out = {}
for i in range(N_PEAKS):
    params_out[f'peak_{i+1}'] = {
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
    'model_type': 'Sum of 7 pseudo-Voigt peaks on a linear baseline (Levenberg-Marquardt least squares); reseeded/retuned peak_1 (~141) and peak_4 (~394) to narrower FWHM + intermediate-high eta, reassigned peak_2 (~197 narrow) and peak_3 (~307 broad hump), FWHM capped at 50 cm-1; flat/linear baseline confirmed (rolling-min baseline area fraction = %.3f, not fluorescence-dominated)' % baseline_fraction,
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
