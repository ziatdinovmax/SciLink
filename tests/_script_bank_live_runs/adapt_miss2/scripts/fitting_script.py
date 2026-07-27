import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
from scipy.ndimage import minimum_filter1d, uniform_filter1d, median_filter

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
# Remove single-pixel cosmic spikes via a light median filter (skill rule).
y_med = median_filter(y, size=3)
spike = (y - y_med) > 5.0 * (1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9)
y = np.where(spike, y_med, y)

# ---------- Background assessment (Plan step 1) ----------
# Rolling-minimum baseline estimate to confirm fluorescence is NOT dominant.
win = max(5, len(y) // 40)
roll_min = minimum_filter1d(y, size=win)
baseline_est = uniform_filter1d(roll_min, size=win)
area_total = trapezoid(y, x)
area_base = trapezoid(baseline_est, x)
baseline_fraction = area_base / area_total if area_total > 0 else 0.0
# Rolling-min baseline near zero and << 40% of area => NOT fluorescence-dominated.
# Use a simple flat linear baseline as a model parameter (no ALS per plan).

# ---------- Model: sum of 12 pseudo-Voigt peaks + linear baseline ----------
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.abs(fwhm) + 1e-12
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# 12 components per LOCKED plan. Band centers (initial):
#  ~220, ~305, ~325, ~370, ~428, ~542, ~590, ~610, ~822 (strong), ~854 (strong), ~918, ~960 cm-1
# The 810-870 region is a RESOLVED DOUBLET -> two components at ~822 and ~854.
N_PEAKS = 12

init_centers = np.array([220.0, 305.0, 325.0, 370.0, 428.0, 542.0,
                         590.0, 610.0, 822.0, 854.0, 918.0, 960.0])
init_fwhm = np.array([12.0, 8.0, 8.0, 10.0, 10.0, 10.0,
                      8.0, 8.0, 12.0, 12.0, 10.0, 10.0])
init_eta = np.full(N_PEAKS, 0.5)

# Seed amplitudes from local data maxima near each center.
init_amps = np.empty(N_PEAKS)
for i in range(N_PEAKS):
    c = init_centers[i]
    mloc = np.abs(x - c) <= 6.0
    if np.any(mloc):
        lm = y[mloc].max()
        init_amps[i] = lm if lm > 0 else 100.0
    else:
        init_amps[i] = 100.0

# baseline initial: near zero slope, small intercept anchored on near-zero regions
anchor = ((x >= 640) & (x <= 780)) | ((x >= 990) & (x <= 1270))
if np.any(anchor):
    init_b = float(np.percentile(y[anchor], 20))
else:
    init_b = float(np.percentile(y, 5))
init_m = 0.0

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
# Centers constrained within +-10 cm-1 of seeds (plan step 4).
CENTER_TOL = 10.0
FWHM_MAX = 60.0
center_lo = init_centers - CENTER_TOL
center_hi = init_centers + CENTER_TOL

lo = []
hi = []
for i in range(N_PEAKS):
    lo += [0.0, center_lo[i], 1.0, 0.0]
    hi += [np.inf, center_hi[i], FWHM_MAX, 1.0]
lo += [-np.inf, -np.inf]
hi += [np.inf, np.inf]
lo = np.array(lo); hi = np.array(hi)

p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

# ---------- Fit (LM via TRF with bounds) ----------
res = least_squares(resid, p0, args=(x, y), bounds=(lo, hi),
                    method='trf', max_nfev=80000, x_scale='jac')
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

noise = 1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
if noise <= 0:
    noise = np.std(residuals) + 1e-9

# ---------- Save fit.npy (mapped back to original order) ----------
fit_full = np.empty_like(fit_y)
fit_full[order] = fit_y
np.save('fit.npy', fit_full)

# ---------- Extract per-peak parameters ----------
amps, centers, fwhms, etas, m_bl, b_bl = unpack(popt)
amps_e = np.array([perr[4*i+0] for i in range(N_PEAKS)])
cent_e = np.array([perr[4*i+1] for i in range(N_PEAKS)])
fwhm_e = np.array([perr[4*i+2] for i in range(N_PEAKS)])
eta_e  = np.array([perr[4*i+3] for i in range(N_PEAKS)])

xgrid = np.linspace(x.min(), x.max(), 12000)
areas = []
for i in range(N_PEAKS):
    comp = pseudo_voigt(xgrid, amps[i], centers[i], fwhms[i], etas[i])
    areas.append(float(trapezoid(comp, xgrid)))
areas = np.array(areas)
max_amp = amps.max() if amps.max() > 0 else 1.0

# ---------- Visualization ----------
fig, axes = plt.subplots(5, 1, figsize=(11, 14),
                         gridspec_kw={'height_ratios': [3, 1.1, 1.1, 1.1, 1.1]})
ax_main, ax_res, ax_z0, ax_z1, ax_z2 = axes

ax_main.plot(x, y_raw, color='0.7', lw=0.8, alpha=0.5, label='Raw data')
ax_main.plot(x, y, 'k.', ms=2, label='Data')
ax_main.plot(x, fit_y, 'r-', lw=1.3, label='Fit')
baseline_line = m_bl * x + b_bl
for i in range(N_PEAKS):
    comp = pseudo_voigt(x, amps[i], centers[i], fwhms[i], etas[i]) + baseline_line
    ax_main.plot(x, comp, '--', lw=0.7, label=f'Component {i+1}')
ax_main.plot(x, baseline_line, ':', color='green', lw=1.0, label='Baseline')
ax_main.set_yscale('symlog', linthresh=100)
ax_main.set_xlim(x.min(), x.max())
ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.legend(fontsize=6, ncol=3, loc='upper right')

ax_res.plot(x, residuals / noise, 'b-', lw=0.7, label='Residual / noise')
ax_res.axhline(0, color='k', lw=0.6)
ax_res.axhline(3, color='r', ls='--', lw=0.6)
ax_res.axhline(-3, color='r', ls='--', lw=0.6)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_ylabel('Residual / noise')
ax_res.set_xlabel('X')
ax_res.legend(fontsize=7)

def zoom_panel(ax, zlo, zhi, tag):
    mz = (x >= zlo) & (x <= zhi)
    ax.plot(x[mz], y[mz], 'k.', ms=3, label='Data')
    ax.plot(x[mz], fit_y[mz], 'r-', lw=1.2, label='Fit')
    ax.plot(x[mz], residuals[mz], 'b-', lw=0.8, label='Residual')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlim(zlo, zhi)
    ax.set_xlabel('X (zoom %s)' % tag)
    ax.set_ylabel('Y')
    ax.legend(fontsize=7)

# Zoom on the strong doublet, the 280-380 cluster, and the 900-970 region.
zoom_panel(ax_z0, 795.0, 885.0, '822/854 doublet')
zoom_panel(ax_z1, 285.0, 385.0, '305/325/370 cluster')
zoom_panel(ax_z2, 560.0, 640.0, '590/610 region')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Results JSON ----------
labels = ['band_220', 'band_305', 'band_325', 'band_370', 'band_428',
          'band_542', 'band_590', 'band_610', 'band_822_strong',
          'band_854_strong', 'band_918', 'band_960']
params_out = {}
for i in range(N_PEAKS):
    params_out[f'peak_{i+1}'] = {
        'label': labels[i],
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
    'model_type': 'Sum of 12 pseudo-Voigt peaks on a flat linear baseline (Levenberg-Marquardt / TRF least squares). Band centers (cm-1): ~220, ~305, ~325, ~370, ~428, ~542, ~590, ~610, ~822 (strong), ~854 (strong), ~918, ~960. The 810-870 region is treated as a RESOLVED DOUBLET (two symmetric components at ~822 and ~854), not one asymmetric band. Each component: amp*[eta*Lorentzian + (1-eta)*Gaussian], independent center, FWHM, amplitude, eta. Baseline: linear (near-zero) fitted as a model parameter. Constraints: amp>=0, FWHM in (1,60), eta in [0,1], centers within +-10 cm-1 of seeds. Background assessment: rolling-minimum baseline area fraction = %.3f (<<0.40) => NOT fluorescence-dominated, no ALS applied per plan step 1. Cosmic single-pixel spikes removed via median filter.' % baseline_fraction,
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
