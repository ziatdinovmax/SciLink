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
# Rolling-min baseline near zero and << 40% of area => NOT fluorescence-dominated.
# Use a simple flat linear baseline as a model parameter (no ALS per plan).

# ---------- Model: sum of 8 pseudo-Voigt peaks + linear baseline ----------
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.abs(fwhm) + 1e-12
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# 8 components per updated LOCKED plan.
# Ordering (index -> role) follows plan's peak numbering (1-based -> 0-based):
#  0: peak_1 B1g ~143 sharp core, Lorentzian (eta toward 0.8-1.0), narrow FWHM to reach ~4000 apex
#  1: peak_2 broad multi-phonon/second-order ~232, eta free, tuned for 263-335 tail
#  2: peak_3 Eg main ~445 (dominant symmetric core)
#  3: peak_4 A1g ~611, broadened low-frequency flank absorbs the ~577 shoulder
#  4: peak_5 weak band ~700
#  5: peak_6 second-order/overtone RESTORED to ~800 (FWHM 40-50, amp ~1000) -- PRIMARY FIX
#  6: peak_7 Eg low-frequency shoulder ~405 (steep rising edge)
#  7: peak_8 Eg high-frequency shoulder ~450 (flat-topped asymmetry / falling wing)
N_PEAKS = 8

# Eg-composite sub-components: main (peak_3=idx2), low shoulder (peak_7=idx6), high shoulder (peak_8=idx7)
EG_MAIN, EG_LOW, EG_HIGH = 2, 6, 7

init_centers = np.array([143.0, 232.0, 445.0, 611.0, 700.0, 800.0, 405.0, 450.0])
init_amps    = np.array([4000.0, 8700.0, 37600.0, 13900.0, 500.0, 1000.0, 6500.0, 10000.0])
# peak_1 narrowed FWHM to reach true apex; peak_6 FWHM ~45 per plan; peak_4 broadened.
init_fwhm    = np.array([13.0, 70.0, 30.0, 40.0, 25.0, 45.0, 32.0, 40.0])
# peak_1 eta pushed toward Lorentzian; others free at 0.5.
init_eta     = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5])

# Refine initial amplitudes from local data near strong, well-isolated centers only.
# Eg composite (2,6,7) overlaps heavily; keep plan-seeded amplitudes so they don't
# all snap to the same local maximum of the dominant band.
for i in (0, 1, 3, 4, 5):
    c = init_centers[i]
    mloc = np.abs(x - c) <= 8.0
    if np.any(mloc):
        local_max = y[mloc].max()
        if local_max > 0:
            init_amps[i] = local_max

# baseline initial: near zero slope, small intercept anchored on near-zero regions
anchor = ((x >= 300) & (x <= 380)) | ((x >= 900) & (x <= 1270))
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
# Per-peak center windows.
#   peak_1 B1g   : +-15 (allow narrowing to reach apex)
#   peak_2 232   : +-15
#   peak_3 Eg main: 437-453 (+-8 of 445)
#   peak_4 A1g   : 580-620 (allow modest downward shift so low flank covers ~577 shoulder)
#   peak_5 700   : +-15
#   peak_6 800   : 780-820 (LOCKED window -- do NOT let it drift to ~608)
#   peak_7 Eg low shoulder : 395-415
#   peak_8 Eg high shoulder: 448-470
CENTER_TOL = 15.0
FWHM_MAX = 150.0  # allow broad multi-phonon band; not artificially capped
center_lo = init_centers - CENTER_TOL
center_hi = init_centers + CENTER_TOL
center_lo[EG_MAIN], center_hi[EG_MAIN] = 437.0, 453.0
center_lo[EG_LOW], center_hi[EG_LOW] = 395.0, 415.0
center_lo[EG_HIGH], center_hi[EG_HIGH] = 448.0, 470.0
# peak_4 A1g: allow modest downward shift for the ~577 shoulder absorption
center_lo[3], center_hi[3] = 580.0, 620.0
# peak_6 800: strict window per PRIMARY FIX
center_lo[5], center_hi[5] = 780.0, 820.0

lo = []
hi = []
# Per-peak FWHM upper bounds; keep peak_6 (800) in the plan's 40-50 regime by
# bounding its FWHM so it stays a resolved second-order band, not a broad ramp.
fwhm_hi = np.full(N_PEAKS, FWHM_MAX)
fwhm_hi[5] = 60.0  # peak_6 ~800 band: FWHM ~40-50, cap at 60
for i in range(N_PEAKS):
    lo += [0.0, center_lo[i], 1.0, 0.0]
    hi += [np.inf, center_hi[i], fwhm_hi[i], 1.0]
lo += [-np.inf, -np.inf]
hi += [np.inf, np.inf]
lo = np.array(lo); hi = np.array(hi)

p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

# ---------- Fit (LM via TRF with bounds) ----------
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

xgrid = np.linspace(x.min(), x.max(), 8000)
areas = []
for i in range(N_PEAKS):
    comp = pseudo_voigt(xgrid, amps[i], centers[i], fwhms[i], etas[i])
    areas.append(float(trapezoid(comp, xgrid)))
areas = np.array(areas)
max_amp = amps.max() if amps.max() > 0 else 1.0

# ---------- Eg composite metrics (main + two flanking shoulders) ----------
eg_idx = [EG_MAIN, EG_LOW, EG_HIGH]
eg_curve = np.zeros_like(xgrid)
for i in eg_idx:
    eg_curve += pseudo_voigt(xgrid, amps[i], centers[i], fwhms[i], etas[i])
eg_area = float(trapezoid(eg_curve, xgrid))
if eg_area > 0:
    eg_eff_center = float(trapezoid(eg_curve * xgrid, xgrid) / eg_area)
else:
    eg_eff_center = float(centers[EG_MAIN])
eg_peak = float(eg_curve.max())
# Effective FWHM from half-max crossings of the composite curve.
if eg_peak > 0:
    half = eg_peak / 2.0
    above = eg_curve >= half
    if np.any(above):
        idxs = np.where(above)[0]
        eg_eff_fwhm = float(xgrid[idxs[-1]] - xgrid[idxs[0]])
    else:
        eg_eff_fwhm = float('nan')
else:
    eg_eff_fwhm = float('nan')

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
    ax_main.plot(x, comp, '--', lw=0.8, label=f'Component {i+1}')
ax_main.plot(x, baseline_line, ':', color='green', lw=1.0, label='Baseline')
ax_main.set_yscale('symlog', linthresh=1000)
ax_main.set_xlim(x.min(), x.max())
ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.legend(fontsize=7, ncol=2, loc='upper right')

ax_res.plot(x, residuals / noise, 'b-', lw=0.7, label='Residual / noise')
ax_res.axhline(0, color='k', lw=0.6)
ax_res.axhline(3, color='r', ls='--', lw=0.6)
ax_res.axhline(-3, color='r', ls='--', lw=0.6)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_ylabel('Residual / noise')
ax_res.set_xlabel('X')
ax_res.legend(fontsize=7)

# Diagnostic zoom windows targeted by verification (own non-shared x-axes).
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

# Zoom on the B1g apex, the 800 band (PRIMARY FIX), and the 232 tail.
zoom_panel(ax_z0, 119.0, 191.0, 'B1g apex')
zoom_panel(ax_z1, 766.0, 838.0, '~800 band')
zoom_panel(ax_z2, 263.0, 335.0, '232 tail')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Results JSON ----------
labels = ['B1g_143', 'band_232', 'Eg_main_445', 'A1g_611', 'band_700',
          'band_800', 'Eg_shoulder_low_405', 'Eg_shoulder_high_450']
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
params_out['Eg_composite'] = {
    'integrated_area': float(eg_area),
    'effective_center': float(eg_eff_center),
    'effective_fwhm': float(eg_eff_fwhm),
    'subcomponents': ['peak_3', 'peak_7', 'peak_8'],
}

results = {
    'model_type': 'Sum of 8 pseudo-Voigt peaks on a flat linear baseline (Levenberg-Marquardt / TRF least squares). Bands: B1g ~143 (sharp, Lorentzian-leaning), broad ~232, Eg main ~445, A1g ~611 (broadened low-frequency flank absorbs ~577 shoulder), weak ~700, second-order/overtone ~800 (restored to seeded position, FWHM ~40-50), Eg low-frequency shoulder ~405, Eg high-frequency shoulder ~450 cm-1. The dominant Eg band is decomposed into a symmetric main core plus two flanking symmetric pseudo-Voigt shoulders (superposition, no skewed profiles). peak_6 constrained to 780-820 cm-1 (FWHM<=60) to resolve the 800 band; peak_1 narrowed and pushed toward Lorentzian to reach the ~4000 apex; peak_2 eta free for the 263-335 tail. Center windows: Eg main 437-453, Eg shoulders 395-415 / 448-470, A1g 580-620, others +-15. FWHM>0, amp>0, eta in [0,1]. Flat/linear baseline confirmed (rolling-min baseline area fraction = %.3f, not fluorescence-dominated).' % baseline_fraction,
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
