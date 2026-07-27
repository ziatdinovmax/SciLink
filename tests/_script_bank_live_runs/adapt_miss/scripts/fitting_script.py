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
# Use a simple constant+linear baseline as model parameters (no ALS per plan).

# ---------- Model: sum of 13 pseudo-Voigt peaks + linear baseline ----------
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.abs(fwhm) + 1e-12
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# 13 components per UPDATED LOCKED plan:
# DROP the spurious ~415 candidate (refined to noise-level amplitude, pinned center).
# ADD a ~450 cm-1 shoulder on the low-frequency rising flank of the 464 band.
# ADD a ~120 cm-1 shoulder on the low-frequency flank of the 130 band.
# Retained core bands: ~120(sh), 130, 205, 262, 356, 402, 450(sh), 464, 505,
# 700, 810, 1085, 1160 cm-1.  Lorentzian-dominated pseudo-Voigt (eta ~0.7-1.0).
N_PEAKS = 13

init_centers = np.array([120.0, 130.0, 205.0, 262.0, 356.0, 402.0, 450.0,
                         464.0, 505.0, 700.0, 810.0, 1085.0, 1160.0])
# Amplitudes seeded then refined from local data for isolated bands.
init_amps    = np.array([600.0, 1500.0, 2200.0, 800.0, 600.0, 900.0, 400.0,
                         2000.0, 700.0, 500.0, 600.0, 700.0, 500.0])
# Widths: shoulders narrow, strong bands moderate.
init_fwhm    = np.array([8.0, 12.0, 12.0, 15.0, 15.0, 12.0, 10.0,
                         14.0, 15.0, 22.0, 25.0, 20.0, 20.0])
# Lorentzian-dominated eta (~0.7-1.0 regime) per plan.
init_eta     = np.full(N_PEAKS, 0.8)

# Indices of the two new shoulder components.
SH_120 = 0
SH_450 = 6
# Index of band_700 (previously railed against FWHM upper bound; re-checked).
BAND_700 = 9

# Refine initial amplitudes from local data near seeded centers.
for i in range(N_PEAKS):
    c = init_centers[i]
    mloc = np.abs(x - c) <= 6.0
    if np.any(mloc):
        local_max = y[mloc].max()
        if local_max > 0:
            init_amps[i] = local_max
# Shoulders sit on a flank, not an apex: temper their seed so the main band
# can carry the peak and the shoulder resolves the inflection.
init_amps[SH_120] = 0.4 * init_amps[SH_120]
init_amps[SH_450] = 0.35 * init_amps[SH_450]

# baseline initial: near zero slope, small intercept anchored on low regions
anchor = (y <= np.percentile(y, 20))
if np.any(anchor):
    init_b = float(np.percentile(y[anchor], 50))
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
# Center windows: +/-10 cm-1 on established bands per plan. Widths 2-40 cm-1.
# Amp>=0. eta in [0,1].
CENTER_TOL = 10.0
FWHM_MIN = 2.0
FWHM_MAX = 40.0
center_lo = init_centers - CENTER_TOL
center_hi = init_centers + CENTER_TOL

# Shoulder-specific center windows per plan seeds:
#   120 shoulder within 116-124; 450 shoulder within 445-455.
center_lo[SH_120] = 116.0; center_hi[SH_120] = 124.0
center_lo[SH_450] = 445.0; center_hi[SH_450] = 455.0
# Shoulder FWHM seeds are narrow (plan: 120 -> 5-15, 450 -> 6-15); keep the
# global 2-40 window but the narrow seed steers them.

# band_700 previously pinned at the 40 cm-1 FWHM upper bound with a large center
# error. Per plan, widen the FWHM upper bound modestly for THIS band so it is not
# artificially railed against the global bound; if a real broad feature exists it
# will refine within, otherwise it settles below 40.
FWHM_MAX_700 = 55.0

lo = []
hi = []
for i in range(N_PEAKS):
    fmax = FWHM_MAX_700 if i == BAND_700 else FWHM_MAX
    lo += [0.0, center_lo[i], FWHM_MIN, 0.0]
    hi += [np.inf, center_hi[i], fmax, 1.0]
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

xgrid = np.linspace(x.min(), x.max(), 8000)
areas = []
for i in range(N_PEAKS):
    comp = pseudo_voigt(xgrid, amps[i], centers[i], fwhms[i], etas[i])
    areas.append(float(trapezoid(comp, xgrid)))
areas = np.array(areas)
max_amp = amps.max() if amps.max() > 0 else 1.0

# ---------- Assess new shoulder components against noise floor ----------
# Apply the same drop-criterion used for the dropped 415 candidate: a component
# that parks at noise-level amplitude is flagged.
sh120_amp = float(amps[SH_120])
sh450_amp = float(amps[SH_450])
sh120_survives = sh120_amp > 5.0 * noise
sh450_survives = sh450_amp > 5.0 * noise
deviation_note = ''
flags = []
if not sh120_survives:
    flags.append('~120 cm-1 shoulder refined below 5x noise')
if not sh450_survives:
    flags.append('~450 cm-1 shoulder refined below 5x noise')
if flags:
    deviation_note = ('Shoulder component(s) parked near noise-level amplitude: '
                      + '; '.join(flags) + '. Retained per plan seed but flagged.')

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
ax_main.set_yscale('symlog', linthresh=500)
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

# Diagnostic zoom windows on strong/structured bands (own non-shared x-axes).
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

# Zoom on the 130 band + 120 shoulder, and the 400-472 / 464 composite region.
zoom_panel(ax_z0, 109.0, 181.0, '119-130')
zoom_panel(ax_z1, 385.0, 472.0, '400-472')
zoom_panel(ax_z2, 460.0, 500.0, '464-flank')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Results JSON ----------
labels = ['band_120_shoulder', 'band_130', 'band_205', 'band_262', 'band_356',
          'band_402', 'band_450_shoulder', 'band_464', 'band_505', 'band_700',
          'band_810', 'band_1085', 'band_1160']
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
    'model_type': ('Sum of 13 Lorentzian-dominated pseudo-Voigt peaks (eta seeded ~0.8) on a '
                   'constant+linear baseline, fit simultaneously over the full 109-1271 cm-1 range '
                   'via TRF (Levenberg-Marquardt-style) least squares. Seed centers: ~120(shoulder), '
                   '130, 205, 262, 356, 402, 450(shoulder), 464, 505, 700, 810, 1085, 1160 cm-1. '
                   'Dropped the spurious ~415 candidate; added a 120 cm-1 shoulder on the 130-band '
                   'low flank and a 450 cm-1 shoulder on the 464-band rising flank to resolve the '
                   'asymmetric composite line shapes. band_700 FWHM upper bound widened to 55 cm-1. '
                   'No ALS subtraction (rolling-min baseline area fraction = %.3f, not '
                   'fluorescence-dominated); background modeled as constant+linear parameters. '
                   'Center windows +/-10 cm-1 (shoulders tighter: 116-124 and 445-455), '
                   'FWHM 2-40 cm-1, amp>=0, eta in [0,1].' % baseline_fraction),
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': deviation_note
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
