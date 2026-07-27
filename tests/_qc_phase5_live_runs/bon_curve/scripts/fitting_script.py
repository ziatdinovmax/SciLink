import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid

# ---------------------------------------------------------------
# 1. Load RAW data
# ---------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x, y = data[0].astype(float), data[1].astype(float)
    elif data.shape[1] == 2:
        x, y = data[:, 0].astype(float), data[:, 1].astype(float)
    else:
        # assume rows are [x, y]
        x, y = data[0].astype(float), data[1].astype(float)
else:
    # 1-D: treat as y, build x from metadata range
    y = data.astype(float)
    x = np.linspace(134.647, 1271.48, y.size)

# ensure ascending x
if x[0] > x[-1]:
    x = x[::-1]
    y = y[::-1]

y_raw = y.copy()

# ---------------------------------------------------------------
# 2. Preprocess: intensity-type Raman -> clip negatives, remove
#    single-pixel cosmic spikes with a mild median filter guard.
# ---------------------------------------------------------------
from scipy.ndimage import median_filter
# cosmic spike removal (single-pixel): replace points that deviate
# strongly from local median
med = median_filter(y, size=5)
spike_mask = np.abs(y - med) > 8.0 * np.std(y - med)
y = np.where(spike_mask, med, y)
# clip negatives (noise) for intensity spectrum
y = np.clip(y, 0, None)

# ---------------------------------------------------------------
# Plan step 1: baseline. The plan states signal returns to ~0
# between bands and baseline is a small fraction of area, so a
# simple LINEAR baseline as a model parameter is used (per plan),
# NOT ALS (ALS is mandated only when fluorescence dominates).
# The linear baseline is fit jointly as model parameters.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 3. Model: sum of 13 pseudo-Voigt components + linear baseline
#    Expanded from 11 to 13 components to resolve two verified
#    band-shape deficiencies (per updated plan):
#      - nu1 band split into a narrow Lorentzian CORE + broader BASE
#      - nu4 doublet resolved into TWO components (~702 / ~706)
# ---------------------------------------------------------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1.0 - eta) * g)

# Component index map (0-based) -> plan peak number:
#  0: 161  (peak1)  narrow shoulder
#  1: 183  (peak2)  broad lattice base
#  2: 190  (peak3)  narrow lattice
#  3: 213  (peak4)  strong lattice
#  4: 272  (peak5)  doublet member
#  5: 285  (peak6)  doublet member
#  6: 702  (peak7)  nu4 first member
#  7: 706  (peak8)  nu4 SECOND member (newly added)
#  8: 1085 (peak9)  nu1 CORE (very narrow, high-eta Lorentzian)
#  9: 1086 (peak10) nu1 BASE (broader, lower-eta, newly added)
# 10: 1105 (peak11) weak satellite
# 11: 1210 (peak12) high-wavenumber pair
# 12: 1240 (peak13) high-wavenumber pair
seed_centers = [161., 183., 190., 213., 272., 285.,
                702., 706., 1085., 1086., 1105., 1210., 1240.]
n_comp = len(seed_centers)

# seed FWHM per plan guidance
seed_fwhm = [10., 22., 10., 10., 8., 8.,
             6., 6., 3., 12., 8., 25., 25.]
# eta: nu1 core high-Lorentzian, nu1 base low (Gaussian-leaning)
seed_eta = [0.5, 0.3, 0.5, 0.5, 0.5, 0.5,
            0.6, 0.6, 0.9, 0.3, 0.6, 0.5, 0.5]

# amplitude seeds from data near each center
def local_amp(c):
    m = np.abs(x - c) < 8
    if np.any(m):
        return max(y[m].max(), 1.0)
    return 1.0
seed_amp = [local_amp(c) for c in seed_centers]
# nu1 core (idx 8) seeded near data max; base (idx 9) a fraction of core
seed_amp[8] = max(y.max() * 0.95, 1.0)
seed_amp[9] = seed_amp[8] * 0.2
# low-x amplitude targets per plan (190~1100, 213~1900)
seed_amp[2] = max(seed_amp[2], 1100.0)
seed_amp[3] = max(seed_amp[3], 1900.0)

# linear baseline seed anchored at signal-free regions (~350-650, ~750-1000)
anchor_mask = ((x > 350) & (x < 650)) | ((x > 750) & (x < 1000))
if np.count_nonzero(anchor_mask) > 5:
    pc = np.polyfit(x[anchor_mask], y[anchor_mask], 1)
    b_slope, b_off = pc[0], pc[1]
else:
    b_slope, b_off = 0.0, float(np.min(y))

# ---------------------------------------------------------------
# parameter vector: [slope, offset,
#   (amp, center, fwhm, eta) x n_comp]
#
# Special width bounds:
#   - nu1 CORE (idx 8): FWHM in [2,5] (near lower bound, sharp core),
#     eta in [0.8,1.0] (Lorentzian)
#   - nu1 BASE (idx 9): FWHM in [8,40] (must stay broader than core to
#     prevent degenerate swapping), eta in [0.0,0.5] (Gaussian-leaning)
#   - 183 broad base (idx 1): FWHM in [18,40] (held broad, not allowed
#     to migrate to 190/213)
#   - nu4 members (idx 6,7): FWHM in [4,12] (narrow); centers separated
#     via disjoint center windows (699-704 and 704-709)
# ---------------------------------------------------------------
p0 = [b_slope, b_off]
lo = [-np.inf, -np.inf]
hi = [np.inf, np.inf]
for i in range(n_comp):
    # default bounds
    c_lo, c_hi = seed_centers[i] - 5.0, seed_centers[i] + 5.0
    f_lo, f_hi = 2.0, 40.0
    e_lo, e_hi = 0.0, 1.0

    if i == 1:  # 183 broad lattice base: keep broad
        f_lo, f_hi = 18.0, 40.0
    elif i == 6:  # nu4 first member ~702: window below 704 to avoid collapse
        c_lo, c_hi = 699.0, 704.0
        f_lo, f_hi = 4.0, 12.0
    elif i == 7:  # nu4 second member ~706: window above 704
        c_lo, c_hi = 704.0, 709.0
        f_lo, f_hi = 4.0, 12.0
    elif i == 8:  # nu1 CORE: very narrow, Lorentzian
        f_lo, f_hi = 2.0, 5.0
        e_lo, e_hi = 0.8, 1.0
    elif i == 9:  # nu1 BASE: broader than core, Gaussian-leaning
        f_lo, f_hi = 8.0, 40.0
        e_lo, e_hi = 0.0, 0.5

    p0 += [seed_amp[i], seed_centers[i], seed_fwhm[i], seed_eta[i]]
    lo += [0.0, c_lo, f_lo, e_lo]
    hi += [np.inf, c_hi, f_hi, e_hi]

p0 = np.array(p0, float)
lo = np.array(lo, float)
hi = np.array(hi, float)
p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

def model(params, xx):
    slope, off = params[0], params[1]
    out = slope * xx + off
    for i in range(n_comp):
        a, c, f, e = params[2 + 4*i: 6 + 4*i]
        out = out + pseudo_voigt(xx, a, c, f, e)
    return out

# fit domain = full measured range (plan: fit full range)
fit_mask = np.ones_like(x, dtype=bool)
xf, yf = x[fit_mask], y[fit_mask]

def resid(params):
    return model(params, xf) - yf

# ---------------------------------------------------------------
# Two-stage fit (per plan step 8) to stabilize the co-located
# nu1 core/base pair:
#   Stage A: fit with nu1 base amplitude pinned small (core carries
#            the band) so low-x and mid regions settle first.
#   Stage B: release everything and refit from stage-A result.
# ---------------------------------------------------------------
# Stage A: temporarily tighten base amplitude upper bound so the core
# is established first, then relax.
hiA = hi.copy()
base_amp_idx = 2 + 4*9
hiA[base_amp_idx] = max(seed_amp[9], 1.0)  # cap base amp low in stage A
p0A = np.clip(p0.copy(), lo + 1e-9, hiA - 1e-9)
try:
    resA = least_squares(resid, p0A, bounds=(lo, hiA), method='trf',
                         max_nfev=20000, x_scale='jac')
    pA = resA.x
except Exception:
    pA = p0.copy()

# Stage B: full release from stage-A solution
pB0 = np.clip(pA, lo + 1e-9, hi - 1e-9)
res = least_squares(resid, pB0, bounds=(lo, hi), method='trf',
                    max_nfev=20000, x_scale='jac')
popt = res.x

# parameter uncertainties from Jacobian
try:
    J = res.jac
    dof = max(1, len(yf) - len(popt))
    resvar = 2.0 * res.cost / dof
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * resvar
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
except Exception:
    perr = np.full_like(popt, np.nan)

# ---------------------------------------------------------------
# evaluate model at ALL x-points
# ---------------------------------------------------------------
fit_full = model(popt, x)

# fit quality on saved arrays (data=y processed, fit=fit_full)
residuals = y - fit_full
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1.0 - ss_res / ss_tot
rmse = np.sqrt(np.mean(residuals**2))

# noise estimate for normalized residual
noise = np.std(residuals[anchor_mask]) if np.count_nonzero(anchor_mask) > 5 else np.std(residuals)
if noise <= 0:
    noise = np.std(residuals) or 1.0

# ---------------------------------------------------------------
# save fit.npy (length N, same x-points)
# ---------------------------------------------------------------
np.save('fit.npy', fit_full)

# ---------------------------------------------------------------
# component curves (peaks only, without baseline) for plotting
# ---------------------------------------------------------------
slope, off = popt[0], popt[1]
baseline = slope * x + off
components = []
for i in range(n_comp):
    a, c, f, e = popt[2 + 4*i: 6 + 4*i]
    components.append(pseudo_voigt(x, a, c, f, e))

# ---------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------
fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 2], hspace=0.4, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x, y_raw, color='0.7', lw=0.8, alpha=0.6, label='Raw')
ax1.plot(x, y, color='black', lw=0.8, label='Data')
ax1.plot(x, fit_full, color='red', lw=1.2, label='Fit')
for i, comp in enumerate(components):
    ax1.plot(x, comp + baseline, lw=0.7, alpha=0.7,
             label=f'Component {i+1}')
ax1.plot(x, baseline, color='blue', lw=0.6, ls='--', alpha=0.6, label='Baseline')
ax1.set_xlim(x.min(), x.max())
ax1.set_ylabel('Y')
ax1.set_title('Data and Fit')
ax1.legend(fontsize=6, ncol=3, loc='upper left')

# raw residual panel
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.plot(x, residuals, color='purple', lw=0.7)
ax2.axhline(0, color='k', lw=0.5)
ax2.set_ylabel('Residual')

# normalized residual panel
ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
ax3.plot(x, residuals / noise, color='green', lw=0.7)
ax3.axhline(0, color='k', lw=0.5)
ax3.axhline(3, color='r', lw=0.4, ls=':')
ax3.axhline(-3, color='r', lw=0.4, ls=':')
ax3.set_ylabel('Residual / noise')
ax3.set_xlabel('X')
ax1.set_xlim(x.min(), x.max())

# zoom sub-panels (own non-shared x-axes) over the two critical regions
ax4 = fig.add_subplot(gs[3, 0])
m_nu4 = (x > 690) & (x < 720)
ax4.plot(x[m_nu4], y[m_nu4], color='black', lw=0.9, label='Data')
ax4.plot(x[m_nu4], fit_full[m_nu4], color='red', lw=1.1, label='Fit')
for i in (6, 7):
    ax4.plot(x[m_nu4], components[i][m_nu4] + baseline[m_nu4], lw=0.8,
             alpha=0.8, label=f'Component {i+1}')
ax4.set_xlim(690, 720)
ax4.set_title('Zoom ~700-710')
ax4.set_xlabel('X')
ax4.legend(fontsize=6)

ax5 = fig.add_subplot(gs[3, 1])
m_nu1 = (x > 1070) & (x < 1100)
ax5.plot(x[m_nu1], y[m_nu1], color='black', lw=0.9, label='Data')
ax5.plot(x[m_nu1], fit_full[m_nu1], color='red', lw=1.1, label='Fit')
for i in (8, 9):
    ax5.plot(x[m_nu1], components[i][m_nu1] + baseline[m_nu1], lw=0.8,
             alpha=0.8, label=f'Component {i+1}')
ax5.set_xlim(1070, 1100)
ax5.set_title('Zoom ~1085')
ax5.set_xlabel('X')
ax5.legend(fontsize=6)

fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------
# 7. Results JSON
# ---------------------------------------------------------------
params_out = {}
amps = [popt[2 + 4*i] for i in range(n_comp)]
max_amp = max(amps) if amps else 1.0
for i in range(n_comp):
    a, c, f, e = popt[2 + 4*i: 6 + 4*i]
    ea, ec, ef, ee = perr[2 + 4*i: 6 + 4*i]
    area = trapezoid(components[i], x)
    params_out[f'peak_{i+1}'] = {
        'center': float(c), 'center_err': float(ec),
        'fwhm': float(f), 'fwhm_err': float(ef),
        'amplitude': float(a), 'amplitude_err': float(ea),
        'eta': float(e), 'eta_err': float(ee),
        'area': float(area),
        'relative_intensity': float(a / max_amp),
    }
params_out['baseline'] = {
    'slope': float(popt[0]), 'slope_err': float(perr[0]),
    'offset': float(popt[1]), 'offset_err': float(perr[1]),
}

results = {
    'model_type': 'Linear baseline + sum of 13 pseudo-Voigt components (nu1 core+base split; nu4 doublet resolved)',
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
