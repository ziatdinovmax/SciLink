import numpy as np
import json
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Load ----------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x_raw, y_raw = data[0].astype(float), data[1].astype(float)
    elif data.shape[1] == 2:
        x_raw, y_raw = data[:, 0].astype(float), data[:, 1].astype(float)
    else:
        x_raw = np.arange(data.shape[0], dtype=float)
        y_raw = data.ravel().astype(float)
else:
    y_raw = data.astype(float)
    x_raw = np.linspace(134.647, 1271.48, len(y_raw))

# sort by x
order = np.argsort(x_raw)
x_raw = x_raw[order]
y_raw = y_raw[order]
N = len(y_raw)

# ---------------- Preprocess (intensity spectrum) ----------------
# Clip negatives (noise) for Raman intensity data
y_proc = np.clip(y_raw, 0, None)

# ---------------- Step 1: ALS baseline (light, near-flat) ----------------
def als_baseline(y, lam=1e6, p=0.01, niter=8):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = y.copy()
    for _ in range(niter):
        W.setdiag(w)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# Step 1 fix (verifier 987-1058 negative-dip issue): raise lambda for a stiffer,
# near-flat baseline that does not bend up into the pre-peak onset region.
baseline = als_baseline(y_proc, lam=1e6, p=0.01, niter=10)
y_corr = y_proc - baseline

# baseline fraction diagnostic
baseline_fraction = float(trapezoid(np.clip(baseline, 0, None)) / max(trapezoid(np.clip(y_proc, 0, None)), 1e-12))

# ---------------- Pseudo-Voigt models ----------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

def split_pseudo_voigt(x, amp, center, fwhm_l, fwhm_r, eta):
    # apex-preserving split pseudo-Voigt: left/right independent widths
    fwhm = np.where(x < center, fwhm_l, fwhm_r)
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# LOCKED 12-component model (component count unchanged = 12). Peak_8 (dominant,
# ~1085) uses a SKEWED (split) pseudo-Voigt per the locked-plan contingency to
# resolve the persistent dispersive S-shaped apex residual. Peak_11 is RELOCATED
# from the degenerate 1230-cluster to seed the real, previously-unmodeled ~190
# cm-1 band; peak_12 becomes a single asymmetric (split) band absorbing the
# entire 1225-1240 region. All other 10 components remain symmetric pseudo-Voigt.
#
# Re-seeded centers (per updated locked plan):
#  (1) 142  (2) 160  (3) 213  (4) 272  (5) 284
#  (6) 701  (7) 705  (8) 1085 [skewed]  (9) 1112
#  (10) 1210  (11) 190 [RELOCATED]  (12) 1230 [single asymmetric band, split]
seeds = [
    (142.0, 12.0),   # 1 low-freq shoulder
    (160.0, 12.0),   # 2 155-165 shoulder cluster
    (213.0, 8.0),    # 3 sharp (tails retuned)
    (272.0, 20.0),   # 4
    (284.0, 20.0),   # 5 doublet partner
    (701.0, 8.0),    # 6 sharp
    (705.0, 8.0),    # 7 doublet partner
    # 8 handled separately (skewed)
    (1112.0, 15.0),  # 9 main-peak asymmetric tail/shoulder
    (1210.0, 20.0),  # 10 weak separate band anchoring 1205-1240 cluster
    (190.0, 12.0),   # 11 RELOCATED: real ~190 cm-1 band (~1100 counts)
    # 12 handled separately (single asymmetric 1230 band, split)
]
# peak_8 skewed seed: center, fwhm_l, fwhm_r
peak8_seed = (1085.0, 8.0, 8.0)
# peak_12 skewed seed (single asymmetric 1230 band): center, fwhm_l, fwhm_r
peak12_seed = (1230.0, 15.0, 15.0)
NC = 12  # total locked components

# fit domain: full range
xmask = np.ones(N, dtype=bool)
xf = x_raw[xmask]
yf = y_corr[xmask]

# amplitude seed from local data near each center
def local_amp(c):
    idx = np.argmin(np.abs(xf - c))
    lo = max(0, idx - 5); hi = min(len(xf), idx + 6)
    return max(np.max(yf[lo:hi]), 1.0)

# ---------------- Parameter vector layout ----------------
# peaks 1-7:   [amp, center, fwhm, eta]           (7*4 = 28 params, idx 0..27)
# peak_8:      [amp, center, fwhm_l, fwhm_r, eta]  (skewed, 5 params, idx 28..32)
# peak_9:      [amp, center, fwhm, eta]            (idx 33..36)
# peak_10:     [amp, center, fwhm, eta]            (idx 37..40)
# peak_11:     [amp, center, fwhm, eta]            (idx 41..44) RELOCATED ~190
# peak_12:     [amp, center, fwhm_l, fwhm_r, eta]  (skewed, 5 params, idx 45..49)
# offset:      last param
p0 = []
lo_b = []
hi_b = []

# symmetric peaks 1-7 (seeds indices 0..6)
for (c, w) in seeds[:7]:
    a0 = local_amp(c)
    p0 += [a0, c, w, 0.5]
    lo_b += [0.0, c - 5.0, 1.0, 0.0]
    hi_b += [np.inf, c + 5.0, 80.0, 1.0]

# peak_8 skewed (dominant): constrain low-freq (left) width so it does not
# overshoot into the 987-1058 pre-peak region (verifier fix); keep right
# width bounded. Retune of asymmetry/eta and relaxed L/R coupling kills the
# apex derivative-like residual. amp = apex height.
c8, fl8, fr8 = peak8_seed
a8 = local_amp(c8)
p0 += [a8, c8, fl8, fr8, 0.5]
lo_b += [0.0, c8 - 5.0, 2.0, 2.0, 0.0]
hi_b += [np.inf, c8 + 5.0, 25.0, 40.0, 1.0]

# symmetric peaks 9, 10, 11 (seeds indices 7..9)
for (c, w) in seeds[7:]:
    a0 = local_amp(c)
    p0 += [a0, c, w, 0.5]
    lo_b += [0.0, c - 5.0, 1.0, 0.0]
    hi_b += [np.inf, c + 5.0, 80.0, 1.0]

# peak_12 skewed (single asymmetric band absorbing 1225-1240 cluster):
# tight, physically-bounded widths/amplitude to prevent recurrence of the
# former peak_11/peak_12 degeneracy while representing the whole region.
c12, fl12, fr12 = peak12_seed
a12 = local_amp(c12)
p0 += [a12, c12, fl12, fr12, 0.5]
lo_b += [0.0, c12 - 5.0, 3.0, 3.0, 0.0]
hi_b += [np.inf, c12 + 5.0, 35.0, 35.0, 1.0]

# Parameter index bookkeeping (must match assembly order above)
#   peaks1-7: 7*4 = 28 params (idx 0..27)
#   peak_8:   5 params -> idx 28..32
#   peak_9 @ 33, peak_10 @ 37, peak_11 @ 41
#   peak_12 (skewed 5 params) @ 45..49
P8 = 28
P9 = 33
P10 = 37
P11 = 41
P12 = 45

# Physical caps on the RELOCATED 190 band (peak_11) and the single 1230 band
# (peak_12) to prevent degeneracy recurrence and unbounded growth.
cap_amp = float(np.max(yf)) * 0.5
# peak_11 (190 cm-1): amplitude cap and width cap (moderate band ~10-15 cm-1)
hi_b[P11 + 0] = cap_amp
hi_b[P11 + 2] = 30.0
# peak_12 (1230 cm-1, skewed): amplitude cap
hi_b[P12 + 0] = cap_amp

# constant offset (fixed non-negative floor >= 0 to prevent negative dip)
p0 += [0.0]
lo_b += [0.0]
hi_b += [abs(np.max(yf)) * 0.5 + 1.0]

p0 = np.array(p0, float)
lo_b = np.array(lo_b, float)
hi_b = np.array(hi_b, float)
p0 = np.clip(p0, lo_b, hi_b)

OFF = len(p0) - 1

def model_eval(params, x):
    out = np.full_like(x, params[OFF], dtype=float)
    # peaks 1-7 symmetric
    for i in range(7):
        amp, center, fwhm, eta = params[4*i:4*i+4]
        out = out + pseudo_voigt(x, amp, center, fwhm, eta)
    # peak_8 skewed
    a8, c8, fl8, fr8, e8 = params[P8:P8+5]
    out = out + split_pseudo_voigt(x, a8, c8, fl8, fr8, e8)
    # peaks 9,10,11 symmetric
    for base in (P9, P10, P11):
        amp, center, fwhm, eta = params[base:base+4]
        out = out + pseudo_voigt(x, amp, center, fwhm, eta)
    # peak_12 skewed
    a12, c12, fl12, fr12, e12 = params[P12:P12+5]
    out = out + split_pseudo_voigt(x, a12, c12, fl12, fr12, e12)
    return out

def resid(params):
    return model_eval(params, xf) - yf

res = least_squares(resid, p0, bounds=(lo_b, hi_b), method='trf',
                    max_nfev=30000, xtol=1e-10, ftol=1e-10)
popt = res.x

# ---------------- Parameter errors ----------------
try:
    J = res.jac
    dof = max(len(yf) - len(popt), 1)
    resvar = 2.0 * res.cost / dof
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * resvar
    perr = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
except Exception:
    perr = np.full(len(popt), np.nan)

# ---------------- Build full-domain fit in RAW space ----------------
fit_corr_full = model_eval(popt, x_raw)
fit_raw_full = fit_corr_full + baseline  # overlay on raw/processed data

# Save fit evaluated at all N points (same space as saved data = y_proc)
np.save('fit.npy', fit_raw_full.astype(float))

# ---------------- R^2 / RMSE from saved arrays ----------------
saved_data = y_proc
saved_fit = fit_raw_full
ssr = float(np.sum((saved_data - saved_fit) ** 2))
sst = float(np.sum((saved_data - np.mean(saved_data)) ** 2))
r2 = 1.0 - ssr / sst if sst > 0 else float('nan')
rmse = float(np.sqrt(ssr / len(saved_data)))

# noise estimate for normalized residual
noise = float(np.median(np.abs(np.diff(y_corr))) / 0.6745) if N > 1 else 1.0
if noise <= 0:
    noise = 1.0

# ---------------- Helper: evaluate a single component ----------------
def comp_eval(idx, x):
    # idx is 0-based component index 0..11
    if idx < 7:
        amp, center, fwhm, eta = popt[4*idx:4*idx+4]
        return pseudo_voigt(x, amp, center, fwhm, eta), amp, center
    elif idx == 7:
        a8, c8, fl8, fr8, e8 = popt[P8:P8+5]
        return split_pseudo_voigt(x, a8, c8, fl8, fr8, e8), a8, c8
    elif idx == 8:
        amp, center, fwhm, eta = popt[P9:P9+4]
        return pseudo_voigt(x, amp, center, fwhm, eta), amp, center
    elif idx == 9:
        amp, center, fwhm, eta = popt[P10:P10+4]
        return pseudo_voigt(x, amp, center, fwhm, eta), amp, center
    elif idx == 10:
        amp, center, fwhm, eta = popt[P11:P11+4]
        return pseudo_voigt(x, amp, center, fwhm, eta), amp, center
    else:
        a12, c12, fl12, fr12, e12 = popt[P12:P12+5]
        return split_pseudo_voigt(x, a12, c12, fl12, fr12, e12), a12, c12

# ---------------- Component relative intensities (by area) & amps ----------------
areas = []
amps = []
for i in range(NC):
    comp, amp, center = comp_eval(i, x_raw)
    areas.append(float(trapezoid(comp, x_raw)))
    amps.append(float(amp))
max_amp = max(amps) if amps else 1.0

# ---------------- JSON parameters ----------------
params_out = {}
for i in range(NC):
    if i < 7:
        amp, center, fwhm, eta = popt[4*i:4*i+4]
        ea, ec, ef, ee = perr[4*i:4*i+4]
        params_out[f'peak_{i+1}'] = {
            'center': float(center), 'center_err': float(ec),
            'fwhm': float(fwhm), 'fwhm_err': float(ef),
            'amplitude': float(amp), 'amplitude_err': float(ea),
            'eta': float(eta), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(amp / max_amp) if max_amp > 0 else float('nan'),
        }
    elif i == 7:
        a8, c8, fl8, fr8, e8 = popt[P8:P8+5]
        ea, ec, efl, efr, ee = perr[P8:P8+5]
        params_out['peak_8'] = {
            'center': float(c8), 'center_err': float(ec),
            'fwhm_left': float(fl8), 'fwhm_left_err': float(efl),
            'fwhm_right': float(fr8), 'fwhm_right_err': float(efr),
            'fwhm': float(0.5 * (fl8 + fr8)),
            'asymmetry_ratio': float(fr8 / fl8) if fl8 > 0 else float('nan'),
            'amplitude': float(a8), 'amplitude_err': float(ea),
            'eta': float(e8), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(a8 / max_amp) if max_amp > 0 else float('nan'),
            'profile': 'skewed_pseudo_voigt',
        }
    elif i == 8:
        amp, center, fwhm, eta = popt[P9:P9+4]
        ea, ec, ef, ee = perr[P9:P9+4]
        params_out['peak_9'] = {
            'center': float(center), 'center_err': float(ec),
            'fwhm': float(fwhm), 'fwhm_err': float(ef),
            'amplitude': float(amp), 'amplitude_err': float(ea),
            'eta': float(eta), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(amp / max_amp) if max_amp > 0 else float('nan'),
        }
    elif i == 9:
        amp, center, fwhm, eta = popt[P10:P10+4]
        ea, ec, ef, ee = perr[P10:P10+4]
        params_out['peak_10'] = {
            'center': float(center), 'center_err': float(ec),
            'fwhm': float(fwhm), 'fwhm_err': float(ef),
            'amplitude': float(amp), 'amplitude_err': float(ea),
            'eta': float(eta), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(amp / max_amp) if max_amp > 0 else float('nan'),
        }
    elif i == 10:
        amp, center, fwhm, eta = popt[P11:P11+4]
        ea, ec, ef, ee = perr[P11:P11+4]
        params_out['peak_11'] = {
            'center': float(center), 'center_err': float(ec),
            'fwhm': float(fwhm), 'fwhm_err': float(ef),
            'amplitude': float(amp), 'amplitude_err': float(ea),
            'eta': float(eta), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(amp / max_amp) if max_amp > 0 else float('nan'),
        }
    else:
        a12, c12, fl12, fr12, e12 = popt[P12:P12+5]
        ea, ec, efl, efr, ee = perr[P12:P12+5]
        params_out['peak_12'] = {
            'center': float(c12), 'center_err': float(ec),
            'fwhm_left': float(fl12), 'fwhm_left_err': float(efl),
            'fwhm_right': float(fr12), 'fwhm_right_err': float(efr),
            'fwhm': float(0.5 * (fl12 + fr12)),
            'asymmetry_ratio': float(fr12 / fl12) if fl12 > 0 else float('nan'),
            'amplitude': float(a12), 'amplitude_err': float(ea),
            'eta': float(e12), 'eta_err': float(ee),
            'area': float(areas[i]),
            'rel_intensity': float(a12 / max_amp) if max_amp > 0 else float('nan'),
            'profile': 'skewed_pseudo_voigt',
        }
params_out['offset'] = {'value': float(popt[OFF]), 'err': float(perr[OFF])}
params_out['baseline_fraction'] = baseline_fraction

# ---------------- Visualization ----------------
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1.3], hspace=0.35, wspace=0.3)

ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(x_raw, y_raw, color='0.75', lw=0.8, alpha=0.6, label='Raw')
ax_main.plot(x_raw, y_proc, color='k', lw=1.0, label='Data')
ax_main.plot(x_raw, fit_raw_full, color='r', lw=1.2, label='Fit')
ax_main.plot(x_raw, baseline, color='b', lw=0.8, ls='--', alpha=0.6, label='Baseline')
colors = plt.cm.viridis(np.linspace(0, 0.9, NC))
for i in range(NC):
    comp, _, _ = comp_eval(i, x_raw)
    ax_main.plot(x_raw, comp + baseline, color=colors[i], lw=0.7, alpha=0.7,
                 label=f'Component {i+1}')
ax_main.set_xlim(x_raw.min(), x_raw.max())
ax_main.set_ylabel('Y')
ax_main.set_xlabel('X')
ax_main.set_title('Data and Fit')
ax_main.legend(fontsize=6, ncol=3, loc='upper right')

# residual panel (normalized)
residual = y_proc - fit_raw_full
ax_res = fig.add_subplot(gs[1, :], sharex=ax_main)
ax_res.plot(x_raw, residual / noise, color='0.3', lw=0.7)
ax_res.axhline(0, color='r', lw=0.6)
ax_res.axhline(3, color='orange', lw=0.5, ls=':')
ax_res.axhline(-3, color='orange', lw=0.5, ls=':')
ax_res.set_ylabel('Residuals / noise')
ax_res.set_xlim(x_raw.min(), x_raw.max())

# zoom sub-panels on flagged regions (own non-shared x-axis)
zoom_regions = [(134, 206), (205, 277), (1058, 1129)]
for j, (zlo, zhi) in enumerate(zoom_regions):
    axz = fig.add_subplot(gs[2, j])
    m = (x_raw >= zlo) & (x_raw <= zhi)
    axz.plot(x_raw[m], y_proc[m], color='k', lw=0.8, label='Data')
    axz.plot(x_raw[m], fit_raw_full[m], color='r', lw=1.0, label='Fit')
    axz.set_xlim(zlo, zhi)
    axz.set_title(f'{zlo}-{zhi}', fontsize=7)
    axz.tick_params(labelsize=6)

# ensure main panel spans full domain
ax_main.set_xlim(x_raw.min(), x_raw.max())
fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------- Output ----------------
results = {
    'model_type': 'Sum of 12 pseudo-Voigt components (peak_8 and peak_12 skewed/split pseudo-Voigt; peak_11 relocated to ~190 cm-1 real band) on stiff ALS-corrected near-flat baseline with non-negative constant offset; full 135-1271 cm-1 fit',
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r2), 'rmse': float(rmse)},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
