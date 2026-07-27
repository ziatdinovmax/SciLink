import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import wofz
from lmfit import Parameters, minimize
from scipy.signal import medfilt

# ---------- Load ----------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x = data[0].astype(float); y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float); y = data[:, 1].astype(float)
    else:
        x = np.arange(data.shape[-1]).astype(float); y = data.reshape(-1).astype(float)
else:
    y = data.astype(float)
    x = np.linspace(134.647, 1271.48, y.size)

x_raw = x.copy(); y_raw = y.copy()

# ---------- Preprocess ----------
# Cosmic-spike removal (single-pixel) via mild median filter comparison
med = medfilt(y, 5)
spike = np.abs(y - med) > 8.0 * np.std(y - med)
y[spike] = med[spike]
# Intensity spectrum: clip negatives to zero (noise)
y = np.clip(y, 0, None)

# ---------- Stiff ALS residual-offset removal ----------
def als_baseline(yy, lam=1e7, p=0.001, niter=15):
    L = len(yy)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = yy.copy()
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * yy)
        w = p * (yy > z) + (1 - p) * (yy < z)
    return z

baseline = als_baseline(y, lam=1e7, p=0.001, niter=15)
y_corr = y - baseline

from scipy.integrate import trapezoid
tot_area = trapezoid(np.abs(y), x)
base_area = trapezoid(np.abs(baseline), x)
baseline_fraction = float(base_area / tot_area) if tot_area > 0 else 0.0

# noise estimate from high-frequency content
noise = np.std(np.diff(y_corr)) / np.sqrt(2.0)
if noise <= 0:
    noise = 1.0

# ---------- Voigt model ----------
def voigt(x, amp, center, sigma, gamma):
    sigma = max(sigma, 1e-6)
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

# ----- Component definitions per the locked plan (10 Voigt components) -----
# Low-wavenumber cluster RE-PARTITIONED: the former single broad Gaussian near
# 203 (FWHM~39) is eliminated in favor of two narrow components at ~190 and ~214.
# The ~214 feature is the DOMINANT low-wavenumber band and carries larger amp.
# peak_3 (previously spurious near-zero) is repurposed as the strong ~214 band.
# Main 1085 band is modeled with a sharp Lorentzian core (peak_7) PLUS a
# co-located narrow apex component (peak_8) to remove the derivative oscillation.
# Seeds: (1)143 (2)190 (3)214 (4)278 doublet region (5)701 (6)706
#        (7)1085 main (8)1085 apex (9)1210 (10)1240
seeds = [143.0, 190.0, 214.0, 278.0, 701.0, 706.0, 1085.0, 1085.0, 1210.0, 1240.0]
n_comp = len(seeds)

def fwhm_to_sigma(f):
    return f / 2.3548
def fwhm_to_gamma(f):
    return f / 2.0

# amplitude seed from local data
def local_amp(c, halfwin=8.0):
    m = np.abs(x - c) <= halfwin
    if not np.any(m):
        return noise * 5
    return max(np.max(y_corr[m]), noise)

params = Parameters()
for i, c in enumerate(seeds):
    pre = f'p{i}_'
    if i == 0:  # ~143 lattice mode
        gf, lf = 8.0, 6.0
        cmin, cmax = 138.0, 150.0
        amp0 = local_amp(c, 8) * 8.0
    elif i == 1:  # ~190 sharp narrow (pinned, no re-merge)
        gf, lf = 6.0, 5.0
        cmin, cmax = 185.0, 197.0
        amp0 = local_amp(c, 6) * 8.0
    elif i == 2:  # ~214 DOMINANT low-wavenumber, narrow, larger amp
        gf, lf = 6.0, 6.0
        cmin, cmax = 208.0, 220.0
        amp0 = local_amp(c, 6) * 10.0
    elif i == 3:  # 272/284 doublet region
        gf, lf = 8.0, 6.0
        cmin, cmax = 268.0, 290.0
        amp0 = local_amp(c, 10) * 8.0
    elif i in (4, 5):  # 700-region doublet (retained as-is)
        gf, lf = 6.0, 5.0
        cmin, cmax = c - 5.0, c + 5.0
        amp0 = local_amp(c, 6) * 8.0
    elif i == 6:  # 1085 main sharp Lorentzian core (high L-mixing, small G)
        gf, lf = 3.0, 6.0
        cmin, cmax = 1081.0, 1089.0
        amp0 = local_amp(c, 6) * 10.0
    elif i == 7:  # 1085 apex sharpening component (very narrow, co-located)
        gf, lf = 1.5, 2.5
        cmin, cmax = 1082.0, 1088.0
        amp0 = local_amp(c, 4) * 6.0
    elif i == 8:  # ~1210
        gf, lf = 10.0, 8.0
        cmin, cmax = 1204.0, 1216.0
        amp0 = local_amp(c, 10) * 6.0
    else:  # ~1240 broad structure
        gf, lf = 16.0, 12.0
        cmin, cmax = 1230.0, 1250.0
        amp0 = local_amp(c, 12) * 6.0
    params.add(pre + 'center', value=c, min=cmin, max=cmax)
    params.add(pre + 'gfwhm', value=gf, min=0.5, max=60.0)
    params.add(pre + 'lfwhm', value=lf, min=0.5, max=60.0)
    params.add(pre + 'amp', value=max(amp0, noise), min=0.0)

# constrain the two narrow low-wavenumber components' widths so they stay narrow
# and do not re-merge into a single broad band (FWHM well below the old ~39)
for i in (1, 2):
    params[f'p{i}_gfwhm'].set(max=20.0)
    params[f'p{i}_lfwhm'].set(max=20.0)
# apex component (peak_8) kept very narrow to sharpen the 1085 core
params['p7_gfwhm'].set(max=6.0)
params['p7_lfwhm'].set(max=8.0)

# flat near-zero baseline (constant offset residual)
params.add('offset', value=0.0, min=-50.0, max=50.0)

def model_eval(pars, xv):
    total = np.full_like(xv, pars['offset'].value, dtype=float)
    for i in range(n_comp):
        pre = f'p{i}_'
        total = total + voigt(xv, pars[pre + 'amp'].value,
                              pars[pre + 'center'].value,
                              fwhm_to_sigma(pars[pre + 'gfwhm'].value),
                              fwhm_to_gamma(pars[pre + 'lfwhm'].value))
    return total

# weighting 1/sqrt(y) so huge 1085 doesn't swamp weak lattice modes
weights = 1.0 / np.sqrt(np.clip(y_corr, noise, None) + noise)

def residual(pars, xv, yv, w):
    return (model_eval(pars, xv) - yv) * w

out = minimize(residual, params, args=(x, y_corr, weights), method='leastsq', max_nfev=30000)
pars = out.params

fit_corr = model_eval(pars, x)
# saved fit lives in RAW space: peaks + ALS baseline overlays raw data
fit_full = fit_corr + baseline

# ---------- Quality (from saved arrays: y_raw vs fit_full) ----------
resid = y_raw - fit_full
ss_res = np.sum(resid ** 2)
ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = np.sqrt(np.mean(resid ** 2))

# ---------- per-band residual RMS ----------
def band_rms(lo, hi):
    m = (x >= lo) & (x <= hi)
    if not np.any(m):
        return None
    return float(np.sqrt(np.mean(resid[m] ** 2)))

per_band_residual_rms = {
    '134_216': band_rms(134.0, 216.0),
    '260_300': band_rms(260.0, 300.0),
    '690_715': band_rms(690.0, 715.0),
    '1075_1100': band_rms(1075.0, 1100.0),
    '1200_1255': band_rms(1200.0, 1255.0),
}

# ---------- Save fit ----------
np.save('fit.npy', fit_full)

# ---------- Build parameter report ----------
def voigt_fwhm(gf, lf):
    return 0.5346 * lf + np.sqrt(0.2166 * lf ** 2 + gf ** 2)

amps = [pars[f'p{i}_amp'].value for i in range(n_comp)]
max_amp = max(amps) if amps else 1.0

parameters = {}
for i in range(n_comp):
    pre = f'p{i}_'
    c = pars[pre + 'center']
    gf = pars[pre + 'gfwhm']
    lf = pars[pre + 'lfwhm']
    a = pars[pre + 'amp']
    tot_f = voigt_fwhm(gf.value, lf.value)
    eta = lf.value / (gf.value + lf.value) if (gf.value + lf.value) > 0 else 0.0
    parameters[f'peak_{i+1}'] = {
        'center': float(c.value),
        'center_err': float(c.stderr) if c.stderr is not None else None,
        'amplitude': float(a.value),
        'amplitude_err': float(a.stderr) if a.stderr is not None else None,
        'gaussian_fwhm': float(gf.value),
        'lorentzian_fwhm': float(lf.value),
        'voigt_fwhm': float(tot_f),
        'gl_mixing': float(eta),
        'rel_intensity': float(a.value / max_amp) if max_amp > 0 else 0.0,
    }

# ---------- Visualization ----------
comps = []
for i in range(n_comp):
    pre = f'p{i}_'
    ci = voigt(x, pars[pre + 'amp'].value, pars[pre + 'center'].value,
               fwhm_to_sigma(pars[pre + 'gfwhm'].value),
               fwhm_to_gamma(pars[pre + 'lfwhm'].value)) + baseline
    comps.append(ci)

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False,
                         gridspec_kw={'height_ratios': [3, 1.2, 1.2, 1.2]})
ax0, ax1, ax2, ax3 = axes

ax0.plot(x_raw, y_raw, color='0.7', lw=0.8, alpha=0.6, label='Raw')
ax0.plot(x, y, color='black', lw=0.8, label='Data')
ax0.plot(x, fit_full, color='red', lw=1.2, label='Fit')
ax0.plot(x, baseline, color='green', lw=0.8, ls='--', alpha=0.7, label='Baseline')
for i, ci in enumerate(comps):
    ax0.plot(x, ci, lw=0.7, alpha=0.6, label=f'Component {i+1}')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.set_xlim(x.min(), x.max())
ax0.legend(fontsize=6, ncol=3)

# normalized residual
nres = resid / noise
ax1.plot(x, nres, color='purple', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.axhline(3, color='r', lw=0.4, ls=':'); ax1.axhline(-3, color='r', lw=0.4, ls=':')
ax1.set_ylabel('Resid / noise')
ax1.set_xlim(x.min(), x.max())

# zoom on low-wavenumber cluster 134-230 (own x-axis, not shared)
zmask = (x >= 134) & (x <= 300)
if np.any(zmask):
    ax2.plot(x[zmask], y[zmask], color='black', lw=0.8, label='Data')
    ax2.plot(x[zmask], fit_full[zmask], color='red', lw=1.0, label='Fit')
    for i, ci in enumerate(comps):
        cv = pars[f'p{i}_center'].value
        if 134 <= cv <= 300:
            ax2.plot(x[zmask], ci[zmask], lw=0.7, alpha=0.7, label=f'Component {i+1}')
    ax2.set_xlim(134, 300)
    ax2.legend(fontsize=6)
ax2.set_ylabel('Y')
ax2.set_xlabel('X')

# zoom on dominant cluster 1060-1130 (own x-axis, not shared)
zmask2 = (x >= 1060) & (x <= 1130)
if np.any(zmask2):
    ax3.plot(x[zmask2], y[zmask2], color='black', lw=0.8, label='Data')
    ax3.plot(x[zmask2], fit_full[zmask2], color='red', lw=1.0, label='Fit')
    for i, ci in enumerate(comps):
        cv = pars[f'p{i}_center'].value
        if 1050 <= cv <= 1140:
            ax3.plot(x[zmask2], ci[zmask2], lw=0.7, alpha=0.7, label=f'Component {i+1}')
    ax3.set_xlim(1060, 1130)
    ax3.legend(fontsize=6)
ax3.set_ylabel('Y')
ax3.set_xlabel('X')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------- Print JSON ----------
results = {
    'model_type': 'Sum of 10 Voigt profiles on flat near-zero baseline with stiff-ALS (lam=1e7) residual-offset removal; low-wavenumber cluster re-partitioned into narrow 143/190/214 components (214 dominant), 1085 main+narrow-apex dual component to remove derivative oscillation; 1/sqrt(y)-weighted Levenberg-Marquardt over full range',
    'parameters': parameters,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'extra_metrics': {
        'baseline_fraction': baseline_fraction,
        'per_band_residual_rms': per_band_residual_rms,
        'noise_level': float(noise),
    },
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
