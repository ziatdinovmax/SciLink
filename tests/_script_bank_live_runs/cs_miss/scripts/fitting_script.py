import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lmfit import Model

# --- Load raw data ---
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
    y = data.astype(float)
    x = np.linspace(0, 500, y.size)

x_raw = x.copy()
y_raw = y.copy()

# --- Preprocessing: PL intensity data, clip negatives to zero (noise) ---
y = np.clip(y, 0, None)

# --- Biexponential + constant offset model ---
def biexp(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

model = Model(biexp)
params = model.make_params(A1=6.0, tau1=30.0, A2=4.0, tau2=150.0, C=0.2)

# Constraints: all amplitudes/lifetimes/offset >= 0, enforce tau1 < tau2
params['A1'].set(min=0)
params['A2'].set(min=0)
params['C'].set(min=0)
params['tau1'].set(min=1e-6)
# tau2 = tau1 + delta with delta > 0 to keep tau1 < tau2
params.add('dtau', value=120.0, min=1e-6)
params['tau2'].set(expr='tau1 + dtau')

# Weighting by 1/sqrt(y) (Poisson-like); guard against zeros
weights = 1.0 / np.sqrt(np.clip(y, 1e-6, None))

result = model.fit(y, params, t=x, weights=weights)

fit_y = result.eval(t=x)

# --- Fit quality over full domain ---
resid = y - fit_y
ss_res = np.sum(resid**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = np.sqrt(np.mean(resid**2))

# --- Extract parameters ---
def gv(name):
    p = result.params.get(name)
    return (float(p.value) if p is not None and p.value is not None else float('nan'))

def ge(name):
    p = result.params.get(name)
    if p is not None and p.stderr is not None:
        return float(p.stderr)
    return None

A1, tau1 = gv('A1'), gv('tau1')
A2, tau2 = gv('A2'), gv('tau2')
C = gv('C')

# Amplitude-weighted average lifetime
denom = A1 + A2
tau_avg = (A1 * tau1 + A2 * tau2) / denom if denom > 0 else float('nan')

# --- Save fitted model at all N x-points ---
fit_full = model.eval(result.params, t=x_raw)
np.save('fit.npy', np.asarray(fit_full, dtype=float))

# --- Visualization ---
comp1 = A1 * np.exp(-x / tau1) + C
comp2 = A2 * np.exp(-x / tau2) + C
noise = np.std(resid)
norm_resid = resid / noise if noise > 0 else resid

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False,
                         gridspec_kw={'height_ratios': [3, 1, 1]})

ax0 = axes[0]
ax0.plot(x_raw, y_raw, color='lightgrey', alpha=0.6, lw=1, label='Raw data')
ax0.plot(x, y, '.', color='steelblue', ms=3, alpha=0.6, label='Data')
ax0.plot(x, fit_y, 'r-', lw=1.8, label='Fit')
ax0.plot(x, comp1, '--', color='green', lw=1, label='Component 1')
ax0.plot(x, comp2, '--', color='orange', lw=1, label='Component 2')
ax0.set_yscale('log')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=8)
ax0.set_xlim(x.min(), x.max())

ax1 = axes[1]
ax1.plot(x, resid, color='purple', lw=0.8)
ax1.axhline(0, color='k', lw=0.6)
ax1.set_ylabel('Residuals')
ax1.set_xlim(x.min(), x.max())

ax2 = axes[2]
ax2.plot(x, norm_resid, color='darkred', lw=0.8)
ax2.axhline(0, color='k', lw=0.6)
ax2.set_ylabel('Norm. residuals')
ax2.set_xlabel('X')
ax2.set_xlim(x.min(), x.max())

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# --- Results JSON ---
results = {
    "model_type": "Biexponential decay with constant offset: I(t)=A1*exp(-t/tau1)+A2*exp(-t/tau2)+C",
    "parameters": {
        "component_1_fast": {
            "amplitude": A1, "amplitude_err": ge('A1'),
            "lifetime_ns": tau1, "lifetime_ns_err": ge('tau1')
        },
        "component_2_slow": {
            "amplitude": A2, "amplitude_err": ge('A2'),
            "lifetime_ns": tau2, "lifetime_ns_err": ge('tau2')
        },
        "baseline_offset": {"C": C, "C_err": ge('C')},
        "tau_avg_amplitude_weighted_ns": tau_avg
    },
    "fit_quality": {"r_squared": float(r_squared), "rmse": float(rmse)},
    "deviation_note": ""
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
