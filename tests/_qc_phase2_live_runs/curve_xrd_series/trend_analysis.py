import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Load data ----------------
# The provided payload may be a raw list or wrapped in {'results': ...}
with open('series_fit_results.json', 'r') as f:
    raw = json.load(f)

if isinstance(raw, dict):
    results = raw.get('results', raw.get('data', []))
    series_metadata = raw.get('series_metadata', {})
else:
    results = raw
    series_metadata = {}

# Fallback metadata (matches this dehydration series)
if not series_metadata:
    series_metadata = {
        'variable': 'temperature',
        'unit': 'C',
        'values': [28.7, 35.9, 40.9, 48.3, 55.8, 65.7]
    }

var_name = series_metadata.get('variable', 'temperature')
var_unit = series_metadata.get('unit', 'C')
temp_vals = series_metadata.get('values', [])

# Sort results by their index to keep alignment with metadata values
results = sorted(results, key=lambda r: r.get('index', 0))

flagged_indices = set()  # none flagged

# ---------------- Extract per-spectrum info ----------------
n = len(results)
temps = np.array([temp_vals[r['index']] if r['index'] < len(temp_vals) else r['index'] for r in results], dtype=float)

r2 = np.array([r['fit_quality'].get('r_squared', np.nan) for r in results])
n_peaks = np.array([r['fit_quality'].get('n_peaks_fitted', np.nan) for r in results])
scherrer = np.array([r['fit_quality'].get('scherrer_mean_size_nm', np.nan) for r in results])

# Collect all peaks as (center, area, amplitude) per spectrum
peak_lists = []
for r in results:
    plist = []
    for k, v in r['parameters'].items():
        if not k.startswith('peak'):
            continue
        c = v.get('center', np.nan)
        a = v.get('area', 0.0)
        amp = v.get('amplitude', 0.0)
        plist.append((c, a, amp))
    plist.sort(key=lambda x: x[0])
    peak_lists.append(plist)


def dominant_in_window(plist, lo, hi):
    """Return (center, area) of the largest-area peak within [lo,hi], else (nan,0)."""
    best = None
    for c, a, amp in plist:
        if lo <= c <= hi:
            if best is None or a > best[1]:
                best = (c, a)
    return best if best is not None else (np.nan, 0.0)


# ---------------- Diagnostic windows ----------------
# Hydrate phase fingerprints (dominant at low T): ~11.0, ~12.0, ~17.1, ~22.2
# Product phase fingerprints (appear at high T): ~11.9, ~15.95, ~19.99, ~17.93
windows = {
    '~7.3/7.9 (surface)':   (6.8, 8.2),
    '~11.0 hydrate':        (10.7, 11.3),
    '~11.9 product':        (11.5, 12.2),
    '~15.95 product':       (15.6, 16.6),
    '~17.1/17.9':           (16.9, 18.3),
    '~19.99 product':       (19.6, 20.5),
    '~22.2 hydrate':        (21.9, 22.6),
}

# Build center & area matrices [window x spectrum]
win_centers = {}
win_areas = {}
for wname, (lo, hi) in windows.items():
    cs, ars = [], []
    for plist in peak_lists:
        c, a = dominant_in_window(plist, lo, hi)
        cs.append(c)
        ars.append(a)
    win_centers[wname] = np.array(cs)
    win_areas[wname] = np.array(ars)

# ---------------- Detect transition ----------------
# The hydrate peak at ~11.0 deg drops out where the ~19.99 product peak turns on.
hydrate_area = win_areas['~11.0 hydrate']
product_area = win_areas['~19.99 product']

# Transition index: first spectrum where product area exceeds hydrate area
transition_idx = None
for i in range(n):
    if (np.nan_to_num(product_area[i]) > np.nan_to_num(hydrate_area[i])) and np.nan_to_num(product_area[i]) > 0:
        transition_idx = i
        break
if transition_idx is None:
    transition_idx = n // 2

# Midpoint temperature of transition band
if 0 < transition_idx < n:
    t_trans = 0.5 * (temps[transition_idx - 1] + temps[transition_idx])
else:
    t_trans = temps[transition_idx]

pre = np.arange(0, transition_idx)
post = np.arange(transition_idx, n)


def linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    coef = np.polyfit(x[m], y[m], 1)
    yp = np.polyval(coef, x[m])
    ss_res = np.sum((y[m] - yp) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    r2v = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, r2v, x[m]


# ---------------- Figure ----------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Powder Pattern Evolution Through Dehydration  (phase transition near %.1f %s)'
             % (t_trans, var_unit), fontsize=15, fontweight='bold')

band_kw = dict(color='orange', alpha=0.15, zorder=0)


def mark_flags(ax, x, y):
    for i in flagged_indices:
        if i < len(x):
            ax.plot(x[i], y[i], 'rx', markersize=14, markeredgewidth=3, zorder=5)


def add_band(ax):
    if 0 < transition_idx < n:
        ax.axvspan(temps[transition_idx - 1], temps[transition_idx], **band_kw)
        ax.axvline(t_trans, color='orange', ls='--', lw=1.5, alpha=0.8)


# (1) Peak position map vs T -------------------------------------------
ax = axes[0, 0]
cmap = plt.cm.tab10
for j, (wname, cs) in enumerate(win_centers.items()):
    m = np.isfinite(cs)
    ax.plot(temps[m], cs[m], 'o-', color=cmap(j % 10), label=wname, markersize=6)
add_band(ax)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Peak center (2$\\theta$, deg)')
ax.set_title('Diagnostic peak positions vs T')
ax.legend(fontsize=7, ncol=2, loc='center right')
ax.grid(alpha=0.3)

# (2) Number of reflections vs T ---------------------------------------
ax = axes[0, 1]
ax.plot(temps, n_peaks, 's-', color='navy', markersize=8)
add_band(ax)
mark_flags(ax, temps, n_peaks)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Number of fitted reflections')
ax.set_title('Reflection count (peak merging/splitting)')
ax.grid(alpha=0.3)

# (3) Order parameter: hydrate vs product peak areas -------------------
ax = axes[0, 2]
ax.plot(temps, hydrate_area, 'o-', color='steelblue', label='~11.0 deg hydrate area', markersize=7)
ax.plot(temps, product_area, 's-', color='firebrick', label='~19.99 deg product area', markersize=7)
# normalized order parameter
tot = np.nan_to_num(hydrate_area) + np.nan_to_num(product_area)
frac_product = np.where(tot > 0, np.nan_to_num(product_area) / tot, np.nan)
ax2 = ax.twinx()
ax2.plot(temps, frac_product, 'k^--', label='product fraction', markersize=6, alpha=0.7)
ax2.set_ylabel('Product phase fraction', color='k')
ax2.set_ylim(-0.05, 1.05)
add_band(ax)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Integrated area')
ax.set_title('Order parameter (hydrate \u2192 product)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center left')
ax.grid(alpha=0.3)

# (4) Dominant strong-peak 2theta shift with branch regression ---------
ax = axes[1, 0]
# Strong peak: hydrate ~22.2 (pre) and product ~19.99 (post) are the strongest
strong_center = np.full(n, np.nan)
for i in range(n):
    if i < transition_idx:
        c, a = dominant_in_window(peak_lists[i], 21.9, 22.6)
    else:
        c, a = dominant_in_window(peak_lists[i], 19.6, 20.5)
    strong_center[i] = c
ax.plot(temps[pre], strong_center[pre], 'o', color='steelblue', markersize=9,
        label='hydrate strong peak (~22.2)')
ax.plot(temps[post], strong_center[post], 's', color='firebrick', markersize=9,
        label='product strong peak (~20.0)')
# Regression on each branch
for idxs, col, lab in [(pre, 'steelblue', 'hydrate fit'), (post, 'firebrick', 'product fit')]:
    fit = linfit(temps[idxs], strong_center[idxs])
    if fit is not None:
        coef, r2v, xm = fit
        xs = np.linspace(xm.min(), xm.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), '--', color=col,
                label='%s: slope=%.4f, R\u00b2=%.2f' % (lab, coef[0], r2v))
add_band(ax)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Strongest peak 2$\\theta$ (deg)')
ax.set_title('Dominant reflection position (per-phase regression)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (5) Fit R2 vs T ------------------------------------------------------
ax = axes[1, 1]
ax.plot(temps, r2, 'D-', color='green', markersize=7)
add_band(ax)
mark_flags(ax, temps, r2)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Fit R\u00b2')
ax.set_title('Fit quality (R\u00b2) across series')
ax.grid(alpha=0.3)
ax.ticklabel_format(useOffset=False, axis='y')

# (6) Mean Scherrer crystallite size vs T ------------------------------
ax = axes[1, 2]
ax.plot(temps, scherrer, 'o-', color='purple', markersize=7)
fit = linfit(temps, scherrer)
if fit is not None:
    coef, r2v, xm = fit
    xs = np.linspace(xm.min(), xm.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), 'k--',
            label='linear: slope=%.4f nm/%s, R\u00b2=%.2f' % (coef[0], var_unit, r2v))
    ax.legend(fontsize=8)
add_band(ax)
mark_flags(ax, temps, scherrer)
ax.set_xlabel('%s (%s)' % (var_name, var_unit))
ax.set_ylabel('Mean Scherrer size (nm)')
ax.set_title('Mean crystallite size vs T')
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')

# ---------------- Console summary ----------------
print('=== Dehydration phase-transition analysis ===')
print('Temperatures (%s): %s' % (var_unit, temps.tolist()))
print('Number of reflections: %s' % n_peaks.tolist())
print('Detected transition between index %d (%.1f %s) and %d (%.1f %s); midpoint ~%.1f %s'
      % (max(transition_idx-1,0), temps[max(transition_idx-1,0)], var_unit,
         transition_idx, temps[transition_idx], var_unit, t_trans, var_unit))
print('Hydrate ~11 deg area: %s' % np.round(hydrate_area, 1).tolist())
print('Product ~20 deg area: %s' % np.round(product_area, 1).tolist())
print('Product fraction: %s' % np.round(frac_product, 3).tolist())
