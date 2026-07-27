import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - REQUIRED
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

# The provided payload may be a bare list or a dict with 'results'.
if isinstance(data, dict):
    results = data.get('results', data.get('series', []))
    series_metadata = data.get('series_metadata', {})
else:
    results = data
    series_metadata = {}

# Fallback metadata if not embedded in the results file
if not series_metadata:
    series_metadata = {
        'variable': 'temperature',
        'unit': 'C',
        'values': [33.4, 35.9]
    }

var_name = series_metadata.get('variable', 'index')
var_unit = series_metadata.get('unit', '')
var_values_list = series_metadata.get('values', [])

# ------------------------------------------------------------------
# Build per-spectrum x values (primary control variable)
# ------------------------------------------------------------------
x_vals = []
names = []
flagged = []
flagged_set = set()  # no spectra flagged in this series

for r in results:
    idx = r.get('index', len(x_vals))
    if idx < len(var_values_list):
        x_vals.append(var_values_list[idx])
    else:
        x_vals.append(idx)
    nm = r.get('name', 'spec_%d' % idx)
    names.append(nm)
    flagged.append(nm in flagged_set or idx in flagged_set)

x_vals = np.array(x_vals, dtype=float)
flagged = np.array(flagged, dtype=bool)
order = np.argsort(x_vals)

# ------------------------------------------------------------------
# Helper: gather peaks list from a result
# ------------------------------------------------------------------
def get_peaks(r):
    params = r.get('parameters', {})
    peaks = []
    for k, v in params.items():
        if k.startswith('peak_') and isinstance(v, dict) and 'center' in v:
            peaks.append(v)
    return peaks

# Match a target center to nearest peak in a result (within tolerance)
def find_peak_near(r, target_center, tol=0.25):
    best = None
    best_d = tol
    for p in get_peaks(r):
        d = abs(p['center'] - target_center)
        if d < best_d:
            best_d = d
            best = p
    return best

# ------------------------------------------------------------------
# Choose dominant reflections to track (by center) using largest-area peaks
# in the reference (first) spectrum
# ------------------------------------------------------------------
ref_peaks = sorted(get_peaks(results[0]), key=lambda p: p.get('area', 0.0), reverse=True)
target_centers = [p['center'] for p in ref_peaks[:5]]
target_centers = sorted(target_centers)

# ------------------------------------------------------------------
# Extract fit-quality metrics
# ------------------------------------------------------------------
r2 = []
r2_peak = []
scherrer_mean = []
wh_size = []
wh_strain = []
for r in results:
    fq = r.get('fit_quality', {})
    r2.append(fq.get('r_squared', np.nan))
    r2_peak.append(fq.get('peak_region_r2', np.nan))
    scherrer_mean.append(fq.get('scherrer_mean_size_nm', np.nan))
    wh = fq.get('williamson_hall', {})
    wh_size.append(wh.get('size_nm', np.nan))
    wh_strain.append(wh.get('strain', np.nan))

r2 = np.array(r2)
r2_peak = np.array(r2_peak)
scherrer_mean = np.array(scherrer_mean)
wh_size = np.array(wh_size)
wh_strain = np.array(wh_strain)

# ------------------------------------------------------------------
# Build trend arrays for the tracked peaks
# ------------------------------------------------------------------
peak_area = {c: [] for c in target_centers}
peak_fwhm = {c: [] for c in target_centers}
peak_center = {c: [] for c in target_centers}

for r in results:
    for c in target_centers:
        p = find_peak_near(r, c)
        if p is not None:
            peak_area[c].append(p.get('area', np.nan))
            peak_fwhm[c].append(p.get('fwhm', np.nan))
            peak_center[c].append(p.get('center', np.nan))
        else:
            peak_area[c].append(np.nan)
            peak_fwhm[c].append(np.nan)
            peak_center[c].append(np.nan)

for c in target_centers:
    peak_area[c] = np.array(peak_area[c])
    peak_fwhm[c] = np.array(peak_fwhm[c])
    peak_center[c] = np.array(peak_center[c])

# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------
xlabel = '%s (%s)' % (var_name.capitalize(), var_unit) if var_unit else var_name.capitalize()
colors = plt.cm.viridis(np.linspace(0, 0.85, len(target_centers)))

def mark_flagged(ax, xarr, yarr):
    if np.any(flagged):
        ax.scatter(xarr[flagged], yarr[flagged], marker='x', s=140,
                   color='red', zorder=10, linewidths=2.5, label='flagged')

def add_trend(ax, xarr, yarr, color='k'):
    m = np.isfinite(xarr) & np.isfinite(yarr)
    if np.sum(m) >= 2 and len(np.unique(xarr[m])) >= 2:
        coef = np.polyfit(xarr[m], yarr[m], 1)
        xs = np.linspace(np.min(xarr[m]), np.max(xarr[m]), 50)
        ax.plot(xs, np.polyval(coef, xs), '--', color=color, alpha=0.5, lw=1.2)

# ------------------------------------------------------------------
# Figure: 2 x 3 dashboard
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Parameter Trends vs %s  (in-situ dehydration series)' % var_name.capitalize(),
             fontsize=15, fontweight='bold')

xs_o = x_vals[order]

# (0,0) Peak areas of dominant reflections
ax = axes[0, 0]
for i, c in enumerate(target_centers):
    y = peak_area[c][order]
    ax.plot(xs_o, y, 'o-', color=colors[i], label='%.2f\u00b0' % c, markersize=7)
    add_trend(ax, xs_o, y, colors[i])
    mark_flagged(ax, xs_o, y)
ax.set_xlabel(xlabel)
ax.set_ylabel('Integrated area')
ax.set_title('Dominant peak areas')
ax.legend(fontsize=8, title='2\u03b8 center')
ax.grid(alpha=0.3)

# (0,1) Peak FWHM
ax = axes[0, 1]
for i, c in enumerate(target_centers):
    y = peak_fwhm[c][order]
    ax.plot(xs_o, y, 's-', color=colors[i], label='%.2f\u00b0' % c, markersize=7)
    add_trend(ax, xs_o, y, colors[i])
    mark_flagged(ax, xs_o, y)
ax.set_xlabel(xlabel)
ax.set_ylabel('FWHM (\u00b0 2\u03b8)')
ax.set_title('Dominant peak FWHM (broadening)')
ax.legend(fontsize=8, title='2\u03b8 center')
ax.grid(alpha=0.3)

# (0,2) Peak center positions (relative shift)
ax = axes[0, 2]
for i, c in enumerate(target_centers):
    y = peak_center[c][order]
    y0 = y[np.isfinite(y)][0] if np.any(np.isfinite(y)) else np.nan
    dy = y - y0
    ax.plot(xs_o, dy, 'd-', color=colors[i], label='%.2f\u00b0' % c, markersize=7)
    add_trend(ax, xs_o, dy, colors[i])
    mark_flagged(ax, xs_o, dy)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(xlabel)
ax.set_ylabel('\u0394 center (\u00b0 2\u03b8)')
ax.set_title('Peak position shift (lattice expansion/contraction)')
ax.legend(fontsize=8, title='2\u03b8 center')
ax.grid(alpha=0.3)

# (1,0) Mean Scherrer size + Williamson-Hall size
ax = axes[1, 0]
y = scherrer_mean[order]
ax.plot(xs_o, y, 'o-', color='tab:blue', label='Scherrer mean', markersize=8)
add_trend(ax, xs_o, y, 'tab:blue')
mark_flagged(ax, xs_o, y)
y2 = wh_size[order]
ax.plot(xs_o, y2, '^-', color='tab:orange', label='W-H size', markersize=8)
add_trend(ax, xs_o, y2, 'tab:orange')
mark_flagged(ax, xs_o, y2)
ax.set_xlabel(xlabel)
ax.set_ylabel('Crystallite size (nm)')
ax.set_title('Crystallite size metrics')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (1,1) Williamson-Hall microstrain
ax = axes[1, 1]
y = wh_strain[order]
ax.plot(xs_o, y, 'o-', color='tab:green', markersize=8)
add_trend(ax, xs_o, y, 'tab:green')
mark_flagged(ax, xs_o, y)
ax.set_xlabel(xlabel)
ax.set_ylabel('Microstrain (W-H)')
ax.set_title('Williamson-Hall microstrain')
ax.grid(alpha=0.3)

# (1,2) Fit quality
ax = axes[1, 2]
y = r2[order]
ax.plot(xs_o, y, 'o-', color='tab:red', label='global R\u00b2', markersize=8)
mark_flagged(ax, xs_o, y)
y2 = r2_peak[order]
ax.plot(xs_o, y2, 's-', color='tab:purple', label='peak-region R\u00b2', markersize=8)
mark_flagged(ax, xs_o, y2)
ax.set_xlabel(xlabel)
ax.set_ylabel('R\u00b2')
ax.set_title('Fit quality')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Handle degenerate x-axis (few unique points) for readability
for ax in axes.ravel():
    uniq = np.unique(xs_o[np.isfinite(xs_o)])
    if len(uniq) <= 4:
        ax.set_xticks(uniq)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display
