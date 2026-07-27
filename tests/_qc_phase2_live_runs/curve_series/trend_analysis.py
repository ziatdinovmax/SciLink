import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Load data ----------------
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

# Support either {'results':[...]} or a bare list
if isinstance(data, dict):
    results = data.get('results', data.get('series', []))
    series_metadata = data.get('series_metadata', {})
else:
    results = data
    series_metadata = {}

# Fallback metadata if not embedded
if not series_metadata:
    series_metadata = {
        'variable': 'temperature',
        'unit': 'K',
        'values': [243.2, 258.2, 278.2, 293.5, 313.2]
    }

var_name = series_metadata.get('variable', 'temperature')
var_unit = series_metadata.get('unit', 'K')
temp_values = series_metadata.get('values', [])

# ---------------- Extract per-spectrum info ----------------
# temp_values is aligned to results by results[i]['index']
records = []
for r in results:
    idx = r.get('index', None)
    if idx is not None and idx < len(temp_values):
        T = temp_values[idx]
    else:
        T = np.nan

    params = r.get('parameters', {})
    fitq = r.get('fit_quality', {})
    r2 = fitq.get('r_squared', np.nan)
    flagged = bool(r.get('flagged', False))

    # Collect all peak sub-dicts (those having a center)
    peaks = []
    for k, v in params.items():
        if isinstance(v, dict):
            c = v.get('center_ppm', v.get('center', None))
            if c is None:
                continue
            peaks.append({
                'name': k,
                'center': c,
                'fwhm': v.get('fwhm_ppm', np.nan),
                'area': v.get('area', v.get('integrated_area', np.nan)),
                'relpop': v.get('relative_population', np.nan),
                'center_err': v.get('center_err', np.nan)
            })

    ncomp = len(peaks)

    # Dominant peak = largest relative_population (fallback: largest area)
    if peaks:
        def dom_key(p):
            rp = p['relpop']
            if rp is None or (isinstance(rp, float) and np.isnan(rp)):
                return p['area'] if not np.isnan(p['area']) else -np.inf
            return rp
        dom = max(peaks, key=dom_key)
    else:
        dom = None

    total_area = np.nansum([p['area'] for p in peaks]) if peaks else np.nan

    records.append({
        'name': r.get('name', ''),
        'T': T,
        'r2': r2,
        'flagged': flagged,
        'ncomp': ncomp,
        'peaks': peaks,
        'dom': dom,
        'total_area': total_area
    })

# Sort by temperature
records.sort(key=lambda d: (np.nan_to_num(d['T'], nan=1e9)))

T_arr = np.array([d['T'] for d in records], dtype=float)
r2_arr = np.array([d['r2'] for d in records], dtype=float)
ncomp_arr = np.array([d['ncomp'] for d in records], dtype=float)
dom_center = np.array([d['dom']['center'] if d['dom'] else np.nan for d in records], dtype=float)
dom_center_err = np.array([
    d['dom']['center_err'] if (d['dom'] and d['dom']['center_err'] is not None and not (isinstance(d['dom']['center_err'], float) and np.isnan(d['dom']['center_err']))) else np.nan
    for d in records], dtype=float)
dom_fwhm = np.array([d['dom']['fwhm'] if d['dom'] else np.nan for d in records], dtype=float)
tot_area = np.array([d['total_area'] for d in records], dtype=float)
flag_mask = np.array([d['flagged'] for d in records], dtype=bool)

# ---------------- Regressions ----------------
def linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    coef = np.polyfit(x[m], y[m], 1)
    yhat = np.polyval(coef, x[m])
    ss_res = np.sum((y[m] - yhat) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'slope': coef[0], 'intercept': coef[1], 'r2': r2, 'x': x[m]}

fit_center = linfit(T_arr, dom_center)
fit_fwhm = linfit(T_arr, dom_fwhm)

regression_summary = {}
if fit_center:
    regression_summary['dominant_center_vs_T'] = {
        'slope_ppm_per_K': fit_center['slope'],
        'intercept_ppm': fit_center['intercept'],
        'r2': fit_center['r2']
    }
if fit_fwhm:
    regression_summary['dominant_fwhm_vs_T'] = {
        'slope_ppm_per_K': fit_fwhm['slope'],
        'intercept_ppm': fit_fwhm['intercept'],
        'r2': fit_fwhm['r2']
    }
print('Regression summary:')
print(json.dumps(regression_summary, indent=2))

# ---------------- Dashboard ----------------
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('31P NMR mixed G2/TEP Mg-electrolyte: fitted parameter trends vs temperature',
             fontsize=14, fontweight='bold')

xlabel = f'{var_name} ({var_unit})'

# Panel 1: dominant peak center vs T with regression
ax = axes[0, 0]
yerr = np.where(np.isfinite(dom_center_err), dom_center_err, 0.0)
ax.errorbar(T_arr, dom_center, yerr=yerr, fmt='o', color='navy',
            capsize=3, label='dominant 31P peak', zorder=3)
# overlay all peak centers (minor components) as faint points
for d in records:
    for p in d['peaks']:
        if d['dom'] and p is d['dom']:
            continue
        ax.scatter(d['T'], p['center'], color='gray', alpha=0.6, s=30, zorder=2)
if fit_center:
    xs = np.linspace(np.nanmin(T_arr), np.nanmax(T_arr), 50)
    ax.plot(xs, np.polyval([fit_center['slope'], fit_center['intercept']], xs),
            'r--', label=f"fit: {fit_center['slope']:.4f} ppm/K\nR2={fit_center['r2']:.3f}")
if flag_mask.any():
    ax.scatter(T_arr[flag_mask], dom_center[flag_mask], marker='x', color='red',
               s=140, linewidths=3, zorder=5, label='flagged')
ax.set_xlabel(xlabel); ax.set_ylabel('center (ppm)')
ax.set_title('Dominant peak chemical shift'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.invert_yaxis()  # NMR convention

# Panel 2: dominant peak FWHM vs T (coalescence / exchange broadening)
ax = axes[0, 1]
ax.plot(T_arr, dom_fwhm, 'o-', color='darkgreen', label='dominant FWHM', zorder=3)
if fit_fwhm:
    xs = np.linspace(np.nanmin(T_arr), np.nanmax(T_arr), 50)
    ax.plot(xs, np.polyval([fit_fwhm['slope'], fit_fwhm['intercept']], xs),
            'r--', label=f"fit: {fit_fwhm['slope']:.4f} ppm/K\nR2={fit_fwhm['r2']:.3f}")
if flag_mask.any():
    ax.scatter(T_arr[flag_mask], dom_fwhm[flag_mask], marker='x', color='red',
               s=140, linewidths=3, zorder=5, label='flagged')
ax.set_xlabel(xlabel); ax.set_ylabel('FWHM (ppm)')
ax.set_title('Dominant peak linewidth (exchange broadening)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: number of resolved components (coalescence map)
ax = axes[0, 2]
ax.plot(T_arr, ncomp_arr, 's-', color='purple', markersize=9, zorder=3)
for xi, yi in zip(T_arr, ncomp_arr):
    if np.isfinite(xi):
        ax.annotate(f'{int(yi)}', (xi, yi), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)
if flag_mask.any():
    ax.scatter(T_arr[flag_mask], ncomp_arr[flag_mask], marker='x', color='red',
               s=140, linewidths=3, zorder=5)
ax.set_xlabel(xlabel); ax.set_ylabel('resolved components')
ax.set_title('Resolved 31P species count\n(multi-peak at low T -> single at high T)')
ax.set_ylim(0, max(4, np.nanmax(ncomp_arr) + 1)); ax.grid(alpha=0.3)

# Panel 4: all peak centers colored by relative population (species landscape)
ax = axes[1, 0]
sc_x, sc_y, sc_c = [], [], []
for d in records:
    for p in d['peaks']:
        rp = p['relpop']
        sc_x.append(d['T']); sc_y.append(p['center'])
        sc_c.append(rp if (rp is not None and np.isfinite(rp)) else 1.0)
sc = ax.scatter(sc_x, sc_y, c=sc_c, cmap='viridis', s=90, edgecolor='k',
                vmin=0, vmax=1, zorder=3)
cb = fig.colorbar(sc, ax=ax); cb.set_label('relative population')
ax.set_xlabel(xlabel); ax.set_ylabel('center (ppm)')
ax.set_title('All resolved peak positions vs T'); ax.grid(alpha=0.3)
ax.invert_yaxis()

# Panel 5: integrated / total area vs T
ax = axes[1, 1]
ax.plot(T_arr, tot_area, 'D-', color='chocolate', zorder=3)
if flag_mask.any():
    ax.scatter(T_arr[flag_mask], tot_area[flag_mask], marker='x', color='red',
               s=140, linewidths=3, zorder=5)
ax.set_xlabel(xlabel); ax.set_ylabel('total integrated area')
ax.set_title('Total 31P integrated area (conservation check)')
ax.grid(alpha=0.3)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Panel 6: fit quality R2 vs T
ax = axes[1, 2]
ax.plot(T_arr, r2_arr, 'o-', color='teal', zorder=3)
if flag_mask.any():
    ax.scatter(T_arr[flag_mask], r2_arr[flag_mask], marker='x', color='red',
               s=140, linewidths=3, zorder=5, label='flagged')
ax.set_xlabel(xlabel); ax.set_ylabel('R2')
ax.set_title('Fit quality (R-squared)')
ax.set_ylim(min(0.99, np.nanmin(r2_arr) - 0.002), 1.0005)
ax.grid(alpha=0.3)

# Annotate regression summary as figure text
txt_lines = []
if fit_center:
    txt_lines.append(f"Dominant shift: {fit_center['slope']:.4f} ppm/K (R2={fit_center['r2']:.3f})")
if fit_fwhm:
    txt_lines.append(f"Dominant FWHM: {fit_fwhm['slope']:.4f} ppm/K (R2={fit_fwhm['r2']:.3f})")
if txt_lines:
    fig.text(0.5, 0.005, '  |  '.join(txt_lines), ha='center', fontsize=9,
             style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')

print('Saved parameter_trends.png')
