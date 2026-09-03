"""``derive_phase_products`` tool — piston-immune observables from a wrapped
relative-phase stack: steady-state difference maps between the first and last
condition of a run, band-profile ROI traces vs time, and a template-amplitude
trace, each with same-stem JSON sidecars so downstream curve/image agents can
consume them directly.

Why not per-pixel temporal unwrapping: an interferometer's global piston
fluctuates randomly between frames (often > pi), and a fast transition can
change the phase by more than pi per frame, so temporal unwrapping aliases.
Every observable here is either a within-frame spatial difference or a
circular-mean field over a steady window (the piston factors out exactly).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..._shared._spec import ToolSpec


def _disc_frac(u, valid):
    dy = np.abs(np.diff(u, axis=0)); vy = valid[1:] & valid[:-1] & np.isfinite(dy)
    dx = np.abs(np.diff(u, axis=1)); vx = valid[:, 1:] & valid[:, :-1] & np.isfinite(dx)
    n = vy.sum() + vx.sum()
    return float(((dy[vy] > np.pi).sum() + (dx[vx] > np.pi).sum()) / max(n, 1))


def _unwrap2d(d, valid):
    from skimage.restoration import unwrap_phase
    u = np.asarray(unwrap_phase(np.ma.array(d, mask=~valid))).astype(np.float64)
    u[~valid] = np.nan
    return u


def find_dense_fringe_edges(Z: np.ndarray, valid: np.ndarray, max_gradient: float = 1.05,
                            pad: int = 2) -> list:
    """Columns at the left/right EDGE of the field where the steady-state wrapped
    phase changes faster than ``max_gradient`` rad per binned px (fringes closer
    than ~6 px: unresolvable / aliased). Measured from the circular-mean field
    of the last quarter vs the first quarter of the run. Returns [[x0, x1], ...]
    in binned columns, only contiguous zones touching an edge."""
    n = Z.shape[0]; q = max(n // 4, 1)
    FA = Z[:q].mean(axis=0); FB = Z[-q:].mean(axis=0)
    D = np.angle(FB * np.conj(FA))
    dx = np.abs(np.angle(np.exp(1j * (D[:, 1:] - D[:, :-1]))))
    v = valid[:, 1:] & valid[:, :-1]
    col = np.array([np.median(dx[:, j][v[:, j]]) if v[:, j].sum() > 10 else 0.0 for j in range(dx.shape[1])])
    dense = col > max_gradient
    zones = []
    W = Z.shape[2]
    if dense[:5].any():
        x1 = 0
        while x1 < len(dense) and (dense[x1] or dense[x1:x1 + 3].any()):
            x1 += 1
        zones.append([0, min(W, x1 + 1 + pad)])
    if dense[-5:].any():
        x0 = len(dense) - 1
        while x0 > 0 and (dense[x0] or dense[max(0, x0 - 3):x0].any()):
            x0 -= 1
        zones.append([max(0, x0 - pad), W])
    return zones


def _read_timeline(path: Optional[str], n: int, state_column: str, time_column: str):
    """Return (elapsed_s[n], states[n], extra columns dict). Without a
    timeline every frame is one state and elapsed = frame index."""
    if not path:
        return np.arange(n, dtype=float), ["all"] * n, {}
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if len(rows) != n:
        raise ValueError(f"timeline has {len(rows)} rows but the stack has {n} frames")
    t = np.array([float(r.get(time_column, i)) for i, r in enumerate(rows)])
    states = [str(r.get(state_column, "all")) for r in rows]
    extras = {}
    for k in rows[0]:
        if k in (state_column, time_column):
            continue
        try:
            extras[k] = [float(r[k]) if r[k] not in ("", None) else float("nan") for r in rows]
        except ValueError:
            continue          # text columns never enter the trace CSV (curve loaders need all-numeric)
    return t, states, extras


def derive_phase_products(
    wrapped_phase_npy: str,
    output_dir: str,
    timeline_csv: Optional[str] = None,
    state_column: str = "magnet_state",
    time_column: str = "capture_elapsed_s",
    transition_states: Optional[list] = None,
    steady_window_frames: int = 25,
    bin_factor: int = 2,
    smoothing_sigma: float = 1.0,
    coherence_min: float = 0.6,
    exclude_columns: Optional[list] = None,
    auto_exclude_dense_fringes: bool = False,
    dense_fringe_max_gradient_rad_per_px: float = 1.05,
    band_rows: Optional[list] = None,
    roi_x_ranges: Optional[dict] = None,
    reference_roi: Optional[str] = None,
    template_amplitude_grid: Optional[list] = None,
    frame_index_map: Optional[list] = None,
    stem: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    """See ``TOOL_SPEC``. Returns a dict of product paths and summary scalars."""
    t0 = time.time()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    W = np.load(wrapped_phase_npy, mmap_mode="r")
    n, H0, W0 = W.shape
    stem = stem or Path(wrapped_phase_npy).stem.replace("_wrapped_phase", "")
    b = int(bin_factor)
    H, Wd = H0 // b, W0 // b
    from scipy.ndimage import gaussian_filter
    Z = np.zeros((n, H, Wd), dtype=np.complex64)
    for t in range(n):
        z = np.exp(1j * np.asarray(W[t][: H * b, : Wd * b], dtype=np.float32)).reshape(H, b, Wd, b).mean(axis=(1, 3))
        if smoothing_sigma and smoothing_sigma > 0:
            z = gaussian_filter(z.real, smoothing_sigma) + 1j * gaussian_filter(z.imag, smoothing_sigma)
        Z[t] = z
    valid = np.abs(Z).mean(axis=0) >= coherence_min
    excluded = []
    for rng in (exclude_columns or []):
        x0, x1 = int(rng[0]) // b, int(rng[1]) // b
        valid[:, x0:x1] = False; excluded.append([x0 * b, x1 * b])
    Z /= np.maximum(np.abs(Z), 1e-6)
    if auto_exclude_dense_fringes:
        auto = find_dense_fringe_edges(Z, valid, max_gradient=dense_fringe_max_gradient_rad_per_px)
        for x0, x1 in auto:
            valid[:, x0:x1] = False; excluded.append([x0 * b, x1 * b])
    base_valid = np.abs(Z).mean(axis=0) >= coherence_min if False else None  # noqa: F841 (kept for clarity)
    kept_cols = valid.any(axis=0).sum()
    if kept_cols < 0.5 * Wd:
        raise ValueError(
            f"exclusion leaves only {kept_cols}/{Wd} columns: a dense-fringe zone must be MEASURED "
            "(auto_exclude_dense_fringes=True), never assumed; excluding most of the frame discards the signal")
    # ---- conditions and steady windows ----
    elapsed, states, extras = _read_timeline(timeline_csv, n, state_column, time_column)
    trans = set(transition_states or [])
    first_state, last_state = states[0], states[-1]
    single = all(s == first_state for s in states)
    N = int(steady_window_frames)
    if single:
        first_end, last_start = N - 1, n - N
    else:
        first_end = max(i for i in range(n) if all(s == first_state for s in states[: i + 1]))
        last_start = min(i for i in range(n) if all(s == last_state for s in states[i:]))
    winA = (max(0, first_end - N + 1), first_end + 1)
    winB = (max(last_start, n - N), n)
    FA = Z[winA[0]:winA[1]].mean(axis=0); FB = Z[winB[0]:winB[1]].mean(axis=0)
    # ---- steady-state map ----
    dmap = _unwrap2d(np.angle(FB * np.conj(FA)), valid)
    dmap -= np.nanmedian(dmap[valid])
    map_disc = _disc_frac(dmap, valid)
    h1 = Z[winB[0]:(winB[0] + winB[1]) // 2].mean(axis=0); h2 = Z[(winB[0] + winB[1]) // 2:winB[1]].mean(axis=0)
    m1 = _unwrap2d(np.angle(h1 * np.conj(FA)), valid); m2 = _unwrap2d(np.angle(h2 * np.conj(FA)), valid)
    hres = np.angle(np.exp(1j * (m1[valid] - m2[valid] - np.nanmedian(m1[valid] - m2[valid]))))
    lab = label or (f"{last_state}_minus_{first_state}" if not single else f"end_minus_start_{first_state}")
    map_path = out / f"{stem}_diffmap_{lab}.npy"
    np.save(map_path, np.nan_to_num(dmap, nan=0.0).astype(np.float32), allow_pickle=False)
    np.save(out / f"{stem}_diffmap_{lab}_nan_outside_valid.npy", dmap.astype(np.float32), allow_pickle=False)
    np.save(out / f"{stem}_valid_mask_x{b}.npy", valid, allow_pickle=False)
    # ---- band-profile traces ----
    band = band_rows or [int(H0 * 0.33), int(H0 * 0.66)]
    r0, r1 = int(band[0]) // b, int(band[1]) // b
    rois = roi_x_ranges or {"left": [int(W0 * 0.25), int(W0 * 0.35)],
                            "mid": [int(W0 * 0.45), int(W0 * 0.55)],
                            "right": [int(W0 * 0.65), int(W0 * 0.75)]}
    roi_cols = {k: (int(v[0]) // b, int(v[1]) // b) for k, v in rois.items()}
    ref = reference_roi or list(rois)[-1]
    bandvalid = valid[r0:r1]
    C = Z * np.conj(FA)[None]
    P = np.array([np.where(bandvalid, C[t][r0:r1], 0).sum(axis=0) / np.maximum(bandvalid.sum(axis=0), 1) for t in range(n)])
    colvalid = None; thr_used = None
    for thr in (0.3, 0.15, 0.05, 0.0):
        colvalid = (bandvalid.mean(axis=0) > 0.8) & (np.abs(P).mean(axis=0) > thr)
        thr_used = thr
        if colvalid.sum() > 0.6 * valid.any(axis=0).sum():
            break
    prof = np.full((n, Wd), np.nan)
    for t in range(n):
        prof[t][colvalid] = np.unwrap(np.angle(P[t][colvalid]))
    # ---- template amplitude ----
    grid = np.asarray(template_amplitude_grid or np.arange(-0.5, 1.5001, 0.01), dtype=float)
    sub = (slice(None, None, 2), slice(None, None, 2)); vs = valid[sub]
    T = np.nan_to_num(dmap, nan=0.0)[sub][vs]
    amp = np.zeros(n); amp_coh = np.zeros(n)
    for t in range(n):
        ph = np.angle(C[t][sub][vs])
        cohs = np.abs(np.exp(1j * (ph[None, :] - grid[:, None] * T[None, :])).mean(axis=1))
        i = int(np.argmax(cohs)); amp[t] = grid[i]; amp_coh[t] = cohs[i]
    # ---- rows ----
    def roi_val(t, k):
        a, c = roi_cols[k]; return float(np.nanmean(prof[t][a:c]))
    rows = []
    for t in range(n):
        r = {k: roi_val(t, k) for k in roi_cols}
        row = {"elapsed_s": float(elapsed[t])}
        for k in roi_cols:
            if k != ref:
                row[f"phase_{k}_minus_{ref}_rad"] = r[k] - r[ref]
        row.update({"template_amplitude_fraction": float(amp[t]), "template_fit_coherence": float(amp_coh[t]),
                    "transition": 1 if states[t] in trans else 0,
                    "frame_index": int(frame_index_map[t]) if frame_index_map else t,
                    "band_profile_coherence_mean": float(np.abs(P[t]).mean())})
        for k in roi_cols:
            a, c = roi_cols[k]; row[f"coherence_{k}"] = float(np.abs(P[t][a:c]).mean())
        for k, v in extras.items():
            row[k] = v[t]
        rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    # numeric state code for curve loaders
    uniq = [s for i, s in enumerate(states) if s not in states[:i]]
    code = {s: (0.5 if s in trans else float(i)) for i, s in enumerate([u for u in uniq if u not in trans])}
    for t, row in enumerate(rows):
        row["state_code"] = code.get(states[t], 0.5)
    csv_path = out / f"{stem}_roi_phase_vs_time.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    def map_roi(k):
        a, c = roi_cols[k]; return float(np.nanmean(dmap[r0:r1, a:c]))
    map_rois = {k: map_roi(k) for k in roi_cols}
    keys = [k for k in roi_cols if k != ref]
    stepA = {k: float(np.mean([r[f"phase_{k}_minus_{ref}_rad"] for r in rows[winA[0]:winA[1]]])) for k in keys}
    stepB = {k: float(np.mean([r[f"phase_{k}_minus_{ref}_rad"] for r in rows[winB[0]:winB[1]]])) for k in keys}
    steps = {k: stepB[k] - stepA[k] for k in keys}
    xcheck = {"steady_map_discontinuity_fraction": map_disc,
              "split_half_circular_rms_rad": float(np.sqrt(np.mean(hres ** 2))),
              "band_column_coherence_threshold_used": thr_used,
              "band_trace_steps_rad": steps,
              "steady_map_roi_differences_rad": {k: map_rois[k] - map_rois[ref] for k in keys}}
    # ---- quicklook ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(16, 9)); tt = [r["elapsed_s"] for r in rows]
        for k in keys: ax[0, 0].plot(tt, [r[f"phase_{k}_minus_{ref}_rad"] for r in rows], label=f"{k} - {ref}")
        for i in range(1, n):
            if states[i] != states[i - 1]: ax[0, 0].axvline(tt[i], color="k", ls="--", lw=0.8)
        ax[0, 0].set_title(f"{stem}: band-profile ROI differences ({first_state} -> {last_state})"); ax[0, 0].legend(fontsize=7); ax[0, 0].set_ylabel("rad")
        ax[0, 1].plot(tt, amp, label="template amplitude"); ax[0, 1].plot(tt, amp_coh, label="template fit coherence"); ax[0, 1].legend(fontsize=7)
        im = ax[1, 0].imshow(dmap, cmap="RdBu_r"); plt.colorbar(im, ax=ax[1, 0]); ax[1, 0].set_title(f"steady map {lab} [rad]")
        for k, (a, c) in roi_cols.items():
            ax[1, 0].add_patch(plt.Rectangle((a, r0), c - a, r1 - r0, fill=False, ec="k", lw=0.8)); ax[1, 0].text(a, r0, k, fontsize=7)
        ax[1, 1].plot(np.arange(Wd) * b, np.nanmean(dmap[r0:r1], axis=0)); ax[1, 1].set_title("band-mean profile of the steady map vs x [rad]")
        plt.tight_layout(); plt.savefig(out / f"{stem}_quicklook.png", dpi=80); plt.close()
        quicklook = str(out / f"{stem}_quicklook.png")
    except Exception:  # noqa: BLE001 - figures are a convenience
        quicklook = None
    # ---- sidecars ----
    limits = ["relative optical phase in radians only; no conversion to concentration or refractive index",
              "global piston fluctuates between frames and is NOT reported; all columns are within-frame spatial differences or shape-matched amplitudes (piston-immune)",
              f"condition is confounded with time within the run (windows: {first_state} frames {winA[0]}-{winA[1]-1} vs {last_state} frames {winB[0]}-{winB[1]-1})"]
    if excluded:
        limits.append(f"columns {excluded} (full-res x) excluded from every product: unresolvable fringe density there")
    common = {"technique": "off-axis interferometry / quantitative phase imaging, time-resolved",
              "source_phase_stack": str(wrapped_phase_npy), "n_frames": n, "conditions": uniq,
              "steady_windows_frames": {"A_first_state": [first_state, list(winA)], "B_last_state": [last_state, list(winB)]},
              "interpretation_limits": limits, "cross_checks": xcheck}
    # Top-level NUMERIC condition fields: the orchestrator's sidecar-series extraction
    # builds a physically meaningful series variable from numeric keys shared by all
    # sidecars (has_transition / last_state_code ...), instead of falling back to a
    # file-order index that says nothing about the conditions.
    trans_frames = [i for i, s_ in enumerate(states) if s_ in trans]
    cond_scalars = {
        "n_frames": int(n), "n_conditions": len([u for u in uniq if u not in trans]),
        "has_transition": 1 if first_state != last_state else 0,
        "first_state_code": float(code.get(first_state, 0.5)), "last_state_code": float(code.get(last_state, 0.5)),
        "transition_start_s": float(elapsed[trans_frames[0]]) if trans_frames else (float(elapsed[first_end + 1]) if first_state != last_state else -1.0),
        "condition_sequence": " -> ".join(u for u in uniq),
    }
    csv_meta = dict(common, **cond_scalars, measurement_type="relative_optical_phase_time_series", data_file=csv_path.name,
        format="CSV, one header row, ALL NUMERIC (state labels are encoded in state_code; see state_code_map)",
        state_code_map={str(v): k for k, v in code.items()} | ({"0.5": "transition (" + ", ".join(sorted(trans)) + ")"} if trans else {}),
        columns={"elapsed_s": "time axis (x)",
                 **{f"phase_{k}_minus_{ref}_rad": f"PRIMARY: band-profile phase (rows {band[0]}:{band[1]}) averaged over x {rois[k][0]}:{rois[k][1]} minus over the reference ROI '{ref}' (x {rois[ref][0]}:{rois[ref][1]}), relative to the first steady state; piston-immune" for k in keys},
                 "template_amplitude_fraction": "PRIMARY (shape-matched): fraction of the run's final steady-state map present in the frame (0 = first state, 1 = final state)",
                 "template_fit_coherence": "goodness of the template fit (1 = frame is exactly a scaled steady map + piston)",
                 "state_code": "numeric condition code (0, 1, ... in order of appearance; 0.5 during transitions)",
                 "transition": "1 while in a transition state", "frame_index": "source frame index",
                 "band_profile_coherence_mean": "mean |row-averaged unit field| across the band (1 = phase flat along y; low = fringes/tilt along y, lower confidence)",
                 "coherence_<roi>": "same within the ROI's x range"},
        primary_columns={"x": "elapsed_s", "y": f"phase_{keys[0]}_minus_{ref}_rad" if keys else "template_amplitude_fraction"},
        steady_state_steps_rad=steps, sibling_diff_map=str(map_path))
    csv_path.with_suffix(".json").write_text(json.dumps(csv_meta, indent=2))
    map_meta = dict(common, **cond_scalars, measurement_type="relative_optical_phase_difference_map", data_file=map_path.name,
        format=f"2D float32 numpy array ({H}, {Wd}) = {b}x{b}-binned, smoothed (sigma {smoothing_sigma}) phase frame; radians; global piston removed (field median = 0); 0 outside the valid mask; NOT an intensity image",
        semantics=f"spatially unwrapped phase of (circular-mean field over the last steady window [{last_state}]) x conj(first steady window [{first_state}])",
        valid_mask=str(out / f"{stem}_valid_mask_x{b}.npy"), valid_fraction=float(valid.mean()),
        roi_band_rows_binned=[r0, r1], roi_x_ranges_binned=roi_cols, steady_map_roi_means_rad=map_rois,
        sibling_time_series=str(csv_path))
    map_path.with_suffix(".json").write_text(json.dumps(map_meta, indent=2))
    return {"status": "success", "diff_map": str(map_path), "diff_map_sidecar": str(map_path.with_suffix(".json")),
            "roi_curve": str(csv_path), "roi_curve_sidecar": str(csv_path.with_suffix(".json")),
            "valid_mask": str(out / f"{stem}_valid_mask_x{b}.npy"), "quicklook": quicklook,
            "conditions": uniq, "steady_windows_frames": common["steady_windows_frames"],
            "steady_state_steps_rad": steps, "steady_map_roi_means_rad": map_rois,
            "cross_checks": xcheck, "seconds": round(time.time() - t0, 1)}


TOOL_SPEC = ToolSpec(
    name="derive_phase_products",
    description=(
        "From a wrapped relative-phase stack (n_frames, y, x) produce piston-immune, "
        "analysis-ready products: a steady-state phase DIFFERENCE MAP (last condition minus "
        "first, spatially unwrapped, global piston removed) for image analysis, and a "
        "TIME-SERIES CSV of band-profile ROI differences plus a template-amplitude trace for "
        "curve analysis, each with a same-stem JSON sidecar. Joins the frames with a "
        "per-frame condition timeline (e.g. magnet state) when given."
    ),
    import_line=("from scilink.skills.data_preparation.mmzi_hologram_reconstruction"
                 ".derive import derive_phase_products"),
    signature=("derive_phase_products(wrapped_phase_npy: str, output_dir: str, timeline_csv: str | None = None, "
               "state_column: str = 'magnet_state', time_column: str = 'capture_elapsed_s', "
               "transition_states: list[str] | None = None, steady_window_frames: int = 25, bin_factor: int = 2, "
               "smoothing_sigma: float = 1.0, coherence_min: float = 0.6, exclude_columns: list[[x0, x1]] | None = None, "
               "band_rows: [y0, y1] | None = None, roi_x_ranges: dict[str, [x0, x1]] | None = None, "
               "reference_roi: str | None = None, template_amplitude_grid: list[float] | None = None, "
               "frame_index_map: list[int] | None = None, stem: str | None = None, label: str | None = None) -> dict"),
    parameters={
        "wrapped_phase_npy": {"type": "str", "description": "Output of reconstruct_offaxis_hologram_stack (float32 (n, y, x) wrapped phase, radians)."},
        "output_dir": {"type": "str", "description": "Directory for the products (outside the source bundle)."},
        "timeline_csv": {"type": "str", "description": "Per-frame condition table (one row per frame, in frame order) with a state column and a time column; extra numeric columns are copied into the trace CSV. Omit for a single-condition run (drift only)."},
        "state_column": {"type": "str", "description": "Column holding the condition label per frame (default 'magnet_state')."},
        "time_column": {"type": "str", "description": "Column holding elapsed seconds (default 'capture_elapsed_s'; falls back to frame index)."},
        "transition_states": {"type": "list[str]", "description": "State labels that mean 'moving between conditions' (excluded from steady windows, flagged transition=1)."},
        "steady_window_frames": {"type": "int", "description": "Frames averaged at the end of the first condition and the end of the last condition (default 25). LOWER if the conditions are short; RAISE for more averaging when the run is long and stable."},
        "bin_factor": {"type": "int", "description": "Spatial binning (circular mean of the unit field; default 2). Fringes must stay >= ~6 binned px apart: LOWER (1) when fringes are dense, RAISE (4) for noisy low-fringe-density data."},
        "smoothing_sigma": {"type": "float", "description": "Gaussian smoothing of the complex field in binned px (default 1.0; 0 disables)."},
        "coherence_min": {"type": "float", "description": "Valid-pixel rule: mean |smoothed unit field| over the run must exceed this (default 0.6). LOWER to keep dim regions; RAISE to drop noisy edges."},
        "exclude_columns": {"type": "list", "description": "Full-resolution x ranges [[x0, x1], ...] to drop from every product (a MEASURED zone, e.g. from a previous run's auto exclusion). An exclusion leaving fewer than half the columns raises."},
        "auto_exclude_dense_fringes": {"type": "bool", "description": "Measure and exclude edge zones where the steady-state fringes are denser than the sampling limit (median |dphi/dx| > dense_fringe_max_gradient_rad_per_px per binned px, contiguous from an edge). Default False; set True for any run with a strong localized perturbation. Excluded ranges are reported in the sidecar."},
        "dense_fringe_max_gradient_rad_per_px": {"type": "float", "description": "Gradient threshold for the auto exclusion (default 1.05 rad/px = fringe spacing ~6 binned px). LOWER to exclude more aggressively, RAISE to keep steeper gradients."},
        "band_rows": {"type": "list", "description": "Full-resolution [y0, y1] row band for the profile traces (default the middle third)."},
        "roi_x_ranges": {"type": "dict", "description": "Named full-resolution x ranges for the traces, e.g. {'near_wall': [350, 500], 'mid': [650, 800], 'far': [950, 1100]} (default thirds). Choose them from the steady map so at least one ROI is far from the perturbation."},
        "reference_roi": {"type": "str", "description": "ROI subtracted from the others (default: the last named ROI). Pick the ROI least affected by the perturbation."},
        "template_amplitude_grid": {"type": "list[float]", "description": "Scale grid for the template fit (default -0.5..1.5 step 0.01)."},
        "frame_index_map": {"type": "list[int]", "description": "Source frame index per stack frame when the stack is a subset."},
        "stem": {"type": "str", "description": "Output file stem (default derived from the input name)."},
        "label": {"type": "str", "description": "Label in the difference-map file name (default '<last>_minus_<first>')."},
    },
    required=["wrapped_phase_npy", "output_dir"],
    returns=("dict: diff_map (.npy) + diff_map_sidecar, roi_curve (.csv) + roi_curve_sidecar, valid_mask, "
             "quicklook (.png), conditions, steady_windows_frames, steady_state_steps_rad (per ROI minus "
             "reference), steady_map_roi_means_rad, cross_checks (unwrap discontinuity fraction, split-half "
             "rms, band-trace vs map agreement)."),
    when_to_use=("Right after reconstruct_offaxis_hologram_stack, to turn the phase stack into "
                 "a map for image analysis and a CSV for curve analysis. Inspect the quicklook and "
                 "the cross_checks; if the steady map shows sharp-edged plateaus (unwrap residues), "
                 "lower bin_factor or exclude the dense-fringe columns and rerun."),
    agents=["data_preparation"],
)
