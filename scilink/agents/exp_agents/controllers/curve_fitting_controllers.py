# controllers/curve_fitting_controllers.py

"""
Curve Fitting Controllers - Complete Module

This module contains:
1. Original controllers for single-spectrum analysis steps
2. Unified controllers that handle both single spectrum (n=1) and series (n>1) analysis

Key principle for series analysis: Single spectrum = Series of 1

Quality control features:
- Automatic model retry when R² is inadequate
- Statistical outlier detection for series
- Human feedback integration for unresolved quality issues
"""

# Set non-interactive backend BEFORE importing pyplot anywhere
import matplotlib
matplotlib.use('Agg')

import subprocess
import json
import logging
import os
import base64
import copy
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List
import numpy as np

from .._locked_exec import (
    stage_and_run, script_uses_canonical_input, DATA_NAME, CANDIDATES_DIR_NAME,
    atomic_np_save,
)
from ....utils.codegen_parse import parse_codegen_response
from ....utils.synthesis_parse import salvage_synthesis_from_response

# Canonical fitted-curve output the fit script saves alongside visualization.png.
# Best-effort: when present it powers controller-side residual diagnostics; when
# absent (older/refit scripts that didn't save it) diagnostics are simply skipped.
FIT_NAME = "fit.npy"


def _robust_noise_sigma(residual: np.ndarray) -> float:
    """Per-point noise sigma from successive differences — robust to systematic
    structure in the residual (a trend/oscillation barely affects neighbour diffs)."""
    d = np.diff(residual)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = 1.4826 * mad / np.sqrt(2.0)          # MAD of diffs -> per-point sigma
    if not sigma or not np.isfinite(sigma):
        sigma = float(np.std(d)) / np.sqrt(2.0)
    return float(sigma)


def _residual_diagnostics(x, y, fit, n_windows: int = 16):
    """Structure-aware residual diagnostics from data and the fitted curve.

    Pure NumPy on the saved fit array — reliable, unlike asking the vision model
    to read small systematic residuals off a dynamic-range-crushed plot. Returns a
    dict of global metrics plus the most systematic windows, or ``None`` if the
    inputs can't be aligned (so the caller degrades to no-diagnostics gracefully).
    """
    try:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        fit = np.asarray(fit, float)
        if fit.ndim == 2:                        # accept [N,2] -> take y column
            fit = fit[:, -1]
        fit = fit.ravel()
        if fit.shape[0] != y.shape[0] or x.shape[0] != y.shape[0]:
            return None
        resid = y - fit
        m = np.isfinite(resid) & np.isfinite(x)
        if int(m.sum()) < 8:
            return None
        x, resid = x[m], resid[m]
        order = np.argsort(x)
        x, resid = x[order], resid[order]

        sigma = _robust_noise_sigma(resid) or (float(np.std(resid)) or 1.0)
        nresid = resid / sigma
        rms = float(np.sqrt(np.mean(resid ** 2)))
        r0 = resid - resid.mean()
        denom = float(np.sum(r0 * r0)) or 1.0
        autocorr1 = float(np.sum(r0[:-1] * r0[1:]) / denom)
        frac_gt3 = float(np.mean(np.abs(nresid) > 3.0))

        edges = np.linspace(x.min(), x.max(), n_windows + 1)
        windows = []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            wm = (x >= a) & (x <= b) if i == n_windows - 1 else (x >= a) & (x < b)
            if int(wm.sum()) < 4:
                continue
            wr = resid[wm]
            wn = nresid[wm]
            # Sign-changes of the BIN-AVERAGED residual, not the raw points:
            # averaging cancels point noise so this counts *systematic*
            # oscillation (a single bump -> 0, a peak shift -> 1, multiple
            # unresolved peaks -> several), not noise crossings.
            nb = int(min(10, max(3, wr.size // 8)))
            cuts = np.linspace(0, wr.size, nb + 1).astype(int)
            bmeans = np.array([wr[cuts[k]:cuts[k + 1]].mean()
                               for k in range(nb) if cuts[k + 1] > cuts[k]])
            bsigns = np.sign(bmeans)
            bsigns = bsigns[bsigns != 0]
            sign_changes = int(np.sum(bsigns[:-1] != bsigns[1:])) if bsigns.size > 1 else 0
            j = int(np.argmax(np.abs(wn)))
            windows.append({
                "x_lo": float(a), "x_hi": float(b),
                "rms_over_noise": float(np.sqrt(np.mean(wr ** 2)) / sigma),
                "max_abs_norm": float(np.abs(wn[j])),
                "x_at_max": float(x[wm][j]),
                "sign_changes": sign_changes,
            })
        windows.sort(key=lambda w: w["rms_over_noise"], reverse=True)
        return {
            "noise_sigma": sigma,
            "global_rms": rms,
            "global_rms_over_noise": float(rms / sigma),
            "autocorr_lag1": autocorr1,
            "frac_points_gt_3sigma": frac_gt3,
            "worst_windows": windows[:5],
        }
    except Exception:
        return None


def _canonical_r2(y, fit):
    """R² of the *saved* fitted curve vs the data, over the finite,
    length-matched points (same alignment guards as ``_residual_diagnostics``).

    This is computed from the canonical ``data.npy`` / ``fit.npy`` arrays — the
    exact curve that is plotted and shown to the verifier — so it can't diverge
    from the displayed fit the way a script's self-reported R² can. Returns
    ``None`` when the arrays can't be aligned (length mismatch, e.g. a partial
    fit saved as a short array) or there is too little finite signal, so the
    caller keeps the self-reported value. Callers use it to raise a
    broken-low self-report (``max(self, recompute)``) — never to lower a
    deliberate windowed/partial fit's number.
    """
    try:
        y = np.asarray(y, float).ravel()
        fit = np.asarray(fit, float)
        if fit.ndim == 2:                       # accept [N,2] -> take y column
            fit = fit[:, -1]
        fit = fit.ravel()
        if fit.shape[0] != y.shape[0]:
            return None
        m = np.isfinite(y) & np.isfinite(fit)
        if int(m.sum()) < 8:
            return None
        yy, ff = y[m], fit[m]
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        if ss_tot <= 0:
            return None
        ss_res = float(np.sum((yy - ff) ** 2))
        return 1.0 - ss_res / ss_tot
    except Exception:
        return None


def _format_residual_diagnostics(diag) -> str:
    """Compact text block of residual diagnostics for the verifier prompt — gives
    the LLM numbers to reason over instead of eyeballing a compressed plot."""
    if not diag:
        return ""
    lines = [
        "\n**RESIDUAL DIAGNOSTICS (computed from data − fit; use to locate "
        "systematic structure the plot's dynamic range may hide):**",
        f"- Noise σ (successive-difference estimate): {diag['noise_sigma']:.3g}",
        f"- Global residual RMS: {diag['global_rms']:.3g} "
        f"({diag['global_rms_over_noise']:.1f}× noise)",
        f"- Lag-1 autocorrelation: {diag['autocorr_lag1']:.2f} "
        f"(≳ 0.3 ⇒ systematic, not white noise)",
        f"- Points beyond 3σ: {diag['frac_points_gt_3sigma'] * 100:.1f}%",
    ]
    flagged = [w for w in (diag.get("worst_windows") or []) if w["rms_over_noise"] >= 1.5]
    if flagged:
        lines.append("- Most systematic regions (RMS/noise · peak |resid|/σ · sign-changes):")
        for w in flagged:
            lines.append(
                f"    • {w['x_lo']:.1f}–{w['x_hi']:.1f}: "
                f"{w['rms_over_noise']:.1f}× · "
                f"{w['max_abs_norm']:.0f}σ at x≈{w['x_at_max']:.1f} · "
                f"{w['sign_changes']} sign-changes"
            )
    return "\n".join(lines)


def _render_region_zoom_panels(x, y, fit, diag, max_panels: int = 3,
                               rms_floor: float = 1.5, pad_frac: float = 0.15):
    """Zoomed, locally-rescaled views of the most systematic residual regions.

    The numeric residual diagnostics tell the verifier *where* the misfit is; the
    full-range plot squashes the corresponding fine structure under a tall peak so
    the verifier can't *see* what's missing. For each flagged window (already
    sorted by severity) this crops the data + fit + residual to that x-range and
    rescales the y-axis to the local data, so an unmodeled maximum/shoulder
    becomes visible. The x-axis is the TRUE data axis and the title states the
    real x-range, so the verifier can reference/seed components at correct
    positions. Returns ``[(label, png_bytes), ...]`` (empty if nothing systematic
    or inputs can't be rendered — the caller degrades gracefully).
    """
    if not diag or not diag.get("worst_windows"):
        return []
    try:
        from io import BytesIO
        from matplotlib import pyplot as plt
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        fit = np.asarray(fit, float)
        if fit.ndim == 2:
            fit = fit[:, -1]
        fit = fit.ravel()
        if not (x.shape[0] == y.shape[0] == fit.shape[0]):
            return []
        order = np.argsort(x)
        x, y, fit = x[order], y[order], fit[order]
        panels = []
        for w in diag["worst_windows"]:
            if len(panels) >= max_panels:
                break
            if float(w.get("rms_over_noise", 0.0)) < rms_floor:
                continue
            lo, hi = float(w["x_lo"]), float(w["x_hi"])
            pad = (hi - lo) * pad_frac
            mask = (x >= lo - pad) & (x <= hi + pad)
            if int(mask.sum()) < 4:
                continue
            xs, ys, fs = x[mask], y[mask], fit[mask]
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(6, 4.2), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(xs, ys, "o", ms=3, color="#1f77b4", label="Data")
            ax1.plot(xs, fs, "-", lw=1.8, color="#d62728", label="Fit")
            ax1.legend(loc="best", fontsize=8)
            ax1.set_ylabel("Intensity")
            ax1.set_title(
                f"Region {lo:.1f}–{hi:.1f} (true x axis)  |  "
                f"RMS/noise={float(w.get('rms_over_noise', 0.0)):.1f}, "
                f"{int(w.get('sign_changes', 0))} sign-changes",
                fontsize=9)
            ax2.plot(xs, ys - fs, "-", lw=1.0, color="#555555")
            ax2.axhline(0, color="k", lw=0.7)
            ax2.set_ylabel("Residual")
            ax2.set_xlabel("x (data axis)")
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            panels.append((f"Region {lo:.1f}–{hi:.1f}", buf.getvalue()))
        return panels
    except Exception:
        return []


def _extract_xy(curve_data):
    """(x, y) from a 1-D curve array, mirroring AnalyzeDataController's
    heuristic: 1-D -> (index, data); [2,N] -> rows; [N,2] -> columns. Returns
    None if the shape isn't a recognizable single curve."""
    try:
        d = np.asarray(curve_data, float)
        if d.ndim == 1:
            return np.arange(d.size, dtype=float), d
        if d.ndim == 2 and d.shape[0] == 2:
            return d[0], d[1]
        if d.ndim == 2 and d.shape[1] == 2:
            return d[:, 0], d[:, 1]
    except Exception:
        pass
    return None


_STRUCTURE_RMS_FLOOR = 2.5  # window structure must be this many × noise to count


def _score_scale(x, y, sigma, global_span, n_windows):
    """Score sliding windows at ONE scale. Window centers step by width/4 so a
    window lands ON each feature regardless of grid alignment (a fixed grid
    splits a narrow feature across a boundary and misses it). Returns the
    windows whose structure clears the noise floor."""
    width = (x.max() - x.min()) / n_windows
    if width <= 0:
        return []
    half = width / 2.0
    out = []
    for c in np.arange(x.min() + half, x.max() - half + 1e-9, width / 4.0):
        lo, hi = c - half, c + half
        wm = (x >= lo) & (x <= hi)
        if int(wm.sum()) < 6:
            continue
        xw, yw = x[wm], y[wm]
        xc = xw - xw.mean()
        try:                                    # local LINE = the resolvable trend
            dr = yw - np.polyval(np.polyfit(xc, yw, 1), xc)
        except Exception:
            dr = yw - yw.mean()
        rms_over_noise = float(np.sqrt(np.mean(dr ** 2)) / sigma)
        if rms_over_noise < _STRUCTURE_RMS_FLOOR:    # no structure above the noise
            continue
        # Sign-changes of the BIN-AVERAGED detrend residual: systematic local
        # structure (a shoulder -> 1, an unresolved doublet -> several), not noise.
        nb = int(min(10, max(3, dr.size // 8)))
        cuts = np.linspace(0, dr.size, nb + 1).astype(int)
        bmeans = np.array([dr[cuts[k]:cuts[k + 1]].mean()
                           for k in range(nb) if cuts[k + 1] > cuts[k]])
        bsigns = np.sign(bmeans)
        bsigns = bsigns[bsigns != 0]
        sign_changes = int(np.sum(bsigns[:-1] != bsigns[1:])) if bsigns.size > 1 else 0
        local_span = float(np.ptp(yw)) or sigma
        compression = max(1.0, global_span / local_span)
        j = int(np.argmax(np.abs(dr)))
        out.append({
            "x_lo": float(lo), "x_hi": float(hi), "x_mid": float(c),
            "rms_over_noise": rms_over_noise,
            "max_abs_norm": float(np.abs(dr[j]) / sigma),
            "x_at_max": float(xw[j]),
            "sign_changes": sign_changes,
            "compression": compression,
            # structured AND squashed sorts to the top. Compression is weighted
            # LINEARLY so a small feature hidden under a tall one — the actually-
            # hard-to-resolve case — outranks the tall feature's own (already-
            # visible) flanks. Safe because the rms floor excludes squashed noise.
            "_score": rms_over_noise * compression,
        })
    return out


def _data_structure_diagnostics(x, y, scales=(8, 16, 32)):
    """Fit-FREE "where is the hard-to-resolve structure?" diagnostics for the
    PLANNING stage. The planner sees only the full-range plot, which squashes
    fine structure under the dominant features. ANALYSIS-AGNOSTIC — no assumed
    model: per window it removes a local LINE (the part a glance already
    resolves) and scores the leftover structure, so it flags bumps / shoulders /
    unresolved doublets for spectra AND fine structure on edges / steep knees
    for decays, steps and monotonic curves.

    MULTI-SCALE: runs several window scales and merges (cross-scale non-max
    suppression), so it adapts to ANY feature width — narrow shoulders and broad
    humps alike — with no single ``n_windows`` to tune. Precision-first: a window
    must clear the noise floor to count, so featureless / noisy data flags
    nothing (a clean no-op). Returns ``None`` on unusable input.
    """
    try:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 16:
            return None
        x, y = x[m], y[m]
        order = np.argsort(x)
        x, y = x[order], y[order]
        sigma = _robust_noise_sigma(y) or (float(np.std(y)) or 1.0)
        global_span = float(np.ptp(y)) or 1.0
        cand = []
        for nw in scales:
            cand.extend(_score_scale(x, y, sigma, global_span, nw))
        # Cross-scale non-max suppression: highest score first; drop any window
        # that overlaps an already-kept one (its center inside the kept range or
        # vice-versa) so the result is up to 5 DISTINCT regions, each at the
        # scale that best resolved it.
        cand.sort(key=lambda w: w["_score"], reverse=True)
        kept = []
        for w in cand:
            if not any((k["x_lo"] <= w["x_mid"] <= k["x_hi"])
                       or (w["x_lo"] <= k["x_mid"] <= w["x_hi"]) for k in kept):
                kept.append(w)
            if len(kept) >= 5:
                break
        return {"noise_sigma": sigma, "global_span": global_span,
                "worst_windows": kept}
    except Exception:
        return None


def _render_data_zoom_panels(x, y, diag, max_panels: int = 3, pad_frac: float = 0.15):
    """Zoomed, locally-rescaled DATA views of the most structured-but-squashed
    regions, for the PLANNING stage (no fit exists yet). Returns
    ``[(label, png_bytes), ...]`` (empty when nothing systematic / unrenderable
    — caller degrades gracefully)."""
    if not diag or not diag.get("worst_windows"):
        return []
    try:
        from io import BytesIO
        from matplotlib import pyplot as plt
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        if x.shape[0] != y.shape[0]:
            return []
        order = np.argsort(x)
        x, y = x[order], y[order]
        panels = []
        for w in diag["worst_windows"]:
            if len(panels) >= max_panels:
                break
            lo, hi = float(w["x_lo"]), float(w["x_hi"])
            pad = (hi - lo) * pad_frac
            mask = (x >= lo - pad) & (x <= hi + pad)
            if int(mask.sum()) < 4:
                continue
            xs, ys = x[mask], y[mask]
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(xs, ys, "-o", ms=3, lw=1.0, color="#1f77b4")
            ax.set_title(
                f"Region {lo:.1f}–{hi:.1f} (true x axis)  |  local structure "
                f"{float(w.get('rms_over_noise', 0.0)):.1f}×noise, "
                f"{int(w.get('sign_changes', 0))} sign-changes",
                fontsize=9)
            ax.set_xlabel("x (data axis)")
            ax.set_ylabel("Intensity (local scale)")
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            panels.append((f"Region {lo:.1f}–{hi:.1f}", buf.getvalue()))
        return panels
    except Exception:
        return []


def _append_structure_zoom(prompt: list, state: dict) -> bool:
    """Append fit-free 'hard-to-resolve region' zoom panels to a planning /
    validation prompt, so the planner sees fine structure the full-range plot
    squashes. No-op (returns False) for multi-spectrum stacks or unstructured
    data. Analysis-agnostic — works for spectra, decays, edges, steps."""
    try:
        xy = _extract_xy(state.get("curve_data"))
        if xy is None:
            return False
        panels = _render_data_zoom_panels(
            xy[0], xy[1], _data_structure_diagnostics(xy[0], xy[1]))
        if not panels:
            return False
        prompt.append(
            "\n## Candidate hard-to-resolve regions (zoomed, fit-free — ADVISORY)\n"
            "The full-range plot above is the PRIMARY evidence. The windows below "
            "are candidate regions where a smooth local trend leaves leftover "
            "structure — cropped to their true x-range and y-rescaled to the LOCAL "
            "data so squashed detail becomes visible. They are detected "
            "GEOMETRICALLY (NO assumed model), so they apply whatever the analysis "
            "type: a shoulder may need an extra component, a knee an extra decay "
            "term, a split edge two features.\n"
            "Treat them as HINTS, not findings: VERIFY each against the full plot "
            "and the noise level before acting on it, and IGNORE any that look "
            "like noise or are already clearly resolved in the full view. They "
            "never override the full plot — at worst they are redundant. Use the "
            "ones that hold up to choose the model/approach and seed feature "
            "positions; absence of a flagged window does not mean absence of "
            "structure. If a domain skill is loaded, cross-reference these regions "
            "against the features that technique expects."
        )
        for label, png in panels:
            prompt.append(f"\n**{label}:**")
            prompt.append({"mime_type": "image/png", "data": png})
        return True
    except Exception:
        return False


def _active_skill_names(state: dict) -> list[str]:
    """Return names of all currently-loaded skills from a pipeline state dict.

    Mirrors the image_analysis helper of the same name. Falls back to the
    legacy singular ``skill_name`` field when ``skills_loaded`` is absent.
    """
    loaded = state.get("skills_loaded")
    if loaded:
        return [s.get("name") for s in loaded if s and s.get("name")]
    legacy = state.get("skill_name")
    return [legacy] if legacy else []


def _gate(state: dict):
    """Return the effective QualityGate for this analysis.

    The agent stashes the resolved gate at ``state['quality_gate']`` in
    ``CurveFittingAgent.analyze``. When absent (e.g. legacy callers
    constructing a controller directly), falls back to the framework
    default — R² ≥ 0.95 — so existing behavior is unchanged.
    """
    from ..quality_gate import R_SQUARED_DEFAULT
    g = state.get("quality_gate")
    if g is None:
        return R_SQUARED_DEFAULT
    return g


def _safe_r2(result_or_quality: dict, default: float = 0.0) -> float:
    """Extract r_squared from a fit_result (or fit_quality dict), defaulting
    to ``default`` when the key is missing OR present with value None.

    Workflow-style skills (xrd structure-matching, future Raman / EELS
    libraries with FOM-based scoring) emit FIT_RESULTS_JSON without a
    meaningful r_squared — the natural emitted value is ``null`` or
    omitted. ``.get('r_squared', 0)`` returns the default only when the
    key is missing; if the key is present with a None value it returns
    None, which then crashes downstream arithmetic / comparison.

    Accepts either a full fit result (with a ``fit_quality`` sub-dict)
    or a fit_quality dict directly.
    """
    if not isinstance(result_or_quality, dict):
        return float(default)
    fq = result_or_quality.get("fit_quality", result_or_quality)
    if not isinstance(fq, dict):
        return float(default)
    val = fq.get("r_squared")
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _tool_inventory_text(state: dict) -> str:
    """Render the curve-fitting tool inventory for the active skills.

    Returns an empty string when no skill is active and no shared tools
    target the curve_fitting agent — avoids polluting prompts with an
    empty "Available Tools" header.
    """
    from ...skills._shared._registry import format_tool_inventory

    return format_tool_inventory(
        "curve_fitting", active_skills=_active_skill_names(state),
    )


def _parse_script_markers(stdout: Optional[str]) -> dict:
    """Parse FIT_RESULTS_JSON and DB_MATCHES_JSON markers from script stdout.

    The first parseable ``FIT_RESULTS_JSON:`` line wins. Once that marker
    has been seen (even with malformed JSON) later instances are ignored —
    matches the long-standing first-wins behavior.

    ``DB_MATCHES_JSON:`` is emitted by ``search_structures`` in the
    structure_matching skill. When present, the parsed payload is merged
    in at ``fit_results['db_matches']`` so the synthesis stage and HTML
    report can surface candidates without per-script glue code.
    """
    fit_results: dict = {}
    fit_seen = False
    db_matches: Optional[dict] = None
    for line in (stdout or "").splitlines():
        if line.startswith("FIT_RESULTS_JSON:") and not fit_seen:
            fit_seen = True
            try:
                fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
            except json.JSONDecodeError:
                pass
        elif line.startswith("DB_MATCHES_JSON:") and db_matches is None:
            try:
                db_matches = json.loads(line.replace("DB_MATCHES_JSON:", "").strip())
            except json.JSONDecodeError:
                pass
    if db_matches is not None:
        fit_results.setdefault("db_matches", db_matches)
    return fit_results


def _resolve_parallel_workers(value: Optional[int]) -> int:
    """Resolve the effective non-anchor worker count.

    Precedence: explicit constructor value (when not None) > env var
    ``SCILINK_CURVE_FIT_WORKERS`` > 1. Values <1 are clamped to 1.
    """
    if value is None:
        env = os.environ.get("SCILINK_CURVE_FIT_WORKERS")
        if env:
            try:
                value = int(env)
            except ValueError:
                value = 1
        else:
            value = 1
    return max(int(value), 1)


def build_verification_prompt_with_history(
    current_fit: dict,
    previous_iterations: List[dict],
) -> str:
    """Build history context string for verification prompt."""
    if not previous_iterations:
        return ""
    
    lines = [
        "\n\n## PREVIOUS VERIFICATION ATTEMPTS",
        "Review what was tried before. Don't suggest fixes that already failed.\n"
    ]
    
    for i, prev in enumerate(previous_iterations, 1):
        lines.append(f"\n### Attempt {i}")
        label = prev.get('metric_label', 'R²')
        mv = prev.get('metric_value', prev.get('r_squared'))
        bm = prev.get('best_metric_value', prev.get('best_so_far'))
        parts = [f"{label} = {mv:.4f}" if mv is not None else f"{label} = N/A"]
        if bm is not None:
            parts.append(f"best-so-far = {bm:.4f}")
        lines.append("- " + " | ".join(parts))
        lines.append(f"- Config: {prev.get('config_used', {}).get('physical_model', 'N/A')}")
        lines.append(f"- Assessment: {prev.get('overall_assessment', 'N/A')}")
        
        issues = prev.get('issues_found', [])
        if issues:
            lines.append(f"- Issues ({len(issues)}):")
            for issue in issues:
                lines.append(f"  • {issue.get('location', '?')}: {issue.get('problem', '?')}")
        
        if prev.get('recommended_action'):
            lines.append(f"- Action taken: {prev['recommended_action']}")

        if prev.get('refinement_error'):
            lines.append(
                f"- **NOTE: The recommended fix was NOT applied** because "
                f"the refinement LLM call failed ({prev['refinement_error']}). "
                f"The results below are UNCHANGED from this attempt — "
                f"do not penalize for identical output. Re-evaluate the "
                f"recommended action and suggest concrete fixes."
            )

    lines.extend([
        "\n\n## IMPORTANT",
        "1. Check if previous issues were RESOLVED or still PERSIST",
        "2. If a fix didn't work AND the best metric is still below the accept "
        "threshold, suggest something DIFFERENT. But if the best is already above "
        "the accept threshold and has not improved for the last 2 iterations, do "
        "NOT propose another change — accept and record any remaining concern as a "
        "caveat (the plateau/convergence rule takes precedence).",
        "3. If a previous fix was NOT applied due to an API error, "
        "re-suggest it or propose an alternative",
        "4. A previously-raised issue may have been MISTAKEN. RETRACT it (drop it; "
        "stop demanding fixes) when STRONG evidence shows the concern was unfounded "
        "— the plot, a registered tool's documented behaviour/guarantees, clear "
        "physics, or an independent cross-check. Absent strong evidence, keep "
        "scrutinizing: 'persists' means still demonstrably real, not merely "
        "un-disproven.",
    ])

    return "\n".join(lines)


def _append_deviation_note(prompt: list, fit_results: dict) -> None:
    """Append the fitting-stage deviation_note if non-empty.

    The note is labeled clearly as process notes (NOT findings) so the
    interpretation LLM does not treat it as a pre-drawn conclusion. Reads
    the new `deviation_note` field with a fallback to the legacy `summary`
    field for any in-flight states; the fallback can be removed after one
    release.
    """
    raw = fit_results.get("deviation_note")
    if raw is None:
        raw = fit_results.get("summary")  # back-compat fallback
    if not raw:
        return
    note = str(raw).strip()
    if not note:
        return
    prompt.append(
        "\n## Fitting-stage process notes (not findings)\n"
        "The fitter recorded the following note about deviations from the plan "
        "or unusual adjustments during the fit. This is process context, not a "
        "scientific finding — do not treat it as a conclusion.\n"
        f"{note}"
    )


def _sanitize_aux_name(label: str, idx: int) -> str:
    """Filesystem-safe stem for a per-auxiliary temp file."""
    safe = re.sub(r'[^0-9A-Za-z_-]', '_', str(label)).strip('_')
    return safe or f"aux{idx}"


def _auxiliary_display_items(state: dict) -> list:
    """Auxiliary datasets to show the LLM as context — items with a rendered
    plot, from the multi-aux ``auxiliary_items`` list. (#226)"""
    return [it for it in (state.get("auxiliary_items") or []) if it.get("plot_bytes")]


def _append_auxiliary_context(prompt: list, state: dict) -> None:
    """Append auxiliary reference dataset(s) to an LLM prompt if available."""
    items = _auxiliary_display_items(state)
    if not items:
        return
    prompt.append("\n## Auxiliary Reference Data")
    prompt.append(
        "The user provided the following auxiliary reference dataset(s). Take "
        "them into account in your analysis and interpretation, but do NOT fit "
        "or quantitatively analyze the auxiliary data as if it were a measurement."
    )
    for it in items:
        prompt.append(f"\n### {it.get('label', 'Auxiliary data')}")
        if it.get("summary"):
            prompt.append(f"Data summary: {it['summary']}")
        prompt.append({
            "mime_type": it.get("mime_type", "image/png"),
            "data": it["plot_bytes"],
        })


def _append_column_structure(prompt: list, state: dict) -> None:
    """Surface a >2-column file's structure so the planner can choose X/Y and
    decide how to treat extra columns. No-op for ordinary <=2-column data."""
    info = state.get("column_info")
    if not info:
        return
    lines = [
        f"\n## Column Structure",
        f"This data file has {info['n_columns']} columns "
        f"({'named' if info.get('names_known') else 'unnamed — referenced by index'}). "
        "Decide which column is X and which is Y to fit, and note the role of the rest.",
    ]
    for c in info.get("per_column", []):
        rng = (f"[{c['min']:.6g}, {c['max']:.6g}]"
               if c.get("min") is not None else "(non-numeric)")
        mono = ", monotonic" if c.get("monotonic") else ""
        lines.append(f"- index {c['index']} \"{c['name']}\": range {rng}{mono}")
    preview = info.get("preview_rows")
    if preview:
        lines.append("First rows: " + json.dumps(preview))
    prompt.append("\n".join(lines))


def _resolve_column_mapping(state: dict):
    """Resolve the LLM's column_mapping against column_info into concrete indices.

    Returns ``{x_index, y_index, names, note, extras}`` or None — None means fall
    back to the deterministic heuristic (unresolvable / missing / x==y)."""
    info = state.get("column_info")
    cm = state.get("column_mapping")
    if not info or not isinstance(cm, dict):
        return None
    names = info.get("names") or []
    n = info["n_columns"]

    def resolve(ref):
        if isinstance(ref, bool):
            return None
        if isinstance(ref, int) and 0 <= ref < n:
            return ref
        if isinstance(ref, str):
            low = ref.strip().lower()
            for i, c in enumerate(names):
                if str(c).strip().lower() == low:
                    return i
            if low.isdigit() and 0 <= int(low) < n:
                return int(low)
        return None

    xi, yi = resolve(cm.get("x")), resolve(cm.get("y"))
    if xi is None or yi is None or xi == yi:
        return None
    # Resolve usable extra columns (skip role=ignore, unresolvable, or x/y dups)
    # to concrete indices so the fit can stage them as per-spectrum operands.
    extras_resolved = []
    for e in (cm.get("extras") or []):
        if not isinstance(e, dict):
            continue
        if str(e.get("role", "")).strip().lower() == "ignore":
            continue
        ei = resolve(e.get("ref"))
        if ei is None or ei in (xi, yi):
            continue
        extras_resolved.append({
            "index": ei,
            "name": names[ei] if ei < len(names) else f"col_{ei}",
            "role": str(e.get("role", "")),
            "use": str(e.get("use", "")),
        })
    return {"x_index": xi, "y_index": yi, "names": names,
            "note": state.get("column_mapping_note", ""),
            "extras": cm.get("extras", []),
            "extras_resolved": extras_resolved}


def _operand_filename(name: str) -> str:
    """Canonical per-spectrum operand filename for an extra column."""
    safe = re.sub(r"[^0-9A-Za-z_-]", "_", str(name)).strip("_") or "operand"
    return f"{safe}.npy"


def _append_fit_domain_guidance(prompt: list, state: dict) -> None:
    """Surface a custom processing instruction to the planner as fit-domain
    guidance.

    A "fit only the decay / this range" or "ignore the background" request is a
    fit-domain decision (a fit window + a background parameter), not data
    preprocessing — preprocessing stays length-preserving so the raw data is
    fit. The instruction is otherwise only buried in the metadata JSON dump.
    """
    instruction = (state.get("system_info") or {}).get("custom_processing_instruction")
    if not instruction:
        return
    prompt.append(
        "\n## Fit-domain & background guidance\n"
        f"User processing note: {instruction}\n"
        "Express any region-of-interest as the FIT DOMAIN and any "
        "background/baseline as a FIT PARAMETER — not as preprocessing."
    )


def _append_skill_context(prompt: list, state: dict, stage: str) -> None:
    """Append domain skill knowledge to an LLM prompt for the given stage.

    With multiple skills loaded, each skill's section is appended in order
    (most-relevant first) so the LLM can attribute guidance to its source.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``skills_loaded`` (or the legacy
            ``skill_sections`` / ``skill_name`` for single-skill state dicts).
        stage: One of ``"planning"``, ``"analysis"``, ``"interpretation"``, ``"validation"``.
    """
    skills = state.get("skills_loaded") or (
        [state["skill_sections"]] if state.get("skill_sections") else []
    )
    if not skills:
        return

    intro_appended = False
    for sections in skills:
        if not sections:
            continue
        content = sections.get(stage, "")
        if not content:
            continue
        skill_name = sections.get("name", "domain skill")

        prompt.append(f"\n## MANDATORY Domain Skill Rules: {skill_name} ({stage})")
        if not intro_appended:
            prompt.append(
                "The following rules are MANDATORY. Your analysis plan and implementation "
                "MUST conform to these domain-specific requirements. These rules encode "
                "validated domain expertise and take precedence over general-purpose defaults. "
                "Do NOT substitute your own preferences where these rules specify a method, "
                "treatment, or constraint."
            )
            intro_appended = True
        prompt.append(content)

        # Include validation rules during planning and interpretation
        # so the LLM knows quality criteria upfront
        if stage in ("planning", "interpretation"):
            validation = sections.get("validation", "")
            if validation:
                prompt.append(f"\n## MANDATORY Domain Validation Rules ({skill_name})")
                prompt.append(validation)


def _collect_codegen_recipe(state: dict) -> list:
    """Per-skill codegen recipes for every co-active skill that authored one.

    Returns ``[(skill_name, recipe_text), …]`` in ranked order (most-relevant
    first), preferring each skill's ``implementation`` section over its
    ``analysis`` synonym. When several skills are active each may own a
    different pipeline stage (e.g. preprocessing vs fitting), so all their
    recipes are returned and the generated script applies each to its stage in
    the plan's order — the top-ranked skill is NOT the sole recipe. Falls back
    to the legacy singular ``skill_sections`` field.
    """
    skills = state.get("skills_loaded") or (
        [state["skill_sections"]] if state.get("skill_sections") else []
    )
    recipes = []
    for s in skills:
        if not s:
            continue
        recipe = s.get("implementation") or s.get("analysis")
        if recipe:
            recipes.append((s.get("name", "skill"), recipe))
    return recipes


def _render_codegen_recipe(recipes: list) -> str:
    """Render collected recipes into one codegen block.

    Single skill: the recipe verbatim (unchanged from the pre-multi-skill
    behavior). Multiple: a short composition note plus each recipe labeled by
    skill, so the codegen LLM maps each recipe to its pipeline stage.
    """
    if len(recipes) == 1:
        return recipes[0][1]
    note = (
        " Multiple skills are active; each recipe below may cover a different "
        "stage of the analysis (e.g. preprocessing vs fitting). Apply each to "
        "its stage in the plan's order and produce ONE script.\n\n"
    )
    return note + "\n\n".join(f"### Recipe — {n}\n{r}" for n, r in recipes)


def _append_prior_knowledge_context(prompt: list, state: dict) -> None:
    """Append prior knowledge from reference analyses to an LLM prompt.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``prior_knowledge`` list.
    """
    knowledge = state.get("prior_knowledge", [])
    if not knowledge:
        return
    prompt.append("\n## Prior Knowledge from Reference Analyses")
    prompt.append(
        "The following knowledge was derived from prior reference analyses. "
        "Use it to inform your analysis approach, model selection, and interpretation."
    )
    for entry in knowledge:
        prompt.append(f"\n### {entry.get('focus', 'Reference findings')}")
        prompt.append(entry.get("summary", ""))
        findings = entry.get("key_findings", [])
        if findings:
            prompt.append("\nKey findings:")
            for f in findings:
                prompt.append(f"- {f}")


def _load_prior_curve_fit_state(raw_path):
    """Locate a prior curve-fit run's artifacts for a single path.

    Accepts a directory or a file inside one. Looks for
    ``series_fit_results.json`` (the structured fit record) and a saved
    fitting script under ``scripts/``. Returns ``(anchor_dir, summary,
    script_text, script_label)`` or ``(None, None, None, None)`` on any
    failure — a missing or malformed prior run silently contributes nothing.
    """
    p = Path(raw_path)
    dir_candidates = (
        [p.parent, p.parent.parent] if p.is_file() else [p, p.parent]
    )
    anchor_dir = None
    sfr_path = None
    for cand in dir_candidates:
        candidate = cand / "series_fit_results.json"
        if candidate.is_file():
            anchor_dir = cand
            sfr_path = candidate
            break
    if anchor_dir is None:
        return None, None, None, None
    try:
        data = json.loads(sfr_path.read_text())
    except Exception:  # noqa: BLE001 - a malformed prior run is skipped
        return None, None, None, None

    results = data.get("results") or []
    model_types = sorted({
        r.get("model_type") for r in results
        if isinstance(r, dict) and r.get("model_type")
    })
    summary = {
        "series_variable": (data.get("series_metadata") or {}).get("variable"),
        "total_spectra": data.get("total_spectra"),
        "successful": data.get("successful"),
        "model_types": model_types,
        "locked_config": data.get("locked_config"),
    }

    # Locate a representative fitting script. A single-spectrum run writes
    # `scripts/fitting_script.py`; a series writes one `scripts/<spectrum>.py`
    # per spectrum — all share the locked model, so the first is a
    # representative template.
    script_text = None
    script_label = None
    scripts_dir = anchor_dir / "scripts"
    single = scripts_dir / "fitting_script.py"
    candidate = None
    if single.is_file():
        candidate, script_label = single, single.name
    elif scripts_dir.is_dir():
        py_files = sorted(scripts_dir.glob("*.py"))
        if py_files:
            candidate = py_files[0]
            script_label = f"{candidate.name} (representative of the series)"
    if candidate is not None:
        try:
            script_text = candidate.read_text()
        except Exception:  # noqa: BLE001
            script_text = None
            script_label = None

    return anchor_dir, summary, script_text, script_label


def _prior_curve_fit_block(state: dict) -> str:
    """A reference-context block for prior curve-fit runs named in
    ``state['prior_analysis_paths']`` — a compact fit summary plus each
    run's saved fitting script.

    Returns an empty string when no prior paths are given, so callers can
    append it unconditionally without affecting a normal (no-prior) run.
    """
    paths = state.get("prior_analysis_paths") or []
    if not paths:
        return ""
    blocks = []
    for raw_path in paths:
        anchor_dir, summary, script_text, script_label = (
            _load_prior_curve_fit_state(raw_path)
        )
        if anchor_dir is None:
            continue
        lines = [f"\n### Prior run: {anchor_dir.name or anchor_dir}"]
        if summary:
            lines.append(f"- Fit summary: {json.dumps(summary, default=str)}")
        if script_text:
            lines.append(f"- Saved fitting script ({script_label}):")
            lines.append(f"```python\n{script_text}\n```")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return (
        "\n## Prior Curve-Fit Runs\n"
        "Artifacts from earlier curve-fit analyses, provided as reference. "
        "Decide for yourself how to use them given the goal: reuse the saved "
        "script as-is to extend/reproduce a consistent fit, adapt it if the "
        "model needs adjusting, or write a fresh script. If the goal is to "
        "VERIFY or re-examine a prior result, derive the fit independently "
        "rather than re-running the prior script — treat the prior numbers as a "
        "hypothesis to test, since re-running the script that produced them "
        "only reproduces them.\n"
        + "\n".join(blocks)
    )


def _first_prior_curve_fit_script(state: dict):
    """Return the first reusable fitting script for locked-script reuse (#172).

    Scans ``state['prior_analysis_paths']`` and returns
    ``(script_text, source_label)`` for the first prior curve-fit run that
    carries a saved fitting script, or ``(None, None)`` when no prior paths
    are given or none have a script. The empty-case gate keeps a normal
    (no-prior) run byte-identical.
    """
    paths = state.get("prior_analysis_paths") or []
    for raw_path in paths:
        anchor_dir, _summary, script_text, _label = (
            _load_prior_curve_fit_state(raw_path)
        )
        if anchor_dir is not None and script_text:
            return script_text, (anchor_dir.name or str(anchor_dir))
    return None, None


def _append_objective_context(prompt: list, state: dict) -> None:
    """Append high-level scientific objective to an LLM prompt.

    The objective is injected as a top-level framing directive that tells the
    LLM *why* the analysis is being performed and *what question* to answer.
    It is distinct from ``analysis_hints`` which provide tactical guidance on
    *how* to analyze.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``analysis_objective``.
    """
    objective = state.get("analysis_objective")
    if not objective:
        return
    prompt.append(
        f"\n## Analysis Objective\n"
        f"The overarching scientific objective of this analysis is: {objective}\n"
        f"Frame your analysis, model selection, and interpretation around "
        f"answering this objective. All findings should be evaluated in terms "
        f"of how they contribute to resolving this question."
    )


class AnalyzeDataController:
    """Compute data statistics and create initial visualization."""

    def __init__(self, logger: logging.Logger, plot_fn: Callable):
        self.logger = logger
        self.plot_fn = plot_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🔍 --- Analyzing Data ---\n")

        try:
            data = state["curve_data"]

            if data.ndim == 1:
                x = np.arange(len(data))
                y = data
            elif data.shape[0] == 2:
                x, y = data[0], data[1]
            elif data.shape[1] == 2:
                x, y = data[:, 0], data[:, 1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            state["data_statistics"] = {
                "n_points": len(x),
                "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
                "y_mean": float(np.nanmean(y)),
                "y_std": float(np.nanstd(y)),
                "has_nans": bool(np.any(np.isnan(data))),
            }

            plot_bytes = self.plot_fn(state["curve_data"], state.get("system_info", {}))
            state["original_plot_bytes"] = plot_bytes
            state["analysis_images"] = [{"label": "Raw Data", "data": plot_bytes}]

            self.logger.info(f"  Points: {state['data_statistics']['n_points']}")
            self.logger.info(f"  X: {state['data_statistics']['x_range']}")
            self.logger.info(f"  Y: {state['data_statistics']['y_range']}")

        except Exception as e:
            self.logger.error(f"❌ Data analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Data analysis failed", "details": str(e)}

        return state


class CurveFittingSkillSuggestionController:
    """Auto-suggest domain skill(s) when none were explicitly provided.

    Runs after data analysis and before planning. Shows the LLM the curve's
    metadata, summary statistics, and the raw-data plot alongside a catalog
    of available curve-fitting skills, and asks which (if any) match the
    measurement technique. No-op when a skill was already loaded (e.g. by the
    orchestrator or user). Selection is conservative and technique-aware
    (see issue #251); it may return zero, one, or several skills.
    """

    def __init__(self, model, logger, generation_config, safety_settings,
                 parse_fn, load_skills_fn, domain="curve_fitting"):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self._load_skills = load_skills_fn
        self.domain = domain

    def execute(self, state: dict) -> dict:
        if (state.get("error_dict") or state.get("skills_loaded")
                or state.get("skill_sections")):
            return state

        from ....skills._shared._skill_selector import select_relevant_skills

        context_parts = []
        sysinfo = state.get("system_info")
        if isinstance(sysinfo, dict) and sysinfo:
            context_parts.append(f"Metadata: {str(sysinfo)[:1500]}")
        elif isinstance(sysinfo, str) and sysinfo.strip():
            context_parts.append(f"Metadata: {sysinfo.strip()[:1500]}")
        stats = state.get("data_statistics")
        if stats:
            context_parts.append(f"Data statistics: {stats}")
        plot_bytes = state.get("original_plot_bytes")
        if plot_bytes:
            # plot_fn renders PNG; declare it correctly (Bedrock's converse
            # API validates the declared type and rejects a mismatch).
            context_parts.append({"mime_type": "image/png", "data": plot_bytes})
        if not context_parts:
            return state

        self.logger.info("\n--- Skill Suggestion ---\n")

        # Curve-fitting skills are authoritative, mutually-exclusive techniques
        # (a 1D spectrum is XPS *or* EPR, never a blend) and inject MANDATORY
        # rules — so select at most one, the single best technique match.
        custom_skills = state.get("custom_skills") or {}
        selected = select_relevant_skills(
            model=self.model,
            parse_fn=self._parse,
            domain=self.domain,
            context_parts=context_parts,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            exclusive=True,
            hint=state.get("skill_hint"),
            custom_skills=custom_skills,
            logger=self.logger,
        )
        if selected:
            # Resolve selected custom-skill name(s) to their registered path(s).
            resolved = [custom_skills.get(n, n) for n in selected]
            state.update(self._load_skills(resolved, domain=self.domain))

        return state


class SeriesScoutController:
    """Scout representative spectra across a series before planning.

    For series with n > 1, loads and plots representative spectra
    (evenly spaced, capped at 7) so the LLM can see how data evolves
    across the series and plan fitting regimes proactively.

    For n == 1: no-op (state passes through unchanged).
    """

    def __init__(
        self,
        logger: logging.Logger,
        plot_fn: Callable,
    ):
        self.logger = logger
        self.plot_fn = plot_fn

    @staticmethod
    def _select_scout_indices(num_spectra: int) -> list:
        """Select evenly spaced representative indices, capped at 7."""
        if num_spectra <= 3:
            return list(range(num_spectra))
        if num_spectra <= 6:
            mid = num_spectra // 2
            return sorted({0, mid, num_spectra - 1})
        if num_spectra <= 15:
            indices = {0, num_spectra // 4, num_spectra // 2,
                       3 * num_spectra // 4, num_spectra - 1}
            return sorted(indices)
        # Large series: 7 evenly spaced
        step = (num_spectra - 1) / 6
        indices = {round(i * step) for i in range(7)}
        return sorted(indices)

    @staticmethod
    def _compute_statistics(curve_data: np.ndarray) -> dict:
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")
        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    @staticmethod
    def _extract_xy(curve_data: np.ndarray):
        """Extract x, y arrays from various data shapes."""
        if curve_data.ndim == 1:
            return np.arange(len(curve_data)), curve_data
        elif curve_data.shape[0] == 2:
            return curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            return curve_data[:, 0], curve_data[:, 1]
        raise ValueError(f"Unexpected data shape: {curve_data.shape}")

    @staticmethod
    def _create_overlay_plot(
        scout_curves: list,
        system_info: dict,
    ) -> str:
        """Create a single overlay figure with all scout spectra.

        Returns base64-encoded PNG (preserved shape for the existing prompt
        consumer at `state["scout_overlay_plot"]`). Rendering is delegated
        to `scilink.utils.curve_preview.render_curve_overlay` so the lit-
        search optimizer can reuse the same plotting logic.
        """
        from ....utils.curve_preview import render_curve_overlay

        png_bytes = render_curve_overlay(scout_curves, system_info)
        return base64.b64encode(png_bytes).decode("utf-8")

    def _load_spectrum(self, idx: int, state: dict) -> np.ndarray:
        spectrum_stack = state.get("spectrum_stack")
        if spectrum_stack is not None:
            return spectrum_stack[idx]
        data_path = state.get("spectrum_paths", [])[idx]
        try:
            from ....skills._shared.curve_fitting_tools import load_curve_data
            return load_curve_data(data_path)
        except ImportError:
            if data_path.endswith('.npy'):
                return np.load(data_path)
            return np.loadtxt(data_path, delimiter=',')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("is_single_spectrum", True):
            return state

        num_spectra = state.get("num_spectra", 1)
        if num_spectra <= 1:
            return state

        self.logger.info("\n🔭 --- Scouting Series ---\n")

        scout_indices = self._select_scout_indices(num_spectra)
        series_metadata = state.get("series_metadata", {})
        values = series_metadata.get("values", [])
        variable = series_metadata.get("variable", "index")
        unit = series_metadata.get("unit", "")

        scout_data = []
        scout_curves = []  # for overlay plot
        for idx in scout_indices:
            try:
                curve_data = self._load_spectrum(idx, state)

                stats = self._compute_statistics(curve_data)

                if idx < len(values):
                    label = f"{variable}={values[idx]} {unit}".strip()
                else:
                    label = f"index {idx}"

                plot_bytes = self.plot_fn(
                    curve_data, state.get("system_info", {}),
                    title_suffix=f" [{label}]",
                )

                scout_data.append({
                    "index": idx,
                    "label": label,
                    "statistics": stats,
                    "plot_bytes": plot_bytes,
                })
                scout_curves.append({
                    "label": label,
                    "curve_data": curve_data,
                })
                self.logger.info(f"  Scouted spectrum {idx}: {label}")
            except Exception as e:
                self.logger.warning(f"  Failed to scout spectrum {idx}: {e}")

        # Generate overlay comparison plot
        if len(scout_curves) >= 2:
            try:
                overlay_bytes = self._create_overlay_plot(
                    scout_curves, state.get("system_info", {})
                )
                state["scout_overlay_plot"] = overlay_bytes
                self.logger.info("  Generated overlay comparison plot")
            except Exception as e:
                self.logger.warning(f"  Failed to create overlay plot: {e}")
                state["scout_overlay_plot"] = None
        else:
            state["scout_overlay_plot"] = None

        state["scout_data"] = scout_data
        self.logger.info(f"  Scouted {len(scout_data)} of {num_spectra} spectra")

        return state


class LiteratureSearchController:
    """Search literature if enabled and query provided.

    DEPRECATED: prefer the orchestrator-level `search_literature` tool, which
    fetches lit context BEFORE planning so the planner can produce a
    literature-informed plan. This in-pipeline controller is retained as a
    fallback for direct-Python-API callers using `use_literature=True`.
    """

    def __init__(
        self,
        logger: logging.Logger,
        literature_agent: Any | None,
        output_dir: str,
    ):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_results(self, query: str, report: str) -> dict:
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, "w") as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["report_file"] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save literature: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if state.get("literature_context"):
            self.logger.info("\n📚 --- Skipping Literature (pre-fetched via search_literature tool) ---\n")
            return state

        if self.literature_agent is None:
            self.logger.info("\n📚 --- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n📚 --- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n📚 --- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  ✅ Success")
            else:
                state["literature_context"] = None
                self.logger.warning("  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class GenerateCurveFittingReportController:
    """Generates a human-readable HTML report for curve fitting analysis."""
    
    DEFAULT_R2_THRESHOLD = 0.95

    def __init__(self, logger: logging.Logger, output_dir: str, r2_threshold: float = None):
        self.logger = logger
        self.output_dir = output_dir
        self.r2_threshold = r2_threshold if r2_threshold is not None else self.DEFAULT_R2_THRESHOLD

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get("is_single_spectrum", True):
            return state
            
        self.logger.info("\n📄 --- Generating HTML Report ---\n")
        
        result_json = state.get("result_json", {})
        fit_results = state.get("fit_results", {})
        synthesis_result = state.get("synthesis_result", {})
        
        detailed_analysis = synthesis_result.get("detailed_analysis") or result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = synthesis_result.get("scientific_claims") or result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        model_type = fit_results.get("model_type", result_json.get("model_type", "N/A"))
        parameters = fit_results.get("parameters", result_json.get("fitting_parameters", {}))
        fit_quality = fit_results.get("fit_quality", result_json.get("fit_quality", {}))
        caveats = synthesis_result.get("caveats") or result_json.get("caveats", "")
        
        quality_warning = None
        series_results = state.get("series_results", [])
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]
        
        original_plot = state.get("original_plot_bytes")
        fit_plot = state.get("final_plot_bytes")
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CurveFitting_Report_{file_timestamp}.html"
        filepath = output_dir / filename

        params_html = self._format_parameters(parameters)
        quality_html = self._format_fit_quality(fit_quality, quality_warning, gate=_gate(state))

        # In identification mode, surface the ranked candidate list. Empty
        # string in fitting mode so the HTML report is unchanged for the
        # default path.
        candidate_identifications = []
        if state.get("task_mode") == "identification":
            candidate_identifications = (
                synthesis_result.get("candidate_identifications")
                or result_json.get("candidate_identifications")
                or []
            )
        candidates_html = self._format_candidate_identifications(
            candidate_identifications,
            literature_used=bool(state.get("literature_context")),
        )

        html_content = self._build_html_report(
            timestamp, state, system_info, model_type, quality_html,
            detailed_analysis, original_plot, fit_plot, params_html,
            scientific_claims, caveats, candidates_html,
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"  ✅ Report saved: {filepath}")
            state["report_path"] = str(filepath)
        except Exception as e:
            self.logger.error(f"  ❌ Failed to write report: {e}")

        return state

    def _build_html_report(self, timestamp, state, system_info, model_type,
                          quality_html, detailed_analysis, original_plot,
                          fit_plot, params_html, scientific_claims, caveats,
                          candidates_html=""):
        """Build the complete HTML report."""
        system_info_str = self._format_system_info(system_info)
        
        images_html = ""
        if original_plot:
            b64_original = self._image_to_base64(original_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_original}" alt="Original Data"><div class="image-label">Original Data</div></div>'
        if fit_plot:
            b64_fit = self._image_to_base64(fit_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_fit}" alt="Fit Visualization"><div class="image-label">Fit Result with Residuals</div></div>'

        claims_html = ""
        if not scientific_claims:
            claims_html = "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f"""
        <div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Literature Search Query:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>"""

        caveats_html = ""
        if caveats:
            caveats_html = f"""
        <h2>5. Caveats & Limitations</h2>
        <div class="caveats">{caveats}</div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .model-box {{ background-color: #e8f4fc; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .params-table th, .params-table td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        .params-table th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }}
        .params-table tr:hover {{ background-color: #f5f5f5; }}
        .quality-badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; margin-right: 10px; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-ok {{ background-color: #fff3cd; color: #856404; }}
        .quality-poor {{ background-color: #f8d7da; color: #721c24; }}
        .quality-warning-box {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px 15px; margin-top: 10px; border-radius: 0 5px 5px 0; font-size: 0.9em; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Curve Fitting Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {state.get('data_path', 'N/A')}</p>
            <p><strong>Sample Info:</strong> {system_info_str}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <h3>Fitting Model</h3>
        <div class="model-box">{model_type}</div>
        <h3>Fit Quality</h3>
        {quality_html}
        <h3>Interpretation</h3>
        <div class="analysis-text">{detailed_analysis}</div>
        <h2>2. Visualizations</h2>
        <div class="image-grid">{images_html}</div>
        <h2>3. Fitted Parameters</h2>
        {params_html}
        <h2>4. Scientific Claims</h2>
        {claims_html}
        {candidates_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Analysis Agent</div>
    </div>
</body>
</html>"""

    def _format_system_info(self, system_info: dict) -> str:
        if not system_info:
            return "N/A"
        parts = [f"{k}: {v}" for k, v in system_info.items() if v]
        return ", ".join(parts) if parts else "N/A"

    def _format_parameters(self, parameters: dict) -> str:
        if not parameters:
            return "<p>No parameters extracted.</p>"

        rows = ""
        for component, params in parameters.items():
            if isinstance(params, dict):
                first_row = True
                for param_name, value in params.items():
                    if param_name.endswith("_err"):
                        continue
                    err_key = f"{param_name}_err"
                    err_value = params.get(err_key, "—")
                    if isinstance(err_value, (int, float)):
                        err_value = f"± {err_value:.4g}"
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4g}"
                    else:
                        value_str = str(value)
                    component_display = component if first_row else ""
                    rows += f"<tr><td><strong>{component_display}</strong></td><td>{param_name}</td><td>{value_str}</td><td>{err_value}</td></tr>"
                    first_row = False
            else:
                rows += f"<tr><td><strong>{component}</strong></td><td>—</td><td>{parameters[component]}</td><td>—</td></tr>"

        return f"""<table class="params-table">
            <thead><tr><th>Component</th><th>Parameter</th><th>Value</th><th>Uncertainty</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def _format_candidate_identifications(self, candidates, literature_used: bool = False) -> str:
        """Render the id-mode ranked candidate list as an HTML section.

        Returns an empty string when `candidates` is empty or missing —
        so the rest of the report is unchanged for fitting-mode runs.
        Missing per-candidate fields fall back to "—" rather than
        raising; malformed entries are skipped.

        `literature_used` controls the provenance caveat: when a literature
        search was consulted, the disclaimer must not claim the candidates
        rest on model knowledge alone.
        """
        if not candidates:
            return ""

        def _esc(x):
            return str(x) if x is not None else "—"

        rows = ""
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            name = _esc(cand.get("name", "(unnamed)"))
            rank = _esc(cand.get("rank", "—"))
            consistency = _esc(cand.get("consistency", "—"))
            peaks_present = cand.get("discriminating_peaks_present") or []
            peaks_absent = cand.get("discriminating_peaks_absent") or []
            distinguishing = _esc(cand.get("distinguishing_evidence", "—"))

            peaks_present_html = (
                "<br>".join(f"• {_esc(p)}" for p in peaks_present)
                if peaks_present else "—"
            )
            peaks_absent_html = (
                "<br>".join(f"• {_esc(p)}" for p in peaks_absent)
                if peaks_absent else "—"
            )

            rows += (
                f"<tr>"
                f"<td><strong>{rank}</strong></td>"
                f"<td>{name}</td>"
                f"<td>{consistency}</td>"
                f"<td>{peaks_present_html}</td>"
                f"<td>{peaks_absent_html}</td>"
                f"<td>{distinguishing}</td>"
                f"</tr>"
            )

        if not rows:
            return ""

        if literature_used:
            provenance = (
                "Candidates enumerated by the model from the spectral evidence "
                "and a consulted literature search; ranks and consistency grades "
                "remain qualitative model judgments (not database-verified)."
            )
        else:
            provenance = (
                "LLM-enumerated candidates from the spectral evidence; ranks and "
                "consistency grades are qualitative LLM judgments (not database-verified)."
            )

        return f"""
        <h2>Candidate Identifications (id-mode)</h2>
        <p><em>{provenance}
        Use <strong>Distinguishing evidence</strong> to plan a follow-up measurement
        that would separate the top candidates.</em></p>
        <table class="params-table">
            <thead><tr>
                <th>Rank</th><th>Candidate</th><th>Consistency</th>
                <th>Peaks present</th><th>Peaks absent</th>
                <th>Distinguishing evidence</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def _format_fit_quality(self, fit_quality: dict, quality_warning: str = None,
                            gate=None) -> str:
        if not fit_quality:
            return "<p>No quality metrics available.</p>"

        r_squared = fit_quality.get("r_squared", fit_quality.get("r2"))
        rmse = fit_quality.get("rmse")
        chi_squared = fit_quality.get("chi_squared_reduced", fit_quality.get("reduced_chi_squared"))

        html = "<div>"
        # Badge on the GATE's acceptance metric. For a non-R² goodness-of-fit
        # gate (e.g. peak_region_r2) the badge must reflect that metric and its
        # thresholds — a global-R² badge mislabels a verifier-approved low-SNR
        # fit as "Poor". The R² path is unchanged.
        gate_value = gate.extract(fit_quality) if (gate is not None and gate.metric != "r_squared") else None
        if gate_value is not None:
            if gate.is_accept(gate_value):
                badge_class, label = "quality-good", "Good"
            elif gate.is_hard_reject(gate_value):
                badge_class, label = "quality-poor", "Poor"
            else:
                badge_class, label = "quality-ok", "Marginal"
            html += f'<span class="quality-badge {badge_class}">{label}</span><strong>{gate.label} = {gate_value:.4f}</strong>'
            if r_squared is not None:
                html += f" &nbsp;|&nbsp; <span>R² = {r_squared:.4f}</span>"
        elif r_squared is not None:
            if r_squared >= self.r2_threshold + 0.04:
                badge_class, label = "quality-good", "Excellent"
            elif r_squared >= self.r2_threshold:
                badge_class, label = "quality-ok", "Good"
            else:
                badge_class, label = "quality-poor", "Poor"
            html += f'<span class="quality-badge {badge_class}">{label}</span><strong>R² = {r_squared:.4f}</strong>'

        if rmse is not None:
            html += f" &nbsp;|&nbsp; <strong>RMSE = {rmse:.4g}</strong>"
        if chi_squared is not None:
            html += f" &nbsp;|&nbsp; <strong>χ²/DOF = {chi_squared:.3f}</strong>"
        html += "</div>"
        
        if quality_warning:
            html += f'<div class="quality-warning-box">⚠️ <strong>Note:</strong> {quality_warning}.</div>'
        
        return html


# ============================================================================
# UNIFIED CONTROLLERS (for series analysis support)
# ============================================================================

class HumanFeedbackRefinementController:
    """
    Facilitates human-in-the-loop parameter refinement for the first spectrum.
    
    Works identically for single spectra and series:
    - Single spectrum: Refine fitting, then process that one spectrum
    - Series: Refine fitting on first spectrum, then apply to all
    """
    
    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        instructions: str,
        output_dir: str,
        enable_human_feedback: bool = False,
        max_iterations: int = 5
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.instructions = instructions
        self.output_dir = Path(output_dir)
        self.enable_human_feedback = enable_human_feedback
        self.max_iterations = max_iterations

    def _display_plan(self, state: dict) -> None:
        is_single = state.get("is_single_spectrum", True)
        num_spectra = state.get("num_spectra", 1)
        
        print("\n" + "=" * 60)
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        print(f"📋 PROPOSED FITTING PLAN - {mode_str}")
        print("=" * 60)
        
        if state.get("observations"):
            print(f"\n🔍 Observations:\n   {state['observations']}")
        
        print(f"\n📊 Approach:\n   {state.get('analysis_approach', 'N/A')}")
        print(f"\n📐 Physical Model:\n   {state.get('physical_model', 'N/A')}")
        print(f"\n🎯 Parameters to Extract:\n   {', '.join(state.get('parameters_to_extract', [])) or 'N/A'}")
        import re as _re
        _strategy = state.get("fitting_strategy", "N/A")
        # Put each numbered step on its own line with consistent indentation.
        # Only split on step numbers that follow a sentence-ending ". " to
        # avoid mangling numbers in text (e.g. "cm-1.", "8.7").
        _strategy = _re.sub(r"\. (\d+)\. ", r".\n   \1. ", _strategy)
        print(f"\n⚙️  Fitting Strategy:\n   {_strategy}")

        # Display regime plan if present
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes") and not is_single:
            regimes = series_plan["regimes"]
            print(f"\n{'=' * 60}")
            print(f"📦 SERIES FITTING REGIMES ({len(regimes)} regimes)")
            print(f"{'=' * 60}")
            if series_plan.get("rationale"):
                print(f"\nRationale: {series_plan['rationale']}")

            series_metadata = state.get("series_metadata", {})
            values = series_metadata.get("values", [])
            # Defensive: a filename-keyed dict should be normalized upstream
            # (_normalize_series_values), but tolerate it here so a stray dict
            # can't crash planning (min/max over dicts) — display only needs the
            # scalar range, so value-order is irrelevant.
            if isinstance(values, dict):
                values = list(values.values())
            unit = series_metadata.get("unit", "")

            for i, regime in enumerate(regimes, 1):
                indices = regime.get("spectrum_indices", [])
                if values and indices:
                    valid_vals = [values[idx] for idx in indices if idx < len(values)]
                    range_str = f" ({min(valid_vals)}-{max(valid_vals)} {unit})" if valid_vals else ""
                else:
                    range_str = ""
                # Fall back to the top-level plan fields so a regime the LLM left
                # sparse never renders as all-N/A.
                model = regime.get("physical_model") or series_plan.get("physical_model") or "N/A"
                strategy = regime.get("fitting_strategy") or series_plan.get("fitting_strategy") or "N/A"
                params = regime.get("parameters_to_extract") or series_plan.get("parameters_to_extract", [])
                print(f"\n  Regime {i}: {regime.get('name', 'Unnamed')}")
                print(f"    Spectra: indices {indices}{range_str}")
                print(f"    Model: {model}")
                print(f"    Strategy: {strategy}")
                print(f"    Parameters: {', '.join(params)}")

            transitions = series_plan.get("transition_points", [])
            if transitions:
                print(f"\n  Transition Points:")
                for t in transitions:
                    print(f"    Between indices {t.get('between_indices', '?')}: "
                          f"{t.get('description', 'N/A')}")
        elif not is_single:
            print(f"\n📦 **Note:** This fitting model will be LOCKED and applied to all {num_spectra} spectra.")

        print("\n" + "=" * 60)

    def _get_human_feedback(self, state: dict) -> dict:
        self._display_plan(state)
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        if feedback == "":
            print("✅ Plan accepted.")
            return state
        else:
            state["_refine_requested"] = True
            state["_refine_feedback"] = feedback
            return state

    def _apply_column_mapping_to_arrays(self, state: dict, mapping: dict) -> None:
        """Re-slice in-memory array/DataFrame inputs to the LLM-chosen (x, y).

        File inputs re-load lazily with the locked mapping (see the load path), so
        only array/DataFrame inputs — whose data was reduced heuristically at
        ingestion — need this in-memory correction. No-op otherwise.
        """
        raw = state.get("raw_first_spectrum_full")
        if raw is None:
            return
        raw = np.asarray(raw)
        if raw.ndim != 2:
            return
        if raw.shape[0] < raw.shape[1]:          # orient to (n_points, n_cols)
            raw = raw.T
        xi, yi = mapping["x_index"], mapping["y_index"]
        if xi >= raw.shape[1] or yi >= raw.shape[1]:
            return
        xy = np.vstack([raw[:, xi], raw[:, yi]])  # (2, n)
        stack = state.get("spectrum_stack")
        if stack is not None and stack.shape[0] >= 1:
            new_stack = stack.copy()
            new_stack[0] = xy
            state["spectrum_stack"] = new_stack
        state["curve_data"] = xy
        x, y = xy[0], xy[1]
        state["data_statistics"] = {
            "n_points": int(len(x)),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(xy))),
        }
        self.logger.info(
            f"  Re-sliced in-memory data to X=col {xi}, Y=col {yi} per column mapping."
        )

    def _plan_analysis(self, state: dict) -> dict:
        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        # Fit-free zoom into hard-to-resolve regions the full plot squashes.
        _append_structure_zoom(prompt, state)
        _append_objective_context(prompt, state)
        _append_fit_domain_guidance(prompt, state)
        _append_column_structure(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)
        _prior_runs = _prior_curve_fit_block(state)
        if _prior_runs:
            prompt.append(_prior_runs)

        # Withhold lit context from the planner in identification mode — it
        # would re-anchor the planner to specific known materials/phases and
        # defeat the unbiased-fit purpose. Lit context still reaches Stage-2
        # candidate enumeration via the synthesis prompt.
        if state.get("literature_context") and state.get("task_mode") != "identification":
            prompt.append("\n## Literature\n" + state["literature_context"])

        # Identification mode: require a generic, material-agnostic fit plan.
        if state.get("task_mode") == "identification":
            from ..instruct import ID_MODE_PLANNING_ADDENDUM
            prompt.append(ID_MODE_PLANNING_ADDENDUM)

        # Series context: use scout data if available, otherwise basic notice
        num_spectra = state.get("num_spectra", 1)
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_spectrum", True):
            self._append_scout_context(prompt, state, scout_data)
        elif not state.get("is_single_spectrum", True):
            prompt.append(
                f"\n## Series Context\nThis is the first spectrum in a series of {num_spectra}. "
                "The fitting model you choose will be applied to ALL spectra in the series."
            )

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            raise ValueError(f"Failed to parse: {error}")

        state["observations"] = result.get("observations", "")
        state["analysis_approach"] = result.get("analysis_approach", "Curve fitting")
        state["physical_model"] = result.get("physical_model", "Appropriate model")
        state["parameters_to_extract"] = result.get("parameters_to_extract", [])
        state["fitting_strategy"] = result.get("fitting_strategy", "Standard fitting")
        state["literature_query"] = result.get("literature_query")

        # Multi-column inputs: record the LLM's column decision (resolved + locked
        # later). Only present when a Column Structure block was shown.
        if state.get("column_info"):
            state["column_mapping"] = result.get("column_mapping")
            state["column_mapping_note"] = result.get("column_mapping_note", "")
            if state["column_mapping"]:
                self.logger.info(
                    f"  Column mapping (LLM): {state['column_mapping']} "
                    f"— {state['column_mapping_note']}"
                )

        # Extract series analysis plan if present
        self._extract_series_plan(state, result)

        return state

    def _validate_plan(self, state: dict) -> dict:
        """Validate the proposed fitting plan against the data (and skill rules
        when a skill is loaded).

        Always runs (matching ImageAnalysisAgent): the data-grounded sanity
        check — do the planned peaks exist, is the plot consistent with the
        model — is useful even without a skill. Skill-conformance is enforced
        only when skill rules are present (the validation prompt applies the
        "MANDATORY Domain Skill Rules" clause conditionally).
        """
        from ..instruct import CURVE_FITTING_PLAN_VALIDATION_PROMPT

        regime_section = ""
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes"):
            lines = ["\n**Regimes:**"]
            for regime in series_plan["regimes"]:
                lines.append(
                    f"- {regime.get('name', 'Unnamed')}: "
                    f"model={regime.get('physical_model', 'N/A')}, "
                    f"params={', '.join(regime.get('parameters_to_extract', []))}"
                )
            regime_section = "\n".join(lines)

        prompt_text = CURVE_FITTING_PLAN_VALIDATION_PROMPT.format(
            analysis_approach=state.get("analysis_approach", "N/A"),
            physical_model=state.get("physical_model", "N/A"),
            parameters_to_extract=", ".join(state.get("parameters_to_extract", [])),
            fitting_strategy=state.get("fitting_strategy", "N/A"),
            regime_section=regime_section,
        )

        prompt_parts = [prompt_text]
        # Inject the user's objective so the validator judges the plan against
        # what was actually asked — not the data plot alone. Without this, an
        # explicit requirement (a region to exclude, a parameter to report) is
        # invisible here and gets silently stripped when the data looks
        # ambiguous. Mirrors the planning prompt and ImageAnalysis._validate_plan.
        _append_objective_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "planning")

        # For a series, show the multi-spectrum scout overlay (the single
        # first-spectrum plot is uninformative — and can render blank — for a
        # series); fall back to the single-spectrum plot otherwise.
        data_plot = state.get("scout_overlay_plot") or state.get("original_plot_bytes")
        if data_plot:
            prompt_parts.append("\n**Data:**")
            prompt_parts.append({"mime_type": "image/png", "data": data_plot})
        # Same fit-free zoom into hard-to-resolve regions so the validator can
        # catch unresolved structure the plan mischaracterized (no-op for series
        # stacks, where _extract_xy returns None).
        _append_structure_zoom(prompt_parts, state)

        try:
            response = self.model.generate_content(
                prompt_parts, generation_config=self.generation_config,
            )
            result, error = self._parse(response)

            if error or not result:
                self.logger.warning("  Plan validation parse failed, keeping plan")
                return state

            if result.get("valid", True):
                self.logger.info("  Plan validation: approved")
                return state

            issues = result.get("issues", [])
            self.logger.info(f"  Plan validation: {len(issues)} issue(s) found, revising")
            for issue in issues:
                self.logger.info(f"    - {issue}")

            if result.get("physical_model"):
                state["physical_model"] = result["physical_model"]
            if result.get("parameters_to_extract"):
                state["parameters_to_extract"] = result["parameters_to_extract"]
            if result.get("fitting_strategy"):
                state["fitting_strategy"] = result["fitting_strategy"]
            if result.get("series_analysis_plan"):
                self._extract_series_plan(state, result)

        except Exception as e:
            self.logger.warning(f"  Plan validation failed: {e}, keeping plan")

        return state

    def _append_scout_context(self, prompt: list, state: dict, scout_data: list) -> None:
        """Append scout spectrum plots and series regime planning instructions."""
        from ..instruct import SERIES_REGIME_PLANNING_SUPPLEMENT

        num_spectra = state.get("num_spectra", 1)
        series_metadata = state.get("series_metadata", {})

        prompt.append(f"\n## Series Overview ({num_spectra} spectra)")
        prompt.append(
            "Below are representative spectra from across the series. "
            "Examine how the data changes. If the spectral character changes "
            "significantly (e.g., peak splitting, new features, major shape "
            "changes), plan multiple fitting regimes. Otherwise, a single "
            "model is fine."
        )

        if series_metadata.get("variable"):
            values = series_metadata.get("values", [])
            unit = series_metadata.get("unit", "")
            prompt.append(
                f"\nSeries variable: {series_metadata['variable']} ({unit})"
            )
            if values:
                prompt.append(f"Range: {values[0]} to {values[-1]} {unit}")
            secondary = series_metadata.get("secondary_variables") or []
            if secondary:
                names = "; ".join(
                    f"{s.get('variable')}"
                    + (f" ({s.get('unit')})" if s.get("unit") else "")
                    for s in secondary
                )
                prompt.append(
                    f"Additional control variable(s) co-varying across the "
                    f"series: {names}. The series is ordered by "
                    f"{series_metadata['variable']}, but these also change "
                    f"between spectra — account for their effect when "
                    f"interpreting how the data evolves."
                )

        # Overlay comparison plot (all scouts on one figure)
        overlay = state.get("scout_overlay_plot")
        if overlay:
            prompt.append(
                "\n### Overlay Comparison\n"
                "All scout spectra plotted together for direct visual comparison. "
                "Look for shifts, shape changes, peak splitting, or new features "
                "emerging across the series."
            )
            prompt.append({
                "mime_type": "image/png",
                "data": overlay,
            })

        prompt.append("\n### Individual Scout Spectra")
        for scout in scout_data:
            prompt.append(
                f"\n### Spectrum at {scout['label']} (index {scout['index']})"
            )
            prompt.append(f"Statistics: {json.dumps(scout['statistics'], indent=2)}")
            prompt.append({
                "mime_type": "image/png",
                "data": scout["plot_bytes"],
            })

        prompt.append(SERIES_REGIME_PLANNING_SUPPLEMENT.format(
            num_spectra=num_spectra,
            num_spectra_minus_1=num_spectra - 1,
        ))

    def _extract_series_plan(self, state: dict, result: dict) -> None:
        """Extract and validate series_analysis_plan from LLM response."""
        series_plan = result.get("series_analysis_plan")
        if not isinstance(series_plan, dict) or state.get("is_single_spectrum", True):
            state["series_analysis_plan"] = None
            return

        num_spectra = state.get("num_spectra", 1)
        # Defensively drop malformed (non-dict) regimes — an LLM/validator
        # revision can return a regime as a bare string, which would otherwise
        # crash regime.get(...) below.
        regimes = [r for r in series_plan.get("regimes", []) if isinstance(r, dict)]
        series_plan["regimes"] = regimes

        if not regimes:
            state["series_analysis_plan"] = None
            return

        # Validate index coverage
        all_indices = set()
        for regime in regimes:
            indices = regime.get("spectrum_indices", [])
            # Filter to valid range
            regime["spectrum_indices"] = [i for i in indices if 0 <= i < num_spectra]
            all_indices.update(regime["spectrum_indices"])

        missing = set(range(num_spectra)) - all_indices
        if missing:
            self.logger.warning(
                f"  Series plan missing indices {sorted(missing)}, "
                f"assigning to first regime"
            )
            regimes[0]["spectrum_indices"] = sorted(
                set(regimes[0]["spectrum_indices"]) | missing
            )

        # Drop regimes that ended up fitting no spectra. The LLM sometimes names
        # an aspirational regime (e.g. "split doublet") but commits every
        # spectrum to one regime; the orphan-reassignment above then leaves the
        # other empty. An empty regime fits nothing — it would pollute the banner,
        # the report, the regime-locking step, and the per-index regime loops — so
        # it is removed here rather than carried forward.
        dropped = [r.get("name", "unnamed") for r in regimes if not r["spectrum_indices"]]
        regimes = [r for r in regimes if r["spectrum_indices"]]
        series_plan["regimes"] = regimes
        if dropped:
            self.logger.warning(
                f"  Dropped {len(dropped)} empty regime(s) (no spectra assigned): {dropped}"
            )
        if not regimes:
            state["series_analysis_plan"] = None
            return

        # Backfill per-regime model/strategy/params from the top-level plan when
        # the LLM populated only the top level — otherwise the display and report
        # show "N/A" for a regime whose model is in fact known.
        for regime in regimes:
            if not regime.get("physical_model"):
                regime["physical_model"] = (
                    series_plan.get("physical_model") or "Model to be determined from the data"
                )
            if not regime.get("fitting_strategy"):
                regime["fitting_strategy"] = series_plan.get("fitting_strategy")
            if not regime.get("parameters_to_extract"):
                regime["parameters_to_extract"] = series_plan.get("parameters_to_extract", [])

        state["series_analysis_plan"] = series_plan
        self.logger.info(
            f"  Series analysis plan: {len(regimes)} regime(s)"
        )
        for regime in regimes:
            self.logger.info(
                f"    {regime.get('name', 'unnamed')}: "
                f"indices {regime.get('spectrum_indices', [])}, "
                f"model: {regime.get('physical_model', 'N/A')}"
            )

    def _refine_plan(self, state: dict, feedback: str) -> dict:
        current_plan = (
            f"Observations: {state.get('observations', 'N/A')}\n"
            f"Approach: {state.get('analysis_approach', 'N/A')}\n"
            f"Physical Model: {state.get('physical_model', 'N/A')}\n"
            f"Parameters: {', '.join(state.get('parameters_to_extract', []))}\n"
            f"Strategy: {state.get('fitting_strategy', 'N/A')}"
        )
        # For >2-column inputs, show the currently selected columns so the
        # feedback (e.g. "use the other columns") is anchored to what is wrong.
        if state.get("column_info") and state.get("column_mapping"):
            current_plan += (
                f"\nColumn Mapping: {json.dumps(state['column_mapping'])}"
                f" — {state.get('column_mapping_note', '')}"
            )

        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
            f"\n## Current Plan\n{current_plan}",
            f"\n## User Feedback\nAdjust the plan based on this feedback: \"{feedback}\"",
        ]

        _append_column_structure(prompt, state)
        _append_objective_context(prompt, state)
        _append_fit_domain_guidance(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## Original Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)

        if state.get("literature_context") and state.get("task_mode") != "identification":
            prompt.append("\n## Literature\n" + state["literature_context"])

        # Include current series plan and scout data in refinement context
        if state.get("series_analysis_plan"):
            prompt.append(
                f"\n## Current Series Analysis Plan\n"
                f"{json.dumps(state['series_analysis_plan'], indent=2)}"
            )
            prompt.append(
                "\nThe user may want to adjust regime boundaries, merge regimes, "
                "change models for specific regimes, or switch to a single model. "
                "Adjust the series_analysis_plan accordingly, or remove it entirely "
                "if the user wants a single model."
            )
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_spectrum", True):
            self._append_scout_context(prompt, state, scout_data)

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            self.logger.warning(f"Refinement failed: {error}. Keeping current plan.")
            return state

        state["observations"] = result.get("observations", state.get("observations", ""))
        state["analysis_approach"] = result.get("analysis_approach", state.get("analysis_approach"))
        state["physical_model"] = result.get("physical_model", state.get("physical_model"))
        state["parameters_to_extract"] = result.get("parameters_to_extract", state.get("parameters_to_extract", []))
        state["fitting_strategy"] = result.get("fitting_strategy", state.get("fitting_strategy"))
        state["literature_query"] = result.get("literature_query", state.get("literature_query"))

        # Re-extract the column decision so a feedback-corrected X/Y choice is
        # honored at lock time. Mirrors _plan_analysis; guarded so a refinement
        # that omits column_mapping keeps the prior selection rather than nulling
        # it (the locked fit reads state["column_mapping"] via _resolve_column_mapping).
        if state.get("column_info"):
            state["column_mapping"] = result.get("column_mapping", state.get("column_mapping"))
            state["column_mapping_note"] = result.get(
                "column_mapping_note", state.get("column_mapping_note", "")
            )
            if state.get("column_mapping"):
                self.logger.info(
                    f"  Column mapping (refined): {state['column_mapping']} "
                    f"— {state['column_mapping_note']}"
                )

        # Re-extract series plan (may have been updated or removed)
        self._extract_series_plan(state, result)

        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_spectrum", True)
        mode_str = "SINGLE SPECTRUM" if is_single else "SERIES"
        self.logger.info(f"\n🧠 --- Planning Analysis ({mode_str}) ---\n")

        try:
            state = self._plan_analysis(state)
            state = self._validate_plan(state)
            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Model: {state['physical_model']}")

            # On a verbatim locked-script reuse turn (#172) the plan is
            # foreordained to re-run the prior script unchanged, so re-approving
            # it is pointless interruption. Planning still ran above (downstream
            # stages need its fields); we only skip the display/approval gate.
            if self.enable_human_feedback and not state.get("reuse_locked_script"):
                iteration = 0
                while iteration < self.max_iterations:
                    state = self._get_human_feedback(state)
                    if state.pop("_refine_requested", False):
                        feedback = state.pop("_refine_feedback", "")
                        if feedback:
                            state.setdefault("human_feedback_log", []).append(str(feedback))
                        self.logger.info(f"  Refining with feedback: {feedback}")
                        print("\n🔄 Refining plan...\n")
                        state = self._refine_plan(state, feedback)
                        iteration += 1
                    else:
                        break

                if iteration >= self.max_iterations:
                    self.logger.warning("  Max iterations reached.")
                    print("⚠️  Max refinements reached. Proceeding with current plan.")

            # Resolve + lock the column mapping (>2-col inputs only). It applies to
            # every spectrum (column roles are a file-structure property), so store
            # it at a stable key the load path reads, and re-slice array/DataFrame
            # inputs whose data is already in memory.
            column_mapping = _resolve_column_mapping(state)
            state["column_mapping_locked"] = column_mapping
            if column_mapping:
                self.logger.info(
                    f"  ✅ Locked column mapping: X=col {column_mapping['x_index']}, "
                    f"Y=col {column_mapping['y_index']}."
                )
                self._apply_column_mapping_to_arrays(state, column_mapping)
            elif state.get("column_info"):
                self.logger.info(
                    "  Column mapping unresolved/absent — heuristic X/Y selection."
                )

            self._lock_config(state, column_mapping)

        except Exception as e:
            self.logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Fit the data with an appropriate model"
            state["physical_model"] = "To be determined"
            state["parameters_to_extract"] = []
            state["fitting_strategy"] = "Standard curve fitting"
            state["literature_query"] = None
            # Mirror the success-path config shape (not None) so downstream
            # consumers that do `locked_fitting_config.copy()` /
            # `.get("physical_model")` don't crash on the fallback path.
            state["locked_fitting_config"] = {
                "analysis_approach": state.get("analysis_approach"),
                "physical_model": state.get("physical_model"),
                "parameters_to_extract": state.get("parameters_to_extract", []),
                "fitting_strategy": state.get("fitting_strategy"),
                "column_mapping": None,
            }
            state["column_mapping_locked"] = None
            state["series_analysis_plan"] = None
            state["regime_configs"] = None

        return state

    def _lock_config(self, state: dict, column_mapping) -> None:
        """Freeze the current fitting-plan fields into ``locked_fitting_config``
        (and per-regime configs). Shared by ``execute`` and the per-candidate
        ``replan_headless`` so both lock identically. ``column_mapping`` is a
        file-structure property resolved once and passed in — candidates
        inherit the primary plan's mapping rather than re-resolving it."""
        state["locked_fitting_config"] = {
            "analysis_approach": state.get("analysis_approach"),
            "physical_model": state.get("physical_model") or "Model to be determined from the data",
            "parameters_to_extract": state.get("parameters_to_extract", []),
            "fitting_strategy": state.get("fitting_strategy"),
            "column_mapping": column_mapping,
        }

        # Build per-regime configs if series plan has multiple regimes
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes"):
            regime_configs = {}
            for regime in series_plan["regimes"]:
                regime_config = {
                    "analysis_approach": state.get("analysis_approach"),
                    "physical_model": regime.get(
                        "physical_model", state.get("physical_model")
                    ),
                    "parameters_to_extract": regime.get(
                        "parameters_to_extract",
                        state.get("parameters_to_extract", []),
                    ),
                    "fitting_strategy": regime.get(
                        "fitting_strategy", state.get("fitting_strategy")
                    ),
                    # Column roles are a file property — same across regimes.
                    "column_mapping": column_mapping,
                }
                for idx in regime.get("spectrum_indices", []):
                    regime_configs[idx] = regime_config
            state["regime_configs"] = regime_configs
            self.logger.info(
                f"  ✅ Locked {len(series_plan['regimes'])} regime "
                f"configuration(s) for series processing."
            )
        else:
            state["regime_configs"] = None
            self.logger.info(
                "  ✅ Fitting configuration locked for series processing."
            )

    def replan_headless(self, state: dict) -> dict:
        """Generate a fresh, INDEPENDENT fitting plan for one best-of-N
        candidate.

        Mirrors ``execute``'s planning — one ``_plan_analysis`` + ``_validate_plan``
        + lock — with NO human feedback and NO candidate pre-selection.
        Divergence comes from inherent sampling, which is especially valuable
        for SKILL-LESS curves where initial-plan variance (model family, peak
        count, background) is high; when an authoritative technique skill is
        active the plans naturally converge on the mandated model (correct).
        COLUMN MAPPING is a file-structure property already resolved and
        applied to the shared data by the primary plan, so candidates INHERIT
        it (no re-resolve, no re-slice). Mutates and returns ``state``.
        """
        state = self._plan_analysis(state)
        self.logger.info(f"  Approach: {state['analysis_approach']}")
        self.logger.info(f"  Model: {state['physical_model']}")
        state = self._validate_plan(state)
        self._lock_config(state, state.get("column_mapping_locked"))
        return state


def _write_series_fit_results(output_dir, state, series_results, quality_settings):
    """Write ``series_fit_results.json`` from the current ``series_results``.

    Called after initial fitting AND re-called after the adaptive refit, so the
    file reflects adopted refits. It feeds the BO/planning feature table
    (``write_feature_table``) and the #172 prior-run reference summary
    (``_load_prior_curve_fit_state``); a stale copy would carry pre-refit values
    for refitted spectra. Counts are recomputed from ``series_results`` so the
    re-write stays correct regardless of caller.
    """
    output_dir = Path(output_dir)
    rows = [r for r in series_results if isinstance(r, dict)]
    num_spectra = len(rows)
    successful = sum(1 for r in rows if r.get("success"))
    flagged_count = sum(1 for r in rows if r.get("flagged"))
    serializable_results = [
        {k: v for k, v in r.items() if k not in ("visualization_bytes", "_winning_config")}
        for r in rows
    ]
    payload = {
        "timestamp": datetime.now().isoformat(),
        "total_spectra": num_spectra,
        "successful": successful,
        "flagged_count": flagged_count,
        "is_single_spectrum": state.get("is_single_spectrum", num_spectra <= 1),
        "series_metadata": state.get("series_metadata", {}),
        "quality_settings": quality_settings or {},
        "locked_config": state.get("locked_fitting_config"),
        "series_analysis_plan": state.get("series_analysis_plan"),
        "locked_preprocessing_strategy": state.get("locked_preprocessing_strategy"),
        "results": serializable_results,
    }
    results_path = output_dir / "series_fit_results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(results_path)


class UnifiedSeriesProcessingController:
    """
    Processes ALL spectra using the locked fitting model.

    Quality control features:
    - Verification loop iterates with adaptive constraint annealing.  Patience
      counter + iteration floor guarantee escalation to the hot annealing level
      so the LLM can restructure the model from scratch when local refits stall.
    - End-of-loop judge weighs all attempts on physics + R² when the verifier
      kept rejecting the high-water best.
    - If still inadequate and human feedback enabled, asks for guidance.
    - Otherwise proceeds with best available fit.
    - For series: detects statistical outliers that may indicate interesting physics
    """
    
    MAX_ATTEMPTS = 5
    DEFAULT_R2_THRESHOLD = 0.95
    DEFAULT_MAX_MODEL_RETRIES = 1
    DEFAULT_OUTLIER_SIGMA = 2.0
    DEFAULT_MAX_VERIFICATION_ITERATIONS = 7
    # Maximum width (in R² units) of the soft band below the acceptance
    # threshold.  A flat 0.05 doesn't scale: at r2_threshold=0.999 it
    # would put the hard-reject floor at 0.949, making 50× the gap-to-
    # perfection eligible for soft-band acceptance.  See _r2_soft_margin.
    SOFT_BAND_MAX_WIDTH = 0.05

    @classmethod
    def _r2_soft_margin(cls, r2_threshold: float) -> float:
        """Width of the soft band below the acceptance threshold.

        Width is capped at ``SOFT_BAND_MAX_WIDTH`` (0.05 — preserves
        backward-compatible behavior at the default threshold of 0.95)
        and at ``1 - r2_threshold`` (the gap to perfection — keeps the
        band narrow when the user demands very high R², e.g. 0.999 for
        XRD).  Always non-negative.
        """
        return max(min(cls.SOFT_BAND_MAX_WIDTH, 1.0 - r2_threshold), 0.0)

    JUDGE_PROMPT = '''You are a scientific data fitting expert acting as a judge.

Multiple fitting attempts were made but none passed automated verification. 
Review all attempts and select the most physically reasonable fit, or declare all unacceptable.

**SELECTION CRITERIA:**
1. Physical plausibility - are the model parameters reasonable for this type of data?
2. Residual structure - random noise is good, systematic patterns are bad
3. Component necessity - each component should fit a real feature in the data, not noise or baseline artifacts
4. Parsimony - prefer simpler models if fit quality is similar

**ATTEMPTS:**
{attempts_summary}

**VISUALIZATIONS:**
(See images below for each attempt)

Examine each fit carefully. Look at:
- Whether the model captures the key features in the data
- Whether component parameters are physically reasonable
- Whether residuals show random scatter or systematic patterns
- Whether any components appear to be fitting noise rather than real features

**Return JSON:**
{{
    "selected_index": <1, 2, 3, etc. matching the Attempt numbers above, or null if ALL are unacceptable>,
    "acceptable": true/false,
    "reasoning": "detailed explanation of your choice or why all are unacceptable",
    "issues_with_selected": "any remaining concerns with the chosen fit, or null if none"
}}

IMPORTANT: If one fit is clearly better than others (better residuals, more physical parameters),
select it even if it's not perfect. Only return acceptable=false if ALL fits are fundamentally flawed.
'''

    BEST_OF_N_JUDGE_PROMPT = '''You are a scientific data fitting expert selecting the best result among {num_candidates} independent fitting runs of the SAME data under the SAME fitting plan.

The runs differ only by sampling randomness in code generation and refinement.
Each completed its own verification loop and passed the R² gate. Select the
run whose RESULT is best.

## Candidates
{candidates_formatted}

The original data plot is attached first, followed by each candidate's fit
visualization in order.

## Selection Criteria
R² is objective, but a marginally higher R² does NOT automatically win —
inspect the fit plots:
1. Physical plausibility — are the model and its parameters reasonable for
   this type of data? A higher R² achieved by an unphysical model or by
   fitting noise/baseline artifacts loses to a slightly lower R² from a
   physically sound fit.
2. Residual structure — random scatter is good; systematic patterns mean the
   model misses real features regardless of R².
3. Parsimony — when fits are comparable, prefer the simpler model and the run
   with fewer verification iterations (it stayed closer to the planned model).

**Return JSON:**
{{
    "selected_index": <0-based index of the best run>,
    "reasoning": "Brief comparison: why this run's fit is best and what the others got wrong"
}}
'''

    HUMAN_FEEDBACK_PROMPT = '''## Fit Quality Issue

The automated fitting could not achieve adequate fit quality.

**Best Result:** R² = {best_r2:.4f} (threshold: {threshold})
**Models Tried:**
{models_tried}

**Options:**
1. Suggest a different model or approach
2. Adjust the R² threshold for this analysis (e.g., "threshold {example_threshold:.2f}")
3. Accept the best available fit (type "accept")

Your guidance: '''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
        plot_fn: Callable,
        r2_threshold: float = None,
        max_model_retries: int = None,
        enable_human_feedback: bool = False,
        outlier_sigma: float = None,
        max_verification_iterations: int = None,
        conformance_instructions: str = "",
        parallel_workers: Optional[int] = None,
        replanner: Any = None,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.script_instructions = script_instructions
        self.correction_instructions = correction_instructions
        self.quality_instructions = quality_instructions
        self.output_dir = Path(output_dir)
        self.plot_fn = plot_fn
        # Planning controller used to give each best-of-N fan-out candidate
        # (>=1) its OWN independent fitting plan (ensemble diversity; most
        # valuable for skill-less curves). None -> candidates share the plan.
        self.replanner = replanner
        self.r2_threshold = r2_threshold if r2_threshold is not None else self.DEFAULT_R2_THRESHOLD
        # Vestigial: the alternative-models loop was removed in favor of
        # patience-counter-driven hot annealing inside the verification
        # loop.  Parameter is accepted for backward compatibility but no
        # longer affects behavior.
        self.max_model_retries = max_model_retries if max_model_retries is not None else self.DEFAULT_MAX_MODEL_RETRIES
        self.enable_human_feedback = enable_human_feedback
        self.outlier_sigma = outlier_sigma if outlier_sigma is not None else self.DEFAULT_OUTLIER_SIGMA
        self.max_verification_iterations = max_verification_iterations if max_verification_iterations is not None else self.DEFAULT_MAX_VERIFICATION_ITERATIONS
        # Non-anchor parallel fan-out. Defaults to 1 (serial, byte-identical
        # to pre-feature behavior). Anchor processing always runs serially.
        self.parallel_workers = _resolve_parallel_workers(parallel_workers)
        self.conformance_instructions = conformance_instructions

    def _extract_extra_operands(self, state: dict, data_path: str) -> dict:
        """Per-spectrum extra columns the planner flagged for use (Phase 2).

        Reloads the raw file (or the retained array for array/DataFrame input),
        slices each resolved extra column into a 1-D array aligned to the data
        rows, and returns ``{canonical_filename: array}`` to stage as operands.
        """
        mapping = state.get("column_mapping_locked")
        if not mapping or not mapping.get("extras_resolved"):
            return {}
        raw = None
        p = Path(data_path)
        if p.exists() and p.is_file():
            try:
                from ....skills._shared.curve_fitting_tools import load_curve_data
                raw = np.asarray(load_curve_data(str(p), auto_orient=False))
            except Exception:  # noqa: BLE001
                raw = None
        if raw is None and state.get("raw_first_spectrum_full") is not None:
            raw = np.asarray(state["raw_first_spectrum_full"])
        if raw is None or raw.ndim != 2:
            return {}
        if raw.shape[0] < raw.shape[1]:          # orient to (n_points, n_cols)
            raw = raw.T
        operands = {}
        for e in mapping["extras_resolved"]:
            i = e["index"]
            if i < raw.shape[1]:
                operands[_operand_filename(e["name"])] = raw[:, i]
        return operands

    def _extra_operand_block(self, state: dict) -> str:
        """Codegen-prompt description of the staged extra-column operands."""
        mapping = state.get("column_mapping_locked")
        if not mapping or not mapping.get("extras_resolved"):
            return ""
        lines = []
        for e in mapping["extras_resolved"]:
            use = e.get("use") or e.get("role") or "an additional measured column"
            lines.append(f"- `{_operand_filename(e['name'])}` — column "
                         f"\"{e['name']}\"; intended use: {use}")
        return (
            "\n**Per-point operand arrays the planner SELECTED from the same file** "
            "(1-D, aligned to the data rows, in the working directory):\n"
            + "\n".join(lines) +
            "\n- These were chosen deliberately. Load each with `np.load` and incorporate "
            "it into the fit as its intended use describes; do not ignore a provided "
            "operand or substitute your own assumption for what it provides. Skip one "
            "only if it is genuinely unusable, and say why in the summary.\n"
        )

    def _generate_fitting_script(
        self,
        state: dict,
        data_path: str,
        stats: dict,
        prior_script: Optional[str] = None,
        prior_r2: float = 0.0,
        prior_issues: Optional[list] = None,
        extra_operand_block: str = "",
    ) -> str:
        config = state.get("locked_fitting_config", {})
        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])
        # Codegen recipe from ALL co-active skills (not just the top-ranked):
        # with several skills active each may own a different pipeline stage,
        # so none is dropped. Single-skill output is unchanged. Prefers each
        # skill's `implementation` section over its `analysis` synonym.
        recipes = _collect_codegen_recipe(state)
        if recipes:
            level = state.get("_annealing_level", 0)
            preamble = self._SKILL_STRICTNESS_SCHEDULE[
                min(level, len(self._SKILL_STRICTNESS_SCHEDULE) - 1)
            ].format(name=", ".join(n for n, _ in recipes))
            context_parts.append(preamble + _render_codegen_recipe(recipes))
        prior_runs = _prior_curve_fit_block(state)
        if prior_runs:
            context_parts.append(prior_runs)

        # Optional auxiliary operand(s) (#226): for each 1D auxiliary curve aligned
        # with the primary (same length), write it next to the spectrum and list
        # it in a manifest the generated script MAY use (e.g. baseline subtraction,
        # reference division). Misaligned ones stay context-only (no resampling in
        # v1) — their rendered plot still reaches the planning/interpretation LLM.
        operand_lines = []
        for j, it in enumerate(state.get("auxiliary_items") or []):
            arr = it.get("array")
            axis = it.get("axis")
            label = it.get("label") or f"reference_{j}"
            if arr is None or axis is None:
                continue
            arr = np.asarray(arr)
            axis = np.asarray(axis)
            if arr.ndim == 1 and arr.shape[0] == stats["n_points"]:
                safe = _sanitize_aux_name(label, j)
                aux_path = self.output_dir / f"temp_auxiliary_{safe}.npy"
                # Atomic: best-of-N attempts stage this concurrently.
                atomic_np_save(aux_path, np.column_stack([axis, arr]))
                operand_lines.append(
                    f"- \"{label}\": `{aux_path}` — a 2-column [x, y] array, "
                    f"{arr.shape[0]} points, same x-axis as the primary "
                    f"(x range [{float(np.nanmin(axis)):.6g}, {float(np.nanmax(axis)):.6g}])."
                )
                self.logger.info(
                    f"🧩 Offering auxiliary '{label}' ({arr.shape[0]} pts) as an "
                    f"optional fit-script operand."
                )
            else:
                self.logger.info(
                    f"Auxiliary '{label}' not aligned with the primary "
                    f"({getattr(arr, 'shape', None)} vs {stats['n_points']} pts); "
                    f"kept as context only (not a fit-script operand)."
                )

        auxiliary_block = ""
        if operand_lines:
            auxiliary_block = (
                "\n**Optional reference/baseline operand(s):**\n"
                + "\n".join(operand_lines)
                + "\n- You MAY load any of these and use it numerically — e.g. "
                "subtract or divide its y-column from the primary — ONLY if your "
                "method needs it (background/baseline removal, normalization). The "
                "primary data is the base input; references are optional, never "
                "required. Do NOT report findings about a reference as if it were a "
                "measurement; it is an operand for transforming the primary.\n"
            )
        # Phase 2: extra columns from the same file (e.g. an uncertainty column
        # the planner flagged), staged per-spectrum as canonical operand files.
        auxiliary_block += extra_operand_block

        prompt = self.script_instructions.format(
            analysis_approach=config.get("analysis_approach", "Fit the data"),
            physical_model=config.get("physical_model", "Appropriate model"),
            parameters_to_extract=", ".join(config.get("parameters_to_extract", [])) or "relevant parameters",
            fitting_strategy=config.get("fitting_strategy", "Standard fitting"),
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=data_path,
            n_points=stats["n_points"],
            x_min=stats["x_range"][0],
            x_max=stats["x_range"][1],
            y_min=stats["y_range"][0],
            y_max=stats["y_range"][1],
            auxiliary_block=auxiliary_block,
            tool_inventory=_tool_inventory_text(state),
        )

        if prior_script:
            issues_text = "  - (no specific issues recorded)"
            if prior_issues:
                issues_lines = []
                for issue in prior_issues:
                    if isinstance(issue, dict):
                        loc = issue.get("location", "")
                        prob = issue.get("problem", "")
                        fix = issue.get("suggested_fix", "")
                        bullet = prob or fix or str(issue)
                        if loc:
                            bullet = f"{loc}: {bullet}"
                        if fix and fix not in bullet:
                            bullet = f"{bullet} — fix: {fix}"
                        issues_lines.append(f"  - {bullet}")
                    else:
                        issues_lines.append(f"  - {issue}")
                if issues_lines:
                    issues_text = "\n".join(issues_lines)

            prompt += (
                "\n\n## REFINEMENT MODE\n"
                f"A previous fitting attempt produced the script below "
                f"(R² = {prior_r2:.4f}). The verifier rejected it for these reasons:\n"
                f"{issues_text}\n\n"
                "Adapt this script to the (possibly updated) locked plan above. "
                "Preserve working scaffolding (data loading, output paths, numpy "
                "formatting, 1D-vs-2D handling, FIT_RESULTS_JSON output, and the "
                "fit.npy save if present) verbatim; "
                "only modify the model components, initial guesses, bounds, or "
                "background treatment needed to address the issues. Do NOT "
                "regenerate from scratch. If the RESIDUAL DIAGNOSTICS flagged a "
                "localized region with RMS far above noise and repeated "
                "sign-changes, treat that as under-resolved real structure there "
                "(add a physically-nameable component or fix the peak shape), not "
                "as noise.\n\n"
                "```python\n"
                f"{prior_script}\n"
                "```\n"
            )

        response = self.model.generate_content(prompt)
        result, error = parse_codegen_response(response, field="script", logger=self.logger)

        if error or not result or "script" not in result:
            raise ValueError(f"Script generation failed: {error or 'no script'}")

        return result["script"]

    def _correct_script(self, state: dict, script: str, error_msg: str) -> tuple[str, str]:
        """Return ``(corrected_script, diagnosis)``."""
        config = state.get("locked_fitting_config", {})
        prompt = self.correction_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            physical_model=config.get("physical_model", ""),
            failed_script=script,
            error_message=error_msg,
            tool_inventory=_tool_inventory_text(state),
        )
        # Codegen recipe from ALL co-active skills (see _generate_fitting_script).
        recipes = _collect_codegen_recipe(state)
        if recipes:
            level = state.get("_annealing_level", 0)
            preamble = self._SKILL_STRICTNESS_SCHEDULE[
                min(level, len(self._SKILL_STRICTNESS_SCHEDULE) - 1)
            ].format(name=", ".join(n for n, _ in recipes))
            prompt += "\n\n" + preamble + _render_codegen_recipe(recipes)

        response = self.model.generate_content(prompt)
        result, error = parse_codegen_response(response, field="script", logger=self.logger)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        diagnosis = result.get("diagnosis", "")
        if diagnosis:
            self.logger.info(f"    Diagnosis: {diagnosis}")

        return result["script"], diagnosis

    def _check_plan_conformance(self, state: dict, script: str) -> dict | None:
        """Use the LLM to verify a generated script implements the locked plan.

        Returns a dict with ``conformant``, ``justified_deviations``,
        ``unjustified_deviations``, and ``summary`` keys, or ``None`` if the
        check cannot be performed (missing config, LLM error, etc.).
        """
        config = state.get("locked_fitting_config", {})
        if not config or not config.get("physical_model"):
            return None
        if not self.conformance_instructions:
            return None

        # Build skill rules text for conformance checking
        skill_rules_text = ""
        skill_sections = state.get("skill_sections")
        if skill_sections:
            skill_name = state.get("skill_name", "domain skill")
            rules_parts = []
            for stage in ("planning", "analysis", "validation"):
                content = skill_sections.get(stage, "")
                if content:
                    rules_parts.append(f"### {stage.title()} rules\n{content}")
            if rules_parts:
                skill_rules_text = (
                    f"\n**MANDATORY Domain Skill Rules ({skill_name}):**\n"
                    + "\n".join(rules_parts)
                    + "\n"
                )

        prompt = self.conformance_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            physical_model=config.get("physical_model", ""),
            parameters_to_extract=", ".join(
                config.get("parameters_to_extract", [])
            ),
            fitting_strategy=config.get("fitting_strategy", ""),
            skill_rules=skill_rules_text,
            script=script,
        )

        try:
            response = self.model.generate_content(contents=[prompt])
            result, error = self._parse(response)
            if error or not result:
                self.logger.debug(
                    "Plan conformance check parse failed: %s", error
                )
                return None
            return result
        except Exception as exc:
            self.logger.debug("Plan conformance check failed: %s", exc)
            return None


    def _compute_statistics(self, curve_data: np.ndarray) -> dict:
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")

        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    def _load_curve_data(self, data_path: str, column_mapping: dict = None) -> np.ndarray:
        """Load curve data from file, handling various formats. When a locked
        ``column_mapping`` is given (>2-col inputs), it selects the LLM-chosen
        X/Y columns; otherwise the deterministic heuristic applies."""
        # Try using the project's load_curve_data function first
        try:
            from ....skills._shared.curve_fitting_tools import load_curve_data
            if column_mapping:
                return load_curve_data(
                    data_path,
                    system_info={"x_column": column_mapping.get("x_index"),
                                 "y_column": column_mapping.get("y_index")},
                    column_names=column_mapping.get("names"))
            return load_curve_data(data_path)
        except ImportError:
            pass
        
        
        # Fallback: handle common formats manually
        if data_path.endswith('.npy'):
            return np.load(data_path)
        elif data_path.endswith('.csv'):
            # Try to load CSV, handling potential headers
            try:
                # First try without header
                return np.loadtxt(data_path, delimiter=',')
            except ValueError:
                # If that fails, try skipping first row (header)
                try:
                    return np.loadtxt(data_path, delimiter=',', skiprows=1)
                except ValueError:
                    # Try pandas as last resort for complex CSVs
                    try:
                        import pandas as pd
                        df = pd.read_csv(data_path)
                        return df.values.T  # Transpose to get (2, n_points) shape
                    except ImportError:
                        # Re-raise the original error if pandas not available
                        raise ValueError(f"Could not parse CSV file: {data_path}. "
                                        "File may have headers or non-numeric data.")
        elif data_path.endswith('.txt'):
            # Try common text formats
            try:
                return np.loadtxt(data_path)
            except ValueError:
                try:
                    return np.loadtxt(data_path, skiprows=1)
                except ValueError:
                    # Try tab-delimited
                    try:
                        return np.loadtxt(data_path, delimiter='\t', skiprows=1)
                    except:
                        raise ValueError(f"Could not parse text file: {data_path}")
        else:
            # Generic attempt
            try:
                return np.loadtxt(data_path)
            except ValueError:
                return np.loadtxt(data_path, skiprows=1)

    def _fit_single_spectrum(
        self,
        state: dict,
        curve_data: np.ndarray,
        data_path: str,
        spectrum_name: str,
        spectrum_idx: int,
        base_script: Optional[str] = None,
        refine_from_script: Optional[str] = None,
        refine_from_r2: float = 0.0,
        refine_from_issues: Optional[list] = None,
    ) -> dict:
        stats = self._compute_statistics(curve_data)
        # Per-spectrum working dir: the locked script runs VERBATIM here with data
        # staged as the canonical DATA_NAME and viz written canonically — no
        # per-spectrum source rewriting, no cross-item glob hazard.
        # Best-of-N anchor attempts nest under _candidates/cand_NN so concurrent
        # attempts never share a working dir.
        output_prefix = f"spectrum_{spectrum_idx:04d}"
        item_dir = self.output_dir / output_prefix
        candidate_subdir = state.get("_candidate_subdir")
        if candidate_subdir:
            item_dir = item_dir / candidate_subdir

        # Phase 2: extra columns the planner flagged (e.g. an uncertainty column)
        # are staged per-spectrum as canonical operand files and described to the
        # codegen LLM, which decides how to use them (e.g. weighted least-squares).
        extra_operands = self._extract_extra_operands(state, data_path)
        operand_block = self._extra_operand_block(state)

        script = None
        last_error = ""
        run = None
        script_errors: list[dict] = []

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                if base_script is not None and attempt == 1:
                    script = base_script   # reuse VERBATIM (loads DATA_NAME from cwd)
                elif attempt == 1:
                    script = self._generate_fitting_script(
                        state, DATA_NAME, stats,
                        prior_script=refine_from_script,
                        prior_r2=refine_from_r2,
                        prior_issues=refine_from_issues,
                        extra_operand_block=operand_block,
                    )
                    if not script_uses_canonical_input(script):
                        last_error = (
                            f"Script must load the data from '{DATA_NAME}' in the "
                            "current working directory (np.load), not another path."
                        )
                        continue
                    # Check conformance with locked plan on fresh generation
                    conformance = self._check_plan_conformance(state, script)
                    if conformance and not conformance.get("conformant", True):
                        issues = "; ".join(
                            conformance.get("unjustified_deviations", [])
                        )
                        self.logger.warning(
                            "    \u26a0\ufe0f Plan conformance issue: %s", issues
                        )
                        last_error = (
                            "PLAN CONFORMANCE: Script deviates from the "
                            "locked plan without justification. Issues: "
                            f"{issues}. Plan model: "
                            f"{state.get('locked_fitting_config', {}).get('physical_model', '')}. "
                            "Either fix the script to match the plan, or if "
                            "the plan cannot work, implement the closest "
                            "viable alternative and explain why in the summary."
                        )
                        continue
                    if conformance and conformance.get("justified_deviations"):
                        self.logger.info(
                            "    \u2139\ufe0f Justified plan deviations: %s",
                            "; ".join(conformance["justified_deviations"]),
                        )
                else:
                    script, diagnosis = self._correct_script(state, script, last_error)
                    script_errors.append({"error": last_error, "diagnosis": diagnosis})

                state["_verify_working_dir"] = str(item_dir)
                run = stage_and_run(self.executor, script, curve_data, item_dir,
                                    aux=extra_operands)
                exec_result = run["exec"]

                if run["status"] == "success":
                    has_fit_results = "FIT_RESULTS_JSON:" in run["stdout"]
                    has_visualization = run["visualization_path"] is not None

                    if has_fit_results and has_visualization:
                        break
                    else:
                        missing = []
                        if not has_fit_results:
                            missing.append("FIT_RESULTS_JSON output")
                        if not has_visualization:
                            missing.append("visualization file")
                        last_error = (
                            f"Script executed but did not produce expected outputs. "
                            f"Missing: {', '.join(missing)}. The script must print "
                            f"'FIT_RESULTS_JSON:{{...}}' with fit results and save "
                            f"'visualization.png' in the working directory."
                        )
                        self.logger.warning(f"    ⚠️ Attempt {attempt}: Script ran but missing outputs: {', '.join(missing)}")
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(f"    ⚠️ Attempt {attempt} failed: {last_error[:100]}")
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"    ❌ Attempt {attempt} error: {e}")

        # Success iff the final run produced BOTH the marker and the viz (matches
        # the break condition); run["status"] alone is a snapshot and can be
        # "success" even when outputs are missing.
        ok = (run is not None and run["status"] == "success"
              and run["visualization_path"] is not None
              and "FIT_RESULTS_JSON:" in run["stdout"])
        if not ok:
            return {
                "index": spectrum_idx,
                "name": spectrum_name,
                "success": False,
                "error": last_error,
                "parameters": {},
                "fit_quality": {},
                "script": script,
                "script_errors": script_errors,
            }

        fit_results = _parse_script_markers(run["stdout"])

        # Best-effort residual diagnostics from the saved fitted curve (vision aid):
        # reliable per-region structure metrics the verifier can reason over instead
        # of eyeballing a dynamic-range-crushed plot. Skipped silently if fit.npy
        # is absent (older/refit scripts) so this never breaks the fit path.
        fit_quality = dict(fit_results.get("fit_quality", {}) or {})
        residual_diag = None
        residual_zoom_panels = []
        try:
            fit_path = Path(item_dir) / FIT_NAME
            if fit_path.exists():
                cd = np.asarray(curve_data, float)
                if cd.ndim == 2 and cd.shape[1] >= 2:
                    xx, yy = cd[:, 0], cd[:, 1]
                else:
                    yy = cd.ravel()
                    xx = np.arange(yy.shape[0], dtype=float)
                fit_arr = np.load(fit_path)
                # The generated script can save fit.npy in a different x-ordering
                # than the data it was given (NMR ppm is usually DESCENDING, but
                # a script that sorts ascending for fitting saves the fit on that
                # ascending grid). Pairing fit.npy[i] with data[i] is then
                # reversed — corrupting the residual diagnostics below and the
                # saved fit.npy artifact. Detect the reversal against the data and
                # realign (and re-save the corrected fit.npy). Length/other
                # mismatches are left untouched.
                if fit_arr.shape == yy.shape:
                    fwd = _canonical_r2(yy, fit_arr)
                    rev = _canonical_r2(yy, fit_arr[::-1])
                    if rev is not None and (fwd is None or rev > fwd + 0.05):
                        fit_arr = np.ascontiguousarray(fit_arr[::-1])
                        try:
                            np.save(fit_path, fit_arr)
                        except Exception:
                            pass
                residual_diag = _residual_diagnostics(xx, yy, fit_arr)
                # Zoomed, locally-rescaled views of the flagged regions so the
                # verifier can SEE unmodeled fine structure (e.g. crystal-field
                # sub-peaks) the full-range plot squashes. x-axis is the true
                # data axis so seed positions it suggests are correct.
                residual_zoom_panels = _render_region_zoom_panels(
                    xx, yy, fit_arr, residual_diag)

                # Trust the saved fit over a broken self-reported R². The
                # self-report is computed inside the (LLM-generated) script and
                # can diverge from the curve it actually saved/plotted. We only
                # override UPWARD — a recompute higher than the self-report means
                # the saved fit is genuinely better than the script claimed. A
                # *lower* recompute is left alone: it usually means a deliberate
                # windowed/partial fit, where the script's own (windowed) number
                # is the meaningful one. None (length mismatch / no signal) also
                # keeps the self-report.
                recomputed_r2 = _canonical_r2(yy, fit_arr)
                self_r2 = fit_quality.get("r_squared")
                if recomputed_r2 is not None:
                    if isinstance(self_r2, (int, float)) and abs(recomputed_r2 - self_r2) > 0.05:
                        self.logger.info(
                            f"   ⚠️  R² from saved fit ({recomputed_r2:.4f}) "
                            f"diverges from self-reported ({self_r2:.4f})."
                        )
                    if self_r2 is None or recomputed_r2 > self_r2:
                        if isinstance(self_r2, (int, float)):
                            fit_quality["r_squared_self_reported"] = self_r2
                        fit_quality["r_squared"] = recomputed_r2
        except Exception:
            residual_diag = None

        return {
            "index": spectrum_idx,
            "name": spectrum_name,
            "data_path": data_path,
            "success": True,
            "error": None,
            "model_type": fit_results.get("model_type"),
            "parameters": fit_results.get("parameters", {}),
            "fit_quality": fit_quality,
            "deviation_note": fit_results.get("deviation_note") or fit_results.get("summary"),
            "visualization_path": run["visualization_path"],
            "visualization_bytes": run["visualization_bytes"],
            "residual_diagnostics": residual_diag,
            "residual_zoom_panels": residual_zoom_panels,
            "statistics": stats,
            "script": script,
            "script_errors": script_errors,
        }

    FIT_VERIFICATION_PROMPT = '''You are a scientific data analysis expert reviewing a curve/spectral fit.

**TASK:** Examine this fit visualization and determine if the fit is acceptable for scientific use.

**FIT STATISTICS:**
- R² = {r_squared:.4f}{metric_stat_line}
- Model: {model_type}
- Number of components: {n_components}

**FITTED PARAMETERS:**
{parameters}
{prior_best_section}{residual_diagnostics}
## STEP 1: CHECK FOR BROKEN FITS (reject immediately if ANY are true)

- **Wrong x-range?** Does the plot show a completely different x-range than where the model components are defined? (e.g., plot shows 135-200 but components are at 300, 520, 860) → REJECT
- **Featureless fit?** Is R² ≈ 1.0 but the plot shows only a simple line/curve with no actual data structure being fitted? → REJECT
- **RMSE ≈ 0 with trivial fit?** Near-zero error but no meaningful features captured suggests fitting wrong data subset → REJECT
- **Model components outside plot?** Legend shows components at positions not visible in the plotted x-range? → REJECT

If ANY box above is checked: set fit_acceptable: FALSE, explain the data range or data loading problem.

---

## STEP 2: IF STEP 1 PASSED, evaluate fit quality

The two {metric_label} thresholds form a **soft band** derived from the user's
configured acceptance target:
- **{accept_threshold:.2f}** = acceptance target ("accept floor")
- **{reject_threshold:.2f}** = hard-reject floor{reject_floor_note}

**Accept if:**
- {metric_label} {accept_cmp} {accept_threshold:.2f} AND residuals are mostly random noise AND main data features are captured

**Stop on plateau (convergence):** the PREVIOUS VERIFICATION ATTEMPTS section
below lists, per iteration, the metric that drives acceptance ({metric_label}) —
its value that step and the best-so-far. Track the best, not the latest (which can
regress). **Plateau = the last two iterations produced no new best**, where
"improvement" is judged relative to the accept threshold ({accept_threshold:.2f}):
once the best sits comfortably past the threshold, a change small compared to its
margin beyond the threshold does not count as a new best. When the best
{metric_label} is {accept_cmp} {accept_threshold:.2f} AND it has plateaued in this
sense, the fit has converged: set `fit_acceptable: true`, `recommended_action:
"none"`, and record any remaining residual concern in `overall_assessment` as a
caveat for the user, rather than continuing to refine.

**Reject if:**
- {metric_label} {reject_cmp} {reject_threshold:.2f} (hard-reject floor — numerical fit is too poor)
- Major systematic residual pattern across ENTIRE spectrum (any {metric_label})
- A prominent data feature is completely missed by the model (any {metric_label})
- **Under-resolved structure:** a *localized* region shows clearly systematic
  residuals — RMS well above the noise AND repeated sign-changes (an
  oscillation), at a named, visible spectral position — even if the rest of the
  spectrum fits well. Use the RESIDUAL DIAGNOSTICS block above (if present): a
  window with RMS ≫ noise and several sign-changes means the model is
  *under-resolving real, repeating structure* there (an unresolved component, or
  the wrong peak shape) — not random noise. When the data warrants it (and the
  active constraint level permits model changes), the fix may be to add a
  physically-nameable component or change the peak shape, not just retune.

**Soft band ({soft_band_desc}):**
- Numerical {metric_label} is borderline. Reject ONLY if you find concrete physics
  problems (systematic residuals, missing features, unphysical parameters).
  Don't reject solely because {metric_label} is in the band.
- When you reject in this band, **state the physics reason** in
  `overall_assessment` rather than just citing the {metric_label} number, so the
  trace is interpretable.

**Never claim {metric_label} is "below the threshold" unless the number truly is below
{accept_threshold:.2f}** — when you reject a fit whose {metric_label} is at or above the
accept floor, give only the physics reason for rejection (the systematic
residual, missed feature, or unphysical parameter), never the {metric_label} value, so the
report stays factually correct.

**Residual adequacy — the goal is residuals consistent with noise, i.e.
*structureless* (no coherent shape, trend, or repeated oscillation), NOT residuals
driven toward zero.** Once the residuals carry no systematic structure, the fit is
as good as the data supports: accept it, and do not add components or keep retuning
to shrink residual amplitude further (that is overfitting). It is the *structure*
of a residual, not its amplitude or σ-multiple, that signals a real deficiency —
reject only for a *structured* residual (a coherent local oscillation =
under-resolved structure per above, or a global trend) or a genuine physics defect.

For **count / shot-noise-limited data** (photon- or electron-counting — EELS, XPS,
XRD, raw spectroscopy counts), refine this further: the noise grows with the
signal (≈√counts), so a structured residual sitting on a tall, bright peak that is
only a fraction of a percent of the local signal is within counting statistics —
its large σ-multiple overstates it, so don't chase it with extra components. This
refinement applies ONLY to count data; for **constant-noise data** (normalized,
derivative, or processed signals with roughly uniform noise across the spectrum) a
structured many-σ residual is significant at any signal level — do not discount it.

**Do NOT reject for:**
- Ambiguous or subtle features — but distinguish "subtle" (small, noise-level,
  non-repeating) from "under-resolved" (localized, RMS ≫ noise, oscillating);
  the latter is a real defect per the bullet above, not a subtlety to wave off.
- Minor position offsets (<5%)
- Large parameter uncertainties (that's just uncertainty, not failure)
- "Could try different model" suggestions

---

## STEP 3: COMPARATIVE ASSESSMENT (only if a previous best fit was shown above)

If — and ONLY if — a "PREVIOUS BEST FIT" section was shown above the visualization,
also rate whether THIS fit is **physically better than the previous best**, on:
- Parameter values (closer to expected physics: ratios, splittings, widths in
  benchmark ranges)
- Component decomposition (fewer missing features, fewer spurious components)
- Residual quality (more random / less systematic)
- Convergence health (fewer parameters at bounds, fewer zero-error parameters)

Set **`physically_better_than_best: true`** if THIS fit improves on the previous
best in at least one of these dimensions WITHOUT introducing comparable new
problems. Set **`false`** otherwise — including when the two fits have similar
problems, or when this fit fixes one issue but breaks another.

If no previous best fit was shown, set `physically_better_than_best: false` (the
field is unused in that case).

---

## RESPONSE FORMAT

Return JSON:
{{
    "fit_acceptable": true/false,
    "issues_found": [
        {{
            "location": "where in the data",
            "problem": "what is wrong",
            "evidence": "what you see in the plot/residuals",
            "suggested_fix": "how to fix it"
        }}
    ],
    "spurious_components": ["list of components fitting noise, not real features"],
    "missing_features": ["list of obvious data features not captured by model"],
    "physically_better_than_best": true/false,
    "comparison_note": "one line on what improved or didn't (or 'N/A' if no previous best shown)",
    "overall_assessment": "one sentence summary",
    "recommended_action": "specific fix OR 'none'"
}}


Remember: Rejecting a good fit ({metric_label} {accept_cmp} {accept_threshold:.2f}) to chase marginal improvements often makes things WORSE through overfitting or convergence failures.
'''

    # Constraint annealing: gradually raise the "temperature" so the
    # verifier can explore more of the model space when early iterations
    # fail to find an adequate fit.  Like simulated annealing
    # (P ∝ exp(−ΔE/kT)), low T freezes the system to the locked plan
    # while high T lets it explore freely.
    _CONSTRAINT_ANNEALING_SCHEDULE = (
        # T=0  frozen: must stay within the locked model.
        "\n**Plan-aware constraint:**\n"
        "The fitting model is LOCKED by the analysis plan. "
        "Your suggested fixes must work within the current model — "
        "do not recommend changing the model itself.\n",
        # T=1  warm: prefer small deviations, but allow them.
        "\n**Plan-aware constraint (eased — earlier fixes did not resolve the issues):**\n"
        "Prefer the smallest change that could fix the remaining issues. "
        "If you believe a model change is necessary, suggest it, but explain "
        "why a parameter-level fix is insufficient.\n",
        # T=2  hot: full freedom, justify from data.
        "\n**Plan constraint (open):**\n"
        "Earlier iterations stayed within tighter model constraints. If the fit "
        "still needs work, you now have full freedom to suggest any change the data "
        "warrants, from small parameter adjustments to a completely different model; "
        "justify every deviation from what you observe in the data and residuals. "
        "This freedom does NOT oblige a change: if the best metric is already above "
        "the accept threshold and has plateaued, accept per the plateau rule instead "
        "of proposing further changes.\n",
    )

    # Same annealing applied to domain skill strictness during fitting.
    # Planning and interpretation stages always keep skills mandatory.
    _SKILL_STRICTNESS_SCHEDULE = (
        # T=0: mandatory
        "## MANDATORY Domain Skill Rules ({name})\n"
        "These rules encode validated domain expertise and take precedence "
        "over defaults.\n\n",
        # T=1: preferred
        "## Domain Skill Guidance ({name})\n"
        "Follow these rules unless the data clearly requires a different "
        "approach. If you deviate, explain why.\n\n",
        # T=2: reference
        "## Domain Skill Reference ({name})\n"
        "Use as context. Override any rule if the data warrants it — "
        "explain the deviation.\n\n",
    )

    def _verify_fit_with_llm(self, state: dict, fit_result: dict, history: List[dict] = None, verification_iter: int = 0, annealing_level: int | None = None, best_result: dict | None = None, best_verification: dict | None = None) -> Optional[dict]:
        """
        Use LLM to verify fit quality by examining the visualization.
        Returns verification result with any issues found, or None if verification fails.

        When ``best_result`` is supplied AND it is a different object than
        ``fit_result``, a "PREVIOUS BEST FIT" section is injected so the
        verifier can rate whether the current fit is physically better than
        the prior high-water mark.  ``best_verification`` (the verifier's
        last verdict on best) is used to summarize prior issues.

        Workflow-style skills whose gate sets ``physical_review=False`` (e.g.
        xrd's figure_of_merit) bypass this verifier — the skill's own scoring
        tools (e.g. score_xrd_match_robust) ARE the verification, and the
        goodness-of-fit-shaped prompt would not apply. Goodness-of-fit gates
        (r_squared, peak_region_r2, BIC, …) keep ``physical_review=True`` and
        run the verifier below, framed against the gate's own metric.
        """
        gate = _gate(state)
        if not gate.physical_review:
            value = gate.extract(fit_result.get("fit_quality"))
            # Canonical verdict schema — must match the keys downstream
            # consumers actually read (curve_fitting_controllers.py:2839, :3094
            # read `fit_acceptable`; :2871 reads `issues_found`; :2848 reads
            # `physically_better_than_best`). The earlier short-circuit
            # emitted `should_accept` / `issues`, which silently defaulted
            # downstream — non-R² gates were effectively inert. Reviewer
            # caught it on PR #193.
            if gate.is_accept(value):
                cmp = "≥" if gate.direction == "higher_is_better" else "≤"
                return {
                    "fit_acceptable": True,
                    "overall_assessment": (
                        f"Skill workflow gate satisfied: {gate.metric} = "
                        f"{value:.4f} {cmp} {gate.accept_threshold:.4f}. "
                        f"Curve-fit R² verifier bypassed for non-R² gates."
                    ),
                    "issues_found": [],
                    "recommended_action": "none",
                    "physically_better_than_best": False,
                    "comparison_note": "N/A — non-R² gate (no prior-best comparison)",
                }
            elif gate.is_hard_reject(value):
                value_str = f"{value:.4f}" if isinstance(value, (int, float)) else "missing"
                return {
                    "fit_acceptable": False,
                    "overall_assessment": (
                        f"Skill workflow gate hard-rejects: {gate.metric} = "
                        f"{value_str} vs hard-reject threshold "
                        f"{gate.hard_reject_threshold:.4f}."
                    ),
                    "issues_found": [{
                        "location": "Workflow scoring",
                        "problem": f"{gate.metric} below acceptable range",
                        "suggested_fix": (
                            "Re-plan the workflow — widen the database query, "
                            "broaden the chemistry hypothesis, or verify the "
                            "wavelength / experimental metadata."
                        ),
                    }],
                    "recommended_action": "retry_fitting_attempt_with_changes",
                    "physically_better_than_best": False,
                    "comparison_note": "N/A — non-R² gate (no prior-best comparison)",
                }
            else:
                # Marginal: between accept and hard-reject. Treat as
                # acceptable (don't trigger retry), but flag as marginal
                # in the verdict so the synthesis layer can qualify it.
                return {
                    "fit_acceptable": True,
                    "overall_assessment": (
                        f"Skill workflow gate marginal: {gate.metric} = "
                        f"{value:.4f}. Below accept threshold "
                        f"{gate.accept_threshold:.4f} but above hard-reject; "
                        f"synthesis will report as marginal."
                    ),
                    "issues_found": [{
                        "location": "Workflow scoring",
                        "problem": f"{gate.metric} marginal (below accept threshold)",
                        "suggested_fix": (
                            "Acceptable as-is; downstream synthesis will "
                            "qualify confidence as marginal."
                        ),
                    }],
                    "recommended_action": "none",
                    "physically_better_than_best": False,
                    "comparison_note": "N/A — non-R² gate (no prior-best comparison)",
                }
        if not fit_result.get("visualization_bytes"):
            self.logger.warning("      No visualization available for LLM verification")
            return None

        # Gather fit info
        r_squared = fit_result.get("fit_quality", {}).get("r_squared") or 0
        model_type = fit_result.get("model_type", "Unknown")
        parameters = fit_result.get("parameters", {})

        # Count components
        n_components = len(parameters) if isinstance(parameters, dict) else 0

        # Format parameters for prompt
        params_str = json.dumps(parameters, indent=2) if parameters else "No parameters extracted"

        # Build the comparative "previous best" context only when fit_result
        # is a different object than best_result.  When they are the same
        # (initial verification, or just-promoted refit), no comparison is
        # meaningful and we leave the section empty so the LLM ignores
        # STEP 3's comparative assessment.
        prior_best_section = ""
        if best_result is not None and best_result is not fit_result:
            best_r2 = best_result.get("fit_quality", {}).get("r_squared") or 0
            best_issues_lines = []
            if best_verification:
                for issue in (best_verification.get("issues_found") or [])[:6]:
                    if isinstance(issue, dict):
                        loc = issue.get("location", "")
                        prob = issue.get("problem", "")
                        bullet = f"{loc}: {prob}" if loc else prob
                        if bullet:
                            best_issues_lines.append(f"  - {bullet}")
            best_issues_text = (
                "\n".join(best_issues_lines)
                if best_issues_lines
                else "  - (no specific issues recorded)"
            )
            prior_best_section = (
                "\n**PREVIOUS BEST FIT (for comparative assessment):**\n"
                f"- R² = {best_r2:.4f}\n"
                f"- Issues the verifier flagged on the previous best:\n"
                f"{best_issues_text}\n"
            )

        # Frame the acceptance criterion against the GATE's metric, not always
        # R². For the r_squared gate this reproduces the previous wording
        # exactly (label "R²", the controller's r2_threshold + soft margin). For
        # a goodness-of-fit gate with a different metric (peak_region_r2, BIC, …)
        # the verifier judges that metric, with the gate's own thresholds and
        # comparison direction, and the metric value is surfaced in the stats.
        if gate.metric == "r_squared":
            metric_label = "R²"
            accept_thr = self.r2_threshold
            reject_thr = self.r2_threshold - self._r2_soft_margin(self.r2_threshold)
            metric_stat_line = ""
        else:
            gate_value = gate.extract(fit_result.get("fit_quality"))
            metric_label = gate.label
            accept_thr = gate.accept_threshold
            reject_thr = gate.hard_reject_threshold
            metric_stat_line = (
                f"\n- {metric_label} = {gate_value:.4f} (acceptance metric)"
                if isinstance(gate_value, (int, float)) else ""
            )
        # Direction-aware soft-band descriptor (byte-identical to the original
        # "{reject} ≤ R² < {accept}" for the higher-is-better r_squared path).
        if gate.direction == "higher_is_better":
            soft_band_desc = f"{reject_thr:.2f} ≤ {metric_label} < {accept_thr:.2f}"
        else:
            soft_band_desc = f"{accept_thr:.2f} < {metric_label} ≤ {reject_thr:.2f}"
        # The "(= accept floor − margin)" note is the R² path's original wording
        # (a fixed margin below accept). Keep it only for r_squared; omit it for
        # metrics where a subtractive margin is not the framing.
        reject_floor_note = (
            f" (= accept floor − {accept_thr - reject_thr:.2f})"
            if gate.metric == "r_squared" else ""
        )

        prompt_text = self.FIT_VERIFICATION_PROMPT.format(
            r_squared=r_squared,
            metric_label=metric_label,
            metric_stat_line=metric_stat_line,
            soft_band_desc=soft_band_desc,
            reject_floor_note=reject_floor_note,
            accept_cmp=gate.accept_cmp,
            reject_cmp=gate.reject_cmp,
            model_type=model_type,
            n_components=n_components,
            parameters=params_str,
            accept_threshold=accept_thr,
            reject_threshold=reject_thr,
            prior_best_section=prior_best_section,
            residual_diagnostics=_format_residual_diagnostics(
                fit_result.get("residual_diagnostics")
            ),
        )

        # Constraint annealing: use caller-supplied level (adaptive) or fall
        # back to the legacy iteration-proportional formula.
        schedule = self._CONSTRAINT_ANNEALING_SCHEDULE
        if annealing_level is not None:
            level = min(annealing_level, len(schedule) - 1)
        else:
            n_levels = len(schedule)
            max_iter = max(self.max_verification_iterations, 1)
            level = min(verification_iter * n_levels // max_iter, n_levels - 1)
        prompt_text += schedule[level]

        # Add history context
        history_context = build_verification_prompt_with_history(
            current_fit={
                "r_squared": r_squared,
                "model_type": model_type,
                "parameters": parameters,
            },
            previous_iterations=history or [],
        )

        prompt_parts = [
            prompt_text + history_context,
            "\n\n**FIT VISUALIZATION (examine carefully, especially the residual plot):**",
        ]
        
        # Add the actual fit visualization
        prompt_parts.append({
            "mime_type": "image/png", 
            "data": fit_result["visualization_bytes"]
        })
        
        # Preprocessing is now done INSIDE the fit script; when it preprocesses,
        # its visualization shows the raw data faintly behind the fitted data.
        # Always remind the verifier to check for preprocessing-induced
        # distortion (otherwise invisible because the fit is plotted against the
        # processed curve).
        prompt_parts.append(
            "\n\n**PREPROCESSING CHECK:** Any preprocessing is done inside the "
            "fit script. If the visualization shows a faint raw trace behind the "
            "fitted data, verify the preprocessing did not distort the fitted "
            "features (e.g. over-smoothing broadening a peak/linewidth, or a "
            "baseline removing real signal). If it did, add an issues_found entry "
            "with location 'preprocessing' and set recommended_action to 'none' — "
            "recorded as a caveat, not a refit trigger."
        )
        # Per-region zoom panels: the flagged residual windows rendered zoomed and
        # locally y-rescaled, so fine structure squashed on the full-range plot is
        # visible. This turns the residual-diagnostics "where" into a "what" the
        # model can see, and disambiguates add-a-component vs retune-the-shape.
        zoom_panels = fit_result.get("residual_zoom_panels") or []
        if zoom_panels:
            prompt_parts.append(
                "\n\n**RESOLVED RESIDUAL REGIONS** — each flagged window below is "
                "zoomed and y-rescaled to its local range (x-axis is the TRUE data "
                "axis). For each, look at the DATA (blue) vs FIT (red): if the data "
                "shows a maximum or shoulder the fit does NOT cover, the model is "
                "UNDER-RESOLVED there — ADD a component seeded at that x position "
                "(report the position in recommended_action). Only retune "
                "width/shape if the feature is already modelled. Do not treat a "
                "clearly-real maximum as noise."
            )
            for label, png in zoom_panels:
                prompt_parts.append(f"\n_{label}_")
                prompt_parts.append({"mime_type": "image/png", "data": png})

        # Original (raw) data for reference.
        if state.get("original_plot_bytes"):
            prompt_parts.append("\n\n**ORIGINAL (RAW) DATA for reference:**")
            prompt_parts.append({"mime_type": "image/png", "data": state["original_plot_bytes"]})

        # Scrutinize-don't-reimplement: when a registered curve-fitting tool
        # (e.g. fit_pattern, fit_sideband_manifold) produced the fit, judge it by
        # the tool's QC + domain knowledge + cross-checks, not by re-deriving.
        from ....skills._shared._registry import (
            VERIFIER_TOOL_SCRUTINY_PRINCIPLE, get_tools_for)
        _tool_inv = _tool_inventory_text(state)
        if _tool_inv:
            prompt_parts.append(
                "\n\n**REGISTERED TOOLS AVAILABLE TO THIS FIT** — judge the result "
                "against what each tool actually does and what its outputs mean; do not "
                "re-derive a failure mode the tool already controls for:\n" + _tool_inv)
            # Which of those tools THIS iteration's script actually called
            # (authoritative — the prose pipeline description can deviate from the
            # executed code). Parsed from the saved fitting script; only the name
            # list is injected, never the script source.
            try:
                import glob as _g
                _wd = state.get("_verify_working_dir")
                _names = [t.name for t in get_tools_for(
                    "curve_fitting", active_skills=_active_skill_names(state))]
                _src = ""
                if _wd:
                    _hits = (_g.glob(os.path.join(_wd, "scripts", "*.py"))
                             or _g.glob(os.path.join(_wd, "*.py")))
                    if _hits:
                        with open(_hits[0]) as _sf:
                            _src = _sf.read()
                import re as _re
                _used = [n for n in _names
                         if _re.search(rf"\b{_re.escape(n)}\b", _src)]
                state["_last_tools_used"] = _used   # persisted into quality_history
                if _used:
                    prompt_parts.append(
                        "\n\n**Registered tools this iteration's script actually CALLED:** "
                        + ", ".join(_used) + " — apply each one's documented behaviour "
                        "(above) when judging the result.")
            except Exception:
                pass
        prompt_parts.append("\n\n" + VERIFIER_TOOL_SCRUTINY_PRINCIPLE)

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      LLM verification parse failed: {error}")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"      LLM verification failed: {e}")
            return None

    def _apply_llm_verification_feedback(self, state: dict, verification: dict) -> dict:
        """
        Apply LLM verification feedback to refine the fitting configuration.
        Returns updated config.
        """
        config = (state.get("locked_fitting_config") or {}).copy()

        recommended_action = verification.get("recommended_action", "")
        if not recommended_action or recommended_action.lower() == "none":
            return config

        # Build a refinement prompt based on verification results
        issues_summary = []
        for issue in verification.get("issues_found", []):
            issues_summary.append(f"- {issue.get('location', 'Unknown')}: {issue.get('problem', '')} -> {issue.get('suggested_fix', '')}")

        spurious = verification.get("spurious_components", [])
        missing = verification.get("missing_features", [])

        # Inject the same constraint annealing directive so the refinement
        # LLM respects the current temperature level.
        annealing_level = state.get("_annealing_level", 0)
        schedule = self._CONSTRAINT_ANNEALING_SCHEDULE
        constraint_text = schedule[min(annealing_level, len(schedule) - 1)]

        refinement_prompt = f"""Refine the fitting approach based on automated verification feedback.

**CURRENT APPROACH:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}
{constraint_text}
**VERIFICATION FINDINGS:**
{chr(10).join(issues_summary) if issues_summary else 'No specific issues listed'}

**SPURIOUS COMPONENTS TO REMOVE:** {', '.join(spurious) if spurious else 'None identified'}

**MISSING FEATURES TO ADD:** {', '.join(missing) if missing else 'None identified'}

**RECOMMENDED ACTION:** {recommended_action}

Return JSON with the refined fitting approach:
{{
    "physical_model": "updated model description incorporating the fixes",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[refinement_prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      Could not parse refinement: {error}")
                return config
            
            # Update config with refinements
            config.update(result)
            return config
            
        except Exception as e:
            self.logger.error(f"      Refinement failed: {e}")
            config["_refinement_error"] = str(e)
            return config

    def _get_human_feedback_for_poor_fit(self, state: dict, best_result: dict, all_attempts: List[dict]) -> Optional[dict]:
        # No successful fit to review (every attempt failed → best_result is None).
        # Don't solicit poor-fit feedback on a non-existent fit; let the caller
        # fall through to graceful failure handling rather than dereferencing None.
        if not best_result:
            self.logger.warning("   All fitting attempts failed — no fit to review; skipping poor-fit feedback.")
            return None
        models_tried = "\n".join([f"  - {a['model']}: R² = {a['r2']:.4f}" for a in all_attempts])
        
        print("")
        print("=" * 60)
        print("⚠️  FIT QUALITY BELOW THRESHOLD")
        print("=" * 60)
        
        if best_result.get("visualization_bytes"):
            viz_path = self.output_dir / "quality_review_fit.png"
            with open(viz_path, 'wb') as f:
                f.write(best_result["visualization_bytes"])
            print(f"[Best fit visualization saved to: {viz_path}]")
        
        prompt = self.HUMAN_FEEDBACK_PROMPT.format(
            best_r2=best_result.get("fit_quality", {}).get("r_squared") or 0,
            threshold=self.r2_threshold,
            models_tried=models_tried,
            example_threshold=self.r2_threshold - self._r2_soft_margin(self.r2_threshold),
        )
        print(prompt)
        
        feedback = input("\nYour input: ").strip()
        
        if not feedback:
            print("No feedback provided. Proceeding with best available fit.")
            return None
        
        if "accept" in feedback.lower() or "proceed" in feedback.lower():
            print("✓ Accepting best available fit.")
            return None
        
        if "threshold" in feedback.lower():
            try:
                match = re.search(r'(\d+\.?\d*)', feedback)
                if match:
                    new_threshold = float(match.group(1))
                    if new_threshold <= 1.0:
                        print(f"✓ Adjusting threshold to {new_threshold}")
                        return {"action": "adjust_threshold", "new_threshold": new_threshold}
            except:
                pass
        
        print("🔄 Will retry with your suggested approach...")
        return {"action": "retry", "feedback": feedback}

    def _get_user_feedback_on_fit(self, state: dict, fit_result: dict, r2: float) -> Optional[str]:
        """
        Show user the first spectrum fit and ask for optional feedback.
        Returns feedback string if user wants changes, None if satisfied.
        """
        print("\n" + "=" * 70)
        print("📊 FIRST SPECTRUM FIT RESULT - Review Before Processing Series")
        print("=" * 70)
        
        # Save and display fit visualization path
        review_viz_path = None
        if fit_result.get("visualization_bytes"):
            review_viz_path = self.output_dir / "first_spectrum_fit_review.png"
            with open(review_viz_path, 'wb') as f:
                f.write(fit_result["visualization_bytes"])
            print(f"\n[Fit visualization saved to: {review_viz_path}]")
        
        # Show fit summary
        print(f"\n📈 Model: {fit_result.get('model_type', 'N/A')}")
        print(f"📊 R² = {r2:.4f} (threshold: {self.r2_threshold})")
        
        params = fit_result.get("parameters", {})
        if params:
            print("\n📋 Fitted Parameters:")
            for comp, values in params.items():
                if isinstance(values, dict):
                    print(f"   {comp}:")
                    for k, v in values.items():
                        if not k.endswith('_err'):
                            if isinstance(v, float):
                                print(f"      {k}: {v:.4g}")
                            else:
                                print(f"      {k}: {v}")
        
        num_spectra = state.get("num_spectra", 1)
        print(f"\n⚠️  This fitting model will be applied to all {num_spectra} spectra in the series.")
        print("\n" + "-" * 60)
        print("Options:")
        print("  • Press Enter to accept this fit and proceed with series")
        print("  • Type feedback to modify the fitting approach (e.g., 'add baseline', ")
        print("    'use Voigt instead of Gaussian', 'fit two peaks instead of one')")
        print("-" * 60)
        
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        # Clean up the review file - it's only for user viewing during this step
        if review_viz_path and review_viz_path.exists():
            try:
                os.remove(review_viz_path)
            except:
                pass
        
        if not feedback:
            print("✅ Fit accepted. Proceeding with series...")
            return None
        
        return feedback

    def _ask_keep_user_guided_fit(self, user_r2: float, original_r2: float) -> bool:
        """Ask user whether to keep the user-guided fit even if R² is worse."""
        print("\n" + "-" * 60)
        print(f"⚠️  User-guided fit has lower R² ({user_r2:.4f}) than original ({original_r2:.4f})")
        print("-" * 60)
        print("Options:")
        print(f"  • Type 'keep' to use the user-guided fit anyway (R² = {user_r2:.4f})")
        print(f"  • Press Enter to revert to original fit (R² = {original_r2:.4f})")
        
        response = input("\nYour choice: ").strip().lower()
        
        if response == 'keep':
            print("✅ Keeping user-guided fit.")
            return True
        else:
            print("✅ Reverting to original fit.")
            return False

    def _refine_model_from_feedback(self, state: dict, feedback: str) -> dict:
        # Persist the applied feedback so it survives to end-of-run (the staging
        # hook distills human corrections into skills). Feedback is otherwise
        # consumed transiently in-flight.
        if feedback:
            state.setdefault("human_feedback_log", []).append(str(feedback))
        config = state.get("locked_fitting_config", {})
        prompt = f"""Refine the fitting approach based on user feedback.

**Current Approach:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}

**User Feedback:** {feedback}

Return JSON with:
{{
    "physical_model": "updated model description",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            if error or not result:
                return config
            updated = config.copy()
            updated.update(result)
            return updated
        except Exception as e:
            self.logger.error(f"Failed to refine model from feedback: {e}")
            return config

    def _fit_with_quality_control(self, state: dict, curve_data: np.ndarray, data_path: str, spectrum_name: str, spectrum_idx: int, is_regime_anchor: bool = False, reuse_script: Optional[str] = None, reuse_source: Optional[str] = None) -> dict:
        """
        Fit a single spectrum with quality control, verification, and optional judge selection.

        #172 locked-script reuse: when ``reuse_script`` is supplied (an anchor
        fed a prior run's saved fitting script via ``prior_analysis_paths``),
        the prior script is run verbatim on the new data first. If it executes,
        its result is kept — regardless of R² — so the extracted-parameter
        schema stays consistent across an incremental measurement campaign by
        construction; R² is attached as a ``reuse_validity`` verdict the
        orchestrator can act on, not a gate that re-derives the model. Full QC
        re-derivation runs only when the prior script cannot execute at all.

        Flow:
        1. Initial fit attempt.
        2. For anchor spectrum (first in series or first in regime): LLM
           verification loop with adaptive constraint annealing.  Patience
           counter and iteration floor guarantee escalation to the hot
           annealing level when refits stall.  At hot, the LLM regenerates
           the script from scratch with full freedom to restructure the
           model — this subsumes the prior "alternative models" path.
        3. If human feedback enabled: allow user to guide refinement.
        4. Unified judge evaluates ALL attempts when verifier kept rejecting
           the high-water best (Option B threshold gating).
        5. Attach quality_history to result for downstream synthesis.
        """
        all_attempts = []
        verification_history = []
        best_result = None
        best_r2 = -1.0
        best_config = (state.get("locked_fitting_config") or {}).copy()
        # Option B gate: set to True if the verifier ever rejects best_result
        # without later approving it.  Drives the threshold short-circuit
        # at the post-loop checkpoint.
        best_ever_rejected = False

        # Anchor = first spectrum overall OR first in a regime; gets full QC
        _is_anchor = spectrum_idx == 0 or is_regime_anchor

        # --- #172: locked-script reuse fast path ---
        # A prior curve-fit run supplied via prior_analysis_paths means the new
        # data is point N+1 of that series: reuse the prior run's locked fitting
        # script verbatim instead of re-deriving the model. This keeps the
        # extracted-parameter schema consistent across an incremental campaign
        # by construction. R² is a validity *signal* (attached as reuse_validity
        # for the orchestrator), never a gate that re-derives the model — a
        # re-derived model could change the feature columns. The only fallback
        # to full QC is a prior script that cannot execute at all.
        if reuse_script and _is_anchor:
            self.logger.info(
                f"   ♻️  Reusing locked fitting script from prior run "
                f"'{reuse_source or 'prior'}'..."
            )
            reuse_result = self._fit_single_spectrum(
                state=state, curve_data=curve_data, data_path=data_path,
                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx,
                base_script=reuse_script,
            )
            if reuse_result.get("success"):
                reuse_r2 = (
                    reuse_result.get("fit_quality", {}).get("r_squared") or 0
                    or 0.0
                )
                verdict = "good" if reuse_r2 >= self.r2_threshold else "poor"
                if verdict == "good":
                    self.logger.info(
                        f"   ✅ Reused script fits well (R² = {reuse_r2:.4f} ≥ "
                        f"{self.r2_threshold:.3f}) — model re-derivation skipped"
                    )
                    message = (
                        f"Reused the locked fitting script from prior run "
                        f"'{reuse_source or 'prior'}'; R² = {reuse_r2:.4f} "
                        f"meets the acceptance threshold "
                        f"{self.r2_threshold:.3f}."
                    )
                else:
                    self.logger.warning(
                        f"   ⚠️  Reused script fits poorly (R² = "
                        f"{reuse_r2:.4f} < {self.r2_threshold:.3f}). Keeping "
                        f"the result to preserve feature-schema consistency; "
                        f"flagging it as low-confidence."
                    )
                    message = (
                        f"Reused the locked fitting script from prior run "
                        f"'{reuse_source or 'prior'}', but R² = "
                        f"{reuse_r2:.4f} is below the acceptance threshold "
                        f"{self.r2_threshold:.3f}. The new measurement may "
                        f"not belong to this series, or measurement "
                        f"conditions shifted. Extracted parameters are "
                        f"schema-consistent but should be treated as "
                        f"low-confidence."
                    )
                reuse_result["reuse_validity"] = {
                    "reused": True,
                    "source": reuse_source,
                    "r_squared": reuse_r2,
                    "threshold": self.r2_threshold,
                    "verdict": verdict,
                    "message": message,
                }
                if verdict == "poor":
                    reuse_result["quality_warning"] = message
                return reuse_result
            self.logger.warning(
                f"   ⚠️  Prior fitting script could not execute on this data "
                f"(even after correction). Falling back to full model "
                f"re-derivation — the extracted-feature schema may differ "
                f"from the prior run."
            )

        # --- Initial fit (annealing schedule starts here; default T=0) ---
        # A re-run may start the schedule HIGHER (e.g. hot) via
        # `_starting_annealing_level`, so it does not repeat early constraint
        # stages a prior run already found inadequate. Default 0 = unchanged.
        _start_level = max(0, min(int(state.get("_starting_annealing_level") or 0),
                                  len(self._CONSTRAINT_ANNEALING_SCHEDULE) - 1))
        state["_annealing_level"] = _start_level
        initial_model = state.get('locked_fitting_config', {}).get('physical_model') or 'Initial model'
        self.logger.info(f"   Attempt 1: {str(initial_model)[:80]}...")

        result = self._fit_single_spectrum(
            state=state, curve_data=curve_data, data_path=data_path,
            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
        )

        if result["success"]:
            r2 = result.get("fit_quality", {}).get("r_squared") or 0
            all_attempts.append({
                "model": initial_model, "r2": r2, "result": result,
                "config": (state.get("locked_fitting_config") or {}).copy(),
            })

            if r2 > best_r2:
                best_r2 = r2
                best_result = result
                best_config = (state.get("locked_fitting_config") or {}).copy()

            # --- Verification loop (for anchor spectra: first overall or first in regime) ---
            fit_was_approved = False
            if (_is_anchor and self.max_verification_iterations <= 0
                    and best_result and best_result.get("success")):
                # Explicit verification bypass (max_verification_iterations=0):
                # the caller asked for a fast / in-situ turnaround. Accept the
                # initial successful fit as-is, with no LLM verification or
                # refit loop. Only triggers at <= 0, so the default thorough
                # path (>= 1) is unaffected. A failed/degenerate initial fit
                # (no success) still falls through to the loop below for the
                # recovery path rather than locking garbage.
                self.logger.info(
                    f"   ⏩ Verification bypassed (max_verification_iterations=0); "
                    f"accepting initial fit (R² = {best_r2:.4f})")
                fit_was_approved = True
            elif _is_anchor:
                # Skip verification ONLY when there is no successful fit to work
                # with. A fit that executed but is degenerate (low/zero R²) must
                # still enter the loop: the verifier + adaptive annealing (which
                # reaches hot/fresh-generation) + residual diagnostics are the
                # recovery path for a "ran-but-garbage" fit. Previously a
                # `best_r2 < 0.1` clause skipped these cases, locking a degenerate
                # script with no recovery (and, in a series, reusing it for every
                # spectrum). Matches image-analysis, which gates on success only. (#245)
                if not best_result or not best_result.get("success"):
                    self.logger.warning(f"   Initial fit failed (no successful result, R²={best_r2:.4f}), skipping verification")
                else:
                    # Adaptive annealing state: start frozen, escalate via
                    # three complementary mechanisms so hot annealing
                    # (level n-1) is reliably reached when refits stall:
                    #   (a) rate-based escalation: improvement too slow to
                    #       reach threshold within the remaining iterations
                    #   (b) patience counter: N consecutive iterations with
                    #       best stuck → escalate (mirrors image-analysis
                    #       _PATIENCE = 2 at image_analysis_controllers.py:3001)
                    #   (c) iteration floor: floor(iter / floor_divisor) is
                    #       the minimum allowed level — guarantees we hit the
                    #       hot level by the end of the budget regardless of
                    #       what the rate/patience say.
                    _annealing_level = _start_level
                    # Annealing level at which the PREVIOUS refit actually ran.
                    # Used to detect the escalation INTO the hot level (see the
                    # fresh-generation trigger below). Must track the prior
                    # refit's level — not a same-iteration snapshot of
                    # _annealing_level — because escalation happens at the end of
                    # an iteration, so a start-of-iteration snapshot already
                    # equals the (escalated) current level and never registers a
                    # transition. Mirrors image_analysis's _previous_annealing_level.
                    # max(start-1,0): starting AT hot still registers the
                    # transition into hot (fires fresh generation); start at 0
                    # restores the original `= 0`.
                    _previous_annealing_level = max(_start_level - 1, 0)
                    _prev_best_r2 = best_r2
                    _n_anneal_levels = len(self._CONSTRAINT_ANNEALING_SCHEDULE)
                    _PATIENCE = 2
                    _stall_count = 0
                    # Floor divisor chosen so the loop reaches level n-1 by
                    # roughly the last third of the iteration budget.
                    _floor_divisor = max(self.max_verification_iterations // _n_anneal_levels, 1)

                    # current_result tracks the latest refit (what the verifier
                    # diagnoses next); best_result is the high-water mark used
                    # as the refinement anchor and final return value.
                    current_result = best_result
                    current_r2 = best_r2

                    # best_ever_rejected is initialized at function scope.
                    # Reset on promotion (new best hasn't been verified yet).
                    # Used to gate the threshold short-circuit so a high-R²
                    # but verifier-rejected best falls through to the
                    # end-of-loop judge.
                    best_verification = None  # last verifier verdict on best

                    # R² floor for "in-band" promotion on physics grounds.
                    # Catastrophic regressions (script bugs, complete failure)
                    # are always rejected; small dips are admissible if the
                    # verifier signals physical improvement.
                    R2_FLOOR = max(self.r2_threshold - self._r2_soft_margin(self.r2_threshold), 0.0)

                    for verification_iter in range(self.max_verification_iterations):
                        self.logger.info(f"   Verification {verification_iter + 1}/{self.max_verification_iterations} (annealing level {_annealing_level})...")

                        # Pass best_result for comparative assessment.  The
                        # verifier emits physically_better_than_best only when
                        # current and best are different objects.
                        verification = self._verify_fit_with_llm(
                            state, current_result,
                            history=verification_history,
                            verification_iter=verification_iter,
                            annealing_level=_annealing_level,
                            best_result=best_result,
                            best_verification=best_verification,
                        )

                        if verification is None:
                            self.logger.warning(f"   Verification failed, skipping")
                            break

                        _cur_level = _annealing_level
                        _was_rejected = not verification.get("fit_acceptable", True)

                        # Retroactive physics-based promotion: if a previous
                        # iteration deferred current_result (in-band lower R²,
                        # awaiting a verifier verdict), this verification just
                        # rated it.  Promote if physics improved over best.
                        if (current_result is not best_result
                                and current_r2 >= R2_FLOOR
                                and verification.get("physically_better_than_best", False)):
                            note = (verification.get("comparison_note") or "physics improvement")[:90]
                            best_r2 = current_r2
                            best_result = current_result
                            best_config = (state.get("locked_fitting_config") or {}).copy()
                            state["locked_fitting_config"] = best_config
                            self.logger.info(
                                f"   Retroactively promoted current (R² = {current_r2:.4f}) on physics — {note}"
                            )

                        # If the verifier just inspected best_result itself
                        # (either was already best or just promoted above),
                        # record its verdict so the next iteration's prompt
                        # can include best's complaint summary and the
                        # post-loop threshold gate can know whether best is
                        # under suspicion.
                        if current_result is best_result:
                            best_verification = verification
                            best_ever_rejected = best_ever_rejected or _was_rejected

                        # Surface the GATE's driving metric (not always R²) per
                        # iteration, so the verifier judges the plateau on the
                        # actual acceptance metric — its value this step and the
                        # best-so-far. For the r_squared gate these are just R².
                        _g = _gate(state)
                        if _g is not None and getattr(_g, "metric", "r_squared") != "r_squared":
                            try:
                                _cur_metric = _g.extract(current_result.get("fit_quality"))
                                _best_metric = _g.extract(best_result.get("fit_quality"))
                                _metric_label = _g.label
                            except Exception:
                                _cur_metric, _best_metric, _metric_label = current_r2, best_r2, "R²"
                        else:
                            _cur_metric, _best_metric, _metric_label = current_r2, best_r2, "R²"

                        # Store in history for next iteration's context
                        verification_history.append({
                            "r_squared": current_r2,
                            "best_so_far": best_r2,
                            "metric_value": _cur_metric,
                            "best_metric_value": _best_metric,
                            "metric_label": _metric_label,
                            "config_used": state.get("locked_fitting_config", {}),
                            "issues_found": verification.get("issues_found", []),
                            "overall_assessment": verification.get("overall_assessment", ""),
                            "recommended_action": verification.get("recommended_action", ""),
                            "physically_better_than_best": verification.get("physically_better_than_best", False),
                            "comparison_note": verification.get("comparison_note", ""),
                            "annealing_level": _cur_level,
                        })

                        if not _was_rejected:
                            # Verifier approval trumps the R² high-water mark —
                            # the verifier may accept a lower-R² fit on physics
                            # grounds (e.g. better peak shape).  Promote.
                            best_r2 = current_r2
                            best_result = current_result
                            best_config = (state.get("locked_fitting_config") or {}).copy()
                            best_verification = verification
                            best_ever_rejected = False
                            state["locked_fitting_config"] = best_config
                            self.logger.info(f"   ✅ Fit approved (R² = {best_r2:.4f})")
                            fit_was_approved = True
                            break

                        # Log issues
                        self._log_verification_issues(verification)

                        # Apply LLM's recommended fixes
                        refined_config = self._apply_llm_verification_feedback(state, verification)

                        # If the refinement LLM call failed (transient
                        # API error), tag the history so the next verifier
                        # knows the fix was never applied.
                        refinement_error = refined_config.pop(
                            "_refinement_error", None
                        )
                        if refinement_error:
                            verification_history[-1]["refinement_error"] = (
                                refinement_error
                            )

                        if refined_config == state.get("locked_fitting_config", {}):
                            # No changes at current temperature — escalate to
                            # give the LLM more freedom before giving up.
                            _annealing_level = min(_annealing_level + 1, _n_anneal_levels - 1)
                            if _annealing_level == _cur_level:
                                self.logger.info(f"   No config changes at max annealing level, stopping verification")
                                break
                            self.logger.info(f"   No config changes suggested, escalating to annealing level {_annealing_level}")
                            continue

                        # Clean up old visualization (but not the best result's
                        # viz — best_result and current_result share the same
                        # path when current was just promoted).
                        old_viz_path = current_result.get("visualization_path")
                        if (old_viz_path
                                and Path(old_viz_path).exists()
                                and current_result is not best_result):
                            try:
                                os.remove(old_viz_path)
                            except:
                                pass

                        state["locked_fitting_config"] = refined_config

                        # Sync skill strictness with adaptive annealing level
                        state["_annealing_level"] = _annealing_level

                        # Anchor the refinement on best_result.script (the
                        # working version) so the LLM adapts known-good code
                        # rather than regenerating from scratch.  Drop the
                        # script when escalating to the hot annealing level
                        # so the LLM can restructure freely.
                        _just_escalated_to_hot = (
                            _annealing_level >= _n_anneal_levels - 1
                            and _previous_annealing_level < _n_anneal_levels - 1
                        )
                        _refine_from = (
                            None if _just_escalated_to_hot
                            else (best_result or {}).get("script")
                        )

                        if _just_escalated_to_hot:
                            self.logger.info(f"   Refitting with verification feedback (fresh generation — hot annealing)...")
                        elif _refine_from:
                            self.logger.info(f"   Refitting with verification feedback (refining prior script)...")
                        else:
                            self.logger.info(f"   Refitting with verification feedback...")

                        verified_result = self._fit_single_spectrum(
                            state=state, curve_data=curve_data, data_path=data_path,
                            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx,
                            base_script=None,
                            refine_from_script=_refine_from,
                            refine_from_r2=best_r2,
                            refine_from_issues=verification.get("issues_found", []),
                        )
                        # Stamp the annealing level this refit was generated at,
                        # so a downstream consumer can tell whether the WINNING
                        # result came from a hot (fresh-generation) regeneration
                        # vs. the original plan — used by T=2 auto-distillation
                        # to decide a fit is a "novel pipeline". Travels with the
                        # result dict through every promotion / judge path.
                        if isinstance(verified_result, dict):
                            verified_result["_produced_at_level"] = _annealing_level
                        # Record the level this refit ran at, so the next
                        # iteration can detect the escalation into hot. Updated
                        # only on an actual refit (not the no-config-change
                        # escalate-and-continue branch above).
                        _previous_annealing_level = _annealing_level

                        if verified_result["success"]:
                            verified_r2 = verified_result.get("fit_quality", {}).get("r_squared") or 0

                            all_attempts.append({
                                "model": f"Verification-{verification_iter + 1}",
                                "r2": verified_r2,
                                "result": verified_result,
                                "config": (state.get("locked_fitting_config") or {}).copy(),
                                "verification": verification,
                            })

                            # Latest is always what the next verifier judges.
                            current_result = verified_result
                            current_r2 = verified_r2

                            # Promotion rule (post-refit, no LLM call here):
                            # 1. Strict R² improvement → promote immediately.
                            # 2. Catastrophic regression (R² < floor) →
                            #    reject; roll back the locked config so the
                            #    next refit anchors on best.
                            # 3. In-band lower R² → DEFER promotion to the
                            #    next iteration's verifier, which will rate
                            #    physically_better_than_best with the new
                            #    fit's visualization.  Keep refined_config
                            #    as the locked one so it matches current.
                            if verified_r2 > best_r2:
                                best_r2 = verified_r2
                                best_result = verified_result
                                best_config = (state.get("locked_fitting_config") or {}).copy()
                                state["locked_fitting_config"] = best_config
                                best_ever_rejected = False
                                best_verification = None
                                self.logger.info(
                                    f"   Refit R² = {verified_r2:.4f} promoted (best now {best_r2:.4f})"
                                )
                            elif verified_r2 < R2_FLOOR:
                                state["locked_fitting_config"] = best_config
                                self.logger.info(
                                    f"   Refit R² = {verified_r2:.4f} below R² floor "
                                    f"{R2_FLOOR:.2f} → rejected (best stays {best_r2:.4f})"
                                )
                            else:
                                self.logger.info(
                                    f"   Refit R² = {verified_r2:.4f} "
                                    f"(best stays {best_r2:.4f}; deferred to next verifier for physics check)"
                                )

                            # Adaptive annealing — three escalation
                            # triggers, applied in order; each can lift
                            # _annealing_level (capped at n-1).
                            improvement = best_r2 - _prev_best_r2
                            remaining = max(self.max_verification_iterations - verification_iter - 1, 1)
                            required_rate = max(self.r2_threshold - best_r2, 0.0) / remaining

                            # (a) Rate-based: improvement too slow to reach
                            #     threshold in remaining budget.
                            rate_escalated = False
                            if improvement < required_rate:
                                _annealing_level = min(
                                    _annealing_level + 1, _n_anneal_levels - 1
                                )
                                rate_escalated = True
                                self.logger.info(
                                    f"   Annealing: improvement {improvement:.4f} < required rate {required_rate:.4f}, "
                                    f"escalating to level {_annealing_level}"
                                )

                            # (b) Patience-based: best stalled for _PATIENCE
                            #     consecutive iterations.  Resets on any
                            #     forward movement of best.
                            if best_r2 > _prev_best_r2:
                                _stall_count = 0
                            else:
                                _stall_count += 1
                                if _stall_count >= _PATIENCE and not rate_escalated:
                                    new_level = min(
                                        _annealing_level + 1, _n_anneal_levels - 1
                                    )
                                    if new_level > _annealing_level:
                                        _annealing_level = new_level
                                        self.logger.info(
                                            f"   Annealing: best stalled for {_stall_count} iterations, "
                                            f"escalating to level {_annealing_level}"
                                        )
                                    _stall_count = 0

                            # (c) Iteration floor: guarantees the hot level
                            #     is reached even when rate/patience say
                            #     otherwise (e.g., best ≥ threshold so
                            #     required_rate degenerates to 0).
                            _floor = min(
                                (verification_iter + 1) // _floor_divisor,
                                _n_anneal_levels - 1,
                            )
                            if _floor > _annealing_level:
                                self.logger.info(
                                    f"   Annealing: iteration floor lifting "
                                    f"level {_annealing_level} → {_floor}"
                                )
                                _annealing_level = _floor
                                _stall_count = 0

                            if not rate_escalated and _stall_count == 0 and _floor <= _annealing_level:
                                # No escalation this iteration; log the
                                # rate decision for diagnostic continuity.
                                pass  # already implicit; suppress duplicate logs

                            _prev_best_r2 = best_r2
                        else:
                            self.logger.warning(f"   Refit failed, stopping verification")
                            break

                    else:
                        # Loop exhausted without approval - one final pass to
                        # rate the latest state.  If current was deferred
                        # (in-band, awaiting physics verdict), this is its
                        # last chance to be promoted.
                        self.logger.info(f"   Verifying final refit...")
                        final_verification = self._verify_fit_with_llm(
                            state, current_result,
                            verification_iter=self.max_verification_iterations,
                            annealing_level=_annealing_level,
                            best_result=best_result,
                            best_verification=best_verification,
                        )

                        if final_verification:
                            _final_rejected = not final_verification.get("fit_acceptable", True)

                            # Retroactive promotion of deferred current
                            if (current_result is not best_result
                                    and current_r2 >= R2_FLOOR
                                    and final_verification.get("physically_better_than_best", False)):
                                note = (final_verification.get("comparison_note") or "physics improvement")[:90]
                                best_r2 = current_r2
                                best_result = current_result
                                best_config = (state.get("locked_fitting_config") or {}).copy()
                                self.logger.info(
                                    f"   Post-loop promoted current (R² = {current_r2:.4f}) on physics — {note}"
                                )

                            # Update best's verdict tracking
                            if current_result is best_result:
                                best_verification = final_verification
                                if not _final_rejected:
                                    self.logger.info(f"   ✅ Final fit approved (R² = {best_r2:.4f})")
                                    fit_was_approved = True
                                    best_ever_rejected = False
                                else:
                                    best_ever_rejected = True
                                    self._log_verification_issues(final_verification)
                            else:
                                # current still differs from best (no physics
                                # promotion).  best's last verdict stands.
                                if _final_rejected:
                                    self._log_verification_issues(final_verification)

                    # Restore config to match best result after verification loop
                    state["locked_fitting_config"] = best_config

            # --- Verifier-approved fits bypass the R² threshold check ---
            if fit_was_approved:
                self.logger.info(f"✅ Verifier approved fit (R² = {best_r2:.4f})")
                quality_history = self._build_quality_history(
                    best_r2, self.r2_threshold, all_attempts,
                    verification_history, None,
                    best_result.get("script_errors"),
                )
                quality_history["approved"] = True
                quality_history["approved_by"] = "verifier"
                best_result["quality_history"] = quality_history
                self._stamp_hot_deviation(best_result)
                return best_result

            # --- Check if we meet threshold ---
            # Option B: when the verifier explicitly rejected best at some
            # point and never approved it later, fall through to the judge
            # even if R² meets the numerical threshold.  This catches the
            # "high-R² but wrong-physics" trap where the verifier kept
            # complaining about best on physics grounds.
            if best_r2 >= self.r2_threshold and not best_ever_rejected:
                self.logger.info(f"✅ R² = {best_r2:.4f} (meets threshold {self.r2_threshold})")
                best_result["quality_history"] = self._build_quality_history(
                    best_r2, self.r2_threshold, all_attempts,
                    verification_history, None,
                    best_result.get("script_errors"),
                )
                self._stamp_hot_deviation(best_result)
                return best_result
            elif best_r2 >= self.r2_threshold:
                self.logger.info(
                    f"⚠️ R² = {best_r2:.4f} meets threshold {self.r2_threshold}, "
                    f"but verifier rejected best — deferring to judge"
                )
            else:
                self.logger.warning(f"⚠️ R² = {best_r2:.4f} (below threshold {self.r2_threshold})")
        else:
            self.logger.error(f"   Initial fit failed: {result.get('error', 'Unknown')[:50]}")
            all_attempts.append({"model": initial_model, "r2": 0, "result": result})

        # NOTE: the alternative-model loop was removed.  Hot annealing
        # (level n-1) inside the verification loop now drops the script
        # anchor and grants the LLM the same freedom to restructure the
        # model.  Patience counter and iteration floor guarantee the hot
        # level is reached when refits stall.

        # --- Human feedback for poor fit (if enabled) ---
        # Guard `best_result`: when every fitting attempt failed it is None, and
        # there is no fit to review — skip straight to graceful failure handling.
        # Suppressed inside best-of-N candidate attempts: interactive prompts
        # from N worker threads would interleave (and the threshold adjustment
        # below mutates shared self.r2_threshold).
        if (
            self.enable_human_feedback
            and _is_anchor
            and best_result
            and not state.get("_suppress_human_feedback")
        ):
            feedback_result = self._get_human_feedback_for_poor_fit(state, best_result, all_attempts)

            if feedback_result:
                if feedback_result.get("action") == "adjust_threshold":
                    self.r2_threshold = feedback_result["new_threshold"]
                    if best_r2 >= self.r2_threshold:
                        self.logger.info(f"✅ Best fit now meets adjusted threshold")
                        best_result["quality_history"] = self._build_quality_history(
                            best_r2, self.r2_threshold, all_attempts,
                            verification_history, None,
                            best_result.get("script_errors"),
                        )
                        return best_result

                elif feedback_result.get("action") == "retry":
                    refined_config = self._refine_model_from_feedback(state, feedback_result["feedback"])
                    original_config = state.get("locked_fitting_config")
                    state["locked_fitting_config"] = refined_config

                    human_guided_result = self._fit_single_spectrum(
                        state=state, curve_data=curve_data, data_path=data_path,
                        spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                    )

                    if human_guided_result["success"]:
                        human_r2 = human_guided_result.get("fit_quality", {}).get("r_squared") or 0
                        self.logger.info(f"   Human-guided fit: R² = {human_r2:.4f}")

                        if human_r2 > best_r2:
                            best_r2 = human_r2
                            best_result = human_guided_result
                            best_config = refined_config.copy()
                            if _is_anchor:
                                state["locked_fitting_config"] = refined_config
                        else:
                            state["locked_fitting_config"] = original_config
                    else:
                        state["locked_fitting_config"] = original_config

        # --- Unified judge: evaluate ALL attempts (verification + alternatives) ---
        judge_result = None
        successful_attempts = [a for a in all_attempts if a.get("r2", 0) > 0]
        if _is_anchor and len(successful_attempts) > 1:
            judge_result = self._judge_select_best_fit(successful_attempts)

            selected_index = judge_result.get("selected_index")
            is_acceptable = judge_result.get("acceptable", False)

            if selected_index is not None:
                selected_attempt = successful_attempts[selected_index]
                best_result = selected_attempt["result"]
                best_r2 = selected_attempt["r2"]
                if selected_attempt.get("config"):
                    state["locked_fitting_config"] = selected_attempt["config"]

                if is_acceptable:
                    if judge_result.get("issues_with_selected"):
                        best_result["judge_note"] = judge_result["issues_with_selected"]
                    self.logger.info(f"   ✅ Using judge-selected fit (R² = {best_r2:.4f})")
                else:
                    best_result["judge_warning"] = (
                        f"Judge selected this as best available (R² = {best_r2:.4f}) "
                        f"but noted it does not meet acceptance criteria. "
                        f"Reason: {judge_result.get('reasoning', 'No reason provided')[:200]}"
                    )
                    self.logger.warning(
                        f"   ⚠️ Using judge-selected fit (R² = {best_r2:.4f}) "
                        f"despite not meeting acceptance criteria"
                    )
            else:
                best_result["judge_warning"] = (
                    f"Judge could not select any acceptable fit. "
                    f"Reason: {judge_result.get('reasoning', 'No reason provided')[:200]}"
                )
                self.logger.warning(f"   ⚠️ Judge could not select any fit - keeping current best (R² = {best_r2:.4f})")

        # --- Return best available result ---
        if best_result:
            # This is the "best available" fallback (the accept/threshold paths
            # return earlier). A fit can land here two ways: (a) R² genuinely
            # below threshold, or (b) R² meets threshold but the verifier kept
            # rejecting on PHYSICS grounds. Word the warning to match reality —
            # never claim "below threshold" when the number is at/above it.
            if best_r2 >= self.r2_threshold:
                best_result["quality_warning"] = (
                    f"R² = {best_r2:.4f} meets the threshold {self.r2_threshold} but the "
                    f"fit was not accepted on physical grounds (see verifier notes)"
                )
            else:
                best_result["quality_warning"] = (
                    f"R² = {best_r2:.4f} below threshold {self.r2_threshold}"
                )
            best_result["attempted_models"] = [a["model"] for a in all_attempts]
            best_result["quality_history"] = self._build_quality_history(
                best_r2, self.r2_threshold, all_attempts,
                verification_history, judge_result,
                best_result.get("script_errors"),
            )
            if best_r2 >= self.r2_threshold:
                self.logger.info(
                    f"✅ Accepting best available fit (R² = {best_r2:.4f} meets threshold {self.r2_threshold})"
                )
            else:
                self.logger.warning(
                    f"⚠️ Proceeding with best available fit (R² = {best_r2:.4f}, below threshold {self.r2_threshold})"
                )

            if _is_anchor:
                state["locked_fitting_config"] = best_config

            return best_result
        else:
            return {
                "index": spectrum_idx, "name": spectrum_name, "success": False,
                "error": "All fitting attempts failed", "attempts": len(all_attempts),
                "parameters": {}, "fit_quality": {},
            }

    def _fit_with_quality_control_best_of_n(
        self,
        state: dict,
        curve_data: np.ndarray,
        data_path: str,
        spectrum_name: str,
        spectrum_idx: int,
        is_regime_anchor: bool = False,
        reuse_script: Optional[str] = None,
        reuse_source: Optional[str] = None,
    ) -> dict:
        """Run N independent anchor fits in parallel and keep the best.

        Mirrors the image controller's ``_execute_and_verify_best_of_n``:
        each attempt is a full ``_fit_with_quality_control`` run in its own
        working subdir with its own copy of the locked config; attempts
        differ only by sampling randomness. R²-gated survivors go to an LLM
        judge that inspects the fit plots (a marginally higher R² from an
        unphysical fit loses); the winner's (possibly QC-refined) config is
        propagated back into ``state`` for the regime.

        ``n_candidates == 1`` and the #172 ``reuse_script`` fast path bypass
        the fan-out entirely (byte-identical to a direct call).

        With ``candidate_escalation`` set (the orchestrator's auto-default
        path), attempt 0 runs alone first and is fast-accepted when strong
        (``_candidate_fast_accept``); the remaining attempts launch only when
        it is weak. In CO_PILOT/AUTOPILOT a join-approval prompt lets the
        user accept the judge's pick, override it, or demand the remaining
        attempts after a fast-accept.
        """
        n = max(1, int(state.get("n_candidates") or 1))
        escalation = bool(state.get("candidate_escalation"))
        # Skill-gated auto-escalation: the n>1 + escalation default is an AUTO
        # ensemble (no explicit user count, used when curve fitting has no
        # skill to guide it — high plan variance). A loaded domain skill PINS
        # the technique, so independent candidates would just converge on the
        # mandated model; run one skill-guided fit instead. An EXPLICIT user
        # count (candidate_escalation False, e.g. "run 3 candidates") is always
        # honored, skill or not.
        if escalation and n > 1:
            _active = _active_skill_names(state)
            if _active:
                self.logger.info(
                    f"   Skill active ({', '.join(_active)}) — single "
                    f"skill-guided fit; auto best-of-N suppressed (pass an "
                    f"explicit n_candidates to force it)."
                )
                n = 1
        if n == 1 or reuse_script:
            return self._fit_with_quality_control(
                state=state, curve_data=curve_data, data_path=data_path,
                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx,
                is_regime_anchor=is_regime_anchor,
                reuse_script=reuse_script, reuse_source=reuse_source,
            )

        spectrum_config = state.get("locked_fitting_config", {})

        import threading as _threading
        from ....utils.log_context import register_worker, unregister_worker
        _parent_thread = _threading.get_ident()

        def _run_candidate(i: int, tagged: bool = True) -> tuple:
            job_state = dict(state)
            job_state["locked_fitting_config"] = copy.deepcopy(spectrum_config)
            job_state["_candidate_tag"] = f"cand_{i:02d}"
            job_state["_candidate_subdir"] = (
                f"{CANDIDATES_DIR_NAME}/cand_{i:02d}"
            )
            job_state["_suppress_human_feedback"] = True
            # Always register so this worker's log records route to the calling
            # (chat) thread and stay visible in the UI verbose panel. The [cand]
            # PREFIX is added only when several candidates run concurrently; a
            # lone candidate keeps clean, unprefixed — but still visible — logs.
            register_worker(_parent_thread, f"cand_{i:02d}", prefix=tagged)
            try:
                # Ensemble diversity: each fan-out candidate (>=1) generates its
                # OWN independent fitting plan — like running the agent again.
                # Especially valuable for skill-less curves (high plan variance);
                # with an authoritative skill the plans converge on the mandated
                # model. Candidate 0 keeps the (human-approved) primary plan.
                # Toggle off with state["independent_candidate_plans"] = False.
                if (i >= 1 and self.replanner is not None
                        and job_state.get("independent_candidate_plans", True)):
                    self.logger.info(
                        "Planning an independent approach for this candidate..."
                    )
                    self.replanner.replan_headless(job_state)
                result = self._fit_with_quality_control(
                    state=job_state, curve_data=curve_data,
                    data_path=data_path, spectrum_name=spectrum_name,
                    spectrum_idx=spectrum_idx,
                    is_regime_anchor=is_regime_anchor,
                )
            finally:
                unregister_worker()
            return result, job_state

        candidates = []

        def _run_attempts(indices) -> None:
            indices = list(indices)
            # Prefix worker logs with the candidate tag only when more than one
            # candidate runs at once; a single candidate stays unprefixed.
            tagged = len(indices) > 1
            with ThreadPoolExecutor(max_workers=min(len(indices), 6)) as pool:
                future_to_attempt = {
                    pool.submit(_run_candidate, i, tagged): i for i in indices
                }
                done_count = 0
                for future in as_completed(future_to_attempt):
                    attempt = future_to_attempt[future]
                    done_count += 1
                    try:
                        result, job_state = future.result()
                    except Exception as exc:
                        self.logger.error(f"Candidate {attempt} raised: {exc}")
                        result = {
                            "index": spectrum_idx, "name": spectrum_name,
                            "success": False, "error": str(exc),
                            "parameters": {}, "fit_quality": {},
                        }
                        job_state = {}
                    qh = result.get("quality_history") or {}
                    candidates.append({
                        "attempt": attempt,
                        "result": result,
                        "config_after": job_state.get(
                            "locked_fitting_config", spectrum_config
                        ),
                        "score": qh.get("final_r2", 0.0) or 0.0,
                        "approved": bool(qh.get("approved", False)),
                        "success": bool(result.get("success", False)),
                        "iterations": len(qh.get("verification_iterations", [])),
                        "visualization_path": result.get("visualization_path"),
                    })
                    self.logger.info(
                        f"Candidate {attempt} finished "
                        f"({done_count}/{len(indices)}): "
                        f"R²={candidates[-1]['score']:.4f}, "
                        f"approved={candidates[-1]['approved']}, "
                        f"iterations={candidates[-1]['iterations']}"
                    )
                    # Persist a JSON-safe snapshot of the attempt's numbers
                    # into its candidate dir (audit + ground-truth scoring
                    # of losers; mirrors the image controller).
                    try:
                        cdir = (
                            self.output_dir / f"spectrum_{spectrum_idx:04d}"
                            / CANDIDATES_DIR_NAME / f"cand_{attempt:02d}"
                        )
                        cdir.mkdir(parents=True, exist_ok=True)
                        with open(cdir / "attempt_result.json", "w") as f:
                            json.dump({
                                "attempt": attempt,
                                "score": candidates[-1]["score"],
                                "approved": candidates[-1]["approved"],
                                "success": candidates[-1]["success"],
                                "iterations": candidates[-1]["iterations"],
                                "model_type": result.get("model_type"),
                                "parameters": result.get("parameters"),
                                "fit_quality": result.get("fit_quality"),
                            }, f, indent=2, default=str)
                    except Exception as e:
                        self.logger.debug(
                            f"attempt_result.json not written: {e}"
                        )
            candidates.sort(key=lambda c: c["attempt"])

        def _select() -> tuple:
            survivors = [
                c for c in candidates if c["success"] and c["approved"]
            ]
            if len(survivors) >= 2:
                judge_info = self._select_best_fit_candidate(state, survivors)
                return survivors[judge_info["selected_index"]], judge_info
            if len(survivors) == 1:
                return survivors[0], {
                    "reasoning": "Only one candidate passed the R² gate.",
                    "fallback": False,
                }
            successful = [c for c in candidates if c["success"]]
            if successful:
                return max(successful, key=lambda c: c["score"]), {
                    "reasoning": (
                        "No candidate passed the R² gate; kept the "
                        "highest-R² result."
                    ),
                    "fallback": True,
                }
            self.logger.error(f"All {len(candidates)} candidates failed")
            return None, None

        # --- Fan-out ---
        escalated = False
        if escalation:
            self.logger.info(
                f"Best-of-{n} (escalation): running attempt 0 alone; "
                f"fanning out only if it is weak"
            )
            _run_attempts([0])
            if not self._candidate_fast_accept(candidates[0], _gate(state)):
                escalated = True
                self.logger.info(
                    f"First attempt not a clean win "
                    f"(R²={candidates[0]['score']:.4f}, "
                    f"iterations={candidates[0]['iterations']}, "
                    f"climbed_to_hot="
                    f"{self._candidate_climbed_to_hot(candidates[0])}) - "
                    f"escalating to {n} candidates"
                )
                _run_attempts(range(1, n))
        else:
            self.logger.info(
                f"Best-of-{n}: launching {n} independent anchor fits "
                f"in parallel"
            )
            _run_attempts(range(n))

        # --- Selection ---
        if escalation and not escalated:
            winner = candidates[0]
            judge_info = {
                "reasoning": (
                    "First attempt passed the fast-accept gate - "
                    "no escalation."
                ),
                "fallback": False,
            }
        else:
            winner, judge_info = _select()
            if winner is None:
                return candidates[0]["result"]

        self.logger.info(
            f"Best-of-{n}: selected candidate {winner['attempt']} "
            f"(R²={winner['score']:.4f}) - {judge_info['reasoning'][:120]}"
        )

        # --- Join approval (CO_PILOT/AUTOPILOT) ---
        # Only prompt when there is more than one candidate to compare. A single
        # fast-accepted candidate (the escalation probe that passed the gate, or
        # a single non-escalation attempt) just proceeds — the best-of-N
        # comparison menu is meaningless for one result.
        if (
            self.enable_human_feedback
            and not state.get("_suppress_human_feedback")
            and len(candidates) > 1
        ):
            choice = self._get_bestofn_join_approval(
                candidates, winner, judge_info, allow_more=False,
            )
            if isinstance(choice, int):
                winner = next(
                    c for c in candidates if c["attempt"] == choice
                )
                judge_info = dict(judge_info)
                judge_info["human_override"] = True
                self.logger.info(
                    f"Human override: candidate {choice} selected"
                )

        # --- Lock the winner ---
        # Winner's (possibly QC-refined) config becomes the regime's config;
        # the caller's existing propagation distributes it.
        state["locked_fitting_config"] = winner["config_after"]

        result = winner["result"]
        self._promote_candidate_artifacts(
            result, spectrum_idx, winner["attempt"]
        )
        winner["visualization_path"] = result.get("visualization_path")

        result["anchor_candidates"] = [
            {
                "attempt": c["attempt"],
                "score": c["score"],
                "approved": c["approved"],
                "success": c["success"],
                "iterations": c["iterations"],
                "visualization_path": c["visualization_path"],
                "selected": c is winner,
            }
            for c in candidates
        ]
        result["anchor_judge"] = {
            "reasoning": judge_info.get("reasoning", ""),
            "fallback": bool(judge_info.get("fallback", False)),
            "escalated": escalated,
            "human_override": bool(judge_info.get("human_override", False)),
        }
        return result

    # Escalation fast-accept gate: attempt 0 skips the fan-out only when it
    # cleared the acceptance metric WITH MARGIN and did NOT have to anneal
    # "hot" to get there. Fit quality has TWO parts — the numeric metric
    # (objective goodness-of-fit: R², or a skill's χ²/RMSE/FOM) and physical
    # correctness (a verifier judgment, folded into `approved`). The metric
    # half is a strong, objective signal; the physical-correctness half is
    # SUBJECTIVE, and is exactly where a hot-annealed OVER-FIT (T=2 grants full
    # model freedom to add components) can post a great metric, be physically
    # wrong, and slip past a lenient verifier. The annealing-"struggle" check
    # (max level < hot) corroborates that approval was earned WITHOUT relaxing
    # the model. (Replaces the old `iterations <= 2` proxy, which fanned out
    # needlessly on good-but-slow fits and did nothing about over-fit risk.)
    #
    # The margin is METRIC-AGNOSTIC: a fraction of the gate's accept↔hard-reject
    # band, applied in the gate's direction, capped near the metric's optimum
    # (QualityGate.clears_by_fast_margin). For the default R² gate (band 0.05,
    # best 1.0) the fraction 0.4 reproduces the old absolute +0.02 / 0.97 bar
    # exactly; a lower-is-better χ² gate instead requires value <= accept -
    # margin. The R² `final_r2` score is NOT used as the bar — the gate reads
    # its own metric from `fit_quality`, so a skill scored by χ²/RMSE is no
    # longer wrongly required to also post a high R².
    ESCALATION_MARGIN_FRACTION = 0.4

    @property
    def _hot_annealing_level(self) -> int:
        return len(self._CONSTRAINT_ANNEALING_SCHEDULE) - 1

    @staticmethod
    def _candidate_max_annealing_level(c: dict) -> int:
        """Highest annealing level the candidate's verification loop reached."""
        iters = (
            (c["result"].get("quality_history") or {})
            .get("verification_iterations") or []
        )
        return max(
            (it.get("annealing_level", 0) for it in iters), default=0
        )

    def _candidate_climbed_to_hot(self, c: dict) -> bool:
        """True only if the loop ESCALATED into hot annealing under stall — it
        started below hot and had to climb there. A fit that STARTED at hot (a
        caller / re-run set the starting annealing level) did not struggle, so
        reaching hot is not held against it (that must NOT auto-escalate)."""
        iters = (
            (c["result"].get("quality_history") or {})
            .get("verification_iterations") or []
        )
        if not iters:
            return False
        levels = [it.get("annealing_level", 0) for it in iters]
        hot = self._hot_annealing_level
        return max(levels) >= hot and levels[0] < hot

    def _candidate_fast_accept(self, c: dict, gate) -> bool:
        value = gate.extract((c["result"] or {}).get("fit_quality") or {})
        return (
            c["success"]
            and c["approved"]
            and gate.clears_by_fast_margin(value, self.ESCALATION_MARGIN_FRACTION)
            and not self._candidate_climbed_to_hot(c)
        )

    def _get_bestofn_join_approval(
        self, candidates: List[dict], winner: dict, judge_info: dict,
        allow_more: bool,
    ):
        """CO_PILOT/AUTOPILOT approval of the best-of-N winner.

        Saves each candidate's fit plot as a ``*review*`` png (the UI's
        feedback modal discovers and renders those), prints a comparison
        block, and asks. Returns ``None`` (accept the judge's pick), an
        ``int`` (human-overridden attempt index), or ``"more"`` (run the
        remaining attempts after an escalation fast-accept). Never mutates
        ``self.r2_threshold`` (shared across attempts).
        """
        review_paths = []
        try:
            print("\n" + "=" * 70)
            print("BEST-OF-N CANDIDATES - Review Before Locking Anchor")
            print("=" * 70)
            for c in candidates:
                viz = c["result"].get("visualization_bytes")
                if viz:
                    p = self.output_dir / (
                        f"bestofn_candidate_{c['attempt']:02d}_review.png"
                    )
                    with open(p, "wb") as f:
                        f.write(viz)
                    review_paths.append(p)
                    print(
                        f"[Candidate {c['attempt']} fit saved to: {p}]"
                    )
            print()
            for c in candidates:
                mark = "  <- judge pick" if c is winner else ""
                print(
                    f"  Candidate {c['attempt']}: R²={c['score']:.4f}, "
                    f"approved={c['approved']}, "
                    f"iterations={c['iterations']}{mark}"
                )
            reasoning = judge_info.get("reasoning", "")
            if reasoning:
                print(f"\nJudge: {reasoning[:500]}")
            print("\n" + "-" * 60)
            print("Options:")
            print(
                f"  - Press Enter to accept candidate {winner['attempt']}"
            )
            print("  - Type a candidate number to use that one instead")
            if allow_more:
                print(
                    "  - Type 'more' to run the remaining candidates "
                    "and compare"
                )
            print("-" * 60)

            for _ in range(3):
                response = input(
                    f"\nYour choice (Enter = accept candidate "
                    f"{winner['attempt']}): "
                ).strip()

                if not response:
                    return None
                if allow_more and response.lower() == "more":
                    print("Running the remaining candidates...")
                    return "more"
                if response.isdigit():
                    idx = int(response)
                    if any(c["attempt"] == idx and c["success"]
                           for c in candidates):
                        print(f"Using candidate {idx}.")
                        return idx
                    print(f"No successful candidate {idx} to use.")
                    continue
                # SELECTION step, not a refine step: do not silently discard
                # unrecognized free text — re-prompt with the valid options so
                # the input is never quietly dropped.
                print(
                    "Unrecognized input. Press Enter to accept candidate "
                    f"{winner['attempt']}, or type a candidate number"
                    + (" or 'more'" if allow_more else "") + "."
                )
            print("No valid choice entered; accepting the judge's pick.")
            return None
        finally:
            for p in review_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _promote_candidate_artifacts(
        self, result: dict, spectrum_idx: int, attempt: int
    ) -> None:
        """Copy the winning attempt's files up into the canonical per-spectrum dir.

        Everything downstream (feature tables, prior_analysis_paths, the
        orchestrator's viz search) expects artifacts directly under
        ``spectrum_NNNN/``; loser attempts stay under ``_candidates/`` for
        audit.
        """
        item_dir = self.output_dir / f"spectrum_{spectrum_idx:04d}"
        cand_dir = item_dir / CANDIDATES_DIR_NAME / f"cand_{attempt:02d}"
        if not cand_dir.is_dir():
            return
        try:
            for src in cand_dir.iterdir():
                dest = item_dir / src.name
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest)
            viz_path = result.get("visualization_path")
            if viz_path:
                promoted = item_dir / Path(viz_path).name
                if promoted.exists():
                    result["visualization_path"] = str(promoted)
        except Exception as e:
            self.logger.warning(
                f"Could not promote winning candidate artifacts: {e}"
            )

    def _select_best_fit_candidate(
        self, state: dict, candidates: List[dict]
    ) -> dict:
        """LLM judge: compare finished candidate fits, pick the best.

        Returns ``{"selected_index": <index into candidates>, "reasoning": str,
        "fallback": bool}``. Any judge failure falls back to the highest R².
        """
        def _fallback(reason: str) -> dict:
            best = max(
                range(len(candidates)), key=lambda i: candidates[i]["score"]
            )
            self.logger.warning(
                f"Best-of-N judge fallback ({reason}); using highest R² "
                f"(candidate {candidates[best]['attempt']})."
            )
            return {
                "selected_index": best,
                "reasoning": "judge unavailable - fell back to highest R²",
                "fallback": True,
            }

        blocks = []
        for i, c in enumerate(candidates):
            r = c["result"]
            qh = r.get("quality_history") or {}
            iters = qh.get("verification_iterations", [])
            max_anneal = max(
                (it.get("annealing_level", 0) or 0 for it in iters), default=0
            )
            params = json.dumps(r.get("parameters", {}), default=str)[:600]
            blocks.append(
                f"### Candidate {i}\n"
                f"  R²: {c['score']:.4f} (approved: {c['approved']})\n"
                f"  Model: {r.get('model_type', 'unknown')}\n"
                f"  Verification iterations: {c['iterations']} "
                f"(max annealing level: {max_anneal})\n"
                f"  Parameters (truncated): {params}\n"
            )

        prompt_text = self.BEST_OF_N_JUDGE_PROMPT.format(
            num_candidates=len(candidates),
            candidates_formatted="\n".join(blocks),
        )

        prompt_parts: List[Any] = [prompt_text]
        original_plot = state.get("original_plot_bytes")
        if original_plot:
            prompt_parts.append("\n**ORIGINAL DATA PLOT:**")
            prompt_parts.append({"mime_type": "image/png", "data": original_plot})
        for i, c in enumerate(candidates):
            viz = c["result"].get("visualization_bytes")
            if not viz:
                return _fallback(f"candidate {i} has no visualization bytes")
            prompt_parts.append(f"\n**Candidate {i} fit:**")
            prompt_parts.append({"mime_type": "image/png", "data": viz})

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            parsed, error = self._parse(response)
            if error or not parsed:
                return _fallback(f"parse failed: {error}")
            idx = parsed.get("selected_index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                return _fallback(f"invalid selected_index {idx!r}")
            return {
                "selected_index": idx,
                "reasoning": parsed.get("reasoning", ""),
                "fallback": False,
            }
        except Exception as e:
            return _fallback(str(e))

    def _log_verification_issues(self, verification: dict) -> None:
        """Log verification issues in a readable format."""
        issues_count = len(verification.get("issues_found", []))
        overall_assessment = verification.get('overall_assessment', 'No assessment provided')

        self.logger.info(f"   ⚠️ Found {issues_count} issue(s)")
        self.logger.info("")
        self.logger.info(f"   Assessment:")
        for line in self._wrap_text(overall_assessment, width=70):
            self.logger.info(f"      {line}")

        if verification.get("issues_found"):
            self.logger.info("")
            self.logger.info(f"   Issues:")
            
            for i, issue in enumerate(verification.get("issues_found", []), 1):
                location = issue.get('location', 'Unknown')
                problem = issue.get('problem', 'No description')
                suggested_fix = issue.get('suggested_fix', '')
                
                self.logger.info("")
                self.logger.info(f"   [{i}] {location}")

                # Wrap problem text
                problem_lines = self._wrap_text(problem, width=65)
                self.logger.info(f"       Problem: {problem_lines[0]}")
                for line in problem_lines[1:]:
                    self.logger.info(f"                {line}")

                # Wrap fix text
                if suggested_fix:
                    fix_lines = self._wrap_text(suggested_fix, width=65)
                    self.logger.info(f"       Fix: {fix_lines[0]}")
                    for line in fix_lines[1:]:
                        self.logger.info(f"            {line}")

        recommended = verification.get("recommended_action", "")
        if recommended and recommended.lower() != "none":
            self.logger.info("")
            self.logger.info(f"   Recommended action:")
            for line in self._wrap_text(recommended, width=65):
                self.logger.info(f"      {line}")
        
        self.logger.info("")

    def _apply_user_feedback(
        self, 
        state: dict, 
        user_feedback: str, 
        best_result: dict, 
        best_r2: float,
        curve_data: np.ndarray, 
        data_path: str, 
        spectrum_name: str, 
        spectrum_idx: int,
        all_attempts: list
    ) -> tuple:
        """
        Apply user feedback to refine the fit.
        
        Returns:
            Tuple of (best_result, best_r2) after applying feedback
        """
        refined_config = self._refine_model_from_feedback(state, user_feedback)
        original_config = state.get("locked_fitting_config")
        state["locked_fitting_config"] = refined_config

        # Clean up old visualization — _fit_single_spectrum will
        # create a fresh file at the same canonical path.
        old_viz_path = best_result.get("visualization_path")
        if old_viz_path and Path(old_viz_path).exists():
            try:
                os.remove(old_viz_path)
            except Exception:
                pass

        self.logger.info("   Refitting with user feedback...")
        user_guided_result = self._fit_single_spectrum(
            state=state, curve_data=curve_data, data_path=data_path,
            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
        )

        if user_guided_result["success"]:
            user_r2 = user_guided_result.get("fit_quality", {}).get("r_squared") or 0
            self.logger.info(f"   User-guided fit: R² = {user_r2:.4f}")
            all_attempts.append({"model": "User-guided", "r2": user_r2, "result": user_guided_result})

            if user_r2 > best_r2:
                return user_guided_result, user_r2
            else:
                # Save the new fit as a review image so the UI can display it
                if user_guided_result.get("visualization_bytes"):
                    review_viz_path = self.output_dir / "first_spectrum_fit_review.png"
                    with open(review_viz_path, 'wb') as f:
                        f.write(user_guided_result["visualization_bytes"])
                # User-guided was worse - ask what to do
                keep_user = self._ask_keep_user_guided_fit(user_r2, best_r2)
                if keep_user:
                    return user_guided_result, user_r2
                else:
                    # Restore original viz on disk so the file matches
                    # what best_result describes (re-analysis overwrote
                    # or deleted the original file).
                    if best_result.get("visualization_bytes") and old_viz_path:
                        try:
                            with open(old_viz_path, "wb") as f:
                                f.write(best_result["visualization_bytes"])
                        except Exception:
                            pass
                    state["locked_fitting_config"] = original_config
                    return best_result, best_r2
        else:
            self.logger.warning("   User-guided fit failed, keeping previous")
            # Restore original viz on disk (re-analysis deleted it)
            if best_result.get("visualization_bytes") and old_viz_path:
                try:
                    with open(old_viz_path, "wb") as f:
                        f.write(best_result["visualization_bytes"])
                except Exception:
                    pass
            state["locked_fitting_config"] = original_config
            return best_result, best_r2

    def _detect_outliers(self, series_results: List[dict], gate=None) -> List[dict]:
        # Score each fit by the GATE's metric, not always global R². For a
        # non-R² goodness-of-fit gate (e.g. peak_region_r2) a correct low-SNR
        # fit has a high gate metric but a low global R² — flagging on global R²
        # false-flags it. Fall back to r_squared when there is no such gate
        # (legacy behavior, unchanged).
        self._outlier_gate = gate if (gate is not None and gate.metric != "r_squared") else None

        def _score(r):
            fq = r.get("fit_quality", {})
            if self._outlier_gate is not None:
                return self._outlier_gate.extract(fq)
            return fq.get("r_squared")

        self._score_fn = _score
        r2_values = []
        for r in series_results:
            if r["success"]:
                r2 = _score(r)
                if r2 is not None:
                    r2_values.append(r2)

        if len(r2_values) < 3:
            return []

        r2_array = np.array(r2_values)
        # Robust center/scale (median + MAD). A single bad fit can't mask itself
        # by inflating the statistic the way mean/std let it — with mean/std a
        # lone outlier in an n-point series caps at √(n-1)σ (exactly 2.0 for
        # n=5), so it could never exceed a 2σ threshold and got mislabeled
        # "consistent with series". MAD is unaffected by the outlier itself.
        median_r2 = float(np.median(r2_array))
        mad = float(np.median(np.abs(r2_array - median_r2)))
        # 1.4826·MAD ≈ σ for normal data; floor it so a near-identical series
        # (MAD≈0) doesn't flag trivial scatter while a real gap still registers.
        robust_scale = max(1.4826 * mad, 0.02)

        flagged = []

        for r in series_results:
            if not r["success"]:
                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": "fit_failed",
                    "r_squared": None, "series_mean": median_r2, "series_std": robust_scale,
                    "deviation_sigma": None,
                    "recommendation": "Check data quality and consider manual inspection. The fitting script failed to execute successfully."
                })
                continue

            r2 = self._score_fn(r)
            if r2 is None:
                continue

            g = self._outlier_gate
            if g is not None:
                below_threshold = not g.is_accept(r2)
                worse = (median_r2 - r2) if g.direction == "higher_is_better" else (r2 - median_r2)
            else:
                below_threshold = r2 < self.r2_threshold
                worse = median_r2 - r2
            # Robust z-score; only a fit *worse* than the series median can be an
            # outlier (a better-than-typical fit is never flagged).
            deviation_sigma = worse / robust_scale
            is_outlier = deviation_sigma > self.outlier_sigma
            
            if below_threshold or is_outlier:
                if is_outlier and not below_threshold:
                    reason = "statistical_outlier"
                    recommendation = "Fit quality significantly worse than series average. Possible causes: phase transition, sample change, or instrument artifact. Consider detailed inspection - may indicate interesting physics."
                elif below_threshold and not is_outlier:
                    reason = "below_threshold"
                    recommendation = "Fit quality below threshold but in line with the rest of the series (R² is not a statistical outlier) — the chosen model may be suboptimal for this data type."
                else:
                    reason = "outlier_and_below_threshold"
                    recommendation = "Significant fit quality issue. This spectrum behaves differently from others in the series. Strongly recommend manual review - could indicate interesting physics, phase transition, or data quality issue."

                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": reason,
                    "r_squared": float(r2), "series_mean": median_r2, "series_std": robust_scale,
                    "deviation_sigma": float(deviation_sigma) if deviation_sigma else None,
                    "recommendation": recommendation
                })
        
        return flagged

    def _generate_outlier_report(self, flagged: List[dict], series_results: List[dict]) -> str:
        if not flagged:
            return ""
        
        lines = ["", "=" * 60, "⚠️  FLAGGED SPECTRA - REQUIRE ATTENTION", "=" * 60, ""]
        
        total = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        r2_values = [r.get("fit_quality", {}).get("r_squared") or 0 for r in series_results if r["success"]]
        
        if r2_values:
            lines.append(f"Series statistics: {successful}/{total} successful fits")
            lines.append(f"R² range: {min(r2_values):.4f} - {max(r2_values):.4f}")
            lines.append(f"R² mean ± std: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            lines.append(f"Quality threshold: {self.r2_threshold}")
            lines.append(f"Outlier detection: {self.outlier_sigma}σ below median (robust/MAD)")
            lines.append("")
        
        by_reason = {}
        for f in flagged:
            reason = f["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)
        
        reason_labels = {
            "fit_failed": "❌ Failed Fits",
            "statistical_outlier": "📊 Statistical Outliers (possible interesting physics)",
            "below_threshold": "⚠️ Below Threshold",
            "outlier_and_below_threshold": "🔴 Critical: Outlier + Below Threshold"
        }
        
        for reason, items in by_reason.items():
            lines.append(f"\n{reason_labels.get(reason, reason)} ({len(items)} spectra):")
            lines.append("-" * 50)
            
            for f in items:
                lines.append(f"  • {f['name']} (index {f['index']})")
                if f["r_squared"] is not None:
                    lines.append(f"    R² = {f['r_squared']:.4f} (series median: {f['series_mean']:.4f})")
                    if f["deviation_sigma"] is not None:
                        lines.append(f"    Deviation: {f['deviation_sigma']:.1f}σ below median")
                lines.append(f"    → {f['recommendation']}")
                lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_config_for_spectrum(self, state: dict, idx: int) -> dict:
        """Return the fitting config for a given spectrum index.

        If regime_configs is present, return the regime-specific config.
        Otherwise, return the single locked_fitting_config (backward compatible).
        """
        regime_configs = state.get("regime_configs")
        if regime_configs and idx in regime_configs:
            return regime_configs[idx]
        return state.get("locked_fitting_config", {})

    def _get_regime_for_spectrum(self, state: dict, idx: int) -> Optional[str]:
        """Return the regime name for a given spectrum index, or None."""
        series_plan = state.get("series_analysis_plan")
        if not series_plan:
            return None
        for regime in series_plan.get("regimes", []):
            if idx in regime.get("spectrum_indices", []):
                return regime.get("name", "unnamed")
        return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        self.logger.info("")
        self.logger.info(f"⚙️ FITTING: {mode_str}")
        _gate_obj = _gate(state)
        if _gate_obj.metric == "r_squared":
            _accept = float(self.r2_threshold)
            _floor = max(_accept - self._r2_soft_margin(_accept), 0.0)
            self.logger.info(
                f"   R² targets: accept ≥ {_accept:.3f} (with clean residuals); "
                f"hard-reject below {_floor:.3f}; soft band {_floor:.3f}–{_accept:.3f} "
                f"(verifier may reject on physics)"
            )
        else:
            _cmp = "≥" if _gate_obj.direction == "higher_is_better" else "≤"
            self.logger.info(
                f"   Quality: {_gate_obj.metric} {_cmp} {_gate_obj.accept_threshold:.3f} "
                f"accepts ({_gate_obj.direction}); skill scoring is the verification "
                f"(curve-fit R² verifier bypassed)."
            )
        self.logger.info(f"   Max verification iterations: {self.max_verification_iterations}")
        if not is_single:
            self.logger.info(f"   Outlier detection: {self.outlier_sigma}σ")
        
        spectrum_paths = state.get("spectrum_paths", [])
        spectrum_stack = state.get("spectrum_stack")

        # Determine regime structure for per-regime execution
        series_plan = state.get("series_analysis_plan")
        regime_configs = state.get("regime_configs")

        # Regime-boundary markers, keyed by the regime's first spectrum index.
        # Emitted as the loop reaches each boundary so transitions are visible
        # while the fit streams — not only in the upfront plan (the inline
        # "(regime: …)" tag alone made boundaries hard to spot in the log).
        regime_markers: Dict[int, list] = {}
        if series_plan and regime_configs:
            first_in_regime: set = set()
            regimes = series_plan.get("regimes", [])
            series_metadata = state.get("series_metadata", {})
            _values = series_metadata.get("values", [])
            _unit = series_metadata.get("unit", "")
            for rnum, regime in enumerate(regimes, 1):
                indices = sorted(regime.get("spectrum_indices", []))
                if not indices:
                    continue
                first_in_regime.add(indices[0])
                if len(regimes) > 1:
                    rng = ""
                    if _values:
                        vv = [_values[i] for i in indices if i < len(_values)]
                        if vv:
                            unit_str = f" {_unit}" if _unit else ""
                            span = f"{min(vv)}" if min(vv) == max(vv) else f"{min(vv)}-{max(vv)}"
                            rng = f" ({span}{unit_str})"
                    line = f"  ▸ Regime {rnum}/{len(regimes)}: {regime.get('name', 'Unnamed')}{rng}"
                    bar = "  " + "─" * max(len(line) - 2, 0)  # rule spans the text width
                    regime_markers[indices[0]] = ["", bar, line, bar]
            self.logger.info(f"   Regimes: {len(regimes)}")
            self.logger.info(
                f"   First-in-regime spectra (full QC): {sorted(first_in_regime)}"
            )
        else:
            first_in_regime = {0}  # Only spectrum 0 needs full QC

        # #172: locked-script reuse. When prior_analysis_paths supplies an
        # earlier curve-fit run, the spectrum-0 anchor reuses that run's saved
        # fitting script instead of re-deriving the model. Skipped for
        # multi-regime runs (a single prior script has no regime mapping).
        # Locked-script reuse (#172) is an explicit opt-in (reuse_locked_script),
        # not the default for a prior run being supplied. By default a follow-up
        # that names prior_analysis_paths is AGENT-JUDGED: the prior fit summary +
        # script are surfaced as reference in codegen and the agent decides
        # whether to reuse, adapt, or rewrite. Verbatim reuse is for extending an
        # ongoing campaign with a fixed model/feature schema — it cannot verify or
        # deepen a prior result (re-running the script that produced it only
        # reproduces it).
        if state.get("reuse_locked_script"):
            reuse_script, reuse_source = _first_prior_curve_fit_script(state)
        else:
            reuse_script, reuse_source = None, None
        if reuse_script and regime_configs:
            self.logger.info(
                "   ♻️  Locked-script reuse requested, but this run is "
                "multi-regime — reuse skipped."
            )
            reuse_script, reuse_source = None, None
        elif reuse_script:
            self.logger.info(
                f"   ♻️  Locked-script reuse (opt-in) — anchor will reuse the "
                f"fitting script from prior run '{reuse_source}' (#172)."
            )
        elif state.get("prior_analysis_paths"):
            self.logger.info(
                "   🧭 Prior curve-fit run(s) supplied as reference — the agent "
                "will decide whether to reuse, adapt, or rewrite the fit."
            )

        results_by_idx: Dict[int, dict] = {}
        deferred_non_anchors: List[dict] = []
        base_scripts: Dict[str, str] = {}  # keyed by regime name
        locked_preprocessing_strategy = None
        original_locked_config = state.get("locked_fitting_config", {})
        if original_locked_config:
            original_locked_config = original_locked_config.copy()

        run_parallel = self.parallel_workers > 1 and num_spectra > 1
        if run_parallel:
            self.logger.info(
                f"   Parallel non-anchor fan-out: up to {self.parallel_workers} workers"
            )

        for idx in range(num_spectra):
            if spectrum_stack is not None:
                curve_data = spectrum_stack[idx]
                spectrum_name = f"spectrum_{idx:04d}"
                data_path = f"stack_index_{idx}"
            else:
                data_path = spectrum_paths[idx]
                spectrum_name = Path(data_path).stem
                curve_data = self._load_curve_data(
                    data_path, column_mapping=state.get("column_mapping_locked"))

            # Raw data is fed straight to the fit script, which owns preprocessing
            # (see docs/preprocessing_in_fit_loop.md). No separate preprocessing
            # step here.

            # Determine regime and set appropriate config
            regime_name = self._get_regime_for_spectrum(state, idx) or "default"
            spectrum_config = self._get_config_for_spectrum(state, idx)

            # Temporarily set the config for this spectrum
            state["locked_fitting_config"] = spectrum_config

            # Regime-transition marker at each regime boundary.
            for _line in regime_markers.get(idx, ()):
                self.logger.info(_line)

            if is_single:
                self.logger.info(f"Fitting: {spectrum_name}")
            elif regime_configs:
                self.logger.info(
                    f"[{idx + 1}/{num_spectra}] Fitting: {spectrum_name} "
                    f"(regime: {regime_name})"
                )
            else:
                self.logger.info(f"[{idx + 1}/{num_spectra}] Fitting: {spectrum_name}")

            if idx in first_in_regime:
                if regime_configs and idx != 0:
                    self.logger.info(
                        f"  First in regime '{regime_name}' — full quality control"
                    )

                # For regime anchors that aren't spectrum 0, temporarily swap
                # original_plot_bytes and data_statistics so the verification
                # and script generation steps reference the correct spectrum.
                _saved_original_plot = None
                _saved_data_statistics = None
                if idx != 0 and idx in first_in_regime:
                    _saved_original_plot = state.get("original_plot_bytes")
                    _saved_data_statistics = state.get("data_statistics")
                    try:
                        anchor_plot = self.plot_fn(
                            curve_data, state.get("system_info", {})
                        )
                        state["original_plot_bytes"] = anchor_plot
                    except Exception:
                        pass
                    state["data_statistics"] = self._compute_statistics(curve_data)

                result = self._fit_with_quality_control_best_of_n(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx,
                    is_regime_anchor=(idx != 0 and idx in first_in_regime),
                    reuse_script=(reuse_script if idx == 0 else None),
                    reuse_source=(reuse_source if idx == 0 else None),
                )

                # Restore original state
                if _saved_original_plot is not None:
                    state["original_plot_bytes"] = _saved_original_plot
                if _saved_data_statistics is not None:
                    state["data_statistics"] = _saved_data_statistics

                # #172: reuse was attempted for the anchor but the result
                # carries no reuse_validity verdict -> the prior script could
                # not execute and full QC re-derived the model. Record the
                # schema-drift caveat so the orchestrator can react.
                if idx == 0 and reuse_script and not result.get("reuse_validity"):
                    result["reuse_validity"] = {
                        "reused": False,
                        "source": reuse_source,
                        "verdict": "script_failed",
                        "message": (
                            f"The locked fitting script from prior run "
                            f"'{reuse_source or 'prior'}' could not execute "
                            f"on this data; the model was re-derived from "
                            f"scratch and the extracted-feature schema may "
                            f"differ from the prior run."
                        ),
                    }
                    result["quality_warning"] = result["reuse_validity"]["message"]

                if result["success"] and result.get("script"):
                    base_scripts[regime_name] = result["script"]
                    if idx == 0:
                        state["base_fitting_script"] = result["script"]
                    self.logger.info(
                        f"📝 Base fitting script locked for regime '{regime_name}'."
                    )

                    # If QC changed the config, propagate to all spectra in this regime
                    updated_config = state.get("locked_fitting_config", spectrum_config)
                    if regime_configs and updated_config != spectrum_config:
                        for r_regime in (series_plan or {}).get("regimes", []):
                            if idx in r_regime.get("spectrum_indices", []):
                                for other_idx in r_regime["spectrum_indices"]:
                                    regime_configs[other_idx] = updated_config
                                break
            else:
                base_script = base_scripts.get(regime_name)
                if run_parallel:
                    deferred_non_anchors.append({
                        "idx": idx,
                        "regime_name": regime_name,
                        "spectrum_name": spectrum_name,
                        "curve_data": curve_data,
                        "data_path": data_path,
                        "base_script": base_script,
                        # Capture the per-spectrum locked config so retries via
                        # _correct_script see the correct regime's config, not
                        # whichever value happens to be left in shared `state`
                        # at drain time.
                        "spectrum_config": spectrum_config,
                    })
                    self.logger.info(
                        f"   ⏳ Queued spectrum {idx} ({spectrum_name}) for parallel fan-out"
                    )
                    continue  # tagging + logging happen in the drain phase
                result = self._fit_single_spectrum(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx,
                    base_script=base_script,
                )

            # Tag result with regime info
            if regime_configs:
                result["regime"] = regime_name

            results_by_idx[idx] = result

            if result["success"]:
                r2 = result.get("fit_quality", {}).get("r_squared")
                r2_str = f"R²: {r2:.4f}" if r2 else "R²: N/A"
                self.logger.info(f"✅ {result.get('model_type', 'Fit')} - {r2_str}")
            else:
                self.logger.error(f"❌ Failed: {result.get('error', 'Unknown')[:50]}")

        # Phase 2: drain deferred non-anchor fits in parallel. Anchors and any
        # state mutations they perform have already completed by this point,
        # so non-anchor workers see a stable snapshot of `state`.
        if deferred_non_anchors:
            workers = min(self.parallel_workers, len(deferred_non_anchors))
            self.logger.info(
                f"⚙️ Parallel non-anchor phase: {len(deferred_non_anchors)} spectra, "
                f"{workers} workers"
            )

            def _run_deferred(job: dict) -> dict:
                # Shallow-copy state per job and pin its regime's
                # locked_fitting_config. Other state fields are shared
                # read-only references; this is cheap and keeps the
                # retry path (which reads locked_fitting_config) per-spectrum
                # correct without mutating shared state.
                job_state = dict(state)
                job_state["locked_fitting_config"] = job["spectrum_config"]
                return self._fit_single_spectrum(
                    state=job_state,
                    curve_data=job["curve_data"],
                    data_path=job["data_path"],
                    spectrum_name=job["spectrum_name"],
                    spectrum_idx=job["idx"],
                    base_script=job["base_script"],
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_job = {pool.submit(_run_deferred, job): job for job in deferred_non_anchors}
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    idx = job["idx"]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.error(
                            f"❌ Spectrum {idx} ({job['spectrum_name']}) raised: {exc}"
                        )
                        result = {
                            "index": idx,
                            "name": job["spectrum_name"],
                            "data_path": job["data_path"],
                            "success": False,
                            "error": str(exc),
                            "parameters": {},
                            "fit_quality": {},
                            "script": job["base_script"],
                            "script_errors": [],
                        }
                    if regime_configs:
                        result["regime"] = job["regime_name"]
                    results_by_idx[idx] = result
                    if result.get("success"):
                        r2 = result.get("fit_quality", {}).get("r_squared")
                        r2_str = f"R²: {r2:.4f}" if r2 else "R²: N/A"
                        self.logger.info(
                            f"✅ [{idx + 1}/{num_spectra}] {result.get('model_type', 'Fit')} - {r2_str}"
                        )
                    else:
                        self.logger.error(
                            f"❌ [{idx + 1}/{num_spectra}] Failed: "
                            f"{(result.get('error') or 'Unknown')[:50]}"
                        )

        series_results = [results_by_idx[i] for i in range(num_spectra)]

        # Restore original locked config
        if original_locked_config:
            state["locked_fitting_config"] = original_locked_config
        
        flagged_spectra = []
        if num_spectra > 1:
            flagged_spectra = self._detect_outliers(series_results, gate=_gate(state))
            
            if flagged_spectra:
                report = self._generate_outlier_report(flagged_spectra, series_results)
                self.logger.warning(report)
                
                flagged_indices = {f["index"] for f in flagged_spectra}
                for r in series_results:
                    if r["index"] in flagged_indices:
                        flag_info = next(f for f in flagged_spectra if f["index"] == r["index"])
                        r["flagged"] = True
                        r["flag_reason"] = flag_info["reason"]
                        r["flag_recommendation"] = flag_info["recommendation"]
                        r["deviation_sigma"] = flag_info.get("deviation_sigma")
                
                flagged_report_path = self.output_dir / "flagged_spectra.json"
                with open(flagged_report_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "r2_threshold": self.r2_threshold,
                        "outlier_sigma": self.outlier_sigma,
                        "total_spectra": num_spectra,
                        "flagged_count": len(flagged_spectra),
                        "flagged_spectra": flagged_spectra
                    }, f, indent=2)
                
                state["flagged_spectra_path"] = str(flagged_report_path)
        
        state["series_results"] = series_results
        state["flagged_spectra"] = flagged_spectra

        # Best-of-N: per-anchor candidate tables (index -> table) for the
        # final result dict.
        anchor_candidate_tables = {
            r["index"]: {
                "candidates": r["anchor_candidates"],
                "judge": r.get("anchor_judge", {}),
            }
            for r in series_results
            if r.get("anchor_candidates")
        }
        if anchor_candidate_tables:
            state["anchor_candidates"] = anchor_candidate_tables

        if is_single and series_results and series_results[0]["success"]:
            first_result = series_results[0]
            state["fit_results"] = {
                "model_type": first_result.get("model_type"),
                "parameters": first_result.get("parameters", {}),
                "fit_quality": first_result.get("fit_quality", {}),
                "deviation_note": first_result.get("deviation_note") or first_result.get("summary"),
            }
            state["final_script"] = first_result.get("script")
            state["final_plot_bytes"] = first_result.get("visualization_bytes")
            
            if first_result.get("visualization_bytes"):
                state["analysis_images"].append({
                    "label": first_result.get("model_type", "Fit"),
                    "data": first_result["visualization_bytes"],
                })
        
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_spectra)
        
        self.logger.info("")
        self.logger.info(f"✅ Fitting complete: {successful}/{num_spectra} successful")
        if flagged_count > 0:
            self.logger.warning(f"⚠️ {flagged_count} spectra flagged for review")
        
        state["series_results_path"] = _write_series_fit_results(
            self.output_dir, state, series_results,
            quality_settings={
                "r2_threshold": self.r2_threshold,
                "max_model_retries": self.max_model_retries,
                "outlier_sigma": self.outlier_sigma,
            },
        )

        return state
    
    def _wrap_text(self, text: str, width: int = 70) -> list:
        """Wrap text to specified width, preserving words."""
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [""]
    
    def _stamp_hot_deviation(self, best_result: dict | None) -> None:
        """Ensure a successful, hot-produced fit carries a ``deviation_note``.

        Reaching the hot (T = n-1) annealing level means the verification loop
        dropped the prior script and let the LLM regenerate the model from
        scratch (``_just_escalated_to_hot``). A fit whose WINNING result was
        produced at that level is, by construction, a departure from the locked
        plan — a "novel pipeline" — regardless of whether the LLM happened to
        fill in a free-text ``deviation_note``. T=2 auto-distillation keys its
        novelty gate on this note, so synthesize a deterministic one when it is
        absent. No-op for non-hot or unsuccessful results, so the gate still
        excludes fits that merely succeeded on the original plan.
        """
        if not isinstance(best_result, dict) or not best_result.get("success"):
            return
        hot = len(self._CONSTRAINT_ANNEALING_SCHEDULE) - 1
        if (best_result.get("_produced_at_level") or 0) < hot:
            return
        if (best_result.get("deviation_note") or "").strip():
            return
        model = best_result.get("model_type") or "a regenerated model"
        best_result["deviation_note"] = (
            f"Abandoned the locked plan during hot annealing (T={hot}) and "
            f"regenerated the fit from scratch, arriving at {model}."
        )

    @staticmethod
    def _build_quality_history(
        best_r2: float,
        r2_threshold: float,
        all_attempts: list,
        verification_history: list,
        judge_result: dict | None,
        script_errors: list | None = None,
    ) -> dict:
        """Build a compact quality history dict for the best result.

        Captures problem-solution pairs at every level: script errors,
        verification iterations, alternative approaches, and judge reasoning.
        """
        return {
            "final_r2": best_r2,
            "threshold": r2_threshold,
            "approved": best_r2 >= r2_threshold,
            "verification_iterations": [
                {
                    "r_squared": entry.get("r_squared"),
                    "annealing_level": entry.get("annealing_level", 0),
                    # The model in force at this iteration — lets a consumer
                    # (e.g. the self-evolution figure) show the per-attempt
                    # model alongside its R² and issues.
                    "model": (entry.get("config_used") or {}).get("physical_model", ""),
                    "issues": [
                        {
                            "location": iss.get("location", ""),
                            "problem": iss.get("problem", ""),
                        }
                        for iss in entry.get("issues_found", [])
                    ],
                    "fix_applied": entry.get("recommended_action", ""),
                }
                for entry in verification_history
            ],
            "alternative_models": [
                {
                    "model": a.get("model", ""),
                    "r2": a.get("r2", 0),
                    "diagnosis": a.get("diagnosis", ""),
                }
                for a in all_attempts[1:]
                if not str(a.get("model", "")).startswith("Verification")
            ],
            "script_errors": script_errors or [],
            "judge_reasoning": (judge_result or {}).get("reasoning"),
        }

    def _judge_select_best_fit(self, attempts: List[dict]) -> dict:
        """
        Present all attempts to a judge LLM to select the best one.

        Called after all retries (verification + alternatives) are exhausted.

        Args:
            attempts: List of dicts with keys:
                - model: display name of the model/attempt
                - result: the fit result dict (includes visualization_bytes)
                - r2: the R² value
                - config (optional): the fitting config used
                - verification (optional): LLM verification dict
        
        Returns:
            Dict with:
                - selected_index: int or None
                - acceptable: bool
                - reasoning: str
                - issues_with_selected: str or None
        """
        self.logger.info("")
        self.logger.info("⚖️ Calling judge to select best fit from all attempts...")

        # Build attempts summary
        attempts_summary = []
        for i, attempt in enumerate(attempts):
            r2 = attempt.get("r2", 0)
            model = attempt.get("model") or attempt.get("config", {}).get("physical_model", "Unknown")
            verification = attempt.get("verification", {})
            assessment = verification.get("overall_assessment", "No assessment available")
            issues = verification.get("issues_found", [])
            
            issues_brief = []
            for issue in issues[:3]:  # Limit to first 3 issues
                issues_brief.append(f"  - {issue.get('location', '?')}: {issue.get('problem', '?')}")
            issues_str = "\n".join(issues_brief) if issues_brief else "  (no specific issues listed)"
            
            summary = f"""
    **Attempt {i + 1}:**
    - Model: {model}
    - R² = {r2:.4f}
    - Assessment: {assessment}
    - Issues ({len(issues)} found):
    {issues_str}
    """
            attempts_summary.append(summary)
        
        prompt_parts = [
            self.JUDGE_PROMPT.format(attempts_summary="\n".join(attempts_summary))
        ]
        
        # Add all visualizations
        for i, attempt in enumerate(attempts):
            viz_bytes = attempt["result"].get("visualization_bytes")
            if viz_bytes:
                prompt_parts.append(f"\n\n**Attempt {i + 1} Visualization:**")
                prompt_parts.append({
                    "mime_type": "image/png",
                    "data": viz_bytes
                })
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"   Judge failed to parse response: {error}")
                return {
                    "selected_index": None, 
                    "acceptable": False, 
                    "reasoning": f"Judge parse failed: {error}"
                }
            
            # Convert 1-indexed response to 0-indexed, with bounds check
            selected = result.get("selected_index")
            if selected is not None:
                selected = int(selected) - 1  # prompt uses 1-indexed labels
                if selected < 0 or selected >= len(attempts):
                    self.logger.warning(
                        f"   Judge returned out-of-range index {selected + 1}, ignoring"
                    )
                    selected = None
                result["selected_index"] = selected

            acceptable = result.get("acceptable", False)
            reasoning = result.get("reasoning", "No reasoning provided")

            if acceptable and selected is not None:
                self.logger.info(f"   ✅ Judge selected attempt {selected + 1}")
            else:
                self.logger.warning(f"   ⚠️ Judge found no acceptable fit")
            
            # Wrap reasoning for readability
            self.logger.info(f"   Reasoning:")
            for line in self._wrap_text(reasoning, width=70):
                self.logger.info(f"      {line}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"   Judge call failed: {e}")
            return {
                "selected_index": None, 
                "acceptable": False, 
                "reasoning": f"Judge call failed: {str(e)}"
            }



class AdaptiveRefitController:
    """
    Post-processing recovery step that re-analyzes flagged spectra independently.

    After the locked-config series processing completes, this controller:
    1. Identifies spectra flagged for quality reasons (below_threshold, fit_failed)
    2. Re-runs each one with full LLM planning + model selection + verification
    3. Updates series_results with improved fits where possible
    4. Re-runs outlier detection on updated results

    Statistical outliers (reason="statistical_outlier") are NOT re-fitted,
    because their low R² may reflect genuine physical phenomena rather
    than model inadequacy.
    """

    REFIT_REASONS = frozenset({"below_threshold", "fit_failed", "outlier_and_below_threshold"})

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
        plot_fn: Callable,
        r2_threshold: float = 0.95,
        max_model_retries: int = 1,
        max_verification_iterations: int = 7,
        enable_human_feedback: bool = False,
        conformance_instructions: str = "",
    ):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.plot_fn = plot_fn
        self.r2_threshold = r2_threshold
        self.enable_human_feedback = enable_human_feedback

        # Compose a fitting helper to reuse _fit_with_quality_control
        self._fitting_helper = UnifiedSeriesProcessingController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=script_instructions,
            correction_instructions=correction_instructions,
            quality_instructions=quality_instructions,
            output_dir=output_dir,
            plot_fn=plot_fn,
            r2_threshold=r2_threshold,
            max_model_retries=max_model_retries,
            enable_human_feedback=False,
            max_verification_iterations=max_verification_iterations,
            conformance_instructions=conformance_instructions,
        )

    def _load_spectrum(self, idx, spectrum_paths, spectrum_stack, column_mapping=None):
        """Load spectrum data for re-analysis (honors the locked column mapping)."""
        if spectrum_stack is not None:
            return spectrum_stack[idx]
        if spectrum_paths and idx < len(spectrum_paths):
            try:
                return self._fitting_helper._load_curve_data(
                    spectrum_paths[idx], column_mapping=column_mapping)
            except Exception as e:
                self.logger.error(f"Failed to load {spectrum_paths[idx]}: {e}")
                return None
        return None

    def _build_refit_state(self, state, curve_data, idx, name):
        """Build a temporary state dict for independent re-analysis."""
        locked_config = state.get("locked_fitting_config", {})
        original_result = state["series_results"][idx]
        original_r2 = original_result.get("fit_quality", {}).get("r_squared") or 0

        # Build experimental context so the LLM knows what it's fitting
        system_info = state.get("system_info", {})
        series_metadata = state.get("series_metadata", {})
        num_spectra = state.get("num_spectra", 0)

        # Serialize full system_info so the LLM gets all metadata
        # regardless of key structure (flat or nested)
        exp_context_parts = []
        if system_info:
            exp_context_parts.append(json.dumps(system_info, indent=2, default=str))
        if series_metadata.get("variable") and series_metadata.get("values"):
            values = series_metadata["values"]
            units = series_metadata.get("units", "")
            if idx < len(values):
                exp_context_parts.append(
                    f"Series position: spectrum {idx + 1}/{num_spectra}, "
                    f"{series_metadata['variable']} = {values[idx]} {units}"
                )
        exp_context = "\n".join(exp_context_parts)

        # Summarize series context: what worked, what failed, neighbors
        series_results = state.get("series_results", [])
        series_context_parts = []
        successful = [r for r in series_results if r.get("success") and not r.get("flagged")]
        if successful:
            r2_vals = [r.get("fit_quality", {}).get("r_squared") or 0 for r in successful]
            series_context_parts.append(
                f"Successful fits (locked model): {len(successful)}/{len(series_results)} spectra, "
                f"R² range {min(r2_vals):.4f}–{max(r2_vals):.4f}, "
                f"model: {successful[0].get('model_type', 'N/A')}"
            )
        flagged = [r for r in series_results if r.get("flagged") or not r.get("success")]
        if flagged:
            flagged_indices = [str(r["index"]) for r in flagged]
            series_context_parts.append(f"Failed spectra indices: [{', '.join(flagged_indices)}]")
        # Nearest successful neighbor summary
        for offset in (-1, 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(series_results):
                nr = series_results[neighbor_idx]
                if nr.get("success") and not nr.get("flagged"):
                    nr2 = nr.get("fit_quality", {}).get("r_squared") or 0
                    series_context_parts.append(
                        f"Neighbor spectrum [{neighbor_idx}] fitted successfully: "
                        f"model={nr.get('model_type', 'N/A')}, R²={nr2:.4f}"
                    )
        series_context = "\n".join(series_context_parts)

        refit_context = (
            f"This spectrum was previously fitted using the locked series model but achieved "
            f"inadequate fit quality (R² = {original_r2:.4f}, threshold = {self.r2_threshold}).\n\n"
            f"The locked model was: {locked_config.get('physical_model', 'Unknown')}\n"
            f"The locked strategy was: {locked_config.get('fitting_strategy', 'Unknown')}\n\n"
        )

        # Add regime context if available
        series_plan = state.get("series_analysis_plan")
        if series_plan:
            for regime in series_plan.get("regimes", []):
                if idx in regime.get("spectrum_indices", []):
                    refit_context += (
                        f"**Regime context:** This spectrum was assigned to regime "
                        f"'{regime.get('name', 'unnamed')}' with expected model: "
                        f"{regime.get('physical_model', 'Unknown')}.\n\n"
                    )
                    break

        if exp_context:
            refit_context += f"**Experimental context:**\n{exp_context}\n\n"
        if series_context:
            refit_context += f"**Series context:**\n{series_context}\n\n"
        refit_context += (
            f"IMPORTANT: The locked model failed for this specific spectrum. You MUST try a DIFFERENT "
            f"fitting approach. Consider:\n"
            f"1. Different functional forms (the locked model's shape may not match this spectrum)\n"
            f"2. Additional components (this spectrum may have features others don't)\n"
            f"3. Different physical models (this spectrum may represent a different physical regime)\n\n"
            f"Do NOT simply retry the same model with different initial parameters.\n\n"
            f"PARSIMONY: Use the SIMPLEST model that achieves R² ≥ {self.r2_threshold}. "
            f"Do not add extra components beyond what the data clearly requires. "
            f"If two peaks are visible, use a two-component model — not three or more."
        )

        fresh_config = {
            "analysis_approach": refit_context,
            "physical_model": f"Alternative to: {locked_config.get('physical_model', 'Unknown')}",
            "fitting_strategy": "Independent analysis - try different approach than locked model",
            "parameters_to_extract": locked_config.get("parameters_to_extract", []),
        }

        spectrum_paths = state.get("spectrum_paths", [])
        data_path = spectrum_paths[idx] if spectrum_paths and idx < len(spectrum_paths) else name

        stats = self._fitting_helper._compute_statistics(curve_data)
        plot_bytes = self.plot_fn(curve_data, state.get("system_info", {}))

        return {
            "data_path": data_path,
            "curve_data": curve_data,
            "original_plot_bytes": plot_bytes,
            "data_statistics": stats,
            "locked_fitting_config": fresh_config,
            "system_info": state.get("system_info", {}),
            "literature_context": state.get("literature_context"),
            "analysis_hints": state.get("analysis_hints"),
            "analysis_objective": state.get("analysis_objective"),
            "skill_name": state.get("skill_name"),
            "skill_sections": state.get("skill_sections"),
            # Carry the multi-aux item list so display + operands flow through
            # this sub-state path too (#226).
            "auxiliary_items": state.get("auxiliary_items", []),
            "prior_knowledge": state.get("prior_knowledge", []),
            "analysis_images": [],
        }

    def _ask_user_for_consensus(self, improved, model_counts):
        """Ask user which model to use when refits found no consensus."""
        print("\n" + "=" * 60)
        print("🔄 ADAPTIVE REFIT: No model consensus among re-fitted spectra")
        print("=" * 60)
        print("\nThe re-fitted spectra used different models:")
        for i, (model, count) in enumerate(
            sorted(model_counts.items(), key=lambda x: -x[1]), 1
        ):
            indices = [str(r["index"]) for r in improved if r["new_model"] == model]
            r2s = [r["new_r2"] for r in improved if r["new_model"] == model]
            r2_str = ", ".join(f"{v:.4f}" for v in r2s)
            print(f"  {i}. '{model}' — spectra [{', '.join(indices)}], R²: {r2_str}")

        print("\nOptions:")
        print("  • Enter a number (1, 2, ...) to use that model for all re-fitted spectra")
        print("  • Type a model name to suggest a different model")
        print("  • Press Enter to keep the independent results as-is")
        print("-" * 60)

        response = input("\n🤔 Your choice: ").strip()
        if not response:
            print("✅ Keeping independent refit results.")
            return None

        # Check if user entered a number
        try:
            choice = int(response)
            models = sorted(model_counts.keys(), key=lambda m: -model_counts[m])
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                print(f"✅ Will re-fit with '{selected}'")
                return selected
        except ValueError:
            pass

        # User typed a model name directly
        print(f"✅ Will re-fit with '{response}'")
        return response

    def _run_consistency_refit(
        self, minority, target_model, improved, state, series_results,
        spectrum_paths, spectrum_stack,
    ):
        """Re-fit minority spectra using the target model."""
        peer_r2 = [r["new_r2"] for r in improved if r["new_model"] == target_model]
        peer_count = len(peer_r2)

        for entry in minority:
            idx = entry["index"]
            name = entry["name"]
            self.logger.info(f"  Re-fitting [{idx}] {name} with '{target_model}'")

            curve_data = self._load_spectrum(
                idx, spectrum_paths, spectrum_stack,
                column_mapping=state.get("column_mapping_locked"))
            if curve_data is None:
                continue

            refit_state = self._build_refit_state(state, curve_data, idx, name)
            if peer_r2:
                refit_state["locked_fitting_config"]["analysis_approach"] += (
                    f"\n\n**Peer evidence:** {peer_count} other spectra in this series "
                    f"were successfully refitted with '{target_model}' "
                    f"(R² {min(peer_r2):.4f}–{max(peer_r2):.4f}). "
                    f"Strongly prefer this model unless the data clearly "
                    f"requires something different."
                )
            refit_state["locked_fitting_config"]["physical_model"] = target_model

            data_path = (spectrum_paths[idx]
                         if spectrum_paths and idx < len(spectrum_paths) else name)
            try:
                result = self._fitting_helper._fit_with_quality_control(
                    state=refit_state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=name, spectrum_idx=idx,
                    is_regime_anchor=True,  # enable full verification for independent refit
                )
            except Exception as e:
                self.logger.error(f"  Consistency refit failed for {name}: {e}")
                continue

            new_r2 = result.get("fit_quality", {}).get("r_squared") or 0
            prev_r2 = entry["new_r2"] or 0

            if result["success"] and new_r2 >= prev_r2 * 0.99:
                self.logger.info(f"  ✅ Consistent: R² {new_r2:.4f} with '{target_model}'")
                result["adaptively_refitted"] = True
                result["original_r2"] = entry["original_r2"]
                result["refit_model_type"] = result.get("model_type")
                result["locked_model_type"] = state.get(
                    "locked_fitting_config", {}
                ).get("physical_model")
                series_results[idx] = result
                entry["new_r2"] = new_r2
                entry["new_model"] = result.get("model_type")
            elif self.enable_human_feedback:
                keep = self._ask_keep_consistency_result(
                    name, idx, target_model, new_r2,
                    entry["new_model"], prev_r2,
                )
                if keep:
                    result["adaptively_refitted"] = True
                    result["original_r2"] = entry["original_r2"]
                    result["refit_model_type"] = result.get("model_type")
                    result["locked_model_type"] = state.get(
                        "locked_fitting_config", {}
                    ).get("physical_model")
                    series_results[idx] = result
                    entry["new_r2"] = new_r2
                    entry["new_model"] = result.get("model_type")
                else:
                    self.logger.info(f"  Keeping original refit for [{idx}] {name}")
            else:
                self.logger.info(
                    f"  Keeping original refit: consensus R²={new_r2:.4f} "
                    f"vs previous R²={prev_r2:.4f}"
                )

    def _ask_keep_consistency_result(
        self, name, idx, consensus_model, consensus_r2,
        original_model, original_r2,
    ):
        """Ask user whether to keep consensus model when R² dropped."""
        print("\n" + "-" * 60)
        print(f"⚠️  Spectrum [{idx}] {name}: consensus model has lower R²")
        print("-" * 60)
        print(f"  Consensus: '{consensus_model}' → R² = {consensus_r2:.4f}")
        print(f"  Independent: '{original_model}' → R² = {original_r2:.4f}")
        print("\nOptions:")
        print(f"  • Type 'consensus' to use '{consensus_model}' for consistency")
        print(f"  • Press Enter to keep '{original_model}'")

        response = input("\nYour choice: ").strip().lower()
        if response == "consensus":
            print(f"✅ Using consensus model for [{idx}] {name}")
            return True
        print(f"✅ Keeping independent model for [{idx}] {name}")
        return False

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if state.get("is_single_spectrum", True):
            return state

        # Scoring-gated (non-R²) skills — e.g. XRD phase identification — lock a
        # phase set on the anchor frame by design. Adaptive refit re-derives the
        # model per frame (for ID that means re-identifying), which defeats the
        # lock; and a lower score on a later frame is the physical signal (a phase
        # being consumed / a transformation), NOT a bad fit to repair. So skip the
        # refit for these — the flagged frames are surfaced in the report for
        # interpretation instead. (This is also what eliminated the pathological
        # 100+ min "re-analysis" grind on real in-situ series: each flagged frame
        # was being re-fit under the R² verification loop.) R² skills (all curve
        # fitting) are unaffected — they fall through to the refit below.
        if _gate(state).metric != "r_squared":
            self.logger.info(
                "\n🔄 Adaptive refit: scoring-gated skill (non-R² gate) — the phase "
                "set is locked by design; skipping per-frame model re-derivation. "
                "Lower per-frame scores are reported as physical evolution."
            )
            return state

        flagged_spectra = state.get("flagged_spectra", [])
        if not flagged_spectra:
            self.logger.info("\n🔄 Adaptive refit: No flagged spectra, skipping.")
            return state

        refit_candidates = [f for f in flagged_spectra if f["reason"] in self.REFIT_REASONS]
        if not refit_candidates:
            self.logger.info("\n🔄 Adaptive refit: Flagged spectra are statistical outliers only, skipping.")
            return state

        self.logger.info(f"\n🔄 ADAPTIVE REFIT: {len(refit_candidates)} spectra to re-analyze independently")

        series_results = state.get("series_results", [])
        spectrum_paths = state.get("spectrum_paths", [])
        spectrum_stack = state.get("spectrum_stack")
        refit_summary = []

        for flagged in refit_candidates:
            idx = flagged["index"]
            name = flagged["name"]
            original_r2 = flagged.get("r_squared")

            self.logger.info(f"\n  Re-analyzing [{idx}] {name} (original R²={original_r2})")

            curve_data = self._load_spectrum(
                idx, spectrum_paths, spectrum_stack,
                column_mapping=state.get("column_mapping_locked"))
            if curve_data is None:
                self.logger.warning(f"  Could not load spectrum data for {name}, skipping")
                continue


            refit_state = self._build_refit_state(state, curve_data, idx, name)
            spectrum_paths_list = state.get("spectrum_paths", [])
            data_path = spectrum_paths_list[idx] if spectrum_paths_list and idx < len(spectrum_paths_list) else name

            try:
                refit_result = self._fitting_helper._fit_with_quality_control(
                    state=refit_state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=name, spectrum_idx=idx,
                    is_regime_anchor=True,  # enable full verification for independent refit
                )
            except Exception as e:
                self.logger.error(f"  Refit failed for {name}: {e}")
                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": None,
                    "improved": False,
                })
                continue

            new_r2 = refit_result.get("fit_quality", {}).get("r_squared") or 0
            locked_model = state.get("locked_fitting_config", {}).get("physical_model")

            if refit_result["success"] and (original_r2 is None or new_r2 > original_r2):
                self.logger.info(f"  ✅ Improved: R² {original_r2} → {new_r2:.4f}")
                refit_result["adaptively_refitted"] = True
                refit_result["original_r2"] = original_r2
                refit_result["refit_model_type"] = refit_result.get("model_type")
                refit_result["locked_model_type"] = locked_model
                series_results[idx] = refit_result

                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": new_r2,
                    "original_model": locked_model,
                    "new_model": refit_result.get("model_type"),
                    "improved": True,
                })
            else:
                self.logger.info(f"  No improvement: R² {original_r2} → {new_r2:.4f}, keeping original")
                # The refit ran in the same spectrum directory and overwrote the
                # kept fit's visualization.png. Restore the kept fit's plot so the
                # report shows the fit whose metrics it records — not the
                # discarded refit's. (fit.npy is realigned at read time, below.)
                kept = series_results[idx] if idx < len(series_results) else None
                if isinstance(kept, dict):
                    vpath = kept.get("visualization_path")
                    vbytes = kept.get("visualization_bytes")
                    if vpath and vbytes:
                        try:
                            with open(vpath, "wb") as fh:
                                fh.write(vbytes)
                        except Exception as e:  # noqa: BLE001
                            self.logger.warning(
                                f"  Could not restore original plot for {name}: {e}"
                            )
                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": new_r2,
                    "improved": False,
                })

        # --- Consistency pass ---
        # If a majority of improved refits converged on the same model type,
        # re-refit outlier models using the consensus as guidance.
        # If no consensus, optionally ask the user for guidance.
        improved = [r for r in refit_summary if r["improved"] and r.get("new_model")]
        if len(improved) >= 2:
            model_counts = {}
            for r in improved:
                model_counts[r["new_model"]] = model_counts.get(r["new_model"], 0) + 1
            top_model, top_count = max(model_counts.items(), key=lambda x: x[1])
            has_majority = top_count > len(improved) / 2
            minority = [r for r in improved if r["new_model"] != top_model]

            if has_majority and minority:
                self.logger.info(
                    f"\n🔄 Consistency pass: majority model is '{top_model}' "
                    f"({top_count}/{len(improved)}), re-fitting {len(minority)} outlier(s)"
                )
                self._run_consistency_refit(
                    minority, top_model, improved, state, series_results,
                    spectrum_paths, spectrum_stack,
                )
            elif not has_majority and len(model_counts) > 1:
                if self.enable_human_feedback:
                    # No consensus — ask user for guidance
                    user_model = self._ask_user_for_consensus(improved, model_counts)
                    if user_model:
                        user_minority = [r for r in improved if r["new_model"] != user_model]
                        if user_minority:
                            self.logger.info(
                                f"\n🔄 User-guided consistency: re-fitting "
                                f"{len(user_minority)} spectra with '{user_model}'"
                            )
                            self._run_consistency_refit(
                                user_minority, user_model, improved, state,
                                series_results, spectrum_paths, spectrum_stack,
                            )
                else:
                    # No human feedback and no consensus — keep independent results.
                    # The parsimony prompt should minimize this case; if models still
                    # disagree, the inconsistency is noted in the synthesis.
                    self.logger.info(
                        f"\n⚠️ No model consensus among refitted spectra "
                        f"({dict(model_counts)}). Keeping independent results."
                    )

        state["series_results"] = series_results
        state["refit_summary"] = refit_summary

        # Re-run outlier detection with updated results
        updated_flagged = self._fitting_helper._detect_outliers(series_results)
        state["flagged_spectra"] = updated_flagged

        improved_count = sum(1 for r in refit_summary if r["improved"])
        self.logger.info(f"\n🔄 Adaptive refit complete: {improved_count}/{len(refit_candidates)} spectra improved")

        # Refresh series_fit_results.json so adopted refits are reflected. It was
        # written before this step (post-initial-fit) and feeds the BO/planning
        # feature table and the #172 prior-run reference summary; without this,
        # refitted spectra would carry their pre-refit values there.
        if improved_count > 0:
            fh = self._fitting_helper
            _write_series_fit_results(
                self.output_dir, state, series_results,
                quality_settings={
                    "r2_threshold": fh.r2_threshold,
                    "max_model_retries": fh.max_model_retries,
                    "outlier_sigma": fh.outlier_sigma,
                },
            )

        return state


class ConditionalTrendAnalysisController:
    """Generates and executes custom Python script for trend analysis. Only for n>=2."""
    
    TREND_ANALYSIS_INSTRUCTIONS = '''You are analyzing a series of fitted spectra/curves to identify trends.
{objective}
**SERIES SUMMARY:**
{series_summary}

**SERIES METADATA:**
{series_metadata}

**FLAGGED SPECTRA:**
{flagged_info}

**CRITICAL REQUIREMENTS:**
1. DO NOT use plt.show() anywhere in the script - only save figures with plt.savefig()
2. DO NOT include individual spectrum fit visualizations - only create parameter trend dashboard
3. Use plt.close('all') after saving each figure to free memory

**VISUALIZATION SCOPE - TRENDS:**
Create a SINGLE dashboard figure showing how fitted PARAMETERS evolve across the series.
DO NOT recreate individual spectrum fits - those already exist separately.

The series may vary ONE control variable or SEVERAL at once (a factorial /
grid design). Inspect `series_metadata` for a `secondary_variables` entry and
choose the representation to match:
- ONE control variable: parameter values (y-axis) vs that variable (x-axis),
  with error bars where available - the standard trend dashboard.
- TWO control variables: represent BOTH. If their values define a regular
  lattice (grid sampling), use a heatmap or filled contour of each key
  parameter over the 2-D space. If the sampling is scattered, use a scatter
  plot positioned by the two variables and colored by the parameter value.
  Detect grid vs scattered from the data itself.
- THREE OR MORE: there is no single canonical N-D trend plot - produce a
  best-effort view: plot each parameter against the primary variable and
  facet or color by the remaining variable(s), or use pairwise panels.
- In every case also show fit quality (R²) evolution and mark flagged
  spectra with distinct markers.
State the representation you chose (and why) in `analysis_approach`.

**FIGURE REQUIREMENTS:**
- Create ONE summary dashboard figure (parameter_trends.png)
- 2x2 or 2x3 subplot layout with 4-6 most important parameters
- Clean, publication-quality appearance
- Mark flagged spectra with red X markers
- Include linear regression trend lines where appropriate
- NO plt.show() calls
- Use plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
- Call plt.close('all') at the end

**DATA EXTRACTION PATTERN:**
```python
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - REQUIRED
import matplotlib.pyplot as plt

# Load data
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
series_metadata = data.get('series_metadata', {{}})
# PRIMARY control variable:
#   series_metadata['variable'] (name), series_metadata['unit'],
#   series_metadata['values'] -> a list aligned to results by index:
#   results[i] primary value = series_metadata['values'][results[i]['index']].
# ADDITIONAL control variables (present only for a grid / factorial design):
#   series_metadata.get('secondary_variables', []) -> a list of entries
#   {{'variable': name, 'unit': unit, 'values': {{filename: value}}}}.
#   Each secondary 'values' is a dict keyed by file name; align it to a
#   result via key = os.path.basename(results[i]['data_path']).

# Extract series variable and parameters...
# Create figure with subplots...
# Plot parameter trends (NOT individual fits)...

plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display
```

Return JSON with:
{{
    "analysis_approach": "brief description",
    "key_metrics": ["list", "of", "parameters", "tracked"],
    "flagged_handling": "how flagged spectra are marked",
    "expected_outputs": ["parameter_trends.png"],
    "script": "full python script - NO plt.show()"
}}
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, executor: Any, output_dir: str, max_corrections: int = 3):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.max_corrections = max_corrections

    def _generate_trend_script(self, state: dict) -> Optional[Dict]:
        series_results = state.get("series_results", [])
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])

        param_summary = []
        for r in series_results:
            if r["success"]:
                summary = {"index": r["index"], "name": r["name"], "model_type": r.get("model_type"),
                          "parameters": r.get("parameters", {}), "fit_quality": r.get("fit_quality", {})}
                if r.get("flagged"):
                    summary["flagged"] = True
                    summary["flag_reason"] = r.get("flag_reason")
                param_summary.append(summary)

        flagged_info = json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."

        objective = state.get("analysis_objective")
        objective_block = (
            f"\n**ANALYSIS OBJECTIVE:**\n{objective}\n"
            "Frame the trend analysis around answering this objective. "
            "If the objective involves calibration or quantitative modeling, "
            "the script must compute and output regression models.\n"
        ) if objective else ""

        prompt = self.TREND_ANALYSIS_INSTRUCTIONS.format(
            series_summary=json.dumps(param_summary, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            flagged_info=flagged_info,
            objective=objective_block,
        )
        
        try:
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = parse_codegen_response(response, field="script", logger=self.logger)
            if error_dict and not (result_json and 'script' in result_json):
                return None
            return result_json
        except Exception as e:
            self.logger.error(f"Error generating trend script: {e}")
            return None

    def _execute_script(self, script: str) -> tuple:
        # Remove any plt.show() calls that might have slipped through
        script = re.sub(r'plt\.show\s*\(\s*\)', '# plt.show() removed', script)
        
        # Ensure matplotlib backend is set at the top
        if 'matplotlib.use' not in script:
            script = "import matplotlib\nmatplotlib.use('Agg')\n" + script
        
        script_path = self.output_dir / "trend_analysis.py"
        with open(script_path, 'w') as f:
            f.write(script)
        result = self.executor.execute_script(script, working_dir=str(self.output_dir))
        return result.get("status") == "success", result.get("stdout", ""), result.get("message", "")

    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        if len(error_message) > 1000:
            error_message = error_message[:500] + "\n...[truncated]...\n" + error_message[-500:]
        
        prompt = f"""Fix this Python script that failed:

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_message}
```

Return JSON with: {{"diagnosis": "...", "script": "corrected script"}}
"""
        
        try:
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, _ = parse_codegen_response(response, field="script", logger=self.logger)
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        if is_single or num_spectra < 2:
            self.logger.info("\n📊 Trend analysis skipped (single spectrum mode).\n")
            state["trend_analysis_results"] = {"success": True, "skipped": True, "reason": "Single spectrum - no trend analysis applicable"}
            return state
        
        self.logger.info("")
        self.logger.info("📈 TREND ANALYSIS")
        
        flagged_count = len(state.get("flagged_spectra", []))
        if flagged_count > 0:
            self.logger.info(f"   Note: {flagged_count} flagged spectra will be highlighted in visualizations")
        
        script_result = self._generate_trend_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate trend analysis script.")
            state["trend_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        self.logger.info(f"   📊 Approach: {script_result.get('analysis_approach', 'unknown')}")
        self.logger.info(f"   📈 Metrics: {script_result.get('key_metrics', [])}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_corrections + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Trend analysis completed!")
                break
            
            self.logger.warning(f"   ⚠️ Script failed: {stderr[:200]}...")
            
            if attempt < self.max_corrections:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        # Only include trend analysis outputs, NOT individual fit images or review files
        for f in self.output_dir.glob('*.png'):
            # Exclude individual fit visualizations (they have _fit.png suffix or spectrum_ prefix with _fit)
            fname = f.name
            if '_fit.png' in fname:
                continue
            if fname.startswith('spectrum_') and fname.endswith('.png'):
                continue
            # Exclude other known non-trend files
            if fname in ['quality_review_fit.png', 'first_spectrum_fit_review.png']:
                continue
            generated_files.append(str(f))
        
        # Include any CSV/JSON outputs from trend analysis
        for f in self.output_dir.glob('*.csv'):
            if f.name not in ['series_fit_results.json', 'flagged_spectra.json']:
                generated_files.append(str(f))
        
        state["trend_analysis_results"] = {
            "success": success, "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics"),
            "flagged_handling": script_result.get("flagged_handling"),
            "stdout": stdout, "stderr": stderr if not success else None,
            "generated_files": generated_files,
            "script_path": str(self.output_dir / "trend_analysis.py")
        }
        
        return state


class UnifiedCurveSynthesisController:
    """Synthesizes findings into scientific claims. Adapts to single vs series.

    Series synthesis uses a staged (model-blind) prompt structure: Stage 1
    enumerates candidate causes from evidence without knowing the locked
    model; Stage 2 discloses the locked model (and any refit models) for
    reconciliation; Stage 3 is the output schema. Dropped leading examples
    that previously pre-seeded explanations like "phase transition."
    """

    SERIES_STAGE1 = '''You are synthesizing findings from a curve fitting analysis of a spectral series. \
You will do this in three stages. Do the stages in order — do NOT skip ahead and do NOT preview \
the locked model name (disclosed only in Stage 2).

## Stage 1 — Hypothesis from the evidence (model-blind)

**Series overview:**
- Total spectra: {num_spectra}
- Successfully fit: {successful_fits}
- Flagged for review: {flagged_count}

Below you will see:
- Per-spectrum parameter values with uncertainties, R², and status flags (flagged / adaptively refitted).
- Flagged-spectrum visualizations (where the fit is poor).
- Trend plots for the series, if generated.
- Series metadata (independent variable, values, unit) and sample metadata.

The model class used for the fit will NOT be shown at this stage. Refitted spectra \
are marked as having been refitted, but the alternate model name is withheld for Stage 2.

Working from the evidence alone, address these questions:
- What parameter trends (or non-trends) does the series show — smooth, stepwise, noisy, non-monotonic?
- Do the flagged/refitted spectra cluster at particular values of the series variable, or appear scattered?
- From residual patterns on flagged spectra, does the original fit model look systematically inadequate, \
or only occasionally off?
- Enumerate 2-3 candidate physical causes that could produce the observed trend and anomalies. \
For each candidate, state the evidence that supports it AND the evidence that would contradict it.

Do NOT assert a single cause yet.

**Per-spectrum fit summaries (Stage 1 view — model name withheld):**
{fit_summaries}

**Flagged spectra:**
{flagged_summary}

**Adaptive refit status (counts only; alternate models withheld):**
{refit_status}

**Trend analysis:**
{trend_results}

**Series metadata:**
{series_metadata}

**System information:**
{system_info}
'''

    SERIES_STAGE2_TMPL = '''

## Stage 2 — Reconcile with the fitted model(s)

The locked mathematical model used for this series was: **{locked_model}**

{refit_disclosure}

Compare to your Stage 1 hypothesis:
- Does the locked model align with what the evidence would have suggested?
- For any refitted spectra, do the alternate models correspond to the candidate causes you enumerated \
in Stage 1, or do they suggest a different explanation?
- Qualify each physical claim by how well the data constrains the model and by any model-vs-evidence \
divergence you identify.
'''

    SERIES_STAGE3 = '''

## Stage 3 — Output JSON

Return a single JSON object with these keys. Leave fields blank when the evidence does not support a \
claim — do NOT fill them speculatively, and do NOT assert a specific cause (e.g., phase transition, \
instrumental drift) without explicit evidence.

```json
{
    "stage1_hypothesis": "your evidence-only conclusion from Stage 1, including the 2-3 candidate causes enumerated and what evidence supports or contradicts each",
    "model_reconciliation": "whether the locked (and any refit) models align with Stage 1, and what that implies",
    "detailed_analysis": "comprehensive scientific interpretation integrating Stage 1 and Stage 2",
    "scientific_claims": [
        {
            "claim": "specific claim statement",
            "scientific_impact": "why this matters",
            "has_anyone_question": "research question formulation",
            "keywords": ["keyword1", "keyword2"]
        }
    ],
    "parameter_trends": {
        "parameter_name": {"trend": "increasing/decreasing/stable/non-monotonic", "interpretation": "physical meaning"}
    },
    "flagged_spectra_analysis": {
        "summary": "",
        "possible_causes": [],
        "recommended_followup": [],
        "scientific_significance": ""
    },
    "refit_analysis": {
        "summary": "",
        "model_changes": [],
        "scientific_implications": ""
    },
    "caveats": "limitations and considerations, including model-vs-data divergence from Stage 2"
}
```

**Number of claims:** emit **at most 2** `scientific_claims` for a series — the \
dominant trend across the spectra (1 claim) plus, only if independent, a \
secondary finding (e.g. a flagged-spectrum anomaly that doesn't fit the \
trend). One claim is the right answer when the series tells a single \
coherent story. Never more than 2. Do not pad with restatements of the \
same trend.
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, single_spectrum_instructions: str, output_dir: str):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.single_spectrum_instructions = single_spectrum_instructions
        self.output_dir = Path(output_dir)

    def _synthesize_single_spectrum(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("🔬 SINGLE SPECTRUM INTERPRETATION")

        from ..instruct import (
            FITTING_INTERPRETATION_STAGE1,
            FITTING_INTERPRETATION_STAGE2_TMPL,
            FITTING_INTERPRETATION_STAGE3,
            ID_MODE_INTERPRETATION_STAGE1_ADDENDUM,
            ID_MODE_OUTPUT_ADDENDUM,
        )

        fit_results = state.get("fit_results", {})
        series_results = state.get("series_results", [])
        is_id_mode = state.get("task_mode") == "identification"

        quality_warning = None
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]

        # Stage 1 — model-blind, instructs the LLM not to look ahead.
        prompt_parts = [FITTING_INTERPRETATION_STAGE1]
        if is_id_mode:
            prompt_parts.append(ID_MODE_INTERPRETATION_STAGE1_ADDENDUM)

        # Evidence block. Appears BEFORE the model name is disclosed (Stage 2).
        prompt_parts.extend([
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
        ])
        if state.get("final_plot_bytes"):
            prompt_parts.extend([
                "\n## Fit and Residuals (plot labels are neutral: Data/Fit/Components/Residuals only)",
                {"mime_type": "image/png", "data": state["final_plot_bytes"]},
            ])
        prompt_parts.extend([
            "\n## Fitted parameters (with uncertainties)\n"
            + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit quality metrics\n"
            + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Sample metadata\n"
            + json.dumps(state.get("system_info", {}), indent=2),
        ])

        if quality_warning:
            prompt_parts.append(
                f"\n## Quality warning\n{quality_warning}\n"
                "Note: Alternative models were attempted but this was the best fit achieved."
            )

        if series_results and series_results[0].get("quality_history"):
            prompt_parts.append(
                "\n## Quality history (verification & retry context)\n"
                + json.dumps(series_results[0]["quality_history"], indent=2)
            )

        if state.get("literature_context"):
            prompt_parts.extend(["\n## Literature", state["literature_context"]])

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_deviation_note(prompt_parts, fit_results)

        # Stage 2 — disclose the fitted model AFTER all evidence has been shown.
        prompt_parts.append(
            FITTING_INTERPRETATION_STAGE2_TMPL.format(
                model_type=fit_results.get("model_type", "Unknown model")
            )
        )

        # Stage 3 — output schema (+ id-mode addendum when applicable).
        prompt_parts.append(FITTING_INTERPRETATION_STAGE3)
        if is_id_mode:
            prompt_parts.append(ID_MODE_OUTPUT_ADDENDUM)

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)

            if error_dict:
                salvaged = salvage_synthesis_from_response(response)
                if salvaged:
                    self.logger.warning("Synthesis JSON parse failed; salvaged detailed_analysis from raw text.")
                    state["synthesis_result"] = salvaged
                else:
                    self.logger.error(f"Synthesis failed: {error_dict}")
                    state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Single spectrum synthesis complete.")
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}

        return state

    def _synthesize_series(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("🔬 SERIES SYNTHESIS")

        from ..instruct import (
            ID_MODE_INTERPRETATION_STAGE1_ADDENDUM,
            ID_MODE_OUTPUT_ADDENDUM,
        )

        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])
        is_id_mode = state.get("task_mode") == "identification"

        successful_fits = [r for r in series_results if r["success"]]

        # Stage 1 per-spectrum summaries are model-blind: the `model` name and
        # `refit_model`/`locked_model` fields are withheld. Flags that describe
        # the fit's status (flagged, adaptively_refitted) are retained because
        # they describe outcome, not physics.
        fit_summaries_stage1 = []
        for r in successful_fits[:15]:
            s = {
                "index": r["index"],
                "name": r["name"],
                "key_params": r.get("parameters", {}),
                "r_squared": r.get("fit_quality", {}).get("r_squared"),
            }
            if r.get("flagged"):
                s["flagged"] = True
                s["flag_reason"] = r.get("flag_reason")
            if r.get("adaptively_refitted"):
                s["adaptively_refitted"] = True
                s["original_r2"] = r.get("original_r2")
            if r.get("quality_history"):
                s["quality_history"] = r["quality_history"]
            fit_summaries_stage1.append(s)

        n_refitted = sum(1 for r in successful_fits if r.get("adaptively_refitted"))
        if n_refitted == 0:
            refit_status = "No spectra were adaptively refitted."
        else:
            refit_status = (
                f"{n_refitted} spectrum/spectra were refitted with an alternate configuration. "
                "Alternate model names are withheld until Stage 2 — for Stage 1, treat 'was refitted' "
                "as a status flag indicating the original model performed poorly on that spectrum."
            )

        flagged_summary = (
            json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."
        )

        # Stage 2 disclosure: locked model name + any refit model names.
        locked_model = (
            state.get("locked_fitting_config", {}).get("physical_model")
            or (successful_fits[0].get("model_type") if successful_fits else "Unknown")
        )
        refit_rows = [
            {
                "index": r["index"],
                "name": r["name"],
                "locked_model": r.get("locked_model_type"),
                "refit_model": r.get("refit_model_type"),
            }
            for r in successful_fits if r.get("adaptively_refitted")
        ]
        if not refit_rows:
            refit_disclosure = "No spectra were adaptively refitted."
        else:
            refit_disclosure = (
                f"{len(refit_rows)} spectrum/spectra were refitted with alternate models:\n"
                + json.dumps(refit_rows, indent=2)
            )

        stage1_text = self.SERIES_STAGE1.format(
            num_spectra=state.get("num_spectra", 1),
            successful_fits=len(successful_fits),
            flagged_count=len(flagged_spectra),
            fit_summaries=json.dumps(fit_summaries_stage1, indent=2),
            flagged_summary=flagged_summary,
            refit_status=refit_status,
            trend_results=json.dumps(trend_results, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            system_info=json.dumps(state.get("system_info", {}), indent=2),
        )
        prompt_parts = [stage1_text]
        if is_id_mode:
            prompt_parts.append(ID_MODE_INTERPRETATION_STAGE1_ADDENDUM)

        if flagged_spectra:
            prompt_parts.append("\n\n**Flagged spectra — visual evidence:**")
            flagged_indices = {f["index"] for f in flagged_spectra}
            included_count = 0
            for r in series_results:
                if r["index"] in flagged_indices and r.get("visualization_bytes") and included_count < 5:
                    prompt_parts.append(
                        f"\n{r['name']} (flagged: {r.get('flag_reason', 'unknown')}):"
                    )
                    prompt_parts.append(
                        {"mime_type": "image/png", "data": r["visualization_bytes"]}
                    )
                    included_count += 1

        if trend_results.get("success") and trend_results.get("generated_files"):
            prompt_parts.append("\n\n**Trend visualizations:**")
            for file_path in trend_results["generated_files"][:5]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({"mime_type": "image/png", "data": f.read()})

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)

        # Aggregate non-empty per-spectrum deviation_note strings (process notes
        # only). Inject with neutral framing so the LLM does not treat them as
        # findings.
        deviation_lines = []
        for r in successful_fits:
            note = (r.get("deviation_note") or "").strip()
            if note:
                name = r.get("name", f"spectrum_{r.get('index', '?')}")
                deviation_lines.append(f"- {name}: {note}")
        if deviation_lines:
            prompt_parts.append(
                "\n## Fitting-stage process notes (not findings)\n"
                "The fitter recorded the following notes about deviations from the plan "
                "or unusual adjustments for specific spectra. Process context only — do "
                "not treat as scientific conclusions.\n"
                + "\n".join(deviation_lines)
            )

        # Stage 2 — disclose locked and refit models AFTER all evidence shown.
        prompt_parts.append(
            self.SERIES_STAGE2_TMPL.format(
                locked_model=locked_model,
                refit_disclosure=refit_disclosure,
            )
        )

        # Stage 3 — output schema (+ id-mode addendum when applicable).
        prompt_parts.append(self.SERIES_STAGE3)
        if is_id_mode:
            prompt_parts.append(ID_MODE_OUTPUT_ADDENDUM)

        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse(response)

            if error_dict:
                salvaged = salvage_synthesis_from_response(response)
                if salvaged:
                    self.logger.warning("Series synthesis JSON parse failed; salvaged detailed_analysis from raw text.")
                    state["synthesis_result"] = salvaged
                else:
                    self.logger.error(f"Series synthesis failed: {error_dict}")
                    state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Series synthesis complete.")
        except Exception as e:
            self.logger.error(f"Series synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            return self._synthesize_single_spectrum(state)
        else:
            return self._synthesize_series(state)


class UnifiedCurveReportController:
    """Generates final HTML report for series analysis. Only for n>=2."""
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = Path(output_dir)

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def _generate_flagged_spectra_section(self, flagged_spectra: List[dict], series_results: List[dict], synthesis: dict) -> str:
        if not flagged_spectra:
            return ""
        
        flagged_analysis = synthesis.get("flagged_spectra_analysis", {})
        
        html = f"""
        <h2>⚠️ Flagged Spectra</h2>
        <div class="flagged-summary">
            <p><strong>{len(flagged_spectra)} spectra flagged for review</strong></p>
            <p>{flagged_analysis.get("summary", "Some spectra showed anomalous fitting behavior.")}</p>
        </div>
"""
        
        causes = flagged_analysis.get("possible_causes", [])
        if causes:
            html += "<h3>Possible Causes</h3><ul>"
            for cause in causes:
                html += f"<li>{cause}</li>"
            html += "</ul>"
        
        followup = flagged_analysis.get("recommended_followup", [])
        if followup:
            html += "<h3>Recommended Follow-up</h3><ul>"
            for item in followup:
                html += f"<li>{item}</li>"
            html += "</ul>"
        
        significance = flagged_analysis.get("scientific_significance", "")
        if significance:
            html += f"<h3>Scientific Significance</h3><p>{significance}</p>"
        
        html += '<h3>Flagged Spectra Details</h3><div class="flagged-grid">'
        
        badge_colors = {
            "fit_failed": ("#dc3545", "Failed"),
            "statistical_outlier": ("#fd7e14", "Outlier"),
            "below_threshold": ("#ffc107", "Low R²"),
            "outlier_and_below_threshold": ("#dc3545", "Critical"),
        }
        
        for f in flagged_spectra:
            result = next((r for r in series_results if r["index"] == f["index"]), None)
            color, label = badge_colors.get(f["reason"], ("#6c757d", "Flagged"))
            
            html += f'<div class="flagged-card" style="border-color: {color};">'
            html += f'<div class="flagged-card-header"><strong>{f["name"]}</strong>'
            html += f'<span class="flagged-badge" style="background-color: {color};">{label}</span></div>'
            
            if f.get("r_squared") is not None:
                html += f'<p><strong>R²:</strong> {f["r_squared"]:.4f} (series median: {f["series_mean"]:.4f})</p>'
                if f.get("deviation_sigma") is not None:
                    html += f'<p><strong>Deviation:</strong> {f["deviation_sigma"]:.1f}σ below median</p>'
            
            html += f'<p class="flagged-recommendation">{f["recommendation"]}</p>'
            
            if result and result.get("visualization_path") and Path(result["visualization_path"]).exists():
                with open(result["visualization_path"], 'rb') as img_f:
                    b64 = self._image_to_base64(img_f.read())
                html += f'<img src="data:image/png;base64,{b64}" alt="{f["name"]}">'
            
            html += '</div>'
        
        html += '</div>'
        return html

    def _generate_refit_section(self, refit_summary: List[dict], series_results: List[dict]) -> str:
        """Generate HTML section for adaptive refit results."""
        if not refit_summary:
            return ""

        improved = [r for r in refit_summary if r["improved"]]
        not_improved = [r for r in refit_summary if not r["improved"]]

        html = f"""
        <h2>🔄 Adaptive Re-Fitting Results</h2>
        <div class="refit-summary">
            <p><strong>{len(improved)}/{len(refit_summary)}</strong> spectra improved through independent re-analysis</p>
        </div>
"""

        if improved:
            html += '<h3>Improved Fits</h3><table class="params-table"><thead><tr>'
            html += '<th>Spectrum</th><th>Original R²</th><th>New R²</th><th>Original Model</th><th>New Model</th>'
            html += '</tr></thead><tbody>'
            for r in improved:
                orig_r2 = f"{r['original_r2']:.4f}" if r.get("original_r2") is not None else "Failed"
                new_r2 = f"{r['new_r2']:.4f}" if r.get("new_r2") is not None else "N/A"
                html += f'<tr><td>{r["name"]}</td><td>{orig_r2}</td><td>{new_r2}</td>'
                html += f'<td>{r.get("original_model", "N/A")}</td><td>{r.get("new_model", "N/A")}</td></tr>'
            html += '</tbody></table>'

        if not_improved:
            html += '<h3>Unchanged Fits</h3><p>The following spectra could not be improved with alternative models:</p><ul>'
            for r in not_improved:
                html += f'<li>{r["name"]} (R² remained {r.get("original_r2", "N/A")})</li>'
            html += '</ul>'

        # Include visualizations for improved spectra
        for r in improved:
            result = next((sr for sr in series_results if sr.get("index") == r["index"]), None)
            if result and result.get("visualization_path") and Path(result["visualization_path"]).exists():
                with open(result["visualization_path"], 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                html += f'<div class="image-card" style="border-left: 4px solid #17a2b8;"><img src="data:image/png;base64,{b64}" alt="{r["name"]}"><div class="image-label">{r["name"]} (Re-fitted, R²: {r.get("new_r2", 0):.4f})</div></div>'

        return html

    def _generate_individual_fits_section(self, series_results: List[dict], num_spectra: int) -> str:
        results_with_viz = [(i, r) for i, r in enumerate(series_results) 
                           if r.get("visualization_path") and Path(r["visualization_path"]).exists()]
        
        if not results_with_viz:
            return ""
        
        failed_indices = {i for i, r in enumerate(series_results) if not r["success"]}
        flagged_indices = {i for i, r in enumerate(series_results) if r.get("flagged")}
        priority_indices = failed_indices | flagged_indices
        
        if num_spectra <= 10:
            indices_to_show = set(range(num_spectra))
            section_note = ""
        elif num_spectra <= 30:
            indices_to_show = set(range(min(3, num_spectra))) | set(range(max(0, num_spectra - 3), num_spectra))
            if num_spectra > 6:
                step = (num_spectra - 6) // 5
                for i in range(3, num_spectra - 3, max(1, step)):
                    if len(indices_to_show) < 10:
                        indices_to_show.add(i)
            indices_to_show.update(priority_indices)
            not_shown = num_spectra - len(indices_to_show)
            section_note = f"<p><em>Showing {len(indices_to_show)} of {num_spectra} fits. {not_shown} fits not displayed.</em></p>"
        else:
            indices_to_show = {0, 1, num_spectra - 2, num_spectra - 1}
            indices_to_show.update(list(priority_indices)[:10])
            section_note = f"<p><em>Large series ({num_spectra} spectra): Showing boundary fits and flagged/failed spectra.</em></p>"
        
        indices_to_show = sorted(indices_to_show)
        
        html = f"\n        <h2>Individual Fit Results</h2>\n{section_note}"
        html += '        <div class="image-grid" style="grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));">\n'
        
        for idx in indices_to_show:
            if idx >= len(series_results):
                continue
            r = series_results[idx]
            viz_path = r.get("visualization_path")
            
            if viz_path and Path(viz_path).exists():
                with open(viz_path, 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                
                if not r["success"]:
                    status, status_color = "✗ FAILED", "#e74c3c"
                elif r.get("adaptively_refitted"):
                    status, status_color = "🔄 Re-fitted", "#17a2b8"
                elif r.get("flagged"):
                    status, status_color = f"⚠ {r.get('flag_reason', 'Flagged')}", "#fd7e14"
                else:
                    status, status_color = "✓", "#27ae60"

                r_squared = r.get("fit_quality", {}).get("r_squared") or 0
                r2_str = f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else ""
                refit_note = ""
                if r.get("adaptively_refitted") and r.get("original_r2") is not None:
                    refit_note = f"<br><small>Original R²: {r['original_r2']:.4f}</small>"

                html += f'''
            <div class="image-card" style="border-left: 4px solid {status_color};">
                <img src="data:image/png;base64,{b64}" alt="{r['name']}">
                <div style="margin-top: 8px;">
                    <strong>{r['name']}</strong><br>
                    <span style="color: {status_color};">{status}</span> {r2_str}{refit_note}
                </div>
            </div>
'''
        
        html += "        </div>\n"
        return html

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            self.logger.info("")
            self.logger.info("📄 Single spectrum report handled by standard controller")
            return state
        
        self._generate_series_report(state)
        return state

    def _generate_series_report(self, state: dict) -> None:
        self.logger.info("")
        self.logger.info("📄 GENERATING SERIES REPORT")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        synthesis = state.get("synthesis_result", {})
        series_metadata = state.get("series_metadata", {})
        locked_config = state.get("locked_fitting_config", {})
        flagged_spectra = state.get("flagged_spectra", [])
        refit_summary = state.get("refit_summary", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_spectra = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_spectra)
        refitted_count = sum(1 for r in refit_summary if r.get("improved"))
        
        # Quality status indicator
        if flagged_count == 0:
            quality_indicator = '<span class="quality-indicator quality-good">✓ All fits acceptable</span>'
        elif flagged_count <= num_spectra * 0.1:
            quality_indicator = f'<span class="quality-indicator quality-warning">⚠ {flagged_count} spectra flagged</span>'
        else:
            quality_indicator = f'<span class="quality-indicator quality-critical">⚠ {flagged_count} spectra flagged ({100*flagged_count/num_spectra:.0f}%)</span>'
        
        # Build trend visualizations HTML
        trend_viz_html = ""
        if trend_results.get("success") and trend_results.get("generated_files"):
            trend_viz_html = '<h2>3. Trend Visualizations</h2><div class="image-grid">'
            for file_path in trend_results["generated_files"]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        b64 = self._image_to_base64(f.read())
                    name = Path(file_path).stem.replace('_', ' ').title()
                    trend_viz_html += f'<div class="image-card"><img src="data:image/png;base64,{b64}" alt="{name}"><div class="image-label">{name}</div></div>'
            trend_viz_html += '</div>'
        
        # Parameter trends HTML
        param_trends_html = ""
        param_trends = synthesis.get('parameter_trends', {})
        if param_trends:
            param_trends_html = "<h2>2. Parameter Trends</h2>"
            for param_name, trend_info in param_trends.items():
                if isinstance(trend_info, dict):
                    param_trends_html += f'<div class="trend-card"><strong>{param_name}</strong><br>Trend: {trend_info.get("trend", "N/A")}<br><em>{trend_info.get("interpretation", "")}</em></div>'
        
        # Scientific claims HTML
        claims_html = ""
        scientific_claims = synthesis.get('scientific_claims', [])
        if scientific_claims:
            claims_html = "<h2>5. Scientific Claims</h2>"
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f'''<div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Literature Search Query:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>'''
        
        # Caveats HTML
        caveats_html = ""
        caveats = synthesis.get('caveats', '')
        if caveats:
            caveats_html = f'<h2>6. Caveats & Limitations</h2><div class="caveats">{caveats}</div>'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Series Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .trend-card {{ background-color: #fef9e7; border-left: 5px solid #f39c12; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        .quality-indicator {{ display: inline-block; padding: 5px 12px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-warning {{ background-color: #fff3cd; color: #856404; }}
        .quality-critical {{ background-color: #f8d7da; color: #721c24; }}
        .flagged-summary {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0; }}
        .flagged-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 15px; }}
        .flagged-card {{ background: white; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; }}
        .flagged-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .flagged-badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.85em; color: white; }}
        .flagged-card img {{ max-width: 100%; margin-top: 10px; border-radius: 4px; }}
        .flagged-recommendation {{ margin: 10px 0; font-size: 0.9em; color: #666; }}
        .refit-summary {{ background-color: #d1ecf1; border-left: 5px solid #17a2b8; padding: 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .params-table th, .params-table td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
        .params-table th {{ background-color: #e9ecef; font-weight: bold; }}
        .params-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Spectral Series Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Spectra Processed:</strong> {successful}/{num_spectra}</p>
            <p><strong>Series Variable:</strong> {series_metadata.get('variable', 'N/A')}</p>
            <p><strong>Fitting Model:</strong> {locked_config.get('physical_model', 'N/A')}{f' ({refitted_count} spectra re-fitted with alternative models)' if refitted_count > 0 else ''}</p>
            <p><strong>Quality Status:</strong> {quality_indicator}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{synthesis.get('detailed_analysis', 'No analysis available.')}</div>
        {param_trends_html}
        {trend_viz_html}
        {self._generate_individual_fits_section(series_results, num_spectra)}
        {self._generate_refit_section(refit_summary, series_results) if refit_summary else ''}
        {self._generate_flagged_spectra_section(flagged_spectra, series_results, synthesis) if flagged_spectra else ''}
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Series Analysis Agent</div>
    </div>
</body>
</html>"""

        report_path = self.output_dir / "series_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        state["report_path"] = str(report_path)
        self.logger.info(f"   ✅ Report saved: {report_path}")