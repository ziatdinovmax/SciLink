"""Technique-agnostic reconciliation of two complementary series analyses.

The recurring shape across in-situ / operando spectroscopy: the same series
of frames is analysed two ways that answer different questions and depend on
different things —

* a **profile-fitting** pass (peak / line / band fitting) that gives per-frame
  FEATURES — position, width, area — by fitting a lineshape model. It uses a
  model (pseudo-Voigt, split-PV, a size–strain-convolved profile, …), but it
  is **database-independent**: it needs NO reference and assigns NO identity.
  It describes HOW the material evolves.
* an **identification** pass that gives per-frame LABELS (phase, species,
  element, …) by matching against a REFERENCE database — WHAT the material is,
  but only where a reference exists.

The distinction is database-independence vs reference-matching (NOT "model-free
vs model-based" — both use models). Neither alone is the whole picture. This
module joins them: it tracks the fitted features across frames, attributes
them to the identified labels, and — for a one-step transformation — locates
the transition each pass finds independently. Where identification produced no
label, the regime stays honestly ``None`` (unidentified) rather than named.

Two layers, so the tool degrades honestly:

1. **Feature tracking + labelling (general).** Cluster fitted peaks into
   tracked features across frames and attach the identified label. This is
   the always-valid output — it makes no assumption about the series shape.
2. **Transition detection (a specific model).** ``transition_model`` selects
   how the transition is located. Only ``'single_crossover'`` is implemented
   (v1): it splits features into a low- and high-endpoint family and finds
   the weight-share crossover — correct for a ONE-STEP transformation ramp
   (dehydration, calcination, a polymorphic transition). A titration, a
   reversible transition, or a multi-step A→B→C series is NOT one-step: the
   transition may be meaningless there, but the layer-1 tracked features are
   still the real deliverable. ``transition_model`` is the named seam for a
   multi-regime model.

Package-neutral CORE: per-technique skills wrap it with their own vocabulary
and feature/label extraction (XRD's ``reconcile_series_phases`` maps 2θ peaks
→ features, phases → labels). Vocabulary here is generic — ``features``
(position, weight), ``labels`` (value, label) — so NMR chemical shifts,
Raman/IR band positions, XPS binding energies all reuse it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _frame_features(fr: dict) -> list[tuple]:
    """(position, weight) list for one frame. Accepts 'features' with
    position/weight, or the leniency aliases center/area/amplitude so a
    caller can pass profile-fit peaks with minimal reshaping."""
    out = []
    for p in (fr.get("features") or fr.get("peaks") or []):
        pos = p.get("position", p.get("center"))
        w = p.get("weight", p.get("area", p.get("amplitude", 0.0)))
        try:
            out.append((float(pos), float(w)))
        except (TypeError, ValueError):
            continue
    return out


def _auto_tol(feature_frames) -> float:
    """A data-driven tracking tolerance so the default is NOT tied to any
    technique's x-units (2θ° for XRD, ppm for NMR, eV for XPS differ by orders
    of magnitude). Uses ~1/4 of the median nearest-neighbour peak spacing
    within a frame — tight enough to keep distinct peaks apart, loose enough to
    follow one peak that drifts frame-to-frame."""
    gaps = []
    for f in feature_frames:
        pos = sorted(p for p, _ in _frame_features(f))
        gaps += [b - a for a, b in zip(pos, pos[1:]) if b > a]
    if not gaps:
        return 0.25
    return float(0.25 * np.median(gaps))


def reconcile_series(
    feature_frames: Sequence[dict],
    label_frames: Sequence[dict],
    tol: Optional[float] = None,
    min_presence_frac: float = 0.2,
    agreement_units: float = 15.0,
    regime_window_frac: float = 0.33,
    crossover_threshold: float = 0.5,
    transition_model: str = "single_crossover",
) -> dict[str, Any]:
    """Join a profile-fitting feature series with an identification label series.

    ``feature_frames``: per-frame, in series order,
        ``[{'value': <series var>, 'features': [{'position': x, 'weight': w}, ...]}]``
        (``'peaks'`` / ``center`` / ``area`` accepted as aliases).
    ``label_frames``: aligned per-frame labels,
        ``[{'value': <series var>, 'label': <str or None>}]``.

    Knobs (robust defaults; turn them when the data needs it):

    * ``tol`` — feature-tracking tolerance IN THE DATA'S X-UNITS (2θ° for XRD,
      ppm for NMR, eV for XPS). ``None`` (default) auto-scales it from the peak
      spacing, so no technique-specific default is baked in. Set it explicitly
      to override: RAISE for features that drift a lot across the series
      (thermal expansion, shifting environments); LOWER for sharp,
      well-separated features that must not be merged.
    * ``min_presence_frac`` — keep a tracked feature only if it appears in at
      least this fraction of frames (default 0.2), filtering transient noise
      peaks. RAISE to keep only persistent features; LOWER to retain a
      short-lived one (a transient intermediate's peaks).
    * ``agreement_units`` — in the SERIES-VARIABLE units (°C, time, pH): the
      two transition estimates within this are called ``consistent`` (default
      15). RAISE for a slow ramp / coarse frame spacing; LOWER for a
      fine-grained scan where you expect the two to pin the same frame.
    * ``regime_window_frac`` — fraction of frames at EACH end used to classify
      which features belong to the low- vs high-endpoint family (default 0.33 =
      first/last third). LOWER (e.g. 0.15) when the endpoints are pure (a clean
      start/end phase); RAISE toward 0.5 for a gradual transformation whose
      extremes are barely separated.
    * ``crossover_threshold`` — the high-family weight-share level that defines
      the transition (default 0.5 = equal shares / ~50 % conversion). Change
      only to define the transition at a different conversion fraction.
    * ``transition_model`` — how the transition is located. Only
      ``'single_crossover'`` (v1, one-step transformation) is implemented; it
      is the named seam for a future multi-regime model. The tracked-feature
      output is returned regardless of this setting.

    Returns tracked features (each attributed to a regime + label), the two
    transition estimates (feature-weight-share crossover vs label switch), and
    an agreement verdict. See module docstring for the two-layer / model note.
    """
    if transition_model != "single_crossover":
        raise ValueError(
            f"transition_model={transition_model!r} not implemented; only "
            "'single_crossover' is available (v1, one-step transformation). "
            "It is the named seam for a multi-regime model.")
    n = min(len(feature_frames), len(label_frames))
    if n < 3:
        raise ValueError("reconcile_series needs at least 3 aligned frames.")
    feature_frames = list(feature_frames)[:n]
    label_frames = list(label_frames)[:n]
    V = np.array([float(f.get("value", i)) for i, f in enumerate(feature_frames)])
    if tol is None:
        tol = _auto_tol(feature_frames)
    tol = float(tol)

    # --- layer 1: cluster feature positions into tracked features ---
    allp = sorted(p for f in feature_frames for p, _ in _frame_features(f))
    refs: list[float] = []
    for p in allp:
        if not refs or abs(p - refs[-1]) > tol:
            refs.append(p)
        else:
            refs[-1] = 0.5 * (refs[-1] + p)

    def _seen(rp):
        return float(np.mean([any(abs(p - rp) <= tol for p, _ in _frame_features(f))
                              for f in feature_frames]))
    refs = [rp for rp in refs if _seen(rp) >= float(min_presence_frac)]
    if not refs:
        raise ValueError("no feature persisted across the series — raise tol or "
                         "lower min_presence_frac.")

    weight = np.zeros((n, len(refs)))
    for i, f in enumerate(feature_frames):
        ff = _frame_features(f)
        for j, rp in enumerate(refs):
            near = [(abs(p - rp), w) for p, w in ff if abs(p - rp) <= tol]
            if near:
                weight[i, j] = min(near)[1]

    # --- layer 2 (single_crossover): endpoint families + share crossover ---
    win = max(1, int(round(float(regime_window_frac) * n)))
    early, late = weight[:win].mean(axis=0), weight[-win:].mean(axis=0)
    low = np.where(early > late)[0]
    high = np.where(late >= early)[0]
    frac = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-9)
    low_share = frac[:, low].sum(axis=1) if len(low) else np.zeros(n)
    high_share = frac[:, high].sum(axis=1) if len(high) else np.zeros(n)

    thr = float(crossover_threshold)
    t_profile = None
    for i in range(1, n):
        if high_share[i - 1] < thr <= high_share[i]:
            g = (thr - high_share[i - 1]) / (high_share[i] - high_share[i - 1] + 1e-9)
            t_profile = float(V[i - 1] + g * (V[i] - V[i - 1]))
            break

    def _lab(i):
        lb = label_frames[i].get("label")
        return lb if lb else None
    lo_idx = [i for i in range(n) if low_share[i] >= high_share[i]]
    hi_idx = [i for i in range(n) if high_share[i] > low_share[i]]

    def _dom(idxs):
        names = [_lab(i) for i in idxs if _lab(i)]
        return max(set(names), key=names.count) if names else None
    lo_label, hi_label = _dom(lo_idx), _dom(hi_idx)

    t_id = None
    if lo_label and hi_label and lo_label != hi_label:
        last_lo = max([i for i in range(n) if _lab(i) == lo_label], default=None)
        first_hi = min([i for i in range(n) if _lab(i) == hi_label], default=None)
        if last_lo is not None and first_hi is not None and first_hi >= last_lo:
            t_id = float(0.5 * (V[last_lo] + V[first_hi]))

    if t_profile is not None and t_id is not None:
        gap = abs(t_profile - t_id)
        agree = {"units_apart": round(gap, 2),
                 "verdict": "consistent" if gap <= float(agreement_units) else "divergent"}
    else:
        agree = {"units_apart": None, "verdict": "one_sided"}

    tracked = []
    for j, rp in enumerate(refs):
        regime = "low" if j in low else "high"
        tracked.append({
            "position": round(float(rp), 3),
            "regime": regime,
            "label": lo_label if regime == "low" else hi_label,
            "weight_series": [round(float(v), 4) for v in weight[:, j]],
        })

    return {
        "values": [float(v) for v in V],
        "tolerance_used": tol,
        "low_regime_label": lo_label,
        "high_regime_label": hi_label,
        "tracked_features": tracked,
        "transition_profile": t_profile,
        "transition_identification": t_id,
        "agreement": agree,
        "transition_model": transition_model,
        # arrays kept so a technique wrapper can plot without re-deriving
        "_low_idx": low.tolist(), "_high_idx": high.tolist(),
        "_weight": weight.tolist(), "_low_share": low_share.tolist(),
        "_high_share": high_share.tolist(), "_refs": [float(r) for r in refs],
    }


# --- generic extraction from stored analysis results (orchestrator seam) -------
#
# The orchestrator reconciles two PRIOR analyses (a profile-fitting pass and an
# identification pass) it already ran and stored. Both write an
# ``analysis_results.json`` (per-frame ``individual_results``) and a
# ``series_fit_results.json`` (series values) in their output dir. Extraction
# is technique-agnostic: features are any per-frame fitted peaks (center/area),
# labels are the per-frame identity under one of the common label keys. So a
# single orchestrator step serves XRD today and NMR/Raman/XPS as their
# identification skills mature — no per-technique orchestration.

_LABEL_KEYS = ("identified_phase", "identified_species", "identified_compound",
               "identity", "phase", "species", "compound", "label")


def _read_json(path):
    import json
    from pathlib import Path
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def _series_values(output_dir, n):
    from pathlib import Path
    sfr = _read_json(Path(output_dir) / "series_fit_results.json") or {}
    vals = ((sfr.get("series_metadata") or {}).get("values")
            or sfr.get("values"))
    if vals and len(vals) >= n:
        return [float(v) for v in vals[:n]]
    return [float(i) for i in range(n)]


def _extract_features(analysis_dir) -> list[dict]:
    """Per-frame (position, weight) peaks from a profile-fitting pass's stored
    result — any ``parameters`` entry that is a dict with a center/position."""
    from pathlib import Path
    d = _read_json(Path(analysis_dir) / "analysis_results.json") or {}
    res = d.get("individual_results") or []
    vals = _series_values(analysis_dir, len(res))
    frames = []
    for i, r in enumerate(res):
        feats = []
        for _, v in (r.get("parameters") or {}).items():
            if isinstance(v, dict) and ("center" in v or "position" in v):
                pos = v.get("position", v.get("center"))
                w = v.get("weight", v.get("area", v.get("amplitude", 0.0)))
                feats.append({"position": pos, "weight": w})
        frames.append({"value": vals[i], "features": feats})
    return frames


def _extract_labels(analysis_dir) -> list[dict]:
    """Per-frame identity label from an identification pass's stored result."""
    from pathlib import Path
    d = _read_json(Path(analysis_dir) / "analysis_results.json") or {}
    res = d.get("individual_results") or []
    vals = _series_values(analysis_dir, len(res))
    frames = []
    for i, r in enumerate(res):
        params = r.get("parameters") or {}
        label = next((params[k] for k in _LABEL_KEYS if params.get(k)), None)
        frames.append({"value": vals[i], "label": label})
    return frames


def reconcile_analysis_dirs(profile_dir: str, identification_dir: str,
                            output_figure: Optional[str] = None,
                            output_report: Optional[str] = None,
                            series_variable: str = "series variable",
                            tol: Optional[float] = None,
                            min_presence_frac: float = 0.2,
                            agreement_units: float = 15.0,
                            regime_window_frac: float = 0.33,
                            crossover_threshold: float = 0.5,
                            interpretation: Optional[str] = None) -> dict[str, Any]:
    """Reconcile two PRIOR analyses given their output directories: extract
    features from the profile-fitting pass and labels from the identification
    pass, call :func:`reconcile_series`, plot a generic figure, and (if
    ``output_report``) render a self-contained HTML report. Technique-agnostic
    — the orchestrator's coupled-series step calls this. Knobs pass straight
    through to :func:`reconcile_series` (see there). ``interpretation`` is
    forwarded to the report renderer (the orchestrator's synthesis prose); on
    the first pass it is usually ``None`` (the numbers are not known until this
    call returns), and the orchestrator re-renders with it via a finalize
    step."""
    ff = _extract_features(profile_dir)
    lf = _extract_labels(identification_dir)
    if not any(f.get("features") for f in ff):
        raise ValueError(
            f"no per-frame fitted peaks found in {profile_dir} — the "
            "profile-fitting pass must be a peak/line-fitting run (its "
            "'parameters' carry peak center/area per frame). Check the "
            "profile / identification arguments are not swapped.")
    r = reconcile_series(ff, lf, tol=tol, min_presence_frac=min_presence_frac,
                         agreement_units=agreement_units,
                         regime_window_frac=regime_window_frac,
                         crossover_threshold=crossover_threshold)
    if output_figure:
        try:
            _plot_generic(r, output_figure, series_variable=series_variable)
            r["figure"] = output_figure
        except Exception:
            r["figure"] = None
    if output_report:
        try:
            render_reconcile_report(r, output_report, series_variable=series_variable,
                                    interpretation=interpretation)
            r["report"] = output_report
        except Exception:
            r["report"] = None
    return {k: v for k, v in r.items() if not k.startswith("_") or k in ("figure", "report")}


def render_reconcile_report(result: dict, out_html: str,
                            series_variable: str = "series variable",
                            title: str = "Profile fitting + Identification, reconciled",
                            interpretation: Optional[str] = None) -> str:
    """Render a self-contained HTML report for a reconcile result: the
    transition summary (both estimates + agreement), the phase/species labels
    (with honest 'unidentified' callouts), an optional scientific
    interpretation, the embedded figure, a tracked-feature table, and
    methodological guidance. Technique-agnostic — reads only the generic
    reconcile-result keys. Returns the path written.

    ``interpretation`` (optional): a free-prose scientific synthesis of the
    reconciliation, authored by the caller (the analysis orchestrator LLM,
    which holds both prior reports + session context). Rendered as the primary
    Interpretation section so the reconcile report carries a narrative like the
    component profile-fitting and identification reports do, rather than only
    computed numbers. When absent, the report still renders — the deterministic
    methodological note at the foot is always present as the fallback guidance.
    Split on blank lines into paragraphs. The transition numbers and agreement
    verdict stay computed (this prose narrates them; it does not replace them)."""
    import base64
    import html as _html
    from pathlib import Path

    lo, hi = result.get("low_regime_label"), result.get("high_regime_label")
    t_p, t_i = result.get("transition_profile"), result.get("transition_identification")
    agree = result.get("agreement") or {}
    verdict = agree.get("verdict")
    tracked = result.get("tracked_features") or []
    fig = result.get("figure")

    def esc(x):
        return _html.escape(str(x)) if x is not None else ""

    # agreement badge
    badge = {"consistent": ("#155724", "#d4edda", "✓ consistent"),
             "divergent": ("#721c24", "#f8d7da", "⚠ divergent — investigate"),
             "one_sided": ("#856404", "#fff3cd", "one-sided (only one method timed it)")
             }.get(verdict, ("#383d41", "#e2e3e5", esc(verdict)))
    ap = agree.get("units_apart")
    agree_line = (f"{badge[2]}" + (f" ({ap} {esc(series_variable)} apart)" if ap is not None else ""))

    def label_html(lab, which):
        if lab:
            return f"<strong>{esc(lab)}</strong>"
        return (f'<strong style="color:#856404;">UNIDENTIFIED</strong> '
                f'<span style="color:#666;">— the {which}-regime phase is not in the '
                f'searched database (organic / novel product). Its trends are real but '
                f'unnamed; use an empirical reference or the indexing route.</span>')

    rows = "".join(
        f"<tr><td>{t['position']}</td>"
        f"<td>{esc(t['regime'])}</td>"
        f"<td>{esc(t['label']) if t.get('label') else '<em>unidentified</em>'}</td></tr>"
        for t in tracked)

    img = ""
    if fig and Path(fig).is_file():
        b64 = base64.b64encode(Path(fig).read_bytes()).decode()
        img = (f'<div class="fig"><img src="data:image/png;base64,{b64}" '
               f'alt="reconciled figure"></div>')

    t_p_s = f"{t_p:.3g} {esc(series_variable)}" if t_p is not None else "not detected"
    t_i_s = f"{t_i:.3g} {esc(series_variable)}" if t_i is not None else "not detected"

    interp_html = ""
    if interpretation and str(interpretation).strip():
        paras = "".join(f"<p>{esc(p.strip())}</p>"
                        for p in str(interpretation).split("\n\n") if p.strip())
        interp_html = (
            '<h2>Interpretation</h2>'
            f'<div class="box interp">{paras}'
            '<p class="attr">— scientific synthesis by the analysis orchestrator, '
            'grounded in the reconciled result above (the transition values and '
            'agreement verdict are computed; this narrates them).</p></div>')

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>{esc(title)}</title><style>
body{{font-family:'Segoe UI',Tahoma,sans-serif;line-height:1.6;color:#333;max-width:1200px;margin:0 auto;padding:20px;background:#f4f4f9;}}
.container{{background:#fff;padding:36px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);}}
h1{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;}}
h2{{color:#2980b9;margin-top:26px;}}
.box{{background:#ecf0f1;padding:14px 18px;border-radius:5px;border-left:5px solid #3498db;margin:14px 0;}}
.badge{{display:inline-block;padding:5px 12px;border-radius:14px;font-weight:bold;color:{badge[0]};background:{badge[1]};}}
.interp{{background:#eef7ff;border-left:5px solid #2980b9;}}
.interp p{{margin:8px 0;}}
.attr{{color:#666;font-size:.85em;font-style:italic;margin-top:12px;}}
.fig img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-top:12px;}}
table{{width:100%;border-collapse:collapse;margin-top:10px;}}
th,td{{border:1px solid #dee2e6;padding:7px 11px;text-align:left;}}
th{{background:#e9ecef;}}
tr:nth-child(even){{background:#f8f9fa;}}
.note{{background:#fff8e6;border-left:5px solid #f0ad4e;padding:12px 16px;border-radius:0 5px 5px 0;margin-top:16px;font-size:.95em;}}
</style></head><body><div class="container">
<h1>📊 {esc(title)}</h1>
<p>Two complementary passes over the same series, joined: <em>profile fitting</em>
(how the structure evolves — database-independent) and <em>identification</em>
(which phases — database-dependent).</p>

<h2>Transition</h2>
<div class="box">
<p><strong>Profile-fit transition:</strong> {t_p_s} &nbsp;·&nbsp;
   <strong>Identification transition:</strong> {t_i_s}</p>
<p><span class="badge">{agree_line}</span></p>
</div>

<h2>Phases</h2>
<div class="box">
<p><strong>Start-phase (low regime):</strong> {label_html(lo, 'start')}</p>
<p><strong>End-phase (high regime):</strong> {label_html(hi, 'end')}</p>
</div>

{interp_html}

{img}

<h2>Tracked features</h2>
<table><thead><tr><th>position</th><th>regime</th><th>phase / species</th></tr></thead>
<tbody>{rows}</tbody></table>

<div class="note"><strong>How to read this.</strong> {esc(result.get('note', ''))}
Agreement of the two transitions is corroboration; a <em>divergent</em> verdict
flags mis-tracked features, a mid-series false ID, or a multi-step process the
single-crossover model does not capture. Transition detection uses the
{esc(result.get('transition_model', 'single_crossover'))} model.</div>
</div></body></html>"""

    Path(out_html).write_text(doc, encoding="utf-8")
    return out_html


def _plot_generic(r, path, series_variable: str = None):
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Label the x-axis with the actual series variable when the caller knows it
    # (temperature / time / pH), else fall back to the generic term.
    xlabel = series_variable or r.get("series_variable") or "series variable"
    V = _np.array(r["values"]); w = _np.array(r["_weight"])
    low, high = r["_low_idx"], r["_high_idx"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    for j in low:
        ax[0].plot(V, w[:, j], "-", color="tab:blue", alpha=0.6, lw=1)
    for j in high:
        ax[0].plot(V, w[:, j], "-", color="tab:red", alpha=0.6, lw=1)
    ax[0].plot([], [], color="tab:blue",
               label=f"low-endpoint family: {r['low_regime_label'] or 'unidentified'}")
    ax[0].plot([], [], color="tab:red",
               label=f"high-endpoint family: {r['high_regime_label'] or 'unidentified'}")
    ax[0].set_ylabel("fitted feature weight (area)")
    ax[0].set_title("Profile fitting + Identification, reconciled\n"
                    "fitted-feature evolution + phase/species labels")
    ax[0].legend(fontsize=9, loc="upper right")
    ax[1].plot(V, r["_low_share"], "o-", color="tab:blue", label="low-family weight share")
    ax[1].plot(V, r["_high_share"], "s-", color="tab:red", label="high-family weight share")
    if r["transition_profile"] is not None:
        ax[1].axvline(r["transition_profile"], color="k", ls="--",
                      label=f"profile-fit transition ≈ {r['transition_profile']:.3g}")
    if r["transition_identification"] is not None:
        ax[1].axvline(r["transition_identification"], color="tab:green", ls=":",
                      label=f"identification transition ≈ {r['transition_identification']:.3g}")
    ax[1].set_xlabel(xlabel); ax[1].set_ylabel("weight share")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
