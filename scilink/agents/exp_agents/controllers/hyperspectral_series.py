"""Hyperspectral series analysis — the datacube counterpart of the image /
curve series machinery.

A series is a list of datacubes measured along a control variable (one cube
per temperature, dose, time step, ...). The shape mirrors the image agent's
"anchor + locked recipe" loop, instantiated with what the hyperspectral agent
already has:

1. **Anchor.** The first dataset gets the full single-cube pipeline
   (decomposition, planning, dynamic analysis with per-map QC). Its APPROVED
   dynamic-analysis scripts become the locked recipe.
2. **Locked replay.** Every later dataset replays those scripts verbatim
   through the existing ``reuse_locked_script`` path (#172): no fresh plan,
   no codegen, decomposition skipped, per-map QC still verifies the outputs.
   That is what makes the per-dataset magnitudes method-comparable.
3. **Flagging.** Failed datasets are flagged ``analysis_failed``; datasets
   whose feature values sit more than ``outlier_sigma`` from the series mean
   are flagged ``statistical_outlier`` — reported, never re-analysed (the
   anomaly may be the physics).
4. **Adaptive refit.** Failed datasets are re-analysed independently with a
   fresh plan (budgeted by ``max_series_refits``, as in the curve agent).
5. **Trend + synthesis.** A trend script is generated over the per-dataset
   feature table, then one series-level interpretation produces the claims.

The per-dataset rows are written to ``series_analysis_results.json`` in the
SAME shape the image agent uses, so the shared feature-table adapter
(``feature_table._image_series_rows``) emits ``features.csv`` unchanged.

Everything here is a free function (or a thin controller) taking plain state,
so it is testable without an agent or an LLM.
"""

from __future__ import annotations

import base64
import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .image_analysis_controllers import ConditionalImageTrendController
from .hyperspectral_controllers import (
    _append_skill_context,
    _append_prior_knowledge_context,
    _append_literature_context,
)

# Per-dataset working directory name inside the series output directory.
UNIT_DIR_FMT = "dataset_{idx:04d}"
SERIES_RESULTS_FILENAME = "series_analysis_results.json"
FLAGGED_FILENAME = "flagged_datasets.json"


# ---------------------------------------------------------------------------
# Feature flattening
# ---------------------------------------------------------------------------

def flatten_feature_records(records: Any) -> Dict[str, float]:
    """Flatten the hyperspectral ``extracted_features`` LIST (per-map records
    ``{name, units, stats:{min,max,mean}}`` / global ``scalar`` records /
    honest-null ``not_measurable`` records) into a flat ``{column: value}``
    dict — the shape the outlier detector, the trend codegen and the feature
    table consume. Column naming matches the single-cube
    ``analysis_results.json`` writer so a series row and a standalone run of
    the same cube produce identical columns."""
    feats: Dict[str, float] = {}
    for m in records or []:
        if not isinstance(m, dict):
            continue
        nm = m.get("not_measurable")
        if isinstance(nm, dict) and nm.get("feature"):
            base = str(nm["feature"]).strip().replace(" ", "_")[:60]
            feats[f"{base}_not_measurable"] = 1
            continue
        stats = m.get("stats")
        sc = m.get("scalar")
        is_scalar = isinstance(sc, (int, float)) and np.isfinite(sc)
        if not isinstance(stats, dict) and not is_scalar:
            continue
        base = str(m.get("name") or "feature").strip().replace(" ", "_")
        units = str(m.get("units") or "").strip()
        suffix = ("_" + units.replace(" ", "").replace("/", "per")
                  if units and units.lower() not in ("a.u.", "au", "")
                  else "")
        if is_scalar:
            feats[f"{base}{suffix}"] = float(sc)
            continue
        for k, v in stats.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                feats[f"{base}_{k}{suffix}"] = float(v)
    return feats


# ---------------------------------------------------------------------------
# Per-dataset rows
# ---------------------------------------------------------------------------

def build_series_row(index: int, data_path: str, result: Dict[str, Any],
                     role: str, output_dir: str) -> Dict[str, Any]:
    """Reduce one dataset's ``analyze()`` result to a series row.

    ``success`` means the run produced at least one committed feature — a
    'partial' run with honest caveats counts, a run that answered nothing
    (error, or total dynamic-analysis failure) does not.
    """
    status = result.get("status")
    features = flatten_feature_records(result.get("extracted_features"))
    records = result.get("dynamic_analysis_records") or []
    n_ok = sum(1 for r in records if isinstance(r, dict) and r.get("task_success"))
    success = status in ("success", "partial") and bool(features)
    error = None
    if not success:
        err = result.get("error")
        if isinstance(err, dict):
            error = err.get("details") or err.get("error") or "analysis failed"
        elif err:
            error = str(err)
        elif result.get("warnings"):
            error = "; ".join(str(w) for w in result["warnings"])[:500]
        else:
            error = "no feature passed verification"
    row: Dict[str, Any] = {
        "index": index,
        "name": Path(data_path).stem,
        "data_path": str(data_path),
        "success": success,
        # A run can commit features from a SALVAGED attempt in which no
        # required output passed QC (only diagnostic maps did). Such a row
        # is usable but not verified: it is flagged, excluded from the outlier
        # statistics, and re-analysed like a failure when the budget allows.
        "verified": success and (n_ok > 0 or not records),
        "status": status,
        "role": role,
        "confidence": result.get("confidence"),
        "error": error,
        "output_directory": str(output_dir),
        "extracted_features": features,
        "feature_records": result.get("extracted_features") or [],
        "quality_metrics": {
            "n_targets": len(records),
            "n_approved": n_ok,
            "approved_fraction": (n_ok / len(records)) if records else None,
        },
        "warnings": list(result.get("warnings") or []),
        "scientific_claims": result.get("scientific_claims") or [],
        "detailed_analysis": result.get("detailed_analysis") or "",
    }
    reuse = result.get("script_reuse")
    if reuse:
        row["reuse_validity"] = {
            "reused": True,
            "verbatim": bool(reuse.get("verbatim")),
            "n_replayed": reuse.get("n_replayed"),
            # "good" only when the replay was byte-exact AND its required
            # outputs verified on this dataset; a salvaged replay is degraded.
            "verdict": ("good" if success and reuse.get("verbatim") and row["verified"]
                        else "degraded" if success else "script_failed"),
            **({"scope_degraded": True} if reuse.get("scope_degraded") else {}),
        }
    return row


# ---------------------------------------------------------------------------
# Outlier detection (pure statistics — no LLM)
# ---------------------------------------------------------------------------

def detect_outliers(series_results: List[dict], outlier_sigma: float = 2.0,
                    control_values: Optional[list] = None,
                    feature_prefixes: Optional[List[str]] = None,
                    groups: Optional[Dict[str, List[int]]] = None) -> List[dict]:
    """Flag failed datasets and statistical outliers among the successes.

    ``groups`` (regime name -> dataset indices) scores each group as its own
    population. The driver passes the planned regimes when they INTERLEAVE
    along the series axis (axis not coherent): there the "trend of the other
    datasets" mixes two states and half the series looks anomalous. With
    contiguous regimes the whole series is scored, so a transition into a
    one-dataset regime is still flagged as the abrupt change it is.

    Every ``success is False`` row is flagged ``analysis_failed``. With >= 3
    successful rows, each numeric feature column is scanned for datasets that
    deviate more than ``outlier_sigma`` from what the OTHER datasets predict;
    honest-null ``*_not_measurable`` indicator columns are excluded (an
    absence is data, not a magnitude).

    ``feature_prefixes`` restricts the scan to the series' PRIMARY features —
    the locked required output names (``Ti_L3_Position`` → every
    ``Ti_L3_Position_*`` column). Diagnostic columns (fit R², width, coverage)
    are then ignored: a refit or regime anchor runs a different method, so
    its diagnostics are not comparable to the locked script's and would flag
    the dataset for the wrong reason. Falls back to every column when no
    prefix matches.

    The image agent's population z-score caps at sqrt(n-1) sigma for a single
    outlier, so a five-dataset series could never flag anything at 2 sigma
    (observed live: a planted +2.5 eV white-line jump went unflagged). A
    series is also expected to TREND, so a plain leave-one-out score flags a
    linear trend's endpoints. Hence, per column and dataset:

    * >= 3 other datasets: deviation from a straight line fitted through the
      others against the control variable (``control_values``, else the
      index), scaled by the OLS prediction standard error (residual scatter
      x leverage of the held-out point x a small-sample inflation) — a plain
      leave-one-out mean would flag the endpoints of a linear trend, and a
      bare residual scatter over-flags an endpoint extrapolated from a few
      tightly scattered points;
    * exactly 3 successes: the population score, as in the image agent.

    The scale is floored at 5 % of the column's range so a noise-free column
    cannot produce infinite scores, and only datasets whose strongest
    deviation is at least half of the series' strongest deviation are
    flagged — the datasets carrying the dominant anomaly — which keeps
    noise-level trips in near-constant diagnostic columns (R², coverage) from
    flagging every dataset once one real outlier exists.
    """
    flagged: List[dict] = []
    for r in series_results:
        if not r.get("success"):
            flagged.append({
                "index": r["index"], "name": r["name"],
                "reason": "analysis_failed",
                "details": r.get("error") or "analysis failed",
                "recommendation": (
                    "The locked series pipeline did not produce a verified "
                    "result on this dataset. It is re-analysed independently "
                    "when the refit budget allows."),
            })
    for r in series_results:
        if r.get("success") and r.get("verified") is False:
            flagged.append({
                "index": r["index"], "name": r["name"],
                "reason": "unverified",
                "details": ("features come from a salvaged attempt: no required "
                            "output passed verification "
                            f"({(r.get('quality_metrics') or {}).get('n_approved', 0)} of "
                            f"{(r.get('quality_metrics') or {}).get('n_targets', 0)} targets approved)"),
                "recommendation": (
                    "Not comparable to the verified datasets. It is re-analysed "
                    "with the series' locked targets when the refit budget allows."),
            })
    if groups:
        by_idx = {r["index"]: r for r in series_results}
        for name, members in groups.items():
            sub = [by_idx[i] for i in members if i in by_idx]
            for f in detect_outliers(sub, outlier_sigma, control_values, feature_prefixes):
                if f["reason"] == "statistical_outlier":   # failures/unverified already listed
                    f["details"] = f"[regime {name}] " + f["details"]
                    flagged.append(f)
        return flagged
    successful = [r for r in series_results
                  if r.get("success") and r.get("verified") is not False]
    if len(successful) < 3:
        return flagged

    def _x(idx: int) -> float:
        if isinstance(control_values, (list, tuple)) and idx < len(control_values):
            try:
                return float(control_values[idx])
            except (TypeError, ValueError):
                pass
        return float(idx)

    columns: Dict[str, Dict[int, float]] = {}
    for r in successful:
        for k, v in (r.get("extracted_features") or {}).items():
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and np.isfinite(v) and not k.endswith("_not_measurable")):
                columns.setdefault(k, {})[r["index"]] = float(v)
    columns = {k: c for k, c in columns.items() if len(c) >= 3}
    if feature_prefixes:
        primary = {k: c for k, c in columns.items()
                   if any(k == p or k.startswith(p + "_") for p in feature_prefixes)}
        # A map's min / max are extreme-value statistics (one pixel decides
        # them); score the mean and global scalars, not the tails.
        central = {k: c for k, c in primary.items()
                   if not any(k.startswith(p + "_min") or k.startswith(p + "_max")
                              for p in feature_prefixes)}
        if central:
            columns = central
        elif primary:
            columns = primary
    if not columns:
        return flagged

    scores: Dict[int, tuple] = {}
    for r in successful:
        idx = r["index"]
        best, details = 0.0, []
        for k, col in columns.items():
            if idx not in col:
                continue
            v = col[idx]
            floor = 0.05 * (max(col.values()) - min(col.values()))
            others = [(i, x) for i, x in col.items() if i != idx]
            if len(others) >= 3:
                xo = np.asarray([_x(i) for i, _ in others]); vo = np.asarray([x for _, x in others])
                m_ = len(others)
                if np.ptp(xo) > 0:
                    b, a = np.polyfit(xo, vo, 1)
                    pred, resid = a + b * _x(idx), vo - (a + b * xo)
                    # Prediction standard error of an OLS line: leverage of
                    # the held-out x (an endpoint is extrapolated, so it is
                    # less certain) and a small-sample inflation for the
                    # residual scatter estimated from m points with 2 dof.
                    sxx = float(((xo - xo.mean()) ** 2).sum())
                    lever = np.sqrt(1.0 + 1.0 / m_ + ((_x(idx) - xo.mean()) ** 2) / sxx)
                    dof = np.sqrt(m_ / max(m_ - 2, 1))
                else:
                    pred, resid = vo.mean(), vo - vo.mean()
                    lever, dof = np.sqrt(1.0 + 1.0 / m_), np.sqrt(m_ / max(m_ - 1, 1))
                scale, label = float(resid.std()) * float(lever * dof), "the other datasets' trend"
            else:
                allv = np.asarray(list(col.values()))
                pred, scale, label = float(allv.mean()), float(allv.std()), "the series mean"
            scale = max(scale, floor)
            if scale <= 1e-12:
                continue
            z = abs(v - pred) / scale
            if z > outlier_sigma:
                details.append(f"{k}={v:.4g} ({z:.1f}sigma from {label} {pred:.4g})")
                best = max(best, z)
        if details:
            scores[idx] = (best, details)
    if not scores:
        return flagged
    top = max(s[0] for s in scores.values())
    for r in successful:
        sc = scores.get(r["index"])
        if sc and sc[0] >= 0.5 * top:
            flagged.append({
                "index": r["index"], "name": r["name"],
                "reason": "statistical_outlier",
                "details": "; ".join(sc[1]),
                "deviation_sigma": float(sc[0]),
                "recommendation": (
                    "Feature values significantly different from what the "
                    "rest of the series predicts. Possible causes: a phase or "
                    "chemical transition, a measurement artifact, or a "
                    "pipeline mismatch. May indicate interesting physics — "
                    "not re-analysed."),
            })
    return flagged


def select_refit_candidates(flagged: List[dict], max_refits: Optional[int]
                            ) -> tuple[List[dict], List[dict]]:
    """Split ``analysis_failed`` / ``unverified`` flags into (refit now,
    skipped by budget), in that priority order.

    ``None`` = unlimited, ``0`` = no refits. Statistical outliers are never
    refit candidates. Candidates are taken in series order, failures first.
    """
    ordered = ([f for f in flagged if f.get("reason") == "analysis_failed"]
               + [f for f in flagged if f.get("reason") == "unverified"])
    cands, seen = [], set()          # a dataset with two reasons is one candidate
    for f in ordered:
        if f.get("index") not in seen:
            seen.add(f.get("index")); cands.append(f)
    if max_refits is None:
        return cands, []
    n = max(0, int(max_refits))
    skipped = [{**c, "skip_reason": f"max_series_refits={n} exhausted"}
               for c in cands[n:]]
    return cands[:n], skipped


# ---------------------------------------------------------------------------
# Series results file (feature-table contract)
# ---------------------------------------------------------------------------

def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bytes):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


_ROW_FILE_DROP = ("detailed_analysis", "feature_records", "scientific_claims")


def write_series_results(output_dir, state: Dict[str, Any]) -> str:
    """Write ``series_analysis_results.json`` — one row per dataset plus the
    series metadata and the locked recipe. Same filename and row keys as the
    image agent's file so ``feature_table._image_series_rows`` reads it
    unchanged (one ``features.csv`` row per successful dataset, conditions
    from sidecars or ``series_metadata``). Re-written after the refit stage so
    the file reflects adopted refits, never a stale pre-refit copy."""
    rows = []
    for r in state.get("series_results", []):
        rows.append({k: v for k, v in r.items() if k not in _ROW_FILE_DROP})
    payload = {
        "timestamp": datetime.now().isoformat(),
        "agent_type": "hyperspectral",
        "total_datasets": len(rows),
        "successful": sum(1 for r in rows if r.get("success")),
        "flagged_count": len(state.get("flagged_images") or []),
        "is_single_image": False,
        "series_metadata": state.get("series_metadata") or {},
        "quality_settings": {"outlier_sigma": state.get("outlier_sigma")},
        "locked_config": state.get("locked_config"),
        "series_analysis_plan": state.get("series_analysis_plan"),
        "results": rows,
    }
    dest = Path(output_dir) / SERIES_RESULTS_FILENAME
    dest.write_text(json.dumps(_serializable(payload), indent=2),
                    encoding="utf-8")
    return str(dest)


def write_flagged_file(output_dir, state: Dict[str, Any]) -> Optional[str]:
    flagged = state.get("flagged_images") or []
    if not flagged:
        return None
    dest = Path(output_dir) / FLAGGED_FILENAME
    dest.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "outlier_sigma": state.get("outlier_sigma"),
        "total_datasets": len(state.get("series_results") or []),
        "flagged_count": len(flagged),
        "flagged_datasets": flagged,
    }, indent=2), encoding="utf-8")
    return str(dest)


# ---------------------------------------------------------------------------
# Trend analysis — reuse the image trend codegen over the same JSON contract
# ---------------------------------------------------------------------------

class HyperspectralSeriesTrendController(ConditionalImageTrendController):
    """The image trend-codegen controller re-worded for datacubes. The data
    contract (``series_analysis_results.json`` → ``results[i].extracted_features``
    aligned to ``series_metadata.values``) is identical, so only the framing
    changes: per-dataset values are statistics of spectral feature MAPS."""

    TREND_ANALYSIS_INSTRUCTIONS = (
        ConditionalImageTrendController.TREND_ANALYSIS_INSTRUCTIONS
        .replace(
            "You are analyzing a series of image analysis results to identify trends.",
            "You are analyzing a series of hyperspectral datacube analysis "
            "results to identify trends. Each dataset's features are per-map "
            "statistics (min/max/mean of a spectral feature map, e.g. a peak "
            "position or an intensity ratio) or global scalars; the same "
            "locked analysis script produced them for every dataset, so the "
            "values are directly comparable across the series.")
        .replace("FLAGGED IMAGES", "FLAGGED DATASETS")
        .replace("flagged images", "flagged datasets")
        .replace("individual image analysis visualizations",
                 "individual datacube analysis visualizations")
        .replace("individual image analyses", "individual datacube analyses")
    )


# ---------------------------------------------------------------------------
# Series synthesis (one LLM call)
# ---------------------------------------------------------------------------

SERIES_SYNTHESIS_INSTRUCTIONS = """You are an expert in hyperspectral / spectroscopic materials characterization. \
A SERIES of {num_datasets} datacubes was measured along a control variable and analysed with ONE locked \
analysis pipeline (the same verified script applied to every dataset), so the per-dataset feature values \
are directly comparable. {successful} dataset(s) succeeded, {flagged_count} were flagged.

**LOCKED PIPELINE (analysis targets):**
{locked_targets}

**SERIES REGIMES (planned from the scouted spectra; one locked script per regime):**
{regimes}

**ANCHOR INTERPRETATION (first dataset, full analysis):**
{anchor_analysis}

**PER-DATASET FEATURES:**
{dataset_summaries}

**FLAGGED DATASETS:**
{flagged_summary}

**ADAPTIVE RE-ANALYSES:**
{refit_summary}

**TREND ANALYSIS RESULTS:**
{trend_results}

**SERIES METADATA:**
{series_metadata}

**SYSTEM INFORMATION:**
{system_info}

Synthesize the series as a whole. Interpret how the spectral features evolve with the control variable \
and what that implies physically (phase or chemical transitions, thresholds, monotonic trends, saturation). \
Treat statistical outliers as possible physics, not noise, unless the evidence says otherwise. Ground every \
statement in the feature values and trend results above; do not invent magnitudes. Note failed or degraded \
datasets as limitations, not evidence.

You MUST output a valid JSON object with these keys:
1. "detailed_analysis": (String) A thorough narrative of the series — overall quality, key trends, \
physical interpretation, treatment of flagged datasets, caveats.
2. "scientific_claims": (List) 1-2 specific claims about the series-level behavior. Each object has: \
"claim" (String), "spectroscopic_evidence" (String), "scientific_impact" (String), "has_anyone_question" \
(String starting with "Has anyone"), "keywords" (List of 4-6 strings). Questions must be portable — no \
"this", "that" or "the observed".
3. "feature_trends": (Object) feature name -> {{"trend": "increasing|decreasing|non-monotonic|stable|transition", \
"interpretation": "..."}} for the 3-6 most informative features.
4. "flagged_analysis": (Object) {{"summary": "...", "possible_causes": [...], "scientific_significance": "..."}} \
or null when nothing was flagged.
5. "caveats": (List of Strings).
Output ONLY the JSON object.
"""


def _trend_block(trend: Optional[dict]) -> str:
    if not trend:
        return "Trend analysis was not run."
    if trend.get("skipped"):
        return f"Trend analysis skipped: {trend.get('reason')}"
    if not trend.get("success"):
        return f"Trend analysis FAILED: {trend.get('error') or (trend.get('stderr') or '')[:300]}"
    out = trend.get("stdout") or ""
    if len(out) > 2500:
        out = out[:1200] + "\n...[truncated]...\n" + out[-1200:]
    return (f"Approach: {trend.get('approach')}\n"
            f"Metrics tracked: {trend.get('metrics_tracked')}\n"
            f"Script output:\n{out}")


def build_series_synthesis_prompt(state: Dict[str, Any]) -> list:
    """Assemble the multi-part synthesis prompt (text + trend PNGs)."""
    results = state.get("series_results") or []
    meta = state.get("series_metadata") or {}
    values = meta.get("values") if isinstance(meta.get("values"), list) else []
    var, unit = meta.get("variable"), meta.get("unit")
    summaries = []
    for r in results:
        s: Dict[str, Any] = {
            "index": r["index"], "name": r["name"], "success": r.get("success"),
            "status": r.get("status"), "role": r.get("role"),
        }
        if r["index"] < len(values):
            s["control_value"] = f"{var}={values[r['index']]} {unit or ''}".strip()
        if r.get("success"):
            s["features"] = r.get("extracted_features")
            s["confidence"] = r.get("confidence")
        else:
            s["error"] = r.get("error")
        if r.get("flagged"):
            s["flagged"] = r.get("flag_reason")
        if r.get("adaptively_refitted"):
            s["adaptively_refitted"] = True
        if r.get("regime"):
            s["regime"] = r["regime"]
        if r.get("locked_schema_gap"):
            s["locked_schema_gap"] = r["locked_schema_gap"]
        summaries.append(s)

    anchor = next((r for r in results if r.get("role") == "anchor"), None)
    anchor_text = (anchor or {}).get("detailed_analysis") or "(no anchor interpretation)"
    if len(anchor_text) > 3000:
        anchor_text = anchor_text[:3000] + " ...[truncated]"
    flagged = state.get("flagged_images") or []
    refits = state.get("refit_summary") or []
    locked = state.get("locked_config") or {}
    targets = locked.get("targets") or []
    prompt: list = [SERIES_SYNTHESIS_INSTRUCTIONS.format(
        num_datasets=len(results),
        successful=sum(1 for r in results if r.get("success")),
        flagged_count=len(flagged),
        locked_targets=json.dumps(targets, indent=1) if targets else "(no locked recipe — every dataset was analysed independently)",
        regimes=(json.dumps([{k: r.get(k) for k in ("name", "dataset_indices", "description")}
                             for r in (state.get("series_analysis_plan") or {}).get("regimes") or []],
                            indent=1)
                 if state.get("series_analysis_plan") else "One regime: the whole series shares one locked script."),
        anchor_analysis=anchor_text,
        dataset_summaries=json.dumps(_serializable(summaries), indent=1),
        flagged_summary=json.dumps(flagged, indent=1) if flagged else "None.",
        refit_summary=json.dumps(refits, indent=1) if refits else "None.",
        trend_results=_trend_block(state.get("trend_analysis_results")),
        series_metadata=json.dumps(_serializable(meta), indent=1),
        system_info=json.dumps(_serializable(state.get("system_info") or {}), indent=1)[:4000],
    )]
    if state.get("analysis_objective"):
        prompt.append(f"\n## Analysis Objective\n{state['analysis_objective']}\n"
                      "Frame the synthesis around answering this objective.")
    if state.get("analysis_hints"):
        prompt.append(f"\n## Analyst Hints\n{state['analysis_hints']}")
    _append_skill_context(prompt, state, "interpretation")
    _append_prior_knowledge_context(prompt, state)
    _append_literature_context(prompt, state)
    # Trend dashboards as images (up to 5), like the image synthesis does.
    n_img = 0
    for f in ((state.get("trend_analysis_results") or {}).get("generated_files") or []):
        if n_img >= 5 or not str(f).lower().endswith(".png"):
            continue
        try:
            data = Path(f).read_bytes()
        except OSError:
            continue
        prompt.append(f"\nTrend figure: {Path(f).name}")
        prompt.append({"mime_type": "image/png", "data": data})
        n_img += 1
    return prompt


def synthesize_series(model, generation_config, safety_settings,
                      parse_fn: Callable, state: Dict[str, Any],
                      logger: logging.Logger) -> Dict[str, Any]:
    """Run the series synthesis LLM call; never raises."""
    prompt = build_series_synthesis_prompt(state)
    try:
        response = model.generate_content(
            contents=prompt, generation_config=generation_config,
            safety_settings=safety_settings)
        result, err = parse_fn(response)
        if err or not isinstance(result, dict):
            logger.warning(f"Series synthesis parse failed: {err}")
            return {"error": err or "unparseable synthesis response"}
        return result
    except Exception as e:  # noqa: BLE001 - synthesis must not kill the series
        logger.exception(f"Series synthesis failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_series_report(output_dir, state: Dict[str, Any]) -> Optional[str]:
    """Compact HTML report: series table, trend figures, synthesis, claims."""
    try:
        results = state.get("series_results") or []
        synth = state.get("synthesis_result") or {}
        meta = state.get("series_metadata") or {}
        values = meta.get("values") if isinstance(meta.get("values"), list) else []
        esc = html.escape
        parts = [
            "<html><head><meta charset='utf-8'><title>Hyperspectral Series Report</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 6px;"
            "font-size:13px;text-align:left}.bad{background:#fde8e8}.flag{background:#fff4d6}"
            "img{max-width:100%}pre{white-space:pre-wrap}</style></head><body>",
            "<h1>Hyperspectral Series Analysis Report</h1>",
            f"<p>{esc(str(datetime.now().isoformat(timespec='seconds')))} — "
            f"{len(results)} datasets, variable <b>{esc(str(meta.get('variable') or '?'))}</b> "
            f"({esc(str(meta.get('unit') or ''))})</p>",
        ]
        locked = state.get("locked_config") or {}
        if locked:
            parts.append("<h2>Locked pipeline</h2><ul>")
            for t in locked.get("targets") or []:
                parts.append(f"<li>{esc(str(t))}</li>")
            parts.append(f"</ul><p>Anchor dataset index: {locked.get('anchor_index')}</p>")
        plan = state.get("series_analysis_plan")
        if plan:
            parts.append("<h2>Series regimes</h2><ul>")
            for rg in plan.get("regimes") or []:
                parts.append(f"<li><b>{esc(str(rg.get('name')))}</b> — datasets "
                             f"{esc(str(rg.get('dataset_indices')))}: "
                             f"{esc(str(rg.get('description') or ''))}</li>")
            parts.append("</ul>")
        parts.append("<h2>Datasets</h2><table><tr><th>#</th><th>Name</th><th>Value</th>"
                     "<th>Regime</th><th>Role</th><th>Status</th><th>Features</th><th>Flag</th></tr>")
        for r in results:
            cls = "" if r.get("success") else " class='bad'"
            if r.get("success") and r.get("flagged"):
                cls = " class='flag'"
            val = values[r["index"]] if r["index"] < len(values) else ""
            feats = r.get("extracted_features") or {}
            ftxt = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in list(feats.items())[:8])
            if len(feats) > 8:
                ftxt += f", … (+{len(feats) - 8})"
            parts.append(
                f"<tr{cls}><td>{r['index']}</td><td>{esc(str(r['name']))}</td>"
                f"<td>{esc(str(val))}</td><td>{esc(str(r.get('regime') or ''))}</td>"
                f"<td>{esc(str(r.get('role')))}"
                f"{' (refit)' if r.get('adaptively_refitted') else ''}</td>"
                f"<td>{esc(str(r.get('status')))}</td><td>{esc(ftxt)}</td>"
                f"<td>{esc(str(r.get('flag_reason') or ''))}</td></tr>")
        parts.append("</table>")
        trend = state.get("trend_analysis_results") or {}
        pngs = [f for f in (trend.get("generated_files") or []) if str(f).lower().endswith(".png")]
        if pngs:
            parts.append("<h2>Trend visualizations</h2>")
            for f in pngs[:5]:
                try:
                    b64 = base64.b64encode(Path(f).read_bytes()).decode("ascii")
                    parts.append(f"<p><b>{esc(Path(f).name)}</b></p>"
                                 f"<img src='data:image/png;base64,{b64}'/>")
                except OSError:
                    continue
        if synth.get("detailed_analysis"):
            parts.append("<h2>Series interpretation</h2>"
                         f"<pre>{esc(str(synth['detailed_analysis']))}</pre>")
        ft = synth.get("feature_trends")
        if isinstance(ft, dict) and ft:
            parts.append("<h2>Feature trends</h2><table><tr><th>Feature</th><th>Trend</th>"
                         "<th>Interpretation</th></tr>")
            for k, v in ft.items():
                v = v if isinstance(v, dict) else {"trend": str(v)}
                parts.append(f"<tr><td>{esc(str(k))}</td><td>{esc(str(v.get('trend', '')))}</td>"
                             f"<td>{esc(str(v.get('interpretation', '')))}</td></tr>")
            parts.append("</table>")
        claims = synth.get("scientific_claims") or []
        if claims:
            parts.append("<h2>Scientific claims</h2><ol>")
            for c in claims:
                if isinstance(c, dict):
                    parts.append(f"<li><b>{esc(str(c.get('claim', '')))}</b><br/>"
                                 f"<i>{esc(str(c.get('has_anyone_question', '')))}</i></li>")
            parts.append("</ol>")
        cav = synth.get("caveats") or []
        if cav:
            parts.append("<h2>Caveats</h2><ul>" + "".join(
                f"<li>{esc(str(c))}</li>" for c in cav) + "</ul>")
        parts.append("</body></html>")
        dest = Path(output_dir) / "hyperspectral_series_report.html"
        dest.write_text("\n".join(parts), encoding="utf-8")
        return str(dest)
    except Exception as e:  # noqa: BLE001 - report is best-effort
        logging.getLogger(__name__).warning(f"Series report skipped: {e}")
        return None


# ---------------------------------------------------------------------------
# Series-level feature table (for programmatic consumers)
# ---------------------------------------------------------------------------

def series_feature_matrix(series_results: List[dict]) -> Dict[str, list]:
    """``{feature: [value per dataset or None]}`` in series order — the
    hyperspectral analogue of the curve agent's parameter trends, for callers
    that want the numbers without re-reading the JSON file."""
    keys: List[str] = []
    for r in series_results:
        for k in (r.get("extracted_features") or {}):
            if k not in keys:
                keys.append(k)
    return {k: [(r.get("extracted_features") or {}).get(k) for r in series_results]
            for k in keys}


# ---------------------------------------------------------------------------
# Scout stage: mean spectra, overlay, full-series change detection
# ---------------------------------------------------------------------------

_SCOUT_ALL_MAX = 16
_REDUCTION_MAX_DATASETS = 256


def select_scout_indices(n: int, scout_all: bool = False) -> List[int]:
    """Evenly spaced representative indices, capped at 7 — or every dataset
    (capped at ``_SCOUT_ALL_MAX``) when the series axis is not coherent. Same
    rule as the curve / image scouts."""
    if scout_all and n <= _SCOUT_ALL_MAX:
        return list(range(n))
    if n <= 3:
        return list(range(n))
    if n <= 6:
        return sorted({0, n // 2, n - 1})
    if n <= 15:
        return sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    step = (n - 1) / 6
    return sorted({round(i * step) for i in range(7)})


def _overlay_png(curves: List[dict], axis_label: str, title: str) -> Optional[bytes]:
    """Overlay of the scouted mean spectra (one line per dataset)."""
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.get_cmap("viridis")
        for i, c in enumerate(curves):
            ax.plot(c["x"], c["y"], lw=1.4, label=c["label"],
                    color=cmap(i / max(1, len(curves) - 1)))
        ax.set_xlabel(axis_label)
        ax.set_ylabel("mean intensity (a.u.)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - overlay is evidence, never a blocker
        return None


def scout_series(data_paths: List[str], load_fn: Callable, axis_fn: Callable,
                 series_metadata: dict, logger: logging.Logger) -> Dict[str, Any]:
    """Scout a datacube series before planning.

    Every cube is loaded once (one at a time) and reduced to its spatially
    averaged spectrum. The curve agent's full-series SVD change detection
    (``series_reduction.reduce_curves``) runs over ALL mean spectra along the
    control variable, locating where the series changes and whether the
    control variable orders the datasets into contiguous regimes; the
    scouted subset is overlaid for the planner's eyes. Never raises.

    Returns ``{"scout_data", "overlay_png", "reduction", "mean_spectra"}``.
    """
    from ....skills._shared.series_reduction import reduce_curves

    n = len(data_paths)
    values = series_metadata.get("values") if isinstance(series_metadata, dict) else None
    variable = (series_metadata or {}).get("variable") or "index"
    unit = (series_metadata or {}).get("unit") or ""
    mean_spectra: Dict[int, tuple] = {}
    stats: Dict[int, dict] = {}
    axis_label = "channel"
    indices = list(range(n))
    if n > _REDUCTION_MAX_DATASETS:
        step = (n - 1) / (_REDUCTION_MAX_DATASETS - 1)
        indices = sorted({round(i * step) for i in range(_REDUCTION_MAX_DATASETS)})
    for idx in indices:
        try:
            cube = load_fn(data_paths[idx])
            flat = np.asarray(cube, dtype=float).reshape(-1, cube.shape[-1])
            mean = np.nanmean(flat, axis=0)
            try:
                x, axis_label, _ = axis_fn(cube.shape[-1])
                x = np.asarray(x, dtype=float)
            except Exception:  # noqa: BLE001 - fall back to channel index
                x = np.arange(cube.shape[-1], dtype=float)
            mean_spectra[idx] = (x, mean)
            stats[idx] = {
                "shape": list(cube.shape),
                "mean_intensity": float(np.nanmean(flat)),
                "peak_position": float(x[int(np.nanargmax(mean))]),
                "peak_intensity": float(np.nanmax(mean)),
                "n_nonfinite": int(np.sum(~np.isfinite(flat))),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  Scout: could not load dataset {idx}: {e}")

    reduction = None
    if len(mean_spectra) >= 4:
        controls = []
        for idx in mean_spectra:
            cv = None
            if isinstance(values, list) and idx < len(values):
                try:
                    cv = float(values[idx])
                except (TypeError, ValueError):
                    cv = None
            controls.append(cv)
        if all(c is not None for c in controls):
            ctrl, source = controls, variable
        else:
            ctrl, source = None, "index"
        try:
            result = reduce_curves([mean_spectra[i] for i in mean_spectra],
                                   controls=ctrl, control_source=source,
                                   label="series", return_figure=True)
            if result.get("status") == "success":
                if len(mean_spectra) < n:
                    result["subsampled"] = f"{len(mean_spectra)} of {n} datasets"
                # scores_by_index is in reduce_curves' input order → map back
                # to dataset indices when the scout skipped some.
                if len(mean_spectra) != n:
                    remap = {k: i for k, i in enumerate(mean_spectra)}
                    result["dataset_index_of_score"] = [remap[k] for k in range(len(mean_spectra))]
                reduction = result
                logger.info(
                    f"  Change detection ({result['n_points']} datasets): "
                    f"change point ≈ {result['change_point']:g} ({source}), "
                    f"sharpness {result['change_sharpness']}")
            else:
                logger.info(f"  Change detection skipped: {result.get('error')}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  Change detection failed: {e}")
    else:
        logger.info("  Change detection needs >= 4 datasets — skipped.")

    coherent = ((reduction or {}).get("axis_coherence") or {}).get("coherent", True)
    scout_idx = set(i for i in select_scout_indices(n, scout_all=not coherent) if i in mean_spectra)
    # A sharp change lands BETWEEN two datasets; the evenly spaced scouts may
    # skip both, so the planner never sees the transition (or a corrupt cube
    # that produced it). Always scout the pair bracketing a sharp change point.
    if reduction and (reduction.get("change_sharpness") or 0) >= 0.5:
        cp = reduction.get("change_point")
        ctrls = reduction.get("controls_by_index") or []
        idx_of = reduction.get("dataset_index_of_score") or list(range(len(ctrls)))
        pairs = sorted(zip(ctrls, idx_of))
        for (c0, i0), (c1, i1) in zip(pairs, pairs[1:]):
            if c0 <= cp <= c1:
                scout_idx.update({i0, i1})
                break
    scout_idx = sorted(scout_idx)
    scout_data, curves = [], []
    for idx in scout_idx:
        if isinstance(values, list) and idx < len(values):
            label = f"{variable}={values[idx]} {unit}".strip()
        else:
            label = f"index {idx}"
        scout_data.append({"index": idx, "label": label, "statistics": stats.get(idx, {})})
        x, y = mean_spectra[idx]
        curves.append({"x": x, "y": y, "label": label})
    overlay = _overlay_png(curves, axis_label, f"Scouted mean spectra ({len(curves)} of {n} datasets)") \
        if len(curves) >= 2 else None
    logger.info(f"  Scouted {len(scout_data)} of {n} datasets"
                + ("" if coherent else " (axis not coherent → all scouted)"))
    return {"scout_data": scout_data, "overlay_png": overlay,
            "reduction": reduction, "n_loaded": len(mean_spectra)}


# ---------------------------------------------------------------------------
# Regime planning (one LLM call, before any dataset is analysed)
# ---------------------------------------------------------------------------

HYPERSPECTRAL_SERIES_REGIME_SUPPLEMENT = """
## Series Regime Planning

You are planning the analysis of a SERIES of {num_datasets} hyperspectral datacubes measured along a \
control variable. One verified per-pixel analysis script will be LOCKED on the first dataset of each \
regime and replayed VERBATIM on the other datasets of that regime, so a regime must be a set of datasets \
the SAME script (same fit windows, same peak model, same thresholds) can analyse. The spatially averaged \
spectra above are your evidence; the computed change detection tells you WHERE the series changes.

**If the spectra appear UNIFORM** (same edges / peaks at the same positions, similar shapes, only gradual \
intensity changes): return ONE regime containing every dataset.

**If the spectra change SIGNIFICANTLY** (peaks shift beyond a fit window, new features appear, an edge \
splits, a component vanishes): return several regimes.

Return ONLY a JSON object:
```json
{{
  "observations": "what changes across the series and where",
  "series_analysis_plan": {{
    "rationale": "why one regime, or why several",
    "regimes": [
      {{
        "name": "descriptive regime name",
        "dataset_indices": [0, 1, 2],
        "description": "spectral character of this regime and what the locked script must handle"
      }}
    ],
    "transition_points": [
      {{"between_indices": [2, 3], "variable_value": null, "description": "what changes"}}
    ]
  }}
}}
```

**Rules:**
- Every dataset index (0 through {num_datasets_minus_1}) must appear in exactly ONE regime.
- Each regime must have at least one dataset. Regimes need not be contiguous if the axis is not coherent.
- Only split when a single script clearly cannot serve every dataset; when in doubt use ONE regime — a \
dataset the locked script fails on is re-analysed adaptively later.
- The same analysis TARGETS (quantities and output names) are extracted in every regime; regimes differ \
in HOW they are extracted, not in what is reported.
"""


def build_regime_plan_prompt(state: Dict[str, Any], scout: Dict[str, Any]) -> list:
    n = state.get("num_images") or len(state.get("data_paths") or [])
    meta = state.get("series_metadata") or {}
    values = meta.get("values") if isinstance(meta.get("values"), list) else []
    unit = meta.get("unit") or ""
    prompt: list = [
        "You are an expert in hyperspectral / spectroscopic materials characterization "
        "planning the analysis of a datacube series.",
        f"\n## Series Overview ({n} datasets)",
        f"Series variable: {meta.get('variable', 'index')} ({unit})"
        + (f" — range {values[0]} to {values[-1]} {unit}" if values else ""),
        f"\n## System Information\n{json.dumps(_serializable(state.get('system_info') or {}), indent=1)[:3000]}",
    ]
    if state.get("analysis_objective"):
        prompt.append(f"\n## Analysis Objective\n{state['analysis_objective']}")
    if state.get("analysis_hints"):
        prompt.append(f"\n## Analyst Hints\n{state['analysis_hints']}")
    if scout.get("overlay_png"):
        prompt.append("\n### Overlay of scouted mean spectra\nAll scouted datasets' spatially "
                      "averaged spectra on one figure. Look for shifts, new features, "
                      "splitting or vanishing components across the series.")
        prompt.append({"mime_type": "image/png", "data": scout["overlay_png"]})
    for s in scout.get("scout_data") or []:
        prompt.append(f"- dataset {s['index']} ({s['label']}): {json.dumps(s.get('statistics', {}))}")
    red = scout.get("reduction")
    if red:
        axis = red.get("control_variable", {}).get("source", "index")
        lines = [
            "\n### Full-Series Change Detection (computed)",
            f"Unsupervised SVD change detection on the mean spectra of "
            f"{red.get('subsampled') or ('all ' + str(red['n_points']) + ' datasets')}.",
            f"- Change point: {axis} ≈ {red['change_point']:g} {unit}".rstrip(),
            f"- Change sharpness: {red['change_sharpness']} (near 1 = abrupt, small = gradual)",
            f"- Variance explained by the first two components: {red['variance_explained']}",
        ]
        flags = [k for k in ("shift_dominated", "intensity_drift", "resampled_to_common_grid")
                 if (red.get("flags") or {}).get(k)]
        if flags:
            lines.append(f"- Flags: {', '.join(flags)}")
        if red.get("caution"):
            lines.append(f"- Caution: {red['caution']}")
        ac = red.get("axis_coherence") or {}
        scores = red.get("scores_by_index") or []
        if ac and not ac.get("coherent", True):
            lines.append(
                f"- AXIS NOT COHERENT: the PC1 score makes {ac.get('n_large_steps')} large jumps "
                f"with {ac.get('n_reversals')} reversals along '{axis}', so this variable does not "
                "order the datasets into contiguous regimes. Assign membership per dataset from "
                "the scores and spectra; a regime's dataset_indices may be non-contiguous.")
        lines.append("- PC1 score per dataset index: "
                     + ", ".join(f"{i}: {v:+.3g}" for i, v in enumerate(scores)))
        prompt.append("\n".join(lines))
        if red.get("score_curve_png"):
            prompt.append({"mime_type": "image/png", "data": red["score_curve_png"]})
    _append_skill_context(prompt, state, "planning")
    _append_prior_knowledge_context(prompt, state)
    prompt.append(HYPERSPECTRAL_SERIES_REGIME_SUPPLEMENT.format(
        num_datasets=n, num_datasets_minus_1=n - 1))
    return prompt


def extract_series_plan(result: Any, n: int, reduction: Optional[dict],
                        logger: logging.Logger) -> Optional[dict]:
    """Validate the planner's ``series_analysis_plan``: dict regimes only,
    indices clamped to range, missing indices assigned by the curve agent's
    rule (nearest neighbour along a coherent axis, nearest PC1 score
    otherwise), empty regimes dropped. ``None`` means one regime."""
    from .curve_fitting_controllers import _assign_missing_indices

    plan = (result or {}).get("series_analysis_plan") if isinstance(result, dict) else None
    if not isinstance(plan, dict):
        return None
    regimes = [r for r in (plan.get("regimes") or []) if isinstance(r, dict)]
    if not regimes:
        return None
    seen: set = set()
    for r in regimes:
        idx = r.get("dataset_indices") or r.get("spectrum_indices") or r.get("image_indices") or []
        clean = []
        for i in idx:
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and i not in seen:
                clean.append(i); seen.add(i)
        r["dataset_indices"] = sorted(clean)
        r["spectrum_indices"] = r["dataset_indices"]   # key the shared helper mutates
    missing = set(range(n)) - seen
    if missing:
        rule = _assign_missing_indices(regimes, missing, reduction)
        logger.warning(f"  Series plan missing indices {sorted(missing)}, assigned by {rule}")
    for r in regimes:
        r["dataset_indices"] = sorted(set(r.get("spectrum_indices") or []))
        r.pop("spectrum_indices", None)
    dropped = [r.get("name", "unnamed") for r in regimes if not r["dataset_indices"]]
    regimes = [r for r in regimes if r["dataset_indices"]]
    if dropped:
        logger.warning(f"  Dropped {len(dropped)} empty regime(s): {dropped}")
    if not regimes:
        return None
    for k, r in enumerate(regimes):
        r.setdefault("name", f"regime_{k + 1}")
    plan["regimes"] = regimes
    plan["transition_points"] = [t for t in (plan.get("transition_points") or []) if isinstance(t, dict)]
    return plan


def render_regime_plan(plan: Optional[dict], series_metadata: dict,
                       scout: Optional[dict] = None, n: int = 0) -> str:
    """Console rendering of a regime plan for the human gate."""
    meta = series_metadata or {}
    values = meta.get("values") if isinstance(meta.get("values"), list) else []
    var, unit = meta.get("variable") or "index", meta.get("unit") or ""

    def _lab(i):
        return f"{i} ({var}={values[i]} {unit})".replace(" )", ")") if i < len(values) else str(i)
    lines = ["", "=" * 60, "📋 PROPOSED SERIES REGIME PLAN", "=" * 60]
    red = (scout or {}).get("reduction") or {}
    if red:
        ac = red.get("axis_coherence") or {}
        lines.append(f"🔎 Change detection: change point ≈ {red.get('change_point'):g} "
                     f"({var}), sharpness {red.get('change_sharpness')}, axis "
                     f"{'coherent' if ac.get('coherent', True) else 'NOT coherent (regimes interleave)'}")
    if not plan:
        lines.append(f"\n1 regime — all {n} datasets share one locked script.")
    else:
        if plan.get("rationale"):
            lines.append(f"\n💡 Rationale: {plan['rationale']}")
        for k, r in enumerate(plan.get("regimes") or [], 1):
            idx = r.get("dataset_indices") or []
            lines.append(f"\n{k}. {r.get('name')}  — datasets {', '.join(_lab(i) for i in idx)}")
            lines.append(f"   anchor: dataset {idx[0] if idx else '?'} (full analysis; its script is "
                         f"locked and replayed on the rest)")
            if r.get("description"):
                lines.append(f"   {r['description']}")
        for tp in plan.get("transition_points") or []:
            lines.append(f"\n↕ transition between {tp.get('between_indices')}: {tp.get('description')}")
    lines.append("")
    return "\n".join(lines)


def plan_series_regimes(model, generation_config, safety_settings, parse_fn: Callable,
                        state: Dict[str, Any], scout: Dict[str, Any],
                        logger: logging.Logger, feedback: Optional[str] = None,
                        previous_plan: Optional[dict] = None) -> Optional[dict]:
    """One planning call over the scout evidence; ``None`` = single regime.
    With ``feedback`` (the human gate), the previous plan and the analyst's
    words are appended and the planner revises. Never raises — a planning
    failure means one regime, not no analysis."""
    n = state.get("num_images") or 0
    if n < 2:
        return None
    try:
        prompt = build_regime_plan_prompt(state, scout)
        if feedback:
            prev = (json.dumps({k: previous_plan.get(k) for k in ("rationale", "regimes", "transition_points")},
                               indent=1, default=str)
                    if previous_plan else "one regime containing every dataset")
            prompt.append(
                "\n## Analyst feedback on the previous plan\n"
                f"Previous plan:\n{prev}\n\nThe analyst says: {feedback}\n"
                "Revise the plan to honour this feedback (it overrides your own reading "
                "of the evidence where they conflict) and return the full JSON again.")
        response = model.generate_content(
            contents=prompt, generation_config=generation_config,
            safety_settings=safety_settings)
        result, err = parse_fn(response)
        if err or not isinstance(result, dict):
            # Observed live: a long answer cut off mid-JSON. One retry asking
            # for a compact object; a second failure means one regime.
            logger.warning(f"  Regime planning unparseable ({(err or {}).get('error') if isinstance(err, dict) else err}); "
                           "retrying once with a compact-JSON instruction.")
            retry_prompt = prompt + [
                "\nYour previous answer was not valid JSON (it may have been cut off). "
                "Reply again with ONLY the JSON object. Keep 'observations' and each "
                "regime 'description' under 300 characters; no prose outside the JSON."]
            response = model.generate_content(
                contents=retry_prompt, generation_config=generation_config,
                safety_settings=safety_settings)
            result, err = parse_fn(response)
            if err or not isinstance(result, dict):
                logger.warning("  Regime planning unparseable twice; using one regime.")
                return None
        if result.get("observations"):
            logger.info(f"  Planner observations: {str(result['observations'])[:400]}")
        plan = extract_series_plan(result, n, scout.get("reduction"), logger)
        if plan:
            for r in plan["regimes"]:
                logger.info(f"  Regime '{r['name']}': datasets {r['dataset_indices']}"
                            f" — {str(r.get('description', ''))[:120]}")
        else:
            logger.info("  Planner: one regime for the whole series.")
        return plan
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  Regime planning failed ({e}); using one regime.")
        return None


# ---------------------------------------------------------------------------
# Locked-schema completion (feature names must align across datasets)
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def complete_locked_schema(row: Dict[str, Any], locked_columns: List[str],
                           logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Make a fresh-code dataset row report the series' locked column names.

    Regime anchors and refits run with the locked TARGETS and required output
    names, so the primary columns already match; what still drifts is a
    units suffix (``..._mean_eV`` vs ``..._mean``) or a prefix on the extra
    diagnostic maps (``Fit_R2_mean`` vs ``R2_mean``). For every locked column
    the row lacks, a unique row column whose normalised name contains, or is
    contained in, the locked name is aliased under the locked name (the
    original is dropped so the feature table does not carry two half-empty
    siblings). Whatever cannot be matched is recorded as
    ``locked_schema_gap`` on the row — reported, not silently NaN.
    """
    feats = dict(row.get("extracted_features") or {})
    gap: List[str] = []
    aliased: Dict[str, str] = {}
    for col in locked_columns:
        if col in feats:
            continue
        target = _norm_name(col)
        cands = []
        for k in feats:
            if k in locked_columns or k in aliased.values():
                continue
            nk = _norm_name(k)
            if nk and target and (nk in target or target in nk) and min(len(nk), len(target)) >= 4:
                cands.append(k)
        if len(cands) == 1:
            feats[col] = feats.pop(cands[0])
            aliased[col] = cands[0]
        else:
            gap.append(col)
    row["extracted_features"] = feats
    row["locked_schema_gap"] = gap
    if aliased:
        row["schema_aliases"] = aliased
    if logger:
        if aliased:
            logger.info(f"  🧩 Schema aliases for dataset {row.get('index')}: "
                        + ", ".join(f"{v}→{k}" for k, v in aliased.items()))
        if gap:
            logger.warning(f"  🧩 Locked-schema gap for dataset {row.get('index')}: "
                           f"{len(gap)} column(s) missing ({', '.join(gap[:6])}"
                           f"{'…' if len(gap) > 6 else ''})")
    return row


# ---------------------------------------------------------------------------
# Parallel replays (non-anchor fan-out) — the curve agent's series_workers
# ---------------------------------------------------------------------------

def resolve_series_workers(value: Optional[int]) -> int:
    """Explicit value > ``SCILINK_HS_SERIES_WORKERS`` env var > 1 (serial)."""
    import os
    if value is None:
        env = os.environ.get("SCILINK_HS_SERIES_WORKERS")
        try:
            value = int(env) if env else 1
        except ValueError:
            value = 1
    return max(int(value), 1)


def replay_worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run ONE locked replay in a child agent. Executed in a spawned process
    (default) or a thread (``SCILINK_HS_SERIES_POOL=thread``, used by the
    offline tests so monkeypatches apply). Everything in ``spec`` is plain
    data; the agent is built here from ``agent_kwargs``. Never raises."""
    import logging
    import os
    idx = spec["index"]
    unit_dir = Path(spec["unit_dir"])
    unit_dir.mkdir(parents=True, exist_ok=True)
    handler = None
    if spec.get("log_to_file"):
        handler = logging.FileHandler(unit_dir / "replay.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    try:
        # The parent process already holds the sandbox approval; a spawned
        # child cannot answer an interactive prompt, so the approval travels.
        if spec.get("sandbox_approved"):
            os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
        from ..hyperspectral_analysis_agent import HyperspectralAnalysisAgent
        agent = HyperspectralAnalysisAgent(**spec["agent_kwargs"])
        agent._series_role = "replay"
        agent._skill_autoselect = False
        agent._light_synthesis = True
        res = agent.analyze(spec["data_path"], **spec["analyze_kwargs"])
    except Exception as e:  # noqa: BLE001 - one replay must not kill the pool
        res = {"status": "error", "error": {"error": type(e).__name__, "details": str(e)}}
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
    return {"index": idx, "result": _serializable(res)}


class ReplayPool:
    """Runs locked replays on ``workers`` processes (or threads under
    ``SCILINK_HS_SERIES_POOL=thread``) as they are queued, so replays overlap
    with the anchors still running in the parent. ``submit`` per spec,
    ``collect`` once at the end (``{index: result}``; a worker failure
    becomes an error result)."""

    def __init__(self, workers: int, logger: logging.Logger):
        import os
        self.workers = max(int(workers), 1)
        self.logger = logger
        self.kind = os.environ.get("SCILINK_HS_SERIES_POOL", "process").lower()
        if self.kind == "thread":
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"))
        self._futures: Dict[int, Any] = {}
        self.logger.info(f"⚡ Replay pool: {self.workers} {self.kind} worker(s)")

    def submit(self, spec: Dict[str, Any]) -> None:
        self._futures[spec["index"]] = self._pool.submit(replay_worker, spec)
        self.logger.info(f"   ⚡ replay dataset {spec['index']} submitted to the pool")

    def collect(self) -> Dict[int, Dict[str, Any]]:
        from concurrent.futures import as_completed
        results: Dict[int, Dict[str, Any]] = {}
        by_future = {f: i for i, f in self._futures.items()}
        try:
            for fut in as_completed(list(by_future)):
                idx = by_future[fut]
                try:
                    results[idx] = fut.result()["result"]
                except Exception as e:  # noqa: BLE001
                    self.logger.exception(f"Replay worker for dataset {idx} failed: {e}")
                    results[idx] = {"status": "error",
                                    "error": {"error": type(e).__name__, "details": str(e)}}
                st = results[idx].get("status")
                self.logger.info(f"   {'✅' if st in ('success', 'partial') else '❌'} replay "
                                 f"dataset {idx} finished: {st}")
        finally:
            self._pool.shutdown(wait=True)
        return results


def run_replays(specs: List[Dict[str, Any]], workers: int,
                logger: logging.Logger) -> Dict[int, Dict[str, Any]]:
    """Convenience: submit all ``specs`` and collect (kept for callers that
    have the full queue up front)."""
    if not specs:
        return {}
    pool = ReplayPool(workers, logger)
    for sp in specs:
        pool.submit(sp)
    return pool.collect()
