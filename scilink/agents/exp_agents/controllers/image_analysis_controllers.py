# controllers/image_analysis_controllers.py

"""
Image Analysis Controllers - Complete Module

This module contains:
1. Controllers for single-image analysis steps
2. Unified controllers that handle both single image (n=1) and series (n>1) analysis

Key principle for series analysis: Single image = Series of 1

Quality control features:
- LLM-based quality verification of analysis results
- Automatic pipeline retry when quality is inadequate
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
import re
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List
import numpy as np

from .._locked_exec import stage_and_run, script_uses_canonical_input, DATA_NAME, VIZ_NAME


# Anthropic's API rejects images over 5 MB. Cap below that with headroom
# for base64 expansion and other margin so generated visualizations don't
# crash the verification call when scripts pile on diagnostic subplots.
_VERIFICATION_IMAGE_CAP_BYTES = int(4.5 * 1024 * 1024)


def _fit_image_under_api_cap(
    image_bytes: bytes, cap: int = _VERIFICATION_IMAGE_CAP_BYTES
) -> tuple[bytes, str]:
    """Return image bytes guaranteed (best-effort) to fit under ``cap``.

    No-op for inputs already at or below the cap — returns them with the
    detected mime type. Otherwise decodes via PIL, re-encodes as JPEG
    with progressively smaller dimensions until under the cap or a
    floor size is reached. Returns the last attempt either way (better
    a slightly-too-large image than crashing the API call).
    """
    if len(image_bytes) <= cap:
        mime = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
        return image_bytes, mime

    try:
        from io import BytesIO
        from PIL import Image as PILImage

        img = PILImage.open(BytesIO(image_bytes))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        target_w, target_h = img.size
        last_bytes = image_bytes
        for _ in range(6):
            buf = BytesIO()
            img.resize((target_w, target_h), PILImage.LANCZOS).save(
                buf, format="JPEG", quality=85, optimize=True
            )
            last_bytes = buf.getvalue()
            if len(last_bytes) <= cap:
                return last_bytes, "image/jpeg"
            target_w = max(200, int(target_w * 0.7))
            target_h = max(200, int(target_h * 0.7))
            if target_w == 200 and target_h == 200:
                break
        return last_bytes, "image/jpeg"
    except Exception:
        # If PIL can't decode or anything else goes wrong, return the
        # original bytes; the caller's API call may still fail but we
        # don't make it worse.
        mime = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
        return image_bytes, mime


def load_image_file(image_path: str) -> np.ndarray:
    """Load image data from file, handling various formats.

    Shared helper used by multiple controllers.  Tries the canonical
    ``load_image_data`` from tools first, then falls back to cv2/PIL.
    """
    try:
        from ...skills._shared.image_analysis_tools import load_image_data
        return load_image_data(image_path)
    except ImportError:
        pass

    if image_path.endswith('.npy'):
        return np.load(image_path)
    elif image_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        try:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            # Convert BGR→RGB; 2-channel images need no conversion
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return img
        except ImportError:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            return np.array(img)
    else:
        try:
            return np.load(image_path)
        except Exception:
            try:
                import cv2
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img
            except ImportError:
                pass
            raise ValueError(f"Could not load image: {image_path}")


def compute_image_statistics(image: np.ndarray) -> dict:
    """Compute statistics for a single image.

    Shared helper used by multiple controllers and the agent itself.
    """
    stats = {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "has_nans": (
            bool(np.any(np.isnan(image)))
            if np.issubdtype(image.dtype, np.floating)
            else False
        ),
    }

    if image.ndim == 2:
        stats["channels"] = 1
        h, w = image.shape
    elif image.ndim == 3:
        h, w = image.shape[:2]
        stats["channels"] = image.shape[2]
    else:
        h, w = image.shape[:2]
        stats["channels"] = image.shape[2] if image.ndim > 2 else 1

    stats["aspect_ratio"] = round(w / h, 3) if h > 0 else 0
    stats["intensity_range"] = [float(np.nanmin(image)), float(np.nanmax(image))]
    stats["intensity_mean"] = float(np.nanmean(image))
    stats["intensity_std"] = float(np.nanstd(image))

    return stats


def build_verification_prompt_with_history(
    current_result: dict,
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
        score = prev.get('quality_score')
        lines.append(f"- Quality score = {score:.2f}" if score is not None else "- Quality score = N/A")
        lines.append(f"- Pipeline: {prev.get('config_used', {}).get('processing_pipeline', 'N/A')}")
        lines.append(f"- Assessment: {prev.get('overall_assessment', 'N/A')}")

        issues = prev.get('issues_found', [])
        if issues:
            lines.append(f"- Issues ({len(issues)}):")
            for issue in issues:
                lines.append(f"  - {issue.get('location', '?')}: {issue.get('problem', '?')}")

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
        "2. If a fix didn't work, suggest something DIFFERENT",
        "3. If a previous fix was NOT applied due to an API error, "
        "re-suggest it or propose an alternative",
    ])

    return "\n".join(lines)


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


def _append_tool_inventory(
    prompt: list,
    agent: str = "image_analysis",
    active_skills: list[str] | None = None,
) -> None:
    """Append the registered tool inventory and library list for ``agent``.

    Tools come from ``scilink.skills._shared._registry.get_tools_for(agent, active_skills)``;
    libraries come from ``IMAGE_ANALYSIS_LIBRARIES``. Injected before skill
    context so skill prose can reference tools introduced here by name.
    """
    from ....skills._shared._registry import (
        format_library_inventory,
        get_tools_for,
    )

    specs = get_tools_for(agent, active_skills=active_skills)
    if specs:
        prompt.append("\n## Available Tools")
        prompt.append(
            "The following tools are registered and callable from generated scripts. "
            "Prefer a tool when it fits; combine with custom numpy/scipy/skimage/cv2 "
            "code for post-processing. A tool call anchoring the hard step followed "
            "by custom code is usually more reliable than an all-custom pipeline."
        )
        for spec in specs:
            prompt.append(spec.to_prompt())

    prompt.append("\n## Available Libraries")
    prompt.append(
        "These libraries are importable in the execution sandbox. Use them for "
        "custom code when no registered tool fits."
    )
    prompt.append(format_library_inventory())


def _active_skill_names(state: dict) -> list[str]:
    """Return names of all currently-loaded skills from a pipeline state dict.

    Falls back to the legacy singular field if ``skills_loaded`` is absent
    so older state dicts continue to work.
    """
    loaded = state.get("skills_loaded")
    if loaded:
        return [s.get("name") for s in loaded if s and s.get("name")]
    legacy = state.get("skill_name")
    return [legacy] if legacy else []


def _append_skill_context(prompt: list, state: dict, stage: str) -> None:
    """Append domain skill knowledge to an LLM prompt for the given stage.

    With multiple skills loaded, sections from each are appended in order
    so the LLM can attribute guidance to its source.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``skills_loaded`` (or the
            legacy ``skill_sections`` / ``skill_name`` for single-skill).
        stage: One of ``"planning"``, ``"analysis"``, ``"interpretation"``, ``"validation"``.
    """
    skills = state.get("skills_loaded") or (
        [state["skill_sections"]]
        if state.get("skill_sections")
        else []
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
        prompt.append(f"\n## Domain Expertise: {skill_name} ({stage})")
        if not intro_appended:
            prompt.append(
                "The following guidance is from validated domain expertise. "
                "Use it to inform your approach."
            )
            intro_appended = True
        prompt.append(content)

        # Include validation rules during planning and interpretation
        if stage in ("planning", "interpretation"):
            validation = sections.get("validation", "")
            if validation:
                prompt.append(f"\n## Domain Validation Guidance: {skill_name}")
                prompt.append(validation)


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


def _load_prior_state(raw_path):
    """Locate and merge prior-analysis JSON for a single path.

    Accepts a directory or a file inside one. Looks for
    ``analysis_results.json`` in the directory or its parent, then merges
    per-image fields from a sibling ``series_analysis_results.json``
    (where ``extracted_features``, ``saved_arrays``, and
    ``quality_metrics`` actually live for series runs). Returns
    ``(anchor_dir, merged_data)`` or ``(None, None)`` on any failure.
    """
    p = Path(raw_path)
    if p.is_file():
        dir_candidates = [p.parent, p.parent.parent]
    else:
        dir_candidates = [p, p.parent]
    results_json = None
    anchor_dir = None
    for cand in dir_candidates:
        candidate_json = cand / "analysis_results.json"
        if candidate_json.is_file():
            results_json = candidate_json
            anchor_dir = cand
            break
    if not results_json:
        return None, None
    try:
        data = json.loads(results_json.read_text())
    except Exception:
        return None, None

    series_json = anchor_dir / "series_analysis_results.json"
    if series_json.is_file():
        try:
            series_data = json.loads(series_json.read_text())
            per_image_results = series_data.get("results") or []

            # Propagate the regime plan and per-regime representative
            # features so the planner sees the multi-pipeline structure
            # for follow-up runs on a multi-regime series.
            series_plan = series_data.get("series_analysis_plan") or {}
            regimes = series_plan.get("regimes") or []
            if series_plan:
                data["series_analysis_plan"] = series_plan
            if regimes and per_image_results:
                regime_summaries = []
                for regime in regimes:
                    indices = regime.get("image_indices") or []
                    rep_idx = indices[0] if indices else None
                    rep_features = {}
                    if rep_idx is not None and rep_idx < len(per_image_results):
                        rep_features = (
                            (per_image_results[rep_idx] or {}).get(
                                "extracted_features"
                            ) or {}
                        )
                    regime_summaries.append({
                        "name": regime.get("name", "Unnamed"),
                        "image_indices": indices,
                        "processing_pipeline": regime.get(
                            "processing_pipeline", ""
                        ),
                        "features_to_extract": regime.get(
                            "features_to_extract", []
                        ),
                        "representative_features": rep_features,
                    })
                data["_regime_summaries"] = regime_summaries

            if per_image_results:
                first = per_image_results[0] or {}
                # First per-image result acts as the representative for
                # planner-facing scalar fields. For a single-image run
                # this is exhaustive; for a series it is one
                # representative image (series-wide aggregates like
                # `summary` and `feature_trends` come from
                # `analysis_results.json` itself).
                for key in (
                    "extracted_features",
                    "quality_metrics",
                ):
                    if key not in data and first.get(key):
                        data[key] = first[key]
                if "analysis_type" not in data and first.get("analysis_type"):
                    data["analysis_type"] = first["analysis_type"]
                if "detailed_analysis" not in data and first.get(
                    "detailed_analysis"
                ):
                    data["detailed_analysis"] = first["detailed_analysis"]

                # Aggregate `saved_arrays` across ALL per-image results so
                # files from every image_<NNNN>/ subdir can be annotated
                # by basename in the code-gen listing. Same basename
                # across images carries the same description/shape/dtype
                # by construction (the analysis writes the same array
                # name per image), so first-seen wins.
                aggregated_saved: dict = dict(data.get("saved_arrays") or {})
                for r in per_image_results:
                    if not isinstance(r, dict):
                        continue
                    for name, meta in (r.get("saved_arrays") or {}).items():
                        if name not in aggregated_saved:
                            aggregated_saved[name] = meta
                if aggregated_saved:
                    data["saved_arrays"] = aggregated_saved
        except Exception:
            pass

    return anchor_dir, data


def _first_prior_image_script(state: dict):
    """Return the first reusable analysis script for locked-script reuse (#172).

    Mirrors the curve-fit helper: scans ``state['prior_analysis_paths']`` and
    returns ``(script_text, source_label)`` for the first prior image-analysis
    run that carries a saved analysis script under ``scripts/`` — a single-
    image run writes ``scripts/analysis_script.py``; a series writes one
    ``scripts/<image>.py`` per image (all share the locked pipeline, so the
    first is a representative template). Returns ``(None, None)`` when no
    prior paths are given or none carry a script, which keeps a normal
    (no-prior) run byte-identical.
    """
    paths = state.get("prior_analysis_paths") or []
    for raw_path in paths:
        anchor_dir, _data = _load_prior_state(raw_path)
        if anchor_dir is None:
            continue
        scripts_dir = anchor_dir / "scripts"
        single = scripts_dir / "analysis_script.py"
        candidate = None
        if single.is_file():
            candidate = single
        elif scripts_dir.is_dir():
            py_files = sorted(scripts_dir.glob("*.py"))
            if py_files:
                candidate = py_files[0]
        if candidate is not None:
            try:
                return candidate.read_text(), (anchor_dir.name or str(anchor_dir))
            except Exception:  # noqa: BLE001 - a malformed prior run is skipped
                continue
    return None, None


def _append_prior_analysis_state(prompt: list, state: dict) -> None:
    """Surface a compact state summary from prior analyses to the planner.

    For each path in ``state['prior_analysis_paths']`` that points at (or
    contains) a directory holding ``analysis_results.json``, append a
    state block with the analysis's pipeline, quality score, extracted
    features, scientific claims, saved-arrays catalog, and detailed-
    analysis narrative. Missing or malformed JSON silently skips the
    path (the file-listing block for code-gen stays the single source
    of truth for "which files exist"; this helper only adds planner-
    facing context).

    Called during the planning stage; the code-gen stage consumes the
    same prior state via ``_load_prior_state`` in
    ``_generate_analysis_script``.
    """
    paths = state.get("prior_analysis_paths", [])
    if not paths:
        return

    entries = []
    for raw_path in paths:
        anchor_dir, data = _load_prior_state(raw_path)
        if anchor_dir is None:
            continue
        entries.append((anchor_dir, data))

    if not entries:
        return

    prompt.append("\n## Prior Analysis State")
    prompt.append(
        "Compact summaries of prior analyses whose full artifacts are "
        "listed to the code generator below. Use these to inform your "
        "plan — what has already been measured, what features are "
        "available, and how reliable the prior results were."
    )
    for anchor_dir, data in entries:
        label = anchor_dir.name or str(anchor_dir)
        prompt.append(f"\n### {label}")

        approach = data.get("analysis_approach") or data.get("analysis_type")
        if approach:
            prompt.append(f"- Approach: {approach}")

        qh = data.get("quality_history") or {}
        score = qh.get("final_score")
        approved = qh.get("approved")
        if score is not None or approved is not None:
            parts = []
            if score is not None:
                parts.append(f"score={score}")
            if approved is not None:
                parts.append(f"approved={approved}")
            prompt.append(f"- Quality: {', '.join(parts)}")

        # Series-level summary (only present when prior run was a series).
        summary = data.get("summary") or {}
        if summary:
            summary_parts = []
            if summary.get("total_images") is not None:
                summary_parts.append(
                    f"{summary['total_images']} images "
                    f"({summary.get('successful_analyses', '?')} successful)"
                )
            if summary.get("flagged_count"):
                summary_parts.append(
                    f"{summary['flagged_count']} flagged"
                )
            if summary.get("locked_approach"):
                summary_parts.append(
                    f"approach: {summary['locked_approach']}"
                )
            if summary_parts:
                prompt.append(f"- Series: {', '.join(summary_parts)}")

        # Multi-regime structure: when the prior run split the series
        # into regimes with different pipelines, surface each regime so
        # the follow-up planner knows which images share an analysis
        # pipeline. Single-regime series fall through to the
        # representative-features block below.
        regime_summaries = data.get("_regime_summaries") or []
        is_multi_regime = len(regime_summaries) > 1
        if is_multi_regime:
            series_plan = data.get("series_analysis_plan") or {}
            regime_lines = [f"- Regimes ({len(regime_summaries)}):"]
            if series_plan.get("rationale"):
                rationale = series_plan["rationale"]
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                regime_lines.append(f"  - Rationale: {rationale}")
            for r in regime_summaries:
                indices = r.get("image_indices") or []
                if len(indices) > 6:
                    idx_str = (
                        f"{indices[0]}–{indices[-1]} "
                        f"({len(indices)} images)"
                    )
                else:
                    idx_str = ", ".join(str(i) for i in indices)
                regime_lines.append(
                    f"  - \"{r.get('name', 'Unnamed')}\" "
                    f"(images {idx_str}):"
                )
                pipeline = r.get("processing_pipeline") or ""
                if pipeline:
                    if len(pipeline) > 200:
                        pipeline = pipeline[:200] + "..."
                    regime_lines.append(f"    - Pipeline: {pipeline}")
                rep = r.get("representative_features") or {}
                if rep:
                    feat_strs = []
                    for k, v in list(rep.items())[:6]:
                        v_str = str(v)
                        if len(v_str) > 80:
                            v_str = v_str[:80] + "..."
                        feat_strs.append(f"`{k}`={v_str}")
                    if feat_strs:
                        regime_lines.append(
                            f"    - Sample features (image "
                            f"{(r.get('image_indices') or [0])[0]}): "
                            f"{', '.join(feat_strs)}"
                        )
            prompt.append("\n".join(regime_lines))

        features = data.get("extracted_features") or {}
        if features and not is_multi_regime:
            heading = (
                "- Extracted features (representative image):"
                if summary else "- Extracted features:"
            )
            feat_lines = [heading]
            for k, v in list(features.items())[:12]:
                v_str = str(v)
                if len(v_str) > 120:
                    v_str = v_str[:120] + "..."
                feat_lines.append(f"  - `{k}`: {v_str}")
            if len(features) > 12:
                feat_lines.append(f"  - ... ({len(features) - 12} more)")
            prompt.append("\n".join(feat_lines))

        # Series feature trends (e.g., monotonic increase, drift) live in
        # `analysis_results.json` for series runs.
        trends = data.get("feature_trends") or {}
        if trends:
            trend_lines = ["- Feature trends across series:"]
            for k, v in list(trends.items())[:8]:
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                trend_lines.append(f"  - `{k}`: {v_str}")
            if len(trends) > 8:
                trend_lines.append(f"  - ... ({len(trends) - 8} more)")
            prompt.append("\n".join(trend_lines))

        claims = data.get("scientific_claims") or []
        if claims:
            claim_lines = ["- Scientific claims:"]
            for c in claims[:6]:
                c_str = c if isinstance(c, str) else (
                    c.get("claim") or c.get("text") or str(c)
                )
                if len(c_str) > 200:
                    c_str = c_str[:200] + "..."
                claim_lines.append(f"  - {c_str}")
            if len(claims) > 6:
                claim_lines.append(f"  - ... ({len(claims) - 6} more)")
            prompt.append("\n".join(claim_lines))

        saved = data.get("saved_arrays") or {}
        if saved:
            saved_lines = ["- Saved arrays:"]
            for name, meta in saved.items():
                desc = ""
                shape = ""
                if isinstance(meta, dict):
                    desc = meta.get("description", "")
                    shape_val = meta.get("shape")
                    if shape_val:
                        shape = f" shape {shape_val}"
                desc_str = f" — {desc}" if desc else ""
                saved_lines.append(f"  - `{name}`{shape}{desc_str}")
            prompt.append("\n".join(saved_lines))

        # Rich narrative from the prior run; mirrors what direct-API
        # Tier 2 receives via `tier1_summary` in IMAGE_ANALYSIS_TIER2_
        # PLANNING_INSTRUCTIONS.format(...).
        detailed = data.get("detailed_analysis") or ""
        if detailed:
            if len(detailed) > 2000:
                detailed = detailed[:2000] + "\n... (truncated)"
            prompt.append(f"- Detailed analysis:\n{detailed}")


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


def _append_subagent_context(prompt: list, state: dict) -> None:
    """Append FFT/SAM sub-agent preprocessing results to an LLM prompt.

    Includes text summaries, visualization thumbnails, and available
    array file paths with shapes so the LLM can plan accordingly.
    """
    fft = state.get("fft_preprocessing")
    if fft:
        prompt.append("\n## FFT/NMF Preprocessing Results")
        prompt.append(
            "The image was analyzed with sliding FFT + NMF decomposition "
            "before your analysis. Use these findings to inform your "
            "approach if relevant."
        )
        if fft.get("detailed_analysis"):
            prompt.append(f"\n**Findings:**\n{fft['detailed_analysis']}")
        claims = fft.get("scientific_claims", [])
        if claims:
            prompt.append("\n**Key claims:**")
            for c in claims[:5]:
                prompt.append(f"- {c.get('claim', '')}")

        if fft.get("visualization_bytes"):
            prompt.append("\n**FFT/NMF abundance map:**")
            prompt.append({
                "mime_type": "image/jpeg",
                "data": fft["visualization_bytes"],
            })

        paths = fft.get("array_paths", {})
        shapes = fft.get("array_shapes", {})
        if paths:
            prompt.append("\n**Available arrays in working directory:**")
            for name, path in paths.items():
                shape = shapes.get(name, "unknown")
                if "components" in name:
                    desc = "each component is a local FFT power spectrum pattern"
                elif "abundances" in name:
                    desc = "spatial weight map showing where each component is dominant"
                else:
                    desc = ""
                prompt.append(f"- `{name}` shape {shape} — {desc}")

    sam = state.get("sam_preprocessing")
    if sam:
        prompt.append("\n## SAM Segmentation Preprocessing Results")
        prompt.append(
            "The image was segmented using the Segment Anything Model "
            "before your analysis. Use these findings to inform your "
            "approach if relevant."
        )
        if sam.get("detailed_analysis"):
            prompt.append(f"\n**Findings:**\n{sam['detailed_analysis']}")
        if sam.get("particle_count") is not None:
            prompt.append(f"\n**Particle count:** {sam['particle_count']}")
        claims = sam.get("scientific_claims", [])
        if claims:
            prompt.append("\n**Key claims:**")
            for c in claims[:5]:
                prompt.append(f"- {c.get('claim', '')}")

        if sam.get("visualization_bytes"):
            prompt.append("\n**SAM segmentation overlay:**")
            prompt.append({
                "mime_type": "image/jpeg",
                "data": sam["visualization_bytes"],
            })

        paths = sam.get("array_paths", {})
        shapes = sam.get("array_shapes", {})
        if paths:
            prompt.append("\n**Available files in working directory:**")
            for name, path in paths.items():
                shape = shapes.get(name, "")
                if "label_map" in name:
                    desc = "0=background, 1..K=particle IDs (sorted by area)"
                elif "statistics" in name:
                    desc = "per-particle area, centroid, circularity, aspect ratio, solidity"
                else:
                    desc = ""
                shape_str = f" shape {shape}" if shape else ""
                prompt.append(f"- `{name}`{shape_str} — {desc}")


class AnalyzeImageController:
    """Compute image statistics and create initial thumbnail."""

    def __init__(self, logger: logging.Logger, image_to_bytes_fn: Callable):
        self.logger = logger
        self.image_to_bytes_fn = image_to_bytes_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n--- Analyzing Image ---\n")

        try:
            image = state["image_data"]

            state["image_statistics"] = compute_image_statistics(image)

            # Create thumbnail for LLM prompts
            thumbnail_bytes = self.image_to_bytes_fn(image)
            state["original_image_bytes"] = thumbnail_bytes
            state["analysis_images"] = [{"label": "Original Image", "data": thumbnail_bytes}]

            self.logger.info(f"  Shape: {state['image_statistics']['shape']}")
            self.logger.info(f"  Dtype: {state['image_statistics']['dtype']}")
            self.logger.info(f"  Channels: {state['image_statistics']['channels']}")
            self.logger.info(f"  Intensity: {state['image_statistics']['intensity_range']}")

        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Image analysis failed", "details": str(e)}

        return state


class ImageSeriesScoutController:
    """Scout representative images across a series before planning.

    For series with n > 1, loads and thumbnails representative images
    (evenly spaced, capped at 7) so the LLM can see how data evolves
    across the series and plan analysis regimes proactively.

    For n == 1: no-op (state passes through unchanged).
    """

    def __init__(
        self,
        logger: logging.Logger,
        image_to_bytes_fn: Callable,
        montage_fn: Callable,
    ):
        self.logger = logger
        self.image_to_bytes_fn = image_to_bytes_fn
        self.montage_fn = montage_fn

    @staticmethod
    def _select_scout_indices(num_images: int) -> list:
        """Select evenly spaced representative indices, capped at 7."""
        if num_images <= 3:
            return list(range(num_images))
        if num_images <= 6:
            mid = num_images // 2
            return sorted({0, mid, num_images - 1})
        if num_images <= 15:
            indices = {0, num_images // 4, num_images // 2,
                       3 * num_images // 4, num_images - 1}
            return sorted(indices)
        # Large series: 7 evenly spaced
        step = (num_images - 1) / 6
        indices = {round(i * step) for i in range(7)}
        return sorted(indices)

    @staticmethod
    def _load_image(idx: int, state: dict) -> np.ndarray:
        image_stack = state.get("image_stack")
        if image_stack is not None:
            return image_stack[idx]
        image_path = state.get("image_paths", [])[idx]
        return load_image_file(image_path)

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("is_single_image", True):
            return state

        num_images = state.get("num_images", 1)
        if num_images <= 1:
            return state

        self.logger.info("\n--- Scouting Series ---\n")

        scout_indices = self._select_scout_indices(num_images)
        series_metadata = state.get("series_metadata", {})
        values = series_metadata.get("values", [])
        variable = series_metadata.get("variable", "index")
        unit = series_metadata.get("unit", "")

        scout_data = []
        scout_images = []  # for montage
        scout_labels = []
        for idx in scout_indices:
            try:
                image = self._load_image(idx, state)

                stats = compute_image_statistics(image)

                if idx < len(values):
                    label = f"{variable}={values[idx]} {unit}".strip()
                else:
                    label = f"index {idx}"

                thumbnail_bytes = self.image_to_bytes_fn(image)

                scout_data.append({
                    "index": idx,
                    "label": label,
                    "statistics": stats,
                    "thumbnail_bytes": thumbnail_bytes,
                })
                scout_images.append(image)
                scout_labels.append(label)
                self.logger.info(f"  Scouted image {idx}: {label}")
            except Exception as e:
                self.logger.warning(f"  Failed to scout image {idx}: {e}")

        # Generate montage comparison
        if len(scout_images) >= 2:
            try:
                montage_bytes = self.montage_fn(scout_images, scout_labels)
                state["scout_montage_bytes"] = montage_bytes
                self.logger.info("  Generated montage comparison")
            except Exception as e:
                self.logger.warning(f"  Failed to create montage: {e}")
                state["scout_montage_bytes"] = None
        else:
            state["scout_montage_bytes"] = None

        state["scout_data"] = scout_data
        self.logger.info(f"  Scouted {len(scout_data)} of {num_images} images")

        return state


class SkillSuggestionController:
    """Auto-suggest a domain skill when none was explicitly provided.

    Runs after scouting and before planning. Shows the LLM the image(s)
    alongside a catalog of available skills and asks whether any is relevant.
    No-op when a skill was already loaded (e.g. by the orchestrator or user).
    """

    def __init__(self, model, logger, generation_config, safety_settings,
                 parse_fn, domain="image_analysis"):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.domain = domain

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("skill_sections"):
            return state

        from ....skills.loader import list_skills, load_skill

        available = list_skills(domain=self.domain)
        if not available:
            return state

        catalog, cache = [], {}
        for name in available:
            try:
                parsed = load_skill(name, domain=self.domain)
                cache[name] = parsed
                catalog.append(f"- **{name}**: {parsed.get('overview', '').strip()}")
            except Exception:
                continue
        if not catalog:
            return state

        self.logger.info("\n--- Skill Suggestion ---\n")

        prompt = [
            "Based on the image(s) below, decide whether any of these "
            "domain skills is relevant.\n\n## Available Skills\n"
            + "\n".join(catalog),
        ]
        image_bytes = (state.get("scout_montage_bytes")
                       or state.get("original_image_bytes"))
        if image_bytes:
            prompt.append({"mime_type": "image/jpeg", "data": image_bytes})
        prompt.append(
            'Respond with JSON: {"skill": "<name>"} if clearly relevant, '
            'or {"skill": null} if none applies.'
        )

        try:
            resp = self.model.generate_content(
                contents=prompt, generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, _ = self._parse(resp)
            suggested = (result or {}).get("skill")
            if suggested and suggested in cache:
                state["skill_name"] = cache[suggested]["name"]
                state["skill_sections"] = cache[suggested]
                print(f"  Auto-selected domain skill: {suggested}")
            else:
                self.logger.info("  No skill auto-selected")
        except Exception as e:
            self.logger.warning(f"  Skill suggestion failed: {e}")

        return state


class ImagePlanningController:
    """
    Plan image analysis approach via LLM, with optional human feedback.

    Works identically for single images and series:
    - Single image: Plan analysis, then process that one image
    - Series: Plan analysis on first image, then apply to all
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
        max_iterations: int = 5,
        num_plan_candidates: int = 1,
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
        self.num_plan_candidates = num_plan_candidates

    def _get_instructions(self, state: dict) -> str:
        """Return planning instructions, using state override if present."""
        return state.get("planning_instructions_override") or self.instructions

    def _display_plan(self, state: dict) -> None:
        is_single = state.get("is_single_image", True)
        num_images = state.get("num_images", 1)

        print("\n" + "=" * 60)
        mode_str = "SINGLE IMAGE" if is_single else f"SERIES ({num_images} images)"
        print(f"📋 PROPOSED ANALYSIS PLAN - {mode_str}")
        print("=" * 60)

        if state.get("observations"):
            print(f"\n🔍 Observations:\n   {state['observations']}")

        print(f"\n📊 Approach:\n   {state.get('analysis_approach', 'N/A')}")

        _pipeline = state.get("processing_pipeline", "N/A")
        _pipeline = re.sub(r"\. (\d+)\. ", r".\n   \1. ", _pipeline)
        print(f"\n⚙️  Pipeline:\n   {_pipeline}")

        print(f"\n🎯 Features to Extract:\n   {', '.join(state.get('features_to_extract', [])) or 'N/A'}")
        print(f"\n✅ Quality Criteria:\n   {state.get('quality_criteria', 'N/A')}")

        if state.get("expected_outputs"):
            print(f"\n📄 Expected Outputs:\n   {', '.join(state.get('expected_outputs', []))}")

        # Display regime plan if present
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes") and not is_single:
            regimes = series_plan["regimes"]
            print(f"\n{'=' * 60}")
            print(f"📦 IMAGE ANALYSIS REGIMES ({len(regimes)} regimes)")
            print(f"{'=' * 60}")
            if series_plan.get("rationale"):
                print(f"\nRationale: {series_plan['rationale']}")

            series_metadata = state.get("series_metadata", {})
            values = series_metadata.get("values", [])
            unit = series_metadata.get("unit", "")

            for i, regime in enumerate(regimes, 1):
                indices = regime.get("image_indices", [])
                if values and indices:
                    valid_vals = [values[idx] for idx in indices if idx < len(values)]
                    range_str = f" ({min(valid_vals)}-{max(valid_vals)} {unit})" if valid_vals else ""
                else:
                    range_str = ""
                print(f"\n  Regime {i}: {regime.get('name', 'Unnamed')}")
                print(f"    Images: indices {indices}{range_str}")
                print(f"    Pipeline: {regime.get('processing_pipeline', 'N/A')}")
                print(f"    Features: {', '.join(regime.get('features_to_extract', []))}")

            transitions = series_plan.get("transition_points", [])
            if transitions:
                print(f"\n  Transition Points:")
                for t in transitions:
                    print(f"    Between indices {t.get('between_indices', '?')}: "
                          f"{t.get('description', 'N/A')}")
        elif not is_single:
            print(f"\n📦 **Note:** This analysis pipeline will be LOCKED and applied to all {num_images} images.")

        print("\n" + "=" * 60)

    def _get_human_feedback(self, state: dict) -> dict:
        self._display_plan(state)
        feedback = input("\nYour feedback (or Enter to accept): ").strip()

        if feedback == "":
            print("Plan accepted.")
            return state
        else:
            state["_refine_requested"] = True
            state["_refine_feedback"] = feedback
            return state

    def _build_planning_prompt(self, state: dict, extra_suffix: str = "") -> list:
        """Build the planning prompt, optionally with a diversity suffix."""
        instructions = self._get_instructions(state)
        if extra_suffix:
            instructions = instructions + extra_suffix

        prompt = [
            instructions,
            "\n## Image",
            {"mime_type": "image/jpeg", "data": state["original_image_bytes"]},
            "\n## Image Statistics\n" + json.dumps(state["image_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        _append_objective_context(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_tool_inventory(prompt, agent="image_analysis", active_skills=_active_skill_names(state))
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)
        _append_prior_analysis_state(prompt, state)
        _append_subagent_context(prompt, state)

        if state.get("literature_context"):
            prompt.append("\n## Literature\n" + state["literature_context"])

        # Series context: use scout data if available, otherwise basic notice
        num_images = state.get("num_images", 1)
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_image", True):
            self._append_scout_context(prompt, state, scout_data)
        elif not state.get("is_single_image", True):
            prompt.append(
                f"\n## Series Context\nThis is the first image in a series of {num_images}. "
                "The analysis pipeline you choose will be applied to ALL images in the series."
            )

        return prompt

    def _plan_analysis(self, state: dict, extra_suffix: str = "") -> dict:
        """Generate a single analysis plan. Returns the plan as a dict."""
        prompt = self._build_planning_prompt(state, extra_suffix)
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            raise ValueError(f"Failed to parse: {error}")

        pipeline = result.get("processing_pipeline", "Standard processing")
        if isinstance(pipeline, list):
            pipeline = " -> ".join(str(s) for s in pipeline)

        return {
            "observations": result.get("observations", ""),
            "analysis_approach": result.get("analysis_approach", "Image analysis"),
            "processing_pipeline": pipeline,
            "features_to_extract": result.get("features_to_extract", []),
            "quality_criteria": result.get("quality_criteria", "Visual inspection"),
            "expected_outputs": result.get("expected_outputs", []),
            "literature_query": result.get("literature_query"),
            "series_analysis_plan": result.get("series_analysis_plan"),
        }

    def _apply_plan_to_state(self, state: dict, plan: dict) -> dict:
        """Apply a plan dict's fields to state and extract series plan."""
        state["observations"] = plan["observations"]
        state["analysis_approach"] = plan["analysis_approach"]
        state["processing_pipeline"] = plan["processing_pipeline"]
        state["features_to_extract"] = plan["features_to_extract"]
        state["quality_criteria"] = plan["quality_criteria"]
        state["expected_outputs"] = plan["expected_outputs"]
        state["literature_query"] = plan["literature_query"]

        # Extract series analysis plan if present
        self._extract_series_plan(state, plan)

        return state

    def _generate_candidate_plans(self, state: dict) -> dict:
        """Generate N candidate plans with different approaches, select the best."""
        from ..instruct import IMAGE_ANALYSIS_PLAN_DIVERSITY_SUFFIX

        n = self.num_plan_candidates
        candidates = []

        for i in range(n):
            self.logger.info(f"  Generating plan candidate {i + 1}/{n}...")

            # Build diversity suffix from previously generated pipelines
            if i == 0:
                extra_suffix = ""
            else:
                prev_lines = []
                for j, c in enumerate(candidates):
                    prev_lines.append(
                        f"- Plan {j + 1}: {c['analysis_approach']} | "
                        f"Pipeline: {c['processing_pipeline']}"
                    )
                extra_suffix = IMAGE_ANALYSIS_PLAN_DIVERSITY_SUFFIX.format(
                    previous_approaches="\n".join(prev_lines)
                )

            try:
                plan = self._plan_analysis(state, extra_suffix=extra_suffix)
                candidates.append(plan)
                self.logger.info(
                    f"    Approach: {plan['analysis_approach'][:80]}"
                )
                self.logger.info(
                    f"    Pipeline: {plan['processing_pipeline'][:80]}..."
                )
            except Exception as e:
                self.logger.warning(f"    Candidate {i + 1} failed: {e}")

        if not candidates:
            raise ValueError("All plan candidates failed to generate")

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            winner = self._select_best_plan(candidates, state)

        return self._apply_plan_to_state(state, winner)

    def _select_best_plan(self, candidates: list, state: dict) -> dict:
        """Use LLM to select the best plan from candidates."""
        from ..instruct import IMAGE_ANALYSIS_PLAN_SELECTION_PROMPT

        # Format candidates for the prompt
        candidates_text = []
        for i, c in enumerate(candidates):
            regime_info = ""
            sp = c.get("series_analysis_plan")
            if sp and sp.get("regimes"):
                regimes_str = "; ".join(
                    f"{r.get('name', 'unnamed')}: indices {r.get('image_indices', [])}"
                    for r in sp["regimes"]
                )
                regime_info = f"\n  Regimes: {regimes_str}"
            candidates_text.append(
                f"### Plan {i}\n"
                f"  Observations: {c['observations'][:200]}\n"
                f"  Approach: {c['analysis_approach']}\n"
                f"  Pipeline: {c['processing_pipeline']}\n"
                f"  Features: {', '.join(c['features_to_extract'])}\n"
                f"  Quality Criteria: {c['quality_criteria']}"
                f"{regime_info}"
            )

        prompt_text = IMAGE_ANALYSIS_PLAN_SELECTION_PROMPT.format(
            num_candidates=len(candidates),
            candidates_formatted="\n\n".join(candidates_text),
        )

        prompt_parts = [prompt_text]

        # Include image for reference
        prompt_parts.append("\n## Image for Reference")
        prompt_parts.append({
            "mime_type": "image/jpeg",
            "data": state["original_image_bytes"],
        })

        # Include scout montage for series
        montage = state.get("scout_montage_bytes")
        if montage:
            prompt_parts.append("\n## Series Montage")
            prompt_parts.append({
                "mime_type": "image/jpeg",
                "data": montage,
            })

        try:
            response = self.model.generate_content(
                prompt_parts, generation_config=self.generation_config,
            )
            result, error = self._parse(response)

            if error or not result:
                self.logger.warning(
                    f"  Plan selection failed: {error}. Using candidate 0."
                )
                return candidates[0]

            idx = result.get("selected_index", 0)
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                self.logger.warning(
                    f"  Invalid selected_index {idx}. Using candidate 0."
                )
                return candidates[0]

            reasoning = result.get("reasoning", "")
            self.logger.info(f"  Selected plan {idx}: {reasoning[:120]}")
            return candidates[idx]

        except Exception as e:
            self.logger.warning(
                f"  Plan selection error: {e}. Using candidate 0."
            )
            return candidates[0]

    def _append_scout_context(self, prompt: list, state: dict, scout_data: list) -> None:
        """Append scout image thumbnails and series regime planning instructions."""
        from ..instruct import IMAGE_ANALYSIS_SERIES_REGIME_SUPPLEMENT

        num_images = state.get("num_images", 1)
        series_metadata = state.get("series_metadata", {})

        prompt.append(f"\n## Series Overview ({num_images} images)")
        prompt.append(
            "Below are representative images from across the series. "
            "Examine how the data changes. If the image character changes "
            "significantly (e.g., new features, structural transitions, "
            "major contrast changes), plan multiple analysis regimes. "
            "Otherwise, a single pipeline is fine."
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
                    f"between images — account for their effect when "
                    f"interpreting how the data evolves."
                )

        # Montage comparison (all scouts in one figure)
        montage = state.get("scout_montage_bytes")
        if montage:
            prompt.append(
                "\n### Montage Comparison\n"
                "All scout images shown together for direct visual comparison. "
                "Look for changes in features, contrast, structural transitions, "
                "or new features emerging across the series."
            )
            prompt.append({
                "mime_type": "image/jpeg",
                "data": montage,
            })

        prompt.append("\n### Individual Scout Images")
        for scout in scout_data:
            prompt.append(
                f"\n### Image at {scout['label']} (index {scout['index']})"
            )
            prompt.append(f"Statistics: {json.dumps(scout['statistics'], indent=2)}")
            prompt.append({
                "mime_type": "image/jpeg",
                "data": scout["thumbnail_bytes"],
            })

        prompt.append(IMAGE_ANALYSIS_SERIES_REGIME_SUPPLEMENT.format(
            num_images=num_images,
            num_images_minus_1=num_images - 1,
        ))

    def _extract_series_plan(self, state: dict, result: dict) -> None:
        """Extract and validate series_analysis_plan from LLM response."""
        series_plan = result.get("series_analysis_plan")
        if not series_plan or state.get("is_single_image", True):
            state["series_analysis_plan"] = None
            return

        num_images = state.get("num_images", 1)
        regimes = series_plan.get("regimes", [])

        if not regimes:
            state["series_analysis_plan"] = None
            return

        # Validate index coverage
        all_indices = set()
        for regime in regimes:
            indices = regime.get("image_indices", [])
            # Filter to valid range
            regime["image_indices"] = [i for i in indices if 0 <= i < num_images]
            all_indices.update(regime["image_indices"])

        missing = set(range(num_images)) - all_indices
        if missing:
            self.logger.warning(
                f"  Series plan missing indices {sorted(missing)}, "
                f"assigning to first regime"
            )
            regimes[0]["image_indices"] = sorted(
                set(regimes[0]["image_indices"]) | missing
            )

        state["series_analysis_plan"] = series_plan
        self.logger.info(
            f"  Series analysis plan: {len(regimes)} regime(s)"
        )
        for regime in regimes:
            self.logger.info(
                f"    {regime.get('name', 'unnamed')}: "
                f"indices {regime.get('image_indices', [])}, "
                f"pipeline: {regime.get('processing_pipeline', 'N/A')}"
            )

    def _refine_plan(self, state: dict, feedback: str) -> dict:
        current_plan = (
            f"Observations: {state.get('observations', 'N/A')}\n"
            f"Approach: {state.get('analysis_approach', 'N/A')}\n"
            f"Pipeline: {state.get('processing_pipeline', 'N/A')}\n"
            f"Features: {', '.join(state.get('features_to_extract', []))}\n"
            f"Quality Criteria: {state.get('quality_criteria', 'N/A')}"
        )

        prompt = [
            self._get_instructions(state),
            "\n## Image",
            {"mime_type": "image/jpeg", "data": state["original_image_bytes"]},
            "\n## Image Statistics\n" + json.dumps(state["image_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
            f"\n## Current Plan\n{current_plan}",
            f"\n## User Feedback\nAdjust the plan based on this feedback: \"{feedback}\"",
        ]

        _append_objective_context(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## Original Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_tool_inventory(prompt, agent="image_analysis", active_skills=_active_skill_names(state))
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)
        _append_prior_analysis_state(prompt, state)
        _append_subagent_context(prompt, state)

        if state.get("literature_context"):
            prompt.append("\n## Literature\n" + state["literature_context"])

        # Include current series plan and scout data in refinement context
        if state.get("series_analysis_plan"):
            prompt.append(
                f"\n## Current Series Analysis Plan\n"
                f"{json.dumps(state['series_analysis_plan'], indent=2)}"
            )
            prompt.append(
                "\nThe user may want to adjust regime boundaries, merge regimes, "
                "change pipelines for specific regimes, or switch to a single pipeline. "
                "Adjust the series_analysis_plan accordingly, or remove it entirely "
                "if the user wants a single pipeline."
            )
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_image", True):
            self._append_scout_context(prompt, state, scout_data)

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            self.logger.warning(f"Refinement failed: {error}. Keeping current plan.")
            return state

        state["observations"] = result.get("observations", state.get("observations", ""))
        state["analysis_approach"] = result.get("analysis_approach", state.get("analysis_approach"))
        pipeline = result.get("processing_pipeline", state.get("processing_pipeline"))
        if isinstance(pipeline, list):
            pipeline = " -> ".join(str(s) for s in pipeline)
        state["processing_pipeline"] = pipeline
        state["features_to_extract"] = result.get("features_to_extract", state.get("features_to_extract", []))
        state["quality_criteria"] = result.get("quality_criteria", state.get("quality_criteria"))
        state["expected_outputs"] = result.get("expected_outputs", state.get("expected_outputs", []))
        state["literature_query"] = result.get("literature_query", state.get("literature_query"))

        # Re-extract series plan (may have been updated or removed)
        self._extract_series_plan(state, result)

        return state

    def _validate_plan(self, state: dict) -> dict:
        """Validate the selected plan against the actual images."""
        from ..instruct import IMAGE_ANALYSIS_PLAN_VALIDATION_PROMPT

        is_single = state.get("is_single_image", True)

        # Build regime section
        regime_section = ""
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes"):
            lines = ["\n**Regimes:**"]
            for regime in series_plan["regimes"]:
                lines.append(
                    f"- {regime.get('name', 'Unnamed')}: "
                    f"indices {regime.get('image_indices', [])}, "
                    f"pipeline: {regime.get('processing_pipeline', 'N/A')}, "
                    f"features: {', '.join(regime.get('features_to_extract', []))}"
                )
            regime_section = "\n".join(lines)

        prompt_text = IMAGE_ANALYSIS_PLAN_VALIDATION_PROMPT.format(
            analysis_approach=state.get("analysis_approach", "N/A"),
            processing_pipeline=state.get("processing_pipeline", "N/A"),
            features_to_extract=", ".join(
                state.get("features_to_extract", [])
            ),
            quality_criteria=state.get("quality_criteria", "N/A"),
            regime_section=regime_section,
        )

        prompt_parts = [prompt_text]

        # Include skill context so validator understands domain guidance
        _append_skill_context(prompt_parts, state, "planning")

        # Show challenging images for validation
        scout_data = state.get("scout_data", [])
        if not is_single and scout_data:
            if series_plan and series_plan.get("regimes"):
                # Show last scouted image per regime (most challenging)
                shown = set()
                for regime in series_plan["regimes"]:
                    indices = regime.get("image_indices", [])
                    for scout in reversed(scout_data):
                        if scout["index"] in indices and scout["index"] not in shown:
                            prompt_parts.append(
                                f"\n**{regime.get('name', 'Regime')} "
                                f"— image at {scout['label']} "
                                f"(index {scout['index']}):**"
                            )
                            prompt_parts.append({
                                "mime_type": "image/jpeg",
                                "data": scout["thumbnail_bytes"],
                            })
                            shown.add(scout["index"])
                            break
            else:
                # Show last scout
                last_scout = scout_data[-1]
                prompt_parts.append(
                    f"\n**Most challenging image "
                    f"(index {last_scout['index']}, {last_scout['label']}):**"
                )
                prompt_parts.append({
                    "mime_type": "image/jpeg",
                    "data": last_scout["thumbnail_bytes"],
                })
        else:
            prompt_parts.append("\n**Image:**")
            prompt_parts.append({
                "mime_type": "image/jpeg",
                "data": state["original_image_bytes"],
            })

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
            self.logger.info(
                f"  Plan validation: {len(issues)} issue(s) found, revising"
            )
            for issue in issues:
                self.logger.info(f"    - {issue}")

            if result.get("processing_pipeline"):
                pipeline = result["processing_pipeline"]
                if isinstance(pipeline, list):
                    pipeline = " -> ".join(str(s) for s in pipeline)
                state["processing_pipeline"] = pipeline
            if result.get("features_to_extract"):
                state["features_to_extract"] = result["features_to_extract"]
            if result.get("quality_criteria"):
                state["quality_criteria"] = result["quality_criteria"]

            # Update series plan if revised
            if result.get("series_analysis_plan"):
                self._extract_series_plan(state, result)

        except Exception as e:
            self.logger.warning(f"  Plan validation failed: {e}, keeping plan")

        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_image", True)
        mode_str = "SINGLE IMAGE" if is_single else "SERIES"
        self.logger.info(f"\n--- Planning Analysis ({mode_str}) ---\n")

        try:
            if self.num_plan_candidates > 1:
                state = self._generate_candidate_plans(state)
            else:
                plan = self._plan_analysis(state)
                state = self._apply_plan_to_state(state, plan)
            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Pipeline: {state['processing_pipeline']}")

            # Validate plan against actual images
            state = self._validate_plan(state)

            if self.enable_human_feedback:
                iteration = 0
                while iteration < self.max_iterations:
                    state = self._get_human_feedback(state)
                    if state.pop("_refine_requested", False):
                        feedback = state.pop("_refine_feedback", "")
                        self.logger.info(f"  Refining with feedback: {feedback}")
                        print("\nRefining plan...\n")
                        state = self._refine_plan(state, feedback)
                        iteration += 1
                    else:
                        break

                if iteration >= self.max_iterations:
                    self.logger.warning("  Max iterations reached.")
                    print("Max refinements reached. Proceeding with current plan.")

            state["locked_analysis_config"] = {
                "analysis_approach": state.get("analysis_approach"),
                "processing_pipeline": state.get("processing_pipeline"),
                "features_to_extract": state.get("features_to_extract", []),
                "quality_criteria": state.get("quality_criteria"),
                "expected_outputs": state.get("expected_outputs", []),
            }

            # Build per-regime configs if series plan has multiple regimes
            series_plan = state.get("series_analysis_plan")
            if series_plan and series_plan.get("regimes"):
                regime_configs = {}
                for regime in series_plan["regimes"]:
                    regime_config = {
                        "analysis_approach": state.get("analysis_approach"),
                        "processing_pipeline": regime.get(
                            "processing_pipeline", state.get("processing_pipeline")
                        ),
                        "features_to_extract": regime.get(
                            "features_to_extract",
                            state.get("features_to_extract", []),
                        ),
                        "quality_criteria": state.get("quality_criteria"),
                        "expected_outputs": state.get("expected_outputs", []),
                    }
                    for idx in regime.get("image_indices", []):
                        regime_configs[idx] = regime_config
                state["regime_configs"] = regime_configs
                self.logger.info(
                    f"  Locked {len(series_plan['regimes'])} regime "
                    f"configuration(s) for series processing."
                )
            else:
                state["regime_configs"] = None
                self.logger.info(
                    "  Analysis configuration locked for series processing."
                )

        except Exception as e:
            self.logger.warning(f"Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Analyze the image with appropriate methods"
            state["processing_pipeline"] = "To be determined"
            state["features_to_extract"] = []
            state["quality_criteria"] = "Visual inspection"
            state["expected_outputs"] = []
            state["literature_query"] = None
            state["locked_analysis_config"] = {}
            state["series_analysis_plan"] = None
            state["regime_configs"] = None

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
        literature_agent: Any | None = None,
        output_dir: str = "",
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
            self.logger.info("\n--- Skipping Literature (pre-fetched via search_literature tool) ---\n")
            return state

        if self.literature_agent is None:
            self.logger.info("\n--- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n--- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n--- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  Success")
            else:
                state["literature_context"] = None
                self.logger.warning("  No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class UnifiedImageProcessingController:
    """
    Processes ALL images using the locked analysis pipeline.

    Quality control features:
    - LLM-based quality verification of analysis results
    - Annealing-driven refinement loop inside the verification iterations
    - If quality remains inadequate and human feedback is enabled, asks for guidance
    - Otherwise proceeds with best available result
    - For series: detects statistical outliers that may indicate interesting physics
    """

    MAX_ATTEMPTS = 5
    DEFAULT_OUTLIER_SIGMA = 2.0
    DEFAULT_MAX_VERIFICATION_ITERATIONS = 7
    DEFAULT_QUALITY_THRESHOLD = 0.7

    # Quality verification is rubric-based scoring — not generative work.
    # Using a deterministic config (T=0) so the same visualization gets the
    # same sub-scores across iterations, which the annealing /
    # stall-counter logic depends on. Other LLM calls (planner, refiner,
    # code-gen) continue to use provider-default temperature.
    # Note: only temperature is set — Anthropic's Messages API rejects
    # temperature and top_p together.
    _VERIFIER_GEN_CONFIG = {"temperature": 0.0}

    # Constraint annealing: gradually raise the "temperature" so the
    # verifier can explore more of the pipeline space when early iterations
    # fail to produce an adequate result.  Like simulated annealing
    # (P ∝ exp(−ΔE/kT)), low T keeps the system near the locked plan
    # while high T lets it explore freely.
    #
    # NOTE: Image analysis starts softer than curve fitting — the baseline
    # (T=0) is "guidance", not "mandatory", because image analysis pipelines
    # are inherently more flexible than parametric curve models.
    _CONSTRAINT_ANNEALING_SCHEDULE = (
        # T=0  guidance: prefer the locked pipeline, suggest parameter tweaks.
        "\n**Plan-aware constraint:**\n"
        "The processing pipeline listed above is the analysis plan. "
        "Your suggested fixes should work within the planned method — recommend "
        "parameter adjustments, preprocessing improvements, or cleanup steps. "
        "Do not recommend replacing the core analysis method with a different one.\n",
        # T=1  warm: allow pipeline modifications if justified.
        "\n**Plan-aware constraint (eased — earlier fixes did not resolve the issues):**\n"
        "Prefer the smallest change that could fix the remaining issues. "
        "If you believe a pipeline change is necessary, suggest it, but explain "
        "why a parameter-level fix is insufficient.\n",
        # T=2  hot: full freedom, justify from what you see in the images.
        "\n**Plan constraint (open — previous iterations could not fix the analysis):**\n"
        "You have full freedom to suggest any pipeline or method change the data "
        "warrants. The only requirement is that you justify every deviation from "
        "the original plan based on what you observe in the images.\n",
    )

    # Tool-use constraint, graded with the same annealing ladder.
    # At T=0 the verifier and refinement LLM must stay within the tool's
    # documented parameters; at T=1 they prefer to but may replace the
    # tool with justification; at T=2 there is no extra constraint because
    # the main annealing directive already grants full pipeline freedom.
    _TOOL_CONSTRAINT_SCHEDULE = (
        # T=0 strict
        "\n**Registered tool constraints:**\n"
        "When the pipeline uses a registered tool (any function imported from "
        "`scilink.skills.*`), your suggested fixes must be achievable via that "
        "tool's documented parameters OR via preprocessing / postprocessing that "
        "happens OUTSIDE the tool call. Do not suggest modifications to the "
        "tool's internal algorithm. If the tool's documented parameters cannot "
        "fix the issue, say so explicitly rather than inventing internal "
        "modifications that would force the code generator to bypass the tool.\n",
        # T=1 preferred
        "\n**Registered tool constraints (eased):**\n"
        "Prefer keeping registered tools in the pipeline and expressing fixes "
        "through their documented parameters. If you must recommend replacing a "
        "tool with custom code, briefly state why the tool's parameters could "
        "not address the issue.\n",
        # T=2 open — main annealing already grants full freedom
        "",
    )

    # Same annealing applied to domain skill strictness during analysis.
    # Planning and interpretation stages always keep skills at T=0 (guidance).
    _SKILL_STRICTNESS_SCHEDULE = (
        # T=0: guidance (default for image analysis — softer than curve fitting's "mandatory")
        "## Domain Expertise Guidance ({name})\n"
        "The following guidance is from validated domain expertise. "
        "Use it to inform your implementation.\n\n",
        # T=1: light reference
        "## Domain Expertise Reference ({name})\n"
        "Use as reference. If the data clearly requires a different approach, "
        "deviate and explain why.\n\n",
        # T=2: context only
        "## Domain Expertise Context ({name})\n"
        "Use as background context only. Override any guidance if the data "
        "warrants it — explain the deviation.\n\n",
    )

    JUDGE_PROMPT = '''You are a scientific image analysis expert acting as a judge.

Multiple analysis attempts were made but none passed automated quality verification.
Review all attempts and select the most physically reasonable result, or declare all unacceptable.

**SELECTION CRITERIA:**
1. Physical plausibility - are the extracted features and measurements reasonable for this type of image?
2. Segmentation/detection quality - do the detected features correspond to real structures in the image?
3. False positive rate - are artifacts being incorrectly identified as features?
4. Completeness - are obvious features being missed?
5. Parsimony - prefer simpler pipelines if result quality is similar

**ATTEMPTS:**
{attempts_summary}

**VISUALIZATIONS:**
(See images below for each attempt)

Examine each analysis result carefully. Look at:
- Whether the analysis correctly identifies features visible in the original image
- Whether detected features match real structures
- Whether quantitative metrics are physically reasonable
- Whether the visualization faithfully represents the analysis output

**Return JSON:**
{{
    "selected_index": <0, 1, 2, etc., or null if ALL are unacceptable>,
    "acceptable": true/false,
    "reasoning": "detailed explanation of your choice or why all are unacceptable",
    "issues_with_selected": "any remaining concerns with the chosen result, or null if none"
}}

IMPORTANT: If one result is clearly better than others (better feature detection, fewer false positives,
more complete coverage), select it even if it is not perfect. Only return acceptable=false if ALL
results are fundamentally flawed.
'''

    HUMAN_FEEDBACK_PROMPT = '''## Analysis Quality Issue

The automated image analysis could not achieve adequate quality.

**Best Result:** Quality score = {best_score:.2f}
**Pipelines Tried:**
{pipelines_tried}

**Options:**
1. Suggest a different analysis approach or pipeline
2. Accept the best available result (type "accept")
3. Provide specific guidance (e.g., "use watershed segmentation", "threshold at 128")

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
        image_to_bytes_fn: Callable,
        enable_human_feedback: bool = False,
        outlier_sigma: float = None,
        max_verification_iterations: int = None,
        conformance_instructions: str = "",
        refinement_instructions: str = "",
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.script_instructions = script_instructions
        self.correction_instructions = correction_instructions
        self.refinement_instructions = refinement_instructions
        self.quality_instructions = quality_instructions
        self.output_dir = Path(output_dir)
        self.image_to_bytes_fn = image_to_bytes_fn
        self.enable_human_feedback = enable_human_feedback
        self.outlier_sigma = outlier_sigma if outlier_sigma is not None else self.DEFAULT_OUTLIER_SIGMA
        self.max_verification_iterations = max_verification_iterations if max_verification_iterations is not None else self.DEFAULT_MAX_VERIFICATION_ITERATIONS
        self.quality_threshold = self.DEFAULT_QUALITY_THRESHOLD
        self.conformance_instructions = conformance_instructions

    def _generate_analysis_script(
        self,
        state: dict,
        data_path: str,
        stats: dict,
        base_script: str | None = None,
    ) -> str:
        """Generate an image analysis script using the locked config.

        When ``base_script`` is provided (non-empty), uses the refinement
        prompt to adapt the previous attempt's script to the refined plan
        rather than regenerating from scratch. When ``None`` or empty,
        falls back to fresh generation via ``self.script_instructions``.
        """
        config = state.get("locked_analysis_config", {})
        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])
        skill_sections = state.get("skill_sections")
        # Prefer the `implementation` section for codegen. CLAUDE.md: implementation
        # is the going-forward name; the loader's analysis<->implementation synonym
        # fold makes single-section skills populate both, while dual-section skills
        # keep `implementation` as the runnable recipe (not `analysis`).
        codegen_recipe = (skill_sections or {}).get("implementation") or (skill_sections or {}).get("analysis")
        if codegen_recipe:
            level = state.get("_annealing_level", 0)
            preamble = self._SKILL_STRICTNESS_SCHEDULE[
                min(level, len(self._SKILL_STRICTNESS_SCHEDULE) - 1)
            ].format(name=state.get("skill_name", "skill"))
            context_parts.append(preamble + codegen_recipe)

        # Add sub-agent preprocessing array paths
        for key in ("fft_preprocessing", "sam_preprocessing"):
            preproc = state.get(key)
            if preproc and preproc.get("array_paths"):
                source = "FFT/NMF" if "fft" in key else "SAM"
                lines = [f"## {source} Preprocessing Arrays (available in working directory)"]
                for name, path in preproc["array_paths"].items():
                    shape = preproc.get("array_shapes", {}).get(name, "")
                    shape_str = f" shape {shape}" if shape else ""
                    lines.append(f"- `{name}`{shape_str}")
                context_parts.append("\n".join(lines))

        # Add prior-analysis file listing (available via absolute path;
        # NOT copied into working dir — use the absolute paths directly).
        # Annotate each file with description / shape / dtype from the
        # prior run's `saved_arrays` catalog where available, mirroring
        # the direct-API Tier 2 file listing built in
        # `image_analysis_agent._build_tier2_state`.
        prior_paths = state.get("prior_analysis_paths") or []
        if prior_paths:
            listing: list[tuple[str, dict]] = []
            file_globs = ("*.npy", "*.json", "*.csv", "*.png", "*.py")
            saved_arrays_by_basename: dict = {}
            for raw in prior_paths:
                _, prior_data = _load_prior_state(raw)
                if prior_data:
                    for name, meta in (prior_data.get("saved_arrays") or {}).items():
                        if isinstance(meta, dict):
                            saved_arrays_by_basename[name] = meta

                p = Path(raw)
                if p.is_file():
                    listing.append((str(p.resolve()), {}))
                elif p.is_dir():
                    # Include files in this dir and one level of subdirs
                    # (typical analysis-results layout: analysis_<id>/ and
                    # analysis_<id>/image_0000/). Skip the `tier1/` archive
                    # subdir to avoid duplicate entries when the prior run
                    # was a direct-API auto/deep run.
                    matched: set = set()
                    for pat in file_globs:
                        matched.update(p.glob(pat))
                        matched.update(p.glob(f"*/{pat}"))
                    matched = {
                        f for f in matched
                        if "tier1" not in f.relative_to(p).parts
                    }
                    for f in sorted(matched):
                        meta = saved_arrays_by_basename.get(f.name, {})
                        listing.append((str(f.resolve()), meta))
            if listing:
                lines = [
                    "## Prior Analysis Files (available via absolute path)",
                    "The following files from previous analyses are "
                    "accessible. Load them with `np.load` (.npy), "
                    "`json.load` or `pd.read_json` (.json), "
                    "`pd.read_csv` (.csv), `cv2.imread` (.png) as "
                    "appropriate. Use the absolute paths — files are "
                    "NOT copied into the working directory.",
                ]
                for path_str, meta in listing:
                    desc = meta.get("description", "") if meta else ""
                    shape = meta.get("shape", "") if meta else ""
                    dtype = meta.get("dtype", "") if meta else ""
                    if desc or shape or dtype:
                        annotation_parts = []
                        if desc:
                            annotation_parts.append(desc)
                        if shape:
                            annotation_parts.append(f"shape={shape}")
                        if dtype:
                            annotation_parts.append(f"dtype={dtype}")
                        lines.append(
                            f"- `{path_str}` — {', '.join(annotation_parts)}"
                        )
                    else:
                        lines.append(f"- `{path_str}`")
                context_parts.append("\n".join(lines))

        # Optional auxiliary operand(s) (#226): for each co-registered companion
        # image aligned with the primary (same H×W), write it next to the image
        # and list it in a manifest the generated script MAY use (e.g. a
        # co-registered channel masking/informing the primary). Misaligned ones
        # stay context-only (no resampling in v1).
        primary_shape = tuple(stats.get("shape") or ())
        operand_lines = []
        for j, it in enumerate(state.get("auxiliary_items") or []):
            arr = it.get("array")
            label = it.get("label") or f"reference_{j}"
            if arr is None:
                continue
            arr = np.asarray(arr)
            aligned = (
                len(primary_shape) >= 2
                and arr.ndim >= 2
                and tuple(arr.shape[:2]) == primary_shape[:2]
            )
            if aligned:
                safe = _sanitize_aux_name(label, j)
                aux_path = Path(data_path).parent / f"temp_auxiliary_{safe}.npy"
                np.save(aux_path, arr)
                operand_lines.append(
                    f"- \"{label}\": `{aux_path}` — a co-registered array of shape "
                    f"{tuple(arr.shape)} (same H×W as the primary image)."
                )
                self.logger.info(
                    f"🧩 Offering auxiliary '{label}' {tuple(arr.shape)} as an "
                    f"optional image-script operand."
                )
            else:
                self.logger.info(
                    f"Auxiliary '{label}' shape {tuple(arr.shape)} not aligned with "
                    f"primary {primary_shape}; kept as context only (not an operand)."
                )

        auxiliary_block = ""
        if operand_lines:
            auxiliary_block = (
                "\n**Optional companion operand(s):**\n"
                + "\n".join(operand_lines)
                + "\n- You MAY load any of these (np.load) and use it numerically — "
                "e.g. mask, normalize, divide, or correlate it with the primary — "
                "ONLY if your method needs it. The primary image is the base input; "
                "companions are optional, never required. Do NOT report findings "
                "about a companion as if it were the measurement; it is an operand "
                "for analyzing the primary.\n"
            )

        from ....skills._shared._registry import format_tool_inventory

        active_skills = _active_skill_names(state)
        format_kwargs = dict(
            auxiliary_block=auxiliary_block,
            analysis_approach=config.get("analysis_approach", "Analyze the image"),
            processing_pipeline=config.get("processing_pipeline", "Standard processing"),
            features_to_extract=", ".join(config.get("features_to_extract", [])) or "relevant features",
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=data_path,
            shape=stats.get("shape", "unknown"),
            dtype=stats.get("dtype", "unknown"),
            intensity_min=stats.get("intensity_range", [0, 255])[0],
            intensity_max=stats.get("intensity_range", [0, 255])[1],
            tool_inventory=format_tool_inventory("image_analysis", active_skills=active_skills),
        )

        if base_script and self.refinement_instructions:
            prompt = self.refinement_instructions.format(
                base_script=base_script,
                **format_kwargs,
            )
        else:
            prompt = self.script_instructions.format(**format_kwargs)

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Script generation failed: {error or 'no script'}")

        return result["script"]

    def _correct_script(
        self, state: dict, script: str, error_msg: str
    ) -> tuple:
        """Generate a corrected script after an execution failure.

        Returns:
            (corrected_script, diagnosis) — *diagnosis* is the LLM's
            explanation of what went wrong and how it was fixed, or None.
        """
        config = state.get("locked_analysis_config", {})
        from ....skills._shared._registry import format_tool_inventory

        active_skills = _active_skill_names(state)
        prompt = self.correction_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            processing_pipeline=config.get("processing_pipeline", ""),
            failed_script=script,
            error_message=error_msg,
            tool_inventory=format_tool_inventory("image_analysis", active_skills=active_skills),
        )
        skill_sections = state.get("skill_sections")
        # Prefer `implementation` for codegen (see the generate-script path).
        codegen_recipe = (skill_sections or {}).get("implementation") or (skill_sections or {}).get("analysis")
        if codegen_recipe:
            level = state.get("_annealing_level", 0)
            preamble = self._SKILL_STRICTNESS_SCHEDULE[
                min(level, len(self._SKILL_STRICTNESS_SCHEDULE) - 1)
            ].format(name=state.get("skill_name", "skill"))
            prompt += "\n\n" + preamble + codegen_recipe

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        diagnosis = result.get("diagnosis")
        if diagnosis:
            self.logger.info(f"    Diagnosis: {diagnosis}")

        return result["script"], diagnosis

    def _check_conformance(self, state: dict, script: str) -> dict | None:
        """Use the LLM to verify a generated script implements the locked plan.

        Returns a dict with ``conformant``, ``justified_deviations``,
        ``unjustified_deviations``, and ``summary`` keys, or ``None`` if the
        check cannot be performed (missing config, LLM error, etc.).
        """
        config = state.get("locked_analysis_config", {})
        if not config or not config.get("processing_pipeline"):
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
                    f"\n**Domain Expertise ({skill_name}):**\n"
                    + "\n".join(rules_parts)
                    + "\n"
                )

        prompt = self.conformance_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            processing_pipeline=config.get("processing_pipeline", ""),
            features_to_extract=", ".join(
                config.get("features_to_extract", [])
            ),
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

    @staticmethod
    def _load_image_data(image_path: str) -> np.ndarray:
        """Load image data from file, handling various formats."""
        return load_image_file(image_path)

    def _sanitize_script(self, script: str) -> str:
        """Sanitize analysis script for non-interactive execution."""
        # Remove plt.show() calls
        script = re.sub(r'plt\.show\s*\(\s*\)', '# plt.show() removed', script)
        # Ensure matplotlib backend is set at the top
        if 'matplotlib.use' not in script:
            script = "import matplotlib\nmatplotlib.use('Agg')\n" + script
        return script

    def _process_single_image(
        self,
        state: dict,
        image_data: np.ndarray,
        data_path: str,
        image_name: str,
        image_idx: int,
        base_script: Optional[str] = None,
        refine_from_script: Optional[str] = None,
    ) -> dict:
        """Execute analysis pipeline on a single image with retry logic.

        Two independent ways to carry a previous script forward:

        - ``base_script`` — series-level reuse. When provided, the first attempt
          runs it VERBATIM (no LLM) in this image's working directory, where the
          image is staged as the canonical ``data.npy``. Used to keep the
          pipeline consistent across images in a series.
        - ``refine_from_script`` — refinement-iteration adaptation. When
          provided, the first attempt calls ``_generate_analysis_script``
          with ``base_script=refine_from_script`` so the refinement prompt
          is used (LLM adapts the previous script to the refined plan).
          Used when carrying a working script forward across verification
          refinement iterations at lower annealing levels.

        ``base_script`` wins if both are supplied (series consistency takes
        precedence over refinement adaptation).
        """
        stats = compute_image_statistics(image_data)

        # Per-image working directory: the locked script runs VERBATIM here with the
        # image staged as the canonical DATA_NAME and the viz written canonically —
        # no per-image source rewriting, no cross-item glob hazard.
        working_dir = self.output_dir / f"image_{image_idx:04d}"
        working_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = f"image_{image_idx:04d}"

        script = None
        last_error = ""
        run = None
        script_errors = []
        last_diagnosis = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                if base_script is not None and attempt == 1:
                    script = base_script   # reuse VERBATIM (loads DATA_NAME from cwd)
                elif attempt == 1:
                    script = self._generate_analysis_script(
                        state,
                        DATA_NAME,
                        stats,
                        base_script=refine_from_script,
                    )
                    if not script_uses_canonical_input(script):
                        last_error = (
                            f"Script must load the image from '{DATA_NAME}' in the "
                            "current working directory (np.load), not another path."
                        )
                        continue
                    # Check conformance with locked plan on fresh generation
                    conformance = self._check_conformance(state, script)
                    if conformance and not conformance.get("conformant", True):
                        issues = "; ".join(
                            conformance.get("unjustified_deviations", [])
                        )
                        self.logger.warning(
                            "    Plan conformance issue: %s", issues
                        )
                        last_error = (
                            "PLAN CONFORMANCE: Script deviates from the "
                            "locked plan without justification. Issues: "
                            f"{issues}. Plan pipeline: "
                            f"{state.get('locked_analysis_config', {}).get('processing_pipeline', '')}. "
                            "Either fix the script to match the plan, or if "
                            "the plan cannot work, implement the closest "
                            "viable alternative and explain why in the summary."
                        )
                        continue
                    if conformance and conformance.get("justified_deviations"):
                        self.logger.info(
                            "    Justified plan deviations: %s",
                            "; ".join(conformance["justified_deviations"]),
                        )
                else:
                    script, last_diagnosis = self._correct_script(
                        state, script, last_error
                    )

                # Sanitize the script
                script = self._sanitize_script(script)

                run = stage_and_run(self.executor, script, image_data, working_dir)
                exec_result = run["exec"]

                if run["status"] == "success":
                    has_results = "IMAGE_ANALYSIS_RESULTS_JSON:" in run["stdout"]
                    has_visualization = run["visualization_path"] is not None

                    if has_results and has_visualization:
                        break
                    else:
                        missing = []
                        if not has_results:
                            missing.append("IMAGE_ANALYSIS_RESULTS_JSON output")
                        if not has_visualization:
                            missing.append("visualization file")
                        last_error = (
                            f"Script executed but did not produce expected outputs. "
                            f"Missing: {', '.join(missing)}. The script must print "
                            f"'IMAGE_ANALYSIS_RESULTS_JSON:{{...}}' with analysis results "
                            f"and save 'visualization.png' in the working directory."
                        )
                        self.logger.warning(
                            f"    Attempt {attempt}: Script ran but missing outputs: "
                            f"{', '.join(missing)}"
                        )
                        script_errors.append({
                            "attempt": attempt,
                            "error": last_error[:300],
                            "fix": (last_diagnosis or "")[:300] or None,
                        })
                        last_diagnosis = None
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(
                        f"    Attempt {attempt} failed: {last_error[:100]}"
                    )
                    script_errors.append({
                        "attempt": attempt,
                        "error": last_error[:300],
                        "fix": (last_diagnosis or "")[:300] or None,
                    })
                    last_diagnosis = None
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"    Attempt {attempt} error: {e}")
                script_errors.append({
                    "attempt": attempt,
                    "error": last_error[:300],
                    "fix": None,
                })

        # Success iff the final run produced BOTH the marker and the viz (matches
        # the break condition); run["status"] alone is a snapshot.
        ok = (run is not None and run["status"] == "success"
              and run["visualization_path"] is not None
              and "IMAGE_ANALYSIS_RESULTS_JSON:" in run["stdout"])
        if not ok:
            return {
                "index": image_idx,
                "name": image_name,
                "success": False,
                "error": last_error,
                "extracted_features": {},
                "quality_metrics": {},
                "script": script,
                "script_errors": script_errors,
            }

        # Parse results from stdout
        analysis_results = {}
        for line in run["stdout"].splitlines():
            if line.startswith("IMAGE_ANALYSIS_RESULTS_JSON:"):
                try:
                    analysis_results = json.loads(
                        line.replace("IMAGE_ANALYSIS_RESULTS_JSON:", "").strip()
                    )
                except json.JSONDecodeError:
                    pass
                break

        return {
            "index": image_idx,
            "name": image_name,
            "data_path": data_path,
            "success": True,
            "error": None,
            "analysis_type": analysis_results.get("analysis_type"),
            "extracted_features": analysis_results.get("extracted_features", {}),
            "quality_metrics": analysis_results.get("quality_metrics", {}),
            "summary": analysis_results.get("summary"),
            "saved_arrays": analysis_results.get("saved_arrays", {}),
            "visualization_path": run["visualization_path"],
            "visualization_bytes": run["visualization_bytes"],
            "statistics": stats,
            "script": script,
            "script_errors": script_errors,
        }

    QUALITY_VERIFICATION_PROMPT = '''You are a scientific image analysis expert reviewing an analysis result.

**TASK:** Compare the analysis visualization against the original image and score the result.

**ANALYSIS APPROACH:** {analysis_approach}
**PROCESSING PIPELINE:** {processing_pipeline}
**QUALITY CRITERIA (defined during planning):** {quality_criteria}

**EXTRACTED FEATURES:**
{features}

**QUALITY METRICS:**
{metrics}

---

## SCORING RUBRIC

Score the analysis by evaluating three dimensions, then combine into a single quality_score.

### A. Completeness — how much of what should be captured was captured?
Compare the visualization against the original image.
- For detection/segmentation tasks: what fraction of visible target structures were identified?
- For boundary/edge tasks: what fraction of visible boundaries were traced?
- For texture/phase tasks: what fraction of the image area was correctly classified?
- For measurement tasks: were all requested quantities extracted?
Score: 0.0 (nothing captured) to 1.0 (everything captured).

### B. Correctness — how much of the output corresponds to real structures?
- For detection/segmentation: are detected regions real features, not artifacts?
- For boundary/edge tasks: do detected edges follow real boundaries, not noise?
- For texture/phase tasks: do classified regions match what you see in the original?
- For measurement tasks: do extracted values match visual estimates from the image?
Score: 0.0 (entirely wrong) to 1.0 (all output is correct).

### C. Relevance — does the output address what was asked?
- Does the analysis extract the features listed in the quality criteria?
- Are the outputs scientifically usable for the stated objective?
- Would a domain scientist find the results informative?
Score: 0.0 (output is irrelevant) to 1.0 (directly answers the analysis goal).

**quality_score = 0.4 * Completeness + 0.4 * Correctness + 0.2 * Relevance**, rounded to 2 decimal places.

### Decision thresholds:
- **quality_score >= {quality_threshold}** → is_acceptable: TRUE (good enough for scientific use)
- **quality_score < {quality_threshold}** → is_acceptable: FALSE (needs improvement)

### Important:
- Score against what you SEE in the images, not against what would be ideal.
- Be honest: estimate counts and compute the ratios. Do not default to 0.5 when uncertain — make your best estimate from the images.

---

## STEP 1: CHECK FOR BROKEN ANALYSIS (score 0.0 if ANY are true)

- **Empty result?** Analysis produced no features when features are clearly visible → score 0.0
- **All-black or all-white segmentation?** Trivially empty or full mask → score 0.0
- **Wrong region?** Analysis applied to wrong part of the image → score 0.0
- **Numerical artifacts?** Values obviously impossible (negative area, NaN, etc.) → score 0.0

---

## STEP 2: IF STEP 1 PASSED, score using the rubric above

Estimate Completeness, Precision, and Plausibility separately, then average.

**Do NOT reject for:**
- Minor imperfections in segmentation boundaries
- A few missed small or ambiguous features if the majority are captured
- Slight over- or under-segmentation if the overall result is scientifically usable
- An abundance map or decomposition component that does not delineate every visible region of the image. Data-driven decompositions (NMF, PCA, ICA, FFT-NMF) produce basis patterns, not semantic segmentations — they are not required to match what you would segment by eye. If the output is coherent and faithful to the image content, score it on that basis, not on whether it matched the regions you visually identified.

---

{tool_constraint}

---

{plan_constraint}

---

## RESPONSE FORMAT

Return JSON:
{{
    "completeness": 0.0-1.0,
    "correctness": 0.0-1.0,
    "relevance": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "is_acceptable": true/false,
    "issues_found": [
        {{
            "location": "where in the image",
            "problem": "what is wrong",
            "evidence": "what you see in the visualization",
            "suggested_fix": "how to fix it"
        }}
    ],
    "missed_features": ["list of obvious features not captured"],
    "false_positives": ["list of artifacts incorrectly identified"],
    "overall_assessment": "one sentence summary",
    "recommended_action": "specific fix OR 'none'"
}}
'''

    def _verify_quality(
        self,
        state: dict,
        result: dict,
        history: List[dict] = None,
        verification_iter: int = 0,
        annealing_level: int | None = None,
    ) -> Optional[dict]:
        """Use LLM to verify analysis quality by examining the visualization.

        Returns verification result with any issues found, or None if
        verification fails.
        """
        if not result.get("visualization_bytes"):
            self.logger.warning("      No visualization available for LLM verification")
            return None

        config = state.get("locked_analysis_config", {})
        features = result.get("extracted_features", {})
        metrics = result.get("quality_metrics", {})

        features_str = json.dumps(features, indent=2) if features else "No features extracted"
        metrics_str = json.dumps(metrics, indent=2) if metrics else "No metrics available"

        # Constraint annealing: use caller-supplied level (adaptive) or fall
        # back to the legacy iteration-proportional formula.
        schedule = self._CONSTRAINT_ANNEALING_SCHEDULE
        if annealing_level is not None:
            level = min(annealing_level, len(schedule) - 1)
        else:
            n_levels = len(schedule)
            max_iter = max(self.max_verification_iterations, 1)
            level = min(verification_iter * n_levels // max_iter, n_levels - 1)
        plan_constraint = schedule[level]
        tool_schedule = self._TOOL_CONSTRAINT_SCHEDULE
        tool_constraint = tool_schedule[min(level, len(tool_schedule) - 1)]

        prompt_text = self.QUALITY_VERIFICATION_PROMPT.format(
            analysis_approach=config.get("analysis_approach", "Unknown"),
            processing_pipeline=config.get("processing_pipeline", "Unknown"),
            quality_criteria=config.get("quality_criteria", "Visual inspection"),
            features=features_str,
            metrics=metrics_str,
            quality_threshold=self.quality_threshold,
            plan_constraint=plan_constraint,
            tool_constraint=tool_constraint,
        )

        # Add history context
        history_context = build_verification_prompt_with_history(
            current_result={
                "quality_score": result.get("_quality_score"),
                "pipeline": config.get("processing_pipeline"),
                "features": features,
            },
            previous_iterations=history or [],
        )

        prompt_parts = [
            prompt_text + history_context,
            "\n\n**ANALYSIS VISUALIZATION (examine carefully):**",
        ]

        # Add the analysis visualization, capped under Anthropic's 5 MB
        # per-image limit so retry-loop figure inflation doesn't crash
        # the verification call.
        viz_bytes, viz_mime = _fit_image_under_api_cap(
            result["visualization_bytes"]
        )
        if viz_bytes is not result["visualization_bytes"]:
            self.logger.info(
                f"      Visualization shrunk for verification: "
                f"{len(result['visualization_bytes'])} → {len(viz_bytes)} bytes"
            )
        prompt_parts.append({"mime_type": viz_mime, "data": viz_bytes})

        # Also include original image for comparison
        if state.get("original_image_bytes"):
            prompt_parts.append("\n\n**ORIGINAL IMAGE (for reference):**")
            prompt_parts.append({
                "mime_type": "image/jpeg",
                "data": state["original_image_bytes"]
            })

        # Include domain-specific validation criteria from skill
        skill_sections = state.get("skill_sections")
        if skill_sections and skill_sections.get("validation"):
            skill_name = state.get("skill_name", "domain skill")
            prompt_parts.append(
                f"\n\n**Domain Validation Criteria ({skill_name}):**\n"
                "Use these criteria when scoring completeness and correctness.\n\n"
                + skill_sections["validation"]
            )

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self._VERIFIER_GEN_CONFIG,
                safety_settings=self.safety_settings,
            )
            result_parsed, error = self._parse(response)

            if error or not result_parsed:
                self.logger.warning(f"      LLM verification parse failed: {error}")
                return None

            return result_parsed

        except Exception as e:
            self.logger.error(f"      LLM verification failed: {e}")
            return None

    def _format_refinement_history(self, history: list) -> str:
        """Compact per-iteration trajectory for the refinement LLM.

        Gives the refiner visibility into prior scores and pipelines so it
        can recognize when a recent change regressed and consider backing
        off toward a better-scoring earlier config. Leaner than
        ``build_verification_prompt_with_history`` (used by the verifier) —
        the refiner needs scores + pipeline summaries, not full issue lists.
        """
        if not history:
            return ""

        # Cap at the most recent 6 entries to bound token cost.
        recent = history[-6:]
        scores = [h.get("quality_score", 0.0) for h in recent]
        if not scores:
            return ""
        best = max(scores)

        lines = [
            "**ITERATION HISTORY (oldest first, current last):**",
            "Use this trajectory to decide whether the current pipeline is "
            "improving or has regressed from an earlier version. If a recent "
            "change dropped the score, consider reverting toward the "
            "better-scoring config and making a smaller adjustment rather "
            "than piling on further changes. You may still accept a temporary "
            "regression if your reasoning supports it.",
            "",
        ]
        n = len(recent)
        for i, h in enumerate(recent):
            score = h.get("quality_score", 0.0)
            cfg = h.get("config_used", {}) or {}
            pipeline = (cfg.get("processing_pipeline", "") or "").strip()
            if len(pipeline) > 200:
                pipeline = pipeline[:200] + "..."
            markers = []
            if abs(score - best) < 1e-6:
                markers.append("BEST")
            if i == n - 1:
                markers.append("CURRENT")
            tag = f" [{', '.join(markers)}]" if markers else ""
            lines.append(
                f"- score={score:.2f}{tag}: "
                f"{pipeline or '(no pipeline captured)'}"
            )
        return "\n".join(lines)

    def _apply_verification_feedback(
        self,
        state: dict,
        verification: dict,
        history: list | None = None,
    ) -> dict:
        """Apply LLM verification feedback to refine the analysis configuration.

        Args:
            state: Pipeline state dict.
            verification: Latest verifier output (scores, issues, recommended_action).
            history: Optional list of prior ``verification_history`` entries. When
                provided, a compact trajectory is included in the refinement prompt
                so the LLM can reason about regressions. Leave ``None`` to preserve
                previous behavior (no history injected).

        Returns:
            Updated analysis config dict.
        """
        config = state.get("locked_analysis_config", {}).copy()

        recommended_action = verification.get("recommended_action", "")
        if not recommended_action or recommended_action.lower() == "none":
            return config

        # Build a refinement prompt based on verification results
        issues_summary = []
        for issue in verification.get("issues_found", []):
            issues_summary.append(
                f"- {issue.get('location', 'Unknown')}: "
                f"{issue.get('problem', '')} -> {issue.get('suggested_fix', '')}"
            )

        missed = verification.get("missed_features", [])
        false_positives = verification.get("false_positives", [])

        # Inject the same constraint annealing directive so the refinement
        # LLM respects the current temperature level.
        annealing_level = state.get("_annealing_level", 0)
        schedule = self._CONSTRAINT_ANNEALING_SCHEDULE
        constraint_text = schedule[min(annealing_level, len(schedule) - 1)]
        tool_schedule = self._TOOL_CONSTRAINT_SCHEDULE
        tool_constraint = tool_schedule[
            min(annealing_level, len(tool_schedule) - 1)
        ]

        history_text = self._format_refinement_history(history or [])
        history_section = f"\n{history_text}\n" if history_text else ""

        refinement_prompt = f"""Refine the image analysis approach based on automated verification feedback.

**CURRENT APPROACH:**
- Pipeline: {config.get('processing_pipeline', 'Unknown')}
- Approach: {config.get('analysis_approach', 'Unknown')}
{constraint_text}
{tool_constraint}
{history_section}
**VERIFICATION FINDINGS:**
{chr(10).join(issues_summary) if issues_summary else 'No specific issues listed'}

**MISSED FEATURES:** {', '.join(missed) if missed else 'None identified'}

**FALSE POSITIVES:** {', '.join(false_positives) if false_positives else 'None identified'}

**RECOMMENDED ACTION:** {recommended_action}

Return JSON with the refined analysis approach:
{{
    "processing_pipeline": "updated pipeline description",
    "analysis_approach": "updated approach",
    "features_to_extract": ["list", "of", "features"],
    "quality_criteria": "updated quality criteria"
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

            config.update(result)
            return config

        except Exception as e:
            self.logger.error(f"      Refinement failed: {e}")
            config["_refinement_error"] = str(e)
            return config

    def _get_human_feedback_for_poor_quality(
        self, state: dict, best_result: dict, all_attempts: List[dict]
    ) -> Optional[dict]:
        """Ask user for guidance when automated quality is poor."""
        # Build concise summary grouped by attempt type
        lines = []
        initial = [a for a in all_attempts if not str(a["pipeline"]).startswith(("Verification", "User"))]
        verifications = [a for a in all_attempts if str(a["pipeline"]).startswith("Verification")]
        user_guided = [a for a in all_attempts if str(a["pipeline"]).startswith("User")]

        if initial:
            lines.append(f"  Initial pipeline: score = {initial[0]['score']:.2f}")
        if verifications:
            scores = [f"{v['score']:.2f}" for v in verifications]
            lines.append(f"  Verification refinements ({len(verifications)}): scores = {', '.join(scores)}")
        for a in user_guided:
            lines.append(f"  User-guided: score = {a['score']:.2f}")

        pipelines_tried = "\n".join(lines)

        print("\n\n")
        print("─" * 60)
        print()
        print("⚠️  ANALYSIS QUALITY BELOW THRESHOLD")
        print()
        print("─" * 60)

        if best_result.get("visualization_bytes"):
            viz_path = self.output_dir / "quality_review_analysis.png"
            with open(viz_path, 'wb') as f:
                f.write(best_result["visualization_bytes"])
            print(f"\n📊 [Best result saved to: {viz_path}]")

        prompt = self.HUMAN_FEEDBACK_PROMPT.format(
            best_score=best_result.get("_quality_score", 0.0),
            pipelines_tried=pipelines_tried,
        )
        print(prompt)

        feedback = input("\nYour input: ").strip()

        if not feedback:
            print("No feedback provided. Proceeding with best available result.")
            return None

        if "accept" in feedback.lower() or "proceed" in feedback.lower():
            print("Accepting best available result.")
            return None

        print("Will retry with your suggested approach...")
        return {"action": "retry", "feedback": feedback}

    def _get_user_feedback_on_result(
        self, state: dict, analysis_result: dict, quality_score: float
    ) -> Optional[str]:
        """Show user the analysis result and ask for optional feedback."""
        is_single = state.get("is_single_image", True)
        num_images = state.get("num_images", 1)

        print("\n" + "=" * 70)
        if is_single:
            print("ANALYSIS RESULT - Review Before Synthesis")
        else:
            print("FIRST IMAGE RESULT - Review Before Processing Series")
        print("=" * 70)

        review_viz_path = None
        if analysis_result.get("visualization_bytes"):
            review_viz_path = self.output_dir / "first_image_analysis_review.png"
            with open(review_viz_path, 'wb') as f:
                f.write(analysis_result["visualization_bytes"])
            print(f"\n[Analysis visualization saved to: {review_viz_path}]")

        print(f"\nAnalysis: {analysis_result.get('analysis_type', 'N/A')}")
        print(f"Quality Score: {quality_score:.2f}")

        features = analysis_result.get("extracted_features", {})
        if features:
            print("\nExtracted Features:")
            for k, v in features.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4g}")
                else:
                    print(f"   {k}: {v}")

        if not is_single:
            regime_name = state.get("_current_regime_name")
            if regime_name:
                print(f"\nThis analysis pipeline will be applied to all images in regime '{regime_name}'.")
            else:
                print(f"\nThis analysis pipeline will be applied to all {num_images} images in the series.")

        print("\n" + "-" * 60)
        print("Options:")
        if is_single:
            print("  - Press Enter to accept and proceed to interpretation")
        else:
            print("  - Press Enter to accept this result and proceed with series")
        print("  - Type feedback to modify the analysis approach")
        print("-" * 60)

        feedback = input("\nYour feedback (or Enter to accept): ").strip()

        if review_viz_path and review_viz_path.exists():
            try:
                os.remove(review_viz_path)
            except Exception:
                pass

        if not feedback:
            if is_single:
                print("Result accepted. Proceeding to interpretation...")
            else:
                print("Result accepted. Proceeding with series...")
            return None

        return feedback

    def _ask_keep_user_guided_result(
        self, user_score: float, original_score: float
    ) -> bool:
        """Ask user whether to keep user-guided result even if quality is worse."""
        print("\n" + "-" * 60)
        print(
            f"User-guided result has lower quality ({user_score:.2f}) "
            f"than original ({original_score:.2f})"
        )
        print("-" * 60)
        print("Options:")
        print(f"  - Type 'keep' to use the user-guided result anyway (score = {user_score:.2f})")
        print(f"  - Press Enter to revert to original result (score = {original_score:.2f})")

        response = input("\nYour choice: ").strip().lower()

        if response == 'keep':
            print("Keeping user-guided result.")
            return True
        else:
            print("Reverting to original result.")
            return False

    def _refine_config_from_feedback(self, state: dict, feedback: str) -> dict:
        """Refine analysis config based on user feedback."""
        config = state.get("locked_analysis_config", {})
        prompt = f"""Refine the image analysis approach based on user feedback.

**Current Approach:**
- Pipeline: {config.get('processing_pipeline', 'Unknown')}
- Approach: {config.get('analysis_approach', 'Unknown')}

**User Feedback:** {feedback}

Return JSON with:
{{
    "processing_pipeline": "updated pipeline description",
    "analysis_approach": "updated approach",
    "features_to_extract": ["list", "of", "features"],
    "quality_criteria": "updated criteria"
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
            self.logger.error(f"Failed to refine config from feedback: {e}")
            return config

    def _execute_and_verify(
        self,
        state: dict,
        image_data: np.ndarray,
        data_path: str,
        image_name: str,
        image_idx: int,
        is_regime_anchor: bool = False,
        reuse_script: Optional[str] = None,
        reuse_source: Optional[str] = None,
    ) -> dict:
        """Execute analysis with quality control, verification, and optional judge selection.

        Flow:
        1. Initial analysis attempt
        2. For anchor image (first in series or first in regime): LLM verification loop
           - Each iteration: verify current result -> if issues, re-analyze
           - After loop: verify final result
           - If still not approved: call judge to select best
        3. If still below quality threshold: try alternative pipelines
        4. If human feedback enabled: allow user to guide refinement

        #172 locked-script reuse: when ``reuse_script`` is supplied (an anchor
        fed a prior run's saved analysis script via ``prior_analysis_paths``),
        the prior script is run verbatim on the new image first. If it
        executes, its result is kept — regardless of the quality score — so
        the extracted-feature schema stays consistent across an incremental
        measurement campaign by construction. A single vision-verification
        pass supplies a soft ``reuse_validity`` verdict the orchestrator can
        act on; it never re-derives the pipeline. Full QC runs only when the
        prior script cannot execute at all.
        """
        all_attempts = []
        verification_history = []
        judge_result = None
        best_result = None
        best_score = -1.0
        best_config = state.get("locked_analysis_config", {}).copy()
        quality_threshold = self.quality_threshold

        # Anchor = first image overall OR first in a regime; gets full QC
        _is_anchor = image_idx == 0 or is_regime_anchor

        # --- #172: locked-script reuse fast path ---
        # A prior image-analysis run supplied via prior_analysis_paths means
        # the new image is unit N+1 of that series: reuse the prior run's
        # locked analysis script verbatim instead of re-deriving the pipeline.
        # This keeps the extracted-feature schema consistent across an
        # incremental campaign by construction. The vision verifier runs once
        # as a soft validity *signal* (attached as reuse_validity for the
        # orchestrator) — it never re-derives the pipeline, since a re-derived
        # pipeline could change the feature columns. The only fallback to full
        # QC is a prior script that cannot execute at all.
        if reuse_script and _is_anchor:
            self.logger.info(
                f"   ♻️  Reusing locked analysis script from prior run "
                f"'{reuse_source or 'prior'}'..."
            )
            reuse_result = self._process_single_image(
                state=state, image_data=image_data, data_path=data_path,
                image_name=image_name, image_idx=image_idx,
                base_script=reuse_script,
            )
            if reuse_result.get("success"):
                # Softer validity guard: a single vision-verification pass,
                # no iterative re-derivation.
                verification = self._verify_quality(
                    state, reuse_result, history=[],
                    verification_iter=0, annealing_level=0,
                )
                v_score = 0.0
                if verification and isinstance(
                    verification.get("quality_score"), (int, float)
                ):
                    v_score = verification["quality_score"]
                verdict = "good" if v_score >= quality_threshold else "poor"
                reuse_result["_quality_score"] = v_score
                if verdict == "good":
                    self.logger.info(
                        f"   ✅ Reused script verified (score = {v_score:.2f} "
                        f"≥ {quality_threshold}) — pipeline re-derivation "
                        f"skipped"
                    )
                    message = (
                        f"Reused the locked analysis script from prior run "
                        f"'{reuse_source or 'prior'}'; vision verification "
                        f"score {v_score:.2f} meets the threshold "
                        f"{quality_threshold}."
                    )
                else:
                    self.logger.warning(
                        f"   ⚠️  Reused script verified low (score = "
                        f"{v_score:.2f} < {quality_threshold}). Keeping the "
                        f"result to preserve feature-schema consistency; "
                        f"flagging it as low-confidence."
                    )
                    message = (
                        f"Reused the locked analysis script from prior run "
                        f"'{reuse_source or 'prior'}', but vision "
                        f"verification scored {v_score:.2f}, below the "
                        f"threshold {quality_threshold}. The new image may "
                        f"not belong to this series, or imaging conditions "
                        f"shifted. Extracted features are schema-consistent "
                        f"but should be treated as low-confidence."
                    )
                reuse_result["reuse_validity"] = {
                    "reused": True,
                    "source": reuse_source,
                    "quality_score": v_score,
                    "threshold": quality_threshold,
                    "verdict": verdict,
                    "message": message,
                }
                reuse_result["quality_history"] = self._build_quality_history(
                    v_score, quality_threshold, [],
                    [verification] if verification else [], None,
                )
                if verdict == "poor":
                    reuse_result["quality_warning"] = message
                return reuse_result
            self.logger.warning(
                f"   ⚠️  Prior analysis script could not execute on this "
                f"image (even after correction). Falling back to full "
                f"pipeline re-derivation — the extracted-feature schema may "
                f"differ from the prior run."
            )

        # --- Initial analysis (skills at T=0 — guidance) ---
        state["_annealing_level"] = 0

        # --- Initial analysis ---
        initial_pipeline = state.get(
            'locked_analysis_config', {}
        ).get('processing_pipeline', 'Initial pipeline')
        self.logger.info(f"   Attempt 1: {initial_pipeline[:80]}...")

        result = self._process_single_image(
            state=state, image_data=image_data, data_path=data_path,
            image_name=image_name, image_idx=image_idx, base_script=None,
        )

        if result["success"]:
            # Initial score is provisional — LLM verification is authoritative
            score = 0.0  # Will be set by verification
            result["_quality_score"] = score
            all_attempts.append({
                "pipeline": initial_pipeline,
                "score": score,
                "result": result,
            })

            if score > best_score:
                best_score = score
                best_result = result
                best_config = state.get("locked_analysis_config", {}).copy()

            user_accepted = False

            # --- Verification loop (for anchor images) ---
            if _is_anchor:
                if not best_result or not best_result.get("success"):
                    self.logger.warning(
                        "   Initial analysis failed, skipping verification"
                    )
                else:
                    verification_attempts = []
                    analysis_was_approved = False
                    current_result = best_result  # track latest for verification

                    # Adaptive annealing state: start frozen, escalate
                    # only when the best score stalls for multiple iterations.
                    # Unlike curve fitting (deterministic R²), quality_score is
                    # LLM-assigned and noisy, so we use a patience counter
                    # instead of a rate-based formula.
                    _annealing_level = 0
                    _previous_annealing_level = 0
                    _stall_count = 0
                    _PATIENCE = 2
                    _prev_best_score = best_score
                    _n_anneal_levels = len(self._CONSTRAINT_ANNEALING_SCHEDULE)

                    for verification_iter in range(self.max_verification_iterations):
                        self.logger.info(
                            f"   Verification {verification_iter + 1}/"
                            f"{self.max_verification_iterations}"
                            f" (annealing level {_annealing_level})..."
                        )

                        verification = self._verify_quality(
                            state, current_result, history=verification_history,
                            verification_iter=verification_iter,
                            annealing_level=_annealing_level,
                        )

                        if verification is None:
                            self.logger.warning("   Verification failed, skipping")
                            break

                        # Extract score first
                        v_score = verification.get("quality_score", 0.0)
                        if not isinstance(v_score, (int, float)):
                            v_score = 0.0
                        current_result["_quality_score"] = v_score

                        # Update score in all_attempts for this result
                        if all_attempts:
                            all_attempts[-1]["score"] = v_score

                        # Track best result (high-water mark)
                        if v_score > best_score:
                            best_score = v_score
                            best_result = current_result
                            best_config = state.get("locked_analysis_config", {}).copy()

                        # Patience-based adaptive annealing: check if
                        # best_score improved since last iteration.
                        if best_score > _prev_best_score:
                            _stall_count = 0
                        else:
                            _stall_count += 1
                            if _stall_count >= _PATIENCE:
                                _annealing_level = min(
                                    _annealing_level + 1,
                                    _n_anneal_levels - 1,
                                )
                                _stall_count = 0
                                self.logger.info(
                                    f"   Annealing: {_PATIENCE} stalled "
                                    f"iterations, escalating to level "
                                    f"{_annealing_level}"
                                )
                        _prev_best_score = best_score

                        # Iteration-based floor: guarantees progression when
                        # noisy quality scores keep resetting the stall
                        # counter. Each level gets ~max_iter/n_levels slots
                        # before the floor forces the next one.
                        _floor = min(
                            verification_iter // 2,
                            _n_anneal_levels - 1,
                        )
                        if _floor > _annealing_level:
                            self.logger.info(
                                f"   Annealing: iteration floor lifting "
                                f"level {_annealing_level} -> {_floor}"
                            )
                            _annealing_level = _floor
                            _stall_count = 0

                        _cur_level = _annealing_level

                        verification_attempts.append({
                            "result": current_result.copy() if current_result else {},
                            "verification": verification,
                            "config": state.get("locked_analysis_config", {}).copy(),
                            "score": v_score,
                        })

                        verification_history.append({
                            "quality_score": v_score,
                            "config_used": state.get("locked_analysis_config", {}),
                            "issues_found": verification.get("issues_found", []),
                            "overall_assessment": verification.get(
                                "overall_assessment", ""
                            ),
                            "recommended_action": verification.get(
                                "recommended_action", ""
                            ),
                            "annealing_level": _cur_level,
                        })

                        # Log sub-scores if available
                        c = verification.get("completeness")
                        cr = verification.get("correctness")
                        r = verification.get("relevance")
                        if c is not None or cr is not None or r is not None:
                            self.logger.info(
                                f"   Sub-scores: completeness={c}, "
                                f"correctness={cr}, relevance={r}"
                            )

                        if best_score >= quality_threshold:
                            self.logger.info(
                                f"   Analysis approved (score = {best_score:.2f})"
                            )
                            analysis_was_approved = True
                            break

                        # Log issues and save last recommended action
                        self._log_verification_issues(verification)
                        if current_result:
                            current_result["_last_recommended_action"] = (
                                verification.get("recommended_action", "")
                            )

                        # Apply LLM's recommended fixes. Pass the
                        # accumulated verification_history so the refiner
                        # can see prior scores/pipelines and recognize
                        # regressions instead of iterating blindly on the
                        # previous (possibly degraded) config.
                        refined_config = self._apply_verification_feedback(
                            state, verification, history=verification_history
                        )

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

                        if refined_config == state.get("locked_analysis_config", {}):
                            # No changes at current temperature — escalate to
                            # give the LLM more freedom before giving up.
                            _annealing_level = min(
                                _annealing_level + 1, _n_anneal_levels - 1
                            )
                            if _annealing_level == _cur_level:
                                self.logger.info(
                                    "   No config changes at max annealing level, "
                                    "stopping verification"
                                )
                                break
                            self.logger.info(
                                f"   No config changes suggested, escalating "
                                f"to annealing level {_annealing_level}"
                            )
                            continue

                        # Clean up old visualization (but not the best result's viz)
                        old_viz_path = current_result.get("visualization_path")
                        if (old_viz_path and Path(old_viz_path).exists()
                                and current_result is not best_result):
                            try:
                                os.remove(old_viz_path)
                            except Exception:
                                pass

                        state["locked_analysis_config"] = refined_config

                        # Sync skill strictness with adaptive annealing level
                        state["_annealing_level"] = _annealing_level

                        # Carry the previous attempt's script forward so the
                        # code generator can adapt rather than regenerate
                        # from scratch — except on the single iteration
                        # where the annealing level escalates from < 2 to
                        # = 2 (hot). That escalation step deliberately
                        # gets fresh generation so the code generator can
                        # restructure without anchor bias from the
                        # structure that prompted the escalation.
                        _just_escalated_to_hot = (
                            _annealing_level >= 2
                            and _previous_annealing_level < 2
                        )
                        _refine_from_script = (
                            None if _just_escalated_to_hot
                            else (current_result or {}).get("script")
                        )

                        # Re-analyze with refined config
                        self.logger.info(
                            "   Re-analyzing with verification feedback..."
                        )
                        verified_result = self._process_single_image(
                            state=state, image_data=image_data,
                            data_path=data_path, image_name=image_name,
                            image_idx=image_idx, base_script=None,
                            refine_from_script=_refine_from_script,
                        )
                        _previous_annealing_level = _annealing_level

                        if verified_result["success"]:
                            verified_result["_quality_score"] = 0.0
                            self.logger.info(
                                "   Re-analysis complete, awaiting verification..."
                            )

                            # Track as latest result for next verification,
                            # but preserve best result/score separately
                            current_result = verified_result
                            all_attempts.append({
                                "pipeline": f"Verification-{verification_iter + 1}",
                                "score": 0.0,  # will be updated by next verification
                                "result": verified_result,
                            })
                        else:
                            self.logger.warning(
                                "   Re-analysis failed, stopping verification"
                            )
                            break

                    else:
                        # Loop exhausted without approval - verify final result
                        self.logger.info("   Verifying final re-analysis...")
                        final_verification = self._verify_quality(
                            state, current_result,
                            verification_iter=self.max_verification_iterations,
                            annealing_level=_annealing_level,
                        )

                        if final_verification:
                            v_score = final_verification.get(
                                "quality_score", 0.0
                            )
                            if isinstance(v_score, (int, float)):
                                current_result["_quality_score"] = v_score
                                if v_score > best_score:
                                    best_score = v_score
                                    best_result = current_result
                                    best_config = state.get("locked_analysis_config", {}).copy()

                            verification_attempts.append({
                                "result": current_result.copy() if current_result else {},
                                "verification": final_verification,
                                "config": state.get(
                                    "locked_analysis_config", {}
                                ).copy(),
                                "score": v_score,
                            })

                            if best_score >= quality_threshold:
                                self.logger.info(
                                    f"   Final analysis approved "
                                    f"(score = {best_score:.2f})"
                                )
                                analysis_was_approved = True
                            else:
                                self._log_verification_issues(final_verification)

                        # If verification loop exhausted without approval,
                        # keep best result and let alternative approaches try next.
                        # Judge is only called after all alternatives are exhausted.

                    # Restore config to match best result after verification loop
                    state["locked_analysis_config"] = best_config

            # --- Check if quality is acceptable ---
            if best_score >= quality_threshold:
                self.logger.info(
                    f"Quality score = {best_score:.2f} (meets threshold "
                    f"{quality_threshold})"
                )

                # In CO_PILOT mode, show anchor result and ask for approval
                if (
                    _is_anchor
                    and self.enable_human_feedback
                    and best_result.get("visualization_bytes")
                ):
                    user_feedback = self._get_user_feedback_on_result(
                        state, best_result, best_score
                    )
                    if user_feedback:
                        best_result, best_score = self._apply_user_feedback(
                            state, user_feedback, best_result, best_score,
                            image_data, data_path, image_name,
                            image_idx, all_attempts,
                        )

                best_result["quality_history"] = self._build_quality_history(
                    best_score, quality_threshold, all_attempts,
                    verification_history, judge_result,
                    best_result.get("script_errors"),
                )
                return best_result
            else:
                self.logger.warning(
                    f"Quality score = {best_score:.2f} (below threshold "
                    f"{quality_threshold})"
                )

        else:
            self.logger.error(
                f"   Initial analysis failed: "
                f"{result.get('error', 'Unknown')[:50]}"
            )
            all_attempts.append({
                "pipeline": initial_pipeline,
                "score": 0,
                "result": result,
            })

        # --- Human feedback for poor quality (if enabled) ---
        if self.enable_human_feedback and _is_anchor:
            feedback_result = self._get_human_feedback_for_poor_quality(
                state, best_result, all_attempts
            )

            if feedback_result and feedback_result.get("action") == "retry":
                refined_config = self._refine_config_from_feedback(
                    state, feedback_result["feedback"]
                )
                original_config = state.get("locked_analysis_config")
                state["locked_analysis_config"] = refined_config

                human_guided_result = self._process_single_image(
                    state=state, image_data=image_data, data_path=data_path,
                    image_name=image_name, image_idx=image_idx,
                    base_script=None,
                )

                if human_guided_result["success"]:
                    human_score = human_guided_result.get(
                        "quality_metrics", {}
                    ).get("quality_score", 0.5)
                    if isinstance(human_score, str):
                        try:
                            human_score = float(human_score)
                        except (ValueError, TypeError):
                            human_score = 0.5
                    human_guided_result["_quality_score"] = human_score
                    self.logger.info(
                        f"   Human-guided result: score = {human_score:.2f}"
                    )

                    if human_score > best_score:
                        best_score = human_score
                        best_result = human_guided_result
                        best_config = refined_config.copy()
                        if _is_anchor:
                            state["locked_analysis_config"] = refined_config
                    else:
                        state["locked_analysis_config"] = original_config
                else:
                    state["locked_analysis_config"] = original_config

        # --- Judge: select best from all attempts if still below threshold ---
        if best_score < quality_threshold and len(all_attempts) > 1:
            self.logger.info(
                "No result approved after all attempts - calling judge..."
            )
            judge_attempts = [
                {
                    "result": a.get("result", {}),
                    "verification": a.get("verification", {}),
                    "config": a.get("config", {}),
                    "score": a.get("score", 0),
                }
                for a in all_attempts
            ]
            judge_result = self._judge_select_best(judge_attempts)

            selected_index = judge_result.get("selected_index")
            if selected_index is not None:
                idx = selected_index
                selected = all_attempts[idx]
                best_result = selected["result"]
                best_score = selected["score"]
                if selected.get("config"):
                    best_config = selected["config"].copy()
                    state["locked_analysis_config"] = selected["config"]
                self.logger.info(
                    f"   Judge selected attempt {idx + 1} "
                    f"(score = {best_score:.2f})"
                )
                if judge_result.get("reasoning"):
                    best_result["judge_reasoning"] = judge_result[
                        "reasoning"
                    ][:500]

        # --- Summarize plan history ---
        multiple_attempts = len(all_attempts) > 1
        if multiple_attempts:
            summary_lines = []
            summary_lines.append(
                f"Original plan: {initial_pipeline[:100]}"
            )
            summary_lines.append(
                f"Original plan score: "
                f"{all_attempts[0].get('score', 'N/A')}"
            )
            for a in all_attempts[1:]:
                summary_lines.append(
                    f"{a.get('pipeline', 'N/A')[:80]} "
                    f"(score: {a.get('score', 'N/A')})"
                )
            summary_lines.append(
                "Selected: original plan (best available)"
            )

            plan_summary = "\n".join(summary_lines)

            print("\n" + "=" * 60)
            print("📋 REFINEMENT HISTORY")
            print("=" * 60)
            for line in summary_lines:
                print(f"   {line}")
            print("=" * 60)

        # --- Return best available result ---
        if best_result:
            best_result["quality_warning"] = (
                f"Quality score = {best_score:.2f} below threshold "
                f"{quality_threshold}"
            )
            best_result["attempted_pipelines"] = [
                a["pipeline"] for a in all_attempts
            ]
            if multiple_attempts:
                best_result["plan_deviation_summary"] = plan_summary
            self.logger.warning(
                f"Proceeding with best available result "
                f"(score = {best_score:.2f})"
            )

            if _is_anchor:
                state["locked_analysis_config"] = best_config

            best_result["quality_history"] = self._build_quality_history(
                best_score, quality_threshold, all_attempts,
                verification_history, judge_result,
                best_result.get("script_errors"),
            )
            return best_result
        else:
            return {
                "index": image_idx,
                "name": image_name,
                "success": False,
                "error": "All analysis attempts failed",
                "attempts": len(all_attempts),
                "extracted_features": {},
                "quality_metrics": {},
            }

    def _log_verification_issues(self, verification: dict) -> None:
        """Log verification issues in a readable format."""
        issues_count = len(verification.get("issues_found", []))
        overall_assessment = verification.get(
            "overall_assessment", "No assessment provided"
        )

        self.logger.info(f"   Found {issues_count} issue(s)")
        self.logger.info("")
        self.logger.info("   Assessment:")
        for line in self._wrap_text(overall_assessment, width=70):
            self.logger.info(f"      {line}")

        if verification.get("issues_found"):
            self.logger.info("")
            self.logger.info("   Issues:")

            for i, issue in enumerate(verification.get("issues_found", []), 1):
                location = issue.get('location', 'Unknown')
                problem = issue.get('problem', 'No description')
                suggested_fix = issue.get('suggested_fix', '')

                self.logger.info("")
                self.logger.info(f"   [{i}] {location}")

                problem_lines = self._wrap_text(problem, width=65)
                self.logger.info(f"       Problem: {problem_lines[0]}")
                for line in problem_lines[1:]:
                    self.logger.info(f"                {line}")

                if suggested_fix:
                    fix_lines = self._wrap_text(suggested_fix, width=65)
                    self.logger.info(f"       Fix: {fix_lines[0]}")
                    for line in fix_lines[1:]:
                        self.logger.info(f"            {line}")

        recommended = verification.get("recommended_action", "")
        if recommended and recommended.lower() != "none":
            self.logger.info("")
            self.logger.info("   Recommended action:")
            for line in self._wrap_text(recommended, width=65):
                self.logger.info(f"      {line}")

        self.logger.info("")

    @staticmethod
    def _build_quality_history(
        best_score: float,
        quality_threshold: float,
        all_attempts: list,
        verification_history: list,
        judge_result: dict | None,
        script_errors: list | None = None,
    ) -> dict:
        """Build a compact quality history dict for the best result.

        Captures problem→solution pairs at every level: script errors,
        verification iterations, alternative approaches, and judge reasoning.
        """
        return {
            "final_score": best_score,
            "threshold": quality_threshold,
            "approved": best_score >= quality_threshold,
            "verification_iterations": [
                {
                    "score": entry["quality_score"],
                    "annealing_level": entry.get("annealing_level", 0),
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
            "script_errors": script_errors or [],
            "judge_reasoning": (
                (judge_result or {}).get("reasoning")
            ),
        }

    USER_FEEDBACK_SCRIPT_PROMPT = '''You have a working image analysis script and user feedback requesting changes.

**USER FEEDBACK:** {user_feedback}

**CURRENT WORKING SCRIPT:**
```python
{current_script}
```

**CURRENT ANALYSIS CONFIG:**
- Approach: {analysis_approach}
- Pipeline: {processing_pipeline}

Decide how to address the user's feedback and classify the change:
- **cosmetic** — visualization, colormaps, labels, fonts, legend, axis ranges, \
subplot layout, or other display-only changes. Modify the plotting sections \
only; keep all analysis logic and extracted values untouched.
- **analytical** — parameter / threshold / filter changes, or any change that \
affects the extracted values. Modify the relevant parts of the script.
- **rewrite** — the feedback demands a fundamentally different analysis \
approach. Write a new script from scratch.

In all cases, preserve the output contract: the script must print \
'IMAGE_ANALYSIS_RESULTS_JSON:{{...}}' and save a visualization PNG.

Return JSON: {{"change_type": "cosmetic" | "analytical" | "rewrite", \
"diagnosis": "what you changed and why", "script": "full Python script"}}
'''

    def _apply_user_feedback(
        self,
        state: dict,
        user_feedback: str,
        best_result: dict,
        best_score: float,
        image_data: np.ndarray,
        data_path: str,
        image_name: str,
        image_idx: int,
        all_attempts: list,
    ) -> tuple:
        """Apply user feedback to refine the analysis.

        Gives the LLM the existing script and lets it decide whether to
        patch it or rewrite from scratch based on the feedback.

        Returns:
            Tuple of (best_result, best_score) after applying feedback
        """
        existing_script = best_result.get("script", "")
        original_config = state.get("locked_analysis_config", {})

        if existing_script:
            # Let the LLM decide: patch existing script or rewrite
            self.logger.info("   Applying user feedback to existing script...")
            config = state.get("locked_analysis_config", {})
            prompt = self.USER_FEEDBACK_SCRIPT_PROMPT.format(
                user_feedback=user_feedback,
                current_script=existing_script,
                analysis_approach=config.get("analysis_approach", ""),
                processing_pipeline=config.get("processing_pipeline", ""),
            )

            try:
                response = self.model.generate_content(prompt)
                result, error = self._parse(response)

                if not error and result and result.get("script"):
                    diagnosis = result.get("diagnosis", "")
                    change_type = (
                        result.get("change_type", "").strip().lower()
                    )
                    if diagnosis:
                        self.logger.info(f"    Diagnosis: {diagnosis}")
                    if change_type:
                        self.logger.info(f"    Change type: {change_type}")
                    patched_script = result["script"]

                    # Execute the patched script VERBATIM in the per-image dir
                    working_dir = self.output_dir / f"image_{image_idx:04d}"
                    canonical_viz = working_dir / VIZ_NAME
                    patched_script = self._sanitize_script(patched_script)
                    run = stage_and_run(self.executor, patched_script, image_data, working_dir)
                    exec_result = run["exec"]

                    if run["status"] == "success":
                        analysis_results = {}
                        for line in run["stdout"].splitlines():
                            if line.startswith("IMAGE_ANALYSIS_RESULTS_JSON:"):
                                try:
                                    analysis_results = json.loads(
                                        line.replace(
                                            "IMAGE_ANALYSIS_RESULTS_JSON:", ""
                                        ).strip()
                                    )
                                except json.JSONDecodeError:
                                    pass
                                break

                        viz_bytes = run["visualization_bytes"]

                        if analysis_results and viz_bytes:
                            user_guided_result = {
                                "index": image_idx,
                                "name": image_name,
                                "data_path": data_path,
                                "success": True,
                                "error": None,
                                "analysis_type": analysis_results.get(
                                    "analysis_type"
                                ),
                                "extracted_features": analysis_results.get(
                                    "extracted_features", {}
                                ),
                                "quality_metrics": analysis_results.get(
                                    "quality_metrics", {}
                                ),
                                "summary": analysis_results.get("summary"),
                                "visualization_bytes": viz_bytes,
                                "visualization_path": str(canonical_viz),
                                "script": patched_script,
                                "script_errors": [],
                            }

                            # Cosmetic-only changes: the analysis outputs did
                            # not change, so skip re-verification and keep the
                            # previous best_score. Accept the patched script
                            # and new visualization as the final result.
                            if change_type == "cosmetic":
                                user_guided_result["_quality_score"] = best_score
                                self.logger.info(
                                    "   Cosmetic change applied — skipping "
                                    "re-verification, keeping existing score "
                                    f"({best_score:.2f})"
                                )
                                all_attempts.append({
                                    "pipeline": "User-guided (cosmetic)",
                                    "score": best_score,
                                    "result": user_guided_result,
                                })
                                return user_guided_result, best_score

                            # Analytical / rewrite changes: verify quality.
                            verification = self._verify_quality(
                                state, user_guided_result
                            )
                            if verification:
                                user_score = verification.get(
                                    "quality_score", 0.5
                                )
                                if not isinstance(user_score, (int, float)):
                                    user_score = 0.5
                            else:
                                user_score = user_guided_result.get(
                                    "quality_metrics", {}
                                ).get("quality_score", 0.5)
                                if isinstance(user_score, str):
                                    try:
                                        user_score = float(user_score)
                                    except (ValueError, TypeError):
                                        user_score = 0.5

                            user_guided_result["_quality_score"] = user_score
                            self.logger.info(
                                f"   User-guided result: score = {user_score:.2f}"
                            )
                            all_attempts.append({
                                "pipeline": "User-guided",
                                "score": user_score,
                                "result": user_guided_result,
                            })

                            if user_score >= best_score:
                                return user_guided_result, user_score
                            else:
                                if viz_bytes:
                                    review_path = (
                                        self.output_dir
                                        / "first_image_analysis_review.png"
                                    )
                                    with open(review_path, "wb") as f:
                                        f.write(viz_bytes)
                                keep = self._ask_keep_user_guided_result(
                                    user_score, best_score
                                )
                                if keep:
                                    return user_guided_result, user_score
                                # Restore original viz on disk so the
                                # file matches what best_result describes
                                if best_result.get("visualization_bytes"):
                                    try:
                                        with open(canonical_viz, "wb") as f:
                                            f.write(
                                                best_result[
                                                    "visualization_bytes"
                                                ]
                                            )
                                    except Exception:
                                        pass
                                return best_result, best_score

                    self.logger.warning(
                        "   Patched script failed, falling back to full re-analysis"
                    )
            except Exception as e:
                self.logger.warning(
                    f"   Script patching failed ({e}), falling back to full re-analysis"
                )

        # Fallback: full re-analysis (original behavior)
        refined_config = self._refine_config_from_feedback(state, user_feedback)
        state["locked_analysis_config"] = refined_config

        # Clean up old visualization
        old_viz_path = best_result.get("visualization_path")
        if old_viz_path and Path(old_viz_path).exists():
            try:
                os.remove(old_viz_path)
            except Exception:
                pass

        self.logger.info("   Re-analyzing with user feedback...")
        user_guided_result = self._process_single_image(
            state=state, image_data=image_data, data_path=data_path,
            image_name=image_name, image_idx=image_idx, base_script=None,
        )

        if user_guided_result["success"]:
            verification = self._verify_quality(state, user_guided_result)
            if verification:
                user_score = verification.get("quality_score", 0.5)
                if not isinstance(user_score, (int, float)):
                    user_score = 0.5
            else:
                user_score = user_guided_result.get(
                    "quality_metrics", {}
                ).get("quality_score", 0.5)
                if isinstance(user_score, str):
                    try:
                        user_score = float(user_score)
                    except (ValueError, TypeError):
                        user_score = 0.5
            user_guided_result["_quality_score"] = user_score
            self.logger.info(f"   User-guided result: score = {user_score:.2f}")
            all_attempts.append({
                "pipeline": "User-guided",
                "score": user_score,
                "result": user_guided_result,
            })

            if user_score >= best_score:
                return user_guided_result, user_score
            else:
                if user_guided_result.get("visualization_bytes"):
                    review_viz_path = (
                        self.output_dir / "first_image_analysis_review.png"
                    )
                    with open(review_viz_path, "wb") as f:
                        f.write(user_guided_result["visualization_bytes"])
                keep_user = self._ask_keep_user_guided_result(
                    user_score, best_score
                )
                if keep_user:
                    return user_guided_result, user_score
                else:
                    # Restore original viz on disk so the file matches
                    # what best_result describes (the re-analysis
                    # overwrote or deleted the original file).
                    if best_result.get("visualization_bytes") and best_result.get("visualization_path"):
                        try:
                            with open(best_result["visualization_path"], "wb") as f:
                                f.write(best_result["visualization_bytes"])
                        except Exception:
                            pass
                    state["locked_analysis_config"] = original_config
                    return best_result, best_score
        else:
            self.logger.warning(
                "   User-guided analysis failed, keeping previous"
            )
            # Restore original viz on disk (re-analysis deleted it)
            if best_result.get("visualization_bytes") and best_result.get("visualization_path"):
                try:
                    with open(best_result["visualization_path"], "wb") as f:
                        f.write(best_result["visualization_bytes"])
                except Exception:
                    pass
            state["locked_analysis_config"] = original_config
            return best_result, best_score

    def _detect_outliers(self, series_results: List[dict]) -> List[dict]:
        """Detect statistical outliers in extracted feature values across a series.

        For each numeric feature, compute mean/std across the series and flag
        images where any feature deviates more than outlier_sigma standard
        deviations from the mean.
        """
        # Collect all numeric features across successful results
        successful = [r for r in series_results if r["success"]]
        if len(successful) < 3:
            return []

        # Gather all numeric feature keys
        all_feature_keys = set()
        for r in successful:
            features = r.get("extracted_features", {})
            for k, v in features.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_feature_keys.add(k)

        if not all_feature_keys:
            # Fallback: use quality_score if available
            scores = []
            for r in successful:
                score = r.get("quality_metrics", {}).get("quality_score")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        pass

            if len(scores) < 3:
                return []

            score_array = np.array(scores)
            mean_score = np.mean(score_array)
            std_score = np.std(score_array)

            flagged = []
            score_idx = 0
            for r in series_results:
                if not r["success"]:
                    flagged.append({
                        "index": r["index"],
                        "name": r["name"],
                        "reason": "analysis_failed",
                        "details": "Analysis script failed to execute",
                        "recommendation": (
                            "Check image quality and consider manual inspection. "
                            "The analysis script failed to execute successfully."
                        ),
                    })
                    continue

                score = r.get("quality_metrics", {}).get("quality_score")
                if score is None:
                    continue
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    continue

                if std_score > 0.001:
                    deviation = (mean_score - score) / std_score
                    is_outlier = deviation > self.outlier_sigma
                else:
                    deviation = 0
                    is_outlier = False

                if is_outlier:
                    flagged.append({
                        "index": r["index"],
                        "name": r["name"],
                        "reason": "statistical_outlier",
                        "details": (
                            f"Quality score {score:.2f} deviates "
                            f"{deviation:.1f} sigma from mean "
                            f"{mean_score:.2f}"
                        ),
                        "deviation_sigma": float(deviation),
                        "recommendation": (
                            "Analysis quality significantly worse than series "
                            "average. Possible causes: structural change, "
                            "imaging artifact, or pipeline mismatch."
                        ),
                    })

            return flagged

        # Build per-feature arrays
        feature_values = {k: [] for k in all_feature_keys}
        feature_indices = {k: [] for k in all_feature_keys}
        for r in successful:
            features = r.get("extracted_features", {})
            for k in all_feature_keys:
                v = features.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    feature_values[k].append(float(v))
                    feature_indices[k].append(r["index"])

        # Compute per-feature stats
        feature_stats = {}
        for k in all_feature_keys:
            vals = np.array(feature_values[k])
            if len(vals) >= 3:
                feature_stats[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

        # Flag outliers
        flagged = []
        flagged_set = set()

        for r in series_results:
            if not r["success"]:
                flagged.append({
                    "index": r["index"],
                    "name": r["name"],
                    "reason": "analysis_failed",
                    "details": "Analysis script failed to execute",
                    "recommendation": (
                        "Check image quality and consider manual inspection. "
                        "The analysis script failed to execute successfully."
                    ),
                })
                flagged_set.add(r["index"])
                continue

            features = r.get("extracted_features", {})
            outlier_reasons = []

            for k, stats in feature_stats.items():
                v = features.get(k)
                if v is None or not isinstance(v, (int, float)):
                    continue
                mean = stats["mean"]
                std = stats["std"]
                if std > 1e-9:
                    deviation = abs(float(v) - mean) / std
                    if deviation > self.outlier_sigma:
                        outlier_reasons.append(
                            f"{k}={float(v):.4g} "
                            f"({deviation:.1f}sigma from mean {mean:.4g})"
                        )

            if outlier_reasons and r["index"] not in flagged_set:
                flagged.append({
                    "index": r["index"],
                    "name": r["name"],
                    "reason": "statistical_outlier",
                    "details": "; ".join(outlier_reasons),
                    "deviation_sigma": None,
                    "recommendation": (
                        "Feature values significantly different from series "
                        "average. Possible causes: structural transition, "
                        "phase change, imaging artifact, or analysis pipeline "
                        "mismatch. May indicate interesting physics."
                    ),
                })
                flagged_set.add(r["index"])

        return flagged

    def _generate_outlier_report(
        self, flagged: List[dict], series_results: List[dict]
    ) -> str:
        """Generate a human-readable outlier report."""
        if not flagged:
            return ""

        lines = [
            "",
            "=" * 60,
            "FLAGGED IMAGES - REQUIRE ATTENTION",
            "=" * 60,
            "",
        ]

        total = len(series_results)
        successful = sum(1 for r in series_results if r["success"])

        lines.append(f"Series statistics: {successful}/{total} successful analyses")
        lines.append(
            f"Outlier detection: {self.outlier_sigma} sigma threshold"
        )
        lines.append("")

        by_reason = {}
        for f in flagged:
            reason = f["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)

        reason_labels = {
            "analysis_failed": "Failed Analyses",
            "statistical_outlier": "Statistical Outliers (possible interesting physics)",
            "below_threshold": "Below Quality Threshold",
            "outlier_and_below_threshold": "Critical: Outlier + Below Threshold",
        }

        for reason, items in by_reason.items():
            lines.append(
                f"\n{reason_labels.get(reason, reason)} ({len(items)} images):"
            )
            lines.append("-" * 50)

            for f in items:
                lines.append(f"  - {f['name']} (index {f['index']})")
                if f.get("details"):
                    lines.append(f"    {f['details']}")
                if f.get("deviation_sigma") is not None:
                    lines.append(
                        f"    Deviation: {f['deviation_sigma']:.1f} sigma"
                    )
                lines.append(f"    -> {f['recommendation']}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_config_for_image(self, state: dict, idx: int) -> dict:
        """Return the analysis config for a given image index.

        If regime_configs is present, return the regime-specific config.
        Otherwise, return the single locked_analysis_config.
        """
        regime_configs = state.get("regime_configs")
        if regime_configs and idx in regime_configs:
            return regime_configs[idx]
        return state.get("locked_analysis_config", {})

    def _get_regime_for_image(self, state: dict, idx: int) -> Optional[str]:
        """Return the regime name for a given image index, or None."""
        series_plan = state.get("series_analysis_plan")
        if not series_plan:
            return None
        for regime in series_plan.get("regimes", []):
            if idx in regime.get("image_indices", []):
                return regime.get("name", "unnamed")
        return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", True)

        mode_str = "SINGLE IMAGE" if is_single else f"SERIES ({num_images} images)"
        self.logger.info("")
        self.logger.info(f"IMAGE ANALYSIS: {mode_str}")
        if not is_single:
            self.logger.info(
                f"   Outlier detection: {self.outlier_sigma} sigma"
            )

        image_paths = state.get("image_paths", [])
        image_stack = state.get("image_stack")

        # Determine regime structure for per-regime execution
        series_plan = state.get("series_analysis_plan")
        regime_configs = state.get("regime_configs")

        if series_plan and regime_configs:
            first_in_regime: set = set()
            for regime in series_plan.get("regimes", []):
                indices = sorted(regime.get("image_indices", []))
                if indices:
                    first_in_regime.add(indices[0])
            self.logger.info(
                f"   Regimes: {len(series_plan.get('regimes', []))}"
            )
            self.logger.info(
                f"   First-in-regime images (full QC): "
                f"{sorted(first_in_regime)}"
            )
        else:
            first_in_regime = {0}

        # #172: locked-script reuse. When prior_analysis_paths supplies an
        # earlier image-analysis run, the image-0 anchor reuses that run's
        # saved analysis script instead of re-deriving the pipeline. Skipped
        # for multi-regime runs (a single prior script has no regime mapping).
        reuse_script, reuse_source = _first_prior_image_script(state)
        if reuse_script and regime_configs:
            self.logger.info(
                "   ♻️  Prior image-analysis run supplied, but this run is "
                "multi-regime — locked-script reuse skipped."
            )
            reuse_script, reuse_source = None, None
        elif reuse_script:
            self.logger.info(
                f"   ♻️  Prior image-analysis run '{reuse_source}' supplied "
                f"— the anchor will reuse its locked analysis script (#172)."
            )

        series_results = []
        base_scripts: Dict[str, str] = {}
        original_locked_config = state.get("locked_analysis_config", {})
        if original_locked_config:
            original_locked_config = original_locked_config.copy()

        for idx in range(num_images):
            if image_stack is not None:
                image_data = image_stack[idx]
                image_name = f"image_{idx:04d}"
                data_path = f"stack_index_{idx}"
            else:
                data_path = image_paths[idx]
                image_name = Path(data_path).stem
                image_data = self._load_image_data(data_path)

            # Determine regime and set appropriate config
            regime_name = self._get_regime_for_image(state, idx) or "default"
            image_config = self._get_config_for_image(state, idx)

            # Temporarily set the config for this image
            state["locked_analysis_config"] = image_config
            state["_current_regime_name"] = regime_name if regime_configs else None

            if is_single:
                self.logger.info(f"Analyzing: {image_name}")
            elif regime_configs:
                self.logger.info(
                    f"[{idx + 1}/{num_images}] Analyzing: {image_name} "
                    f"(regime: {regime_name})"
                )
            else:
                self.logger.info(
                    f"[{idx + 1}/{num_images}] Analyzing: {image_name}"
                )

            if idx in first_in_regime:
                if regime_configs and idx != 0:
                    self.logger.info(
                        f"  First in regime '{regime_name}' - full quality control"
                    )

                # For regime anchors that aren't image 0, temporarily swap
                # original_image_bytes and image_statistics
                _saved_original_bytes = None
                _saved_image_statistics = None
                if idx != 0 and idx in first_in_regime:
                    _saved_original_bytes = state.get("original_image_bytes")
                    _saved_image_statistics = state.get("image_statistics")
                    try:
                        anchor_bytes = self.image_to_bytes_fn(image_data)
                        state["original_image_bytes"] = anchor_bytes
                    except Exception:
                        pass
                    state["image_statistics"] = compute_image_statistics(
                        image_data
                    )

                result = self._execute_and_verify(
                    state=state, image_data=image_data, data_path=data_path,
                    image_name=image_name, image_idx=idx,
                    is_regime_anchor=(idx != 0 and idx in first_in_regime),
                    reuse_script=(reuse_script if idx == 0 else None),
                    reuse_source=(reuse_source if idx == 0 else None),
                )

                # Restore original state
                if _saved_original_bytes is not None:
                    state["original_image_bytes"] = _saved_original_bytes
                if _saved_image_statistics is not None:
                    state["image_statistics"] = _saved_image_statistics

                # #172: reuse was attempted for the anchor but the result
                # carries no reuse_validity verdict -> the prior script could
                # not execute and full QC re-derived the pipeline. Record the
                # schema-drift caveat so the orchestrator can react.
                if idx == 0 and reuse_script and not result.get("reuse_validity"):
                    result["reuse_validity"] = {
                        "reused": False,
                        "source": reuse_source,
                        "verdict": "script_failed",
                        "message": (
                            f"The locked analysis script from prior run "
                            f"'{reuse_source or 'prior'}' could not execute "
                            f"on this image; the pipeline was re-derived from "
                            f"scratch and the extracted-feature schema may "
                            f"differ from the prior run."
                        ),
                    }
                    result["quality_warning"] = result["reuse_validity"]["message"]

                if result["success"] and result.get("script"):
                    base_scripts[regime_name] = result["script"]
                    if idx == 0:
                        state["base_analysis_script"] = result["script"]
                    self.logger.info(
                        f"Base analysis script locked for regime "
                        f"'{regime_name}'."
                    )

                    # If QC changed the config, propagate to all images
                    # in this regime
                    updated_config = state.get(
                        "locked_analysis_config", image_config
                    )
                    if regime_configs and updated_config != image_config:
                        for r_regime in (series_plan or {}).get("regimes", []):
                            if idx in r_regime.get("image_indices", []):
                                for other_idx in r_regime["image_indices"]:
                                    regime_configs[other_idx] = updated_config
                                break
            else:
                base_script = base_scripts.get(regime_name)
                result = self._process_single_image(
                    state=state, image_data=image_data, data_path=data_path,
                    image_name=image_name, image_idx=idx,
                    base_script=base_script,
                )

            # Tag result with regime info
            if regime_configs:
                result["regime"] = regime_name

            series_results.append(result)

            if result["success"]:
                analysis_type = result.get("analysis_type", "Analysis")
                self.logger.info(f"  {analysis_type} - complete")
            else:
                self.logger.error(
                    f"  Failed: {result.get('error', 'Unknown')[:50]}"
                )

        # Restore original locked config
        if original_locked_config:
            state["locked_analysis_config"] = original_locked_config

        # Outlier detection for series
        flagged_images = []
        if num_images > 1:
            flagged_images = self._detect_outliers(series_results)

            if flagged_images:
                report = self._generate_outlier_report(
                    flagged_images, series_results
                )
                self.logger.warning(report)

                flagged_indices = {f["index"] for f in flagged_images}
                for r in series_results:
                    if r["index"] in flagged_indices:
                        flag_info = next(
                            f for f in flagged_images
                            if f["index"] == r["index"]
                        )
                        r["flagged"] = True
                        r["flag_reason"] = flag_info["reason"]
                        r["flag_recommendation"] = flag_info["recommendation"]
                        r["flag_details"] = flag_info.get("details")

                flagged_report_path = self.output_dir / "flagged_images.json"
                with open(flagged_report_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "outlier_sigma": self.outlier_sigma,
                        "total_images": num_images,
                        "flagged_count": len(flagged_images),
                        "flagged_images": flagged_images,
                    }, f, indent=2)

                state["flagged_images_path"] = str(flagged_report_path)

        state["series_results"] = series_results
        state["flagged_images"] = flagged_images

        if is_single and series_results and series_results[0]["success"]:
            first_result = series_results[0]
            state["analysis_result"] = {
                "analysis_type": first_result.get("analysis_type"),
                "extracted_features": first_result.get(
                    "extracted_features", {}
                ),
                "quality_metrics": first_result.get("quality_metrics", {}),
                "summary": first_result.get("summary"),
                "saved_arrays": first_result.get("saved_arrays", {}),
                "quality_history": first_result.get("quality_history"),
            }
            state["final_script"] = first_result.get("script")
            state["final_viz_bytes"] = first_result.get("visualization_bytes")

            if first_result.get("visualization_bytes"):
                state["analysis_images"].append({
                    "label": first_result.get("analysis_type", "Analysis"),
                    "data": first_result["visualization_bytes"],
                })

        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_images)

        self.logger.info("")
        self.logger.info(
            f"Analysis complete: {successful}/{num_images} successful"
        )
        if flagged_count > 0:
            self.logger.warning(
                f"{flagged_count} images flagged for review"
            )

        # Save series results
        results_path = self.output_dir / "series_analysis_results.json"
        with open(results_path, 'w') as f:
            serializable_results = []
            for r in series_results:
                r_copy = {
                    k: v for k, v in r.items()
                    if k not in (
                        "visualization_bytes", "_winning_config",
                        "_quality_score", "_quality_issues",
                    )
                }
                serializable_results.append(r_copy)

            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": num_images,
                "successful": successful,
                "flagged_count": flagged_count,
                "is_single_image": is_single,
                "series_metadata": state.get("series_metadata", {}),
                "quality_settings": {
                    "outlier_sigma": self.outlier_sigma,
                },
                "locked_config": state.get("locked_analysis_config"),
                "series_analysis_plan": state.get("series_analysis_plan"),
                "results": serializable_results,
            }, f, indent=2, default=str)

        state["series_results_path"] = str(results_path)

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

    def _judge_select_best(self, attempts: List[dict]) -> dict:
        """Present all verification attempts to a judge LLM to select the best one.

        Called when the verification loop exhausts without any result being approved.
        """
        self.logger.info("")
        self.logger.info(
            "No result approved after verification loop - calling judge..."
        )

        attempts_summary = []
        for i, attempt in enumerate(attempts):
            score = attempt.get("score", 0)
            pipeline = attempt["config"].get("processing_pipeline", "Unknown")
            verification = attempt.get("verification", {})
            assessment = verification.get(
                "overall_assessment", "No assessment available"
            )
            issues = verification.get("issues_found", [])

            issues_brief = []
            for issue in issues[:3]:
                issues_brief.append(
                    f"  - {issue.get('location', '?')}: "
                    f"{issue.get('problem', '?')}"
                )
            issues_str = (
                "\n".join(issues_brief)
                if issues_brief
                else "  (no specific issues listed)"
            )

            summary = f"""
    **Attempt {i + 1}:**
    - Pipeline: {pipeline}
    - Quality score = {score:.2f}
    - Assessment: {assessment}
    - Issues ({len(issues)} found):
    {issues_str}
    """
            attempts_summary.append(summary)

        prompt_parts = [
            self.JUDGE_PROMPT.format(
                attempts_summary="\n".join(attempts_summary)
            )
        ]

        # Add all visualizations
        for i, attempt in enumerate(attempts):
            viz_bytes = attempt["result"].get("visualization_bytes")
            if viz_bytes:
                prompt_parts.append(
                    f"\n\n**Attempt {i + 1} Visualization:**"
                )
                prompt_parts.append({
                    "mime_type": "image/png",
                    "data": viz_bytes,
                })

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)

            if error or not result:
                self.logger.warning(
                    f"   Judge failed to parse response: {error}"
                )
                return {
                    "selected_index": None,
                    "acceptable": False,
                    "reasoning": f"Judge parse failed: {error}",
                }

            selected = result.get("selected_index")
            acceptable = result.get("acceptable", False)
            reasoning = result.get("reasoning", "No reasoning provided")

            if acceptable and selected is not None:
                self.logger.info(f"   Judge selected attempt {selected + 1}")
            else:
                self.logger.warning("   Judge found no acceptable result")

            self.logger.info("   Reasoning:")
            for line in self._wrap_text(reasoning, width=70):
                self.logger.info(f"      {line}")

            return result

        except Exception as e:
            self.logger.error(f"   Judge call failed: {e}")
            return {
                "selected_index": None,
                "acceptable": False,
                "reasoning": f"Judge call failed: {str(e)}",
            }


class ImageAdaptiveRefitController:
    """
    Post-processing recovery step that re-analyzes flagged images independently.

    After the locked-config series processing completes, this controller:
    1. Identifies images flagged for quality reasons (below_threshold, analysis_failed)
    2. Re-runs each one with full LLM planning + pipeline selection + verification
    3. Updates series_results with improved analyses where possible
    4. Re-runs outlier detection on updated results

    Statistical outliers (reason="statistical_outlier") are NOT re-analyzed,
    because their anomalous features may reflect genuine physical phenomena
    rather than pipeline inadequacy.
    """

    REFIT_REASONS = frozenset({
        "below_threshold", "analysis_failed", "outlier_and_below_threshold"
    })

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
        image_to_bytes_fn: Callable,
        max_verification_iterations: int = 7,
        enable_human_feedback: bool = False,
        conformance_instructions: str = "",
        refinement_instructions: str = "",
    ):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.image_to_bytes_fn = image_to_bytes_fn
        self.enable_human_feedback = enable_human_feedback

        # Compose a processing helper to reuse _execute_and_verify
        self._processing_helper = UnifiedImageProcessingController(
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
            image_to_bytes_fn=image_to_bytes_fn,
            enable_human_feedback=False,
            max_verification_iterations=max_verification_iterations,
            conformance_instructions=conformance_instructions,
            refinement_instructions=refinement_instructions,
        )

    def _load_image(self, idx, image_paths, image_stack):
        """Load image data for re-analysis."""
        if image_stack is not None:
            return image_stack[idx]
        if image_paths and idx < len(image_paths):
            try:
                return self._processing_helper._load_image_data(
                    image_paths[idx]
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load {image_paths[idx]}: {e}"
                )
                return None
        return None

    def _build_refit_state(self, state, image_data, idx, name):
        """Build a temporary state dict for independent re-analysis."""
        locked_config = state.get("locked_analysis_config", {})
        original_result = state["series_results"][idx]

        system_info = state.get("system_info", {})
        series_metadata = state.get("series_metadata", {})
        num_images = state.get("num_images", 0)

        exp_context_parts = []
        if system_info:
            exp_context_parts.append(
                json.dumps(system_info, indent=2, default=str)
            )
        if series_metadata.get("variable") and series_metadata.get("values"):
            values = series_metadata["values"]
            units = series_metadata.get("units", "")
            if idx < len(values):
                exp_context_parts.append(
                    f"Series position: image {idx + 1}/{num_images}, "
                    f"{series_metadata['variable']} = {values[idx]} {units}"
                )
        exp_context = "\n".join(exp_context_parts)

        # Summarize series context
        series_results = state.get("series_results", [])
        series_context_parts = []
        successful = [
            r for r in series_results
            if r.get("success") and not r.get("flagged")
        ]
        if successful:
            series_context_parts.append(
                f"Successful analyses (locked pipeline): "
                f"{len(successful)}/{len(series_results)} images, "
                f"pipeline: {successful[0].get('analysis_type', 'N/A')}"
            )
        flagged = [
            r for r in series_results
            if r.get("flagged") or not r.get("success")
        ]
        if flagged:
            flagged_indices = [str(r["index"]) for r in flagged]
            series_context_parts.append(
                f"Failed image indices: [{', '.join(flagged_indices)}]"
            )
        # Nearest successful neighbor summary
        for offset in (-1, 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(series_results):
                nr = series_results[neighbor_idx]
                if nr.get("success") and not nr.get("flagged"):
                    series_context_parts.append(
                        f"Neighbor image [{neighbor_idx}] analyzed "
                        f"successfully: "
                        f"type={nr.get('analysis_type', 'N/A')}"
                    )
        series_context = "\n".join(series_context_parts)

        refit_context = (
            f"This image was previously analyzed using the locked series "
            f"pipeline but produced inadequate results.\n\n"
            f"The locked pipeline was: "
            f"{locked_config.get('processing_pipeline', 'Unknown')}\n"
            f"The locked approach was: "
            f"{locked_config.get('analysis_approach', 'Unknown')}\n\n"
        )

        # Add regime context if available
        series_plan = state.get("series_analysis_plan")
        if series_plan:
            for regime in series_plan.get("regimes", []):
                if idx in regime.get("image_indices", []):
                    refit_context += (
                        f"**Regime context:** This image was assigned to regime "
                        f"'{regime.get('name', 'unnamed')}' with expected pipeline: "
                        f"{regime.get('processing_pipeline', 'Unknown')}.\n\n"
                    )
                    break

        if exp_context:
            refit_context += (
                f"**Experimental context:**\n{exp_context}\n\n"
            )
        if series_context:
            refit_context += (
                f"**Series context:**\n{series_context}\n\n"
            )
        refit_context += (
            f"IMPORTANT: The locked pipeline failed for this specific image. "
            f"You MUST try a DIFFERENT analysis approach. Consider:\n"
            f"1. Different preprocessing (different denoising, different "
            f"contrast enhancement)\n"
            f"2. Different segmentation method (the locked method's "
            f"thresholding may not work for this image)\n"
            f"3. Different feature extraction approach\n\n"
            f"Do NOT simply retry the same pipeline with different parameters."
        )

        fresh_config = {
            "analysis_approach": refit_context,
            "processing_pipeline": (
                f"Alternative to: "
                f"{locked_config.get('processing_pipeline', 'Unknown')}"
            ),
            "features_to_extract": locked_config.get(
                "features_to_extract", []
            ),
            "quality_criteria": locked_config.get(
                "quality_criteria", "Visual inspection"
            ),
            "expected_outputs": locked_config.get("expected_outputs", []),
        }

        image_paths = state.get("image_paths", [])
        data_path = (
            image_paths[idx]
            if image_paths and idx < len(image_paths)
            else name
        )

        stats = compute_image_statistics(image_data)
        thumbnail_bytes = self.image_to_bytes_fn(image_data)

        return {
            "data_path": data_path,
            "image_data": image_data,
            "original_image_bytes": thumbnail_bytes,
            "image_statistics": stats,
            "locked_analysis_config": fresh_config,
            "system_info": state.get("system_info", {}),
            "literature_context": state.get("literature_context"),
            "analysis_hints": state.get("analysis_hints"),
            "analysis_objective": state.get("analysis_objective"),
            "skill_name": state.get("skill_name"),
            "skill_sections": state.get("skill_sections"),
            "auxiliary_items": state.get("auxiliary_items", []),
            "prior_knowledge": state.get("prior_knowledge", []),
            "analysis_images": [],
        }

    def _ask_user_for_consensus(self, improved, pipeline_counts):
        """Ask user which pipeline to use when refits found no consensus."""
        print("\n" + "=" * 60)
        print("ADAPTIVE REFIT: No pipeline consensus among re-analyzed images")
        print("=" * 60)
        print("\nThe re-analyzed images used different pipelines:")
        for i, (pipeline, count) in enumerate(
            sorted(pipeline_counts.items(), key=lambda x: -x[1]), 1
        ):
            indices = [
                str(r["index"]) for r in improved
                if r["new_pipeline"] == pipeline
            ]
            scores = [
                r["new_score"] for r in improved
                if r["new_pipeline"] == pipeline
            ]
            score_str = ", ".join(f"{v:.2f}" for v in scores)
            print(
                f"  {i}. '{pipeline}' - images "
                f"[{', '.join(indices)}], scores: {score_str}"
            )

        print("\nOptions:")
        print(
            "  - Enter a number (1, 2, ...) to use that pipeline for "
            "all re-analyzed images"
        )
        print("  - Type a pipeline description to suggest a different approach")
        print("  - Press Enter to keep the independent results as-is")
        print("-" * 60)

        response = input("\nYour choice: ").strip()
        if not response:
            print("Keeping independent refit results.")
            return None

        try:
            choice = int(response)
            pipelines = sorted(
                pipeline_counts.keys(),
                key=lambda m: -pipeline_counts[m],
            )
            if 1 <= choice <= len(pipelines):
                selected = pipelines[choice - 1]
                print(f"Will re-analyze with '{selected}'")
                return selected
        except ValueError:
            pass

        print(f"Will re-analyze with '{response}'")
        return response

    def _run_consistency_refit(
        self,
        minority,
        target_pipeline,
        improved,
        state,
        series_results,
        image_paths,
        image_stack,
    ):
        """Re-analyze minority images using the target pipeline."""
        peer_scores = [
            r["new_score"] for r in improved
            if r["new_pipeline"] == target_pipeline
        ]
        peer_count = len(peer_scores)

        for entry in minority:
            idx = entry["index"]
            name = entry["name"]
            self.logger.info(
                f"  Re-analyzing [{idx}] {name} with '{target_pipeline}'"
            )

            image_data = self._load_image(idx, image_paths, image_stack)
            if image_data is None:
                continue

            refit_state = self._build_refit_state(
                state, image_data, idx, name
            )
            if peer_scores:
                refit_state["locked_analysis_config"]["analysis_approach"] += (
                    f"\n\n**Peer evidence:** {peer_count} other images in "
                    f"this series were successfully re-analyzed with "
                    f"'{target_pipeline}' "
                    f"(scores {min(peer_scores):.2f}-{max(peer_scores):.2f}). "
                    f"Strongly prefer this pipeline unless the image clearly "
                    f"requires something different."
                )
            refit_state["locked_analysis_config"][
                "processing_pipeline"
            ] = target_pipeline

            data_path = (
                image_paths[idx]
                if image_paths and idx < len(image_paths)
                else name
            )
            try:
                result = self._processing_helper._execute_and_verify(
                    state=refit_state, image_data=image_data,
                    data_path=data_path, image_name=name, image_idx=idx,
                )
            except Exception as e:
                self.logger.error(
                    f"  Consistency refit failed for {name}: {e}"
                )
                continue

            new_score = result.get(
                "quality_metrics", {}
            ).get("quality_score", 0.0)
            if isinstance(new_score, str):
                try:
                    new_score = float(new_score)
                except (ValueError, TypeError):
                    new_score = 0.0
            prev_score = entry["new_score"] or 0

            if result["success"] and new_score >= prev_score * 0.99:
                self.logger.info(
                    f"  Consistent: score {new_score:.2f} with "
                    f"'{target_pipeline}'"
                )
                result["adaptively_refitted"] = True
                result["original_analysis_type"] = entry.get(
                    "original_pipeline"
                )
                result["refit_analysis_type"] = result.get("analysis_type")
                result["locked_pipeline"] = state.get(
                    "locked_analysis_config", {}
                ).get("processing_pipeline")
                series_results[idx] = result
                entry["new_score"] = new_score
                entry["new_pipeline"] = result.get("analysis_type")
            elif self.enable_human_feedback:
                keep = self._ask_keep_consistency_result(
                    name, idx, target_pipeline, new_score,
                    entry["new_pipeline"], prev_score,
                )
                if keep:
                    result["adaptively_refitted"] = True
                    result["original_analysis_type"] = entry.get(
                        "original_pipeline"
                    )
                    result["refit_analysis_type"] = result.get("analysis_type")
                    result["locked_pipeline"] = state.get(
                        "locked_analysis_config", {}
                    ).get("processing_pipeline")
                    series_results[idx] = result
                    entry["new_score"] = new_score
                    entry["new_pipeline"] = result.get("analysis_type")
                else:
                    self.logger.info(
                        f"  Keeping original refit for [{idx}] {name}"
                    )
            else:
                self.logger.info(
                    f"  Keeping original refit: consensus "
                    f"score={new_score:.2f} vs previous "
                    f"score={prev_score:.2f}"
                )

    def _ask_keep_consistency_result(
        self,
        name,
        idx,
        consensus_pipeline,
        consensus_score,
        original_pipeline,
        original_score,
    ):
        """Ask user whether to keep consensus pipeline when score dropped."""
        print("\n" + "-" * 60)
        print(
            f"Image [{idx}] {name}: consensus pipeline has lower quality"
        )
        print("-" * 60)
        print(
            f"  Consensus: '{consensus_pipeline}' -> "
            f"score = {consensus_score:.2f}"
        )
        print(
            f"  Independent: '{original_pipeline}' -> "
            f"score = {original_score:.2f}"
        )
        print("\nOptions:")
        print(
            f"  - Type 'consensus' to use '{consensus_pipeline}' "
            f"for consistency"
        )
        print(f"  - Press Enter to keep '{original_pipeline}'")

        response = input("\nYour choice: ").strip().lower()
        if response == "consensus":
            print(f"Using consensus pipeline for [{idx}] {name}")
            return True
        print(f"Keeping independent pipeline for [{idx}] {name}")
        return False

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if state.get("is_single_image", True):
            return state

        flagged_images = state.get("flagged_images", [])
        if not flagged_images:
            self.logger.info(
                "\nAdaptive refit: No flagged images, skipping."
            )
            return state

        refit_candidates = [
            f for f in flagged_images
            if f["reason"] in self.REFIT_REASONS
        ]
        if not refit_candidates:
            self.logger.info(
                "\nAdaptive refit: Flagged images are statistical "
                "outliers only, skipping."
            )
            return state

        self.logger.info(
            f"\nADAPTIVE REFIT: {len(refit_candidates)} images to "
            f"re-analyze independently"
        )

        series_results = state.get("series_results", [])
        image_paths = state.get("image_paths", [])
        image_stack = state.get("image_stack")
        refit_summary = []

        for flagged in refit_candidates:
            idx = flagged["index"]
            name = flagged["name"]
            original_details = flagged.get("details", "")

            self.logger.info(
                f"\n  Re-analyzing [{idx}] {name} "
                f"({original_details[:60]})"
            )

            image_data = self._load_image(idx, image_paths, image_stack)
            if image_data is None:
                self.logger.warning(
                    f"  Could not load image data for {name}, skipping"
                )
                continue

            refit_state = self._build_refit_state(
                state, image_data, idx, name
            )
            image_paths_list = state.get("image_paths", [])
            data_path = (
                image_paths_list[idx]
                if image_paths_list and idx < len(image_paths_list)
                else name
            )

            try:
                refit_result = self._processing_helper._execute_and_verify(
                    state=refit_state, image_data=image_data,
                    data_path=data_path, image_name=name, image_idx=idx,
                )
            except Exception as e:
                self.logger.error(f"  Refit failed for {name}: {e}")
                refit_summary.append({
                    "index": idx,
                    "name": name,
                    "original_pipeline": state.get(
                        "locked_analysis_config", {}
                    ).get("processing_pipeline"),
                    "new_pipeline": None,
                    "new_score": None,
                    "improved": False,
                })
                continue

            new_score = refit_result.get(
                "quality_metrics", {}
            ).get("quality_score", 0.0)
            if isinstance(new_score, str):
                try:
                    new_score = float(new_score)
                except (ValueError, TypeError):
                    new_score = 0.0
            locked_pipeline = state.get(
                "locked_analysis_config", {}
            ).get("processing_pipeline")

            if refit_result["success"]:
                self.logger.info(
                    f"  Refit score: {new_score:.2f}"
                )
                refit_result["adaptively_refitted"] = True
                refit_result["original_analysis_type"] = locked_pipeline
                refit_result["refit_analysis_type"] = refit_result.get(
                    "analysis_type"
                )
                refit_result["locked_pipeline"] = locked_pipeline
                series_results[idx] = refit_result

                refit_summary.append({
                    "index": idx,
                    "name": name,
                    "original_pipeline": locked_pipeline,
                    "new_pipeline": refit_result.get("analysis_type"),
                    "new_score": new_score,
                    "improved": True,
                })
            else:
                self.logger.info(
                    f"  Refit failed, keeping original"
                )
                refit_summary.append({
                    "index": idx,
                    "name": name,
                    "original_pipeline": locked_pipeline,
                    "new_pipeline": None,
                    "new_score": new_score,
                    "improved": False,
                })

        # --- Consistency pass ---
        improved = [
            r for r in refit_summary
            if r["improved"] and r.get("new_pipeline")
        ]
        if len(improved) >= 2:
            pipeline_counts = {}
            for r in improved:
                pipeline_counts[r["new_pipeline"]] = (
                    pipeline_counts.get(r["new_pipeline"], 0) + 1
                )
            top_pipeline, top_count = max(
                pipeline_counts.items(), key=lambda x: x[1]
            )
            has_majority = top_count > len(improved) / 2
            minority = [
                r for r in improved if r["new_pipeline"] != top_pipeline
            ]

            if has_majority and minority:
                self.logger.info(
                    f"\nConsistency pass: majority pipeline is "
                    f"'{top_pipeline}' ({top_count}/{len(improved)}), "
                    f"re-analyzing {len(minority)} outlier(s)"
                )
                self._run_consistency_refit(
                    minority, top_pipeline, improved, state,
                    series_results, image_paths, image_stack,
                )
            elif not has_majority and len(pipeline_counts) > 1:
                if self.enable_human_feedback:
                    user_pipeline = self._ask_user_for_consensus(
                        improved, pipeline_counts
                    )
                    if user_pipeline:
                        user_minority = [
                            r for r in improved
                            if r["new_pipeline"] != user_pipeline
                        ]
                        if user_minority:
                            self.logger.info(
                                f"\nUser-guided consistency: re-analyzing "
                                f"{len(user_minority)} images with "
                                f"'{user_pipeline}'"
                            )
                            self._run_consistency_refit(
                                user_minority, user_pipeline, improved,
                                state, series_results, image_paths,
                                image_stack,
                            )
                else:
                    self.logger.info(
                        f"\nNo pipeline consensus among refitted images "
                        f"({dict(pipeline_counts)}). Keeping independent "
                        f"results."
                    )

        state["series_results"] = series_results
        state["refit_summary"] = refit_summary

        # Re-run outlier detection with updated results
        updated_flagged = self._processing_helper._detect_outliers(
            series_results
        )
        state["flagged_images"] = updated_flagged

        improved_count = sum(1 for r in refit_summary if r["improved"])
        self.logger.info(
            f"\nAdaptive refit complete: {improved_count}/"
            f"{len(refit_candidates)} images improved"
        )

        return state


class ConditionalImageTrendController:
    """Generates and executes custom Python script for trend analysis across
    image series. Only for n >= 2."""

    TREND_ANALYSIS_INSTRUCTIONS = '''You are analyzing a series of image analysis results to identify trends.
{objective}
**SERIES SUMMARY:**
{series_summary}

**SERIES METADATA:**
{series_metadata}

**FLAGGED IMAGES:**
{flagged_info}

**CRITICAL REQUIREMENTS:**
1. DO NOT use plt.show() anywhere in the script - only save figures with plt.savefig()
2. DO NOT include individual image analysis visualizations - only create feature trend dashboard
3. Use plt.close('all') after saving each figure to free memory

**VISUALIZATION SCOPE - TRENDS:**
Create a SINGLE dashboard figure showing how extracted FEATURES evolve across the series.
DO NOT recreate individual image analyses - those already exist separately.

The series may vary ONE control variable or SEVERAL at once (a factorial /
grid design). Inspect `series_metadata` for a `secondary_variables` entry and
choose the representation to match:
- ONE control variable: feature values (y-axis) vs that variable (x-axis),
  with error bars where available - the standard trend dashboard.
- TWO control variables: represent BOTH. If their values define a regular
  lattice (grid sampling), use a heatmap or filled contour of each key
  feature over the 2-D space. If the sampling is scattered, use a scatter
  plot positioned by the two variables and colored by the feature value.
  Detect grid vs scattered from the data itself.
- THREE OR MORE: there is no single canonical N-D trend plot - produce a
  best-effort view: plot each feature against the primary variable and
  facet or color by the remaining variable(s), or use pairwise panels.
- In every case also show quality-metric evolution and mark flagged images
  with distinct markers.
State the representation you chose (and why) in `analysis_approach`.

**FIGURE REQUIREMENTS:**
- Create ONE summary dashboard figure (feature_trends.png)
- 2x2 or 2x3 subplot layout with 4-6 most important features
- Clean, publication-quality appearance
- Mark flagged images with red X markers
- Include linear regression trend lines where appropriate
- NO plt.show() calls
- Use plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
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
with open('series_analysis_results.json', 'r') as f:
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

# Extract series variable and features...
# Create figure with subplots...
# Plot feature trends (NOT individual analyses)...

plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display
```

Return JSON with:
{{
    "analysis_approach": "brief description",
    "key_metrics": ["list", "of", "features", "tracked"],
    "flagged_handling": "how flagged images are marked",
    "expected_outputs": ["feature_trends.png"],
    "script": "full python script - NO plt.show()"
}}
'''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        output_dir: str,
        max_corrections: int = 3,
    ):
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
        flagged_images = state.get("flagged_images", [])

        feature_summary = []
        for r in series_results:
            if r["success"]:
                summary = {
                    "index": r["index"],
                    "name": r["name"],
                    "analysis_type": r.get("analysis_type"),
                    "extracted_features": r.get("extracted_features", {}),
                    "quality_metrics": r.get("quality_metrics", {}),
                }
                if r.get("flagged"):
                    summary["flagged"] = True
                    summary["flag_reason"] = r.get("flag_reason")
                feature_summary.append(summary)

        flagged_info = (
            json.dumps(flagged_images, indent=2)
            if flagged_images
            else "No images were flagged."
        )

        objective = state.get("analysis_objective")
        objective_block = (
            f"\n**ANALYSIS OBJECTIVE:**\n{objective}\n"
            "Frame the trend analysis around answering this objective. "
            "If the objective involves calibration or quantitative modeling, "
            "the script must compute and output regression models.\n"
        ) if objective else ""

        prompt = self.TREND_ANALYSIS_INSTRUCTIONS.format(
            series_summary=json.dumps(feature_summary, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            flagged_info=flagged_info,
            objective=objective_block,
        )

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)
            if error_dict and not (result_json and 'script' in result_json):
                return None
            return result_json
        except Exception as e:
            self.logger.error(f"Error generating trend script: {e}")
            return None

    def _execute_script(self, script: str) -> tuple:
        # Remove any plt.show() calls
        script = re.sub(
            r'plt\.show\s*\(\s*\)', '# plt.show() removed', script
        )

        # Ensure matplotlib backend is set
        if 'matplotlib.use' not in script:
            script = "import matplotlib\nmatplotlib.use('Agg')\n" + script

        script_path = self.output_dir / "trend_analysis.py"
        with open(script_path, 'w') as f:
            f.write(script)
        result = self.executor.execute_script(
            script, working_dir=str(self.output_dir)
        )
        return (
            result.get("status") == "success",
            result.get("stdout", ""),
            result.get("message", ""),
        )

    def _correct_script(
        self, original_script: str, error_message: str, attempt: int
    ) -> Optional[str]:
        self.logger.info(
            f"   Attempting script correction (attempt {attempt})..."
        )
        if len(error_message) > 1000:
            tail = "\n".join(error_message.splitlines()[-30:])
            error_message = (
                error_message[:500]
                + "\n...[truncated]...\n"
                + error_message[-500:]
                + "\n\n--- Last 30 lines ---\n"
                + tail
            )

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
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, _ = self._parse(response)
            if result_json:
                self.logger.info(
                    f"   Diagnosis: {result_json.get('diagnosis', 'N/A')}"
                )
                return result_json.get("script")
            return None
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", True)

        if is_single or num_images < 2:
            self.logger.info(
                "\nTrend analysis skipped (single image mode).\n"
            )
            state["trend_analysis_results"] = {
                "success": True,
                "skipped": True,
                "reason": "Single image - no trend analysis applicable",
            }
            return state

        self.logger.info("")
        self.logger.info("TREND ANALYSIS")

        flagged_count = len(state.get("flagged_images", []))
        if flagged_count > 0:
            self.logger.info(
                f"   Note: {flagged_count} flagged images will be "
                f"highlighted in visualizations"
            )

        script_result = self._generate_trend_script(state)

        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate trend analysis script.")
            state["trend_analysis_results"] = {
                "success": False,
                "error": "Script generation failed",
            }
            return state

        self.logger.info(
            f"   Approach: {script_result.get('analysis_approach', 'unknown')}"
        )
        self.logger.info(
            f"   Metrics: {script_result.get('key_metrics', [])}"
        )

        script = script_result["script"]
        success, stdout, stderr = False, "", ""

        for attempt in range(self.max_corrections + 1):
            if attempt > 0:
                self.logger.info(f"   Execution attempt {attempt + 1}")

            success, stdout, stderr = self._execute_script(script)

            if success:
                self.logger.info("   Trend analysis completed!")
                break

            self.logger.warning(f"   Script failed: {stderr[:200]}...")

            if attempt < self.max_corrections:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break

        generated_files = []
        for f in self.output_dir.glob('*.png'):
            fname = f.name
            if '_analysis.png' in fname:
                continue
            if fname.startswith('image_') and fname.endswith('.png'):
                continue
            if fname in [
                'quality_review_analysis.png',
                'first_image_analysis_review.png',
            ]:
                continue
            generated_files.append(str(f))

        for f in self.output_dir.glob('*.csv'):
            if f.name not in [
                'series_analysis_results.json',
                'flagged_images.json',
            ]:
                generated_files.append(str(f))

        state["trend_analysis_results"] = {
            "success": success,
            "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics"),
            "flagged_handling": script_result.get("flagged_handling"),
            "stdout": stdout,
            "stderr": stderr if not success else None,
            "generated_files": generated_files,
            "script_path": str(self.output_dir / "trend_analysis.py"),
        }

        return state


class UnifiedImageSynthesisController:
    """Synthesizes findings into scientific claims. Adapts to single vs series."""

    SERIES_SYNTHESIS_INSTRUCTIONS = '''You are synthesizing findings from an image analysis of a series of images.

**SERIES OVERVIEW:**
- Total images: {num_images}
- Successful analyses: {successful_analyses}
- Analysis pipeline: {analysis_pipeline}
- Flagged images: {flagged_count}

**INDIVIDUAL ANALYSIS SUMMARIES:**
{analysis_summaries}

**FLAGGED IMAGES (require attention):**
{flagged_summary}

**ADAPTIVE REFIT RESULTS:**
{refit_summary}

**TREND ANALYSIS RESULTS:**
{trend_results}

**SERIES METADATA:**
{series_metadata}

**SYSTEM INFORMATION:**
{system_info}

Provide comprehensive scientific synthesis including:
1. Overall quality assessment
2. Key trends in extracted features
3. Physical interpretation of feature evolution
4. **Analysis of flagged images** - what might explain why these analyzed poorly?
5. **1-2 scientific claims** supported by the data (not more). One focused,
   well-supported claim is preferred — only add a second when it covers a
   genuinely distinct finding. Do not pad with redundant or speculative claims.
6. Caveats and limitations
7. **Analysis of adaptively re-analyzed images** - if any images were re-analyzed with different pipelines, interpret what this means scientifically (e.g., structural transition, different morphology, instrumental change)

Return JSON with:
{{
    "detailed_analysis": "comprehensive scientific interpretation",
    "scientific_claims": [
        {{
            "claim": "specific claim statement",
            "scientific_impact": "why this matters",
            "has_anyone_question": "research question formulation",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "feature_trends": {{
        "feature_name": {{"trend": "increasing/decreasing/stable", "interpretation": "physical meaning"}}
    }},
    "flagged_images_analysis": {{
        "summary": "interpretation of why images were flagged",
        "possible_causes": ["list of explanations"],
        "recommended_followup": ["suggested investigations"],
        "scientific_significance": "whether outliers represent interesting physics"
    }},
    "refit_analysis": {{
        "summary": "interpretation of why different pipelines were needed",
        "pipeline_changes": [{{"index": 0, "from_pipeline": "...", "to_pipeline": "...", "interpretation": "..."}}],
        "scientific_implications": "what the pipeline changes tell us about the system"
    }},
    "caveats": "limitations and considerations"
}}
'''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        single_image_instructions: str,
        output_dir: str,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.single_image_instructions = single_image_instructions
        self.output_dir = Path(output_dir)

    def _synthesize_single_image(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("SINGLE IMAGE INTERPRETATION")

        analysis_result = state.get("analysis_result", {})
        series_results = state.get("series_results", [])

        quality_warning = None
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]

        formatted = self.single_image_instructions.format(
            analysis_type=analysis_result.get("analysis_type", "Image analysis"),
            summary=analysis_result.get("summary", "Analysis complete"),
        )

        prompt_parts = [
            formatted,
            "\n## Original Image",
            {"mime_type": "image/jpeg", "data": state["original_image_bytes"]},
        ]

        if state.get("final_viz_bytes"):
            prompt_parts.extend([
                "\n## Analysis Visualization",
                {"mime_type": "image/png", "data": state["final_viz_bytes"]},
            ])

        prompt_parts.extend([
            "\n## Extracted Features\n" + json.dumps(
                analysis_result.get("extracted_features", {}), indent=2
            ),
            "\n## Quality Metrics\n" + json.dumps(
                analysis_result.get("quality_metrics", {}), indent=2
            ),
            "\n## Metadata\n" + json.dumps(
                state.get("system_info", {}), indent=2
            ),
        ])

        if quality_warning:
            prompt_parts.append(
                f"\n## Quality Warning\n{quality_warning}\n"
                "Note: this is the best result achieved after verification "
                "refinement — quality remained below the threshold."
            )

        qh = analysis_result.get("quality_history")
        if qh:
            qh_lines = ["## Analysis Quality Context"]
            approved = qh.get("approved", True)
            qh_lines.append(
                f"- Final score: {qh.get('final_score', 'N/A')} "
                f"({'approved' if approved else 'best available, below threshold'})"
            )
            iters = qh.get("verification_iterations", [])
            if len(iters) > 1:
                qh_lines.append(
                    f"- Verification iterations: {len(iters)}"
                )
            # Remaining issues from last verification (known limitations)
            if iters:
                last_issues = iters[-1].get("issues", [])
                if last_issues:
                    qh_lines.append("- Known limitations in this result:")
                    for iss in last_issues[:5]:
                        qh_lines.append(
                            f"  - {iss.get('location', '')}: "
                            f"{iss.get('problem', '')}"
                        )
            # Script errors
            se = qh.get("script_errors", [])
            if se:
                qh_lines.append(
                    f"- {len(se)} script correction(s) needed: "
                    + "; ".join(e["error"][:80] for e in se[:3])
                )
            qh_lines.append(
                "\nNote these known limitations where relevant "
                "in your interpretation."
            )
            prompt_parts.append("\n".join(qh_lines))

        if state.get("literature_context"):
            prompt_parts.extend([
                "\n## Literature", state["literature_context"]
            ])

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_subagent_context(prompt_parts, state)

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)

            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("Single image synthesis complete.")
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}

        return state

    def _synthesize_series(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("SERIES SYNTHESIS")

        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        flagged_images = state.get("flagged_images", [])

        successful_analyses = [r for r in series_results if r["success"]]
        refit_summary_data = state.get("refit_summary", [])

        analysis_summaries = []
        for r in successful_analyses[:15]:
            summary = {
                "index": r["index"],
                "name": r["name"],
                "analysis_type": r.get("analysis_type"),
                "key_features": r.get("extracted_features", {}),
                "quality_metrics": r.get("quality_metrics", {}),
            }
            vh = r.get("quality_history")
            if vh:
                summary["quality_score"] = vh.get("final_score")
                summary["quality_approved"] = vh.get("approved")
            if r.get("flagged"):
                summary["flagged"] = True
                summary["flag_reason"] = r.get("flag_reason")
            if r.get("adaptively_refitted"):
                summary["adaptively_refitted"] = True
                summary["refit_analysis_type"] = r.get("refit_analysis_type")
                summary["locked_pipeline"] = r.get("locked_pipeline")
            analysis_summaries.append(summary)

        flagged_summary = (
            json.dumps(flagged_images, indent=2)
            if flagged_images
            else "No images were flagged."
        )
        refit_summary_str = (
            json.dumps(refit_summary_data, indent=2)
            if refit_summary_data
            else "No images were adaptively re-analyzed."
        )

        # Handle mixed pipeline types when refitting occurred
        pipeline_types_used = set()
        for r in successful_analyses:
            at = r.get("analysis_type")
            if at:
                pipeline_types_used.add(at)

        if len(pipeline_types_used) <= 1:
            analysis_pipeline = (
                successful_analyses[0].get("analysis_type")
                if successful_analyses
                else "Unknown"
            )
        else:
            locked_pipeline = state.get(
                "locked_analysis_config", {}
            ).get("processing_pipeline", "Unknown")
            refitted_pipelines = [
                r.get("refit_analysis_type")
                for r in successful_analyses
                if r.get("adaptively_refitted") and r.get("refit_analysis_type")
            ]
            unique_refit = sorted(set(refitted_pipelines))
            analysis_pipeline = (
                f"Primary: {locked_pipeline}; "
                f"Re-analyzed: {', '.join(unique_refit)}"
            )

        prompt = self.SERIES_SYNTHESIS_INSTRUCTIONS.format(
            num_images=state.get("num_images", 1),
            successful_analyses=len(successful_analyses),
            analysis_pipeline=analysis_pipeline,
            flagged_count=len(flagged_images),
            analysis_summaries=json.dumps(analysis_summaries, indent=2),
            flagged_summary=flagged_summary,
            refit_summary=refit_summary_str,
            trend_results=json.dumps(trend_results, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            system_info=json.dumps(state.get("system_info", {}), indent=2),
        )

        prompt_parts = [prompt]

        # Aggregate quality verification summary
        verified = [
            r for r in successful_analyses
            if r.get("quality_history")
        ]
        if verified:
            scores = [
                r["quality_history"]["final_score"]
                for r in verified
                if r["quality_history"].get("final_score") is not None
            ]
            approved_n = sum(
                1 for r in verified if r["quality_history"].get("approved")
            )
            qv_lines = [
                "\n**QUALITY VERIFICATION SUMMARY:**",
                f"- Verified images: {len(verified)}/{len(successful_analyses)}",
                f"- Approved: {approved_n}/{len(verified)}",
            ]
            if scores:
                qv_lines.append(
                    f"- Score range: {min(scores):.2f} - {max(scores):.2f} "
                    f"(mean: {sum(scores)/len(scores):.2f})"
                )
            prompt_parts.append("\n".join(qv_lines))

        if flagged_images:
            prompt_parts.append("\n\n**FLAGGED IMAGE VISUALIZATIONS:**")
            flagged_indices = {f["index"] for f in flagged_images}
            included_count = 0
            for r in series_results:
                if (
                    r["index"] in flagged_indices
                    and r.get("visualization_bytes")
                    and included_count < 5
                ):
                    prompt_parts.append(
                        f"\n{r['name']} (flagged: "
                        f"{r.get('flag_reason', 'unknown')}):"
                    )
                    prompt_parts.append({
                        "mime_type": "image/png",
                        "data": r["visualization_bytes"],
                    })
                    included_count += 1

        if trend_results.get("success") and trend_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND VISUALIZATIONS:**")
            for file_path in trend_results["generated_files"][:5]:
                if (
                    file_path.endswith('.png')
                    and Path(file_path).exists()
                ):
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({
                            "mime_type": "image/png",
                            "data": f.read(),
                        })

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_subagent_context(prompt_parts, state)

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse(response)

            if error_dict:
                self.logger.error(f"Series synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("Series synthesis complete.")
        except Exception as e:
            self.logger.error(f"Series synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}

        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_image", True)

        if is_single:
            return self._synthesize_single_image(state)
        else:
            return self._synthesize_series(state)


class GenerateImageReportController:
    """Generates a human-readable HTML report for image analysis."""

    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = Path(output_dir)

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_image", True)

        if is_single:
            return self._generate_single_image_report(state)
        else:
            return self._generate_series_report(state)

    # ------------------------------------------------------------------
    # Single image report
    # ------------------------------------------------------------------

    def _generate_single_image_report(self, state: dict) -> dict:
        self.logger.info("\n--- Generating HTML Report (Single Image) ---\n")

        analysis_result = state.get("analysis_result", {})
        synthesis_result = state.get("synthesis_result", {})

        detailed_analysis = (
            synthesis_result.get("detailed_analysis")
            or analysis_result.get("summary", "No analysis provided.")
        )
        scientific_claims = (
            synthesis_result.get("scientific_claims")
            or []
        )
        system_info = state.get("system_info", {})
        analysis_type = analysis_result.get("analysis_type", "N/A")
        caveats = synthesis_result.get("caveats", "")

        quality_warning = None
        series_results = state.get("series_results", [])
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]

        original_image = state.get("original_image_bytes")
        analysis_viz = state.get("final_viz_bytes")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ImageAnalysis_Report_{file_timestamp}.html"
        filepath = output_dir / filename

        # Images section
        images_html = ""
        if original_image:
            b64 = self._image_to_base64(original_image)
            images_html += (
                f'<div class="image-card">'
                f'<img src="data:image/jpeg;base64,{b64}" alt="Original Image">'
                f'<div class="image-label">Original Image</div></div>'
            )
        if analysis_viz:
            b64 = self._image_to_base64(analysis_viz)
            images_html += (
                f'<div class="image-card">'
                f'<img src="data:image/png;base64,{b64}" alt="Analysis Result">'
                f'<div class="image-label">Analysis Visualization</div></div>'
            )

        # Claims
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
        <h2>4. Caveats & Limitations</h2>
        <div class="caveats">{caveats}</div>"""

        system_info_str = self._format_system_info(system_info)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis Report</title>
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
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {state.get('data_path', 'N/A')}</p>
            <p><strong>Sample Info:</strong> {system_info_str}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <h3>Analysis Type</h3>
        <div class="model-box">{analysis_type}</div>
        <h3>Interpretation</h3>
        <div class="analysis-text">{detailed_analysis}</div>
        <h2>2. Visualizations</h2>
        <div class="image-grid">{images_html}</div>
        <h2>3. Scientific Claims</h2>
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Image Analysis Agent</div>
    </div>
</body>
</html>"""

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"  Report saved: {filepath}")
            state["report_path"] = str(filepath)
        except Exception as e:
            self.logger.error(f"  Failed to write report: {e}")

        return state

    # ------------------------------------------------------------------
    # Series report
    # ------------------------------------------------------------------

    def _generate_series_report(self, state: dict) -> None:
        self.logger.info("")
        self.logger.info("GENERATING SERIES REPORT")

        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        synthesis = state.get("synthesis_result", {})
        series_metadata = state.get("series_metadata", {})
        locked_config = state.get("locked_analysis_config", {})
        flagged_images = state.get("flagged_images", [])
        refit_summary = state.get("refit_summary", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_images = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_images)
        refitted_count = sum(1 for r in refit_summary if r.get("improved"))

        # Quality status indicator
        if flagged_count == 0:
            quality_indicator = (
                '<span class="quality-indicator quality-good">'
                'All analyses acceptable</span>'
            )
        elif flagged_count <= num_images * 0.1:
            quality_indicator = (
                f'<span class="quality-indicator quality-warning">'
                f'{flagged_count} images flagged</span>'
            )
        else:
            quality_indicator = (
                f'<span class="quality-indicator quality-critical">'
                f'{flagged_count} images flagged '
                f'({100 * flagged_count / num_images:.0f}%)</span>'
            )

        # Trend visualizations
        trend_viz_html = ""
        if trend_results.get("success") and trend_results.get("generated_files"):
            trend_viz_html = (
                '<h2>3. Trend Visualizations</h2><div class="image-grid">'
            )
            for file_path in trend_results["generated_files"]:
                if (
                    file_path.endswith('.png')
                    and Path(file_path).exists()
                ):
                    with open(file_path, 'rb') as f:
                        b64 = self._image_to_base64(f.read())
                    name = Path(file_path).stem.replace('_', ' ').title()
                    trend_viz_html += (
                        f'<div class="image-card">'
                        f'<img src="data:image/png;base64,{b64}" '
                        f'alt="{name}">'
                        f'<div class="image-label">{name}</div></div>'
                    )
            trend_viz_html += '</div>'

        # Feature trends
        feature_trends_html = ""
        feature_trends = synthesis.get('feature_trends', {})
        if feature_trends:
            feature_trends_html = "<h2>2. Feature Trends</h2>"
            for feature_name, trend_info in feature_trends.items():
                if isinstance(trend_info, dict):
                    feature_trends_html += (
                        f'<div class="trend-card">'
                        f'<strong>{feature_name}</strong><br>'
                        f'Trend: {trend_info.get("trend", "N/A")}<br>'
                        f'<em>{trend_info.get("interpretation", "")}</em>'
                        f'</div>'
                    )

        # Scientific claims
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

        # Caveats
        caveats_html = ""
        caveats = synthesis.get('caveats', '')
        if caveats:
            caveats_html = (
                f'<h2>6. Caveats & Limitations</h2>'
                f'<div class="caveats">{caveats}</div>'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Series Analysis Report</title>
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
        <h1>Image Series Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Images Processed:</strong> {successful}/{num_images}</p>
            <p><strong>Series Variable:</strong> {series_metadata.get('variable', 'N/A')}</p>
            <p><strong>Analysis Pipeline:</strong> {locked_config.get('processing_pipeline', 'N/A')}{f' ({refitted_count} images re-analyzed with alternative pipelines)' if refitted_count > 0 else ''}</p>
            <p><strong>Quality Status:</strong> {quality_indicator}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{synthesis.get('detailed_analysis', 'No analysis available.')}</div>
        {feature_trends_html}
        {trend_viz_html}
        {self._generate_individual_results_section(series_results, num_images)}
        {self._generate_refit_section(refit_summary, series_results) if refit_summary else ''}
        {self._generate_flagged_images_section(flagged_images, series_results, synthesis) if flagged_images else ''}
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Image Analysis Agent</div>
    </div>
</body>
</html>"""

        report_path = self.output_dir / "series_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        state["report_path"] = str(report_path)
        self.logger.info(f"   Report saved: {report_path}")

        return state

    # ------------------------------------------------------------------
    # Report helper methods
    # ------------------------------------------------------------------

    def _format_system_info(self, system_info: dict) -> str:
        if not system_info:
            return "N/A"
        parts = [f"{k}: {v}" for k, v in system_info.items() if v]
        return ", ".join(parts) if parts else "N/A"

    def _generate_flagged_images_section(
        self,
        flagged_images: List[dict],
        series_results: List[dict],
        synthesis: dict,
    ) -> str:
        if not flagged_images:
            return ""

        flagged_analysis = synthesis.get("flagged_images_analysis", {})

        html = f"""
        <h2>Flagged Images</h2>
        <div class="flagged-summary">
            <p><strong>{len(flagged_images)} images flagged for review</strong></p>
            <p>{flagged_analysis.get("summary", "Some images showed anomalous analysis behavior.")}</p>
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
            html += (
                f"<h3>Scientific Significance</h3><p>{significance}</p>"
            )

        html += '<h3>Flagged Image Details</h3><div class="flagged-grid">'

        badge_colors = {
            "analysis_failed": ("#dc3545", "Failed"),
            "statistical_outlier": ("#fd7e14", "Outlier"),
            "below_threshold": ("#ffc107", "Low Quality"),
            "outlier_and_below_threshold": ("#dc3545", "Critical"),
        }

        for f in flagged_images:
            result = next(
                (r for r in series_results if r["index"] == f["index"]),
                None,
            )
            color, label = badge_colors.get(
                f["reason"], ("#6c757d", "Flagged")
            )

            html += (
                f'<div class="flagged-card" style="border-color: {color};">'
            )
            html += (
                f'<div class="flagged-card-header">'
                f'<strong>{f["name"]}</strong>'
                f'<span class="flagged-badge" '
                f'style="background-color: {color};">{label}</span></div>'
            )

            if f.get("details"):
                html += f'<p>{f["details"]}</p>'

            html += (
                f'<p class="flagged-recommendation">'
                f'{f["recommendation"]}</p>'
            )

            if (
                result
                and result.get("visualization_path")
                and Path(result["visualization_path"]).exists()
            ):
                with open(result["visualization_path"], 'rb') as img_f:
                    b64 = self._image_to_base64(img_f.read())
                html += (
                    f'<img src="data:image/png;base64,{b64}" '
                    f'alt="{f["name"]}">'
                )

            html += '</div>'

        html += '</div>'
        return html

    def _generate_refit_section(
        self, refit_summary: List[dict], series_results: List[dict]
    ) -> str:
        if not refit_summary:
            return ""

        improved = [r for r in refit_summary if r["improved"]]
        not_improved = [r for r in refit_summary if not r["improved"]]

        html = f"""
        <h2>Adaptive Re-Analysis Results</h2>
        <div class="refit-summary">
            <p><strong>{len(improved)}/{len(refit_summary)}</strong> images improved through independent re-analysis</p>
        </div>
"""

        if improved:
            html += (
                '<h3>Improved Analyses</h3>'
                '<table class="params-table"><thead><tr>'
                '<th>Image</th><th>Original Pipeline</th>'
                '<th>New Pipeline</th><th>Score</th>'
                '</tr></thead><tbody>'
            )
            for r in improved:
                new_score = (
                    f"{r['new_score']:.2f}"
                    if r.get("new_score") is not None
                    else "N/A"
                )
                html += (
                    f'<tr><td>{r["name"]}</td>'
                    f'<td>{r.get("original_pipeline", "N/A")}</td>'
                    f'<td>{r.get("new_pipeline", "N/A")}</td>'
                    f'<td>{new_score}</td></tr>'
                )
            html += '</tbody></table>'

        if not_improved:
            html += (
                '<h3>Unchanged Analyses</h3>'
                '<p>The following images could not be improved with '
                'alternative pipelines:</p><ul>'
            )
            for r in not_improved:
                html += f'<li>{r["name"]}</li>'
            html += '</ul>'

        # Include visualizations for improved images
        for r in improved:
            result = next(
                (sr for sr in series_results
                 if sr.get("index") == r["index"]),
                None,
            )
            if (
                result
                and result.get("visualization_path")
                and Path(result["visualization_path"]).exists()
            ):
                with open(result["visualization_path"], 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                new_score = (
                    f"{r.get('new_score', 0):.2f}"
                    if r.get('new_score') is not None
                    else "N/A"
                )
                html += (
                    f'<div class="image-card" '
                    f'style="border-left: 4px solid #17a2b8;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'alt="{r["name"]}">'
                    f'<div class="image-label">{r["name"]} '
                    f'(Re-analyzed, Score: {new_score})</div></div>'
                )

        return html

    def _generate_individual_results_section(
        self, series_results: List[dict], num_images: int
    ) -> str:
        results_with_viz = [
            (i, r)
            for i, r in enumerate(series_results)
            if r.get("visualization_path")
            and Path(r["visualization_path"]).exists()
        ]

        if not results_with_viz:
            return ""

        failed_indices = {
            i for i, r in enumerate(series_results) if not r["success"]
        }
        flagged_indices = {
            i for i, r in enumerate(series_results) if r.get("flagged")
        }
        priority_indices = failed_indices | flagged_indices

        if num_images <= 10:
            indices_to_show = set(range(num_images))
            section_note = ""
        elif num_images <= 30:
            indices_to_show = (
                set(range(min(3, num_images)))
                | set(range(max(0, num_images - 3), num_images))
            )
            if num_images > 6:
                step = (num_images - 6) // 5
                for i in range(3, num_images - 3, max(1, step)):
                    if len(indices_to_show) < 10:
                        indices_to_show.add(i)
            indices_to_show.update(priority_indices)
            not_shown = num_images - len(indices_to_show)
            section_note = (
                f"<p><em>Showing {len(indices_to_show)} of {num_images} "
                f"results. {not_shown} results not displayed.</em></p>"
            )
        else:
            indices_to_show = {0, 1, num_images - 2, num_images - 1}
            indices_to_show.update(list(priority_indices)[:10])
            section_note = (
                f"<p><em>Large series ({num_images} images): Showing "
                f"boundary results and flagged/failed images.</em></p>"
            )

        indices_to_show = sorted(indices_to_show)

        html = (
            f"\n        <h2>Individual Analysis Results</h2>\n{section_note}"
        )
        html += (
            '        <div class="image-grid" '
            'style="grid-template-columns: repeat(auto-fit, '
            'minmax(350px, 1fr));">\n'
        )

        for idx in indices_to_show:
            if idx >= len(series_results):
                continue
            r = series_results[idx]
            viz_path = r.get("visualization_path")

            if viz_path and Path(viz_path).exists():
                with open(viz_path, 'rb') as f:
                    b64 = self._image_to_base64(f.read())

                if not r["success"]:
                    status, status_color = "FAILED", "#e74c3c"
                elif r.get("adaptively_refitted"):
                    status, status_color = "Re-analyzed", "#17a2b8"
                elif r.get("flagged"):
                    status = r.get('flag_reason', 'Flagged')
                    status_color = "#fd7e14"
                else:
                    status, status_color = "OK", "#27ae60"

                refit_note = ""
                if r.get("adaptively_refitted"):
                    refit_note = (
                        f"<br><small>Original pipeline: "
                        f"{r.get('locked_pipeline', 'N/A')}</small>"
                    )

                html += f'''
            <div class="image-card" style="border-left: 4px solid {status_color};">
                <img src="data:image/png;base64,{b64}" alt="{r['name']}">
                <div style="margin-top: 8px;">
                    <strong>{r['name']}</strong><br>
                    <span style="color: {status_color};">{status}</span>{refit_note}
                </div>
            </div>
'''

        html += "        </div>\n"
        return html
