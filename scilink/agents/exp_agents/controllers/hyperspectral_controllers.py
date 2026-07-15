import logging
import numpy as np
import json
import os
import re
import inspect
from datetime import datetime
import base64
import cv2
from typing import Callable

import traceback

from ....skills.hyperspectral.eels import eels as tools
from ....skills._shared.image_processor import load_image
from ..preprocess import HyperspectralPreprocessingAgent
from ..metadata_converter import resolve_axis_spec, describe_axes_for_prompt
from ..instruct import (
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS,
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS,
    SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS,
    SPECTROSCOPY_REFLECTION_INSTRUCTIONS,
    SPECTROSCOPY_REFLECTION_UPDATE_INSTRUCTIONS,
    SPECTROSCOPY_VALIDATION_INTERPRETATION_INSTRUCTIONS,
    SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS,
    SPECTROSCOPY_PHYSICS_SANITY_INSTRUCTIONS,
    SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS,
    SPECTROSCOPY_SALVAGE_JUDGE_INSTRUCTIONS,
)

from ....skills.hyperspectral.eels.eels import AGENT_METADATA_KEYS_TO_STRIP
from ....skills._shared.curve_fitting_tools import plot_curve_to_bytes
from ....skills._shared._registry import (
    get_tools_for, get_tool_function, VERIFIER_TOOL_SCRUTINY_PRINCIPLE,
)
from ....executors import ExecutionTimeout
from .._qc_engine import CodegenQCEngine, QCEngineSpec, QCItemContext
from ....utils.codegen_parse import parse_codegen_response


def _render_skill_block(state: dict, stage: str) -> str:
    """Render the active domain skill(s) block for ``stage`` as a single string.

    With multiple skills loaded, each skill's block is rendered in order
    (most-relevant first) — including at the ``implementation`` (codegen)
    stage, where co-active skills may each own a different pipeline stage, so
    none is dropped. Returns ``""`` when no skill is active or no content
    exists for the requested stage. Use this from prompt builders that
    assemble a single string (e.g. ``build_code_generation_prompt``); the
    list-mode wrapper ``_append_skill_context`` below uses this internally.
    """
    skills = state.get("skills_loaded") or (
        [state["skill_sections"]] if state.get("skill_sections") else []
    )
    if not skills:
        return ""

    parts: list = []
    intro_appended = False
    for sections in skills:
        if not sections:
            continue
        content = sections.get(stage, "")
        if not content:
            continue
        skill_name = sections.get("name", "domain skill")

        parts.append(f"\n## MANDATORY Domain Skill Rules: {skill_name} ({stage})")
        if not intro_appended:
            parts.append(
                "The following rules are MANDATORY. Your analysis plan and implementation "
                "MUST conform to these domain-specific requirements. These rules encode "
                "validated domain expertise and take precedence over general-purpose defaults. "
                "Do NOT substitute your own preferences where these rules specify a method, "
                "treatment, or constraint."
            )
            intro_appended = True
        parts.append(content)

        # Include validation rules during planning, interpretation, and
        # implementation so the LLM knows quality criteria upfront — at planning
        # to shape the approach, at interpretation/implementation to shape the
        # output and the generated code.
        if stage in ("planning", "interpretation", "implementation"):
            validation = sections.get("validation", "")
            if validation:
                parts.append(f"\n## MANDATORY Domain Validation Rules ({skill_name})")
                parts.append(validation)

    return "\n".join(parts)


def _append_skill_context(prompt: list, state: dict, stage: str) -> None:
    """Append domain skill knowledge to an LLM prompt list for the given stage.

    Thin list-mode wrapper around ``_render_skill_block`` for prompt
    builders that assemble multi-part (text + image) prompts.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``skill_sections`` and ``skill_name``.
        stage: One of ``"planning"``, ``"analysis"``, ``"interpretation"``,
               ``"validation"``, ``"implementation"``.
    """
    block = _render_skill_block(state, stage)
    if block:
        prompt.append(block)


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


def _active_skill_names(state: dict) -> list:
    """Names of the skills currently active for this run (for registry tool
    scoping). Empty when running skill-free (cold)."""
    return [s.get("name") for s in (state.get("skills_loaded") or [])
            if isinstance(s, dict) and s.get("name")]


def _hyperspectral_tool_specs(state: dict):
    """ToolSpecs the _shared registry exposes to the hyperspectral agent given
    the active skills — the optional tools generated code may call."""
    try:
        return get_tools_for("hyperspectral", active_skills=_active_skill_names(state) or None)
    except Exception:
        return []


def _planning_tool_awareness(state: dict) -> str | None:
    """Minimal names + when_to_use for the downstream code tools, plus a
    discipline directive telling the PLANNER to stay method-level.

    The planner does NOT get the full tool inventory (that stays at the codegen
    step, to keep planning prompts tight). It gets just enough to avoid the
    failure this fixes: a tool-unaware plan prescribes hand-rolled physics
    ("embed a NIST mu tabulation", "use a 76-80 / 81-86 keV window") which the
    codegen then follows *instead of* calling the vetted tool — reintroducing
    the very bias the tool removes. Naming the tools + forbidding hand-rolled
    coefficient tables / prescribed windows keeps method selection in the plan
    and implementation detail in the tool. Returns None when no tools apply.
    """
    specs = _hyperspectral_tool_specs(state)
    if not specs:
        return None
    lines = [
        "\n\n--- Downstream Code Tools (available to the implementation step) ---",
        "The code implementing your plan can call these vetted helpers. Plan at the "
        "METHOD level and rely on them for the details they own. Do NOT embed your own "
        "coefficient/attenuation tables, and do NOT prescribe specific numeric windows "
        "or parameters that a tool below already handles — name the method (e.g. "
        "'measure the K-edge step', 'obtain mu from tables') and leave the exact "
        "windows / coefficients to the tool.",
        "When a tool below matches a required deliverable, NAME that tool in the "
        "target description as the method of record (e.g. 'quantify the edge jump "
        "with `measure_edge_step`') — the implementation must call it rather than "
        "hand-roll its own estimator.",
    ]
    for s in specs:
        wtu = getattr(s, "when_to_use", "") or getattr(s, "description", "")
        lines.append(f"- `{s.name}`: {wtu}")
    return "\n".join(lines)


def _registry_tool_callables(state: dict) -> dict:
    """{name: callable} for the registered hyperspectral tools, resolved from
    each spec's ``import_line`` (``from MODULE import NAME``). Best-effort — a
    tool whose module/callable can't be imported is skipped, so a missing
    optional dependency never breaks the sandbox setup.
    """
    import importlib
    out = {}
    for spec in _hyperspectral_tool_specs(state):
        try:
            il = getattr(spec, "import_line", "") or ""
            if il.startswith("from ") and " import " in il:
                mod_path, _, nm = il[len("from "):].partition(" import ")
                out[spec.name] = getattr(importlib.import_module(mod_path.strip()), nm.strip())
            else:
                out[spec.name] = get_tool_function(spec.name, active_skills=_active_skill_names(state))
        except Exception:
            continue
    return out


def _used_tool_descriptions(state: dict, code_str: str, max_bytes: int = 1500) -> str:
    """Descriptions of the registered tools the generated code actually CALLS,
    for the required-output review.

    Tells the verifier WHAT each used tool already handles robustly (window
    selection, flux gating, measurability, …) so it does not reject the result
    by second-guessing internals the tool owns — without feeding any per-run
    output. Names + description + when_to_use only. Bounded so it can never
    dominate the prompt.
    """
    if not code_str:
        return ""
    lines = []
    for s in _hyperspectral_tool_specs(state):
        if f"{s.name}(" in code_str:
            desc = (getattr(s, "description", "") or "").strip()
            wtu = (getattr(s, "when_to_use", "") or "").strip()
            entry = f"- `{s.name}`: {desc}"
            if wtu:
                entry += f" (When to use: {wtu})"
            lines.append(entry)
    blob = "\n".join(lines)
    return blob[:max_bytes] + ("…" if len(blob) > max_bytes else "")


def _auxiliary_display_items(state: dict) -> list:
    """Auxiliary datasets to show the LLM as context — items with a rendered
    plot, from the multi-aux ``auxiliary_items`` list. (#226)"""
    return [it for it in (state.get("auxiliary_items") or []) if it.get("plot_bytes")]


def _resample_ref_to_signal_axis(arr, ref_axis, energy_axis, e_):
    """Resample a 1D reference sampled on its OWN axis onto the data's signal
    (energy) axis, so it can serve as a per-channel codegen operand.

    Returns the resampled length-``e_`` array, or ``None`` when resampling is
    not warranted — i.e. the reference is not a 1D curve carrying its own axis,
    the signal axis is unknown/degenerate, or the two axes don't overlap (so
    they plausibly describe different quantities and interpolation would
    fabricate values rather than align them). Only the previously-dropped
    misaligned case is affected; aligned operands never reach here.
    """
    if ref_axis is None or arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    ref_axis = np.asarray(ref_axis, dtype=float)
    if arr.ndim != 1 or ref_axis.shape != arr.shape:
        return None
    # Need a real, length-matched signal axis to resample onto (not the
    # channel-index fallback shape).
    if energy_axis is None or np.asarray(energy_axis).shape != (e_,):
        return None
    energy_axis = np.asarray(energy_axis, dtype=float)
    order = np.argsort(ref_axis)
    ra, rv = ref_axis[order], arr[order]
    # Require axis overlap — guards against interpolating, say, a keV table
    # onto a channel-index axis (no shared range => not the same quantity).
    lo, hi = max(ra.min(), float(energy_axis.min())), min(ra.max(), float(energy_axis.max()))
    if not (hi > lo):
        return None
    return np.interp(energy_axis, ra, rv)


def _codegen_retry_feedback(failures: int, critique: str) -> str:
    """Annealed retry guidance for the per-pixel code-gen loop.

    Flat retries (re-prompt at the same setting with "fix the math") cannot
    escape a wrong-but-self-consistent method — they re-sample the same basin.
    So the guidance *anneals*: early failures patch the logic, repeated
    failures push the model to question and then ABANDON the method for a
    structurally different estimator. "Temperature" here is the prompt's
    structural freedom — modern Claude/Bedrock models omit the sampling
    temperature, so escalation is expressed in the instruction, not the knob.
    The escape-hatch families are generic to per-pixel quantitative extraction,
    so this helps any struggling analysis, not one technique.
    """
    block = [
        "\n\n### ❌ PREVIOUS ATTEMPT FAILED",
        f"Critique:\n```text\n{critique}\n```",
    ]
    if failures <= 1:
        block.append("Fix the logic/math to address this critique.")
    elif failures == 2:
        block.append(
            "This is the SECOND failure with the same approach — stop tweaking "
            "parameters and question the METHOD itself. A common cause of a "
            "non-physical or mostly-masked quantitative map is a fit performed "
            "across the FULL measurement axis: channels that violate the model "
            "(signal saturation, heavy absorption / near-zero transmission, low "
            "SNR, near-zero reference) bias a global fit. Restrict the extraction "
            "to the informative sub-range, or use a measure insensitive to those "
            "channels."
        )
    else:
        block.append(
            "The current APPROACH has failed repeatedly. Do NOT patch it again — "
            "ABANDON it and choose a STRUCTURALLY DIFFERENT estimator. If your "
            "previous attempts fit a model across the whole measurement axis, "
            "switch families: (a) restrict the fit to a narrow, informative "
            "window around the diagnostic feature instead of the full axis; "
            "(b) use a DIFFERENTIAL measure (difference of the signal just across "
            "the feature/edge) that cancels smooth backgrounds and is feature-"
            "specific; or (c) use a ROBUST estimator that down-weights saturated "
            "or outlier channels. Negative / NaN / mostly-masked outputs are "
            "strong evidence the global model is biased — change families, do not "
            "re-fit the same way."
        )
    if failures >= 2:
        block.append(
            "Note: 'method' and 'method family' refer to YOUR estimator "
            "structure — a vetted REGISTERED TOOL that fits the task remains "
            "the preferred implementation: first change how you drive it "
            "(windows, parameters, fallback edge/feature). Replace a "
            "registered tool with custom code only if you state which of its "
            "parameters you already tried and why no setting of them can "
            "address the failure."
        )
    return "\n".join(block)


def _retry_annealing_level(failures: int) -> int:
    """Map the codegen retry ladder onto the shared 0/1/2 annealing scale.

    0 = first attempt (no retry feedback); 1 = warm (patch the math /
    question the method); 2 = hot (abandon the method family) — see
    ``_codegen_retry_feedback``. Used for the per-attempt verification
    record so hyperspectral participates in the same hot-success staging
    gate as the other modalities.
    """
    if failures <= 0:
        return 0
    if failures <= 2:
        return 1
    return 2


def _retry_stage_label(failures: int) -> str:
    """Human label of the retry-feedback stage applied after this failure."""
    if failures <= 0:
        return ""
    if failures == 1:
        return "patch the logic/math"
    if failures == 2:
        return "question the method"
    return "abandon the method family"


def _hs_attempt_entry(level: int, passed_fraction, qc_failures: list,
                      recommended_action: str, error: str | None = None) -> dict:
    """One dynamic-analysis attempt as a verification-record entry.

    ``qc_failures`` strings are "feature: critique" — split into the shared
    issues shape; a hard execution error becomes a single issue.
    """
    issues = []
    for f in qc_failures or []:
        loc, _, prob = str(f).partition(":")
        issues.append({"location": loc.strip(), "problem": prob.strip() or loc.strip()})
    if error:
        issues.append({"location": "execution", "problem": str(error)[:500]})
    return {
        "passed_fraction": passed_fraction,
        "annealing_level": level,
        "issues_found": issues,
        "recommended_action": recommended_action,
    }


def _render_band_flux_table(data, axis, axis_units: str, aux: dict | None = None,
                            n_bands: int = 16) -> str:
    """Deterministic field-mean counts per axis band, for the result review.

    The review's flux/measurability arguments must rest on numbers, not on a
    visual read of the mean-spectrum plot: on a linear-scale plot whose
    y-range is set by low-energy spikes, a perfectly usable ~100-200
    counts/channel region is rendered a few pixels above the zero line and
    reviewers systematically misjudge it. Computed once per run from the
    primary cube and every spectrum-aligned auxiliary operand (e.g. I0).
    """
    try:
        axis = np.asarray(axis, dtype=float)
        e = axis.size
        series = {"sample": np.asarray(data).reshape(-1, e).mean(0)}
        for label, arr in (aux or {}).items():
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == e:
                series[label] = arr.astype(float)
            elif arr.ndim >= 2 and arr.shape[-1] == e:
                series[label] = arr.reshape(-1, e).mean(0)
        edges = np.linspace(axis.min(), axis.max(), n_bands + 1)
        idx = np.clip(np.digitize(axis, edges) - 1, 0, n_bands - 1)
        header = "  ".join(f"{name:>14s}" for name in series)
        lines = [
            "### MEASURED FLUX BY BAND (deterministic — computed from the data)",
            f"Field-mean counts per channel in equal {axis_units} bands:",
            f"{'band':>19s}  {header}",
        ]
        for b in range(n_bands):
            m = idx == b
            if not m.any():
                continue
            vals = "  ".join(f"{float(v[m].mean()):>14.1f}" for v in series.values())
            lines.append(f"{edges[b]:>8.1f}-{edges[b + 1]:<10.1f}{vals}")
        lines.append(
            "These numbers OVERRIDE any visual estimate from the spectrum plot. "
            "Note analysis windows aggregate MANY channels, so usable SNR can be "
            "high even where the plotted curve looks near zero."
        )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - advisory block, never break the run
        return ""


def _render_attempt_history(entries: list) -> str:
    """Compact prior-attempt block for the combined result review.

    Gives the reviewer the trajectory (what each earlier attempt was allowed
    to change, what fraction of its maps passed, and why it was rejected) so
    its stance can move under mounting evidence — the hyperspectral analogue
    of the verification history the curve/image verifiers receive. Returns
    ``""`` on the first attempt, leaving that prompt unchanged.
    """
    if not entries:
        return ""
    lines = [
        "### PRIOR ATTEMPTS ON THIS TASK",
        "Earlier attempts and their review outcomes (constraint level 0 = "
        "plan-constrained; 1 = method questioned; 2 = method family "
        "abandoned):",
    ]
    for k, e in enumerate(entries, 1):
        frac = e.get("passed_fraction")
        frac_s = f"{frac:.2f}" if isinstance(frac, (int, float)) else "n/a"
        issues = "; ".join(
            f"{i.get('location')}: {str(i.get('problem'))[:220]}"
            for i in (e.get("issues_found") or [])[:3]
        ) or "none recorded"
        lines.append(
            f"- Attempt {k} (level {e.get('annealing_level')}, maps passed "
            f"{frac_s}): {issues}"
        )
    lines.append(
        "If methodologically DIFFERENT attempts keep converging on the same "
        "measured magnitude, weigh that agreement as evidence in its own "
        "right when judging the current result."
    )
    return "\n".join(lines)


def _append_literature_context(prompt: list, state: dict) -> None:
    """Append pre-fetched literature context (Channel A passthrough).

    Populated when the caller supplies ``literature_file`` (typically the
    orchestrator's ``search_literature`` tool). Framed as advisory context —
    it must not override what the data actually shows.
    """
    lit = state.get("literature_context")
    if not lit:
        return
    prompt.append(
        "\n\n--- Literature Context ---\n"
        "Relevant literature findings were provided for this analysis. Use "
        "them to inform interpretation and target selection, but do not let "
        "them override what the data actually shows.\n"
        f"{str(lit)}"
    )


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
        f"\n\n--- Analysis Objective ---\n"
        f"The overarching scientific objective of this analysis is: {objective}\n"
        f"Frame your analysis, model selection, and interpretation around "
        f"answering this objective. All findings should be evaluated in terms "
        f"of how they contribute to resolving this question."
    )


def build_code_generation_prompt(
    target_desc: str,
    h: int, w: int, e: int,
    axis_units: str,
    axis_start: float,
    axis_end: float,
    processing_note: str,
    hints: str | None = None,
    objective: str | None = None,
    required_outputs: list[str] | None = None,
    skill_implementation: str | None = None,
    reconstruction_available: bool = False,
    auxiliary_operands: dict | None = None,
) -> str:
    skill_section = ""
    if skill_implementation:
        skill_section = f"""

### 0. ACTIVE DOMAIN SKILL — IMPLEMENTATION GUIDANCE
{skill_implementation}

If the skill's guidance suggests map key names that differ from the
REQUIRED OUTPUTS block (when present), the REQUIRED OUTPUTS keys are
authoritative — produce maps under those exact names even if the skill
recommends different naming.
"""

    hints_section = ""
    if required_outputs:
        keys_str = ", ".join(f'"{n}"' for n in required_outputs)
        hints_section += f"""

### REQUIRED OUTPUTS
Your `maps` dict MUST contain the following keys (exact spelling):
{keys_str}

These keys represent quantities the user specifically asked for. Failure
to include any of them — or producing them with values that visually fail
quality checks (e.g. rail-gazing at parameter bounds, all-NaN, salt-and-
pepper noise) — will cause this task to fail and force a retry. You MAY
return additional keys, but the listed ones must be present and
physically meaningful.

If a previous attempt failed because a required output rail-gazed at
parameter bounds, the fix is usually one of: (a) widen the parameter
bounds, (b) use a better per-pixel initial guess (e.g. argmax of the
spectrum after smoothing), (c) switch lineshape (Lorentzian vs Gaussian
vs Voigt), or (d) add light per-pixel smoothing before fitting.
"""
    if objective:
        hints_section += f"""

### 5. ANALYSIS OBJECTIVE
The overarching scientific objective is: {objective}
Frame your feature extraction around answering this objective. Prioritize extracting parameters that are directly relevant to resolving this question.
"""
    if hints:
        hints_section += f"""

### {'6' if objective else '5'}. USER GUIDANCE
The user has indicated interest in: {hints}
Prioritize this guidance in your analysis, but also capture any other significant features present in the data.
"""
    _sig_extra = []
    if reconstruction_available:
        _sig_extra.append("reconstruction=None")
    if auxiliary_operands:
        _sig_extra.append("auxiliary=None")
    signature = "analyze_feature(data, axis" + "".join(f", {p}" for p in _sig_extra) + ")"

    auxiliary_section = ""
    if auxiliary_operands:
        operand_lines = "\n".join(
            f'  - `auxiliary["{name}"]`: array of shape {shape}'
            for name, shape in auxiliary_operands.items()
        )
        auxiliary_section = f"""

### OPTIONAL COMPANION OPERAND(S) — `auxiliary`
Your function is also handed `auxiliary`, a dict of user-supplied companion
dataset(s) that are shape-aligned with the primary data and so may be used
*numerically*:
{operand_lines}

Use these ONLY if your method needs them — e.g. divide the primary by a
reference/baseline spectrum, normalize against an I₀ reference, or use one
map to mask/weight another. The RAW `data` remains the base input; `auxiliary`
is an option, never required. Do NOT report findings about an auxiliary as if
it were a measurement — it is an operand for transforming the primary. Always
guard with `if auxiliary and "<name>" in auxiliary:` before use; `auxiliary`
may be `None` or empty.
"""

    reconstruction_section = ""
    if reconstruction_available:
        reconstruction_section = f"""

### OPTIONAL DENOISED INPUT — `reconstruction`
Your function is also handed a third argument, `reconstruction`: the rank-k
decomposition (NMF/PCA/ICA) approximation of the cube, same shape as `data`
({h}, {w}, {e}). It is a **denoised** version of the data, but it lives in the
decomposition's processed intensity space — so trust it for *shape*-based
quantities (peak position, peak width, edge onset) on noisy features, and use
the RAW `data` for anything intensity/quantification-related.

This is an **option, not a requirement**: `data` (raw) is always the base input.
Use `reconstruction` only when the raw signal for your feature is too noisy to
fit reliably; otherwise fit `data` directly. `reconstruction` may be `None` —
always guard with `if reconstruction is not None:` before using it.
"""

    return f"""
You are a Python Data Scientist specialized in Spectroscopy.
The standard NMF tool failed to model a spectral feature described as: "{target_desc}".

Your task: Write a Python function to mathematically model this feature.
Since complex features often require multiple parameters (e.g., Peak Position AND Peak Width), your function must be able to return MULTIPLE maps.
{skill_section}
### 1. DATA CONTEXT
- Input Data `hspy_data`: Shape ({h}, {w}, {e}) (Numpy array)
- X-Axis `axis`: Shape ({e},) (Numpy array). **Units: {axis_units}**
- **Axis Range:** {axis_start:.2f} to {axis_end:.2f} {axis_units}

### 2. EXECUTION ENVIRONMENT (STRICT)
Your code will run in a restricted `exec()` sandbox. 

**PRE-IMPORTED LIBRARIES (Available Globally):**
- `np`: The full NumPy library.
- `scipy`: The top-level SciPy module.
- `sklearn`: The top-level Scikit-Learn module.
- `lmfit`: Model-based curve fitting library (Parameters, Model, built-in models like GaussianModel, LorentzianModel, VoigtModel).

**PRE-IMPORTED FUNCTIONS (Direct Shortcuts):**
- `curve_fit`, `nnls` (from scipy.optimize)
- `linregress` (from scipy.stats)
- `find_peaks` (from scipy.signal)
- `gaussian_filter` (from scipy.ndimage)

**Performance Note:** `lmfit` adds per-fit setup overhead (~0.1-0.5ms) that can accumulate over thousands of pixels. For simple single-peak fits on large datasets, prefer raw `curve_fit` for speed. Use `lmfit` when you need its advantages: multi-peak composite models, parameter constraints/bounds, or built-in line shapes.

**SIZE BUDGET (single-process sandbox — no multiprocessing):** this cube has {h}x{w} = {h * w} pixels. A per-pixel iterative fit at ~2-5 ms costs roughly {max(1, (h * w) // 25000)}-{max(1, (h * w) // 12000)} minutes over the full frame, and the execution is time-capped — budget accordingly:
- fit ONLY the pixels your objective/mask actually needs (compute a mask first, fit inside it);
- vectorize or linearize wherever possible (batched linear algebra, log-linear fits, moment/centroid estimators) — vectorized NumPy also gets multithreaded BLAS for free, per-pixel Python loops do not;
- for full-frame maps, go coarse-to-fine: fit a spatially binned copy (e.g. 4x4 mean) first, then refine at full resolution only where the binned map shows structure.

### 3. CODING CONSTRAINTS
1. **NO External Imports:** Do not import `os`, `sys`, `matplotlib`, or `warnings`. The sandbox does not support them.
2. **SciPy Submodules:** If you need a specific SciPy submodule that is NOT in the shortcuts list (e.g., `scipy.interpolate` or `scipy.integrate`), you MUST write `import scipy.interpolate` **inside** your function definition before using it.
3. **Standard Math:** Use `np.exp`, `np.log`, etc., instead of the `math` library.
4. **NumPy 2.x:** This sandbox runs NumPy 2.x, where aliases removed in NumPy 2.0 raise `AttributeError` — notably use `np.trapezoid` (NOT `np.trapz`); prefer `scipy.integrate.trapezoid` for integration.
5. **Return Format:** You must return a dictionary, not a print statement or a plot.

### 4. YOUR GOAL
Write a function `{signature}` that:
1. Reshapes data to (pixels, energy).
2. Implements the specific math required.
3. Returns a DICTIONARY containing the results.

### ADDITIONAL NOTES
The variable `hspy_data` passed to your function contains: **{processing_note}**.
This is the RAW cube — no smoothing/clipping/despiking has been applied for you
(any cleaning done for decomposition is NOT in this array). Apply whatever
noise/spike/negative handling you judge necessary for a stable per-pixel fit —
the goal is fittable spectra — but do NOT erase the feature you are measuring.
If performing derivative-based operations (like `find_peaks` or `curve_fit`) on noisy data, apply appropriate smoothing to ensure convergence.
{reconstruction_section}{auxiliary_section}{hints_section}
### REQUIRED RETURN FORMAT
{{
    "maps": {{
        "Feature_Name_1": np.ndarray, 
        "Feature_Name_2": np.ndarray
    }},
    "units": {{                 
        "Feature_Name_1": "{axis_units}",
        "Feature_Name_2": "a.u."
    }},    
    "description": "Brief physics explanation"
}}

### RESPONSE FORMAT
Return a JSON object with:
- "code": The valid Python code string.
- "explanation": Brief logic summary.
"""

def _fmt(val, fmt=".4f"):
    """Format a numeric value, or return 'N/A'."""
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return "N/A"
    
def _sanitize_filename(text: str) -> str:
    """Helper to create safe filenames from labels."""
    # Replace spaces with underscores, remove non-alphanumeric chars except _ and -
    safe_text = re.sub(r'[^\w\-\_]', '', text.replace(" ", "_"))
    return safe_text


def _invoke_analyze_feature(func, data, axis, reconstruction=None, *, auxiliary=None):
    """Call the generated ``analyze_feature``, passing the optional operands
    (``reconstruction``, ``auxiliary``) only when the function declares them.

    Both are *options*, not a contract: a function that keeps the legacy
    two-argument signature is valid and simply fits the raw ``data`` (issue
    #219 for ``reconstruction``; #226 for ``auxiliary``). We never force an
    extra argument on a function that doesn't accept it — that would raise
    ``TypeError`` and break a perfectly good fit.

    ``auxiliary`` is a ``{label: array}`` dict of shape-aligned companion
    operands (e.g. a reference spectrum to divide by); passed by keyword only.
    """
    optional = {}
    if reconstruction is not None:
        optional["reconstruction"] = reconstruction
    if auxiliary:  # non-empty dict
        optional["auxiliary"] = auxiliary
    if not optional:
        return func(data, axis)

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return func(data, axis)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs = {
        name: val for name, val in optional.items()
        if name in params or accepts_var_kw
    }
    if kwargs:
        return func(data, axis, **kwargs)

    # Back-compat: a legacy 3-positional ``def f(d, a, r)`` whose 3rd param is
    # named something other than ``reconstruction`` — honored only when
    # reconstruction is the sole operand (the #219 shape).
    if list(optional) == ["reconstruction"]:
        positional = [
            p for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 3 or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
        ):
            return func(data, axis, reconstruction)
    return func(data, axis)

class RunPreprocessingController:
    """
    [🛠️ Tool Step] The decomposition tool's own input-prep substage.

    Cleans the cube *for decomposition* (despike / clip-iff-non-negative /
    mask-threshold) and emits the SNR + mask metadata the component planner
    consumes. It is no longer a standalone, universal preprocessing stage:
    the per-pixel codegen receives the RAW cube and owns its own fittability
    denoising. So this runs only when decomposition will run (gated on
    ``skip_decomposition``). See docs/hyperspectral_codegen_relocation.md.
    """
    def __init__(self, logger: logging.Logger, preprocessor: HyperspectralPreprocessingAgent):
        self.logger = logger
        self.preprocessor = preprocessor

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        self.logger.info("\n\n🛠️ --- DECOMPOSITION PREP --- 🛠️\n")

        # If decomposition is skipped, the per-pixel codegen uses the RAW cube
        # anyway — no decomposition-prep is needed. Just surface SNR/mask so any
        # downstream prompt that reads them stays populated.
        if state.get("skip_decomposition"):
            self.logger.info("Decomposition skipped → no decomposition-prep (codegen uses the raw cube).")
            if "preprocessing_mask" not in state:
                state["preprocessing_mask"] = np.ones(state["hspy_data"].shape[:2], dtype=bool)
            state.setdefault("data_quality", {"reasoning": "Decomposition skipped; raw data used downstream."})
            return state

        if not self.preprocessor:
            self.logger.warning("Preprocessing skipped: agent not initialized.")
            state["data_quality"] = {"reasoning": "Preprocessing skipped: agent not initialized."}
            return state

        # Check the runtime flag set by the agent
        if not state.get("settings", {}).get("run_preprocessing", True):
            self.logger.info("Preprocessing skipped for this refinement iteration (run_preprocessing=False).")
            self.logger.info("Calculating statistics on *current* masked data for the next step...")

            try:
                # We still need stats (like SNR and shape) for the *next* controller
                stats = self.preprocessor._calculate_statistics(state["hspy_data"])
                snr_value, snr_reasoning = self.preprocessor._calculate_snr(stats)
                state["data_quality"] = {
                    "snr_estimate": snr_value,
                    "reasoning": f"SNR of *current iteration* data: {snr_reasoning}"
                }
                # Preserve the original preprocessing mask if available;
                # only fall back to all-ones if no mask was ever computed.
                if "preprocessing_mask" not in state:
                    state["preprocessing_mask"] = np.ones(state["hspy_data"].shape[:2], dtype=bool)
                self.logger.info(f"✅ Tool Complete: Statistics calculated. SNR = {snr_value:.2f}")
                return state
            except Exception as e:
                self.logger.error(f"❌ Tool Failed: Stat calculation on refinement data failed: {e}", exc_info=True)
                state["error_dict"] = {"error": "Stat calculation on refinement data failed", "details": str(e)}
                return state

        try:
            processed_data, mask, data_quality = self.preprocessor.run_preprocessing(
                state["hspy_data"],
                state["system_info"]
            )
            state["hspy_data"] = processed_data
            state["preprocessing_mask"] = mask
            state["data_quality"] = data_quality
            self.logger.info("✅ Tool Complete: Full preprocessing finished.")
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Preprocessing failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Preprocessing failed", "details": str(e)}
        return state

class GetInitialComponentParamsController:
    """
    [🧠 LLM Step]
    Asks LLM for initial n_components and decomposition method (NMF, PCA, or ICA).
    """
    VALID_METHODS = ("nmf", "pca", "ica")

    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: ESTIMATE INITIAL N_COMPONENTS & METHOD --- 🧠\n")

        h, w, e = state["hspy_data"].shape
        data_quality = state.get("data_quality", {})
        if not data_quality.get("snr_estimate"):
            # Decomposition-prep now runs AFTER this estimate (it is the
            # decomposition's own prep substage), so derive SNR from the raw cube
            # here. See docs/hyperspectral_codegen_relocation.md.
            try:
                _snr = float(tools.estimate_global_snr(state["hspy_data"]))
                data_quality = {
                    "snr_estimate": round(_snr, 1),
                    "reasoning": f"SNR estimated from the raw cube (≈{_snr:.1f}); "
                                 f"decomposition preprocessing runs next.",
                }
                state["data_quality"] = data_quality
            except Exception as _e:
                self.logger.debug(f"raw SNR estimate failed: {_e}")
        axis_spec = resolve_axis_spec(state.get("system_info"))

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Hyperspectral Data Information ---")
        prompt_parts.append(f"Data dimensions: {describe_axes_for_prompt((h, w, e), axis_spec)}")
        prompt_parts.append(f"\n--- Data Quality Assessment (from Preprocessor) ---")
        prompt_parts.append(f"- Robust SNR Estimate: {data_quality.get('snr_estimate', 'N/A')}")
        prompt_parts.append(f"- Assessment: {data_quality.get('reasoning', 'N/A')}")

        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        _append_objective_context(prompt_parts, state)

        if state.get("analysis_hints"):
            prompt_parts.append(
                f"\n\n--- User Guidance ---\n"
                f"The user has provided the following guidance for this analysis. "
                f"Prioritize these suggestions but also report any other significant features you discover.\n"
                f"{state['analysis_hints']}"
            )

        _append_skill_context(prompt_parts, state, "planning")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)

        prompt_parts.append("\n\nBased on the system description and data characteristics, choose the decomposition method and estimate the optimal number of spectral components.")

        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict:
                self.logger.warning(f"LLM initial estimation failed: {error_dict}. Using defaults.")
                n_components = 4
                selected_method = "nmf"
                run_decomposition = True
            else:
                # Defensive default: missing field => run decomposition. The
                # skip path is opt-in via an explicit objective-driven decision.
                run_decomposition = bool(result_json.get('run_decomposition', True))
                n_components = result_json.get('estimated_components', 4)
                selected_method = result_json.get('method', 'nmf').lower().strip()
                reasoning = result_json.get('reasoning', 'No reasoning provided.')
                self.logger.info(
                    f"LLM initial estimate: run_decomposition={run_decomposition}, "
                    f"method={selected_method}, {n_components} components. "
                    f"Reasoning: {reasoning}"
                )

                print("\n" + "="*80)
                print("🧠 LLM REASONING (GetInitialComponentParamsController)")
                print(f"  Run decomposition: {run_decomposition}")
                print(f"  Selected method: {selected_method.upper()}")
                print(f"  Suggested n_components: {n_components}")
                print(f"  Explanation: {reasoning}")
                print("="*80 + "\n")

                if not (isinstance(n_components, int) and 2 <= n_components <= 15):
                    self.logger.warning(f"Invalid LLM estimate {n_components}, using default 4.")
                    n_components = 4

                if selected_method not in self.VALID_METHODS:
                    self.logger.warning(f"Invalid LLM method '{selected_method}', using default 'nmf'.")
                    selected_method = "nmf"

            state["initial_n_components"] = n_components
            state["selected_method"] = selected_method
            state["settings"]["method"] = selected_method
            state["skip_decomposition"] = not run_decomposition
            if not run_decomposition:
                self.logger.info(
                    "🚦 LLM gate selected SKIP: proceeding directly to dynamic "
                    "analysis without decomposition."
                )
            self.logger.info(
                f"✅ LLM Step Complete: skip_decomposition={state['skip_decomposition']}, "
                f"method={selected_method.upper()}, initial components={n_components}."
            )

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Initial component estimation: {e}", exc_info=True)
            state["initial_n_components"] = 4
            state["selected_method"] = "nmf"
            state["settings"]["method"] = "nmf"
            state["skip_decomposition"] = False

        return state

class RunComponentTestLoopController:
    """
    [🛠️ Tool Step]
    Loops from min to max components, runs spectral unmixing.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        if state.get("skip_decomposition"):
            self.logger.info("Skip-decomposition gate active — bypassing component test loop.")
            return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: COMPONENT TEST LOOP --- 🛠️\n")

        method_name = state.get("settings", {}).get("method", "nmf").upper()

        # ICA has no meaningful reconstruction-error trend in n_components, so
        # the elbow loop is uninformative. Skip it and let the downstream
        # selection controller fall back to the LLM's initial estimate.
        if method_name == "ICA":
            self.logger.info(
                "ICA mode: skipping component test loop (no informative elbow). "
                "Final n_components will use the LLM's initial estimate."
            )
            state["component_test_range"] = []
            state["component_test_errors"] = []
            state["component_test_visuals"] = []
            return state

        tool_settings = self.settings.copy()
        for key in AGENT_METADATA_KEYS_TO_STRIP:
            tool_settings.pop(key, None)

        initial_estimate = state.get("initial_n_components", 4)
        min_c = self.settings.get('min_auto_components', 2)
        max_c = self.settings.get('max_auto_components', min(initial_estimate + 4, 12))
        component_range = list(range(min_c, max_c + 1))

        errors = []
        visual_examples = []

        for n_comp in component_range:
            try:
                components, abundance_maps, error = tools.run_spectral_unmixing(
                    state["hspy_data"], n_comp, tool_settings, self.logger
                )
                errors.append(error)
                self.logger.info(f"  (Loop {n_comp}/{max_c}): Error = {error:.4f}")

                if n_comp == min_c or n_comp == max_c or n_comp == initial_estimate:
                    summary_bytes = tools.create_nmf_summary_plot(
                        components, abundance_maps, n_comp, state["system_info"], self.logger,
                        method_name=method_name
                    )
                    if summary_bytes:
                        visual_examples.append({
                            'n_components': n_comp,
                            'image': summary_bytes,
                            'label': f"{n_comp} Components ({'Min' if n_comp==min_c else 'Max' if n_comp==max_c else 'Initial Estimate'})"
                        })
                        
                        try:
                            output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                            os.makedirs(output_dir, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            iter_title = _sanitize_filename(state.get('iteration_title', 'iter'))
                            filename = f"{iter_title}_TestLoop_{n_comp}comp_{timestamp}.jpeg"
                            filepath = os.path.join(output_dir, filename)
                            
                            with open(filepath, 'wb') as f:
                                f.write(summary_bytes)
                            self.logger.info(f"📸 Saved component test plot to: {filepath}")
                        except Exception as e:
                            self.logger.warning(f"Failed to save component test plot: {e}")
            except Exception as e:
                self.logger.warning(f"  (Loop {n_comp}/{max_c}): Failed. {e}")
                errors.append(np.inf)
        
        state["component_test_range"] = component_range
        state["component_test_errors"] = errors
        state["component_test_visuals"] = visual_examples
        self.logger.info("✅ Tool Complete: Component test loop finished.")
        return state

class CreateElbowPlotController:
    """
    [🛠️ Tool Step]
    Generates the elbow plot.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        if state.get("skip_decomposition"):
            self.logger.info("Skip-decomposition gate active — bypassing elbow plot.")
            state["elbow_plot_bytes"] = None
            return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE ELBOW PLOT --- 🛠️\n")

        method_name = state.get("settings", {}).get("method", "nmf").upper()
        plot_bytes = tools.create_elbow_plot(
            state["component_test_range"],
            state["component_test_errors"],
            self.logger,
            method_name=method_name
        )
        state["elbow_plot_bytes"] = plot_bytes
        if plot_bytes:
            self.logger.info("✅ Tool Complete: Elbow plot created.")
            try:
                output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                iter_title = _sanitize_filename(state.get('iteration_title', 'iter'))
                filename = f"{iter_title}_Elbow_Plot_{timestamp}.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(plot_bytes)
                self.logger.info(f"📸 Saved elbow plot to: {filepath}")
            except Exception as e:
                self.logger.warning(f"Failed to save elbow plot: {e}")
        else:
            self.logger.warning("Tool Warning: Elbow plot creation failed.")
        return state

class GetFinalComponentSelectionController:
    """
    [🧠 LLM Step]
    Asks LLM to pick the best n_components.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        if state.get("skip_decomposition"):
            self.logger.info("Skip-decomposition gate active — bypassing final component selection.")
            return state
        self.logger.info("\n\n🧠 --- LLM STEP: SELECT FINAL N_COMPONENTS --- 🧠\n")

        initial_estimate = state.get("initial_n_components", 4)
        component_range = state.get("component_test_range", [])
        
        if not state.get("elbow_plot_bytes") or not state.get("component_test_visuals"):
            self.logger.warning("Missing elbow plot or visual examples. Using initial estimate.")
            state["final_n_components"] = initial_estimate
            return state

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Context ---")
        prompt_parts.append(f"Initial LLM estimate: {initial_estimate} components")
        prompt_parts.append(f"Tested component range: {component_range}")
        
        prompt_parts.append(f"\n\n--- Quantitative Analysis: Reconstruction Error ---")
        prompt_parts.append("Elbow Plot (Error vs. Number of Components):")
        prompt_parts.append({"mime_type": "image/jpeg", "data": state["elbow_plot_bytes"]})
        
        prompt_parts.append(f"\n\n--- Qualitative Analysis: Visual Examples ---")
        for viz in state.get("component_test_visuals", []):
            prompt_parts.append(f"\n\n**{viz['label']}:**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": viz['image']})

        _append_objective_context(prompt_parts, state)

        if state.get("analysis_hints"):
            prompt_parts.append(
                f"\n\n--- User Guidance ---\n"
                f"The user has provided the following guidance for this analysis. "
                f"Prioritize these suggestions but also report any other significant features you discover.\n"
                f"{state['analysis_hints']}"
            )

        _append_auxiliary_context(prompt_parts, state)

        prompt_parts.append(f"\n\nBased on the elbow plot AND the visual examples, decide the optimal number of components.")

        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM final selection failed: {error_dict}. Using initial estimate.")
                final_n_components = initial_estimate
            else:
                final_n_components = result_json.get('final_components', initial_estimate)
                reasoning = result_json.get('reasoning', 'No reasoning provided.')
                self.logger.info(f"LLM final decision: {final_n_components} components. Reasoning: {reasoning}")

                print("\n" + "="*80)
                print("🧠 LLM REASONING (GetFinalComponentSelectionController)")
                print(f"  Final n_components: {final_n_components}")
                print(f"  Explanation: {reasoning}")
                print("="*80 + "\n")

                if not (isinstance(final_n_components, int) and final_n_components in component_range):
                    self.logger.warning(f"Invalid LLM final choice {final_n_components}, using initial estimate.")
                    final_n_components = initial_estimate
            
            state["final_n_components"] = final_n_components
            self.logger.info(f"✅ LLM Step Complete: Final component selection = {final_n_components}.")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Final component selection: {e}", exc_info=True)
            state["final_n_components"] = initial_estimate 
            
        return state

class RunFinalSpectralUnmixingController:
    """
    [🛠️ Tool Step]
    Runs spectral unmixing one last time.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        if state.get("skip_decomposition"):
            self.logger.info("Skip-decomposition gate active — bypassing final spectral unmixing.")
            return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: FINAL SPECTRAL UNMIXING --- 🛠️\n")

        final_n_components = state.get("final_n_components")
        if not final_n_components:
            final_n_components = self.settings.get('n_components', 4)
            self.logger.warning(f"Auto-selection failed. Using fixed component count: {final_n_components}")
            state["final_n_components"] = final_n_components

        tool_settings = self.settings.copy()
        for key in AGENT_METADATA_KEYS_TO_STRIP:
            tool_settings.pop(key, None)
            
        try:
            components, abundance_maps, error = tools.run_spectral_unmixing(
                state["hspy_data"], final_n_components, tool_settings, self.logger
            )
            state["final_components"] = components
            state["final_abundance_maps"] = abundance_maps
            state["final_reconstruction_error"] = error
            self.logger.info(f"✅ Tool Complete: Final unmixing done. Error: {error:.4f}")
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Final unmixing: {e}", exc_info=True)
            state["error_dict"] = {"error": "Final spectral unmixing failed", "details": str(e)}
        return state

class CreateAnalysisPlotsController:
    """
    [🛠️ Tool Step]
    Generates high-quality validation plots using reconstruction comparison.
    
    UPDATED: Now uses create_validated_component_pair_reconstruction() 
    to fix the "all components look the same" problem.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        if state.get("skip_decomposition"):
            self.logger.info("Skip-decomposition gate active — bypassing analysis plots.")
            return state

        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE ANALYSIS PLOTS --- 🛠️\n")

        components = state.get("final_components")
        abundance_maps = state.get("final_abundance_maps")

        iter_title_raw = state.get("iteration_title", "Global_Analysis")
        iter_prefix = _sanitize_filename(iter_title_raw)

        if components is None or abundance_maps is None:
            self.logger.warning("Skipping plot creation: final components/maps not found.")
            return state

        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
        method_name = state.get("settings", {}).get("method", "nmf").upper()

        final_plots = []
        validated_bytes_list = []

        if method_name in ("PCA", "ICA"):
            # --- PCA / ICA MODE: Summary-only (no per-component validation) ---
            # Both produce signed components that don't map directly to physical
            # phases, so we skip per-component reconstruction validation.
            self.logger.info(
                f"{method_name} mode: Generating summary plot for {components.shape[0]} components..."
            )
            summary_bytes = tools.create_nmf_summary_plot(
                components, abundance_maps, components.shape[0],
                state["system_info"], self.logger, method_name=method_name
            )
            if summary_bytes:
                label = f"{method_name} Summary Grid"
                final_plots.append({'label': label, 'bytes': summary_bytes, 'metrics': {}})
                tools.save_image_bytes(
                    summary_bytes, output_dir,
                    f"{iter_prefix}_{_sanitize_filename(label)}.jpeg", self.logger
                )
                state["analysis_images"].append({"label": label, "data": summary_bytes})

        else:
            # --- NMF MODE: Per-component validation plots + summary grid ---
            self.logger.info(
                f"Generating High-Purity Reconstruction Validation Plots "
                f"for {components.shape[0]} components..."
            )

            for i in range(components.shape[0]):
                result = tools.create_validated_component_pair_reconstruction(
                    state["hspy_data"],       # Raw data
                    components,               # ALL components (needed for reconstruction)
                    abundance_maps,           # ALL abundance maps
                    i,                        # Current component index
                    state["system_info"],
                    self.logger,
                    purity_percentile=90.0,   # Top 10% (adjustable)
                    show_basis_component=True, # Show orange reference line
                    method_name=method_name
                )

                if result is not None:
                    plot_bytes, metrics = result
                else:
                    plot_bytes, metrics = None, {}

                if plot_bytes:
                    label = f"Component {i+1} Analysis"
                    final_plots.append({'label': label, 'bytes': plot_bytes, 'metrics': metrics})
                    validated_bytes_list.append(plot_bytes)

                    # Save using tool
                    label_safe = _sanitize_filename(label)
                    tools.save_image_bytes(
                        plot_bytes, output_dir,
                        f"{iter_prefix}_{label_safe}.jpeg", self.logger
                    )

            for plot in final_plots:
                state["analysis_images"].append({"label": plot['label'], "data": plot['bytes'], "metrics": plot.get('metrics', {})})

            # Create Summary Grid from validated plots
            try:
                self.logger.info("  (Tool Info: Stitching validated plots into Summary Grid...)")
                summary_bytes = tools.create_image_grid(validated_bytes_list, self.logger)

                if summary_bytes:
                    label = f"{method_name} Summary Grid"
                    tools.save_image_bytes(
                        summary_bytes, output_dir,
                        f"{iter_prefix}_{_sanitize_filename(label)}.jpeg", self.logger
                    )

                    state["analysis_images"].append({"label": label, "data": summary_bytes})

            except Exception as e:
                self.logger.warning(f"Failed to create/save {method_name} summary plot: {e}")

        state["component_pair_plots"] = final_plots

        # --- 3. Structure Overlays (UNCHANGED) ---
        if state.get("structure_image_path"):
            try:
                # Load image
                structure_img = load_image(state["structure_image_path"])
                if structure_img.ndim == 3:
                    structure_img = cv2.cvtColor(structure_img, cv2.COLOR_RGB2GRAY)
                
                # Create overlays
                overlay_bytes = tools.create_multi_abundance_overlays(
                    structure_img, abundance_maps, threshold_percentile=85.0 
                )
                state["structure_overlay_bytes"] = overlay_bytes
                
                if overlay_bytes:
                    label = "Structure-Abundance Overlays"
                    tools.save_image_bytes(
                        overlay_bytes, output_dir, 
                        f"{iter_prefix}_{_sanitize_filename(label)}.jpeg", self.logger
                    )
                    state["analysis_images"].append({"label": label, "data": overlay_bytes})
                
            except Exception as e:
                self.logger.warning(f"Failed to create structure overlays: {e}")

        self.logger.info("✅ Tool Complete: Final analysis plots created and saved.")
        return state

class DecompositionController:
    """
    [🧩 Composite Step] The cohesive spectral-decomposition step.

    Wraps the whole decomposition sub-pipeline — its own input prep, initial
    component/method estimation, the elbow scan, the LLM component selection,
    the final unmixing, and the validation plots — behind one controller that
    leaves the decomposition contract in ``state`` (``final_components``,
    ``final_abundance_maps``, ``final_reconstruction_error``, ``data_quality``
    SNR, ``preprocessing_mask``). The per-pixel codegen downstream reads that
    contract; this remains a model-bearing step *inside* the iteration loop, so
    the decompose↔re-plan feedback edge is preserved (see issue #220 and
    docs/hyperspectral_codegen_relocation.md).

    Internally it composes the same sub-controllers that previously sat as
    separate pipeline entries, run in order. Each sub-controller already guards
    on ``error_dict`` and ``skip_decomposition``, so the composite is a thin
    sequential driver — gating on ``auto_components`` / ``run_preprocessing`` is
    applied here at construction time exactly as the pipeline factory did before.
    """
    def __init__(self, model, logger, generation_config, safety_settings,
                 settings: dict, preprocessor: HyperspectralPreprocessingAgent,
                 parse_fn: Callable):
        self.logger = logger

        stages = []
        # [🧠 LLM] Initial component/method guess — also decides
        # skip_decomposition, using SNR estimated from the raw cube
        # (prep runs after this).
        if settings.get('auto_components', True):
            stages.append(GetInitialComponentParamsController(
                model, logger, generation_config, safety_settings, parse_fn
            ))

        # [🛠️ Tool] Decomposition's own prep substage — runs after the skip
        # decision; internally gated on skip_decomposition.
        if settings.get('run_preprocessing', True):
            stages.append(RunPreprocessingController(logger, preprocessor))

        # Rest of the auto-component workflow operates on the cleaned cube.
        if settings.get('auto_components', True):
            stages.append(RunComponentTestLoopController(logger, settings))
            stages.append(CreateElbowPlotController(logger, settings))
            stages.append(GetFinalComponentSelectionController(
                model, logger, generation_config, safety_settings, parse_fn
            ))

        stages.append(RunFinalSpectralUnmixingController(logger, settings))
        stages.append(CreateAnalysisPlotsController(logger, settings))

        self.stages = stages

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        self.logger.info("\n\n🧩 --- SPECTRAL DECOMPOSITION --- 🧩\n")
        for stage in self.stages:
            state = stage.execute(state)
            if state.get("error_dict"):
                self.logger.warning(
                    f"Decomposition halted: {stage.__class__.__name__} reported an error."
                )
                break
        return state

class BuildHyperspectralPromptController:
    """
    [📝 Prep Step]
    Assembles per-iteration decomposition results into the interpretation
    prompt for this iteration. Distinct from
    BuildHolisticSynthesisPromptController, which assembles the synthesis
    prompt run after dynamic analysis completes.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        self.logger.info("\n\n📝 --- PREP STEP: BUILDING ITERATION INTERPRETATION PROMPT --- 📝\n")
        
        # 1. Base Instruction & Context
        prompt_parts = [state["instruction_prompt"]]

        # 2. Skip-decomposition framing (when the gate selected direct analysis)
        if state.get("skip_decomposition"):
            prompt_parts.append("""

### 🚦 CONTEXT: Unsupervised decomposition was skipped

The user's objective is best served by direct per-pixel quantitative
analysis (e.g. curve fitting, peak finding, integration) rather than
unsupervised source separation. No NMF/PCA/ICA components are available.
Frame your interpretation around the raw data characteristics and propose
`custom_code` refinement targets that operate per-pixel on the (preprocessed)
raw spectra.
""")

        # 3. Data Metadata
        h, w, e = state["hspy_data"].shape
        axis_spec = resolve_axis_spec(state.get("system_info"))
        _, energy_xlabel, _ = tools.create_axis(e, state["system_info"], axis_index=2)

        metadata_info = f"""

Hyperspectral Data Information:
- Data shape: ({h}, {w}, {e}) = {describe_axes_for_prompt((h, w, e), axis_spec)}
- X-axis: {energy_xlabel}
"""
        
        if state.get("final_components") is not None:
            metadata_info += f"""- Spectral unmixing method: {state['settings'].get('method', 'nmf').upper()}
- Number of components: {state['final_n_components']}
- Final Reconstruction Error: {_fmt(state.get('final_reconstruction_error'))}
"""
        
        prompt_parts.append(metadata_info)

        # 4. Component Analysis (Dynamic Instructions based on depth and method)
        current_depth = state.get("current_depth", 0)
        method_name = state.get("settings", {}).get("method", "nmf").upper()

        if state.get("component_pair_plots"):
            prompt_parts.append("\n\n**Spectral Component Analysis:**")

            if method_name == "PCA":
                # PCA mode: summary-only, exploratory framing
                prompt_parts.append(f"""
Below is a PCA decomposition summary of the dataset.
Top row: Principal Component spectra. Bottom row: Corresponding spatial loading maps.
PCA components are exploratory — they capture variance directions, not necessarily physical phases.
Identify spectral features of interest (peaks, edges, shifts) for custom code modeling.
""")
                # Append only the summary image (no per-component metrics)
                for plot in state["component_pair_plots"]:
                    prompt_parts.append(f"\n{plot['label']}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": plot['bytes']})
            elif method_name == "ICA":
                # ICA mode: summary-only, independent-source framing
                prompt_parts.append(f"""
Below is an ICA decomposition summary of the dataset.
Top row: Independent Component spectra. Bottom row: Corresponding spatial loading maps.
ICA components represent statistically independent sources rather than variance directions;
they may overlap spectrally and can have signed loadings. Use them to identify candidate
distinct contributions for custom code modeling, but do not treat them as physical phases
without further validation.
""")
                # Append only the summary image (no per-component metrics)
                for plot in state["component_pair_plots"]:
                    prompt_parts.append(f"\n{plot['label']}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": plot['bytes']})
            else:
                # NMF mode: per-component validation with metrics
                if current_depth == 0:
                    prompt_parts.append(f"""
Below are the {method_name} components extracted from the global dataset.
For each component, the LEFT image is the Spectral Signature and the RIGHT image is the Spatial Abundance.
""")
                else:
                    prompt_parts.append(SPECTROSCOPY_VALIDATION_INTERPRETATION_INSTRUCTIONS)

                # Append the plots with per-component metrics
                for plot in state["component_pair_plots"]:
                    metrics = plot.get('metrics', {})
                    prompt_parts.append(f"\n{plot['label']}:")
                    if metrics:
                        prompt_parts.append(f"  Reconstruction RMSE: {_fmt(metrics.get('rmse'))}")
                        prompt_parts.append(f"  Cosine Similarity (Measured vs Reconstruction): {_fmt(metrics.get('cosine_similarity'))}")
                        prompt_parts.append(f"  Cosine Similarity (Measured vs Basis): {_fmt(metrics.get('basis_cosine_similarity'))}")
                        prompt_parts.append(f"  High-Purity Region: {_fmt(metrics.get('purity_pixel_percent'), '.1f')}% of pixels")
                        prompt_parts.append(f"  Residual Autocorrelation: {_fmt(metrics.get('residual_autocorrelation'), '.3f')}")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": plot['bytes']})

        # 5. Structure Overlays (if available)
        if state.get("structure_overlay_bytes"):
            prompt_parts.append("""

**Structure-Abundance Correlation Analysis:**
Overlays showing where components are concentrated on the structural image.
""")
            prompt_parts.append({"mime_type": "image/jpeg", "data": state["structure_overlay_bytes"]})
            
            # Ensure storage for synthesis
            found = False
            for img in state.get("analysis_images", []):
                if img.get("label") == "Structure-Abundance Overlays": 
                    found = True
            if not found:
                state["analysis_images"].append({
                    "label": "Structure-Abundance Overlays",
                    "data": state["structure_overlay_bytes"]
                })

        # 6. System Metadata
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\nAdditional System Information (Metadata):\n{sys_info_str}")

        # 7. Domain skill context
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_literature_context(prompt_parts, state)

        # 8. Final instructions
        prompt_parts.append("\n\nProvide your analysis in the requested JSON format.")

        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Iteration interpretation prompt is ready.")
        return state


class SelectRefinementTargetController:
    """
    [🧠 LLM Step]
    Asks the LLM if a refinement (zoom-in) is needed and where.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = SPECTROSCOPY_REFINEMENT_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        # Log header reflects whether this is "select a plan from scratch"
        # (skip-decomposition mode — no prior analysis to refine) vs.
        # "refine the existing decomposition results".
        skip_mode = bool(state.get("skip_decomposition"))
        header = (
            "🧠 --- LLM STEP: SELECT ANALYSIS PLAN --- 🧠"
            if skip_mode
            else "🧠 --- LLM STEP: SELECT REFINEMENT TARGET --- 🧠"
        )
        self.logger.info(f"\n\n{header}\n")

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Current Analysis: {state.get('iteration_title', 'Analysis')} ---")

        if skip_mode:
            prompt_parts.append("""

🚦 NOTE: Unsupervised decomposition was skipped for this run because the
user's objective specifies a direct per-pixel measurement. Only `custom_code`
refinement targets are meaningful here — do not request `spatial` or
`spectral` zoom refinement.
""")

        # Give the planner MINIMAL awareness of the downstream code tools (names +
        # when_to_use) plus a method-level discipline directive, so the plan stops
        # prescribing hand-rolled physics that overrides those tools. See
        # _planning_tool_awareness.
        tool_note = _planning_tool_awareness(state)
        if tool_note:
            prompt_parts.append(tool_note)

        # Add system info
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        # Add plots from the current iteration
        prompt_parts.append("\n\n--- Analysis Results ---")
        analysis_images = state.get("analysis_images", [])
        if not analysis_images:
            # In skip-decomposition mode the absence of analysis images is
            # expected (no decomposition → no decomposition plots), not a
            # warning condition. In normal mode it indicates a real upstream
            # issue worth flagging.
            if skip_mode:
                self.logger.info("Skip-decomposition mode — no decomposition images to review (expected).")
            else:
                self.logger.warning("No analysis images found for refinement selection.")
            prompt_parts.append("(No visual results available)")

        for img in analysis_images:
            image_bytes = img.get('data') or img.get('bytes')
            if image_bytes:
                prompt_parts.append(f"\n{img['label']}:")
                # Surface metrics if available (from component plots)
                metrics = img.get('metrics', {})
                if metrics:
                    prompt_parts.append(f"  CosSim: {_fmt(metrics.get('cosine_similarity'))} | Residual AutoCorr: {_fmt(metrics.get('residual_autocorrelation'), '.3f')}")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
            else:
                self.logger.warning(f"Could not find image bytes for plot: {img.get('label')}")

        _append_objective_context(prompt_parts, state)

        if state.get("analysis_hints"):
            prompt_parts.append(
                f"\n\n--- User Guidance ---\n"
                f"The user has provided the following guidance for this analysis. "
                f"Prioritize these suggestions but also report any other significant features you discover.\n"
                f"{state['analysis_hints']}"
            )

        _append_skill_context(prompt_parts, state, "planning")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_literature_context(prompt_parts, state)

        prompt_parts.append("\n\nBased on these results, decide if a focused refinement is needed.")

        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict:
                self.logger.error(f"LLM refinement selection failed: {error_dict}. Stopping loop.")
                state["refinement_decision"] = {"refinement_needed": False, "reasoning": "LLM selection failed."}
                return state

            # Get Raw Targets
            raw_targets = result_json.get("targets", [])
            is_needed = result_json.get("refinement_needed", False)

            # Priority Filtering (Custom Code vs Standard)
            custom_code_targets = [t for t in raw_targets if t.get('type') == 'custom_code']
            standard_targets = [t for t in raw_targets if t.get('type') != 'custom_code']
            
            final_targets = []
            requires_custom_code = False
            
            if custom_code_targets:
                # Winner-Takes-All: If code is needed, focus ONLY on that.
                # We pick the first custom target and ignore standard zooms for this turn.
                top_target = custom_code_targets[0]
                label = "Analysis Plan" if skip_mode else "Priority Target Selected"
                self.logger.info(f"🎯 {label} (Custom Code): {top_target.get('description')}")
                final_targets = [top_target]
                requires_custom_code = True
            else:
                # Otherwise, proceed with standard targets
                final_targets = standard_targets
                requires_custom_code = False

            # Store the final decision with the filtered targets and the FLAG
            state["refinement_decision"] = {
                "refinement_needed": is_needed,
                "reasoning": result_json.get("reasoning", "No reasoning provided."),
                "targets": final_targets,
                "requires_custom_code": requires_custom_code
            }

            step_label = "Analysis plan" if skip_mode else "Refinement decision"
            self.logger.info(f"✅ LLM Step Complete: {step_label}: {state['refinement_decision']['reasoning']}")

            print("\n" + "="*80)
            print("🧠 LLM REASONING (SelectRefinementTargetController)")
            if skip_mode:
                print(f"  Analysis Plan Ready: {is_needed}")
            else:
                print(f"  Refinement Needed: {is_needed}")
            print(f"  Custom Code Triggered: {requires_custom_code}")
            print(f"  Explanation: {state['refinement_decision']['reasoning']}")
            print(f"  Targets Found: {len(final_targets)}")
            if final_targets:
                for i, t in enumerate(final_targets):
                    print(f"    Target {i+1} ({t.get('type')}): {t.get('description')}")
            print("="*80 + "\n")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Refinement selection: {e}", exc_info=True)
            state["refinement_decision"] = {"refinement_needed": False, "reasoning": f"Exception: {e}"}
            
        return state
    

class BuildHolisticSynthesisPromptController:
    """
    [📝 Prep Step]
    Assembles ALL iteration results into the final prompt for synthesis.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.instructions = SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n📝 --- PREP STEP: BUILDING FINAL SYNTHESIS PROMPT --- 📝\n")
        
        prompt_parts = [self.instructions]
        
        all_results = state.get("all_iteration_results", [])
        if not all_results:
            self.logger.error("No iteration results found to synthesize.")
            state["error_dict"] = {"error": "No iteration results found for synthesis."}
            return state

        # 1. System Info
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        _append_objective_context(prompt_parts, state)

        if state.get("analysis_hints"):
            prompt_parts.append(
                f"\n\n--- User Guidance ---\n"
                f"The user has provided the following guidance for this analysis. "
                f"Prioritize these suggestions but also report any other significant features you discover.\n"
                f"{state['analysis_hints']}"
            )

        # 2. Build Context for Each Iteration
        all_images = []

        for i, iter_result in enumerate(all_results):
            raw_title = iter_result.get('iteration_title', f'Iteration_{i}')
            iter_ref_id = _sanitize_filename(raw_title)
            
            prompt_parts.append(f"\n\n### SECTION {i+1}: {raw_title}")

            # --- DYNAMIC ANALYSIS INJECTION
            # Retrieve the list of features generated by the custom code
            custom_meta_list = iter_result.get("custom_analysis_metadata_list")
            
            if custom_meta_list:
                prompt_parts.append(f"\n**🔍 DYNAMIC ANALYSIS FINDINGS (Physics-Based Mapping):**")
                prompt_parts.append("The following features were mathematically modeled using custom Python code:")
                
                # Loop through every feature in the list
                for idx, meta in enumerate(custom_meta_list, 1):
                    name = meta.get('name', 'Custom Feature')
                    desc = meta.get('description', 'N/A')
                    units = meta.get('units', 'a.u.')
                    stats = meta.get('stats', {})
                    
                    prompt_parts.append(f"\n   **Feature {idx}: {name}**")
                    prompt_parts.append(f"   - Physical Interpretation: {desc}")
                    prompt_parts.append(f"   - Units: {units}")
                    
                    # Crash Fix: Use .get(key, 0.0) to handle missing stats gracefully
                    if stats:
                        s_min = stats.get('min', 0.0)
                        s_max = stats.get('max', 0.0)
                        s_mean = stats.get('mean', 0.0)
                        prompt_parts.append(f"   - Statistics: Min {s_min:.2f}, Max {s_max:.2f}, Mean {s_mean:.2f}")
                
                prompt_parts.append("\n-> **INSTRUCTION:** Use these specific physical maps to validate or correct the NMF results.")
            
            # Text Summary (Standard NMF Analysis)
            iter_analysis = iter_result.get('iteration_analysis_text')
            if iter_analysis:
                prompt_parts.append(f"\n**Previous NMF Analysis Summary:**\n{iter_analysis}")
            
            # Visual Evidence
            iter_images = iter_result.get('analysis_images', [])
            if iter_images:
                prompt_parts.append(f"\n**Visual Evidence for {raw_title}:**")
                for img in iter_images:
                    image_bytes = img.get('data') or img.get('bytes')
                    raw_label = img.get('label', 'Unknown_Plot')

                    if image_bytes:
                        # Create a unique semantic ID for citation
                        unique_ref = f"[{iter_ref_id}] {raw_label}"

                        prompt_parts.append(f"\n**{unique_ref}**")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})

                        # Update label in the image object itself for the Report Generation step
                        # (This ensures the HTML report filters correctly)
                        img['label'] = unique_ref
                        all_images.append(img)

        # 3. Domain skill context
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_literature_context(prompt_parts, state)

        # 4. EXPLICIT REPORTING INSTRUCTIONS
        prompt_parts.append("\n\n### 📝 CRITICAL REPORTING INSTRUCTIONS")
        prompt_parts.append("1. **AT THE END of your 'detailed_analysis' text**, you MUST append a section titled **'### Key Evidence'**.")
        prompt_parts.append("2. In that section, you MUST list the supporting figures using their **EXACT bolded titles** provided above.")
        prompt_parts.append("\n**Required Format for Evidence Section:**")
        prompt_parts.append("### Key Evidence")
        prompt_parts.append("- **[Exact_ID_From_Above] Image Title**: Explanation of evidence.")

        prompt_parts.append("\n\nProvide your final, synthesized analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        state["analysis_images"] = all_images 
        
        self.logger.info("✅ Prep Step Complete: Final synthesis prompt is ready.")
        return state
    

class GenerateHTMLReportController:
    """
    [🛠️ Tool Step]
    Generates a beautiful, human-readable HTML report.
    
    - Citation-Based Filtering: Scans the 'detailed_analysis' text. 
      Only displays images that the LLM explicitly referenced by name.
    - Fallback: If the LLM references nothing, falls back to 'Smart Filtering' 
      (showing Grids and hiding redundant components) to ensure the report isn't empty.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Helper to convert bytes to base64 string for HTML embedding."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def _filter_by_citations(self, text: str, all_images: list) -> list:
        """
        Selects images based on 'Concept Triggers' rather than strict string matching.
        If the text discusses a scientific method (e.g., NMF), the relevant summary plots are forced to display.
        """
        cited_images = []
        lower_text = text.lower()
        
        for img in all_images:
            raw_label = img.get('label', '')
            label_lower = raw_label.lower()
            
            # --- 1. Exact & Direct Match ---
            if raw_label in text:
                cited_images.append(img)
                continue
            
            # Check for label without the [ID] prefix
            # e.g. Label: "[Global_Analysis] NMF Summary Grid" -> Match: "NMF Summary Grid"
            clean_name = re.sub(r'\[.*?\]', '', label_lower).strip()
            if clean_name and clean_name in lower_text:
                cited_images.append(img)
                continue

            # --- 2. Concept Triggers (The Safety Net) ---

            # TRIGGER: Decomposition Summary (NMF, PCA, or ICA)
            # If the plot is a summary grid and the text mentions the method or "Components", show it.
            if "summary grid" in label_lower:
                if "nmf" in lower_text or "pca" in lower_text or "ica" in lower_text or "component" in lower_text or "unmixing" in lower_text or "decomposition" in lower_text:
                    cited_images.append(img)
                    continue

            # TRIGGER: Custom / Dynamic Analysis
            # If the plot is a Custom Analysis, check if the specific feature name (e.g. "Peak Center") is mentioned.
            if "custom analysis" in label_lower and ":" in label_lower:
                # Extract feature name: "[ID] Custom Analysis: Peak Center" -> "peak center"
                try:
                    feature_name = label_lower.split(":", 1)[1].strip()
                    if feature_name and feature_name in lower_text:
                        cited_images.append(img)
                        continue
                except IndexError:
                    pass

            # TRIGGER: Structure / Morphology
            # If the plot is a Structure Overlay and text mentions Structure/Correlation, show it.
            if "structure" in label_lower and "overlay" in label_lower:
                if "structure" in lower_text or "morphology" in lower_text or "correlation" in lower_text:
                    cited_images.append(img)
                    continue

            # --- 3. Iteration Context Match ---
            # If the text explicitly names an iteration (e.g. "Global Analysis"), 
            # ensure the main summary grid for that iteration is shown.
            match = re.match(r"\[(.*?)\]", raw_label)
            if match:
                iter_id_clean = match.group(1).replace("_", " ").lower() # e.g. "global analysis"
                if iter_id_clean in lower_text and ("grid" in label_lower or "custom" in label_lower):
                    cited_images.append(img)
                    continue

        # --- Deduplicate ---
        unique_images = []
        seen = set()
        for img in cited_images:
            if img['label'] not in seen:
                unique_images.append(img)
                seen.add(img['label'])

        # --- 4. Final Fail-Safe ---
        # If the filter returned <= 1 image, force the Global Summary to appear
        # to ensure the report always has context.
        if len(unique_images) <= 1:
            for img in all_images:
                if "global" in img['label'].lower() and "summary" in img['label'].lower():
                    if img['label'] not in seen:
                        unique_images.insert(0, img) # Insert at top
                        seen.add(img['label'])

        return unique_images

    def _filter_redundant_heuristic(self, all_images: list) -> list:
        """
        Backup Strategy: If LLM fails to cite images, use logic to pick the best ones.
        Hides individual components if a Grid exists.
        """
        iterations_with_grid = set()
        for img in all_images:
            label = img.get('label', '')
            if "Summary Grid" in label:
                match = re.match(r"\[(.*?)\]", label)
                if match: iterations_with_grid.add(match.group(1))

        filtered_images = []
        for img in all_images:
            label = img.get('label', '')
            match = re.match(r"\[(.*?)\]", label)
            if match and match.group(1) in iterations_with_grid:
                if "Component" in label and "Analysis" in label:
                    continue # Skip component if grid exists
            filtered_images.append(img)
        return filtered_images

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n📄 --- TOOL STEP: GENERATING HTML REPORT --- 📄\n")
        
        result_json = state.get("result_json")
        if not result_json:
            self.logger.warning("Skipping report generation: No result_json found.")
            return state

        # Extract Data
        detailed_analysis = result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        all_images = state.get("analysis_images", [])
        
        # --- SELECTION LOGIC ---
        # 1. Try Strict Citation
        display_images = self._filter_by_citations(detailed_analysis, all_images)
        selection_method = "Strict Text Citation"

        # 2. Fallback to Heuristic if strict failed (LLM didn't follow instructions)
        if not display_images:
            self.logger.warning("LLM did not explicitly cite any images. Falling back to heuristic filter.")
            display_images = self._filter_redundant_heuristic(all_images)
            selection_method = "Heuristic (Backup)"

        self.logger.info(f"Report Generation: Selected {len(display_images)} images using method: {selection_method}")

        # Output Setup
        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Hyperspectral_Report_{file_timestamp}.html"
        filepath = os.path.join(output_dir, filename)

        # --- HTML CONSTRUCTION ---
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hyperspectral Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
                .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                h3 {{ color: #16a085; }}
                .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #bdc3c7; margin-bottom: 20px; }}
                .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
                .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; }}
                .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; margin-top: 20px; }}
                .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; cursor: pointer; transition: transform 0.2s; }}
                .image-card img:hover {{ transform: scale(1.01); }}
                .image-label {{ margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }}
                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 Hyperspectral Analysis Report</h1>
                <div class="metadata-box">
                    <p><strong>Date:</strong> {timestamp}</p>
                    <p><strong>Data Source:</strong> {state.get('image_path', 'N/A')}</p>
                    <p><strong>System Info:</strong> {json.dumps(system_info)}</p>
                </div>

                <h2>1. Synthesized Scientific Analysis</h2>
                <div class="analysis-text">{detailed_analysis}</div>

                <h2>2. Key Evidence (Visual Gallery)</h2>
                <p>These figures are explicitly cited in the analysis above.</p>
                <div class="image-grid">
        """

        for img in display_images:
            label = img.get('label', 'Unknown Figure')
            data = img.get('data') or img.get('bytes')
            
            if data:
                b64_str = self._image_to_base64(data)
                safe_id = _sanitize_filename(label)
                
                html_content += f"""
                    <div class="image-card" id="{safe_id}">
                        <img src="data:image/jpeg;base64,{b64_str}" alt="{label}" loading="lazy">
                        <div class="image-label">{label}</div>
                    </div>
                """

        html_content += """
                </div>
                <h2>3. Key Scientific Claims</h2>
        """

        if not scientific_claims:
            html_content += "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                html_content += f"""
                <div class="claim-card">
                    <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
                    <p><strong>Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
                    <p><strong>Literature Search Query:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
                </div>
                """

        html_content += """
                <div class="footer">
                    Generated by SciLink Hyperspectral Analysis Agent
                </div>
            </div>
        </body>
        </html>
        """

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"✅ REPORT GENERATED: {filepath}")
            if "result_paths" not in state: state["result_paths"] = []
            state["result_paths"].append(filepath)
        except Exception as e:
            self.logger.error(f"❌ Failed to write HTML report: {e}")

        return state
    

class RunDynamicAnalysisController:
    """
    [🧠 + 💻] The 'Code Interpreter' / 'Dynamic Analyst'.
    Generates, executes, and validates Python code to model spectral features.

    Unlike for other agents, we use in-process exec() because:

    - Hyperspectral cubes are large (100MB+). Serializing to disk for a
      subprocess to reload would add significant I/O overhead.
    - The generated code is a pure function (data in → arrays out), not a
      standalone program that needs matplotlib or file I/O.
    - Results are numpy arrays that would be painful to serialize via stdout.

    """
    MAX_RETRIES = 5
    SUCCESS_THRESHOLD = 0.5  # If >50% of maps in a script pass QC, accept the run.

    # Engine plumbing (#327 phase 5). The retry ladder has 3 structural
    # rungs — first try / patch-or-question / abandon-family, expressed in
    # _codegen_retry_feedback — the engine reads only the ladder LENGTH.
    _CONSTRAINT_ANNEALING_SCHEDULE = (
        "first attempt (no retry feedback)",
        "warm — patch the logic / question the method",
        "hot — abandon the method family",
    )
    _QC_ENGINE_SPEC = QCEngineSpec(
        config_key=None,       # no locked-config plumbing in dynamic analysis
        refine_anchor="none",  # regenerate from prompt; freedom lives in the feedback text
        refit_fail_msg="    \u274c Attempt could not run",  # unreachable (attempts never fail at engine level)
    )

    @property
    def max_verification_iterations(self) -> int:
        """Engine loop budget: initial attempt + N refits, the last verdict
        checked by the for/else final pass == ``MAX_RETRIES`` total attempts,
        exactly the pre-port while-loop budget."""
        return self.MAX_RETRIES - 1

    def __init__(self, model, logger, generation_config, safety_settings, parse_fn,
                 executor_timeout: int = 600,
                 qc_time_budget_s: float = 1800.0):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        # In-process ExecutionTimeout for the generated code's exec()
        # sandbox. Plumbed from HyperspectralAnalysisAgent's
        # executor_timeout kwarg so the user's chosen limit is honored
        # and the log line below reports the actual value.
        self.executor_timeout = executor_timeout
        # Cumulative wall-clock budget for the QC verification loop (#358):
        # the attempt cap alone doesn't bound time (each attempt can run up
        # to executor_timeout). Consumed by CodegenQCEngine's verification
        # loop; falsy disables. Hyperspectral is the only host that sets
        # this — curve/image behavior is byte-identical.
        self.qc_time_budget_s = qc_time_budget_s
        self._parse_llm_response = parse_fn

    def execute(self, state: dict) -> dict:
        decision = state.get("refinement_decision", {})
        targets = decision.get("targets", [])
        
        # Filter strictly for custom code requests
        custom_targets = [t for t in targets if t.get('type') == 'custom_code']
        
        # Gatekeeping: If no code requested, skip
        if not custom_targets and not decision.get("requires_custom_code", False):
            return state

        self.logger.info(f"\n\n💻 --- DYNAMIC ANALYSIS: PROCESSING {len(custom_targets)} TASKS --- 💻\n")

        # --- SETUP OUTPUT PATHS ---
        output_dir = state.get("settings", {}).get("output_dir", "spectroscopy_output")
        os.makedirs(output_dir, exist_ok=True)
        
        iter_title = _sanitize_filename(state.get("iteration_title", "iter"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- PREPARE DATA CONTEXT ---
        h, w, e = state["hspy_data"].shape

        # Axis & Unit Detection — reads through resolve_axis_spec so non-energy
        # axes (time, voltage, frequency, ...) work the same as the legacy
        # energy_range path. The state key remains "energy_axis" for backward
        # compatibility with downstream consumers.
        sys_info = state.get("system_info", {})
        axis_spec = resolve_axis_spec(sys_info)
        axis_2 = axis_spec["axis_2"]
        axis_units = axis_2.get("units", "arbitrary units")

        if "energy_axis" not in state:
            if "start" in axis_2 and "end" in axis_2:
                state["energy_axis"] = np.linspace(axis_2["start"], axis_2["end"], e)
            else:
                state["energy_axis"] = np.arange(e)
                axis_units = "channels"

        self.logger.info(f"Data Axis Units detected as: {axis_units}")

        # Master containers for ALL scripts run in this session
        all_valid_maps = []
        all_valid_meta = []

        # Feed the per-pixel codegen the RAW cube, not the cube cleaned for
        # decomposition (despike/clip/mask is tuned for NMF and can distort the
        # features being fit — e.g. clipping real low-loss negatives). The codegen
        # owns its own fittability denoising. See docs/hyperspectral_codegen_relocation.md.
        optimal_data, processing_note = tools.get_optimal_analysis_data(state["original_hspy_data"])
        self.logger.info(f"📊 Dynamic Analysis Prep: {processing_note}")

        # Script-bank retrieval inputs (#346 step 2): fingerprint the cube once
        # per run; the per-target lookup below adds the target text to the
        # context query. Failure-isolated; None disables retrieval entirely.
        _bank_cube_fp = None
        _bank_ctx = None
        try:
            from scilink.skills._shared import _script_bank
            if _script_bank.bank_enabled() and not state.get("prior_analysis_paths"):
                _bank_cube_fp = _script_bank.hyperspectral_fingerprint(
                    optimal_data, state.get("energy_axis"), axis_units)
                _bank_ctx = _script_bank.measurement_context(sys_info)
        except Exception as e:
            self.logger.warning(f"Bank fingerprint skipped: {e}")

        # Offer the rank-k decomposition reconstruction as an OPTIONAL, denoised
        # fit target alongside the raw cube (issue #219). The generated code may
        # use it for shape-based features on noisy data, but raw stays the base
        # input. None when decomposition was skipped or produced no components.
        reconstruction = None
        components = state.get("final_components")
        abundance_maps = state.get("final_abundance_maps")
        if components is not None and abundance_maps is not None:
            try:
                reconstruction = tools.reconstruct_cube(components, abundance_maps)
                if reconstruction.shape != optimal_data.shape:
                    self.logger.warning(
                        f"Reconstruction shape {reconstruction.shape} != raw "
                        f"{optimal_data.shape}; not offering it to codegen."
                    )
                    reconstruction = None
                else:
                    self.logger.info(
                        "🧩 Offering rank-k decomposition reconstruction as an "
                        "optional denoised codegen input."
                    )
            except Exception as e:
                self.logger.warning(f"Could not build reconstruction cube: {e}")
                reconstruction = None

        # Offer shape-aligned auxiliary dataset(s) as OPTIONAL numerical operands
        # (issue #226): the user-supplied companions (reference/baseline/other
        # channels) the generated code MAY divide by / subtract / mask with,
        # keyed by label. Aligned companions pass through as index-matched
        # operands. A 1D reference sampled on its OWN axis (e.g. a tabulated
        # cross-section / attenuation / standard pulled from a database, never
        # on the detector's grid) is resampled onto the signal axis here so it
        # becomes a per-channel operand instead of being dropped — the operand
        # stays index-aligned, so the codegen contract is unchanged. Anything
        # still unalignable is kept context-only. Raw `data` stays base.
        auxiliary_operands = {}
        h_, w_, e_ = optimal_data.shape
        energy_axis = np.asarray(state.get("energy_axis")) if state.get("energy_axis") is not None else None
        for it in (state.get("auxiliary_items") or []):
            arr = it.get("array")
            if arr is None:
                continue
            arr = np.asarray(arr)
            label = it.get("label") or "auxiliary"
            aligned = (
                (arr.ndim == 1 and arr.shape[0] == e_)            # reference spectrum (per channel)
                or (arr.ndim == 2 and arr.shape == (h_, w_))      # per-pixel map (mask/normalize)
                or (arr.shape == optimal_data.shape)              # full companion cube
            )
            resampled = (
                None if aligned
                else _resample_ref_to_signal_axis(arr, it.get("axis"), energy_axis, e_)
            )
            if aligned:
                auxiliary_operands[label] = arr
                self.logger.info(
                    f"🧩 Offering auxiliary '{label}' {arr.shape} as an optional "
                    f"codegen operand."
                )
            elif resampled is not None:
                auxiliary_operands[label] = resampled
                self.logger.info(
                    f"🧩 Auxiliary '{label}' ({arr.shape[0]} pts on its own axis) "
                    f"resampled onto the {e_}-channel signal axis as an operand."
                )
            else:
                self.logger.info(
                    f"Auxiliary '{label}' shape {arr.shape} not aligned with primary "
                    f"{optimal_data.shape}; kept as context only (not an operand)."
                )

        # Deterministic flux-by-band table for the result reviews (computed
        # once; same data for every target/attempt this run).
        flux_table = _render_band_flux_table(
            optimal_data, state["energy_axis"], axis_units, auxiliary_operands
        )

        # --- MAIN LOOP: Process each target description separately ---
        for i, target in enumerate(custom_targets, 1):
            target_desc = target.get("description", "Analyze feature")
            # Objective-aware required outputs: when the refinement-LLM marked
            # specific map keys as mandatory (driven by the user's stated
            # objective), failing any of them triggers a retry instead of
            # being silently dropped by the partial-success threshold.
            required_outputs = list(target.get("required_outputs") or [])
            if required_outputs:
                self.logger.info(
                    f"👉 Task {i}/{len(custom_targets)} (required outputs: {required_outputs}): {target_desc}"
                )
            else:
                self.logger.info(f"👉 Task {i}/{len(custom_targets)}: {target_desc}")

            # 1. Define Prompt for this specific task
            base_prompt = build_code_generation_prompt(
                target_desc=target_desc,
                h=h, w=w, e=e,
                axis_units=axis_units,
                axis_start=state['energy_axis'][0],
                axis_end=state['energy_axis'][-1],
                processing_note=processing_note,
                hints=state.get("analysis_hints"),
                objective=state.get("analysis_objective"),
                required_outputs=required_outputs,
                # Inject the active skill's `implementation` section so the
                # code-gen LLM gets domain-specific recipe guidance (lineshape,
                # baseline, library choice). Empty string when no skill is
                # active or no implementation section is defined.
                skill_implementation=_render_skill_block(state, "implementation"),
                reconstruction_available=reconstruction is not None,
                auxiliary_operands={k: v.shape for k, v in auxiliary_operands.items()},
            )

            # Registered tools from the _shared registry (this agent + active
            # skills). Pre-loaded into the sandbox globals below, so generated
            # code MAY call them by name (no import) when one fits — the same
            # optional-tool mechanism the image / curve agents use.
            _tool_specs = _hyperspectral_tool_specs(state)
            if _tool_specs:
                base_prompt += (
                    "\n\n### REGISTERED TOOLS (pre-loaded — call by name, no import)\n"
                    "These functions are already in scope. Prefer one when it fits "
                    "the task (e.g. deriving physical constants) over reimplementing it."
                )
                for _spec in _tool_specs:
                    base_prompt += "\n" + _spec.to_prompt()

            # Append a preprocessing-mask hint when one exists and identifies
            # excluded pixels. The mask is already applied to the data (zero-
            # filled), so per-pixel fits will produce garbage values on
            # excluded pixels; the LLM should be told to filter on the mask.
            mask = state.get("preprocessing_mask")
            if mask is not None and not bool(mask.all()):
                n_kept = int(mask.sum())
                n_total = int(mask.size)
                base_prompt += f"""

### PREPROCESSING MASK
A boolean preprocessing mask of shape ({mask.shape[0]}, {mask.shape[1]}) is
available indicating which (axis_0, axis_1) samples carry valid signal:
{n_kept} of {n_total} samples are True (kept). The mask itself is NOT passed
into the analysis function — operate on the raw spectra and, if your output
maps should mark excluded samples, set them to np.nan in your returned maps.
"""

            # --- Run the per-target attempt ladder on the shared engine ---
            # (#327 phase 5). The engine drives the outer loop: initial
            # attempt -> verdict check -> annealed retry-feedback refits ->
            # salvage fallback. One "attempt" (generate -> in-process exec ->
            # per-map QC -> success decision) moved verbatim into
            # _run_attempt; the per-map inner loop (voted combined review /
            # visual QC + SUCCESS_THRESHOLD/required-outputs gate) and the
            # salvage judge stay native, per the plan's HS-3 scoping.
            ctx = QCItemContext(
                state=state, data=optimal_data, data_path="",
                item_name=target_desc, item_idx=i - 1,
                is_regime_anchor=True,  # every target gets the full ladder
            )
            ctx.target_index = i
            ctx.required_outputs = required_outputs
            ctx.base_prompt = base_prompt
            ctx.current_prompt = base_prompt
            # Script-bank exemplar (#346): appended to the FIRST attempt's
            # prompt only — retries rebuild from the clean base_prompt, so
            # the escalation ladder (up to "abandon the method family")
            # stays exemplar-free. Failure-isolated.
            if _bank_cube_fp is not None:
                try:
                    from scilink.skills._shared import _script_bank
                    matches = _script_bank.find_exemplar(
                        "hyperspectral", _bank_cube_fp,
                        {**(_bank_ctx or {}), "analysis_target": target_desc},
                    )
                    if matches:
                        match = matches[0]
                        ctx.current_prompt = (
                            base_prompt + "\n\n"
                            + _script_bank.render_exemplar_block(match)
                        )
                        _script_bank.mark_retrieved(
                            "hyperspectral", match["record"]["id"])
                        self.logger.info(
                            f"   🏦 Bank exemplar offered: "
                            f"id={match['record']['id']} score={match['score']} "
                            f"({str(match['record'].get('technique_signals', {}).get('analysis_target') or '')[:60]})"
                        )
                except Exception as e:
                    self.logger.warning(f"Bank retrieval skipped: {e}")
            ctx.retries = 0
            ctx.best_attempt = {"req_passed": -1, "valid_count": -1,
                                "images": [], "maps": [], "meta": []}
            ctx.attempt_entries = []
            ctx.last_code = ""
            ctx.mean_spec_bytes = None
            ctx.session = {
                "h": h, "w": w,
                "output_dir": output_dir,
                "iter_title": iter_title,
                "timestamp": timestamp,
                "optimal_data": optimal_data,
                "reconstruction": reconstruction,
                "auxiliary_operands": auxiliary_operands,
                "processing_note": processing_note,
                "all_valid_maps": all_valid_maps,
                "all_valid_meta": all_valid_meta,
                "flux_table": flux_table,
            }
            engine = CodegenQCEngine(host=self, spec=self._QC_ENGINE_SPEC)
            out = engine.run_item(ctx) or {}
            record = out.get("record")
            if record is not None:
                state.setdefault("dynamic_analysis_records", []).append(record)

        # --- FINAL AGGREGATION ---
        if not all_valid_maps:
            self.logger.warning("⚠️ All dynamic analysis tasks failed.")
            state["dynamic_analysis_failed"] = True
            return state

        # Commit per-feature metadata; downstream synthesis reads
        # custom_analysis_metadata_list. We no longer overwrite the NMF/PCA/ICA
        # state["final_abundance_maps"] key with the stacked custom maps —
        # that write had no downstream reader after the Phase C cleanup and
        # would have collided semantically with the decomposition's own
        # abundance maps. method_used and new_tasks writes are likewise gone
        # (no consumers).
        state["custom_analysis_metadata_list"] = all_valid_meta

        self.logger.info(f"✅ Dynamic Analysis Complete. Total unique maps generated: {len(all_valid_maps)}")
        return state

    # --- CodegenQCEngine hooks (#327 phase 5) -----------------------------
    # The attempt bodies below moved verbatim from the pre-port while-loop.
    # An "attempt" never fails at the ENGINE level (exceptions become the
    # outcome's failure verdict — the retry trigger), so the engine's
    # refit-failure break is unreachable; the task verdict lives in the
    # outcome's `task_success` and is what qc_check_accept keys on.

    def qc_setup(self, ctx: QCItemContext) -> None:
        pass  # per-target context is prepared in execute() before run_item

    def qc_try_reuse(self, ctx: QCItemContext):
        return None  # no locked-script reuse path for dynamic analysis

    def qc_run_initial(self, ctx: QCItemContext) -> dict:
        return self._run_attempt(ctx)

    def qc_record_initial(self, ctx: QCItemContext, result: dict) -> None:
        ctx.best_result = result

    def qc_record_initial_failure(self, ctx: QCItemContext, result: dict) -> None:
        pass  # unreachable: attempts always succeed at the engine level

    def qc_verification_bypass(self, ctx: QCItemContext) -> bool:
        return False

    def qc_log_skip_verification(self, ctx: QCItemContext) -> None:
        pass  # unreachable: attempts always succeed at the engine level

    def qc_loop_setup(self, ctx: QCItemContext) -> None:
        pass

    def qc_verify(self, ctx: QCItemContext) -> dict:
        # The attempt already carries its own verdict (per-map QC + the
        # required-outputs/threshold decision ran inside it) — no extra
        # verification call.
        return ctx.current_result

    def qc_assess(self, ctx: QCItemContext, verification: dict) -> None:
        pass  # attempt entries are recorded at attempt time (_run_attempt)

    def qc_check_accept(self, ctx: QCItemContext, verification: dict) -> bool:
        return bool(verification.get("task_success"))

    def qc_refine(self, ctx: QCItemContext, verification: dict) -> dict:
        # Annealed retry: escalate from "patch the math" to "abandon the
        # method" as failures accumulate, so retries can leave a wrong-but-
        # self-consistent method basin instead of resampling it. See
        # _codegen_retry_feedback.
        ctx.current_prompt = ctx.base_prompt + _codegen_retry_feedback(
            ctx.retries, verification.get("error_msg") or "")
        if ctx.retries >= 2:
            self.logger.info(
                f"    ↻ Retry annealing engaged (failure {ctx.retries}): "
                "escalating from parameter-patch toward method-change."
            )
        # Level of the NEXT attempt, for the engine's _produced_at_level stamp.
        ctx.annealing_level = _retry_annealing_level(ctx.retries)
        return {"prompt": ctx.current_prompt}

    def qc_refit(self, ctx: QCItemContext, verification: dict,
                 refine_from, just_escalated_to_hot: bool) -> dict:
        # refine_from is always None (spec.refine_anchor="none"): structural
        # freedom is granted through the retry-feedback text, not a script
        # anchor drop.
        return self._run_attempt(ctx)

    def qc_after_refit(self, ctx: QCItemContext, refit_result: dict,
                       verification: dict) -> None:
        ctx.current_result = refit_result

    def qc_final_verify(self, ctx: QCItemContext) -> None:
        # Loop budget exhausted — the last attempt's verdict was not yet
        # checked by qc_check_accept; accept it here if the task succeeded.
        if ctx.current_result and ctx.current_result.get("task_success"):
            ctx.approved = True

    def qc_post_verification(self, ctx: QCItemContext):
        if ctx.approved:
            return {"record": self._build_target_record(ctx, task_success=True)}
        return None

    def qc_fallback(self, ctx: QCItemContext) -> dict:
        state = ctx.state
        required_outputs = ctx.required_outputs
        best_attempt = ctx.best_attempt
        optimal_data = ctx.session["optimal_data"]
        output_dir = ctx.session["output_dir"]
        code_str = ctx.last_code
        mean_spec_bytes = ctx.mean_spec_bytes

        self.logger.error(f"    ⚠️ Task {ctx.target_index} failed after {self.MAX_RETRIES} attempts.")
        # Salvage the strongest attempt rather than discarding all work:
        # commit its passing maps so a recoverable output survives an
        # unsatisfiable required one. The required-output failure is still
        # surfaced (the task did not cleanly succeed).
        if best_attempt["valid_count"] > 0:
            # Physics-aware salvage Judge: decide (from physics, not just
            # QC-pass count) whether the best partial is a defensible
            # APPROXIMATE result to present with honest caveats, or is
            # meaningless and should be withheld. Simple: one call.
            if mean_spec_bytes is None:
                try:
                    _ms = np.asarray(optimal_data).reshape(
                        -1, optimal_data.shape[-1]).mean(0)
                    mean_spec_bytes = plot_curve_to_bytes(
                        np.column_stack(
                            [np.asarray(state["energy_axis"]), _ms]),
                        {"title": "Representative mean spectrum"})
                except Exception:
                    mean_spec_bytes = b""
            passed_names = [m["name"] for m in best_attempt["meta"]]
            missing = [n for n in required_outputs if n not in passed_names]
            result_summary = (
                f"{best_attempt['req_passed']}/{len(required_outputs)} required "
                f"outputs passed verification. Committed (passed QC): "
                f"{passed_names or 'none'}. Never passed / withheld: "
                f"{missing or 'none'}.")
            if ctx.session.get("flux_table"):
                result_summary += "\n\n" + ctx.session["flux_table"]
            rep_dash = (best_attempt["images"][0]["data"]
                        if best_attempt["images"] else None)
            present, conf, caveat = self._judge_salvage(
                rep_dash, code_str, mean_spec_bytes or None,
                state.get("system_info"), state.get("analysis_objective"),
                _used_tool_descriptions(state, code_str), result_summary)

            # Record the degradation so the top-level status reflects it
            # (a salvage is NOT a clean success). Threaded up to analyze().
            state.setdefault("degradation_notes", []).append({
                "kind": "withheld" if not present else "approximate",
                "confidence": "none" if not present else conf,
                "missing_required": missing,
                "caveat": caveat,
            })

            if not present:
                self.logger.warning(
                    f"    ⚖️  Salvage judge WITHHELD the partial result "
                    f"(no physically defensible signal): {caveat}")
            else:
                self.logger.warning(
                    f"    ⚖️  Salvage judge: presenting APPROXIMATE result "
                    f"({conf} confidence) — {caveat}")
                self.logger.warning(
                    f"    ⛑️  Committing best partial attempt "
                    f"({best_attempt['req_passed']}/{len(required_outputs)} "
                    f"required outputs, {best_attempt['valid_count']} map(s) "
                    f"passed QC); some required output(s) never passed.")
                marker = f"[APPROXIMATE — {conf} confidence: {caveat}] "
                for img_item in best_attempt["images"]:
                    tools.save_image_bytes(img_item['data'], output_dir, img_item['filename'], self.logger)
                    state.setdefault("analysis_images", []).append(img_item)
                for m in best_attempt["meta"]:
                    m["description"] = marker + m.get("description", "")
                    m["confidence"] = conf
                    m["salvage_caveat"] = caveat
                ctx.session["all_valid_maps"].extend(best_attempt["maps"])
                ctx.session["all_valid_meta"].extend(best_attempt["meta"])

        return {"record": self._build_target_record(ctx, task_success=False)}

    def _build_target_record(self, ctx: QCItemContext, task_success: bool):
        # --- HS-1: per-target verification record (additive) ---
        # Same shape family as the curve/image quality_history so the
        # T=2 staging gate and downstream consumers read all three
        # modalities uniformly. Failure-isolated: never affects results.
        try:
            from .._verification_record import (
                HS_HISTORY_KEYMAP,
                build_quality_history,
            )
            _final_frac = (
                ctx.attempt_entries[-1].get("passed_fraction")
                if ctx.attempt_entries else None
            )
            _hist = build_quality_history(
                best_value=(_final_frac if _final_frac is not None else 0.0),
                threshold=self.SUCCESS_THRESHOLD,
                all_attempts=None,
                verification_history=ctx.attempt_entries,
                judge_result=None,
                script_errors=None,
                keymap=HS_HISTORY_KEYMAP,
            )
            # The task verdict also encodes the required-outputs gate,
            # which the numeric fraction alone cannot — overwrite
            # (curve's verifier-approved path does the same).
            _hist["approved"] = bool(task_success)
            return {
                "target": ctx.item_name,
                "required_outputs": ctx.required_outputs,
                "task_success": bool(task_success),
                "salvaged": (not task_success) and ctx.best_attempt["valid_count"] > 0,
                "script": ctx.last_code or None,
                "quality_history": _hist,
            }
        except Exception as _rec_err:  # noqa: BLE001 - record is additive
            self.logger.warning(
                f"    dynamic-analysis record skipped: {_rec_err}"
            )
            return None

    def _run_attempt(self, ctx: QCItemContext) -> dict:
        """One generate → exec → per-map-QC → success-decision attempt.

        Body moved verbatim from the pre-port retry loop. Exceptions become
        the returned outcome's failure verdict (the retry trigger), so the
        engine-level ``success`` is always True. Attempt entries and the
        retry counter are updated here, at attempt time, exactly as before.
        """
        state = ctx.state
        target_desc = ctx.item_name
        required_outputs = ctx.required_outputs
        h = ctx.session["h"]
        w = ctx.session["w"]
        output_dir = ctx.session["output_dir"]
        iter_title = ctx.session["iter_title"]
        timestamp = ctx.session["timestamp"]
        i = ctx.target_index
        optimal_data = ctx.session["optimal_data"]
        reconstruction = ctx.session["reconstruction"]
        auxiliary_operands = ctx.session["auxiliary_operands"]
        processing_note = ctx.session["processing_note"]
        retries = ctx.retries

        try:
            # --- A. CLEAN SLATE FOR THIS ATTEMPT ---
            # Prevents "Ghost Data" from failed previous attempts accumulating
            current_run_valid_images = []
            current_run_valid_maps = []
            current_run_valid_meta = []
            qc_failures = []
            # Level of THIS attempt = the retry-feedback stage it was
            # generated under; total defaults to 0 until stage E runs
            # so the except-path record never hits an unbound name.
            attempt_level = _retry_annealing_level(retries)
            total_maps_expected = 0
            ctx.mean_spec_bytes = None  # rendered lazily for the sanity check

            # --- B. GENERATE CODE ---
            self.logger.info(f"    (Attempt {retries+1}) Asking LLM to write code...")
            response = self.model.generate_content(ctx.current_prompt, generation_config=self.generation_config)
            result_json, _ = parse_codegen_response(response, field="code", logger=self.logger)
            code_str = (result_json or {}).get("code", "")
            ctx.last_code = code_str

            # --- C. SANDBOX SETUP ---
            local_scope = {}
            global_scope = {
                "np": np,
                "scipy": __import__("scipy"),
                "sklearn": __import__("sklearn"),
                "lmfit": __import__("lmfit"),
                "curve_fit": __import__("scipy.optimize", fromlist=["curve_fit"]).curve_fit,
                "nnls": __import__("scipy.optimize", fromlist=["nnls"]).nnls,
                "linregress": __import__("scipy.stats", fromlist=["linregress"]).linregress,
                "find_peaks": __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks,
                "gaussian_filter": __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter,
            }
            # Inject registered tools (from the _shared registry) for the
            # hyperspectral agent + active skills, so generated code can
            # optionally call them by name — same mechanism the image /
            # curve agents use, not domain code hardcoded in this generic
            # controller.
            global_scope.update(_registry_tool_callables(state))

            # Execute Code
            with ExecutionTimeout(seconds=self.executor_timeout):
                exec(code_str, global_scope, local_scope)

                if "analyze_feature" not in local_scope:
                    raise ValueError("Function 'analyze_feature' was not found in generated code.")

                # --- D. RUN ON DATA ---
                self.logger.info(f"    Executing generated code (timeout: {self.executor_timeout}s)...")
                func = local_scope["analyze_feature"]
                result_dict = _invoke_analyze_feature(
                    func, optimal_data, state["energy_axis"], reconstruction,
                    auxiliary=auxiliary_operands,
                )

            # Validation
            if not isinstance(result_dict, dict): raise ValueError("Function return must be a dict.")
            maps_dict = result_dict.get("maps")
            if not maps_dict or not isinstance(maps_dict, dict):
                raise ValueError("Return dict must contain a 'maps' key.")

            # Save Script (for debugging)
            safe_task_name = _sanitize_filename(target_desc)[:30]
            script_filename = f"{iter_title}_T{i}_{safe_task_name}_{timestamp}.py"
            try:
                with open(os.path.join(output_dir, script_filename), "w", encoding="utf-8") as f:
                    f.write(f"# Auto-generated Script\n# Task: {target_desc}\n\n{code_str}")
            except: pass

            # --- E. PROCESS MAPS (Dashboard + QC) ---
            total_maps_expected = len(maps_dict)
            raw_units = result_dict.get("units", "a.u.")
            desc = result_dict.get("description", "")

            for feature_name, result_map in maps_dict.items():
                # Shape/NaN Check
                if result_map.shape != (h, w):
                    self.logger.warning(f"    Skipping {feature_name}: Shape mismatch.")
                    continue
                if np.all(np.isnan(result_map)):
                    self.logger.warning(f"    Skipping {feature_name}: Map contains only NaNs.")
                    continue

                # 1. Determine Units (Fixes UnboundLocalError)
                current_unit = "a.u."
                if isinstance(raw_units, dict):
                    current_unit = raw_units.get(feature_name, "a.u.")
                elif isinstance(raw_units, str):
                    current_unit = raw_units

                safe_feat = _sanitize_filename(feature_name)

                # 2. Generate Dashboard (Map + Histogram). Pass
                # axis_spec so non-spatial leading axes get axis-
                # name-driven labels ("Voltage-Time Map" / "Sample
                # Count") instead of "Spatial Map" / "Pixel Count".
                dashboard_bytes = tools.create_feature_dashboard(
                    result_map, feature_name, current_unit,
                    axis_spec=resolve_axis_spec(state.get("system_info")),
                )

                if dashboard_bytes:
                    if feature_name in required_outputs:
                        # Combined review (visual + physical + tool
                        # evidence) in ONE voted pass for the user-asked-
                        # for deliverables. Merges what were two gates so a
                        # single reviewer weighs the dashboard, the method,
                        # the spectrum AND the deterministic tool evidence
                        # together — preventing the split-brain false
                        # reject (visual 'noise' vs physical 'saturation',
                        # each blind to the tool's measurability proof).
                        if ctx.mean_spec_bytes is None:
                            try:
                                _ms = np.asarray(optimal_data).reshape(
                                    -1, optimal_data.shape[-1]).mean(0)
                                ctx.mean_spec_bytes = plot_curve_to_bytes(
                                    np.column_stack(
                                        [np.asarray(state["energy_axis"]), _ms]),
                                    {"title": "Representative mean spectrum"})
                            except Exception:
                                ctx.mean_spec_bytes = b""  # tolerate; runs without it
                        # Coverage: fraction of pixels carrying a real
                        # (finite, non-zero) value. A tiny coverage for a
                        # feature expected to fill a coherent region is a
                        # masking/segmentation COLLAPSE the value stats
                        # alone don't reveal (a few dozen plausible-valued
                        # pixels look fine by range/mean).
                        _finite = np.isfinite(result_map)
                        _real = _finite & (np.abs(result_map) > 1e-9)
                        _cov = 100.0 * float(_real.sum()) / max(result_map.size, 1)
                        summary = (
                            f"value range [{float(np.nanmin(result_map)):.4g}, "
                            f"{float(np.nanmax(result_map)):.4g}], mean "
                            f"{float(np.nanmean(result_map)):.4g} {current_unit}; "
                            f"valid coverage {_cov:.1f}% "
                            f"({int(_real.sum())} of {result_map.size} pixels finite & non-zero)")
                        if ctx.session.get("flux_table"):
                            summary += "\n\n" + ctx.session["flux_table"]
                        tool_descriptions = _used_tool_descriptions(state, code_str)
                        self.logger.info(f"    🔎 Combined review on {feature_name}...")
                        is_valid, critique = self._review_required_output(
                            dashboard_bytes, code_str, ctx.mean_spec_bytes or None,
                            state.get("system_info"),
                            state.get("analysis_objective"),
                            feature_name, summary, tool_descriptions,
                            attempt_history=_render_attempt_history(ctx.attempt_entries))
                        if not is_valid:
                            self.logger.warning(
                                f"    🔎 Combined review rejected "
                                f"{feature_name}: {critique}")
                    else:
                        # Diagnostic (non-required) map: lighter single
                        # visual QC — no method/physics gate needed.
                        self.logger.info(f"    👀 Performing Visual QC on {feature_name}...")
                        is_valid, critique = self._check_result_visually(
                            dashboard_bytes, f"{target_desc} ({feature_name})")

                    if is_valid:
                        # STAGE DATA (Do not commit to state yet)
                        current_run_valid_images.append({
                            "label": f"Custom Analysis: {feature_name}",
                            "data": dashboard_bytes,
                            "filename": f"{iter_title}_T{i}_{safe_feat}_Dashboard_{timestamp}.jpeg"
                        })
                        current_run_valid_maps.append(result_map)
                        current_run_valid_meta.append({
                            "name": feature_name,
                            "units": current_unit,
                            "description": f"{desc}. [Data Source: {processing_note}]",
                            "stats": {
                                "min": float(np.nanmin(result_map)),
                                "max": float(np.nanmax(result_map)),
                                "mean": float(np.nanmean(result_map))
                            }
                        })
                    else:
                        self.logger.warning(f"    ❌ Visual QC rejected {feature_name}: {critique}")
                        qc_failures.append(f"{feature_name}: {critique}")

            # --- F. SUCCESS DECISION (Threshold + Required-Outputs Logic) ---
            valid_count = len(current_run_valid_maps)
            success_rate = valid_count / total_maps_expected if total_maps_expected > 0 else 0

            # Required-outputs gate: every named output must be
            # present AND QC-pass. Failure here forces a retry, so
            # the partial-success threshold never silently drops the
            # user-asked-for quantity.
            valid_names = {m['name'] for m in current_run_valid_meta}

            # Track the strongest attempt so an UNSATISFIABLE required
            # output (e.g. a QC criterion a valid result can never meet)
            # doesn't discard the recoverable maps this attempt produced.
            # Ranked by (#required passed, #maps passed); committed as a
            # self-consistent partial result only if every attempt fails.
            n_req_passed = sum(1 for n in required_outputs if n in valid_names)
            if (n_req_passed, valid_count) > (ctx.best_attempt["req_passed"], ctx.best_attempt["valid_count"]):
                ctx.best_attempt = {
                    "req_passed": n_req_passed,
                    "valid_count": valid_count,
                    "images": list(current_run_valid_images),
                    "maps": list(current_run_valid_maps),
                    "meta": list(current_run_valid_meta),
                }

            missing_required = [n for n in required_outputs if n not in valid_names]
            if missing_required:
                relevant_critiques = [
                    c for c in qc_failures
                    if any(req in c for req in missing_required)
                ]
                absent_from_output = [
                    n for n in missing_required if n not in maps_dict
                ]
                detail_parts = []
                if absent_from_output:
                    detail_parts.append(
                        f"keys absent from your `maps` dict: {absent_from_output}"
                    )
                if relevant_critiques:
                    detail_parts.append(
                        f"QC critiques on required outputs: {relevant_critiques}"
                    )
                detail = "; ".join(detail_parts) or "no further detail"
                raise ValueError(
                    f"Required outputs failed: {missing_required}. {detail}"
                )

            if valid_count > 0 and success_rate >= self.SUCCESS_THRESHOLD:
                status_msg = "✅ Success" if valid_count == total_maps_expected else "⚠️ Partial Success"
                self.logger.info(f"    {status_msg} ({valid_count}/{total_maps_expected} passed). Committing valid maps.")

                # 1. COMMIT Valid Images
                for img_item in current_run_valid_images:
                    tools.save_image_bytes(img_item['data'], output_dir, img_item['filename'], self.logger)
                    if "analysis_images" not in state: state["analysis_images"] = []
                    state["analysis_images"].append(img_item)

                # 2. COMMIT Data
                ctx.session["all_valid_maps"].extend(current_run_valid_maps)
                ctx.session["all_valid_meta"].extend(current_run_valid_meta)

                # HS-1: record the successful attempt (any residual
                # qc_failures are the maps a partial success dropped).
                ctx.attempt_entries.append(_hs_attempt_entry(
                    attempt_level, success_rate, qc_failures, ""))
                return {"success": True, "task_success": True,
                        "error_msg": None}
            else:
                raise ValueError(f"Too many QC failures ({len(qc_failures)}/{total_maps_expected}). Critiques: {qc_failures}")

        except Exception as e:
            error_msg = traceback.format_exc()
            if "QC failures" in str(e): error_msg = str(e) # Clean message for LLM

            self.logger.warning(f"    ❌ Attempt {retries+1} failed: {error_msg}")
            # HS-1: record the failed attempt with the escalation
            # stage that will be applied to the next one.
            ctx.attempt_entries.append(_hs_attempt_entry(
                attempt_level,
                (len(current_run_valid_maps) / total_maps_expected
                 if total_maps_expected else None),
                qc_failures,
                _retry_stage_label(retries + 1),
                error=str(e),
            ))
            ctx.retries = retries + 1
            return {"success": True, "task_success": False,
                    "error_msg": error_msg}

    def _check_result_visually(self, dashboard_bytes: bytes, feature_desc: str) -> tuple[bool, str]:
        """
        Judge the Dashboard (Map + Histogram) with SPARSE SIGNAL AWARENESS.
        """
        check_prompt = [
            SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS.format(feature_desc=feature_desc)
        ]
        check_prompt.append({"mime_type": "image/jpeg", "data": dashboard_bytes})
        
        try:
            # Low temperature for strict consistency
            config = None#GenerationConfig(response_mime_type="application/json", temperature=0.1)
            resp = self.model.generate_content(
                check_prompt, 
                generation_config=config,
                safety_settings=self.safety_settings
            )
            result, _ = self._parse_llm_response(resp)
            return result.get("valid", True), result.get("critique", "")
        except Exception as e:
            self.logger.warning(f"QC check crashed: {e}")
            return True, ""

    # Number of independent judgments and the votes-to-reject majority. A single
    # judgment is too noisy near the decision boundary (it false-rejects a
    # correct result ~half the time in validation), but it never false-PASSES a
    # gross error — so requiring a majority of independent judgments to agree
    # before flagging removes the false-rejects while keeping the true catches.
    SANITY_VOTES = 3
    SANITY_REJECT_MAJORITY = 2

    def _sanity_check_one(self, dashboard_bytes, code_str, mean_spec_bytes,
                          system_info, objective, feature_desc, result_summary):
        meta_str = (json.dumps(system_info, default=str)[:1500]
                    if system_info else "(none)")
        prompt = [SPECTROSCOPY_PHYSICS_SANITY_INSTRUCTIONS.format(
            objective=objective or "(not specified)",
            metadata=meta_str,
            method=(code_str or "(unavailable)")[:6000],
            result_summary=f"Output '{feature_desc}': {result_summary}",
        )]
        prompt.append("Result dashboard (map + histogram):")
        prompt.append({"mime_type": "image/jpeg", "data": dashboard_bytes})
        if mean_spec_bytes:
            prompt.append("Representative mean spectrum of the data:")
            prompt.append({"mime_type": "image/png", "data": mean_spec_bytes})
        resp = self.model.generate_content(
            prompt, generation_config=None, safety_settings=self.safety_settings,
        )
        result, _ = self._parse_llm_response(resp)
        return bool(result.get("valid", True)), result.get("critique", "")

    def _sanity_check_result(self, dashboard_bytes, code_str, mean_spec_bytes,
                             system_info, objective, feature_desc,
                             result_summary) -> tuple[bool, str]:
        """LLM physical-soundness check: given the objective, data context, the
        METHOD (generated code) and a representative spectrum, judge whether the
        result is physically plausible and the method sound. Complements visual
        QC, which only sees the output dashboard and so cannot catch a smooth-
        but-wrong VALUE (e.g. a biased global fit).

        Votes ``SANITY_VOTES`` independent judgments and rejects only when at
        least ``SANITY_REJECT_MAJORITY`` agree the result is flawed — robust to
        single-judgment noise, and biased toward accept (so it does not suppress
        surprising-but-sound results). Fails open on error, and short-circuits
        once the outcome is decided.
        """
        reject_votes, last_crit = 0, ""
        for i in range(self.SANITY_VOTES):
            try:
                ok, crit = self._sanity_check_one(
                    dashboard_bytes, code_str, mean_spec_bytes,
                    system_info, objective, feature_desc, result_summary)
            except Exception as e:
                self.logger.warning(f"Physics sanity check crashed: {e}")
                return True, ""  # fail-open
            if not ok:
                reject_votes += 1
                last_crit = crit or last_crit
            # short-circuit: decided either way
            remaining = self.SANITY_VOTES - (i + 1)
            if reject_votes >= self.SANITY_REJECT_MAJORITY:
                return False, last_crit
            if reject_votes + remaining < self.SANITY_REJECT_MAJORITY:
                return True, ""
        return True, ""

    def _review_required_output_one(self, dashboard_bytes, code_str, mean_spec_bytes,
                                    system_info, objective, feature_desc,
                                    result_summary, tool_descriptions,
                                    attempt_history: str = ""):
        meta_str = (json.dumps(system_info, default=str)[:1500]
                    if system_info else "(none)")
        prompt = [SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS.format(
            objective=objective or "(not specified)",
            metadata=meta_str,
            method=(code_str or "(unavailable)")[:6000],
            result_summary=f"Output '{feature_desc}': {result_summary}",
            tool_descriptions=tool_descriptions or "(no registered tool was called)",
            tool_scrutiny=VERIFIER_TOOL_SCRUTINY_PRINCIPLE,
            attempt_history=attempt_history,
        )]
        prompt.append("Result dashboard (map + histogram):")
        prompt.append({"mime_type": "image/jpeg", "data": dashboard_bytes})
        if mean_spec_bytes:
            prompt.append("Representative mean spectrum of the data:")
            prompt.append({"mime_type": "image/png", "data": mean_spec_bytes})
        resp = self.model.generate_content(
            prompt, generation_config=None, safety_settings=self.safety_settings,
        )
        result, _ = self._parse_llm_response(resp)
        return bool(result.get("valid", True)), result.get("critique", "")

    def _review_required_output(self, dashboard_bytes, code_str, mean_spec_bytes,
                                system_info, objective, feature_desc,
                                result_summary, tool_descriptions,
                                attempt_history: str = "") -> tuple[bool, str]:
        """Combined visual + physical review for a REQUIRED output, in ONE voted
        pass. Replaces the separate visual-QC then physics-sanity gates for
        required deliverables: a single reviewer weighs the dashboard, the
        method, the spectrum AND the deterministic tool evidence together, so it
        cannot split-brain into 'looks like noise' (visual) vs 'saturation'
        (physical) each blind to the tool's measurability proof. Same voting
        policy as the physics sanity check (majority-to-reject, fail-open,
        short-circuit); diagnostics (non-required maps) keep the lighter single
        visual QC.
        """
        reject_votes, last_crit = 0, ""
        for i in range(self.SANITY_VOTES):
            try:
                ok, crit = self._review_required_output_one(
                    dashboard_bytes, code_str, mean_spec_bytes,
                    system_info, objective, feature_desc,
                    result_summary, tool_descriptions,
                    attempt_history=attempt_history)
            except Exception as e:
                self.logger.warning(f"Combined result review crashed: {e}")
                return True, ""  # fail-open
            if not ok:
                reject_votes += 1
                last_crit = crit or last_crit
            remaining = self.SANITY_VOTES - (i + 1)
            if reject_votes >= self.SANITY_REJECT_MAJORITY:
                return False, last_crit
            if reject_votes + remaining < self.SANITY_REJECT_MAJORITY:
                return True, ""
        return True, ""

    def _judge_salvage(self, dashboard_bytes, code_str, mean_spec_bytes,
                       system_info, objective, tool_descriptions, result_summary):
        """Final salvage judge (single call). When ALL attempts fail, decide from
        the physics whether the best partial result is a defensible APPROXIMATE
        answer worth presenting with caveats, or is meaningless and should be
        withheld. Returns (present: bool, confidence: 'low'|'medium', caveat).
        Fails OPEN (present, low, generic caveat) so a crash never silently drops
        recoverable data — honesty via the caveat, not by discarding.
        """
        try:
            meta_str = (json.dumps(system_info, default=str)[:1500]
                        if system_info else "(none)")
            prompt = [SPECTROSCOPY_SALVAGE_JUDGE_INSTRUCTIONS.format(
                objective=objective or "(not specified)",
                metadata=meta_str,
                method=(code_str or "(unavailable)")[:6000],
                tool_descriptions=tool_descriptions or "(no registered tool was called)",
                result_summary=result_summary,
            )]
            if dashboard_bytes:
                prompt.append("Representative result dashboard (map + histogram):")
                prompt.append({"mime_type": "image/jpeg", "data": dashboard_bytes})
            if mean_spec_bytes:
                prompt.append("Representative mean spectrum of the data:")
                prompt.append({"mime_type": "image/png", "data": mean_spec_bytes})
            resp = self.model.generate_content(
                prompt, generation_config=None, safety_settings=self.safety_settings)
            result, _ = self._parse_llm_response(resp)
            present = bool(result.get("present", True))
            conf = str(result.get("confidence", "low")).lower()
            if conf not in ("low", "medium"):
                conf = "low"
            caveat = (result.get("caveat") or
                      "Result did not pass full verification; treat as approximate.")
            return present, conf, caveat
        except Exception as e:
            self.logger.warning(f"Salvage judge crashed: {e}")
            return True, "low", "Result did not pass full verification; treat as approximate."


# Moved to base_controllers (modality-agnostic critic/editor pair) —
# re-exported under the historical names so existing imports keep working.
from .base_controllers import (  # noqa: E402,F401
    RunSelfReflectionController,
    ApplyReflectionUpdatesController,
)
