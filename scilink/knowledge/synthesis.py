"""Programmatic API for knowledge synthesis and reuse across analyses."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# Map synthesis_type to prompt template names
_SYNTHESIS_PROMPT_MAP = {
    "reference": "KNOWLEDGE_SYNTHESIS_INSTRUCTIONS",
    "trend": "KNOWLEDGE_TREND_INSTRUCTIONS",
    "failure": "KNOWLEDGE_FAILURE_INSTRUCTIONS",
    "method": "KNOWLEDGE_METHOD_INSTRUCTIONS",
}


def synthesize_knowledge(
    analysis_results: List[Dict[str, Any]],
    focus: str,
    *,
    model: Any = None,
    model_name: str = "gemini-3.1-pro-preview",
    api_key: Optional[str] = None,
    knowledge_id: Optional[str] = None,
    synthesis_type: str = "reference",
) -> Dict[str, Any]:
    """Distill findings from completed analyses into reusable prior knowledge.

    Takes one or more result dicts (as returned by ``agent.analyze()``) and
    uses an LLM to extract actionable, quantitative findings focused on a
    specific topic.  The returned dict can be passed directly as an element
    of the ``prior_knowledge`` list in subsequent ``agent.analyze()`` calls.

    Args:
        analysis_results: List of result dicts from ``agent.analyze()``.
            Each dict must contain a ``"detailed_analysis"`` key.
        focus: What to extract — e.g. ``"peak position vs. concentration
            calibration"``.
        model: An already-initialised LLM model instance (anything with a
            ``generate_content`` method).  When *None*, a
            :class:`LiteLLMGenerativeModel` is created from *model_name*
            and *api_key*.
        model_name: Model name used when *model* is *None*.
        api_key: API key used when *model* is *None*.
        knowledge_id: Optional identifier for the entry.  Defaults to
            ``"knowledge_001"``.
        synthesis_type: Type of synthesis to perform. One of:
            ``"reference"`` (default) — calibration/reference extraction,
            ``"trend"`` — cross-sample trend detection,
            ``"failure"`` — failure pattern learning,
            ``"method"`` — method selection heuristics.

    Returns:
        A knowledge entry dict::

            {
                "id": "knowledge_001",
                "focus": "...",
                "synthesis_type": "reference",
                "summary": "...",
                "key_findings": ["...", ...],
                "timestamp": "..."
            }

    Raises:
        ValueError: If no ``detailed_analysis`` text is found in any result,
            or if *synthesis_type* is not recognized.
        RuntimeError: If the LLM call fails or returns unparseable output.

    Example::

        from scilink.knowledge import synthesize_knowledge

        calibration_result = agent.analyze(spectra, objective="calibrate ...")
        knowledge = synthesize_knowledge(
            [calibration_result],
            focus="peak position vs boron concentration calibration",
        )

        # Apply to a new spectrum
        prediction = agent.analyze(
            "unknown.csv",
            prior_knowledge=[knowledge],
        )
    """
    if synthesis_type not in _SYNTHESIS_PROMPT_MAP:
        raise ValueError(
            f"Unknown synthesis_type '{synthesis_type}'. "
            f"Must be one of: {list(_SYNTHESIS_PROMPT_MAP.keys())}"
        )

    # ── Collect detailed_analysis texts ──────────────────────────────────
    analysis_texts: list[str] = []
    human_feedback_texts: list[str] = []
    for i, result in enumerate(analysis_results):
        label = result.get("analysis_id", f"analysis_{i}")

        text = result.get("detailed_analysis", "")
        if text:
            analysis_texts.append(f"### Analysis: {label}\n{text}")

        # Collect fitting parameters if present
        fitting_params = result.get("fitting_parameters")
        if fitting_params:
            params_str = json.dumps(fitting_params, indent=2, default=str)
            analysis_texts.append(
                f"### Fitting Parameters ({label}):\n```json\n{params_str}\n```"
            )

        # Collect status for failure-pattern learning
        status = result.get("status")
        if status:
            analysis_texts.append(f"### Status ({label}): {status}")

        # Collect quality history for failure/method synthesis.
        # For tiered results, also check tier2_results.
        qh_entries = []
        t2 = result.get("tier2_results") or {}
        if result.get("quality_history"):
            tier_label = "Tier 1" if t2 else ""
            qh_entries.append((tier_label, result["quality_history"]))
        if isinstance(t2, dict) and t2.get("quality_history"):
            qh_entries.append(("Tier 2", t2["quality_history"]))

        for qh_label, qh in qh_entries:
            if not (qh and synthesis_type in ("failure", "method")):
                continue
            section_label = f" — {qh_label}" if qh_label else ""
            qh_lines = [f"### Quality History ({label}{section_label})"]

            for se in qh.get("script_errors", []):
                qh_lines.append(f"- Script error: {se.get('error', '')}")
                if se.get("fix"):
                    qh_lines.append(f"  Fix: {se['fix']}")

            iterations = qh.get("verification_iterations", [])
            for j, it in enumerate(iterations):
                issues = [x.get("problem", "") for x in it.get("issues", [])]
                qh_lines.append(
                    f"- Verification {j + 1} (score={it.get('score', 0):.2f})"
                    f": issues={issues}"
                )
                fix = it.get("fix_applied")
                if fix:
                    qh_lines.append(f"  Fix applied: {fix}")
                    if j + 1 < len(iterations):
                        next_score = iterations[j + 1].get("score", 0)
                        improved = next_score > it.get("score", 0)
                        qh_lines.append(
                            f"  Outcome: score "
                            f"{'improved' if improved else 'worsened'}"
                            f" to {next_score:.2f}"
                        )

            if len(qh_lines) > 1:
                analysis_texts.append("\n".join(qh_lines))

        # Collect human feedback
        hf = result.get("human_feedback", {})
        if isinstance(hf, dict):
            user_fb = hf.get("user_feedback", "")
            if user_fb:
                human_feedback_texts.append(
                    f"### User Feedback ({label}):\n{user_fb}"
                )

    if not analysis_texts:
        raise ValueError(
            "No detailed_analysis text found in the provided results."
        )

    # ── Build human feedback section ──────────────────────────────────────
    if human_feedback_texts:
        human_feedback_section = (
            "**Human Feedback / Domain Corrections:**\n"
            + "\n\n".join(human_feedback_texts)
            + "\n\nIncorporate these corrections and domain expertise into your findings."
        )
    else:
        human_feedback_section = ""

    # ── Build LLM prompt ─────────────────────────────────────────────────
    from scilink.agents.exp_agents.instruct import (
        KNOWLEDGE_SYNTHESIS_INSTRUCTIONS,
        KNOWLEDGE_TREND_INSTRUCTIONS,
        KNOWLEDGE_FAILURE_INSTRUCTIONS,
        KNOWLEDGE_METHOD_INSTRUCTIONS,
    )

    prompt_map = {
        "reference": KNOWLEDGE_SYNTHESIS_INSTRUCTIONS,
        "trend": KNOWLEDGE_TREND_INSTRUCTIONS,
        "failure": KNOWLEDGE_FAILURE_INSTRUCTIONS,
        "method": KNOWLEDGE_METHOD_INSTRUCTIONS,
    }

    template = prompt_map[synthesis_type]

    format_kwargs = {
        "focus": focus,
        "analysis_texts": "\n\n".join(analysis_texts),
    }
    # The reference template doesn't have a human_feedback_section placeholder
    if synthesis_type != "reference":
        format_kwargs["human_feedback_section"] = human_feedback_section
    else:
        # For reference type, append human feedback to analysis texts if present
        if human_feedback_section:
            format_kwargs["analysis_texts"] += "\n\n" + human_feedback_section

    prompt_text = template.format(**format_kwargs)

    # ── Resolve model ────────────────────────────────────────────────────
    if model is None:
        from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
        model = LiteLLMGenerativeModel(model=model_name, api_key=api_key)

    # ── Call LLM ─────────────────────────────────────────────────────────
    try:
        response = model.generate_content(
            contents=[prompt_text],
            generation_config=None,
            safety_settings=None,
        )
    except Exception as exc:
        raise RuntimeError(f"Knowledge synthesis LLM call failed: {exc}") from exc

    response_text = response.text if hasattr(response, "text") else str(response)

    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        raise RuntimeError("LLM did not return valid JSON.")

    llm_output = json.loads(json_match.group())

    # ── Build knowledge entry ────────────────────────────────────────────
    return {
        "id": knowledge_id or "knowledge_001",
        "focus": focus,
        "synthesis_type": synthesis_type,
        "summary": llm_output.get("summary", ""),
        "key_findings": llm_output.get("key_findings", []),
        "timestamp": datetime.now().isoformat(),
    }
