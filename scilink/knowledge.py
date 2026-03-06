"""Programmatic API for knowledge synthesis and reuse across analyses."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def synthesize_knowledge(
    analysis_results: List[Dict[str, Any]],
    focus: str,
    *,
    model: Any = None,
    model_name: str = "gemini-3.1-pro-preview",
    api_key: Optional[str] = None,
    knowledge_id: Optional[str] = None,
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

    Returns:
        A knowledge entry dict::

            {
                "id": "knowledge_001",
                "focus": "...",
                "summary": "...",
                "key_findings": ["...", ...],
                "timestamp": "..."
            }

    Raises:
        ValueError: If no ``detailed_analysis`` text is found in any result.
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
    # ── Collect detailed_analysis texts ──────────────────────────────────
    analysis_texts: list[str] = []
    for i, result in enumerate(analysis_results):
        text = result.get("detailed_analysis", "")
        if text:
            label = result.get("analysis_id", f"analysis_{i}")
            analysis_texts.append(f"### Analysis: {label}\n{text}")

    if not analysis_texts:
        raise ValueError(
            "No detailed_analysis text found in the provided results."
        )

    # ── Build LLM prompt ─────────────────────────────────────────────────
    from scilink.agents.exp_agents.instruct import KNOWLEDGE_SYNTHESIS_INSTRUCTIONS

    prompt_text = KNOWLEDGE_SYNTHESIS_INSTRUCTIONS.format(
        focus=focus,
        analysis_texts="\n\n".join(analysis_texts),
    )

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
        "summary": llm_output.get("summary", ""),
        "key_findings": llm_output.get("key_findings", []),
        "timestamp": datetime.now().isoformat(),
    }
