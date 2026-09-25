"""Acquisition skills — what a technique's knobs buy and cost.

The analysis skills (``skills/curve_fitting/raman`` ...) say how to FIT a
measurement; they serve live and chat runs alike and live mode adds nothing to
them. What is specific to a running experiment is how to STEER it: which
parameter trades what, where the sample or the instrument gets hurt, what a bad
frame looks like, in what order to move. That knowledge lives in a knowledge-only
skill domain, ``skills/acquisition/<technique>/``, read by the slow-clock
consumers of a live loop (today: :class:`~scilink.live.recommend.LLMRecommender`).

Per technique, not per instrument: a vendor's API, its limits and its file
format belong to the :class:`~scilink.live.instruments.Instrument` subclass and
its ``schema``. Two Raman spectrometers share ``acquisition/raman``.

The domain declares its own section vocabulary in ``skills/loader.py``:
``overview · tradeoffs · limits · quality · strategy``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

DOMAIN = "acquisition"
SECTIONS = ("overview", "tradeoffs", "limits", "quality", "strategy")
_TITLES = {"overview": "What is being measured", "tradeoffs": "What each parameter buys and costs",
           "limits": "Limits — never recommend past these", "quality": "Reading the stream",
           "strategy": "How to move"}


def select_acquisition_skill(model: Any, context: Any, schema: Any = None,
                             logger: Optional[logging.Logger] = None) -> Optional[str]:
    """The one acquisition skill whose technique matches this experiment, or
    None. One model call through the shared selector (exclusive: a measurement
    is made with one technique). Never raises."""
    from ..skills._shared._graduation import parse_json_response
    from ..skills._shared._skill_selector import select_relevant_skills

    if isinstance(context, dict):
        text = "\n".join(f"- {k}: {v}" for k, v in context.items())
    else:
        text = str(context or "")
    if not text.strip():
        return None          # nothing to match a technique against
    if schema is not None:
        text += "\n\nAcquisition parameters of the instrument:\n" + schema.describe()

    def _parse(resp):
        return parse_json_response(resp.text if hasattr(resp, "text") else str(resp)), None

    def _generate(contents=None, generation_config=None, safety_settings=None):
        prompt = "\n".join(str(p) for p in contents) if isinstance(contents, list) else contents
        return model.generate_content(prompt)

    picked = select_relevant_skills(
        model=SimpleNamespace(generate_content=_generate), parse_fn=_parse, domain=DOMAIN, context_parts=[text], exclusive=True, logger=logger)
    return picked[0] if picked else None


def render_guidance(skill: str) -> str:
    """The skill's sections as one prompt block (empty string if it cannot be
    loaded — guidance is grounding, never a dependency)."""
    from ..skills.loader import load_skill
    try:
        parsed: Dict[str, Any] = load_skill(skill, domain=DOMAIN)
    except Exception:  # noqa: BLE001
        return ""
    parts = [f"### {_TITLES[s]}\n{parsed[s].strip()}" for s in SECTIONS if (parsed.get(s) or "").strip()]
    return "\n\n".join(parts)
