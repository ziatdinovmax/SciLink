"""Offline tests: user fit-domain requests are visible to the plan validator.

A crop / region-of-interest request routed through
custom_processing_instruction was visible to the planner but NOT to
_validate_plan — so a skill's mandatory "fit the full measured range" rule
made the validator revert the user's restricted fit domain. The guidance is
now injected into the validation prompt too, and states the restriction is
user-authorized.

The change is strictly conditional: with no custom_processing_instruction
the validation prompt must be byte-identical to before (asserted here as
"contains no trace of the guidance block").

  conda run -n scilink python -m pytest tests/test_fitdomain_validator_visibility.py -q
"""
import io
import logging
import tempfile

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    CurveFittingPlanningController, _append_fit_domain_guidance)


class CaptureModel:
    def __init__(self):
        self.prompts = []

    def generate_content(self, prompt_parts, **kw):
        self.prompts.append(prompt_parts)
        raise RuntimeError("capture only")


def _png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(np.arange(10), np.arange(10))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


PLOT = _png()


def make_state(cpi=None, skill=True):
    s = {
        "analysis_approach": "peak fitting",
        "physical_model": "Two Voigt peaks on a linear baseline",
        "parameters_to_extract": ["center", "fwhm"],
        "fitting_strategy": "Fit both peaks",
        "original_plot_bytes": PLOT,
        "is_single_spectrum": True,
        "num_spectra": 1,
        "system_info": {"technique": "IR"},
    }
    if skill:
        s["skills_loaded"] = [{
            "name": "ir",
            "planning": ("MANDATORY: Fit the full measured range. Never "
                         "restrict fitting to one region unless the user "
                         "asked for it."),
        }]
    if cpi:
        s["system_info"]["custom_processing_instruction"] = cpi
    return s


def validation_prompt_text(state):
    model = CaptureModel()
    ctrl = CurveFittingPlanningController(
        model=model, logger=logging.getLogger("t"),
        generation_config=None, safety_settings=None,
        parse_fn=lambda r: (None, {"error": "n/a"}),
        instructions="", output_dir=tempfile.mkdtemp())
    ctrl._validate_plan(state)  # model raises; _validate_plan swallows
    assert model.prompts, "validator never called the model"
    return "\n".join(p for p in model.prompts[-1] if isinstance(p, str))


def test_no_cpi_prompt_has_no_guidance_block():
    text = validation_prompt_text(make_state(cpi=None))
    assert "Fit-domain & background guidance" not in text
    assert "User processing note" not in text
    assert "user-authorized" not in text


def test_cpi_reaches_validator_with_authorization_principle():
    text = validation_prompt_text(make_state(cpi="crop to 1000-1800 cm-1"))
    assert "User processing note: crop to 1000-1800 cm-1" in text
    # The principle that defuses mandatory full-range skill rules:
    assert "user-authorized" in text
    assert "unless the user asked" in text
    # And the skill rule it must coexist with is present in the same prompt.
    assert "Fit the full measured range" in text


def test_helper_is_noop_without_instruction():
    prompt = ["base"]
    _append_fit_domain_guidance(prompt, {"system_info": {"technique": "IR"}})
    assert prompt == ["base"]
    _append_fit_domain_guidance(prompt, {})
    assert prompt == ["base"]


def test_helper_text_carries_principle():
    prompt = []
    _append_fit_domain_guidance(
        prompt, {"system_info": {
            "custom_processing_instruction": "fit only the decay"}})
    assert len(prompt) == 1
    assert "must not be reverted" in prompt[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
