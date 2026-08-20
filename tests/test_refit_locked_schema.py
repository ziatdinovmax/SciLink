"""AdaptiveRefit: a refit that adopts a different model must still report the
locked run's parameter names for that unit (series trends and the feature
table align units by name). Deterministic gap check + one completion pass
that only ADDS reporting, accepted only if no existing key is lost and R²
does not degrade; the residual gap is recorded on the result.
"""
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    AdaptiveRefitController,
)

LOCKED = {"peak_1": {"center": 532.0, "amplitude": 0.7, "fwhm": 13.0},
          "baseline": {"c0": 0.05}, "extinction_at_400nm": 0.9}
REFIT = {"powerlaw": {"A": 1.2, "n": 1.1}, "baseline": {"c0": 0.05}}


def test_gap_is_flattened_locked_names_missing_from_refit():
    gap = AdaptiveRefitController.locked_schema_gap(REFIT, LOCKED)
    assert gap == ["peak_1_center", "peak_1_amplitude", "peak_1_fwhm",
                   "extinction_at_400nm"]
    assert AdaptiveRefitController.locked_schema_gap(LOCKED, LOCKED) == []
    # a null counts as missing (it is an empty feature-table cell)
    assert "extinction_at_400nm" in AdaptiveRefitController.locked_schema_gap(
        {**REFIT, "extinction_at_400nm": None}, LOCKED)


class _FakeExec:
    """Runs nothing; returns a canned FIT_RESULTS_JSON per script marker."""
    timeout = 60

    def __init__(self, payloads):
        self.payloads = payloads   # script-substring -> params dict

    def execute_script(self, script, working_dir=None, timeout=None):
        for key, params in self.payloads.items():
            if key in script:
                out = {"model_type": "x", "parameters": params,
                       "fit_quality": {"r_squared": 0.98}}
                Path(working_dir, "visualization.png").write_bytes(b"png")
                return {"status": "success",
                        "stdout": "FIT_RESULTS_JSON:" + json.dumps(out)}
        return {"status": "error", "stdout": "", "message": "boom"}


class _FakeModel:
    def __init__(self, script):
        self.script = script
        self.prompts = []

    def generate_content(self, contents=None, **kw):
        self.prompts.append(contents[0])
        return SimpleNamespace(text=json.dumps({"script": self.script}))


def _controller(tmp, model, executor):
    c = object.__new__(AdaptiveRefitController)
    c.logger = logging.getLogger("t")
    c.output_dir = Path(tmp)
    c.model = model
    c.generation_config = None
    c.safety_settings = None
    c.executor = executor
    return c


def _refit_result(script="ORIGINAL"):
    return {"success": True, "parameters": dict(REFIT),
            "fit_quality": {"r_squared": 0.97}, "script": script}


def test_completion_adds_missing_under_locked_names():
    with tempfile.TemporaryDirectory() as tmp:
        completed = {**REFIT, "peak_1": {"center": None, "amplitude": None,
                                         "fwhm": None},
                     "extinction_at_400nm": 0.88}
        model = _FakeModel("COMPLETED")
        c = _controller(tmp, model, _FakeExec({"COMPLETED": completed}))
        res = c._complete_locked_schema(_refit_result(), {"parameters": LOCKED},
                                        np.zeros((5, 2)), 3, {"locked_fitting_config": {}})
        assert res["parameters"]["extinction_at_400nm"] == 0.88
        assert res["script"] == "COMPLETED"
        # nulls stay "missing" (empty cells) and are reported as the residual gap
        assert res["locked_schema_gap"] == ["peak_1_center", "peak_1_amplitude",
                                            "peak_1_fwhm"]
        assert "extinction_at_400nm" in model.prompts[0]
        assert "Do NOT change the model" in model.prompts[0]


def test_completion_rejected_if_existing_keys_lost():
    with tempfile.TemporaryDirectory() as tmp:
        broken = {"extinction_at_400nm": 0.88}          # dropped powerlaw_*
        c = _controller(tmp, _FakeModel("BROKEN"), _FakeExec({"BROKEN": broken}))
        res = c._complete_locked_schema(_refit_result(), {"parameters": LOCKED},
                                        np.zeros((5, 2)), 3, {})
        assert res["parameters"] == REFIT and res["script"] == "ORIGINAL"
        assert res["locked_schema_gap"] == ["peak_1_center", "peak_1_amplitude",
                                            "peak_1_fwhm", "extinction_at_400nm"]


def test_no_gap_is_a_noop_without_llm_call():
    with tempfile.TemporaryDirectory() as tmp:
        model = _FakeModel("X")
        c = _controller(tmp, model, _FakeExec({}))
        r = {"success": True, "parameters": dict(LOCKED), "fit_quality": {}, "script": "S"}
        res = c._complete_locked_schema(r, {"parameters": LOCKED}, np.zeros((5, 2)), 0, {})
        assert res["locked_schema_gap"] == [] and not model.prompts


def test_script_failure_keeps_refit_and_reports_gap():
    with tempfile.TemporaryDirectory() as tmp:
        c = _controller(tmp, _FakeModel("NEVER_RUNS"), _FakeExec({}))
        res = c._complete_locked_schema(_refit_result(), {"parameters": LOCKED},
                                        np.zeros((5, 2)), 1, {})
        assert res["parameters"] == REFIT
        assert len(res["locked_schema_gap"]) == 4
