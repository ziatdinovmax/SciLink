"""Verifier issues are logged even when the model returns a field as a list.

Before the fix, a list-valued suggested_fix (or recommended_action)
reached ``_wrap_text``, whose ``text.split()`` raised
"'list' object has no attribute 'split'". The exception escaped the logging
call and failed the whole candidate (score 0) after it had already run.
"""
import logging

import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
from scilink.agents.exp_agents.controllers.image_analysis_controllers import UnifiedImageProcessingController

CONTROLLERS = [UnifiedImageProcessingController, UnifiedSeriesProcessingController]


def _ctrl(cls, name):
    ctrl = object.__new__(cls)  # only the logging helpers are exercised
    ctrl.logger = logging.getLogger(name)
    return ctrl


@pytest.mark.parametrize("cls", CONTROLLERS)
def test_list_valued_fields_are_logged_not_raised(cls, caplog):
    ctrl = _ctrl(cls, f"verifier_log_{cls.__name__}")
    verification = {
        "overall_assessment": ["Segmentation over-merged.", "Twin stats unusable."],
        "issues_found": [{
            "location": "whole field",
            "problem": "Only 15 grains remain against about 50-80 visible.",
            "suggested_fix": ["Lower the merge threshold to 3", "re-run the RAG merge"],
        }],
        "recommended_action": ["Re-segment", "then recount twins"],
    }
    with caplog.at_level(logging.INFO, logger=ctrl.logger.name):
        ctrl._log_verification_issues(verification)
    text = caplog.text
    assert "Lower the merge threshold to 3; re-run the RAG merge" in text
    assert "Re-segment; then recount twins" in text
    assert "Could not format" not in text


@pytest.mark.parametrize("cls", CONTROLLERS)
def test_malformed_payload_warns_instead_of_raising(cls, caplog):
    ctrl = _ctrl(cls, f"verifier_log_bad_{cls.__name__}")
    with caplog.at_level(logging.INFO, logger=ctrl.logger.name):
        ctrl._log_verification_issues({"issues_found": ["a bare string, not a dict"]})
    assert "Could not format verification issues" in caplog.text


@pytest.mark.parametrize("cls", CONTROLLERS)
@pytest.mark.parametrize("value, expected", [
    (None, [""]), ("", [""]), ([], [""]), (3.5, ["3.5"]),
    (["a", "b"], ["a; b"]), ("plain text", ["plain text"]),
])
def test_wrap_text_accepts_non_strings(cls, value, expected):
    assert _ctrl(cls, "wrap")._wrap_text(value, width=70) == expected
