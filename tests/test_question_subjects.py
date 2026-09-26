"""The gates that declare what is under review (scilink.hitl ``subject``):
their block builders, and that the request they ask carries it."""

from types import SimpleNamespace

import pytest

from scilink import hitl
from scilink.agents.exp_agents.human_feedback import (
    SimpleFeedbackCollector, analysis_result_subject)
from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cfc


@pytest.fixture(autouse=True)
def _reset():
    yield
    hitl.set_default_channel(None)


class Capture:
    def __init__(self, answer=""):
        self.answer = answer
        self.req = None

    def ask(self, req):
        self.req = req
        return self.answer


def test_subject_block_vocabulary_is_checked():
    b = hitl.subject_block("chips", label="P", items=["a"])
    assert b == {"type": "chips", "label": "P", "items": ["a"]}
    with pytest.raises(ValueError):
        hitl.subject_block("hologram")
    s = hitl.make_subject("T", [b, None, {}])
    assert s == {"title": "T", "blocks": [b]}


def test_numbered_steps_split_only_after_a_sentence_end():
    assert cfc._numbered_steps("1. Baseline with 8.7 cm-1 window. 2. Fit two peaks") == \
        ["Baseline with 8.7 cm-1 window.", "Fit two peaks."]
    assert cfc._numbered_steps("Fit one Gaussian.") == ["Fit one Gaussian."]
    assert cfc._numbered_steps("") == []


def test_fitting_plan_subject_single_and_series():
    state = {"is_single_spectrum": True, "observations": "two peaks",
             "analysis_approach": "peak fit", "physical_model": "2 Lorentzians",
             "parameters_to_extract": ["center", "fwhm"],
             "fitting_strategy": "1. Subtract baseline. 2. Fit both peaks."}
    s = cfc.fitting_plan_subject(state)
    assert s["title"] == "📋 Proposed fitting plan — single spectrum"
    assert [(b["type"], b["label"]) for b in s["blocks"]] == [
        ("text", "🔍 Observations"), ("text", "📊 Approach"), ("text", "📐 Physical model"),
        ("text", "🎯 Parameters to extract"), ("steps", "⚙️ Fitting strategy")]
    assert s["blocks"][2]["markdown"] == "2 Lorentzians"
    assert s["blocks"][3]["markdown"] == "center, fwhm"
    assert s["blocks"][4]["items"] == ["Subtract baseline.", "Fit both peaks."]

    series = dict(state, is_single_spectrum=False, num_spectra=6,
                  series_metadata={"values": [100, 200, 300, 400, 500, 600], "unit": "K"},
                  series_analysis_plan={
                      "rationale": "a phase change",
                      # a regime left sparse falls back to the series plan's
                      # fields, as the printer does
                      "physical_model": "2 Lorentzians",
                      "parameters_to_extract": ["center", "fwhm"],
                      "regimes": [{"name": "low", "spectrum_indices": [0, 1, 2]},
                                  {"name": "high", "spectrum_indices": [3, 4, 5],
                                   "physical_model": "1 Lorentzian"}],
                      "transition_points": [{"between_indices": [2, 3], "description": "split"}]})
    s = cfc.fitting_plan_subject(series)
    table = next(b for b in s["blocks"] if b["type"] == "table")
    assert table["label"] == "📦 Series fitting regimes (2)"
    assert table["rows"][0] == [1, "low", "[0, 1, 2] (100–300 K)", "2 Lorentzians", "center, fwhm"]
    assert table["rows"][1][3] == "1 Lorentzian"
    assert [b["type"] for b in s["blocks"]][-3:] == ["table", "text", "table"]
    locked = cfc.fitting_plan_subject(dict(state, is_single_spectrum=False, num_spectra=3))
    assert locked["blocks"][-1]["type"] == "notice"


def test_fitting_plan_gate_asks_with_the_subject(capsys):
    cap = Capture(answer="use Voigt")
    hitl.set_default_channel(cap)
    owner = SimpleNamespace(_display_plan=lambda state: None)
    state = {"is_single_spectrum": True, "physical_model": "G"}
    out = cfc.CurveFittingPlanningController._get_human_feedback(owner, state)
    assert cap.req.kind == "review_plan" and cap.req.origin == {"stage": "fitting_plan"}
    assert cap.req.subject["title"].startswith("📋 Proposed fitting plan")
    assert out["_refine_feedback"] == "use Voigt"


def test_analysis_result_subject_and_gate(capsys):
    result = {"detailed_analysis": "The doublet narrows.",
              "scientific_claims": [{"claim": "Peak A narrows", "scientific_impact": "strain",
                                     "has_anyone_question": "Has anyone…", "keywords": ["raman"]}]}
    s = analysis_result_subject(result)
    assert s["title"] == "🤖 Agent's analysis results"
    assert s["blocks"][0] == {"type": "text", "label": "📋 Detailed analysis",
                              "markdown": "The doublet narrows."}
    assert s["blocks"][1]["label"] == "🎯 Scientific claims (1)"
    assert s["blocks"][1]["items"][0]["keywords"] == ["raman"]
    assert analysis_result_subject({})["blocks"][1]["type"] == "notice"

    cap = Capture()
    hitl.set_default_channel(cap)
    assert SimpleFeedbackCollector().collect_optional_feedback(result) is None
    assert cap.req.kind == "review_result" and cap.req.subject == s
    assert "CLAIM 1:" in capsys.readouterr().out       # the console printout is unchanged
