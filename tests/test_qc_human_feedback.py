"""Offline tests for the human-feedback paths through the QC engine hooks.

These interactive paths (curve ``adjust_threshold`` / ``retry`` in
qc_fallback, image poor-quality ``retry``, image CO_PILOT accept approval in
qc_post_verification) were moved verbatim in the #327 phase-4 extraction but
had no automated coverage (all other suites run AUTONOMOUS). Here the agents
run under a ScriptedModel with ``builtins.input`` monkeypatched; responses
key on the prompt text so unrelated interactive prompts default to accept.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ["UNSAFE_EXECUTION_OK"] = "true"

sys.path.insert(0, str(Path(__file__).parent))

from qc_golden.fixtures import write_blob_image, write_gaussian_spectrum  # noqa: E402
from qc_golden.harness import Rule, ScriptedModel  # noqa: E402
from qc_golden.scenarios import (  # noqa: E402
    curve_rules,
    curve_script,
    image_rules,
    image_script,
)

logging.basicConfig(level=logging.INFO)


# --- scripted "never approved" rule sets -----------------------------------

def _curve_fail_rules():
    """Happy-shaped rules whose fits stay at R² = 0.60 and never get accepted."""
    rules = curve_rules("happy")
    for r in rules:
        if r.name == "codegen":
            r.responses = [lambda n, t: {"script": curve_script(
                f"attempt-{n}", "linear background only", 0.60)}]
        elif r.name == "verify":
            from qc_golden.scenarios import _curve_verify_reject
            r.responses = [_curve_verify_reject()]
    rules.append(Rule(
        "human_refine", "Refine the fitting approach based on user feedback.",
        [{
            "physical_model": "Gaussian + linear background (human-guided)",
            "fitting_strategy": "follow the user's suggestion",
            "parameters_to_extract": ["amplitude", "center", "sigma"],
            "analysis_approach": "human-guided single peak",
        }], repeat_last=True))
    rules.append(Rule(
        "judge", "You are a scientific data fitting expert acting as a judge.",
        [{"selected_index": 0, "acceptable": True, "issues_with_selected": "",
          "reasoning": "best available canned fit"}], repeat_last=True))
    return rules


def _image_fail_rules():
    """Image rules whose attempts never carry the accepted tag (score 0.4)."""
    rules = image_rules("happy")
    for r in rules:
        if r.name in ("codegen", "codegen_refit"):
            r.responses = [lambda n, t: {"script": image_script(f"attempt-{n}")}]
        elif r.name == "verify":
            from qc_golden.scenarios import _image_verify_reject
            r.responses = [_image_verify_reject()]
    rules.append(Rule(
        "human_refine", "Refine the image analysis approach based on user feedback.",
        [{
            "processing_pipeline": "human-guided threshold -> label -> regionprops",
            "analysis_approach": "human-guided blob segmentation",
            "features_to_extract": ["blob_count", "mask_fraction"],
            "quality_criteria": "all blobs segmented",
        }], repeat_last=True))
    rules.append(Rule(
        "judge", "You are a scientific image analysis expert acting as a judge.",
        [{"selected_index": 0, "reasoning": "best available canned attempt",
          "score_explanation": "verifier kept rejecting the canned mask"}],
        repeat_last=True))
    return rules


def _patch_input(monkeypatch, poor_fit_response: str):
    """input() stub: answer the poor-fit/poor-quality prompt with
    ``poor_fit_response``; every other interactive prompt gets Enter."""
    seen = []

    def fake_input(prompt=""):
        seen.append(str(prompt))
        if "Your input" in str(prompt):
            return poor_fit_response
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    return seen


# --- curve ------------------------------------------------------------------

def _run_curve(tmp_path, monkeypatch, response):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")
    seen = _patch_input(monkeypatch, response)

    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"),
        enable_human_feedback=True, use_literature=False,
    )
    model = ScriptedModel(_curve_fail_rules())
    agent.model = model
    result = agent.analyze(str(spectrum), max_verification_iterations=2)
    return result, model, seen


def test_curve_adjust_threshold(tmp_path, monkeypatch):
    result, model, seen = _run_curve(tmp_path, monkeypatch, "threshold 0.5")
    assert result["status"] == "success", result.get("error")
    assert any("Your input" in p for p in seen)
    # adjust_threshold returns the best fit directly: history attached and
    # approved under the adjusted threshold; the judge is never consulted.
    qh = result["quality_history"]
    assert qh["approved"] is True
    assert qh["threshold"] == 0.5
    assert not any(c["rule"] == "judge" for c in model.calls)
    assert abs(result["fit_quality"]["r_squared"] - 0.60) < 1e-6


def test_curve_retry_then_judge(tmp_path, monkeypatch):
    result, model, seen = _run_curve(
        tmp_path, monkeypatch, "retry: add a Gaussian peak component")
    assert result["status"] == "success", result.get("error")
    calls = [c["rule"] for c in model.calls]
    # the human feedback refined the model (LLM call), the refit ran, and the
    # judge adjudicated all attempts afterwards
    assert "human_refine" in calls
    assert "judge" in calls
    assert result.get("quality_warning")  # still below the 0.95 threshold
    assert result["quality_history"]["approved"] is False


# --- image ------------------------------------------------------------------

def test_image_retry_then_judge(tmp_path, monkeypatch):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = write_blob_image(data_dir / "blobs.npy")
    seen = _patch_input(monkeypatch, "retry: lower the threshold further")

    agent = ImageAnalysisAgent(
        output_dir=str(tmp_path / "out"),
        enable_human_feedback=True, use_literature=False,
    )
    model = ScriptedModel(_image_fail_rules())
    agent.model = model
    result = agent.analyze(str(image), max_verification_iterations=2)

    assert result["status"] == "success", result.get("error")
    calls = [c["rule"] for c in model.calls]
    assert "human_refine" in calls
    assert "judge" in calls
    # the judge's score_explanation is folded into the quality warning
    assert "verifier kept rejecting" in (result.get("quality_warning") or "")


def test_image_copilot_accept_approval(tmp_path, monkeypatch):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = write_blob_image(data_dir / "blobs.npy")
    seen = _patch_input(monkeypatch, "")  # Enter everywhere = accept

    agent = ImageAnalysisAgent(
        output_dir=str(tmp_path / "out"),
        enable_human_feedback=True, use_literature=False,
    )
    model = ScriptedModel(image_rules("happy"))
    agent.model = model
    result = agent.analyze(str(image))

    assert result["status"] == "success", result.get("error")
    assert result["quality_history"]["approved"] is True
    # the CO_PILOT accept-approval prompt was actually shown
    assert any("feedback" in p.lower() or "Enter to accept" in p for p in seen)
