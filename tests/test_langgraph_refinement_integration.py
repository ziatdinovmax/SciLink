"""
Integration tests for the refinement-subgraph migration — pytest-discoverable.

Unlike tests/test_langgraph_refinement.py (pure subgraph logic, every
controller method stubbed out), these tests call each controller's real
``execute()`` top to bottom: real prompt-building, real adapters, real
``_refinement_subgraph`` construction and invocation, real image/FFT/SAM
post-processing helpers. Only the true external boundary is mocked:

* ``self.model.generate_content`` / the parse callback — no live LLM calls,
  no API key required.
* ``run_sam_analysis`` — wraps atomai's SAM model (~2.5GB checkpoint), the
  one dependency genuinely too heavy to run in CI.

Everything else (SlidingFFTNMF's real NMF decomposition, atomai's
ParticleAnalyzer visualization/statistics code, the controllers' own
prompt-building and state-mutation methods) runs for real. This is what
catches wiring bugs unit tests with stubbed controller methods cannot:
adapters passing the wrong state shape into a real method, or realistic
numpy-scalar-laden data breaking LangGraph's MemorySaver checkpointing.

No live LLM calls are made; no API keys are required.
"""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


class _SequencedParser:
    """
    Fake ``parse_fn``/``parse_llm_response`` callback that returns a
    scripted (dict, error) pair per call, repeating the last entry once
    exhausted. The controllers under test only ever pass the parsed dict
    onward — the raw ``response`` object is opaque to them, so a bare
    ``MagicMock()`` model response is fine.
    """

    def __init__(self, responses: List[Tuple[Optional[Dict[str, Any]], Optional[str]]]):
        self._responses = responses
        self.calls = 0

    def __call__(self, response) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _FakeModel:
    """``self.model`` stand-in — generate_content is never inspected by
    the controllers under test, only its return value is passed to the
    parse callback."""

    def generate_content(self, *args, **kwargs):
        return object()


# ===========================================================================
# Site #1 — ImagePlanningController (image_analysis_controllers.py)
# ===========================================================================

class TestImagePlanningControllerIntegration:
    def _make_state(self):
        return {
            "is_single_image": True,
            "num_images": 1,
            "original_image_bytes": b"fake-jpeg-bytes",
            "image_statistics": {"mean": np.float64(1.2), "std": np.float64(0.3)},
            "system_info": {},
        }

    def test_real_execute_accept_after_one_refine(self, monkeypatch):
        from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
            ImagePlanningController,
        )

        parse_fn = _SequencedParser([
            # 1: _plan_analysis
            ({
                "observations": "obs", "analysis_approach": "approach-v1",
                "processing_pipeline": "step1 -> step2", "features_to_extract": ["f1"],
                "quality_criteria": "qc", "expected_outputs": [], "literature_query": None,
                "series_analysis_plan": None,
            }, None),
            # 2: _validate_plan — empty dict is a safe no-op
            ({}, None),
            # 3: _refine_plan
            ({
                "observations": "obs2", "analysis_approach": "approach-v2",
                "processing_pipeline": "step1 -> step2 -> step3", "features_to_extract": ["f1", "f2"],
                "quality_criteria": "qc2",
            }, None),
        ])

        inputs = iter(["make it better", ""])  # one refine round, then accept
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        ctrl = ImagePlanningController(
            model=_FakeModel(), logger=_logger("img-plan-it"), generation_config=None,
            safety_settings=None, parse_fn=parse_fn, instructions="instructions",
            output_dir="/tmp", enable_human_feedback=True, max_iterations=3,
        )
        result = ctrl.execute(self._make_state())

        assert result["analysis_approach"] == "approach-v2"
        assert result["locked_analysis_config"]["analysis_approach"] == "approach-v2"
        assert result["locked_analysis_config"]["features_to_extract"] == ["f1", "f2"]


# ===========================================================================
# Site #2 — HumanFeedbackRefinementController (curve_fitting_controllers.py)
# ===========================================================================

class TestCurveFittingPlanningControllerIntegration:
    def _make_state(self):
        return {
            "is_single_spectrum": True,
            "num_spectra": 1,
            "original_plot_bytes": b"fake-png-bytes",
            "data_statistics": {"mean": np.float64(1.2)},
            "system_info": {},
        }

    def test_real_execute_max_iterations_exhaustion(self, monkeypatch):
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            HumanFeedbackRefinementController,
        )

        refine_dict = {
            "observations": "obs2", "analysis_approach": "approach", "physical_model": "model-v2",
            "parameters_to_extract": ["p1"], "fitting_strategy": "strategy-v2", "literature_query": None,
        }
        parse_fn = _SequencedParser([
            ({  # 1: _plan_analysis
                "observations": "obs", "analysis_approach": "approach", "physical_model": "model-v1",
                "parameters_to_extract": ["p1"], "fitting_strategy": "strategy", "literature_query": None,
                "series_analysis_plan": None,
            }, None),
            ({}, None),          # 2: _validate_plan — no-op
            (refine_dict, None),  # 3: _refine_plan (round 1)
            (refine_dict, None),  # 4: _refine_plan (round 2)
            (refine_dict, None),  # 5: _refine_plan (round 3)
        ])

        # Always ask for refinement — never accept — to exercise exhaustion.
        monkeypatch.setattr(builtins, "input", lambda prompt="": "keep tweaking")
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        ctrl = HumanFeedbackRefinementController(
            model=_FakeModel(), logger=_logger("cf-plan-it"), generation_config=None,
            safety_settings=None, parse_fn=parse_fn, instructions="instructions",
            output_dir="/tmp", enable_human_feedback=True, max_iterations=3,
        )
        result = ctrl.execute(self._make_state())

        assert result["physical_model"] == "model-v2"
        assert result["locked_fitting_config"]["physical_model"] == "model-v2"


# ===========================================================================
# Site #3 — fft_microscopy_controllers.HumanFeedbackRefinementController
# Runs the REAL SlidingFFTNMF decomposition (pure numpy/sklearn, no
# downloaded weights) — only the LLM feedback-to-params conversion is mocked.
# ===========================================================================

class TestFFTRefinementControllerIntegration:
    def test_real_execute_with_real_nmf_decomposition(self, monkeypatch, tmp_path):
        from scilink.agents.exp_agents.controllers.fft_microscopy_controllers import (
            HumanFeedbackRefinementController,
        )

        rng = np.random.default_rng(0)
        image = rng.random((200, 200))  # big enough for >1 window at the default 64px window size

        # First round: natural-language feedback -> LLM converts to params.
        # Second round: empty input -> accept.
        parse_fn = _SequencedParser([
            ({"n_components": 2}, None),
        ])
        inputs = iter(["use 2 components", ""])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        ctrl = HumanFeedbackRefinementController(
            model=_FakeModel(), logger=_logger("fft-it"), generation_config=None,
            safety_settings=None, parse_fn=parse_fn,
            settings={"max_feedback_iterations": 3, "output_dir": str(tmp_path)},
        )
        state = {
            "enable_human_feedback": True,
            "is_single_image": True,
            "preprocessed_image_array": image,
            "llm_params": {"window_size_nm": None, "n_components": 4},
            "current_params": {"n_components": 4},
            "nm_per_pixel": 1.0,
        }
        result = ctrl.execute(state)

        assert result["llm_params"]["n_components"] == 2
        assert result["locked_params"]["n_components"] == 2
        # Real SlidingFFTNMF actually ran and produced real arrays.
        assert result["fft_components"].shape[0] == 2
        assert isinstance(result["fft_components"], np.ndarray)


# ===========================================================================
# Sites #4/#5 — sam_microscopy_controllers
# Only run_sam_analysis (the ~2.5GB SAM checkpoint) is mocked. Everything
# else — visualize_sam_results, convert_numpy_to_jpeg_bytes,
# calculate_sam_statistics — runs for real against a realistic,
# numpy-scalar-laden sam_result fixture (mirrors atomai's actual output
# shape closely enough to exercise the real code, not a hand-picked dict).
# ===========================================================================

def _make_sam_result(n: int, size: int = 16, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rgb_image = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    particles = []
    masks = []
    for i in range(n):
        mask = np.zeros((size, size), dtype=bool)
        y, x = (i * 3) % size, (i * 5) % size
        mask[y:y + 2, x:x + 2] = True
        particle = {
            "id": i,
            "mask": mask,
            "centroid": (np.float64(x), np.float64(y)),
            "area": np.float64(4.0),
            "bbox": (x, y, 2, 2),
        }
        particles.append(particle)
        masks.append(mask)
    return {
        "total_count": np.int64(n),
        "rgb_image": rgb_image,
        "original_image": rgb_image[:, :, 0],
        "particles": particles,
        "masks": masks,
        "areas": [np.float64(4.0)] * n,
        "parameters": {"min_area": 500},
    }


class TestSAMHumanRefinementControllerIntegration:
    def test_real_execute_accept_with_realistic_numpy_fixture(self, monkeypatch):
        from scilink.agents.exp_agents.controllers.sam_microscopy_controllers import (
            HumanFeedbackRefinementController,
        )

        parse_fn = _SequencedParser([({"needs_refinement": False, "reasoning": "looks good", "evaluation": {"overall_quality": "good"}}, None)])
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # accept LLM recommendation
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        ctrl = HumanFeedbackRefinementController(
            model=_FakeModel(), logger=_logger("sam-hf-it"), generation_config=None,
            safety_settings=None, parse_fn=parse_fn, settings={"max_feedback_iterations": 3},
        )
        state = {
            "enable_human_feedback": True,
            "is_single_image": True,
            "image_blob": {"data": b"fake-jpeg"},
            "sam_result": _make_sam_result(5),
            "summary_stats": {},
            "current_params": {"min_area": 500},
        }
        result = ctrl.execute(state)

        assert result["refinement_complete"] is True
        assert result["final_params_for_batch"] == {"min_area": 500}
        assert len(result["refinement_history"]) == 1

    def test_real_execute_refine_then_accept_with_real_sam_rerun_mocked(self, monkeypatch):
        from scilink.agents.exp_agents.controllers.sam_microscopy_controllers import (
            HumanFeedbackRefinementController,
        )
        import scilink.agents.exp_agents.controllers.sam_microscopy_controllers as sammod

        parse_fn = _SequencedParser([
            ({"needs_refinement": True, "reasoning": "too few particles",
              "evaluation": {"overall_quality": "poor"},
              "recommended_parameters": {"min_area": 100}}, None),
            ({"needs_refinement": False, "reasoning": "looks good now",
              "evaluation": {"overall_quality": "good"}}, None),
        ])
        # Round 1: empty input + needs_refinement=True -> "use_llm" (applies
        # the recommendation, does NOT accept). Round 2: empty input +
        # needs_refinement=False -> real accept.
        inputs = iter(["", ""])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        rerun_calls = []

        def fake_run_sam_analysis(image_array, params):
            rerun_calls.append(dict(params))
            return _make_sam_result(9, seed=1)

        monkeypatch.setattr(sammod, "run_sam_analysis", fake_run_sam_analysis)

        ctrl = HumanFeedbackRefinementController(
            model=_FakeModel(), logger=_logger("sam-hf-it2"), generation_config=None,
            safety_settings=None, parse_fn=parse_fn, settings={"max_feedback_iterations": 3},
        )
        state = {
            "enable_human_feedback": True,
            "is_single_image": True,
            "image_blob": {"data": b"fake-jpeg"},
            "sam_result": _make_sam_result(5),
            "summary_stats": {},
            "current_params": {"min_area": 500},
            "preprocessed_image_array": np.zeros((16, 16)),
            "image_path": "fake.png",
            "nm_per_pixel": np.float64(1.0),
        }
        result = ctrl.execute(state)

        assert rerun_calls == [{"min_area": 100}]
        assert result["current_params"] == {"min_area": 100}
        # Real calculate_sam_statistics ran against the mocked rerun's realistic fixture.
        assert result["summary_stats"]["total_particles"] == 9
        assert len(result["refinement_history"]) == 2


class TestSAMAutomatedRefinementControllerIntegration:
    def test_real_execute_judge_on_exhaustion_with_real_helpers(self, monkeypatch):
        from scilink.agents.exp_agents.controllers.sam_microscopy_controllers import (
            AutomatedLLMRefinementController,
        )
        import scilink.agents.exp_agents.controllers.sam_microscopy_controllers as sammod

        # Two rounds of "refine", each with different recommended params;
        # then real _judge_select_best_iteration runs (mocked LLM response
        # inside it) and picks a winner.
        eval_parse = _SequencedParser([
            ({"decision": "refine", "reasoning": "r1",
              "evaluation": {"overall_quality": "poor", "coverage_score": 3, "accuracy_score": 3},
              "recommended_parameters": {"min_area": 50}}, None),
            ({"decision": "refine", "reasoning": "r2",
              "evaluation": {"overall_quality": "ok", "coverage_score": 7, "accuracy_score": 8},
              "recommended_parameters": {"min_area": 30}}, None),
            # Judge's own parse call — 1-based iteration 2 is the winner.
            ({"selected_iteration": 2, "reasoning": "second round covered more particles",
              "confidence": "high"}, None),
        ])
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        rerun_calls = []

        def fake_run_sam_analysis(image_array, params):
            rerun_calls.append(dict(params))
            return _make_sam_result(6 + len(rerun_calls), seed=len(rerun_calls))

        monkeypatch.setattr(sammod, "run_sam_analysis", fake_run_sam_analysis)

        ctrl = AutomatedLLMRefinementController(
            model=_FakeModel(), logger=_logger("sam-auto-it"), generation_config=None,
            safety_settings=None, parse_fn=eval_parse,
            settings={"max_auto_refinement_iterations": 2},
        )
        state = {
            "enable_human_feedback": False,
            "is_single_image": True,
            "image_blob": {"data": b"fake-jpeg"},
            "sam_result": _make_sam_result(5),
            "summary_stats": {},
            "current_params": {"min_area": 500},
            "preprocessed_image_array": np.zeros((16, 16)),
            "image_path": "fake.png",
            "nm_per_pixel": np.float64(1.0),
        }
        result = ctrl.execute(state)

        assert rerun_calls == [{"min_area": 50}, {"min_area": 30}]
        assert result["judge_invoked"] is True
        assert result["judge_selected_iteration"] == 2
        # Iteration 2's params_used is the snapshot that PRODUCED the
        # judged sam_result (50, applied after round 1) — not 30, which was
        # only recommended, never itself applied/evaluated.
        assert result["current_params"] == {"min_area": 50}
        assert len(result["refinement_history"]) == 2
