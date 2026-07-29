"""Offline tests for the Tier-A synthesis re-entry (issue #322 / #327 phase 3).

Covers: SynthesisReEntryController.revise (payload sources, feature grounding,
no mutation of the prior result), BaseAnalysisAgent.reenter_interpretation
(append-only storage, string→payload wrapping, claims validation), the moved
reflection pair re-exports, and the orchestrator reenter_interpretation tool
(record lookup, revision composition, novelty staleness ripple).
"""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ["UNSAFE_EXECUTION_OK"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

from qc_golden.harness import Rule, ScriptedModel  # noqa: E402

from scilink.agents.exp_agents._critique import CritiquePayload  # noqa: E402
from scilink.agents.exp_agents.controllers.base_controllers import (  # noqa: E402
    SYNTHESIS_REENTRY_INSTRUCTIONS,
    SynthesisReEntryController,
)

logging.basicConfig(level=logging.INFO)

_MARKER = "You are revising the final interpretation of a completed scientific analysis."

REVISED = {
    "detailed_analysis": "REVISED: the trend shows two regimes with a break at 270 K.",
    "scientific_claims": [{
        "claim": "The parameter trend exhibits a regime change near 270 K.",
        "scientific_impact": "Indicates a phase or exchange-regime transition.",
        "has_anyone_question": "Has anyone reported a 270 K transition in this system?",
        "keywords": ["regime change", "270 K", "transition"],
    }],
    "revision_summary": "Split the single-regime reading into two regimes per the critique.",
}

PRIOR = {
    "status": "success",
    "detailed_analysis": "ORIGINAL: the parameter trend is a single smooth regime.",
    "scientific_claims": [{
        "claim": "The trend is single-regime.",
        "scientific_impact": "x", "has_anyone_question": "Has anyone ...?",
        "keywords": ["single"],
    }],
    "model_type": "Voigt x2",
    "fitting_parameters": {"center": 4.2},
    "fit_quality": {"r_squared": 0.997},
}


def _parse(resp):
    return json.loads(resp.text), None


def _controller(model):
    return SynthesisReEntryController(
        model=model, logger=logging.getLogger("reentry_test"),
        generation_config=None, safety_settings=None, parse_fn=_parse,
    )


def test_revise_grounds_prompt_and_returns_revision():
    model = ScriptedModel([Rule("reentry", _MARKER, [REVISED], repeat_last=True)])
    payload = CritiquePayload(source="human",
                              critique="There is a slope break at 270 K.")
    prior_snapshot = json.dumps(PRIOR, sort_keys=True)
    revision, err = _controller(model).revise(
        PRIOR, payload,
        features_block=json.dumps({"fit_quality": {"r_squared": 0.997}}),
    )
    assert err is None
    assert revision["detailed_analysis"].startswith("REVISED")
    assert revision["source"] == "human"
    prompt = model.calls[0]["prompt"]
    assert "ORIGINAL: the parameter trend" in prompt
    assert "There is a slope break at 270 K." in prompt
    assert '"r_squared": 0.997' in prompt
    assert "(source: human)" in prompt
    # prior result untouched
    assert json.dumps(PRIOR, sort_keys=True) == prior_snapshot


def test_revise_error_paths():
    model = ScriptedModel([Rule("reentry", _MARKER,
                                [{"revision_summary": "no text"}], repeat_last=True)])
    payload = CritiquePayload(source="verifier", critique="bad")
    revision, err = _controller(model).revise(PRIOR, payload)
    assert revision is None and err is not None


def test_agent_reenter_interpretation_appends():
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent

    class _Concrete(BaseAnalysisAgent):
        def analyze(self, data, system_info=None, **kwargs):  # pragma: no cover
            raise NotImplementedError

    agent = _Concrete.__new__(_Concrete)
    agent.logger = logging.getLogger("agent_reentry_test")
    agent.model = ScriptedModel([Rule("reentry", _MARKER, [REVISED], repeat_last=True)])
    agent.generation_config = None
    agent.safety_settings = None
    agent._stored_analysis_images = []
    agent._stored_analysis_metadata = {}

    result = json.loads(json.dumps(PRIOR))  # deep copy
    out = agent.reenter_interpretation(result, "slope break at 270 K")
    assert out["status"] == "success"
    revs = result["interpretation_revisions"]
    assert len(revs) == 1
    assert revs[0]["revised_analysis"].startswith("REVISED")
    assert revs[0]["source"] == "human"
    assert revs[0]["revised_claims"][0]["claim"].startswith("The parameter trend")
    # original text untouched
    assert result["detailed_analysis"].startswith("ORIGINAL")
    # second call appends
    out2 = agent.reenter_interpretation(
        result, CritiquePayload(source="orchestrator", critique="also mention hysteresis"))
    assert out2["status"] == "success"
    assert len(result["interpretation_revisions"]) == 2
    assert result["interpretation_revisions"][1]["source"] == "orchestrator"


def test_surface_features_modalities():
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent as B
    f = B.surface_features_for_reentry(PRIOR)
    assert set(f) == {"model_type", "fit_quality", "fitting_parameters"}
    f2 = B.surface_features_for_reentry(
        {"parameter_trends": {"fwhm": "decreasing"}, "fitting_parameters": {"a": 1}})
    assert "parameter_trends" in f2 and "fitting_parameters" not in f2
    f3 = B.surface_features_for_reentry({"extracted_features": {"blob_count": 3}})
    assert f3 == {"extracted_features": {"blob_count": 3}}


def test_reflection_pair_reexports_and_defaults():
    from scilink.agents.exp_agents.controllers import base_controllers as bc
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    from scilink.agents.exp_agents.instruct import SPECTROSCOPY_REFLECTION_INSTRUCTIONS

    assert hc.RunSelfReflectionController is bc.RunSelfReflectionController
    assert hc.ApplyReflectionUpdatesController is bc.ApplyReflectionUpdatesController
    ctrl = bc.RunSelfReflectionController(
        model=None, logger=logging.getLogger("x"), generation_config=None,
        safety_settings=None, parse_fn=_parse)
    assert ctrl.instructions == SPECTROSCOPY_REFLECTION_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Orchestrator tool
# ---------------------------------------------------------------------------

def _fake_orch(model, records):
    return SimpleNamespace(
        analysis_results=records,
        model=model,
        futurehouse_api_key=None,
        current_metadata=None,
        base_dir=Path("/tmp"),
    )


def _make_tools(orch):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        AnalysisOrchestratorTools,
    )
    tools = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    tools.orch = orch
    tools.functions_map = {}
    tools.openai_schemas = []
    tools._register_all_tools()
    return tools


def test_orchestrator_reenter_tool_end_to_end():
    model = ScriptedModel([Rule("reentry", _MARKER, [REVISED], repeat_last=True)])
    record = {
        "analysis_id": "raman_curve_001",
        "full_result": json.loads(json.dumps(PRIOR)),
        "novelty_assessment": {"summary_stats": {"average_score": 3.0}},
        "output_directory": "/nonexistent",
    }
    tools = _make_tools(_fake_orch(model, [record]))
    assert "reenter_interpretation" in tools.functions_map

    out = json.loads(tools.functions_map["reenter_interpretation"](
        critique="the trend has a break at 270 K"))
    assert out["status"] == "success", out
    assert out["analysis_id"] == "raman_curve_001"
    assert out["prior_novelty_assessment_stale"] is True
    revs = record["interpretation_revisions"]
    assert len(revs) == 1 and revs[0]["revised_analysis"].startswith("REVISED")
    assert record["novelty_assessment"]["stale"] is True
    # original interpretation untouched
    assert record["full_result"]["detailed_analysis"].startswith("ORIGINAL")

    # successive critique composes: prompt #2 must contain revision #1's text
    out2 = json.loads(tools.functions_map["reenter_interpretation"](
        critique="also mention hysteresis"))
    assert out2["status"] == "success"
    assert "REVISED: the trend shows two regimes" in model.calls[1]["prompt"]
    assert len(record["interpretation_revisions"]) == 2


def test_orchestrator_reenter_tool_errors():
    model = ScriptedModel([Rule("reentry", _MARKER, [REVISED], repeat_last=True)])
    tools = _make_tools(_fake_orch(model, []))
    out = json.loads(tools.functions_map["reenter_interpretation"](critique="x"))
    assert out["status"] == "error"
    out = json.loads(tools.functions_map["reenter_interpretation"](critique="  "))
    assert out["status"] == "error"

    tools2 = _make_tools(_fake_orch(model, [{"analysis_id": "a", "full_result": {}}]))
    out = json.loads(tools2.functions_map["reenter_interpretation"](
        critique="x", analysis_id="missing"))
    assert out["status"] == "error"
    out = json.loads(tools2.functions_map["reenter_interpretation"](critique="x"))
    assert out["status"] == "error"  # no interpretation text on record


def test_refine_interpretation_features_include_extracted():
    """The #323 tool's feature surfacing now covers image/HS records."""
    import inspect
    from scilink.agents.exp_agents import analysis_orchestrator_tools as aot
    src = inspect.getsource(aot.AnalysisOrchestratorTools._register_all_tools)
    # the additive line exists inside refine_interpretation's feature block
    assert 'full_result.get("extracted_features")' in src
