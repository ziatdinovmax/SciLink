"""Offline tests for the analysis-side evidence contract on
recommend_measurements (#383): the optional measurement_history prompt
section and the machine-readable per-recommendation `target` field."""

import json
import logging

from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    """Captures the prompt and returns a canned recommendations JSON."""

    def __init__(self, payload):
        self.payload = payload
        self.last_contents = None

    def generate_content(self, contents, generation_config=None,
                         safety_settings=None, **kw):
        self.last_contents = contents
        return _FakeResponse(json.dumps(self.payload))


def _bare_agent(payload):
    agent = object.__new__(CurveFittingAgent)
    agent.logger = logging.getLogger("test_rec_contract")
    agent.model = _FakeModel(payload)
    agent.generation_config = None
    agent.safety_settings = None
    agent._stored_analysis_images = []
    agent._stored_analysis_metadata = {}
    return agent


PAYLOAD = {
    "analysis_integration": "test",
    "measurement_recommendations": [
        {"description": "acquire Fe L2,3", "scientific_justification": "valence",
         "priority": 1,
         "target": {"setting": "660-750 eV", "expected_signature": "Fe L2,3"}},
        {"description": "prep thinner region", "scientific_justification": "t/lambda",
         "priority": 2, "target": "not-a-dict"},
        {"description": "no target given", "scientific_justification": "x",
         "priority": 3},
    ],
}


def _prompt_text(model):
    return "\n".join(p for p in model.last_contents if isinstance(p, str))


def test_target_field_passes_through_and_normalizes():
    agent = _bare_agent(PAYLOAD)
    out = agent._generate_recommendations({"detailed_analysis": "spectrum shows X"})
    assert out["status"] == "success"
    recs = out["measurement_recommendations"]
    assert len(recs) == 3
    assert recs[0]["target"] == {"setting": "660-750 eV",
                                 "expected_signature": "Fe L2,3"}
    # non-dict and absent targets both normalize to None
    assert recs[1]["target"] is None
    assert recs[2]["target"] is None


def test_measurement_history_section_present_only_when_given():
    agent = _bare_agent(PAYLOAD)
    agent._generate_recommendations({"detailed_analysis": "x"})
    assert "Measurements Already Performed" not in _prompt_text(agent.model)

    agent._generate_recommendations(
        {"detailed_analysis": "x"},
        measurement_history=["low-loss -34-160 eV", "core-loss 660-750 eV"])
    text = _prompt_text(agent.model)
    assert "Measurements Already Performed" in text
    assert "660-750 eV" in text
    assert "Do not recommend re-acquiring" in text


def test_recommend_measurements_threads_history():
    agent = _bare_agent(PAYLOAD)
    out = agent.recommend_measurements(
        analysis_result={"detailed_analysis": "x"},
        measurement_history=["window A"])
    assert out["status"] == "success"
    assert len(out["measurement_recommendations"]) == 3
    assert "window A" in _prompt_text(agent.model)


def test_prompt_variants_declare_target():
    from scilink.agents.exp_agents.instruct import (
        CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS as C,
        SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS as S,
        IMAGE_ANALYSIS_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS as I,
    )
    for v in (C, S, I):
        assert "target" in v and "expected_signature" in v


def test_image_mime_type_sniffing():
    """#382: stored figures must be declared with their real media type."""
    import base64
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent
    sniff = BaseAnalysisAgent._image_mime_type
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 8
    assert sniff(png) == "image/png"
    assert sniff(jpg) == "image/jpeg"
    assert sniff(base64.b64encode(png).decode()) == "image/png"
    assert sniff(base64.b64encode(jpg).decode()) == "image/jpeg"
    assert sniff("garbage") == "image/jpeg"  # safe default


def test_stored_png_declared_as_png_in_recommendation_prompt():
    import base64
    agent = _bare_agent(PAYLOAD)
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    agent._stored_analysis_images = [{"label": "Fit", "data": png_b64}]
    out = agent._generate_recommendations({"detailed_analysis": "x"})
    assert out["status"] == "success"
    img_parts = [p for p in agent.model.last_contents if isinstance(p, dict)]
    assert img_parts and img_parts[0]["mime_type"] == "image/png"
