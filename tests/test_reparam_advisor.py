"""ReparameterizationAdvisor recommends how to fix a flagged force field.

Given the inconsistent pure-component properties the critic flagged, it proposes
a concrete corrective action (and whether a human must approve/supply it). The
LLM is stubbed; no API key.
"""

import json
import logging

from scilink.agents.sim_agents.critics import ReparameterizationAdvisor


def _stub(fake_text: str):
    obj = ReparameterizationAdvisor.__new__(ReparameterizationAdvisor)
    obj.logger = logging.getLogger("test_reparam_advisor")
    obj.futurehouse_api_key = None
    captured = {}

    class _Model:
        def generate_content(self, prompt, generation_config=None):
            captured["prompt"] = prompt

            class _Resp:
                text = fake_text

            return _Resp()

    obj.model = _Model()
    return obj, captured


FLAGGED = [
    {"component": "EIS", "property": "density", "consistent": False,
     "reasoning": "Far below the known density of a sulfone."},
]


def test_flagged_properties_and_backend_reach_the_prompt():
    adv, captured = _stub(json.dumps({"recommended_action": "add_force_field"}))
    adv.advise(FLAGGED, system_description="aqueous sulfone electrolyte",
               backend="openff")
    prompt = captured["prompt"]
    assert "EIS" in prompt and "density" in prompt
    assert "Far below the known density" in prompt        # the critic's reasoning
    assert "openff" in prompt


def test_recommendation_passes_through():
    adv, _ = _stub(json.dumps({
        "status": "success",
        "diagnosis": "EIS density is low; the sulfone vdW/charges are under-parameterized.",
        "recommended_action": "add_force_field",
        "detail": "Supplement EIS with a validated sulfone parameter set via extra_force_fields.",
        "requires_human": True,
        "rationale": "Base Sage does not cover sulfones well.",
    }))
    rec = adv.advise(FLAGGED, backend="openff")
    assert rec["recommended_action"] == "add_force_field"
    assert rec["requires_human"] is True
    assert "extra_force_fields" in rec["detail"]


def test_defaults_when_model_underspecifies():
    # Model returns only a diagnosis → advisor fills safe defaults (escalate,
    # human-required) rather than emit an actionable-looking blank.
    adv, _ = _stub(json.dumps({"diagnosis": "unclear"}))
    rec = adv.advise(FLAGGED)
    assert rec["recommended_action"] == "escalate"
    assert rec["requires_human"] is True


def test_no_flagged_is_a_noop_escalate_without_llm():
    adv, captured = _stub("SHOULD NOT BE CALLED")
    rec = adv.advise([])
    assert rec["recommended_action"] == "escalate"
    assert "prompt" not in captured                        # no LLM call
