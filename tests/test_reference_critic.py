"""ReferencePropertyCritic reasons over pre-run reference measurements.

Given each pure component's measured property, it decides whether the value is
consistent with known behaviour and, when one clearly isn't, blames the force
field — the same ``poor`` / ``force_field`` vocabulary the post-run RunCritic
uses. The LLM is stubbed; no API key.
"""

import json
import logging

from scilink.agents.sim_agents.critics import ReferencePropertyCritic


def _stub(fake_text: str):
    obj = ReferencePropertyCritic.__new__(ReferencePropertyCritic)
    obj.logger = logging.getLogger("test_reference_critic")
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


MEASURED = [
    {"component": "water", "smiles": "O", "status": "measured",
     "value": 0.99, "units": "g/cm^3"},
    {"component": "EIS", "smiles": "CCS(=O)(=O)C(C)C", "status": "measured",
     "value": 1.03, "units": "g/cm^3"},
]


def test_measurements_reach_the_prompt():
    rc, captured = _stub(json.dumps({"verdict": "good", "per_component": []}))
    rc.assess(MEASURED, system_description="aqueous sulfone electrolyte")
    prompt = captured["prompt"]
    assert "EIS" in prompt and "1.03" in prompt
    assert "aqueous sulfone electrolyte" in prompt
    # The conservative guard against vetoing surprising-but-sound results is present.
    assert "must not veto" in prompt


def test_inconsistent_component_flags_force_field():
    rc, _ = _stub(json.dumps({
        "status": "success",
        "verdict": "poor",
        "failure_class": "force_field",
        "per_component": [
            {"component": "water", "consistent": True, "reasoning": "≈1.0, fine."},
            {"component": "EIS", "consistent": False,
             "reasoning": "Far below the known density of a sulfone."},
        ],
        "reasoning": "The sulfone is under-dense; the force field is at fault.",
    }))
    report = rc.assess(MEASURED)
    assert report["verdict"] == "poor"
    assert report["failure_class"] == "force_field"
    assert any(not c["consistent"] for c in report["per_component"])


def test_all_consistent_is_good_with_null_cause():
    rc, _ = _stub(json.dumps({
        "verdict": "good",
        "per_component": [
            {"component": "water", "consistent": True, "reasoning": "ok"},
            {"component": "EIS", "consistent": True, "reasoning": "plausible"},
        ],
    }))
    report = rc.assess(MEASURED)
    assert report["verdict"] == "good"
    assert report["failure_class"] is None


def test_no_measurements_fails_open_without_llm_call():
    # All unmeasured → returns good immediately; the stubbed model must not fire.
    rc, captured = _stub("SHOULD NOT BE CALLED")
    report = rc.assess([
        {"component": "x", "status": "unmeasured", "error": "packmol failed"},
    ])
    assert report["verdict"] == "good"
    assert report["failure_class"] is None
    assert "prompt" not in captured        # no LLM call was made
