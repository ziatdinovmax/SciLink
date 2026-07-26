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
     "property": "density", "value": 1.03, "units": "g/cm^3"},
]


def test_measurements_reach_the_prompt():
    rc, captured = _stub(json.dumps({"verdict": "good", "per_measurement": []}))
    rc.assess(MEASURED, system_description="aqueous sulfone electrolyte")
    prompt = captured["prompt"]
    assert "EIS" in prompt and "1.03" in prompt
    assert "density" in prompt                   # the property is shown per value
    assert "aqueous sulfone electrolyte" in prompt
    # The conservative guard against vetoing surprising-but-sound results is present.
    assert "must not veto" in prompt


def test_inconsistent_value_flags_the_model_cause():
    rc, _ = _stub(json.dumps({
        "status": "success",
        "verdict": "poor",
        "failure_class": "force_field",
        "per_measurement": [
            {"component": "water", "property": "density",
             "consistent": True, "reasoning": "≈1.0, fine."},
            {"component": "EIS", "property": "density", "consistent": False,
             "reasoning": "Far below the known density of a sulfone."},
        ],
        "reasoning": "The sulfone is under-dense; the force field is at fault.",
    }))
    report = rc.assess(MEASURED)
    assert report["verdict"] == "poor"
    assert report["failure_class"] == "force_field"
    assert any(not m["consistent"] for m in report["per_measurement"])


def test_cause_is_named_in_the_systems_terms():
    # A DFT context: the miscalibrated model is the functional, not a force field.
    rc, _ = _stub(json.dumps({
        "verdict": "poor", "failure_class": "functional",
        "per_measurement": [
            {"component": "germanium", "property": "band gap",
             "consistent": False, "reasoning": "Metallic, but Ge is a semiconductor."},
        ],
    }))
    report = rc.assess([{"component": "germanium", "status": "measured",
                         "property": "band gap", "value": 0.0, "units": "eV"}])
    assert report["failure_class"] == "functional"


def test_two_properties_of_one_component_are_judged_separately():
    # Right density, wrong dielectric → one entry per measured value.
    rc, _ = _stub(json.dumps({
        "verdict": "poor", "failure_class": "force_field",
        "per_measurement": [
            {"component": "water", "property": "density",
             "consistent": True, "reasoning": "Accurate."},
            {"component": "water", "property": "dielectric constant",
             "consistent": False, "reasoning": "Far too low."},
        ],
    }))
    report = rc.assess([
        {"component": "water", "status": "measured", "property": "density",
         "value": 0.997, "units": "g/cm^3"},
        {"component": "water", "status": "measured",
         "property": "dielectric constant", "value": 30, "units": ""},
    ])
    by_prop = {m["property"]: m["consistent"] for m in report["per_measurement"]}
    assert by_prop == {"density": True, "dielectric constant": False}


def test_all_consistent_is_good_with_null_cause():
    rc, _ = _stub(json.dumps({
        "verdict": "good",
        "per_measurement": [
            {"component": "water", "property": "density",
             "consistent": True, "reasoning": "ok"},
            {"component": "EIS", "property": "density",
             "consistent": True, "reasoning": "plausible"},
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
