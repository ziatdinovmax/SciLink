"""ReferencePropertySelector chooses which reference property to check.

Keeps the validation general — density isn't hardwired; this step decides per
component (density for a liquid, a lattice constant for a crystal, ...). The
LLM is stubbed; no API key.
"""

import json
import logging

from scilink.agents.sim_agents.critics import ReferencePropertySelector


def _stub(fake_text: str):
    obj = ReferencePropertySelector.__new__(ReferencePropertySelector)
    obj.logger = logging.getLogger("test_reference_selector")
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


COMPONENTS = [
    {"name": "water", "smiles": "O", "role": "solvent"},
    {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C", "role": "cosolvent"},
]


def test_components_reach_the_prompt():
    sel, captured = _stub(json.dumps({"selections": []}))
    sel.select(COMPONENTS, system_description="aqueous sulfone electrolyte")
    prompt = captured["prompt"]
    assert "water" in prompt and "EIS" in prompt
    assert "CCS(=O)(=O)C(C)C" in prompt          # SMILES carries the chemistry
    assert "cosolvent" in prompt                 # role passed through
    assert "aqueous sulfone electrolyte" in prompt


def test_selections_pass_through():
    sel, _ = _stub(json.dumps({
        "status": "success",
        "selections": [
            {"component": "water", "property": "density",
             "measurable": True, "rationale": "Liquid; density is well known."},
            {"component": "EIS", "property": "density",
             "measurable": True, "rationale": "Molecular liquid cosolvent."},
        ],
    }))
    report = sel.select(COMPONENTS)
    props = {s["component"]: s["property"] for s in report["selections"]}
    assert props == {"water": "density", "EIS": "density"}
    assert all(s["measurable"] for s in report["selections"])


def test_non_measurable_component_is_allowed():
    sel, _ = _stub(json.dumps({
        "selections": [
            {"component": "novelX", "property": None, "measurable": False,
             "rationale": "No independently-known reference for this compound."},
        ],
    }))
    report = sel.select([{"name": "novelX", "smiles": "C#N"}])
    s = report["selections"][0]
    assert s["measurable"] is False and s["property"] is None


def test_empty_components_makes_no_llm_call():
    sel, captured = _stub("SHOULD NOT BE CALLED")
    report = sel.select([])
    assert report["selections"] == []
    assert "prompt" not in captured
