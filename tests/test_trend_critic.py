"""TrendCritic judges whether a swept trend moves the physically-expected way.

The reliable catch for a composition series — a single density can look fine
while the trend across the sweep is backwards. Grounded in the sourced
constituent properties. LLM stubbed; no API key.
"""

import json
import logging

from scilink.agents.sim_agents.critics import TrendCritic

# The real UC2 wrong-trend data: density FALLS as EIS is added.
UC2_SERIES = [
    {"point": "EIS x=0.03", "value": 1.148},
    {"point": "EIS x=0.06", "value": 1.138},
    {"point": "EIS x=0.10", "value": 1.128},
    {"point": "EIS x=0.15", "value": 1.119},
]
REF = ("Literature: pure ethyl-isopropyl-sulfone density is 1.13 g/cm^3, denser "
       "than water (1.00 g/cm^3).")


def _stub(fake_text: str):
    obj = TrendCritic.__new__(TrendCritic)
    obj.logger = logging.getLogger("test_trend_critic")
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


def test_trend_and_reference_reach_the_prompt():
    tc, captured = _stub(json.dumps({"verdict": "good"}))
    tc.assess(UC2_SERIES, quantity="mass density",
              parameter="EIS mole fraction",
              system_description="aqueous Zn electrolyte + EIS cosolvent",
              reference_context=REF, units="g/cm^3")
    p = captured["prompt"]
    assert "1.148" in p and "1.119" in p            # the trend points
    assert "EIS mole fraction" in p                 # the swept parameter
    assert "denser than water" in p                 # the sourced anchor
    assert "mass density" in p


def test_inverted_trend_flags_the_model():
    tc, _ = _stub(json.dumps({
        "status": "success",
        "expected_direction": "increasing",
        "observed_direction": "decreasing",
        "consistent": False,
        "verdict": "poor",
        "failure_class": "force_field",
        "reasoning": "EIS is denser than water, so density should rise as it is "
                     "added; the simulation shows it falling.",
    }))
    r = tc.assess(UC2_SERIES, reference_context=REF)
    assert r["verdict"] == "poor"
    assert r["failure_class"] == "force_field"
    assert r["consistent"] is False


def test_consistent_trend_is_good():
    tc, _ = _stub(json.dumps({
        "expected_direction": "increasing", "observed_direction": "increasing",
        "consistent": True, "verdict": "good"}))
    r = tc.assess([{"point": "x=0.03", "value": 1.22},
                   {"point": "x=0.10", "value": 1.24}], reference_context=REF)
    assert r["verdict"] == "good"
    assert r["failure_class"] is None
    assert r["consistent"] is True


def test_single_point_is_a_noop_without_llm():
    tc, captured = _stub("SHOULD NOT BE CALLED")
    r = tc.assess([{"point": "x=0.03", "value": 1.148}])
    assert r["verdict"] == "good"
    assert "prompt" not in captured                 # no trend → no LLM call


def test_consistent_defaults_from_verdict_when_model_omits_it():
    tc, _ = _stub(json.dumps({"verdict": "poor", "failure_class": "force_field"}))
    r = tc.assess(UC2_SERIES, reference_context=REF)
    assert r["consistent"] is False                 # inferred from poor verdict
