"""What to measure next is a contract, not an algorithm.

BO is one recommender among several — a rule table, a surrogate optimizer, a
language model writing acquisition parameters as JSON or a revised protocol.
What the loop requires of all of them is pinned here: an instrument schema that
makes a recommendation mappable onto a controller, validation by the LOOP (an
unknown name or an out-of-bounds value is refused, never clamped), and a clock
(a slow recommender never makes a frame wait).

No LLM calls anywhere — the LLM recommender is driven by a scripted model.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.live import (GPRecommender, InstrumentSchema, LLMRecommender,
                          MeasurementLoop, ParameterSpec, RuleTableRecommender)
from scilink.live.recommend import finalize

SCHEMA = InstrumentSchema.from_dict({
    "dwell_ms": {"low": 1, "high": 500, "units": "ms", "description": "pixel dwell time"},
    "averages": {"kind": "int", "low": 1, "high": 64},
    "grating": {"kind": "choice", "choices": ["600", "1200", "1800"]},
    "shutter_open": {"kind": "bool"},
})


# ──────────────────────────────────────────────────────────────
# the schema
# ──────────────────────────────────────────────────────────────

class TestSchema:
    def test_a_partial_valid_set_is_accepted(self):
        assert SCHEMA.validate({"dwell_ms": 120.5}) == []
        assert SCHEMA.validate({"averages": 8, "grating": "1200", "shutter_open": True}) == []

    def test_an_unknown_name_cannot_be_mapped_onto_the_controller(self):
        [p] = SCHEMA.validate({"exposure": 3})
        assert "not a parameter of this instrument" in p and "dwell_ms" in p

    @pytest.mark.parametrize("params, fragment", [
        ({"dwell_ms": 900}, "outside [1, 500] ms"),
        ({"dwell_ms": 0}, "outside"),
        ({"averages": 2.5}, "expected an integer"),
        ({"averages": "8"}, "expected a number"),
        ({"grating": "2400"}, "not one of"),
        ({"shutter_open": 1}, "expected true/false"),
        ({"dwell_ms": float("nan")}, "not a finite number"),
        ({}, "no parameters were proposed"),
    ])
    def test_refusals(self, params, fragment):
        assert any(fragment in p for p in SCHEMA.validate(params))

    def test_the_caller_must_state_the_limits(self):
        with pytest.raises(ValueError, match="needs low < high"):
            ParameterSpec("dwell_ms")
        with pytest.raises(ValueError, match="needs `choices`"):
            ParameterSpec("grating", kind="choice")
        with pytest.raises(ValueError, match="duplicate"):
            InstrumentSchema([ParameterSpec("a", low=0, high=1), ParameterSpec("a", low=0, high=1)])

    def test_round_trip_and_description(self):
        assert InstrumentSchema.from_dict(SCHEMA.to_dict()).to_dict() == SCHEMA.to_dict()
        text = SCHEMA.describe()
        assert "dwell_ms: float in [1, 500] ms — pixel dwell time" in text
        assert "grating: one of ['600', '1200', '1800']" in text


# ──────────────────────────────────────────────────────────────
# the loop's side of the contract
# ──────────────────────────────────────────────────────────────

class TestFinalize:
    def test_a_valid_recommendation(self):
        r = finalize({"params": {"dwell_ms": 200}, "rationale": "SNR low"}, SCHEMA,
                     source="llm", based_on_step=7)
        assert r["valid"] and r["params"] == {"dwell_ms": 200} and r["based_on_step"] == 7
        assert r["requires_approval"] is True            # advisory by default

    def test_closed_loop_lifts_approval_for_valid_params_only(self):
        ok = finalize({"params": {"dwell_ms": 200}}, SCHEMA, source="gp",
                      based_on_step=1, closed_loop=True)
        bad = finalize({"params": {"dwell_ms": 9000}}, SCHEMA, source="gp",
                       based_on_step=1, closed_loop=True)
        proto = finalize({"protocol": "raster at 2 nm"}, SCHEMA, source="llm",
                         based_on_step=1, closed_loop=True)
        assert ok["requires_approval"] is False
        assert bad["requires_approval"] is True and proto["requires_approval"] is True

    def test_an_invalid_recommendation_withholds_its_params(self):
        r = finalize({"params": {"dwell_ms": 9000, "exposure": 1}}, SCHEMA,
                     source="llm", based_on_step=3)
        assert r["valid"] is False and r["params"] is None      # nothing to act on by accident
        assert r["rejected_params"] == {"dwell_ms": 9000, "exposure": 1}
        assert len(r["problems"]) == 2

    def test_out_of_bounds_is_refused_never_clamped(self):
        r = finalize({"params": {"dwell_ms": 501}}, SCHEMA, source="x", based_on_step=1)
        assert r["params"] is None and "outside" in r["problems"][0]

    def test_a_protocol_is_a_different_kind_of_thing(self):
        r = finalize({"protocol": "1. lower dose\\n2. re-acquire", "rationale": "damage"},
                     SCHEMA, source="llm", based_on_step=9)
        assert r["kind"] == "protocol" and r["valid"] and r["params"] is None
        assert r["requires_approval"] is True

    def test_garbage_from_a_recommender(self):
        for raw in (None, "set dwell to 200", 42, {}):
            assert finalize(raw, SCHEMA, source="x", based_on_step=1)["valid"] is False


# ──────────────────────────────────────────────────────────────
# fast-clock recommenders
# ──────────────────────────────────────────────────────────────

class TestRuleTable:
    RULES = [
        {"when": {"feature": "snr", "op": "<", "value": 10},
         "then": {"param": "dwell_ms", "scale": 2.0}, "why": "SNR below target"},
        {"when": {"feature": "saturated_fraction", "op": ">", "value": 0.01},
         "then": {"param": "dwell_ms", "scale": 0.5}, "why": "detector clipping"},
    ]

    def _rec(self):
        return RuleTableRecommender(SCHEMA, self.RULES)

    def test_a_symptom_maps_to_a_correction(self):
        r = self._rec()
        r.observe({"dwell_ms": 50}, {"snr": 4.0, "saturated_fraction": 0.0})
        out = r.suggest()
        assert out["params"] == {"dwell_ms": 100.0} and out["rationale"] == "SNR below target"

    def test_first_matching_rule_wins_and_no_rule_means_hold(self):
        r = self._rec()
        r.observe({"dwell_ms": 50}, {"snr": 4.0, "saturated_fraction": 0.5})
        assert r.suggest()["params"] == {"dwell_ms": 100.0}
        r.observe({"dwell_ms": 100}, {"snr": 40.0, "saturated_fraction": 0.0})
        out = r.suggest()
        assert out["params"] == {"dwell_ms": 100} and "no rule fired" in out["rationale"]

    def test_a_bound_is_a_stop_not_an_error(self):
        r = self._rec()
        r.observe({"dwell_ms": 400}, {"snr": 2.0})
        assert r.suggest()["params"] == {"dwell_ms": 500.0}          # clipped at the limit
        r.observe({"dwell_ms": 500.0}, {"snr": 2.0})
        out = r.suggest()
        assert out["params"] is None and "already at its limit" in out["problems"][0]

    def test_rules_are_checked_against_the_schema_up_front(self):
        with pytest.raises(ValueError, match="not in the instrument schema"):
            RuleTableRecommender(SCHEMA, [{"when": {"feature": "snr", "op": "<", "value": 1},
                                           "then": {"param": "gain", "set": 2}}])
        with pytest.raises(ValueError, match="op must be"):
            RuleTableRecommender(SCHEMA, [{"when": {"feature": "snr", "op": "~", "value": 1},
                                           "then": {"param": "dwell_ms", "set": 2}}])


class TestGPRecommender:
    NUM = InstrumentSchema.from_dict({"x": {"low": 0.0, "high": 10.0},
                                      "n": {"kind": "int", "low": 1, "high": 20}})

    def test_starts_with_a_space_filling_design_inside_the_bounds(self):
        r = GPRecommender(self.NUM, "signal", n_init=4)
        seen = []
        for _ in range(4):
            out = r.suggest()
            assert self.NUM.validate(out["params"]) == [] and "design point" in out["rationale"]
            seen.append(out["params"]["x"])
            r.observe(out["params"], {"signal": 1.0})
        assert len({round(v) for v in seen}) >= 3                     # actually spread out
        assert isinstance(out["params"]["n"], int)

    def test_then_the_locked_strategy_climbs_a_simple_objective(self):
        one = InstrumentSchema.from_dict({"x": {"low": 0.0, "high": 10.0}})
        r = GPRecommender(one, "signal", n_init=4, seed=1)
        best = -1e9
        for _ in range(12):
            p = r.suggest()["params"]
            assert one.validate(p) == []
            y = -(p["x"] - 7.0) ** 2                                  # maximum at x = 7
            best = max(best, y)
            r.observe(p, {"signal": y})
        assert best > -0.5, f"best {best}"
        assert "log_ei" in r.suggest()["rationale"]

    def test_minimize_and_numeric_only(self):
        one = InstrumentSchema.from_dict({"x": {"low": 0.0, "high": 10.0}})
        r = GPRecommender(one, "width", direction="minimize", n_init=3, seed=2)
        for _ in range(10):
            p = r.suggest()["params"]
            r.observe(p, {"width": (p["x"] - 2.0) ** 2 + 1.0})
        assert min(h["features"]["width"] for h in r.history) < 1.6
        with pytest.raises(ValueError, match="numeric parameters only"):
            GPRecommender(SCHEMA, "signal")


def test_change_nothing_is_an_answer_not_a_failure():
    # Observed live: an AFM recommender that judged precision already met
    # returned no parameters and was logged as refused.
    for raw in ({"params": None, "rationale": "precision already meets the goal"},
                {"params": {}, "rationale": "hold"}):
        rec = finalize(raw, SCHEMA, source="llm", based_on_step=4)
        assert rec["kind"] == "hold" and rec["valid"] and rec["params"] is None
        assert rec["requires_approval"] is False
    # ...but silence is not: no parameters AND no reason is still a failure,
    assert finalize({"params": None}, SCHEMA, source="llm", based_on_step=4)["valid"] is False
    # and so is a recommender that reported its own problem.
    broken = finalize({"params": None, "problems": ["the model did not return JSON"],
                       "rationale": "garbled"}, SCHEMA, source="llm", based_on_step=4)
    assert broken["valid"] is False and broken["kind"] == "params"


# ──────────────────────────────────────────────────────────────
# the slow clock: an LLM recommender
# ──────────────────────────────────────────────────────────────

class ScriptedModel:
    def __init__(self, replies, delay=0.0):
        self.replies, self.delay, self.prompts = list(replies), delay, []

    def generate_content(self, prompt, **kw):
        self.prompts.append(prompt)
        time.sleep(self.delay)
        return SimpleNamespace(text=self.replies.pop(0))


class TestLLMRecommender:
    def test_the_prompt_carries_the_goal_the_interface_and_the_run(self):
        m = ScriptedModel(['{"params": {"dwell_ms": 200}, "rationale": "SNR is 4; double dwell"}'])
        r = LLMRecommender(m, SCHEMA, "reach SNR 10 with the least dose", feature_keys=["snr"])
        r.observe({"dwell_ms": 100}, {"snr": 4.0, "snr_err": 0.1}, step=3)
        out = r.suggest()
        assert out["params"] == {"dwell_ms": 200}
        [prompt] = m.prompts
        assert "reach SNR 10 with the least dose" in prompt
        assert "dwell_ms: float in [1, 500] ms" in prompt
        assert 'step 3: {"dwell_ms": 100} -> {"snr": 4.0}' in prompt
        r.observe({"dwell_ms": 100}, {"snr": 3.0}, step=4, flags=["gate_poor"])
        assert '{"snr": 3.0}  [flags: gate_poor]' in r.build_prompt()

    def test_it_is_shown_its_own_earlier_decisions(self):
        # Observed live (in-situ Raman): shorten, restore, shorten again.
        m = ScriptedModel(['{"params": {"dwell_ms": 50}, "rationale": "precision met; go faster"}',
                           '{"params": {}, "rationale": "still fine"}', '{"params": {"dwell_ms": 60}}'])
        r = LLMRecommender(m, SCHEMA, "goal", skill=None)
        r.observe({"dwell_ms": 100}, {"snr": 9.0}, step=8)
        r.suggest()
        assert "first recommendation" in m.prompts[0]
        r.observe({"dwell_ms": 50}, {"snr": 6.0}, step=16)
        r.suggest(); r.suggest()
        assert 'after step 8: {"dwell_ms": 50} — precision met; go faster' in m.prompts[1]
        assert "after step 16: no change — still fine" in m.prompts[2]

    def test_json_in_a_fence_and_non_json(self):
        m = ScriptedModel(['```json\\n{"params": {"averages": 16}}\\n```'.replace("\\\\n", "\\n"),
                           "I would increase the dwell time."])
        r = LLMRecommender(m, SCHEMA, "goal")
        assert r.suggest()["params"] == {"averages": 16}
        assert "did not return JSON" in r.suggest()["problems"][0]

    def test_protocol_output(self):
        m = ScriptedModel(['{"protocol": "1. halve the dose\\n2. re-acquire", "rationale": "damage"}'])
        r = LLMRecommender(m, SCHEMA, "goal", output="protocol")
        assert "handed to a person for review" in r.build_prompt()
        assert r.suggest()["protocol"].startswith("1. halve")


def _loop(tmp_path, recommender, **kw):
    from tests.test_measurement_loop import FakeAgent, make_anchor
    FakeAgent.calls, FakeAgent.replies = [], {}
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=FakeAgent,
                           recommender=recommender, schema=SCHEMA, **kw)
    loop.setup(anchor=str(make_anchor(tmp_path)))
    return loop


class TestLoopWithRecommenders:
    def test_a_slow_recommender_never_makes_a_frame_wait(self, tmp_path):
        m = ScriptedModel(['{"params": {"dwell_ms": 200}, "rationale": "more signal"}'], delay=0.6)
        loop = _loop(tmp_path, LLMRecommender(m, SCHEMA, "goal", every=1))
        t0 = time.perf_counter()
        first = loop.step("f1.csv", {"dwell_ms": 100})
        assert time.perf_counter() - t0 < 0.4                    # the call is 0.6 s long
        assert first["recommendation"] == {"pending": True}
        loop._slot.wait(5)
        second = loop.step("f2.csv", {"dwell_ms": 100})
        rec = second["recommendation"]
        assert rec["params"] == {"dwell_ms": 200} and rec["based_on_step"] == 1
        assert rec["source"] == "llm" and rec["requires_approval"] is True
        assert [e["event"] for e in loop.read_log()].count("recommendation") == 1

    def test_the_model_is_told_what_the_experiment_is(self, tmp_path):
        # Observed live (in-situ Raman): with no context the model read an
        # anneal's G-band shift as laser heating and cut the power.
        m = ScriptedModel(['{"params": {"dwell_ms": 200}}'] * 2)
        rec = LLMRecommender(m, SCHEMA, "goal", every=1)
        loop = _loop(tmp_path, rec, system_info={"sample": "carbon film, annealed in situ"})
        loop.step("f1.csv", {"dwell_ms": 100})
        loop._slot.wait(5)
        assert "- sample: carbon film, annealed in situ" in m.prompts[0]
        own = LLMRecommender(m, SCHEMA, "goal", context="a beam-sensitive polymer")
        _loop(tmp_path / "b", own, system_info={"sample": "x"}).step("f1.csv", {"dwell_ms": 100})
        assert "a beam-sensitive polymer" in own.build_prompt()      # the caller's own wins

    def test_the_loop_refuses_what_the_model_got_wrong(self, tmp_path):
        m = ScriptedModel(['{"params": {"dwell_ms": 5000, "laser_power": 3}}'])
        loop = _loop(tmp_path, LLMRecommender(m, SCHEMA, "goal", every=1))
        loop.step("f1.csv", {"dwell_ms": 100})
        loop._slot.wait(5)
        rec = loop.step("f2.csv", {"dwell_ms": 100})["recommendation"]
        assert rec["valid"] is False and rec["params"] is None
        assert any("outside [1, 500]" in p for p in rec["problems"])
        assert any("laser_power" in p for p in rec["problems"])

    def test_a_protocol_is_written_out_never_run(self, tmp_path):
        m = ScriptedModel(['{"protocol": "lower the dose, then re-acquire", "rationale": "damage"}'])
        loop = _loop(tmp_path, LLMRecommender(m, SCHEMA, "goal", output="protocol", every=1),
                     closed_loop=True)
        loop.step("f1.csv", {"dwell_ms": 100})
        loop._slot.wait(5)
        rec = loop.step("f2.csv", {"dwell_ms": 100})["recommendation"]
        assert rec["kind"] == "protocol" and rec["requires_approval"] is True
        assert Path(rec["protocol_path"]).read_text() == "lower the dose, then re-acquire"

    def test_every_n_clean_frames(self, tmp_path):
        m = ScriptedModel(['{"params": {"averages": 4}}'] * 5)
        loop = _loop(tmp_path, LLMRecommender(m, SCHEMA, "goal", every=3))
        for i in range(7):
            loop.step(f"f{i}.csv", {"averages": 2})
            loop._slot.wait(5)
        assert len(m.prompts) == 2                               # after frames 3 and 6

    def test_a_fast_recommender_answers_in_the_same_frame(self, tmp_path):
        from tests.test_measurement_loop import FakeAgent
        rules = [{"when": {"feature": "fit_r_squared", "op": "<", "value": 0.995},
                  "then": {"param": "averages", "scale": 2}, "why": "noisy fit"}]
        loop = _loop(tmp_path, RuleTableRecommender(SCHEMA, rules), closed_loop=True)
        rec = loop.step("f1.csv", {"averages": 4})["recommendation"]
        assert rec["params"] == {"averages": 8} and rec["based_on_step"] == 1
        assert rec["requires_approval"] is False and rec["source"] == "rule_table"


def test_a_recommenders_llm_call_is_not_charged_to_a_frame():
    """The LLM counters are process-wide so that a stage's worker threads are
    included; a recommender thinking beside the pipeline must not be."""
    import threading
    from scilink import tracing
    before = tracing.llm_counters()

    def think():
        with tracing.off_path():
            tracing.note_llm_call(3.0, 100, 20)
    th = threading.Thread(target=think)
    th.start(); th.join()
    tracing.note_llm_call(1.0)                       # an ordinary in-pipeline call
    after = tracing.llm_counters()
    assert after["calls"] - before["calls"] == 1
    assert after["off_path_calls"] - before["off_path_calls"] == 1
