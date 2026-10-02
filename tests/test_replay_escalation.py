"""#712 follow-up: a replay that fails its checks, or whose regime the data
cannot tell, is explained by a JUDGE — one model call, an opinion on the
record, never a verdict.

The deterministic checks (the gate, the drift monitor's state distance, the
identity check) say THAT something differs and decide verified / not
verified; they cannot say what a difference means. The judge is shown the
findings as quoted data (between markers), the replayed fit and the new
curve over the regime's anchor, and the skill's interpretation guidance, and
asked which regime the measurement belongs to and what changed. Its answer
goes on ``reuse_validity.escalation``, into the message, onto the board as a
PROVISIONAL claim; the verdict and ``interpretation_checked`` do not move,
nothing is re-run. It never fires on a clean pass, on the fast clock, when
the caller asked for no review, on a replay that did not execute, and at
most once per item.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_series_verdict_path as curve  # noqa: E402
import test_regime_choice as rc  # noqa: E402

from scilink.agents.exp_agents import _replay  # noqa: E402
from scilink.agents.exp_agents._verification_record import (  # noqa: E402
    analysis_verdict, final_verdict_record, replay_escalation, unit_verdict_for)
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (  # noqa: E402
    UnifiedSeriesProcessingController, _overlay_png)
from scilink.agents.meta_agent import board as board_mod  # noqa: E402


class ScriptedJudge:
    """The controller's model: answers the escalation with a canned JSON and
    records every prompt it was shown (text parts and image parts)."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def generate_content(self, contents=None, **kw):
        self.prompts.append(contents)
        if isinstance(self.answer, Exception):
            raise self.answer
        return SimpleNamespace(text=json.dumps(self.answer) if isinstance(self.answer, dict) else str(self.answer))


def _judged(tmp_path, judge, *, max_verification_iterations=None, feedback=False):
    def controller(out, ex):
        return UnifiedSeriesProcessingController(
            model=judge, logger=curve.logging.getLogger("escalation"), generation_config=None, safety_settings=None,
            parse_fn=lambda r: (json.loads(r.text), None), executor=ex, script_instructions="",
            correction_instructions="", quality_instructions="", output_dir=str(out), plot_fn=lambda d, i: b"plot",
            r2_threshold=curve.THRESHOLD, parallel_workers=1, enable_human_feedback=feedback,
            **({"max_verification_iterations": max_verification_iterations} if max_verification_iterations is not None else {}))
    return controller


def _text_of(prompt):
    return "\n".join(p for p in prompt if isinstance(p, str))


def _images_of(prompt):
    return [p for p in prompt if isinstance(p, dict) and p.get("mime_type") == "image/png"]


def test_the_trigger_fires_on_the_three_findings_and_on_nothing_else():
    base = {"reused": True, "verdict": "good", "state_distance": 0.01, "state_flag": False,
            "identity": {"checked": True, "within": True, "spread_known": True, "drifted": []},
            "regime_choice": {"ambiguous": False}}
    assert _replay.escalation_trigger(base) is None                                             # a clean pass
    assert _replay.escalation_trigger({**base, "state_distance": 0.6, "state_flag": True}) == "state"
    assert _replay.escalation_trigger({**base, "state_distance": 0.6}) == "state"                 # the distance alone says it too
    drift = {"checked": True, "within": False, "spread_known": True, "drifted": [{"name": "position", "value": 27.3}]}
    assert _replay.escalation_trigger({**base, "identity": drift}) == "identity"
    flag = {**drift, "spread_known": False}
    assert _replay.escalation_trigger({**base, "identity": flag}) is None                       # attended: a flag is read
    assert _replay.escalation_trigger({**base, "identity": flag}, attended=False) == "flag"      # nobody is looking
    assert _replay.escalation_trigger({**base, "regime_choice": {"ambiguous": True}}) == "ambiguous"
    assert _replay.escalation_trigger({**base, "state_distance": 0.6, "regime_choice": {"ambiguous": True}}) == "state"
    assert _replay.escalation_trigger({**base, "verdict": "failed"}) is None                    # did not execute
    assert _replay.escalation_trigger({"reused": False, "verdict": "poor"}) is None             # not a replay
    assert _replay.escalation_trigger({**base, "verdict": "poor"}) is None                      # the gate alone is no question for a judge
    assert _replay.escalation_trigger(None) is None


def test_the_question_quotes_the_evidence_as_data_and_the_answer_is_normalised():
    rv = {"reused": True, "verdict": "poor", "metric": "R²", "score": 0.99, "threshold": 0.95, "state_distance": 0.9987,
          "identity": {"checked": True, "within": False, "spread_known": True, "n_units": 3,
                       "drifted": [{"name": "position", "value": 27.3, "reference": "no strong feature of the regime near it"},
                                   {"name": "space_group", "value": "p42mnm", "reference": ["i41amd"]}]},
          "regime_choice": {"chosen_regime": "anatase", "ambiguous": False,
                            "ranking": [{"regime": "anatase", "distance": 0.9987}, {"regime": "rutile", "distance": 0.0}]}}
    ev = _replay.escalation_evidence(rv)
    assert ev["state"] == {"distance": 0.9987, "bar": _replay.SAME_STATE_BAR,
                           "meaning": "the share of this measurement the regime's own curves cannot describe"}
    assert ev["identity"]["drifted"] == [{"kind": "strong_feature_new", "position": 27.3, "regime_has": "no strong feature of the regime near it"},
                                         {"kind": "name", "name": "space_group", "value": "p42mnm", "regime_has": ["i41amd"]}]
    assert ev["regimes"]["ranking"][1] == {"regime": "rutile", "distance": 0.0}
    regimes = [{"regime": "anatase", "model": "anatase TiO2, four Raman-active modes", "unit": "xrd_T300K", "n_units": 3},
               {"regime": "rutile", "model": None, "unit": "xrd_T800K", "n_units": 3}]
    q = _replay.escalation_question(ev, regimes, trigger="state")
    body = q[q.index(_replay.ESCALATION_MARK_OPEN) + len(_replay.ESCALATION_MARK_OPEN):q.index(_replay.ESCALATION_MARK_CLOSE)]
    assert json.loads(body) == ev                                            # the evidence is data between markers
    assert "'anatase' — model: anatase TiO2, four Raman-active modes — anchor unit: xrd_T300K — 3 units" in q
    assert "does not change the pipeline's verdict and triggers no re-run" in q
    assert "['anatase', 'rutile', 'none', 'cannot_tell']" in q and "cannot_tell" in q
    # answers: a known regime (case-insensitive), the two words, anything else is cannot_tell; strings coerced
    assert _replay.read_escalation_answer({"belongs_to": "Rutile", "same_interpretation": "false", "what_changed": "new 27.3° line",
                                          "confidence": "HIGH"}, regimes) == \
        {"belongs_to": "rutile", "same_interpretation": False, "what_changed": "new 27.3° line", "confidence": "high"}
    assert _replay.read_escalation_answer({"belongs_to": "a third phase"}, regimes)["belongs_to"] == "cannot_tell"
    assert _replay.read_escalation_answer({"belongs_to": "none", "confidence": "sure"}, regimes) == \
        {"belongs_to": "none", "same_interpretation": None, "what_changed": "", "confidence": "low"}
    assert _replay.read_escalation_answer("not a dict", regimes)["belongs_to"] == "cannot_tell"
    assert len(_replay.read_escalation_answer({"what_changed": "x" * 5000}, regimes)["what_changed"]) == 1200


def test_the_overlay_is_a_png_or_nothing():
    xy = (np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    png = _overlay_png(xy, (np.linspace(0, 10, 25), np.cos(np.linspace(0, 10, 25))), "a")
    assert isinstance(png, bytes) and png[:4] == b"\x89PNG"
    assert _overlay_png(xy, None, "a") is None and _overlay_png(None, xy, "a") is None


def test_a_replay_that_fails_its_checks_is_explained_and_the_verdict_does_not_move(tmp_path, monkeypatch):
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    judge = ScriptedJudge({"belongs_to": "high", "same_interpretation": False,
                           "what_changed": "New strong reflections near 27.3°, 36.0° and 54.3° (rutile 110/101/211); the anatase 25.3° line is gone.",
                           "confidence": "high"})
    # the anatase recipe forced on rutile data: poor on state and identity → escalated
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=2),
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=91), "space_group": "P 42/m n m"}},
                                  controller=_judged(tmp_path, judge))
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["state_flag"] is True and rv["identity"]["flagged"] is True   # the gate's verdict; the checks' flags
    esc = rv["escalation"]
    assert esc["trigger"] == "state" and esc["decided_by"] == "judge" and esc["belongs_to"] == "high"
    assert esc["same_interpretation"] is False and "27.3°" in esc["what_changed"] and esc["confidence"] == "high"
    assert "JUDGE (no gate): belongs to 'high', same interpretation: False — New strong reflections" in rv["message"]
    uv = unit_verdict_for({**res, "success": True})
    assert uv["verified"] and uv["interpretation_checked"] is False                            # the judge moved nothing
    assert len(judge.prompts) == 1
    text = _text_of(judge.prompts[0])
    assert _replay.ESCALATION_MARK_OPEN in text and "because the new measurement is NOT the chosen regime's state" in text
    assert "'low'" in text and "anchor unit: spectrum_0000" in text and "3 units" in text
    # the judge sees the distance to EVERY regime of the prior run, nearest first, although one recipe was replayed
    body = text[text.index(_replay.ESCALATION_MARK_OPEN) + len(_replay.ESCALATION_MARK_OPEN):text.index(_replay.ESCALATION_MARK_CLOSE)]
    shown = json.loads(body)["regimes"]
    assert [x["regime"] for x in shown["ranking"]] == ["high", "low"] and shown["ranking"][0]["distance"] < 0.05
    assert shown["ranking"][1]["distance"] > _replay.SAME_STATE_BAR and shown["chosen"] is None
    assert len(_images_of(judge.prompts[0])) >= 1                                              # the overlay at least
    # the row and the board: an opinion beside the verdict, a PROVISIONAL claim with the judge gate
    full = {"status": "success", "reuse_validity": rv}
    full["verdict"] = final_verdict_record(full)
    row = {"analysis_id": "r1", "status": "success", "agent_name": "CurveFittingAgent", "output_directory": str(tmp_path / "none"),
           **analysis_verdict(full), "escalation": replay_escalation(full)}
    assert row["verified"] is True and row["interpretation_checked"] is False
    assert row["escalation"]["belongs_to"] == "high" and row["escalation"]["decided_by"] == "judge"
    recs = board_mod.records_for({"index": 1, "label": "replay", "mode": "analysis", "status": "success"},
                                 {"key_findings": ["[r1] the pattern is anatase"], "analyses": [row]})
    kinds = [(r["kind"], r["status"]) for r in recs]
    assert kinds == [("claim", "provisional"), ("claim", "provisional")]     # the claim withheld, the judge's reading beside it
    judge_rec = recs[1]
    assert judge_rec["payload"]["text"].startswith("Replay of analysis r1 escalated (state): the judge reads it as belonging to 'high'")
    assert judge_rec["evidence"]["gate"].startswith("replay escalation: a judge's reading") and "no gate" in judge_rec["evidence"]["gate"]
    assert replay_escalation({"reuse_validity": {}}) is None and replay_escalation({"reuse_validity": {"escalation": {"error": "x"}}}) is None


def test_the_judge_is_not_asked_on_a_clean_pass_the_fast_clock_or_no_review(tmp_path, monkeypatch):
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    judge = ScriptedJudge({"belongs_to": "high", "what_changed": "nothing", "confidence": "high"})
    # a clean pass: no call, no record
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=1),
                                  extra_params={"HIGH": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=90, n_noise=6), "space_group": "P42/mnm"}},
                                  controller=_judged(tmp_path, judge))
    assert res["reuse_validity"]["verdict"] == "good" and "escalation" not in res["reuse_validity"] and judge.prompts == []
    # the fast clock: the same failing replay, strict → no call
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    data = rc.spectrum(rc.RUTILE, shift=0.5, seed=2)
    params = {"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=91), "space_group": "P 42/m n m"}}
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, strict=True, controller=_judged(tmp_path, judge))
    assert res["reuse_validity"]["state_flag"] is True and "escalation" not in res["reuse_validity"] and judge.prompts == []
    # the caller asked for no review (max_verification_iterations == 0) → no call
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, controller=_judged(tmp_path, judge, max_verification_iterations=0))
    assert res["reuse_validity"]["state_flag"] is True and "escalation" not in res["reuse_validity"] and judge.prompts == []
    # a replay that did not execute: failed, no call
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, strict=True, controller=_judged(tmp_path, judge))
    assert res["reuse_validity"]["verdict"] == "failed" and judge.prompts == []
    # the gate alone failing (R² low, the state and identity fine) is no question for a judge
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.999, "HIGH": 0.70}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=13),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=54),
                                                "LOW": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=55)},
                                  controller=_judged(tmp_path, judge))
    # the gate failing on HIGH falls through to LOW, whose gate passes: kept (good), its state flagged → escalated on "state"
    # (the flag is the reason a judge is asked; the verdict is the gate's)
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["recipes_tried"] == 2
    assert res["reuse_validity"]["state_flag"] is True and res["reuse_validity"]["escalation"]["trigger"] == "state"
    judge.prompts.clear()
    # a state distance merely above the certification bar (nothing flagged): nothing to explain, no call
    rv_band = {"reused": True, "verdict": "good", "state_distance": 0.15, "identity": {"checked": True, "within": True, "drifted": []},
               "regime_choice": {"ambiguous": False}}
    assert _replay.escalation_trigger(rv_band) is None


def test_an_ambiguous_choice_is_escalated_and_the_judges_regime_is_listed_not_taken(tmp_path, monkeypatch):
    run = tmp_path / "prior"
    (run / "scripts").mkdir(parents=True)
    rows = []
    for i in range(6):
        (run / f"spectrum_{i:04d}").mkdir()
        np.save(run / f"spectrum_{i:04d}" / "data.npy", rc.spectrum(rc.ANATASE, seed=i))
        rows.append({"index": i, "name": f"spectrum_{i:04d}", "success": True, "regime": "a" if i < 3 else "b",
                     "parameters": rc.auto_detect_parameters(rc.ANATASE, seed=i, n_noise=4)})
    (run / "series_fit_results.json").write_text(json.dumps({"results": rows}))
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "a": {"unit": "spectrum_0000", "index": 0, "regime": "a", "script": "LOW", "verdict": {"verified": True}, "model": "the cold phase"},
        "b": {"unit": "spectrum_0003", "index": 3, "regime": "b", "script": "HIGH", "verdict": {"verified": True}, "model": "the warm phase"}}}))
    judge = ScriptedJudge({"belongs_to": "b", "same_interpretation": True, "what_changed": "A 0.4° shift toward the warm phase; same peaks.",
                           "confidence": "medium"})
    res, ex, _, _ = curve._replay(tmp_path / "t", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=run,
                                  data=rc.spectrum(rc.ANATASE, seed=22),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.ANATASE, seed=60), "LOW": rc.auto_detect_parameters(rc.ANATASE, seed=61)},
                                  controller=_judged(tmp_path, judge))
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["regime_choice"]["ambiguous"] is True
    assert rv["escalation"]["trigger"] == "ambiguous" and rv["escalation"]["belongs_to"] == "b"
    assert rv["regime_choice"]["chosen_regime"] == "a" and rv["regime_choice"]["suggested"] == "b"   # listed, not taken
    assert res["script"] == "LOW"                                                                     # nothing re-run
    assert unit_verdict_for({**res, "success": True})["verified"]                                     # the gate's verdict stands
    text = _text_of(judge.prompts[0])
    assert "because the data does not tell the two nearest regimes apart" in text and "model: the cold phase" in text


def test_a_judge_that_fails_or_cannot_tell_is_recorded_as_such_and_asked_once(tmp_path, monkeypatch):
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    data = rc.spectrum(rc.RUTILE, shift=0.5, seed=2)
    params = {"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=91), "space_group": "P 42/m n m"}}
    # the model raises: the record says so, the verdict is the checks'
    judge = ScriptedJudge(RuntimeError("provider down"))
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, controller=_judged(tmp_path, judge))
    esc = res["reuse_validity"]["escalation"]
    assert esc["trigger"] == "state" and "provider down" in esc["error"] and "belongs_to" not in esc
    assert res["reuse_validity"]["verdict"] == "good" and "JUDGE" not in res["reuse_validity"]["message"]
    assert replay_escalation({"reuse_validity": res["reuse_validity"]}) is None
    # cannot_tell is a fine answer
    judge = ScriptedJudge({"belongs_to": "cannot_tell", "what_changed": "The pattern is too noisy to place.", "confidence": "low"})
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, controller=_judged(tmp_path, judge))
    assert res["reuse_validity"]["escalation"]["belongs_to"] == "cannot_tell" and "belongs to 'cannot_tell'" in res["reuse_validity"]["message"]
    # a text answer that is not JSON: recorded as an error, never raised into the run
    judge = ScriptedJudge("I think it is rutile.")
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, controller=_judged(tmp_path, judge))
    assert "error" in res["reuse_validity"]["escalation"] and res["reuse_validity"]["verdict"] == "good"
    # the budget: a second escalation of the same item is skipped
    judge = ScriptedJudge({"belongs_to": "high", "what_changed": "x", "confidence": "low"})
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=data, extra_params=params, controller=_judged(tmp_path, judge),
                                  state_extra={"_replay_escalations": {0: 1}})
    assert res["reuse_validity"]["escalation"] == {"trigger": "state", "skipped": "budget of 1 per item spent"} and judge.prompts == []


def test_a_flag_against_one_unit_is_escalated_only_when_nobody_attends(tmp_path, monkeypatch):
    single = tmp_path / "single"
    (single / "scripts").mkdir(parents=True)
    (single / "spectrum_0000").mkdir()
    np.save(single / "spectrum_0000" / "data.npy", rc.spectrum(rc.ANATASE, seed=5))
    (single / "scripts" / "fitting_script.py").write_text("ONE")
    (single / "series_fit_results.json").write_text(json.dumps({"results": [
        {"index": 0, "name": "s", "success": True, "parameters": rc.auto_detect_parameters(rc.ANATASE, seed=6)}]}))
    params = {"ONE": {**rc.auto_detect_parameters(rc.ANATASE, seed=95), "peak_9": {"center": 450.0, "amplitude": 0.9}}}
    judge = ScriptedJudge({"belongs_to": "none", "what_changed": "A new strong line at 450.", "confidence": "medium"})
    # a person attends (enable_human_feedback): the flag is theirs to read, no call
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, seed=7),
                                  extra_params=params, controller=_judged(tmp_path, judge, feedback=True))
    assert res["reuse_validity"]["identity"].get("flagged") is True and "escalation" not in res["reuse_validity"] and judge.prompts == []
    # unattended (an automatic chain): the judge is asked; the verdict stays good, the flag stays
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, seed=7),
                                  extra_params=params, controller=_judged(tmp_path, judge, feedback=False))
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["identity"].get("flagged") is True
    assert rv["escalation"]["trigger"] == "flag" and rv["escalation"]["belongs_to"] == "none" and len(judge.prompts) == 1
    assert unit_verdict_for({**res, "success": True})["verified"] and unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
