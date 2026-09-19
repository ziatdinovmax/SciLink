"""MeasurementLoop — the live analysis driver for 1D spectra.

Lock once → execute per frame → gate as drift detector → flag on breach. These
tests pin the driver's contract with a fake curve agent: what ``setup()`` locks,
what a ``step()`` record says, when a run of bad frames becomes
``needs_escalation``, that an amendment is atomic, and that a crashed loop
resumes without losing its place. No LLM calls anywhere.
"""

import json
from pathlib import Path

import pytest

from scilink.live import LOOP_LOG_NAME, LoopNotReady, MeasurementLoop
from scilink.live import measurement_loop as ml

SCRIPT = "WINDOW = (5.0, 7.0)\nTHRESH = 0.5\nprint('FIT_RESULTS_JSON: {}')\n"


def make_anchor(root: Path, features=None) -> Path:
    d = root / "reference_run"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "fitting_script.py").write_text(SCRIPT)
    (d / "series_fit_results.json").write_text(json.dumps(
        {"results": [{"model_type": "two peaks"}], "locked_config": {"physical_model": "two peaks"}}))
    (d / "analysis_results.json").write_text(json.dumps({
        "status": "success",
        "fitting_parameters": features or {"peak_1": {"center": 6.0, "fwhm": 1.0}},
        "fit_quality": {"r_squared": 0.99}}))
    return d


class FakeAgent:
    """Stands in for CurveFittingAgent: replies from a per-file script."""
    calls = []
    replies = {}

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def analyze(self, data, **kw):
        FakeAgent.calls.append({"data": data, "output_dir": self.output_dir, **kw})
        reply = FakeAgent.replies.get(Path(str(data)).name, good(6.0))
        if isinstance(reply, Exception):
            raise reply
        return {"output_directory": self.output_dir, **reply}


def good(center, r2=0.99, drift="none", llm_calls=0):
    return {"status": "success",
            "fitting_parameters": {"peak_1": {"center": center, "fwhm": 1.0}},
            "fit_quality": {"r_squared": r2},
            "reuse_validity": {"verdict": "good" if r2 >= 0.95 else "poor",
                               "r_squared": r2, "threshold": 0.95, "drift": drift,
                               "fingerprint_similarity": 0.99 if drift == "none" else 0.6},
            "stage_timings": {"llm_calls": llm_calls}}


@pytest.fixture(autouse=True)
def _reset():
    FakeAgent.calls, FakeAgent.replies = [], {}


def loop_at(tmp_path, **kw):
    return MeasurementLoop(str(tmp_path / "loop"), agent_factory=FakeAgent, **kw)


# ──────────────────────────────────────────────────────────────
# setup
# ──────────────────────────────────────────────────────────────

class TestSetup:
    def test_adopting_a_prior_run_costs_no_analysis(self, tmp_path):
        loop = loop_at(tmp_path)
        rec = loop.setup(anchor=str(make_anchor(tmp_path)))
        assert FakeAgent.calls == []
        assert rec["source"] == "anchor" and rec["recipe"]["n_edits"] == 0
        assert rec["reference_features"] == {"peak_1_center": 6.0, "peak_1_fwhm": 1.0,
                                             "fit_r_squared": 0.99}
        assert loop.status()["armed"] is True

    def test_a_reference_file_is_analysed_once_and_becomes_the_anchor(self, tmp_path):
        anchor = make_anchor(tmp_path)

        class RefAgent(FakeAgent):
            def analyze(self, data, **kw):
                FakeAgent.calls.append({"data": data, **kw})
                return {**good(6.0), "output_directory": str(anchor),
                        "stage_timings": {"llm_calls": 7}}
        loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=RefAgent,
                               targets=["peak position"])
        rec = loop.setup(reference="ref.csv")
        [call] = FakeAgent.calls
        assert call["targets"] == ["peak position"] and "profile" not in call
        assert rec["source"] == "reference:thorough" and rec["llm_calls"] == 7
        assert loop.anchor_dir == anchor.resolve()

    def test_a_bank_served_reference_says_so(self, tmp_path):
        anchor = make_anchor(tmp_path)

        class BankAgent(FakeAgent):
            def analyze(self, data, **kw):
                return {**good(6.0), "output_directory": str(anchor),
                        "cold_start": {"id": "abc"}}
        loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=BankAgent)
        assert loop.setup(reference="ref.csv", profile="extract")["source"] == "bank"

    def test_exactly_one_source(self, tmp_path):
        loop = loop_at(tmp_path)
        with pytest.raises(ValueError, match="exactly one"):
            loop.setup()
        with pytest.raises(ValueError, match="exactly one"):
            loop.setup(reference="a.csv", anchor=str(tmp_path))

    def test_a_directory_with_no_run_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no reusable curve-fit run"):
            loop_at(tmp_path).setup(anchor=str(tmp_path))

    def test_a_failed_reference_analysis_does_not_arm_the_loop(self, tmp_path):
        class Failing(FakeAgent):
            def analyze(self, data, **kw):
                return {"status": "error", "error": {"error": "no peaks"}}
        loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=Failing)
        with pytest.raises(RuntimeError, match="did not succeed"):
            loop.setup(reference="ref.csv")
        assert loop.status()["armed"] is False

    def test_objective_key_is_checked_against_the_recipes_features(self, tmp_path):
        loop = loop_at(tmp_path, objective_key="peak_9_center")
        with pytest.raises(ValueError, match="Available: .*peak_1_center"):
            loop.setup(anchor=str(make_anchor(tmp_path)))

    def test_step_before_setup(self, tmp_path):
        with pytest.raises(LoopNotReady):
            loop_at(tmp_path).step("f.csv")


# ──────────────────────────────────────────────────────────────
# step
# ──────────────────────────────────────────────────────────────

class TestStep:
    def _armed(self, tmp_path, **kw):
        loop = loop_at(tmp_path, **kw)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        return loop

    def test_a_frame_is_a_locked_zero_llm_replay(self, tmp_path):
        loop = self._armed(tmp_path, objective_key="peak_1_center")
        rec = loop.step("f1.csv", params={"T": 300})
        [call] = FakeAgent.calls
        assert call["profile"] == "realtime" and call["reuse_locked_script"] is True
        assert call["strict_replay"] is True          # the fast clock never repairs in-frame
        assert call["prior_analysis_paths"] == [str(loop.anchor_dir)]
        assert "script_edits" not in call
        assert rec["step"] == 1 and rec["flags"] == [] and rec["llm_calls"] == 0
        assert rec["features"]["peak_1_center"] == 6.0 and rec["objective"] == 6.0
        assert rec["params"] == {"T": 300} and rec["gate"]["verdict"] == "good"
        assert rec["recipe_id"] == loop.recipe["id"] and rec["needs_escalation"] is False
        assert Path(rec["frame_dir"]).name == "frame_000001"

    def test_every_event_is_one_json_line(self, tmp_path):
        loop = self._armed(tmp_path)
        loop.step("f1.csv")
        loop.step("f2.csv")
        lines = (tmp_path / "loop" / LOOP_LOG_NAME).read_text().splitlines()
        events = [json.loads(ln) for ln in lines]
        assert [e["event"] for e in events] == ["setup", "frame", "frame"]
        assert all(e["v"] == ml.SCHEMA_VERSION and e["timestamp"] for e in events)

    def test_a_bad_frame_is_flagged_never_raised(self, tmp_path):
        loop = self._armed(tmp_path)
        FakeAgent.replies = {"boom.csv": RuntimeError("detector glitch"),
                             "err.csv": {"status": "error", "error": {"error": "empty"}}}
        a, b = loop.step("boom.csv"), loop.step("err.csv")
        assert a["flags"] == ["fit_failed"] and "detector glitch" in a["error"]
        assert b["flags"] == ["fit_failed"] and "empty" in b["error"]
        assert loop.step("ok.csv")["flags"] == []          # and the loop carries on

    def test_poor_gate_and_drift_are_separate_signals(self, tmp_path):
        loop = self._armed(tmp_path)
        FakeAgent.replies = {"poor.csv": good(6.0, r2=0.80),
                             "drift.csv": good(6.0, r2=0.99, drift="suspected")}
        assert loop.step("poor.csv")["flags"] == ["gate_poor"]
        assert loop.step("drift.csv")["flags"] == ["drift_suspected"]

    def test_one_glitch_is_tolerated_a_run_of_them_escalates(self, tmp_path):
        loop = self._armed(tmp_path, breach_patience=2)
        FakeAgent.replies = {"bad.csv": good(6.0, r2=0.5)}
        assert loop.step("bad.csv")["needs_escalation"] is False
        assert loop.step("ok.csv")["needs_escalation"] is False     # streak broken
        loop.step("bad.csv")
        rec = loop.step("bad.csv")
        assert rec["needs_escalation"] is True and rec["consecutive_breaches"] == 2
        assert loop.status()["needs_escalation"] is True

    def test_leaving_the_zero_llm_path_is_visible(self, tmp_path):
        loop = self._armed(tmp_path)
        FakeAgent.replies = {"f.csv": good(6.0, llm_calls=1)}
        rec = loop.step("f.csv")
        assert "llm_used" in rec["flags"] and rec["llm_calls"] == 1
        assert loop.status()["llm_calls_in_frames"] == 1

    def test_deadline_is_flagged_not_enforced(self, tmp_path, monkeypatch):
        loop = self._armed(tmp_path, frame_deadline_s=0.5)
        ticks = iter([0.0, 2.0])
        monkeypatch.setattr(ml.time, "perf_counter", lambda: next(ticks))
        rec = loop.step("slow.csv")
        assert rec["flags"] == ["deadline_missed"] and rec["latency_s"] == 2.0
        assert rec["needs_escalation"] is False      # slow is not wrong

    def test_a_value_far_outside_what_clean_frames_produced_is_flagged(self, tmp_path):
        loop = self._armed(tmp_path)
        FakeAgent.replies = {f"f{i}.csv": good(6.0 + 0.1 * i) for i in range(5)}
        for i in range(5):
            assert loop.step(f"f{i}.csv")["flags"] == []     # range walks with the data
        FakeAgent.replies["jump.csv"] = good(9.5)
        rec = loop.step("jump.csv")
        assert rec["flags"] == ["out_of_reference_range"]
        assert rec["out_of_range"]["peak_1_center"]["value"] == 9.5
        assert rec["needs_escalation"] is False              # a flag, not a breach
        # a flagged frame must not teach the range
        assert loop._ranges["peak_1_center"][1] == pytest.approx(6.4)

    def test_the_range_gate_only_learns_during_warm_up(self, tmp_path):
        # Observed live: armed on the single reference value, the gate flagged
        # every frame of a healthy stream.
        loop = self._armed(tmp_path, range_warmup=5)
        FakeAgent.replies = {"far.csv": good(9.5)}
        assert loop.step("far.csv")["flags"] == []           # frame 1: nothing to judge by
        assert loop._ranges["peak_1_center"] == [6.0, 9.5]   # ...but it learns

    def test_fit_uncertainties_are_never_gated(self, tmp_path):
        loop = loop_at(tmp_path, range_warmup=0)
        loop.setup(anchor=str(make_anchor(tmp_path, features={
            "peak_1": {"center": 6.0, "center_err": 0.003, "eta": 0.03}})))

        def frame(err, eta=0.03):
            return {**good(6.0), "fitting_parameters": {
                "peak_1": {"center": 6.0, "center_err": err, "eta": eta}}}
        FakeAgent.replies = {"noisy_err.csv": frame(0.05), "eta.csv": frame(0.003, eta=0.4)}
        assert loop.step("noisy_err.csv")["flags"] == []     # 17x the reference error: fine
        assert loop.step("eta.csv")["flags"] == ["out_of_reference_range"]

    def test_gate_keys_and_objective_key_narrow_the_gate(self, tmp_path):
        loop = loop_at(tmp_path, range_warmup=0, objective_key="peak_1_center")
        loop.setup(anchor=str(make_anchor(tmp_path)))
        FakeAgent.replies = {"wide.csv": {**good(6.0), "fitting_parameters": {
            "peak_1": {"center": 6.0, "fwhm": 9.0}}}}
        assert loop.step("wide.csv")["flags"] == []          # fwhm is not the objective

    def test_declared_outputs_are_what_the_gate_watches(self, tmp_path):
        # Observed live (beamline XRD): an ill-determined mixing fraction
        # wandering over decades flagged clean frames; the user never asked for it.
        loop = loop_at(tmp_path, range_warmup=0)
        loop.setup(anchor=str(make_anchor(tmp_path, features={
            "peak_1": {"center": 6.0, "eta": 1e-9}})))
        loop.outputs = {"peak_1_center": "centre of the first peak"}

        def frame(center, eta):
            return {**good(center), "fitting_parameters": {"peak_1": {"center": center, "eta": eta}}}
        FakeAgent.replies = {"eta.csv": frame(6.0, 0.9), "moved.csv": frame(9.5, 0.9)}
        assert loop.step("eta.csv")["flags"] == []
        assert loop.step("moved.csv")["flags"] == ["out_of_reference_range"]

    def test_a_persistent_new_level_on_good_fits_is_adopted(self, tmp_path):
        loop = self._armed(tmp_path, range_warmup=0, range_adopt_after=3)
        FakeAgent.replies = {f"hi{i}.csv": good(9.5) for i in range(4)}
        assert loop.step("hi0.csv")["flags"] == ["out_of_reference_range"]
        assert loop.step("hi1.csv")["flags"] == ["out_of_reference_range"]
        rec = loop.step("hi2.csv")
        assert rec["flags"] == [] and rec["range_adopted"] == ["peak_1_center"]
        assert loop.step("hi3.csv")["flags"] == []

    def test_a_bad_fit_is_not_also_range_judged(self, tmp_path):
        loop = self._armed(tmp_path, range_warmup=0)
        FakeAgent.replies = {"bad.csv": good(9.5, r2=0.4)}
        assert loop.step("bad.csv")["flags"] == ["gate_poor"]

    def test_recommender_sees_flagged_frames_too_and_cannot_fail_a_frame(self, tmp_path):
        """Observed live: shown clean frames only, a GP exploring focus starved
        (defocus reads as drift) and an LLM asked to fix low SNR was never
        asked (every frame was below the fit gate)."""
        class Rec:
            seen = []

            def observe(self, params, features, step=None, flags=None):
                Rec.seen.append((params, flags))

            def suggest(self):
                if len(Rec.seen) == 3:
                    raise RuntimeError("singular kernel")
                return {"params": {"T": 310 + len(Rec.seen)}}
        loop = self._armed(tmp_path, recommender=Rec())
        FakeAgent.replies = {"noisy.csv": good(6.0, r2=0.3),
                             "dead.csv": {"status": "error", "error": {"error": "empty"}}}
        assert loop.step("a.csv", {"T": 300})["recommendation"]["params"] == {"T": 311}
        assert loop.step("noisy.csv", {"T": 305})["recommendation"]["params"] == {"T": 312}
        assert loop.step("dead.csv", {"T": 306})["recommendation"]["params"] == {"T": 312}
        rec = loop.step("b.csv", {"T": 311})
        assert rec["flags"] == [] and rec["recommendation"]["valid"] is False
        assert "singular kernel" in rec["recommendation"]["problems"][0]
        # the noisy frame was observed WITH its flag; the dead one had nothing to observe
        assert Rec.seen == [({"T": 300}, []), ({"T": 305}, ["gate_poor"]), ({"T": 311}, [])]

    def test_the_range_gate_stands_down_when_conditions_are_being_changed(self, tmp_path):
        class Hold:
            def observe(self, *a, **k): pass
            def suggest(self): return {"params": {"T": 1}}
        loop = self._armed(tmp_path, recommender=Hold(), range_warmup=0)
        FakeAgent.replies = {"far.csv": good(9.5)}
        assert loop.step("far.csv", {"T": 1})["flags"] == []
        watched = loop_at(tmp_path / "w", recommender=Hold(), range_warmup=0,
                          gate_keys=["peak_1_center"])
        watched.setup(anchor=str(loop.anchor_dir))
        FakeAgent.replies = {"far.csv": good(9.5)}
        assert watched.step("far.csv", {"T": 1})["flags"] == ["out_of_reference_range"]


# ──────────────────────────────────────────────────────────────
# amend / resume
# ──────────────────────────────────────────────────────────────

class TestAmendAndResume:
    def test_amend_changes_the_recipe_id_and_travels_with_every_frame(self, tmp_path):
        loop = loop_at(tmp_path)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        before = loop.recipe["id"]
        edit = {"old_text": "WINDOW = (5.0, 7.0)", "new_text": "WINDOW = (4.0, 8.0)"}
        rec = loop.amend([edit], note="peak walking to the window edge")
        assert rec["previous_recipe_id"] == before and rec["recipe"]["id"] != before
        frame = loop.step("f.csv")
        assert FakeAgent.calls[-1]["script_edits"] == [edit]
        assert frame["recipe_id"] == rec["recipe"]["id"]

    def test_amendments_stack_against_the_recipe_as_it_runs_now(self, tmp_path):
        loop = loop_at(tmp_path)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.amend([{"old_text": "THRESH = 0.5", "new_text": "THRESH = 0.4"}])
        loop.amend([{"old_text": "THRESH = 0.4", "new_text": "THRESH = 0.3"}])
        assert loop.recipe["n_edits"] == 2

    def test_an_edit_that_does_not_apply_changes_nothing(self, tmp_path):
        loop = loop_at(tmp_path)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        before = dict(loop.recipe)
        with pytest.raises(ValueError, match="do not apply"):
            loop.amend([{"old_text": "THRESH = 0.5", "new_text": "THRESH = 0.4"},
                        {"old_text": "NO_SUCH_LINE", "new_text": "x"}])
        assert loop.recipe == before and loop._edits == []
        assert [e["event"] for e in loop.read_log()] == ["setup"]

    def test_setup_with_edits_is_tomorrows_loop_from_yesterdays_recipe(self, tmp_path):
        loop = loop_at(tmp_path)
        edit = {"old_text": "THRESH = 0.5", "new_text": "THRESH = 0.7"}
        assert loop.setup(anchor=str(make_anchor(tmp_path)),
                          script_edits=[edit])["recipe"]["n_edits"] == 1
        loop.step("f.csv")
        assert FakeAgent.calls[-1]["script_edits"] == [edit]

    def test_resume_continues_where_the_log_stopped(self, tmp_path):
        loop = loop_at(tmp_path, breach_patience=3, frame_deadline_s=5.0)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.amend([{"old_text": "THRESH = 0.5", "new_text": "THRESH = 0.4"}])
        loop.step("f1.csv")
        loop.step("f2.csv")
        recipe = dict(loop.recipe)

        again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=FakeAgent)
        assert again.recipe == recipe and again.breach_patience == 3
        assert again.frame_deadline_s == 5.0
        rec = again.step("f3.csv")
        assert rec["step"] == 3 and rec["recipe_id"] == recipe["id"]
        assert FakeAgent.calls[-1]["script_edits"][0]["new_text"] == "THRESH = 0.4"
        assert [e["event"] for e in again.read_log()] == [
            "setup", "amend", "frame", "frame", "resume", "frame"]

    def test_a_torn_log_line_is_skipped(self, tmp_path):
        loop = loop_at(tmp_path)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.step("f.csv")
        with open(loop.log_path, "a") as fh:
            fh.write('{"event": "fra')
        assert len(loop.read_log()) == 2 and loop.status()["frames"] == 1


# ──────────────────────────────────────────────────────────────
# escalation
# ──────────────────────────────────────────────────────────────

class FakeEscalation:
    """A re-anchor whose completion the test controls."""
    last = None

    def __init__(self, spec):
        self.spec, self.result = spec, None
        FakeEscalation.last = self

    def poll(self):
        return self.result


def new_anchor(root, name="regime2", features=None):
    d = root / name
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "fitting_script.py").write_text("THREE_PEAKS = True\n")
    (d / "series_fit_results.json").write_text(json.dumps({"results": []}))
    (d / "analysis_results.json").write_text(json.dumps({
        "status": "success", "fit_quality": {"r_squared": 0.99},
        "fitting_parameters": features or {"peak_1": {"center": 6.9},
                                           "peak_3": {"center": 16.5}}}))
    return d


class TestEscalation:
    def _breaching(self, tmp_path, **kw):
        loop = loop_at(tmp_path, breach_patience=2, escalation_runner=FakeEscalation,
                       targets=["first peak"], **kw)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        FakeAgent.replies = {"bad.csv": good(6.9, r2=0.4, drift="suspected")}
        return loop

    def test_step_keeps_answering_while_the_reanchor_runs(self, tmp_path):
        loop = self._breaching(tmp_path)
        loop.step("bad.csv")
        assert loop.step("bad.csv")["needs_escalation"] is True
        started = loop.escalate("bad.csv")
        spec = FakeEscalation.last.spec
        assert started["event"] == "escalation_started" and loop.escalating
        assert spec["analyze_kwargs"]["profile"] == "extract"          # the cost ladder
        assert spec["analyze_kwargs"]["targets"] == ["first peak"]
        assert spec["data_path"] == "bad.csv"
        old = loop.recipe["id"]
        rec = loop.step("bad.csv")                                     # still the old recipe
        assert rec["escalation"] == "running" and rec["recipe_id"] == old
        assert rec["flags"] == ["gate_poor", "drift_suspected"]

    def test_the_new_recipe_is_adopted_between_frames(self, tmp_path):
        loop = self._breaching(tmp_path)
        loop.step("bad.csv"); loop.step("bad.csv")
        loop.amend([{"old_text": "THRESH = 0.5", "new_text": "THRESH = 0.4"}])
        loop.escalate("bad.csv")
        loop.step("bad.csv")
        old = loop.recipe["id"]
        anchor2 = new_anchor(tmp_path)
        FakeEscalation.last.result = {"status": "success", "output_directory": str(anchor2),
                                      "llm_calls": 4, "seconds": 61.0}
        FakeAgent.replies = {"new.csv": good(6.9)}
        rec = loop.step("new.csv")
        assert rec["recipe_id"] != old and rec["flags"] == [] and "escalation" not in rec
        assert FakeAgent.calls[-1]["prior_analysis_paths"] == [str(anchor2.resolve())]
        assert "script_edits" not in FakeAgent.calls[-1]      # old amendments do not carry
        [re] = [e for e in loop.read_log() if e["event"] == "reanchor"]
        assert re["from_recipe"] == old and re["source"] == "reanchor:extract"
        assert re["llm_calls"] == 4 and re["frames_answered_meanwhile"] == 1
        assert loop._n_learned == 1 and loop._consecutive_breaches == 0   # ranges re-learn
        assert loop.status()["reanchors"] == 1 and not loop.status()["escalating"]

    def test_a_regime_the_bank_already_knows_costs_no_model_call(self, tmp_path):
        loop = self._breaching(tmp_path)
        loop.escalate("bad.csv")
        FakeEscalation.last.result = {"status": "success", "llm_calls": 0, "seconds": 2.1,
                                      "output_directory": str(new_anchor(tmp_path)),
                                      "cold_start": {"id": "abc"}}
        loop.step("x.csv")
        [re] = [e for e in loop.read_log() if e["event"] == "reanchor"]
        assert re["source"] == "bank" and re["llm_calls"] == 0

    def test_a_failed_reanchor_keeps_the_old_recipe(self, tmp_path):
        loop = self._breaching(tmp_path)
        old = loop.recipe["id"]
        loop.escalate("bad.csv")
        FakeEscalation.last.result = {"status": "error", "error": "503 from the endpoint"}
        rec = loop.step("bad.csv")
        assert rec["recipe_id"] == old and not loop.escalating
        [failed] = [e for e in loop.read_log() if e["event"] == "escalation_failed"]
        assert "503" in failed["error"]
        loop.escalate("bad.csv")                                # and it may be retried

    def test_one_escalation_at_a_time(self, tmp_path):
        loop = self._breaching(tmp_path)
        loop.escalate("bad.csv")
        with pytest.raises(RuntimeError, match="already running"):
            loop.escalate("bad.csv")

    def test_auto_escalate_starts_on_the_frame_that_raised_it(self, tmp_path):
        loop = self._breaching(tmp_path, auto_escalate=True)
        assert "escalation" not in loop.step("bad.csv")
        rec = loop.step("bad.csv")
        assert rec["needs_escalation"] and rec["escalation"] == "started"
        assert FakeEscalation.last.spec["data_path"] == "bad.csv"
        loop.step("bad.csv")                                    # no second one
        assert [e["event"] for e in loop.read_log()].count("escalation_started") == 1

    def test_never_escalates_by_itself_unless_asked(self, tmp_path):
        loop = self._breaching(tmp_path)
        for _ in range(4):
            loop.step("bad.csv")
        assert FakeEscalation.last is None or not loop.escalating

    def test_the_spec_is_never_written_to_disk(self, tmp_path):
        loop = self._breaching(tmp_path)
        loop.api_key = "sk-SECRET"
        loop.escalate("bad.csv")
        assert FakeEscalation.last.spec["agent_kwargs"]["api_key"] == "sk-SECRET"
        for f in (tmp_path / "loop").rglob("*"):
            if f.is_file():
                assert "sk-SECRET" not in f.read_text(errors="ignore")


class TestReanchorSubprocess:
    """The real worker plumbing, without a model: a frame that does not exist
    fails fast, which is enough to prove the spec arrives over stdin, the
    result comes back as a file, and nothing secret touches the disk."""

    def test_spec_over_stdin_result_as_a_file(self, tmp_path):
        import time as _t
        out = tmp_path / "escalation_001"
        esc = ml._ProcessEscalation({
            "out_dir": str(out), "data_path": str(tmp_path / "no_such_frame.csv"),
            "agent_kwargs": {"api_key": "sk-SECRET-123", "model_name": "gpt-4o",
                             "base_url": None},
            "analyze_kwargs": {"system_info": None, "profile": "extract"},
            "sandbox_approved": False})
        deadline = _t.time() + 120
        result = None
        while result is None and _t.time() < deadline:
            result = esc.poll()
            _t.sleep(0.5)
        assert result is not None, "worker never reported"
        assert result["status"] != "success" and "seconds" in result
        assert (out / "result.json").exists()
        for f in out.rglob("*"):
            if f.is_file():
                assert "sk-SECRET-123" not in f.read_text(errors="ignore"), f

    def test_it_is_not_a_multiprocessing_child(self):
        # A spawned multiprocessing child re-imports the caller's __main__ —
        # observed live: an unguarded driving script re-ran itself inside the
        # worker. The worker must be a `-m` subprocess.
        import inspect
        src = inspect.getsource(ml._ProcessEscalation)
        assert '"-m", "scilink.live._reanchor"' in src
        assert "get_context" not in src


class TestStrictReplay:
    """Observed live: back in a two-peak regime, the three-peak recipe could not
    execute, and the realtime profile's forgiving fallback repaired / re-derived
    it INSIDE step() — three frames at ~40 s and an LLM call each. Under strict
    replay the failure is the result."""

    def _try_reuse(self, strict):
        from types import SimpleNamespace
        from scilink.agents.exp_agents._qc_engine import QCItemContext
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController as C)
        host = SimpleNamespace(
            logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
            _fit_single_spectrum=lambda **kw: {"success": False, "error": "needs 3 peaks",
                                               "parameters": {}, "fit_quality": {}})
        ctx = QCItemContext(state={"_strict_replay": strict}, data=None, data_path="f.csv",
                            item_name="f", item_idx=0, reuse_script="s",
                            reuse_source="anchor")
        return C.qc_try_reuse(host, ctx)

    def test_strict_returns_the_failure_instead_of_rederiving(self):
        out = self._try_reuse(strict=True)
        assert out is not None and out["success"] is False
        assert out["reuse_validity"]["verdict"] == "failed"

    def test_default_keeps_the_forgiving_fallback(self):
        assert self._try_reuse(strict=False) is None      # None = re-derive (today)


# ──────────────────────────────────────────────────────────────
# the reference of a stream: told it will be replayed, and planned from
# several frames when there are several
# ──────────────────────────────────────────────────────────────

class TestStreamReference:
    def test_the_reference_analysis_is_told_it_will_be_replayed(self, tmp_path):
        anchor = make_anchor(tmp_path)

        class RefAgent(FakeAgent):
            def analyze(self, data, **kw):
                FakeAgent.calls.append({"data": data, **kw})
                return {**good(6.0), "output_directory": str(anchor)}
        MeasurementLoop(str(tmp_path / "loop"), agent_factory=RefAgent).setup(reference="ref.csv")
        assert FakeAgent.calls[0]["stream_reference"] is True
        from scilink.agents.exp_agents._qc_profile import planning_addendum
        text = planning_addendum({"stream_reference": True, "analysis_targets": ["peak position"]})
        assert "## Scope" in text and "## Replay" in text and "replayed unchanged" in text
        assert planning_addendum({"stream_reference": False}) is None

    def _series_run(self, root, n=4, regimes=2):
        d = root / "series_run"
        d.mkdir()
        results = []
        for i in range(n):
            (d / f"spectrum_{i:04d}").mkdir()
            (d / f"spectrum_{i:04d}" / "marker.txt").write_text(str(i))
            results.append({"index": i, "name": f"f{i}", "success": i != n - 1 or True,
                            "model_type": "two peaks" if i >= n - 2 else "one peak",
                            "script": f"# script for frame {i}\n" + SCRIPT,
                            "parameters": {"peak_1": {"center": 6.0 + i}},
                            "fit_quality": {"r_squared": 0.99}})
        (d / "series_fit_results.json").write_text(json.dumps({
            "total_spectra": n, "is_single_spectrum": False,
            "locked_config": {"physical_model": "one peak"},
            "series_analysis_plan": {"regimes": [{}] * regimes}, "results": results}))
        return d

    def test_several_frames_are_a_series_and_the_recipe_locks_on_the_last(self, tmp_path):
        series = self._series_run(tmp_path)

        class SeriesAgent(FakeAgent):
            def analyze(self, data, **kw):
                FakeAgent.calls.append({"data": data, **kw})
                if isinstance(data, list):
                    return {"status": "success", "output_directory": str(series),
                            "stage_timings": {"llm_calls": 9}}
                return {**good(9.0), "output_directory": self.output_dir}
        loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=SeriesAgent,
                               check_portability=False)
        files = [f"f{i}.csv" for i in range(4)]
        rec = loop.setup(reference=files)
        call = FakeAgent.calls[0]
        assert call["data"] == files and call["series_metadata"]["values"] == [0, 1, 2, 3]
        assert call["profile"] == {"base": "thorough", "trend": False, "synthesis": "none",
                                   "adaptive_refit": False}
        assert rec["source"] == "reference:thorough:4 frames" and rec["llm_calls"] == 9
        assert rec["reference_frames"] == {"n": 4, "fitted": 4, "anchored_on": 3, "regimes": 2,
                                           "model": "two peaks", "llm_calls": 9}
        anchor = loop.anchor_dir
        assert anchor.name == "reference_anchor"
        assert (anchor / "scripts" / "fitting_script.py").read_text().startswith("# script for frame 3")
        assert (anchor / "spectrum_0000" / "marker.txt").read_text() == "3"
        single = json.loads((anchor / "series_fit_results.json").read_text())
        assert single["locked_config"]["physical_model"] == "two peaks"      # the last regime's
        assert single["is_single_spectrum"] and len(single["results"]) == 1
        assert rec["reference_features"]["peak_1_center"] == 9.0
        # and the stream replays against that single-frame anchor
        loop.step("next.csv")
        assert FakeAgent.calls[-1]["prior_analysis_paths"] == [str(anchor)]

    def test_a_series_with_no_fitted_frame_does_not_arm(self, tmp_path):
        series = self._series_run(tmp_path)
        data = json.loads((series / "series_fit_results.json").read_text())
        for r in data["results"]:
            r["success"] = False
        (series / "series_fit_results.json").write_text(json.dumps(data))

        class SeriesAgent(FakeAgent):
            def analyze(self, data, **kw):
                return {"status": "success", "output_directory": str(series)}
        loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=SeriesAgent)
        with pytest.raises(RuntimeError, match="no frame of the reference series"):
            loop.setup(reference=["a.csv", "b.csv"])


# ──────────────────────────────────────────────────────────────
# a re-anchor planned from the recent frames, not one snapshot
# ──────────────────────────────────────────────────────────────

class TestReanchorWindow:
    def _frames(self, tmp_path, n):
        d = tmp_path / "incoming"
        d.mkdir(exist_ok=True)
        paths = []
        for i in range(n):
            p = d / f"frame_{i:03d}.csv"
            p.write_text("x,y\n0,1\n")
            paths.append(str(p))
        return paths

    def test_the_window_is_the_recent_frames_ending_in_the_breaching_one(self, tmp_path):
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation, reanchor_frames=4)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        paths = self._frames(tmp_path, 7)
        FakeAgent.replies = {Path(paths[4]).name: RuntimeError("detector glitch")}
        for p in paths:
            loop.step(p)
        loop.escalate(paths[-1])
        spec = FakeEscalation.last.spec
        # four frames, newest last, the dead frame 4 left out
        assert spec["data_path"] == [paths[2], paths[3], paths[5], paths[6]]
        assert spec["analyze_kwargs"]["stream_reference"] is True
        started = next(e for e in loop.read_log() if e["event"] == "escalation_started")
        assert started["window"] == 4 and started["data"] == paths[-1]

    def test_one_frame_when_asked_or_when_that_is_all_there_is(self, tmp_path):
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation, reanchor_frames=1)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        paths = self._frames(tmp_path, 3)
        for p in paths:
            loop.step(p)
        loop.escalate(paths[-1])
        assert FakeEscalation.last.spec["data_path"] == paths[-1]
        gone = loop_at(tmp_path / "b", escalation_runner=FakeEscalation, reanchor_frames=5)
        gone.setup(anchor=str(make_anchor(tmp_path / "b")))
        gone.step("not_on_disk_1.csv"); gone.step("not_on_disk_2.csv")
        gone.escalate("not_on_disk_2.csv")                     # earlier frames no longer exist
        assert FakeEscalation.last.spec["data_path"] == "not_on_disk_2.csv"

    def test_explicit_frames_and_a_restart(self, tmp_path):
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation, reanchor_frames=3)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        paths = self._frames(tmp_path, 5)
        for p in paths[:4]:
            loop.step(p)
        again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=FakeAgent,
                                       escalation_runner=FakeEscalation)
        assert again.reanchor_frames == 3 and again._recent_frames == paths[:4]
        again.escalate(paths[4], frames=paths[:2])
        assert FakeEscalation.last.spec["data_path"] == [paths[0], paths[1], paths[4]]

    def test_the_window_is_recorded_when_the_recipe_is_adopted(self, tmp_path):
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.step("a.csv")
        loop.escalate("a.csv")
        FakeEscalation.last.result = {"status": "success", "seconds": 60, "llm_calls": 6,
                                      "output_directory": str(new_anchor(tmp_path)),
                                      "window": {"n": 5, "fitted": 5, "anchored_on": 4, "regimes": 2}}
        loop.step("b.csv")
        event = next(e for e in loop.read_log() if e["event"] == "reanchor")
        assert event["window"]["n"] == 5 and event["window"]["regimes"] == 2


class TestReanchorWorker:
    """scilink.live._reanchor with a stand-in for the curve agent."""

    def _agent(self, monkeypatch, tmp_path, bank_hit):
        import scilink.agents.exp_agents.curve_fitting_agent as mod
        calls = []
        series = TestStreamReference()._series_run(tmp_path)

        class Agent:
            model = None

            def __init__(self, output_dir, **kw):
                self.output_dir = output_dir

            def analyze(self, data, **kw):
                calls.append({"data": data, **kw})
                if kw.get("bank_only"):
                    return ({"status": "success", "output_directory": str(make_anchor(tmp_path)),
                             "cold_start": {"id": "abc"}} if bank_hit
                            else {"status": "no_bank_recipe"})
                return {"status": "success", "output_directory": str(series),
                        "stage_timings": {"llm_calls": 5}}
        monkeypatch.setattr(mod, "CurveFittingAgent", Agent)
        return calls

    def _run(self, tmp_path, data):
        from scilink.live._reanchor import reanchor
        out = tmp_path / "esc"
        reanchor({"out_dir": str(out), "data_path": data, "agent_kwargs": {},
                  "analyze_kwargs": {"profile": "extract", "stream_reference": True}})
        return json.loads((out / "result.json").read_text())

    def test_the_bank_is_asked_about_the_newest_frame_first(self, tmp_path, monkeypatch):
        calls = self._agent(monkeypatch, tmp_path, bank_hit=True)
        res = self._run(tmp_path, ["f0.csv", "f1.csv", "f2.csv"])
        assert [(c["data"], c.get("bank_only")) for c in calls] == [("f2.csv", True)]
        assert res["status"] == "success" and res["cold_start"] and res["window"] is None

    def test_no_banked_recipe_means_a_series_over_the_window(self, tmp_path, monkeypatch):
        calls = self._agent(monkeypatch, tmp_path, bank_hit=False)
        files = ["f0.csv", "f1.csv", "f2.csv", "f3.csv"]
        res = self._run(tmp_path, files)
        assert calls[1]["data"] == files and "bank_only" not in calls[1]
        assert calls[1]["profile"] == {"base": "extract", "trend": False, "synthesis": "none",
                                       "adaptive_refit": False}
        assert res["window"] == {"n": 4, "fitted": 4, "anchored_on": 3, "regimes": 2,
                                 "model": "two peaks"}
        anchor = Path(res["output_directory"])
        assert anchor.name == "anchor" and (
            anchor / "scripts" / "fitting_script.py").read_text().startswith("# script for frame 3")

    def test_a_single_frame_is_analysed_as_before(self, tmp_path, monkeypatch):
        calls = self._agent(monkeypatch, tmp_path, bank_hit=False)
        self._run(tmp_path, "only.csv")
        assert len(calls) == 1 and calls[0]["data"] == "only.csv" and "bank_only" not in calls[0]


class TestDriftRebase:
    """After a window-planned re-anchor the anchor FRAME is stale by
    construction. Observed live (XRD through a phase transition): the new recipe
    fitted every frame at R² 0.99 and the loop rebuilt it at once, because pure
    phase B no longer looked like the mid-transition anchor frame."""

    def _curve(self, tmp_path, name, centers):
        import numpy as np
        x = np.linspace(0, 20, 600)
        y = 0.2 + sum(5.0 * np.exp(-0.5 * ((x - c) / 0.25) ** 2) for c in centers)
        y = y + np.random.default_rng(len(name)).normal(0, 0.01, x.size)
        p = tmp_path / name
        np.savetxt(p, np.column_stack([x, y]), delimiter=",", header="x,y", comments="")
        return str(p)

    def _adopted(self, tmp_path, window):
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation, breach_patience=3,
                       noise_gate_tolerance=None)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.step("a.csv")
        loop.escalate("a.csv")
        FakeEscalation.last.result = {"status": "success", "seconds": 60, "llm_calls": 6,
                                      "output_directory": str(new_anchor(tmp_path)),
                                      "window": window}
        return loop

    def _reply(self, path, r2=0.99):
        FakeAgent.replies[Path(path).name] = good(6.9, r2=r2, drift="suspected")

    def test_a_stable_stream_that_fits_moves_the_reference(self, tmp_path):
        loop = self._adopted(tmp_path, {"n": 5, "regimes": 2})
        frames = [self._curve(tmp_path, f"b{i}.csv", [5, 9, 14]) for i in range(3)]
        other = self._curve(tmp_path, "changed.csv", [3, 5, 7, 9, 11, 14, 17])
        for f in frames + [other]:
            self._reply(f)
        first, second, third = (loop.step(f) for f in frames)
        assert first["flags"] == ["drift_suspected"]              # one frame is not yet a pattern
        assert second["flags"] == [] and third["flags"] == []
        assert third["gate"]["fingerprint_similarity_rebased"] > 0.92
        [event] = [e for e in loop.read_log() if e["event"] == "drift_rebased"]
        assert event["recipe_id"] == loop.recipe["id"]
        assert loop.step(other)["flags"] == ["drift_suspected"]   # a real change still shows
        again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=FakeAgent)
        assert again._drift_reference == loop._drift_reference

    def test_only_a_window_planned_recipe_earns_it(self, tmp_path):
        loop = self._adopted(tmp_path, None)                      # rebuilt from one snapshot
        frames = [self._curve(tmp_path, f"b{i}.csv", [5, 9, 14]) for i in range(3)]
        for f in frames:
            self._reply(f)
        assert [loop.step(f)["flags"] for f in frames] == [["drift_suspected"]] * 3

    def test_a_poor_fit_or_a_moving_stream_cancels_it(self, tmp_path):
        loop = self._adopted(tmp_path, {"n": 5})
        bad = self._curve(tmp_path, "bad.csv", [5, 9, 14])
        self._reply(bad, r2=0.5)
        ok = [self._curve(tmp_path, f"b{i}.csv", [5, 9, 14]) for i in range(2)]
        for f in ok:
            self._reply(f)
        assert set(loop.step(bad)["flags"]) == {"gate_poor", "drift_suspected"}
        assert [loop.step(f)["flags"] for f in ok] == [["drift_suspected"]] * 2
        assert not [e for e in loop.read_log() if e["event"] == "drift_rebased"]

        moving = self._adopted(tmp_path / "m", {"n": 5})
        a = self._curve(tmp_path, "m0.csv", [5])
        b = self._curve(tmp_path, "m1.csv", [2, 4, 6, 8, 10, 12, 14, 16, 18])
        self._reply(a); self._reply(b)
        assert moving.step(a)["flags"] == ["drift_suspected"]
        assert moving.step(b)["flags"] == ["drift_suspected"]    # frames disagree: still changing
