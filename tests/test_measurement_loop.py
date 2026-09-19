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

    def test_recommender_sees_clean_frames_only_and_cannot_fail_a_frame(self, tmp_path):
        class Rec:
            seen = []

            def observe(self, params, features):
                Rec.seen.append((params, features["peak_1_center"]))

            def suggest(self):
                if len(Rec.seen) == 2:
                    raise RuntimeError("singular kernel")
                return {"T": 310 + len(Rec.seen)}
        loop = self._armed(tmp_path, recommender=Rec())
        FakeAgent.replies = {"bad.csv": good(6.0, r2=0.3)}
        assert loop.step("a.csv", {"T": 300})["recommendation"] == {"T": 311}
        assert loop.step("bad.csv", {"T": 305})["recommendation"] == {"T": 311}
        rec = loop.step("b.csv", {"T": 311})
        assert "singular kernel" in rec["recommendation"]["error"] and rec["flags"] == []
        assert [p for p, _ in Rec.seen] == [{"T": 300}, {"T": 311}]


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
