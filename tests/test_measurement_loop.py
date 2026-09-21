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

    def test_an_open_ended_run_does_not_keep_every_frames_folder(self, tmp_path):
        # Live: a recipe's tool left 770 MB in each frame's folder, 12 GB in eleven frames.
        loop = self._armed(tmp_path, keep_frame_dirs=3)
        for i in range(1, 8):
            rec = loop.step(f"f{i}.csv")
            Path(rec["frame_dir"]).mkdir(parents=True, exist_ok=True)      # what a replay leaves
            (Path(rec["frame_dir"]) / "heavy.bin").write_bytes(b"x")
        kept = sorted(p.name for p in (loop.output_dir / "frames").iterdir())
        assert kept == ["frame_000005", "frame_000006", "frame_000007"]
        assert len([e for e in loop.read_log() if e["event"] == "frame"]) == 7   # the record is whole
        everything = self._armed(tmp_path / "all", keep_frame_dirs=None)
        for i in range(1, 6):
            Path(everything.step(f"g{i}.csv")["frame_dir"]).mkdir(parents=True, exist_ok=True)
        assert len(list((everything.output_dir / "frames").iterdir())) == 5

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


class TestChangeSignalAndAudit:
    """The loop's own change signal (live/drift.py) and what the slow clock does
    about a stream that changed: a rebuild when the recipe fails, an independent
    audit when it still fits."""

    OUT = {"peak_1_center": "centre of the first peak"}

    def _curve(self, tmp_path, name, centers, seed=0):
        import numpy as np
        x = np.linspace(0, 20, 600)
        y = 0.2 + sum(5.0 * np.exp(-0.5 * ((x - c) / 0.25) ** 2) for c in centers)
        y = y + np.random.default_rng(seed).normal(0, 0.02, x.size)
        p = tmp_path / name
        np.savetxt(p, np.column_stack([x, y]), delimiter=",", header="x,y", comments="")
        return str(p)

    def _armed(self, tmp_path, **kw):
        ref = self._curve(tmp_path, "ref.csv", [5, 9, 14], seed=1)
        loop = loop_at(tmp_path, escalation_runner=FakeEscalation, breach_patience=2,
                       auto_escalate=True, check_portability=False, **kw)
        loop.setup(anchor=str(make_anchor(tmp_path)), reference_data=ref)
        loop.outputs = dict(self.OUT)                   # named outputs, without a pinning model
        for i in range(5):
            assert loop.step(self._curve(tmp_path, f"n{i}.csv", [5, 9, 14], seed=10 + i))["flags"] == []
        return loop

    def _changed(self, tmp_path, n=4):
        return [self._curve(tmp_path, f"c{i}.csv", [5, 9, 14, 17], seed=30 + i) for i in range(n)]

    def _finish(self, tmp_path, center, **extra):
        FakeEscalation.last.result = {
            "status": "success", "seconds": 40, "llm_calls": 5, "pin_edits": [{"old_text": "THREE_PEAKS = True", "new_text": "THREE_PEAKS = True  # pinned"}],
            "output_directory": str(new_anchor(tmp_path, name=f"audit_{center}")),
            "pin_features": {"peak_1_center": center, "peak_1_center_err": 0.001}, **extra}

    def test_the_signal_is_graded_and_on_every_frame(self, tmp_path):
        loop = self._armed(tmp_path)
        gate = loop.read_log()[-1]["gate"]
        assert gate["drift_fraction"] < 0.05 and "drift_from_reference" in gate
        changed = loop.step(self._changed(tmp_path)[0])
        assert changed["flags"] == ["drift_suspected"] and changed["gate"]["drift_fraction"] > 0.2

    # ── novelty: in discovery work the announcement IS the result ────────────
    def test_a_lasting_change_is_announced_with_how_much_and_where(self, tmp_path):
        loop = self._armed(tmp_path)                                  # default: on_change="report"
        frames = self._changed(tmp_path)                              # a new peak at x = 17
        first, second = loop.step(frames[0]), loop.step(frames[1])
        assert first["flags"] == second["flags"] == ["drift_suspected"]
        [novelty] = [e for e in loop.read_log() if e["event"] == "novelty"]
        assert novelty["since_step"] == first["step"] and novelty["recipe_fits"] is True
        assert novelty["fraction"] > 0.2 and novelty["frames"] == frames[:2]
        [where] = novelty["where"]
        assert where["kind"] == "new" and abs(where["x_peak"] - 17.0) < 0.3
        # The frame that announces it never also accepts it: a driver may pause there,
        # and the decision made at the pause comes before the state is taken as normal.
        assert second["novelty"]["since_step"] == first["step"]
        assert not [e for e in loop.read_log() if e["event"] == "state_accepted"]
        third = loop.step(frames[2])
        # accepted for tracking, with no model call and nothing started in the background
        accepted = next(e for e in loop.read_log() if e["event"] == "state_accepted")
        assert accepted["step"] == third["step"] and accepted["verified"] is False
        assert "agree with each other" in accepted["how"]
        assert not loop.escalating and third.get("escalation") is None and "novelty" not in third
        assert loop.step(frames[3])["flags"] == []
        assert [e["event"] for e in loop.read_log()].count("novelty") == 1     # once per change

    # ── a change that arrives slowly never makes one frame look new ─────────
    def _growing(self, tmp_path, i, per_frame=0.12):
        import numpy as np
        x = np.linspace(0, 20, 600)
        y = 0.2 + sum(5.0 * np.exp(-0.5 * ((x - c) / 0.25) ** 2) for c in (5, 9, 14))
        y = y + per_frame * i * np.exp(-0.5 * ((x - 17.0) / 0.3) ** 2)      # a peak growing in at x = 17
        y = y + np.random.default_rng(100 + i).normal(0, 0.02, x.size)
        p = tmp_path / f"g{i}.csv"
        np.savetxt(p, np.column_stack([x, y]), delimiter=",", header="x,y", comments="")
        return str(p)

    def test_a_gradual_onset_is_announced_from_the_distance_to_the_reference(self, tmp_path):
        loop = self._armed(tmp_path)
        records = [loop.step(self._growing(tmp_path, i)) for i in range(1, 41)]
        assert all(r["flags"] == [] for r in records)                # no frame ever looked new
        events = [e for e in loop.read_log() if e["event"] == "novelty"]
        assert events and all(e["onset"] == "gradual" for e in events)
        first = events[0]
        assert first["fraction"] > 0.25 and first["recipe_fits"] is True
        assert first["where"][0]["kind"] == "new" and abs(first["where"][0]["x_peak"] - 17.0) < 0.3
        assert [r for r in records if r.get("novelty")][0]["novelty"]["onset"] == "gradual"
        # the same distance is not news twice: each announcement is at least double the last
        levels = [e["fraction"] for e in events]
        assert len(events) <= 3 and all(b >= 2 * a for a, b in zip(levels, levels[1:]))
        assert not [e for e in loop.read_log() if e["event"] in ("escalation_started", "state_accepted")]

    def test_the_slow_alarm_can_be_turned_off_and_does_not_repeat_an_abrupt_change(self, tmp_path):
        (tmp_path / "quiet").mkdir()
        (tmp_path / "abrupt").mkdir()
        quiet = self._armed(tmp_path / "quiet", gradual_bar=None)
        for i in range(1, 41):
            quiet.step(self._growing(tmp_path / "quiet", i))
        assert not [e for e in quiet.read_log() if e["event"] == "novelty"]
        loop = self._armed(tmp_path / "abrupt")
        for f in self._changed(tmp_path / "abrupt", n=8):            # a new peak, all at once
            loop.step(f)
        events = [e for e in loop.read_log() if e["event"] == "novelty"]
        assert len(events) == 1 and "onset" not in events[0]         # announced once, as what it was

    def test_one_odd_frame_is_a_flag_not_a_discovery(self, tmp_path):
        loop = self._armed(tmp_path)
        assert loop.step(self._changed(tmp_path)[0])["flags"] == ["drift_suspected"]
        assert loop.step(self._curve(tmp_path, "back.csv", [5, 9, 14], seed=77))["flags"] == []
        assert not [e for e in loop.read_log() if e["event"] in ("novelty", "state_accepted")]

    def test_a_change_the_recipe_fails_on_is_announced_and_rebuilt(self, tmp_path):
        loop = self._armed(tmp_path)
        bad = [self._curve(tmp_path, f"bad{i}.csv", [5, 9, 14, 17], seed=50 + i) for i in range(2)]
        FakeAgent.replies = {Path(b).name: good(6.0, r2=0.4) for b in bad}
        loop.step(bad[0])
        assert loop.step(bad[1])["escalation"] == "started"
        novelty = next(e for e in loop.read_log() if e["event"] == "novelty")
        assert novelty["recipe_fits"] is False and novelty["where"][0]["kind"] == "new"

    def test_a_stream_still_changing_is_accepted_eventually_and_says_so(self, tmp_path):
        loop = self._armed(tmp_path)
        moving = [self._curve(tmp_path, f"m{i}.csv", [5, 9, 14] + [2.0 + 1.9 * i], seed=90 + i)
                  for i in range(9)]
        for f in moving:
            loop.step(f)
        accepted = [e for e in loop.read_log() if e["event"] == "state_accepted"]
        assert accepted and accepted[0]["settled"] is False and "still changing" in accepted[0]["how"]

    def test_the_recommender_is_told(self, tmp_path):
        notes = []

        class Listening:
            clock, name = "fast", "listening"

            def observe(self, *a, **k): pass
            def suggest(self): return {"params": {}, "rationale": "hold"}
            def notify(self, event): notes.append(event)
        loop = self._armed(tmp_path)
        loop.recommender = Listening()
        frames = self._changed(tmp_path)
        loop.step(frames[0]); loop.step(frames[1])
        assert [n["event"] for n in notes] == ["novelty"]
        assert notes[0]["where"][0]["kind"] == "new" and "frames" not in notes[0]

    def test_a_frame_the_recipe_fails_on_still_gets_a_change_reading(self, tmp_path):
        loop = self._armed(tmp_path)
        bad = self._changed(tmp_path)[0]
        FakeAgent.replies = {Path(bad).name: RuntimeError("the script crashed")}
        rec = loop.step(bad)
        assert rec["flags"] == ["fit_failed"] and rec["gate"]["drift_fraction"] > 0.2

    def test_a_change_that_still_fits_gets_an_audit_and_agreement_keeps_the_recipe(self, tmp_path):
        loop = self._armed(tmp_path, on_change="audit")
        recipe = loop.recipe["id"]
        frames = self._changed(tmp_path)
        loop.step(frames[0])
        second = loop.step(frames[1])
        assert second["escalation"] == "audit_started"
        spec = FakeEscalation.last.spec
        assert spec["analyze_kwargs"]["profile"] == "quick" and spec["pin_outputs"] == self.OUT
        assert "bank_exclude" in spec["analyze_kwargs"]              # a second opinion, not an echo
        self._finish(tmp_path, 6.0)                                   # the audit agrees (locked: 6.0)
        third = loop.step(frames[2])
        [audit] = [e for e in loop.read_log() if e["event"] == "audit"]
        assert audit["agrees"] and audit["reason"] == "change"
        accepted = next(e for e in loop.read_log() if e["event"] == "state_accepted")
        assert accepted["verified"] is True and accepted["how"] == "an audit agreed"
        assert audit["outputs"]["peak_1_center"]["relative_difference"] == 0.0
        assert loop.recipe["id"] == recipe and not [e for e in loop.read_log() if e["event"] == "reanchor"]
        assert loop.step(frames[3])["flags"] == []                   # the new state is normal now
        assert third["recipe_id"] == recipe

    def test_disagreement_on_a_changed_stream_adopts_the_audits_recipe(self, tmp_path):
        loop = self._armed(tmp_path, on_change="audit")
        old = loop.recipe["id"]
        frames = self._changed(tmp_path)
        loop.step(frames[0]); loop.step(frames[1])
        self._finish(tmp_path, 6.9)                                   # locked says 6.0
        loop.step(frames[2])
        log = loop.read_log()
        audit = next(e for e in log if e["event"] == "audit")
        assert audit["agrees"] is False and audit["outputs"]["peak_1_center"]["agrees"] is False
        adopted = next(e for e in log if e["event"] == "reanchor")
        assert adopted["after_audit"] is True and loop.recipe["id"] != old

    def test_without_named_outputs_or_with_a_failing_fit_it_rebuilds(self, tmp_path):
        loop = self._armed(tmp_path, on_change="audit")
        loop.outputs = {}
        frames = self._changed(tmp_path)
        loop.step(frames[0])
        assert loop.step(frames[1])["escalation"] == "started"        # nothing to compare: re-anchor
        assert FakeEscalation.last.spec["analyze_kwargs"]["profile"] == "extract"

        (tmp_path / "b").mkdir()
        failing = self._armed(tmp_path / "b")
        bad = [self._curve(tmp_path, f"bad{i}.csv", [5, 9, 14, 17], seed=50 + i) for i in range(2)]
        FakeAgent.replies = {Path(b).name: good(6.0, r2=0.4) for b in bad}
        failing.step(bad[0])
        assert failing.step(bad[1])["escalation"] == "started"        # the recipe fails: rebuild it

    def test_an_audit_that_cannot_be_had_accepts_the_state_and_says_so(self, tmp_path):
        loop = self._armed(tmp_path, on_change="audit")
        frames = self._changed(tmp_path)
        loop.step(frames[0]); loop.step(frames[1])
        FakeEscalation.last.result = {"status": "error", "error": "no model", "seconds": 3}
        first = FakeEscalation.last
        loop.step(frames[2])
        # one retry at a deeper profile, then it gives up
        assert FakeEscalation.last is not first
        assert FakeEscalation.last.spec["analyze_kwargs"]["profile"] == "thorough"
        FakeEscalation.last.result = {"status": "error", "error": "no model", "seconds": 3}
        loop.step(frames[3])
        events = [e["event"] for e in loop.read_log()]
        assert events.count("audit_failed") == 2
        accepted = next(e for e in loop.read_log() if e["event"] == "state_accepted")
        assert accepted["verified"] is False
        more = self._curve(tmp_path, "c_more.csv", [5, 9, 14, 17], seed=77)
        assert loop.step(more)["flags"] == []                        # and does not ask again

    def test_periodic_audits_report_and_never_act(self, tmp_path):
        loop = self._armed(tmp_path, audit_every=3)
        started = [e for e in loop.read_log() if e["event"] == "audit_started"]
        assert [e["step"] for e in started] == [3] and started[0]["reason"] == "periodic"
        recipe = loop.recipe["id"]
        self._finish(tmp_path, 7.5)                                   # disagrees with the locked 6.0
        loop.step(self._curve(tmp_path, "n9.csv", [5, 9, 14], seed=99))
        assert loop.recipe["id"] == recipe                           # reported, not acted on
        last = loop.status()["last_audit"]
        assert last["agrees"] is False and last["outputs"]["peak_1_center"]["audit"] == 7.5
        again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=FakeAgent)
        assert again.status()["last_audit"]["agrees"] is False and again.audit_every == 3

    def test_a_breach_run_takes_the_background_from_a_periodic_audit(self, tmp_path):
        loop = self._armed(tmp_path, audit_every=3)
        assert loop.escalating                                        # the periodic audit from step 3
        bad = [self._curve(tmp_path, f"bad{i}.csv", [5, 9, 14], seed=70 + i) for i in range(2)]
        FakeAgent.replies = {Path(b).name: good(6.0, r2=0.4) for b in bad}
        loop.step(bad[0])
        assert loop.step(bad[1])["escalation"] == "started"
        events = [e["event"] for e in loop.read_log()]
        assert "escalation_cancelled" in events and events[-1] == "escalation_started"

    def test_values_within_their_uncertainties_agree(self, tmp_path):
        loop = self._armed(tmp_path)
        c = loop._compare({"peak_1_center": 6.00, "peak_1_center_err": 0.2},
                          {"peak_1_center": 6.45, "peak_1_center_err": 0.2})
        assert c["peak_1_center"]["agrees"]                           # 7 % apart, under 3 combined sigma
        c = loop._compare({"peak_1_center": 6.00}, {"peak_1_center": 6.45})
        assert not c["peak_1_center"]["agrees"]
        assert loop._compare({}, {"peak_1_center": 6.0})["peak_1_center"]["why"] == "missing"

    def test_two_analyses_need_not_agree_better_than_one_agrees_with_itself(self, tmp_path):
        # Seen live on real EELS frames: a width 5.2 % apart counted as a
        # disagreement and swapped the recipe, while the locked recipe's own
        # width moved more than that from one frame to the next.
        loop = self._armed(tmp_path)
        loop._recent_features = [{"peak_1_center": v} for v in (6.0, 6.3, 5.8, 6.2, 5.9, 6.25, 5.85)]
        c = loop._compare({"peak_1_center": 6.0}, {"peak_1_center": 6.45})["peak_1_center"]
        assert c["agrees"] and c["frame_scatter"] > 0.15
        loop._recent_features = [{"peak_1_center": 6.0 + 0.001 * i} for i in range(8)]
        assert not loop._compare({"peak_1_center": 6.0}, {"peak_1_center": 6.45})["peak_1_center"]["agrees"]



# ──────────────────────────────────────────────────────────────
# review of PR 656: shutdown, and metadata that lives next to the data
# ──────────────────────────────────────────────────────────────

class TestClose:
    class Running(FakeEscalation):
        terminated = 0

        def terminate(self):
            type(self).terminated += 1

        def wait(self, timeout=None):
            self.result = {"status": "success", "seconds": 1, "llm_calls": 1,
                           "output_directory": self.anchor}

    def _loop(self, tmp_path):
        loop = loop_at(tmp_path, escalation_runner=self.Running)
        loop.setup(anchor=str(make_anchor(tmp_path)))
        loop.step("a.csv")
        loop.escalate("a.csv")
        self.Running.terminated = 0
        return loop

    def test_a_running_reanchor_is_stopped_not_orphaned(self, tmp_path):
        loop = self._loop(tmp_path)
        assert loop.close() is None and self.Running.terminated == 1 and not loop.escalating
        [event] = [e for e in loop.read_log() if e["event"] == "escalation_cancelled"]
        assert "closed before" in event["why"]
        assert loop.close() is None                              # idempotent

    def test_or_waited_for_and_adopted(self, tmp_path):
        loop = self._loop(tmp_path)
        FakeEscalation.last.anchor = str(new_anchor(tmp_path))
        adopted = loop.close(wait=True)
        assert adopted["event"] == "reanchor" and self.Running.terminated == 0

    def test_a_context_manager_closes(self, tmp_path):
        with self._loop(tmp_path) as loop:
            assert loop.escalating
        assert self.Running.terminated == 1


def test_sidecar_metadata_reaches_the_analysis_and_truth_never_does(tmp_path):
    # Review: a technique needing an instrument constant aborted at setup; the
    # value was in the file's sidecar, which the loop never read.
    ref = tmp_path / "ref.csv"
    ref.write_text("x,y\n0,1\n")
    ref.with_suffix(".json").write_text(json.dumps({
        "meta": {"microwave_frequency_GHz": 9.41, "technique": "from the sidecar"},
        "params": {"power": 2}, "truth": {"g_factor": 2.0023}}))
    native = tmp_path / "frame.csv"
    native.write_text("x,y\n0,1\n")
    native.with_suffix(".json").write_text(json.dumps({
        "temperature_K": 120, "truth": {"x": 1},
        # a small description of scalars travels (a datacube's spectral axis does this) ...
        "energy_range": {"start": 0.2, "end": 1.13, "units": "eV"},
        # ... arrays and bulk do not
        "spectrum": [1, 2, 3], "table": {"rows": [[1, 2], [3, 4]]},
        "big": {f"k{i}": i for i in range(40)}}))
    anchor = make_anchor(tmp_path)

    class RefAgent(FakeAgent):
        def analyze(self, data, **kw):
            FakeAgent.calls.append({"data": data, **kw})
            return {**good(6.0), "output_directory": str(anchor) if "ref" in str(data) else self.output_dir}
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=RefAgent, check_portability=False,
                           system_info={"technique": "EPR"})
    loop.setup(reference=str(ref))
    loop.step(str(native))
    setup_info, frame_info = FakeAgent.calls[0]["system_info"], FakeAgent.calls[-1]["system_info"]
    assert setup_info == {"microwave_frequency_GHz": 9.41, "technique": "EPR"}     # the caller's word wins
    assert frame_info == {"temperature_K": 120, "technique": "EPR",
                          "energy_range": {"start": 0.2, "end": 1.13, "units": "eV"}}
    assert "truth" not in json.dumps([setup_info, frame_info])


def test_targets_do_not_override_a_skills_mandate():
    from scilink.agents.exp_agents._qc_profile import planning_addendum
    text = planning_addendum({"analysis_targets": ["G band position"]})
    assert "the skill's rule stands" in text and "not what is fitted" in text


def test_the_worker_process_is_really_terminated(tmp_path):
    import subprocess
    import sys
    esc = ml._ProcessEscalation.__new__(ml._ProcessEscalation)
    esc.out_dir = tmp_path
    esc._log = open(tmp_path / "worker.out", "w")
    esc._proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    assert esc._proc.poll() is None
    esc.terminate()
    assert esc._proc.poll() is not None and esc._log.closed
    esc.terminate()                                               # idempotent
