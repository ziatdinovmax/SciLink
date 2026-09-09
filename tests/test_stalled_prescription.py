"""#568 — the image verification loop pivots off a stalled prescription: a
fix the verifier keeps re-issuing that never moves the score is named to
the refiner and the verifier as stalled (do not re-issue), stamped on the
result, and if it comes back once more the loop stops with the best
result so far and a legible reason — instead of burning the whole budget
repeating it."""
import logging
from types import SimpleNamespace

from scilink.agents.exp_agents._qc_engine import CodegenQCEngine, QCEngineSpec, QCItemContext
from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    UnifiedImageProcessingController as Host,
)


def _host():
    h = Host.__new__(Host)
    h.logger = logging.getLogger("test.stalled")
    return h


def _ctx(actions_scores, state=None):
    """A QC context whose verification history carries the given
    (recommended_action, quality_score) pairs, oldest first."""
    c = SimpleNamespace(state=state if state is not None else {},
                        verification_history=[{"recommended_action": a, "quality_score": s}
                                              for a, s in actions_scores],
                        best_score=max((s for _, s in actions_scores), default=0.0),
                        best_result={"success": True}, stop_reason=None)
    return c


SAM = "Invoke the SAM fallback (run_sam_analysis) and substitute its masks for the watershed output."
SAM2 = "Substitute SAM masks: call run_sam_analysis as the fallback and replace the watershed segmentation."
OTHER = "Increase the Gaussian blur sigma before thresholding to merge fragmented particles."


def test_same_prescription_is_token_overlap_not_exact_text():
    assert Host._same_prescription(SAM, SAM2)
    assert not Host._same_prescription(SAM, OTHER)
    assert not Host._same_prescription("", SAM) and not Host._same_prescription("none", "none")


def test_detects_three_repeats_without_improvement_only():
    h = _host()
    # two repeats: not yet
    assert h._detect_stalled_prescription(_ctx([(SAM, 0.5), (SAM2, 0.5)]), {"recommended_action": SAM}) is None
    # three repeats, flat score: stalled
    st = h._detect_stalled_prescription(_ctx([(OTHER, 0.4), (SAM, 0.5), (SAM2, 0.51), (SAM, 0.5)]),
                                        {"recommended_action": SAM})
    assert st and st["times"] == 3 and st["score_at_first"] == 0.5 and st["best_since"] == 0.51
    # three repeats but the score climbed: the fix is landing, not stalled
    assert h._detect_stalled_prescription(_ctx([(SAM, 0.4), (SAM2, 0.5), (SAM, 0.6)]),
                                          {"recommended_action": SAM}) is None
    # a different prescription in between breaks the run
    assert h._detect_stalled_prescription(_ctx([(SAM, 0.5), (OTHER, 0.5), (SAM, 0.5)]),
                                          {"recommended_action": SAM}) is None
    # "none" is never a prescription
    assert h._detect_stalled_prescription(_ctx([("none", 0.5)] * 4), {"recommended_action": "none"}) is None


def test_first_stall_pivots_then_a_repeat_stops_the_loop(caplog):
    h = _host()
    ctx = _ctx([(SAM, 0.5), (SAM2, 0.5), (SAM, 0.5)])
    st = h._detect_stalled_prescription(ctx, {"recommended_action": SAM})
    with caplog.at_level(logging.WARNING, logger="test.stalled"):
        h._note_stalled_prescription(ctx, st)
    assert ctx.stop_reason is None
    assert ctx.state["_stalled_prescriptions"][0]["prescription"] == SAM
    assert "did not take after 3 tries" in caplog.text and "pivoting" in caplog.text
    # the prompts now carry the block
    block = Host._stalled_prescriptions_block(ctx.state["_stalled_prescriptions"])
    assert "STALLED PRESCRIPTIONS" in block and "prescribed 3 times" in block
    # re-issued once more after the pivot → stop with a reason, no new entry
    ctx.verification_history.append({"recommended_action": SAM2, "quality_score": 0.5})
    st = h._detect_stalled_prescription(ctx, {"recommended_action": SAM2})
    h._note_stalled_prescription(ctx, st)
    assert len(ctx.state["_stalled_prescriptions"]) == 1
    assert ctx.state["_stalled_prescriptions"][0]["times"] == 4
    assert "did not take after 4 tries" in ctx.stop_reason
    # and the result is stamped for legibility
    h._stamp_stalled(ctx)
    assert ctx.best_result["stalled_prescriptions"][0]["times"] == 4


def test_refinement_prompt_names_the_stalled_fix():
    h = _host()
    prompts = []
    h.model = SimpleNamespace(generate_content=lambda **kw: prompts.append(kw["contents"][0]) or SimpleNamespace(text="{}"))
    h.generation_config = None
    h.safety_settings = None
    h._parse = lambda r: ({}, "no json")
    state = {"locked_analysis_config": {"processing_pipeline": "watershed", "analysis_approach": "x"},
             "_annealing_level": 0}
    stalled = [{"prescription": SAM, "times": 3, "score_at_first": 0.5, "best_since": 0.5}]
    h._apply_verification_feedback(state, {"recommended_action": SAM, "issues_found": []},
                                   history=[], stalled=stalled)
    assert prompts and "STALLED PRESCRIPTIONS — DO NOT RE-ISSUE" in prompts[0]
    assert "materially different method path" in prompts[0]


class _LoopHost:
    """Minimal engine host: every verification prescribes the same fix; the
    refine hook stops the loop on its second call the way the image host
    does after a pivot is ignored."""
    _CONSTRAINT_ANNEALING_SCHEDULE = ("T0", "T1", "T2")
    max_verification_iterations = 7

    def __init__(self):
        self.logger = logging.getLogger("test.loophost")
        self.calls = []

    def qc_loop_setup(self, ctx):
        pass

    def qc_verify(self, ctx):
        self.calls.append("verify")
        return {"quality_score": 0.5, "recommended_action": SAM}

    def qc_assess(self, ctx, v):
        ctx.verification_history.append({"quality_score": 0.5, "recommended_action": SAM})

    def qc_check_accept(self, ctx, v):
        return False

    def qc_refine(self, ctx, v):
        self.calls.append("refine")
        if self.calls.count("refine") == 2:
            ctx.stop_reason = "prescribed fix did not take"
        return {"pipeline": f"p{len(self.calls)}"}

    def qc_refit(self, ctx, v, refine_from, hot):
        self.calls.append("refit")
        return {"success": True}

    def qc_after_refit(self, ctx, r, v):
        self.calls.append("after_refit")

    def qc_final_verify(self, ctx):
        self.calls.append("final_verify")


def test_engine_stops_on_the_host_stop_reason():
    host = _LoopHost()
    engine = CodegenQCEngine(host, QCEngineSpec(config_key=None, refine_anchor="none", refit_fail_msg="refit failed"))
    ctx = QCItemContext(state={}, data=None, data_path="x", item_name="i", item_idx=0,
                        is_regime_anchor=True)
    ctx.n_levels = 3
    ctx.best_result = {"success": True}
    ctx.best_score = 0.5
    engine._verification_loop(ctx)
    # verify/refine twice; the second refine set stop_reason → final verify, no more refits
    assert host.calls == ["verify", "refine", "refit", "after_refit", "verify", "refine", "final_verify"]
