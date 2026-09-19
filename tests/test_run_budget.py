"""The run-level soft budget: always return something within the time asked.

``qc_time_budget_s`` bounded only the verification loop, was set only by the
hyperspectral host, and on expiry still spent an LLM call on a final verify.
A run budget (``QCProfile.time_budget_s``) is broader and stricter about what
it spends once it has run out:

  - between pipeline stages, the optional ones (literature, adaptive refit,
    trend, synthesis) are skipped once the deadline has passed — the stages
    that produce and persist the result always run;
  - inside the QC loop, expiry returns the best result so far WITHOUT the
    final LLM verify / judge, flagged ``unverified`` — never silently as if it
    had been approved.

It is a soft budget: an in-flight call or script is not interrupted.

No LLM calls anywhere.
"""

import logging
import time
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents._qc_engine import (
    CodegenQCEngine, QCEngineSpec, QCItemContext)
from scilink.agents.exp_agents._stage_timing import RunBudget, StageTimer


class TestRunBudget:
    def test_unbounded_never_expires(self):
        b = RunBudget(None)
        assert b.seconds is None and not b.expired and b.remaining is None
        assert b.deadline is None

    def test_expiry_and_remaining(self):
        b = RunBudget(0.05)
        assert not b.expired and 0 < b.remaining <= 0.05
        time.sleep(0.06)
        assert b.expired and b.remaining == 0.0

    def test_from_state_round_trip(self):
        b = RunBudget(30)
        state = {}
        b.stamp(state)
        again = RunBudget.from_state(state)
        assert again.deadline == b.deadline and not again.expired
        assert RunBudget.from_state({}).seconds is None


class _Stage:
    def __init__(self, log):
        self.log = log

    def execute(self, state):
        self.log.append(self.__class__.__name__)
        return state


class UnifiedSeriesProcessingController(_Stage): pass
class AdaptiveRefitController(_Stage): pass
class ConditionalTrendAnalysisController(_Stage): pass
class UnifiedCurveSynthesisController(_Stage): pass
class LiteratureSearchController(_Stage): pass
class StoreAnalysisResultsController(_Stage): pass
class GenerateCurveFittingReportController(_Stage): pass


class TestDeferrableStages:
    def _run(self, budget):
        log = []
        timer = StageTimer()
        state = {}
        budget.stamp(state)
        for cls in (LiteratureSearchController, UnifiedSeriesProcessingController,
                    AdaptiveRefitController, ConditionalTrendAnalysisController,
                    UnifiedCurveSynthesisController, StoreAnalysisResultsController,
                    GenerateCurveFittingReportController):
            state = timer.run_within_budget(cls(log), state, budget,
                                            logger=logging.getLogger("t"))
        return log, state, timer

    def test_everything_runs_inside_the_budget(self):
        log, state, _ = self._run(RunBudget(60))
        assert len(log) == 7 and not state.get("_budget_skipped")

    def test_optional_stages_are_skipped_after_the_deadline(self):
        b = RunBudget(0.01)
        time.sleep(0.02)
        log, state, timer = self._run(b)
        # What produces and persists the result always runs.
        assert log == ["UnifiedSeriesProcessingController",
                       "StoreAnalysisResultsController",
                       "GenerateCurveFittingReportController"]
        assert state["_budget_skipped"] == [
            "LiteratureSearchController", "AdaptiveRefitController",
            "ConditionalTrendAnalysisController", "UnifiedCurveSynthesisController"]
        skipped = [r for r in timer.records if r["status"] == "skipped_budget"]
        assert len(skipped) == 4 and all(r["seconds"] == 0 for r in skipped)

    def test_tier2_and_image_names_are_deferrable_too(self):
        from scilink.agents.exp_agents._stage_timing import is_deferrable
        for name in ("ImageAdaptiveRefitController", "ConditionalImageTrendController",
                     "UnifiedImageSynthesisController", "tier2:UnifiedImageProcessingController",
                     "Tier2Evaluation", "synthesis:RunSelfReflectionController"):
            assert is_deferrable(name), name
        for name in ("UnifiedImageProcessingController", "AnalyzeImageController",
                     "StoreAnalysisResultsController", "GenerateImageReportController",
                     "synthesis:GenerateHTMLReportController",
                     "CurveFittingPlanningController", "RunDynamicAnalysisController"):
            assert not is_deferrable(name), name


# ──────────────────────────────────────────────────────────────
# Inside the QC loop
# ──────────────────────────────────────────────────────────────

class _Host:
    _CONSTRAINT_ANNEALING_SCHEDULE = [0, 1, 2]
    max_verification_iterations = 7

    def __init__(self):
        self.logger = logging.getLogger("t")
        self.calls = []

    def qc_setup(self, ctx): pass
    def qc_try_reuse(self, ctx): return None
    def qc_verification_bypass(self, ctx): return False
    def qc_loop_setup(self, ctx): pass

    def qc_run_initial(self, ctx):
        return {"success": True, "fit_quality": {"r_squared": 0.97}, "script": "s"}

    def qc_record_initial(self, ctx, result):
        ctx.best_result, ctx.best_score = result, 0.97

    def qc_verify(self, ctx):
        self.calls.append("verify")
        return {"fit_acceptable": True}

    def qc_assess(self, ctx, verification): pass
    def qc_check_accept(self, ctx, verification): return True

    def qc_final_verify(self, ctx):
        self.calls.append("final_verify")

    def qc_post_verification(self, ctx):
        self.calls.append("post")
        return dict(ctx.best_result, approved=ctx.approved,
                    budget_expired=getattr(ctx, "budget_expired", False))

    def qc_fallback(self, ctx):
        self.calls.append("fallback")
        return {"success": False}


def _ctx(state):
    return QCItemContext(state=state, data=None, data_path="x", item_name="a", item_idx=0)


SPEC = QCEngineSpec(config_key=None, refine_anchor="none", refit_fail_msg="refit failed")


class TestEngineHonoursTheRunDeadline:
    def test_inside_the_budget_the_loop_is_unchanged(self):
        host, state = _Host(), {}
        RunBudget(60).stamp(state)
        out = CodegenQCEngine(host, SPEC).run_item(_ctx(state))
        assert host.calls == ["verify", "post"] and out["approved"] is True

    def test_no_deadline_is_todays_behavior(self):
        host = _Host()
        out = CodegenQCEngine(host, SPEC).run_item(_ctx({}))
        assert host.calls == ["verify", "post"] and out["budget_expired"] is False

    def test_expired_run_budget_returns_best_without_spending_a_call(self):
        host, state = _Host(), {}
        b = RunBudget(0.01)
        b.stamp(state)
        time.sleep(0.02)
        out = CodegenQCEngine(host, SPEC).run_item(_ctx(state))
        # No verify, and — unlike the loop's own budget — no final verify.
        assert host.calls == ["post"]
        assert out["budget_expired"] is True and out["approved"] is False

    def test_the_hosts_own_loop_budget_still_final_verifies(self):
        # Hyperspectral's pre-existing qc_time_budget_s path, byte-identical.
        host = _Host()
        host.qc_time_budget_s = 1e-9
        time.sleep(0.001)
        CodegenQCEngine(host, SPEC).run_item(_ctx({}))
        assert host.calls == ["final_verify", "post"]


class TestProfileCarriesTheBudget:
    def test_budget_is_a_profile_field(self):
        from scilink.agents.exp_agents._qc_profile import resolve_profile
        p = resolve_profile({"base": "quick", "time_budget_s": 120})
        assert p.time_budget_s == 120 and p.name == "quick"
        assert RunBudget(p.time_budget_s).seconds == 120
