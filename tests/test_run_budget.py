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


class _RejectingHost(_Host):
    """A verifier that never approves."""
    max_verification_iterations = 2

    def qc_check_accept(self, ctx, verification):
        return False

    def qc_refine(self, ctx, verification):
        self.calls.append("refine")
        return {}

    def qc_refit(self, ctx, verification, refine_from, just_escalated_to_hot):
        self.calls.append("refit")
        return {"success": True, "fit_quality": {"r_squared": 0.97}, "script": "s2"}

    def qc_after_refit(self, ctx, refit_result, verification):
        ctx.current_result = refit_result


class TestCappedLoopUnderAReducedProfile:
    """Observed live (an `extract` Raman fit): after verification 2/2 rejected,
    the loop refined and refitted a third time, which then cost a final verify
    and a judge — four LLM calls, ~170 s — only for the best fit to be accepted
    on the deterministic gate anyway."""

    def test_the_last_rejection_does_not_start_another_refit(self):
        host = _RejectingHost()
        ctx = _ctx({"_verification_mode": "purpose"})
        CodegenQCEngine(host, SPEC).run_item(ctx)
        # two verifies, ONE refine (after the first), no final verify
        assert host.calls.count("verify") == 2
        assert host.calls.count("refine") == 1 and host.calls.count("refit") == 1
        assert "final_verify" not in host.calls
        assert ctx.capped is True

    def test_strict_verification_keeps_todays_loop(self):
        host = _RejectingHost()
        ctx = _ctx({"_verification_mode": "strict"})
        CodegenQCEngine(host, SPEC).run_item(ctx)
        assert host.calls.count("refine") == 2 and host.calls.count("refit") == 2
        assert "final_verify" in host.calls and ctx.capped is False


# ── a first fit that produced nothing gets one re-plan, not an `error` ──────
# Review of #656: `extract` on a hard single spectrum exhausted its passes and
# returned status=error where the default profile converges. Every later stage
# returns the best attempt so far; a first fit whose every script pass failed
# has none, and nothing recovered it (adaptive refit is a series stage, reduced
# depth has no best-of-N).

class _FailingHost(_Host):
    def __init__(self, recovered=None):
        super().__init__()
        self.recovered = recovered

    def qc_run_initial(self, ctx):
        return {"success": False, "error": "Optimal parameters not found", "script": "s"}

    def qc_record_initial_failure(self, ctx, result):
        self.calls.append("failed")

    def qc_recover_initial_failure(self, ctx, result):
        self.calls.append("recover")
        return self.recovered


class TestAFirstFitThatProducedNothing:
    def test_a_recovered_fit_re_enters_the_normal_flow(self):
        host = _FailingHost({"success": True, "fit_quality": {"r_squared": 0.97}, "script": "s2"})
        out = CodegenQCEngine(host, SPEC).run_item(_ctx({}))
        assert host.calls == ["failed", "recover", "verify", "post"]
        assert out["script"] == "s2" and out["approved"] is True

    def test_a_failed_recovery_is_todays_failure(self):
        for recovered in (None, {"success": False, "error": "still"}):
            host = _FailingHost(recovered)
            out = CodegenQCEngine(host, SPEC).run_item(_ctx({}))
            assert host.calls == ["failed", "recover", "fallback"] and out == {"success": False}

    def test_a_host_without_the_hook_is_unchanged(self):
        class Plain(_Host):
            def qc_run_initial(self, ctx): return {"success": False}
            def qc_record_initial_failure(self, ctx, result): self.calls.append("failed")
        host = Plain()
        CodegenQCEngine(host, SPEC).run_item(_ctx({}))
        assert host.calls == ["failed", "fallback"]


class TestTheCurveHostsRecovery:
    FAILED = {"success": False, "script": "s", "error": "Traceback...\nRuntimeError: Optimal parameters not found",
              "script_errors": [{"error": "x\nValueError: `x0` is infeasible"}]}

    def _host(self, fit_ok=True):
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController)
        host = object.__new__(UnifiedSeriesProcessingController)
        host.logger = logging.getLogger("t")
        host.seen = []

        def replan(state):
            host.seen.append(dict(state["_failed_plan"]))
            state["physical_model"] = "a simpler model"
        host.replanner = SimpleNamespace(replan_headless=replan)
        host._fit_single_spectrum = lambda **kw: (
            {"success": True, "fit_quality": {"r_squared": 0.98}, "script": "s2"} if fit_ok
            else {"success": False, "error": "again", "script": "s2"})
        return host

    def _state(self, **kw):
        return {"_recover_failed_fit": True, "physical_model": "edge + locked linear background",
                "fitting_strategy": "power-law", **kw}

    def test_one_replan_told_what_failed_and_how(self):
        host, state = self._host(), self._state()
        ctx = _ctx(state)
        out = host.qc_recover_initial_failure(ctx, self.FAILED)
        assert out["success"] and out["recovered_from"]["failed_model"].startswith("edge")
        [told] = host.seen
        assert told["physical_model"].startswith("edge")
        assert told["errors"] == ["ValueError: `x0` is infeasible",
                                  "RuntimeError: Optimal parameters not found"]
        assert "_failed_plan" not in state and ctx.initial_label == "a simpler model"
        assert state["_recovered_from"] == out["recovered_from"]            # survives a later refit
        assert host.qc_recover_initial_failure(ctx, self.FAILED) is None        # once per run

    def test_the_planner_reads_it_as_a_principle(self):
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _failed_plan_block
        block = _failed_plan_block({"physical_model": "M", "fitting_strategy": "S", "errors": ["E1"]})
        assert "M" in block and "- E1" in block and "this data can actually support" in block

    def test_a_recovery_that_fails_too_is_returned_as_the_failure(self):
        out = self._host(fit_ok=False).qc_recover_initial_failure(_ctx(self._state()), self.FAILED)
        assert out["success"] is False and "recovered_from" not in out

    @pytest.mark.parametrize("state, result", [
        ({"_recover_failed_fit": False}, {}),                       # realtime frame / strict replay
        ({"_candidate_subdir": "_candidates/cand_01"}, {}),         # its siblings are the recovery
        ({}, {"kind": "timeout"}),                                  # a new plan is not a faster machine
        ({}, {"script": None}),                                     # pre-flight refusal: nothing was tried
        ({"_run_deadline": 0.0}, {}),                               # the run's budget is spent
    ])
    def test_where_it_does_not_apply(self, state, result):
        host = self._host()
        assert host.qc_recover_initial_failure(
            _ctx(self._state(**state)), {**self.FAILED, **result}) is None
        assert host.seen == []

    def test_not_for_a_series_unit_or_without_a_planner(self):
        host = self._host()
        ctx = _ctx(self._state())
        ctx.is_anchor = False
        assert host.qc_recover_initial_failure(ctx, self.FAILED) is None
        host.replanner = None
        assert host.qc_recover_initial_failure(_ctx(self._state()), self.FAILED) is None

    def test_it_is_a_profile_field_and_realtime_turns_it_off(self):
        from scilink.agents.exp_agents._qc_profile import resolve_profile
        assert [resolve_profile(p).recover_failed_fit for p in ("thorough", "quick", "extract", "realtime")] \
            == [True, True, True, False]
