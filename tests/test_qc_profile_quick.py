"""Depth is a property of the request: the ``quick`` / ``extract`` profiles.

``QCProfile`` used to name only the QC-loop toggles, and nothing but the
literal name "realtime" was ever consumed. These tests pin the two halves of
step 3a:

  - the profile owns the stages AROUND the QC loop too (plan validation,
    adaptive refit, trend, synthesis, image tier 2), with two new presets
    between ``thorough`` and ``realtime``;
  - the curve pipeline builder honours those fields — a stage a profile turns
    off is not in the pipeline at all, so it cannot cost a call.

No LLM calls anywhere.
"""

import logging

import pytest

from scilink.agents.exp_agents._qc_profile import (
    EXTRACT, QUICK, REALTIME, THOROUGH, QCProfile, resolve_profile)
from scilink.agents.exp_agents.pipelines.curve_fitting_pipelines import (
    create_unified_curve_fitting_pipeline)


class TestPresets:
    def test_thorough_is_todays_behavior(self):
        assert THOROUGH.max_verification_iterations == 7
        assert THOROUGH.plan_validation and THOROUGH.adaptive_refit
        assert THOROUGH.trend and THOROUGH.tier2
        assert THOROUGH.synthesis == "full"
        assert THOROUGH.time_budget_s is None

    def test_quick_cuts_llm_judgement_but_keeps_a_readable_result(self):
        assert QUICK.name == "quick"
        assert 0 < QUICK.max_verification_iterations < THOROUGH.max_verification_iterations
        assert not QUICK.check_plan_conformance and not QUICK.plan_validation
        assert not QUICK.literature and not QUICK.best_of_n_eligible
        assert not QUICK.adaptive_refit and not QUICK.trend and not QUICK.tier2
        assert QUICK.synthesis == "light"
        # Human gates belong to the autonomy mode, not to depth.
        assert QUICK.human_feedback is True

    def test_extract_is_quick_without_any_narrative(self):
        assert EXTRACT.name == "extract"
        assert EXTRACT.synthesis == "none"
        assert EXTRACT.with_overrides(name="quick", synthesis="light") == QUICK

    def test_realtime_skips_every_downstream_stage(self):
        assert not REALTIME.adaptive_refit and not REALTIME.trend
        assert REALTIME.synthesis == "none" and not REALTIME.plan_validation

    def test_resolve(self):
        assert resolve_profile(None) is THOROUGH
        assert resolve_profile(" Quick ") is QUICK
        assert resolve_profile("extract") is EXTRACT
        assert resolve_profile(QUICK) is QUICK
        with pytest.raises(ValueError, match="extract.*quick.*realtime.*thorough"):
            resolve_profile("fast")

    def test_resolve_accepts_overrides_on_a_base(self):
        p = resolve_profile({"base": "quick", "synthesis": "none", "trend": True})
        assert p.name == "quick" and p.synthesis == "none" and p.trend is True
        assert p.adaptive_refit is False          # the rest of the preset holds
        with pytest.raises(ValueError, match="unknown"):
            resolve_profile({"base": "quick", "no_such_field": 1})

    def test_validation(self):
        with pytest.raises(ValueError, match="synthesis"):
            QCProfile(name="x", synthesis="medium")
        with pytest.raises(ValueError, match="time_budget_s"):
            QCProfile(name="x", time_budget_s=0)


def _pipeline(profile, **kw):
    return create_unified_curve_fitting_pipeline(
        **kw,
        model=None, logger=logging.getLogger("t"), generation_config=None,
        safety_settings=None, parse_fn=lambda *a, **k: None,
        store_fn=lambda *a, **k: None, plot_fn=lambda *a, **k: None,
        executor=None, output_dir="out", load_skills_fn=lambda *a, **k: None,
        profile=profile)


def _names(pipeline):
    return [c.__class__.__name__ for c in pipeline]


class TestCurvePipelineHonoursTheProfile:
    def test_thorough_pipeline_is_unchanged(self):
        assert _names(_pipeline(None)) == _names(_pipeline("thorough")) == [
            "AnalyzeDataController", "SeriesScoutController",
            "CurveFittingSkillSuggestionController", "CurveFittingPlanningController",
            "LiteratureSearchController", "UnifiedSeriesProcessingController",
            "AdaptiveRefitController", "ConditionalTrendAnalysisController",
            "UnifiedCurveSynthesisController", "StoreAnalysisResultsController",
            "GenerateCurveFittingReportController", "UnifiedCurveReportController"]

    def test_quick_drops_literature_refit_and_trend(self):
        names = _names(_pipeline("quick"))
        for gone in ("LiteratureSearchController", "AdaptiveRefitController",
                     "ConditionalTrendAnalysisController"):
            assert gone not in names
        for kept in ("CurveFittingSkillSuggestionController",
                     "CurveFittingPlanningController",
                     "UnifiedSeriesProcessingController",
                     "UnifiedCurveSynthesisController"):
            assert kept in names

    def test_extract_also_drops_synthesis(self):
        names = _names(_pipeline("extract"))
        assert "UnifiedCurveSynthesisController" not in names
        assert "UnifiedSeriesProcessingController" in names

    def test_quick_turns_off_the_llm_checks_inside_the_kept_stages(self):
        by_name = {c.__class__.__name__: c for c in _pipeline("quick")}
        proc = by_name["UnifiedSeriesProcessingController"]
        assert proc.conformance_instructions is None
        assert proc.max_verification_iterations == QUICK.max_verification_iterations
        assert by_name["CurveFittingPlanningController"].validate_plan is False
        thorough = {c.__class__.__name__: c for c in _pipeline("thorough")}
        assert thorough["UnifiedSeriesProcessingController"].conformance_instructions
        assert thorough["CurveFittingPlanningController"].validate_plan is True

    def test_an_explicit_iteration_budget_beats_the_preset(self):
        p = create_unified_curve_fitting_pipeline(
            model=None, logger=logging.getLogger("t"), generation_config=None,
            safety_settings=None, parse_fn=None, store_fn=None, plot_fn=None,
            executor=None, output_dir="out", profile="quick",
            max_verification_iterations=5, explicit_verification_budget=True)
        proc = next(c for c in p if c.__class__.__name__ == "UnifiedSeriesProcessingController")
        assert proc.max_verification_iterations == 5

    def test_overridden_profile_object_is_honoured(self):
        names = _names(_pipeline(QUICK.with_overrides(trend=True)))
        assert "ConditionalTrendAnalysisController" in names
        assert "AdaptiveRefitController" not in names

    def test_realtime_pipeline_is_unchanged(self):
        assert _names(_pipeline("realtime")) == [
            "AnalyzeDataController", "UnifiedSeriesProcessingController",
            "StoreAnalysisResultsController", "GenerateCurveFittingReportController",
            "UnifiedCurveReportController"]

    def test_a_live_frame_writes_no_html_report(self):
        # Observed in the web UI: fifty per-frame reports listed under one chat turn.
        assert _names(_pipeline("realtime", write_reports=False)) == [
            "AnalyzeDataController", "UnifiedSeriesProcessingController",
            "StoreAnalysisResultsController"]


# ──────────────────────────────────────────────────────────────
# Image and hyperspectral honour the same fields
# ──────────────────────────────────────────────────────────────

def _image_pipeline(profile, **kw):
    from scilink.agents.exp_agents.pipelines.image_analysis_pipelines import (
        create_unified_image_analysis_pipeline)
    return create_unified_image_analysis_pipeline(
        model=None, logger=logging.getLogger("t"), generation_config=None,
        safety_settings=None, parse_fn=None, store_fn=None,
        image_to_bytes_fn=None, montage_fn=None, executor=None,
        output_dir="out", load_skills_fn=lambda *a, **k: None,
        profile=profile, **kw)


class TestImagePipelineHonoursTheProfile:
    def test_thorough_is_unchanged(self):
        assert _names(_image_pipeline(None)) == _names(_image_pipeline("thorough"))
        assert len(_image_pipeline(None)) == 11

    def test_quick_and_extract(self):
        quick = _names(_image_pipeline("quick"))
        for gone in ("LiteratureSearchController", "ImageAdaptiveRefitController",
                     "ConditionalImageTrendController"):
            assert gone not in quick
        assert "UnifiedImageSynthesisController" in quick
        assert "UnifiedImageSynthesisController" not in _names(_image_pipeline("extract"))

    def test_quick_turns_off_the_llm_checks_inside_the_kept_stages(self):
        by_name = {c.__class__.__name__: c
                   for c in _image_pipeline("quick", num_plan_candidates=3)}
        proc = by_name["UnifiedImageProcessingController"]
        assert not proc.conformance_instructions
        assert proc.max_verification_iterations == QUICK.max_verification_iterations
        planner = by_name["ImagePlanningController"]
        assert planner.validate_plan is False and planner.num_plan_candidates == 1


class TestHyperspectralSynthesisLevels:
    def _agent(self, profile=None, light=False):
        from types import SimpleNamespace
        from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
            HyperspectralAnalysisAgent as HS)
        names = ["BuildHolisticSynthesisPromptController", "RunFinalInterpretationController",
                 "RunSelfReflectionController", "ApplyReflectionUpdatesController",
                 "GenerateHTMLReportController", "StoreAnalysisResultsController"]
        a = HS.__new__(HS)
        a.synthesis_pipeline = [type(n, (), {})() for n in names]
        if profile is not None:
            a._qc_profile = resolve_profile(profile)
        a._light_synthesis = light
        return a

    def _kept(self, a):
        return [c.__class__.__name__ for c in a._synthesis_controllers()]

    def test_full_by_default(self):
        assert len(self._kept(self._agent())) == 6

    def test_light_drops_the_critic_and_editor(self):
        for a in (self._agent("quick"), self._agent(light=True)):
            kept = self._kept(a)
            assert "RunSelfReflectionController" not in kept
            assert "ApplyReflectionUpdatesController" not in kept
            assert "RunFinalInterpretationController" in kept

    def test_none_keeps_only_report_and_store(self):
        assert self._kept(self._agent("extract")) == [
            "GenerateHTMLReportController", "StoreAnalysisResultsController"]

    def test_profile_round_trips_as_plain_data(self):
        # Series children receive the profile across a process boundary.
        from dataclasses import asdict
        custom = QUICK.with_overrides(trend=True, max_verification_iterations=3)
        assert resolve_profile(asdict(custom)) == custom


def test_a_directory_is_refused_before_any_llm_call(tmp_path):
    """Observed live: a folder passed straight to the curve agent cost ~2 min
    and three LLM calls (file normalization asking for scripts that open() a
    directory) before failing. The orchestrator expands folders; a direct
    caller gets the fix in the message, for free."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    agent = CurveFittingAgent.__new__(CurveFittingAgent)   # no model, no I/O
    agent.output_dir = tmp_path
    err = agent._reject_directory_input(str(tmp_path))
    assert err["status"] == "error" and "list" in err["error"]["details"]
    f = tmp_path / "a.csv"
    f.write_text("x,y\n1,2\n")
    assert agent._reject_directory_input(str(f)) is None
    assert agent._reject_directory_input(None) is None


class TestLightSynthesisIsActuallyLighter:
    """Measured live: `quick` on an easy spectrum took as long as `thorough`
    (119 s vs 118 s) because the calls it removed were the short ones; the
    time is in what the model WRITES. `light` therefore asks for less."""

    def test_addendum_only_under_light(self):
        from scilink.agents.exp_agents._qc_profile import (
            LIGHT_SYNTHESIS_ADDENDUM, synthesis_addendum)
        assert synthesis_addendum({"_synthesis_level": "light"}) == LIGHT_SYNTHESIS_ADDENDUM
        for state in ({"_synthesis_level": "full"}, {"_synthesis_level": "none"}, {}, None):
            assert synthesis_addendum(state) is None

    def test_addendum_states_a_principle_and_keeps_the_schema(self):
        from scilink.agents.exp_agents._qc_profile import LIGHT_SYNTHESIS_ADDENDUM as a
        assert "one short paragraph" in a and "at most three" in a
        assert "required field" in a        # brevity must not break parsing


class TestPurposeScopedVerification:
    """Measured live: the iteration cap is not what makes the agents slow on a
    hard item — every refinement round the verifier asks for is a 40–150 s
    multimodal call. Under quick / extract the verifier is told what the
    result is for and rejects only what would change it."""

    def test_presets(self):
        assert THOROUGH.verification == "strict" and REALTIME.verification == "strict"
        assert QUICK.verification == "purpose" and EXTRACT.verification == "purpose"
        with pytest.raises(ValueError, match="verification"):
            QCProfile(name="x", verification="lenient")

    def test_no_addendum_under_strict(self):
        from scilink.agents.exp_agents._qc_profile import verification_addendum
        for state in ({}, None, {"_verification_mode": "strict",
                                 "analysis_objective": "count particles"}):
            assert verification_addendum(state) is None

    def test_purpose_comes_from_the_objective(self):
        from scilink.agents.exp_agents._qc_profile import verification_addendum
        a = verification_addendum({"_verification_mode": "purpose",
                                   "analysis_objective": "the G-band position for a BO loop"})
        assert "the G-band position for a BO loop" in a
        assert "materially change" in a and "accept" in a

    def test_purpose_falls_back_to_the_locked_plan(self):
        from scilink.agents.exp_agents._qc_profile import verification_addendum
        a = verification_addendum({
            "_verification_mode": "purpose",
            "locked_fitting_config": {"parameters_to_extract": ["center_1", "fwhm_1"]}})
        assert "center_1, fwhm_1" in a
        b = verification_addendum({"_verification_mode": "purpose"})
        assert "set out to extract" in b        # never just "be lenient"


def test_profile_is_stamped_where_results_are_published():
    """Observed live: a 5-spectrum `extract` series wrote series_fit_results.json
    with no per-item profile mark, because the stamp was applied at result
    compilation — after the store stage had written the file."""
    from scilink.agents.exp_agents._qc_profile import stamp_profile
    rows = [{"name": "a", "quality_history": {"approved": True}}, {"name": "b"}, "junk"]
    stamp_profile({"_qc_profile": "extract"}, rows)
    assert rows[0]["quality_history"] == {"approved": True, "produced_under_profile": "extract"}
    assert rows[1]["quality_history"] == {"produced_under_profile": "extract"}
    untouched = [{"name": "a"}]
    stamp_profile({"_qc_profile": "thorough"}, untouched)
    stamp_profile({}, untouched)
    stamp_profile(None, None)
    assert untouched == [{"name": "a"}]


class TestBankFirstUnderQuick:
    """Step 3c: with a verbatim audition winner the record IS the plan, so the
    pipeline has no planning / skill-selection / literature stage at all."""

    def _recipe(self, profile):
        return _names(create_unified_curve_fitting_pipeline(
            model=None, logger=logging.getLogger("t"), generation_config=None,
            safety_settings=None, parse_fn=None, store_fn=None, plot_fn=None,
            executor=None, output_dir="out", load_skills_fn=lambda *a, **k: None,
            profile=profile, bank_recipe=True))

    def test_quick_recipe_keeps_synthesis_only(self):
        assert self._recipe("quick") == [
            "AnalyzeDataController", "UnifiedSeriesProcessingController",
            "UnifiedCurveSynthesisController", "StoreAnalysisResultsController",
            "GenerateCurveFittingReportController", "UnifiedCurveReportController"]

    def test_extract_recipe_has_no_llm_stage_at_all(self):
        assert self._recipe("extract") == [
            "AnalyzeDataController", "UnifiedSeriesProcessingController",
            "StoreAnalysisResultsController",
            "GenerateCurveFittingReportController", "UnifiedCurveReportController"]

    def test_bank_recipe_flag_is_inert_without_it(self):
        assert "CurveFittingPlanningController" in _names(_pipeline("quick"))


class TestReducedDepthScriptsMustEarnVerbatimReuse:
    @pytest.fixture(autouse=True)
    def _bank(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.setenv("SCILINK_SCRIPT_BANK", "1")

    def _agent(self, tmp_path, runs):
        from types import SimpleNamespace
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        import scilink.agents.exp_agents._locked_exec as le
        a = CurveFittingAgent.__new__(CurveFittingAgent)
        a.output_dir = tmp_path / "run"
        a.executor = None
        a.logger = logging.getLogger("t")
        return a

    def _xy(self, shift=0.0, seed=0):
        import numpy as np
        x = np.linspace(0, 100, 2000)
        y = (5 * np.exp(-(x - 20 - shift) ** 2 / 4) + 3 * np.exp(-(x - 60) ** 2 / 9)
             + np.random.RandomState(seed).normal(0, 0.05, x.size))
        return np.vstack([x, y])

    def _audition(self, tmp_path, monkeypatch, born, extra_verbatim=0):
        import numpy as np
        from scilink.skills._shared import _script_bank as sb
        import scilink.agents.exp_agents._locked_exec as le
        data = self._xy()
        rid = sb.add_record("curve_fitting", {
            "working_script": "print('FIT_RESULTS_JSON: {}')",
            "data_fingerprint": sb.curve_fingerprint(data[0], data[1]),
            "provenance": {"session": "s1", "profile": born}})["id"]
        for i in range(extra_verbatim):
            d = self._xy(shift=0.3 * (i + 1), seed=i + 1)
            sb.record_success("curve_fitting", rid, session=f"v{i}",
                              fingerprint=sb.curve_fingerprint(d[0], d[1]))
        ran = []
        monkeypatch.setattr(le, "stage_and_run", lambda ex, script, d, work: (
            ran.append(script) or {"status": "success", "visualization_path": "p.png",
                                   "stdout": 'FIT_RESULTS_JSON: {"fit_quality": {"r_squared": 0.99}}'}))
        a = self._agent(tmp_path, ran)
        a._parse_audition_r2 = lambda stdout: 0.99
        new = self._xy(shift=0.2, seed=9)
        return a._bank_cold_start_audition(new, {}, 0.95), ran

    def test_a_gate_pass_on_DIFFERENT_data_is_not_a_verbatim_win(self, tmp_path, monkeypatch):
        """Observed live: a two-peak script with a flexible baseline absorbed a
        new third peak (R² 0.996) and was locked verbatim while the drift check
        said the data had changed (similarity 0.858 < 0.92)."""
        import numpy as np
        from scilink.skills._shared import _script_bank as sb
        import scilink.agents.exp_agents._locked_exec as le
        two = self._xy()
        sb.add_record("curve_fitting", {
            "working_script": "print('FIT_RESULTS_JSON: {}')",
            "data_fingerprint": sb.curve_fingerprint(two[0], two[1]),
            "provenance": {"session": "s1", "profile": "thorough"}})
        ran = []
        monkeypatch.setattr(le, "stage_and_run", lambda *a, **k: (
            ran.append(1) or {"status": "success", "visualization_path": "p.png",
                              "stdout": 'FIT_RESULTS_JSON: {}'}))
        a = self._agent(tmp_path, ran)
        a._parse_audition_r2 = lambda stdout: 0.996          # the gate WOULD pass
        x = two[0]
        three = np.vstack([x, two[1] + 6 * np.exp(-(x - 85) ** 2 / 6)])
        fp = sb.curve_fingerprint(three[0], three[1])
        [cand] = sb.find_exemplar("curve_fitting", fp, min_score=0.55)
        assert cand["fingerprint_score"] < 0.92               # retrievable, but different data
        assert a._bank_cold_start_audition(three, {}, 0.95) is None and ran == []

    def test_the_recipe_that_just_breached_is_not_auditioned(self, tmp_path, monkeypatch):
        from scilink.skills._shared import _script_bank as sb
        win, ran = self._audition(tmp_path, monkeypatch, born="thorough")
        assert win is not None
        ran.clear()
        a = self._agent(tmp_path, ran)
        a._parse_audition_r2 = lambda stdout: 0.99
        h = sb.script_hash("print('FIT_RESULTS_JSON: {}')")
        assert a._bank_cold_start_audition(self._xy(shift=0.2, seed=9), {}, 0.95,
                                           exclude_hashes=[h]) is None
        assert ran == []

    def test_thorough_born_script_is_eligible_at_once(self, tmp_path, monkeypatch):
        win, ran = self._audition(tmp_path, monkeypatch, born="thorough")
        assert win is not None and len(ran) == 1

    def test_quick_born_script_is_not_run_unreviewed_on_its_first_reuse(self, tmp_path, monkeypatch):
        win, ran = self._audition(tmp_path, monkeypatch, born="extract")
        assert win is None and ran == []

    def test_quick_born_script_earns_it_on_a_second_dataset(self, tmp_path, monkeypatch):
        win, ran = self._audition(tmp_path, monkeypatch, born="extract", extra_verbatim=1)
        assert win is not None and len(ran) == 1
