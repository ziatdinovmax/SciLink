"""Offline tests for exposing max_verification_iterations on the image and
hyperspectral agents (#271), mirroring the curve-fitting fast/bypass mode.

Covers: analyze() signatures (orchestrator forwards by signature
introspection), per-call override validation, the image controller's
<=0 bypass (successful initial only) + its qc_post_verification fast
path, the hyperspectral controller's retry-budget override (constructor
+ per-run via state) and its single-attempt bypass, and pipeline-factory
threading. Default paths must be behavior-identical.

  conda run -n scilink python tests/test_max_verification_expose.py
"""
import inspect
import logging
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

results = {}
LOG = logging.getLogger("mvi_test")


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _Ctx:
    """Minimal stand-in for QCItemContext."""

    def __init__(self, best_result=None, is_anchor=True):
        self.is_anchor = is_anchor
        self.best_result = best_result
        self.best_score = 0.0
        self.approved = False
        self.all_attempts = []
        self.verification_history = []
        self.judge_result = None
        self.quality_threshold = 6.0
        self.attempt_entries = []


def main():
    # ------------------------------------------------------------------
    print("1) analyze() signatures (orchestrator introspection contract):")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    for cls in (CurveFittingAgent, ImageAnalysisAgent,
                HyperspectralAnalysisAgent):
        check(f"{cls.__name__}.analyze accepts max_verification_iterations",
              "max_verification_iterations"
              in inspect.signature(cls.analyze).parameters)
    check("ImageAnalysisAgent constructor default unchanged (7)",
          inspect.signature(ImageAnalysisAgent.__init__)
          .parameters["max_verification_iterations"].default == 7)
    check("HyperspectralAnalysisAgent constructor default is None (built-in)",
          inspect.signature(HyperspectralAnalysisAgent.__init__)
          .parameters["max_verification_iterations"].default is None)

    # ------------------------------------------------------------------
    print("2) image controller bypass (successful initial only):")
    from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
        UnifiedImageProcessingController)
    c = object.__new__(UnifiedImageProcessingController)
    c.logger = LOG

    c.max_verification_iterations = 0
    ctx = _Ctx(best_result={"success": True})
    check("bypass fires at 0 with a successful initial result",
          c.qc_verification_bypass(ctx) is True and ctx.approved
          and getattr(ctx, "_verification_bypassed", False))

    ctx = _Ctx(best_result={"success": False})
    check("failed initial result does NOT bypass (recovery path preserved)",
          c.qc_verification_bypass(ctx) is False and not ctx.approved)

    ctx = _Ctx(best_result=None)
    check("no initial result does NOT bypass",
          c.qc_verification_bypass(ctx) is False)

    ctx = _Ctx(best_result={"success": True}, is_anchor=False)
    check("non-anchor items do NOT bypass",
          c.qc_verification_bypass(ctx) is False)

    c.max_verification_iterations = 7
    ctx = _Ctx(best_result={"success": True})
    check("default budget (7) does NOT bypass — thorough path unchanged",
          c.qc_verification_bypass(ctx) is False and not ctx.approved)

    # ------------------------------------------------------------------
    print("3) image qc_post_verification fast path:")
    c.max_verification_iterations = 0
    seen = {}
    c._build_quality_history = (
        lambda *a, **k: seen.setdefault("hist", {"attempts": len(a)}))
    c._stamp_hot_deviation = lambda r: seen.setdefault("stamped", True)
    ctx = _Ctx(best_result={"success": True})
    c.qc_verification_bypass(ctx)
    out = c.qc_post_verification(ctx)
    check("bypassed run returns the result directly (no accept gate)",
          out is ctx.best_result and seen.get("stamped"))
    check("bypassed run stamped approved_by=bypass in quality_history",
          out.get("quality_history", {}).get("approved") is True
          and out["quality_history"].get("approved_by") == "bypass")

    class _Gate:
        def is_accept(self, s):
            seen["gate_consulted"] = True
            return False

    ctx2 = _Ctx(best_result={"success": True})
    ctx2.accept_gate = _Gate()
    check("non-bypassed run still consults the accept gate (default path)",
          c.qc_post_verification(ctx2) is None
          and seen.get("gate_consulted") is True)

    # ------------------------------------------------------------------
    print("4) hyperspectral controller retry-budget override:")
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        RunDynamicAnalysisController)
    h = object.__new__(RunDynamicAnalysisController)
    h.logger = LOG

    h._ctor_max_verification = None
    h._max_verification_override = None
    check("default budget unchanged (MAX_RETRIES-1 = 4)",
          h.max_verification_iterations == 4)

    h._max_verification_override = 2
    check("override replaces the retry count",
          h.max_verification_iterations == 2)
    h._max_verification_override = -3
    check("negative override clamped to 0",
          h.max_verification_iterations == 0)

    # Per-run refresh: execute() reads state before the gatekeeping return.
    h._ctor_max_verification = 1
    h._max_verification_override = None
    h.execute({"refinement_decision": {}})
    check("execute() falls back to the constructor override",
          h.max_verification_iterations == 1)
    h.execute({"refinement_decision": {}, "max_verification_iterations": 0})
    check("execute() honors the per-run state override over the constructor",
          h.max_verification_iterations == 0)
    h.execute({"refinement_decision": {}})
    check("stale per-run override cleared on the next run",
          h.max_verification_iterations == 1)

    print("5) hyperspectral single-attempt bypass:")
    h._ctor_max_verification = None
    h._max_verification_override = 0
    ctx = _Ctx(best_result={"success": True, "task_success": True})
    check("bypass fires at 0 when the task succeeded",
          h.qc_verification_bypass(ctx) is True and ctx.approved)
    ctx = _Ctx(best_result={"success": True, "task_success": False})
    check("failed task does NOT bypass (salvage path preserved)",
          h.qc_verification_bypass(ctx) is False and not ctx.approved)
    h._max_verification_override = None
    ctx = _Ctx(best_result={"success": True, "task_success": True})
    check("default budget does NOT bypass — retry ladder unchanged",
          h.qc_verification_bypass(ctx) is False)

    # ------------------------------------------------------------------
    print("6) pipeline factory threading:")
    from scilink.agents.exp_agents.pipelines.hyperspectral_pipelines import (
        create_hyperspectral_iteration_pipeline)
    pipe = create_hyperspectral_iteration_pipeline(
        model=None, logger=LOG, generation_config=None, safety_settings=None,
        settings={"enabled": False}, preprocessor=None, parse_fn=lambda r: r,
        max_verification_iterations=2)
    dyn = [p for p in pipe if isinstance(p, RunDynamicAnalysisController)]
    check("factory threads the override into RunDynamicAnalysisController",
          len(dyn) == 1 and dyn[0]._ctor_max_verification == 2
          and dyn[0].max_verification_iterations == 2)
    pipe = create_hyperspectral_iteration_pipeline(
        model=None, logger=LOG, generation_config=None, safety_settings=None,
        settings={"enabled": False}, preprocessor=None, parse_fn=lambda r: r)
    dyn = [p for p in pipe if isinstance(p, RunDynamicAnalysisController)]
    check("factory default leaves the built-in budget",
          dyn[0].max_verification_iterations == 4)

    from scilink.agents.exp_agents.pipelines.image_analysis_pipelines import (
        create_unified_image_analysis_pipeline)
    check("image factory already accepts the knob (threaded from analyze)",
          "max_verification_iterations"
          in inspect.signature(create_unified_image_analysis_pipeline)
          .parameters)

    # ------------------------------------------------------------------
    print("7) per-call validation (< 0 rejected before any work):")
    import tempfile
    img_agent = ImageAnalysisAgent(
        api_key="sk-dummy", output_dir=tempfile.mkdtemp(prefix="mvi_img_"),
        enable_human_feedback=False)
    try:
        img_agent.analyze("nonexistent.npy", max_verification_iterations=-1)
        check("image analyze(-1) raises ValueError", False)
    except ValueError:
        check("image analyze(-1) raises ValueError", True)
    hs_agent = HyperspectralAnalysisAgent(
        api_key="sk-dummy", output_dir=tempfile.mkdtemp(prefix="mvi_hs_"),
        enable_human_feedback=False)
    try:
        hs_agent.analyze("nonexistent.npy", max_verification_iterations=-1)
        check("hyperspectral analyze(-1) raises ValueError", False)
    except ValueError:
        check("hyperspectral analyze(-1) raises ValueError", True)
    r = img_agent.analyze("nonexistent.npy", max_verification_iterations=0)
    check("image analyze(0) proceeds past validation (fails later on data)",
          isinstance(r, dict) and r.get("status") == "error")
    r = hs_agent.analyze("nonexistent.npy", max_verification_iterations=0)
    check("hyperspectral analyze(0) proceeds past validation",
          isinstance(r, dict) and r.get("status") == "error")

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"MAX-VERIFICATION EXPOSE: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
