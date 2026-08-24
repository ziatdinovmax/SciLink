"""One temperature governs the whole skill at every stage (issue #498).

Skill strictness used to anneal only inside the codegen/verification loop;
planning, plan validation, interpretation and (image) conformance injected
the skill as hard-MANDATORY at every temperature, so a hot re-run re-derived
the same skill-bound plan and could only deviate while patching the fit.

Now a per-agent ``_SKILL_STRICTNESS_FRAMES`` table, resolved by
``_skill_annealing_level(state)`` (live ``_annealing_level``, falling back to
the run's ``_starting_annealing_level``), frames the skill identically at
all stages: mandatory (T=0) -> guidance (T=1) -> reference/overridable (T=2).

Invariants pinned here:
  * T=0 framing is BYTE-IDENTICAL to the pre-#498 hardcoded wording (per agent).
  * planning / interpretation framing relaxes at T=1 / T=2.
  * a hot RE-RUN (``_starting_annealing_level`` set, loop not started) re-plans
    with the skill already relaxed.
  * conformance keeps checking the locked plan; only its skill framing anneals.
  * best-of-N winner lock propagates the reached level so interpretation reads
    the FINAL temperature.
  * hyperspectral's codegen retry ladder re-renders the skill block at the
    attempt's rung; the first attempt is unchanged.
"""

import inspect
import logging
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc
from scilink.agents.exp_agents.controllers import image_analysis_controllers as ic
from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
from scilink.agents.exp_agents._qc_engine import QCItemContext


_SKILL = {
    "name": "xps",
    "planning": "PLAN-RULES",
    "analysis": "ANALYSIS-RULES",
    "implementation": "IMPL-RULES",
    "interpretation": "INTERP-RULES",
    "validation": "VALID-RULES",
}


def _state(**extra):
    st = {"skills_loaded": [dict(_SKILL)],
          "skill_sections": dict(_SKILL), "skill_name": "xps"}
    st.update(extra)
    return st


def _render(mod, state, stage):
    parts = []
    mod._append_skill_context(parts, state, stage)
    return parts


# ---------------------------------------------------------------------------
# Level resolver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_resolver_first_run_is_frozen(mod):
    assert mod._skill_annealing_level({}) == 0
    assert mod._skill_annealing_level({"_starting_annealing_level": None}) == 0


@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_resolver_falls_back_to_starting_level_before_loop_seeds_it(mod):
    # Planning runs BEFORE the QC engine seeds _annealing_level: a hot re-run
    # must resolve to its requested starting level.
    assert mod._skill_annealing_level({"_starting_annealing_level": 2}) == 2
    assert mod._skill_annealing_level({"_starting_annealing_level": 1}) == 1


@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_resolver_live_level_wins_and_clamps(mod):
    assert mod._skill_annealing_level(
        {"_annealing_level": 1, "_starting_annealing_level": 2}) == 1
    assert mod._skill_annealing_level(
        {"_annealing_level": 0, "_starting_annealing_level": 2}) == 0
    assert mod._skill_annealing_level({"_annealing_level": 99}) == 2
    assert mod._skill_annealing_level({"_annealing_level": -3}) == 0


# ---------------------------------------------------------------------------
# T=0 byte-identity with the pre-#498 hardcoded wording
# ---------------------------------------------------------------------------

_CURVE_MANDATORY_INTRO = (
    "The following rules are MANDATORY. Your analysis plan and implementation "
    "MUST conform to these domain-specific requirements. These rules encode "
    "validated domain expertise and take precedence over general-purpose defaults. "
    "Do NOT substitute your own preferences where these rules specify a method, "
    "treatment, or constraint."
)


def test_curve_t0_planning_framing_is_byte_identical():
    assert _render(cc, _state(), "planning") == [
        "\n## MANDATORY Domain Skill Rules: xps (planning)",
        _CURVE_MANDATORY_INTRO,
        "PLAN-RULES",
        "\n## MANDATORY Domain Validation Rules (xps)",
        "VALID-RULES",
    ]
    # analysis stage: no validation block (unchanged)
    assert _render(cc, _state(), "analysis") == [
        "\n## MANDATORY Domain Skill Rules: xps (analysis)",
        _CURVE_MANDATORY_INTRO,
        "ANALYSIS-RULES",
    ]


def test_image_t0_planning_framing_is_byte_identical():
    assert _render(ic, _state(), "planning") == [
        "\n## Domain Expertise: xps (planning)",
        "The following guidance is from validated domain expertise. "
        "Use it to inform your approach.",
        "PLAN-RULES",
        "\n## Domain Validation Guidance: xps",
        "VALID-RULES",
    ]


def test_hyperspectral_t0_framing_is_byte_identical():
    assert _render(hc, _state(), "planning") == [
        "\n## MANDATORY Domain Skill Rules: xps (planning)\n"
        + _CURVE_MANDATORY_INTRO
        + "\nPLAN-RULES"
        "\n\n## MANDATORY Domain Validation Rules (xps)"
        "\nVALID-RULES"
    ]
    # implementation (codegen) also carries validation at T=0 — unchanged.
    block = hc._render_skill_block(_state(), "implementation")
    assert block.startswith("\n## MANDATORY Domain Skill Rules: xps (implementation)")
    assert "\n## MANDATORY Domain Validation Rules (xps)\nVALID-RULES" in block
    assert hc._render_skill_block(_state(), "implementation", level=0) == block


@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_t0_frame_matches_the_legacy_single_skill_state(mod):
    # legacy skill_sections-only state renders exactly like skills_loaded
    legacy = {"skill_sections": dict(_SKILL), "skill_name": "xps"}
    assert _render(mod, legacy, "planning") == _render(mod, _state(), "planning")


# ---------------------------------------------------------------------------
# Planning / interpretation relax with temperature
# ---------------------------------------------------------------------------

def test_curve_planning_relaxes_with_live_level():
    warm = _render(cc, _state(_annealing_level=1), "planning")
    hot = _render(cc, _state(_annealing_level=2), "planning")
    assert warm[0] == "\n## Domain Skill Guidance: xps (planning)"
    assert warm[3] == "\n## Domain Validation Guidance (xps)"
    assert hot[0] == "\n## Domain Skill Reference: xps (planning)"
    assert hot[3] == "\n## Domain Validation Reference (xps)"
    for parts in (warm, hot):
        assert not any("MANDATORY" in p for p in parts)
        assert "PLAN-RULES" in parts and "VALID-RULES" in parts  # content intact
    assert "Override any rule" in hot[1]
    assert "explain why" in warm[1]


def test_image_planning_relaxes_with_live_level():
    warm = _render(ic, _state(_annealing_level=1), "planning")
    hot = _render(ic, _state(_annealing_level=2), "planning")
    assert warm[0] == "\n## Domain Expertise (reference): xps (planning)"
    assert warm[3] == "\n## Domain Validation Reference: xps"
    assert hot[0] == "\n## Domain Expertise (context): xps (planning)"
    assert hot[3] == "\n## Domain Validation Context: xps"
    assert "Override any guidance" in hot[1]


def test_hyperspectral_block_relaxes_with_level():
    hot = hc._render_skill_block(_state(), "implementation", level=2)
    assert hot.startswith("\n## Domain Skill Reference: xps (implementation)")
    assert "MANDATORY" not in hot
    assert "\n## Domain Validation Reference (xps)\nVALID-RULES" in hot
    # state-level resolution works too (no level kwarg)
    assert hc._render_skill_block(_state(_annealing_level=1), "planning").startswith(
        "\n## Domain Skill Guidance: xps (planning)")


@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_hot_rerun_replans_with_relaxed_skill(mod):
    # A re-run launched hot sets ONLY _starting_annealing_level; the loop has
    # not seeded _annealing_level yet when planning runs.
    parts = _render(mod, _state(_starting_annealing_level=2), "planning")
    text = "\n".join(parts)
    assert "MANDATORY" not in text
    assert mod._SKILL_STRICTNESS_FRAMES[2][0] in text
    assert "PLAN-RULES" in text


@pytest.mark.parametrize("mod", [cc, ic, hc], ids=["curve", "image", "hyper"])
def test_interpretation_reads_final_level(mod):
    # After a fit that went hot, state["_annealing_level"] == 2 (engine sync /
    # winner lock) -> interpretation reads the skill as reference.
    hot = "\n".join(_render(mod, _state(_annealing_level=2), "interpretation"))
    frozen = "\n".join(_render(mod, _state(), "interpretation"))
    assert mod._SKILL_STRICTNESS_FRAMES[2][0] in hot
    assert mod._SKILL_STRICTNESS_FRAMES[2][2] in hot
    assert mod._SKILL_STRICTNESS_FRAMES[0][0] in frozen
    assert "INTERP-RULES" in hot and "INTERP-RULES" in frozen


def test_multi_skill_intro_appears_once_per_frame():
    st = _state()
    st["skills_loaded"].append({"name": "raman", "planning": "P2", "validation": "V2"})
    hot = _render(cc, dict(st, _annealing_level=2), "planning")
    intro = cc._SKILL_STRICTNESS_FRAMES[2][1]
    assert hot.count(intro) == 1
    assert hot[0] == "\n## Domain Skill Reference: xps (planning)"
    assert "\n## Domain Skill Reference: raman (planning)" in hot


# ---------------------------------------------------------------------------
# Conformance: still checks the locked plan; only the skill framing anneals
# ---------------------------------------------------------------------------

class _CapturingModel:
    def __init__(self):
        self.prompts = []

    def generate_content(self, contents=None, **_):
        self.prompts.append(contents[0])
        raise RuntimeError("stop after capture")


def _curve_conformance_prompt(state):
    ctl = object.__new__(cc.UnifiedSeriesProcessingController)
    ctl.model = _CapturingModel()
    ctl.logger = logging.getLogger("test")
    ctl.conformance_instructions = (
        "PLAN={physical_model}|{analysis_approach}|{parameters_to_extract}|"
        "{fitting_strategy}\nSKILL={skill_rules}\nSCRIPT={script}")
    state = dict(state)
    state["locked_fitting_config"] = {
        "physical_model": "two Voigts", "analysis_approach": "fit",
        "parameters_to_extract": ["fwhm"], "fitting_strategy": "lm"}
    assert ctl._check_plan_conformance(state, "print(1)") is None
    return ctl.model.prompts[0]


def test_curve_conformance_t0_framing_unchanged_and_plan_still_checked():
    prompt = _curve_conformance_prompt(_state())
    assert "PLAN=two Voigts|fit|fwhm|lm" in prompt
    assert "## MANDATORY Domain Skill Rules (xps)" in prompt
    assert "### Planning rules\nPLAN-RULES" in prompt
    assert "### Validation rules\nVALID-RULES" in prompt


def test_curve_conformance_framing_anneals_with_starting_level():
    # T=1: rules stay, framed as guidance with plan precedence.
    prompt = _curve_conformance_prompt(_state(_starting_annealing_level=1))
    assert "PLAN=two Voigts|fit|fwhm|lm" in prompt  # locked plan still checked
    assert "MANDATORY" not in prompt
    assert "## Domain Skill Guidance (xps)" in prompt
    assert "follow the plan" in prompt
    assert "PLAN-RULES" in prompt  # content intact, only framing relaxed
    # T=2 (hot): the skill is disregarded — the checker judges plan fidelity
    # only. Rules offered as "reference" were still read as mandatory
    # criteria live, re-flagging a plan that deliberately left the skill.
    prompt = _curve_conformance_prompt(_state(_starting_annealing_level=2))
    assert "PLAN=two Voigts|fit|fwhm|lm" in prompt
    assert "SKILL=\n" in prompt
    assert "PLAN-RULES" not in prompt and "Domain Skill" not in prompt


@pytest.mark.parametrize("mod", [cc, ic], ids=["curve", "image"])
def test_codegen_schedule_plan_precedence_above_frozen_only(mod):
    cls = (cc.UnifiedSeriesProcessingController if mod is cc
           else ic.UnifiedImageProcessingController)
    sched = cls._SKILL_STRICTNESS_SCHEDULE
    assert "follow the plan" not in sched[0]  # T=0 text byte-identical
    assert all("follow the plan" in t for t in sched[1:])


def _image_conformance_prompt(state):
    ctl = object.__new__(ic.UnifiedImageProcessingController)
    ctl.model = _CapturingModel()
    ctl.logger = logging.getLogger("test")
    ctl.conformance_instructions = (
        "PLAN={processing_pipeline}|{analysis_approach}|{features_to_extract}\n"
        "SKILL={skill_rules}\nSCRIPT={script}")
    state = dict(state)
    state["locked_analysis_config"] = {
        "processing_pipeline": "flatten->segment", "analysis_approach": "seg",
        "features_to_extract": ["area"]}
    assert ctl._check_conformance(state, "print(1)") is None
    return ctl.model.prompts[0]


def test_image_conformance_t0_framing_unchanged_and_anneals():
    frozen = _image_conformance_prompt(_state())
    assert "PLAN=flatten->segment|seg|area" in frozen
    assert "\n**Domain Expertise (xps):**\n" in frozen
    warm = _image_conformance_prompt(_state(_annealing_level=1))
    assert "PLAN=flatten->segment|seg|area" in warm
    assert "\n**Domain Expertise (reference) (xps):**\n" in warm
    assert "**Domain Expertise (xps):**" not in warm
    hot = _image_conformance_prompt(_state(_annealing_level=2))
    assert "PLAN=flatten->segment|seg|area" in hot
    assert "SKILL=\n" in hot and "PLAN-RULES" not in hot  # skill disregarded


# ---------------------------------------------------------------------------
# Codegen / correction read the same resolver (no divergent level reads)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [cc, ic], ids=["curve", "image"])
def test_codegen_skill_preamble_reads_the_shared_resolver(mod):
    src = inspect.getsource(mod)
    # Every _SKILL_STRICTNESS_SCHEDULE lookup is driven by the resolver;
    # no stage reads a raw state.get("_annealing_level", 0) for the skill.
    for m in __import__("re").finditer(r"_SKILL_STRICTNESS_SCHEDULE\[", src):
        window = src[m.start() - 400: m.end() + 200]
        assert "_skill_annealing_level(state)" in window, window
    assert "level = state.get(\"_annealing_level\", 0)\n            preamble" not in src


# ---------------------------------------------------------------------------
# Best-of-N winner lock propagates the reached level (interpretation reads it)
# ---------------------------------------------------------------------------

def _winner(levels, produced_at=None):
    result = {"success": True, "quality_history": {"verification_iterations": [
        {"annealing_level": lv} for lv in levels]}}
    if produced_at is not None:
        result["_produced_at_level"] = produced_at
    return {"result": result, "attempt": 1}


@pytest.mark.parametrize("mod,cls", [
    (cc, "UnifiedSeriesProcessingController"),
    (ic, "UnifiedImageProcessingController"),
], ids=["curve", "image"])
def test_winner_lock_records_reached_level(mod, cls):
    ctl = object.__new__(getattr(mod, cls))
    st = _state()
    ctl._record_winner_annealing_level(st, _winner([0, 1, 2]))
    assert st["_annealing_level"] == 2
    st = _state()
    ctl._record_winner_annealing_level(st, _winner([0, 0]))
    assert st["_annealing_level"] == 0
    st = _state()
    ctl._record_winner_annealing_level(st, _winner([], produced_at=2))
    assert st["_annealing_level"] == 2
    # monotone: never lowers a hot re-run's starting level
    st = _state(_starting_annealing_level=2)
    ctl._record_winner_annealing_level(st, _winner([]))
    assert st["_annealing_level"] == 2
    # interpretation then reads the FINAL level
    assert mod._SKILL_STRICTNESS_FRAMES[2][0] in "\n".join(
        _render(mod, st, "interpretation"))


@pytest.mark.parametrize("mod", [cc, ic], ids=["curve", "image"])
def test_winner_lock_calls_recorder(mod):
    src = inspect.getsource(mod)
    i = src.index("# --- Lock the winner ---")
    assert "_record_winner_annealing_level(state, winner)" in src[i:i + 500]


# ---------------------------------------------------------------------------
# Hyperspectral retry ladder re-renders the skill block at the attempt's rung
# ---------------------------------------------------------------------------

def _hyper_refine(retries, with_builder=True):
    ctl = object.__new__(hc.RunDynamicAnalysisController)
    ctl.logger = logging.getLogger("test")
    ctl._max_verification_override = None
    st = _state()
    ctx = QCItemContext(state=st, data=None, data_path="", item_name="t", item_idx=0,
                        is_regime_anchor=True)
    ctx.retries = retries
    ctx.last_code = "print(0)"
    ctx.attempt_entries = []
    ctx.last_passed_names = None

    def _builder(level=0):
        return "HEAD\n" + hc._render_skill_block(st, "implementation", level=level)

    ctx.base_prompt = _builder(0)
    if with_builder:
        ctx.base_prompt_builder = _builder
    out = ctl.qc_refine(ctx, {"error_msg": "bad"})
    return out["prompt"], ctx


def test_hyper_retry_ladder_relaxes_skill_block_with_rung():
    p0, ctx0 = _hyper_refine(0)
    assert ctx0.annealing_level == 0
    assert "## MANDATORY Domain Skill Rules: xps (implementation)" in p0
    p1, ctx1 = _hyper_refine(1)
    assert ctx1.annealing_level == 1
    assert "## Domain Skill Guidance: xps (implementation)" in p1
    assert "MANDATORY" not in p1.split("HEAD", 1)[1].split("\n\n", 3)[0]
    p2, ctx2 = _hyper_refine(3)
    assert ctx2.annealing_level == 2
    assert "## Domain Skill Reference: xps (implementation)" in p2
    assert "MANDATORY Domain Skill Rules" not in p2
    for p in (p0, p1, p2):
        assert p.startswith("HEAD\n")
        assert "IMPL-RULES" in p  # recipe content is never dropped


def test_hyper_retry_without_builder_keeps_base_prompt():
    # Defensive: a ctx without the builder (older callers) behaves as before.
    p, _ = _hyper_refine(3, with_builder=False)
    assert "## MANDATORY Domain Skill Rules: xps (implementation)" in p


def test_hyper_builder_is_wired_on_ctx():
    src = inspect.getsource(hc.RunDynamicAnalysisController)
    assert "ctx.base_prompt_builder = _render_base_prompt" in src
    assert "base_prompt = _render_base_prompt(0)" in src
