"""
Tests for agent-side multi-skill auto-selection.

Covers the shared relevance selector (`scilink.skills._shared._skill_selector`)
and the multi-skill prompt rendering in the curve-fitting and hyperspectral
controllers. The LLM call is mocked, so these are fully offline.
"""

import json

import pytest

from scilink.skills._shared._skill_selector import (
    build_skill_catalog,
    select_relevant_skills,
)
from scilink.agents.exp_agents.controllers import (
    curve_fitting_controllers as cc,
    hyperspectral_controllers as hc,
    image_analysis_controllers as ic,
)


class _MockModel:
    """Returns a fixed payload from generate_content."""

    def __init__(self, payload):
        self.payload = payload

    def generate_content(self, contents, generation_config=None, safety_settings=None):
        return self.payload


def _parse(resp):
    return json.loads(resp), None


def _selector(payload, **kw):
    return select_relevant_skills(
        model=_MockModel(json.dumps(payload)),
        parse_fn=_parse,
        domain=kw.pop("domain", "curve_fitting"),
        context_parts=["some data"],
        **kw,
    )


class _RecordingModel:
    """Captures the prompt `contents` passed to generate_content."""

    def __init__(self, payload):
        self.payload = payload
        self.contents = None

    def generate_content(self, contents, generation_config=None, safety_settings=None):
        self.contents = contents
        return self.payload


def _prompt_text(model):
    return "\n".join(p for p in model.contents if isinstance(p, str))


class TestHintInjection:
    def test_valid_hint_surfaces_as_suggestion(self):
        m = _RecordingModel(json.dumps({"skills": []}))
        select_relevant_skills(model=m, parse_fn=_parse, domain="curve_fitting",
                               context_parts=["data"], hint="xps")
        txt = _prompt_text(m)
        assert "Orchestrator suggestion" in txt and "xps" in txt

    def test_invalid_hint_is_dropped(self):
        m = _RecordingModel(json.dumps({"skills": []}))
        select_relevant_skills(model=m, parse_fn=_parse, domain="curve_fitting",
                               context_parts=["data"], hint="does_not_exist")
        assert "Orchestrator suggestion" not in _prompt_text(m)

    def test_no_hint_no_suggestion_block(self):
        m = _RecordingModel(json.dumps({"skills": []}))
        select_relevant_skills(model=m, parse_fn=_parse, domain="curve_fitting",
                               context_parts=["data"])
        assert "Orchestrator suggestion" not in _prompt_text(m)


def _write_custom_skill(tmp_path, name, *, technique=None, domain=None, body="Overview text."):
    fm = []
    if technique:
        fm.append(f"technique: {technique}")
    if domain:
        fm.append(f"domain: {domain}")
    front = ("---\n" + "\n".join(fm) + "\n---\n") if fm else ""
    p = tmp_path / f"{name}.md"
    p.write_text(f"{front}## overview\n{body}\n")
    return str(p)


class TestCustomSkills:
    """#256 fix #1 — registered custom skills are visible to the agent selector."""

    def test_custom_skill_folded_into_catalog(self, tmp_path):
        path = _write_custom_skill(tmp_path, "myraman", technique="Raman")
        cat, names = build_skill_catalog("curve_fitting", custom_skills={"myraman": path})
        assert "myraman" in names
        assert any("myraman" in line for line in cat)

    def test_custom_skill_with_foreign_modality_is_skipped(self, tmp_path):
        # A custom skill that explicitly declares a DIFFERENT modality domain
        # must not leak into this agent's catalog.
        path = _write_custom_skill(tmp_path, "imgthing", domain="image_analysis")
        _, names = build_skill_catalog("curve_fitting", custom_skills={"imgthing": path})
        assert "imgthing" not in names

    def test_custom_skill_without_domain_is_included(self, tmp_path):
        path = _write_custom_skill(tmp_path, "plainskill")
        _, names = build_skill_catalog("curve_fitting", custom_skills={"plainskill": path})
        assert "plainskill" in names

    def test_selector_can_return_a_custom_skill(self, tmp_path):
        path = _write_custom_skill(tmp_path, "myraman", technique="Raman")
        out = select_relevant_skills(
            model=_MockModel(json.dumps({"skills": ["myraman"]})),
            parse_fn=_parse, domain="curve_fitting", context_parts=["Raman data"],
            custom_skills={"myraman": path},
        )
        assert out == ["myraman"]

    def test_custom_name_invalid_without_registration(self, tmp_path):
        # Same name but NOT registered -> not in catalog -> dropped.
        out = select_relevant_skills(
            model=_MockModel(json.dumps({"skills": ["myraman"]})),
            parse_fn=_parse, domain="curve_fitting", context_parts=["x"],
        )
        assert out == []


class TestCatalog:
    def test_curve_fitting_catalog_has_known_skills(self):
        _, names = build_skill_catalog("curve_fitting")
        assert {"epr", "xps", "xrd_profile"} <= names

    def test_catalog_lines_carry_technique_tags(self):
        lines, _ = build_skill_catalog("curve_fitting")
        joined = "\n".join(lines)
        assert "[technique:" in joined  # epr/xps/xrd_profile declare them


class TestSelector:
    def test_multiple_skills_selected_in_order(self):
        assert _selector({"skills": ["xps", "epr"]}) == ["xps", "epr"]

    def test_invalid_names_dropped(self):
        assert _selector({"skills": ["xps", "does_not_exist"]}) == ["xps"]

    def test_empty_when_none_apply(self):
        assert _selector({"skills": []}) == []

    def test_singular_reply_back_compat(self):
        assert _selector({"skill": "xrd_profile"}) == ["xrd_profile"]

    def test_string_reply_tolerated(self):
        assert _selector({"skills": "xps"}) == ["xps"]

    def test_max_skills_cap(self):
        out = _selector({"skills": ["xps", "epr", "xrd_profile"]}, max_skills=2)
        assert out == ["xps", "epr"]

    def test_duplicates_deduped(self):
        assert _selector({"skills": ["xps", "xps", "epr"]}) == ["xps", "epr"]

    def test_exclusive_caps_to_single(self):
        # Authoritative/mutually-exclusive domains (curve fitting) take at most
        # one skill even if the model returns several.
        assert _selector({"skills": ["xps", "epr"]}, exclusive=True) == ["xps"]

    def test_non_exclusive_allows_multiple(self):
        assert _selector({"skills": ["xps", "epr"]}) == ["xps", "epr"]

    def test_hint_does_not_force_selection(self):
        # Orchestrator hints 'epr' but the model (agent) chooses 'xps' from the
        # data — the agent has final authority; the hint is non-binding.
        assert _selector({"skills": ["xps"]}, hint="epr") == ["xps"]

    def test_malformed_reply_returns_empty(self):
        # parse_fn raises on non-JSON -> selector swallows and returns []
        out = select_relevant_skills(
            model=_MockModel("not json"),
            parse_fn=_parse,
            domain="curve_fitting",
            context_parts=["x"],
        )
        assert out == []


# Two loaded skills, most-relevant first.
_TWO = {
    "skills_loaded": [
        {"name": "epr", "planning": "EPR planning rules", "validation": "EPR validation"},
        {"name": "xps", "planning": "XPS planning rules"},
    ]
}
# Legacy single-skill state (no skills_loaded).
_LEGACY = {"skill_sections": {"name": "eels", "planning": "EELS rules"}, "skill_name": "eels"}


class TestCurveMultiSkillRender:
    def test_both_skills_and_validation_rendered(self):
        prompt = []
        cc._append_skill_context(prompt, _TWO, "planning")
        joined = "\n".join(prompt)
        assert "EPR planning rules" in joined
        assert "XPS planning rules" in joined
        assert "EPR validation" in joined

    def test_legacy_single_skill_still_renders(self):
        prompt = []
        cc._append_skill_context(prompt, _LEGACY, "planning")
        assert "EELS rules" in "\n".join(prompt)

    def test_no_skill_is_noop(self):
        prompt = []
        cc._append_skill_context(prompt, {}, "planning")
        assert prompt == []


class TestHyperspectralMultiSkillRender:
    def test_both_skills_rendered(self):
        block = hc._render_skill_block(_TWO, "planning")
        assert "EPR planning rules" in block and "XPS planning rules" in block

    def test_legacy_single_skill_still_renders(self):
        assert "EELS rules" in hc._render_skill_block(_LEGACY, "planning")

    def test_no_skill_is_empty_string(self):
        assert hc._render_skill_block({}, "planning") == ""

    def test_codegen_renders_all_co_active_recipes(self):
        # Co-active skills may each own a different stage — codegen must keep
        # all their implementation recipes, not just the top-ranked one.
        state = {
            "skills_loaded": [
                {"name": "eels", "implementation": "EELS recipe"},
                {"name": "eds", "implementation": "EDS recipe"},
            ]
        }
        block = hc._render_skill_block(state, "implementation")
        assert "EELS recipe" in block and "EDS recipe" in block


# Two skills where the TOP-ranked one carries NO implementation recipe — the
# afm-first case: codegen must still surface the recipe-owner's recipe.
_TOP_HAS_NO_RECIPE = {
    "skills_loaded": [
        {"name": "afm", "planning": "flatten guidance"},  # no implementation/analysis
        {"name": "overlapping_objects", "analysis": "segmentation recipe"},
    ]
}


class TestCodegenRecipeSourcing:
    """fix #2 — codegen sources recipes from ALL co-active skills, not primary."""

    def test_collects_all_recipe_bearing_skills_in_order(self):
        state = {"skills_loaded": [
            {"name": "epr", "implementation": "EPR recipe"},
            {"name": "xps", "implementation": "XPS recipe"},
        ]}
        for mod in (cc, ic):
            recipes = mod._collect_codegen_recipe(state)
            assert [n for n, _ in recipes] == ["epr", "xps"]

    def test_prose_only_skills_contribute_no_recipe(self):
        # _TWO has planning/validation but no implementation -> no codegen recipe.
        for mod in (cc, ic):
            assert mod._collect_codegen_recipe(_TWO) == []

    def test_top_skill_without_recipe_does_not_hide_owner(self):
        # afm (top, no recipe) must NOT shadow overlapping_objects' recipe.
        for mod in (cc, ic):
            recipes = mod._collect_codegen_recipe(_TOP_HAS_NO_RECIPE)
            assert recipes == [("overlapping_objects", "segmentation recipe")]

    def test_single_skill_render_is_verbatim(self):
        # Single-skill codegen output unchanged: just the recipe, no headers.
        for mod in (cc, ic):
            out = mod._render_codegen_recipe([("xps", "XPS recipe body")])
            assert out == "XPS recipe body"

    def test_multi_skill_render_labels_each(self):
        for mod in (cc, ic):
            out = mod._render_codegen_recipe(
                [("afm", "flatten code"), ("overlapping_objects", "segment code")]
            )
            assert "### Recipe — afm" in out and "### Recipe — overlapping_objects" in out
            assert "flatten code" in out and "segment code" in out
            assert "plan's order" in out  # composition note present

    def test_implementation_preferred_over_analysis(self):
        state = {"skills_loaded": [{"name": "x", "implementation": "impl", "analysis": "anal"}]}
        for mod in (cc, ic):
            assert mod._collect_codegen_recipe(state) == [("x", "impl")]

    def test_no_skill_is_empty(self):
        for mod in (cc, ic):
            assert mod._collect_codegen_recipe({}) == []
