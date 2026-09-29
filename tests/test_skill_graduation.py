"""Unit tests for sim_agents skill graduation.

The graduate_to_skill_file helper takes the LLM as a callable, so we
inject a fake one whose return value we control. No real LLM calls.
"""

import copy
import json
from pathlib import Path

import pytest

from scilink.agents.sim_agents.skill_graduation import (
    GRADUATED_SKILLS_DIR,
    KnowledgeStore,
    WORD_COUNT_WARN_THRESHOLD,
    _format_knowledge,
    format_graduated_skills_block,
    format_skill_as_markdown,
    graduate_to_skill_file,
    load_graduated_skills,
    parse_json_response,
)


# ── Templates and JSON fakes ─────────────────────────────────

# Minimal templates that just echo their inputs back; the real
# prompts in scilink.agents.sim_agents.instruct ask the LLM for
# JSON. Plumbing tests don't care about prompt text.
FAKE_FRESH = "FRESH skill={skill_name} domain={domain}\n{knowledge_text}"
FAKE_UPDATE = "UPDATE skill={skill_name}\nEXISTING:\n{existing_skill}\nNEW:\n{new_knowledge}"

# What a well-formed LLM JSON response looks like.
VALID_SKILL_JSON = json.dumps({
    "description": "test rule for graduation",
    "overview": "Covers a single VASP error class.",
    "planning": "Apply this rule when X.",
    "implementation": "Set INCAR key Y to value Z.",
    "interpretation": "Log line 'X' indicates the rule applies.",
    "validation": "After applying, verify W.",
})


def _fake_llm(response: str):
    """Return a callable matching the llm_call signature."""
    def _fn(prompt: str) -> str:
        return response
    return _fn


# ──────────────────────────────────────────────────────────────
# KnowledgeStore
# ──────────────────────────────────────────────────────────────

class TestKnowledgeStore:
    def test_record_and_get(self):
        ks = KnowledgeStore()
        i1 = ks.record({"summary": "first"})
        i2 = ks.record({"summary": "second"})
        assert i1 != i2
        assert ks.get(i1)["summary"] == "first"
        assert ks.get(i2)["summary"] == "second"

    def test_get_unknown_id_returns_none(self):
        ks = KnowledgeStore()
        ks.record({"summary": "x"})
        assert ks.get("does-not-exist") is None

    def test_list_returns_copy(self):
        ks = KnowledgeStore()
        ks.record({"summary": "x"})
        first = ks.list()
        first.append({"id": "fake", "summary": "should not stick"})
        # Mutating the returned list shouldn't affect the store.
        assert len(ks.list()) == 1

    def test_remove_specific(self):
        ks = KnowledgeStore()
        i1 = ks.record({"summary": "one"})
        i2 = ks.record({"summary": "two"})
        n = ks.remove(i1)
        assert n == 1
        assert ks.get(i1) is None
        assert ks.get(i2) is not None

    def test_remove_all(self):
        ks = KnowledgeStore()
        ks.record({"summary": "a"})
        ks.record({"summary": "b"})
        ks.record({"summary": "c"})
        n = ks.remove()
        assert n == 3
        assert ks.list() == []


# ──────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_pure_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_strips_json_fence(self):
        wrapped = '```json\n{"a": 1}\n```'
        assert parse_json_response(wrapped) == {"a": 1}

    def test_strips_bare_fence(self):
        wrapped = '```\n{"a": 1}\n```'
        assert parse_json_response(wrapped) == {"a": 1}

    def test_extracts_json_from_surrounding_prose(self):
        wrapped = 'Here is the result:\n{"a": 1}\nHope that helps!'
        assert parse_json_response(wrapped) == {"a": 1}

    def test_raises_when_no_json_present(self):
        with pytest.raises(ValueError):
            parse_json_response("nothing useful here")


# ──────────────────────────────────────────────────────────────
# Markdown emission
# ──────────────────────────────────────────────────────────────

class TestFormatSkillAsMarkdown:
    def test_basic(self):
        data = {
            "description": "test rule",
            "overview": "an overview paragraph",
            "planning": "planning notes",
        }
        out = format_skill_as_markdown(data)
        assert out.startswith("---\n")
        assert "description: test rule" in out
        assert "## overview" in out
        assert "an overview paragraph" in out
        assert "## planning" in out
        assert "planning notes" in out

    def test_round_trips_through_loader(self, tmp_path):
        """Files emitted by format_skill_as_markdown must be parseable
        by scilink.skills.loader without warnings."""
        from scilink.skills.loader import load_skill

        data = {
            "description": "round-trip test",
            "overview": "ov",
            "planning": "pl",
            "implementation": "im",
            "validation": "va",
        }
        out = format_skill_as_markdown(data)
        skill_dir = tmp_path / "vasp" / "round_trip"
        skill_dir.mkdir(parents=True)
        path = skill_dir / "round_trip.md"
        path.write_text(out)
        parsed = load_skill(str(path), domain="vasp")
        assert parsed["meta"]["description"] == "round-trip test"
        assert parsed.get("overview", "").strip() == "ov"
        assert parsed.get("planning", "").strip() == "pl"

    def test_yaml_special_descriptions_round_trip(self, tmp_path):
        """A description starting with `{` (which YAML treats as a
        flow-mapping opener if unquoted) must still parse cleanly."""
        from scilink.skills.loader import load_skill

        data = {
            "description": "{All,Veryfast} + ISMEAR=-5 produces a tetrahedron warning",
            "overview": "ov",
        }
        out = format_skill_as_markdown(data)
        skill_dir = tmp_path / "vasp" / "yaml_special"
        skill_dir.mkdir(parents=True)
        path = skill_dir / "yaml_special.md"
        path.write_text(out)
        parsed = load_skill(str(path), domain="vasp")
        assert parsed["meta"]["description"].startswith("{All,Veryfast}")

    def test_skips_empty_sections(self):
        data = {
            "description": "x",
            "overview": "",
            "planning": "real content",
            "implementation": None,
        }
        out = format_skill_as_markdown(data)
        assert "## planning" in out
        assert "## overview" not in out
        assert "## implementation" not in out

    def test_default_description_when_missing(self):
        out = format_skill_as_markdown({"overview": "x"})
        assert "description:" in out


# ──────────────────────────────────────────────────────────────
# Graduation: fresh
# ──────────────────────────────────────────────────────────────

class TestGraduateFresh:
    def test_creates_file_at_expected_path(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"id": "abc", "summary": "the thing"},
            skill_name="my_rule",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["status"] == "success"
        assert result["method"] == "created"
        skill_path = Path(result["skill_path"])
        assert skill_path == tmp_path / "vasp" / "my_rule" / "my_rule.md"
        assert skill_path.exists()
        assert (skill_path.parent / "__init__.py").exists()

    def test_emitted_file_has_well_formed_yaml_frontmatter(self, tmp_path):
        graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="r",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        content = (tmp_path / "vasp" / "r" / "r.md").read_text()
        # Frontmatter must be parseable by the loader.
        from scilink.skills.loader import load_skill
        parsed = load_skill(str(tmp_path / "vasp" / "r" / "r.md"), domain="vasp")
        assert parsed["meta"]["description"] == "test rule for graduation"
        assert "## overview" in content
        assert "Covers a single VASP error class." in parsed.get("overview", "")

    def test_handles_fenced_json(self, tmp_path):
        wrapped = f"```json\n{VALID_SKILL_JSON}\n```"
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="r",
            domain="vasp",
            llm_call=_fake_llm(wrapped),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["status"] == "success"

    def test_no_warning_below_word_threshold(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="r",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["warning"] is None
        assert 0 < result["word_count"] < WORD_COUNT_WARN_THRESHOLD


class TestGraduateLargeFile:
    def test_warning_fires_above_threshold(self, tmp_path):
        big_section = "word " * (WORD_COUNT_WARN_THRESHOLD + 100)
        big_json = json.dumps({
            "description": "huge skill",
            "overview": big_section,
        })
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="big_rule",
            domain="vasp",
            llm_call=_fake_llm(big_json),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["warning"] is not None
        assert "consolidation" in result["warning"]
        assert result["word_count"] > WORD_COUNT_WARN_THRESHOLD


# ──────────────────────────────────────────────────────────────
# Graduation: update existing
# ──────────────────────────────────────────────────────────────

class TestGraduateUpdate:
    def test_update_path_passes_existing_skill_as_json(self, tmp_path):
        # Step 1: create the skill via fresh path.
        graduate_to_skill_file(
            knowledge_entry={"summary": "first"},
            skill_name="r",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )

        # Step 2: capture the prompt the LLM sees on the second call.
        seen = {"prompt": None}

        def capture_llm(prompt: str) -> str:
            seen["prompt"] = prompt
            updated_json = json.dumps({
                "description": "test rule for graduation (updated)",
                "overview": "Now covers two error classes.",
                "planning": "Apply this rule when X or Y.",
                "implementation": "Set Y or Z.",
                "interpretation": "Log line X or Y indicates the rule applies.",
                "validation": "After applying, verify W.",
            })
            return updated_json

        result = graduate_to_skill_file(
            knowledge_entry={"summary": "second"},
            skill_name="r",
            domain="vasp",
            llm_call=capture_llm,
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        assert result["method"] == "updated"
        # The prompt should have been built from the update template.
        assert seen["prompt"].startswith("UPDATE skill=r")
        assert "EXISTING:" in seen["prompt"]
        assert "NEW:" in seen["prompt"]
        # The existing-skill payload is JSON; the original description
        # should be in there.
        assert "test rule for graduation" in seen["prompt"]
        # The merged file uses the LLM's new content.
        from scilink.skills.loader import load_skill
        parsed = load_skill(
            str(tmp_path / "vasp" / "r" / "r.md"), domain="vasp"
        )
        assert "(updated)" in parsed["meta"]["description"]
        assert "two error classes" in parsed.get("overview", "")


# ──────────────────────────────────────────────────────────────
# Loading + prompt-block formatting
# ──────────────────────────────────────────────────────────────

class TestLoadGraduatedSkills:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_graduated_skills("vasp", skills_root=tmp_path) == []

    def test_loads_a_graduated_skill_after_creation(self, tmp_path):
        graduate_to_skill_file(
            knowledge_entry={"summary": "first"},
            skill_name="my_rule",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        skills = load_graduated_skills("vasp", skills_root=tmp_path)
        assert len(skills) == 1
        sk = skills[0]
        assert sk["name"] == "my_rule"
        assert sk["meta"]["description"] == "test rule for graduation"

    def test_skips_dotted_and_underscored_dirs(self, tmp_path):
        graduate_to_skill_file(
            knowledge_entry={"summary": "x"},
            skill_name="real",
            domain="vasp",
            llm_call=_fake_llm(VALID_SKILL_JSON),
            fresh_template=FAKE_FRESH,
            update_template=FAKE_UPDATE,
            skills_root=tmp_path,
        )
        (tmp_path / "vasp" / "_internal").mkdir()
        (tmp_path / "vasp" / ".cache").mkdir()
        skills = load_graduated_skills("vasp", skills_root=tmp_path)
        assert len(skills) == 1
        assert skills[0]["name"] == "real"


class TestFormatGraduatedSkillsBlock:
    def test_empty_list_returns_empty_string(self):
        assert format_graduated_skills_block([]) == ""

    def test_renders_skill_with_description(self):
        skill = {
            "name": "demo",
            "meta": {"description": "demo rule for testing"},
            "planning": "Do X.",
            "implementation": "Set Y = Z.",
            "validation": "Verify W.",
        }
        block = format_graduated_skills_block([skill])
        assert "## LEARNED RULES" in block
        assert "### demo" in block
        assert "demo rule for testing" in block
        assert "#### planning" in block

    def test_default_includes_all_canonical_sections(self):
        """Default section list should include every canonical section
        so newly-graduated rules can't be hidden by an over-restrictive
        filter."""
        skill = {
            "name": "demo",
            "meta": {"description": "x"},
            "overview": "O",
            "planning": "P",
            "analysis": "A",
            "interpretation": "I",
            "validation": "V",
            "implementation": "Im",
        }
        block = format_graduated_skills_block([skill])
        for section in ["overview", "planning", "analysis",
                        "interpretation", "validation", "implementation"]:
            assert f"#### {section}" in block

    def test_section_filter(self):
        skill = {
            "name": "demo",
            "meta": {"description": "x"},
            "planning": "P",
            "implementation": "I",
            "validation": "V",
        }
        block = format_graduated_skills_block(
            [skill], sections=["planning"],
        )
        assert "#### planning" in block
        assert "#### implementation" not in block
        assert "#### validation" not in block


# ── technique frontmatter (live audit 2026-09-26: a consolidated skill had no
# ``technique`` and the technique-driven selector never picked it) ──────────

def test_merge_techniques_normalises_and_dedupes():
    from scilink.skills._shared._graduation import merge_techniques
    out = merge_techniques(["Raman spectroscopy", " raman  spectroscopy ", None, "unknown"],
                           "micro-Raman", ["XRD", "Raman Spectroscopy"])
    assert out == ["Raman spectroscopy", "micro-Raman", "XRD"]
    assert merge_techniques(None, []) == []
    assert len(merge_techniques([f"t{i}" for i in range(20)])) == 8


def test_format_skill_writes_technique_list():
    from scilink.skills._shared._graduation import format_skill_as_markdown
    from scilink.skills.loader import _split_frontmatter
    md = format_skill_as_markdown({"description": "d", "technique": ["Raman", "XRD"],
                                   "overview": "o"})
    meta, _ = _split_frontmatter(md, source="t")
    assert meta["technique"] == ["Raman", "XRD"] and meta["description"] == "d"


def test_graduate_merges_measured_and_model_techniques(tmp_path):
    import json
    from scilink.skills._shared._graduation import graduate_to_skill_file

    def llm(prompt):
        return json.dumps({"description": "d", "technique": ["XRD", "powder diffraction"],
                           "overview": "o", "planning": "p"})
    res = graduate_to_skill_file(
        knowledge_entry={"x": 1}, skill_name="t_skill", domain="curve_fitting",
        llm_call=llm, fresh_template="{skill_name} {domain} {knowledge_text}",
        update_template="{skill_name} {domain} {existing_skill} {new_knowledge}",
        skills_root=tmp_path, extra_meta={"technique": ["xrd", "Powder XRD"], "provisional": True})
    text = open(res["skill_path"]).read()
    from scilink.skills.loader import _split_frontmatter
    meta, _ = _split_frontmatter(text, source="t")
    # measured first, the model's aliases after, no case-duplicates
    assert meta["technique"] == ["xrd", "Powder XRD", "powder diffraction"]
    assert meta["provisional"] is True


def test_consolidate_routes_on_the_records_measurement_technique(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    from scilink.skills._shared import _staging
    from scilink.skills.loader import _split_frontmatter
    for i in range(2):
        _staging.stage_solution("curve_fitting", "peaks", {
            "provenance": "bank_nominated", "model": "one gaussian",
            "measurement_context": {"technique": "Raman spectroscopy" if i else "raman"},
            "session": f"s{i}", "working_script": "print(1)\n"})
    res = _staging.consolidate_technique(
        "curve_fitting", "peaks",
        llm_call=lambda p: json.dumps({"description": "d", "technique": ["micro-Raman"],
                                       "overview": "o"}),
        consolidation_template="{skill_name} {domain} {knowledge_text}",
        update_template="{skill_name} {domain} {existing_skill} {new_knowledge}")
    assert res["status"] == "success"
    meta, _ = _split_frontmatter(open(res["skill_path"]).read(), source="t")
    # the two measured spellings first (staging order), the model's alias last
    assert meta["technique"][-1] == "micro-Raman"
    assert sorted(meta["technique"][:2]) == ["Raman spectroscopy", "raman"]
