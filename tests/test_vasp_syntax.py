"""Deterministic-layer tests for the VASP INCAR syntax tools.

Covers the engine-native syntax check that lives in the VASP skill
bundle (``scilink/skills/periodic_dft/vasp/vasp_syntax.py``): unrecognized
tag detection, closest-match suggestions, high/low confidence rating,
the in-place auto-rename, the engine-neutral ``check_input_syntax``
entry point, and resolution through the skill registry. No LLM, no VASP
run, no API key required.

Requires pymatgen for the canonical tag list; tests that depend on it
skip cleanly when it is unavailable.
"""

import pytest

pytest.importorskip("pymatgen")

from scilink.skills.periodic_dft.vasp.vasp_syntax import (
    apply_incar_syntax_fixes,
    apply_input_syntax_fixes,
    check_incar_syntax,
    check_input_syntax,
)


CLEAN_INCAR = "ISPIN = 2\nENCUT = 400\nGGA = PE\nISMEAR = 1\nSIGMA = 0.1\n"
TYPO_INCAR = "ISPN = 2\nENCUT = 400\nGGA = PE\n"  # ISPN should be ISPIN


# ── check_incar_syntax ────────────────────────────────────────

class TestCheckIncarSyntax:
    def test_clean_incar_has_no_issues(self):
        assert check_incar_syntax(CLEAN_INCAR) == []

    def test_typo_is_flagged_with_high_confidence_suggestion(self):
        issues = check_incar_syntax(TYPO_INCAR)
        assert len(issues) == 1
        issue = issues[0]
        assert issue["tag"] == "ISPN"
        assert issue["suggested"] == "ISPIN"
        assert issue["confidence"] == "high"
        assert issue["category"] == "incar_tag"

    def test_prose_without_assignments_returns_empty(self):
        # No ``key = value`` lines means no tags to validate — the pass
        # is specifically for valid-looking INCARs with a fake tag.
        assert check_incar_syntax("just some notes\nnothing assigned here\n") == []

    def test_empty_content_returns_empty(self):
        assert check_incar_syntax("") == []


# ── check_input_syntax (engine-neutral entry point) ───────────

class TestCheckInputSyntax:
    def test_selects_incar_from_mapping(self):
        issues = check_input_syntax({"INCAR": TYPO_INCAR, "KPOINTS": "auto"})
        assert len(issues) == 1
        assert issues[0]["tag"] == "ISPN"

    def test_no_incar_present_returns_empty(self):
        assert check_input_syntax({"KPOINTS": "auto", "POSCAR": "..."}) == []

    def test_clean_incar_in_mapping_returns_empty(self):
        assert check_input_syntax({"INCAR": CLEAN_INCAR}) == []


# ── apply_incar_syntax_fixes ──────────────────────────────────

class TestApplyIncarSyntaxFixes:
    def test_high_confidence_typo_is_renamed(self):
        fixed, applied = apply_incar_syntax_fixes(TYPO_INCAR)
        assert "ISPIN = 2" in fixed
        assert "ISPN = 2" not in fixed
        assert len(applied) == 1
        assert applied[0]["renamed_from"] == "ISPN"
        assert applied[0]["renamed_to"] == "ISPIN"

    def test_clean_incar_is_unchanged(self):
        fixed, applied = apply_incar_syntax_fixes(CLEAN_INCAR)
        assert fixed == CLEAN_INCAR
        assert applied == []

    def test_value_tokens_are_not_renamed(self):
        # A value token that happens to share a tag spelling must not be
        # rewritten — only left-hand-side assignments are touched.
        incar = "LDAUL = 3 3 -1\nISPIN = 2\n"
        fixed, applied = apply_incar_syntax_fixes(incar)
        assert fixed == incar
        assert applied == []


# ── apply_input_syntax_fixes (engine-neutral entry point) ─────

class TestApplyInputSyntaxFixes:
    def test_fixes_incar_in_mapping_and_preserves_others(self):
        files = {"INCAR": TYPO_INCAR, "KPOINTS": "auto"}
        fixed_files, applied = apply_input_syntax_fixes(files)
        assert "ISPIN = 2" in fixed_files["INCAR"]
        assert fixed_files["KPOINTS"] == "auto"
        assert len(applied) == 1
        # Original mapping is not mutated in place.
        assert "ISPN = 2" in files["INCAR"]

    def test_no_incar_present_returns_input_unchanged(self):
        files = {"KPOINTS": "auto", "POSCAR": "..."}
        fixed_files, applied = apply_input_syntax_fixes(files)
        assert fixed_files == files
        assert applied == []

    def test_clean_incar_returns_unchanged(self):
        files = {"INCAR": CLEAN_INCAR}
        fixed_files, applied = apply_input_syntax_fixes(files)
        assert fixed_files["INCAR"] == CLEAN_INCAR
        assert applied == []


# ── registry resolution ───────────────────────────────────────

class TestRegistryResolution:
    def test_check_input_syntax_resolves_via_registry(self):
        from scilink.skills._shared._registry import get_tool_function
        fn = get_tool_function("check_input_syntax", active_skills=["vasp"])
        issues = fn(input_files={"INCAR": TYPO_INCAR})
        assert len(issues) == 1
        assert issues[0]["suggested"] == "ISPIN"

    def test_unknown_skill_raises_lookup_error(self):
        from scilink.skills._shared._registry import get_tool_function
        with pytest.raises(LookupError):
            get_tool_function("check_input_syntax", active_skills=["nonexistent_engine"])

    def test_apply_input_syntax_fixes_resolves_via_registry(self):
        from scilink.skills._shared._registry import get_tool_function
        fn = get_tool_function("apply_input_syntax_fixes", active_skills=["vasp"])
        fixed_files, applied = fn(input_files={"INCAR": TYPO_INCAR})
        assert "ISPIN = 2" in fixed_files["INCAR"]
        assert len(applied) == 1
