"""OpenFF force-field skill: registration, discovery, guarded deps, skill prose.

Offline (no openff/interchange): the build_interchange tool's module imports
(its openff imports are lazy), it registers a TOOL_SPEC discoverable via the
registry for the ``openff`` skill, calling it without the ``[ff]`` deps raises an
actionable error, and the skill markdown parses into the fixed section
vocabulary. The actual Interchange build + LAMMPS export is verified live in the
``[ff]`` env.
"""

from __future__ import annotations

import pytest


def test_build_interchange_module_and_toolspec():
    from scilink.skills.force_field.openff.build_interchange import (
        build_interchange,  # noqa: F401
        TOOL_SPEC,
    )
    assert TOOL_SPEC.name == "build_interchange"
    assert "simulation" in TOOL_SPEC.agents
    assert "coordinates_file" in TOOL_SPEC.required


def test_registry_resolves_build_interchange():
    from scilink.skills._shared._registry import get_tool_function
    fn = get_tool_function("build_interchange", active_skills=["openff"])
    assert fn.__name__ == "build_interchange"


def test_build_interchange_guards_missing_ff_deps():
    from scilink.skills.force_field.openff.build_interchange import build_interchange
    with pytest.raises(ImportError, match=r"scilink\[ff\]"):
        build_interchange([{"name": "w", "smiles": "O", "count": 1}],
                          "/tmp/does_not_exist.extxyz")


def test_openff_skill_loads_with_sections():
    from scilink.skills.loader import load_skill
    sk = load_skill("openff", domain="force_field")
    present = [s for s in ("overview", "planning", "implementation",
                           "interpretation", "validation") if sk.get(s)]
    # All five fixed sections authored.
    assert set(present) == {"overview", "planning", "implementation",
                            "interpretation", "validation"}
