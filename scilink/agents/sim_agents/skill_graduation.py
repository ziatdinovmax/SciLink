"""Backward-compatibility shim.

The graduation helpers moved to ``scilink/skills/_shared/_graduation.py``
so that non-sim code paths (analysis, planning, meta, CLI) can import them
without pulling in ``ase`` (which ``scilink.agents.sim_agents`` hard-imports).
This module re-exports the public API unchanged; existing
``from .skill_graduation import ...`` imports in the sim agents keep working.
"""
from ...skills._shared._graduation import (  # noqa: F401
    GRADUATED_SKILLS_DIR,
    WORD_COUNT_WARN_THRESHOLD,
    _SECTION_KEYS,
    _format_knowledge,
    _read_skill_as_dict,
    KnowledgeStore,
    parse_json_response,
    format_skill_as_markdown,
    graduate_to_skill_file,
    load_graduated_skills,
    format_graduated_skills_block,
)

__all__ = [
    "GRADUATED_SKILLS_DIR",
    "WORD_COUNT_WARN_THRESHOLD",
    "KnowledgeStore",
    "parse_json_response",
    "format_skill_as_markdown",
    "graduate_to_skill_file",
    "load_graduated_skills",
    "format_graduated_skills_block",
]
