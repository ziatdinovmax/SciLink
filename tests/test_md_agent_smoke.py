"""
Local smoke tests for the md_agent branch.

These exercise the post-relocation wiring of:
  - MDSimulationAgent(skill="lammps") — Tier C bundle resolution,
    TOOL_REGISTRY auto-wiring, skill section access, tools-module path
    of analyze_system on a real LAMMPS data file.

No LLM calls are made; the agent's analyze_system() prefers the
tools-module fast path, which is enough to validate the relocation.
"""

import os
import tempfile
from pathlib import Path


CU_DATA = """\
LAMMPS data file for bulk Cu (atom_style atomic)

4 atoms
1 atom types

0.0 3.615 xlo xhi
0.0 3.615 ylo yhi
0.0 3.615 zlo zhi

Masses

1 63.546 # Cu

Atoms # atomic

1 1 0.000 0.000 0.000
2 1 1.808 1.808 0.000
3 1 1.808 0.000 1.808
4 1 0.000 1.808 1.808
"""


def _agent_kwargs():
    """Construction-only kwargs — no real LLM calls will be made."""
    return dict(api_key="sk-smoke-not-real", model_name="gpt-4o-mini")


def test_md_agent_assembly():
    from scilink.agents.sim_agents import MDSimulationAgent

    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / "cu_bulk.data"
        data_path.write_text(CU_DATA)

        agent = MDSimulationAgent(working_dir=td, skill="lammps", **_agent_kwargs())

        assert agent.skill_name == "lammps", (
            f"skill_name should be 'lammps', got {agent.skill_name!r}"
        )
        assert agent.tools_module is not None, (
            "tools_module should be wired via TOOL_REGISTRY['lammps']"
        )
        assert agent.skill_sections is not None
        for section in ("planning", "implementation", "validation"):
            content = agent._get_skill_context(section=section)
            assert content, f"skill section {section!r} should be non-empty"

        info = agent.analyze_system(str(data_path))
        assert info["atom_count"] == 4, f"expected 4 atoms, got {info}"
        assert "Cu" in info.get("elements", []), (
            f"expected Cu in elements, got {info}"
        )

        from scilink.skills.molecular_dynamics.lammps import lammps
        assert agent.tools_module is lammps, (
            "tools_module should be the relocated bundle module"
        )

        print("  MDSimulationAgent(skill='lammps'): OK")
        print(f"    skill_name      = {agent.skill_name}")
        print(f"    tools_module    = {agent.tools_module.__name__}")
        print(f"    analyze_system  = {info['atom_count']} atoms, {info['elements']}")


if __name__ == "__main__":
    print("=== smoke: MDSimulationAgent(skill='lammps') ===")
    test_md_agent_assembly()
    print()
    print("Smoke passed.")
