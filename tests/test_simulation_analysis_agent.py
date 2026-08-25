"""SimulationAnalysisAgent: output classification, the availability gate, and the
end-to-end pipeline (skill catalog + LLM monkeypatched, real sandbox execution)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.simulation_analysis_agent import (  # noqa: E402
    SimulationAnalysisAgent)


def _skill(name, computes, requires, impl="recipe"):
    return {"name": name, "implementation": impl,
            "meta": {"computes": computes, "requires": requires}}


@pytest.fixture
def agent(tmp_path):
    return SimulationAnalysisAgent(output_dir=str(tmp_path / "out"), api_key="test-key")


class TestClassify:
    def test_recognizes_kinds(self, agent, tmp_path):
        (tmp_path / "prod.lammpstrj").write_text("x")
        (tmp_path / "log.lammps").write_text("x")
        (tmp_path / "vasprun.xml").write_text("x")
        (tmp_path / "notes.txt").write_text("x")   # ignored
        by = agent.classify_outputs(str(tmp_path))
        assert set(by) == {"trajectory", "thermo_log", "dft_output"}
        assert by["trajectory"][0].endswith("prod.lammpstrj")

    def test_empty_dir(self, agent, tmp_path):
        assert agent.classify_outputs(str(tmp_path)) == {}

    def test_format_map_is_engine_declared(self, agent):
        # Patterns come from engine skills' `outputs:` frontmatter — the agent
        # itself names no filenames, so adding an engine is a skill-only change.
        fmt = agent._output_format_map()
        assert "vasprun.xml" in fmt.get("dft_output", set())      # from vasp skill
        assert "log.lammps" in fmt.get("thermo_log", set())       # from lammps skill
        assert "lammpstrj" in fmt.get("trajectory", set())        # from lammps skill


class TestAvailabilityGate:
    def test_gates_by_required_data(self, agent):
        cat = [_skill("visc", ["shear_viscosity"], ["trajectory"]),
               _skill("bandgap", ["band_gap"], ["dft_output"]),
               _skill("always", ["energy"], [])]
        names = {s["name"] for s in agent.eligible_skills({"trajectory"}, catalog=cat)}
        assert names == {"visc", "always"}          # DFT skill gated out

    def test_overlap_resolves_by_presence(self, agent):
        cat = [_skill("elastic_md", ["elastic_constants"], ["trajectory"]),
               _skill("elastic_dft", ["elastic_constants"], ["dft_output"])]
        md = {s["name"] for s in agent.eligible_skills({"trajectory"}, catalog=cat)}
        dft = {s["name"] for s in agent.eligible_skills({"dft_output"}, catalog=cat)}
        assert md == {"elastic_md"} and dft == {"elastic_dft"}


class TestPipeline:
    def test_end_to_end(self, agent, tmp_path, monkeypatch):
        (tmp_path / "prod.lammpstrj").write_text("dummy")
        cat = [_skill("viscosity_greenkubo", ["shear_viscosity"], ["trajectory"])]
        monkeypatch.setattr(agent, "_skill_catalog", lambda: cat)

        def fake_llm(prompt):
            if "AVAILABLE TECHNIQUES" in prompt:
                return '{"skills": ["viscosity_greenkubo"]}'
            if "physically plausible" in prompt:
                return '{"plausible": true, "reasoning": "sane"}'
            return 'import json; print(json.dumps({"status":"success","value":0.9,"units":"cP"}))'

        agent._llm = fake_llm
        r = agent.run_analysis("compute the shear viscosity", run_dir=str(tmp_path))
        assert r["status"] == "success"
        assert r["skills_used"] == ["viscosity_greenkubo"]
        assert r["data_kinds"] == ["trajectory"]
        assert r["results"]["shear_viscosity"]["value"] == 0.9
        assert r["results"]["shear_viscosity"]["verification"]["plausible"] is True

    def test_no_output_is_error(self, agent, tmp_path):
        r = agent.run_analysis("anything", run_dir=str(tmp_path))
        assert r["status"] == "error" and r["results"] == {}


class TestRealSkills:
    """The on-disk simulation_analysis skills load and gate through the real loader."""

    def test_viscosity_skill_loads_and_gates(self, agent):
        cat = agent._skill_catalog()
        visc = [c for c in cat if c["name"] == "viscosity_greenkubo"]
        assert visc, "viscosity_greenkubo skill not discovered"
        meta = visc[0]["meta"]
        assert meta["computes"] == ["shear_viscosity"]
        assert meta["requires"] == ["thermo_log"]
        assert visc[0].get("implementation")           # recipe present
        # availability gate: eligible only when its required data is on disk
        elig = lambda kinds: {c["name"] for c in agent.eligible_skills(kinds, catalog=cat)}
        assert "viscosity_greenkubo" in elig({"thermo_log"})
        assert "viscosity_greenkubo" not in elig({"trajectory"})

    def test_t1_skill_loads_and_gates(self, agent):
        cat = agent._skill_catalog()
        t1 = [c for c in cat if c["name"] == "t1_relaxation"]
        assert t1, "t1_relaxation skill not discovered"
        meta = t1[0]["meta"]
        assert meta["computes"] == ["t1_relaxation"]
        assert meta["requires"] == ["trajectory"]
        assert t1[0].get("implementation")
        elig = lambda kinds: {c["name"] for c in agent.eligible_skills(kinds, catalog=cat)}
        assert "t1_relaxation" in elig({"trajectory"})
        assert "t1_relaxation" not in elig({"thermo_log"})

    def test_skills_gate_independently(self, agent):
        cat = agent._skill_catalog()
        elig = lambda kinds: {c["name"] for c in agent.eligible_skills(kinds, catalog=cat)}
        targets = {"viscosity_greenkubo", "t1_relaxation"}
        # a run with both data kinds makes both eligible
        assert targets.issubset(elig({"trajectory", "thermo_log"}))
        # a trajectory-only run: T1 eligible, viscosity not
        assert elig({"trajectory"}) & targets == {"t1_relaxation"}
        # a thermo-only run: viscosity eligible, T1 not
        assert elig({"thermo_log"}) & targets == {"viscosity_greenkubo"}

    def test_forward_model_domain_served(self, agent):
        """The forward_models domain is served by the same agent + gated identically."""
        cat = agent._skill_catalog()
        sf = [c for c in cat if c["name"] == "structure_factor"]
        assert sf, "structure_factor (forward_models) skill not discovered"
        meta = sf[0]["meta"]
        assert meta["computes"] == ["structure_factor"]
        assert meta["requires"] == ["trajectory"]
        assert meta.get("output") == "curve"           # routes compute_property
        elig = lambda kinds: {c["name"] for c in agent.eligible_skills(kinds, catalog=cat)}
        assert "structure_factor" in elig({"trajectory"})
        assert "structure_factor" not in elig({"thermo_log"})

    def test_run_analysis_threads_output_type(self, agent, tmp_path, monkeypatch):
        """A curve skill's `output:` frontmatter reaches compute_property."""
        (tmp_path / "prod.lammpstrj").write_text("x")   # a trajectory data kind
        curve_skill = {"name": "structure_factor", "implementation": "recipe",
                       "meta": {"computes": ["structure_factor"],
                                "requires": ["trajectory"], "output": "curve"}}
        monkeypatch.setattr(agent, "_skill_catalog", lambda: [curve_skill])
        monkeypatch.setattr(agent, "_select_properties", lambda goal, elig: elig)
        captured = {}
        monkeypatch.setattr(
            agent, "compute_property",
            lambda **kw: captured.update(kw) or {"status": "success"})
        agent.run_analysis("compute S(q)", run_dir=str(tmp_path))
        assert captured.get("output_type") == "curve"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
