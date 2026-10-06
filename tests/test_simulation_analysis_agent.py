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


class TestResolveOutputs:
    """Deck-grounded output resolution: a superset of the static filename match
    that also recovers an output written to a self-named file."""

    def _write_run(self, d):
        (d / "run.lammps").write_text(
            "units real\n"
            "fix stresslog all ave/time 5 1 5 v_pxy v_pxz v_pyz v_temp v_vol "
            "file stress.dat\n"
            "thermo 1000\nrun 5000\n")
        (d / "log.lammps").write_text("Step Temp Press\n0 298 1.0\n")
        (d / "stress.dat").write_text("# TimeStep v_pxy v_pxz v_pyz\n5 0.1 0.2 0.3\n")

    def test_adds_deck_named_output_to_static_floor(self, agent, tmp_path):
        self._write_run(tmp_path)
        agent._llm = lambda prompt: '{"thermo_log": ["stress.dat"]}'
        by = agent.resolve_outputs(str(tmp_path))
        names = {Path(p).name for p in by["thermo_log"]}
        assert "log.lammps" in names        # static floor kept
        assert "stress.dat" in names        # LLM-added, deck-grounded

    def test_no_llm_call_when_nothing_unclassified(self, agent, tmp_path):
        (tmp_path / "log.lammps").write_text("Step Temp\n0 298\n")
        (tmp_path / "vasprun.xml").write_text("<modeling/>")
        called = {"n": 0}
        def _boom(prompt):
            called["n"] += 1
            raise AssertionError("LLM must not be called when static is complete")
        agent._llm = _boom
        by = agent.resolve_outputs(str(tmp_path))
        assert called["n"] == 0
        assert set(by) == {"thermo_log", "dft_output"}

    def test_falls_back_to_static_on_llm_error(self, agent, tmp_path):
        self._write_run(tmp_path)
        def _raise(prompt):
            raise RuntimeError("api down")
        agent._llm = _raise
        by = agent.resolve_outputs(str(tmp_path))
        names = {Path(p).name for p in by.get("thermo_log", [])}
        assert "log.lammps" in names        # static floor survived
        assert "stress.dat" not in names    # the LLM add never happened

    def test_ignores_unknown_kinds_and_missing_paths(self, agent, tmp_path):
        self._write_run(tmp_path)
        agent._llm = lambda p: '{"not_a_kind": ["stress.dat"], "thermo_log": ["ghost.dat"]}'
        by = agent.resolve_outputs(str(tmp_path))
        assert "not_a_kind" not in by
        assert all(Path(p).name != "ghost.dat" for p in by.get("thermo_log", []))
        assert any(Path(p).name == "log.lammps" for p in by["thermo_log"])


class TestInputDecks:
    """The run deck is handed to the analysis as INPUT_DECKS so column identity
    comes from the deck (fix ave/time / variable / compute), not header guessing."""

    def test_inputs_frontmatter_declares_lammps_deck(self, agent):
        pats = agent._input_deck_patterns()
        assert "run.lammps" in pats and "in.*" in pats     # from the lammps skill

    def test_gather_reads_deck_and_skips_dryrun(self, agent, tmp_path):
        (tmp_path / "run.lammps").write_text(
            "fix s all ave/time 5 1 5 v_pxy v_pxz v_pyz file stress.dat\n")
        (tmp_path / "_dryrun").mkdir()
        (tmp_path / "_dryrun" / "run.lammps").write_text("DRYRUN\n")
        decks = agent._gather_input_decks(str(tmp_path))
        assert "run.lammps" in decks and "v_pxy" in decks["run.lammps"]
        assert not any("dryrun" in k.lower() for k in decks)

    def test_run_analysis_passes_input_decks_to_codegen(self, agent, tmp_path, monkeypatch):
        (tmp_path / "log.lammps").write_text("Step Temp\n0 298\n")
        (tmp_path / "run.lammps").write_text(
            "units real\nfix s all ave/time 5 1 5 v_pxy v_pxz v_pyz file stress.dat\n")
        captured = {}
        def fake_compute(task, data_files, *, recipe="", output_type="scalar",
                         input_decks=None, **kw):
            captured["input_decks"] = input_decks
            return {"status": "success", "value": 1.0}
        monkeypatch.setattr(agent, "compute_property", fake_compute)
        monkeypatch.setattr(agent, "_skill_catalog",
                            lambda: [_skill("gk", ["shear_viscosity"], ["thermo_log"])])
        monkeypatch.setattr(agent, "_select_properties", lambda g, e: e)
        agent._llm = lambda p: "{}"          # resolver LLM add -> nothing (static floor)
        agent.run_analysis("compute viscosity", run_dir=str(tmp_path))
        assert captured["input_decks"] and "run.lammps" in captured["input_decks"]
        assert "v_pxy" in captured["input_decks"]["run.lammps"]

    def test_run_analysis_resolves_relative_run_dir_to_absolute(self, agent, tmp_path, monkeypatch):
        import os
        (tmp_path / "log.lammps").write_text("Step Temp\n0 298\n")
        captured = {}
        def fake_compute(task, data_files, **kw):
            captured["data_files"] = data_files
            return {"status": "success", "value": 1.0}
        monkeypatch.setattr(agent, "compute_property", fake_compute)
        monkeypatch.setattr(agent, "_skill_catalog",
                            lambda: [_skill("gk", ["shear_viscosity"], ["thermo_log"])])
        monkeypatch.setattr(agent, "_select_properties", lambda g, e: e)
        agent._llm = lambda p: "{}"
        monkeypatch.chdir(tmp_path.parent)
        agent.run_analysis("x", run_dir=tmp_path.name)        # RELATIVE run_dir
        assert captured["data_files"]
        assert all(os.path.isabs(v) for v in captured["data_files"].values())


class TestInputDecksReachCodegen:
    """The blocker Maxim flagged: INPUT_DECKS must reach the code-gen prompts,
    not just the runtime preamble — otherwise the model keeps header-guessing."""

    def _capture_llm(self, agent):
        box = {}
        def fake(prompt):
            box["prompt"] = prompt
            return "print('{\"status\": \"success\", \"value\": 1.0}')"
        agent._llm = fake
        return box

    def test_generate_code_prompt_shows_input_decks(self, agent):
        box = self._capture_llm(agent)
        agent._generate_code(
            task="shear viscosity", data_files={"stress.dat": "/run/stress.dat"},
            recipe="read the pressure tensor", packages=["numpy"],
            input_decks={"run.lammps":
                         "fix s all ave/time 5 1 5 v_pxy v_pxz v_pyz file stress.dat\n"})
        p = box["prompt"]
        assert "INPUT_DECKS" in p and "v_pxy" in p and "run.lammps" in p

    def test_refine_code_prompt_shows_input_decks(self, agent):
        box = self._capture_llm(agent)
        agent._refine_code(
            code="bad()", error_info={"message": "boom"}, task="t",
            recipe="", packages=["numpy"],
            input_decks={"run.lammps": "variable pxy equal pxy\n"})
        assert "INPUT_DECKS" in box["prompt"] and "variable pxy equal pxy" in box["prompt"]


class TestResolveOutputsReviewFixes:
    def test_gather_decks_ignores_dryrun_in_parent_path(self, agent, tmp_path):
        run = tmp_path / "dryrun_vs_prod" / "run1"        # parent contains "dryrun"
        run.mkdir(parents=True)
        (run / "run.lammps").write_text(
            "fix s all ave/time 5 1 5 v_pxy v_pxz v_pyz file stress.dat\n")
        decks = agent._gather_input_decks(str(run))
        assert "run.lammps" in decks                       # not skipped by the parent name

    def test_gather_marks_truncation(self, agent, tmp_path):
        (tmp_path / "run.lammps").write_text("units real\n" * 5000)
        decks = agent._gather_input_decks(str(tmp_path), max_chars=800)
        assert decks["run.lammps"].rstrip().endswith("[deck truncated]")

    def test_no_llm_call_when_only_decks_unclassified(self, agent, tmp_path):
        (tmp_path / "log.lammps").write_text("Step Temp\n0 298\n")     # classified
        (tmp_path / "run.lammps").write_text("units real\nrun 10\n")   # a deck (not a candidate)
        called = {"n": 0}
        def boom(p):
            called["n"] += 1
            raise AssertionError("no model call when only input decks are unclassified")
        agent._llm = boom
        by = agent.resolve_outputs(str(tmp_path))
        assert called["n"] == 0 and "thermo_log" in by

    def test_rejects_model_path_outside_run_dir(self, agent, tmp_path):
        (tmp_path / "run.lammps").write_text(
            "fix s all ave/time 5 1 5 v_pxy v_pxz v_pyz file stress.dat\n")
        (tmp_path / "stress.dat").write_text("# TimeStep v_pxy v_pxz v_pyz\n5 0.1 0.2 0.3\n")
        (tmp_path.parent / "evil.log").write_text("outside the run\n")
        agent._llm = lambda p: '{"thermo_log": ["../evil.log", "stress.dat"]}'
        by = agent.resolve_outputs(str(tmp_path))
        allp = [p for paths in by.values() for p in paths]
        assert any(Path(p).name == "stress.dat" for p in allp)         # legit, kept
        assert all("evil.log" not in p for p in allp)                  # escaped, rejected
