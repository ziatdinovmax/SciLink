"""Tests for the persistent-memory store (PR1).

Covers:
  - the loader's discovery of the persistent graduated-skills store
    (precedence ordering, de-dup, SCILINK_HOME relocation);
  - the graduation helper's extra-meta (provisional) passthrough and
    its byte-identical legacy output for description-only callers;
  - the ase-free import guarantee for the relocated helper;
  - the sim-side back-compat shim.

No real LLM calls — the graduation helper takes the LLM as a callable.
"""

import json
import sys

import pytest

from scilink.skills import loader


@pytest.fixture(autouse=True)
def _enable_memory(monkeypatch):
    """Persistent memory is opt-in (off by default); these tests exercise it, so
    enable the master switch. Off-switch behavior is covered in TestMemorySwitch
    (which overrides this)."""
    monkeypatch.setenv("SCILINK_MEMORY", "1")
from scilink.skills._shared._graduation import (
    format_skill_as_markdown,
    graduate_to_skill_file,
)


def _fake_llm(response: str):
    def _fn(prompt: str) -> str:
        return response
    return _fn


VALID_JSON = json.dumps({
    "description": "distilled curve-fit recipe",
    "overview": "ov",
    "implementation": "recipe\n\n```python\nprint(1)\n```",
})


# ──────────────────────────────────────────────────────────────
# Loader: persistent store discovery
# ──────────────────────────────────────────────────────────────

class TestSkillRootsDiscovery:
    def test_home_redirect_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        assert loader.graduated_skills_dir() == tmp_path / "graduated_skills"

    def test_graduated_dir_appears_in_roots_after_creation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_SKILLS_PATH", raising=False)
        gd = loader.graduated_skills_dir()
        # Absent until it exists on disk.
        assert gd.resolve() not in [r.resolve() for r in loader._skill_roots()]
        (gd / "curve_fitting" / "foo").mkdir(parents=True)
        (gd / "curve_fitting" / "foo" / "foo.md").write_text(
            "---\ndescription: t\n---\n## overview\nhi\n"
        )
        roots = [r.resolve() for r in loader._skill_roots()]
        assert gd.resolve() in roots
        assert "foo" in loader.list_all_skills().get("curve_fitting", [])

    def test_precedence_order_env_then_graduated_then_builtin(self, tmp_path, monkeypatch):
        env_root = tmp_path / "env_skills"
        (env_root / "curve_fitting" / "x").mkdir(parents=True)
        (env_root / "curve_fitting" / "x" / "x.md").write_text("---\ndescription: e\n---\n## overview\ne\n")
        home = tmp_path / "home"
        gd = home / "graduated_skills"
        (gd / "curve_fitting" / "y").mkdir(parents=True)
        (gd / "curve_fitting" / "y" / "y.md").write_text("---\ndescription: g\n---\n## overview\ng\n")

        monkeypatch.setenv("SCILINK_HOME", str(home))
        monkeypatch.setenv("SCILINK_SKILLS_PATH", str(env_root))

        roots = [r.resolve() for r in loader._skill_roots()]
        idx_env = roots.index(env_root.resolve())
        idx_grad = roots.index(gd.resolve())
        idx_builtin = roots.index(loader._SKILLS_DIR.resolve())
        assert idx_env < idx_grad < idx_builtin

    def test_dedupes_when_home_also_in_skills_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        gd = home / "graduated_skills"
        (gd / "curve_fitting" / "y").mkdir(parents=True)
        (gd / "curve_fitting" / "y" / "y.md").write_text("---\ndescription: g\n---\n## overview\ng\n")
        monkeypatch.setenv("SCILINK_HOME", str(home))
        # Point SCILINK_SKILLS_PATH at the same graduated dir.
        monkeypatch.setenv("SCILINK_SKILLS_PATH", str(gd))
        roots = [r.resolve() for r in loader._skill_roots()]
        assert roots.count(gd.resolve()) == 1


# ──────────────────────────────────────────────────────────────
# Graduation: provisional / extra-meta passthrough
# ──────────────────────────────────────────────────────────────

class TestExtraMeta:
    def test_format_passthrough_allowlist(self):
        out = format_skill_as_markdown({
            "description": "d",
            "provisional": True,
            "provenance": "t2_autodistill",
            "r_squared": 0.97,
            "not_allowed": "should be dropped",
            "overview": "o",
        })
        assert "provisional: true" in out
        assert "provenance: t2_autodistill" in out
        assert "r_squared: 0.97" in out
        assert "not_allowed" not in out

    def test_description_only_is_byte_identical_to_legacy(self):
        # The legacy formatter emitted exactly this for a description-only
        # input; the allowlist must not perturb it.
        out = format_skill_as_markdown({"description": "d", "overview": "o"})
        assert out == "---\ndescription: d\n---\n\n## overview\n\no\n"

    def test_graduate_writes_provisional_frontmatter(self, tmp_path):
        result = graduate_to_skill_file(
            knowledge_entry={"summary": "x", "script": "print(1)"},
            skill_name="auto_test",
            domain="curve_fitting",
            llm_call=_fake_llm(VALID_JSON),
            fresh_template="{skill_name} {domain} {knowledge_text}",
            update_template="{skill_name} {existing_skill} {new_knowledge}",
            skills_root=tmp_path,
            extra_meta={"provisional": True, "provenance": "t2_autodistill", "r_squared": 0.97},
        )
        parsed = loader.load_skill(result["skill_path"], domain="curve_fitting")
        assert parsed["meta"].get("provisional") is True
        assert parsed["meta"].get("provenance") == "t2_autodistill"
        assert "print(1)" in parsed["implementation"]


# ──────────────────────────────────────────────────────────────
# ase-free import + shim back-compat
# ──────────────────────────────────────────────────────────────

class TestImportSafety:
    def test_graduation_helper_is_ase_free(self):
        """The relocated helper must import without ase, which the sim
        package hard-requires. Run in a subprocess that blocks ase, so we
        don't pollute this interpreter's module cache."""
        import subprocess
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            class Blocker:
                def find_spec(self, name, path, target=None):
                    if name == 'ase' or name.startswith('ase.'):
                        raise ImportError('ase blocked')
                    return None
            sys.meta_path.insert(0, Blocker())
            from scilink.skills._shared._graduation import graduate_to_skill_file
            from scilink.skills import loader
            assert 'ase' not in sys.modules, 'ase was imported'
            print('OK')
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_sim_shim_reexports_helper(self):
        from scilink.agents.sim_agents.skill_graduation import (
            graduate_to_skill_file as shim_fn,
        )
        from scilink.skills._shared._graduation import graduate_to_skill_file as real_fn
        assert shim_fn is real_fn


# ──────────────────────────────────────────────────────────────
# Routing: provisional skills excluded from the run_analysis menu
# ──────────────────────────────────────────────────────────────

class TestRoutingFilter:
    def _make_skill(self, gd, name, *, provisional):
        d = gd / "curve_fitting" / name
        d.mkdir(parents=True)
        fm = "description: a test skill\n"
        if provisional:
            fm += "provisional: true\n"
        (d / f"{name}.md").write_text(f"---\n{fm}---\n\n## overview\nbody\n")

    def test_provisional_excluded_but_loadable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_SKILLS_PATH", raising=False)
        gd = loader.graduated_skills_dir()
        self._make_skill(gd, "prov_skill", provisional=True)
        self._make_skill(gd, "normal_skill", provisional=False)

        from scilink.agents.exp_agents.analysis_orchestrator_tools import (
            _build_skill_description,
        )
        blurb = _build_skill_description()
        assert "normal_skill" in blurb
        assert "prov_skill" not in blurb

        # Still explicitly loadable.
        parsed = loader.load_skill("prov_skill", domain="curve_fitting")
        assert parsed["meta"].get("provisional") is True


# ──────────────────────────────────────────────────────────────
# T=2 stage-only hook (curve fitting)
# ──────────────────────────────────────────────────────────────

class TestT2StageHook:
    def _fake_agent(self, home, model):
        import logging
        import types
        from pathlib import Path

        a = types.SimpleNamespace()
        a.model = model
        a.logger = logging.getLogger("t2test")
        a.output_dir = Path(home) / "sess"
        a.output_dir.mkdir(parents=True, exist_ok=True)
        a.generation_config = None
        a.safety_settings = None
        return a

    def _hot_state(self):
        return {
            "locked_fitting_config": {"physical_model": "3 Gaussians", "fitting_strategy": "seq"},
            "skills_loaded": [],
            "series_results": [{
                "index": 0, "name": "s0", "success": True,
                "model_type": "2 Voigt + exp tail",
                "fit_quality": {"r_squared": 0.991},
                "deviation_note": "switched model",
                "script": "import numpy as np\n# VERBATIM_MARKER\nprint('fit')\n",
                "quality_history": {"approved": True, "verification_iterations": [
                    {"annealing_level": 0}, {"annealing_level": 2}]},
            }],
        }

    class _Model:
        # The only LLM call in the stage-only hook is the technique label.
        def generate_content(self, contents=None, generation_config=None, safety_settings=None):
            import types
            r = types.SimpleNamespace()
            r.text = "voigt_plus_exp_tail"
            return r

    def test_stages_solution_no_skill_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        from scilink.skills._shared import _memory, _staging

        agent = self._fake_agent(str(tmp_path), self._Model())
        out = CurveFittingAgent._maybe_stage_t2_solutions(agent, self._hot_state())
        assert out, "expected a staged solution id"
        # Staged, not graduated:
        assert _memory.list_memory() == []
        staged = _staging.list_staged("curve_fitting")
        assert len(staged) == 1
        assert staged[0]["technique"] == "voigt_plus_exp_tail"
        assert "VERBATIM_MARKER" in staged[0]["working_script"]

    def test_no_stage_when_not_hot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["verification_iterations"] = [
            {"annealing_level": 0}, {"annealing_level": 1}]
        assert CurveFittingAgent._maybe_stage_t2_solutions(agent, state) == []

    def test_stage_failure_is_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

        class Boom:
            def generate_content(self, *a, **k):
                raise RuntimeError("boom")

        agent = self._fake_agent(str(tmp_path), Boom())
        assert CurveFittingAgent._maybe_stage_t2_solutions(agent, self._hot_state()) == []

    def test_opt_out_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.setenv("SCILINK_T2_AUTODISTILL", "0")
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        assert CurveFittingAgent._maybe_stage_t2_solutions(agent, self._hot_state()) == []

    def test_no_stage_when_below_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["approved"] = False
        assert CurveFittingAgent._maybe_stage_t2_solutions(agent, state) == []


# ──────────────────────────────────────────────────────────────
# Meta-agent review tool (PR3b)
# ──────────────────────────────────────────────────────────────

class TestMetaReviewTool:
    def _seed_provisional(self):
        from scilink.skills._shared._graduation import graduate_to_skill_file
        graduate_to_skill_file(
            knowledge_entry={"summary": "s"},
            skill_name="auto_demo_x", domain="curve_fitting",
            llm_call=lambda p: json.dumps({"description": "d", "overview": "o", "analysis": "recipe"}),
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}",
            extra_meta={"provisional": True, "provenance": "t2_autodistill", "r_squared": 0.99},
        )

    def _tools(self):
        import types
        from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
        # Closures only touch the orchestrator at call time, not registration.
        return MetaOrchestratorTools(types.SimpleNamespace())

    def test_review_tool_registered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        assert "review_distilled_skills" in self._tools().functions_map

    def test_list_show_promote_discard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        self._seed_provisional()
        fn = self._tools().functions_map["review_distilled_skills"]

        listed = json.loads(fn(action="list"))
        assert len(listed["provisional_skills"]) == 1

        shown = json.loads(fn(action="show", skill="curve_fitting/auto_demo_x"))
        assert "recipe" in shown["markdown"]

        promoted = json.loads(fn(action="promote", skill="curve_fitting/auto_demo_x"))
        assert promoted["status"] == "success"
        assert _memory.list_memory(provisional=True) == []
        assert len(_memory.list_memory(provisional=False)) == 1

        discarded = json.loads(fn(action="discard", skill="curve_fitting/auto_demo_x"))
        assert discarded["status"] == "success"
        assert _memory.list_memory() == []

    def test_meta_tools_import_is_ase_free(self):
        import subprocess, sys, textwrap
        script = textwrap.dedent("""
            import sys
            class B:
                def find_spec(self, n, p, t=None):
                    if n == 'ase' or n.startswith('ase.'): raise ImportError('blocked')
                    return None
            sys.meta_path.insert(0, B())
            from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
            assert 'ase' not in sys.modules
            print('OK')
        """)
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert p.returncode == 0 and "OK" in p.stdout, p.stderr


# ──────────────────────────────────────────────────────────────
# Hot-deviation stamping (controllers) — makes a hot success "novel"
# ──────────────────────────────────────────────────────────────

class TestHotDeviationStamp:
    def test_curve_fitting_stamp(self):
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController as U,
        )
        c = U.__new__(U)
        hot = len(U._CONSTRAINT_ANNEALING_SCHEDULE) - 1
        # hot success, no note -> synthesized
        r = {"success": True, "model_type": "2 Voigt", "_produced_at_level": hot}
        c._stamp_hot_deviation(r)
        assert r.get("deviation_note")
        # existing note preserved
        r2 = {"success": True, "_produced_at_level": hot, "deviation_note": "LLM note"}
        c._stamp_hot_deviation(r2)
        assert r2["deviation_note"] == "LLM note"
        # T=0 best -> no false positive
        r3 = {"success": True, "_produced_at_level": 0}
        c._stamp_hot_deviation(r3)
        assert not r3.get("deviation_note")
        # no level recorded / failed -> no note
        assert not _stamped(c, {"success": True})
        assert not _stamped(c, {"success": False, "_produced_at_level": hot})

    def test_image_stamp(self):
        from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
            UnifiedImageProcessingController as U,
        )
        c = U.__new__(U)
        hot = len(U._CONSTRAINT_ANNEALING_SCHEDULE) - 1
        r = {"success": True, "analysis_type": "atom-finder", "_produced_at_level": hot}
        c._stamp_hot_deviation(r)
        assert r.get("plan_deviation_summary")
        r0 = {"success": True, "_produced_at_level": 0}
        c._stamp_hot_deviation(r0)
        assert not r0.get("plan_deviation_summary")


def _stamped(controller, result):
    controller._stamp_hot_deviation(result)
    return result.get("deviation_note")


# ──────────────────────────────────────────────────────────────
# Image-agent T=2 stage-only hook (mirror of curve fitting)
# ──────────────────────────────────────────────────────────────

class TestImageT2Stage:
    class _Model:
        def generate_content(self, contents=None, generation_config=None, safety_settings=None):
            import types
            r = types.SimpleNamespace()
            r.text = "watershed_grain_segmentation"  # technique label
            return r

    def _fake_agent(self, home, model):
        import logging
        import types
        from pathlib import Path
        a = types.SimpleNamespace()
        a.model = model
        a.logger = logging.getLogger("imgt2")
        a.output_dir = Path(home) / "imgsess"
        a.output_dir.mkdir(parents=True, exist_ok=True)
        a.generation_config = None
        a.safety_settings = None
        return a

    def _hot_state(self):
        return {
            "analysis_approach": "threshold + watershed",
            "skills_loaded": [],
            "series_results": [{
                "index": 0, "name": "img0", "success": True,
                "analysis_type": "atom segmentation + RDF",
                "plan_deviation_summary": "switched to watershed after threshold failed",
                "script": "import numpy as np\n# IMG_VERBATIM_MARKER\nprint('seg')\n",
                "quality_history": {"approved": True, "final_score": 0.93, "verification_iterations": [
                    {"annealing_level": 0}, {"annealing_level": 2}]},
            }],
        }

    def test_stages_solution_no_skill_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        from scilink.skills._shared import _memory, _staging
        agent = self._fake_agent(str(tmp_path), self._Model())
        out = ImageAnalysisAgent._maybe_stage_t2_solutions(agent, self._hot_state())
        assert out
        assert _memory.list_memory() == []
        staged = _staging.list_staged("image_analysis")
        assert len(staged) == 1
        assert staged[0]["technique"] == "watershed_grain_segmentation"
        assert "IMG_VERBATIM_MARKER" in staged[0]["working_script"]
        assert staged[0]["quality_score"] == 0.93

    def test_no_stage_when_not_hot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["verification_iterations"] = [
            {"annealing_level": 0}, {"annealing_level": 1}]
        assert ImageAnalysisAgent._maybe_stage_t2_solutions(agent, state) == []

    def test_stage_failure_is_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

        class Boom:
            def generate_content(self, *a, **k):
                raise RuntimeError("boom")

        agent = self._fake_agent(str(tmp_path), Boom())
        assert ImageAnalysisAgent._maybe_stage_t2_solutions(agent, self._hot_state()) == []

    def test_no_stage_when_below_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        agent = self._fake_agent(str(tmp_path), self._Model())
        state = self._hot_state()
        state["series_results"][0]["quality_history"]["approved"] = False
        assert ImageAnalysisAgent._maybe_stage_t2_solutions(agent, state) == []


# ──────────────────────────────────────────────────────────────
# Staging buffer + upgrade@1 / consolidate@N
# ──────────────────────────────────────────────────────────────

class TestStaging:
    def _stub_llm(self, body="MERGED", capture=None):
        def _fn(prompt):
            if capture is not None:
                capture.append(prompt)
            return json.dumps({"description": "d", "overview": "o", "analysis": body})
        return _fn

    def test_crud_and_grouping(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        a = _staging.stage_solution("curve_fitting", "kww",
                                    {"r_squared": 0.99, "working_script": "# A"})
        _staging.stage_solution("curve_fitting", "kww",
                                {"r_squared": 0.98, "working_script": "# B"})
        assert len(_staging.list_staged("curve_fitting")) == 2
        assert list(_staging.group_by_technique("curve_fitting")) == ["kww"]
        assert _staging.get_staged("curve_fitting", a)["r_squared"] == 0.99
        assert _staging.remove_staged("curve_fitting", [a]) == 1
        assert len(_staging.list_staged("curve_fitting")) == 1

    def test_loader_excludes_staging_dir(self, tmp_path, monkeypatch):
        # Staged records must never be discovered as skills.
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        _staging.stage_solution("curve_fitting", "kww", {"working_script": "# A"})
        all_sk = loader.list_all_skills()
        assert "distill_staging" not in all_sk
        # the staging dir is a sibling of graduated_skills, not under a skill root
        assert all("distill_staging" not in str(p) for p in loader._skill_roots())

    def test_upgrade_merges_into_existing_skill(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        # seed an existing skill
        gd = loader.graduated_skills_dir() / "curve_fitting" / "eels"
        gd.mkdir(parents=True); (gd / "__init__.py").touch()
        (gd / "eels.md").write_text("---\ndescription: base\n---\n\n## overview\nthin\n")
        sid = _staging.stage_solution("curve_fitting", "kww",
                                      {"r_squared": 0.99, "working_script": "# UP_MARKER"})
        prompts = []
        res = _staging.upgrade_skill_from_staged(
            "curve_fitting", [sid], target_domain="curve_fitting", target_name="eels",
            llm_call=self._stub_llm("UPGRADED_BODY", capture=prompts),
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="UPD {skill_name} {existing_skill} {new_knowledge}")
        assert res["method"] == "updated"
        parsed = loader.load_skill(res["skill_path"], domain="curve_fitting")
        assert "UPGRADED_BODY" in parsed["analysis"]
        # The LLM SEES the working script as input (to generalize from)…
        assert "UP_MARKER" in prompts[0]
        # …but the verbatim script must NOT be dumped into the saved skill (bloat
        # over-constrains reuse — see the head-to-head regression).
        assert "UP_MARKER" not in parsed.get("implementation", "")
        assert _staging.list_staged("curve_fitting") == []  # consumed

    def test_propose_builds_without_writing(self, tmp_path, monkeypatch):
        # propose must build the merged content for review WITHOUT touching the
        # file or consuming the staged record.
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        gd = loader.graduated_skills_dir() / "curve_fitting" / "eels"
        gd.mkdir(parents=True); (gd / "__init__.py").touch()
        original = "---\ndescription: base\n---\n\n## overview\nthin\n"
        (gd / "eels.md").write_text(original)
        sid = _staging.stage_solution("curve_fitting", "kww",
                                      {"r_squared": 0.99, "working_script": "# x"})
        prop = _staging.propose_skill_upgrade(
            "curve_fitting", [sid], target_domain="curve_fitting", target_name="eels",
            llm_call=self._stub_llm("PROPOSED_BODY"),
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}")
        assert prop["status"] == "success"
        assert "PROPOSED_BODY" in prop["proposed_content"]
        assert prop["existing_content"] == original
        assert (gd / "eels.md").read_text() == original  # not written
        assert len(_staging.list_staged("curve_fitting")) == 1  # not consumed

    def test_apply_writes_and_backs_up(self, tmp_path, monkeypatch):
        # apply writes the approved content, backs up the pre-upgrade file, and
        # consumes the staged record; the backup is not seen as a skill.
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        gd = loader.graduated_skills_dir() / "curve_fitting" / "eels"
        gd.mkdir(parents=True); (gd / "__init__.py").touch()
        original = "---\ndescription: base\n---\n\n## overview\nthin\n"
        (gd / "eels.md").write_text(original)
        sid = _staging.stage_solution("curve_fitting", "kww",
                                      {"r_squared": 0.99, "working_script": "# x"})
        prop = _staging.propose_skill_upgrade(
            "curve_fitting", [sid], target_domain="curve_fitting", target_name="eels",
            llm_call=self._stub_llm("APPLIED_BODY"),
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}")
        res = _staging.apply_skill_upgrade(
            "curve_fitting", prop["staged_ids"], target_domain="curve_fitting",
            target_name="eels", proposed_content=prop["proposed_content"])
        assert res["status"] == "success"
        assert "APPLIED_BODY" in (gd / "eels.md").read_text()
        bak = gd / "eels.md.bak"
        assert bak.exists() and bak.read_text() == original
        assert "eels" in loader.list_skills("curve_fitting")  # .bak not a skill
        assert _staging.list_staged("curve_fitting") == []    # consumed

    def test_resolved_error_lessons(self):
        from scilink.skills._shared import _staging
        qh = {
            "script_errors": [
                {"error": "E1", "diagnosis": "D1"},   # curve-fit shape (fix=diagnosis)
                {"error": "E2", "fix": "F2"},         # image shape
                {"error": "E3"},                       # unresolved -> skip
                {"error": "", "diagnosis": "x"},      # no error -> skip
            ],
            "verification_iterations": [
                {"fix_applied": "VF", "issues": [{"problem": "P1"}, {"problem": "P2"}]},
                {"issues": [{"problem": "P3"}]},      # no fix -> skip
            ],
        }
        lessons = _staging.resolved_error_lessons(qh)
        assert {"error": "E1", "fix": "D1"} in lessons
        assert {"error": "E2", "fix": "F2"} in lessons
        assert any(l["fix"] == "VF" and "P1" in l["error"] for l in lessons)
        assert all(l["error"] and l["fix"] for l in lessons)
        assert len(lessons) == 3

    def test_stage_feedback_and_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        results = [
            {"success": True, "model_type": "kww",
             "quality_history": {"script_errors": [{"error": "bad p0", "diagnosis": "seed from data"}]},
             "fit_quality": {"r_squared": 0.95}},
            {"success": True, "model_type": "kww", "quality_history": {}},        # no lessons -> skip
            {"success": False, "quality_history": {"script_errors": [{"error": "x", "diagnosis": "y"}]}},  # not success -> skip
        ]
        ids = _staging.stage_feedback_and_errors(
            "curve_fitting", results=results,
            feedback_texts=["use a stretched exponential"],
            session="sess1", llm_call=lambda p: "kww",
            label_template="{model} {deviation} {existing}")
        staged = _staging.list_staged("curve_fitting")
        assert sorted(s["provenance"] for s in staged) == ["error_fix", "user_correction"]
        ef = next(s for s in staged if s["provenance"] == "error_fix")
        assert ef["error_lessons"][0]["fix"] == "seed from data" and ef["r_squared"] == 0.95
        uc = next(s for s in staged if s["provenance"] == "user_correction")
        assert "stretched exponential" in uc["user_feedback"]
        assert {s["technique"] for s in staged} == {"kww"}  # grouped under the technique

    def test_feedback_staging_disabled_when_no_signal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        ids = _staging.stage_feedback_and_errors(
            "curve_fitting",
            results=[{"success": True, "model_type": "kww", "quality_history": {}}],
            feedback_texts=[], session="s", llm_call=lambda p: "kww",
            label_template="{model}{deviation}{existing}")
        assert ids == [] and _staging.list_staged("curve_fitting") == []

    def test_consolidate_mixed_provenance(self, tmp_path, monkeypatch):
        # T=2 win + error_fix + user_correction of the same technique consolidate
        # into one skill (records flow through unchanged; provenance preserved).
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        _staging.stage_solution("curve_fitting", "kww",
                                {"provenance": "t2_solution", "r_squared": 0.99, "working_script": "# s"})
        _staging.stage_solution("curve_fitting", "kww",
                                {"provenance": "error_fix", "error_lessons": [{"error": "E", "fix": "F"}]})
        _staging.stage_solution("curve_fitting", "kww",
                                {"provenance": "user_correction", "user_feedback": "prefer X"})
        captured = []
        res = _staging.consolidate_technique(
            "curve_fitting", "kww", llm_call=self._stub_llm("MIXED_BODY", capture=captured),
            consolidation_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}")
        assert res["status"] == "success" and res["n_examples"] == 3
        # all three provenances reached the consolidation prompt
        assert all(p in captured[0] for p in ("t2_solution", "error_fix", "user_correction"))
        parsed = loader.load_skill(res["skill_path"], domain="curve_fitting")
        assert "MIXED_BODY" in parsed["analysis"]

    def test_upgrade_into_missing_target_errors(self, tmp_path, monkeypatch):
        # "upgrade --into <nonexistent>" must error (not silently create), and
        # must NOT call the LLM or consume the staged record.
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        sid = _staging.stage_solution("curve_fitting", "kww",
                                      {"r_squared": 0.99, "working_script": "# x"})
        called = []
        res = _staging.upgrade_skill_from_staged(
            "curve_fitting", [sid], target_domain="curve_fitting", target_name="does_not_exist",
            llm_call=lambda p: called.append(p) or "{}",
            fresh_template="{skill_name}{domain}{knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}")
        assert res["status"] == "error" and "does not exist" in res["message"]
        assert called == [], "LLM should not be called for a missing target"
        assert len(_staging.list_staged("curve_fitting")) == 1, "staged record must be preserved"

    def test_consolidate_builds_new_skill_from_n(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        for i, m in enumerate([0.99, 0.97, 0.98]):
            _staging.stage_solution("curve_fitting", "kww",
                                    {"r_squared": m, "working_script": f"# CMARK{i}"})
        prompts = []
        res = _staging.consolidate_technique(
            "curve_fitting", "kww", llm_call=self._stub_llm("CONSOLIDATED", capture=prompts),
            consolidation_template="CONS {skill_name} {domain} {knowledge_text}",
            update_template="{skill_name}{existing_skill}{new_knowledge}")
        assert res["status"] == "success" and res["n_examples"] == 3
        parsed = loader.load_skill(res["skill_path"], domain="curve_fitting")
        assert parsed["meta"].get("n_examples") == 3
        assert parsed["meta"].get("provenance") == "t2_consolidated"
        assert "CONSOLIDATED" in parsed["analysis"]
        # all 3 example scripts are shown to the LLM as input…
        assert all(f"CMARK{i}" in prompts[0] for i in range(3))
        # …but not dumped verbatim into the consolidated skill.
        assert "CMARK" not in parsed.get("implementation", "")
        assert _staging.list_staged("curve_fitting") == []  # all consumed

    def test_path_traversal_contained(self, tmp_path, monkeypatch):
        # A malformed domain must NOT escape the staging root (path traversal).
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _staging
        _staging.stage_solution("../../evil_domain", "t", {"working_script": "#x"})
        root = _staging.staging_dir()
        outside = list(tmp_path.glob("evil_domain/*.json")) + \
            list(tmp_path.parent.glob("evil_domain/*.json"))
        assert not outside, f"staged record escaped the staging root: {outside}"
        # everything written stays under the staging dir
        assert all(root.resolve() in f.resolve().parents
                   for f in root.rglob("*.json"))

    def test_safe_path_component(self):
        from scilink.skills._shared._graduation import safe_path_component
        assert safe_path_component("../../evil") == "evil"
        assert safe_path_component("a/b/c") == "c"
        assert safe_path_component("..") == "unknown"
        assert safe_path_component("   ") == "unknown"
        assert safe_path_component("curve_fitting") == "curve_fitting"

    def test_ephemeral_store_warning(self, monkeypatch, caplog):
        import logging
        import scilink.skills._shared._graduation as g
        # SCILINK_HOME set -> never warn, even if "in a container"
        monkeypatch.setenv("SCILINK_HOME", "/data")
        monkeypatch.setattr(g.os.path, "exists", lambda p: p == "/.dockerenv")
        g._EPHEMERAL_WARNED = False
        with caplog.at_level(logging.WARNING):
            g.warn_if_ephemeral_store()
        assert not any("will NOT" in r.message for r in caplog.records)
        # In a container, no SCILINK_HOME -> warn ONCE
        monkeypatch.delenv("SCILINK_HOME", raising=False)
        g._EPHEMERAL_WARNED = False
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            g.warn_if_ephemeral_store()
            g.warn_if_ephemeral_store()  # second call: should NOT warn again
        warns = [r for r in caplog.records if "will NOT" in r.message]
        assert len(warns) == 1, f"expected exactly one warning, got {len(warns)}"
        # Not in a container -> never warn
        monkeypatch.setattr(g.os.path, "exists", lambda p: False)
        g._EPHEMERAL_WARNED = False
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            g.warn_if_ephemeral_store()
        assert not any("will NOT" in r.message for r in caplog.records)

    def test_staging_is_ase_free(self):
        import subprocess
        import sys
        import textwrap
        script = textwrap.dedent("""
            import sys
            class B:
                def find_spec(self, n, p, t=None):
                    if n == 'ase' or n.startswith('ase.'): raise ImportError('blocked')
                    return None
            sys.meta_path.insert(0, B())
            from scilink.skills._shared import _staging
            assert 'ase' not in sys.modules
            print('OK')
        """)
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert p.returncode == 0 and "OK" in p.stdout, p.stderr


# ──────────────────────────────────────────────────────────────
# Master on/off switch (opt-in; off by default)
# ──────────────────────────────────────────────────────────────

class TestMemorySwitch:
    def test_default_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_MEMORY", raising=False)  # no env, no config
        assert loader.memory_enabled() is False

    def test_set_persists_and_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.delenv("SCILINK_MEMORY", raising=False)
        loader.set_memory_enabled(True)
        assert loader.memory_enabled() is True
        loader.set_memory_enabled(False)
        assert loader.memory_enabled() is False
        monkeypatch.setenv("SCILINK_MEMORY", "1")   # env overrides config(False)
        assert loader.memory_enabled() is True
        monkeypatch.setenv("SCILINK_MEMORY", "0")
        loader.set_memory_enabled(True)
        assert loader.memory_enabled() is False      # env(False) overrides config(True)

    def test_off_excludes_graduated_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.setenv("SCILINK_MEMORY", "0")
        gd = loader.graduated_skills_dir() / "curve_fitting" / "zz"
        gd.mkdir(parents=True); (gd / "__init__.py").touch()
        (gd / "zz.md").write_text("---\ndescription: t\n---\n## overview\nx\n")
        roots = [r.resolve() for r in loader._skill_roots()]
        assert loader.graduated_skills_dir().resolve() not in roots
        assert "zz" not in loader.list_all_skills().get("curve_fitting", [])
        # ...and ON makes it visible again
        monkeypatch.setenv("SCILINK_MEMORY", "1")
        assert "zz" in loader.list_all_skills().get("curve_fitting", [])


# ──────────────────────────────────────────────────────────────
# Demote — the reverse of promote
# ──────────────────────────────────────────────────────────────

class TestDemote:
    def _write_bundle(self, root, domain, name, provisional):
        d = root / domain / name
        d.mkdir(parents=True)
        fm = "---\nprovisional: true\n---\n" if provisional else ""
        (d / f"{name}.md").write_text(fm + "## overview\nbody text stays\n")

    def test_promote_demote_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        from scilink.skills.loader import graduated_skills_dir

        self._write_bundle(graduated_skills_dir(), "curve_fitting", "sk", True)
        _memory.promote_memory("curve_fitting", "sk")
        assert all(not r["provisional"] for r in _memory.list_memory())

        out = _memory.demote_memory("curve_fitting", "sk")
        assert out["provisional"] is True
        rows = _memory.list_memory()
        assert len(rows) == 1 and rows[0]["provisional"] is True
        # Section bodies untouched by the frontmatter rewrite.
        assert "body text stays" in _memory.show_memory("curve_fitting", "sk")
        # Idempotent + re-promotable.
        _memory.demote_memory("curve_fitting", "sk")
        _memory.promote_memory("curve_fitting", "sk")
        assert all(not r["provisional"] for r in _memory.list_memory())

    def test_demote_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            _memory.demote_memory("curve_fitting", "nope")


# ──────────────────────────────────────────────────────────────
# Unified T=2 nomination: hot wins promote their bank record
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Manual skill editing (validated, backed up)
# ──────────────────────────────────────────────────────────────

class TestEditMemory:
    def _seed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills.loader import graduated_skills_dir
        d = graduated_skills_dir() / "curve_fitting" / "sk"
        d.mkdir(parents=True)
        md = d / "sk.md"
        md.write_text("---\ndescription: d\n---\n## overview\noriginal body\n")
        return md

    def test_edit_round_trip_with_backup(self, tmp_path, monkeypatch):
        from scilink.skills._shared import _memory
        md = self._seed(tmp_path, monkeypatch)
        out = _memory.edit_memory("curve_fitting", "sk",
                                  "---\ndescription: d2\n---\n## overview\nedited body\n")
        assert out["status"] == "success"
        assert "edited body" in md.read_text()
        assert "original body" in md.with_name("sk.md.bak").read_text()

    def test_edit_rejects_broken_content(self, tmp_path, monkeypatch):
        from scilink.skills._shared import _memory
        md = self._seed(tmp_path, monkeypatch)
        bad_yaml = _memory.edit_memory("curve_fitting", "sk",
                                       "---\ndescription: [unclosed\n---\n## overview\nx\n")
        assert bad_yaml["status"] == "error" and "YAML" in bad_yaml["message"]
        no_section = _memory.edit_memory("curve_fitting", "sk",
                                         "---\ndescription: d\n---\njust prose\n")
        assert no_section["status"] == "error" and "section" in no_section["message"]
        assert "original body" in md.read_text()  # untouched on rejection
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            _memory.edit_memory("curve_fitting", "nope", "## overview\nx\n")


# ──────────────────────────────────────────────────────────────
# Fork built-in skills (copy-on-write) + additivity guard
# (ships in the feature-skill-fork PR; these activate on its merge)
# ──────────────────────────────────────────────────────────────

def _has_fork_features() -> bool:
    from scilink.skills._shared import _memory as _m
    return hasattr(_m, "fork_builtin")


@pytest.mark.skipif(not _has_fork_features(),
                    reason="fork_builtin not merged yet (feature-skill-fork)")
class TestForkBuiltin:
    def test_fork_shadows_and_upgradable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        # Shadowing requires the persistent store to be ACTIVE — with memory
        # off the graduated root is not on the skill search path at all.
        monkeypatch.setenv("SCILINK_MEMORY", "1")
        from scilink.skills import loader
        from scilink.skills._shared import _memory

        builtin_text = (loader._SKILLS_DIR / "curve_fitting" / "raman"
                        / "raman.md").read_text()
        out = _memory.fork_builtin("curve_fitting", "raman")
        assert out["status"] == "success"
        assert out["has_sibling_tools"] is False
        # Byte-identical copy -> diff-builtin starts clean.
        d = _memory.diff_builtin("curve_fitting", "raman")
        assert d["identical"] is True
        # The fork appears in the persistent store (upgrade target)…
        assert any(r["name"] == "raman" for r in _memory.list_memory())
        # …and SHADOWS the built-in: edit the fork, loader serves the edit.
        fork_md = tmp_path / "graduated_skills" / "curve_fitting" / "raman" / "raman.md"
        fork_md.write_text(builtin_text + "\nFORK_SENTINEL_LINE\n")
        loaded = loader.load_skill("raman", domain="curve_fitting")
        assert "FORK_SENTINEL_LINE" in str(loaded)
        d = _memory.diff_builtin("curve_fitting", "raman")
        assert d["identical"] is False and "FORK_SENTINEL_LINE" in d["diff"]

    def test_double_fork_refused_and_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        assert _memory.fork_builtin("curve_fitting", "raman")["status"] == "success"
        again = _memory.fork_builtin("curve_fitting", "raman")
        assert again["status"] == "error" and "already forked" in again["message"]
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            _memory.fork_builtin("curve_fitting", "no_such_skill")
        with _pytest.raises(FileNotFoundError):
            _memory.diff_builtin("curve_fitting", "xps")  # not forked

    def test_sibling_tools_flagged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        out = _memory.fork_builtin("curve_fitting", "xrd_profile")
        assert out["status"] == "success"
        assert out["has_sibling_tools"] is True
        assert "fit_pattern.py" in out["sibling_tools"]


@pytest.mark.skipif(not _has_fork_features(),
                    reason="additivity guard not merged yet (feature-skill-fork)")
class TestAdditivityGuard:
    def test_regression_warnings(self):
        from scilink.skills._shared._staging import _regression_warnings
        existing = ("## overview\nlong text " + "word " * 100
                    + "\n## planning\nrules\n## validation\nchecks\n")
        clean = existing + "\nnew lesson appended\n"
        assert _regression_warnings(existing, clean) == []
        dropped = "## overview\n" + "word " * 100
        warns = _regression_warnings(existing, dropped)
        assert any("planning" in w for w in warns)
        assert any("validation" in w for w in warns)
        shrunk = "## overview\nshort\n## planning\nx\n## validation\ny\n"
        assert any("shorter" in w for w in _regression_warnings(existing, shrunk))

    def test_preserve_structure_round_trip(self):
        from scilink.skills._shared._staging import _preserve_structure
        existing = (
            "---\ndescription: old desc\n"
            "technique: [Raman, micro-Raman]\n---\n"
            "# Raman Spectroscopy\n\n"
            "## Overview\nbody A\n\n## Planning\nbody B\n")
        proposed = (  # what the JSON round-trip typically emits
            "---\ndescription: refreshed desc\n---\n"
            "## overview\nbody A plus new lesson\n\n"
            "## analysis\n\n"           # empty artifact section
            "## planning\nbody B\n")
        out = _preserve_structure(existing, proposed)
        assert "technique:" in out and "micro-Raman" in out   # routing meta kept
        assert "refreshed desc" in out                        # LLM description honored
        assert "# Raman Spectroscopy" in out                  # title restored
        assert "## Overview" in out and "## overview" not in out  # casing restored
        assert "## analysis" not in out                       # empty artifact dropped
        assert "new lesson" in out                            # LLM merge kept
        from scilink.skills._shared._staging import _regression_warnings
        assert _regression_warnings(existing, out) == []

    def test_preserve_structure_reinstates_dropped_section(self):
        from scilink.skills._shared._staging import (
            _preserve_structure, _regression_warnings)
        existing = ("---\ndescription: d\n---\n"
                    "## Overview\nA\n\n## Implementation\nCRITICAL RECIPE\n\n"
                    "## Validation\nchecks\n")
        proposed = ("---\ndescription: d\n---\n"
                    "## overview\nA plus lesson\n\n## validation\nchecks\n")
        out = _preserve_structure(existing, proposed)
        assert "## Implementation" in out and "CRITICAL RECIPE" in out
        # reinstated content sits after the surviving sections
        assert out.index("CRITICAL RECIPE") > out.index("A plus lesson")
        assert _regression_warnings(existing, out) == []


# ──────────────────────────────────────────────────────────────
# Unified T=2 nomination: hot wins promote their bank record
# ──────────────────────────────────────────────────────────────

class TestT2BankUnification(TestT2StageHook):
    def test_hot_win_promotes_bank_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        from scilink.skills._shared import _script_bank as sb, _staging

        state = self._hot_state()
        script = state["series_results"][0]["script"]
        # The finalize path banks BEFORE staging — emulate that.
        rid = sb.add_record("curve_fitting", {
            "working_script": script,
            "data_fingerprint": {"kind": "curve", "n_points": 100},
            "measurement_context": {"technique": "Raman"},
            "technique_signals": {"model_type": "2 Voigt + exp tail"},
            "outcome": {"metric": {"name": "r_squared", "value": 0.991}},
            "provenance": {"session": "sess"}})["id"]

        agent = self._fake_agent(str(tmp_path), self._Model())
        out = CurveFittingAgent._maybe_stage_t2_solutions(agent, state)
        assert len(out) == 1
        staged = _staging.get_staged("curve_fitting", out[0])
        # One identity: the staged nomination points at the bank record...
        assert staged["provenance"] == "t2_hot_win"
        assert staged["bank_id"] == rid
        assert staged["technique"] == "voigt_plus_exp_tail"  # LLM label kept
        # ...carries the contrastive T=2 story...
        assert staged["planned_model"] == "3 Gaussians"
        assert staged["deviation_from_plan"] == "switched model"
        assert staged["r_squared"] == 0.991
        assert "VERBATIM_MARKER" in staged["working_script"]
        # ...and the bank record is marked with the nomination reason.
        rec = sb.get_record("curve_fitting", rid)
        assert rec["promoted_to_staging"] == out[0]
        assert rec["promoted_reason"] == "t2_hot_win"
        # No duplicate independent staging happened.
        assert len(_staging.list_staged("curve_fitting")) == 1

    def test_hot_win_fallback_when_bank_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        monkeypatch.setenv("SCILINK_SCRIPT_BANK", "0")  # bank off, memory on
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        from scilink.skills._shared import _staging

        agent = self._fake_agent(str(tmp_path), self._Model())
        out = CurveFittingAgent._maybe_stage_t2_solutions(agent, self._hot_state())
        assert len(out) == 1
        staged = _staging.get_staged("curve_fitting", out[0])
        assert "bank_id" not in staged            # legacy shape
        assert "VERBATIM_MARKER" in staged["working_script"]
