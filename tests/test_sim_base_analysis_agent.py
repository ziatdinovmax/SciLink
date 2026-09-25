"""The sim-side BaseAnalysisAgent codegen engine.

Exercises the reusable core with no real LLM (``_llm`` is monkeypatched to return
canned scripts) but a REAL sandbox executor: static helpers, construction, script
execution + JSON parsing, the compute_property loop, and error-driven refinement.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.base_analysis_agent import BaseAnalysisAgent  # noqa: E402


class _Concrete(BaseAnalysisAgent):
    def run_analysis(self, research_goal, **kwargs):
        return {"status": "success", "results": {}}


@pytest.fixture
def agent(tmp_path):
    return _Concrete(output_dir=str(tmp_path), api_key="test-key")


class TestStatics:
    def test_extract_json_last_object(self):
        out = BaseAnalysisAgent._extract_json('log line\n{"status":"success","value":1.5}')
        assert out == {"status": "success", "value": 1.5}

    def test_extract_json_none_when_absent(self):
        assert BaseAnalysisAgent._extract_json("no json here") is None

    def test_clean_code_strips_fence(self):
        assert BaseAnalysisAgent._clean_code("```python\nx = 1\n```") == "x = 1"


class TestExecution:
    def test_execute_script_parses_value(self, agent):
        r = agent._execute_script('print(\'{"status":"success","value":5,"units":"cP"}\')', "t")
        assert r["status"] == "success" and r["value"] == 5 and r["units"] == "cP"

    def test_execute_script_reports_error(self, agent):
        r = agent._execute_script("raise RuntimeError('boom')", "t")
        assert r["status"] == "error" and "boom" in r["concise_error"]

    def test_execute_script_catches_syntax_error_before_run(self, agent):
        # A syntax error is caught by the compile check, not a sandbox run.
        r = agent._execute_script('print(json.dumps({"a": 1}))extra)', "t")
        assert r["status"] == "error" and "SyntaxError" in r["concise_error"]

    def test_execute_script_reads_injected_globals(self, agent, tmp_path):
        # compute_property injects DATA_FILES/OUTPUT_DIR; a script can read them.
        data = tmp_path / "d.txt"
        data.write_text("21")
        agent._llm = lambda p: (
            "import json\n"
            "with open(DATA_FILES['d']) as f:\n"
            "    v = int(f.read())\n"
            'print(json.dumps({"status": "success", "value": v * 2}))'
        )
        r = agent.compute_property("double it", {"d": str(data)}, verify=False)
        assert r["status"] == "success" and r["value"] == 42


class TestOutputTypes:
    """The non-scalar (curve/image/datacube) output-type generalization."""

    _CURVE_SCRIPT = (
        "import json, numpy as np, os\n"
        "q = np.linspace(0.5, 12.0, 100)\n"
        "S = 1.0 + 0.3 * np.exp(-(q - 2.0) ** 2)\n"
        "np.save(os.path.join(OUTPUT_DIR, 'sq.npy'), np.vstack([q, S]))\n"
        "print(json.dumps({'status': 'success', 'output_type': 'curve',"
        " 'artifact': {'path': 'sq.npy', 'format': 'npy', 'shape': [2, 100]},"
        " 'summary': {'n_points': 100, 'q_max': 12.0, 'peak': 1.3, 'nan': 0}}))"
    )

    def test_curve_output_collects_artifact(self, agent):
        agent._llm = lambda p: self._CURVE_SCRIPT
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=False, output_type="curve")
        assert r["status"] == "success" and r["output_type"] == "curve"
        art = r["artifact"]
        assert os.path.isabs(art["path"]) and os.path.exists(art["path"])
        assert art["format"] == "npy" and art["shape"] == [2, 100]

    def test_curve_missing_artifact_is_error(self, agent):
        # Reports success + an artifact path it never wrote -> fail loud.
        agent._llm = lambda p: (
            "import json\n"
            "print(json.dumps({'status': 'success', 'output_type': 'curve',"
            " 'artifact': {'path': 'ghost.npy', 'format': 'npy'},"
            " 'summary': {}}))"
        )
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=False, output_type="curve")
        assert r["status"] == "error" and "artifact" in r["message"]

    def test_verify_runs_on_curve_summary(self, agent):
        # _llm serves both the codegen prompt (script) and the curve gate (JSON).
        def _llm(prompt):
            if "produced a curve" in prompt:      # the curve verification gate
                return '{"plausible": true, "reasoning": "sensible S(q)"}'
            return TestOutputTypes._CURVE_SCRIPT
        agent._llm = _llm
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=True, output_type="curve")
        assert r["status"] == "success"
        assert r["verification"]["plausible"] is True

    def test_scalar_default_unchanged(self, agent):
        # Regression: the default scalar path is untouched (no artifact).
        agent._llm = lambda p: ('import json; print(json.dumps('
                                '{"status":"success","value":3.14,"units":"x"}))')
        r = agent.compute_property("x", {"d": "/nope"}, verify=False)
        assert r["status"] == "success" and r["value"] == 3.14
        assert "artifact" not in r

    def test_refine_prompt_states_the_artifact_contract(self, agent):
        # The "success without artifact" error routes into _refine_code; the
        # retry prompt must carry the non-scalar contract, not the scalar one.
        prompts, calls = [], {"n": 0}

        def _llm(prompt):
            prompts.append(prompt)
            calls["n"] += 1
            if calls["n"] == 1:                       # first attempt: no artifact written
                return ("import json\n"
                        "print(json.dumps({'status': 'success', 'output_type': "
                        "'curve', 'artifact': {'path': 'ghost.npy', 'format': "
                        "'npy'}, 'summary': {}}))")
            return TestOutputTypes._CURVE_SCRIPT       # refined attempt writes it

        agent._llm = _llm
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=False, output_type="curve")
        assert r["status"] == "success" and r["attempts"] == 2
        # The SECOND prompt is the refine prompt — it must state the artifact
        # contract, i.e. mention writing into OUTPUT_DIR / the curve output type.
        refine_prompt = prompts[1]
        assert "curve observable" in refine_prompt
        assert "WRITE the computed" in refine_prompt

    def test_readback_catches_format_lie(self, agent):
        # Writes a plain-text file but claims it is an npy with a fabricated
        # shape. The deterministic readback fails -> rejected -> error, not a
        # judged "success" on self-reported statistics.
        agent.max_refinement_attempts = 0
        agent._llm = lambda p: (
            "import json, os\n"
            "open(os.path.join(OUTPUT_DIR, 'fake.npy'), 'w').write('not an array')\n"
            "print(json.dumps({'status': 'success', 'output_type': 'curve',"
            " 'artifact': {'path': 'fake.npy', 'format': 'npy', 'shape': [2, 100]},"
            " 'summary': {'n_points': 100, 'nan': 0}}))"
        )
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=False, output_type="curve")
        assert r["status"] == "error" and "artifact" in r["message"]

    def test_artifact_must_be_contained_in_output_dir(self, tmp_path):
        # A script that passes off its INPUT file (outside OUTPUT_DIR) as the
        # artifact is rejected: only files written under OUTPUT_DIR count.
        import numpy as np
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        outside = tmp_path / "input_traj.npy"    # sibling of out_dir, NOT under it
        np.save(str(outside), np.zeros((2, 5)))
        agent = _Concrete(output_dir=str(out_dir), api_key="test-key")
        agent.max_refinement_attempts = 0
        agent._llm = lambda p: (
            "import json\n"
            f"print(json.dumps({{'status': 'success', 'output_type': 'curve',"
            f" 'artifact': {{'path': {str(outside)!r}, 'format': 'npy',"
            f" 'shape': [2, 5]}}, 'summary': {{}}}}))"
        )
        r = agent.compute_property("S(q)", {"traj": str(outside)},
                                   verify=False, output_type="curve")
        assert r["status"] == "error" and "artifact" in r["message"]

    def test_measured_facts_override_reported_shape(self, agent):
        # The script LIES about shape ([9, 9]) but writes a real (2, 100) npy;
        # the resolved artifact carries the MEASURED shape, not the claim.
        agent._llm = lambda p: (
            "import json, os, numpy as np\n"
            "q = np.linspace(0.5, 12.0, 100)\n"
            "np.save(os.path.join(OUTPUT_DIR, 'sq.npy'), np.vstack([q, q]))\n"
            "print(json.dumps({'status': 'success', 'output_type': 'curve',"
            " 'artifact': {'path': 'sq.npy', 'format': 'npy', 'shape': [9, 9]},"
            " 'summary': {'n_points': 100}}))"
        )
        r = agent.compute_property("S(q)", {"traj": "/nope"},
                                   verify=False, output_type="curve")
        assert r["status"] == "success"
        assert r["artifact"]["shape"] == [2, 100]           # measured, not [9, 9]
        assert r["artifact"]["measured"]["nan_count"] == 0

    def test_offvocab_output_type_warns(self, agent, caplog):
        import logging
        agent._llm = lambda p: (
            "import json, os\n"
            "open(os.path.join(OUTPUT_DIR, 'o.npy'), 'wb')\n"
            "import numpy as np; np.save(os.path.join(OUTPUT_DIR,'o.npy'), np.zeros(3))\n"
            "print(json.dumps({'status':'success','output_type':'curl',"
            " 'artifact':{'path':'o.npy','format':'npy','shape':[3]},'summary':{}}))"
        )
        with caplog.at_level(logging.WARNING):
            agent.compute_property("x", {"d": "/nope"}, verify=False,
                                   output_type="curl")   # typo'd frontmatter
        assert any("not one of" in rec.message for rec in caplog.records)


class TestComputePropertyLoop:
    def test_success_first_try(self, agent):
        agent._llm = lambda p: 'import json; print(json.dumps({"status":"success","value":7.0,"units":"x"}))'
        r = agent.compute_property("compute x", {"traj": "/nope"}, verify=False)
        assert r["status"] == "success" and r["value"] == 7.0 and r["attempts"] == 1

    def test_refines_after_failure(self, agent):
        calls = {"n": 0}

        def fake(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "raise RuntimeError('kaboom')"        # first generate -> fails
            return 'import json; print(json.dumps({"status":"success","value":3}))'  # refine -> ok

        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=False)
        assert r["status"] == "success" and r["value"] == 3 and r["attempts"] == 2

    def test_gives_up_after_max_attempts(self, agent):
        agent.max_refinement_attempts = 1
        agent._llm = lambda p: "raise RuntimeError('always')"
        r = agent.compute_property("t", {"traj": "/nope"}, verify=False)
        assert r["status"] == "error" and r["attempts"] == 2

    def test_verification_runs_when_enabled(self, agent):
        def fake(prompt):
            if "physically plausible" in prompt:
                return '{"plausible": true, "reasoning": "sane"}'
            return 'import json; print(json.dumps({"status":"success","value":1.0}))'

        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=True)
        assert r["verification"]["plausible"] is True


class TestQCEngineIntegration:
    """Tests for the CodegenQCEngine-driven verification loop (#439 Tier 2)."""

    def test_verification_retry_on_implausible(self, agent):
        """A plausibility failure triggers a refit; the second attempt passes."""
        calls = {"gen": 0, "verify": 0}

        def fake(prompt):
            if "physically plausible" in prompt or "produced a" in prompt:
                calls["verify"] += 1
                if calls["verify"] <= 1:
                    return '{"plausible": false, "reasoning": "value too high"}'
                return '{"plausible": true, "reasoning": "ok now"}'
            calls["gen"] += 1
            if "PRIOR ATTEMPT FEEDBACK" in prompt:
                return 'import json; print(json.dumps({"status":"success","value":2.0,"units":"x"}))'
            return 'import json; print(json.dumps({"status":"success","value":999.0,"units":"x"}))'

        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=True)
        assert r["status"] == "success"
        assert r["verification"]["plausible"] is True
        assert calls["verify"] >= 2

    def test_verify_false_bypasses_qc_loop(self, agent):
        """verify=False skips the verification loop entirely."""
        calls = {"verify": 0}

        def fake(prompt):
            if "physically plausible" in prompt:
                calls["verify"] += 1
                return '{"plausible": false, "reasoning": "bad"}'
            return 'import json; print(json.dumps({"status":"success","value":1.0}))'

        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=False)
        assert r["status"] == "success" and r["value"] == 1.0
        assert calls["verify"] == 0

    def test_crash_recovery_inside_initial(self, agent):
        """Execution crashes are retried inside qc_run_initial."""
        calls = {"n": 0}

        def fake(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "raise RuntimeError('crash')"
            return 'import json; print(json.dumps({"status":"success","value":42}))'

        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=False)
        assert r["status"] == "success" and r["value"] == 42

    def test_annealing_feedback_in_refit_prompt(self, agent):
        """The refit prompt carries verification feedback and annealing text."""
        prompts = []

        def fake(prompt):
            prompts.append(prompt)
            if "physically plausible" in prompt:
                return '{"plausible": false, "reasoning": "divergent"}'
            return 'import json; print(json.dumps({"status":"success","value":1.0,"units":"x"}))'

        agent.max_refinement_attempts = 1
        agent._llm = fake
        agent.compute_property("t", {"traj": "/nope"}, verify=True)
        refit_prompts = [p for p in prompts if "PRIOR ATTEMPT FEEDBACK" in p]
        assert len(refit_prompts) >= 1
        assert "divergent" in refit_prompts[0]

    def test_fallback_returns_best_on_exhaustion(self, agent):
        """When all verification retries fail, the best result is returned."""
        def fake(prompt):
            if "physically plausible" in prompt:
                return '{"plausible": false, "reasoning": "always bad"}'
            return 'import json; print(json.dumps({"status":"success","value":7.0,"units":"x"}))'

        agent.max_refinement_attempts = 1
        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=True)
        assert r["status"] == "success"
        assert r["value"] == 7.0
        assert r["verification"]["plausible"] is False

    def test_returned_verification_judged_the_returned_value(self, agent):
        """Distinct values per attempt, every verdict implausible: the value
        that comes back must carry the verdict that judged *it*."""
        n = {"gen": 0, "verify": 0}

        def fake(prompt):
            if "physically plausible" in prompt:
                n["verify"] += 1
                return json.dumps({"plausible": False,
                                   "reasoning": f"bad #{n['verify']}"})
            n["gen"] += 1
            return ('import json; print(json.dumps({"status":"success",'
                    f'"value":{float(n["gen"])},"units":"x"}}))')

        agent.max_refinement_attempts = 2
        agent._llm = fake
        r = agent.compute_property("t", {"traj": "/nope"}, verify=True)
        assert n["gen"] == 3 and n["verify"] == 3
        assert r["verification"]["reasoning"] == f"bad #{int(r['value'])}"

    def test_final_verify_keeps_its_verdict_on_reject(self, agent):
        from scilink.agents.exp_agents._qc_engine import QCItemContext
        agent._verify_result = lambda *a, **k: {"plausible": False,
                                                "reasoning": "final no"}
        ctx = QCItemContext(state={"task": "t", "output_type": "scalar",
                                   "verify": True},
                            data={}, data_path=".", item_name="t", item_idx=0)
        ctx.current_result = {"status": "success", "value": 3.0}
        agent.qc_final_verify(ctx)
        assert ctx.approved is False
        assert ctx.current_result["verification"]["reasoning"] == "final no"
        assert ctx.verification_history[-1]["reasoning"] == "final no"

    def test_after_refit_leaves_score_unverified(self, agent):
        from scilink.agents.exp_agents._qc_engine import QCItemContext
        ctx = QCItemContext(state={}, data={}, data_path=".", item_name="t",
                            item_idx=0)
        ctx.current_score = 1.0
        refit = {"status": "success", "value": 1.0, "success": True}
        agent.qc_after_refit(ctx, refit, {})
        assert ctx.current_result is refit
        assert ctx.current_score < 0

    def test_hot_level_keeps_recipe_as_advisory(self, agent):
        """The recipe (how to read the engine output) survives escalation to
        hot; only its instruction loosens."""
        recipe = "LOAD_WITH_THE_MAGIC_READER"
        gen_prompts = []

        def fake(prompt):
            if "physically plausible" in prompt:
                return '{"plausible": false, "reasoning": "nope"}'
            gen_prompts.append(prompt)
            return 'import json; print(json.dumps({"status":"success","value":1.0,"units":"x"}))'

        agent.max_refinement_attempts = 2
        agent._llm = fake
        agent.compute_property("t", {"traj": "/nope"}, recipe=recipe, verify=True)
        assert len(gen_prompts) == 3           # initial + relaxed + hot
        assert all(recipe in p for p in gen_prompts)
        assert "follow this" in gen_prompts[0]
        assert "advisory" in gen_prompts[-1] and "follow this" not in gen_prompts[-1]

    def test_verification_iterations_knob_is_separate(self, tmp_path):
        a = _Concrete(output_dir=str(tmp_path), api_key="k",
                      max_refinement_attempts=1, max_verification_iterations=3)
        assert a.max_verification_iterations == 3
        b = _Concrete(output_dir=str(tmp_path), api_key="k",
                      max_refinement_attempts=1)
        assert b.max_verification_iterations == 1
        b.max_refinement_attempts = 4
        assert b.max_verification_iterations == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
