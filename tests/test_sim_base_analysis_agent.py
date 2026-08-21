"""The sim-side BaseAnalysisAgent codegen engine.

Exercises the reusable core with no real LLM (``_llm`` is monkeypatched to return
canned scripts) but a REAL sandbox executor: static helpers, construction, script
execution + JSON parsing, the compute_property loop, and error-driven refinement.
"""

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
