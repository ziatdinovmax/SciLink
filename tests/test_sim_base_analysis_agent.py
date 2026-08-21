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
