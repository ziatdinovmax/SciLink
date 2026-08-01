"""The sim-side BaseAnalysisAgent codegen engine.

Exercises the reusable core with no real LLM (``_llm`` is monkeypatched to return
canned scripts) but a REAL sandbox executor: static helpers, construction, script
execution + JSON parsing, the compute_property loop, and error-driven refinement.
"""

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
