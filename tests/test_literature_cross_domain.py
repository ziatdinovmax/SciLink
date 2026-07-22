"""Offline tests for cross-domain literature retrieval and parallel search.

Contract pinned here (from the benchmarking that motivated the feature):
- `cross_domain` asks for TRANSFERABLE mechanisms from other fields and
  explicitly forbids a topical subfield review — the grounding wrapper used
  by `hypothesis_context` neutralizes cross-domain intent, which was
  observed live before the fix;
- several search types can be requested at once and run CONCURRENTLY (each
  Edison call is minutes of waiting, so serial pairs double the user's wait);
- merged output is LABELLED per source, so candidates can tell established
  results (usable as constraints) from analogies (mechanism inspiration);
- a cross-domain result carries the constraint-compliance caveat;
- single-type behaviour is unchanged.

No network: the literature agent is a scripted stub.
"""

import json
import threading
import time

import pytest

from scilink.agents.lit_agents.literature_agent import LiteratureSearchAgent


# ---------------------------------------------------------------- stubs

class StubLitAgent:
    """Records queries and how long each search overlapped the others."""

    def __init__(self, delay=0.4, fail=()):
        self.delay = delay
        self.fail = set(fail)
        self.calls = []
        self._active = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def _run(self, kind, query):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        time.sleep(self.delay)
        with self._lock:
            self._active -= 1
        self.calls.append((kind, query))
        if kind in self.fail:
            return {"status": "error", "message": f"{kind} boom"}
        return {"status": "success", "content": f"CONTENT::{kind}"}

    def search_for_hypothesis_context(self, q):
        return self._run("hypothesis_context", q)

    def search_for_cross_domain(self, q):
        return self._run("cross_domain", q)

    def search_for_economic_data(self, q):
        return self._run("economic_data", q)

    def search_for_fitting_models(self, q):
        return self._run("fitting_models", q)


@pytest.fixture
def tool(tmp_path, monkeypatch):
    """Build the registered search_literature callable with stubs."""
    from scilink.agents.planning_agents import orchestrator_tools as ot

    monkeypatch.setattr(ot, "optimize_search_query",
                        lambda objective, model: f"optimized::{objective}")

    lit = StubLitAgent()
    orch = type("O", (), {})()
    orch.lit_agent = lit
    orch.planner = type("P", (), {"model": None})()

    tools = ot.OrchestratorTools.__new__(ot.OrchestratorTools)
    tools.orch = orch
    tools._output_dir = lambda: tmp_path
    captured = {}
    tools._register_tool = lambda func, name, description, parameters, required=None: (
        captured.update({name: (func, description, parameters)}))
    ot.OrchestratorTools._register_all_tools(tools)
    func, desc, params = captured["search_literature"]
    return func, lit, desc, params, tmp_path


# ---------------------------------------------------------------- tests

def test_cross_domain_prompt_forbids_topical_review():
    """The wrapper must ask for transfer, not a subfield review — the
    hypothesis_context wrapper reframes the task and kills cross-domain
    intent (observed live)."""
    agent = LiteratureSearchAgent.__new__(LiteratureSearchAgent)
    sent = {}
    agent._execute_crow_task = lambda q, task_type="general": sent.update(
        q=q, t=task_type) or {"status": "success", "content": ""}
    agent.search_for_cross_domain("selective lithium recovery from brine")

    q = sent["q"]
    assert "ADJACENT and UNRELATED domains" in q
    assert "TRANSFER" in q
    assert "NOT provide a topical review" in q
    assert "selective lithium recovery from brine" in q
    assert sent["t"] == "CrossDomain"
    # must NOT inherit the grounding template
    assert "comprehensive review of experimental methods" not in q


def test_single_type_unchanged(tool):
    func, lit, *_ = tool
    out = json.loads(func("obj", "hypothesis_context"))
    assert out["status"] == "success"
    assert out["searches_run"] == ["hypothesis_context"]
    assert [k for k, _ in lit.calls] == ["hypothesis_context"]
    assert lit.max_concurrent == 1
    assert "caveat" not in out          # caveat is cross-domain-only


def test_multi_type_runs_in_parallel_and_labels_sections(tool):
    func, lit, *_ , tmp = tool
    t0 = time.time()
    out = json.loads(func("obj", "hypothesis_context,cross_domain"))
    elapsed = time.time() - t0

    assert out["status"] == "success"
    assert set(out["searches_run"]) == {"hypothesis_context", "cross_domain"}
    # concurrency: both in flight at once, wall clock ~ one call not two
    assert lit.max_concurrent == 2
    assert elapsed < lit.delay * 1.8

    text = (tmp / "literature_search_hypothesis_context+cross_domain.md").read_text()
    assert "ESTABLISHED IN THIS FIELD" in text
    assert "TRANSFERABLE MECHANISMS FROM OTHER DOMAINS" in text
    assert "NOT established results in this field" in text
    assert text.index("ESTABLISHED IN THIS FIELD") < text.index("TRANSFERABLE")
    assert "CONTENT::hypothesis_context" in text and "CONTENT::cross_domain" in text


def test_cross_domain_carries_constraint_caveat(tool):
    func, *_ = tool
    out = json.loads(func("obj", "cross_domain"))
    assert "caveat" in out
    assert "constraint" in out["caveat"].lower()


def test_partial_failure_keeps_the_successful_half(tool, tmp_path, monkeypatch):
    from scilink.agents.planning_agents import orchestrator_tools as ot
    monkeypatch.setattr(ot, "optimize_search_query", lambda objective, model: objective)
    lit = StubLitAgent(fail=("cross_domain",))
    orch = type("O", (), {})()
    orch.lit_agent = lit
    orch.planner = type("P", (), {"model": None})()
    tools = ot.OrchestratorTools.__new__(ot.OrchestratorTools)
    tools.orch, tools._output_dir = orch, (lambda: tmp_path)
    cap = {}
    tools._register_tool = lambda func, name, description, parameters, required=None: (
        cap.update({name: func}))
    ot.OrchestratorTools._register_all_tools(tools)

    out = json.loads(cap["search_literature"]("obj", "hypothesis_context,cross_domain"))
    assert out["status"] == "success"
    assert out["searches_run"] == ["hypothesis_context"]
    assert out["failed_searches"] == ["cross_domain"]


def test_invalid_type_rejected(tool):
    func, *_ = tool
    out = json.loads(func("obj", "hypothesis_context,nonsense"))
    assert out["status"] == "error"
    assert "nonsense" in out["message"]


def test_tool_description_states_when_to_use_and_when_not(tool):
    _, _, desc, params, _ = tool
    st = params["search_type"]["description"]
    assert "cross_domain" in st
    assert "IDEATION" in st
    assert "AVOID" in st and "constraint" in st.lower()
    assert "parallel" in desc.lower() or "CONCURRENTLY" in desc
