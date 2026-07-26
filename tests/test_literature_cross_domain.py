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
- ANY literature result carries the constraint-coverage caveat (the loss was
  measured for grounding-only retrieval too, so it is not cross-domain-specific);
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
    assert "NOT be a topical review" in q
    assert "selective lithium recovery from brine" in q
    assert sent["t"] == "CrossDomain"
    # must NOT inherit the grounding template
    assert "comprehensive review of experimental methods" not in q

    # The ban is on the BODY, not on naming the target: an absolute "no
    # subfield content" rule deadlocks whenever the objective is phrased as
    # a question about its own field, forcing the model to disobey either
    # the wrapper or the question. State the function, then leave the field.
    assert "FUNCTION to be transferred toward" in q
    assert q.index("FUNCTION to be transferred") < q.index("NOT be a topical")
    # each analogy must carry its failure mode, not just an appealing story
    assert "what would break in transfer" in q
    assert "how it is measured or realized" in q


def test_single_type_unchanged(tool):
    func, lit, *_ = tool
    out = json.loads(func("obj", "hypothesis_context"))
    assert out["status"] == "success"
    assert out["searches_run"] == ["hypothesis_context"]
    assert [k for k, _ in lit.calls] == ["hypothesis_context"]
    assert lit.max_concurrent == 1


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


@pytest.mark.parametrize("types", ["hypothesis_context", "cross_domain",
                                   "hypothesis_context,cross_domain"])
def test_constraint_coverage_caveat_is_literature_general(tool, types):
    """The coverage loss was measured for grounding-only literature too, so
    the caveat must not be cross-domain-specific — and it must point at the
    remedy (map constraints to steps) rather than at withholding literature."""
    func, *_ = tool
    out = json.loads(func("obj", types))
    assert "constraint" in out["caveat"].lower()
    assert "additional_context" in out["caveat"]
    assert "avoid" not in out["caveat"].lower()


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
    # cross_domain must NOT be singled out as constraint-hostile: the coverage
    # loss was measured for grounding-only retrieval too, and the remedy is
    # constraint->step mapping, not withholding a search type.
    assert "AVOID" not in st
    assert "parallel" in desc.lower() or "concurrent" in desc.lower()


# ---------------------------------------------------------------- display

def test_intro_states_question_times_type_multiplication(tool, capsys):
    """The Q-list shows DISTINCT questions; the header must say each one
    runs once per search type — '5 questions but 10 searches' read as a
    mismatch in live sessions — and scale the wait by batch count."""
    func, *_ = tool
    func([f"question {i}" for i in range(1, 5)],
         "hypothesis_context,cross_domain")  # 8 jobs -> batches of 6 + 2
    out = capsys.readouterr().out
    assert ("4 questions, each searched 2 ways "
            "(hypothesis_context, cross_domain) = 8 searches") in out
    # exact batch sizes, not 'batches of 6' — that read as 6 x 2 = 12
    assert "run in 2 sequential batches of 6 then 2" in out
    assert "10-15 minutes per batch" in out
    assert "expect ~20-30 minutes total" in out


def test_intro_single_batch_keeps_flat_eta(tool, capsys):
    func, *_ = tool
    func(["q1", "q2"], "hypothesis_context,cross_domain")  # 4 jobs, 1 batch
    out = capsys.readouterr().out
    assert "2 questions, each searched 2 ways" in out
    assert "sequential batches" not in out
    assert "per batch" not in out
    assert "typically take 10-15 minutes" in out


def test_heartbeat_reports_running_vs_queued(tool, monkeypatch, capsys):
    """Batches run back-to-back, so not-yet-submitted jobs are QUEUED, not
    running — the old single 'N still running' line overstated concurrency
    whenever a call exceeded the 6-wide budget (seen live: '10 of 10 still
    running' with 6 in flight)."""
    from scilink.agents.planning_agents import orchestrator_tools as ot
    func, lit, *_ = tool
    monkeypatch.setattr(ot, "_LIT_HEARTBEAT_SECONDS", 0.15)
    lit.delay = 0.9  # batch 1 spans several heartbeat ticks
    func([f"q{i}" for i in range(1, 5)],
         "hypothesis_context,cross_domain")  # 8 jobs: 6 running + 2 queued
    out = capsys.readouterr().out
    assert "6 running, 2 queued of 8 literature searches" in out
    assert "~10-15 min per batch" in out


def test_search_type_routing_excludes_survey_questions(tool):
    """cross_domain needs a function to transfer TOWARD. A survey question
    ("what are the frontiers in X", "what is hard to measure in X") asks for
    X's own field, which the wrapper forbids in the same breath — the two
    instructions cannot both be satisfied, so the router must not send one
    there."""
    _, _, _, params, _ = tool
    st = params["search_type"]["description"]
    assert "FUNCTION or CHALLENGE" in st
    assert "survey question" in st
    assert "hypothesis_context" in st.split("survey question")[1]


# ------------------------------------------------- per-type objectives

def test_per_type_objectives_send_each_type_its_own_question(tool, tmp_path):
    """A paired call cross-multiplied ONE text across both types, so the
    cross_domain leg received a survey question written for grounding —
    exactly the shape its wrapper tells the model to avoid. Each type can
    now carry its own objective."""
    func, lit, *_ = tool
    out = json.loads(func({
        "hypothesis_context": ["what are the gaps in biomineralization",
                               "what is known about amorphous precursors"],
        "cross_domain": "capture a state that exists only under drive",
    }))
    assert out["status"] == "success"

    by_kind = {}
    for kind, q in lit.calls:
        by_kind.setdefault(kind, []).append(q)
    assert sorted(by_kind) == ["cross_domain", "hypothesis_context"]
    assert len(by_kind["hypothesis_context"]) == 2
    assert by_kind["cross_domain"] == ["optimized::capture a state that "
                                       "exists only under drive"]
    # the grounding questions never reach the transfer leg, and vice versa
    assert not any("gaps in biomineralization" in q
                   for q in by_kind["cross_domain"])
    assert not any("only under drive" in q
                   for q in by_kind["hypothesis_context"])
    assert set(out["searches_run"]) and len(lit.calls) == 3


def test_shared_objective_still_cross_multiplies(tool):
    """Back-compat: a string or list keeps the historical behaviour."""
    func, lit, *_ = tool
    json.loads(func(["q1", "q2"], "hypothesis_context,cross_domain"))
    kinds = sorted(k for k, _ in lit.calls)
    assert kinds == ["cross_domain", "cross_domain",
                     "hypothesis_context", "hypothesis_context"]


def test_per_type_objectives_validate_and_report_honestly(tool, capsys):
    func, lit, *_ = tool
    bad = json.loads(func({"nonsense": "q"}))
    assert bad["status"] == "error" and "nonsense" in bad["message"]

    empty = json.loads(func({"cross_domain": []}))
    assert empty["status"] == "error"

    capsys.readouterr()
    json.loads(func({"hypothesis_context": ["a", "b"], "cross_domain": ["c"]}))
    header = capsys.readouterr().out
    # no "N questions each searched M ways" — that arithmetic is false here
    assert "2 for hypothesis_context, 1 for cross_domain = 3 searches" in header
    assert "each searched" not in header
    assert "[hypothesis_context]" in header and "[cross_domain]" in header


def test_identical_question_for_two_types_is_optimized_once(tool):
    """Dedup by text: the same question asked of both types costs one query
    optimization, but still runs once per type."""
    func, lit, *_ = tool
    json.loads(func({"hypothesis_context": ["same q"],
                     "cross_domain": ["same q"]}))
    assert sorted(k for k, _ in lit.calls) == ["cross_domain",
                                               "hypothesis_context"]
    assert {q for _, q in lit.calls} == {"optimized::same q"}
