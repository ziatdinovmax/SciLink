"""The typed depth channel: meta → run_task → run_analysis → agent.

How good an analysis has to be is known to whoever consumes it. Before this,
the only way to say so from the meta was prose inside ``context``: serialized
as JSON text into the child's prompt, where the child orchestrator's LLM then
had to choose to repeat it as a ``run_analysis`` argument — a decision
travelling through two LLM hops. These tests pin the structural replacement:
typed arguments at every hop, applied deterministically at the bottom.

No LLM calls anywhere.
"""

import json
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents._qc_profile import verification_addendum
from scilink.agents.exp_agents.analysis_orchestrator import (
    AnalysisMode, AnalysisOrchestratorAgent)


# ──────────────────────────────────────────────────────────────
# bottom: targets scope the verifier
# ──────────────────────────────────────────────────────────────

class TestTargetsScopeTheVerifier:
    def test_targets_win_over_the_objective(self):
        a = verification_addendum({
            "_verification_mode": "purpose",
            "analysis_targets": ["G-band position", "D/G ratio"],
            "analysis_objective": "characterise the carbon film"})
        assert "G-band position, D/G ratio" in a and "characterise" not in a

    def test_targets_scope_a_thorough_run_too(self):
        # A caller that names the numbers it needs has said what the result
        # is for; thoroughness is then spent on those numbers.
        a = verification_addendum({"_verification_mode": "strict",
                                   "analysis_targets": ["peak_1 centre"]})
        assert a is not None and "peak_1 centre" in a

    def test_no_targets_and_strict_is_todays_behavior(self):
        assert verification_addendum({"_verification_mode": "strict",
                                      "analysis_targets": []}) is None

    @pytest.mark.parametrize("module, cls", [
        ("curve_fitting_agent", "CurveFittingAgent"),
        ("image_analysis_agent", "ImageAnalysisAgent")])
    def test_agents_accept_targets(self, module, cls):
        import importlib, inspect
        agent = getattr(importlib.import_module(
            f"scilink.agents.exp_agents.{module}"), cls)
        assert "targets" in inspect.signature(agent.analyze).parameters


# ──────────────────────────────────────────────────────────────
# middle: run_task sets the orchestrator's defaults for one call
# ──────────────────────────────────────────────────────────────

def _orch():
    o = AnalysisOrchestratorAgent.__new__(AnalysisOrchestratorAgent)
    o.analysis_mode = AnalysisMode.AUTONOMOUS
    o.max_iterations = 5
    o.analysis_results = []
    o.default_profile = None
    o.default_targets = None
    o.default_time_budget_s = None
    o._last_chat_hit_iter_cap = False
    o._last_chat_error = None
    o.logger = SimpleNamespace(exception=lambda *a, **k: None,
                               info=lambda *a, **k: None,
                               warning=lambda *a, **k: None)
    o.set_analysis_mode = lambda m: setattr(o, "analysis_mode", m)
    o._auto_checkpoint = lambda *a, **k: None
    o.seen = {}

    def chat(prompt):
        o.seen = {"prompt": prompt, "profile": o.default_profile,
                  "targets": o.default_targets, "budget": o.default_time_budget_s}
        return "done"
    o.chat = chat
    return o


class TestRunTaskCarriesDepth:
    def test_defaults_are_set_during_the_call_and_restored_after(self):
        o = _orch()
        res = o.run_task("fit it", profile="extract",
                         targets=["centre_1"], time_budget_s=90)
        assert o.seen["profile"] == "extract" and o.seen["targets"] == ["centre_1"]
        assert o.seen["budget"] == 90.0
        assert (o.default_profile, o.default_targets, o.default_time_budget_s) == (None, None, None)
        assert res["depth"] == {"profile": "extract", "targets": ["centre_1"],
                                "time_budget_s": 90}

    def test_a_session_default_survives_a_delegation(self):
        o = _orch()
        o.default_profile = "quick"          # e.g. scilink analyze --profile quick
        o.run_task("fit it", profile="extract")
        assert o.seen["profile"] == "extract" and o.default_profile == "quick"
        o.run_task("fit it")
        assert o.seen["profile"] == "quick"

    def test_restored_even_when_the_task_raises(self):
        o = _orch()
        o.chat = lambda prompt: (_ for _ in ()).throw(RuntimeError("boom"))
        res = o.run_task("fit it", profile="quick")
        assert res["status"] == "error" and o.default_profile is None

    def test_no_depth_is_todays_prompt_and_result(self):
        o = _orch()
        res = o.run_task("fit it", context={"k": 1})
        assert "depth" not in res and "fixed the analysis depth" not in o.seen["prompt"]

    def test_the_prompt_tells_the_orchestrator_once(self):
        o = _orch()
        o.run_task("fit it", profile="extract")
        assert o.seen["prompt"].count("fixed the analysis depth") == 1
        assert '"profile": "extract"' in o.seen["prompt"]

    def test_a_typo_fails_before_any_work(self):
        with pytest.raises(ValueError, match="Unknown QC profile"):
            _orch().run_task("fit it", profile="fast")


# ──────────────────────────────────────────────────────────────
# top: the meta passes typed arguments, and records them
# ──────────────────────────────────────────────────────────────

class _Child:
    def __init__(self):
        self.calls = []

    def run_task(self, task, context=None, autonomy=None, **depth):
        self.calls.append({"task": task, "context": context, **depth})
        return {"status": "success", "summary": "ok", "key_findings": [],
                "files_produced": [], "suggested_followups": [], "warnings": []}


def _meta(child):
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m.meta_mode = MetaMode.AUTONOMOUS
    m.logger = SimpleNamespace(exception=lambda *a, **k: None)
    m.opened = []
    m._get_analysis_child = lambda: child
    m._get_planning_child = lambda: child
    m._open_delegation = lambda mode, task, ctx, cf, label=None: (
        m.opened.append({"index": len(m.opened), "mode": mode, "task": task})
        or m.opened[-1])
    m._close_delegation = lambda entry, result: None
    m._auto_checkpoint = lambda *a, **k: None
    m._summarize_delegation_result = lambda mode, result, index: json.dumps(
        {"status": result["status"], "delegation_index": index})
    return m


class TestMetaDelegatesTypedDepth:
    def test_depth_reaches_run_task_as_arguments_not_context(self):
        child = _Child()
        m = _meta(child)
        m._delegate("analysis", "fit /data/a.csv", {"note": "x"}, None, "raman fit",
                    depth={"profile": "extract", "targets": ["G position"],
                           "time_budget_s": None})
        [call] = child.calls
        assert call["profile"] == "extract" and call["targets"] == ["G position"]
        assert "time_budget_s" not in call           # empty values are not sent
        assert call["context"] == {"note": "x"}       # context is untouched
        assert m.opened[0]["depth"] == {"profile": "extract", "targets": ["G position"]}

    def test_no_depth_is_todays_call(self):
        child = _Child()
        m = _meta(child)
        m._delegate("analysis", "fit", None, None, "x")
        assert set(child.calls[0]) == {"task", "context"}
        assert "depth" not in m.opened[0]

    def test_depth_is_not_sent_to_other_specialists(self):
        child = _Child()
        m = _meta(child)
        m._delegate("planning", "plan", None, None, "x", depth={"profile": "quick"})
        assert set(child.calls[0]) == {"task", "context"}

    def test_the_tool_schema_offers_the_cold_start_presets_only(self):
        from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
        tools = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
        tools.orch = SimpleNamespace(_delegate=lambda *a, **k: "{}")
        tools.functions, tools.schemas = {}, []
        registered = {}
        tools._register_tool = lambda func, name, description, parameters, required=None, **kw: \
            registered.__setitem__(name, parameters)
        try:
            tools._register_all_tools()
        except Exception:
            pass                                      # later tools need a fuller orch
        params = registered["delegate_to_analysis"]
        assert params["profile"]["enum"] == ["thorough", "quick", "extract"]
        assert params["targets"]["type"] == "array"
