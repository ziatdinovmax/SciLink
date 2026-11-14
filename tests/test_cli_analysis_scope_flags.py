"""The typed analysis controls as flags: ``--targets``, ``--time-budget``,
``--prior-analysis``, ``--reuse-locked-script`` on ``scilink analyze`` and
``--profile`` / ``--targets`` / ``--time-budget`` on the meta.

Before, a command-line caller could only ask for these in prose, so scope,
budget and recipe reuse depended on the orchestrator LLM repeating them as
``run_analysis`` arguments. They are now session defaults applied in
``run_analysis`` itself. No LLM calls.
"""
import json
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
from scilink.cli.shell.app import build_parser
from scilink.cli.shell.modes import AnalyzeAdapter, MetaAdapter


def _parse(adapter, argv):
    parser = build_parser(adapter)
    args = parser.parse_args(argv)
    adapter.validate_args(parser, args)
    return args


def _built_with(adapter, args, monkeypatch, module, cls):
    """The kwargs the adapter builds its orchestrator with."""
    import importlib
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(importlib.import_module(module), cls, fake)
    creds = SimpleNamespace(api_key="k", base_url=None)
    adapter.build(args, creds, "session", restore=False, extras={})
    return seen


# ── scilink analyze ──────────────────────────────────────────────

def test_analyze_flags_reach_the_orchestrator(tmp_path, monkeypatch):
    prior = tmp_path / "run1"
    prior.mkdir()
    monkeypatch.chdir(tmp_path)
    adapter = AnalyzeAdapter()
    args = _parse(adapter, ["--targets", "G band position", "D/G ratio", "--time-budget", "90",
                            "--prior-analysis", "run1", "--reuse-locked-script",
                            "--profile", "quick"])
    seen = _built_with(adapter, args, monkeypatch,
                       "scilink.agents.exp_agents.analysis_orchestrator",
                       "AnalysisOrchestratorAgent")
    assert seen["analysis_targets"] == ["G band position", "D/G ratio"]
    assert seen["analysis_time_budget_s"] == 90.0 and seen["analysis_profile"] == "quick"
    assert seen["prior_analysis_paths"] == [str(prior.absolute())]     # absolute, not session-relative
    assert seen["reuse_locked_script"] is True


def test_analyze_without_the_flags_is_built_as_before(monkeypatch):
    adapter = AnalyzeAdapter()
    seen = _built_with(adapter, _parse(adapter, []), monkeypatch,
                       "scilink.agents.exp_agents.analysis_orchestrator",
                       "AnalysisOrchestratorAgent")
    assert seen["analysis_targets"] is None and seen["analysis_time_budget_s"] is None
    assert seen["prior_analysis_paths"] is None and seen["reuse_locked_script"] is False


@pytest.mark.parametrize("argv, message", [
    (["--reuse-locked-script"], "needs --prior-analysis"),
    (["--prior-analysis", "no_such_run"], "does not exist"),
    (["--time-budget", "0"], "positive"),
    (["--time-budget", "-5"], "positive"),
    (["--time-budget", "soon"], "not a number"),
    (["--time-budget", "inf"], "positive"),
    (["--time-budget", "nan"], "positive"),
])
def test_analyze_refuses_what_cannot_work(argv, message, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _parse(AnalyzeAdapter(), argv)
    assert message in capsys.readouterr().err


# ── the orchestrator keeps them as session defaults ──────────────

def _orchestrator(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return AnalysisOrchestratorAgent(base_dir=str(tmp_path / "session"),
                                     model_name="anthropic/claude-sonnet-4-5", **kwargs)


def test_constructor_sets_the_session_defaults(tmp_path, monkeypatch):
    o = _orchestrator(tmp_path, monkeypatch, analysis_targets=[" centre ", "", "width"],
                      analysis_time_budget_s=120, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    assert o.default_targets == ["centre", "width"] and o.default_time_budget_s == 120.0
    assert o.default_prior_analysis_paths == ["/runs/a"] and o.default_reuse_locked_script is True


def test_constructor_defaults_are_unset(tmp_path, monkeypatch):
    o = _orchestrator(tmp_path, monkeypatch)
    assert o.default_targets is None and o.default_time_budget_s is None
    assert o.default_prior_analysis_paths is None and o.default_reuse_locked_script is False


def test_constructor_refuses_a_replay_with_no_recipe(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="prior_analysis_paths"):
        _orchestrator(tmp_path, monkeypatch, reuse_locked_script=True)
    with pytest.raises(ValueError, match="positive"):
        _orchestrator(tmp_path, monkeypatch, analysis_time_budget_s=-1)


def test_a_tasks_own_depth_still_replaces_the_session_default_for_that_task(tmp_path, monkeypatch):
    """The rule run_task already had (tests/test_depth_channel.py), now also
    for a session default that came from --targets / --time-budget."""
    o = _orchestrator(tmp_path, monkeypatch, analysis_targets=["a"], analysis_time_budget_s=60)
    seen = {}

    def chat(prompt):
        seen.update(targets=o.default_targets, budget=o.default_time_budget_s)
        return "done"
    o.chat = chat
    o.run_task("t")
    assert seen == {"targets": ["a"], "budget": 60.0}
    o.run_task("t", targets=["b"], time_budget_s=30)
    assert seen == {"targets": ["b"], "budget": 30.0}
    assert o.default_targets == ["a"] and o.default_time_budget_s == 60.0


# ── run_analysis applies them, whatever the LLM passed ───────────

class _RecordingAgent:
    """Stands in for an analysis agent: records what analyze() was given."""
    calls = []

    def __init__(self, *a, **k):
        pass

    def analyze(self, data, system_info=None, objective=None, hints=None, targets=None,
                profile=None, prior_analysis_paths=None, reuse_locked_script=False, **kwargs):
        type(self).calls.append({"targets": targets, "profile": profile,
                                 "prior_analysis_paths": prior_analysis_paths,
                                 "reuse_locked_script": reuse_locked_script})
        return {"status": "success", "detailed_analysis": "ok", "scientific_claims": []}


def _run_analysis(o, tmp_path, **tool_args):
    data = tmp_path / "curve.csv"
    data.write_text("x,y\n" + "\n".join(f"{i},{i * i}" for i in range(50)))
    _RecordingAgent.calls = []
    agent_id = max(o._agent_registry) + 1
    o.register_agent(agent_id, _RecordingAgent)
    out = json.loads(o.tools.execute_tool("run_analysis", data_path=str(data),
                                          agent_id=agent_id,
                                          analysis_goal="a test curve; fit it", **tool_args))
    assert _RecordingAgent.calls, out
    return _RecordingAgent.calls[-1]


def test_run_analysis_applies_the_session_scope_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, analysis_targets=["centre"],
                      analysis_time_budget_s=45, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    call = _run_analysis(o, tmp_path)          # the LLM passed none of them
    assert call["targets"] == ["centre"]
    assert call["profile"] == {"base": "thorough", "time_budget_s": 45.0}
    assert call["prior_analysis_paths"] == ["/runs/a"] and call["reuse_locked_script"] is True


def test_the_callers_recipe_comes_first_and_the_calls_own_are_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"])
    call = _run_analysis(o, tmp_path, prior_analysis_paths=["/runs/b", "/runs/a"])
    assert call["prior_analysis_paths"] == ["/runs/a", "/runs/b"]
    assert call["reuse_locked_script"] is False          # reference only: not asked for


def test_no_session_defaults_leaves_the_call_as_it_was(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch)
    call = _run_analysis(o, tmp_path, prior_analysis_paths=["/runs/b"])
    assert call == {"targets": None, "profile": None, "prior_analysis_paths": ["/runs/b"],
                    "reuse_locked_script": False}


# ── the meta ─────────────────────────────────────────────────────

def test_meta_flags_become_the_sessions_default_depth(monkeypatch):
    adapter = MetaAdapter()
    args = _parse(adapter, ["--profile", "extract", "--targets", "D/G ratio",
                            "--time-budget", "300"])
    seen = _built_with(adapter, args, monkeypatch,
                       "scilink.agents.meta_agent.meta_orchestrator", "MetaOrchestratorAgent")
    assert seen["analysis_depth"] == {"profile": "extract", "targets": ["D/G ratio"],
                                      "time_budget_s": 300.0}


def test_meta_builds_its_analysis_children_with_the_default_depth(tmp_path, monkeypatch):
    from scilink.agents.meta_agent import workers
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def meta(name, **kw):
        return MetaOrchestratorAgent(base_dir=str(tmp_path / name),
                                     model_name="anthropic/claude-sonnet-4-5",
                                     meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path), **kw)

    m = meta("m1", analysis_depth={"profile": "quick", "targets": ["width"],
                                   "time_budget_s": 200, "unrelated": 1})
    assert m.default_analysis_depth == {"profile": "quick", "targets": ["width"],
                                        "time_budget_s": 200.0}
    for persistent in (True, False):                     # the specialist and a fan-out worker
        child = workers.build_child(m, "analysis", tmp_path / f"child_{persistent}",
                                    persistent=persistent)
        assert (child.default_profile, child.default_targets, child.default_time_budget_s) == \
            ("quick", ["width"], 200.0)

    plain = meta("m2")
    assert plain.default_analysis_depth == {}
    child = workers.build_child(plain, "analysis", tmp_path / "child_plain")
    assert (child.default_profile, child.default_targets, child.default_time_budget_s) == \
        (None, None, None)

    with pytest.raises(ValueError):
        meta("m3", analysis_depth={"profile": "no-such-profile"})
