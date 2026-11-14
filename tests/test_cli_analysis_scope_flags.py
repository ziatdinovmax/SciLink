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
    (["--targets", ""], "empty string"),
    (["--targets", "width", "  "], "empty string"),
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
    with pytest.raises(ValueError, match="prior_analysis_paths"):
        _orchestrator(tmp_path, monkeypatch, reuse_locked_script=True, prior_analysis_paths=[])


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


class _StrictAgent(_RecordingAgent):
    """A custom agent (--agents) whose analyze() takes none of the reuse arguments."""

    def analyze(self, data, system_info=None, objective=None, hints=None):
        type(self).calls.append({"ran": True})
        return {"status": "success", "detailed_analysis": "ok", "scientific_claims": []}


_UNSET = object()


def _run_analysis(o, tmp_path, agent=_RecordingAgent, name="curve.csv", data_path=_UNSET,
                  must_succeed=True, keep_calls=False, **tool_args):
    data = tmp_path / name
    data.write_text("x,y\n" + "\n".join(f"{i},{i * i}" for i in range(50)))
    if not keep_calls:
        _RecordingAgent.calls = []
        agent.calls = []
    # each call brings its own minimal metadata (analysis_goal), as a fresh
    # dataset does; the previous dataset's is not this one's
    o.current_metadata = None
    o.current_metadata_owner = None
    registered = {type(a) if not isinstance(a, type) else a: i
                  for i, a in o._agent_registry.items()}
    agent_id = registered.get(agent) or max(o._agent_registry) + 1
    if agent not in registered:
        o.register_agent(agent_id, agent)
    if data_path is not None:
        tool_args["data_path"] = str(data) if data_path is _UNSET else data_path
    out = json.loads(o.tools.execute_tool("run_analysis", agent_id=agent_id,
                                          analysis_goal="a test curve; fit it", **tool_args))
    assert agent.calls, out
    if must_succeed:
        assert out.get("status") != "error", out
    return agent.calls[-1]


def test_run_analysis_applies_the_session_scope_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, analysis_targets=["centre"],
                      analysis_time_budget_s=45, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    call = _run_analysis(o, tmp_path)          # the LLM passed none of them
    assert call["targets"] == ["centre"]
    assert call["profile"] == {"base": "thorough", "time_budget_s": 45.0}
    assert call["prior_analysis_paths"] == ["/runs/a"] and call["reuse_locked_script"] is True


def test_a_replay_takes_the_callers_recipe_first(tmp_path, monkeypatch):
    """A replay uses the first recipe it is given: the caller's leads."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    call = _run_analysis(o, tmp_path, prior_analysis_paths=["/runs/b", "/runs/a"])
    assert call["prior_analysis_paths"] == ["/runs/a", "/runs/b"]
    assert call["reuse_locked_script"] is True


def test_reference_material_goes_after_the_calls_own_paths(tmp_path, monkeypatch):
    """Without --reuse-locked-script the caller's runs are reference only:
    appended after the paths of a call that is not itself a replay."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"])
    call = _run_analysis(o, tmp_path, prior_analysis_paths=["/runs/b"])
    assert call["prior_analysis_paths"] == ["/runs/b", "/runs/a"]
    assert call["reuse_locked_script"] is False
    call = _run_analysis(o, tmp_path, name="other.csv")
    assert call["prior_analysis_paths"] == ["/runs/a"] and call["reuse_locked_script"] is False


def test_the_callers_run_is_never_a_fallback_recipe_for_the_calls_own_replay(tmp_path, monkeypatch):
    """A replay and script_edits use the first path that yields a recipe.
    With the caller's run appended, a call whose own path yields none
    replayed or edited the caller's script where main refuses."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"])
    call = _run_analysis(o, tmp_path, prior_analysis_paths=["/runs/b"], reuse_locked_script=True)
    assert call["prior_analysis_paths"] == ["/runs/b"] and call["reuse_locked_script"] is True
    call = _run_analysis(o, tmp_path, reuse_locked_script=True)        # no paths of its own
    assert call["prior_analysis_paths"] is None and call["reuse_locked_script"] is True


def test_a_forced_replay_has_a_way_out(tmp_path, monkeypatch):
    """The replay is forced on a dataset's FIRST analysis. The follow-up the
    system prompt asks for after a replay that did not fit ("re-run without
    reuse_locked_script"), or an independent verification, is not replayed
    again; another dataset's first analysis is."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    first = _run_analysis(o, tmp_path)
    assert first["reuse_locked_script"] is True
    again = _run_analysis(o, tmp_path)                       # same data: the LLM's follow-up
    assert again["reuse_locked_script"] is False
    assert again["prior_analysis_paths"] == ["/runs/a"]      # still there, as reference
    other = _run_analysis(o, tmp_path, name="next.csv")
    assert other["reuse_locked_script"] is True and other["prior_analysis_paths"] == ["/runs/a"]


def test_the_way_out_does_not_depend_on_how_the_path_is_spelled(tmp_path, monkeypatch):
    """Found in review: the follow-up was recognised by comparing the raw
    strings, so the same file as a relative path, with ./, or omitted (the
    orchestrator's absolute current_data_path) was forced to replay again."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    monkeypatch.chdir(tmp_path)
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    assert _run_analysis(o, tmp_path, data_path="curve.csv")["reuse_locked_script"] is True
    for spelling in (str(tmp_path / "curve.csv"), "./curve.csv",
                     str(tmp_path / "sub" / ".." / "curve.csv")):
        (tmp_path / "sub").mkdir(exist_ok=True)
        assert _run_analysis(o, tmp_path, data_path=spelling)["reuse_locked_script"] is False, spelling
    o.current_data_path = str((tmp_path / "curve.csv").absolute())
    assert _run_analysis(o, tmp_path, data_path=None)["reuse_locked_script"] is False


class _FailsOnce(_RecordingAgent):
    """First call: the given outcome. Later calls: a normal result."""
    outcome = "error"

    def analyze(self, data, system_info=None, **kwargs):
        first = not type(self).calls
        type(self).calls.append({"reuse_locked_script": kwargs.get("reuse_locked_script", False)})
        if first and type(self).outcome == "raise":
            raise RuntimeError("transient")
        if first:
            return {"status": "error", "error": "transient"}
        return {"status": "success", "detailed_analysis": "ok", "scientific_claims": []}


@pytest.mark.parametrize("outcome", ["error", "raise"])
def test_a_run_that_failed_outright_does_not_use_up_the_replay(tmp_path, monkeypatch, capsys,
                                                                outcome):
    """An error status used it up (the retry became a fresh derivation with
    new feature names) while an exception did not. Neither produced a result
    nor executed a replay, so the retry is still the caller's replay."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    _FailsOnce.outcome = outcome
    _run_analysis(o, tmp_path, agent=_FailsOnce, must_succeed=False)
    calls = list(_FailsOnce.calls)
    _run_analysis(o, tmp_path, agent=_FailsOnce, keep_calls=True)
    assert [c["reuse_locked_script"] for c in calls + _FailsOnce.calls[len(calls):]] == [True, True]
    capsys.readouterr()
    assert _run_analysis(o, tmp_path, agent=_FailsOnce, keep_calls=True) \
        ["reuse_locked_script"] is False                 # a result exists now: the way out
    # two forced attempts were made and the second SUCCEEDED: nothing was dropped
    assert "reuse is dropped" not in capsys.readouterr().out


class _AlwaysRefuses(_RecordingAgent):
    """Refuses every replay the way an agent with no recipe to replay does,
    and analyses normally otherwise. ``marked`` adds the reuse_validity the
    hyperspectral agent's refusal carries; a custom agent may not."""
    marked = False

    def analyze(self, data, system_info=None, **kwargs):
        reuse = kwargs.get("reuse_locked_script", False)
        type(self).calls.append({"reuse_locked_script": reuse})
        if reuse:
            out = {"status": "error", "error": {"error": "No approved prior script to replay"}}
            if type(self).marked:
                out["reuse_validity"] = {"reused": False, "verdict": "refused"}
            return out
        return {"status": "success", "detailed_analysis": "ok", "scientific_claims": []}


@pytest.mark.parametrize("marked, forced_calls", [(True, 1), (False, 2)])
def test_a_refused_replay_is_not_forced_forever(tmp_path, monkeypatch, capsys, marked, forced_calls):
    """Found in the second review: a refusal is deterministic, and an error
    record with no replay marker was read as a transient failure — the retry
    was forced and refused again on every call, with no argument that got
    out. A refusal that says what it is releases the dataset at once; one
    that does not is bounded at two forced attempts."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/curve_run"],
                      reuse_locked_script=True)
    _AlwaysRefuses.marked = marked
    seen = []
    for i in range(4):
        _run_analysis(o, tmp_path, agent=_AlwaysRefuses, must_succeed=False, keep_calls=i > 0)
        seen = [c["reuse_locked_script"] for c in _AlwaysRefuses.calls]
        if not seen[-1]:
            break
    assert seen == [True] * forced_calls + [False]            # released, then analysed
    assert o.analysis_results[-1]["status"] == "success"
    assert ("reuse is dropped for this dataset" in capsys.readouterr().out) is (not marked)
    # another dataset still gets the caller's replay
    assert _run_analysis(o, tmp_path, agent=_AlwaysRefuses, name="other.csv", must_succeed=False,
                         keep_calls=True)["reuse_locked_script"] is True


def test_the_hyperspectral_refusal_says_what_it_is(tmp_path, monkeypatch):
    """The real agent's 'no recipe to replay' result, read by the rule."""
    import numpy as np
    from scilink.agents.exp_agents.analysis_orchestrator_tools import caller_reuse
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    prior = tmp_path / "curve_run"                      # a run folder with no cube recipe in it
    prior.mkdir()
    (prior / "analysis_results.json").write_text("{}")
    agent = HyperspectralAnalysisAgent(model_name="anthropic/claude-sonnet-4-5",
                                       output_dir=str(tmp_path / "out"),
                                       enable_human_feedback=False)
    cube = tmp_path / "cube.npy"
    np.save(cube, np.zeros((4, 4, 8)))
    result = agent.analyze(str(cube), prior_analysis_paths=[str(prior)],
                           reuse_locked_script=True)
    assert result["status"] == "error"
    assert result["reuse_validity"] == {"reused": False, "verdict": "refused",
                                        "message": "No approved prior script to replay"}
    record = {"agent_id": 3, "status": "error", "data_path": str(cube), "full_result": result,
              "output_directory": str(tmp_path / "out")}
    assert caller_reuse(caller_prior=[str(prior)], caller_replays=True, records=[record],
                        data_path=str(cube)) == ([str(prior)], False, False)


def test_a_replay_of_a_run_this_session_made_is_kept_on_a_first_look(tmp_path, monkeypatch):
    """The session re-derived a model because the caller's recipe did not
    fit; the next dataset replaying THAT run is not forced back onto the
    recipe known not to fit. Any other path the model names is not trusted
    over the caller's."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    _run_analysis(o, tmp_path, name="s2.csv")                 # forced replay of /runs/a
    _run_analysis(o, tmp_path, name="s2.csv")                 # the re-derivation
    rederived = o.analysis_results[-1]["output_directory"]
    call = _run_analysis(o, tmp_path, name="s3.csv", prior_analysis_paths=[rederived],
                         reuse_locked_script=True)
    assert call["prior_analysis_paths"] == [rederived] and call["reuse_locked_script"] is True
    call = _run_analysis(o, tmp_path, name="s4.csv", prior_analysis_paths=["/elsewhere/run"],
                         reuse_locked_script=True)
    assert call["prior_analysis_paths"] == ["/runs/a", "/elsewhere/run"]


def test_the_reuse_rule_on_its_own(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import caller_reuse
    rec = lambda **kw: dict({"agent_id": 1, "status": "success", "data_path": "/d/a.csv",  # noqa: E731
                             "output_directory": "/s/run_a", "full_result": {}}, **kw)
    common = dict(caller_prior=["/runs/a"], data_path="/d/a.csv")
    # no caller: the call is what it was
    assert caller_reuse(caller_prior=None, caller_replays=False, records=[], data_path="/d/a.csv",
                        own_paths=["/x"], own_replay=True) == (["/x"], True, False)
    assert caller_reuse(caller_prior=None, caller_replays=False, records=[],
                        data_path="/d/a.csv") == (None, False, False)
    # a string from the model is one path
    assert caller_reuse(**common, caller_replays=True, records=[], own_paths="/x") == \
        (["/runs/a", "/x"], True, True)
    # preparation and derivation records are not a first analysis
    steps = [rec(agent_id="prepare"), rec(agent_id="derive")]
    assert caller_reuse(**common, caller_replays=True, records=steps)[2] is True
    # a replay that executed and did not fit released the dataset, whatever its status
    poor = [rec(status="error", full_result={"reuse_validity": {"reused": True, "verdict": "poor"}})]
    assert caller_reuse(**common, caller_replays=True, records=poor) == (["/runs/a"], False, False)
    # the backstop: two forced attempts that left nothing, then released
    assert caller_reuse(**common, caller_replays=True, records=[], forced_before=1)[2] is True
    assert caller_reuse(**common, caller_replays=True, records=[], forced_before=2) == \
        (["/runs/a"], False, "dropped")
    # ... and "dropped" is said only when the cap is the reason: a dataset whose
    # second forced attempt produced a result is a follow-up, not a drop
    done = [rec()]
    assert caller_reuse(**common, caller_replays=True, records=done, forced_before=2) == \
        (["/runs/a"], False, False)
    # a member file of a folder already analysed as a series is that dataset
    folder = tmp_path / "series"
    folder.mkdir()
    (folder / "a.csv").write_text("x,y\n0,0\n")
    series = [rec(data_path=str(folder))]
    assert caller_reuse(caller_prior=["/runs/a"], caller_replays=True, records=series,
                        data_path=str(folder / "a.csv"))[2] is False
    # an edit of the call's own run never gets the caller's script appended
    assert caller_reuse(**common, caller_replays=False, records=[], own_paths=["/s/run_a"],
                        own_edits=[{"old": "a", "new": "b"}]) == (["/s/run_a"], False, False)


def test_the_constructor_is_as_strict_as_the_command_line(tmp_path, monkeypatch):
    o = _orchestrator(tmp_path, monkeypatch, analysis_targets="D/G ratio",
                      prior_analysis_paths="/runs/a")
    assert o.default_targets == ["D/G ratio"]                 # not one target per character
    assert o.default_prior_analysis_paths == ["/runs/a"]
    for bad in (float("inf"), float("nan"), True, 0, "90"):
        with pytest.raises(ValueError, match="positive, finite"):
            _orchestrator(tmp_path, monkeypatch, analysis_time_budget_s=bad)


def test_a_custom_agent_that_takes_no_prior_run_is_run_as_before(tmp_path, monkeypatch, capsys):
    """It used to fail on every call with an unexpected keyword argument."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    o = _orchestrator(tmp_path, monkeypatch, prior_analysis_paths=["/runs/a"],
                      reuse_locked_script=True)
    call = _run_analysis(o, tmp_path, agent=_StrictAgent)
    assert call == {"ran": True}
    assert "is not used" in capsys.readouterr().out


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


def test_meta_refuses_an_empty_target_at_the_command_line(capsys):
    """Not as a constructor exception after the session has started."""
    with pytest.raises(SystemExit):
        _parse(MetaAdapter(), ["--targets", ""])
    assert "empty string" in capsys.readouterr().err


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
