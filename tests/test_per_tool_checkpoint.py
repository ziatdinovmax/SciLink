"""Offline tests for per-tool-iteration checkpointing (Phase A).

The chat loops persisted history + state only at turn end (or every N
messages), so a crash mid-turn lost every completed tool call since the
last turn boundary — one turn can contain a whole run_analysis, delegation
or fan-out. Each orchestrator now calls ``_checkpoint_after_tool()`` after
every completed tool iteration: quiet, wall-clock-throttled writes of
chat_history.json + checkpoint.json, mirroring the MCP server's
checkpoint-after-tool. A mid-turn history ends in a dangling tool_use at
worst, which the existing repair machinery heals on the next load.

  python -m pytest tests/test_per_tool_checkpoint.py -v
"""
import json
import os
from types import SimpleNamespace

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest


def _mk_response(tool_calls=None, content=None):
    msg = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg,
                                                    finish_reason="stop")])


def _mk_tool_call(call_id, name, args="{}"):
    return SimpleNamespace(id=call_id,
                           function=SimpleNamespace(name=name,
                                                    arguments=args))


class _Boom(BaseException):
    """Simulates the process dying mid-turn: bypasses every except Exception
    (chat()'s catch-all included), so no turn-end save can run."""


def _scripted_completion(script):
    """Yield canned responses; raise _Boom when the script says so."""
    it = iter(script)

    def fake(*a, **k):
        step = next(it)
        if step == "boom":
            raise _Boom()
        return step
    return fake


# ------------------------------------------------------------ analysis, meta


def _make_analysis(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    return AnalysisOrchestratorAgent(api_key="sk-dummy",
                                     base_dir=str(tmp_path / "an"))


def _make_meta(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    return MetaOrchestratorAgent(api_key="sk-dummy",
                                 base_dir=str(tmp_path / "meta"))


@pytest.mark.parametrize("which", ["analysis", "meta"])
def test_mid_turn_crash_preserves_completed_tool_calls(which, tmp_path,
                                                       monkeypatch):
    orch = (_make_analysis if which == "analysis" else _make_meta)(tmp_path)
    orch.TOOL_CHECKPOINT_MIN_INTERVAL_S = 0.0
    orch.use_openai = False
    monkeypatch.setattr(orch.tools, "execute_tool",
                        lambda name, **kw: f"TOOL-OK::{name}")

    import scilink.wrappers.litellm_wrapper as lw
    monkeypatch.setattr(lw, "litellm_completion", _scripted_completion([
        _mk_response(tool_calls=[_mk_tool_call("c1", "some_tool")]),
        "boom",   # the process dies before the turn can complete
    ]))

    with pytest.raises(_Boom):
        orch.chat("do something")

    # The completed tool call survived on disk despite the mid-turn death.
    hist = json.loads(orch.history_path.read_text(encoding="utf-8"))
    roles = [m.get("role") for m in hist]
    assert "tool" in roles, f"{which}: no tool result persisted mid-turn"
    tool_msgs = [m for m in hist if m.get("role") == "tool"]
    assert any("TOOL-OK::some_tool" in str(m.get("content"))
               for m in tool_msgs)
    assert orch.checkpoint_path.exists()

    # A new session on the same dir picks the mid-turn history up and can
    # run a clean turn (the loop's repair heals the dangling tool_use).
    orch2 = (_make_analysis if which == "analysis" else _make_meta)(tmp_path)
    assert any(m.get("role") == "tool" for m in orch2.messages)
    monkeypatch.setattr(lw, "litellm_completion", _scripted_completion([
        _mk_response(content="recovered fine"),
    ]))
    monkeypatch.setattr(orch2.tools, "execute_tool",
                        lambda name, **kw: "unused")
    reply = orch2.chat("continue where you left off")
    assert "recovered fine" in reply


# --------------------------------------------------- helper contract, all 4


def _make(mode, tmp_path):
    if mode == "analysis":
        return _make_analysis(tmp_path)
    if mode == "meta":
        return _make_meta(tmp_path)
    if mode == "planning":
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent)
        return PlanningOrchestratorAgent(api_key="sk-dummy",
                                         data_dir=str(tmp_path),
                                         base_dir=str(tmp_path / "pl"))
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent)
    return SimulationOrchestratorAgent(api_key="sk-dummy",
                                       base_dir=str(tmp_path / "si"))


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate", "meta"])
def test_checkpoint_after_tool_writes_both_files(mode, tmp_path):
    orch = _make(mode, tmp_path)
    orch.TOOL_CHECKPOINT_MIN_INTERVAL_S = 0.0
    orch.messages.append({"role": "user", "content": "hello"})
    if orch.history_path.exists():
        orch.history_path.unlink()
    if orch.checkpoint_path.exists():
        orch.checkpoint_path.unlink()

    orch._checkpoint_after_tool()

    assert orch.history_path.exists(), f"{mode}: history not written"
    assert orch.checkpoint_path.exists(), f"{mode}: checkpoint not written"
    hist = json.loads(orch.history_path.read_text(encoding="utf-8"))
    assert any(m.get("content") == "hello" for m in hist)


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate", "meta"])
def test_checkpoint_after_tool_is_wall_clock_throttled(mode, tmp_path):
    orch = _make(mode, tmp_path)
    orch.TOOL_CHECKPOINT_MIN_INTERVAL_S = 3600.0

    orch._checkpoint_after_tool()          # first call: writes
    assert orch.history_path.exists()
    orch.history_path.unlink()
    orch._checkpoint_after_tool()          # inside the window: skipped
    assert not orch.history_path.exists(), f"{mode}: throttle did not hold"

    orch._last_tool_checkpoint_ts = 0.0    # window elapsed
    orch._checkpoint_after_tool()
    assert orch.history_path.exists()


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate", "meta"])
def test_checkpoint_after_tool_is_quiet(mode, tmp_path, capsys):
    orch = _make(mode, tmp_path)
    orch.TOOL_CHECKPOINT_MIN_INTERVAL_S = 0.0
    capsys.readouterr()
    orch._checkpoint_after_tool()
    out = capsys.readouterr().out
    assert "Auto-checkpoint saved" not in out, (
        f"{mode}: per-tool cadence must not spam the console")


def test_sim_interval_gate_still_holds_without_force(tmp_path):
    """The simulate orchestrator's N-message auto-checkpoint cadence is
    unchanged: a plain _auto_checkpoint() inside the interval still skips."""
    orch = _make("simulate", tmp_path)
    if orch.checkpoint_path.exists():
        orch.checkpoint_path.unlink()
    orch.message_count = orch.last_checkpoint_message_count  # inside interval
    orch._auto_checkpoint()
    assert not orch.checkpoint_path.exists()
    orch._auto_checkpoint(force=True, quiet=True)
    assert orch.checkpoint_path.exists()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
