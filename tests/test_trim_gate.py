"""The in-loop history-trim trigger must derive from MAX_HISTORY_MESSAGES
(+ TRIM_HYSTERESIS), not the old hardcoded 120 — so configuring the cap
actually moves the trigger, while the default (100 + 20) stays byte-
identical to the literal it replaces.

  python -m pytest tests/test_trim_gate.py -v
"""
import inspect
import os
import types

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest


def _classes():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent)
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    return {"analysis": AnalysisOrchestratorAgent,
            "planning": PlanningOrchestratorAgent,
            "simulate": SimulationOrchestratorAgent,
            "meta": MetaOrchestratorAgent}


def test_default_gate_value_unchanged():
    for mode, cls in _classes().items():
        assert cls.MAX_HISTORY_MESSAGES + cls.TRIM_HYSTERESIS == 120, mode


def test_no_hardcoded_gate_literal():
    for mode, cls in _classes().items():
        for handler in ("_handle_litellm_chat", "_handle_openai_chat"):
            src = inspect.getsource(getattr(cls, handler))
            assert "> 120" not in src, f"{mode}.{handler} keeps the literal"
            assert "MAX_HISTORY_MESSAGES + self.TRIM_HYSTERESIS" in src, \
                f"{mode}.{handler} gate not derived from the cap"


def _fake_completion(**_kw):
    msg = types.SimpleNamespace(content="ok", tool_calls=None)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])


def _make(mode, cls, tmp_path):
    kw = dict(api_key="sk-dummy", base_dir=str(tmp_path / mode))
    if mode == "planning":
        kw["data_dir"] = str(tmp_path)
    return cls(**kw)


def test_small_cap_does_not_grow_history(tmp_path):
    """Pre-fix, max_messages <= 10 made recent_window 0 and history[-0:]
    the WHOLE list — the 'trim' returned first-10 + everything (grew,
    with duplicates)."""
    ag = _make("analysis", _classes()["analysis"], tmp_path)
    history = [{"role": "user", "content": f"m{i}"} for i in range(50)]
    trimmed = ag._trim_history(history, max_messages=10)
    assert len(trimmed) <= 11              # 10 + marker
    contents = [m["content"] for m in trimmed]
    assert len(contents) == len(set(contents))   # no duplicated messages


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate", "meta"])
def test_lowered_cap_moves_the_trigger(mode, tmp_path, monkeypatch):
    import scilink.wrappers.litellm_wrapper as lw
    monkeypatch.setattr(lw, "litellm_completion", _fake_completion)

    ag = _make(mode, _classes()[mode], tmp_path)
    ag.MAX_HISTORY_MESSAGES = 10          # gate is now 10 + 20 = 30

    def _seed(n):
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            ag.messages.append({"role": role, "content": f"filler {i}"})

    # Below the lowered gate: nothing is trimmed.
    _seed(20)
    before = len(ag.messages)
    ag.chat("hello")
    assert len(ag.messages) >= before + 1   # grew; nothing sliced

    # Past the lowered gate: trimmed down to ~cap (pre-fix: nothing would
    # happen until 121 messages).
    _seed(30)
    assert len(ag.messages) > 30
    ag.chat("hello again")
    assert len(ag.messages) <= 10 + 4, (
        f"{mode}: expected trim to ~cap, got {len(ag.messages)}")
    if mode != "simulate":               # sim's trim has no marker on main
        assert any("omitted for context management"
                   in str(m.get("content", "")) for m in ag.messages), mode
