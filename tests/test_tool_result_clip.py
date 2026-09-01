"""Tests for the oversized-tool-result guards.

Found live: a resumed fan-out branch pointed ``load_metadata`` at a large
results JSON; the tool echoed all 7.7MB into its tool result, putting ~5.3M
tokens into every later LLM call of the session — and per-tool
checkpointing makes such a result durable across restores. Two layers now
bound it: ``clip_tool_result`` at the message-build chokepoints (all four
orchestrators), and ``load_metadata``'s conversational echo capped at the
source (the full metadata is still stored for the analysis).

  python -m pytest tests/test_tool_result_clip.py -v
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest

from scilink.utils.tool_media import (TOOL_RESULT_MAX_CHARS, build_tool_message,
                                      clip_tool_result)


def test_normal_results_pass_through_byte_identical():
    s = "x" * (TOOL_RESULT_MAX_CHARS)
    assert clip_tool_result(s) is s
    msg = build_tool_message("c1", '{"status": "success"}', allow_image=False)
    assert msg["content"] == '{"status": "success"}'


def test_oversized_result_is_clipped_with_recovery_marker():
    s = "y" * (TOOL_RESULT_MAX_CHARS + 500)
    out = clip_tool_result(s)
    assert len(out) < len(s)
    assert "TOOL RESULT TRUNCATED" in out
    assert str(len(s)) in out


def test_build_tool_message_clips_both_paths():
    big = json.dumps({"status": "ok", "payload": "z" * (TOOL_RESULT_MAX_CHARS + 100)})
    plain = build_tool_message("c1", big, allow_image=False)
    assert "TOOL RESULT TRUNCATED" in plain["content"]

    with_img = json.dumps({"status": "ok",
                           "payload": "z" * (TOOL_RESULT_MAX_CHARS + 100),
                           "image_base64": "aGVsbG8="})
    multi = build_tool_message("c1", with_img, allow_image=True)
    # The image survives as its own part; the text part is clipped.
    assert any(p.get("type") == "image_url" for p in multi["content"])
    text = next(p["text"] for p in multi["content"] if p.get("type") == "text")
    assert "TOOL RESULT TRUNCATED" in text


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate", "meta"])
def test_giant_tool_result_never_reaches_history_unclipped(mode, tmp_path,
                                                           monkeypatch):
    """End-to-end through each orchestrator's litellm loop: a runaway tool
    result is bounded both in the live messages and in the per-tool
    checkpointed history."""
    if mode == "analysis":
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent)
        orch = AnalysisOrchestratorAgent(api_key="sk-dummy",
                                         base_dir=str(tmp_path / "an"))
    elif mode == "planning":
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent)
        orch = PlanningOrchestratorAgent(api_key="sk-dummy",
                                         data_dir=str(tmp_path),
                                         base_dir=str(tmp_path / "pl"))
    elif mode == "simulate":
        from scilink.agents.sim_agents.simulation_orchestrator import (
            SimulationOrchestratorAgent)
        orch = SimulationOrchestratorAgent(api_key="sk-dummy",
                                           base_dir=str(tmp_path / "si"))
    else:
        from scilink.agents.meta_agent.meta_orchestrator import (
            MetaOrchestratorAgent)
        orch = MetaOrchestratorAgent(api_key="sk-dummy",
                                     base_dir=str(tmp_path / "meta"))
    orch.TOOL_CHECKPOINT_MIN_INTERVAL_S = 0.0
    orch.use_openai = False
    monkeypatch.setattr(orch.tools, "execute_tool",
                        lambda name, **kw: "G" * (TOOL_RESULT_MAX_CHARS + 999))

    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="some_tool", arguments="{}"))
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=[tc], content=None),
            finish_reason="tool_calls")]),
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=None, content="done"),
            finish_reason="stop")]),
    ])
    import scilink.wrappers.litellm_wrapper as lw
    monkeypatch.setattr(lw, "litellm_completion",
                        lambda *a, **k: next(responses))

    orch.chat("go")

    tool_msgs = [m for m in orch.messages if m.get("role") == "tool"]
    assert tool_msgs
    assert all(len(json.dumps(m)) < TOOL_RESULT_MAX_CHARS + 2000
               for m in tool_msgs), f"{mode}: giant result reached history"
    hist = json.loads(Path(orch.history_path).read_text(encoding="utf-8"))
    assert all(len(json.dumps(m)) < TOOL_RESULT_MAX_CHARS + 2000
               for m in hist), f"{mode}: giant result persisted to disk"


def test_load_metadata_echo_is_bounded_but_storage_is_full(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    orch = AnalysisOrchestratorAgent(api_key="sk-dummy",
                                     base_dir=str(tmp_path / "an"))
    big = {"experiment_type": "Diffraction",
           "experiment": {"technique": "XRD"},
           "sample": {"material": "test"},
           "bootstrap_samples": list(range(30000))}
    p = tmp_path / "big_metadata.json"
    p.write_text(json.dumps(big))

    result = json.loads(orch.tools.execute_tool("load_metadata",
                                                json_path=str(p)))
    assert result["status"] in ("success", "warning")
    assert "_truncated_echo" in result["metadata"], (
        "giant metadata echoed verbatim into the tool result")
    assert len(json.dumps(result)) < 25_000
    # The analysis-side storage keeps the FULL object.
    assert orch.current_metadata == big


def test_load_metadata_small_echo_unchanged(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    orch = AnalysisOrchestratorAgent(api_key="sk-dummy",
                                     base_dir=str(tmp_path / "an"))
    small = {"experiment_type": "Diffraction",
             "experiment": {"technique": "XRD"},
             "sample": {"material": "test"}}
    p = tmp_path / "small_metadata.json"
    p.write_text(json.dumps(small))
    result = json.loads(orch.tools.execute_tool("load_metadata",
                                                json_path=str(p)))
    assert result["metadata"] == small


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
