"""Regression test for #364: a programmatic single-turn meta session must
persist its checkpoint on turn completion — interval-based checkpointing never
fires for a one-chat()-per-process session, and before this fix such sessions
restored with an empty delegation ledger.

Offline — the LLM turn is stubbed.

  conda run -n scilink python -m pytest tests/test_meta_turn_checkpoint.py -v
"""
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode


def _entry(index):
    return {
        "index": index,
        "mode": "analysis",
        "task": f"task {index}",
        "status": "success",
        "summary": "ok",
        "key_findings": ["finding"],
        "files_produced": [],
    }


def test_single_turn_session_persists_ledger(tmp_path):
    base = str(tmp_path)
    ag = MetaOrchestratorAgent(
        base_dir=base,
        api_key="sk-dummy",
        model_name="claude-opus-4-6",
        meta_mode=MetaMode.AUTONOMOUS,
    )

    def fake_turn(user_input):
        # Simulate a turn whose tool calls completed two delegations.
        ag._delegation_ledger.append(_entry(1))
        ag._delegation_ledger.append(_entry(2))
        return "done"

    ag._handle_litellm_chat = fake_turn
    assert ag.chat("fan out over my datasets") == "done"

    # One successful turn, no interval crossed, no error — the checkpoint
    # must exist anyway, with the ledger and the turn's message count.
    assert ag.checkpoint_path.exists()
    data = json.loads(ag.checkpoint_path.read_text())
    assert [e["index"] for e in data["delegation_ledger"]] == [1, 2]
    assert data["message_count"] == ag.message_count
    assert ag.last_checkpoint_message_count == ag.message_count

    # "Second process": restore the session and see the first process's
    # delegations without re-running them.
    ag2 = MetaOrchestratorAgent(
        base_dir=base,
        api_key="sk-dummy",
        model_name="claude-opus-4-6",
        restore_checkpoint=True,
    )
    assert [e["index"] for e in ag2._delegation_ledger] == [1, 2]


def test_turn_error_still_checkpoints(tmp_path):
    ag = MetaOrchestratorAgent(
        base_dir=str(tmp_path),
        api_key="sk-dummy",
        model_name="claude-opus-4-6",
        meta_mode=MetaMode.AUTONOMOUS,
    )

    def broken_turn(user_input):
        ag._delegation_ledger.append(_entry(1))
        raise RuntimeError("synthetic turn failure")

    ag._handle_litellm_chat = broken_turn
    out = ag.chat("boom")
    assert "Error" in out

    data = json.loads(ag.checkpoint_path.read_text())
    assert [e["index"] for e in data["delegation_ledger"]] == [1]
