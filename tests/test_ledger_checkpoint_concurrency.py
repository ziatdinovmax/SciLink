"""A checkpoint is not skipped because a branch thread wrote to the ledger.

Fan-out branch threads add keys to their ledger entries (_started_at,
timed_out, late_result, ...) while the coordinator checkpoints. Serializing
the live dicts could raise "dictionary changed size during iteration", and
that checkpoint was skipped with only a warning.
"""

import json
import threading
from types import SimpleNamespace

from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent


def _meta(tmp_path):
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m._checkpoint_lock = threading.Lock()
    m._fanout_lock = threading.RLock()
    m.checkpoint_path = tmp_path / "checkpoint.json"
    m.meta_mode = MetaMode.AUTONOMOUS
    m.message_count = 0
    m._children = {}
    m.knowledge_dir = None
    m._delegation_ledger = [{"index": i + 1, "mode": "analysis", "status": "running",
                             "notes": {f"k{j}": j for j in range(5000)}} for i in range(4)]
    return m


class _WritesDuringSerialization:
    """Serialized through default=str; its str() writes a new key into the
    live entry, exactly what a branch thread does mid-checkpoint, but at a
    deterministic moment."""

    def __init__(self, entry):
        self.entry = entry

    def __str__(self):
        self.entry[f"late_{len(self.entry)}"] = True
        return "value"


def test_a_write_during_serialization_does_not_skip_the_checkpoint(tmp_path):
    m = _meta(tmp_path)
    entry = m._delegation_ledger[0]
    entry["result_object"] = _WritesDuringSerialization(entry)
    assert m._auto_checkpoint(verbose=False) is True
    saved = json.loads(m.checkpoint_path.read_text())["delegation_ledger"][0]
    assert saved["result_object"] == "value"
    assert any(k.startswith("late_") for k in entry)          # the live entry did change


def test_checkpoints_survive_a_busy_branch_writer(tmp_path):
    m = _meta(tmp_path)
    stop = threading.Event()

    def branch_writer():
        n = 0
        while not stop.is_set():
            entry = m._delegation_ledger[n % 4]
            entry["notes"][f"extra_{n}"] = n          # a new key, no lock: a single-key write
            if len(entry["notes"]) > 5200:
                entry["notes"].clear()
                entry["notes"].update({f"k{j}": j for j in range(5000)})
            n += 1

    w = threading.Thread(target=branch_writer, daemon=True)
    w.start()
    try:
        results = [m._auto_checkpoint(verbose=False) for _ in range(20)]
    finally:
        stop.set()
        w.join(timeout=5)
    assert all(results), f"{results.count(False)} of 20 checkpoints skipped"


def test_a_closed_entry_is_updated_in_one_locked_step(tmp_path):
    import inspect
    src = inspect.getsource(MetaOrchestratorAgent._close_delegation)
    assert "with self._fanout_lock:" in src and src.index("with self._fanout_lock:") < src.index("entry.update(")
