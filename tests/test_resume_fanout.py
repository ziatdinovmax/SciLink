"""Offline tests for fan-out branch resume (Phase C).

A coordinator death (or user stop) leaves a fan-out's provisional ledger
entries 'running' — swept to 'interrupted' at the next turn start.
``resume_fanout`` re-creates each such branch IN its original
``fanout/<NN>_<slug>/`` session dir with ``restore_checkpoint=True`` (so it
holds its per-tool-checkpointed partial progress) and re-runs its RECORDED
task verbatim; completed branches and budget-degraded branches are
untouched.

  python -m pytest tests/test_resume_fanout.py -v
"""
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest

import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent


def _entry(index, label, status, task, **kw):
    e = {"index": index, "label": label, "status": status, "task": task,
         "mode": "analysis", "fanout": True, "parallel_group": "fanout_1",
         "key_findings": [], "files_produced": [], "summary": ""}
    e.update(kw)
    return e


@pytest.fixture()
def meta(tmp_path):
    return MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path))


def _capture_factory(calls):
    def fake_child(orch, base_dir, restore=False):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                calls.append({"base_dir": str(base_dir), "restore": restore,
                              "task": task})
                return {"status": "success", "summary": "resumed ok",
                        "key_findings": ["resumed finding"],
                        "files_produced": []}
        return C()
    return fake_child


def test_resume_reruns_interrupted_branches_in_place(meta, monkeypatch):
    recorded_task = ("Analyze X\n\nNOTE — this is ONE branch of a JOINT "
                     "multi-dataset analysis: ...\n\nPRIMARY dataset for "
                     "THIS analysis: /data/a.npy — pass it VERBATIM ...")
    done = _entry(1, "done branch", "success", "Analyze done",
                  key_findings=["orig finding"], completed_at="t")
    i2 = _entry(2, "vib series", "interrupted", recorded_task)
    i3 = _entry(3, "xrd series", "running", "Analyze Y (recorded)")
    meta._delegation_ledger = [done, i2, i3]

    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _capture_factory(calls))
    out = json.loads(meta._resume_fanout())

    assert out["status"] == "success"
    assert out["branches_resumed"] == 2
    assert {r["delegation_index"] for r in out["results"]} == {2, 3}

    # Both branches re-ran with restore=True IN their original dirs.
    assert len(calls) == 2
    assert all(c["restore"] is True for c in calls)
    dirs = {os.path.basename(c["base_dir"]) for c in calls}
    assert dirs == {"02_vib_series", "03_xrd_series"}

    # The RECORDED task is used verbatim (not re-meshed: the mesh block
    # appears exactly once) plus the resume note.
    t2 = next(c["task"] for c in calls if "02_" in c["base_dir"])
    assert t2.startswith(recorded_task)
    assert t2.count("PRIMARY dataset for THIS analysis") == 1
    assert "RESUME NOTE" in t2

    # Ledger updated in place; the completed branch untouched.
    assert i2["status"] == "success" and i2["key_findings"] == ["resumed finding"]
    assert i2["resumed_from_interruption"] is True
    assert i3["status"] == "success"
    assert done["key_findings"] == ["orig finding"]


def test_resume_is_a_noop_without_interrupted_branches(meta, monkeypatch):
    meta._delegation_ledger = [
        _entry(1, "a", "success", "t", completed_at="t"),
        _entry(2, "b", "error", "t", timed_out=True, completed_at="t"),
    ]
    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _capture_factory(calls))
    out = json.loads(meta._resume_fanout())
    assert out["status"] == "no_op"
    assert not calls


def test_budget_degraded_branches_are_not_resumed(meta, monkeypatch):
    """A timed_out verdict is final (#358) — resume must not reopen it."""
    meta._delegation_ledger = [
        _entry(1, "stuck", "interrupted", "t", timed_out=True),
        _entry(2, "lost", "interrupted", "recorded task"),
    ]
    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _capture_factory(calls))
    out = json.loads(meta._resume_fanout())
    assert out["branches_resumed"] == 1
    assert len(calls) == 1 and "02_lost" in calls[0]["base_dir"]
    assert meta._delegation_ledger[0]["status"] == "interrupted"


def test_failed_resume_keeps_honest_error_status(meta, monkeypatch):
    meta._delegation_ledger = [_entry(1, "bad", "interrupted", "t")]

    def failing_child(orch, base_dir, restore=False):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                return {"status": "error", "error": "still broken",
                        "summary": "", "key_findings": [],
                        "files_produced": []}
        return C()
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", failing_child)
    out = json.loads(meta._resume_fanout())
    assert out["status"] == "error"
    assert meta._delegation_ledger[0]["status"] == "error"


def test_restore_notice_reports_interrupted_fanout(tmp_path, capsys):
    m1 = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path))
    m1._delegation_ledger = [
        _entry(1, "ok", "success", "t", completed_at="t"),
        _entry(2, "lost", "running", "t"),
    ]
    m1.save_checkpoint()

    capsys.readouterr()
    m2 = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path),
                               restore_checkpoint=True)
    out = capsys.readouterr().out
    assert "interrupted mid-run" in out and "resume_fanout" in out
    assert len(m2._delegation_ledger) == 2


def test_resume_tool_is_registered(meta):
    schemas = json.dumps(meta.tools.openai_schemas)
    assert "resume_fanout" in schemas


def test_provisional_entries_persisted_before_branches_run(tmp_path,
                                                          monkeypatch):
    """A crash BEFORE the first branch completes must still leave the
    'running' entries on disk — otherwise the fan-out is invisible to
    resume_fanout on restore. Asserted from inside a running branch: the
    meta checkpoint must already hold this branch's running entry."""
    import numpy as np
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401

    paths = [str(tmp_path / f"{n}.npy") for n in "AB"]
    for p in paths:
        np.save(p, np.zeros((4, 4)))
    ag = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path),
                               meta_mode=MetaMode.AUTONOMOUS)

    seen = []

    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                led = json.loads(
                    ag.checkpoint_path.read_text()).get("delegation_ledger", [])
                seen.append([e.get("status") for e in led
                             if e.get("fanout")])
                return {"status": "success", "summary": "ok",
                        "key_findings": ["f"], "files_produced": []}
        return C()

    def fake_llm(orch, prompt, extra_parts=None):
        return {"verdict": "complementary", "confidence": 0.9,
                "rationale": "r", "join_axis": "T",
                "join_type": "shared_parameter_axis",
                "fanout_set": list(paths), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}

    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", fake_child)
    monkeypatch.setattr(fo, "_llm_json", fake_llm)
    json.loads(ag._run_fanout(
        [{"data_path": p, "task": f"Analyze {p}",
          "label": os.path.basename(p)} for p in paths]))

    assert seen, "no branch observed the checkpoint"
    # Every branch found the full fan-out already on disk when it started.
    assert all(len(statuses) == 2 for statuses in seen), seen


def test_harmonize_stamps_persisted_before_followers_run(tmp_path,
                                                         monkeypatch):
    """A crash DURING the harmonized replay phase must leave followers whose
    on-disk entries still carry the verbatim-replay task + harmonized_with
    stamp — otherwise resume_fanout would re-run them from the
    pre-harmonize task, silently unharmonized. Asserted from inside a
    running follower."""
    import numpy as np
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401

    paths = [str(tmp_path / f"{n}.npy") for n in "AB"]
    for p in paths:
        np.save(p, np.zeros((4, 4)))
    ag = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path),
                               meta_mode=MetaMode.AUTONOMOUS)

    seen_followers = []
    n_run = {"n": 0}

    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                n_run["n"] += 1
                if n_run["n"] == 1:      # the donor: write approved records
                    rdir = os.path.join(str(base_dir), "results", "an_fake")
                    os.makedirs(rdir, exist_ok=True)
                    with open(os.path.join(
                            rdir, "dynamic_analysis_records.json"), "w") as fh:
                        json.dump([{"target": "t", "task_success": True,
                                    "required_outputs": [],
                                    "script": "print(1)"}], fh)
                else:                    # a follower: check the disk state
                    led = json.loads(ag.checkpoint_path.read_text()).get(
                        "delegation_ledger", [])
                    mine = [e for e in led if e.get("fanout")
                            and e.get("status") == "running"]
                    seen_followers.append(
                        all("reuse_locked_script=true" in (e.get("task") or "")
                            and e.get("harmonized_with") for e in mine))
                return {"status": "success", "summary": "ok",
                        "key_findings": ["f"], "files_produced": []}
        return C()

    def fake_llm(orch, prompt, extra_parts=None):
        return {"verdict": "complementary", "confidence": 0.9,
                "rationale": "r", "join_axis": "T",
                "join_type": "shared_parameter_axis",
                "fanout_set": list(paths), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}

    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", fake_child)
    monkeypatch.setattr(fo, "_llm_json", fake_llm)
    json.loads(ag._run_fanout(
        [{"data_path": p, "task": f"Analyze {p}",
          "label": os.path.basename(p)} for p in paths], harmonize=True))

    assert seen_followers and all(seen_followers), (
        "follower ran while its on-disk entry lacked the replay task/stamp")


def test_single_delegation_provisional_entry_persisted(tmp_path, monkeypatch):
    """A crash mid-delegation must leave the provisional 'running' entry on
    disk (mirrors the fan-out launch checkpoint) so the restored ledger
    shows the delegation as interrupted, not as never having existed.
    Asserted from inside the running child."""
    meta = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path))
    seen = []

    class FakeChild:
        def run_task(self, task, context=None, autonomy=None):
            led = json.loads(meta.checkpoint_path.read_text()).get(
                "delegation_ledger", [])
            seen.append([(e.get("mode"), e.get("status")) for e in led])
            return {"status": "success", "summary": "ok",
                    "key_findings": ["f"], "files_produced": []}

    monkeypatch.setattr(meta, "_get_analysis_child", lambda: FakeChild())
    meta._delegate("analysis", "analyze the thing", label="one-off")

    assert seen and seen[0] == [("analysis", "running")], (
        "child ran while its provisional entry was not yet on disk")
    # After completion the persisted entry is finalized as usual.
    led = json.loads(meta.checkpoint_path.read_text())["delegation_ledger"]
    assert led[0]["status"] == "success"



if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
