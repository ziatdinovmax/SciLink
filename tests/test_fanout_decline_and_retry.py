"""#557 — the AUTOPILOT fan-out gate: a user decline is sticky for the turn
and directive (no verdict, no re-proposal), a dataset set that already ran
is refused with a fuse-and-retry directive, a mixed-outcome result names the
failed branches, and resume_fanout(retry_failed=True) re-runs only them.

  python -m pytest tests/test_fanout_decline_and_retry.py -v
"""
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import pytest

import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import (
    MetaOrchestratorAgent, MetaMode)


@pytest.fixture()
def paths(tmp_path):
    ps = [str(tmp_path / f"{n}.npy") for n in "AB"]
    for p in ps:
        np.save(p, np.zeros((4, 4)))
    return ps


def _branches(paths):
    return [{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)}
            for p in paths]


def _verdict(paths):
    def fake_llm(orch, prompt, extra_parts=None):
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "T", "join_type": "shared_parameter_axis",
                "fanout_set": list(paths), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    return fake_llm


def _child_factory(fail_paths=(), calls=None):
    def fake_child(orch, base_dir, restore=False):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                if calls is not None:
                    calls.append({"task": task, "restore": restore,
                                  "base_dir": str(base_dir)})
                for fp in fail_paths:
                    if fp in task:
                        raise RuntimeError("synthetic branch failure")
                return {"status": "success", "summary": "ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    return fake_child


def _autopilot(tmp_path):
    return MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path / "s"),
                                 meta_mode=MetaMode.AUTOPILOT)


def _autonomous(tmp_path):
    return MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path / "s"),
                                 meta_mode=MetaMode.AUTONOMOUS)


# ------------------------------------------------------- part 1: decline --

def test_user_decline_is_directive_and_sticky_for_the_turn(tmp_path, paths, monkeypatch):
    meta = _autopilot(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    prompts = []
    monkeypatch.setattr(fo, "request_human_feedback",
                        lambda *a, **k: prompts.append(k.get("origin")) or "n")
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "declined_by_user"
    assert "verdict" not in out                       # no re-motivation
    assert "Do NOT call delegate_to_analyses again" in out["message"]
    assert len(prompts) == 1

    # Same set again in the same turn: refused WITHOUT asking.
    out2 = json.loads(meta._run_fanout(_branches(paths)))
    assert out2["status"] == "declined_by_user"
    assert "already declined" in out2["message"]
    assert len(prompts) == 1
    # Re-labelled / re-worded tasks over the same datasets: still refused.
    relabel = [{"data_path": p, "task": "Different wording", "label": "x" + str(i)}
               for i, p in enumerate(paths)]
    assert json.loads(meta._run_fanout(relabel))["status"] == "declined_by_user"
    assert len(prompts) == 1


def test_decline_resets_at_the_next_user_turn(tmp_path, paths, monkeypatch):
    meta = _autopilot(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    prompts = []
    monkeypatch.setattr(fo, "request_human_feedback",
                        lambda *a, **k: prompts.append(1) or "n")
    meta._run_fanout(_branches(paths))
    assert len(prompts) == 1
    # What chat() does at the start of every user turn.
    meta._fanout_declined_sets = []
    meta._run_fanout(_branches(paths))
    assert len(prompts) == 2                          # asked again


def test_no_input_channel_counts_as_user_decline(tmp_path, paths, monkeypatch):
    meta = _autopilot(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))

    def eof(*a, **k):
        raise EOFError
    monkeypatch.setattr(fo, "request_human_feedback", eof)
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "declined_by_user"
    assert fo._user_already_declined(meta, paths)


def test_user_confirm_runs_and_records_nothing(tmp_path, paths, monkeypatch):
    meta = _autopilot(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "request_human_feedback", lambda *a, **k: "y")
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory())
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "success" and out["branches_with_output"] == 2
    assert not getattr(meta, "_fanout_declined_sets", [])


def test_autonomous_verdict_decline_keeps_verdict_payload(tmp_path, paths, monkeypatch):
    """A machine decline (verdict / caps) is unchanged: it still carries
    the verdict so the model can act on redundant/unrelated groupings."""
    meta = _autonomous(tmp_path)

    def weak(orch, prompt, extra_parts=None):
        v = _verdict(paths)(orch, prompt)
        v["confidence"] = 0.2
        return v
    monkeypatch.setattr(fo, "_llm_json", weak)
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "declined" and "verdict" in out
    assert not getattr(meta, "_fanout_declined_sets", [])


# ------------------------------------------------ part 2: mixed outcomes --

def test_mixed_outcome_names_failed_branches_and_targets_retry(tmp_path, paths, monkeypatch):
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(fail_paths=[paths[1]]))
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "success"
    assert out["branches_with_output"] == 1
    assert [f["label"] for f in out["failed_branches"]] == ["B.npy"]
    assert "synthetic branch failure" in out["failed_branches"][0]["error"]
    assert "resume_fanout(retry_failed=true)" in out["next_step"]
    assert "never the whole fan-out" in out["warning"]


def test_already_ran_set_is_refused_with_fuse_and_retry_directive(tmp_path, paths, monkeypatch):
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(fail_paths=[paths[1]]))
    first = json.loads(meta._run_fanout(_branches(paths)))
    gates = []
    monkeypatch.setattr(fo, "_llm_json",
                        lambda *a, **k: gates.append(1) or _verdict(paths)(*a, **k))
    again = json.loads(meta._run_fanout(_branches(paths)))
    assert again["status"] == "declined" and again["reason"] == "already_ran"
    assert again["parallel_group"] == first["parallel_group"]
    assert [p["delegation_index"] for p in again["productive"]] == [1]
    assert [f["label"] for f in again["failed"]] == ["B.npy"]
    assert "resume_fanout(retry_failed=true)" in again["message"]
    assert "force_rerun=true" in again["message"]
    assert gates == []                                # refused before the gate
    # Explicit override re-runs (the gate verdict is served from its cache).
    forced = json.loads(meta._run_fanout(_branches(paths), force_rerun=True))
    assert forced["status"] == "success" and forced["parallel_group"] != first["parallel_group"]


def test_already_ran_guard_applies_to_the_gate_pruned_set(tmp_path, paths, monkeypatch):
    """Request A+B+C; the gate prunes C; A+B already ran -> refused."""
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory())
    meta._run_fanout(_branches(paths))
    extra = str(tmp_path / "C.npy")
    np.save(extra, np.zeros((4, 4)))

    def prune_c(orch, prompt, extra_parts=None):
        v = _verdict(paths)(orch, prompt)
        v["unrelated"] = [extra]
        return v
    monkeypatch.setattr(fo, "_llm_json", prune_c)
    out = json.loads(meta._run_fanout(_branches(paths + [extra])))
    assert out["status"] == "declined" and out["reason"] == "already_ran"


def test_already_ran_guard_needs_a_productive_branch(tmp_path, paths, monkeypatch):
    """Every branch failed -> a full re-run is the right recovery; not refused."""
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(fail_paths=list(paths)))
    meta._run_fanout(_branches(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory())
    out = json.loads(meta._run_fanout(_branches(paths)))
    assert out["status"] == "success" and out["branches_with_output"] == 2


def test_different_set_is_not_refused(tmp_path, paths, monkeypatch):
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory())
    meta._run_fanout(_branches(paths))
    extra = str(tmp_path / "C.npy")
    np.save(extra, np.zeros((4, 4)))
    ps3 = paths + [extra]
    monkeypatch.setattr(fo, "_llm_json", _verdict(ps3))
    out = json.loads(meta._run_fanout(_branches(ps3)))
    assert out["status"] == "success" and out["branches_run"] == 3


def test_retry_failed_reruns_only_the_failed_branch_of_latest_group(tmp_path, paths, monkeypatch):
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(fail_paths=[paths[1]]))
    meta._run_fanout(_branches(paths))
    failed = [e for e in meta._delegation_ledger if e.get("status") == "error"]
    assert len(failed) == 1 and failed[0]["label"] == "B.npy"

    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(calls=calls))
    out = json.loads(meta._resume_fanout(retry_failed=True))
    assert out["status"] == "success"
    assert out["branches_retried"] == 1 and out["branches_resumed"] == 0
    assert len(calls) == 1
    assert "RETRY NOTE" in calls[0]["task"] and "synthetic branch failure" in calls[0]["task"]
    assert calls[0]["restore"] is False            # no checkpoint was left behind
    e = failed[0]
    assert e["status"] == "success" and e["retries"] == 1
    assert e["key_findings"] == ["finding"] and not e.get("error")
    # Now the set is fully productive: nothing left to retry.
    out2 = json.loads(meta._resume_fanout(retry_failed=True))
    assert out2["status"] == "no_op" and "no failed branches" in out2["message"]


def test_retry_failed_restores_when_the_failed_attempt_left_a_checkpoint(tmp_path, paths, monkeypatch):
    meta = _autonomous(tmp_path)
    monkeypatch.setattr(fo, "_llm_json", _verdict(paths))
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(fail_paths=[paths[1]]))
    meta._run_fanout(_branches(paths))
    e = next(x for x in meta._delegation_ledger if x.get("status") == "error")
    bdir = meta.fanout_dir / f"{e['index']:02d}_{fo._slug(e['label'])}"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "checkpoint.json").write_text("{}")
    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(calls=calls))
    meta._resume_fanout(retry_failed=True)
    assert calls and calls[0]["restore"] is True


def test_retry_failed_leaves_timed_out_and_older_groups_alone(tmp_path, monkeypatch):
    meta = _autonomous(tmp_path)

    def entry(index, label, group, status, **kw):
        d = {"index": index, "label": label, "status": status, "task": "t",
             "mode": "analysis", "fanout": True, "parallel_group": group,
             "key_findings": [], "files_produced": [], "summary": "",
             "data_path": f"/d/{label}"}
        d.update(kw)
        return d
    meta._delegation_ledger = [
        entry(1, "old_fail", "fanout_1", "error", error="old"),
        entry(2, "old_ok", "fanout_1", "success", key_findings=["f"]),
        entry(3, "new_ok", "fanout_3", "success", key_findings=["f"]),
        entry(4, "new_fail", "fanout_3", "error", error="boom"),
        entry(5, "new_timeout", "fanout_3", "error", timed_out=True),
    ]
    calls = []
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        _child_factory(calls=calls))
    out = json.loads(meta._resume_fanout(retry_failed=True))
    assert out["branches_retried"] == 1
    assert [r["label"] for r in out["results"]] == ["new_fail"]
    assert meta._delegation_ledger[0]["status"] == "error"     # older group untouched
    assert meta._delegation_ledger[4]["status"] == "error"     # timed_out untouched


def test_plain_resume_is_unchanged_without_retry_flag(tmp_path, monkeypatch):
    meta = _autonomous(tmp_path)
    meta._delegation_ledger = [
        {"index": 1, "label": "f", "status": "error", "task": "t", "mode": "analysis",
         "fanout": True, "parallel_group": "fanout_1", "key_findings": [],
         "files_produced": [], "summary": "", "error": "boom"}]
    out = json.loads(meta._resume_fanout())
    assert out["status"] == "no_op" and "retry_failed=true" in out["message"]


def test_tools_expose_the_new_flags(tmp_path):
    meta = _autonomous(tmp_path)
    by_name = {s["function"]["name"]: s["function"]
               for s in meta.tools.openai_schemas}
    assert "force_rerun" in by_name["delegate_to_analyses"]["parameters"]["properties"]
    assert "retry_failed" in by_name["resume_fanout"]["parameters"]["properties"]
    assert "retry_failed=true" in by_name["delegate_to_analyses"]["description"]
