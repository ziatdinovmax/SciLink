"""Offline tests for #519: harmonize accepts a partial-but-approved donor.

The donor-script search was gated on the donor delegation's top-level
``status == "success"``. A donor can exhaust its tool-iteration budget (or
die after approval) and honestly report ``error`` while approved
(``task_success``) dynamic-analysis records with a replayable script sit on
disk — and the whole harmonized run then fell back to independent branches.
Now the record search runs regardless of top-level status (the records are
the authority); only a donor with NO approved record triggers the loud
fallback, which these tests also pin.

  conda run -n scilink python -m pytest tests/test_harmonize_partial_donor.py -v
"""
import json
import os
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import pytest

import scilink.agents.meta_agent.fanout as fo
# Pre-warm the worker-side lazy import (see test_wallclock_budget.py).
import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
from scilink.agents.meta_agent.meta_orchestrator import (
    MetaOrchestratorAgent, MetaMode)

CAPTURED_TASKS = []


def _fake_llm(paths):
    def fake(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [],
                    "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "T", "join_type": "shared_parameter_axis",
                "fanout_set": list(paths), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    return fake


def _child_factory(donor_result, donor_writes_records):
    """Donor = first branch to run; followers always succeed."""
    seen = {"n": 0}

    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                CAPTURED_TASKS.append(task)
                seen["n"] += 1
                if seen["n"] == 1:   # the donor (runs alone, first)
                    if donor_writes_records:
                        rdir = os.path.join(base_dir, "results", "an_fake")
                        os.makedirs(rdir, exist_ok=True)
                        with open(os.path.join(
                                rdir, "dynamic_analysis_records.json"),
                                "w") as fh:
                            json.dump([{"target": "t", "task_success": True,
                                        "required_outputs": [],
                                        "script": "print(1)"}], fh)
                    return dict(donor_result)
                return {"status": "success", "summary": "ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    return fake_child


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    paths = [str(tmp_path / f"{n}.npy") for n in "ABC"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    ag = MetaOrchestratorAgent(base_dir=str(tmp_path), api_key="sk-dummy",
                               meta_mode=MetaMode.AUTONOMOUS)
    monkeypatch.setattr(fo, "_llm_json", _fake_llm(paths))
    CAPTURED_TASKS.clear()
    return ag, paths


def _branches(paths):
    return [{"data_path": p, "task": f"Analyze {p}",
             "label": os.path.basename(p)} for p in paths]


def test_partial_donor_with_approved_records_still_harmonizes(rig, monkeypatch):
    ag, paths = rig
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory(
        {"status": "error",
         "error": "⚠️ Reached maximum tool iterations (20).",
         "summary": "", "key_findings": [], "files_produced": []},
        donor_writes_records=True))

    out = json.loads(ag._run_fanout(_branches(paths), harmonize=True))
    harm = out.get("harmonized") or {}
    assert harm.get("status") == "harmonized", (
        "partial donor with an approved script fell back to independent "
        "branches — the #519 bug is back")
    assert harm.get("donor_partial") is True
    assert harm.get("donor_status") == "error"

    # Followers really replay the approved script.
    follower_tasks = CAPTURED_TASKS[1:]
    assert len(follower_tasks) == 2
    assert all("reuse_locked_script=true" in t and harm["reuse_dir"] in t
               for t in follower_tasks)

    # The donor's honest ledger verdict is NOT laundered by harmonization.
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    assert fan[0]["status"] == "error"
    stamped = [e for e in fan if e.get("harmonized_with")]
    assert len(stamped) == 2


def test_donor_with_no_approved_records_still_falls_back(rig, monkeypatch):
    ag, paths = rig
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory(
        {"status": "error", "error": "everything failed", "summary": "",
         "key_findings": [], "files_produced": []},
        donor_writes_records=False))

    out = json.loads(ag._run_fanout(_branches(paths), harmonize=True))
    harm = out.get("harmonized") or {}
    assert harm.get("status") == "fallback"
    assert harm.get("reason") == "no approved donor script"
    follower_tasks = CAPTURED_TASKS[1:]
    assert follower_tasks and all(
        "reuse_locked_script" not in t for t in follower_tasks)


def test_successful_donor_not_marked_partial(rig, monkeypatch):
    ag, paths = rig
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory(
        {"status": "success", "summary": "ok", "key_findings": ["f"],
         "files_produced": []},
        donor_writes_records=True))

    out = json.loads(ag._run_fanout(_branches(paths), harmonize=True))
    harm = out.get("harmonized") or {}
    assert harm.get("status") == "harmonized"
    assert "donor_partial" not in harm
    assert "donor_status" not in harm


def test_donor_task_carries_single_call_nudge(rig, monkeypatch):
    ag, paths = rig
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child", _child_factory(
        {"status": "success", "summary": "ok", "key_findings": ["f"],
         "files_produced": []},
        donor_writes_records=True))

    json.loads(ag._run_fanout(_branches(paths), harmonize=True))
    assert "PIPELINE-DONOR NOTE" in CAPTURED_TASKS[0]
    assert all("PIPELINE-DONOR NOTE" not in t for t in CAPTURED_TASKS[1:])
    # The ledger records the actual (nudged) donor task.
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    assert "PIPELINE-DONOR NOTE" in fan[0]["task"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
