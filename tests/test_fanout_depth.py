"""An analysis delegation and a fan-out branch carry their own depth
(``profile``, ``targets``, ``time_budget_s``), read by one rule
(``fanout.analysis_depth``) and passed to the child's ``run_task``; an
unknown value is refused with its reason before anything runs. The workers
are stand-ins; the meta and its ledger are real. No LLM calls.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent

DEPTH = {"profile": "extract", "targets": ["band position"], "time_budget_s": 120.0}


class Worker:
    """Records the depth each call received."""

    calls: list = []

    def __init__(self, mode, base_dir):
        self.mode, self.base_dir = mode, Path(base_dir)

    def run_task(self, task, context=None, autonomy=None, **depth):
        Worker.calls.append({"mode": self.mode, "task": task, **depth})
        deadline = time.time() + 0.1
        while time.time() < deadline:
            print(".", end="", flush=True)
            time.sleep(0.05)
        return {"status": "success", "summary": f"{task} done", "key_findings": [],
                "files_produced": [], "suggested_followups": [], "warnings": []}


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    Worker.calls = []
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    return m


def _depth_of(call):
    return {k: v for k, v in call.items() if k in fo._BRANCH_DEPTH_KEYS}


# ── one rule on both paths ───────────────────────────────────────────────

@pytest.mark.parametrize("given, applied", [
    (DEPTH, DEPTH),
    ({"profile": "quick", "targets": "band position", "time_budget_s": "90"},
     {"profile": "quick", "targets": ["band position"], "time_budget_s": 90.0}),
    ({"profile": None, "targets": [], "time_budget_s": 0}, {}),
])
def test_the_same_depth_reaches_the_child_on_every_path(meta, monkeypatch, tmp_path, given, applied):
    # a single delegation
    meta._get_analysis_child = lambda: Worker("analysis", tmp_path / "d")
    meta._delegate("analysis", "fit D", None, None, "D", depth=dict(given))
    # a fan-out branch
    paths = []
    for n in "AB":
        paths.append(str(tmp_path / f"{n}.npy"))
        np.save(paths[-1], np.zeros((4, 4)))
    monkeypatch.setattr(fo, "_llm_json", lambda orch, prompt, extra_parts=None: {
        "verdict": "complementary", "confidence": 0.9, "rationale": "r", "join_axis": "T",
        "join_type": "shared_parameter_axis", "fanout_set": list(paths), "redundant_clusters": [],
        "unrelated": [], "excluded_notes": ""})
    monkeypatch.setattr(fo, "_make_ephemeral_analysis_child",
                        lambda orch, base_dir, restore=False: Worker("analysis", base_dir))
    meta._run_fanout([{"data_path": p, "task": f"fit {Path(p).stem}", "label": Path(p).stem, **given}
                      for p in paths])
    seen = {c["task"]: _depth_of(c) for c in Worker.calls}
    assert "fit D" in seen and any(t.startswith("Analyze") or "fit A" in t for t in seen)
    assert len(seen) >= 2
    assert all(d == applied for d in seen.values()), seen


@pytest.mark.parametrize("bad, why", [
    ({"profile": "thorogh"}, "thorogh"),
    ({"targets": [3]}, "targets"),
    ({"time_budget_s": -5}, "time_budget_s"),
    ({"time_budget_s": float("nan")}, "time_budget_s"),
    ({"time_budget_s": float("inf")}, "time_budget_s"),
    ({"time_budget_s": True}, "time_budget_s"),
])
def test_bad_depth_is_refused_with_its_reason_before_anything_runs(meta, monkeypatch, tmp_path, bad, why):
    meta._get_analysis_child = lambda: Worker("analysis", tmp_path / "d")
    out = json.loads(meta._delegate("analysis", "fit D", None, None, "D", depth=bad))
    assert out["status"] == "error" and why in out["message"] and meta._delegation_ledger == []
    out = json.loads(meta._run_fanout([{"data_path": str(tmp_path), "task": f"t{n}", "label": n, **bad}
                                       for n in "AB"]))
    assert out["status"] == "error" and why in out["message"]
    assert Worker.calls == []
