"""Offline tests for the wall-clock budgets (#358).

Three layers, each degrade-never-hang:
  1. Fan-out per-branch deadline: an overdue branch is abandoned (recorded
     degraded, excluded from fusion), the fan-out completes, and a LATE
     completion is recorded for audit without clobbering the verdict.
  2. QC-engine verification loop: an opt-in cumulative budget stops
     iterating and takes the existing final-verify path; hosts that don't
     set it are byte-identical.
  3. Preprocessing custom-script retry loop: a cumulative budget stops new
     attempts and surfaces the budget in the failure message.

  conda run -n scilink python tests/test_wallclock_budget.py
"""
import json
import logging
import os
import tempfile
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
# Pre-warm the worker-side lazy import: in a fresh interpreter the first
# `analysis_orchestrator` import takes longer than this test's sub-second
# branch budget, which would mark even instant branches overdue. (Not a
# production concern — the default budget is an hour.)
import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------- fan-out
BEHAVIORS = {}


def _install_fakes(fanout_set):
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                import re
                m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                beh = BEHAVIORS.get(m.group(1) if m else "", "good")
                if beh == "slow":
                    time.sleep(2.0)
                return {"status": "success", "summary": f"{beh} ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    fo._make_ephemeral_analysis_child = fake_child

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [],
                    "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "T", "join_type": "shared_parameter_axis",
                "fanout_set": list(fanout_set), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake_llm


def test_fanout_budget():
    d = tempfile.mkdtemp()
    paths = [os.path.join(d, f"{n}.npy") for n in "ABC"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    A, B, C = paths
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    BEHAVIORS.clear()
    BEHAVIORS.update({A: "good", B: "good", C: "slow"})
    _install_fakes(paths)
    ag._complementarity_cache.clear()
    orig_poll = fo._FANOUT_POLL_S
    fo._FANOUT_POLL_S = 0.2
    try:
        t0 = time.monotonic()
        out = json.loads(ag._run_fanout(
            [{"data_path": p, "task": f"Analyze {p}",
              "label": os.path.basename(p)} for p in paths],
            branch_time_budget_s=0.5))
        dt = time.monotonic() - t0
    finally:
        fo._FANOUT_POLL_S = orig_poll
    print("1) fan-out branch deadline:")
    check("fan-out completed without waiting for the slow branch",
          out.get("status") == "success" and dt < 1.8)
    check("slow branch recorded timed out",
          out.get("branches_timed_out") == 1)
    slow = [e for e in ag._delegation_ledger
            if e.get("fanout") and e.get("timed_out")]
    check("timed-out entry degraded with the budget in the error",
          len(slow) == 1 and slow[0]["status"] == "error"
          and "wall-clock budget" in (slow[0].get("error") or ""))
    check("warning names the abandonment",
          "abandoned on the branch wall-clock budget" in out.get("warning", ""))
    check("two productive branches remain",
          out.get("branches_with_output") == 2)
    idxs = [r["delegation_index"] for r in out["results"]
            if r["produced_output"]]
    fused = json.loads(ag._fuse_delegations(idxs))
    check("fusion runs over the productive branches",
          fused.get("status") == "success"
          and slow[0]["index"] not in fused.get("fused_from", []))
    # Late completion: the abandoned worker finishes at ~2 s.
    time.sleep(2.2)
    check("late completion recorded WITHOUT clobbering the verdict",
          slow[0]["status"] == "error"
          and (slow[0].get("late_result") or {}).get("status") == "success")

    # Budget disabled -> the slow branch completes normally.
    ag2 = MetaOrchestratorAgent(base_dir=tempfile.mkdtemp(), api_key="sk-dummy",
                                model_name="claude-opus-4-6",
                                meta_mode=MetaMode.AUTONOMOUS)
    _install_fakes(paths); ag2._complementarity_cache.clear()
    out2 = json.loads(ag2._run_fanout(
        [{"data_path": p, "task": f"Analyze {p}",
          "label": os.path.basename(p)} for p in paths],
        branch_time_budget_s=0))
    check("budget <= 0 disables the deadline (slow branch completes)",
          out2.get("branches_with_output") == 3
          and "branches_timed_out" not in out2)


# ------------------------------------------------------------- QC engine
class _StubHost:
    _CONSTRAINT_ANNEALING_SCHEDULE = [0, 1]
    max_verification_iterations = 5
    logger = logging.getLogger("stub")

    def __init__(self, budget=None, verify_sleep=0.15):
        if budget is not None:
            self.qc_time_budget_s = budget
        self.verify_sleep = verify_sleep
        self.iters = 0
        self.final_verify_called = False

    def qc_setup(self, ctx): pass
    def qc_try_reuse(self, ctx): return None
    def qc_run_initial(self, ctx): return {"success": True, "script": "s"}

    def qc_record_initial(self, ctx, result):
        ctx.best_result = result
        ctx.current_result = result

    def qc_verification_bypass(self, ctx): return False
    def qc_log_skip_verification(self, ctx): pass
    def qc_loop_setup(self, ctx): pass

    def qc_verify(self, ctx):
        self.iters += 1
        time.sleep(self.verify_sleep)
        return {"status": "fail"}

    def qc_assess(self, ctx, verification): pass
    def qc_check_accept(self, ctx, verification): return False
    def qc_refine(self, ctx, verification): return {"k": self.iters}
    def qc_refit(self, ctx, verification, refine_from, hot):
        return {"success": True, "script": "s"}
    def qc_after_refit(self, ctx, refit_result, verification):
        ctx.current_result = refit_result

    def qc_final_verify(self, ctx):
        self.final_verify_called = True

    def qc_post_verification(self, ctx):
        return {"record": {"iters": self.iters,
                           "final": self.final_verify_called}}

    def qc_fallback(self, ctx): return {"record": None}
    def qc_record_initial_failure(self, ctx, result): pass


def test_engine_budget():
    from scilink.agents.exp_agents._qc_engine import (
        CodegenQCEngine, QCEngineSpec, QCItemContext)
    spec = QCEngineSpec(config_key=None, refine_anchor="none",
                        refit_fail_msg="refit failed")
    print("2) QC-engine loop budget:")
    host = _StubHost(budget=0.25, verify_sleep=0.15)
    out = CodegenQCEngine(host, spec).run_item(QCItemContext(
        state={}, data=None, data_path="", item_name="i", item_idx=0))
    check("opt-in budget stops the loop early",
          host.iters < 5 and out["record"]["final"] is True)
    host2 = _StubHost(budget=None, verify_sleep=0.01)
    CodegenQCEngine(host2, spec).run_item(QCItemContext(
        state={}, data=None, data_path="", item_name="i", item_idx=0))
    check("no budget attribute -> full iteration count (unchanged hosts)",
          host2.iters == 5)


# ---------------------------------------------------------- preprocess loop
def test_preprocess_budget():
    from scilink.agents.exp_agents.preprocess import HyperspectralPreprocessingAgent

    class _Resp:
        text = "```python\nprint('x')\n```"

    class _Model:
        def generate_content(self, prompt):
            time.sleep(0.4)
            return _Resp()

    class _Exec:
        def execute_script(self, script, working_dir=None):
            return {"status": "error", "message": "synthetic failure"}

    ag = object.__new__(HyperspectralPreprocessingAgent)
    ag.logger = logging.getLogger("prep")
    ag.model = _Model()
    ag.executor = _Exec()
    ag.SCRIPT_LOOP_TIME_BUDGET_S = 0.3
    print("3) preprocessing retry-loop budget:")
    t0 = time.monotonic()
    out = HyperspectralPreprocessingAgent.\
        _generate_and_execute_custom_script_with_retry(
            ag, {"shape": (2, 2, 2)}, "do a thing", "in.npy")
    dt = time.monotonic() - t0
    check("loop stopped on the budget (no 2nd/3rd attempt)",
          out["status"] == "error" and dt < 1.0
          and "wall-clock budget exceeded" in out["message"])
    ag.SCRIPT_LOOP_TIME_BUDGET_S = 0
    out2 = HyperspectralPreprocessingAgent.\
        _generate_and_execute_custom_script_with_retry(
            ag, {"shape": (2, 2, 2)}, "do a thing", "in.npy")
    check("budget disabled -> all attempts run",
          f"after {ag.MAX_SCRIPT_ATTEMPTS} attempts" in out2["message"])


def test_queued_and_multi_timeout():
    # A branch QUEUED behind a slow one (waiting for a worker slot) must not
    # be timed out while waiting — the deadline runs from its actual start.
    d = tempfile.mkdtemp()
    paths = [os.path.join(d, f"{n}.npy") for n in "ABC"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    A, B, C = paths
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    BEHAVIORS.clear()
    BEHAVIORS.update({A: "slow", B: "good", C: "good"})
    _install_fakes(paths)
    ag._complementarity_cache.clear()
    orig_poll, orig_workers = fo._FANOUT_POLL_S, fo.FANOUT_MAX_WORKERS
    fo._FANOUT_POLL_S, fo.FANOUT_MAX_WORKERS = 0.2, 1   # force queueing
    try:
        out = json.loads(ag._run_fanout(
            [{"data_path": p, "task": f"Analyze {p}",
              "label": os.path.basename(p)} for p in paths],
            branch_time_budget_s=0.8))
    finally:
        fo._FANOUT_POLL_S, fo.FANOUT_MAX_WORKERS = orig_poll, orig_workers
    print("4) queued branches vs the deadline:")
    check("only the RUNNING slow branch timed out (queued ones spared)",
          out.get("branches_timed_out") == 1)
    check("queued branches ran after the slot freed and succeeded",
          out.get("branches_with_output") == 2)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    check("no queued branch carries a timeout verdict",
          sum(1 for e in fan if e.get("timed_out")) == 1
          and not any(e.get("timed_out") for e in fan
                      if e["label"] in ("B.npy", "C.npy")))

    # Two branches overdue in the same poll round -> both cut, run completes.
    ag2 = MetaOrchestratorAgent(base_dir=tempfile.mkdtemp(), api_key="sk-dummy",
                                model_name="claude-opus-4-6",
                                meta_mode=MetaMode.AUTONOMOUS)
    BEHAVIORS.update({A: "slow", B: "slow", C: "good"})
    _install_fakes(paths); ag2._complementarity_cache.clear()
    fo._FANOUT_POLL_S = 0.2
    try:
        t0 = time.monotonic()
        out2 = json.loads(ag2._run_fanout(
            [{"data_path": p, "task": f"Analyze {p}",
              "label": os.path.basename(p)} for p in paths],
            branch_time_budget_s=0.5))
        dt = time.monotonic() - t0
    finally:
        fo._FANOUT_POLL_S = orig_poll
    print("5) simultaneous multi-timeout:")
    check("both slow branches timed out in one run",
          out2.get("branches_timed_out") == 2)
    check("run completed promptly with the survivor",
          out2.get("branches_with_output") == 1 and dt < 1.8)
    time.sleep(2.2)   # let abandoned workers finish before interpreter exit


def main():
    test_fanout_budget()
    test_engine_budget()
    test_preprocess_budget()
    test_queued_and_multi_timeout()
    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"WALL-CLOCK BUDGET: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
