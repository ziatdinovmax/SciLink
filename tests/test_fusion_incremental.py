"""Offline tests for incremental fusion + the fusion->re-analysis edge.

Two seams found by asking "what happens next" at the fusion stage:

  1. Adding a NEW dataset to an already-fused set: `fuse_delegations` over a
     MIXED set (fan-out branches + a direct delegation) re-runs the gate,
     which needs `data_path` on every entry — direct delegations now carry
     it, so the enlarged fusion stays gated + computed instead of silently
     demoting to ungated.
  2. Fusion reveals a branch needs re-analysis: the synthesis may emit
     structured `branch_reanalysis` suggestions (rendered as
     suggested_followups), and a re-delegation citing the fusion in
     `context_from` is mechanically stamped informed_via=fusion_feedback,
     gets the additive-only note in its task, and is DISCOUNTED by the next
     fusion.

  conda run -n scilink python tests/test_fusion_incremental.py
"""
import json
import os
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
from test_fusion_numerics import BEHAVIORS, _install_fake_child

GATE_PROMPTS = []
FUSION_PROMPTS = []
REANALYSIS_REPLY = []   # when set, the fake synthesis includes it

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _install_fake_llm(fanout_set):
    def fake(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt:
            return {"method": "qualitative", "rationale": "r",
                    "script": "import json\n"
                              "json.dump({'method': 'qualitative', "
                              "'sigma_available': False, 'quantities': "
                              "{'q': 1.0}, 'notes': []}, "
                              "open('fusion_numerics.json', 'w'))\n"}
        if "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [],
                    "refinement_instructions": ""}
        if "complementary measurements of ONE system" in prompt:
            FUSION_PROMPTS.append(prompt)
            out = {"detailed_analysis": "fused narrative",
                   "scientific_claims": [{"claim": "c"}]}
            if REANALYSIS_REPLY:
                out["branch_reanalysis"] = list(REANALYSIS_REPLY)
            return out
        GATE_PROMPTS.append(prompt)
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "sample temperature",
                "join_type": "shared_parameter_axis",
                "fanout_set": list(fanout_set),
                "redundant_clusters": [], "unrelated": [],
                "excluded_notes": ""}
    fo._llm_json = fake


def _agent_with_fanout():
    d = tempfile.mkdtemp()
    A = os.path.join(d, "A.npy"); B = os.path.join(d, "B.npy")
    C = os.path.join(d, "C.npy")
    np.save(A, np.zeros((8, 8))); np.save(B, np.ones((8, 8)))
    np.save(C, np.full((8, 8), 2.0))
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    BEHAVIORS.clear(); BEHAVIORS[A] = "series"; BEHAVIORS[B] = "single"
    GATE_PROMPTS.clear(); FUSION_PROMPTS.clear(); REANALYSIS_REPLY.clear()
    _install_fake_child()
    _install_fake_llm([A, B])
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": A, "task": f"Analyze {A}", "label": "series branch"},
        {"data_path": B, "task": f"Analyze {B}", "label": "single branch"},
    ]))
    assert out.get("branches_run") == 2
    return ag, A, B, C


def _fake_direct_delegation(ag, C, data_path=True, context_from=None):
    """Simulate delegate_to_analysis via _delegate with a stubbed child."""
    class Child:
        def run_task(self, task, context=None, autonomy=None):
            Child.task = task
            return {"status": "success", "summary": "new dataset ok",
                    "key_findings": ["new finding"], "files_produced": []}
    ag._get_analysis_child = lambda: Child()
    ag._delegate("analysis", f"Analyze {C}", None, context_from,
                 "new dataset", data_path=(C if data_path else None))
    return ag._delegation_ledger[-1], getattr(Child, "task", "")


def main():
    # 1) Incremental fusion WITH the data_path stamp: mixed set re-gates and
    #    stays gated + computed.
    ag, A, B, C = _agent_with_fanout()
    f1 = json.loads(ag._fuse_delegations([1, 2]))
    check("baseline fan-out fusion gated", f1.get("complementarity_gated"))
    entry, _ = _fake_direct_delegation(ag, C, data_path=True)
    check("direct delegation carries data_path", entry.get("data_path") == C)
    n_gate = len(GATE_PROMPTS)
    _install_fake_llm([A, B, C])          # re-gate blesses the enlarged set
    ag._complementarity_cache.clear()
    f2 = json.loads(ag._fuse_delegations([1, 2, entry["index"]]))
    print("1) incremental fusion with data_path:")
    check("mixed set re-ran the gate", len(GATE_PROMPTS) > n_gate)
    check("enlarged fusion is GATED", f2.get("complementarity_gated"))
    check("computed reconciliation ran on the enlarged set",
          (f2.get("computed_reconciliation") or {}).get("status") == "success")
    check("no ungated warning", not f2.get("complementarity_warning"))

    # 2) Without data_path (legacy shape): demotes to ungated — the exact
    #    behavior the stamp exists to avoid.
    ag, A, B, C = _agent_with_fanout()
    entry, _ = _fake_direct_delegation(ag, C, data_path=False)
    f3 = json.loads(ag._fuse_delegations([1, 2, entry["index"]]))
    print("2) incremental fusion without data_path (legacy):")
    check("demotes to ungated", not f3.get("complementarity_gated")
          and f3.get("complementarity_warning"))
    check("ungated fusion computed nothing",
          (f3.get("computed_reconciliation") or {}).get("status") != "success")

    # 3) branch_reanalysis suggestions -> structured output + followups.
    ag, A, B, C = _agent_with_fanout()
    REANALYSIS_REPLY.append({"label": "series branch",
                             "reason": "model order contradicted by companions",
                             "suggestion": "refit testing a two-component model"})
    f4 = json.loads(ag._fuse_delegations([1, 2]))
    print("3) branch re-analysis suggestions:")
    check("synthesis asked for branch_reanalysis",
          "BRANCH RE-ANALYSIS" in FUSION_PROMPTS[-1])
    check("structured suggestions returned",
          (f4.get("branch_reanalysis") or [{}])[0].get("label") == "series branch")
    check("rendered followups carry the provenance instruction",
          f4.get("suggested_followups")
          and "context_from" in f4["suggested_followups"][0]
          and "additive-only" in f4["suggested_followups"][0])
    fus_entry = [e for e in ag._delegation_ledger if e.get("mode") == "fusion"][-1]
    check("fusion ledger entry carries followups + fused labels",
          fus_entry.get("suggested_followups")
          and fus_entry.get("labels") == ["series branch", "single branch"])

    # 4) Re-delegation citing the fusion -> mechanical fusion_feedback
    #    provenance + additive-only note; the NEXT fusion discounts it.
    fus_idx = fus_entry["index"]
    entry, sent_task = _fake_direct_delegation(
        ag, A, data_path=True, context_from=[fus_idx])
    print("4) fusion-feedback provenance:")
    check("informed_by = the fused branch labels",
          entry.get("informed_by") == ["series branch", "single branch"])
    check("informed_via = fusion_feedback",
          entry.get("informed_via") == "fusion_feedback")
    check("additive-only note appended to the child task",
          "ADDITIVE-ONLY" in sent_task and "CROSS-DATASET FUSION" in sent_task)
    check("ledger task matches the sent task (audit trail)",
          entry.get("task") == sent_task)
    _install_fake_llm([A, B])            # re-gate for the mixed re-fusion
    ag._complementarity_cache.clear()
    f5 = json.loads(ag._fuse_delegations([2, entry["index"]]))
    check("next fusion discounts the re-analyzed branch",
          any("re-analyzed with feedback" in str(c)
              for c in f5.get("caveats") or [])
          and "FUSION FEEDBACK" in FUSION_PROMPTS[-1])
    check("independence map lists the re-analyzed branch",
          (f5.get("independence") or {}).get("new dataset")
          == ["series branch", "single branch"])

    # 5) Ordinary delegation (no fusion in context_from) stays unstamped.
    entry, sent_task = _fake_direct_delegation(
        ag, B, data_path=True, context_from=[1])
    print("5) non-fusion context_from unaffected:")
    check("no informed_by", not entry.get("informed_by"))
    check("no note appended", "ADDITIVE-ONLY" not in sent_task)

    # 6) Same-directory pattern branches + a late dataset: the RE-GATE must
    #    carry branch identity (id/task/files) — path-only descriptors
    #    collapse same-dir branches into "redundant" (found live; the #326
    #    lesson resurfacing in the re-gate path).
    from test_fanout_steering import _write_series
    d = tempfile.mkdtemp()
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    up = os.path.join(d, "uploads"); os.makedirs(up)
    pat_a = _write_series(up, "techA")
    pat_b = _write_series(up, "techB")
    C = os.path.join(d, "late.npy"); np.save(C, np.zeros((8, 8)))
    GATE_PROMPTS.clear(); FUSION_PROMPTS.clear(); REANALYSIS_REPLY.clear()
    _install_fake_child()
    _install_fake_llm([f"{up}#1", f"{up}#2"])
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": pat_a, "label": "techA series",
         "task": "Analyze the techA series."},
        {"data_path": up, "pattern": pat_b, "label": "techB series",
         "task": "Analyze the techB series."},
    ]))
    assert out.get("branches_run") == 2
    entry, _ = _fake_direct_delegation(ag, C, data_path=True)
    _install_fake_llm(["techA series#1", "techB series#2",
                       f"new dataset#{entry['index']}"])
    ag._complementarity_cache.clear()
    n_gate = len(GATE_PROMPTS)
    f6 = json.loads(ag._fuse_delegations([1, 2, entry["index"]]))
    print("6) same-dir branches in the incremental re-gate:")
    check("re-gate ran for the mixed same-dir set", len(GATE_PROMPTS) > n_gate)
    gp = GATE_PROMPTS[-1]
    check("re-gate descriptors carry branch identity",
          "techA series#1" in gp and "techB series#2" in gp
          and "analysis_task" in gp)
    check("re-gate descriptors carry per-branch file sets",
          '"n_files"' in gp)
    check("same-dir incremental fusion stays gated",
          f6.get("complementarity_gated"))
    # Partial coverage rule: a partial verdict whose fanout_set misses one
    # fused entry must NOT gate this fusion.
    def _partial_llm(fset):
        def fake(orch, prompt, extra_parts=None):
            if "complementary measurements of ONE system" in prompt:
                FUSION_PROMPTS.append(prompt)
                return {"detailed_analysis": "n", "scientific_claims": []}
            if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
                return {"verdict": "accept", "issues": [],
                        "refinement_instructions": "", "method": "qualitative",
                        "rationale": "r", "script": ""}
            GATE_PROMPTS.append(prompt)
            return {"verdict": "partially_complementary", "confidence": 0.8,
                    "rationale": "r", "join_axis": "T",
                    "join_type": "shared_parameter_axis", "fanout_set": fset,
                    "redundant_clusters": [], "unrelated": [],
                    "excluded_notes": ""}
        fo._llm_json = fake
    _partial_llm(["techA series#1", "techB series#2"])   # misses the newcomer
    ag._complementarity_cache.clear()
    f7 = json.loads(ag._fuse_delegations([1, 2, entry["index"]]))
    check("partial verdict NOT covering all entries -> ungated",
          not f7.get("complementarity_gated"))
    _partial_llm(["techA series#1", "techB series#2",
                  f"new dataset#{entry['index']}"])       # covers all
    ag._complementarity_cache.clear()
    f8 = json.loads(ag._fuse_delegations([1, 2, entry["index"]]))
    check("partial verdict covering all entries -> gated",
          f8.get("complementarity_gated"))

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FUSION INCREMENTAL: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
