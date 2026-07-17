"""Offline tests for the mesh policy (#296 follow-up to #293).

The gate's join type decides the wiring: co-registered sets keep the operand
mesh (recorded as informed_by — a joint computation is not corroboration);
shared-axis / bulk-vs-local / unclassified sets run INDEPENDENT branches.
Steering composes with either. No network — fakes as in the sibling suites.

  conda run -n scilink python tests/test_mesh_policy.py
"""
import json
import os
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import scilink.agents.meta_agent.fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
from test_fanout_steering import _write_series

CAPTURED_TASKS = []
FUSION_PROMPTS = []

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _install_fakes(fanout_set, join_type):
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                CAPTURED_TASKS.append(task)
                return {"status": "success", "summary": "ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    fo._make_ephemeral_analysis_child = fake_child

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt:
            return {"method": "qualitative", "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            FUSION_PROMPTS.append(prompt)
            return {"detailed_analysis": "fused narrative",
                    "scientific_claims": [{"claim": "c"}]}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "grid", "join_type": join_type,
                "fanout_set": list(fanout_set),
                "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake_llm


def _run(join_type, steer_first=False):
    d = tempfile.mkdtemp()
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    up = os.path.join(d, "uploads"); os.makedirs(up)
    pat_a = _write_series(up, "techA")
    pat_b = _write_series(up, "techB")
    CAPTURED_TASKS.clear(); FUSION_PROMPTS.clear()
    _install_fakes([f"{up}#1", f"{up}#2"], join_type)
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": pat_a, "label": "techA series",
         "task": "Analyze the techA series.", "steer": steer_first},
        {"data_path": up, "pattern": pat_b, "label": "techB series",
         "task": "Analyze the techB series."},
    ]))
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    fused = json.loads(ag._fuse_delegations([e["index"] for e in fan]))
    return out, fan, fused, list(CAPTURED_TASKS), (
        FUSION_PROMPTS[-1] if FUSION_PROMPTS else "")


def main():
    # 1) Shared parameter axis -> INDEPENDENT branches (the new default).
    out, fan, fused, tasks, prompt = _run("shared_parameter_axis")
    print("1) shared_parameter_axis -> independent:")
    check("run success, 2 branches", out.get("branches_run") == 2)
    check("result declares mesh=independent",
          out.get("mesh") == "independent"
          and out.get("join_type") == "shared_parameter_axis")
    check("no COMPANION operands in any task",
          all("COMPANION DATASETS" not in t for t in tasks))
    check("PRIMARY glob instruction still present (Fix 3 kept)",
          all("PRIMARY dataset for THIS analysis" in t
              and "FILE PATTERN" in t for t in tasks))
    check("no informed_by stamped", all(not e.get("informed_by") for e in fan))
    check("no independence block in fusion",
          fused.get("independence") is None
          and "INDEPENDENCE PROVENANCE" not in prompt)

    # 2) Co-registered -> operand mesh, recorded and discounted.
    out, fan, fused, tasks, prompt = _run("co_registered")
    print("2) co_registered -> operand mesh:")
    check("result declares mesh=operand", out.get("mesh") == "operand")
    check("companions passed as operands",
          all("COMPANION DATASETS" in t and "auxiliary_data" in t
              for t in tasks))
    check("informed_by mutual with co-registered mode",
          all(e.get("informed_by") and e.get("informed_via")
              == "co_registered_operands" for e in fan))
    check("fusion prompt carries co-registered provenance",
          "INDEPENDENCE PROVENANCE" in prompt
          and "CO-REGISTERED OPERANDS" in prompt)
    check("joint-computation caveat in report",
          any("co-registered numerical operand" in str(c)
              for c in fused.get("caveats") or []))
    check("independence map covers both branches",
          len(fused.get("independence") or {}) == 2)

    # 3) Gate did not classify (old-style verdict) -> independent (fail-safe:
    #    never spend independence by accident).
    out, fan, fused, tasks, prompt = _run(None)
    print("3) unclassified join type -> independent:")
    check("mesh=independent on missing join_type",
          out.get("mesh") == "independent")
    check("no companions, no informed_by",
          all("COMPANION DATASETS" not in t for t in tasks)
          and all(not e.get("informed_by") for e in fan))

    # 4) Steering composes: co-registered mesh + steer on one branch.
    out, fan, fused, tasks, prompt = _run("co_registered", steer_first=True)
    print("4) mesh + steering compose:")
    by_label = {e["label"]: e for e in fan}
    a = by_label["techA series"]
    check("steered branch records both modes",
          a.get("informed_via") == "co_registered_operands+steering")
    check("steered task has STEERING and COMPANION blocks",
          any("STEERING" in t and "COMPANION DATASETS" in t for t in tasks))
    check("both caveat kinds in report",
          any("steered at launch" in str(c) for c in fused.get("caveats") or [])
          and any("co-registered numerical operand" in str(c)
                  for c in fused.get("caveats") or []))

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"MESH POLICY: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
