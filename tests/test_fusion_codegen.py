"""Offline tests for the fusion computed reconciliation (issue #296 phase b).

No network — monkeypatches the gate/codegen/fusion LLM and the ephemeral
child (reusing the artifact-writing fakes from test_fusion_numerics). The
canned codegen replies carry REAL scripts that ScriptExecutor actually runs,
so the generate -> execute -> persist loop and its retry/degrade paths are
exercised end to end.

  UNSAFE_EXECUTION_OK=true conda run -n scilink python tests/test_fusion_codegen.py
"""
import json
import os
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

import numpy as np
import scilink.agents.meta_agent.fanout as fo
import scilink.executors as executors
from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
from test_fusion_numerics import (
    BEHAVIORS, _install_fake_child, _branches)


def _agent():
    d = tempfile.mkdtemp()
    A = os.path.join(d, "A.npy"); B = os.path.join(d, "B.npy")
    np.save(A, np.zeros((8, 8))); np.save(B, np.ones((8, 8)))
    ag = MetaOrchestratorAgent(base_dir=d, api_key="sk-dummy",
                               model_name="claude-opus-4-6",
                               meta_mode=MetaMode.AUTONOMOUS)
    return ag, A, B


GOOD_SCRIPT = """import json
from PIL import Image
res = {"method": "correspondence", "sigma_available": False,
       "quantities": {"xrd_transition_C": 46.6, "ftir_transition_C": 52.1,
                      "offset_C": 5.5},
       "notes": ["computed by offline test script"]}
with open("fusion_numerics.json", "w") as fh:
    json.dump(res, fh)
Image.new("RGB", (8, 8), "red").save("fusion_figure.png")
print("offset 5.5 C between techniques")
"""

GOOD_SCRIPT2 = GOOD_SCRIPT.replace('"offset_C": 5.5', '"offset_C": 6.1')

BAD_SCRIPT = 'raise RuntimeError("synthetic script failure")\n'

NO_OUTPUT_SCRIPT = 'print("ran fine but wrote nothing")\n'


# Scripted codegen/verification replies, consumed in order; repeat the last.
CODEGEN_REPLIES = []
CODEGEN_CALLS = []      # captured codegen prompts
VERIFY_REPLIES = []     # verification verdict dicts (default: accept)
VERIFY_CALLS = []       # captured verification prompts
FUSION_PROMPTS = []     # captured HOLISTIC synthesis prompts

ACCEPT = {"verdict": "accept", "issues": [], "refinement_instructions": ""}
REFINE = {"verdict": "refine",
          "issues": ["degenerate sigmoid width far below sampling interval"],
          "refinement_instructions": "use the dihydrate peak area as the "
                                     "order parameter"}


def _install_fake_llm(fanout_set, join_axis="sample temperature"):
    def fake(orch, prompt, extra_parts=None):
        if "AUDITING a computed" in prompt:                   # verification
            VERIFY_CALLS.append(prompt)
            if not VERIFY_REPLIES:
                return dict(ACCEPT)
            i = min(len(VERIFY_CALLS) - 1, len(VERIFY_REPLIES) - 1)
            return dict(VERIFY_REPLIES[i])
        if "SCRIPT CONTRACT" in prompt:                       # codegen call
            CODEGEN_CALLS.append(prompt)
            i = min(len(CODEGEN_CALLS) - 1, len(CODEGEN_REPLIES) - 1)
            script = CODEGEN_REPLIES[i]
            return {"method": "correspondence", "rationale": "r",
                    "script": script}
        if "complementary measurements of ONE system" in prompt:  # synthesis
            FUSION_PROMPTS.append(prompt)
            return {"detailed_analysis": "fused narrative",
                    "scientific_claims": [{"claim": "c", "keywords": ["k"]}]}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": join_axis, "fanout_set": list(fanout_set),
                "redundant_clusters": [], "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake


def _run(ag, A, B, replies, verify=None):
    BEHAVIORS.clear(); BEHAVIORS[A] = "series"; BEHAVIORS[B] = "single"
    CODEGEN_REPLIES[:] = replies
    VERIFY_REPLIES[:] = (verify or [])
    CODEGEN_CALLS.clear(); VERIFY_CALLS.clear(); FUSION_PROMPTS.clear()
    _install_fake_llm([A, B])
    ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    fused = json.loads(ag._fuse_delegations(idxs))
    return fused, (FUSION_PROMPTS[-1] if FUSION_PROMPTS else "")


results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    _install_fake_child()

    # 1) Happy path: script generated, executed, artifacts persisted,
    #    computed results ground the synthesis prompt.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [GOOD_SCRIPT])
    print("1) codegen happy path:")
    comp = fused.get("computed_reconciliation") or {}
    check("fusion success", fused["status"] == "success")
    check("computed status success", comp.get("status") == "success")
    check("one attempt", comp.get("attempts") == 1)
    check("script persisted", comp.get("script_path")
          and os.path.exists(comp["script_path"]))
    check("fusion_numerics.json persisted", comp.get("numerics_path")
          and os.path.exists(comp["numerics_path"]))
    check("figure persisted", comp.get("figure_path")
          and os.path.exists(comp["figure_path"]))
    check("computed quantities parsed",
          (comp.get("results") or {}).get("quantities", {}).get("offset_C") == 5.5)
    check("COMPUTED RECONCILIATION block in synthesis prompt",
          "COMPUTED RECONCILIATION" in prompt and "offset_C" in prompt)
    check("script path cited in prompt", "fusion_reconciliation.py" in prompt)
    check("computed figure attached", fused.get("figures_used", 0) >= 1)
    report = json.load(open(fused["report_path"]))
    check("report carries computed_reconciliation",
          (report.get("computed_reconciliation") or {}).get("status") == "success")
    fusion_entry = [e for e in ag._delegation_ledger if e.get("mode") == "fusion"][-1]
    check("ledger files_produced includes script + numerics",
          any("fusion_reconciliation.py" in f for f in fusion_entry["files_produced"])
          and any("fusion_numerics.json" in f for f in fusion_entry["files_produced"]))
    check("verification recorded as accept",
          (comp.get("verification") or {}).get("verdict") == "accept")
    check("audit acceptance told to synthesis",
          "audit ACCEPTED this computation" in prompt)
    html = open(fused["report_html_path"]).read()
    check("HTML has computed-reconciliation card",
          "Computed reconciliation" in html and "offset_C" in html)
    check("HTML shows accepted audit badge", "audit: accepted" in html)

    # 2) First attempt crashes -> retry with the error fed back -> success.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [BAD_SCRIPT, GOOD_SCRIPT])
    print("2) retry on runtime error:")
    comp = fused.get("computed_reconciliation") or {}
    check("recovered on attempt 2", comp.get("status") == "success"
          and comp.get("attempts") == 2)
    check("error was fed back to the second attempt",
          len(CODEGEN_CALLS) >= 2
          and "PREVIOUS ATTEMPT FAILED" in CODEGEN_CALLS[1]
          and "synthetic script failure" in CODEGEN_CALLS[1])

    # 3) Script runs but never writes the contract file -> counted as failure.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [NO_OUTPUT_SCRIPT])
    print("3) missing contract output degrades:")
    comp = fused.get("computed_reconciliation") or {}
    check("fusion still success (degrade, never block)",
          fused["status"] == "success")
    check("computed status failed", comp.get("status") == "failed")
    check("all attempts consumed", comp.get("attempts") == 3)
    check("warning in caveats", any("computed reconciliation failed" in str(c)
                                    for c in fused.get("caveats") or []))
    check("synthesis told computation unavailable",
          "COMPUTED RECONCILIATION UNAVAILABLE" in prompt)

    # 4) Persistent crash -> same degrade path.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [BAD_SCRIPT])
    print("4) persistent crash degrades:")
    comp = fused.get("computed_reconciliation") or {}
    check("crash: computed failed after retries", comp.get("status") == "failed")
    check("crash: fusion still success", fused["status"] == "success")

    # 5) Ungated fusion must NOT compute: strip the gate provenance so the
    #    honesty guard fires, and confirm codegen was skipped.
    ag, A, B = _agent()
    BEHAVIORS.clear(); BEHAVIORS[A] = "series"; BEHAVIORS[B] = "single"
    CODEGEN_REPLIES[:] = [GOOD_SCRIPT]
    CODEGEN_CALLS.clear(); FUSION_PROMPTS.clear()
    _install_fake_llm([A, B]); ag._complementarity_cache.clear()
    out = json.loads(ag._run_fanout(_branches(A, B)))
    idxs = [r["delegation_index"] for r in out["results"] if r["produced_output"]]
    for e in ag._delegation_ledger:
        if e.get("fanout"):
            e.pop("fanout"); e.pop("parallel_group"); e.pop("data_path", None)
    fused = json.loads(ag._fuse_delegations(idxs))
    print("5) ungated fusion computes nothing:")
    comp = fused.get("computed_reconciliation") or {}
    check("ungated: codegen skipped", comp.get("status") == "skipped")
    check("ungated: no codegen LLM call", len(CODEGEN_CALLS) == 0)
    check("ungated: warning recorded",
          "ungated" in str(comp.get("warning", "")))

    # 6) Sandbox declined -> skipped with actionable warning, no execution.
    ag, A, B = _agent()
    real_approval = executors.require_sandbox_approval
    executors.require_sandbox_approval = lambda **kw: False
    try:
        fused, prompt = _run(ag, A, B, [GOOD_SCRIPT])
    finally:
        executors.require_sandbox_approval = real_approval
    print("6) sandbox declined:")
    comp = fused.get("computed_reconciliation") or {}
    check("declined: skipped", comp.get("status") == "skipped")
    check("declined: warning names the fix",
          "UNSAFE_EXECUTION_OK" in str(comp.get("warning", "")))
    check("declined: fusion still success", fused["status"] == "success")

    # 7) Audit refine -> regenerate with the audit's instructions -> accept.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [GOOD_SCRIPT, GOOD_SCRIPT2],
                         verify=[REFINE, ACCEPT])
    print("7) audit-driven refinement:")
    comp = fused.get("computed_reconciliation") or {}
    check("refined to accept on attempt 2", comp.get("status") == "success"
          and comp.get("attempts") == 2
          and (comp.get("verification") or {}).get("verdict") == "accept")
    check("audit instructions fed to the second generation",
          len(CODEGEN_CALLS) >= 2
          and "FAILED A SCIENTIFIC AUDIT" in CODEGEN_CALLS[1]
          and "dihydrate peak area" in CODEGEN_CALLS[1])
    check("refined quantities returned",
          (comp.get("results") or {}).get("quantities", {}).get("offset_C") == 6.1)
    check("verification saw script + figure inputs",
          len(VERIFY_CALLS) >= 1 and "EXECUTED SCRIPT" in VERIFY_CALLS[0])

    # 8) Audit never satisfied -> best flagged result kept, honestly marked.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [GOOD_SCRIPT], verify=[REFINE])
    print("8) persistent refine keeps flagged best:")
    comp = fused.get("computed_reconciliation") or {}
    check("kept with status success", comp.get("status") == "success")
    check("verdict recorded as refine",
          (comp.get("verification") or {}).get("verdict") == "refine")
    check("UNRESOLVED warning set",
          "UNRESOLVED" in str(comp.get("warning", "")))
    check("warning propagated to caveats",
          any("UNRESOLVED" in str(c) for c in fused.get("caveats") or []))
    check("synthesis warned about unresolved audit",
          "UNRESOLVED issues" in prompt)
    html = open(fused["report_html_path"]).read()
    check("HTML shows unresolved-audit badge", "audit: unresolved issues" in html)

    # 9) Crash after a refine-flagged success -> artifacts restored to the
    #    returned (best) attempt so the audit trail matches the report.
    ag, A, B = _agent()
    fused, prompt = _run(ag, A, B, [GOOD_SCRIPT, BAD_SCRIPT],
                         verify=[REFINE])
    print("9) artifact restore after late crash:")
    comp = fused.get("computed_reconciliation") or {}
    check("flagged best returned despite later crashes",
          comp.get("status") == "success"
          and "UNRESOLVED" in str(comp.get("warning", "")))
    check("script on disk is the returned attempt's",
          "offset_C" in open(comp["script_path"]).read())
    saved = json.load(open(comp["numerics_path"]))
    check("numerics on disk match the returned results",
          saved.get("quantities", {}).get("offset_C") == 5.5)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FUSION CODEGEN: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
