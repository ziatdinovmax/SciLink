"""TRUE end-to-end chat validation of the #296 stack + mesh policy.

Drives the META CHAT with user-style messages and lets the meta LLM itself
compose the fan-out branches, the fusion, and (turn 2) the steering opt-in —
nothing hand-authored. This exercises the layer every scripted driver
bypasses (delegate_to_analyses composition + schema discoverability), which
is where the original #326 session went wrong.

Turn 1: parallel cross-analysis of the ibuprofen set -> independent
        branches, computed+audited fusion.
Turn 2: re-run with the XRD transition location guiding (steering) the FTIR
        branch -> meta must use the `steer` flag; provenance + discount.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_meta_chat_e2e_live.py [base_dir]
"""
import json
import os
import sys
import tempfile

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
UPLOADS = os.path.expanduser("~/Code/meta_session_20260701_083203/uploads")

USER_MSG_1 = f"""I uploaded three datasets from one in-situ study of ibuprofen \
sodium dihydrate dehydration, all in the folder {UPLOADS}:
1. a TG-DSC thermal analysis curve: 'ISD_TGDSC_temp x axis.txt' (NETZSCH \
export, temperature x-axis);
2. an in-situ FTIR temperature series: the files 'In-situ ibuprofen_*C.txt' \
(35-120 C, two columns: wavenumber cm-1, absorbance);
3. an in-situ XRD temperature series: the files \
'IBD_Insitu_Dehydration_01_*.txt' (38 patterns, temperature in the filename, \
Cu K-alpha 1.5406 A, two columns: 2theta deg, intensity cps).

Please analyze the three datasets in parallel and then fuse the findings \
into one cross-dataset interpretation of the dehydration."""

USER_MSG_2 = """Nice. Now re-run just the FTIR and XRD series in parallel \
once more, but this time let the XRD series' transition location guide \
(steer) the FTIR branch — I know steering trades away that branch's \
independence and I want it anyway. Then fuse the two new analyses."""

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    base = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(
        prefix="chat_e2e_296_")
    print("session dir:", base)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL,
                               meta_mode=MetaMode.AUTONOMOUS)

    # ---------------- turn 1: parallel + fuse ----------------
    print("### TURN 1 ###\n" + USER_MSG_1)
    reply = ag.chat(USER_MSG_1)
    print("\n### META REPLY 1 (first 1500) ###\n" + reply[:1500])

    fan1 = [e for e in ag._delegation_ledger if e.get("fanout")]
    fus1 = [e for e in ag._delegation_ledger if e.get("mode") == "fusion"]
    for e in ag._delegation_ledger:
        print(f"  #{e['index']} mode={e.get('mode')} fanout={e.get('fanout')} "
              f"label={e.get('label')!r} status={e.get('status')} "
              f"informed_by={e.get('informed_by')}")
    check("turn1: meta composed a 3-branch fan-out", len(fan1) == 3)
    t = " || ".join(str(e.get("task", "")) + " " + str(e.get("label", ""))
                    for e in fan1).lower()
    check("turn1: DSC, FTIR and XRD each present",
          all(s in t for s in ("dsc", "ftir", "xrd")))
    check("turn1: series branches carry patterns",
          sum(1 for e in fan1 if e.get("pattern")) >= 2)
    check("turn1: independent wiring (no informed_by, no companions)",
          all(not e.get("informed_by") for e in fan1)
          and all("COMPANION DATASETS" not in (e.get("task") or "")
                  for e in fan1))
    check("turn1: all branches succeeded",
          all(e.get("status") == "success" for e in fan1))
    check("turn1: meta ran the fusion", len(fus1) >= 1)
    comp1 = {}
    if fus1:
        rp = next((f for f in fus1[-1].get("files_produced", [])
                   if str(f).endswith("fusion_report.json")), None)
        rep = json.load(open(rp)) if rp and os.path.exists(rp) else {}
        comp1 = rep.get("computed_reconciliation") or {}
        check("turn1: numerics bundle reached fusion",
              len(rep.get("branch_numerics") or {}) >= 2)
        check("turn1: computed reconciliation ran through chat",
              comp1.get("status") == "success")
        check("turn1: audit verdict recorded",
              (comp1.get("verification") or {}).get("verdict")
              in ("accept", "refine", "unavailable"))
        check("turn1: no independence flags on unsteered run",
              not rep.get("independence"))

    # ---------------- turn 2: steered re-run ----------------
    print("\n### TURN 2 ###\n" + USER_MSG_2)
    reply2 = ag.chat(USER_MSG_2)
    print("\n### META REPLY 2 (first 1500) ###\n" + reply2[:1500])

    fan2 = [e for e in ag._delegation_ledger
            if e.get("fanout") and e not in fan1]
    fus2 = [e for e in ag._delegation_ledger
            if e.get("mode") == "fusion" and e not in fus1]
    for e in fan2 + fus2:
        print(f"  #{e['index']} mode={e.get('mode')} label={e.get('label')!r} "
              f"status={e.get('status')} informed_by={e.get('informed_by')} "
              f"via={e.get('informed_via')}")
    check("turn2: a second fan-out ran", len(fan2) >= 2)
    steered = [e for e in fan2 if e.get("informed_by")]
    check("turn2: meta discovered and used the steer flag",
          len(steered) == 1 and "steering" in (steered[0].get("informed_via")
                                               or ""))
    check("turn2: the steered branch is the FTIR one",
          steered and "ftir" in (str(steered[0].get("label", ""))
                                 + str(steered[0].get("task", ""))).lower())
    check("turn2: steering hint + guardrail in the branch task",
          steered and "STEERING" in (steered[0].get("task") or "")
          and "Never restrict a fit window" in (steered[0].get("task") or ""))
    check("turn2: meta fused the steered pair", len(fus2) >= 1)
    if fus2:
        rp = next((f for f in fus2[-1].get("files_produced", [])
                   if str(f).endswith("fusion_report.json")), None)
        rep2 = json.load(open(rp)) if rp and os.path.exists(rp) else {}
        check("turn2: independence recorded in fusion report",
              bool(rep2.get("independence")))
        check("turn2: independence caveat present",
              any("steered at launch" in str(c)
                  for c in rep2.get("caveats") or []))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"CHAT E2E (#296 + mesh + steering): {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
