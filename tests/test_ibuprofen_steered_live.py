"""Real-data steering validation (#296 phase d) on the ibuprofen set.

The FTIR series branch opts into steering; its companion XRD series' cheap
reduction points at the structural transition (~46 C). FTIR's own evidence
puts its water-loss markers well above that (52-110 C depending on band), so
this is a REAL prior-vs-evidence test: the steered branch must test the ~46 C
hypothesis, keep analyzing its full range, report its own values, and state
the disagreement; fusion must carry the independence discount.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_ibuprofen_steered_live.py
"""
import json
import os
import re
import sys
import tempfile

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
UPLOADS = os.path.expanduser("~/Code/meta_session_20260701_083203/uploads")

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="ibu_steered_")
    print("session dir:", base)
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    subject = "ibuprofen sodium dihydrate undergoing thermal dehydration"
    up = UPLOADS
    out = json.loads(ag._run_fanout([
        {"data_path": up, "pattern": "In-situ ibuprofen_*C.txt",
         "label": "in-situ FTIR series", "steer": True,
         "task": f"Analyze the measurement series in {up} — ONLY the files "
                 f"matching 'In-situ ibuprofen_*C.txt'. They are a series "
                 f"over temperature on {subject}. Track how the bands evolve "
                 "across the series and characterize the transition it "
                 "reveals, reporting the transition temperature(s) your own "
                 "data supports."},
        {"data_path": up, "pattern": "IBD_Insitu_Dehydration_01_*.txt",
         "label": "in-situ XRD series",
         "task": f"Analyze the measurement series in {up} — ONLY the files "
                 f"matching 'IBD_Insitu_Dehydration_01_*.txt'. They are a "
                 f"series over temperature on {subject}. Track the "
                 "reflections and characterize the structural transition "
                 "they reveal."},
    ]))
    check("2 branches ran", out.get("branches_run") == 2)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    by_label = {e["label"]: e for e in fan}
    ftir = by_label.get("in-situ FTIR series", {})
    check("FTIR entry informed_by XRD",
          ftir.get("informed_by") == ["in-situ XRD series"])
    task = ftir.get("task", "")
    m = re.search(r"sharpest change near\s+control ≈ (\d+(?:\.\d+)?)", task)
    hint = float(m.group(1)) if m else None
    print(f"steering hint from XRD reduction: ≈ {hint} C "
          "(XRD structural transition ~46 C)")
    check("hint near the XRD structural transition (40-55 C)",
          hint is not None and 40.0 <= hint <= 55.0)
    check("additive-only guardrail in task",
          "Never restrict a fit window" in task)

    summary = (ftir.get("summary", "") + " "
               + " ".join(str(k) for k in ftir.get("key_findings", [])))
    print("\nFTIR branch summary (first 1500):\n", summary[:1500])
    # FTIR's own evidence: markers in 50-115 C (its bands). It must NOT
    # CLAIM ~46 C as its own transition (explicit claim word within 25
    # chars of the value; a passing mention or a reported disagreement with
    # the hint is fine).
    claim_re = (r"(?:transition|onset|midpoint|breakpoint|change ?point|"
                r"T0|T1/2)[^.;\d]{0,25}(4[3-9](?:\.\d+)?)"
                r"|(4[3-9](?:\.\d+)?)\s*°?\s*C[^.;a-z]{0,10}"
                r"(?:transition|onset|midpoint|breakpoint)")
    ftir_sents = [s for s in re.split(r"(?<=[.;])\s+", summary)
                  if not any(w in s.lower() for w in
                             ("xrd", "companion", "steering", "hint",
                              "structural", "no ", "not ", "absen",
                              "disagree", "does not", "lack"))]
    claimed = [float(a or b) for s in ftir_sents
               for a, b in re.findall(claim_re, s, re.IGNORECASE)]
    print("explicit own transition-claims near 46 C:", claimed)
    check("FTIR does not adopt ~46 C as its own transition", not claimed)
    check("FTIR reports markers in its own range (50-115 C)",
          any(50.0 <= float(v) <= 115.0
              for v in re.findall(r"\b(\d{2,3}(?:\.\d+)?)\s*°?\s*C\b", summary)))

    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Do the datasets agree on where the transition occurs "
                    "along temperature?"))
    check("fusion success", fout.get("status") == "success")
    check("fusion independence provenance",
          (fout.get("independence") or {}).get("in-situ FTIR series")
          == ["in-situ XRD series"])
    check("independence caveat present",
          any("steered at launch" in str(c) for c in fout.get("caveats") or []))
    comp = fout.get("computed_reconciliation") or {}
    print("\ncomputed:", comp.get("status"), "| audit:",
          (comp.get("verification") or {}).get("verdict"),
          "| attempts:", comp.get("attempts"))
    print("QUANTITIES:", json.dumps(
        (comp.get("results") or {}).get("quantities"), default=str)[:1500])
    print("\nNARRATIVE (first 1500):\n",
          str(fout.get("detailed_analysis", ""))[:1500])
    print("\nreport:", fout.get("report_html_path"))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"IBUPROFEN STEERED LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
