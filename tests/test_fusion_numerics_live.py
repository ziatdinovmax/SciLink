"""Live validation for #296 phases (a) + (b) — numerics bundle + computed
reconciliation.

Runs the same 3-dataset in-situ layout as the #326 live test (one standalone
thermal curve + two file series sharing a directory), then asserts the NEW
behavior: each branch's feature tables are schema-previewed into the fusion
prompt (captured via a pass-through spy on the fusion LLM call), the gate's
join_axis is stamped on the branch ledger entries, the fusion report carries
the branch_numerics audit trail, and — phase (b) — a reconciliation script is
generated, EXECUTED over the branch tables, and persisted with its computed
fusion_numerics.json (+ overlay figure), whose quantities ground the
synthesis prompt.

  export AWS_BEARER_TOKEN_BEDROCK=...   AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_fusion_numerics_live.py

Defaults point at the ibuprofen sodium dihydrate dehydration study
(TG-DSC + in-situ FTIR series + in-situ XRD series); override with the same
flags as tests/test_meta_fanout_same_dir_live.py for other data.
"""
import argparse
import glob
import json
import os
import sys
import tempfile

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
DEFAULT_UPLOADS = os.path.expanduser(
    "~/Code/meta_session_20260701_083203/uploads")


def _branches(up, args):
    single = os.path.join(up, args.single)
    subject = args.subject
    out = [{
        "data_path": single,
        "label": args.single_label,
        "task": (f"Analyze the data file at {single}. It is one measurement of "
                 f"{subject}. Characterize the features it resolves and report "
                 "the values that describe them."),
    }]
    for pattern, label in zip(args.series, args.series_label):
        out.append({
            "data_path": up,
            "pattern": pattern,
            "label": label,
            "task": (f"Analyze the measurement series in {up} — ONLY the files "
                     f"matching '{pattern}'. They are a series over the study's "
                     f"control variable on {subject}. Track how the signal "
                     "evolves across the series and characterize the transition "
                     "it reveals."),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", default=DEFAULT_UPLOADS)
    ap.add_argument("--single", default="ISD_TGDSC_temp x axis.txt")
    ap.add_argument("--single-label", default="TG-DSC thermal curve")
    ap.add_argument("--series", action="append", default=None)
    ap.add_argument("--series-label", action="append", default=None)
    ap.add_argument("--subject",
                    default="ibuprofen sodium dihydrate undergoing thermal "
                            "dehydration")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()
    if not args.series:
        args.series = ["In-situ ibuprofen_*C.txt",
                       "IBD_Insitu_Dehydration_01_*.txt"]
        args.series_label = ["in-situ FTIR series", "in-situ XRD series"]

    up = os.path.abspath(os.path.expanduser(args.uploads))
    if not os.path.isfile(os.path.join(up, args.single)):
        print(f"--single file not found under {up}"); sys.exit(2)
    for pat in args.series:
        n = len([f for f in glob.glob(os.path.join(up, pat)) if os.path.isfile(f)])
        print(f"series {pat!r}: {n} files")
        if n == 0:
            sys.exit(2)
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)

    base = args.base_dir or tempfile.mkdtemp(prefix="fusion_numerics_live_")
    print(f"session dir: {base}")

    import scilink.agents.meta_agent.fanout as fo
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)

    # Pass-through spy: record the HOLISTIC fusion prompt (the thing phase (a)
    # is supposed to enrich) while the REAL LLM call still runs.
    FUSION_PROMPTS = []
    _real_llm_json = fo._llm_json

    def _spy(orch, prompt, extra_parts=None):
        if "complementary measurements of ONE system" in prompt:
            FUSION_PROMPTS.append(prompt)
        return _real_llm_json(orch, prompt, extra_parts=extra_parts)

    fo._llm_json = _spy

    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)

    checks = {}

    def check(name, cond):
        checks[name] = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    n_expected = 1 + len(args.series)
    print(f"\n### FAN-OUT: {n_expected} datasets over {up} ###")
    out = json.loads(ag._run_fanout(_branches(up, args)))
    print("FANOUT RESULT:", json.dumps(
        {k: out.get(k) for k in ("status", "reason", "join_axis",
                                 "branches_run", "branches_with_output")},
        indent=2))
    check("fan-out ran (status=success)", out.get("status") == "success")
    check(f"{n_expected} branches ran", out.get("branches_run") == n_expected)

    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    check("branch entries stamped with join_axis (#296)",
          fan and all(e.get("join_axis") for e in fan))
    check("branches carry feature_tables in ledger",
          sum(1 for e in fan if e.get("feature_tables")) >= 2)

    productive = [r["delegation_index"] for r in out.get("results", [])
                  if r.get("produced_output")]
    check("all branches produced output", len(productive) == n_expected)
    if len(productive) < 2:
        print("Cannot fuse; aborting."); sys.exit(1)

    print("\n### FUSION (numerics-aware) ###")
    fout = json.loads(ag._fuse_delegations(
        productive,
        focus="Do the datasets agree on where the transition occurs along "
              "the study's shared control variable?"))
    check("fusion succeeded", fout.get("status") == "success")
    check("fusion was complementarity-gated",
          bool(fout.get("complementarity_gated")))
    check("fusion result carries join_axis", bool(fout.get("join_axis")))
    check("numerics for >= 2 branches",
          (fout.get("numerics_branches") or 0) >= 2)

    # The enriched prompt — the artifact phase (a) exists to produce.
    check("fusion prompt captured", bool(FUSION_PROMPTS))
    prompt = FUSION_PROMPTS[-1] if FUSION_PROMPTS else ""
    check("NUMERICS preamble in prompt", "NUMERICS:" in prompt)
    check(">= 2 feature-table previews in prompt",
          prompt.count("- feature table:") >= 2)
    check("a join-axis column resolved with a range",
          "join-axis column:" in prompt and "(range " in prompt)
    check("a branch trend analysis reached the prompt",
          "trend analysis" in prompt)
    check("audit paths in prompt", "features.csv" in prompt)

    rp = fout.get("report_path")
    report = json.load(open(rp)) if rp and os.path.exists(rp) else {}
    bn = report.get("branch_numerics") or {}
    check("report branch_numerics for >= 2 branches", len(bn) >= 2)
    check("report join_axis recorded", bool(report.get("join_axis")))

    # --- phase (b): computed reconciliation ---
    comp = fout.get("computed_reconciliation") or {}
    check("computed reconciliation ran", comp.get("status") == "success")
    check("reconciliation script persisted", comp.get("script_path")
          and os.path.exists(comp["script_path"]))
    check("fusion_numerics.json persisted", comp.get("numerics_path")
          and os.path.exists(comp["numerics_path"]))
    q = (comp.get("results") or {}).get("quantities") or {}
    check("computed quantities non-empty", bool(q))
    check("sigma honesty recorded",
          "sigma_available" in (comp.get("results") or {}))
    check("COMPUTED RECONCILIATION block in synthesis prompt",
          "COMPUTED RECONCILIATION" in prompt)
    ver = comp.get("verification") or {}
    check("verification pass ran (phase c)",
          ver.get("verdict") in ("accept", "refine"))
    check("flagged result carries UNRESOLVED warning" if
          ver.get("verdict") == "refine" else "accepted result carries no warning",
          bool(comp.get("warning")) == (ver.get("verdict") == "refine"))
    print(f"\nAUDIT VERDICT: {ver.get('verdict')} "
          f"(attempts: {comp.get('attempts')})")
    for i in ver.get("issues") or []:
        print(f"  issue: {i}")
    if comp.get("figure_path"):
        print(f"\n(computed overlay figure: {comp['figure_path']})")

    print("\n--- COMPUTED QUANTITIES ---")
    print(json.dumps(comp.get("results"), indent=2, default=str)[:2500])
    print("\n--- RECONCILIATION SCRIPT STDOUT ---")
    print((comp.get("stdout") or "")[:1500])

    print("\n--- NUMERICS SECTIONS AS THE FUSION LLM SAW THEM ---")
    for chunk in prompt.split("### Dataset:")[1:]:
        head = chunk.splitlines()[0].strip()
        pos = chunk.find("Numerical results on disk")
        print(f"\n### {head}")
        print(chunk[pos:pos + 2000] if pos >= 0
              else "  (no numerics section)")

    print("\n--- FUSED NARRATIVE (first 2000 chars) ---")
    print(str(fout.get("detailed_analysis", ""))[:2000])
    print("\nreport json:", rp)
    print("report html:", fout.get("report_html_path"))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"FUSION NUMERICS LIVE (#296 phases a+b): {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
