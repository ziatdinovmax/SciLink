"""Live regression for #326 — distinct datasets sharing ONE directory.

In-situ / multi-technique studies routinely land several DIFFERENT datasets in
a single upload folder: one file per technique plus one or more file *series*,
distinguished only by their filename pattern. The fan-out used to identify a
branch by its `data_path`, so those branches collapsed — one was run twice and
the other silently dropped — and, even once distinct, a directory-scoped branch
swallowed its siblings' files.

This test drives `delegate_to_analyses` over exactly that layout and asserts:
  * every branch runs (no collapse, no duplication),
  * each series branch carries its file pattern and analyzes ONLY its own files,
  * the branches fuse into one complementarity-gated cross-dataset result.

Bring your own data: a directory holding one standalone data file plus two
file series, each series selected by a glob.

  export AWS_BEARER_TOKEN_BEDROCK=...   AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true conda run -n scilink python \
      tests/test_meta_fanout_same_dir_live.py \
      --uploads /path/to/uploads \
      --single  'thermal_curve.txt'      --single-label 'thermal curve' \
      --series  'spectra_*C.txt'          --series-label 'spectroscopy series' \
      --series  'diffraction_*.txt'       --series-label 'diffraction series' \
      --subject 'the sample under study'
"""
import argparse
import glob
import json
import os
import sys
import tempfile

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")


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
    ap.add_argument("--uploads", required=True,
                    help="Directory holding ALL the datasets (one standalone "
                         "file + >=2 file series).")
    ap.add_argument("--single", required=True,
                    help="Filename of the standalone data file inside --uploads.")
    ap.add_argument("--single-label", default="standalone dataset")
    ap.add_argument("--series", action="append", required=True,
                    help="Glob selecting ONE series inside --uploads (repeatable).")
    ap.add_argument("--series-label", action="append", required=True,
                    help="Label for the corresponding --series (repeatable).")
    ap.add_argument("--subject", default="one physical system",
                    help="What the datasets measure (used in the branch tasks).")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    if len(args.series) != len(args.series_label):
        print("Each --series needs a matching --series-label."); sys.exit(2)
    if len(args.series) < 2:
        print("Need >= 2 series to exercise the same-directory collapse."); sys.exit(2)

    up = os.path.abspath(os.path.expanduser(args.uploads))
    if not os.path.isfile(os.path.join(up, args.single)):
        print(f"--single file not found under {up}"); sys.exit(2)
    expected = {}
    for pat in args.series:
        n = len([f for f in glob.glob(os.path.join(up, pat)) if os.path.isfile(f)])
        if n == 0:
            print(f"pattern {pat!r} matches no files under {up}"); sys.exit(2)
        expected[pat] = n
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)

    base = args.base_dir or tempfile.mkdtemp(prefix="fanout_same_dir_")
    print(f"session dir: {base}")
    print(f"series file counts: {expected}")

    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)

    checks = {}

    def check(name, cond):
        checks[name] = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    n_expected = 1 + len(args.series)
    print(f"\n### FAN-OUT: {n_expected} datasets, {len(args.series)} of them "
          "series SHARING one directory ###")
    out = json.loads(ag._run_fanout(_branches(up, args)))
    print("FANOUT RESULT:", json.dumps(
        {k: out.get(k) for k in ("status", "reason", "join_axis",
                                 "branches_run", "branches_with_output")},
        indent=2))
    if out.get("status") != "success":
        print("verdict:", json.dumps(out.get("verdict", {}), indent=2)[:1200])

    check("fan-out ran (status=success)", out.get("status") == "success")
    check(f"{n_expected} distinct branches ran (no collapse/duplication)",
          out.get("branches_run") == n_expected)
    labels = sorted(r.get("label", "") for r in out.get("results", []))
    check("labels are the distinct datasets",
          labels == sorted([args.single_label] + list(args.series_label)))
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    norm = {" ".join((e.get("task") or "").split()).lower() for e in fan}
    check("no duplicated branch task in ledger", len(norm) == len(fan))
    check("series branches carry their file patterns (#326 Fix 3)",
          sorted(str(e.get("pattern")) for e in fan) ==
          sorted(["None"] + list(args.series)))

    productive = [r["delegation_index"] for r in out.get("results", [])
                  if r.get("produced_output")]
    check("all branches produced output", len(productive) == n_expected)

    if len(productive) >= 2:
        print("\n### FUSION over all productive branches ###")
        fout = json.loads(ag._fuse_delegations(
            productive,
            focus="Do the datasets agree on where the transition occurs along "
                  "the study's shared control variable?"))
        check("fusion succeeded", fout.get("status") == "success")
        check("fusion was complementarity-gated",
              bool(fout.get("complementarity_gated")))
        print("FUSED NARRATIVE (first 1200 chars):\n",
              str(fout.get("detailed_analysis", ""))[:1200])
        print("\nreport:", fout.get("report_html_path"))
    else:
        check("fusion succeeded", False)

    print("\nNOTE: confirm in the log that each series branch reports "
          "\"Pattern '<glob>' matched N data file(s)\" with N equal to its own "
          "file count above — NOT the directory total. That is the scoping fix.")

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"SAME-DIR FAN-OUT LIVE (#326): {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
