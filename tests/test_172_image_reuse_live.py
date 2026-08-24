#!/usr/bin/env python3
"""Live test for #172 — locked extraction-script reuse (image slice).

Runs ImageAnalysisAgent.analyze on one real image and checks:

  A. WITH prior_analysis_paths -> the prior run's locked analysis script is
     reused for the anchor, a single vision-verification pass yields a
     reuse_validity verdict, and the iterative QC loop is skipped.
  B. WITHOUT prior_analysis_paths -> no reuse attempted, no reuse_validity,
     behaviour byte-identical to a normal run.

Ad-hoc live test — NOT committed. Needs ANTHROPIC_API_KEY and, because image
analysis executes generated code, UNSAFE_EXECUTION_OK=true.
"""
import io
import logging
import sys
import tempfile

from scilink import auth

IMAGE = ("/Users/maxim.ziatdinov/Code/SciLink/analysis_session_20260506_174304/"
         "uploads/YBCO_tem.tif")
PRIOR_RUN = ("/Users/maxim.ziatdinov/Code/SciLink/analysis_session_20260506_174304/"
             "results/analysis_YBCO_tem_ImageAnalysis_20260506_174450_001")
SYS_INFO = "A microscopy image."


def _run(model_name, api_key, *, prior):
    """One analyze() run. Returns (result, captured_log_text, raised)."""
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    out = tempfile.mkdtemp(prefix="img172_")
    agent = ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
    )

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)

    kwargs = dict(data=IMAGE, system_info=SYS_INFO)
    if prior:
        kwargs["prior_analysis_paths"] = [PRIOR_RUN]
        # #172 verbatim reuse is now an explicit opt-in (default = agent-judged).
        kwargs["reuse_locked_script"] = True

    raised = None
    result = None
    try:
        result = agent.analyze(**kwargs)
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        root.removeHandler(handler)
    return result, buf.getvalue(), raised


def main() -> int:
    model_name = "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment.")
        return 2

    checks = []

    # --- A: reuse -----------------------------------------------------
    print("=== A: WITH prior_analysis_paths ===")
    res, log, raised = _run(model_name, api_key, prior=True)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        rv = (res or {}).get("reuse_validity") or {}
        reused_msg = "Reusing locked analysis script" in log
        skipped_qc = "Verification 1/" not in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  reuse log line    : {reused_msg}")
        print(f"  reuse_validity    : {rv}")
        print(f"  QC loop skipped   : {skipped_qc}")
        ok_a = (
            res.get("status") == "success"
            and reused_msg
            and rv.get("reused") is True
            and rv.get("verdict") in ("good", "poor")
            and rv.get("quality_score") is not None
            and skipped_qc
        )
        checks.append(ok_a)
        print(f"  -> {'PASS' if ok_a else 'CHECK'}")

    # --- B: no prior -> baseline unchanged ---------------------------
    print("\n=== B: WITHOUT prior_analysis_paths (baseline) ===")
    res, log, raised = _run(model_name, api_key, prior=False)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        no_reuse_field = (res or {}).get("reuse_validity") is None
        no_reuse_log = "Reusing locked analysis script" not in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  no reuse_validity : {no_reuse_field}")
        print(f"  no reuse log line : {no_reuse_log}")
        ok_b = (
            res.get("status") == "success"
            and no_reuse_field
            and no_reuse_log
        )
        checks.append(ok_b)
        print(f"  -> {'PASS' if ok_b else 'CHECK'}")

    ok = all(checks) and len(checks) == 2
    print()
    print("RESULT:", "PASS — image locked-script reuse works; baseline "
          "unaffected" if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
