#!/usr/bin/env python3
"""Live test for #172 — locked extraction-script reuse (curve-fit slice).

Runs CurveFittingAgent.analyze on one real spectrum and checks the three
reuse outcomes:

  A. WITH prior_analysis_paths, default threshold -> the prior run's locked
     fitting script is reused, fits well, and the QC verification loop is
     skipped. result['reuse_validity']['verdict'] == 'good'.
  B. WITH prior_analysis_paths, r2_threshold forced to 0.999 -> the same
     reused fit now scores below threshold -> verdict 'poor', a quality
     warning is attached, but the result is still kept (schema-consistent).
  C. WITHOUT prior_analysis_paths -> no reuse attempted, no reuse_validity,
     behaviour byte-identical to a normal run.

Ad-hoc live test — NOT committed. Needs ANTHROPIC_API_KEY and, because curve
fitting executes generated code, UNSAFE_EXECUTION_OK=true.
"""
import io
import logging
import sys
import tempfile

from scilink import auth

SPECTRUM = ("/Users/maxim.ziatdinov/Code/meta_session_20260517_172726/"
            "uploads/spectrum_T25.0_pH4.0.csv")
PRIOR_RUN = ("/Users/maxim.ziatdinov/Code/meta_session_20260517_172726/"
             "analysis/results/analysis_uploads_CurveFit_20260517_172831_001")
SYS_INFO = "A 1-D intensity-vs-position measurement curve."


def _run(model_name, api_key, *, prior, r2_threshold=None):
    """One analyze() run. Returns (result, captured_log_text, raised)."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    out = tempfile.mkdtemp(prefix="cf172_")
    agent = CurveFittingAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
        run_preprocessing=False,
    )

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    cf_logger = logging.getLogger("CurveFittingAgent")
    cf_logger.addHandler(handler)

    kwargs = dict(data=SPECTRUM, system_info=SYS_INFO)
    if prior:
        kwargs["prior_analysis_paths"] = [PRIOR_RUN]
        # #172 verbatim reuse is now an explicit opt-in (default = agent-judged).
        kwargs["reuse_locked_script"] = True
    if r2_threshold is not None:
        kwargs["r2_threshold"] = r2_threshold

    raised = None
    result = None
    try:
        result = agent.analyze(**kwargs)
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        cf_logger.removeHandler(handler)
    return result, buf.getvalue(), raised


def main() -> int:
    model_name = "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment.")
        return 2

    checks = []

    # --- A: reuse, good verdict --------------------------------------
    print("=== A: WITH prior_analysis_paths, default threshold ===")
    res, log, raised = _run(model_name, api_key, prior=True)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        rv = (res or {}).get("reuse_validity") or {}
        reused_msg = "Reusing locked fitting script" in log
        skipped_qc = "Verification 1/" not in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  reuse log line    : {reused_msg}")
        print(f"  reuse_validity    : {rv}")
        print(f"  QC loop skipped   : {skipped_qc}")
        ok_a = (
            res.get("status") == "success"
            and reused_msg
            and rv.get("reused") is True
            and rv.get("verdict") == "good"
            and skipped_qc
        )
        checks.append(ok_a)
        print(f"  -> {'PASS' if ok_a else 'CHECK'}")

    # --- B: reuse, poor verdict (threshold forced to 0.999) ----------
    print("\n=== B: WITH prior_analysis_paths, r2_threshold=0.999 ===")
    res, log, raised = _run(model_name, api_key, prior=True, r2_threshold=0.999)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        rv = (res or {}).get("reuse_validity") or {}
        poor_msg = "Reused script fits poorly" in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  reuse_validity    : {rv}")
        print(f"  poor log line     : {poor_msg}")
        print(f"  quality_warning   : {bool(res.get('quality_warning'))}")
        ok_b = (
            res.get("status") == "success"
            and rv.get("reused") is True
            and rv.get("verdict") == "poor"
            and rv.get("r_squared") is not None
            and bool(res.get("quality_warning"))
        )
        checks.append(ok_b)
        print(f"  -> {'PASS' if ok_b else 'CHECK'}")

    # --- C: no prior -> no reuse, baseline unchanged -----------------
    print("\n=== C: WITHOUT prior_analysis_paths (baseline) ===")
    res, log, raised = _run(model_name, api_key, prior=False)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        no_reuse_field = (res or {}).get("reuse_validity") is None
        no_reuse_log = "Reusing locked fitting script" not in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  no reuse_validity : {no_reuse_field}")
        print(f"  no reuse log line : {no_reuse_log}")
        ok_c = (
            res.get("status") == "success"
            and no_reuse_field
            and no_reuse_log
        )
        checks.append(ok_c)
        print(f"  -> {'PASS' if ok_c else 'CHECK'}")

    ok = all(checks) and len(checks) == 3
    print()
    print("RESULT:", "PASS — locked-script reuse works; baseline unaffected"
          if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
