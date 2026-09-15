#!/usr/bin/env python3
"""Live test for #173 — prior_analysis_paths support in the curve-fitting agent.

Runs CurveFittingAgent.analyze on one real spectrum with prior_analysis_paths
pointing at a prior curve-fit run, capturing every LLM prompt. Confirms:
  1. the run completes without error (no existing functionality broken), and
  2. the "Prior Curve-Fit Runs" reference block — incl. the prior fitting
     script — reached the planning / script-generation prompts.

Ad-hoc live test — NOT committed. Needs ANTHROPIC_API_KEY and, because curve
fitting executes generated code, UNSAFE_EXECUTION_OK=true.
"""
import sys
import tempfile

from scilink import auth

SPECTRUM = ("/Users/maxim.ziatdinov/Code/meta_session_20260517_172726/"
            "uploads/spectrum_T25.0_pH4.0.csv")
PRIOR_RUN = ("/Users/maxim.ziatdinov/Code/meta_session_20260517_172726/"
             "analysis/results/analysis_uploads_CurveFit_20260517_172831_001")


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(_flatten(x) for x in content)
    if isinstance(content, dict):
        return "" if content.get("mime_type") else str(content)
    return str(content)


def main() -> int:
    model_name = "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment.")
        return 2

    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    out = tempfile.mkdtemp(prefix="cf173_")
    print(f"output dir : {out}")
    agent = CurveFittingAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
        run_preprocessing=False,
    )

    # Capture every LLM prompt the run sends.
    captured = []
    orig = agent.model.generate_content

    def _spy(*args, **kwargs):
        contents = kwargs.get("contents")
        if contents is None and args:
            contents = args[0]
        captured.append(contents)
        return orig(*args, **kwargs)

    agent.model.generate_content = _spy

    print("Running curve-fit analyze WITH prior_analysis_paths ...\n")
    raised = None
    try:
        result = agent.analyze(
            data=SPECTRUM,
            prior_analysis_paths=[PRIOR_RUN],
            system_info="A 1-D intensity-vs-position measurement curve.",
        )
    except Exception as e:  # noqa: BLE001
        raised = e
        result = None

    print()
    if raised is not None:
        print(f"run RAISED: {type(raised).__name__}: {raised}")
    else:
        status = result.get("status") if isinstance(result, dict) else result
        print(f"run completed; status = {status!r}")

    all_text = "\n".join(_flatten(c) for c in captured)
    block_reached = "Prior Curve-Fit Runs" in all_text
    script_reached = "Saved fitting script" in all_text
    print(f"prompts captured                  : {len(captured)}")
    print(f"'Prior Curve-Fit Runs' in a prompt : {block_reached}")
    print(f"prior fitting script in a prompt   : {script_reached}")

    ok = raised is None and block_reached and script_reached
    print()
    print("RESULT:", "PASS — prior artifacts reached the prompts, run completed"
          if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
