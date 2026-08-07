"""Live validation for script_edits (Phase A of the surgical-refinement
plan) on Bedrock Opus 4.8.

  1. wiring   — agent-level, deterministic: a real anchor fit, then a
                reuse run with a test-constructed surgical edit. The
                rerun's saved script must equal the anchor's byte-for-
                byte EXCEPT the edit; provenance recorded; reuse
                validity attached.
  2. routing  — orchestrator chat: "next point of the same series, but
                change one knob". The LLM must read the saved script,
                pass script_edits with a verbatim snippet, and the two
                runs' scripts must differ in only a few lines — not a
                re-derivation, not an unedited reuse.

Needs AWS_BEARER_TOKEN_BEDROCK (+ UNSAFE_EXECUTION_OK=true: curve
fitting executes generated code).

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true python tests/test_script_edits_live.py [1 2]
"""
from __future__ import annotations

import contextlib
import difflib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_script_edits_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


@contextlib.contextmanager
def capture_all(buf):
    """Agents narrate via logging, orchestrators via print — capture both."""
    import logging
    handler = logging.StreamHandler(buf)
    logging.getLogger().addHandler(handler)
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        logging.getLogger().removeHandler(handler)


def make_spectrum(path, center, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 400)
    y = (2.0 * np.exp(-((x - center) ** 2) / (2 * 0.4 ** 2))
         + 0.3 + 0.02 * x + rng.normal(0, 0.02, x.size))
    np.savetxt(path, np.column_stack([x, y]), delimiter=",",
               header="position,intensity", comments="")


SYS_INFO = {"technique": "generic 1D spectroscopy",
            "x_axis": "position (a.u.)", "y_axis": "intensity (a.u.)"}


def _saved_script(run_dir: Path) -> tuple[Path, str]:
    scripts = sorted((Path(run_dir) / "scripts").glob("*.py"))
    assert scripts, f"no saved script under {run_dir}/scripts"
    return scripts[0], scripts[0].read_text()


def part1_wiring():
    print("\n=== 1. agent-level: byte-exact except the edit ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p1"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    spec_a, spec_b = run / "spec_a.csv", run / "spec_b.csv"
    make_spectrum(spec_a, center=5.0, seed=1)
    make_spectrum(spec_b, center=5.1, seed=2)

    agent_a = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "anchor"),
        enable_human_feedback=False, max_verification_iterations=1)
    res_a = agent_a.analyze(str(spec_a), system_info=SYS_INFO)
    check("p1 anchor fit succeeded", res_a.get("status") == "success")
    _, script_a = _saved_script(run / "anchor")

    # A guaranteed-present, behavior-neutral snippet: the numpy import.
    # Part 1 validates the MECHANISM (exact application, provenance,
    # execution); part 2 exercises a real knob chosen by the LLM.
    assert "import numpy as np" in script_a
    edit = {"old_text": "import numpy as np",
            "new_text": "import numpy as np  # SURGICAL-EDIT-MARK"}

    agent_b = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "rerun"),
        enable_human_feedback=False, max_verification_iterations=1)
    buf = Tee()
    with capture_all(buf):
        res_b = agent_b.analyze(
            str(spec_b), system_info=SYS_INFO,
            prior_analysis_paths=[str(run / "anchor")],
            reuse_locked_script=True, script_edits=[edit])
    log = buf.getvalue()

    check("p1 rerun succeeded", res_b.get("status") == "success")
    check("p1 edit applied (log)", "Applied 1 surgical edit" in log)
    check("p1 reuse fast path used (not re-derivation)",
          "Reusing locked fitting script" in log)
    _, script_b = _saved_script(run / "rerun")
    check("p1 mark present in executed script",
          "SURGICAL-EDIT-MARK" in script_b)
    check("p1 byte-identical except the edit",
          script_b.replace("  # SURGICAL-EDIT-MARK", "") == script_a)
    sfr = json.loads((run / "rerun" / "series_fit_results.json").read_text())
    r0 = (sfr.get("results") or [{}])[0]
    rv = r0.get("reuse_validity") or {}
    check("p1 provenance: source labeled with the edit count",
          "+ 1 edit(s)" in str(rv.get("source")))
    check("p1 provenance: edits recorded in the result",
          r0.get("script_edits_applied") == [edit])


def part2_routing():
    print("\n=== 2. orchestrator: single-knob follow-up routes to edits ===")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)

    run = BASE / "p2"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    spec_a, spec_b = run / "spec_a.csv", run / "spec_b.csv"
    make_spectrum(spec_a, center=5.0, seed=3)
    make_spectrum(spec_b, center=5.15, seed=4)

    orch = AnalysisOrchestratorAgent(
        base_dir=str(run / "session"), api_key=None, model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS)

    buf = Tee()
    with capture_all(buf):
        orch.chat(
            f"Fit the 1D spectrum at {spec_a} (single Gaussian peak on a "
            "linear background). Smoke test: max_verification_iterations=1, "
            "accept the first reasonable fit.")
        reply = orch.chat(
            f"{spec_b} is the NEXT measurement of the same series. Re-fit "
            "it reusing the locked script so the feature columns stay "
            "identical — but with ONE surgical change: read the saved "
            "fitting script first, then modify a single numeric value in "
            "it (an initial guess, tolerance, or iteration limit of your "
            "choice). Tell me exactly what you changed.")
    log = buf.getvalue()

    check("p2 edit path used (log)", "Applied 1 surgical edit" in log
          or "surgical edit" in log)
    check("p2 reuse fast path used", "Reusing locked fitting script" in log)

    session = run / "session"
    run_dirs = sorted(d for d in (session / "results").glob("*CurveFit*")
                      if d.is_dir()) if (session / "results").is_dir() else \
        sorted(d for d in session.rglob("series_fit_results.json"))
    # locate the two runs' scripts wherever the session layout put them
    sfrs = sorted(session.rglob("series_fit_results.json"))
    check("p2 two fit runs present", len(sfrs) >= 2)
    if len(sfrs) >= 2:
        s1 = _saved_script(sfrs[0].parent)[1]
        s2 = _saved_script(sfrs[-1].parent)[1]
        ndiff = sum(1 for l in difflib.unified_diff(
            s1.splitlines(), s2.splitlines(), lineterm="")
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
        print(f"     script diff: {ndiff} changed line(s)")
        check("p2 scripts differ minimally (edit, not re-derivation)",
              1 <= ndiff <= 6)
        rec = json.loads(sfrs[-1].read_text())
        r0 = (rec.get("results") or [{}])[0]
        check("p2 provenance recorded", bool(r0.get("script_edits_applied")))


PARTS = {"1": part1_wiring, "2": part2_routing}

if __name__ == "__main__":
    for k in (sys.argv[1:] or sorted(PARTS)):
        PARTS[k]()
    print("\n" + "=" * 60)
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
