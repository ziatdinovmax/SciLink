"""Live validation: a user crop request survives plan validation (Bedrock).

Scenario: an IR spectrum (full range ~374-4000 cm-1) with a user request to
crop to 600-1000 cm-1, routed through custom_processing_instruction — the
channel the orchestrator steers raw-data operations toward. The IR skill
auto-selects and injects the mandatory "fit the full measured range" rule;
pre-fix, the plan validator (which never saw the user note) reverted the
restricted fit domain to full range.

Modes (argv[1]):
  agent  — CurveFittingAgent directly with the instruction seeded in
           system_info (deterministic channel; the pre/post contrast run).
           Set SCILINK_CROP_REPO to a repo checkout to test (e.g. a main
           worktree for the pre-fix contrast) — it is prepended to sys.path.
  meta   — end-to-end MetaOrchestratorAgent session with the crop asked in
           natural language (the complaint's flow).

PASS = the fitted curve (spectrum_0000/fit.npy) spans only the requested
window, not the full spectrum.

    SCILINK_IR_FILE=/path/to/spectrum.csv \
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true python tests/test_crop_fitdomain_live.py agent
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

CROP_LO, CROP_HI = 600.0, 1000.0
TOL = 30.0  # cm-1 slack on the window edges
MODEL = "bedrock/us.anthropic.claude-opus-4-8"


def crop_check(out_dir: Path):
    """Was the fit restricted to the requested window?

    Judged from the EXECUTED plan and script (fit.npy is written by the
    ad-hoc generated script and is not reliable): the locked plan /
    reported model_type must name the window, the generated fitting
    script must contain both bounds, and the run must have succeeded.
    """
    import re

    def has_bounds(text):
        nums = [float(n) for n in re.findall(r"\b\d{3,4}(?:\.\d+)?\b", text)]
        return (any(abs(n - CROP_LO) <= TOL for n in nums)
                and any(abs(n - CROP_HI) <= TOL for n in nums))

    script_says_crop = any(
        has_bounds(sp.read_text()) for sp in out_dir.rglob("fitting_script.py"))

    centers = []
    status_ok = False
    model_type = None
    r_squared = None
    for srp in out_dir.rglob("series_fit_results.json"):
        sr = json.loads(srp.read_text())
        results = sr.get("results") or []
        status_ok = bool(results) and all(r.get("success") for r in results)
        for r in results:
            model_type = r.get("model_type") or model_type
            for p in (r.get("parameters") or {}).values():
                if isinstance(p, dict) and isinstance(
                        p.get("center"), (int, float)):
                    centers.append(float(p["center"]))
    for arp in out_dir.rglob("analysis_results.json"):
        ar = json.loads(arp.read_text())
        fq = ar.get("fit_quality") or {}
        r_squared = fq.get("r_squared")
    if not centers and not status_ok:
        return None
    # Execution-level truth: every fitted band lies inside the window.
    centers_inside = bool(centers) and all(
        CROP_LO - TOL <= c <= CROP_HI + TOL for c in centers)
    return {"script_says_crop": script_says_crop,
            "centers": sorted(round(c, 1) for c in centers),
            "centers_inside": centers_inside,
            "model_type": model_type,
            "status_ok": status_ok, "r_squared": r_squared}


def report(mode, chk):
    if not chk:
        print(f"\n[{mode}] no fit artifacts found")
        print(f"[{mode}] CROP HONORED: NO")
        return False
    ok = chk["centers_inside"] and chk["status_ok"] and chk["script_says_crop"]
    print(f"\n[{mode}] fitted centers: {chk['centers']}; all inside window: "
          f"{chk['centers_inside']}; script bounds: {chk['script_says_crop']}; "
          f"success: {chk['status_ok']}; R2: {chk['r_squared']}")
    print(f"[{mode}] model: {str(chk['model_type'])[:120]}")
    print(f"[{mode}] CROP HONORED: {'YES' if ok else 'NO'}")
    return ok


def run_agent_mode(work: Path, src_csv: Path) -> bool:
    repo = os.environ.get("SCILINK_CROP_REPO")
    if repo:
        sys.path.insert(0, repo)
    import scilink  # noqa: F401  (resolve after path injection)
    print(f"scilink from: {scilink.__file__}")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    out = work / "agent_out"
    agent = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(out),
        enable_human_feedback=False, max_verification_iterations=1,
    )
    res = agent.analyze(
        str(src_csv),
        system_info={
            "technique": "Infrared spectroscopy (FTIR)",
            "x_axis": "wavenumber (cm^-1)",
            "y_axis": "absorbance (a.u.)",
            "custom_processing_instruction": (
                f"crop the spectrum to {CROP_LO:.0f}-{CROP_HI:.0f} cm-1 "
                f"before fitting; the user only cares about this region"
            ),
        },
    )
    print("agent status:", res.get("status"))
    return report("agent", crop_check(out))


def run_meta_mode(work: Path, src_csv: Path) -> bool:
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)

    meta = MetaOrchestratorAgent(
        base_dir=str(work / "meta_session"), api_key=None,
        model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS,
    )
    reply = meta.chat(
        f"Analyze the IR spectrum at {src_csv} (curve fitting). "
        f"Crop the spectrum to {CROP_LO:.0f}-{CROP_HI:.0f} cm-1 before "
        f"fitting — I only care about that region. Smoke test: "
        f"max_verification_iterations=1, accept the first reasonable fit."
    )
    print("meta reply head:", reply[:200])
    return report("meta", crop_check(work / "meta_session"))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "agent"
    src = Path(os.environ.get(
        "SCILINK_IR_FILE",
        "/Users/maxim.ziatdinov/Code/benchmarking_for_paper2/RamanIR/"
        "IR_staged/01_easy_Calcite_R050127_1.csv"))
    if not src.exists():
        print(f"ERROR: IR file not found: {src}", file=sys.stderr)
        return 1
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("ERROR: AWS_BEARER_TOKEN_BEDROCK not set", file=sys.stderr)
        return 1
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    tag = os.environ.get("SCILINK_CROP_TAG", mode)
    work = Path(f"tests/_crop_fitdomain_live_runs/{tag}").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # Copy WITHOUT the sidecar: metadata comes from system_info / the chat,
    # keeping the crop instruction the only special context.
    csv = work / src.name
    shutil.copy(src, csv)
    sidecar = src.with_suffix(".json")
    if sidecar.exists() and mode == "meta":
        # meta needs the technique metadata channel; strip ground truth
        side = json.loads(sidecar.read_text())
        side.pop("ground_truth", None)
        (work / sidecar.name).write_text(json.dumps(side, indent=1))

    ok = run_agent_mode(work, csv) if mode == "agent" else run_meta_mode(work, csv)
    print(f"\nCROP FIT-DOMAIN LIVE [{mode}]: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
