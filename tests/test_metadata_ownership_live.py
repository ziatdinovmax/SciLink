"""Live validation for issue #411 fixes (Bedrock Opus 4.8).

Part A (defect 2, one model call): a wrong-technique plan (XPS, one regime)
is validated against an EPR-looking series overlay. The validator must not
only correct the physical model but return a RE-DERIVED series_analysis_plan
instead of carrying the XPS regime over.

Part B (defects 1+3, two-turn orchestrator chat): dataset A (XPS dir with an
explicitly loaded metadata doc) is analyzed; then dataset B (EPR-like file,
NO sidecar) is requested with no technique info. run_analysis must REFUSE to
reuse A's metadata (naming A), the LLM must recover by creating metadata for
B, and the slot must end bound to B with non-XPS content.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true \
    python tests/test_metadata_ownership_live.py
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_metadata_ownership_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _epr_deriv(B, amp, B0, w):
    """Derivative-of-Lorentzian EPR lineshape."""
    hw = w / 2.0
    return -2.0 * amp * hw ** 2 * (B - B0) / ((B - B0) ** 2 + hw ** 2) ** 2


def make_xps_dataset(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    x = np.linspace(282.0, 290.0, 401)
    y = (_gauss(x, 1.0, 284.6, 0.45) + _gauss(x, 0.55, 286.5, 0.5)
         + 0.05 + rng.normal(0, 0.005, x.size))
    np.savetxt(d / "c1s.csv", np.column_stack([x, y]), delimiter=",",
               header="binding_energy_eV,intensity", comments="")
    (d / "metadata.json").write_text(json.dumps({
        "experiment": {"technique": "XPS", "edge": "C 1s",
                       "x_units": "eV binding energy"},
        "sample": {"material": "synthetic carbon film"},
        "instrument": {"name": "test_instrument"},
    }, indent=2))


def make_epr_file(path: Path) -> None:
    rng = np.random.default_rng(7)
    B = np.linspace(320.0, 340.0, 501)
    y = (_epr_deriv(B, 1.0, 329.0, 1.2) + _epr_deriv(B, 0.6, 331.5, 1.8)
         + rng.normal(0, 0.004, B.size))
    np.savetxt(path, np.column_stack([B, y]), delimiter=",",
               header="field_mT,dI_dB", comments="")


def make_epr_series_overlay() -> bytes:
    """6-spectrum EPR series with a visible regime change at index 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = np.linspace(320.0, 340.0, 501)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(6):
        if i < 3:  # two overlapping derivative features
            y = (_epr_deriv(B, 1.0, 328.5 + 0.2 * i, 1.2)
                 + _epr_deriv(B, 0.7, 331.5 + 0.2 * i, 1.6))
        else:      # collapsed to one narrow isotropic line
            y = _epr_deriv(B, 1.4, 330.0 + 0.1 * i, 0.9)
        ax.plot(B, y + 0.6 * i, label=f"spectrum {i} (T={20 + 15 * i}K)")
    ax.set_xlabel("x")
    ax.set_ylabel("signal (offset for clarity)")
    ax.legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def part_a() -> None:
    print("\n=== PART A: validator re-derives regimes on technique change ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        CurveFittingPlanningController)

    agent = CurveFittingAgent(
        api_key=None, model_name=MODEL,
        output_dir=str(BASE / "part_a_out"),
        enable_human_feedback=False, run_preprocessing=False,
    )
    ctrl = CurveFittingPlanningController(
        model=agent.model, logger=logging.getLogger("live_a"),
        generation_config=agent.generation_config,
        safety_settings=getattr(agent, "safety_settings", None),
        parse_fn=agent._parse_llm_response,
        instructions="", output_dir=str(BASE / "part_a_out"),
    )

    xps_regime = {
        "name": "single_phase",
        "spectrum_indices": list(range(6)),
        "physical_model": "Two Voigt components (C 1s / shifted C 1s) on a "
                          "Shirley background",
        "fitting_strategy": "Voigt doublet, Shirley background per spectrum",
    }
    state = {
        "is_single_spectrum": False,
        "num_spectra": 6,
        "scout_data": [0, 2, 5],
        "analysis_approach": "XPS core-level peak fitting",
        "physical_model": "Two Voigt components on a Shirley background "
                          "(XPS C 1s doublet)",
        "parameters_to_extract": ["binding_energy", "fwhm", "area_ratio"],
        "fitting_strategy": "Fit Voigt doublet with Shirley background to "
                            "every spectrum in the series",
        "series_analysis_plan": {"regimes": [dict(xps_regime)]},
        "scout_overlay_plot": make_epr_series_overlay(),
    }

    out = ctrl._validate_plan(dict(state))
    model_txt = (out.get("physical_model") or "").lower()
    plan = out.get("series_analysis_plan") or {}
    regimes = plan.get("regimes") or []
    regime_txt = json.dumps(regimes).lower()

    (BASE / "part_a_out").mkdir(parents=True, exist_ok=True)
    (BASE / "part_a_out" / "validated_state.json").write_text(json.dumps({
        "physical_model": out.get("physical_model"),
        "parameters_to_extract": out.get("parameters_to_extract"),
        "fitting_strategy": out.get("fitting_strategy"),
        "series_analysis_plan": plan,
    }, indent=2, default=str))

    check("A1: model corrected away from XPS/Voigt+Shirley",
          "shirley" not in model_txt
          and any(k in model_txt for k in
                  ("derivative", "epr", "lorentzian", "resonance", "dysonian")))
    check("A2: series plan returned by the validator (not carried over)",
          bool(regimes)
          and regime_txt != json.dumps([xps_regime]).lower())
    regime_models = " ".join(
        (r.get("physical_model") or "").lower() for r in regimes)
    check("A3: regime physical_model re-derived (EPR-appropriate, "
          "no placeholder, no Shirley)",
          bool(regimes) and "shirley" not in regime_txt
          and "to be determined" not in regime_models
          and any(k in regime_models for k in
                  ("derivative", "epr", "lorentzian", "resonance")))
    print(f"  validated model: {out.get('physical_model')}")
    print(f"  regimes: {[r.get('name') for r in regimes]}")


def part_b() -> None:
    print("\n=== PART B: two-dataset orchestrator session ===")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        _dataset_key)

    data = BASE / "data"
    dir_a = data / "xps_run"
    make_xps_dataset(dir_a)
    file_b = data / "epr_spectrum.csv"
    make_epr_file(file_b)

    log_buf = io.StringIO()
    capture = logging.StreamHandler(log_buf)
    capture.setLevel(logging.INFO)
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    try:
        orch = AnalysisOrchestratorAgent(
            base_dir=str(BASE / "session"),
            api_key=None, model_name=MODEL,
            analysis_mode=AnalysisMode.AUTONOMOUS,
        )

        p1 = (f"Load the metadata file {dir_a / 'metadata.json'}, then analyze "
              f"the XPS spectrum in {dir_a} with the curve fitting agent. "
              f"Two Gaussian-like peaks near 284.6 and 286.5 eV on a flat "
              f"background. This is a smoke test: use "
              f"max_verification_iterations=1 and accept the first "
              f"reasonable fit.")
        print(f"\n>>> turn 1: {p1}\n")
        t0 = time.perf_counter()
        r1 = orch.chat(p1)
        print(f"<<< turn 1 done in {time.perf_counter() - t0:.0f}s: {r1[:200]}")

        n_after_1 = len(orch.analysis_results)
        owner_after_1 = orch.current_metadata_owner

        p2 = (f"Now analyze the spectrum at {file_b} with the curve fitting "
              f"agent. Same smoke-test settings "
              f"(max_verification_iterations=1). Do not ask me questions; "
              f"proceed autonomously.")
        print(f"\n>>> turn 2: {p2}\n")
        t0 = time.perf_counter()
        r2 = orch.chat(p2)
        print(f"<<< turn 2 done in {time.perf_counter() - t0:.0f}s: {r2[:200]}")
    finally:
        logging.getLogger().removeHandler(capture)

    transcript = json.dumps(orch.messages, default=str)
    log_text = log_buf.getvalue()
    (BASE / "session" / "captured.log").write_text(log_text)

    check("B1: first analysis completed", n_after_1 >= 1)
    check("B2: metadata bound to dataset A after turn 1",
          owner_after_1 is not None
          and _same_or_sub(owner_after_1, dir_a))
    check("B3: stale reuse REFUSED for dataset B (error names dataset A)",
          "belongs to a different dataset" in transcript)
    check("B4: LLM recovered and completed the second analysis",
          len(orch.analysis_results) >= n_after_1 + 1
          and (orch.analysis_results[-1].get("status") in
               (None, "success", "completed", "success_with_warnings")))
    md_txt = json.dumps(orch.current_metadata or {}).lower()
    check("B5: final metadata is NOT dataset A's XPS document",
          "c 1s" not in md_txt and "carbon film" not in md_txt)
    check("B6: slot ends bound to dataset B",
          orch.current_metadata_owner == _dataset_key(str(file_b)))

    # Soft report: which skill(s) each run selected (defect 3 signal).
    for line in log_text.splitlines():
        if "skill" in line.lower() and ("select" in line.lower()
                                        or "loaded" in line.lower()):
            print("  [skill-log]", line.strip()[:160])


def _same_or_sub(owner: str, d: Path) -> bool:
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        _same_dataset)
    return _same_dataset(owner, str(d)) or _same_dataset(str(d), owner)


def main() -> int:
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("ERROR: AWS_BEARER_TOKEN_BEDROCK not set", file=sys.stderr)
        return 1
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only is None and BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True, exist_ok=True)

    if only in (None, "a"):
        part_a()
    if only in (None, "b"):
        part_b()

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"METADATA OWNERSHIP LIVE (#411): {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    return 0 if npass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
