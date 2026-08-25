"""Offline tests for dataset-aware metadata ownership (issue #411).

Defect 1: `current_metadata` was one session-global slot — a second dataset
with no sidecar of its own silently inherited the first technique's metadata.
Ownership now binds on first use by run_analysis; a different dataset must
re-resolve (sidecars -> stem backstop -> refuse, naming the previous owner).

Defect 2: plan validation could rewrite the physical model while carrying
over the previous technique's series regimes; the apply path is exercised
here with a mocked validator response.

  conda run -n scilink python tests/test_metadata_dataset_ownership.py
"""
import os
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import json
from pathlib import Path

import numpy as np

import scilink.agents.exp_agents.analysis_orchestrator_tools as aot
from scilink.agents.exp_agents.analysis_orchestrator_tools import (
    _dataset_key, _same_dataset,
)

results = {}
SENTINEL = "SENTINEL_REACHED_AGENT_CREATION"


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def make_orchestrator():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    bd = tempfile.mkdtemp()
    ag = AnalysisOrchestratorAgent(base_dir=bd, api_key="sk-dummy",
                                   model_name="claude-opus-4-6",
                                   restore_checkpoint=False,
                                   analysis_mode=AnalysisMode.AUTONOMOUS)

    def _sentinel(*a, **k):
        raise RuntimeError(SENTINEL)

    ag.create_agent_for_analysis = _sentinel
    ag.selected_agent_id = 1
    return ag


def run(ag, data_path, **kw):
    r = json.loads(ag.tools.functions_map["run_analysis"](
        data_path=data_path, agent_id=1, **kw))
    return r


def reached_agent(r):
    return r.get("status") == "error" and SENTINEL in (r.get("message") or "")


def main():
    # --- Fixture datasets -------------------------------------------------
    root = Path(tempfile.mkdtemp())
    dir_a = root / "xps_run"          # dataset A: dir, global metadata doc
    dir_a.mkdir()
    np.savetxt(dir_a / "xps.csv", np.random.rand(50, 2), delimiter=",")
    (dir_a / "metadata.json").write_text(json.dumps(
        {"technique": "XPS", "x_units": "eV"}))

    file_b = root / "epr_spectrum.csv"   # dataset B: bare file, NO sidecar
    np.savetxt(file_b, np.random.rand(50, 2), delimiter=",")

    dir_c = root / "ftir_series"      # dataset C: series with sidecars
    dir_c.mkdir()
    for t in (30, 40, 50):
        np.savetxt(dir_c / f"ftir_{t}C.txt", np.random.rand(20, 2))
        (dir_c / f"ftir_{t}C.json").write_text(
            json.dumps({"technique": "FTIR", "temperature_C": t}))

    file_d = root / "raman.csv"       # dataset D: file WITH stem sidecar
    np.savetxt(file_d, np.random.rand(50, 2), delimiter=",")
    (root / "raman.json").write_text(json.dumps({"technique": "Raman"}))

    # Avoid any LLM call from series sidecar extraction; return a resolved
    # control variable so the series run proceeds past the user prompt.
    aot._extract_series_from_sidecars = lambda *a, **k: (
        {"variable": "temperature_C", "values": [30, 40, 50], "units": "C"},
        {f"ftir_{t}C.txt": {"technique": "FTIR", "temperature_C": t}
         for t in (30, 40, 50)},
    )

    print("1) _dataset_key / _same_dataset semantics:")
    check("same file matches", _same_dataset(str(file_b), str(file_b)))
    check("dir covers member file", _same_dataset(str(dir_a), str(dir_a / "xps.csv")))
    check("dir covers glob within it",
          _same_dataset(str(dir_c), str(dir_c / "ftir_*.txt")))
    check("sibling file NOT covered by dir",
          not _same_dataset(str(dir_a), str(file_b)))
    check("file does NOT cover its parent dir",
          not _same_dataset(str(dir_a / "xps.csv"), str(dir_a)))
    check("glob covers a file it matches",
          _same_dataset(str(dir_c / "ftir_*.txt"), str(dir_c / "ftir_30C.txt")))
    check("glob does NOT cover non-matching file",
          not _same_dataset(str(dir_c / "ftir_*.txt"), str(file_b)))
    check("distinct globs over one dir are distinct datasets",
          not _same_dataset(str(dir_c / "ftir_3*.txt"), str(dir_c / "ftir_4*.txt")))

    print("2) explicit load binds on first use; different dataset refused:")
    ag = make_orchestrator()
    lm = json.loads(ag.tools.functions_map["load_metadata"](
        str(dir_a / "metadata.json")))
    check("metadata loaded", lm.get("status") in ("success", "warning"))
    check("fresh load is unbound", ag.current_metadata_owner is None)

    r = run(ag, str(dir_a))
    check("first run proceeds to agent creation", reached_agent(r))
    check("slot bound to dataset A", ag.current_metadata_owner == _dataset_key(str(dir_a)))

    r = run(ag, str(dir_a / "xps.csv"))
    check("drill-down into owner dir still allowed", reached_agent(r))

    r = run(ag, str(file_b))
    check("different dataset w/o sidecar is REFUSED", r.get("status") == "error"
          and "different dataset" in (r.get("message") or ""))
    check("refusal names the previous owner",
          _dataset_key(str(dir_a)) in (r.get("message") or ""))
    check("stale metadata not clobbered by refusal",
          (ag.current_metadata or {}).get("technique") == "XPS")

    print("3) stale slot displaced by the new dataset's own sidecars:")
    r = run(ag, str(dir_c))
    check("series with sidecars proceeds despite stale slot", reached_agent(r))
    check("stale XPS doc was cleared",
          (ag.current_metadata or {}).get("technique") != "XPS")
    check("slot rebound to dataset C",
          ag.current_metadata_owner == _dataset_key(str(dir_c)))

    r = run(ag, str(file_d))
    check("stem-sidecar file proceeds despite stale slot", reached_agent(r))
    check("stem sidecar replaced the stale doc",
          (ag.current_metadata or {}).get("technique") == "Raman")
    check("slot rebound to dataset D",
          ag.current_metadata_owner == _dataset_key(str(file_d)))

    print("4) fresh explicit load serves the NEXT dataset (bind-on-first-use):")
    json.loads(ag.tools.functions_map["load_metadata"](
        str(dir_a / "metadata.json")))
    check("re-load unbinds", ag.current_metadata_owner is None)
    r = run(ag, str(file_b))
    check("fresh load usable for new dataset", reached_agent(r))
    check("slot bound to dataset B",
          ag.current_metadata_owner == _dataset_key(str(file_b)))

    print("5) checkpoint round-trip and legacy backfill:")
    ag._auto_checkpoint()
    saved = json.loads(ag.checkpoint_path.read_text())
    check("owner persisted in checkpoint",
          saved.get("current_metadata_owner") == _dataset_key(str(file_b)))

    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    ag2 = AnalysisOrchestratorAgent(base_dir=str(ag.base_dir), api_key="sk-dummy",
                                    model_name="claude-opus-4-6",
                                    restore_checkpoint=True,
                                    analysis_mode=AnalysisMode.AUTONOMOUS)
    check("owner restored", ag2.current_metadata_owner == _dataset_key(str(file_b)))

    # Legacy checkpoint (no owner field): backfill from current_data_path.
    legacy = {k: v for k, v in saved.items() if k != "current_metadata_owner"}
    legacy["current_data_path"] = str(dir_a)
    ag.checkpoint_path.write_text(json.dumps(legacy))
    ag3 = AnalysisOrchestratorAgent(base_dir=str(ag.base_dir), api_key="sk-dummy",
                                    model_name="claude-opus-4-6",
                                    restore_checkpoint=True,
                                    analysis_mode=AnalysisMode.AUTONOMOUS)
    check("legacy checkpoint backfills owner from last data path",
          ag3.current_metadata_owner == str(dir_a))

    print("6) plan validation apply path re-derives series regimes (fix 2):")
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        CurveFittingPlanningController)
    from scilink.agents.exp_agents.instruct import (
        CURVE_FITTING_PLAN_VALIDATION_PROMPT)
    check("prompt states the re-derive principle",
          "regimes are invalid" in CURVE_FITTING_PLAN_VALIDATION_PROMPT
          and "re-derived series_analysis_plan" in CURVE_FITTING_PLAN_VALIDATION_PROMPT)

    corrected = {
        "valid": False,
        "issues": ["wrong technique: data is EPR, not XPS"],
        "physical_model": "EPR axial powder pattern (Lorentzian derivative)",
        "parameters_to_extract": ["g_parallel", "g_perp", "linewidth"],
        "fitting_strategy": "fit derivative lineshape per spectrum",
        "series_analysis_plan": {
            "regimes": [
                {"name": "low_T", "spectrum_indices": [0, 1, 2],
                 "physical_model": "axial powder pattern"},
                {"name": "high_T", "spectrum_indices": [3, 4, 5],
                 "physical_model": "isotropic Lorentzian"},
            ],
        },
    }

    class FakeModel:
        def generate_content(self, *a, **k):
            return corrected

    ctrl = CurveFittingPlanningController(
        model=FakeModel(), logger=aot.logging.getLogger("t"),
        generation_config=None, safety_settings=None,
        parse_fn=lambda resp: (resp, None),
        instructions="", output_dir=tempfile.mkdtemp())

    state = {
        "is_single_spectrum": False,
        "num_spectra": 6,
        "physical_model": "XPS doublet (Voigt)",
        "parameters_to_extract": ["binding_energy"],
        "fitting_strategy": "Voigt peaks with Shirley background",
        "analysis_approach": "peak fitting",
        "series_analysis_plan": {
            "regimes": [{"name": "all", "spectrum_indices": list(range(6)),
                         "physical_model": "XPS doublet (Voigt)"}],
        },
    }
    out = ctrl._validate_plan(dict(state))
    check("model correction applied",
          out["physical_model"].startswith("EPR"))
    new_regimes = (out.get("series_analysis_plan") or {}).get("regimes") or []
    check("series regimes re-derived (not carried over)",
          len(new_regimes) == 2
          and {r["name"] for r in new_regimes} == {"low_T", "high_T"})

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"METADATA OWNERSHIP (#411): {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
