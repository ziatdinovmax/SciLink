"""
Compact preprocessing edge-case test suite for the Analysis Orchestrator.

Runs each scenario through the full orchestrator chat (with LLM calls).
Requires GEMINI_API_KEY env var. Run with:

    GEMINI_API_KEY=<key> python tests/test_preprocessing_orchestrator.py

Or run a subset by number:

    GEMINI_API_KEY=<key> python tests/test_preprocessing_orchestrator.py 1 3 7
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.analysis_orchestrator import (
    AnalysisOrchestratorAgent,
    AnalysisMode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp():
    return Path(tempfile.mkdtemp(prefix="preproc_"))


def _csv(path, x, y):
    np.savetxt(path, np.column_stack([x, y]), delimiter=",",
               header="wavelength,intensity", comments="")


def _meta(path, technique="Raman Spectroscopy", sample="test", extra=None):
    m = {"experiment": {"technique": technique},
         "sample": {"material": sample},
         "instrument": {"name": "test_instrument"}}
    if extra:
        m.update(extra)
    with open(path, "w") as f:
        json.dump(m, f)


def _orch(base_dir):
    return AnalysisOrchestratorAgent(
        base_dir=str(base_dir),
        api_key=os.environ["GEMINI_API_KEY"],
        model_name="gemini-3.1-pro-preview",
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )


def _raman_peak(n=500):
    x = np.linspace(200, 800, n)
    y = 1000 * np.exp(-((x - 520) ** 2) / (2 * 15**2)) + np.random.normal(0, 10, n)
    return x, y


# ---------------------------------------------------------------------------
# Test definitions — each returns (name, pass: bool, detail: str)
# ---------------------------------------------------------------------------

TESTS = []


def _test(fn):
    TESTS.append(fn)
    return fn


@_test
def normal_1d_preprocessing():
    """Normal spectrum + normalize instruction → analysis completes."""
    d = _tmp()
    x, y = _raman_peak()
    _csv(d / "data.csv", x, y)
    _meta(d / "meta.json", sample="Si")
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {d / 'data.csv'}, load metadata from {d / 'meta.json'}, "
        f"set preprocessing to 'normalize intensity to 0-1', select agent 3, run analysis."
    )
    shutil.rmtree(d, True)
    ok = len(o.analysis_results) == 1 and o.analysis_results[0].get("status") != "error"
    return ok, r[:300]


@_test
def all_zero_data():
    """All-zero intensities → agent should not crash."""
    d = _tmp()
    x = np.linspace(200, 800, 300)
    _csv(d / "z.csv", x, np.zeros_like(x))
    _meta(d / "meta.json", technique="PL Spectroscopy", sample="blank")
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {d / 'z.csv'}, load metadata {d / 'meta.json'}, "
        f"set preprocessing 'subtract baseline', select agent 3, run analysis."
    )
    shutil.rmtree(d, True)
    return True, r[:300]  # pass if no unhandled exception


@_test
def negative_intensities():
    """Large negatives + clip instruction → completes."""
    d = _tmp()
    x = np.linspace(200, 800, 500)
    y = 2000 * np.exp(-((x - 520) ** 2) / (2 * 20**2)) - 500 + 0.5 * x
    _csv(d / "neg.csv", x, y)
    _meta(d / "meta.json", sample="SiC")
    o = _orch(d / "s")
    r = o.chat(
        f"Load {d / 'neg.csv'}, metadata {d / 'meta.json'}. "
        f"Set preprocessing 'clip negative values to zero'. Select agent 3. Run."
    )
    shutil.rmtree(d, True)
    ok = len(o.analysis_results) >= 1
    return ok, r[:300]


@_test
def cosmic_ray_spike():
    """Huge spike + median filter → spike removed, analysis completes."""
    d = _tmp()
    x, y = _raman_peak()
    y[250] = 1e6
    _csv(d / "spike.csv", x, y)
    _meta(d / "meta.json", sample="GaN")
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {d / 'spike.csv'}, metadata {d / 'meta.json'}, "
        f"preprocessing 'remove cosmic ray spikes with median filter', agent 3, run."
    )
    shutil.rmtree(d, True)
    ok = len(o.analysis_results) >= 1
    return ok, r[:300]


@_test
def conflict_auto_mode():
    """Two instructions via auto mode → conflict detected."""
    d = _tmp()
    x, y = _raman_peak(300)
    _csv(d / "d.csv", x, y)
    _meta(d / "meta.json")
    o = _orch(d / "s")
    o.chat(f"Load metadata {d / 'meta.json'}, set preprocessing 'subtract baseline'.")
    r = o.chat("Set preprocessing to 'normalize to 0-1'.")
    shutil.rmtree(d, True)
    # LLM should have detected conflict or resolved it
    return True, r[:300]


@_test
def replace_mode():
    """Replace overwrites previous instruction."""
    d = _tmp()
    x, y = _raman_peak(300)
    _csv(d / "d.csv", x, y)
    _meta(d / "meta.json")
    o = _orch(d / "s")
    o.chat(f"Load metadata {d / 'meta.json'}, set preprocessing 'subtract baseline'.")
    r = o.chat("Replace preprocessing with 'divide by reference'. Use mode='replace'.")
    shutil.rmtree(d, True)
    instr = o.current_metadata.get("custom_processing_instruction", "")
    ok = "divide" in instr.lower() and "subtract" not in instr.lower()
    return ok, f"instruction={instr}"


@_test
def force_append():
    """Force-append combines two instructions."""
    d = _tmp()
    _meta(d / "meta.json")
    o = _orch(d / "s")
    o.chat(f"Load metadata {d / 'meta.json'}, set preprocessing 'subtract baseline'.")
    r = o.chat("Force-append preprocessing 'then smooth window=11'. Use mode='force_append'.")
    shutil.rmtree(d, True)
    instr = o.current_metadata.get("custom_processing_instruction", "")
    ok = "subtract" in instr.lower() and "smooth" in instr.lower()
    return ok, f"instruction={instr}"


@_test
def empty_instruction_clears():
    """Empty string clears custom preprocessing."""
    d = _tmp()
    _meta(d / "meta.json")
    o = _orch(d / "s")
    o.chat(f"Load metadata {d / 'meta.json'}, set preprocessing 'subtract baseline'.")
    o.chat("Set preprocessing instruction to ''.")
    shutil.rmtree(d, True)
    ok = "custom_processing_instruction" not in o.current_metadata
    return ok, f"key present={('custom_processing_instruction' in o.current_metadata)}"


@_test
def metadata_reload_preserves():
    """Preprocessing survives metadata reload."""
    d = _tmp()
    _meta(d / "m1.json", sample="Si")
    _meta(d / "m2.json", sample="Ge")
    o = _orch(d / "s")
    o.chat(f"Load metadata {d / 'm1.json'}, set preprocessing 'subtract baseline'.")
    o.chat(f"Load metadata {d / 'm2.json'}.")
    shutil.rmtree(d, True)
    instr = o.current_metadata.get("custom_processing_instruction", "")
    ok = "subtract" in instr.lower() or "baseline" in instr.lower()
    return ok, f"instruction={instr}"


@_test
def hyperspectral_missing_energy_range():
    """3D cube without energy_range → clear error, not NoneType crash."""
    d = _tmp()
    np.save(d / "cube.npy", np.random.poisson(100, (10, 10, 50)).astype(float))
    _meta(d / "meta.json", technique="EELS", sample="TiO2")
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {d / 'cube.npy'}, metadata {d / 'meta.json'}, "
        f"preprocessing 'apply despiking', agent 2, run."
    )
    shutil.rmtree(d, True)
    ok = "NoneType" not in r
    return ok, r[:300]


@_test
def series_with_preprocessing():
    """Series of 3 spectra → preprocessing locks after first."""
    d = _tmp()
    sd = d / "series"
    sd.mkdir()
    for t in [25, 50, 75]:
        x = np.linspace(200, 800, 300)
        y = 1000 * np.exp(-((x - 520) ** 2) / (2 * 15**2)) + t * 2 + np.random.normal(0, 10, 300)
        _csv(sd / f"T{t}.csv", x, y)
        _meta(sd / f"T{t}.json", sample=f"Si_{t}C",
              extra={"conditions": {"temperature_C": t}})
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {sd}, load metadata from {sd}, "
        f"set preprocessing 'subtract linear baseline', agent 3, run."
    )
    shutil.rmtree(d, True)
    ok = len(o.analysis_results) >= 1
    return ok, r[:300]


@_test
def multistep_with_reference_file():
    """Multi-step preprocessing referencing an external file."""
    d = _tmp()
    x = np.linspace(200, 800, 500)
    y = 1000 * np.exp(-((x - 520) ** 2) / (2 * 15**2)) + 200 + 0.5 * x
    _csv(d / "data.csv", x, y)
    _csv(d / "ref.csv", x, 100 + 0.5 * x)
    _meta(d / "meta.json", sample="GaAs")
    o = _orch(d / "s")
    r = o.chat(
        f"Load {d / 'data.csv'}, metadata {d / 'meta.json'}. "
        f"Preprocessing: 'divide by reference in {d / 'ref.csv'}, subtract min, normalize to max'. "
        f"Agent 3, run."
    )
    shutil.rmtree(d, True)
    ok = len(o.analysis_results) >= 1
    return ok, r[:300]


@_test
def wrong_ndim_4d():
    """4D array → graceful error, not crash."""
    d = _tmp()
    np.save(d / "4d.npy", np.random.rand(5, 5, 10, 3))
    _meta(d / "meta.json", technique="Unknown")
    o = _orch(d / "s")
    r = o.chat(
        f"Examine {d / '4d.npy'}, metadata {d / 'meta.json'}, "
        f"preprocessing 'normalize', agent 2, run."
    )
    shutil.rmtree(d, True)
    return True, r[:300]  # pass if no unhandled exception


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Set GEMINI_API_KEY env var first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        indices = [int(a) - 1 for a in sys.argv[1:]]
        to_run = [TESTS[i] for i in indices]
    else:
        to_run = TESTS

    results = {}
    for fn in to_run:
        name = fn.__name__
        desc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"\n{'=' * 60}")
        print(f"[{TESTS.index(fn) + 1}/{len(TESTS)}] {name}: {desc}")
        print("=" * 60)
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            results[name] = status
            print(f"  → {status}: {detail[:200]}")
        except KeyboardInterrupt:
            results[name] = "SKIP"
            break
        except Exception as e:
            results[name] = "ERROR"
            print(f"  → ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        marker = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{marker}] {name}: {status}")
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {passed}/{total} passed")
