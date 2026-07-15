"""Live validation of the hyperspectral size guard on a full-resolution cube.

Bring your own data: point HS_GUARD_CUBE at a large (>= ~50M values) .npy
hyperspectral cube with a matching .json metadata sidecar, and drive it
through the exact path the fan-out branches use
(AnalysisOrchestratorAgent.run_task). The guard's invariant is NO UNBOUNDED
GRIND: either the preprocessing strategy selects spatial binning by itself
(logged "spatial mean-binning (size guard)"), or the analysis completes
quickly anyway via a light path. Optionally pass known spectral features to
assert the reduction did not cost the science.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true \
  HS_GUARD_CUBE=/path/to/cube.npy \
  HS_GUARD_TASK="Analyze the hyperspectral image at {p} (wavelengths in the \
metadata JSON at {m}). Characterize the spectral features and report their \
positions." \
  HS_GUARD_EXPECT_NM="480,575" \
      python tests/test_hs_size_guard_live.py
"""
import io
import logging
import os
import re
import sys
import tempfile
import time

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
CUBE = os.environ.get("HS_GUARD_CUBE")
TASK = os.environ.get(
    "HS_GUARD_TASK",
    "Analyze the hyperspectral image at {p} (wavelengths in the metadata "
    "JSON at {m}). Characterize the spectral features and report their "
    "positions.")
EXPECT_NM = [float(v) for v in
             os.environ.get("HS_GUARD_EXPECT_NM", "").split(",") if v.strip()]
EXPECT_TOL_NM = float(os.environ.get("HS_GUARD_EXPECT_TOL_NM", "10"))
WALL_LIMIT_S = int(os.environ.get("HS_GUARD_WALL_LIMIT_S", "3600"))
FAST_S = 1200      # "completed fast without binning" threshold

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    if not CUBE or not os.path.exists(CUBE):
        print("Set HS_GUARD_CUBE to a large .npy hyperspectral cube "
              "(with a .json metadata sidecar)."); sys.exit(2)
    import numpy as np
    shape = np.load(CUBE, mmap_mode="r").shape
    n_values = int(np.prod(shape))
    print(f"cube: {CUBE} shape={shape} ({n_values / 1e6:.0f}M values)")
    if n_values < 20e6:
        print("NOTE: cube is below the guard's binning threshold (~20M); "
              "this run only demonstrates the no-op path.")

    # Capture INFO logs so the size-guard decision lines are assertable.
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)

    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    base = tempfile.mkdtemp(prefix="hs_guard_")
    print("session:", base)
    ag = AnalysisOrchestratorAgent(
        base_dir=base, api_key=None, base_url=None, model_name=MODEL,
        restore_checkpoint=False, analysis_mode=AnalysisMode.AUTONOMOUS)
    m = os.path.splitext(CUBE)[0] + ".json"
    t0 = time.monotonic()
    res = ag.run_task(TASK.format(p=CUBE, m=m),
                      autonomy=AnalysisMode.AUTONOMOUS)
    dt = time.monotonic() - t0
    log = buf.getvalue()
    print(f"wall time: {dt / 60:.1f} min | status: {res.get('status')}")
    check(f"completed within {WALL_LIMIT_S // 60} min", dt < WALL_LIMIT_S)
    check("run_task success", res.get("status") == "success")
    mm = re.findall(r"Applied (\d)x\1 spatial mean-binning \(size guard\)", log)
    no_bin = re.findall(r"Size guard: spatial_bin_factor=1", log)
    print(f"size-guard decisions: binned={mm} explicit-no-bin={len(no_bin)}")
    # The guard is a judgment heuristic; the INVARIANT is "no unbounded
    # grind": binning fired, or the run finished quickly via a light path.
    check("guard fired (factor >= 2) or run was fast",
          any(int(f) >= 2 for f in mm) or dt < FAST_S)

    summary = (res.get("summary") or "") + " ".join(
        str(k) for k in res.get("key_findings") or [])
    print("summary (first 1200):\n", summary[:1200])
    if EXPECT_NM:
        nums = [float(v) for v in
                re.findall(r"\b(\d{3,4}(?:\.\d+)?)\s*nm\b", summary)]
        for target in EXPECT_NM:
            check(f"expected feature ~{target:g} nm recovered after reduction",
                  any(abs(v - target) <= EXPECT_TOL_NM for v in nums))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"HS SIZE GUARD LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
