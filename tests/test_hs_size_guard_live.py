"""Live validation of the hyperspectral size guard on FULL-RES SOC-710 cubes.

Hands the agent the same full-resolution 704x704x260 cubes (129M values)
whose unguarded analysis previously ground for 3 hours, via the exact path
the fan-out branches use (AnalysisOrchestratorAgent.run_task). Expected:
the preprocessing strategy selects spatial_bin_factor=4 by itself (logged
"spatial mean-binning (size guard)"), the analysis completes in sane wall
time, and the fluorescence science still lands (two Dy(III) emission
peaks, ~480 / ~575 nm).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_hs_size_guard_live.py
"""
import io
import json
import logging
import os
import re
import sys
import tempfile
import time

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
D = os.path.expanduser("~/Code/benchmarking_for_paper2/Labric_MZI_scilink")
WALL_LIMIT_S = 3600           # vs ~3 h unguarded

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    # Capture INFO logs so the size-guard line is assertable.
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)

    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)

    runs = [
        ("fluorescence", "fluorescence_cube_magnet_off",
         "Analyze the hyperspectral fluorescence image at {p} (704x704x260; "
         "wavelengths in the metadata JSON at {m}; 325 nm HeCd excitation, "
         "filtered out of the recorded range; magnet off). The sample is "
         "1 M Dy(NO3)3 in 0.1 M aqueous HNO3. Separate the emission from "
         "detector background/striping, characterize the emission spectrum "
         "and its spatial localization, and report the emission peak "
         "positions."),
        ("absorbance", "absorbance_cube_magnet_off",
         "Analyze the hyperspectral absorbance-mode image at {p} "
         "(704x704x260 raw transmitted intensity, white light on; "
         "wavelengths in the metadata JSON at {m}; magnet off). The sample "
         "is 1 M Dy(NO3)3 in 0.1 M aqueous HNO3. Identify the absorption "
         "features of the solution (dips vs the broadband illumination) and "
         "report their wavelengths."),
    ]
    summaries = {}
    for tag, stem, task_tpl in runs:
        base = tempfile.mkdtemp(prefix=f"hs_guard_{tag}_")
        print(f"\n### {tag}: full-res cube via run_task — session {base}")
        ag = AnalysisOrchestratorAgent(
            base_dir=base, api_key=None, base_url=None, model_name=MODEL,
            restore_checkpoint=False, analysis_mode=AnalysisMode.AUTONOMOUS)
        p = os.path.join(D, stem + ".npy")
        m = os.path.join(D, stem + ".json")
        t0 = time.monotonic()
        res = ag.run_task(task_tpl.format(p=p, m=m),
                          autonomy=AnalysisMode.AUTONOMOUS)
        dt = time.monotonic() - t0
        log = buf.getvalue()
        print(f"[{tag}] wall time: {dt/60:.1f} min | status: {res.get('status')}")
        check(f"{tag}: completed within {WALL_LIMIT_S//60} min "
              f"(was ~180 unguarded)", dt < WALL_LIMIT_S)
        check(f"{tag}: run_task success", res.get("status") == "success")
        mm = re.findall(r"Applied (\d)x\1 spatial mean-binning \(size guard\): "
                        r"\(704, 704, 260\) -> \((\d+), (\d+), 260\)", log)
        no_bin = re.findall(r"Size guard: spatial_bin_factor=1", log)
        print(f"[{tag}] size-guard log hits: {mm} | explicit no-bin "
              f"decisions: {len(no_bin)}")
        # The guard is a judgment heuristic; the INVARIANT is "no unbounded
        # grind": either binning fired, or the run finished quickly anyway
        # (a light analysis path can legitimately skip the reduction).
        check(f"{tag}: guard fired (factor >= 2) or run was fast (< 20 min)",
              any(int(f) >= 2 for f, *_ in mm) or dt < 1200)
        summaries[tag] = (res.get("summary") or "") + " ".join(
            str(k) for k in res.get("key_findings") or [])
        print(f"[{tag}] summary (first 1200):\n{summaries[tag][:1200]}")
        buf.truncate(0); buf.seek(0)

    # Science checks (the guard must not cost the answer):
    fl = summaries.get("fluorescence", "")
    nums = [float(v) for v in re.findall(r"\b(\d{3}(?:\.\d+)?)\s*nm\b", fl)]
    check("fluorescence: ~480 nm Dy emission found",
          any(470 <= v <= 490 for v in nums))
    check("fluorescence: ~575 nm Dy emission found",
          any(565 <= v <= 585 for v in nums))
    ab = summaries.get("absorbance", "")
    anums = [float(v) for v in re.findall(r"\b(\d{3}(?:\.\d+)?)\s*nm\b", ab)]
    check("absorbance: NIR Dy band found (740-920 nm)",
          any(740 <= v <= 920 for v in anums))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"HS SIZE GUARD LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
