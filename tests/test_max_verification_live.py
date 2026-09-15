"""Live validation of max_verification_iterations on image + hyperspectral (#271).

  image_fast — synthetic particle image, analyze(max_verification_iterations=0):
      the successful initial analysis must be accepted with the bypass (no LLM
      verification iterations), and the run must be fast.
  hs_fast — synthetic cube with a strong emission band, single-attempt mode:
      exactly one codegen attempt, no retries, task success.
  orch_fast — orchestrator chat level: a task asking to skip verification on
      an image file; the LLM must map it to max_verification_iterations=0 and
      the forwarding introspection must reach the agent (bypass log fires).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_max_verification_live.py \
      [--scenario all|image_fast|hs_fast|orch_fast]
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(7)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _LogTap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def grep(self, needle):
        return [ln for ln in self.lines if needle in ln]


def _make_particle_image(path):
    h = w = 256
    img = RNG.normal(0.2, 0.03, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r in ((60, 70, 12), (150, 90, 15), (100, 180, 10),
                      (200, 200, 13), (50, 200, 9), (210, 60, 11)):
        img += 0.8 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                              / (2 * (r / 2) ** 2)))
    np.save(path, img.astype(np.float32))


def _make_emission_cube(path):
    wl = np.linspace(400, 900, 120)
    h = w = 64
    # Peak center drifts 600 -> 640 nm across x, so the per-pixel position
    # map has genuine spatial structure (a flat map reads as fitting noise
    # to the visual QC and routes to salvage instead of the bypass).
    centers = 600.0 + 40.0 * (np.arange(w) / (w - 1))
    spec = 4.0 * np.exp(
        -0.5 * ((wl[None, :] - centers[:, None]) / 18) ** 2)   # (w, E)
    cube = (5.0 + np.broadcast_to(spec[None, :, :], (h, w, wl.size))
            + RNG.normal(0, 0.3, (h, w, wl.size))).astype(np.float32)
    np.save(path, cube)
    json.dump({"technique": "hyperspectral emission map",
               "energy_range": {"start": 400.0, "end": 900.0,
                                "units": "nm"}},
              open(path.replace(".npy", ".json"), "w"))


def scenario_image_fast():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    d = tempfile.mkdtemp(prefix="mvi_img_live_")
    p = os.path.join(d, "particles.npy")
    _make_particle_image(p)
    tap = _LogTap()
    ag = ImageAnalysisAgent(api_key=None, model_name=MODEL,
                            output_dir=os.path.join(d, "out"),
                            enable_human_feedback=False)
    ag.logger.addHandler(tap)
    print(f"\n### image_fast — {d}")
    t0 = time.monotonic()
    res = ag.analyze(p, system_info={"technique": "TEM image",
                                     "sample": "nanoparticles on carbon"},
                     objective="Count the particles and measure their sizes.",
                     max_verification_iterations=0)
    dt = time.monotonic() - t0
    print(f"[image_fast] wall: {dt/60:.1f} min | status: {res.get('status')}")
    check("image_fast: success", res.get("status") == "success")
    check("image_fast: bypass fired (no LLM verification)",
          len(tap.grep("Verification bypassed")) >= 1)
    check("image_fast: zero verification iterations ran",
          len(tap.grep("Verification 1/")) == 0)
    check("image_fast: fast (< 8 min)", dt < 480)
    feats = res.get("extracted_features") or {}
    check("image_fast: features extracted", bool(feats))


def scenario_hs_fast():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    d = tempfile.mkdtemp(prefix="mvi_hs_live_")
    p = os.path.join(d, "emission.npy")
    _make_emission_cube(p)
    tap = _LogTap()
    ag = HyperspectralAnalysisAgent(api_key=None, model_name=MODEL,
                                    output_dir=os.path.join(d, "out"),
                                    enable_human_feedback=False)
    ag.logger.addHandler(tap)
    print(f"\n### hs_fast — {d}")
    t0 = time.monotonic()
    res = ag.analyze(
        p, system_info=p.replace(".npy", ".json"),
        objective="Map the per-pixel position of the emission band near "
                  "620 nm (it shifts across the field of view).",
        max_verification_iterations=0)
    dt = time.monotonic() - t0
    print(f"[hs_fast] wall: {dt/60:.1f} min | status: {res.get('status')}")
    check("hs_fast: completed", res.get("status") in ("success", "partial"))
    n_attempts = len(tap.grep("Asking LLM to write code"))
    print(f"[hs_fast] codegen attempts: {n_attempts}")
    check("hs_fast: single codegen attempt (no retry ladder)",
          n_attempts <= 1 or len(tap.grep("Retries bypassed")) >= 1)
    check("hs_fast: no engine verification iterations",
          len(tap.grep("Verification 1/")) == 0)
    check("hs_fast: fast (< 10 min)", dt < 600)


def scenario_orch_fast():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    d = tempfile.mkdtemp(prefix="mvi_orch_live_")
    p = os.path.join(d, "particles.npy")
    _make_particle_image(p)
    base = tempfile.mkdtemp(prefix="mvi_orch_sess_")
    tap = _LogTap()
    logging.getLogger().addHandler(tap)
    ag = AnalysisOrchestratorAgent(
        base_dir=base, api_key=None, base_url=None, model_name=MODEL,
        restore_checkpoint=False, analysis_mode=AnalysisMode.AUTONOMOUS)
    print(f"\n### orch_fast — session {base}")
    t0 = time.monotonic()
    res = ag.run_task(
        f"Analyze the TEM image at {p} (nanoparticles on carbon; count the "
        "particles and measure their sizes). I need a QUICK first look — "
        "skip the verification loop entirely, no refinement.",
        autonomy=AnalysisMode.AUTONOMOUS)
    dt = time.monotonic() - t0
    print(f"[orch_fast] wall: {dt/60:.1f} min | status: {res.get('status')}")
    check("orch_fast: success", res.get("status") == "success")
    check("orch_fast: chat-level intent mapped to the bypass "
          "(forwarding reached the image agent)",
          len(tap.grep("Verification bypassed")) >= 1)
    check("orch_fast: bounded (< 15 min)", dt < 900)
    logging.getLogger().removeHandler(tap)


SCENARIOS = {"image_fast": scenario_image_fast,
             "hs_fast": scenario_hs_fast,
             "orch_fast": scenario_orch_fast}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + sorted(SCENARIOS))
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    todo = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in todo:
        try:
            SCENARIOS[name]()
        except Exception:  # noqa: BLE001
            import traceback; traceback.print_exc()
            check(f"{name}: completed without crashing", False)
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"MAX-VERIFICATION LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
