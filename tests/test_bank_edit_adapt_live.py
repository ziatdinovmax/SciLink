"""Live validation for bank minimal-edit adaptation (Phase B) on Bedrock
Opus 4.8, against an ISOLATED bank (SCILINK_HOME -> temp dir).

  1. adapt    — fit dataset A (auto-banked), then fit similar dataset B
                fresh: the edit-adapt path must fire, the accepted script
                must be a small-diff of the banked one, provenance
                recorded, and the bank record's cross-session success
                count bumped.
  2. fallthrough — dissimilar data (damped oscillation vs peaks): the
                edit-adapt path must NOT anchor to the wrong script; the
                run still succeeds via normal generation.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true python tests/test_bank_edit_adapt_live.py [1 2]
"""
from __future__ import annotations

import contextlib
import difflib
import io
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_bank_edit_adapt_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


@contextlib.contextmanager
def capture_all(buf):
    h = logging.StreamHandler(buf)
    logging.getLogger().addHandler(h)
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        logging.getLogger().removeHandler(h)


def make_peak(path, center, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 400)
    y = (2.0 * np.exp(-((x - center) ** 2) / (2 * 0.4 ** 2))
         + 0.3 + 0.02 * x + rng.normal(0, 0.02, x.size))
    np.savetxt(path, np.column_stack([x, y]), delimiter=",",
               header="position,intensity", comments="")


SYS_INFO = {"technique": "generic 1D spectroscopy",
            "x_axis": "position (a.u.)", "y_axis": "intensity (a.u.)"}


def agent(out):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(api_key=None, model_name=MODEL,
                             output_dir=str(out),
                             enable_human_feedback=False,
                             max_verification_iterations=1)


def bank_records():
    from scilink.skills._shared import _script_bank
    return _script_bank.list_records("curve_fitting")


def part1_adapt():
    print("\n=== 1. strong match: minimal-edit adapt + proven-N bump ===")
    run = BASE / "p1"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_peak(run / "spec_a.csv", center=5.0, seed=1)
    make_peak(run / "spec_b.csv", center=6.5, seed=2)   # shifted peak

    res_a = agent(run / "run_a").analyze(str(run / "spec_a.csv"),
                                         system_info=SYS_INFO)
    check("p1 run A succeeded", res_a.get("status") == "success")
    recs = bank_records()
    check("p1 run A auto-banked (isolated bank)", len(recs) == 1)
    if not recs:
        return
    banked = recs[0]["working_script"]

    buf = Tee()
    with capture_all(buf):
        res_b = agent(run / "run_b").analyze(str(run / "spec_b.csv"),
                                             system_info=SYS_INFO)
    log = buf.getvalue()

    check("p1 run B succeeded", res_b.get("status") == "success")
    check("p1 edit-adapt fired", "Bank edit-adapt: record" in log)
    sfr = json.loads((run / "run_b" / "series_fit_results.json").read_text())
    r0 = (sfr.get("results") or [{}])[0]
    bea = r0.get("bank_edit_adapt") or {}
    check("p1 provenance recorded (id + edits)",
          bea.get("id") == recs[0]["id"] and "edits" in bea)
    if bea:
        script_b = r0.get("script") or ""
        if not script_b:
            scripts = sorted((run / "run_b" / "scripts").glob("*.py"))
            script_b = scripts[0].read_text() if scripts else ""
        ndiff = sum(1 for l in difflib.unified_diff(
            banked.strip().splitlines(), script_b.strip().splitlines(),
            lineterm="")
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
        n_lines = max(len(banked.strip().splitlines()), 1)
        print(f"     adapted-vs-banked diff: {ndiff} changed line(s) "
              f"of {n_lines} ({100 * ndiff // (2 * n_lines)}%)")
        # The invariant is "recognizably the banked script": a
        # regeneration diffs at ~100%; targeted edits stay a small
        # fraction (live: 5 edits -> 23 diff lines on a ~150-line script).
        check("p1 adaptation, not regeneration (diff < 40% of script)",
              ndiff < 0.8 * n_lines)
    after = bank_records()
    n_succ = (after[0].get("stats") or {}).get("n_successes", 1)
    check("p1 proven-N bumped on the SAME record", n_succ >= 2)


def part2_fallthrough():
    print("\n=== 2. dissimilar data: no wrong-script anchoring ===")
    run = BASE / "p2"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 400)
    y = np.exp(-0.3 * x) * np.sin(2 * np.pi * 1.2 * x) \
        + rng.normal(0, 0.02, x.size)
    np.savetxt(run / "osc.csv", np.column_stack([x, y]), delimiter=",",
               header="time,signal", comments="")

    buf = Tee()
    with capture_all(buf):
        res = agent(run / "run_c").analyze(
            str(run / "osc.csv"),
            system_info={"technique": "time-domain damped oscillation",
                         "x_axis": "time (s)", "y_axis": "signal (a.u.)"})
    log = buf.getvalue()
    check("p2 dissimilar run succeeded via normal generation",
          res.get("status") == "success")
    check("p2 edit-adapt did NOT anchor to the peak script",
          "Bank edit-adapt: record" not in log)
    blob = json.dumps(res, default=str)
    # key-specific: the run DIRECTORY name contains the substring
    check("p2 no adapt provenance", '"bank_edit_adapt":' not in blob)


def part3_image_adapt():
    print("\n=== 3. image twin: adapt + bump through the shared core ===")
    from scilink.agents.exp_agents.image_analysis_agent import (
        ImageAnalysisAgent)

    run = BASE / "p3"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)

    def make_image(path, seed):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:128, 0:128]
        img = rng.normal(0.1, 0.02, (128, 128))
        for cx, cy in [(30, 40), (80, 90), (100, 30)]:
            img += 0.8 * np.exp(-(((xx - cx - rng.integers(-4, 5)) ** 2
                                   + (yy - cy - rng.integers(-4, 5)) ** 2)
                                  / (2 * 6.0 ** 2)))
        np.save(path, img.astype(np.float32))

    make_image(run / "img_a.npy", 1)
    make_image(run / "img_b.npy", 2)
    si = {"technique": "AFM height image",
          "description": "nanoparticles on a flat substrate",
          "pixel_size_nm": 2.0}

    def img_agent(out):
        return ImageAnalysisAgent(api_key=None, model_name=MODEL,
                                  output_dir=str(out),
                                  enable_human_feedback=False,
                                  max_verification_iterations=1)

    res_a = img_agent(run / "run_a").analyze(str(run / "img_a.npy"),
                                             system_info=si)
    check("p3 run A succeeded", res_a.get("status") == "success")
    from scilink.skills._shared import _script_bank
    recs = _script_bank.list_records("image_analysis")
    if not recs:
        # The organic write gate requires QC APPROVAL, stochastic at
        # max_verification_iterations=1. The gate is pre-existing #346
        # behavior with its own tests; THIS scenario tests adaptation —
        # seed the bank deterministically from run A's saved script.
        scripts = sorted((run / "run_a" / "scripts").glob("*.py"))
        img_a = np.load(run / "img_a.npy")
        _script_bank.add_record("image_analysis", {
            "technique_signals": {"analysis_type":
                                  "particle blob detection (seeded)"},
            "measurement_context": _script_bank.measurement_context(si),
            "data_fingerprint": _script_bank.image_fingerprint(
                img_a, pixel_size_nm=2.0),
            "outcome": {"metric": {"name": "score", "value": 0.9}},
            "working_script": scripts[0].read_text(),
        })
        recs = _script_bank.list_records("image_analysis")
        print("     (bank seeded from run A's script — organic gate "
              "did not approve at smoke settings)")
    check("p3 bank holds a record", len(recs) >= 1)
    if not recs:
        return

    buf = Tee()
    with capture_all(buf):
        res_b = img_agent(run / "run_b").analyze(str(run / "img_b.npy"),
                                                 system_info=si)
    log = buf.getvalue()
    check("p3 run B succeeded", res_b.get("status") == "success")
    check("p3 edit-adapt fired", "Bank edit-adapt: record" in log)
    # Image analyze() flattens its return; provenance lives in the saved
    # artifacts (same read the corpus matrix uses).
    blob = "".join(p.read_text() for p in (run / "run_b").rglob("*.json"))
    check("p3 provenance present", '"bank_edit_adapt"' in blob)
    after = {r["id"]: (r.get("stats") or {}).get("n_successes", 1)
             for r in _script_bank.list_records("image_analysis")}
    bumped = any(n >= 2 for n in after.values())
    # Clean-acceptance rule: an adaptation kept WITH a quality_warning
    # must not bump (observed live both ways: clean run bumped; a
    # score-0.54 flagged run was correctly withheld).
    flagged = '"quality_warning": "' in blob
    if flagged:
        check("p3 bump correctly WITHHELD (adaptation kept but flagged)",
              not bumped)
    else:
        check("p3 proven-N bumped on a clean adaptation", bumped)


PARTS = {"1": part1_adapt, "2": part2_fallthrough, "3": part3_image_adapt}

if __name__ == "__main__":
    assert os.environ.get("SCILINK_HOME"), \
        "Set SCILINK_HOME to an isolated temp dir before running."
    os.environ.setdefault("SCILINK_SCRIPT_BANK", "1")
    for k in (sys.argv[1:] or sorted(PARTS)):
        PARTS[k]()
    print("\n" + "=" * 60)
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
