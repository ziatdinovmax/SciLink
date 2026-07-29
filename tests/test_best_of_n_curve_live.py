#!/usr/bin/env python3
"""Live test for parallel best-of-N anchor fitting (curve fitting agent).

Runs CurveFittingAgent.analyze on a synthetic noisy spectrum and checks:

  A. n_candidates=3 -> three attempt dirs under spectrum_0000/_candidates/,
     winner artifacts promoted, anchor_candidates table (one selected=True)
     in the result, judge/milestone log lines.
  B. default (no n_candidates) -> no fan-out: no _candidates dir, no table,
     no best-of-N log lines. Curve fitting's default is 1 BY DESIGN.

Needs a vendor key (e.g. AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME) and
UNSAFE_EXECUTION_OK=true.

Usage: python tests/test_best_of_n_curve_live.py [model_name]
"""
import io
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scilink import auth

SYS_INFO = ("Photoluminescence decay trace: time (ns) vs intensity (counts), "
            "room temperature.")
OBJECTIVE = "Fit the decay and extract the lifetime(s)."


def _make_spectrum(tmp: Path) -> str:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 50, 400)
    y = 800 * np.exp(-t / 4.0) + 250 * np.exp(-t / 18.0) + 30
    y = y + rng.normal(0, 8, t.shape)
    p = tmp / "decay.npy"
    np.save(p, np.column_stack([t, y]))
    return str(p)


def _run(model_name, api_key, spectrum, *, n_candidates):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    out = tempfile.mkdtemp(prefix=f"bestofn_curve_{n_candidates or 'def'}_")
    agent = CurveFittingAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
    )

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)

    kwargs = dict(data=spectrum, system_info=SYS_INFO, objective=OBJECTIVE)
    if n_candidates is not None:
        kwargs["n_candidates"] = n_candidates

    raised, result = None, None
    t0 = time.monotonic()
    try:
        result = agent.analyze(**kwargs)
    except Exception as e:  # noqa: BLE001
        raised = e
    finally:
        root.removeHandler(handler)
    return result, buf.getvalue(), raised, Path(out), time.monotonic() - t0


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2

    spectrum = _make_spectrum(Path(tempfile.mkdtemp(prefix="bestofn_curve_in_")))
    checks = []

    print("=== A: n_candidates=3 ===")
    res, log, raised, out, dt = _run(model_name, api_key, spectrum,
                                     n_candidates=3)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        cdir = out / "spectrum_0000" / "_candidates"
        cand_dirs = sorted(p.name for p in cdir.glob("cand_*")) \
            if cdir.is_dir() else []
        table = (res or {}).get("anchor_candidates") or []
        sel = [c for c in table if c.get("selected")]
        judge = (res or {}).get("anchor_judge") or {}
        milestones = log.count("finished (")
        r2 = (res or {}).get("fit_quality", {}).get("r_squared")
        print(f"  status            : {res.get('status')!r}")
        print(f"  wall clock        : {dt:.0f}s")
        print(f"  candidate dirs    : {cand_dirs}")
        print(f"  winner R²         : {r2}")
        for c in table:
            print(f"    - attempt {c['attempt']}: R²={c['score']:.4f}, "
                  f"approved={c['approved']}, selected={c['selected']}")
        print(f"  judge reasoning   : {judge.get('reasoning', '')[:160]!r}")
        print(f"  judge fallback    : {judge.get('fallback')}")
        print(f"  milestone lines   : {milestones}")
        ok_a = (
            res.get("status") == "success"
            and len(cand_dirs) == 3
            and len(table) == 3 and len(sel) == 1
            and milestones == 3
        )
        checks.append(ok_a)
        print(f"  -> {'PASS' if ok_a else 'CHECK'}")
        print(f"  output dir: {out}")

    print("\n=== B: default (n_candidates unset) ===")
    res, log, raised, out, dt = _run(model_name, api_key, spectrum,
                                     n_candidates=None)
    if raised is not None:
        print(f"  RAISED: {type(raised).__name__}: {raised}")
        checks.append(False)
    else:
        no_cand_dir = not (out / "spectrum_0000" / "_candidates").exists()
        no_table = "anchor_candidates" not in (res or {})
        no_logs = "Best-of-" not in log
        print(f"  status            : {res.get('status')!r}")
        print(f"  wall clock        : {dt:.0f}s")
        print(f"  no _candidates dir: {no_cand_dir}")
        print(f"  no candidate table: {no_table}")
        print(f"  no best-of-N logs : {no_logs}")
        ok_b = (res.get("status") == "success"
                and no_cand_dir and no_table and no_logs)
        checks.append(ok_b)
        print(f"  -> {'PASS' if ok_b else 'CHECK'}")
        print(f"  output dir: {out}")

    ok = all(checks) and len(checks) == 2
    print("\nRESULT:", "PASS — curve best-of-N opt-in works; default "
          "unchanged" if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
