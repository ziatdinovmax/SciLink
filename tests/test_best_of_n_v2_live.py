#!/usr/bin/env python3
"""Live test for best-of-N v2: escalation, aux concurrency, join approval.

Scenarios (each on a real image / synthetic spectrum):
  C. Image escalation (n=3, candidate_escalation=True): attempt 0 runs alone;
     dirs/table consistent with anchor_judge.escalated either way.
  D. Image + auxiliary operand (n=2): fan-out no longer disabled; both
     candidate dirs exist; run succeeds.
  E. Image join approval (enable_human_feedback=True, n=2): the BEST-OF-N
     prompt fires, bestofn_candidate_*_review.png exist at prompt time and
     are cleaned afterwards; Enter accepts the judge pick.
  F. Curve escalation (n=3, candidate_escalation=True) on a bi-exponential
     decay: expected fast-accept (table len 1, escalated False) but either
     outcome is accepted if self-consistent.

Needs a vendor key (e.g. AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME) and
UNSAFE_EXECUTION_OK=true.

Usage: python tests/test_best_of_n_v2_live.py [model_name]
"""
import builtins
import io
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scilink import auth

GRAINS = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)
SYS_INFO = "A microscopy image of a polycrystalline material."
OBJECTIVE = "Segment the grains and report grain size statistics."


def _image_agent(model_name, api_key, out, **kw):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    return ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        use_literature=False, **kw,
    )


def _capture_logs():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return buf, handler


def _consistency(out, res, n):
    """Escalation consistency: dirs/table match the escalated flag."""
    judge = (res or {}).get("anchor_judge") or {}
    escalated = judge.get("escalated")
    table = (res or {}).get("anchor_candidates") or []
    cdir = out / "image_0000" / "_candidates"
    dirs = sorted(p.name for p in cdir.glob("cand_*")) if cdir.is_dir() else []
    print(f"  escalated          : {escalated}")
    print(f"  candidate dirs     : {dirs}")
    print(f"  table entries      : {len(table)}")
    if escalated is True:
        return len(dirs) == n and len(table) == n
    if escalated is False:
        return dirs == ["cand_00"] and len(table) == 1
    return False


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2

    checks = []

    # --- C: escalation -------------------------------------------------
    print("=== C: image escalation (n=3, candidate_escalation=True) ===")
    out = Path(tempfile.mkdtemp(prefix="bestofn_v2_esc_"))
    agent = _image_agent(model_name, api_key, str(out),
                         enable_human_feedback=False)
    buf, handler = _capture_logs()
    t0 = time.monotonic()
    try:
        res = agent.analyze(data=GRAINS, system_info=SYS_INFO,
                            objective=OBJECTIVE, n_candidates=3,
                            candidate_escalation=True)
    finally:
        logging.getLogger().removeHandler(handler)
    log = buf.getvalue()
    print(f"  status             : {res.get('status')!r}")
    print(f"  wall clock         : {time.monotonic() - t0:.0f}s")
    esc_log = "(escalation): running attempt 0 alone" in log
    print(f"  escalation log line: {esc_log}")
    ok_c = (res.get("status") == "success" and esc_log
            and _consistency(out, res, 3))
    checks.append(ok_c)
    print(f"  -> {'PASS' if ok_c else 'CHECK'}  ({out})")

    # --- D: auxiliary operand + fan-out ---------------------------------
    print("\n=== D: image + auxiliary operand (n=2) ===")
    img = np.load(GRAINS)
    aux = (img.astype(np.float64) * 0.5 + 10).astype(np.uint8)  # co-registered
    aux_path = Path(tempfile.mkdtemp(prefix="bestofn_v2_aux_in_")) / "ref.npy"
    np.save(aux_path, aux)
    out = Path(tempfile.mkdtemp(prefix="bestofn_v2_aux_"))
    agent = _image_agent(model_name, api_key, str(out),
                         enable_human_feedback=False)
    buf, handler = _capture_logs()
    try:
        res = agent.analyze(data=GRAINS, system_info=SYS_INFO,
                            objective=OBJECTIVE, n_candidates=2,
                            auxiliary_data=str(aux_path),
                            auxiliary_label="co-registered reference channel")
    finally:
        logging.getLogger().removeHandler(handler)
    log = buf.getvalue()
    cdir = out / "image_0000" / "_candidates"
    dirs = sorted(p.name for p in cdir.glob("cand_*")) if cdir.is_dir() else []
    not_disabled = "best-of-N disabled" not in log
    print(f"  status             : {res.get('status')!r}")
    print(f"  candidate dirs     : {dirs}")
    print(f"  no disable log     : {not_disabled}")
    ok_d = (res.get("status") == "success" and len(dirs) == 2 and not_disabled)
    checks.append(ok_d)
    print(f"  -> {'PASS' if ok_d else 'CHECK'}  ({out})")

    # --- E: join approval ------------------------------------------------
    print("\n=== E: join approval (feedback on, n=2, Enter accepts) ===")
    out = Path(tempfile.mkdtemp(prefix="bestofn_v2_join_"))
    agent = _image_agent(model_name, api_key, str(out),
                         enable_human_feedback=True)
    prompts, reviews_at_prompt = [], []
    real_input = builtins.input

    def auto_input(prompt=""):
        prompts.append(prompt)
        reviews_at_prompt.extend(
            out.glob("bestofn_candidate_*_review.png")
        )
        return ""  # accept every prompt

    builtins.input = auto_input
    try:
        res = agent.analyze(data=GRAINS, system_info=SYS_INFO,
                            objective=OBJECTIVE, n_candidates=2)
    finally:
        builtins.input = real_input
    table = (res or {}).get("anchor_candidates") or []
    sel = [c for c in table if c.get("selected")]
    cleaned = not list(out.glob("bestofn_candidate_*_review.png"))
    print(f"  status             : {res.get('status')!r}")
    print(f"  prompts fired      : {len(prompts)}")
    print(f"  reviews at prompt  : {len(set(reviews_at_prompt))}")
    print(f"  reviews cleaned    : {cleaned}")
    print(f"  table/selected     : {len(table)}/{len(sel)}")
    ok_e = (res.get("status") == "success"
            and len(set(reviews_at_prompt)) == 2 and cleaned
            and len(table) == 2 and len(sel) == 1)
    checks.append(ok_e)
    print(f"  -> {'PASS' if ok_e else 'CHECK'}  ({out})")

    # --- F: curve escalation ---------------------------------------------
    print("\n=== F: curve escalation (n=3, candidate_escalation=True) ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    rng = np.random.default_rng(7)
    t = np.linspace(0, 50, 400)
    y = 800 * np.exp(-t / 4.0) + 250 * np.exp(-t / 18.0) + 30
    y = y + rng.normal(0, 8, t.shape)
    spec = Path(tempfile.mkdtemp(prefix="bestofn_v2_curve_in_")) / "decay.npy"
    np.save(spec, np.column_stack([t, y]))
    out = Path(tempfile.mkdtemp(prefix="bestofn_v2_curve_"))
    cagent = CurveFittingAgent(
        api_key=api_key, model_name=model_name, output_dir=str(out),
        enable_human_feedback=False, use_literature=False,
    )
    res = cagent.analyze(
        data=str(spec),
        system_info="Photoluminescence decay trace: time (ns) vs intensity.",
        objective="Fit the decay and extract lifetimes.",
        n_candidates=3, candidate_escalation=True,
    )
    judge = (res or {}).get("anchor_judge") or {}
    table = (res or {}).get("anchor_candidates") or []
    cdir = out / "spectrum_0000" / "_candidates"
    dirs = sorted(p.name for p in cdir.glob("cand_*")) if cdir.is_dir() else []
    escalated = judge.get("escalated")
    print(f"  status             : {res.get('status')!r}")
    print(f"  escalated          : {escalated}")
    print(f"  candidate dirs     : {dirs}")
    print(f"  table entries      : {len(table)}")
    consistent = (
        (escalated is False and dirs == ["cand_00"] and len(table) == 1)
        or (escalated is True and len(dirs) == 3 and len(table) == 3)
    )
    ok_f = res.get("status") == "success" and consistent
    checks.append(ok_f)
    print(f"  -> {'PASS' if ok_f else 'CHECK'}  ({out})")

    ok = all(checks) and len(checks) == 4
    print("\nRESULT:", "PASS — escalation, aux fan-out, and join approval "
          "work live" if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
