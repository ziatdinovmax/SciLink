#!/usr/bin/env python3
"""Live validation of the struggle-based best-of-N escalation gate.

Reproduces the session that motivated the redesign (2026-06-18 11:41:44,
`nanoparticles.npy`): attempt 0 scored ~0.80 and was approved, but the old
`score >= 0.85 AND iters <= 2` gate fanned out 3x — and attempt 0 won anyway.

The new gate (`UnifiedImageProcessingController._candidate_fast_accept`)
fast-accepts unless attempt 0 (a) failed/declined, (b) landed razor-thin
above threshold (< threshold + 0.05), or (c) had to go HOT (max annealing
level == hot) to get there. On escalation, the fan-out is seeded warm
(level 1) when attempt 0 went hot, cold otherwise.

This harness does NOT hard-assert a specific escalated/fast-accept outcome
(LLM runs are non-deterministic); it asserts the gate's decision is
SELF-CONSISTENT with attempt 0's score and the annealing levels parsed from
the verification log. The headline win is reported explicitly: a good,
cold/warm first attempt should fast-accept (no fan-out).

Needs a vendor key (e.g. AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME, or
ANTHROPIC_API_KEY) and UNSAFE_EXECUTION_OK=true.

Usage:
  UNSAFE_EXECUTION_OK=true python tests/test_escalation_struggle_gate_live.py \
      [model_name] [path/to/data.npy]
"""
import io
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from scilink import auth

# Portable default = an in-repo segmentation image. The motivating session
# used `nanoparticles.npy` (not in the repo); pass it as argv[2] to reproduce
# the original case. Validated there: attempt 0 scored 0.83 cold/1-iter and
# fast-accepted (would have fanned out 3x under the old score>=0.85 gate).
DEFAULT_DATA = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)
SYS_INFO = "A microscopy image of a particulate/polycrystalline material."
OBJECTIVE = (
    "Segment the objects and report per-object size/count statistics."
)

# Mirror the controller's threshold + margin so the harness can recompute the
# expected decision independently of the controller's own logging.
QUALITY_THRESHOLD = 0.7
SCORE_MARGIN = 0.05
HOT_LEVEL = 2  # len(_CONSTRAINT_ANNEALING_SCHEDULE) - 1


def _image_agent(model_name, api_key, out):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    return ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        use_literature=False, enable_human_feedback=False,
    )


def _attempt0_max_level(log: str) -> int | None:
    """Highest annealing level attempt 0 (cand_00) reached, from the log.

    Verification lines read 'Verification i/N (annealing level L)...'. With
    escalation, attempt 0 runs alone first, so the levels logged before the
    'escalating to' line all belong to cand_00.
    """
    head = log.split("escalating to", 1)[0]
    levels = [int(m) for m in re.findall(r"annealing level (\d+)", head)]
    return max(levels) if levels else None


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    data_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA
    if not Path(data_path).is_file():
        print("ERROR: data file not found:", data_path)
        return 2
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2
    if os.environ.get("UNSAFE_EXECUTION_OK") != "true":
        print("ERROR: set UNSAFE_EXECUTION_OK=true to run codegen live")
        return 2

    print("=== struggle-gate escalation (n=3, candidate_escalation=True) ===")
    print(f"  model: {model_name}")
    print(f"  data : {data_path}")
    out = Path(tempfile.mkdtemp(prefix="struggle_gate_"))

    agent = _image_agent(model_name, api_key, str(out))
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    t0 = time.monotonic()
    try:
        res = agent.analyze(
            data=data_path, system_info=SYS_INFO, objective=OBJECTIVE,
            n_candidates=3, candidate_escalation=True,
        )
    finally:
        logging.getLogger().removeHandler(handler)
    log = buf.getvalue()

    judge = (res or {}).get("anchor_judge") or {}
    table = (res or {}).get("anchor_candidates") or []
    escalated = judge.get("escalated")
    cdir = out / "image_0000" / "_candidates"
    dirs = sorted(p.name for p in cdir.glob("cand_*")) if cdir.is_dir() else []
    a0 = next((c for c in table if c.get("attempt") == 0), None)
    a0_score = a0.get("score") if a0 else None
    a0_level = _attempt0_max_level(log)

    print(f"  status                 : {res.get('status')!r}")
    print(f"  wall clock             : {time.monotonic() - t0:.0f}s")
    print(f"  attempt 0 score        : {a0_score}")
    print(f"  attempt 0 max level    : {a0_level} (hot == {HOT_LEVEL})")
    print(f"  escalated              : {escalated}")
    print(f"  candidate dirs         : {dirs}")
    print(f"  table entries          : {len(table)}")
    seed_line = re.search(r"fan-out starts at annealing level (\d+)", log)
    if seed_line:
        print(f"  fan-out seed level     : {seed_line.group(1)}")

    checks = []

    # 1. dirs/table consistent with the escalated flag.
    if escalated is False:
        checks.append(("fast-accept: one dir + one table entry",
                       dirs == ["cand_00"] and len(table) == 1))
    elif escalated is True:
        checks.append(("escalated: 3 dirs + 3 table entries",
                       len(dirs) == 3 and len(table) == 3))
    else:
        checks.append(("escalated flag present", False))

    # 2. The gate decision matches attempt 0's score + annealing trajectory.
    if a0_score is not None and a0_level is not None:
        a0_approved = bool(a0.get("approved"))
        expect_fast = (
            a0_approved
            and a0_score >= QUALITY_THRESHOLD + SCORE_MARGIN
            and a0_level < HOT_LEVEL
        )
        checks.append(
            (f"gate self-consistent (expect_fast_accept={expect_fast})",
             (escalated is False) == expect_fast))

    # 3. Seed direction matches attempt 0's level (only when escalated).
    if escalated is True and a0_level is not None and seed_line:
        seed = int(seed_line.group(1))
        expect_seed = 1 if a0_level >= HOT_LEVEL else 0
        checks.append((f"fan-out seed == {expect_seed}", seed == expect_seed))

    # 4. The headline win: a good cold/warm first attempt did NOT fan out.
    if (a0_score is not None and a0_level is not None
            and a0_score >= QUALITY_THRESHOLD + SCORE_MARGIN
            and a0_level < HOT_LEVEL):
        checks.append(("HEADLINE: good cold/warm attempt 0 fast-accepted",
                       escalated is False))

    print("\n  --- checks ---")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n  artifacts: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
