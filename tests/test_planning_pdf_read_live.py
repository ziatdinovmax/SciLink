"""Live validation for #397 phase 0: planning reads PDFs (Bedrock Opus 4.8).

Replays the observed failure: a planning delegation asked to review an
uploaded PDF could not read it (read_file returned FlateDecode byte
streams) and — correctly but terminally — refused the review and reported
a blocker. With extraction routed through the shared parser, the same ask
must proceed: read the actual text, quote it, no blocker.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    python tests/test_planning_pdf_read_live.py
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_planning_pdf_read_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


WHITE_PAPER = """# CDOC White Paper (canonical, 0806)

## Concept and Rationale

The observatory closes the loop between perturbation and phase outcome.
Reference [6] claims the convex hull holds roughly 380,000 entries, of
which about 48,000 are on-hull; the committor formalism in Appendix 2
defines the estimand.

## Appendix 1 — Reciprocity table

Row A: MZI-QPI vs SAXS — reciprocal in q-range.
Row B: SPM vs WAXS — complementary in surface sensitivity.

## References

[6] Merchant et al., scaling deep learning for materials discovery.
"""


def main():
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    from scilink.utils.md_to_pdf import markdown_text_to_pdf

    run = BASE / "p1"
    if run.exists():
        shutil.rmtree(run)
    data_dir = run / "data"
    data_dir.mkdir(parents=True)
    orch = PlanningOrchestratorAgent(
        objective="Chemical dynamics observatory campaign design",
        base_dir=str(run / "session"),
        api_key=None, model_name=MODEL,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    uploads = Path(orch.base_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    pdf = uploads / "CDOC_white_paper_0806.pdf"
    markdown_text_to_pdf(WHITE_PAPER, pdf, title="CDOC White Paper")

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        reply = orch.chat(
            "Read the uploaded white paper at "
            f"{pdf} and answer two things from its ACTUAL text: "
            "(1) what does it claim about the convex hull entry counts, "
            "and (2) which reference number is attached to that claim? "
            "Quote the numbers exactly.")
    log = buf.getvalue()

    check("read_file used on the PDF",
          "Reading file" in log and "CDOC_white_paper_0806.pdf" in log)
    check("no blocker report", not any(
        s in reply for s in ("cannot read", "Blocker", "blocker",
                             "FlateDecode", "no extractable")))
    check("counts quoted from the actual text",
          "380,000" in reply and "48,000" in reply)
    check("reference number correct", "[6]" in reply or "reference 6"
          in reply.lower() or "ref 6" in reply.lower())

    print()
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
