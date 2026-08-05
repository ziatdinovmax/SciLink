"""Live validation for the edit_file tool + PDF-twin refresh (Bedrock
Opus 4.8).

Seeds a standalone planning session holding an existing markdown document
with an exported PDF twin, then drives the REAL chat loop and checks the
agent routes edits correctly:

  1. mechanical_swap   — "point the figure at the new PNG, bump the
                          caption rev": the agent must use edit_file (not
                          re-author the document), prose must survive
                          byte-identical, and the stale PDF twin must be
                          re-exported with the new caption.
  2. content_revision  — "merge two sections and rewrite formally": a
                          content-level ask must route to
                          write_technical_document(revise_path=...), and
                          the revision branch must refresh the PDF twin.
  3. replace_all       — "rename the material everywhere": every
                          occurrence replaced (directly with
                          replace_all=true or by recovering from the
                          ambiguity error), other prose untouched.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    python tests/test_edit_file_live.py [1 2 3]
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_edit_file_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


def _write_png(path: Path, color):
    from PIL import Image
    Image.new("RGB", (64, 32), color).save(path)

OBJ = "Optimize pulsed-laser deposition of BaTiO3 thin films"

REPORT = """# PLD Campaign Report

## Motivation

Stoichiometric transfer in pulsed-laser deposition of BaTiO3 degrades
above a fluence threshold that depends on target density; mapping that
threshold is the campaign's first milestone.

## Workflow

![Campaign workflow](old_diagram.png)

*Figure 1 — campaign workflow (rev A).*

## Approach

A fluence/oxygen-pressure grid is screened first, then Bayesian
optimization refines around the best cell. Film quality is scored by
XRD rocking-curve width and RHEED oscillation persistence.
"""

NOTES = """# Materials Notes

PZT-5H targets sinter at 1250 C. The PZT-5H powder route needs excess
PbO to offset lead loss. Density above 96% is required before PZT-5H
targets survive high-fluence ablation, and archived PZT-5H batches
below that density are reserved for calibration only.
"""


def _pdf_text(pdf_path):
    import fitz
    with fitz.open(pdf_path) as doc:
        return "".join(page.get_text() for page in doc)


def _seed_session(run_dir: Path, docs: dict, with_pdf=()):
    """Fresh AUTONOMOUS planning session with `docs` (name -> text) and a
    real 1-px PNG for every image the report references; names in
    `with_pdf` also get an exported PDF twin (the stale artifact)."""
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    from scilink.utils.md_to_pdf import markdown_to_pdf
    if run_dir.exists():
        shutil.rmtree(run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    orch = PlanningOrchestratorAgent(
        objective=OBJ,
        base_dir=str(run_dir / "session"),
        api_key=None, model_name=MODEL,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    base = Path(orch.base_dir)
    _write_png(base / "old_diagram.png", "gray")
    _write_png(base / "campaign_workflow.png", "navy")
    for name, text in docs.items():
        (base / name).write_text(text)
        if name in with_pdf:
            markdown_to_pdf(base / name)
    return orch


# ---------------------------------------------------------------- parts

def part1_mechanical_swap():
    print("\n=== 1. mechanical swap: edit_file, prose intact, PDF fresh ===")
    run = BASE / "p1"
    orch = _seed_session(run, {"report.md": REPORT}, with_pdf=("report.md",))
    base = Path(orch.base_dir)
    before_lines = set(REPORT.splitlines())

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "The workflow figure in report.md is outdated. Point it at "
            "campaign_workflow.png instead of old_diagram.png, and bump "
            "the caption from '(rev A)' to '(rev B)'. Do not change any "
            "other text in the document.")
    log = buf.getvalue()

    check("p1 edit_file used", "Tool: Editing file" in log)
    check("p1 no LLM re-authoring",
          "Technical Document (revision)" not in log)
    text = (base / "report.md").read_text()
    check("p1 image reference swapped", "campaign_workflow.png" in text
          and "old_diagram.png" not in text)
    check("p1 caption bumped", "(rev B)" in text and "(rev A)" not in text)
    changed = set(text.splitlines()) ^ before_lines
    check("p1 only figure+caption lines differ",
          all(("diagram.png" in l or "workflow.png" in l or "rev " in l)
              for l in changed) and changed)
    check("p1 PDF twin refreshed (log)", "PDF twin refreshed" in log)
    pdf = _pdf_text(base / "report.pdf")
    check("p1 PDF carries the new caption",
          "rev B" in pdf and "rev A" not in pdf)
    baks = list(base.glob("report.before_edit*.md")) \
        + list(base.glob("report.before_edit.md"))
    check("p1 pre-edit backup kept",
          any("old_diagram.png" in b.read_text() for b in baks))


def part2_content_revision():
    print("\n=== 2. content ask routes to revision; twin refreshed ===")
    run = BASE / "p2"
    orch = _seed_session(run, {"report.md": REPORT}, with_pdf=("report.md",))
    base = Path(orch.base_dir)

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "Restructure report.md: merge the Motivation and Approach "
            "sections into a single 'Background' section and rewrite the "
            "whole document in a more formal tone. Keep the workflow "
            "figure as is.")
    log = buf.getvalue()

    check("p2 routed to document revision", "Revised IN PLACE" in log)
    check("p2 edit_file not used for the rewrite",
          "Tool: Editing file" not in log)
    text = (base / "report.md").read_text()
    check("p2 sections merged", "Background" in text
          and "## Motivation" not in text)
    check("p2 PDF twin refreshed (log)", "PDF twin refreshed" in log)
    pdf = _pdf_text(base / "report.pdf")
    check("p2 PDF matches revised content", "Background" in pdf)


def part3_replace_all():
    print("\n=== 3. rename everywhere: replace_all or recovery ===")
    run = BASE / "p3"
    orch = _seed_session(run, {"materials_notes.md": NOTES})
    base = Path(orch.base_dir)

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "In materials_notes.md, replace every mention of PZT-5H with "
            "PZT-4D. Don't touch anything else.")
    log = buf.getvalue()

    check("p3 edit_file used", "Tool: Editing file" in log)
    text = (base / "materials_notes.md").read_text()
    check("p3 all occurrences renamed", "PZT-5H" not in text
          and text.count("PZT-4D") == 4)
    untouched = NOTES.replace("PZT-5H", "PZT-4D")
    check("p3 rest of prose byte-identical", text == untouched)


PARTS = {"1": part1_mechanical_swap, "2": part2_content_revision,
         "3": part3_replace_all}

if __name__ == "__main__":
    want = sys.argv[1:] or sorted(PARTS)
    for k in want:
        PARTS[k]()
    print("\n" + "=" * 60)
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
