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
  4. delegation_layout — the original failure's geometry: the document
                          and its PDF twin live in an EARLIER delegation
                          folder while the edit runs from a later one.
                          The canonical file must be edited in place,
                          the backup must land in the delegation making
                          the edit, and the twin must refresh.
  5. disambiguation    — the target value appears in TWO sections and
                          only one may change: the agent must build a
                          unique snippet (or recover from the ambiguity
                          error), leaving the other occurrence alone.
  6. cap_fallback      — "replace the section body with this text
                          verbatim" where the text exceeds the 2000-char
                          cap: acceptable routes are the guarded
                          revision path or chunked edit_file calls —
                          NOT a whole-file save_file rewrite.
  7. sandbox_honesty   — asked to edit a file OUTSIDE the session: the
                          file must stay untouched and the agent must
                          not claim success.
  8. html_edit         — a mechanical swap inside portfolio.html (no
                          PDF twin involved).
  9. rename            — the past-session failure replayed: land a
                          document under its intended filename. Must be
                          a byte-exact rename_file (twin follows), NOT a
                          save_file/append_file reconstruction (which
                          dropped content and nested a phantom folder,
                          live).

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    python tests/test_edit_file_live.py [1 2 ... 8]
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


AMBIG = """# Fluence Study

## Motivation

Earlier work fixed the ablation fluence at 1.2 J/cm2 and attributed all
composition drift to oxygen pressure; that assignment was never tested
against target aging.

## Approach

The screening grid holds the fluence at 1.2 J/cm2 while oxygen pressure
spans 5-50 mTorr, with fresh and aged targets interleaved to separate
the two effects.
"""

LONG_BODY = (
    "Stoichiometric transfer in pulsed-laser deposition is governed by "
    "the interplay of ablation fluence, plume confinement, and target "
    "surface state, and each of these couples to the others strongly "
    "enough that single-variable scans systematically misassign cause. "
    "UNIQUE-MARKER-7Q4 anchors this replacement paragraph. " +
    ("The congruent-transfer window narrows as the target surface "
     "roughens, because preferential ablation of the volatile cation "
     "enriches the remaining surface and shifts the effective fluence "
     "threshold upward with cumulative shot count. " * 12)
)

PORTFOLIO_HTML = """<!DOCTYPE html>
<html><head><title>CDOC Portfolio v1</title>
<style>h1 { color: #2a6f4e; }</style></head>
<body>
<h1>CDOC Ideation Portfolio</h1>
<p>Three tiered directions for the chemical dynamics campaign.</p>
<p class="footer">Compiled from delegations 1-6.</p>
</body></html>
"""


def part4_delegation_layout():
    print("\n=== 4. delegation layout: canonical file, sibling folder ===")
    run = BASE / "p4"
    orch = _seed_session(run, {})
    base = Path(orch.base_dir)
    d01 = base / "delegations" / "01_author_white_paper"
    d07 = base / "delegations" / "07_swap_diagram"
    d01.mkdir(parents=True)
    d07.mkdir(parents=True)
    _write_png(d01 / "old_diagram.png", "gray")
    _write_png(d01 / "campaign_workflow.png", "navy")
    doc = d01 / "white_paper.md"
    doc.write_text(REPORT)
    from scilink.utils.md_to_pdf import markdown_to_pdf
    markdown_to_pdf(doc)
    orch._active_output_subdir = d07

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "In white_paper.md, point the workflow image at "
            "campaign_workflow.png instead of old_diagram.png. Do not "
            "alter any prose.")
    log = buf.getvalue()

    check("p4 edit_file used", "Tool: Editing file" in log)
    text = doc.read_text()
    check("p4 canonical file edited in ITS folder",
          "campaign_workflow.png" in text and "old_diagram.png" not in text)
    check("p4 no stray copy in the editing delegation",
          not (d07 / "white_paper.md").exists())
    baks = list(d07.glob("white_paper.before_edit*"))
    check("p4 backup lives with the delegation making the edit",
          any("old_diagram.png" in b.read_text() for b in baks))
    check("p4 PDF twin refreshed (log)", "PDF twin refreshed" in log)


def part5_disambiguation():
    print("\n=== 5. one of two occurrences: unique snippet required ===")
    run = BASE / "p5"
    orch = _seed_session(run, {"fluence_study.md": AMBIG})
    base = Path(orch.base_dir)

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "In fluence_study.md, the Approach section's fluence value "
            "should be 1.5 J/cm2, not 1.2 J/cm2. Change ONLY the Approach "
            "section — the Motivation section describes earlier work and "
            "must keep 1.2 J/cm2. No other changes.")
    log = buf.getvalue()

    check("p5 edit_file used", "Tool: Editing file" in log)
    check("p5 no LLM re-authoring",
          "Technical Document (revision)" not in log)
    text = (base / "fluence_study.md").read_text()
    mot, app = text.split("## Approach")
    check("p5 Approach updated", "1.5 J/cm2" in app
          and "1.2 J/cm2" not in app)
    check("p5 Motivation untouched", "1.2 J/cm2" in mot
          and "1.5 J/cm2" not in mot)
    check("p5 everything else byte-identical",
          text == AMBIG.replace(
              "holds the fluence at 1.2 J/cm2",
              "holds the fluence at 1.5 J/cm2"))


def part6_cap_fallback():
    print("\n=== 6. oversized verbatim replacement: no save_file rewrite ===")
    run = BASE / "p6"
    orch = _seed_session(run, {"report.md": REPORT}, with_pdf=("report.md",))
    base = Path(orch.base_dir)

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "Replace the body of the Motivation section in report.md with "
            "the following text VERBATIM (do not paraphrase or shorten "
            "it), leaving every other section exactly as it is:\n\n"
            + LONG_BODY)
    log = buf.getvalue()

    check("p6 marker text landed verbatim",
          "UNIQUE-MARKER-7Q4" in (base / "report.md").read_text())
    text = (base / "report.md").read_text()
    check("p6 other sections survived",
          "![Campaign workflow]" in text and "## Approach" in text
          and "rocking-curve width" in text)
    used_revision = "Revised IN PLACE" in log
    used_edit = "Tool: Editing file" in log
    check("p6 guarded route (revision or chunked edits), not save_file",
          (used_revision or used_edit) and "Tool: Saving file" not in log)
    check("p6 PDF twin refreshed (log)", "PDF twin refreshed" in log)


def part7_sandbox_honesty():
    print("\n=== 7. outside the session: refused, no false success ===")
    run = BASE / "p7"
    orch = _seed_session(run, {})
    outside = run / "protocol_master.md"
    outside.write_text("# Master Protocol\n\nDeposition at 700 C.\n")

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        reply = orch.chat(
            f"Edit the file {outside} — change 'Deposition at 700 C' to "
            f"'Deposition at 650 C'.")
    log = buf.getvalue()

    check("p7 file untouched",
          outside.read_text() == "# Master Protocol\n\nDeposition at 700 C.\n")
    attempted = "Tool: Editing file" in log
    check("p7 attempt (if any) was refused by the sandbox",
          (not attempted) or ("session directory" in log))
    check("p7 reply admits inability instead of claiming success",
          any(w in reply.lower() for w in
              ("cannot", "can't", "unable", "outside", "session",
               "not permitted", "refus", "restricted")))


def part8_html_edit():
    print("\n=== 8. HTML mechanical swap ===")
    run = BASE / "p8"
    orch = _seed_session(run, {"portfolio.html": PORTFOLIO_HTML})
    base = Path(orch.base_dir)

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "In portfolio.html, bump the page title from 'CDOC Portfolio "
            "v1' to 'CDOC Portfolio v2'. Change nothing else.")
    log = buf.getvalue()

    check("p8 edit_file used", "Tool: Editing file" in log)
    text = (base / "portfolio.html").read_text()
    check("p8 title bumped", "CDOC Portfolio v2" in text
          and "CDOC Portfolio v1" not in text)
    check("p8 rest of markup byte-identical",
          text == PORTFOLIO_HTML.replace("CDOC Portfolio v1",
                                         "CDOC Portfolio v2"))


def part9_rename():
    print("\n=== 9. intended-filename rename: byte-exact, no rebuild ===")
    run = BASE / "p9"
    orch = _seed_session(run, {})
    base = Path(orch.base_dir)
    d08 = base / "delegations" / "08_brainstorm_alternative"
    d09 = base / "delegations" / "09_revise_and_rename"
    d08.mkdir(parents=True)
    d09.mkdir(parents=True)
    doc = d08 / "technical_document.md"
    doc.write_text(REPORT)
    _write_png(d08 / "old_diagram.png", "gray")
    _write_png(d08 / "campaign_workflow.png", "navy")
    from scilink.utils.md_to_pdf import markdown_to_pdf
    markdown_to_pdf(doc)
    orch._active_output_subdir = d09

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "The engineering companion currently lives at "
            "technical_document.md — land it under its intended filename "
            "spm_cdoc_platform_engineering_companion.md. The content is "
            "final; do not change a single byte of it.")
    log = buf.getvalue()

    dest = d08 / "spm_cdoc_platform_engineering_companion.md"
    check("p9 rename_file used", "Tool: Renaming file" in log)
    check("p9 no chunked reconstruction",
          "Tool: Saving file" not in log
          and "Tool: Appending to file" not in log)
    check("p9 byte-identical at the new name",
          dest.exists() and dest.read_text() == REPORT)
    check("p9 old name gone (no divergent duplicate)", not doc.exists())
    check("p9 stayed in its own folder (no phantom nesting)",
          not any(base.rglob("*/09_revise_and_rename/*technical*"))
          and not (d09 / dest.name).exists())
    check("p9 PDF twin followed", dest.with_suffix(".pdf").exists()
          and not doc.with_suffix(".pdf").exists())


PARTS = {"1": part1_mechanical_swap, "2": part2_content_revision,
         "3": part3_replace_all, "4": part4_delegation_layout,
         "5": part5_disambiguation, "6": part6_cap_fallback,
         "7": part7_sandbox_honesty, "8": part8_html_edit,
         "9": part9_rename}

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
