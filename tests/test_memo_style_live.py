"""Live: write_technical_document(style="memo") on Bedrock.

Run: python tests/test_memo_style_live.py [1 2 3]
1 — "one-page memo" from a source white paper: routing picks style=memo
    unprompted; shape, budget, .docx checked.
2 — memo with no source and no literature: grounding honesty, still memo.
3 — condense a long white paper INTO a memo in place (revise_path).
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_memo_style_live_runs").resolve()
SRC_WP = Path("meta_session_20260817_162104/planning/delegations/"
              "01_propose_a_killer_proof_of_concept_use_ca/white_paper.md")

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


def _seed(run_dir: Path):
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "data").mkdir(parents=True)
    orch = PlanningOrchestratorAgent(
        objective="Chemical Dynamics Observation & Control platform design",
        base_dir=str(run_dir / "session"), api_key=None, model_name=MODEL,
        autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(run_dir / "data"))
    return orch


def _memo_shape_checks(tag, md: Path, docx: Path | None, log: str):
    text = md.read_text()
    words = len(text.split())
    print(f"    {tag}: {words} words; sections: "
          f"{[l for l in text.splitlines() if l.startswith('## ')]}")
    check(f"{tag} tool called with style=memo", '"style": "memo"' in log
          or "style='memo'" in log or "Technical Memo" in log)
    check(f"{tag} within one-page budget (<=600 words)", words <= 600)
    check(f"{tag} not trivially short (>=250 words)", words >= 250)
    check(f"{tag} header block: Purpose + Date lines",
          re.search(r"\*\*Purpose:\*\*", text) and re.search(r"\*\*Date:\*\*", text))
    check(f"{tag} Date line carries the injected date",
          re.search(r"\*\*Date:\*\*\s*August 1[78], 2026", text))
    check(f"{tag} one CORE line", len(re.findall(r"\*\*CORE [A-Z]+\*\*", text)) == 1)
    n_sec = len([l for l in text.splitlines() if l.startswith("## ")])
    check(f"{tag} 3-5 sections", 3 <= n_sec <= 5)
    check(f"{tag} bold lead-ins used",
          len(re.findall(r"^\*\*[A-Z][^*]{1,40}\.\*\* ", text, re.M)) >= 3)
    check(f"{tag} no References/Executive Summary section",
          not re.search(r"^## (References|Executive Summary|Summary)\s*$", text, re.M))
    check(f"{tag} no auto workflow diagram", "workflow" not in text.lower()
          or "![" not in text)
    check(f"{tag} .docx twin exists", docx is not None and docx.exists())
    if docx and docx.exists():
        from docx import Document
        d = Document(docx)
        paras = [p for p in d.paragraphs if p.text.strip()]
        check(f"{tag} docx opens with TECHNICAL MEMO label",
              paras and paras[0].text == "TECHNICAL MEMO")
        check(f"{tag} docx has Heading 1 sections",
              sum(p.style.name == "Heading 1" for p in paras) == n_sec)
    return text


def part1_memo_from_source():
    print("\n=== 1. one-page memo from a source white paper ===")
    run = BASE / "p1"
    orch = _seed(run)
    base = Path(orch.base_dir)
    shutil.copy(SRC_WP, base / "white_paper.md")

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "Write a one-page technical memo for the directorate that "
            "distils white_paper.md: what the proof-of-concept is, why "
            "state-triggered control benchmarked against a yoked clock is "
            "the decisive test, and what it commits us to in year one. "
            "Base it on white_paper.md; no new literature search.")
    log = buf.getvalue()
    mds = [p for p in base.rglob("*.md") if p.name != "white_paper.md"
           and "before_" not in p.name]
    check("p1 a new memo file was written", len(mds) >= 1)
    if not mds:
        return
    md = max(mds, key=lambda p: p.stat().st_mtime)
    docx = md.with_suffix(".docx")
    text = _memo_shape_checks("p1", md, docx, log)
    check("p1 grounded in source (mentions yoked)", "yoked" in text.lower())
    check("p1 no save_file authoring", "Tool: Saving file" not in log)


def part2_memo_no_source():
    print("\n=== 2. memo phrasing, no source, no literature ===")
    run = BASE / "p2"
    orch = _seed(run)
    base = Path(orch.base_dir)
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "Give me a short memo (one page) defining how a quantum-sensing "
            "track could be added to an operando materials platform "
            "without changing its scientific objective: role, candidate "
            "capability families, relationship to the existing workflow, "
            "and a staged development approach. Skip literature search; "
            "mark what is assumption.")
    log = buf.getvalue()
    mds = [p for p in base.rglob("*.md") if "before_" not in p.name]
    check("p2 a memo file was written", len(mds) >= 1)
    if not mds:
        return
    md = max(mds, key=lambda p: p.stat().st_mtime)
    text = _memo_shape_checks("p2", md, md.with_suffix(".docx"), log)
    check("p2 assumptions marked", re.search(r"assum", text, re.I) is not None)


def part3_condense_in_place():
    print("\n=== 3. condense the white paper INTO a memo in place ===")
    run = BASE / "p3"
    orch = _seed(run)
    base = Path(orch.base_dir)
    d01 = base / "delegations" / "01_white_paper"
    d02 = base / "delegations" / "02_condense"
    d01.mkdir(parents=True); d02.mkdir(parents=True)
    doc = d01 / "white_paper.md"
    shutil.copy(SRC_WP, doc)
    n_before = len(doc.read_text().split())
    orch._active_output_subdir = d02
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            f"Condense {doc} in place into a one-page technical memo — "
            "same file, memo format. Keep the yoked-clock benchmark as the "
            "core idea.")
    log = buf.getvalue()
    text = _memo_shape_checks("p3", doc, doc.with_suffix(".docx"), log)
    print(f"    p3: {n_before} words -> {len(text.split())} words")
    check("p3 revised in place (same path)", "Revised IN PLACE" in log)
    check("p3 pre-revision copy kept in delegation 02",
          list(d02.glob("white_paper.before_revision*")))
    check("p3 was NOT refused as shrinkage", "Revision aborted" not in log)


PARTS = {"1": part1_memo_from_source, "2": part2_memo_no_source,
         "3": part3_condense_in_place}

if __name__ == "__main__":
    for k in (sys.argv[1:] or sorted(PARTS)):
        PARTS[k]()
    print("\n" + "=" * 60)
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
