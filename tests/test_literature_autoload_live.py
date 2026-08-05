"""Live validation for plural literature auto-load, issue #425 (Bedrock
Opus 4.8).

Seeds a planning session whose campaign holds one or several saved
literature files, then exercises the REAL refine / white-paper / chat
paths against a live model:

  1. single_file_refine   — 1 file: auto-load logs '1 file(s)', refined
                            plan's literature_search carries the corpus.
  2. two_file_refine      — 2 files: auto-load logs '2 file(s)' and BOTH
                            corpora markers reach the refined plan (the
                            pre-fix behavior lost the older file).
  3. white_paper_multi    — no plan literature: the rescue restores the
                            UNION and the white paper can cite both.
  4. budget_drop_refine   — tight budget: whole-section drop is logged,
                            refine still succeeds on what was kept.
  5. chat_auto_default    — neutral chat refine: the LLM omits
                            literature_context and auto-load unions all
                            campaign literature.
  6. chat_selection       — chat says only part of the literature is
                            relevant: the LLM consults
                            list_literature_searches and passes a
                            selection instead of loading everything.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    python tests/test_literature_autoload_live.py [1 2 ...]
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_lit_autoload_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


OBJ = ("Control polymorph selection in flash-annealed TiO2 thin films "
       "through pulsed sub-threshold annealing schedules")

Q1 = ("What is established about nucleation path selection in "
      "flash-annealed oxide thin films?")
Q2 = ("Capture a metastable state that exists only under drive and "
      "relaxes within milliseconds?")
Q_TOP = "Which pulse spacings and duty cycles suppress rutile takeover?"

FOUNDATION_MARK = "LEVERAGE-INDEX SCREENING PROTOCOL"
CROSS_MARK = "DELAYED-COUNTER-PULSE FALSIFICATION BATTERY"
TOPUP_MARK = "GRADIENT-FURNACE PULSE-SPACING MAP"


def foundation_text():
    para_a = (
        "Nucleation in flash-annealed anatase/rutile systems proceeds "
        "through transient sub-critical clusters whose lifetime is set by "
        "the quench rate; classical nucleation theory underestimates the "
        "anatase persistence window by 2-3x when pulse heating is used. "
    )
    para_b = (
        "Analogous behavior in driven colloidal crystallization and in "
        "magnetization reversal suggests a one-shot path-selection event "
        "within the first pulse, after which the polymorph identity is "
        "locked. The " + CROSS_MARK + " distinguishes true path selection "
        "from cumulative dose effects: equal-dose, opposite-timing and "
        "delayed-counter-pulse arms must disagree if selection is "
        "one-shot. The " + FOUNDATION_MARK + " ranks candidate control "
        "parameters by the ratio of their effect on the selection event "
        "to their effect on total dose. "
    )
    return (
        "# Literature Search Results (hypothesis_context+cross_domain)\n\n"
        f"# Question 1: {Q1}\n\n"
        "## ESTABLISHED IN THIS FIELD (known methods, parameter ranges, "
        "failure modes)\n" + para_a * 30 + "\n\n"
        f"# Question 2: {Q2}\n\n"
        "## TRANSFERABLE MECHANISMS FROM OTHER DOMAINS (analogies — NOT "
        "established results in this field)\n" + para_b * 30 + "\n"
    )


def topup_text():
    para = (
        "Recent pulse-programmed furnace studies map rutile takeover as a "
        "function of pulse spacing (0.5-50 ms) and duty cycle (5-40%). "
        "The " + TOPUP_MARK + " shows a suppression valley near 5 ms / "
        "10% where anatase fraction exceeds 80% at total doses that "
        "otherwise fully convert. "
    )
    return ("# Literature Search Results (hypothesis_context)\n\n"
            "## ESTABLISHED IN THIS FIELD (known methods, parameter "
            "ranges, failure modes)\n" + para * 20 + "\n")


PLAN = {
    "campaign_id": 1,
    "hypothesis": "one-shot nucleation path selection",
    "proposed_experiments": [{
        "hypothesis": ("Polymorph identity is decided by a one-shot path "
                       "selection event during the first sub-threshold "
                       "pulse, not by cumulative thermal dose."),
        "experiment_name": "Pulse-timing polymorph control",
        "experimental_steps": [
            "Sputter 50 nm amorphous TiO2 films on Si with native oxide.",
            "Apply pulse trains varying spacing (1-20 ms) at fixed total "
            "dose using the flash-anneal stage.",
            "Quantify anatase/rutile fraction by GIXRD and Raman.",
        ],
        "expected_outcome": ("Anatase fraction depends on pulse timing at "
                             "fixed dose."),
        "justification": ("Cross-domain evidence suggests timing-dependent "
                          "one-shot selection."),
    }],
}

RESULTS_TXT = (
    "Run 7 results: at fixed total dose, 5 ms spacing gave 78% anatase; "
    "1 ms gave 22%; 20 ms gave 31%. Raman confirms no brookite. A repeat "
    "at 5 ms with the pulse order reversed reproduced 76% anatase."
)


def _seed_session(run_dir: Path, lit_specs):
    """Fresh orchestrator whose campaign registry holds lit_specs =
    [(name, text, label, questions, age_s), ...] (age_s: older = larger)."""
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    orch = PlanningOrchestratorAgent(
        objective=OBJ,
        base_dir=str(run_dir / "session"),
        api_key=None, model_name=MODEL,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    now = time.time()
    reg = []
    for name, text, label, questions, age_s in lit_specs:
        p = Path(orch.base_dir) / name
        p.write_text(text)
        os.utime(p, (now - age_s, now - age_s))
        entry = {"path": str(p.resolve()), "campaign_id": 1}
        if label:
            entry["label"] = label
        if questions:
            entry["questions"] = questions
        reg.append(entry)
    orch.planner.state = {
        "objective": OBJ,
        "campaign_id": 1,
        "iteration_index": 1,
        "experimental_results": [],
        "action_history": [],
        "current_plan": json.loads(json.dumps(PLAN)),
        "plan_history": [json.loads(json.dumps(PLAN))],
        "campaign_literature": reg,
    }
    return orch


TWO_FILES = [
    ("literature_search_hypothesis_context+cross_domain.md",
     foundation_text(), "hypothesis_context+cross_domain", [Q1, Q2], 3600),
    ("literature_search_hypothesis_context.md",
     topup_text(), "hypothesis_context", [Q_TOP], 60),
]


# ---------------------------------------------------------------- parts

def part1_single_file_refine():
    print("\n=== 1. single-file refine: fast path, loaded whole ===")
    run = BASE / "p1"
    orch = _seed_session(run, TWO_FILES[:1])
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.functions_map["refine_plan_with_results"](
            RESULTS_TXT))
    log = buf.getvalue()
    check("p1 refine succeeded", out.get("status") == "success")
    check("p1 log names 1 file", "1 file(s)" in log)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p1 corpus reached refined plan", FOUNDATION_MARK in lit
          and CROSS_MARK in lit)
    check("p1 corpus verbatim (fast path)",
          lit == foundation_text() or foundation_text() in lit)


def part2_two_file_refine():
    print("\n=== 2. two-file refine: union, both corpora ===")
    run = BASE / "p2"
    orch = _seed_session(run, TWO_FILES)
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.functions_map["refine_plan_with_results"](
            RESULTS_TXT))
    log = buf.getvalue()
    check("p2 refine succeeded", out.get("status") == "success")
    check("p2 log names 2 files", "2 file(s)" in log)
    check("p2 log names both filenames",
          "cross_domain" in log and "hypothesis_context.md" in log)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p2 OLDER foundation corpus present (pre-fix: lost)",
          FOUNDATION_MARK in lit and CROSS_MARK in lit)
    check("p2 newer top-up present", TOPUP_MARK in lit)
    check("p2 foundation leads (oldest first)",
          lit.index(FOUNDATION_MARK) < lit.index(TOPUP_MARK))


def part3_white_paper_multi():
    print("\n=== 3. white paper: rescue restores the union ===")
    run = BASE / "p3"
    orch = _seed_session(run, TWO_FILES)
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        wp_path = orch.tools._write_white_paper()
    log = buf.getvalue()
    check("p3 restore log names both files",
          "restored from" in log and "cross_domain" in log
          and "hypothesis_context.md" in log)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p3 union in state", FOUNDATION_MARK in lit and TOPUP_MARK in lit)
    text = Path(wp_path).read_text()
    check("p3 white paper written", len(text) > 1000)


def part4_budget_drop_refine():
    print("\n=== 4. tight budget: whole-section drop, logged ===")
    from scilink.agents.planning_agents import orchestrator_tools as ot
    run = BASE / "p4"
    orch = _seed_session(run, TWO_FILES)
    keep = ot._LIT_AUTOLOAD_MAX_CHARS
    ot._LIT_AUTOLOAD_MAX_CHARS = len(foundation_text()) + 100
    try:
        buf = Tee()
        with contextlib.redirect_stdout(buf):
            out = json.loads(
                orch.tools.functions_map["refine_plan_with_results"](
                    RESULTS_TXT))
    finally:
        ot._LIT_AUTOLOAD_MAX_CHARS = keep
    log = buf.getvalue()
    check("p4 refine succeeded", out.get("status") == "success")
    check("p4 drop logged", "Literature budget" in log and "dropped" in log)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p4 foundation kept whole", FOUNDATION_MARK in lit
          and CROSS_MARK in lit)
    check("p4 top-up dropped whole", TOPUP_MARK not in lit)


def part5_chat_auto_default():
    print("\n=== 5. chat refine, neutral ask: auto-load all ===")
    run = BASE / "p5"
    orch = _seed_session(run, TWO_FILES)
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(f"Please refine the plan based on these new results: "
                  f"{RESULTS_TXT}")
    log = buf.getvalue()
    check("p5 refine tool ran", "Refining Plan" in log)
    auto = "Auto-loaded literature context from session (2 file(s)" in log
    explicit = "Literature context provided" in log
    check("p5 literature reached refine (auto-union or explicit)",
          auto or explicit)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p5 older corpus not lost", FOUNDATION_MARK in lit)


def part6_chat_selection():
    print("\n=== 6. chat refine, narrow ask: index consulted ===")
    run = BASE / "p6"
    orch = _seed_session(run, TWO_FILES)
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        orch.chat(
            "New results, purely about pulse spacing: " + RESULTS_TXT +
            " These results only bear on the annealing-schedule aspect. "
            "Refine the plan grounding it ONLY on the saved literature "
            "that is actually relevant to pulse spacing/schedules — check "
            "what literature this campaign has saved and pass just the "
            "relevant part, not everything.")
    log = buf.getvalue()
    check("p6 index tool consulted",
          "Listing campaign literature searches" in log)
    check("p6 explicit selection passed",
          "Literature context provided" in log)
    check("p6 no selection skipped as missing",
          "does not contain" not in log)
    lit = (orch.planner.state.get("current_plan") or {}).get(
        "literature_search", "")
    check("p6 relevant (pulse-spacing) corpus reached the refine",
          TOPUP_MARK in lit)


PARTS = {1: part1_single_file_refine, 2: part2_two_file_refine,
         3: part3_white_paper_multi, 4: part4_budget_drop_refine,
         5: part5_chat_auto_default, 6: part6_chat_selection}


if __name__ == "__main__":
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        sys.exit("AWS_BEARER_TOKEN_BEDROCK not set")
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    wanted = [int(a) for a in sys.argv[1:]] or sorted(PARTS)
    if BASE.exists() and set(wanted) == set(PARTS):
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n in wanted:
        try:
            PARTS[n]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"p{n} completed without exception ({e})", False)
    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"LIT AUTOLOAD LIVE: {npass}/{len(results)} passed "
          f"({time.time() - t0:.0f}s)")
    for k, v in results.items():
        if not v:
            print(f"  FAILED: {k}")
    sys.exit(0 if npass == len(results) else 1)
