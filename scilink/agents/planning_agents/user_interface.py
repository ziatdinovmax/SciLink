from typing import Dict, Any, Optional, List
import re


def format_caveats(findings: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Concise, advisory caveat lines from critic findings.

    Assumes ``findings`` is already ordered critical-first (the agent sorts it).
    Minor findings get a ``Minor:`` prefix; each line is ``[dimension] issue``.
    Returns ``[]`` when there are no findings — the single source rendered by
    both the console summary and ``run_task`` warnings.
    """
    lines: List[str] = []
    for f in findings or []:
        dim = f.get("dimension", "")
        issue = (f.get("issue") or "").strip()
        if not issue:
            continue
        prefix = "Minor: " if f.get("severity") == "minor" else ""
        lines.append(f"{prefix}[{dim}] {issue}")
    return lines


def display_plan_summary(result: Dict[str, Any]) -> None:
    """
    Parses the agent's results and prints a structured, pretty-printed 
    summary to the console for human review.
    """
    # 1. Error Handling
    if result.get("error"):
        print(f"\n❌ Agent finished with an error: {result['error']}\n")
        return

    # 2. Structure Validation
    experiments = result.get("proposed_experiments")
    if not experiments or not isinstance(experiments, list):
        print("\n⚠️  The agent returned a result, but no experiments were found.")
        # Optional: Print raw if debugging needed
        # print(json.dumps(result, indent=2))
        return

    # 3. Header
    print("\n" + "="*80)
    print("✅ PROPOSED EXPERIMENTAL PLAN")
    print("="*80)

    # 4. Loop through Experiments
    for i, exp in enumerate(experiments, 1):
        
        # --- Name & Hypothesis ---
        print(f"\n🔬 EXPERIMENT {i}: {exp.get('experiment_name', 'Unnamed Experiment')}")
        print("-" * 80)
        print(f"\n> 🎯 Hypothesis:\n> {exp.get('hypothesis', 'N/A')}")

        # --- Experimental Steps (Numbered) ---
        print("\n--- 🧪 Experimental Steps ---")
        steps = exp.get('experimental_steps', [])
        if steps:
            for j, step in enumerate(steps, 1):
                # Remove leading numbers/bullets provided by LLM
                # Regex removes "1.", "1 -", "1)", etc.
                clean_step = re.sub(r'^[\d\-\.\)\s]+', '', str(step)).strip()
                print(f" {j}. {clean_step}")
        else:
            print("  (No steps provided)")
        
        # --- Equipment ---
        print("\n--- 🛠️  Required Equipment ---")
        equipment = exp.get('required_equipment', [])
        if equipment:
            # Print as a clean comma-separated list if short, or bullets if long
            if len(equipment) > 5:
                for item in equipment: print(f"  * {item}")
            else:
                print(f"  {', '.join(equipment)}")
        else:
            print("  (No equipment specified)")

        # --- Outcome & Justification (Critical for Review) ---
        print("\n--- 📈 Expected Outcome ---")
        print(f"  {exp.get('expected_outcome', 'N/A')}")

        print("\n--- 💡 Justification ---")
        print(f"  {exp.get('justification', 'N/A')}")
        
        # --- Source Documents ---
        print("\n--- 📄 Source Documents ---")
        sources = exp.get('source_documents', [])
        if sources:
            for src in sources:
                print(f"  - {src}")
        else:
            print("  (No sources listed)")

        # --- Code Indicator (If generated) ---
        if "implementation_code" in exp:
            print("\n--- 💻 Implementation Code ---")
            print("  ℹ️  Plan includes implementation script.")

    # --- Plan-level caveats (advisory; from the critic — the plan is unchanged) ---
    caveats = format_caveats(result.get("critic_findings"))
    if caveats:
        print("\n" + "-"*80)
        print("⚠️  Caveats & Potential Limitations")
        for c in caveats:
            print(f"  • {c}")

    print("\n" + "="*80)


def display_plan_candidates(candidates: List[Dict[str, Any]],
                            judge: Dict[str, Any],
                            selected: int,
                            report_paths: Optional[List[str]] = None,
                            pick_caveats: Optional[List[str]] = None) -> None:
    """
    Print the best-of-N candidate summary cards for human review.

    Extractive by design: every line comes verbatim from fields the plan JSON
    already carries (name / hypothesis / expected_outcome / justification) plus
    the judge's scores — no extra LLM call, so a card cannot drift from the
    plan it describes. This ONE stdout block serves both surfaces: the CLI
    user reads it directly, and the UI parser splits it into cards + a radio
    (gated on the "PLAN CANDIDATES" header and the card markers below — keep
    them stable).
    """
    scores_by_idx = {s.get("candidate"): s for s in judge.get("scores", [])
                     if isinstance(s, dict)}

    print("\n" + "=" * 80)
    print(f"🧭 PLAN CANDIDATES — {len(candidates)} distinct strategies "
          f"(judge pick: Candidate {selected})")
    print("=" * 80)

    for i, cand in enumerate(candidates, 1):
        exp = (cand.get("proposed_experiments") or [{}])[0]
        marker = "  ← judge pick" if i == selected else ""
        print(f"\n── Candidate {i}: {exp.get('experiment_name', 'Unnamed')} ──{marker}")
        print(f"🎯 Hypothesis: {exp.get('hypothesis', 'N/A')}")
        print(f"📈 Expected outcome: {exp.get('expected_outcome', 'N/A')}")
        print(f"💡 Justification: {exp.get('justification', 'N/A')}")
        s = scores_by_idx.get(i)
        if s:
            print("🧑‍⚖️ Judge: "
                  f"groundedness {s.get('groundedness', '?')}/5 · "
                  f"testability {s.get('testability', '?')}/5 · "
                  f"actionability {s.get('actionability', '?')}/5 · "
                  f"feasibility {s.get('feasibility', '?')}/5 · "
                  f"info-gain {s.get('information_gain', '?')}/5")
            if s.get("comment"):
                print(f"   {s['comment']}")
        if report_paths and i <= len(report_paths):
            print(f"📄 Full plan: {report_paths[i - 1]}")

    if judge.get("reasoning"):
        print(f"\n🧑‍⚖️ JUDGE REASONING:\n  {judge['reasoning']}")
    if pick_caveats:
        print(f"\n⚠️  Caveats on the pick (Candidate {selected}):")
        for c in pick_caveats:
            print(f"  • {c}")
    print("\n" + "=" * 80)


def get_candidate_selection(n_candidates: int, judge_pick: int) -> int:
    """
    Stage-1 selection prompt: accept the judge's pick (ENTER) or override by
    index. Selection only — free-text refinement belongs to the existing
    stage-2 plan-approval prompt, which runs on whichever candidate wins here.

    The prompt string carries the "accept plan candidate <pick>" marker the UI
    parser gates on; the reply contract is the analysis best-of-N one (bare
    digit, or empty for the pick). Anything unparseable falls back to the
    judge's pick — never blocks.
    """
    print("\n" + "-" * 60)
    print("📝 SELECT A PLAN CANDIDATE")
    print("-" * 60)
    print(f"• Press [ENTER] to accept the judge's pick (Candidate {judge_pick}).")
    print(f"• Or type a candidate number (1-{n_candidates}) to choose a "
          "different plan.")

    reply = input(f"\n> Selection (ENTER to accept plan candidate {judge_pick}): ").strip()

    if reply.isdigit() and 1 <= int(reply) <= n_candidates:
        return int(reply)
    if reply:
        print(f"  - ℹ️  '{reply}' is not a candidate number — keeping the "
              f"judge's pick (Candidate {judge_pick}).")
    return judge_pick


def get_user_feedback() -> Optional[str]:
    """
    Pauses execution to get user input via the CLI. 
    Returns None if the user just presses ENTER (indicating approval).
    """
    print("\n" + "-"*60)
    
    print("📝 REQUESTING FEEDBACK")
    print("-" * 60)
    print("Review the plan and any caveats above.")
    print("• To APPROVE as-is: Press [ENTER] directly.")
    print("• To REVISE (e.g. address the caveats, or your own changes): "
          "Type instructions and press [ENTER].")
    
    feedback = input("\n> Instruction: ").strip()
    
    if not feedback:
        return None # User accepted the plan
        
    return feedback


def get_dataset_description(filename: str) -> str:
    """
    Interactive prompt when metadata is missing.
    """
    print("\n" + "!"*60)
    print(f"⚠️  MISSING METADATA FOR: {filename}")
    print("!"*60)
    print("The agent needs context to understand columns/units in this file.")
    print("• Option 1: Press [ENTER] to skip (Agent will guess based on headers).")
    print("• Option 2: Type a brief description (e.g., 'Yield results from Suzuki coupling').")
    
    desc = input("\n> Context: ").strip()
    return desc