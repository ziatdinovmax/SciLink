from typing import Dict, Any, Optional, List
import json
import re
import textwrap
from pathlib import Path

DELIVERABLES_MANIFEST = "deliverables.json"


def format_path(path: Any) -> str:
    """A file path as an absolute string, ready to copy into a browser.

    Terminal hyperlinks (OSC-8) were tried and removed: detection was wrong
    twice — leaking ']8;;file:///...' into terminals that do not render the
    escape — and a plain absolute path is already selectable and pasteable,
    which is all this needed to be.
    """
    return str(Path(path).resolve())


def record_deliverable(base_dir: Any, path: Any, title: str = "",
                       deliverable: bool = False) -> None:
    """Note a file the session produced, and whether it is a DELIVERABLE.

    A deliverable is the artifact that answers the user's request. Only the
    agent knows which one that is — the filename is chosen at request time
    (`top3_priority_brief.md`), so no allow-list can anticipate it. The
    manifest lets the CLI star it and the UI embed it, while everything else
    is still listed so a forgotten flag can never hide a file.
    """
    try:
        manifest = Path(base_dir) / DELIVERABLES_MANIFEST
        entries = []
        if manifest.exists():
            entries = json.loads(manifest.read_text() or "[]")
        entry = {"path": str(Path(path).resolve()),
                 "title": title or Path(path).stem.replace("_", " ").title(),
                 "deliverable": bool(deliverable)}
        entries = [e for e in entries if e.get("path") != entry["path"]]
        entries.append(entry)
        manifest.write_text(json.dumps(entries, indent=2))
    except Exception:  # noqa: BLE001 - bookkeeping must never break a tool
        pass


def load_deliverables(base_dir: Any) -> List[Dict[str, Any]]:
    """Every recorded artifact under a session root (manifests may exist per
    orchestrator, e.g. <meta>/planning/deliverables.json)."""
    out: List[Dict[str, Any]] = []
    seen = set()
    try:
        for manifest in Path(base_dir).rglob(DELIVERABLES_MANIFEST):
            for e in json.loads(manifest.read_text() or "[]"):
                p = e.get("path")
                if p and p not in seen:
                    seen.add(p)
                    out.append(e)
    except Exception:  # noqa: BLE001
        pass
    return out


def display_files_produced(paths: List[str], base_dir: Any = None) -> None:
    """Print the turn's output files as absolute, pasteable paths.

    Hunting for an agent-written file through a session tree is the single
    most common friction after a run — the assistant's prose cites a short
    relative path and the reader is left to `find` it. This block is
    deterministic (it cannot drift from what was actually written) and
    stars whatever the agent marked as the answer to the request.
    """
    # A turn touches far more than it delivers: session state, the campaign
    # checkpoint, the reviewer's scratch render, per-candidate reports, the
    # literature dump. Listing those as "produced" makes the reader scan
    # eleven lines to find the three that matter, so they collapse to a
    # count and the deliverables lead.
    SUPPORTING_NAMES = {DELIVERABLES_MANIFEST, "planning_state.json",
                        "checkpoint.json", "chat_history.json",
                        "plan_preview.html", "session_log.txt"}
    SUPPORTING_STEMS = ("literature_search", "planning_state")

    def _supporting(p: Path) -> bool:
        return (p.name in SUPPORTING_NAMES
                or p.parent.name == "plan_candidates"
                or any(p.stem.startswith(s) for s in SUPPORTING_STEMS))

    real = [Path(p) for p in paths if p and Path(p).exists()]
    if not real:
        return
    marked = {e["path"]: e for e in load_deliverables(base_dir or "")
              if e.get("deliverable")}
    shown = [p for p in real
             if str(p.resolve()) in marked or not _supporting(p)]
    hidden = [p for p in real if p not in shown]
    if not shown:
        return

    print("\n" + "-" * 60)
    print(f"📎 FILES PRODUCED THIS TURN ({len(shown)})")
    print("-" * 60)
    for p in sorted(shown, key=lambda x: (str(x.resolve()) not in marked,
                                          x.name)):
        info = marked.get(str(p.resolve()))
        size = p.stat().st_size
        unit = f"{size/1024:.0f} KB" if size >= 1024 else f"{size} B"
        star = "★ " if info else "  "
        label = f" — {info['title']}" if info else ""
        print(f" {star}{format_path(p)}  ({unit}){label}")
    if hidden:
        print(f"   (+{len(hidden)} supporting files: candidates, literature, "
              f"session state)")
    if marked:
        print("  ★ = the deliverable(s) for this request")


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


def _print_program(steps: List[Any], numbered: bool) -> None:
    """Render the program/steps list.

    Two structural fixes that apply in both modes: blank entries are spacing
    in the model's output, not steps (they were being NUMBERED, producing
    empty '2.' / '4.' rows), and a banner entry ('==== DOMAIN 1 ... ====')
    is a section divider, not a step. Long entries wrap with a hanging
    indent so the list reads as a list.

    ``numbered=False`` (ideation) leaves the author's own labels — 'PS-1',
    'BA-3', 'BRIDGE-2', 'S4' — to do the numbering: an imposed 1..N counter
    fights those labels and implies a sequential protocol that a research
    portfolio does not have.
    """
    if not steps:
        print("  (Nothing listed)")
        return
    n = 0
    for step in steps:
        raw = str(step).strip()
        if not raw:
            continue
        if re.fullmatch(r"={3,}.*={3,}", raw):
            print(f"\n  ▸ {raw.strip('=').strip()}")
            continue
        if numbered:
            n += 1
            body = re.sub(r'^[\d\-\.\)\s]+', '', raw).strip()
            lead, cont = f" {n}. ", "    "
        else:
            body = raw
            lead, cont = "  • ", "    "
        print(textwrap.fill(body, width=78, initial_indent=lead,
                            subsequent_indent=cont))


def concept_title(c: Dict[str, Any], fallback_index: Any = "") -> str:
    """Best available one-line title for a concept entry.

    Authors do not always emit `title`: when the objective mandates its own
    element list (seen live: "a_title_and_system", "b_operando_only_question",
    ...) the model uses THOSE as the keys, and every card rendered as
    "Untitled". Resolve in order: an exact `title`, any key whose name
    mentions title/name, then the first short-ish string value — so the one
    scannable line per direction is never empty.
    """
    if not isinstance(c, dict):
        return str(c)[:120]
    exact = c.get("title")
    if isinstance(exact, str) and exact.strip():
        return exact.strip()
    for k, v in c.items():
        if not isinstance(v, str) or not v.strip():
            continue
        if "title" in k.lower() or "name" in k.lower():
            t = v.strip()
            # "TITLE: 'Catch the injection' — system description..." -> the
            # quoted/leading clause is the title; keep the rest for the body.
            t = re.sub(r"^TITLE\s*:\s*", "", t, flags=re.I).strip()
            m = re.match(r"^['\"“](.+?)['\"”]", t)
            if m:
                return m.group(1).strip()
            return t.split(". ")[0][:120].strip()
    for k, v in c.items():
        if k in ("id", "tier") or not isinstance(v, str) or not v.strip():
            continue
        return v.strip().split(". ")[0][:120]
    return f"Direction {fallback_index}" if fallback_index else "Untitled"


def humanize_key(key: str) -> str:
    """Render an author-chosen field name for humans.

    Objective-mandated element names arrive as `a_title_and_system` /
    `f_ai_closed_loop`; strip the ordering prefix and the underscores so the
    card reads as prose rather than as JSON keys.
    """
    k = re.sub(r"^[a-z]_", "", str(key))
    return k.replace("_", " ").strip().capitalize()


def _print_concepts(concepts: List[Any]) -> None:
    """Render a `concepts` portfolio: one block per research direction.

    The portfolio is the deliverable of an ideation run, so each direction
    gets its own labelled block rather than being flattened into a step
    list. Unknown keys are printed too — authors add elements the objective
    asked for, and silently dropping them would lose the answer.
    """
    KNOWN = ("id", "tier", "title", "hypothesis", "rationale", "novelty",
             "details")
    for n, c in enumerate(concepts, 1):
        if not isinstance(c, dict):
            print(f"\n  ▸ {n}. {str(c)[:200]}")
            continue
        label = c.get("id") or str(n)
        tier = f"  ·  tier {c['tier']}" if c.get("tier") else ""
        print(f"\n  ▸ {label}: {concept_title(c, n)}{tier}")
        for key in ("hypothesis", "rationale", "novelty"):
            if c.get(key):
                print(f"     {key.capitalize()}:")
                print(_wrap_field(c[key], indent="       "))
        details = c.get("details")
        if isinstance(details, list) and details:
            for d in details:
                print(_wrap_field(str(d), indent="       - "))
        elif details:
            print(_wrap_field(str(details), indent="       "))
        for key, val in c.items():
            if key not in KNOWN and val:
                print(f"     {humanize_key(key)}:")
                if isinstance(val, list):
                    for item in val:
                        print(_wrap_field(str(item), indent="       - "))
                else:
                    print(_wrap_field(val, indent="       "))


def display_plan_summary(result: Dict[str, Any],
                         ideation: bool = False,
                         report_path: Optional[str] = None) -> None:
    """
    Parses the agent's results and prints a structured, pretty-printed
    summary to the console for human review.

    ``ideation=True`` switches the vocabulary from a bench protocol
    ("EXPERIMENT 1", "Experimental Steps", "Required Equipment") to the
    research-portfolio wording the ideation dossier already uses
    ("RESEARCH DIRECTION", "Proposed program", "Key capabilities"), and
    drops the imposed step numbering. The plan JSON schema is identical
    either way — only the presentation differs.
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
    print("✅ PROPOSED RESEARCH DIRECTIONS" if ideation
          else "✅ PROPOSED EXPERIMENTAL PLAN")
    print("="*80)
    # Offered at the TOP: these blocks run to thousands of words in a
    # terminal, and the same content is far easier to read in a browser.
    if report_path:
        print(f"\n📄 Full report (open in a browser): {report_path}")

    # 4. Loop through Experiments
    multi = len(experiments) > 1
    for i, exp in enumerate(experiments, 1):

        # --- Name & Hypothesis ---
        # A single entry needs no ordinal — an ideation run routinely
        # returns ONE entry carrying a whole portfolio, where 'EXPERIMENT 1'
        # reads as the first of a series that does not exist.
        if ideation:
            head = f"💡 RESEARCH DIRECTION{f' {i}' if multi else ''}"
        else:
            head = f"🔬 EXPERIMENT{f' {i}' if multi else ''}"
        print(f"\n{head}: {exp.get('experiment_name', 'Unnamed')}")
        print("-" * 80)
        print("\n🎯 Hypothesis:")
        print(_wrap_field(exp.get('hypothesis')))

        # --- Portfolio of directions, when the plan carries one ---
        concepts = exp.get("concepts")
        if isinstance(concepts, list) and concepts:
            print(f"\n--- 🧠 Research Directions ({len(concepts)}) ---")
            _print_concepts(concepts)

        # --- Program / steps ---
        steps = exp.get('experimental_steps', [])
        if steps or not concepts:
            # With a portfolio present the steps list is shared protocol, if
            # anything — label it so it does not read as the whole plan.
            print("\n--- 🧭 Shared Protocol ---" if (ideation and concepts)
                  else "\n--- 🧭 Proposed Program ---" if ideation
                  else "\n--- 🧪 Experimental Steps ---")
            _print_program(steps, numbered=not ideation)

        # --- Equipment ---
        print("\n--- 🛠️  Key Capabilities ---" if ideation
              else "\n--- 🛠️  Required Equipment ---")
        equipment = exp.get('required_equipment', [])
        if equipment:
            # Print as a clean comma-separated list if short, or bullets if long
            if len(equipment) > 5:
                for item in equipment: print(f"  * {item}")
            else:
                print(f"  {', '.join(equipment)}")
        else:
            print("  (None specified)")

        # --- Outcome & Justification (Critical for Review) ---
        print("\n--- 📈 Expected Outcomes ---" if ideation
              else "\n--- 📈 Expected Outcome ---")
        print(_wrap_field(exp.get('expected_outcome')))

        print("\n--- 💡 Rationale ---" if ideation
              else "\n--- 💡 Justification ---")
        print(_wrap_field(exp.get('justification')))
        
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


def _wrap_field(text: Any, width: int = 78, indent: str = "   ") -> str:
    """Wrap a long plan field into an indented block.

    Candidate hypotheses and justifications run to several hundred words;
    unwrapped they render as one edge-to-edge wall in which the field
    labels and candidate boundaries are invisible. Indenting the body
    subordinates it to its label. Paragraph breaks in the source are kept.
    """
    s = str(text if text not in (None, "") else "N/A")
    paras = [p.strip() for p in s.split("\n") if p.strip()] or [s]
    return "\n\n".join(
        textwrap.fill(p, width=width, initial_indent=indent,
                      subsequent_indent=indent)
        for p in paras
    )


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

    n = len(candidates)
    for i, cand in enumerate(candidates, 1):
        exp = (cand.get("proposed_experiments") or [{}])[0]
        marker = "  ← judge pick" if i == selected else ""
        # Each candidate's body runs to several hundred words, so a bare
        # one-line header disappears between the walls of text. Lead with a
        # banded rule carrying the position ('CANDIDATE 2 of 3'); the
        # '── Candidate N: <name> ──' line below it stays verbatim — the UI
        # radio parser gates on exactly that grammar.
        print(f"\n{'━' * 80}")
        print(f"  CANDIDATE {i} of {n}"
              + ("   ★ JUDGE PICK" if i == selected else ""))
        print("━" * 80)
        # Marker line keeps its exact grammar AND its trailing pick suffix:
        # the UI radio parser reads the index/name from it and detects the
        # pick from what trails the closing '──'.
        print(f"── Candidate {i}: {exp.get('experiment_name', 'Unnamed')} ──{marker}")
        for icon, title, key in (("🎯", "Hypothesis", "hypothesis"),
                                 ("📈", "Expected outcome", "expected_outcome"),
                                 ("💡", "Justification", "justification")):
            print(f"\n{icon} {title}:")
            print(_wrap_field(exp.get(key)))
        s = scores_by_idx.get(i)
        if s:
            print("\n🧑‍⚖️ Judge: "
                  f"groundedness {s.get('groundedness', '?')}/5 · "
                  f"testability {s.get('testability', '?')}/5 · "
                  f"actionability {s.get('actionability', '?')}/5 · "
                  f"feasibility {s.get('feasibility', '?')}/5 · "
                  f"info-gain {s.get('information_gain', '?')}/5")
            if s.get("comment"):
                print(_wrap_field(s["comment"]))
        if report_paths and i <= len(report_paths):
            print(f"\n📄 Full plan: {report_paths[i - 1]}")

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