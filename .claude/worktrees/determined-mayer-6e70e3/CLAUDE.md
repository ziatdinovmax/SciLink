# SciLink — Architecture Notes

Forward-looking design decisions and conventions. Intended for AI assistants
and contributors working on the orchestrator stack. Codebase tour and
per-module docs are elsewhere; this file is about *direction*.

## The mode universe is fixed at three

Every chat-driven orchestrator in SciLink falls into one of three modes —
this is a settled architectural commitment, not a refactoring waypoint:

| Mode | Class | Domain |
|---|---|---|
| `analyze` | `AnalysisOrchestratorAgent` | Experimental data analysis (microscopy, spectroscopy, …) |
| `plan` | `PlanningOrchestratorAgent` | Experimental campaign design |
| `simulate` | `SimulationOrchestratorAgent` *(not yet built — stub at `cli/simulate.py`)* | Computational simulations (DFT today, LAMMPS later) |

Anything in scientific workflow falls under one of these three. There will
**not** be a fourth mode. Future capability growth happens *inside* one of
the three, or as a meta-agent on top (see below).

## Why no `BaseChatOrchestrator` refactor

The three orchestrators share a near-identical chat-loop / message-history /
MCP / autonomy / checkpoint shape (~600 lines each). Reflexively extracting
a base class is tempting and **not what we want at this stage**. The rule
of three says abstract on the third copy when the duplication actually
hurts; bug-fix propagation across three files is acceptable cost.

The trigger to do the refactor is "fixes are diverging across copies" or
"a fourth case appears" — neither holds. The fourth case won't appear
(the universe is fixed at three), so the only legitimate trigger is
maintenance pain. We have not hit it.

When building `SimulationOrchestratorAgent`, copy the structure of
`AnalysisOrchestratorAgent`. Don't refactor the other two.

## What the simulate orchestrator looks like

Structure-centric, iterative, two-surface. **Different from analyze mode**
in three ways: no data file required to start, structure-centric
(not analysis-driven), and includes a post-run feedback loop.

### Tool surface

```
Structure phase
  generate_structure(description)             # one cycle, no validator loop
  validate_structure(path)                    # standalone, post-edit re-run
  refine_structure(path, feedback)            # one refinement cycle
  view_structure(path)                        # 3-axis renders

Inputs phase
  generate_vasp_inputs(poscar, request, method='llm'|'atomate2')
  validate_incar(incar, request)              # literature validation
  apply_incar_improvements(...)

Post-run (currently orphaned in the codebase)
  analyze_vasp_output(output_dir)             # OUTCAR / vasprun.xml summary
  suggest_incar_fixes(log_path)               # wraps existing VaspUpdater

Pipeline shortcut
  run_complete_dft_workflow(description)      # what analyze mode exposes today

Session
  list_generated_structures()
  compare_structures(path_a, path_b)
  set_default_calc_params(...)
```

### Session layout

Structure-centric, not analysis-centric:

```
simulate_session_YYYYMMDD_HHMMSS/
├── structures/
│   └── <structure_slug>/
│       ├── POSCAR / INCAR / KPOINTS
│       ├── script_*.py
│       ├── POSCAR_view_{x,y,z}.png
│       └── outputs/        # user drops VASP run results here
├── chat_history.json
├── checkpoint.json
└── session_log.txt
```

### Two surfaces, one agent

Each orchestrator exposes both an interactive and a non-interactive entry
point sharing the same state and tool registry:

```python
class SimulationOrchestratorAgent:
    def run_chat(self, user_input: str) -> str:
        """Interactive — CLI / UI."""
    def run_task(self, task: str, context: dict) -> dict:
        """Non-interactive — autonomous mode under the hood. Returns:
            {summary, files_produced, key_findings, suggested_followups}
        Used by the future meta agent to delegate sub-tasks.
        """
```

Plan and analyze will eventually need `run_task` too. Not a refactor —
just a contract to honor when *adding* surfaces.

## The meta agent (future)

A meta agent will sit on top of the three modes so users don't switch
manually. It is **not a fourth mode**; it's an orchestrator-of-orchestrators
with a different role (router + context bridge).

### Pattern: agent-as-tool

```
Meta tool registry
  delegate_to_analysis(task, context)     → AnalysisOrchestratorAgent.run_task
  delegate_to_planning(task, context)     → PlanningOrchestratorAgent.run_task
  delegate_to_simulation(task, context)   → SimulationOrchestratorAgent.run_task
  bridge_context(from_session, to_topic)  → extract findings/files between modes
  summarize_session_state()               → cross-child status
```

The meta agent presents a single conversational surface to the user;
under the hood each delegation spins up a child sub-session with its
own chat history, files, checkpoint — all nested under the meta-session
directory.

Because the meta consumes children through their `run_task` contract
(not through inherited internals), no base class is required. The
contract is duck-typed; what the children share is *interface shape*,
not *implementation*.

## Sequencing — hard features first, UI later

Engineering philosophy on this codebase: implement load-bearing logic
first, surface it in CLI / UI later. Reasons:

- UI shaped against an unbuilt feature gets reshaped
- Backend logic is independently testable; UI work depends on it
- Simulating the user's flow without a backend produces wishful UX

Concretely, when the simulate orchestrator work starts, the order is:

1. `SimulationOrchestratorAgent` (copy of analyze structure) +
   `simulate_orchestrator_tools.py` with the granular DFT tool registry
2. `scilink simulate` CLI flesh-out (replace the "Coming Soon!" stub)
3. HPC backend (`scilink/hpc/` — `Connection`, `Scheduler`) — see PR #140;
   self-contained module, no orchestrator dependency
4. HPC tools on the orchestrator (`submit_vasp_job`, `check_job_status`,
   `download_results`, …) wrapping #3
5. UI — sidebar mode, chat panel, possibly a wizard surface coexisting
   with the chat surface

The meta agent comes after all three child orchestrators are stable.

## Connection between modes today

Analyze mode connects to DFT via two tools in
`analysis_orchestrator_tools.py`:

- `recommend_dft_structures` — generates DFT structure recommendations
  from cached analysis text via `RecommendationAgent`
- `run_dft_workflow` — runs the full `DFTOrchestrator` pipeline; takes a
  `structure_description` (free text) or `recommendation_index` (pulls
  from stored recommendations)

When the simulate orchestrator ships, **these stay**. Analyze mode keeps
the one-shot pipeline tool because that's the right shape for "I'm done
analyzing, prepare a calc". Simulate mode adds *granular* alternatives
for iterative work. Don't replace `run_dft_workflow`; add alongside.

## Conventions for prompt patches

When live traces surface bad LLM behavior:

- Encode the **principle** in one short sentence, not a list of phrases
  pulled from the trace. Example-driven prompts overfit and accumulate
  dead weight.
- Trace specifics belong in the commit message and PR description, not
  the prompt itself.
- If a single sentence isn't enough, the rule probably needs structural
  support (a schema field, a separate validation pass) rather than
  more prose.

## Branch hygiene

Non-trivial features start on a dedicated branch off `main`
(`git checkout -b <feature-name>`), never on `main` directly. UI / CLI
exposure for a backend feature can land in the same branch as the
backend, or split into a follow-up PR — depends on review surface area.
