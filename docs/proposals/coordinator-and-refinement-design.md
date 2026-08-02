# Proposal: coordinator vocabulary + refinement-loop design

**Status:** PROPOSAL — for discussion (Sarah ↔ Maxim). Not canonical. Nothing
here is in CLAUDE.md yet; once aligned, the direction-level slice lands in
CLAUDE.md and the detailed design becomes a `docs/` design note.

**Motivation:** "Orchestrator" became overloaded. Several deterministic
helpers borrowed the name without being orchestrators (`StructureOrchestrator`,
`DFTOrchestrator`, `LAMMPSOrchestrator`). Building the engine-neutral simulate
stack surfaced a missing piece — a supervised-execution (refinement) loop —
that `LAMMPSOrchestrator` was quietly standing in for. This proposes a settled
vocabulary and a design for that loop.

---

## Part 1 — Coordinator vocabulary

Reserve the suffix `Orchestrator` for the first row only.

| Shape | Control flow | Role | Examples |
|---|---|---|---|
| **Orchestrator** | LLM-chosen, interactive, session-stateful | route work, own autonomy/skills/MCP/checkpoints | meta (top) + analyze / plan / simulate |
| **Foundation agent** | fixed internal stages, skill-specialized | do one domain's work | PeriodicDFT, MDSimulation, MLIP, CurveFitting |
| **Pipeline** | fixed, deterministic, headless | reproducible multi-step flow | `run_complete_workflow`; `StructureOrchestrator`* |
| **Refinement loop** | fixed cycle (run → critic → fix) | supervise execution to convergence | `LAMMPSOrchestrator`'s real shape* |

\* mis-named today — see Part 4.

**Orchestrator (definition).** A chat-driven, session-stateful coordinator
where the **LLM chooses the control flow** (which sub-agents/tools to invoke),
owning autonomy, skills, MCP, and checkpoints. The universe is fixed: the
**meta orchestrator** (top; decides which mode to use and how) plus the **three
mode orchestrators** (analyze / plan / simulate). Nothing else is an
Orchestrator.

The defining line is *LLM-chosen control flow + stateful session*. (Whether the
formal contract should be `chat()` + `run_task()` or `chat()` alone is an open
question — see Part 5.)

---

## Part 2 — Who authors what

The boundary between fixed scaffolding and filled-in behavior, by author:

| Layer | Owner | What |
|---|---|---|
| Control flow | **source code** | orchestrator chat loops; pipeline sequences; the refinement-loop skeleton; the contracts (foundation-agent, critic, `Executor`, `RefinementPolicy`) |
| Engine/technique knowledge | **skills** | prose + optional tools; human-authored or grown by graduation |
| Per-task artifacts | **runtime LLM** | input files, fixes, analysis scripts |
| Configuration | **user (scientist)** | engine, run command, resources, autonomy level — never control flow |

Consequence for extensibility:
- A **new engine within a scale** (e.g. QE in DFT, GROMACS in MD) = a **skill
  bundle**. No source. Addable by a user or grown by graduation.
- A **new scale** (e.g. molecular DFT) = a **new foundation agent** (source) +
  a skill, reusing the same pipeline/loop unchanged.

---

## Part 3 — The refinement loop

The loop skeleton is **source and deterministic** (reproducible — the right
shape for benchmarking). The *judgments inside* it are LLM, grounded in skills.

```
run_refinement(scale, engine, structure, goal, executor, policy, ctx):
    inputs   = generate(scale, engine, structure, goal)   # foundation agent
    verdict  = InputValidator.validate(inputs, ...)        # pre-run critic
    inputs   = policy.approve_change(inputs, verdict, ctx) # gate (1)
    if inputs is None: return aborted

    while True:
        result  = executor.run(inputs)                    # external run
        verdict = RunCritic.assess(result, ...)           # post-run critic
        ctx.record(inputs, result, verdict)
        if policy.after_run(result, verdict, ctx) == STOP: break   # gate (2)
        fixed   = apply(verdict.suggested_fixes)
        inputs  = policy.approve_change(fixed, verdict, ctx)       # gate (1) reused
        if inputs is None: break
    return result, ctx.history
```

### `RefinementPolicy` — two methods

A refinement loop has exactly two decisions; the policy answers them. The same
gate handles both the pre-run inputs and each proposed fix, because
`InputValidator` and `RunCritic` both produce *a verdict + a proposed input*.

```python
class RefinementPolicy:
    def approve_change(self, proposed_inputs, verdict, ctx) -> dict | None:
        """Gate a proposed input set — used at the pre-run gate (verdict from
        InputValidator) AND after each fix (verdict from RunCritic). Returns
        the inputs to run (possibly edited), or None to abort/stop."""

    def after_run(self, result, verdict, ctx) -> CycleDecision:   # STOP | REFINE
        """Continuation: given the finished run + critic verdict, stop or refine."""
```

`ctx` (a `RefinementContext`) carries shared state: `cycle`, `history`,
`max_cycles`/budget, `autonomy`, and an `interact(...)` handle (SciLink's
existing human-feedback mechanism). **The policy decides whether to call
`interact`** — so autonomy lives inside the policy, reusing existing machinery
rather than inventing a parallel one.

### Autonomy levels ARE the built-in policies

The three autonomy levels are three `RefinementPolicy` implementations — so
"different autonomy → different feedback/interaction" falls out for free:

| | `approve_change` | `after_run` |
|---|---|---|
| **co-pilot** | `interact`: human approves/edits/aborts | `interact`: human decides stop vs refine |
| **autopilot** | apply error-severity adjustments; `interact` only on risky/low-confidence changes | refine while improving + under budget; stop if converged/stalled |
| **autonomous** | apply error-severity adjustments, proceed | refine to `verdict.good` or budget |

The loop **reads the session's existing autonomy level** and uses the matching
built-in policy — it does not reinvent autonomy.

### Three levels of user control (config vs. policy)

- **(a) fixed** — rejected; too rigid.
- **(b) config** — the everyday path: pick autonomy + set knobs (`max_cycles`,
  thresholds, auto-apply vs. approve). No code.
- **(c) pluggable policy** — a power user overrides *one* method of a built-in
  policy for non-standard behavior ("stop when lattice is within 1% of target"
  → override `after_run`; "auto-apply only timestep fixes" → override
  `approve_change`). Never touches the loop.

### Metrics (optional, engine-specific)

No universal quality metric exists (DFT cares about forces/energy, MD about
energy-drift/lattice). So:
- **Built-in policies need no metric** — they stall/continue on the **ordinal
  verdict** (`good > warning > poor > needs_fixes`).
- **Custom policies that want a number get one optionally** — the critic may
  emit an engine-specific `metrics` dict (rides in the verdict, accumulates in
  `ctx.history`); a custom `after_run` reads named metrics for target-based
  stops; built-ins ignore them.

### `Executor` contract

The one genuinely new abstraction. "Run these inputs → give back an output
dir," with implementations for **local / container / HPC**. Today this is
tangled inside `LAMMPSOrchestrator` (it knows `lammps_command` + container) and
absent on the DFT side (DFT runs externally; the critic analyzes after). An
engine-neutral loop needs a clean `Executor` so the same loop works regardless
of where the run happens. The run command + environment are user/config.

---

## Part 4 — Renames the code owes this vocabulary

- `StructureOrchestrator` → `StructurePipeline` (it is a pipeline).
- `DFTOrchestrator` → delete (already a shim over `run_complete_workflow`).
- `LAMMPSOrchestrator` → fold into the engine-neutral refinement loop above;
  stop being a per-engine "Orchestrator."

---

## Part 5 — Open questions for Maxim

1. **Orchestrator contract:** is it formally `chat()` + `run_task()`, or
   `chat()` alone? (Meta has `chat` but not `run_task`.) The litmus
   ("LLM-chosen control flow + stateful session") holds either way; this is
   about the formal definition.
2. **Executor:** is "run inputs → output dir" with local/container/HPC
   implementations the right abstraction, living in source as part of the loop?
3. **Where the loop is driven from:** deterministic source loop (LLM only
   inside the slots; the chat orchestrator *invokes* it as one tool), vs. the
   chat orchestrator driving run/check/fix as separate turns. This proposal
   assumes deterministic-source (for reproducibility); worth confirming.
4. **CLAUDE.md vs. CONTRIBUTING.md:** CLAUDE.md = architecture *direction*
   (keep tight); detailed design → `docs/`. Separately: is a CONTRIBUTING.md
   worth adding for contribution *process* (setup, tests, branch/PR/commit
   conventions) currently scattered in CLAUDE.md?
