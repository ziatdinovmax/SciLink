# Proposal: MLIP is a potential source, not a simulation scale

Follow-up to [#429](https://github.com/ziatdinovmax/SciLink/issues/429), where the
answer was "potential source". This writes down what that implies, what already
exists, and what is missing — so the change can be argued about before it is
written.

> **Status: implemented in this PR.** `machine_learning_potentials` is removed
> from the scale axis; a `potential_selection` skill + LLM step (in
> `potential_selection.py`) chooses classical-FF vs MLIP for a
> `molecular_dynamics` task once the structure exists, and `simulation_pipeline`
> dispatches through it. The explicit scale is kept as a deprecation shim. See
> **Design decisions (resolved)** below. The routing benchmark already labels the
> four inorganic-dynamics cases `molecular_dynamics` (issue #429), so re-running
> `test_router` on this branch turns the four documented misses into hits.

## Context — the four scales are not the same kind of thing

`discover_scale_agents()` presents four scales:

| scale | what it names |
|---|---|
| `periodic_dft` | a physical regime — extended systems, electronic structure |
| `molecular_qc` | a physical regime — finite systems, electronic structure |
| `molecular_dynamics` | a physical regime — classical ensemble sampling |
| `machine_learning_potentials` | **how forces are evaluated** |

The first three answer "what physics is this?". The fourth answers "where do the
forces come from?" — a question that only arises *once* you have decided the
regime is dynamics.

The router's own scale description concedes this. `DEFAULT_SCALE_DESCRIPTIONS
["machine_learning_potentials"]` reads "best for … systems **without good
classical force fields**". That is a statement about force-field coverage for a
chemistry, not about the system's physics — and it is not knowable at routing
time, before anything has looked at what parameters exist for the species
involved.

## The evidence — the router applies that criterion, and it is load-bearing

A 26-query routing sweep (`SciLink-Benchmarks/benchmark/test_router.py`,
claude-opus-4-8, 2026-08-06) scored 22/26. Every one of the four misses went the
same way:

| query | expected | routed to |
|---|---|---|
| thermal conductivity of **amorphous** Si | `molecular_dynamics` | `machine_learning_potentials` |
| melting point of Cu | `molecular_dynamics` | `machine_learning_potentials` |
| water density profile at a TiO₂ surface | `molecular_dynamics` | `machine_learning_potentials` |
| Li diffusion coefficient in LiCoO₂ | `molecular_dynamics` | `machine_learning_potentials` |

The behaviour is coherent, not noisy. MD-sampling tasks on **molecular liquids
and biomolecules** (water density/self-diffusion, aqueous NaCl conductivity,
LiPF₆/EC viscosity, solvated glycine, peptide unfolding, benzene hydration free
energy) all routed to `molecular_dynamics` and all scored correct. MD-sampling
tasks on **inorganic solids, metals, and interfaces** all routed to
`machine_learning_potentials`.

That is exactly the documented criterion, applied consistently. **This is not a
routing bug.** The router is doing what its scale description tells it to. The
defect is that the decision is being asked at the wrong layer — and because it
is a scale, answering it commits the whole downstream pipeline.

There is a second, quieter cost. Splitting one regime across two scale labels
means benchmark ground truth has to pick a side for every dynamics query, and
neither side is defensible: label them `molecular_dynamics` and a correct MLIP
answer scores zero; label them `machine_learning_potentials` and a correct
classical-FF answer does. The routing benchmark currently resolves this by
excluding MLIP entirely, which is only tenable because the answer to #429 makes
those labels the right ones.

## What already exists

Most of the mechanism is in place, which is what makes this tractable:

- `MDSimulationAgent.run(..., potential: Optional[DeployedPotential])` already
  takes a deployed potential and dispatches to `_run_with_potential`.
- The comments there already state the intended layering — the agent can "*run* a
  simulation with any potential — classical FF or MLIP"
  (`md_simulation_agent.py:317`) and "doesn't care whether the potential is an
  MLIP" (`:441`).
- `MLIPAgent.deploy_pretrained()` returns `{"potential": DeployedPotential, …}`,
  annotated `# hand to the MD agent` (`mlip_agent.py:1030`).
- `ForceFieldAgent` is the classical counterpart.

So `DeployedPotential` is already the engine-neutral seam, and both providers
already exist on either side of it.

## What is missing

**Nothing chooses between them, and the pipeline does not use the seam.**

`simulation_pipeline.py` branches on `scale == "machine_learning_potentials"` and
calls `MLIPAgent.deploy_pretrained(..., runner="ase")` directly, taking the
generated ASE script as the pipeline's output. It never passes `potential=` to
`MDSimulationAgent` — `grep -n "potential=" simulation_pipeline.py` is empty. The
MLIP path bypasses the MD agent entirely rather than feeding it.

The missing piece is a **potential-selection step** for a `molecular_dynamics`
task: given the system's chemistry and the research goal, decide between a
classical force field and a deployed MLIP, and hand the result to
`MDSimulationAgent`. That decision has the information the router lacks — by then
the structure exists, so the species are known and force-field coverage is a
checkable fact rather than a guess.

## Shape of the change

1. `machine_learning_potentials` leaves the scale axis: dropped from
   `discover_scale_agents()`, `DEFAULT_SCALE_DESCRIPTIONS`, and the router's
   candidate set. `MLIPAgent` stays exactly as it is.
2. A potential-selection step owns FF-vs-MLIP for `molecular_dynamics` tasks and
   returns a `DeployedPotential`.
3. `simulation_pipeline`'s `molecular_dynamics` branch routes through it and
   calls `MDSimulationAgent.run(potential=…)` — using the seam that already
   exists instead of the parallel ASE path.

### Blast radius

Beyond the router and pipeline, `machine_learning_potentials` is referenced in
`structure_planner.py` (a `simulation_scale` field and its prompt enum),
`simulation_orchestrator.py` and `simulation_orchestrator_tools.py` (prompt text
pointing at `MLIPAgent`), and `available_software.py` (default detection map).
The structure planner is the one worth a second look: it carries the same
four-way axis, so it likely wants the same treatment rather than a mechanical
find-and-replace.

## Design decisions (resolved)

The four open questions, settled for this implementation:

- **Who owns potential selection?** A general **`potential_selection` skill** the
  LLM reasons over — not a new agent and not a method buried on
  `MDSimulationAgent`. The skill carries the FF-vs-MLIP knowledge; a thin
  `select_potential_family()` in `potential_selection.py` loads it, reads the
  structure's species, and asks the model. `simulation_pipeline`'s
  `molecular_dynamics` branch calls it and dispatches.
- **What decides FF-vs-MLIP?** **An LLM judgment guided by the skill**, in
  keeping with the rest of SciLink (skills inject knowledge, agents reason with
  it), rather than a hardcoded coverage rule — so the behaviour is extensible by
  editing the skill bundle and is benchmarkable the same way routing and
  selection already are. It defaults to MLIP when the model is unreachable or
  returns garbage (a universal potential degrades gracefully; a force field with
  an uncovered species fails silently).
- **Back-compatibility.** **Deprecation shim.** `scale="machine_learning_potentials"`
  still resolves — it maps to a `molecular_dynamics` task with a forced MLIP
  potential and logs a warning. No caller breaks; removable later.
- **Does `structure_planner`'s axis move too?** **Deferred to a follow-up.** Its
  `simulation_scale` field still lists `machine_learning_potentials`, which the
  pipeline honours via the shim, so nothing breaks; realigning the planner's
  four-way axis is a separate, non-urgent change. The orchestrator tool
  descriptions that name the MLIP scale are deferred with it.

## What this proposal explicitly does NOT claim

- **Not that the router is broken.** It is applying its own documented criterion
  correctly and consistently; the criterion is at the wrong layer.
- **Not that MLIP is less important.** Nothing about `MLIPAgent`'s capability
  changes — this is about when the choice is made and on what information.
- **No opinion on the selection policy itself.** How aggressively to prefer MLIP
  over a classical FF is a separate, empirical question, and one nothing in the
  suite measures today: `ForceFieldAgent` has no benchmark coverage at all, so
  "was this the right potential for this chemistry?" is currently unmeasured
  either way.
