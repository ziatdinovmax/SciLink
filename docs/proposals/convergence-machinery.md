# Proposal: numeric convergence machinery for scientifically-accurate values

> **Status: proposal — nothing is implemented in this PR.** It writes down the
> gap, what already exists to build on, and the shape of the change, so it can
> be argued about before it is written.

## Context — the benchmark measures accuracy, but only on one side

The simulation benchmark has two tracks. **Pipeline completion** —
`benchmark/test_e2e.py`, tiers T1 structure-built / T2 inputs-valid / T3
objective-met, where T3 is `final == "success" or refinement success`, i.e.
"the run executed and converged", with no value comparison. **Scientific
accuracy** — `benchmark/systems.py` `expected` answer keys + `GROUND_TRUTH.md`
+ the per-class tests, which score a *computed* value against a reference.

The accuracy track is **asymmetric**:

- **DFT is scored on values.** `benchmark/test_dft.py` runs VASP, reads the
  relaxed output, and scores the computed lattice constant within **2.5%** and
  adsorption energy within **0.30 eV** of `systems.py` references (with
  citations).
- **MD transport is not.** `benchmark/test_force_field.py` scores force-field
  *selection*; `benchmark/test_mlip.py` scores *routing*; the transport
  reference numbers (TIP3P viscosity 0.32 mPa·s, self-diffusion 5.5e-5 cm²/s)
  appear only in the reference-property **critic** track — the critic is handed
  a value and asked whether to flag it. Nothing runs MD end-to-end and scores a
  computed transport value.

For a paper whose claim is "SciLink produces scientifically accurate values",
that asymmetry is a hole: MD transport accuracy is asserted more weakly than
DFT accuracy, and closing it is the motivation here.

## The evidence — values are chosen, not tested

In both regimes the numerical parameters that determine accuracy are picked by a
single LLM call and never varied:

- **DFT/QC.** `periodic_dft_agent.generate_inputs` makes one `generate_content`
  call and emits one INCAR/KPOINTS (`periodic_dft_agent.py:302-449`, call at
  `:369-372`); ENCUT/k-points come from skill prose only
  (`skills/periodic_dft/vasp/vasp.md:117,276`; `.../qe/qe.md:44-51,111`).
  `molecular_qc_agent.generate_inputs` likewise emits one deck with an
  LLM-chosen basis (`molecular_qc_agent.py:234-387`). Every `converge` token in
  the skills is SCF/ionic convergence — intrinsic to a single run — never
  parameter convergence.
- **MD transport.** The sampling cadence that a Green-Kubo integral needs is
  never a planned quantity: the planner schema has `production_time` and
  `required_outputs` but no cadence field (`md_simulation_agent.py:279-294`),
  and the one "log this every N, don't under-sample" instruction is gated on
  `required_observables` being non-empty — which the run path leaves empty
  because `derive_observables` defaults off. Live testing on Deception produced
  three "replicas" at 20 fs / 2 ps / 200 fs cadence giving 0.72 / −5.85 / −0.50
  mPa·s (see issue #682).

## What already exists to build on

- **Execution is engine-neutral.** Step 4 of `_run_workflow_once`
  (`simulation_pipeline.py:806-857`) runs *any* scale through `_collect_stages`
  → `run_campaign` when given an `executor` + `run_command`; it is gated only on
  `executor`, not on scale. DFT just never passes one.
- **The fan-out contract is engine-neutral.** `_assemble_fanout_stage`
  (`md_simulation_agent.py:29-59`) and `_collect_stages`
  (`simulation_pipeline.py:952-1031`) read a normalized `stages`/`members` dict;
  members run in isolated `run_dir/<stage>/<member>` directories. Only
  `expand_parameter_sweep` (`skills/molecular_dynamics/lammps/lammps.py`) and
  `_finalize_campaign` are LAMMPS-bound.
- **Skills declare structured facts in frontmatter**, parsed into `meta` and
  consumed engine-neutrally (e.g. `meta["structure_file"]`,
  `periodic_dft_agent.py:267-273`).
- **Output parsers exist** as `snapshot_run` per engine (`vasp_output.py`,
  `nwchem_output.py`), and `run_analysis` already keys fan-out member files by
  path relative to the run dir, feeding all members into one analysis
  (`simulation_analysis_agent.py:183-195`).
- **The observable-requirements machinery** (`ObservableRequirementsDeriver`)
  is fully general and already threaded into both the planner and deck-gen
  prompts — it is simply switched off by default.

## What is missing

1. A **numeric cross-setting comparator** — "re-run at increasing k/ENCUT/basis
   until the observable stops changing." The loop today is verdict-driven
   (`RunCritic` good/warning/poor, `refinement.py:54-66`); fan-out members run
   independently with no cross-member comparison. This is the core new logic.
2. An **engine-neutral declaration** of which parameters need convergence, how
   to step them, the convergence observable, and the tolerance.
3. **Observable extraction** for the convergence quantity — `snapshot_run`
   returns energies but not lattice constant or gap; QE has no parser.
4. The **MD cadence wiring** (turn the deriver on) and a **pooled Green-Kubo
   recipe** across replicas.
5. **MD transport value-accuracy benchmark cases** to score against.

## Shape of the change

### A. DFT/QC parameter convergence (critical path — scoring already exists)

- **A1.** A `convergence:` frontmatter block in the VASP / QE / NWChem skills
  declaring, per parameter, the ladder of settings, the convergence observable,
  and the tolerance (e.g. ENCUT ladder, observable energy/atom, tol 1 meV/atom).
- **A2.** An engine-neutral driver that emits one member per ladder value via an
  engine-declared param-setter (each skill provides its own
  `set_convergence_param(deck, param, value)` — the general analog of the
  LAMMPS-bound `expand_parameter_sweep`), runs them as a batch fan-out through
  the existing `_assemble_fanout_stage` → `run_campaign` path, then applies a
  pure `converged_setting(member_observables, tolerance)` comparator that finds
  the plateau and returns the converged setting/value.
- **A3.** Extend `vasp_output.snapshot_run` / `nwchem_output.snapshot_run` to
  return the convergence observable; add `qe_output.py`.
- **A4.** The driver takes an explicit `run_command` (VASP/NWChem/QE have no
  `default_run_command`; the engines are available on the target cluster).

### B. MD transport convergence (issue #682)

- **B1.** Thread `derive_observables=True` on the MD run path so the deriver
  fires and the cadence instruction reaches the planner and deck generator.
- **B2.** Request a velocity-seed **ensemble up front** (planner
  `requires_multiple_simulations` over seed), reusing the existing fan-out so
  replicas differ only in seed and land under one dir. No reactive fan-out.
- **B3.** Extend the `viscosity_greenkubo` skill recipe to average across the
  member stress series (already delivered to one analysis call) and report a
  pooled value plus cross-replica uncertainty.

### C. Benchmark cases (this repo's benchmark suite)

Add MD transport value-accuracy cases mirroring `test_dft.py`: run the pipeline,
compute the transport value, and score within a band against the **force
field's own** literature value (not experiment) — e.g. TIP3P viscosity vs 0.32
mPa·s. See **Design decisions**.

## Where it lives / blast radius

The comparator and the sweep driver are new engine-neutral code beside the
existing pipeline; the per-engine knowledge (which params, how to set them, the
observable) lives in the skills. `snapshot_run` extensions are per-engine and
additive. The MD changes are a flag thread + a skill-recipe edit. Nothing
changes for a workflow that does not opt into convergence: the driver is a new
entry point, and `derive_observables` stays off except where B1 turns it on.

## Design decisions (resolved)

- **Skill-declared, engine-neutral.** The intelligence (parameters, ladder,
  observable, tolerance, param-setter) lives in each engine's skill; the driver
  and comparator carry no per-engine or per-property branches. Confirmed against
  the general bar with the requester.
- **Batch fan-out, not iterative.** Run the declared ladder in parallel and find
  the plateau, rather than stepping sequentially — simpler and reuses the
  parallel fan-out directly. (Revisit only if compute per case makes the tail of
  the ladder wasteful.)
- **MD transport reference = the force field's own value.** Benchmarking whether
  SciLink correctly computes what the chosen model predicts, not whether the
  model matches experiment (TIP3P's true viscosity is ~0.32, experiment 0.89).
- **No reactive replica loop.** An up-front seed ensemble gives consistent,
  poolable replicas deterministically; prompting the orchestrator to add
  replicas reactively does not (it regenerates the deck and drifts — the #682
  live failure).

## What this proposal explicitly does NOT claim

- It does not add convergence testing to the default pipeline — it is opt-in.
- It does not change the pipeline-completion (T3) track.
- It does not attempt to make a force field match experiment.

## Open questions

- Does `test_molecular_qc.py` already score computed QC values vs `expected`
  within a band (assumed analogous to `test_dft`; to confirm)?
- Ladder definitions per engine/property — fixed in the skill, or partly
  goal-derived?
- Where the converged value is surfaced for the accuracy harness to read —
  through the existing analysis path, or a dedicated convergence-result field.
