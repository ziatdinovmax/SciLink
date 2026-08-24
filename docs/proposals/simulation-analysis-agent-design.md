# Simulation-Analysis Agent — design note

Status: proposal / design notes. Sits under the experiment-validated-prediction
work (`experiment-validated-prediction-design.md`): the validation panel needs
computed observable *values* (viscosity, T1, RDF, …), and nothing currently
produces them engine-neutrally.

## The gap

Computing a property from a finished run is the **data → number** step —
`analyze`-mode work, and the piece the validation panel consumes. The prior
`LAMMPSAnalysisAgent` (`scilink/agents/sim_agents/lammps_analysis.py`) did this by
codegen, but the engine-neutral refactor demoted it to a benchmark baseline: its
*quality-assessment* role moved to `RunCritic`, and its *property-computation*
role was never replaced. That role is what we rebuild.

## The organizing axis: property × technique, in skills — not per-scale, not per-modality

The experimental analysis agents split by **data shape** (1D curves / 2D images /
3D datacubes) because each has shape-specific *pipeline stages* (curve-model
fitting, SAM segmentation). Simulation analysis is different: it is
**codegen-dominated**. The reusable core of the old agent is a
`plan → generate code → sandbox-run → verify → refine` loop with **no
shape-specific stages** — the data shape (a trajectory vs a `vasprun.xml` vs a
thermo table) is absorbed *inside the generated code* and its library
(MDAnalysis / pymatgen / numpy). So the shape-split that fits experimental data
does not transfer here.

The axis that actually varies for simulation analysis is **which property, via
which technique** — and that is exactly what a skill expresses. Decision:

> **One engine-neutral, codegen-based simulation-analysis foundation agent in
> `sim_agents/`. All differentiation lives in skills, each declaring a
> (property, technique, required-input) triple. No per-scale agents
> (`DFTAnalysisAgent`/`MDAnalysisAgent`/`MLIPAnalysisAgent`); no per-data-modality
> split.**

This handles the three property↔engine cases cleanly:

- **Engine-exclusive properties** — `band_gap` (needs DFT output), `viscosity`
  (needs a trajectory): skills selectable only when their required input exists.
- **Overlapping properties, different technique** — `elastic_constants_dft`
  (stress–strain) vs `elastic_constants_md` (strain fluctuations); `vib_spectrum`
  from DFT phonons vs from an MD velocity autocorrelation: two skills, one per
  technique, chosen by what data is present (and cross-checkable when both are).

MD and MLIP-driven MD are the *same* input (a trajectory + thermo log, produced
by one `MDSimulationAgent` via the `DeployedPotential` contract), so they are one
agent's concern by construction — splitting them would re-introduce the N×M
coupling that contract removed.

## The load-bearing mechanism: availability-gated technique selection

The one genuinely new selection rule: a skill declares **what input it needs**
(trajectory / DFT output / thermo log) and **what property it computes**; the
selector matches `(goal property + which output files are actually on disk) →
the applicable technique skill`. This extends the existing data-aware skill
selection (`_shared/_skill_selector.select_relevant_skills`) with an
availability signal, and it is what makes the exclusive/overlap cases resolve
without per-engine branching in shared code.

## Build plan

Build **fresh** (do not reshape the frozen baseline in place — it is a monolithic
LAMMPS-baked class kept for the old-vs-new benchmark):

- **Keep** from `LAMMPSAnalysisAgent`: the codegen → sandbox → refine core
  (`main(data_files, output_dir)` + JSON-on-stdout contract, `ScriptExecutor`
  sandbox, refinement loop).
- **Adopt** from the exp foundation agents (`curve_fitting_agent` + `base_agent`):
  the fixed section vocabulary, skill suggestion/selection, controller pipeline,
  the uniform `analyze()` / `run_analysis` contract, and — the biggest gap in the
  old agent — an **LLM result-verification gate on the computed number** (it
  previously checked only that the code ran).
- **New** for this modality: engine-neutral streaming ingestion (ASE / MDAnalysis
  frame iteration, not whole-file `loadtxt`); an engine-neutral, skill-driven
  **multi-property analysis-plan stage** (one MD run yields RDF *and* MSD *and*
  viscosity, unlike one-fit-per-spectrum).
- **Engine specifics** (reading `.lammpstrj` vs ASE `.traj` vs `vasprun.xml`) stay
  in libraries or engine-skill readers — never in the agent.

## Placement and hand-offs

- Lives in **`scilink/agents/sim_agents/`** (a simulation-side concern; it reads
  simulation output). Its `run_analysis` output feeds the simulate-side
  **validation panel** (`reference_validation.run_validation_panel`) cross-mode —
  compute the observable here, judge it against the reference there.
- **Skill graduation** grows the (property, technique) repertoire: when a
  technique is not yet a skill, the codegen loop improvises it and the verified
  result graduates into a reusable analysis skill — the "prose → skill" path we
  already use for LAMMPS.

## Scope note

No DFT data→number skills exist today (DOS/bands are not even parsed; DFT quality
is `RunCritic`'s job). Under this design a future DFT property is a **new skill on
the same agent**, not a new agent. First skills to build are the UC2 ones —
Green-Kubo viscosity and τc→T1 from a trajectory.
