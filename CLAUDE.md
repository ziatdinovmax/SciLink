# SciLink — Architecture Notes

Forward-looking design decisions and conventions. Intended for AI assistants
and contributors working on the agentic stack — orchestrators, foundation
agents, and the skill subsystem. Codebase tour and per-module docs are
elsewhere; this file is about *direction*.

## The mode universe is fixed at three

Every chat-driven orchestrator in SciLink falls into one of three modes —
this is a settled architectural commitment, not a refactoring waypoint:

| Mode | Class | Domain |
|---|---|---|
| `analyze` | `AnalysisOrchestratorAgent` | Experimental data analysis (microscopy, spectroscopy, …) |
| `plan` | `PlanningOrchestratorAgent` | Experimental campaign design |
| `simulate` | `SimulationOrchestratorAgent` | Computational simulation (DFT, classical MD, MLIP-driven MD) |

Anything in scientific workflow falls under one of these three. There will
**not** be a fourth mode. Future capability growth happens *inside* one of
the three, or as a meta-agent on top (see below).

## Capability expansion through skills, not new agents

Going forward, SciLink intends to extend its agentic capabilities primarily through skill
bundles rather than by adding more specialized subagents. New domains,
techniques, or methods are integrated as skill bundles (knowledge +
tools, co-located) under an existing subagent whose shape already fits;
a new subagent class is justified only when its execution structure
itself cannot be expressed within an existing agent. This applies
across all three modes. For example, adding an XRD or Raman skill for 
existing CurveFittingAgent is strongly preferred over creating two new agents
for Raman and XRD.

## Foundation agents

The architectural shape that makes the "skills, not new agents" preference
work is the **foundation(al) agent**: a single agent class designed to
cover one broad domain (its "modality") and specialized at runtime to
specific techniques within that domain through pluggable skill bundles.
Today the analysis-side agents (`CurveFittingAgent`, `ImageAnalysisAgent`,
`HyperspectralAnalysisAgent`) are the canonical examples; the proposed
`OptimizationAgent` refactor (issue #196) follows this same shape for the
optimization modality, and future simulation foundation agents (e.g., a
DFT-side equivalent) should as well.

A foundation(al) agent has five elements:

1. **A modality-specific pipeline architecture.** Each foundation agent
   defines its own fixed sequence (and graph) of stages — planning,
   execution, verification, refinement, and so on — appropriate to its
   modality. Image analysis, optimization, and DFT calculations don't
   share a pipeline shape; each has stages and branches that fit how that
   kind of work actually proceeds.

2. **Per-stage baseline prompts owned by the agent.** At each stage of
   its pipeline, the agent carries a baseline prompt template encoding
   the technique-independent reasoning for that stage. These baselines
   are the agent's "default expertise" — always present, never replaced
   by specializations.

3. **A fixed section vocabulary defined per modality.** The agent
   declares a small set of named sections (e.g., *planning*,
   *validation*, *interpretation*) that domain specializations are
   authored against. The vocabulary is the contract between
   specialization authors and the agent's pipeline: authors write
   content under those named sections, and the agent's pipeline stages
   know which named sections to splice into which baseline prompts.

4. **Pluggable domain specializations ("skills") combining prose
   guidance with optional code helpers.** A skill is a self-contained
   bundle authored by a domain expert — primarily narrative guidance
   organized under the agent's section vocabulary, optionally
   accompanied by purpose-built helper functions exposed as callable
   tools. Activating a skill for a given run changes both what the LLM
   reads at each stage and which specialized tools it can call; tools
   declared inside a skill are visible to the LLM only when that skill
   is active. This keeps the per-call surface focused on the active
   specialization rather than overwhelming the LLM with every possible
   tool from every possible skill.

5. **An extensibility loop for open-ended per-task work, with
   modality-appropriate verification.** The agent ships with a stable
   surface of specialized tools (e.g., SAM for image analysis) and,
   where the modality needs it, also generates per-task code at
   runtime. Generated artifacts run in a sandbox, and the agent
   verifies the result before accepting it. Both the scope of
   generation and the form of verification vary by modality.

Elements (1)–(3) are the agent's structural contract (pipeline shape,
baseline prompts at each stage, section vocabulary). Element (4) is how
a skill specializes that contract at runtime. Element (5) is what lets
the agent absorb the long tail of techniques without per-technique code.

A note on "modality": the natural axis of variation differs across agent
families. In analysis it's typically data type (1D curves, 2D images,
3D datacubes); in optimization it could be method family (sequential
single-objective BO, multi-objective, DoE, active learning); in
simulation it could be computational method (DFT, classical MD,
machine-learning potentials). Each foundation agent picks its own axis;
the definition is agnostic about *what counts as a modality*.

A note on the `analysis` / `implementation` section pair: codegen-capable
foundation agents inject the active skill's `implementation` section into
per-task code-gen prompts. The skill loader treats `analysis` and
`implementation` as synonyms when only one is authored — copying the
content to the other — so skills written under either name flow into
code-gen identically. When *both* are authored (e.g. `force_field/amber`,
`molecular_dynamics/lammps`, `machine_learning_potentials/chgnet`), they
are left distinct: the author's convention there is `analysis` for input
characterization ("what kind of system is this?") and `implementation`
for the runnable script recipe. This synonym fold is historical: the
section was originally `fitting` in the curve-fitting-only era, renamed
to `analysis` when image_analysis joined, and is now `implementation` in
the most recent sim_agents and hyperspectral work. Going forward, prefer
`implementation`.

**Recommended structure for new analysis-agent skills:** `Overview →
Planning → Implementation → Interpretation → Validation`. This five-
section pattern follows the cognitive flow of an analysis run — what the
technique is, how to plan a use of it, how to write the code, how to
read the output, and how to verify it. New skills should use this
ordering; legacy `analysis` is accepted by the loader for backcompat.

## Series analysis is one shape across the three analysis agents

Curve, image and hyperspectral all run a series as **anchor + locked recipe**:
the first unit is analysed in full, the verified recipe is locked, every later
unit reuses it verbatim, failures are re-analysed within `max_series_refits`,
feature outliers are flagged (never re-analysed — the anomaly may be the
physics), then trend codegen and a series synthesis run over the per-unit
feature table. The per-unit rows are written to `series_analysis_results.json`
in one shape, so `feature_table.write_feature_table` and every downstream
consumer read all three the same way.

The hyperspectral instantiation differs in mechanics, not shape: the single-
cube pipeline is bound to one output directory (decomposition, dynamic-analysis
records, report), so the series driver (`_analyze_series`) runs **one child
agent per dataset** in `dataset_NNNN/` instead of re-pointing controllers, and
"locked recipe" *is* the existing locked-script replay (#172) pointed at the
regime anchor's `dynamic_analysis_records.json`. The scout stage mirrors the
curve agent: every cube's mean spectrum feeds the shared SVD change detection
(`series_reduction.reduce_curves`) and an overlay, and one planning call
declares regimes, each with its own anchor and locked script. The parent owns
the once-per-series decisions (skill choice, script banking, T=2 staging);
replay children skip them.

**Feature names are aligned by construction, then completed.** The first
dataset to lock a recipe is the series' *schema source*. Every later
fresh-code run — another regime's anchor, an adaptive refit — runs in
*locked-targets* mode (`analyze(locked_targets=...)`): the schema source's
targets and required output names are fixed, planning and decomposition are
skipped, and the code is regenerated through the full ladder, so the QC's
required-outputs check enforces the names. What still drifts (a units suffix,
a diagnostic-map prefix) is aliased onto the locked columns by
`complete_locked_schema`; whatever cannot be matched is reported as
`locked_schema_gap`, never silently NaN. Outliers are scored on the locked
primary outputs only — a refit's diagnostics belong to a different method —
and per regime when the planned regimes interleave along the axis.

**Replays are gated on evidence, not re-judged.** Live stress runs showed
the per-map LLM reviewer rejecting, on replays, the very map it approved on
the anchor (judge variance), which punched holes in the series schema and
cost minutes per replay. A locked-script replay therefore runs NO LLM map
review: `_replay_map_gate` accepts a map on valid coverage (within the fit
mask when scoped), a non-collapsed distribution and, for required outputs,
a median inside the anchor's plausible range (its [min, max] widened by one
span; the anchor's per-map stats travel as `replay_reference`). A rejection
means the method broke on that dataset, which the driver answers with a
locked-targets refit. Replay children also skip the synthesis critic/editor
pair (`_light_synthesis`); the series synthesis interprets the series. A run
that committed features from a salvaged attempt with no verified required
output is `unverified`: flagged, excluded from the outlier statistics,
refit-eligible after failures. The scout always includes the two datasets
bracketing a sharp change point.

**The regime plan is human-gated like any other plan.** In co-pilot /
autopilot (`enable_human_feedback`) the driver shows the plan (regimes,
anchors, control values, change-detection summary) and asks; Enter accepts,
any text goes back to the planner as analyst feedback with the previous plan,
up to three rounds (`_regime_plan_gate`). Autonomous runs and locked replays
of a prior run skip it; EOF on the prompt accepts.

**Anchor codegen starts from data facts, not priors.** Live anchors kept
failing on gates set from literature: fit windows centred on textbook peak
positions while the measured peaks sat elsewhere, seeds railing at window
edges, not-measurable declarations built on a wrong-scale sigma. So the
code-generation prompt now carries a deterministic `DATA FACTS` block
(`_render_data_facts`: field-mean peaks with positions, prominence in sigma
of the mean and width bounds, the noise of the mean, the axis) plus the same
band-flux table the judge holds the script to — the model that writes the
gates sees the numbers the reviewer will use. Prompt-side, one principle:
centre windows and seeds on the measured positions and confirm the peaks
survive background subtraction. Two rules give that structural teeth: a
`not_measurable` declaration that contradicts the facts (a >= 5-sigma
field-mean feature) is repaired IN PLACE like an execution error — no judge
call, no ladder budget — and a required map that comes back entirely NaN
or with the wrong shape (a binned estimate not upsampled to the frame) is
diagnosed in the retry critique instead of "no further detail".

**Through the meta agent, a series is ONE delegation.** The meta's routing
guidance and `delegate_to_analysis` say so for spectra, images and cubes
alike: pass the shared directory (or file list) and the control variable;
the child's `run_task` runs `run_analysis` on the directory, so the series
mode engages and the delegation result carries the series claims and its
one `features.csv` (one row per dataset). Harmonized fan-out
(`delegate_to_analyses(harmonize=True)`) predates the series mode and is
kept only for sibling datasets that cannot be staged as one directory. A
fan-out branch that IS a datacube-series directory gets
`FANOUT_SERIES_BUDGET_FACTOR` × the default wall-clock budget (the raw-
instrument rule's shape), because the series mode is a multiple of one run.
Under AUTOPILOT delegation the regime-plan gate reaches the user through the
normal feedback channel, like the specialists' other plan gates.

**Replays fan out.** `series_workers` (or `SCILINK_HS_SERIES_WORKERS`) runs
the locked replays on a spawned-process pool, each submitted the moment its
regime locks so replays overlap with the anchors still running in the
parent — replays are independent, and processes rather than threads keep
matplotlib and the sandbox executor out of each other's way
(`SCILINK_HS_SERIES_POOL=thread` exists for the offline tests). Each replay logs to its own
`dataset_NNNN/replay.log`; the parent's sandbox approval travels with the
spec. Anchors and refits stay serial: they are the LLM-heavy, human-gated
part.

## Data preparation is a stage, not an agent

Some instruments hand over a container that sits *upstream* of what the
analysis agents take — a raw off-axis hologram stack, a raw detector
container with a reconstruction contract. Such data must be **transformed**
(reconstructed, reduced, calibrated, joined with a condition timeline) before
any curve / image / hyperspectral agent may see it, and routing it as an
image or a cube is wrong by construction. This is handled by a preparation
stage on the analysis orchestrator, not by a fourth agent:

- `examine_data` reports `data_type="raw_instrument"` (with
  `preparation_required=true`) when a same-stem sidecar, a
  `reconstruction_manifest.json`, or an embedded HDF5 contract carries
  routing-denial / reconstruction markers (`generic_image_routing_permitted:
  false`, `analysis_status: ready_for_..._reconstruction`, a hologram /
  interferogram `measurement_type`, ...). Detection lives in
  `scilink/agents/exp_agents/data_preparation.py`.
- `prepare_data` (orchestrator tool) selects a skill from the
  `scilink/skills/data_preparation/` domain (auto-selected or named), builds
  an inventory of the container, and runs the shared shape: generate a
  script from the skill's `planning` + `implementation` sections and the
  skill's `TOOL_SPEC` inventory → sandbox run → deterministic gate (products
  exist under `results/prepare_<id>/`, same-stem sidecars, `qc.passed`,
  numeric metrics) → skill-guided LLM verification against the `validation`
  section → retry with feedback. The approved script is kept as
  `scripts/prepare_script.py`; `analysis_results.json` carries the QC
  metrics as `extracted_features` so the feature table works unchanged.
- Products are ordinary data files, so `run_analysis`, meta fan-out and
  fusion consume them without special cases; the meta only needs to route a
  raw container to analysis with "prepare first" in the task.
- **Preparation happens before fan-out, never inside a branch.** Every
  preparation attempt reconstructs the whole container, which alone can
  exceed a branch's wall-clock budget (observed live: a nine-run hologram
  bundle spent its full 3600 s on preparation and was cancelled). `run_fanout`
  therefore declines a raw-instrument branch with a directive to prepare it in
  a standalone delegation first; the fan-out then runs over the products.
  A caller who accepts a long branch opts in with `allow_raw_branches`
  (that branch gets `FANOUT_RAW_INSTRUMENT_BUDGET_FACTOR` × the default
  budget) or sets `branch_time_budget_s` explicitly; budgets are resolved
  per branch and persisted on the ledger entry (`_budget_s`) so a resumed
  fan-out enforces the same value.

Preparation skills follow the standard five-section vocabulary; their
`implementation` recipe is code-in-markdown that calls the bundle's
deterministic helpers (the reliability ladder applies: the contract-exact
numerics live in `TOOL_SPEC` tools, the glue is generated). Preparation
skills are excluded from `run_analysis`'s skill menu. The first skill is
`mmzi_hologram_reconstruction` (contract-exact reconstruction with a
producer-target QC gate, then piston-immune phase maps and ROI traces).

## Plan mode produces three kinds of thing, not one

`generate_initial_plan` designs **lab experiments**. For a long time it was
also the only authoring path in plan mode, so two other kinds of request rode
its schema and were filled by invention:

| the ask | tool | payload |
|---|---|---|
| "design an experiment to test X" | `generate_initial_plan` | `proposed_experiments` — hypothesis, steps, equipment |
| "what directions are worth pursuing" | `generate_ideation_portfolio` | `directions` — id, title, tier, hypothesis, rationale, novelty |
| "write me a roadmap / estimate / memo" | `write_technical_document` | markdown sections; no campaign state at all |

The rule: **if there is no hypothesis to test and nothing to measure, it is
not an experimental plan.** A portfolio forced through the experiment schema
comes back with its directions flattened into `experimental_steps`; a
document forced through it invents `optimization_params` with ranges and
citations for a system nobody has chosen yet. Both were observed live.

**One engine, two contracts.** Ideation is not a new agent or a fourth mode.
`generate_plan(kind="portfolio")` reuses retrieval, best-of-N, the judge, the
critic, campaign scoping, checkpointing and the deliverables ledger, and
swaps only what a candidate *is* — `generate_plan_candidates` takes a
`contract` for that; absent one it is the experiment path unchanged.

**Reading directions.** Never read the payload shape directly. `parser_utils`
exposes `plan_directions` / `plan_is_portfolio` / `plan_thesis`, resolving
top-level `directions` → `proposed_experiments[*].concepts` (PR #394) → a
direction synthesised from the experiment fields (pre-`concepts` plans). That
fallback ladder is what keeps old checkpoints restorable.

**The transition shim.** A portfolio currently carries BOTH shapes:
`directions` as the payload plus a one-entry `proposed_experiments` shim, so
the ~50 legacy readers stay correct instead of seeing a plan with no
experiments (which their validity gates read as *failed*). Consequence worth
remembering: every pass that re-emits a plan edits the shim, so
`resync_portfolio` (called from `_stamp_campaign`) makes the nested copy
authoritative — otherwise a refined portfolio serves stale directions.

**TEA is a grounded, audited step, and the whole assessment travels.**
`run_economic_analysis` authors under two instruction tiers — strict (KB /
literature only) and fallback (general benchmarks, entered only when the
model reports insufficient economic context) — and the result records which
one produced it (`grounding.mode`, mirrored as `generation_mode` on
`latest_tea_results`). A TEA critic (`critique_tea`) then audits every
`(Quantitative)` claim against the same evidence the author saw and stores
advisory `critic_findings`; the evidence itself is written to
`tea_analysis.grounding.md`. Downstream, `_tea_context_block` renders the
full assessment — cost drivers, risks, comparison, **data gaps**, provenance,
caveats — as one block that plan authoring, refinement, portfolio and
technical-document calls all inject; nothing reads the summary sentence
alone. `primary_data_set` takes several tables at once (folder or comma
list), each summarised under its own name, because a TEA routinely needs a
composition, a price list and measured yields together. A TEA-first run is
iteration 0; a TEA run mid-campaign keeps the current iteration (stage
`TEA Update`) and never resets the counter, and the report keys cards on
(iteration, kind) so a TEA and the plan it assesses both render.

## A plan the human approved is settled

In autopilot the planning orchestrator used to re-refine a plan the human had
just approved, on its own reading of the advisory critic. Observed live: three
self-initiated rewrites after the first ENTER, the catalyst switched to the
candidate the human had passed over, each round logged as experimental results
(iteration 4, no experiment run), each later round repairing what the one
before broke, and a final `edit_file` on `plan.json` that forked the disk copy
from the state. The critic never converges — it raises a different set of
advisory findings every time it is asked — so chasing it is unbounded, and a
falling count of critical findings mostly measures a plan that claims less.

**Approval is recorded, and only three things reopen it.** A review gate stamps
`human_review` on the plan and the tool result says `SETTLED`. Every rewrite
tool (`refine_plan_with_results`, `adjust_plan_for_constraints`,
`refine_portfolio`) takes a required `trigger`: `new_results`, `user_request`,
or `blocking_defect` — the plan cannot be run as written, or is unsafe. The
trigger is self-declared, so each value is held to what can be checked
(`_revision_refusal`): a `user_request` needs a user message since the
approval; a `blocking_defect` needs a stated reason, has a budget of one per
authored plan (`MAX_SELF_REVISIONS` — this is what bounds autonomous runs), and
is final once the human has declined one. Anything less is a caveat in the
summary. File tools refuse the files that mirror planner state
(`_STATE_BACKED_FILES`); documents stay editable.

**The reopen gate's default is the approved plan.** An agent-initiated revision
of an approved plan is shown with its reason, and ENTER *keeps the approved
plan* (`_revision_gate`, `get_reopen_decision`): a reviewer who waves a gate
through must get the plan they chose. A declined revision leaves no trace in
the campaign state — no history snapshot, no results entry, same iteration —
only an action-log record. Results and user requests keep the ordinary gate.

**Only an executed plan opens an iteration.** The refinement channel carries
revision requests as well as results (#638), so each `experimental_results`
entry says which it is (`kind`). `new_results` advances the iteration and is
framed as "we executed the plan"; the other triggers keep the iteration, are
framed as "NOT executed", and ask for the requested change only.

**The critic has three tiers and acts on one.** `critical` and `minor` stay
advisory — auto-applying them rescoped plans (a6b946af). `blocking` is reserved
for a conflict that can be stated as two values, the plan's against the limit
it violates — a limit that can be cited and that the plan's value exceeds,
not the critic's estimate of how a step will turn out (the first blocking
findings on a naturally authored plan were of that looser kind); one that
states no `conflict` is filed as `critical`. A blocking
finding gets ONE in-place repair before the plan is shown
(`_auto_repair_blocking`, every autonomy mode, the selected candidate only —
runner-ups stay as authored): a fix-only contract the author may decline, a
deterministic scope guard (`repair_preserves_scope`: experiment count, names
and hypotheses frozen — which is where a plan names its material system — and
the rest at least `REPAIR_MIN_SIMILARITY` alike), and a resolution re-critique
that must find the conflict gone and no critical finding marked `introduced`
by the repair. Anything else keeps the plan as authored with the blocking
caveat and the reason on record. The reviewer sees what changed and can type
`revert`. Counting critical findings before and after does not work as an
acceptance rule; the critic's set differs on every call.

**Selection does not depend on what the repair fixes.** The repair runs on the
selected candidate only — repairing all N before judging triples the cost and
hides from the judge which author made a gross error — so the judge must not
mark a candidate down for a defect a local edit would fix: it scores the
design as it would stand once fixed and lists the defect under that
candidate's `fixable_defects`. Live A/B on one pair of candidates: under the
old prompt an impossible temperature in one step cost the stronger design the
pick in 2 of 3 trials; under this one, 0 of 3. A defect that needs a new
hypothesis or technique is not fixable and counts in full.

**When the pick cannot be repaired.** With a human at the gate nothing
switches: the selection prompt shows the blocking caveat and the discarded
repair, and the human picks another candidate or approves knowingly — a
decision that binds, because the critic can be wrong. With nobody at the gate
(autonomous), the pick falls back down the judge's own ranking to the first
candidate that can be run (`_fallback_order`, bounded by N, recorded as
`plan_candidates.fallback`) — the one automatic switch of a selection, and
only ever on a checked conflict. If none can be run, the judge's pick stands
and nothing is enforced: the tool result carries `unresolved_blocking_finding`
(issue, conflict, and that the repair failed) and the orchestrator decides —
check the cited limit, fix the plan with its one guarded revision, proceed, or
stop and report; the caveat reaches a headless caller in `warnings` either way
(`_standing_blocker`; a human review settles it). A hard stop was tried
(code generation refused, `run_task` forced to `error`) and removed: it failed
runs on the critic's word alone, and live the critic cited three different
limits for one hazard in three passes.

**The agent's own revision is a repair too.** A `blocking_defect` rewrite from
the orchestrator passes the same checks as the automatic one
(`_discard_self_revision`: the scope guard, no critical finding `introduced`);
one that fails is discarded before any gate and the tool says why. If the
defect cannot be fixed locally, the agent reports it and the human decides.

**The gates say what the decision is about.** A review question carries its
subject on `origin`, not in the printed text: the plan gate's `auto_repair`
change lines, the reopen gate's `reason`. The presenter turns them into a
`notice` callout beside the buttons (count in the title, long lines clipped —
the full text is in the review above) plus a one-click "Revert
auto-correction"; the reopen gate reuses the keep/revert widget with its own
words and a text box for "adopt with changes". Keying these on the printed
notice failed in the browser: a real plan puts pages of caveats between the
notice and the prompt, outside the presenter's context tail. `revert` counts
as a declined reopen — the human just refused that exact fix. Mission Control
relays the planning child's questions through the same panel, so the controls
are identical there.

**The critic reads the protocol.** `summarize_plan_for_critic` shows an
experiment plan's steps, equipment, parameter ranges and expected outcome, as
it shows a portfolio's `details`; the conformance pass keeps its
coverage-and-identity view. Before this, the critic called replication "never
defined" and an inert transfer "not specified" on a plan whose steps specified
both, and could not have seen an unrunnable step at all.

## Plan-mode capability boundaries

Two settled conventions on where capability lives in plan mode:

**Plan-mode skills are knowledge-only.** Skill bundles under
`scilink/skills/planning/<name>/` are markdown — no per-skill
`.py` / `TOOL_SPEC` tools. Plan mode reasons and synthesizes; it does
not execute domain numerics. `PlanningAgent` produces plan text, heavy
compute is `BOAgent`'s, and executable artifacts flow through
`generate_implementation_code` — codegen *guided by* the skill's
`implementation` section, so the skill shapes the code rather than
shipping it. Planning subagents deliberately do not consume the
`_shared/_registry` tool inventory. A planning skill that seems to need
a vetted tool is mis-scoped.

**The scalarizer is the lightweight analysis tier.** `ScalarizerAgent`
does simple LLM-generated extraction (pandas / numpy / scipy) over
tabular or otherwise simple data, reduced to scalars plus the BO
input/target schema. It gets no vetted `.py` tools — needing one is the
tripwire that the task is not lightweight and belongs in analyze mode.
Heavy "data → number" extraction is reused, not rebuilt: `run_analysis`
does the hard work with its skill tools, then the scalarizer reduces the
result (`run_analysis → scalarize`). That cross-mode chain is gated on
the future `run_task` contract; until it exists, run the analysis
standalone and feed the resulting scalar in as a data file.

## Knowledge bases are named artifacts; grounding is explicit

RAG knowledge bases live in the persistent store
(`~/.scilink/knowledge_bases/<name>/`, managed by `scilink kb` /
`scilink/knowledge/kb_store.py`) with a `manifest.json` recording the
embedding model that built them — provenance that turns provider
mismatches into upfront warnings instead of opaque query-time failures.
Every `knowledge_dir` surface resolves store names as well as paths (an
existing directory wins over a same-named KB). Two settled rules:

- **No implicit grounding in the meta.** A meta session never silently
  inherits a launch-directory KB; attachment is an explicit choice —
  `--knowledge-dir`, a chat-time confirmation (autopilot), or the
  autonomous relevance decision made from the KB's listed sources.
  Standalone plan mode keeps its stable-KB conventions.
- **Retrieval is grounding, not a dependency.** Retrieval degrades
  through tiers — dense, then model-free keyword (BM25) over the stored
  chunks, then no-context — each with a warning; it must never abort
  generation.

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

## How the simulate orchestrator works

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

Post-run
  analyze_output(output_dir, research_goal, software)   # engine-neutral, via SimulationAnalysisAgent

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

- `chat(user_input: str) -> str` — interactive (CLI / UI).
- `run_task(task, context=None, autonomy=None) -> dict` — programmatic
  entry point. Runs one `chat` turn under the requested autonomy mode, then
  derives a structured summary from the session-state delta. `autonomy=None`
  defaults to AUTONOMOUS — the safe choice for a headless caller (never
  pauses for a nonexistent user). A caller attached to a human passes a
  co-pilot / autopilot mode so the sub-agents' human-feedback prompts reach
  that human.

`run_task` is implemented on **all three** orchestrators with that uniform
signature. The return dict shares `status, task, summary, files_produced,
key_findings, suggested_followups, warnings` (plus `error` on failure); the
domain-specific field differs per mode — `analyses` (analyze),
`campaign_state` (plan), `structures` (simulate). This is the contract the
meta agent delegates through.

## The meta agent

The meta agent sits on top of the mode orchestrators so users don't switch
manually — bare `scilink` (or `scilink explore`) launches it. It is **not a
fourth mode**; it's an orchestrator-of-orchestrators with a different role
(router + context bridge). It lives in `scilink/agents/meta_agent/`
(`MetaOrchestratorAgent` + `MetaOrchestratorTools`), copying the
`AnalysisOrchestratorAgent` chat-loop shape.

**Scope: analysis + planning + simulation.** Simulation delegation is now
wired (`delegate_to_simulation`). Because `scilink.agents.sim_agents`
hard-imports `ase` (an optional dependency), the meta module must stay
importable without it: the tool's body and the orchestrator's
`_get_simulation_child` / `_delegate` "simulation" branch all do the
`simulation_orchestrator` import *inside the function*, never at module scope,
and the tool returns a clean "install scilink[sim]" error if `ase` is absent.
The simulation child is structure-centric, so — unlike the planning child — it
needs no `data_dir` at construction; it lives in `<meta_session>/simulation/`.

### Pattern: agent-as-tool

```
Meta tool registry
  delegate_to_analysis(task, context)   → AnalysisOrchestratorAgent.run_task
  delegate_to_planning(task, context)   → PlanningOrchestratorAgent.run_task
  delegate_to_simulation(task, context) → SimulationOrchestratorAgent.run_task
  summarize_session_state()             → cross-specialist status
  get_delegation_history(limit)         → the delegation ledger
```

There is **no `bridge_context` tool**. `run_task` already accepts a
`context` dict; the meta LLM bridges modes by reading a prior result via
`get_delegation_history` and threading its `key_findings` / `files_produced`
into the next delegation's `context`. The delegation ledger is the
supporting structure.

### Two autonomy levels, not three

The individual modes have a three-level autonomy paradigm (co-pilot /
autopilot / autonomous); `MetaMode` has only **AUTOPILOT** (default) and
**AUTONOMOUS**. A delegation runs the child through its one-shot `run_task`
— a single turn. Co-pilot's model is "pause after every step, wait for the
user's next message," which needs many turns, so it cannot complete a
delegated task. AUTOPILOT and AUTONOMOUS each finish a task in one turn:
AUTOPILOT still pauses at the child's decision points (approve / edit plans
and outputs) via `input()`-based human-feedback prompts — which compose with
`run_task` because they block-and-resume *within* the turn — while AUTONOMOUS
runs end to end. The three-level paradigm is untouched for the standalone
`analyze` / `plan` / `simulate` modes.

### Persistent children, nested sessions

The meta keeps **one persistent child per mode** — lazily created on first
delegation, reused across all delegations so context accumulates — in fixed
sub-directories `<meta_session>/analysis/` and `<meta_session>/planning/`.
After a meta restore a child is re-created with `restore_checkpoint=True`
simply by probing for its `checkpoint.json`. **Each delegation runs the
child under the meta's own autonomy mode** — passed as `run_task`'s
`autonomy` arg (mapped by enum name); the child's resting mode is
irrelevant. So an autopilot delegation keeps the specialist's human-feedback
prompts, which surface to the user driving the meta exactly as in a direct
single-mode session. The planning child is built in CO_PILOT with
`data_dir=None` — the one construction mode that does not require
`data_dir`; `set_autonomy_level` does not re-validate it on the per-call
switch. Per-delegation
isolation: each `run_task` writes into its own sub-directory so a reused
child does not overwrite earlier outputs (analysis already stamps result
dirs; the planning orchestrator writes to a per-delegation
`delegations/<NN>_<slug>/`).

Because the meta consumes children through their `run_task` contract (not
through inherited internals), no base class is required. The contract is
duck-typed; what the children share is *interface shape*, not
*implementation*.

## The terminal is one shell over the four chat modes

`scilink/cli/shell/` is the single REPL behind bare `scilink` (meta),
`scilink analyze`, `scilink plan` and `scilink simulate`; the four
`cli/<mode>.py` files are thin entry points. A `ModeAdapter` (`modes.py`)
carries what differs per mode — flags, how to build/restore the
orchestrator, autonomy get/set, status fields, extra slash commands,
seed turns, headless `run_task` — and nothing else. This does not
contradict the no-`BaseChatOrchestrator` rule above: the agents are
untouched; only the terminal layer, which was four copies, is one.

The shell reuses the web backend's turn machinery rather than
re-implementing it: `RoutedCapture(echo_console=False)` for the
print-driven stop, `ParkingChannel` (the base of the web `HTTPChannel`)
for human-in-the-loop questions, `presenter.present_question` for the
widget vocabulary, `sessions.discover_resumable` for resume. The words
both surfaces show live in `scilink/ui/vocabulary.py` (mode names,
placeholders, status badges, consent sentence, stop messages, the
"Enter = <accept>" hint, activity labels); `scripts/gen_vocabulary.py`
exports them to `webui/src/vocabulary.ts` and a test keeps the two equal.
The narration reader (`scilink/ui/narration.py`, TS twin
`webui/src/narration.ts`) classifies the agents' printed lines and derives
the activity label; the web runner emits it as an `activity` event, the
shell shows it on its status row, and one fixture pins both readers.
**When a chat-surface label or behaviour changes, change it in the
vocabulary or the narration reader, not in one surface.**

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

The meta agent (`scilink/agents/meta_agent/`) was built following this same
backend → CLI → UI order, over the analysis and planning orchestrators;
simulation delegation is wired into its lazy seam once that path is stable.

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

## Self-refinement is one shape

Every simulation agent fits the same loop:

1. **Generate** — structure + inputs. One-shot. Pre-run validation
   (`StructureValidatorAgent`, `IncarValidatorAgent`) lives inside
   this stage as part of generation, not as a separate pre-run phase.
2. **Run** — the engine (VASP, LAMMPS, MD via `DeployedPotential`).
   For MD-shaped runs the engine sweeps multiple phases
   (optim → equilib → production) within this single stage.
3. **Branch on outcome**:
   - **Engine error** → the engine-neutral refinement loop (`refinement.py`)
     parses the log, proposes corrected inputs, loops back to **Run**.
   - **Phase success** → quality check fires *per phase*, not just at
     the end. Pass → next phase or done. Questionable → refine and
     loop back to **Run**.

Both feedback paths terminate at **Run** with updated inputs — the
iteration is around Run, not around Generate. The shape is
scale-agnostic: DFT, classical MD, and MLIP-driven MD instantiate it
with different agents in each slot. When adding a new simulation
agent, fit it into this skeleton rather than reinventing the loop.

## Engine-neutral contracts

Some agents communicate through small, engine-neutral descriptors so
adding a new backend is one skill bundle rather than N×M integrations.

The canonical example today is `DeployedPotential` (in
`scilink/agents/sim_agents/_potential.py`): `MLIPAgent` emits it,
`MDSimulationAgent` consumes it. The descriptor carries `backend`,
`model_name`, `model_file`, `elements`, and an `ASECalculatorSpec`
(three strings: import line, construct expression, device env var).
The MD agent fills its ASE calculator from those strings and never
imports MLIP code itself. Engine-specific bindings (LAMMPS
`pair_style`, GROMACS kernel, …) live with the engine in its skill
bundle.

The payoff is N+M wiring instead of N×M: one new MLIP backend means
one new skill bundle, not one integration per MD engine. The same
pattern should apply to any future producer→consumer agent boundary
that crosses scale or engine — design the contract first, then add
producer and consumer behind it.

## Skill subsystem

Skills are domain-specific LLM context shared across the experimental and
simulation agents.

- **Skill bundles** at `scilink/skills/<domain>/<name>/` — one folder per
  skill containing `<name>.md` plus optional sibling `.py` helpers
  (Anthropic-Skill shape).
- **Cross-skill helpers** at `scilink/skills/_shared/` — modules referenced
  by multiple bundles, plus the `_registry.py` / `_spec.py` discovery
  infrastructure.
- **Non-skill utilities** at `scilink/utils/`. The legacy `scilink/tools/`
  no longer exists.

Skill markdown begins with an optional `---`-delimited YAML frontmatter
block. The only field consumed today is `description` (rendered into the
orchestrator's `run_analysis` tool parameter blurb). Add fields only when
there's a consumer; don't accumulate metadata speculatively.

Section vocabulary is **fixed**: `overview`, `planning`, `analysis`,
`interpretation`, `validation`, `implementation`. Off-vocabulary `## headings`
are preserved under `extras` and a warning is logged so authors get
feedback instead of silent loss. The fixed set is load-bearing — controllers
inject specific sections at decision points
(`_get_skill_context(section="planning")`), which is how prompts stay tight.

Multi-skill is end-to-end. `analyze(skill=...)` and the `run_analysis` tool
both accept `str | list[str]`. `TOOL_SPEC` declarations inside a skill
bundle are visible to the LLM only when that skill is active; `_shared/`
specs are always-on (filtered by their `agents=` tag).

Code blocks inside skill markdown are **LLM-facing reference**, not
executable surfaces — the loader does not extract or run them. Runnable
code lives in sibling `.py` files and is registered via `TOOL_SPEC`.
Domain scientists who write markdown only can ship a skill as a single
`<name>.md` and never touch Python; the engineer-maintained helpers
co-locate as siblings.

This yields a three-rung reliability ladder for a skill's
`implementation` recipe — prose → code-in-markdown → `TOOL_SPEC` helper:

- **Prose recipe.** Narrative guidance; the codegen LLM improvises the
  code. Lowest fidelity, zero packaging.
- **Code-in-markdown.** A concrete snippet in the `implementation`
  section, injected verbatim into the codegen prompt as the recipe to
  follow; the LLM transcribes/adapts it into the generated script (it is
  *not* imported or run directly). **Use when** you want to pin the exact
  algorithm / params / library (much stronger than prose), the op is
  short-to-medium, and you're fine with the LLM adapting it to the data.
  This is the right default for most scientist-authored skills —
  concrete, no Python packaging, composes automatically. The agent's
  verification loop (sandbox run + compile-check + quality gate)
  backstops transcription errors, so a mangled snippet is corrected, not
  silently wrong.
- **`TOOL_SPEC` helper.** A sibling `.py` callable surfaced in the
  codegen tool inventory and *called* by the generated script (byte-exact,
  deterministic, testable, reusable). **Promote to this when** the code
  must run verbatim/deterministically, it's long or numerically
  sensitive, or it's a reusable stage you want tested and called
  identically every time (and you can contribute it to the package).

**Expose a `TOOL_SPEC` helper's tunable parameters to the LLM — robust
defaults, but no locked knobs.** A tool with hidden parameters forces its
defaults on every dataset and is brittle; a tool whose knobs are *surfaced
and explained* lets the agent adapt it to data the defaults don't suit, which
is what makes the tool general rather than overfit to the cases it was built
on. The `TOOL_SPEC.parameters` dict is the only surface the LLM sees, so put
**every meaningful tunable parameter there**, each described by *what it does
and which direction to turn it for which symptom* — e.g. "improve_thresh —
parsimony knob: LOWER to recover weak shoulders, RAISE if adding spurious
peaks", not just a name. Keep the adaptive logic and safe defaults inside the
tool (so a no-arg call still works), and add a test asserting a knob actually
changes behavior. This complements the "adaptive logic in the tool, not prompt
prose" rule: the tool *defaults* are the adaptation, the *exposed knobs* are
the escape hatch when the data needs them.

Note the packaging boundary: `TOOL_SPEC` tools are discovered only from
skills *inside the installed package* (`_registry` walks `_SKILLS_DIR`
and imports them as `scilink.skills.…` modules). Skills added via the UI
uploader or dropped in the persistent `~/.scilink` store are markdown-only
by construction — they compose via the prose / code-in-markdown rungs, not
`TOOL_SPEC`. So code-in-markdown is the highest reliability rung a custom
skill can reach without a package contribution.

A user-registered custom skill (UI uploader / `--skills` → `register_skill`,
held in the orchestrator's `_custom_skills` as `{name: path}`) is still
**auto-selectable by the agent**: `run_analysis` forwards `_custom_skills`
to `analyze(custom_skills=…)`, and `build_skill_catalog` folds them into the
per-domain catalog (skipping a custom skill that explicitly declares a
*different* analysis modality), so the agent's selector treats them like
built-ins. A selected custom name is resolved back to its path before
loading (customs aren't on the loader's search roots). This means the
orchestrator does NOT need to pass an uploaded skill authoritatively just to
make it usable — it pre-loads `skill` only for an explicit user request.

**Multi-skill composition.** When several skills are co-active, each may
own a different pipeline stage (e.g. one skill's preprocessing recipe and
another's analysis recipe), so codegen injects the `implementation`
sections of *all* co-active skills — labeled per skill, applied in the
plan's order — rather than only the top-ranked one. The technique-aware
selector keeps co-activation conservative (complementary skills only), so
this composes stages rather than fusing competing recipes. Authors should
write each `implementation` section as self-contained for *its* stage,
not assuming it is the only active skill.

**Selection policy differs by how authoritative a domain's skills are**
(an `exclusive` flag on `_shared/_skill_selector.select_relevant_skills`):

- **Curve fitting is *exclusive*.** Its skills encode AUTHORITATIVE,
  mutually-exclusive *technique* rules (injected as "MANDATORY Domain
  Skill Rules") — a 1D spectrum is XPS *or* EPR, never a blend, and two
  technique skills would inject contradictory mandates. The agent-side
  selector therefore picks the single best technique match (or none); the
  result is capped to one.
- **Image / hyperspectral are *composable*.** Their skills are advisory
  ("Domain Expertise … use it to inform your approach") and often map to
  distinct pipeline stages (flatten → segment), so multiple may load.

The orchestrator can still pass an explicit multi-skill list to any agent;
the `exclusive` policy governs only the agent's *own* auto-selection.

**The agent-side selectors share one brain but differ in the signal they
feed it.** All three call `select_relevant_skills` and route through
`_load_skills_to_state`, but the *context* each supplies differs in
richness:

- **Image** — the actual pixels (scout montage / image bytes) + metadata;
  runs as a pipeline controller (`SkillSuggestionController`).
- **Curve** — metadata + data statistics + the rendered plot; pipeline
  controller (`CurveFittingSkillSuggestionController`).
- **Hyperspectral** — **metadata only** (`_auto_select_skills` in
  `analyze()`, not a controller; it does *not* inspect the datacube).

So hyperspectral is the weakest selection path and partly redundant with
the orchestrator (no signal the orchestrator lacks) — moot today with a
single `eels` skill, but when a second hyperspectral skill lands the fix is
to give its selector a real data signal (e.g. the datacube's mean
spectrum), making it data-aware like image/curve. The non-redundancy rule:
an agent-side selector earns its keep only where it reads data the
orchestrator can't.

**Authoritative `skill` vs non-binding `skill_hint`.** The orchestrator
defers skill choice to the agents by default, but can influence it two ways
via `run_analysis`:

- `skill` (authoritative) — for an *explicit user request* or a custom
  skill. Pre-loaded into `skills_loaded`; the agent's auto-selector is
  skipped (its skip guard checks `skills_loaded`/`skill_sections`). Honored
  as-is.
- `skill_hint` (non-binding) — for the orchestrator's *own* autonomous guess
  (from the `preview_image`/conversation context the agent can't see). NOT
  pre-loaded; passed into the agent's selector as a prior. The agent
  inspects the data and decides — confirm, augment, or override. **The agent
  has final authority.** Forwarded only to agents whose `analyze()` accepts
  it (signature-introspection, like `max_verification_iterations`).

This resolves the preemption risk: an autonomous orchestrator guess no
longer suppresses the agent's richer, data-level (and possibly multi-skill)
selection, while a genuine user request still binds.

### Live loops reuse the technique skills; steering knowledge is its own domain

A live measurement loop (`scilink/live/`) has no analysis skills of its own.
Its slow clock — the reference analysis and every re-anchor — is an ordinary
`CurveFittingAgent.analyze()`, so the same technique skill (`curve_fitting/raman`,
`xrd_profile`, ...) is auto-selected there as in a chat run; the per-frame fast
path replays a locked script and reads no skill. **Do not add a live-flavoured
copy of an analysis skill**: a technique missing from `curve_fitting/` is
missing for chat runs too, and that is where it gets added.

What *is* specific to a running experiment is how to steer it, and that lives in
the knowledge-only `skills/acquisition/<technique>/` domain with its own section
vocabulary — `overview · tradeoffs · limits · quality · strategy` (declared in
`loader._DOMAIN_VOCABULARIES`, like optimization's). It is read by the loop's
slow-clock consumers — today `LLMRecommender`, which selects one skill through
the shared selector (exclusive: one technique per measurement) on its first
call and records it on every recommendation as `acquisition_skill`. Two
boundaries: acquisition skills are **per technique, not per instrument** (a
vendor's API, limits and file format belong to the `Instrument` subclass and its
`schema`), and they stay out of the `run_analysis` skill menu, like
`data_preparation`.

**A live loop makes three judgements per stream, and they are deliberately
separate.** (1) *Does the recipe still fit this frame?* — the fit gate, the
agent's R² verdict relaxed to what the reference achieved in units of its own
noise (`live/gates.py`). (2) *Has the data changed?* — a graded, model-free
signal (`live/drift.py`): the share of a frame that the frames seen so far
cannot describe, read from the data alone so it cannot depend on which recipe
was locked (it replaced a thresholded peak-counting fingerprint that saturated on
rich patterns and gave opposite verdicts on the same series under two recipes).
(3) *Is a recipe that fits also right?* — no gate can answer that; only a second,
independent analysis can, so the loop runs **audits** (the re-anchor worker with a
different adoption rule) and compares named outputs. What the slow clock does
follows from which judgement fired: a run of frames the recipe fails on is
rebuilt; a run that fits but looks different follows `on_change` — `report`
(default: accept the state for tracking once the changed frames agree, no model
call), `audit` (agreement keeps the recipe, disagreement adopts the audit's) or
`rebuild`; periodic audits only report.

**A lasting change is always announced, and never quietly absorbed.** A locked
recipe is a hypothesis about what the data looks like, and in discovery work the
hypothesis breaking is the result. So before anything is rebuilt or accepted the
loop emits a `novelty` event — how much of a frame is unlike the stream so far
and WHERE on the axis (new / missing / shifted / broad, and `window` when the
frames now cover less of the axis: they are located where they overlap), read from
the data with no model — once per change, after the usual patience so a glitch is not a
discovery. The recommender is told (and asked at once: where the data is new is
usually where to measure next), and the follow-up — a thorough analysis of that
frame in chat — is one click for a person, never automatic. Do not add a path
that makes a change disappear without that event. Free-text notes from the user
travel in `system_info` as context for every model-driven stage; they inform,
they do not constrain. Two analyses are never asked to agree better than one agrees
with itself (the output's own frame-to-frame scatter), and a state accepted once
is remembered so the same kind of region is not asked about twice. The frame that
announces a change never also accepts it: a driver may pause there, and what is
decided at the pause comes before the state is taken as normal. The monitor
works on any 1D curve, which is how it watches a datacube (its mean spectrum,
whole and by region).

**The loop is modality-neutral; what a frame IS lives in `live/modality.py`.**
Two clocks, flags, patience, the change signal, novelty, audits, pausing, the
recommender and the log do not care whether a frame is a spectrum or a datacube.
What differs is a small adapter: which agent locks and replays the recipe, how a
result becomes flat features, what "still fits" means, and which curves the
change signal reads. An instrument declares it (`Instrument.modality`), the loop
follows. `HyperspectralModality` is the datacube instantiation, built from the
series machinery rather than beside it: the fast path is the existing locked
replay made strict (`analyze(strict_replay=True)`: the whole run with ZERO model
calls — no skill selection, no execution repair, no salvage or not-measurable
judge, no synthesis; a script that raises fails the frame) and judged by
`_replay_map_gate` against the reference's own map statistics; a rebuild or an
audit is a `locked_targets` run, so tracked quantities keep their names by
construction and nothing is pinned. Tracked features are per-map MEANS and
scalars (a map's min and max are the extremes of a noisy field). What a modality
does differently is a method or a capability flag on it, never a branch in the
loop: a curve's snippet edits are applied per frame while a cube's are baked
into a copy of the anchor run (`bake_edits`); a curve's portability is judged on
R², a cube's on its OUTPUTS (under a change of signal level each must stay put or
scale with the counts, else the script carries a constant read off the
reference). Pinning and window re-anchors stay curve-only on purpose: a cube
rebuild has fixed targets and no planning step, which is what a window serves,
and it would multiply a rebuild that already takes minutes. A cube is watched by
REGION as well as whole (`DriftBank`: one monitor per curve, the frame is as
changed as its most changed region), on a small pyramid of grids (2x2, 3x3, 4x4
while a region keeps 9 pixels), because a change confined to part of the field
is diluted in the mean spectrum by the area it covers and a feature that
straddles the blocks of one grid sits inside a block of another; the novelty
names the smallest region that holds it. **A region borrows what the whole
field has learned** (`DriftBank._lend`): its own frames are too noisy to learn a
slow real change, which then accumulates until the region misfires (measured:
half the quiet frames of a noisier simulated series, and false novelties in a
region for a shift of the whole field), while the field sees that direction at
full signal. Do not add a per-region monitor that stands on its own history. Several first cubes are a
reference too: they go to the hyperspectral series driver (scout, regime plan,
anchor, replays) and the loop locks the recipe of the regime the LAST cube
belongs to.

`ImageModality` is the third instantiation. Its fast path is
`ImageAnalysisAgent.analyze(strict_replay=True)`: an ordinary image reuse still
made about six model calls (skill suggestion, planning, plan validation, one
vision review, the tier-2 decision, synthesis); a strict replay makes none, does
not repair a script that raises, and writes no report. **An image analysis has no
R², so its replay verdict is evidence of method HEALTH only** (`_replay_feature_gate`:
the approved script still reports every quantity it reported on its reference,
finite, and still finds something). It cannot see a segmentation that runs and
is wrong. Observed live on a simulated coarsening series: when a second
population of small particles nucleated, the locked recipe kept counting only the
large ones (60 of 117) and its gate said good; what caught it was the change
signal, on the exact frame. So for images the change signal and the audit are not
extras, they are the correctness checks, and nobody should be told an image frame
was "verified". Tracked names are only ASKED for (in the objective) and therefore
checked: a rebuilt recipe that does not report a tracked output is refused
(`require_outputs_after_rebuild`). The change signal reads the radially averaged
power spectrum (log power on 96 linear bins from k = 0.01, whole field and
quarters, after a robust normalisation): chosen by benchmark on the simulated
series and on tiles of a real HAADF image, where linear-power variants missed a
defocus blur and doubled noise and misfired on slow coarsening, and an intensity
histogram added nothing. A located change is reported as a LENGTH SCALE
(`annotate_where`), not a spatial frequency.

**For images, a change that "still fits" is audited, and an audit is not a vote.**
`ImageModality.default_on_change` is `"audit"` (curves and cubes report). What
the live runs taught, in order: (1) one quick image audit can be the worse of the
two analyses (an 18 nm diameter against the recipe's 6.6), so a disagreeing audit
asks for a second, deeper one (`audit_needs_second_opinion`); (2) a deeper audit
that is NOT told what changed shares the recipe's blind spot (it left the newly
nucleated particles out exactly as the recipe did, and "agreed"), so **every
analysis made because of a change is told what changed and where**
(`_what_changed` → `hints`: model-free, context and not a constraint; told, the
same audit went from 56 particles to 89 of 105); (3) a vote is not the truth: when
the second audit sides with the recipe the result is an `audit_split`, the recipe
is kept and the state is NOT called verified, and the dissent stays on the record
and on the page; (4) a recipe that two independent analyses both reject is
replaced by the deeper one even when the two do not agree with each other, marked
`contested`, because keeping it is the worst of the three choices. An audit that
could not be formed (no value for a tracked output) is retried once at a deeper
profile. Do not collapse these into a majority rule.

**A pause is when the slow half of a discovery runs.** `loop.assess_change()`
(`live/discovery.py`) is the chain of the first SciLink paper for one frame: an
analysis of the changed frame told what changed, its scientific claims, and with a
literature key a novelty score per claim; the outcome goes on the log as a
`discovery` event and the Live tab shows it on a paused run. Without a key the
claims stand and the result says the literature was not asked. The key comes
from the call or from `FUTUREHOUSE_API_KEY`. Run live once: the chain scored a
claim 4 of 5 on a question that conjoined several specifics, which the
literature rarely matches as a set. The score is only as good as the question,
so the explanation is shown with it and nothing acts on the number.

**What an instrument learns outlives the run** (`live/instrument_home.py`,
`~/.scilink/instruments/<id>/`, opt-in with `remember=True`). Recipes are kept per
instrument identity, not per chat session. At `setup` the instrument's known
recipes are tried on the reference by strict replay before anything is analysed;
one that fits AND reports what is tracked arms the loop with no model call, and a
rebuild tries them too (`recall_known` is shared by setup and the worker). A
recalled recipe is a hypothesis about the new sample: it is replayed and judged
before use, and watched like any other after. Opt-in for that reason. A recipe
taken from the store is COPIED into the run before it is replayed
(`_own_anchor`): the store is trimmed and a person can forget a recipe, and
neither may break a run that is using it. The store is readable without a
session (`known_instruments`, `remembered`, `forget_recipe` in the same module):
`scilink instrument list/show/forget` and the Live tab's Instrument memory card
are two views of those functions, and on a shared multi-user server the card is
read-only. A recipe adopted as `contested` is remembered as contested and tried
after every verified one. Recorded data is not an instrument called "replay": a
`ReplayInstrument` is remembered under the instrument its metadata names
(`system_info["instrument"]`, "Recorded on" in the tab), else under its folder,
so two folders never share a memory by accident.

**The tab has three real sources and one of them is the recommended shape.**
"Your instrument" is an MCP server (any language, works on a shared server) or a
Python `Instrument` class on the machine (local only: importing runs code);
"Data you already have" replays a folder (how a recipe is built and checked
before beam time, and how a recorded series is run); the simulated experiments
are NOT on the form at all: they stay in the library for the tests and the
MCP demo server (`python -m scilink.live.mcp_demo_server <name>` is how one is
tried through the tab, as an MCP instrument). The form opens on the MCP
choice. For all
three real sources the form completes what the source declares (`_complete`,
the same rule as for an MCP server): technique, sample, kind of frame, its
calibration and the tracked names, and what the person enters wins. A class
that only acquires is enough. Run live: a bare image class with the form
supplying the rest armed in 123 s and answered 8 of 8 frames; leaving the kind
on automatic hid the calibration fields and the analysis reported diameters in
nm with no field of view, so the form now points at them.

**What a driven run through the tab taught (real HAADF and EELS tiles, an image
instrument behind MCP), each now structural.** (1) A change that arrives slowly
is a change: the slow alarm applies `on_change` like the abrupt path does (an
image stream's nucleation was announced and nothing looked at the recipe).
(2) A run that ends with an audit still working waits for it
(`pending_work()`, `close(wait=True, stop=...)`, the tab's `finishing` state): a
24-frame run had cancelled the second audit of a disagreed change, which is the
answer the run was asked. Stop ends the wait; the instrument's run summary is
written after it. (3) The pause-time assessment asks for the tracked quantities
by name and its numbers go BESIDE the recipe's for the same frame (`compared`):
that, not the prose, is what a person at a pause decides on. An assessment whose
analysis failed is tried once deeper, and if it still has no measurement it is
`unmeasured` and says its claims come from looking at the frame. (4) For an
image or a cube the tab shows the frame itself (the reference while arming), not
only the curve the change signal reads.

**The fast path runs in one long-lived interpreter** (`executors.WarmScriptExecutor`,
`warm_replay=True`): a fresh process paid the recipe's imports on every frame (a
real atomic-resolution recipe: 7.5 s cold, 1.5 s warm, identical numbers). It is
for REPLAYING a verified script only: module state survives between runs, which
is the point and also why generated, unverified code never runs there. Each run
keeps its own working directory, the timeout is hard (the worker is killed and
replaced), Stop reaches it, and any failure of the worker falls back to a cold run.

Measured and declined, so nobody redoes it blind: finer region grids for IMAGES
(3x3, 4x4) add false novelties on a coarsening series (7 against 0), halve the
detection of a nucleation and gain nothing on real HAADF patches, because small
image regions hold too few objects for a stable power spectrum; quarters stay.
Pinning for images: the names asked for were honoured in every live run, and the
failures were missing VALUES, which the refusal check already catches.

A first `setup` on rich image data is slow-clock work and can take tens of
minutes (a multi-pass analysis, segmentation, the portability replays): it is
paid once per recipe, and with `remember=True` once per instrument.

**What a frame leaves on disk is bounded, and heavy assets are per machine.** A
live run is open-ended, so the loop keeps the newest `keep_frame_dirs` per-frame
folders (the log and the measured data are never pruned). Model weights a skill
tool needs are cached once under `~/.scilink/models` (`SCILINK_MODELS` relocates
it), never relative to the working directory: generated scripts run in a fresh
per-item folder, and a relative default re-downloaded a 770 MB ensemble into
every image's folder (live: 36 s a frame and a full disk; 8 s once cached, of
which 5 s is importing torch in a fresh subprocess).

**A change that arrives slowly has its own alarm.** Every frame of a gradual
onset is explained by the frames just before it, so no frame is ever suspected
and nothing is held. The loop therefore also watches how far the stream has
moved from its REFERENCE frames and announces that as a `novelty` with
`onset="gradual"` and a location read against the reference
(`locate_from_reference`), once the distance stays above `gradual_bar`, then
again only at double the distance; an abrupt change that was announced raises
the level past itself, so the same change is never news twice.

**A rebuild first tries what this run already knows.** The loop remembers the
recipes it has used and left (`_known_recipes`); the worker replays each strictly
on the new frame (no model call) and adopts the first the modality's own verdict
calls good (`source="recalled"`), before any new analysis. It is the in-run
analogue of asking the script bank first, and it is what makes a stream that
returns to a state (a mosaic crossing the same kind of region, a cycled sample)
pay for each state once. An audit never recalls: it must be independent.

**Onboarding an instrument has two halves, and neither is a SciLink class.**
The *driver* — how to talk to the controller, its file formats, its real limits —
is an MCP server in front of the instrument, in any language: one tool that
takes acquisition parameters and returns a measurement.
`scilink.live.MCPInstrument` turns that tool into the loop's instrument and
reads the parameters and their limits from the tool's own `inputSchema` (a
number with no declared limits is held at its default and never steered: the
loop does not invent safe limits). `scilink/live/mcp_demo_server.py` is the
reference server. A measurement is a curve (`x` / `y` in the reply), an image or
a datacube: the server says which in its description (`modality`), and an array
too large for a JSON reply comes back as a `path` to a file the server wrote
(`.npy`, an image file, HDF5), which is how a real microscope hands over a frame
anyway. The *knowledge* — how to steer this kind of measurement — is
an acquisition skill per technique, selected from `system_info["technique"]`,
which the server can supply through a `describe_instrument` tool. A Python
`Instrument` subclass remains the in-process alternative. Uploaded skills stay
markdown-only, so a driver never arrives through the skill uploader.

**Novelty has two layers, and a pause is what connects them.** The loop's
`novelty` event is *statistical*: this frame is unlike the stream so far, here,
with no model. Whether it is *scientifically* new is the question the original
SciLink pipeline asks (observation → falsifiable claims → novelty scored against
the literature → follow-up measurement or theory; arXiv:2508.06569), and
`assess_novelty` still asks it in analyze mode. The first layer is the trigger
for the second; the second is slow-clock work and never runs on a frame's path.
Where an experiment can wait, `run_experiment(pause_on=("novelty", "breach"),
on_pause=...)` makes the change a decision point: the instrument is asked to
hold (`Instrument.pause()` / `resume()`, mapped onto optional `pause` / `resume`
MCP tools; `can_pause` says whether the experiment is really held or only the
acquisition), the slow work happens while the sample is still in the state that
looked new, and the decision — resume, resume with checked parameters, stop —
comes back from a person (the Live tab's Resume / Stop) or a callable. A pause
needs someone who can end it (`on_pause` is required), is off by default, and is
recorded (`paused` / `resumed`) beside what the analysis saw.

**The live layer is written for an instrument-centric SciLink.** The expected
direction is one SciLink per instrument (the paper's "lab of labs"), so
`scilink.live` stays free of session, chat and orchestrator imports: the
contracts are `Instrument` (identity via `describe()`: id, technique, whether it
can pause), `loop_log.jsonl`, and the recommendation schema. Every run records
which instrument it served (`setup.instrument`), so recipes, accepted states and
acquisition history can later be collected per instrument instead of per chat
session. The web session is one host for a loop, not its owner; do not add live
features that only work through `server/live_api.py`.

### Comparison with Anthropic Skills

|  | Anthropic | SciLink |
|---|---|---|
| Folder bundle layout | ✓ (`SKILL.md` + siblings) | ✓ (`<name>.md` + siblings) |
| Description-based selection by the model | ✓ (system prompt, every turn) | ✓ (`run_analysis` tool param, when routing) |
| Section vocabulary | Free-form | Fixed six-section; off-vocab content captured under `extras` |
| Injection granularity | Whole `SKILL.md` once activated | Per-decision via `_get_skill_context(section=…)` |
| Bundled scripts | Model can read and run | Reference-only in markdown; runnable code as sibling `.py` registered via `TOOL_SPEC` |
| Multi-skill loading | Implicit (model loads whichever descriptions match) | Explicit (`skill: str \| list[str]`); active set gates tool visibility |
| Shared library across skills | Not a concept (skills are independent units) | `scilink/skills/_shared/` — always-on infrastructure |

Conceptually: Anthropic skills are *independently distributable units the
model picks at conversation time*; SciLink skills are *in-package knowledge
bundles selected by orchestrator tool routing*, with skill-gated tool
visibility doing what skill activation does upstream. The `_shared/`
carve-out is a deliberate adaptation for in-package code reuse — Anthropic
users would either duplicate the helper or split it into a standalone skill.

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

## API key handling

`SCILINK_API_KEY` is the **proxy** key. It pairs with `base_url=<proxy-url>`
to authenticate against an OpenAI-compatible internal proxy (AI-incubator-
style deployments). It is *not* a vendor-neutral credential and must never
be handed to vendor endpoints (`api.anthropic.com`, `api.openai.com`,
`generativelanguage.googleapis.com`, …) — vendors reject proxy keys.

On the direct LiteLLM path (no `base_url`), the api_key comes from one of
two sources: the caller passes it explicitly, or LiteLLM auto-discovers it
from the conventional vendor env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY`, …). SciLink agent constructors must NOT fall back to
`SCILINK_API_KEY` on this path — when no vendor key is available, raise
`APIKeyNotFoundError` with a message naming both fixes (pass `base_url` to
use the proxy, or set the conventional vendor env var for direct API
access). `BaseAnalysisAgent`'s internal-proxy vs public-LiteLLM branching
is the reference shape; new agents mirror it.
