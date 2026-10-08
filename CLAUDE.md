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
physics; a curve follower whose replay pins a parameter at its bound likewise
keeps the recipe verbatim, is flagged `pinned_at_bound` and is withheld as a
degenerate fit, never relaxed into bounds of its own, #726 — a pin that
withholds is one on a TARGET component: the fitting script declares
`targets`, the components whose parameters answer the plan's
`parameters_to_extract`, and a pin on any other — a background, a baseline,
an overlap — is a caveat (`secondary_pins`, its value reported as no value)
that withholds neither the unit nor, through its anchor, the regime, and is
named by a non-refit `secondary_pin` flag; a fit that declares none is judged
as before. The declaration is FROZEN within a fit — the first stands, a later
attempt can only add to it, a follower starts from its regime anchor's — so a
correction cannot shrink `targets` to clear a target pin, #742; in a curve fit a pure lineshape limit is not a pin — a mixing fraction at exactly 0 or 1, one width of ONE line's Gaussian/Lorentzian pair at a floor small beside the width that carries the line — and a centre held at an end of the measured axis (or at a bound past it, on a window that reaches into the data) is a band peaking outside the range: NOT MEASURED, its whole component reported with no value, target or not, while the other bands stand and the unit carries a non-refit `not_measured` flag — only a fit with no target left measured (its declared targets that are fitted components, else every fitted band — a background measures nothing asked for) is withheld, #761), then trend codegen and a series synthesis run over the per-unit
feature table. The per-unit rows are written to `series_analysis_results.json`
in one shape, so `feature_table.write_feature_table` and every downstream
consumer read all three the same way. Every unit is a row of `features.csv`,
a failed one too, with `verified` / `flag_reason` status columns (annotations,
never counted as missing values; a consumer that skips rows with a missing
value names them; the planning ingestion also skips a row whose `verified` is
False and names it with its `flag_reason`, `include_unverified` opting in, #737),
and the series' control variable is a column unless a
sidecar column already carries it under the name older tables had (#723). **A later reuse of a series run replays
its locked recipe** — the first regime's, recorded in `locked_recipes` when the
anchor's script was locked, the script the feature table rests on — never a
later refit of the anchor (the refit's script stays as that unit's
`scripts/<unit>.py`; a person who wants the better model starts a new series
from it). A curve reuse of a series that locked several regimes replays the
recipes in lock order and keeps the first the R² gate calls good; an image
reuse replays the first regime's and says so (its verdict is one vision
review, not a gate), and so does a hyperspectral reuse, its recipe held to
that regime's own map gate: a folder of several records files (a series run's
datasets, the board's copies of its regimes) is resolved to ONE recipe — the
first by lock order, else by name — never merged, and of several paths the
first is replayed and the rest named (`script_reuse.source`, #751).
`prior_analysis_paths` also takes a recipe FILE (`.py`,
or `dynamic_analysis_records.json` for a cube) through the same replay gate,
so the swarm board's copy under `swarm/recipes/` is replayable — as a recipe
only: a file is not a run, so the realtime profile, the live loop's anchor and
an image strict replay (which need the run's config, fingerprint or reference
features) refuse it. The pick and its reason are `reuse_validity.source`
(`_verification_record.prior_recipe_scripts`).

**What a curve fit reports is held to the fit and the data** (#762). Code
generation is told where x and y sit in `data.npy` (from the array staged), and
every prompt that writes, repairs or judges a replayed script says never to
pick a column, a window or anything else by matching a value read off one
spectrum (bank adapt, whose rules re-derive windows for the new data: never a
column); the repair of a saved-fit mismatch and bank adapt, which never see the
generation prompt, are told the layout too. A saved fit that follows the x axis
is not a fit of this data: the ladder is told, and a unit that still shows it
FAILS (refit-eligible), never verified on its self-reported R². This repairs a
follower's locked recipe, a deliberate exception to #726: the recipe read the
wrong column of that unit's data, it did not merely rail a bound. A REPLAY
whose saved fit does not follow its data even up to a constant offset (R² < −1:
a misfit more than twice the data's variance) where its anchor was not below
zero only WITHHOLDS the unit (`replay_misfit`, a non-refit flag, read by a
follower's and a reuse's verdict): the recipe is kept, nothing is repaired —
unless its gate failed too, which refits it as on `main` (the misfit stays on
`fit_quality`, so the unit is still withheld if no refit replaces it). A
windowed recipe's local baseline carried across the whole axis can land there
on a healthy unit, and in live A/B runs (real series, reuse across a phase
change, the wrong-column replay) the x-axis rule caught every wrong replay
first, so this check withholds a certificate and never decides a repair (#711).
The anchor's reference travels with the recipe
(`certification_reference.saved_fit_r2`), so a reuse is held to it; a bank cold
start, a bare recipe file, a unit script named inside a series run and a run
recorded before this carry none. A saved fit below a flat line on its own is no
error (a peaks-only fit saved without its baseline, a background rising along
the series, a windowed model evaluated outside its window, a level on a flat
control). A band — read only when a bare position name (`center`, `cen`,
`loc`, …) AND a bare width name (`fwhm`, `sigma`, …) sit on one component; a
unit suffix, a derived quantity or `mu` (a level's mean as often as a
position) is not read — whose REPORTED centre lies outside the axis is not
measured like an edge band; a fit with no target left measured is withheld
(`no_target_measured`, read by its own gate, a follower's and a reuse's
verdict), never failed. A band width wider than the axis, the reported
uncertainties of a degenerate fit and a secondary pin's component's
uncertainties are no value (`withheld_values`, a caveat). An uncertainty is
read by structure first (`X_err…` beside a reported `X`, a bare `err` beside a
`value`), then by a word of its name, never a fit metric or a `_std` (as often
a measured scatter); a flag is never nulled. A unit keeps one flag, and
values withheld that the flag does not name are added to its text. All measured on 4,595
saved units. Declined, measured: holding a series' claims on fired follower
`regime_checks` (they fire on 205 of 259 followers, healthy series included).

**A series has a second shape: joint** (#754). Whether a set of files is a
series used to be purely structural (two or more files, one unit each), so a
method whose result exists only over the set was split into units that could
not compute it — and a unit script constructed the measurements it lacked,
measured its own assumption back, and passed every gate. The series planner
of each agent now declares `analysis_shape`: `per_unit` (the default, the
series mode above, byte-for-byte) or `joint`, and a joint plan is run as ONE
analysis whose inputs are every unit, staged with a manifest
(`exp_agents/_joint.py`: the planner rule, the declaration, staging and the
prompt text are shared; each agent supplies only how a unit becomes an
array). A joint run's script reads its own run's measurements by path, so it
is a record, not a recipe: the run's `analysis_results.json` (and a board
copy's sidecar) carries `analysis_shape: joint`, every reuse reader refuses it
with the reason through one check (`_joint.replay_refusal`, consulted by
`prior_recipe_scripts` and the hyperspectral records loader), and it is never
banked — until a replay over a new set of measurements is designed. The live
loop's `setup()` refuses a joint reference outright, and the fan-out's donor
picker never takes one. A board copy separated from its sidecar carries no
marker and cannot be recognised as joint: keep the two together. Every plan
gate shows the shape (a notice when joint), so Enter accepts what was shown.
**The measured inputs are the only inputs** (`_input_integrity.py`):
one principle per role — code generation (never construct an input the
method needs; return null with the reason), plan conformance (constructing
inputs is never a justified deviation), the verifier that drives retries (a
true "the data cannot support this" is the correct result, never fixed by
different input data) and the best-of-N judge (an honest null ranks above a
number resting on constructed data) — imported by all three agents' prompts.
A deterministic detector of constructed inputs waits for a measured corpus: a
word scan cannot tell construction from an axis built from a stated range.

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
field-mean feature INSIDE the `window` the declaration says it examined — a
band elsewhere in the cube is no contradiction, and "the 800 nm band is
absent beside a 523 nm one" is the common honest null; with no window the
judge decides) is repaired IN PLACE like an execution error — no judge
call, no ladder budget — and a required map that comes back entirely NaN
or with the wrong shape (a binned estimate not upsampled to the frame) is
diagnosed in the retry critique instead of "no further detail". The facts
list field-mean features in BOTH directions, labelled: a transmission band
is a dip, a Raman line on fluorescence or an X-ray white line a peak even
where the median sits high. Choosing one direction from where the median
sits was tried and dropped — it lost those peaks (#735 review); choosing
automatically waits for a corpus measurement (#722). The list is the field
mean's peaks and dips only, at most four per direction, not every feature: a weaker band, one local to
part of the field, or a step (an absorption edge) can be real and unlisted,
and the guidance says so, so a missing entry is never read as an absence;
with nothing listed, the facts say no peak or dip stands out — never that the
mean is featureless (#739). A
declaration stands when every REQUIRED output is absent or entirely NaN — a
diagnostic map beside it (a mask, an SNR map) is recorded with the
determination and never committed, since no review looked at it; a
declaration beside a VALUED required output is critiqued as not honoured,
not with the all-NaN text, which pushed live runs away from an honest null
(#723). The series synthesis is told whether a recipe locked and which
outputs a gate checked; a number no gate checked (a task's `scalars`) is
named as such there and in the single-cube synthesis (#722). A scalar that
comes from a fit is returned as `{"value", "role", "bounds"}` and CHECKED
(`_check_scalar`, #722 B1): at its declared bound (the curve agent's
`validate_bound_pinning`), or an amplitude at zero (below `AMPLITUDE_ZERO_SIGMA`
of the field mean's noise — never judged against a declared range — measured on 1,227
scalars of 156 real runs: failed fits at <= 2e-5 sigma, healthy from 0.65), it
is a failed fit: no value in the feature table, `failed_outputs` on the row,
named as FAILED on the board and in both syntheses. One that passes is
`gated: True` — passed against the bounds the SCRIPT DECLARED, nothing ties
them to the bounds the fit used; an amplitude is certified only from
`AMPLITUDE_CERTIFY_SIGMA` (0.5 sigma) up, stays unchecked between the two
bars (a fit to noise lands there) and with no noise estimate at all (empty
facts). A value outside its own declared bounds is unchecked, never passed:
those bounds are another parameterisation's (seen on real data: a Gaussian's
sigma bounds declared for its FWHM hid a width pinned in sigma). A failed
column is never aliased by the locked-schema completion, and a live frame
names a failed tracked number in `withheld`. The completion aliases only drift, never another number: a units suffix on ONE side, read from that side's own records (the schema source's units travel as `schema["units"]`), or a prefix with no uncertainty or noise word in it; two different units are two numbers, and a sibling such as an uncertainty stays under its own name, the locked column a gap (#752). A prefix naming another quantity by some other word is not recognised, and still aliases (`Band2_Depth` → `Depth`), as before. Nothing reads a name: what is checked is what the script
declared it fitted; a plain number stays unchecked, as before.

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
A swarm analysis item with a `data_path` gets the same per-branch budget
(`resolve_branch_budget`); a caller's `item_time_budget_s` is taken as is.
Under AUTOPILOT delegation the regime-plan gate reaches the user through the
normal feedback channel, like the specialists' other plan gates.

**Replays fan out.** `series_workers` (or `SCILINK_HS_SERIES_WORKERS`) runs
the locked replays in worker processes, each submitted the moment its
regime locks so replays overlap with the anchors still running in the
parent — replays are independent, and processes rather than threads keep
matplotlib and the sandbox executor out of each other's way
(`SCILINK_HS_SERIES_POOL=thread` exists for the offline tests). Each replay logs to its own
`dataset_NNNN/replay.log`; the parent's sandbox approval travels with the
spec. Anchors and refits stay serial: they are the LLM-heavy, human-gated
part. **A worker is a fresh interpreter, never a `multiprocessing` spawn
pool** (`utils.child_process.run_in_child`): spawn re-imports the caller's
script in every worker, and a driver script without a `__main__` guard ran a
copy of itself per worker on the same session — a new meta agent, model
calls, an overwritten checkpoint (#721). The child starts on the parent's
`sys.path` exactly (`-P`: a `scilink/` or `signal.py` in the working
directory never shadows it), and the target runs with the caller's own
`PYTHONPATH`, so a replay's generated script sees what a serial one does. Every orchestrator, agent and model
wrapper also refuses to start inside a spawn bootstrap
(`refuse_in_spawn_bootstrap`), which covers pools SciLink does not own. A
replay whose worker returns NO result (killed, or unable to start) is re-run
in the parent the serial way (the lost attempt's files set aside under
`lost_attempt/`) and named in `summary.replays_rerun_in_process`:
a pool failure is not a recipe failure, so it never becomes a refit. A
replay that returns a failure is the method's, and is refit as before. A
new worker pool (the swarm's process workers included) launches through
`run_in_child`.

**The replay gate, the verdict and the regime choice are shared policies.**
The three analysis agents each judged a replay of a locked recipe, said what
"verified" means and chose among a series' regime recipes in their own way,
and the copies drifted (#712). `exp_agents/_replay.py` now holds those
policies by composition — no base class: a `verdict_record` (one dict shape,
`verified` / `reason` / `decided_by` ∈ qc_gate · replay_gate · recipe ·
excluded · none / `interpretation_checked`) that every agent STAMPS where it
decides (a series unit at fit time, a cube or a run when its result is final,
`results["verdict"]`) and that `analysis_verdict` and the swarm board only
read; three replay gates with one verdict shape (`ScoreReplayGate` for the
curve's R² and the image's vision score, `FeatureHealthGate` for an image
strict replay, `MapReplayGate` for a hyperspectral map); `select_recipe`,
the one rule for choosing among a series' regime recipes on a reuse; and the
timeout policy (`_locked_exec.escalate_timeouts`: a slow script gets a doubled
limit, up to the cap, before it is called broken — hyperspectral included,
#699; the FIRST limit is never clamped, a started script finishes; the retries
are bounded by the run's deadline and the loop's budget; the sandbox's limit is
one type on both threads, `executors.SandboxTimeout`, because the message test
missed every worker-thread surface; a failed attempt's arrays are released
before the retry or the repair, the traceback's frames cleared along
`__cause__` / `__context__`). The
reconstruction from result shapes (`reconstructed_verdict`) serves
results from before the stamp and is the reference the stamps are held to
(`tests/test_verdict_parity.py`, no allow-list). A new agent-side decision
about a replay or a verdict goes into `_replay.py`, not into one agent.

**"Verified" covers the numbers; the interpretation only where a check
exists** (#711). A replay gate judges fit quality — R², a vision score, a
map's statistics — and says nothing about WHAT was measured: a recipe can fit
a different phase perfectly. So a replay is held to two more checks, and the
checks may do one thing only: **withhold certification**. They never decide a
verdict. *Did the recipe find what the regime found* — `identity_check`: the
names it assigned (a phase, a space group — normalised, compared as one
string or as token sets, so settings, spellings and a formula beside the
phase are one name and "TiO2 (anatase)" against "TiO2 (rutile)" is not) and
its STRONG positions (≥ 10 % of the strongest) matched to the units' by
nearest neighbour, never by index, with a floor of 1 % of the axis (recorded
with the recipe as `x_range`, so it is the same with and without the data
files). *Is the new data the regime's STATE* — a `live/drift.py`
`DriftMonitor` seeded with the regime's own units' data (`spectrum_NNNN/
data.npy` of the prior run, else the stamp), model-free, the live loop's
change signal. Each is a flag on the record (`identity.flagged`,
`state_flag`; a check that could not run says so: `state_check`,
`identity.reason`, `curves_not_seeded`) and a caveat in the message;
`interpretation_checked` only on clean evidence — identity within, and the
state distance under the CERTIFICATION bar (`CERTIFY_STATE_BAR`, tighter than
the flag bar `SAME_STATE_BAR`: a fixed-position recipe cannot report an
impurity or a low mixture, so the state distance is the only check that sees
them, and they sit between the two bars) — never for a run its own verifier
approved (the verifier reviews the fit, not the claims; a fresh cube's map
reviewer likewise). A one-unit reference certifies on the same evidence (a
spread is not required: a miss only withholds), and a series' followers are
checked the same cheap way against their regime's anchor (`_check_follower`
→ `regime_checks`), so a series whose anchor replayed a prior recipe can
post a verified claim when every unit certified; without that, a single-run
reuse and every series reuse were provisional forever. This is the curve
agent's: an image reuse has no identity check on its `reuse_validity` and
is never certified, so its claims stay provisional. A hyperspectral
replay's `identity_checked` is the map gate's range rule — every required
map inside the anchor's plausible range — a weaker certificate than the
curve's two checks. With no map gate (a single cube's run folder or its board
copy) the run's own `certification_reference` maps certify by the same rule
and never gate: the verdict is the run's, and a single-cube reuse can now be
certified, which it never was before #753. The verdict stays the gate's, on the fast clock and
off it. Why not a verdict: a deterministic check asked an interpretive
question ("same phase, read the same way?" — the question the live section
says no gate can answer) has the precision to withhold a certificate, not to
issue a negative one that drives fall-through, a judge and re-derivation;
three review rounds of a deciding identity check each moved its false
positives to the next threshold (index matching, the strong-peak bar, the
floor and the spread), and a false alarm that only withholds certification
is cheap. A negative verdict on interpretation is a separate, measured step:
a fixed corpus of cases first, a rule adopted only when it measures well
across all of them (the candidate: the monitor's distance after a bounded
alignment, one table for the regime choice and the verdict), the declined
variants recorded here. The swarm board posts a replay's CLAIMS as verified
only when the interpretation was checked — its recipe, a script that ran,
stays verified by the gate — and a series whose ANCHOR replayed a prior
recipe rests on the replay gate like the single replay does (followers that
replay the series' own anchor, a hyperspectral series, are its mechanics).
**Which regime a reuse replays is read from the data** (#710), by the same
monitor: the nearest regime is tried first (`select_recipe(strategy=
"nearest_first")`), the GATE decides whether a candidate is kept (so a flag
on the nearest can never hand the choice to a far regime), and
`reuse_validity.regime_choice` says the distances, the choice and whether two
regimes were too close to tell apart (within a 2× ratio, or both under the
material bar) — said in `source` and `message` too, since the attribution is
read there. **A regime's stamp is its units' curves on the monitor's own
grid** (`drift_state_of_curves`: at most 12 curves evenly spaced along the
regime, each at its own x, a long one reduced by block means — never a point
interpolation, which samples the noise instead of averaging it): the
anchor's curve at lock time, re-stamped with the regime's units when the
series is done (`_restamp_regimes`, every series shape, a single regime
included; `drift_state_units` counts the curves the monitor seeded), because
an anchor-only stamp failed units of its own regime. A peak-counting
fingerprint was tried first and dropped: it separates phases and nothing
below them (a lattice shift, a texture, a background tie at 1.0), as the live
loop had already found. Images keep the first regime's recipe and say so.

**The replay gate is the gate the recipe was approved under, and a withheld
certificate is explained, not re-judged.** A replayed recipe is held first to
the gate recorded with it (`gate_record` on each regime's recipe at lock time
and on every run's results; the swarm board's copy of a script carries it in
a `<stem>.recipe.json` sidecar with the plan's model; the live loop's
series-derived anchor records the locked frame's regime gate) — a reuse run
resolves a gate of its own from whatever skill it was or was not given, which
is not the recipe's. A caller who asks for a gate on the reuse run
wins, with a warning (`quality_gate_explicit`: a full `quality_gate=`, or an
`r2_threshold=` that `resolve_gate` honoured and that the recipe's own R²
metric can take — a bare number cannot replace a recipe's figure of merit,
and a number the skill's metric guard dropped was no ask; a constructor-level
default is not an ask). One predicate carries the decision to every
reader of the gate — "this regime's anchor REPLAYED the recipe"
(`_replayed_regimes`, set where the anchor's result is known: a verbatim or
ladder-repaired replay counts, a reuse that failed into fresh code or a
regime fitted fresh beside a replayed one does not) — through
`_series_gate(state, regime)` and `_unit_gate_resolver` into the outlier
pass, the refit's re-scan and its scoring-gated skip. Off the predicate
everything is `main`'s path (a non-R² skill gate, else the driver's LIVE R²
threshold — never the R² snapshot in state, which once beat a person's
`adjust_threshold`; that adjustment is itself an explicit threshold ask). A
non-reuse series flags, refits and stamps exactly as `main`
(`test_non_reuse_series_are_mains_path_exactly`). Else the run's effective `QualityGate` (`_replay_gate`: an R² gate
at the driver's live `r2_threshold`, a skill's own metric at the skill's
threshold and direction, the rule `_detect_outliers` already applied to a
series' followers; a metric the replayed script does not report is a reject). The
deterministic checks decide verified / not verified and say THAT something
differs; what a difference MEANS — a thermal shift against a new band, an
impurity line, a polymorph — is a skill-bearing model's question, so a replay
that is NOT CERTIFIED for a stated reason — flagged on its state or its
identity, or whose regime the data cannot tell — is escalated to a JUDGE
(`_escalate_reuse`, the live loop's shape: a model-free signal triggers the
slow-clock look; a state distance merely above the certification bar is
withheld with nothing to explain, and is not): one model call shown the findings as
quoted data between markers, the replayed fit and the new curve over the
regime's anchor, and the skill's `interpretation` section, asked which regime
the measurement belongs to and what changed. The answer is an opinion on the
record (`reuse_validity.escalation`, `decided_by: judge`), in the message, on
the `analyses` row and on the board as a PROVISIONAL claim; the verdict and
`interpretation_checked` do not move, nothing is re-run, and a regime the
judge names against an ambiguous choice is `regime_choice.suggested` — listed
for the caller, never taken. Never on a certified or clean pass, on the fast clock, when
the caller asked for no review, on a replay that did not execute, on the
gate alone failing, and at most once per item; a flag against ONE reference —
one unit for identity, one curve for the state, where no spread is known and
a one-sample shift of a sharp step reads past the bar — is escalated only when
nobody attends (`enable_human_feedback` off), and the evidence says "one
reference curve" (#725). The evidence carries WHERE the state differs
(`DriftMonitor.locate_frame`, model-free): without it the judge guessed. With
no named regimes the judge may answer `same_as_reference`.
Do not make the judge a gate: the per-map reviewer that re-judged replays is
why replays are gated on evidence.

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

## Swarms: several delegations at once, on fresh agents

`run_swarm(work_items)` (`scilink/agents/meta_agent/swarm.py`) runs 2-8
delegations of any mode concurrently, each on an **ephemeral worker**
(`workers.build_child`, the one constructor behind the persistent specialists
too) in `<meta_session>/swarm/<NN>_<slug>/`, each an ordinary ledger
delegation. The design, its stages and what each stage left open are in
`docs/proposals/agent-swarms.md`; stages 0, 1 (#697) and 2 (the board, #702)
and 3 (reactions, #708) are on `main`, the replay policies the board's
"verified" rests on were settled between stages (#712: #713, #714, #717,
#715), and stage 4's LOCAL scheduler is on `main` (the worker contract and
its process placement, the measured table per item class, the token budget,
the circuit breaker, the guard shared with the fan-out — the proposal's "Stage
4 on `main`: the local scheduler"); the AWS worker tasks are their own later
PR with the hosted-campaigns work (the proposal's "Between stage 3 and stage
4" says why; "Since the stage-4 scoping" is the contract every placement
implements).
Settled rules, each learned from a live run or a review:

- **A swarm item is a fresh agent.** It does not remember earlier delegations;
  its task and context carry everything. The persistent specialists stay for
  conversation, where accumulating context is the point — and two concurrent
  `run_task` calls on one specialist would each report the other's output.
- **The coordinator is rules, not a model.** Admission by free memory (an
  item is estimated by its LARGEST unit, nested data included — a series runs
  its units one at a time, times the replay workers the agent itself resolves
  plus the parent — and a raw-instrument file, by its own embedded contract or
  its folder's, by its preparation, #724 — unless its CLASS was measured: a
  process worker's peak and tokens are recorded per `mode:kind:size:units`
  in `measured_items.json` under the SciLink home, and the next item of the
  class is sized from the largest run seen, `meta_agent/peaks.py`), a
  capacity plan that refuses an item larger than the machine, a memory guard
  that cancels the running item that HOLDS the most (what its process worker
  was last sampled at, else its estimate; never one running alone) and reruns
  it alone once, a wall-clock budget per item, a token budget reserved per
  item at admission and reconciled on completion (`budget.max_tokens`; a
  cancelled item is charged for what it spends while it winds down), a
  provider circuit breaker per model that pauses admission (never running
  work) while a model's last minute holds at least six retryable failures
  that are at least half its calls — a success counts toward the ratio and
  does not end the hold, since in a brown-out running work keeps succeeding
  now and then (`wrappers/llm_limiter.py`) — no item starting another. The capacity refusal and the guard are the fan-out's
  too (`fanout.plan_capacity`, `fanout.guard_memory`): a branch whose
  MEASURED class is larger than the host is not started (an input-based
  estimate never refuses a fan-out branch; the swarm refuses on it), a
  branch cancelled for memory reruns alone once.
  The model decides between swarm runs and inside each item, never within a
  run.
- **Where an item runs is a placement behind one contract**
  (`meta_agent/placements.py`: `submit(spec) -> handle`, `poll(handle) ->
  {state, result, peak_rss_bytes, stop_reason}`, `cancel(handle)`; states
  queued · running · done · failed · cancelled · out_of_memory ·
  interrupted). An analysis item with data runs as a PROCESS when nobody
  attends the swarm — a fresh interpreter through `run_in_child`, which now
  goes through the executor's tracked runner, so the item's cancel, the
  guard and the turn's Stop end its whole tree (descendants in their own
  sessions included), its console goes to `<item dir>/worker.log` and is
  relayed into the turn, and its memory is measured while it runs (the
  tree's sampled sum, else the child's `ru_maxrss`); a worker killed with
  nothing returned and no cancel asked is `out_of_memory` and runs again
  alone, once. Planning and simulation items, and every item of an attended
  swarm (its questions need the person's channel), stay threads in the
  coordinator's process and measure nothing. The spec is plain data with no
  provider key: each key travels as the NAME of the environment variable
  that holds it and the child reads that variable (an MCP server's `env`
  and `headers` do travel, over stdin, never on disk); a key under no
  variable (the API, embedding or FutureHouse key), or a callable tool
  extension, keeps the item a thread with the reason on its entry. A measured class is refused on its
  raw peak and admitted with ×1.2 headroom; a fan-out branch is refused only
  on a measured peak. `SCILINK_SWARM_PLACEMENT=thread|process`
  overrides the rule (the offline tests pin threads). An HPC job
  (`ClusterExecutor.submit/poll`, whose `cancel_check` now defaults to the
  waiting thread's own cancel) and an ECS task are the later placements of
  the same contract.
- **A gate nobody answers is never a human decision.** Worker questions go
  through one queue (`hitl.QueueChannel`), served on a thread of their own
  (`hitl.QuestionServer`) so the coordinator keeps enforcing its rules, tagged
  with who asks (`WorkerChannel`), and time out to the gate's default — with
  the clock restarting when the question is shown, and the wait left out of
  the item's budget. A timeout, including one inside a channel with its own
  clock (`hitl.mark_timed_out`, the MCP server's), is visible to the gate
  (`last_question_timed_out`): the planner then writes `unattended_gate`, never
  `human_review`. A plan settled by silence would be settled by nobody.
- **A cancel must reach a waiting worker.** A print is not the only place a
  Stop lands: a parked question, an LLM-slot wait, a backoff sleep and a
  best-of-N candidate all poll the thread's cancel
  (`log_context.register_cancel`, `inherited_context`). A new agent that spawns
  its own threads uses `attributed_to_current` or `inherited_context`, or its
  threads run on after their item is cancelled. A chat turn's Stop is added to its thread's cancel for the turn (`register_turn_stop`), never in place of one the thread already carries, so a new wait that polls the cancel ends on the user's Stop and on an item's own cancel alike. And every process SciLink starts is stoppable: model-written code runs through the script executor (`ScriptExecutor`, or `executors.run_generated_script` where a caller needs `subprocess.run`'s shape — SciLink's interpreter, no provider keys, the sandbox limits, consent asked first), and an external engine through `executors.run_engine` (Stop registration, its whole tree killed on a timeout, the parent environment kept); a new call site never calls `subprocess.run` on generated code or an engine (#685).
- **The provider and the stores are shared.** At most `SCILINK_LLM_MAX_INFLIGHT`
  calls per model are in flight (`wrappers/llm_limiter.py`, held per request,
  never across a backoff); usage is charged per worker; the distill staging,
  graduated skills and instrument homes take a lock for every change, and a
  reviewed skill upgrade refuses a skill that changed during the review.
- **The board is the record, and independence is a number.** Every finished
  delegation posts typed, small records to `swarm/board.jsonl`
  (`meta_agent/board.py`): a claim, a measurement, a recipe or structure by
  path, a parameter point, a hazard. One writer, never edited in place (a
  correction `supersedes`, a withdrawal is a `retraction`, and neither takes
  effect from an unchecked record of another author), and `verified` means a
  GATE the author's own pipeline runs passed it — the analysis verifier's
  approval (`analysis_verdict`, read from the shapes the agents write: a
  salvaged, unverified or unapproved result is "success" too and stays
  provisional; a series unit carries the verdict its driver stamped at fit
  time — an anchor or refit by its own gate, a follower by the recipe it
  replayed — and a failed unit the agent already excluded does not block; an
  approved fit with a parameter pinned at its bound is withheld as a
  DEGENERATE fit, named, never called salvaged (#726); a decision about a unit is made where the information is, never reconstructed
  afterwards from markers; the board keeps its own copy of a recipe under
  `swarm/recipes/<NN>_<label>/<analysis_id>/`, written once and never
  rewritten — a series' recipes from the `locked_recipes` its driver records
  where each regime locks (the curve, image and hyperspectral drivers alike;
  a cube's recipe is a records FILE, copied under its unit's own folder by the
  name `prior_analysis_paths` reads, with its map gate in the sidecar, which a
  replay of the copy is held to when its caller passes no reference, #734); and every copy carries, in its sidecar only (never in the board record: it can be large), the opaque `certification_reference` its agent stamped where it recorded the recipe — a curve regime's state and identity from its units, a cube's reference maps — so a replay of the copy is certified, flagged and escalated as a replay of the run is; a copy with none, and an image replay (no interpretation check by design), says why on its record (`_replay.NO_REFERENCE`, `NO_INTERPRETATION_CHECK`), #753 — so an agent's folder is never read again for it and the
  agents' own layout and reuse are untouched by the swarm; a verifier's
  physics approval inside the gate's soft band counts, a bypassed verification
  below threshold does not; a CLAIM of a run that also reported outputs no gate
  checked — a hyperspectral task's `scalars`, marked `gated: False` where they
  are produced and carried as `ungated_outputs` on the `analyses` row — stays
  provisional, its recipe verified, because the gate approved the maps and a
  claim may rest on the numbers beside them, #722 — unless every such number
  was declared and passed its fit-health check, B1; one that FAILED it is
  named as a failed fit, `failed_outputs`), a human-approved plan (an unattended one stays
  provisional; only the delegation that wrote or settled the plan posts it),
  the structure validator. Engine output and advice that passed no gate — a BO
  point, a steering reduction, a TEA summary, a critic's blocking finding — is
  provisional. The board adds
  no judge. A reader gets verified records only, as quoted data between markers,
  clipped and budgeted, under the same additive-only rule as steering; a swarm
  item reads once, at its start (`reads_board`, the newest 24), and a `check`
  item is refused a read by the API, not by prompt text. Every read is
  recorded (`reads`, exactly what was shown), so fusion's `independent_support`
  is computed, not judged: the largest set of fused branches none of which
  read, was steered by, or cited another in the set (the board's read graph
  joined with the ledger's `context_from` and `steered_by` edges, keyed by
  delegation index, exact up to 12 branches and a stated lower bound beyond),
  rendered with what it cannot see (a finding pasted by hand with no citation).
  A shared dataset (co-registered operands) is not a coupling of findings. A
  new way of coupling two delegations becomes a read on the board, not a new
  prompt caveat. A
  record describes its artifact as it was when posted; a later edit is a later
  delegation's record.
- **A reaction is a rule the meta declared, applied by the coordinator.** A
  swarm's `subscriptions` (`meta_agent/reactions.py`) say *when a record of
  this kind, subject and status is posted, enqueue this item*; the coordinator
  fills the item from the record (a fixed vocabulary of `{finding.*}` fields,
  one substitution pass, no model) and launches it as an ordinary item, and
  stamps its cause and causal `chain` on the entry at that moment. Every
  bound is a number: a finding fires a subscription once, `max_fires` per
  subscription, two re-triggers per subject, the swarm's item limit, and a
  cycle — the same `(mode, subject, kind)` hop twice in one chain — or a
  record at the end of a supersede chain of three is refused with the reason
  on the triggering entry. Workers ask (`suggested_followups` become
  `task_request` records, never read by default) and the coordinator decides
  (an item only through a subscription on that kind); no worker starts a
  worker. A reaction's cause is one of its reads (its task quotes the
  finding, as labelled data, never as an instruction), so what the cause
  rests on, the reaction rests on. A withdrawn finding (`retract_finding`)
  taints everything that rested on it — derived in the fold from the read
  graph, so a late post that read it is caught — and the re-run is the next
  swarm, with the items prepared (never a reaction the withdrawn finding
  caused: that is a new decision). Who withdraws: a person at the gate,
  shown the finding and its dependents, Enter keeping it; with nobody there
  the model may withdraw the agents' findings but never a human's decision —
  not a human-approved plan's claim, not a finding such a claim rests on, not
  a retraction a person made (`decided_by`); a retraction is undone by
  retracting it, which retractions stand is decided newest first, and every
  act is judged by its previewed EFFECT (what it withdraws, taints or brings
  back), never by the record it names. A disagreement
  between independent results is reported, never settled by a retraction. A
  hazard on a subject reaches every reader of that subject whatever its
  `kinds` filter and whatever the newest-N cut, marked as the provisional
  record it is.
- **One PR per stage.** A stage is verified as a whole: the full suite against
  a `main` worktree by failing-test ids, and live checks on Bedrock from a
  frozen snapshot, one heavy run at a time.

A new sub-agent or skill inside a mode gets all of this without swarm code;
a fourth mode would not, and there will not be one.

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
vocabulary or the narration reader, not in one surface.** A swarm item's
lines carry its label at the line start (`[XRD 300 K] 💭 …`): the label is
the thread's worker label in `log_context` (registered by the item's
thread, inherited by every thread the item starts through
`attributed_to_current` / `inherited_context` — a best-of-N candidate's
lines are the item's too — and put through `narration.worker_tag`, so a
label the model wrote is always a tag the reader accepts: one line, no
brackets, at most 48 characters); the item's stream wrapper inserts it,
and the process worker's relay only when its thread carries none. Both
readers strip the tag (`split_worker_tag`), classify what follows with a
continuation state kept PER worker (item B's indented line is never item
A's thought), and keep the label on the line and in the activity ("XRD 300
K · Curve Fitting Planning"). The swarm and fan-out coordinators' own lines
are the visible `fanout` kind by a MARK the coordinators put on them
(`vocabulary.COORDINATOR_MARK`, through `fanout.coordinator_line` /
`_cprint` — every coordinator print in `fanout.py`, `swarm.py` and
`meta_orchestrator_tools.py` goes through it, and a test scans the source
for one that does not), never by wording or by an emoji an agent may print
too; the readers give them activity labels of their own. A fan-out
branch's lines carry no tag.

**A human-feedback gate declares what is under review; it does not print
it for the surfaces to parse.** Every gate holds a structured object at ask
time (a plan dict, a fit result and its review figure, a candidate list)
and used to print it and ask, so the web UI and the shell
showed the captured console text and regex-parsed it into widgets. A gate
now passes `subject=` to `request_human_feedback`: a title plus blocks from
the fixed vocabulary in `scilink.hitl.SUBJECT_BLOCKS` (text, fields, chips,
steps, table, figure, candidates, compare, notice), built by a pure
function next to the printer from the same dict (`fitting_plan_subject`,
`analysis_plan_subject`, `refinement_plan_subject`, `regime_plan_subject`,
`plan_subject`, `plan_candidates_subject`, `bestofn_join_subject`,
`consensus_subject`, `consistency_subject`). Every block carries a `label` that keeps the
console's section name and emoji ("🔍 Observations"); both surfaces render
labeled blocks in one aligned label column, so a gate lists its sections
the way its printout does and a short list is plain text, not chips. The decision widget and its words come from the
gate's `kind` (`vocabulary.QUESTION_WIDGETS` / `QUESTION_LABELS`), never
from the prompt text. Printing stays: it is the console, the verbose log and
the record; the surfaces show the blocks, not the captured text as well.
Every live gate declares one (the audit in
`docs/proposals/structured-human-feedback.md` lists them); the presenter
no longer parses console text, and a gate without a subject is shown as
its kind's widget over the captured text. A new gate gets a subject, never
a parser. This work improves how the existing gates are shown; it
adds no gate and re-enables none. Beyond plan approval and the best-of-N
candidate choice, no human review of analysis results is expected, and the
code agrees: attempt 0 of a best-of-N run is a candidate job, and every
candidate job sets `_suppress_human_feedback`, so under the orchestrator's
defaults the result-review, poor-fit, poor-quality and user-guided gates
never fire; the analysis review, the first-spectrum fit review, the mixin's
iteration feedback and the scalarizer's column confirmation (off since
#542) have no live caller at all; the tier-2 approval needs
`analysis_depth="auto"` while the orchestrator passes "basic". None of
these gets a subject. The reachability table is in the proposal.

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

**A skill tool states a caveat about the result itself** (#775). What a tool
knows about the result the script reports (a match that needed a fitted
lattice scale beyond what its reference cell allows) is printed as a
`TOOL_WARNINGS_JSON:` line, a JSON list of sentences, beside the
`DB_MATCHES_JSON:` marker `search_structures` prints; the curve agent lifts
those, and a `warnings` list in the script's own `FIT_RESULTS_JSON`, into the
unit's `caveats` (deduplicated, at most five), so the caveat does not depend
on the generated script copying it. Print it from the call that serves the
result reported (the XRD skill's `register_overlay`, for the plotted match),
not from a call made for every candidate. The figure drawn for a result is
drawn where its number was computed: an identification overlay goes through
the registration the scorer fitted (`register_overlay`), never the raw
reference pattern.

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
path replays a locked script and reads no skill — and is held to the gate the
reference's recipe was approved under (recorded on the anchor, also when the
anchor is a series' last frame laid out as one), not to whatever gate a frame
with no skill would resolve. **Do not add a live-flavoured
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
