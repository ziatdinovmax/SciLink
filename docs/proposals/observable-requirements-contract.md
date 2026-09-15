# Proposal: observable requirements as an upstream contract, checked (not inferred) by the gate

## Context — what we have, and where it's weak

The pre-run dry-run gate now includes an **observable-coverage** check
(`RunCritic.assess(check_observables=True)`): before an expensive HPC run it
reads the research goal and the deck and blocks when a required observable's
data would be permanently unrecoverable. The blocking rule is **recoverability**
— block only if the raw data is neither logged nor reconstructable from a saved
trajectory (no trajectory at all, or a quantity that needs per-step
forces/virial or a flux). Resolution/cadence never blocks.

A variability sweep (opus-4-6, 12-shot, `SciLink-Benchmarks/benchmark/
test_observable_coverage.py`) shows the honest limit: the gate is **1.00-
consistent and correct** on structural/dynamical and no-trajectory cases, but
the **transport case (viscosity) sits near 0.67** — the model reliably reasons
"viscosity → stress tensor → per-step virial → not in a trajectory" only about
two-thirds of the time. Prompt-tuning swings it and can backfire (a domain hint
naming `fix ave/time` made the critic match the deck's *unrelated* `fix ave/time`
lines and conclude coverage was present, dropping the block rate to 0.17).

The gate is fail-open and a strict improvement over the status quo that lost
S2/S5 viscosity, so it stays. But it is doing the hardest reasoning in the worst
place: **inferring the required-observables set in a single headless shot from a
terse goal string, at the last second before submission.**

## The shape — move requirements upstream; the gate checks, not infers

Establish the required observables **earlier and explicitly**, as a first-class
part of planning / experiment design, and carry them into the simulation as
context. The gate then **checks the deck against a stated requirements list**
rather than re-deriving it. Inference becomes carried context — which sidesteps
the unreliable single-shot judgment entirely.

Three reasons this is more reliable than the current inference-at-the-gate:

1. **Inference → explicit context.** When the workflow arrives from analyze/plan
   ("validate NMR T1 and viscosity against these measurements"), the required
   observables are *known*, not guessed. The gate checks presence against a list;
   it does not have to know that viscosity implies a virial-based accumulator.
2. **Dedicated attention.** "What must this campaign measure?" as a focused
   planning decision is far easier than the same question buried inside a
   pre-run critic juggling setup + coverage + physics in one call.
3. **Human-in-the-loop at the seams.** Under co-pilot/autopilot the "your goal
   names viscosity but the deck logs no stress — add it?" question can surface to
   the researcher instead of needing to be fully autonomous.

## Where it lives

- **Producer (upstream):** the planning stage — and, in a real multi-turn
  scenario, the **meta agent's cross-mode context bridge** — emits an
  engine-neutral `required_observables` list as part of the goal/plan handed to
  the simulate orchestrator. This mirrors the `DeployedPotential` pattern: a
  small neutral descriptor crossing an agent boundary.
- **Consumer-declared complement (deterministic):** each analysis/post-processing
  tool already knows the input it needs (Green-Kubo viscosity needs a stress
  series; MSD→diffusion needs positions). Those declarations are a deterministic
  source for the requirements list — adding a property touches only its own tool,
  not a central table.
- **Gate (backstop):** `check_observables` gains a mode where, given a supplied
  `required_observables` list, it verifies each is captured — deterministic
  presence-checking against a known list, with the recoverability rule reserved
  for the "no list supplied" fallback. The gate stays fail-open.
- **Which cases to escalate is measurable.** `majority_coverage`'s vote
  agreement is a validated confidence signal: in the variability study the votes
  split on the borderline transport case (7/8 trials) and were unanimous on the
  clear cases (0/24). So the gate can *tell* which decisions it can auto-make and
  which to defer to this upstream contract or a human — auto-decide on unanimous
  votes, escalate on a split. (Majority voting cannot *fix* the borderline case —
  its base rate drifts around 0.5, so voting amplifies whichever side a run lands
  on — but agreement-as-confidence works.)

## What it needs (not available today)

- **Simulation delegation wired into the meta agent** — currently the documented
  deferred lazy seam (`delegate_to_simulation`). Requirements-carrying is a
  reason to build it.
- **A `required_observables` field on the planning→simulation handoff**
  (`run_task` context already carries a `context` dict; this is a typed entry in
  it), engine-neutral (property names + "must-be-logged-live" flag), realized to
  engine outputs by the skill.
- **A check-against-list mode** on the coverage gate, alongside the existing
  infer mode.

## Relationship to existing decisions

- Keeps the gate (`project_observable_coverage_gate`) as the cheap fail-open
  backstop; this proposal makes it a *checker* when an authoritative list exists.
- This is the structured form of the "observables contract" deferred in the
  earlier critic-vs-orchestrator design discussion — the gate was the 80% now;
  this is the reliable upstream version.
- A deterministic transport-observable check (does a transport-coefficient goal
  emit a stress/flux accumulator?) is the alternative for making viscosity
  reliable *without* upstream requirements; deferred because there is no mature
  library to make it cheap, per the deterministic-critic rule. Upstream
  requirements are preferred because they generalize to any property.

## Open questions

- Author of the list: the planner, the scalarizer, or purely the consumer-tool
  declarations — or a merge of planner intent + tool-declared inputs?
- Reconciliation when both an authoritative list and the gate's own inference
  disagree — does the list win, or does inference augment it (advisory)?
- Does the deterministic transport check belong here as a per-skill declaration
  ("this property needs a live virial accumulator"), which would also fix the
  0.67 case as a side effect?

## Who checks what, where — detection is distributed, the contract is owned

The contract above says *what* must hold; it does not yet say *who checks it and
where*. The tempting answer — a meta-agent pass that reads every agent output and
every input file and hunts for discrepancies — is the wrong shape, for the same
reason the gate's single-shot inference sits at ~0.67 on the transport case:
open-ended "find any contradiction across everything" is maximal-context,
unbounded reasoning, exactly where the model is least reliable. Scaling it to all
artifacts makes it worse. The vote-agreement data is the tell — unanimous on
scoped/clear questions, split only on the one fuzzy judgment.

The reliable shape is the inverse: **turn "find contradictions" into "verify a
declared requirement," and run each verification at the boundary where it first
becomes decidable, with minimal context.** A contradiction is a requirement some
stage cannot satisfy; the stage that produces the deciding artifact is the one
positioned to check it.

| Contradiction (instances seen) | Decidable at | Checked by | Check kind |
|---|---|---|---|
| Goal names an observable the deck never logs (viscosity → no stress) | deck-gen / pre-run | dry-run coverage gate | signal present |
| Observable needs finer sampling than the deck provides (T1 → sub-ps dump) | run-config (pre-run) | gate, vs tool-declared cadence | cadence / duration |
| Required species distinction not realizable (Zn–O_triflate vs Zn–O_sulfone share one atom type) | deck-gen | deck-gen, via type→molecule resolution | selection realizable |
| Property under-converged given the actual dynamics | post-run | refinement assess (per-phase quality) | empirical adequacy |
| Independent runs of one series use inconsistent definitions | across runs | orchestrator / meta | cross-run consistency |

Two roles fall out, and they are *not* the same role:

- **Within a run, checking is local and distributed.** Each boundary verifies its
  own artifact against the carried contract, reasoning about *one* question with
  concrete inputs. No stage re-derives global intent.
- **The meta / orchestrator owns exactly two things:** *carrying* the contract
  across mode boundaries (it is the one place that sees cross-mode intent —
  "validate T1 *and* viscosity" — its existing context-bridge role), and the
  *reconciliation no single stage can see* (the cross-run/series row). It is
  **not** the within-run discrepancy inspector.

This is why series-consistency belongs to the meta agent but the oxygen
resolution belongs to the deck stage: same principle, different boundary. Only the
orchestrator sees the *set* of runs; only deck-gen sees the type map.

### Why this is general and extensible — and where it stops

General: every contradiction we have hit is "a declared requirement a stage
cannot meet," and each lands on one of a *small, fixed set of check kinds* —
signal-present, cadence/duration, selection-realizable, empirical-adequacy,
cross-run-consistency. A new *property* declares its needs in those terms
(Green-Kubo viscosity → "signal present: virial/stress series"; species-RDF →
"selection realizable: these molecule-resolved groups") and touches only its own
consumer tool — no central table, mirroring the `DeployedPotential` neutral-
descriptor pattern. The requirement is engine-neutral; the skill realizes it to
engine outputs, so a new engine is one skill bundle, not N new checks.

Where it stops, stated plainly so the claim stays honest:

- **It only catches contradictions against *declared* requirements.** An
  observable no one put in the contract is checked by no one. Contract
  completeness is the residual burden — but that is a more tractable, auditable
  place to put it (a planning decision with a human seam) than "detect arbitrary
  contradictions," and it is the same limit any approach has.
- **Some adequacy contradictions are only decidable after running** (a property
  under-converges because the system moved slower than expected). Those cannot be
  a pre-run gate; they route to the existing post-run assess stage. The contract
  still defines "good"; the boundary that can decide it is simply later.
- **The check-kind set must stay small.** Extensibility holds because new
  properties *reuse* kinds; a genuinely novel kind is a real (and rare) addition,
  not a free one. If properties routinely needed bespoke kinds, this would
  collapse back toward the holistic inspector — worth watching, not assumed.

This answers two of the open questions above: the list author is a **merge**
(planner intent + consumer-tool-declared inputs), and *who checks* is **the
boundary, not the meta agent** — the meta agent owns the contract and the
cross-run row only.
