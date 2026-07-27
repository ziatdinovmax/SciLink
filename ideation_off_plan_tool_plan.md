# Getting ideation off `generate_initial_plan`

**Status:** proposal, nothing implemented.
**Motivation:** `generate_initial_plan` designs *lab experiments*. Ideation
has been riding it, so a research portfolio is authored into an
experiment-shaped container and every consumer downstream has to unpick it.

## Why this is worth doing

The container mismatch is the common cause behind a cluster of bugs found in
the cdoc session (`meta_session_20260726_141723`) and its replay:

| symptom | cause |
|---|---|
| 56 pseudo-steps printed at a review prompt | portfolio had nowhere to go but `experimental_steps` |
| `plan.html` rendering a dossier as an executable protocol | HTML template is experiment-shaped |
| artifacts keyed off a best-of-N judge knob | no campaign-level notion of "this is ideation" |
| a build roadmap inventing 6 `optimization_params` | *document* squeezed into the same container |

Today's fixes (`concepts`, the shared direction card, `write_technical_document`,
campaign-kind artifact routing) are patches on the container. They are worth
keeping — they make the current shape behave — but they do not remove the
mismatch.

## What ideation actually needs from the plan machinery

Keep (all of this is genuinely shared, and must NOT be reimplemented):

- retrieval / grounding (`run_rag`, KB, campaign literature scoping #396)
- best-of-N authoring + judge + human override
- critic and conformance passes
- campaign state, checkpointing, restore, plan history
- deliverables ledger, run_task summarisation

Drop (this is what the mismatch is):

- `proposed_experiments[]` as the payload
- `experimental_steps`, `required_equipment`, `optimization_params`
- the protocol HTML report
- "experiment" vocabulary in prompts and console

So this is **one engine, two contracts** — not a new agent, not a fourth
mode (see CLAUDE.md: the mode universe is fixed at three, and plan-mode
skills stay knowledge-only).

## Target shape

New tool `generate_ideation_portfolio(objective, n_candidates=3, ...)`,
producing:

```jsonc
{
  "type": "ideation",
  "thesis": "one-line organizing idea for the portfolio",
  "directions": [
    {"id": "PS-1", "title": "...", "tier": "flagship",
     "hypothesis": "...", "rationale": "...", "novelty": "...",
     "details": ["..."]}
  ],
  "shared_protocol": ["optional, genuinely shared method notes"],
  "open_questions": ["what would change the ranking"],
  "literature_search": "...", "campaign_id": 1, "iteration": 2
}
```

`directions` is `concepts` promoted to the top level and renamed for what it
is. `proposed_experiments` does not appear.

## Decisions to make before coding

1. **State slot.** Reuse `current_plan` (with `type="ideation"` and the new
   payload) rather than adding `current_portfolio`. Reusing it inherits
   campaign scoping, literature stamping, checkpointing, history and the
   meta's `campaign_state` for free; a parallel slot duplicates all of it.
   Cost: `current_plan` becomes a union type — mitigated by (2).
2. **One accessor, everywhere.** Add `plan_directions(plan)` and
   `plan_is_portfolio(plan)` in `parser_utils`. Every consumer reads through
   them; nothing else may branch on payload shape. The accessor resolves, in
   order: top-level `directions` → `proposed_experiments[0].concepts` (today's
   shape) → a single synthesised direction from the experiment fields (very
   old plans). This is what keeps restore of existing sessions working.
3. **Refinement verbs.** `refine_plan_with_results` assumes experiments and
   results. Portfolios refine differently: add / drop / re-rank / harden a
   direction, fold in a critique. Needs its own path, not a flag.
4. **Deprecation.** `generate_initial_plan(selection_profile="ideation")`
   warns and forwards for one release, then the parameter goes.

## Amendment (during implementation): the transition shim

Migrating 53 sites at once is the 6–8 day estimate below, and every
intermediate state would be a half-migration. Instead the portfolio carries
BOTH shapes during the transition: `directions` at the top level (the real
payload) plus a minimal one-entry `proposed_experiments` shim derived from
it. Existing consumers keep working untouched; ported consumers read
`plan_directions()`. This is the same additive strategy that let `concepts`
land safely in PR #394, and it means every phase boundary is shippable.

The shim is what phase 6 removes, once no stored session needs it.

## Phases

Each phase ends green on the full suite with the failure set unchanged, and
phases 2+ end with a live check.

**Phase 0 — inventory (0.5 d).** Enumerate the 53 `proposed_experiments`
sites; classify each as lab-only, shared, or ideation-only. Output: a table
in this file. No code.

**Phase 1 — accessor + schema (1 d).** Add `plan_directions` /
`plan_is_portfolio` with the three-tier fallback; route today's `concepts`
readers through them. Pure refactor, no behaviour change — the proof is that
the goldens and the cdoc replay artifacts render byte-identically.

**Phase 2 — the authoring tool (1–2 d).** `generate_ideation_portfolio` +
`IDEATION_PORTFOLIO_INSTRUCTIONS`, reusing `generate_plan_candidates` /
judge / critic with the new contract. Writes `plan.json` with the portfolio
payload. Live check: the cdoc turn-1 brainstorm produces `directions`
directly, no `proposed_experiments`.

**Phase 3 — consumers (2 d).** White paper, `ideation_report`,
`display_plan_summary`, `run_task` summary, deliverables. The HTML generator
gets a portfolio template (this is what finally makes an ideation report
openable in a browser — today it is suppressed because the only template
lies). Live check: the four ideation delegations of the cdoc replay.

**Phase 4 — portfolio refinement (1–2 d).** `refine_portfolio(request)` with
the verbs above. Live check: the cdoc "harden class 2" and "consolidate into
one cross-cutting class" turns, which are exactly these verbs.

**Phase 5 — routing + deprecation (0.5 d).** Point routing at the new tool;
`selection_profile="ideation"` warns and forwards. Live check: full 9-turn
cdoc replay, compared against the recorded session.

**Phase 6 — removal (later release).** Drop the deprecated parameter and the
`concepts`-inside-experiment fallback tier once no stored session needs it.

Estimate: **6–8 working days**, dominated by phase 3.

## Implementation status

Phases 1, 2, 3, 4 and 5 are implemented on
`fix-planning-kb-scope-and-tool-args`. Deviations from the plan above, and
why:

- **The transition shim** (see the amendment) replaced the "port 53 sites"
  reading of phase 3. Every phase boundary is shippable as a result.
- **Phase 3's consumers needed almost no porting.** The white paper, the
  dossier, the console and the candidate cards all read `concepts`, which
  the shim provides, so they rendered a portfolio correctly on the first
  live run. The real phase-3 work turned out to be the *portfolio HTML
  template*, which did not exist — that is why `plan.html` had been
  suppressed for ideation rather than fixed.
- **Phase 4 split.** 4a is `resync_portfolio`: through the transition a
  portfolio carries its directions twice, every re-emitting pass edits the
  shim copy, and the accessor resolves top-level first — so a refined
  portfolio would have served stale directions. Found by reading the refine
  prompt, not by a test. 4b is the refinement verbs.
- **A boundary discovered live, not planned:** with a document tool in the
  inventory, "give me the runnable bench protocol for direction X" was
  authored as prose — the mirror of the original bug, losing conformance,
  critic and refine-with-results. Both tool descriptions now state what they
  are NOT.
- **Phase 6 (removing the shim) is not done** and should wait until stored
  sessions no longer need it. `test_no_new_direct_readers_of_the_experiment
  _schema` ratchets the count so the work does not grow in the meantime.

## Risks

- **Lab-mode regression** is the main one: 53 shared sites. Mitigation is
  phase 1 landing as a provable no-op, and the failure-set-vs-baseline
  discipline used throughout this branch.
- **Two shapes in one state slot.** Any consumer that forgets the accessor
  silently mishandles a portfolio. Mitigation: a test that greps for
  `proposed_experiments` outside the allowed list.
- **Checkpoint restore across versions.** A session checkpointed pre-change
  must still restore; that is what fallback tier 2/3 of the accessor is for,
  and it needs a test fixture built from a real recorded session.
- **UI stdout contracts.** The candidate-card grammar (`── Candidate N: … ──`)
  is parsed by the Streamlit UI; the portfolio path must keep emitting it or
  update both sides together.
- **Scope creep into the meta.** Out of scope: `campaign_state` stays
  opaque, `run_task`'s contract does not change.

## Explicitly not in this plan

- No fourth mode, no new agent class.
- No change to plan-mode skills (knowledge-only stays).
- No change to the analysis or simulation orchestrators.
- `write_technical_document` stays as shipped — documents are a third thing,
  already separated.
