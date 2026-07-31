# Experiment-Validated Prediction — design + UC2 plan

Status: proposal / design notes. **Revises** the UC2 stance in
`reference-property-validation-design.md` (see *Reconciliation* below). Motivated
by the UC2 aqueous Zn-electrolyte study; the capability is system-agnostic.

## What changed, and why

The earlier design asked UC2 to catch the force-field error **without** the
measured data — from pure-component reference checks and physics reasoning
alone — and reveal the experimental spreadsheet only at the end as confirmation.
Direct measurement this session shows that path is not reliable:

| Catch mechanism | Reliability (Opus, N-shot) |
|---|---|
| Pure-component density (EIS 1.03 sim vs ~1.13 ref, ~9% low) | 2/5 — borderline error, careful critic won't always flag |
| Physics reasoning from pure-component anchors ("sulfone is dense → mixture should densify") | 0/5 — the anchor alone didn't move the verdict |
| Autonomous literature trend (Edison) | **wrong** — returned "density falls with EIS"; measured data rises |
| Compare simulated trend to **measured** trend | 5/5 flag / 0/5 false-positive |

A ~9% force-field error is genuinely borderline, so a careful validator is
*right* not to always flag it from a single anchor — that is honest model
behavior, not a bug we can prompt away. And autonomous literature grounding is
not just slow (the trend query completes in ~7 min); when it completes it can be
**confidently wrong** (Edison inverted the trend and cited a tangential MD
thesis), which would flip the catch. The only mechanism that held up was
comparison to measured data.

## Reconciliation with `reference-property-validation-design.md`

That doc rules out comparing to the measured density curve because "that is
validation against the answer, not prediction." The flaw is a conflation:

> **Density and viscosity are *validation observables*; T1 is the *prediction
> target*.** Dave's deliverable is the ¹H **T1 relaxation** (and the
> reorientational dynamics / solvation behind it). Using measured density and
> viscosity to establish model trust, then predicting T1, is textbook
> validate-then-predict — the model is never validated on T1 and then asked to
> "predict" T1.

The old doc's healthy instinct — a predictive demo shouldn't consume its own
answer — is preserved *and correctly placed*: the prediction target (T1) is
never used to validate; its measured value is held back and revealed at the end
as confirmation. Density and viscosity, which *are* consumed, are different
observables — which is what validation is *for*.

Everything else in the earlier doc survives: the self-refinement slot, the
reparameterization fixer, the extensibility bar (new check = skill content, new
fixer = registered), the anti-overfitting discipline, and the ~23-case
generalization benchmark (it tests exactly this panel, system-agnostically).

## The underlying contract: parameters = f(system, observables)

The panel sits on a more foundational principle that generalizes across all
three simulation modalities: **every simulation agent's input parameters are a
function of both the system *and* the target observables.** The observables are
not an implicit substring of a free-text goal — they are a first-class,
engine-neutral co-input that shapes the whole setup, on equal footing with the
system description. Band gap → denser k-points / hybrid functional; viscosity →
per-step stress + long correlation window; T1 → sub-ps dumps; a plain relaxation
→ none of that. Some parameters are system-driven (timestep, thermostat), some
observable-driven (what/how-densely to log, convergence targets), many both — but
the *set* is jointly determined.

Today (verified) observables travel only inside the `research_goal`/`request`
string; the sole structured consumer is the pre-run coverage gate, which
*re-infers* the set in one shot at submission and only checks *presence*. The
typed scaffolding to fix this already exists but is mostly unwired:
`contradictions.Requirement(observable, check_kind, params)` with a `CHECK_KINDS`
vocabulary (`signal_present`, `cadence`, `selection_realizable`, …), of which only
`selection_realizable` has a live checker.

The delta (this is the general library work, engine-neutral, *not* UC2-specific):

1. **Thread `required_observables: List[Requirement]` as a first-class co-input**
   — produced upstream (planning), carried through `run_complete_workflow →
   _generate_inputs →` each agent's Generate, so run length / cadence / logged
   quantities / convergence are chosen *from* the declared observables.
2. **Gate checks the list instead of inferring it** — `RunCritic.assess` gains a
   `required_observables` param and a check-against-list mode (turns the flaky
   ~0.67 viscosity inference into a deterministic presence check).
3. **Implement the empty `signal_present` + `cadence` checkers** — presence
   (tier 1) *and* sampling adequacy (tier 2).

The panel below and this contract are coupled through the *same* observable set:
declaring "validate viscosity" both obligates the run to sample stress (contract)
and tells the panel what to check against the measured curve (validation). This
is CLAUDE.md-level — it promotes `observable-requirements-contract.md` from "an MD
gate" to the Generate contract for all three modalities.

## The capability: a multi-observable validation panel

Generalize the single-property gate into a **panel**: N observables, each with a
reference, each judged, aggregated into a confidence statement that **scopes the
prediction**.

```
for each observable in panel:
    value   = compute_from_MD(observable, run(s))     # density | viscosity | T1 | ...
    ref     = resolve_reference(observable)            # measured > caveated-lit > physics
    verdict = judge(value, ref)                        # pass / fail + magnitude + culprit
aggregate -> confidence envelope, scoped to the prediction
if any fail -> catch -> recommend fix -> human approves -> re-validate   (loop)
else        -> predict(target) with confidence tied to which observables passed
```

**Reference resolution priority (this session's lesson):**
1. **Measured in-house data** — the trustworthy anchor (Dave's density/viscosity
   spreadsheet + T1 JSONs for the exact system and compositions).
2. **Caveated literature single-value** — Edison retrieves single values
   reasonably (pure-EIS ~1.09–1.13) but they *drift run-to-run*; treat as a soft
   anchor, never a hard threshold, and never a bare trend direction.
3. **Physics / first-principles reasoning** — the general engine when no
   reference exists; but weak on borderline errors, so not a sole gate.

**References are cross-checked, not consumed blindly.** The literature-vs-data
conflict this session was resolved by checking each dataset against physics it
does *not* assert: Dave's data has the correct temperature dependence of density
*and* viscosity, correct viscosity-vs-composition, and sensible absolute
magnitude — so it is credible on the disputed axis; Edison offered a bare
direction with no endpoint numbers. The panel should embed this: a reference
earns trust by internal/physical consistency before it is used.

**Why multiple validation observables (the overfitting defense).** Matching
*one* observable (density) is weakly constraining — tune-to-fit is a real risk.
Validating on density **and** viscosity — independent thermo *and* transport
observables — is much harder to overfit and genuinely constrains the model that
then predicts T1. Viscosity is the key one here: it is a dynamic/transport
property driven by the same molecular motions that set the reorientational
correlation times behind T1, so a model that reproduces measured viscosity has
earned real confidence in the T1 prediction. The panel is what makes the
prediction credible, not any single match.

## What we explicitly do NOT claim

- **Not** "SciLink autonomously grounds the fix in literature." Edison-based
  literature grounding stays a *caveated fallback* for systems with no in-house
  data — a hint that must be verified against physics, never the sole reference
  for a trend (it returned the wrong trend direction for UC2). It is future work,
  not part of this capability's core.
- **Not** full autonomy on the fix. Per the use-case framing (max quality *with
  human feedback*), SciLink catches and recommends; the human approves the
  corrected model; SciLink re-validates. That is the version that works.

## UC2 arc (revised)

1. Build the composition series (1.0 M Zn(OTf)₂, water:EIS 80:20 → 50:50),
   parameterize with the default force field.
2. Run each composition (equilibration + production NPT) on Deception.
3. **Validate** simulated density and viscosity against Dave's measured curves
   (per-composition and trend-level). T1 is *not* used here — it is the target.
4. **Catch** where the force field disagrees (reliable: comparison to measured
   data). Expected: the default sulfone model is under-dense / off on transport.
5. **Fix** — SciLink recommends a reparameterization; human approves; rerun.
6. **Re-validate** — the corrected model reproduces measured density + viscosity.
7. **Predict** Dave's deliverable: ¹H T1 relaxation vs composition (from the
   reorientational/translational dynamics of the validated model), with the
   Zn²⁺ solvation structure (RDF, coordination number) as the mechanism behind
   it. Confidence is scoped to the validation the model passed.
8. **Confirm** — only now compare the predicted T1 to Dave's measured T1 curve.

The "wow" is steps 4–5 gating steps 7–8: without the panel you would report a T1
prediction from a model that fails the measured observables; SciLink catches
that and forces a validated model before predicting.

## Grounding: what the existing runs already show

From the completed S1–S5 runs vs the measured spreadsheet:

- **Density is the weak catch.** With a *consistent* series the computed density
  trend is the right direction but ~4–5% low — a borderline magnitude offset
  (attributed to OpenFF Sage giving the sulfone too large a molar volume). A
  careful validator won't reliably flag ~5%, which is honest, not a bug.
- **Viscosity is the strong catch.** Computed viscosity came out unreliable and
  even *backwards* (S4 < S3 where measured rises) — a blatant transport failure,
  the usual signature of the TIP3P water model. This is the unambiguous,
  convincing catch.
- **Two model deficiencies, not one.** Density points at the sulfone
  parameters; viscosity/dynamics points at the water model. Validating multiple
  observables *reveals multiple deficiencies*, each with its own fix — a richer,
  more honest story than a single knob.
- **T1 needs machinery that doesn't exist yet:** τc extraction + sub-ps
  trajectory dumping (a tier-2 sampling requirement). That machinery is *shared
  with UC1*, so building it does double duty. UC2's ¹H T1 is the **water
  proton**, so it probes water dynamics in the solvation shell — mechanistically
  tied to the RDF first shell we predict.

## Deception run plan (staged)

- **Tier 1 — density validation + solvation (tractable, proves the mechanism).**
  Density from production NPT is cheap and reliable. Run the 4 compositions with
  the default FF, compare to measured density, demonstrate the catch, do one
  guided reparameterization, rerun, show the corrected trend. Then extract the
  Zn²⁺ solvation structure (RDFs, coordination numbers) from the validated
  model. A complete catch→fix→re-validate→mechanism story on the cheap observable.
- **Tier 2 — viscosity validation + T1 prediction (the full deliverable).** Add
  viscosity (Green-Kubo stress-ACF or NEMD; long, noisy — real work) to the
  validation panel, then predict ¹H T1 from reorientational/translational
  correlation functions and confirm against Dave's measured T1. Viscosity and T1
  are where a fixed-charge FF may genuinely fall short — if it cannot match
  viscosity, the T1 prediction is correctly scoped down rather than overstated.
- **Prediction — T1 (deliverable) + Zn²⁺ solvation (mechanism).** T1 vs
  composition confirmed against measurement; Zn–O(water), Zn–O(sulfone),
  Zn–O(triflate) RDFs and coordination numbers as the structural explanation.

Honest hard parts: viscosity and T1 from classical MD are the effort sink, and a
non-polarizable FF may not capture them — that boundary is a result, not a
failure, and the panel reports it rather than hiding it.

## Resolved decisions (2026-07-30, Sarah)

- **Maxim gets looped in** on the observable-vs-target reframe before it lands in
  CLAUDE.md (summary drafted).
- **Viscosity is in the validation panel** — invest in the strong, unambiguous
  catch rather than rest on the ~5% density flag. Density + viscosity both
  validate.
- **T1 is in scope for the paper result (Option A)** — build the τc / sub-ps
  sampling machinery (shared with UC1), predict T1, confirm against Dave's
  measured curve. The full deliverable, not a follow-on.

Still open: whether the corrected force field comes from a named validated
parameter set (human-supplied) or a SciLink-recommended one the human approves —
likely *two* fixes (sulfone params for density, water model for
viscosity/dynamics).

## Build sequence

1. **Viscosity observable** — Green-Kubo stress-ACF (or NEMD) computation from a
   production run, as a panel observable with the measured curve as reference.
2. **τc + T1 machinery** — sub-ps trajectory sampling + reorientational
   correlation-time extraction → ¹H T1 estimate. Shared with UC1.
3. **Multi-observable panel** — generalize the single-property gate
   (`reference_validation`) to N observables with per-observable verdicts and an
   aggregate confidence that scopes the prediction.
4. **Deception run** — default FF → validate density + viscosity → catch → guided
   fix(es) → re-validate → predict T1 + solvation → confirm T1 vs measured.
