# Design note: move preprocessing into the verified fit loop

**Status:** proposal (not implemented). Discussion captured from a live
debugging session, 2026-05-28. No code changed.

**Scope:** the `CurveFittingAgent` pipeline. The argument generalizes to any
codegen-capable foundation agent that has a separate preprocessing step, but
this note is written against curve fitting concretely.

---

## TL;DR

Preprocessing currently runs **once, upstream of the fit loop**, its output is
the only thing the verifier ever sees, and nothing downstream can detect or redo
a bad preprocessing choice. Because preprocessing transforms (smoothing,
baseline, cropping, clipping) **bias the very observables being fit**, a wrong
choice silently corrupts the result while still scoring a clean R². The fix is
to fold preprocessing into the `generate → execute → verify → anneal` loop and
into planning, and — critically — to give the verifier the **raw** data so it
can actually see preprocessing damage.

---

## Current architecture (with evidence)

Pipeline factory: `scilink/agents/exp_agents/pipelines/curve_fitting_pipelines.py:136-261`.

```
1.  AnalyzeDataController            stats + initial plot
1.5 SeriesScoutController            infer the series variable
2.  HumanFeedbackRefinementController  LLM plans the MODEL; locks config for the series
3.  LiteratureSearchController       optional
4.  UnifiedSeriesProcessingController  fits all spectra
        └─ preprocessor.run_preprocessing(...)   ← preprocessing happens HERE, once per spectrum
        └─ _fit_single_spectrum(...)             ← contains the verify/anneal loop
5.  AdaptiveRefitController          re-fit flagged spectra
6.  ConditionalTrendAnalysisController
7.  UnifiedCurveSynthesisController
8.  StoreAnalysisResultsController
9.  {Generate,Unified}CurveReportController
```

Three facts that define the problem:

1. **Preprocessing is one-shot, upstream of the loop.**
   `run_preprocessing` is only ever called at
   `curve_fitting_controllers.py:3687`, `:3697`, `:4258` — all *outside*
   `_fit_single_spectrum`. The verify/anneal loop is the `for attempt` loop
   *inside* `_fit_single_spectrum` (`:2015`), which regenerates the fit script
   and anneals model constraints but **never re-invokes the preprocessor**.
   Flow: `preprocess once → [generate script → execute → verify → anneal]×N`.
   The bracket iterates; preprocessing does not.

2. **Planner and preprocessor are independent LLM calls.**
   Preprocessing strategy is chosen by `_llm_select_preprocessing_strategy`
   (`preprocess.py:183`) and locked for series consistency. The planner
   (`HumanFeedbackRefinementController`, step 2) plans the *model*. They reason
   separately; the planner does not own or see the preprocessing transform.

3. **The verifier cannot see preprocessing damage.**
   `_verify_fit_with_llm` (`:2281`) compares the fit to the *already-preprocessed*
   `curve_data` and its plot. It never sees the raw signal, so a fit to
   corrupted data scores a clean R² and is approved.

## Failure mode (not hypothetical)

Preprocessing transforms are **not neutral** — they bias the observables:

- **EPR (observed):** preprocessing applied Savitzky-Golay smoothing
  (window=5). For a sharp derivative feature, smoothing **broadens the line**,
  biasing `lw_G` — the exact quantity being measured. The verifier sees a clean
  fit to the smoothed data and approves; `lw_G` is silently wrong.
- **TRPL:** over-smoothing or a wrong baseline/crop distorts a sub-µs decay
  component before the fit ever runs.
- **Crop dead-end (observed):** a `custom_processing_instruction` to "fit only
  the decay, exclude dark counts" cannot satisfy the 1D preprocessing contract
  that output length == input length (`preprocess.py:846-850`), so it fails and
  silently falls back to original data. Region-of-interest is a *fit-domain*
  concern misfiled as a preprocessing transform.

Common thread: a preprocessing choice corrupts or constrains the result, it is
made once, and **no downstream stage can detect or redo it.**

## Why this violates the agent's own contract

Foundation-agent element 5 (CLAUDE.md): *"Generated artifacts run in a sandbox,
and the agent verifies the result before accepting it."* Preprocessing is a
generated/parameterized transform that **escapes** verify-before-accept. It is
the one stage that mutates the data and is never checked against an independent
reference.

## Target architecture

1. **The fit script owns the full chain, as free-form generated code.** Generate
   one artifact that does `load raw → preprocess → fit → output`, where the
   *preprocess* part is **code the model writes itself** — whatever is
   appropriate for the data — not a call to a fixed menu of operations or any
   prescribed preprocessing tool/helper. The verifier then sees the *entire*
   transform and can reject preprocessing damage, not just model misfit.

   This is a deliberate move away from the current prescriptive approach: today
   `_llm_select_1d_strategy` returns a fixed schema
   (`{apply_clip, apply_smoothing, smoothing_window}`) and `_apply_1d_strategy`
   applies exactly those two operations. That menu is **removed**. Preprocessing
   becomes generated code on the same footing as the fit code the agent already
   writes — consistent with CLAUDE.md (skill code blocks are LLM-facing
   reference, not executable surfaces; runnable specialization is generated, not
   prescribed).

   A direct consequence: the **length-preserving vs length-changing distinction
   dissolves.** It only existed because preprocessing was a separate step handing
   a same-length array to a downstream fitter (the `output length == input
   length` contract). Once preprocessing is code inside the fit script, a crop, a
   bin, a smooth, and a baseline are all just lines the model writes; there is no
   hand-off and no length contract. (Steps 1 & 3 — rerouting crop/ROI to the
   planner — are the *interim* fix while preprocessing is still a separate step;
   once it is in-script, ROI is simply part of the generated preprocessing/fit
   domain.)

2. **Planning decides the preprocessing *approach* jointly with the model**, and
   the skill supplies the *knowledge* (not tools): e.g. "for EPR derivative
   spectra do not smooth — it broadens the line; fit the intrinsic linewidth";
   "TRPL: estimate t₀ at the peak, fit from there, background as a parameter."
   The model reads that guidance and writes appropriate preprocessing code. One
   locked plan covers preprocessing + model so they are coherent.

3. **The anneal loop can revise the preprocessing code** the same way it revises
   the model. If the verifier flags distortion, regenerate with a different
   preprocessing choice (gentler/none). "Preprocessing aggressiveness" becomes an
   annealing knob alongside the model constraint schedule.

## The one detail that makes or breaks it

**The verifier must receive the raw data (and a raw-vs-processed overlay).**
If preprocessing moves into the fit script but the verifier still only sees the
*processed* curve, nothing is gained — it still cannot see the damage.
Raw-vs-processed visibility is the actual safety mechanism; relocating
preprocessing merely *enables* it. This is the required addition to
`_verify_fit_with_llm`'s inputs.

## Consequence: preprocessing becomes skill-specializable

Today preprocessing is **skill-blind**: `_llm_select_preprocessing_strategy`
(`preprocess.py:183`) receives only `stats` + `system_info` (metadata), never
skill context. So domain knowledge that is *fundamentally about preprocessing*
has nowhere to live in a skill — e.g.:

- EPR: "do **not** smooth derivative spectra — it broadens the line and biases
  `lw_G`."
- TRPL: "identify t₀ at the intensity peak; fit from there; background is a
  fit parameter, not a subtraction."
- XPS: "Shirley background, not linear; do not clip."

Because the smoothing/baseline/clip decision happens in a skill-agnostic step
*before* the skill's context is consulted for fitting, the EPR skill cannot stop
the smoothing that corrupts its own observable.

Once preprocessing is part of planning + the generated fit script, it inherits
the skill mechanism for free. The skill's existing fixed sections carry it:

- `planning` → how to approach preprocessing for this technique;
- `implementation` → guidance the model uses to *write its own* preprocessing
  code in the generated fit script (not a prescribed recipe or tool to call);
- `validation` → "check the raw-vs-processed overlay did not distort the line."

**Recommendation:** reuse the existing sections (`planning` + `implementation`)
rather than adding a new `preprocessing` section. The six-section vocabulary is
fixed and load-bearing; do not expand it speculatively. Only introduce a
dedicated section if evidence shows preprocessing guidance is consistently
mis-housed in `planning`/`implementation`. This keeps the change aligned with
the "skills, not new agents / new vocabulary" thesis.

## Disposition of `custom_processing_instruction`

The metadata field does **not** disappear — it gets a better consumer.

Today it routes into `_generate_and_execute_custom_script_1d` (a standalone
preprocessing script bound by the length-preserving contract), which is where
the crop dead-end happens. Under the new design the planner (step 2) consumes
it directly and reconciles it into the locked plan, with two dispositions:

1. **Fit-domain / model intent** (most cases): "fit only the decay, exclude
   dark counts" → a fit *window* (t ≥ t₀) and a background *parameter*, decided
   in planning and emitted in the fit script. No length-changing transform, so
   the contract violation disappears entirely.
2. **Genuine data-transform intent** (rare): "the first 50 points are an
   instrument artifact, drop them"; "subtract this reference spectrum" → a real
   preprocessing step, but now emitted **inside the verified fit script**, so
   the verifier sees raw-vs-processed and can reject it if it distorts.

So the field's *contract* changes from "an imperative for a length-preserving
standalone script" to "free-text intent the planner routes to fit-domain, model
parameter, or in-script transform as appropriate." This is migration step 1
(reject-and-reroute) generalized: the planner, not a standalone preprocessing
path, owns the field.

## Tradeoffs and constraints

- **Series consistency must be preserved.** Today preprocessing is locked once
  and applied uniformly across a series — a real feature for cross-sample
  comparability. The lock must now cover the combined preprocess+model recipe,
  not just the model. Do not let per-spectrum preprocessing drift.
- **Larger codegen surface to verify** — but the fit script already does
  codegen + verify, so this is incremental, not new machinery.
- **Keep a thin deterministic pre-step** for genuinely model-irrelevant,
  non-distorting hygiene only: format/unit parsing, NaN handling, ascending-x
  sorting. Anything that *can* bias an observable (smoothing, baseline, crop,
  clip, normalization that rescales) goes inside the verified loop.

## Migration (incremental, low-risk first)

1. **[DONE — PR #215] Reject-and-reroute, not fail:** the preprocessor refuses
   length-changing `custom_processing_instruction`s (`FitDomainInstruction`) and
   defers them to the planner. (Killed the live crop dead-end.)
2. **[DONE — PR #215] Give the verifier the raw data + raw-vs-processed overlay.**
   The safety mechanism; valuable even before full relocation.
3. **[DONE — PR #215] Surface ROI/background to the planner**
   (`_append_fit_domain_guidance`): domain as a fit window, background as a fit
   parameter — not preprocessing.
4. **[Step 4 — pending] Make preprocessing free-form generated code in the fit
   script.** Remove the prescriptive `_llm_select_1d_strategy` /
   `_apply_1d_strategy` menu. The model writes whatever preprocessing is
   appropriate (guided by the active skill's knowledge), inside the same
   generated artifact as the fit; lock it jointly with the model for series
   consistency; add a preprocessing-aggressiveness knob to the anneal schedule
   so the loop can revise it. The length-preserving/changing distinction goes
   away (no separate-step contract), so steps 1 & 3's reroute becomes a no-op
   that can be retired once this lands.

Steps 1–3 (PR #215) are independently shippable and remove the worst
silent-failure modes; Step 4 completes the relocation.

## Non-goals

- Not a wholesale "one LLM call for everything" merge; the planner and fitter
  stay distinct stages. Only the *responsibility boundaries*, *what the verifier
  sees*, and *who writes the preprocessing code* change.
- Not prescribing preprocessing operations or tools. The point of Step 4 is the
  opposite: the model writes its own preprocessing code from skill guidance. The
  `_llm_select_1d_strategy` / `_apply_1d_strategy` fixed menu is removed, not
  preserved. (A thin deterministic *loader*-level step may remain for genuinely
  model-irrelevant, non-distorting hygiene — format/unit parsing, NaN handling,
  ascending-x sorting — but that is data loading, not preprocessing choices.)
