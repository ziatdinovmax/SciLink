# Design note: hyperspectral preprocessing & denoising — into the decomposition tool and codegen

**Status:** proposal (not implemented). Companion to the curve-fitting work
(`docs/preprocessing_in_fit_loop.md` / PR #215), but **deliberately a different
shape** — see "How this differs from curve fitting."

**Scope:** `HyperspectralAnalysisAgent` (EELS / datacubes). Retire the standalone
preprocessing stage by **folding it into the decomposition tool**, and give the
per-pixel codegen ownership of a separate, fittability-oriented denoising.
**Decomposition itself is NOT relocated** — it stays the iterative,
model-bearing loop it already is.

---

## TL;DR

The hyperspectral iteration pipeline has a standalone preprocessing stage
(despike/clip/mask menu) feeding a 6-substage decomposition block (NMF/PCA/ICA
with an LLM elbow/component selection), feeding a codegen stage that already
generates per-pixel fitting code. This plan:

1. **Retires the standalone preprocessing stage** and **folds it into the
   decomposition tool** — preprocessing exists to make the cube *decomposable*,
   so it belongs with decomposition, as **explicit deterministic primitives the
   (already model-bearing) decomposition step selects** (despike / clip-iff-
   non-negative / mask-threshold / normalize). Not a separate pipeline stage; not
   free-form codegen.
2. **Leaves decomposition as the iterative loop** (it has a real
   decompose → re-plan feedback edge; flattening it would lose the LLM-in-the-loop
   component selection).
3. **Gives the codegen the RAW cube + decomposition outputs + SNR/prep-note** (not
   the decomposition-cleaned cube), and makes it **own a second, free-form
   denoising whose goal is simply: make the per-pixel spectra fittable.**

Two denoisings, two goals, two mechanisms — kept separate on purpose.

## Current architecture (evidence)

Iteration pipeline (`pipelines/hyperspectral_pipelines.py`):

| # | Stage | Today |
|---|---|---|
| 1 | `RunPreprocessingController` → `HyperspectralPreprocessingAgent._apply_preprocessing` | **standalone preprocessing**: despike (median filter) / clip-negatives (iff `signal_is_nonnegative`) / mask-threshold menu, selected by an LLM strategy call. Emits `state["data_quality"]` (SNR + reasoning) and `state["preprocessing_mask"]`. |
| 2a–2f | component-param estimate → test-loop → elbow → LLM picks n → final unmix → validation | **decomposition** (NMF/PCA/ICA via `SpectralUnmixer`), with an **LLM-in-the-loop component selection** (elbow plot → pick n). |
| 3a–3d | prompt / interpret / **refinement decision** / human feedback | the refinement decision (what to extract) **depends on the decomposition output** → this is the decompose ↔ re-plan feedback edge. |
| 3e | `RunDynamicAnalysisController` | **codegen**: writes per-pixel `analyze_feature()`, injects skill `implementation`, `exec` sandbox, per-map visual QC + retry. Receives data from `get_optimal_analysis_data`. |

Three facts that shape the plan:
- **Decomposition is the core analytical step and is interleaved with planning**
  (3a–3c react to 2a–2f). It is *not* feed-forward → it stays a loop.
- **A codegen + visual-QC machine already exists** (3e). It already injects the
  skill `implementation` section and sandboxes execution.
- **PCA denoising was already removed.** `get_optimal_analysis_data`
  (`eels.py:1243`) used to PCA-denoise the cube before codegen; its docstring
  states it was removed because it "could silently remove real spectral features
  … the custom code was specifically asked to model." It now returns the **raw
  cube + an SNR note** and defers noise handling to the codegen. This plan
  continues that direction.

## How this differs from curve fitting (#215)

Curve fitting was feed-forward → preprocessing folded into the single fit script.
Hyperspectral is **not**, for two reasons, so the shape is different:
- **Decomposition is interleaved with planning** → it stays a loop step, not a
  leaf the planner calls once.
- **Preprocessing here serves decomposition** (a bounded, mechanical goal with a
  structured mask output) → it belongs *inside the decomposition tool* as
  **explicit primitives**, not as free-form codegen the way curve preprocessing
  became. (Free-form is reserved for the open-ended *post*-decomposition
  fittability denoising.)

## Target architecture

```
planner (rough: method hint, skill-guided)
  → DECOMPOSITION TOOL   (model-bearing; owns its own prep)
        raw cube
          → internal prep: explicit deterministic primitives the tool's LLM
            SELECTS (despike / clip-iff-nonneg / mask-threshold / normalize),
            informed by SNR + technique + skill; deterministically APPLIED
          → NMF/PCA/ICA + elbow + LLM component selection + validation
        emits: components, abundance_maps, reconstruction, validation,
               AND data-quality metadata (SNR, mask, short prep-note)
  → re-plan on decomposition output            (the feedback edge — preserved)
  → CODEGEN  (per-pixel fitting)
        receives: RAW cube  +  decomposition outputs (components / abundances /
                  reconstruction)  +  SNR  +  prep-note
        owns: free-form, per-task denoising whose goal is FITTABLE spectra
              (may fit raw, the reconstruction, or its own denoise — its call)
```

### Decision 1 — preprocessing folds into the decomposition tool (explicit primitives)
- Retire `RunPreprocessingController` as a stage and `_apply_preprocessing` as a
  standalone menu. The **decomposition tool** does the prep internally.
- Prep is **explicit deterministic primitives, LLM-selected** — NOT free-form
  codegen — because the op set is bounded and mechanical, masking yields a
  structured mask consumed downstream, and NMF mechanics make some ops
  semantically required:
  - **clip-negatives:** deterministic, applies **iff**
    `axis_spec.signal_is_nonnegative` (today's rule).
  - **despike / mask-threshold / normalize:** the *selection + parameters* are
    chosen by the decomposition step's LLM (from SNR, technique, skill — e.g.
    "don't mask the vacuum region", "align the zero-loss peak"); deterministically
    applied.
- This prep is **internal to the tool** (it serves good components) and must
  **not** overwrite the cube handed to codegen.

### Decision 2 — decomposition stays the iterative loop
- The component selection (test-loop → elbow → LLM picks n) is preserved; the
  refinement decision keeps consuming decomposition output. Do **not** linearize
  to planner→decompose→codegen.

### Decision 3 — codegen receives RAW, not the decomposition-cleaned cube
- Base data to codegen = the **raw** cube (after only the deterministic loader:
  format/units/NaN). **Not** the decomposition-prep-cleaned cube — that cleaning
  (masking, clipping, despike thresholds) was tuned **for decomposition** and can
  be wrong for per-pixel fitting (the same "preprocessing biases the observable"
  problem, moved one stage over).
- Codegen **also** receives the decomposition **outputs as optional inputs**
  (components, abundance maps, the rank-k **reconstruction**) + the **SNR** + a
  **prep-note** of what decomposition-prep was applied — so it can choose to fit
  the reconstruction (denoised) or replicate universally-good ops (e.g. cosmic-ray
  despike) without inheriting the decomposition-specific ones.

### Decision 4 — codegen owns post-decomposition denoising (free-form, fittability)
- Goal is narrow and clear: **make the spectra fittable.** The generated code
  decides per task (fit raw / fit reconstruction / its own gentle denoise),
  guided by skill + the "don't erase the feature you're fitting" guardrail.
- **Never a silent global step** — the removed-PCA-denoise is the precedent. The
  denoised reconstruction is *offered as an option*, not forced.

## The info-flow contract (must preserve)

Today the standalone preprocessing emits metadata that flows downstream:
`SNR → component-selection planning` (`hyperspectral_controllers.py:378-379`) and
`processing-note + mask → codegen` (`get_optimal_analysis_data` / build prompt).
When prep folds into the decomposition tool, **the tool must still emit**:
- `components`, `abundance_maps`, `reconstruction`, `validation`;
- `data_quality` = **SNR recomputed deterministically** after prep (not LLM-
  asserted) + reasoning;
- the **mask**;
- a short **prep-note** (operations applied).

These feed the re-planning (component/extraction decisions) and the codegen prompt
exactly as today — so nothing downstream loses information. (The prep-note is
richer than today's terse strategy, so planning/codegen become *better* informed.)

## Migration (incremental)

1. **Expose decomposition (incl. its prep) as a single model-bearing tool/step**
   that encapsulates today's 2a–2f *and* the prep primitives, emitting the
   metadata contract above. The prep primitives are explicit functions
   (despike/clip/mask/normalize) the tool selects.
2. **Stop running the standalone `RunPreprocessingController`**; the decomposition
   tool owns prep. Keep a thin deterministic loader (format/units/NaN).
3. **Feed codegen RAW + decomposition outputs + SNR + prep-note** (extend
   `get_optimal_analysis_data`'s contract — it already returns raw + SNR; add the
   reconstruction/components/prep-note as available inputs).
4. **Codegen owns fittability denoising** — prompt it that any denoising is its
   choice for fittability, offered the reconstruction, never forced; reuse the
   existing visual-QC judge (raw-vs-processed view for whatever it applies).

Steps 1–2 are the load-bearing change; 3–4 wire the codegen contract.

## Tradeoffs and risks

- **Preserve the elbow / component-count LLM judgment** inside the decomposition
  tool — it's a real strength; don't degrade it to "model guesses n."
- **Bigger blast radius than #215** (a whole stage retired into a tool; the
  iteration loop touched). Stage it; keep `final_components` /
  `final_abundance_maps` and the `data_quality`/`mask` state contracts intact.
- **Don't reintroduce silent denoising** (PCA-reconstruction or aggressive
  smoothing) anywhere global — only as an explicit, codegen-chosen option.
- **Decomposition-prep ≠ fitting cube**: enforce that the tool does not overwrite
  the raw cube the codegen receives.

## Non-goals

- Not relocating decomposition out of the loop, and not making it a pure-numeric
  leaf (it keeps its LLM component selection).
- Not making decomposition prep free-form codegen — it's explicit primitives.
- Not forcing reconstruction-denoising on the codegen — it's an option.
- Not changing the synthesis pipeline beyond the inputs it receives.
