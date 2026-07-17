# In-situ synchrotron XRD machinery — plan

*Follow-up to PR #328 (fingerprint/indexing/Le Bail/Rietveld identification
stack). Goal: the curve-fitting agent handles in-situ synchrotron series —
thousands of frames, arbitrary wavelength, evolving phase mixtures — for
identification, tracking, and quantification.*

## What already transfers (do not rebuild)

- **Wavelength-free library** — fingerprints stored as d-spacings; synchrotron
  λ (0.1–0.5 Å) handled by `search_match_pattern(wavelength=...)`.
- **Throughput** — 0.12 s/query deterministic ⇒ frame-wise ID of a 10k-frame
  series is a ~20 min batch, no LLM per frame.
- **Series machinery in `xrd_profile`** — per-frame fitting, freeze-the-
  approach-not-the-values, transition detection (built for the ibuprofen
  dehydration series). The identification (`xrd`) side is frame-wise; the two
  skills are complementary, not overlapping.
- **`score_xrd_match_multiphase`** — joint MILP over candidate phases with
  per-phase activation; exists, exercised for confirmation scoring.
- **Frame-wise lattice tracking** — `validate_cell_lebail` / `refine_rietveld`
  per frame gives cell(T)/cell(t) already.

## Piece 1 — Mixture identification by sequential subtraction (tool)

The classical loop for multi-phase patterns; in-situ series are *by
definition* two-phase during any transformation.

- New tool `identify_mixture` (`xrd` skill, pure Python over existing tools):
  1. `search_match_pattern` on the full peak list → accept best phase.
  2. Remove that phase's **accounted lines** (reuse the Le Bail/absent-lines
     accounting geometry: measured peaks within `tol_deg` of the accepted
     phase's predicted lines).
  3. Re-search the residual peak list; repeat until no convincing match
     (`fom < threshold`) or `max_phases` reached.
  4. Confirm the ensemble with `score_xrd_match_multiphase` (joint MILP —
     catches over-subtraction where one line belongs to two phases).
- Knobs: `max_phases` (default 3), `min_residual_peaks` (stop when too few
  lines remain), `tol_deg`, per-iteration `fom_threshold`. Shared-line
  handling: a peak within tol of BOTH the accepted phase and a residual
  candidate stays in the residual list with reduced weight, not removed
  (document; over-removal is the classical failure of naive subtraction).
- Validation: synthetic two-phase mixes of library entries (deterministic,
  offline: combine two RRUFF patterns at known fractions); the RRUFF set has
  no true mixtures, so synthetic-mix + the ibuprofen mid-transition frames
  (real two-phase coexistence) are the test beds.

## Piece 2 — Temporal workflow (skill prose + thin series driver)

Identification of a series ≠ N independent identifications.

- Prose in `xrd.md` (and cross-reference in `xrd_profile.md`):
  *identify at the series ENDPOINTS first* (start/end frames — usually pure
  phases), then track: each intermediate frame is a mixture of already-
  identified endmembers until residual evidence says otherwise (a transient
  intermediate phase = unaccounted residual peaks appearing mid-series →
  run `identify_mixture` on that frame only).
- Thin deterministic driver `track_phase_series` (tool): given frames +
  endpoint phase IDs, per frame compute each phase's accounted-line intensity
  share (cheap proxy for fraction) → phase-evolution curve + detected onset/
  completion frames. This is the *screening* quantification; Rietveld
  (piece 3) is the rigorous one. Bridges to `xrd_profile`'s transition
  machinery rather than duplicating it: output frame indices feed its series
  fitting.
- No LLM in the per-frame loop; the agent plans, the driver executes.

## Piece 3 — Multi-phase sequential Rietveld (GSAS wrapper extension)

Quantitative phase fractions vs time/T — the standard in-situ deliverable.

- Extend `_gsas_engine.rietveld_refine` → `rietveld_refine_multiphase`:
  multiple phases (CIFs from the identification step's `structure_path`s) in
  one histogram; refine per-phase scale factors + shared background/zero +
  per-phase cell; report **weight fractions** (GSAS computes from scales,
  ZMV formula), per-phase lattice, fit metrics. Same staged protocol;
  `converged` semantics carried over.
- Sequential mode `rietveld_refine_series`: frame k initialized from frame
  k−1's refined parameters (the standard sequential-refinement trick — cells
  evolve smoothly with T); phases whose fraction → 0 get their scale frozen
  (not removed) to avoid instability. GSAS-II has native sequential
  refinement support (`Controls['Seq Data']`) — evaluate wrapping it vs
  looping our single-histogram path with warm starts; prefer whichever is
  simpler to make robust (loop-with-warm-start is the likely v1: same code
  path as today, trivially resumable).
- Knobs: `refine_cell_per_frame` (bool), `freeze_profile_after_first`
  (broadening from frame 1 reused — standard), `fraction_floor`.
- Validation: ibuprofen dehydration series (38 frames, known transition) —
  fractions must go 1→0 / 0→1 monotonically through the transition window
  found in the earlier xrd_profile work; plus a synthetic 2-phase series with
  known ground-truth fractions.

## Sequencing & scope

1. Piece 1 (tool + synthetic-mix tests) — smallest, unblocks 2.
2. Piece 2 (prose + `track_phase_series` + ibuprofen validation).
3. Piece 3 (multi-phase Rietveld, then sequential mode) — largest; its own
   validation gate before any PR.
- One PR for pieces 1+2 (identification-side), a second for piece 3
  (quantification-side), unless 3 lands quickly.
- Explicitly deferred: 2D-detector integration (pyFAI upstream), parametric
  Rietveld (constraints across frames), texture/PO corrections beyond
  documentation.

## Standing constraints (from PR #328 lessons)

- Knobs exposed + explained; adaptive logic in tools, judgment in agent prose.
- No benchmark-fitted defaults; validate on data NOT used to set parameters.
- Live-test through the real CurveFittingAgent before declaring done (the
  full-dress 20-mineral run pattern; ibuprofen series is the in-situ analog).
- Deterministic offline eval first (synthetic mixes/series), LLM runs second.
