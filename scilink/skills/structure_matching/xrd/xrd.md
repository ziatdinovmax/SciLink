---
description: 'XRD phase identification (search-match) — the default first-pass XRD analysis answering "what phase(s) is this?". Queries crystal-structure databases (COD, Materials Project, local CIF), simulates kinematic patterns, and scores by cross-correlation (fast) and Hanawalt / MIP peak-matching (robust). Use this for routine phase ID; the xrd_profile skill is the specialized follow-up for line-broadening (crystallite size / strain) once the phase is known.'
quality_gate:
  metric: figure_of_merit
  accept_threshold: 0.70
  hard_reject_threshold: 0.40
  direction: higher_is_better
  physical_review: false   # workflow-scoring gate: score_xrd_match_robust IS the verification
---
# XRD Structure Matching Skill

## overview

Identify a crystalline phase from an experimental X-ray diffraction (XRD)
pattern by matching against database structures. **This is the default,
highest-frequency XRD question — "what phase(s) is my sample?" — and the usual
first pass for any XRD pattern.** Profile fitting (the `xrd_profile` skill:
per-peak pseudo-Voigt → Scherrer crystallite size / Williamson-Hall strain) is
the *specialized follow-up* once the phase is known, not the starting point.

The skill ships five tools the analysis script chains together:

- `search_structures` — query the **COD** (Crystallography Open Database — the
  recommended default: experimental structures, organic + inorganic, no API key),
  **Materials Project** (computed inorganic; for stability ranking / predicted
  phases), and / or a local CIF directory for candidate structures (chemistry +
  symmetry filters). COD's experimental cells avoid the DFT lattice mismatch that
  MP structures carry.
- `simulate_xrd_pattern` — kinematic XRD pattern from a CIF via pymatgen
  (CuKa default; any wavelength supported).
- `score_xrd_match_fast` — **fast tier**. Cross-correlation of the
  broadened simulated pattern against the experiment, fitting zero-shift
  and lattice scale jointly. Tens of milliseconds per candidate. Use for
  on-the-fly ranking during an experiment or scout passes over many
  candidates.
- `extract_peaks` — extract a peak list from a continuous experimental
  pattern (positions, intensities, FWHMs). Needed before the robust
  tier; can also be called standalone to inspect / report peaks.
- `score_xrd_match_robust` — **robust tier**. Peak-list-based scoring
  with two algorithms: `'hanawalt'` (default, classical figure-of-merit
  search-match) or `'mip'` (mixed-integer linear programming for joint
  shift / scale / assignment optimization). Hundreds of milliseconds
  per candidate. Use for confident identification on real-lab patterns
  after the fast tier narrows the candidate list.
- `search_match_pattern` — **fingerprint search-match, the FIRST-choice
  blind route**. Identifies an unknown whose phase is probably known to
  science (the overwhelming majority of lab work) by Hanawalt-style
  matching of the measured d-lines + intensities against a PRECOMPUTED
  reference library (built once from a CIF collection, e.g. a COD
  mirror, via `build_fingerprint_library`). Deterministic, offline, no
  chemistry needed. Only when it returns no convincing match fall back
  to the indexing route below (possible new phase).
- **Isostructural impostors — read the absent-lines evidence.** A high
  `figure_of_merit` proves the candidate's lines lie where measured peaks
  are; it does NOT prove the phase (isostructural and superlattice phases
  share positions — a ZnS pattern can score 0.97 against a rare-earth
  bixbyite). Each `search_match_pattern` match reports
  `frac_strong_lines_absent` / `absent_strong_lines`: among near-tied
  candidates (within ~0.1 FOM), one with substantially MORE of its strong
  predicted lines absent from the measurement is the impostor. Caveats
  the evidence honestly carries: polytype/polymorph library entries can
  show absent lines for the TRUE phase, and strong texture suppresses
  real line families — so compare candidates against each other rather
  than applying an absolute cutoff, and prefer the chemically plausible
  candidate among ties (a common mineral over an exotic rare-earth
  compound, unless composition evidence says otherwise).
- `identify_mixture` — **blind MULTI-phase identification** (sequential
  subtraction over the fingerprint library). Search-match the peak list,
  accept the dominant phase, subtract its scaled predicted intensities
  (shared lines keep their unexplained remainder — no over-removal),
  re-search the residual, repeat; the accepted ensemble is then confirmed
  by ONE joint multi-phase MILP. Use it instead of a single
  `search_match_pattern` call whenever a mixture is suspected — including
  any in-situ / operando frame taken mid-transformation — or when a
  single-phase match leaves many strong measured peaks unmatched. Trust
  the `multiphase_confirmation` verdict over the per-iteration FOMs
  (those are computed against a residual that still contains other
  phases); each phase carries a `confirmed` flag from the joint solver —
  report `confirmed=False` phases as subtraction artifacts, not
  components. `intensity_share` is a screening abundance proxy only —
  quantify with Rietveld on the returned `structure_path`s. Strong peaks
  left in `residual_peaks` mean a phase missing from the library: run the
  indexing route on that residual list. Known limit: a sparse-line
  minority phase (e.g. a cubic fluorite-type with 3-4 strong lines) can
  be shadowed by a dense-line position-degenerate entry — when
  confirmation leaves peaks unmatched, re-examine them (chemistry
  plausibility, targeted `score_xrd_match_robust` against a suspected
  phase) before reporting.
- `quantify_phases_nnls` — **fast quantitative fractions** over a candidate
  SHORTLIST by non-negative least squares. Fits the continuous pattern as
  `y ≈ Σ cᵢ·patternᵢ`, `cᵢ ≥ 0`, so overlapping reflections are deconvolved in
  ONE joint fit — the complement to `identify_mixture`'s greedy peel-and-
  subtract. The light rung of the quantification ladder: `identify_mixture`
  (WHICH phases) → `quantify_phases_nnls` (fast fractions, screening) →
  `refine_rietveld_multiphase` (rigorous QPA with esds). Run it on the handful
  of candidates `identify_mixture` confirmed (its `structure_path`s), NOT the
  whole library. Returns per-phase `intensity_fraction` (a screening share,
  like `intensity_share` but from a joint fit) plus a ZMV-corrected
  `weight_fraction_est` (approximate — escalate to Rietveld for real QPA). NNLS
  always returns *some* combination, so the tool checks the worst unexplained
  residual against the noise and returns `reliable=False` when a phase is
  missing / off-database (organics, novel products) — treat the fractions as
  untrustworthy and add candidates or run `index_pattern` on the residual first.
- `track_phase_series` — **in-situ / operando series tracking**. After the
  series' establishing frames are identified, one call runs a joint multi-phase MILP
  per frame against that fixed endmember set → per-frame phase shares,
  onset/completion frames, the coexistence (transition) window, per-phase
  lattice-scale drift (thermal expansion), and residual alerts flagging
  frames the endmembers cannot explain (transient intermediate → run
  `identify_mixture` on that frame). Endmembers accept THREE forms: a
  library `source_id`, a simulated pattern, or an EMPIRICAL reference —
  the extracted peak list of a pure frame itself, essential when a phase
  (common for organics) is in no database. Deterministic, ~0.1-0.5 s per
  frame, no per-frame search or LLM. Limits: shares are screening proxies
  (Rietveld quantifies); phases separated by line POSITIONS only — an
  order-disorder transition that redistributes intensity on the same
  lattice needs profile/intensity analysis (`xrd_profile`), not tracking.
- `reconcile_series_phases` — **couple profile fitting with identification**
  over an in-situ series. Profile fitting (the `xrd_profile` skill) gives
  database-INDEPENDENT peak-evolution trends (positions/widths/
  areas, the transition); identification gives phase names where a database
  allows. This tool JOINS them: it attributes the peak-evolution trends to
  the identified phases and cross-checks the transition temperature the two
  methods find independently. Run it after both passes over the same frames.
  A regime whose phase is `null` is honestly UNIDENTIFIED (not in the
  database — organics/novel products): its trends are real but unnamed. When
  the two transitions AGREE it is strong corroboration; when they DIVERGE,
  investigate (mis-tracked peaks, a mid-series false ID, or a two-step
  process). See the in-situ two-pass workflow in `interpretation`.
- `calibrate_zero` — **2θ calibration from an internal standard** (Si /
  LaB₆ / corundum mixed into the sample). Fits zero error + specimen
  displacement from the standard's exactly-known lines and returns the
  corrected sample peak list. When a standard is present, run it FIRST —
  indexing and lattice refinement are exquisitely sensitive to zero
  error. Prefer its `corrected_peaks` over passing `zero_offset` alone
  (the two aberration terms trade off; their sum is what's accurate).
- `index_pattern` — **blind-identification entry point** (optional `gsas`
  extra). Autoindexing: recovers ranked candidate unit cells (lattice
  parameters + Bravais + de Wolff M20) from peak positions alone — no
  chemistry needed. Use it when the sample composition is unknown, then
  search databases with the recovered lattice as a filter; also useful to
  corroborate a chemistry-led ID (indexed cell must match the identified
  phase). Needs ≥10 clean peaks; unreliable for triclinic.
- `validate_cell_lebail` — **cell arbiter** (optional `gsas` extra).
  Structure-free Le Bail whole-pattern fit of a candidate unit cell (no
  atoms — reflection intensities are free). Run it on `index_pattern`'s
  candidates BEFORE any database search: a wrong cell or subcell leaves
  peaks unaccounted (low `profile_corr`), the true cell fits (≥~0.9) and
  its lattice refines to precision with no structure. A supercell always
  fits as well as the true cell — among fitting cells prefer the
  smallest. Do NOT judge a cell by simulating same-cell structures:
  intensities belong to the structure, not the cell, and a correct cell
  with a wrong structure scores badly.
- `determine_space_group` — **systematic-absence analysis** (no extra
  deps). After a cell is Le-Bail-validated, tests the International-
  Tables reflection conditions (centerings, screw axes, glide planes)
  against which lines are observed vs absent, and returns the evidence
  plus frequency-ranked COMMON space groups consistent with it. Powder
  absences often cannot decide a unique group (extinction-symbol twins
  are indistinguishable) — treat the output as evidence, corroborate the
  final choice with a Le Bail fit under that group. New-phase workflow
  only; a database-identified phase already has its group.
- `refine_rietveld_multiphase` — **quantitative phase fractions** (optional
  `gsas` extra). After `identify_mixture` confirmed a phase set, refine ALL
  phases jointly against the whole profile → per-phase WEIGHT fractions
  (the quantitative upgrade of the screening `intensity_share`), per-phase
  lattice + microstrain. Fractions are meaningful only when every
  crystalline phase is included; amorphous content is invisible.
- `refine_rietveld_series` — **in-situ quantification** (optional `gsas`
  extra). Sequential multi-phase Rietveld over a frame series: per-frame
  weight fractions w(T)/w(t) — the standard in-situ deliverable — and
  per-frame effective cells (thermal expansion). Choose
  `establishing_frame` where ALL phases are clearly present. Run AFTER
  `track_phase_series` established the phase set; it quantifies, it does
  not screen.
- `refine_rietveld` — **refinement tier** (optional `gsas` extra). *After*
  the phase is identified, Rietveld-refine that structure against the
  measured pattern to extract accurate lattice parameters (+ esd),
  crystallite size and microstrain, and quantify the whole-profile fit.
  A staged GSAS-II refinement (background+scale → 2θ zero → sample
  broadening → cell). This is a follow-up to identification, **not** part
  of the identification loop — only run it once a candidate is confirmed,
  and read `profile_corr` (not the absolute `Rwp`) as the fit quality when
  the pattern is in arbitrary intensity units.

Install dependencies: `pip install scilink[structure-matching]` (pymatgen
with the XRD analysis module, mp-api, pulp). Rietveld refinement
additionally needs GSAS-II, which is not on PyPI: `pip install "GSAS-II @
git+https://github.com/AdvancedPhotonSource/GSAS-II.git"` (built from source
— see the `simulate_xrd_pattern` `gsas` engine docs for the recipe).

**Extending the backend list.** Materials Project, local CIF, and COD
ship in-package. ICSD, OQMD, AFLOW, NOMAD, and any custom database
plug in through a small public API. A user implements a class
satisfying the `StructureBackend` protocol
(`is_available()` + `query(spec) -> list[StructureCandidate]`) and
either calls `register_backend("icsd", ICSDBackend)` from their code,
or declares it in their package's `pyproject.toml`:

```toml
[project.entry-points."scilink.structure_backends"]
icsd = "my_package.icsd_backend:ICSDBackend"
```

SciLink discovers the entry point on import and the backend is then
addressable as `sources=["icsd"]` in `search_structures`. See
`scilink/skills/structure_matching/_backends/__init__.py` for the full
protocol and the registration helpers.

## planning

**This is pattern-MATCHING, not fitting.** The deliverable is a ranked candidate
match plus a `figure_of_merit` (the skill's quality gate) — *not* a fitted model.
The plan must not list peak-shape fit parameters (pseudo-Voigt mixing `eta`,
per-peak FWHM / amplitude as fit targets); the only quantitative outputs are the
matched phase(s), the scorer's `figure_of_merit`, and its fitted zero-shift /
lattice-scale. Every stage below chains search → simulate → score; no stage fits
the experimental pattern.

**Two-tier identification workflow** — both tiers usually run, never
either-alone:

1. **Pre-fit / triage** with the **fast tier**. Query the DB for top-N
   candidates (N between 5 and 10), simulate each one's pattern, run
   `score_xrd_match_fast` on every candidate. Rank by correlation. This
   establishes which candidates are even plausible — typically 3-4
   survive the `verdict='accept'|'marginal'` filter; the others
   (`verdict='reject'`) can be dropped.

2. **Confident identification** with the **robust tier**. Extract the
   experimental peak list once with `extract_peaks`. For each surviving
   candidate from step 1, run `score_xrd_match_robust(algorithm='hanawalt')`.
   Report the best by figure-of-merit. Switch to `algorithm='mip'` when:
   - The pattern is **suspected multi-phase** (see below — this is the
     dominant trigger).
   - You need the fitted zero-shift and lattice scale reported as
     parameters (e.g., for downstream Rietveld initialization).
   - Hanawalt's top candidates are within ~10% FOM of each other and you
     want a provably optimal tie-breaker.

**Three invocation modes for step 1:**

- **Pre-fit pattern (single-phase)** — chemistry is hypothesized as one
  compound (e.g. "this is a TiO2 sample" or `chemistry_hint=["Ti","O"]`
  for the binary). Query DB with `chemistry=[Ti, O]` (single list);
  optionally add `space_group_hints` if the user names a polymorph.

- **Multi-phase mixture** — when the metadata / notes suggest a mixture
  ("Si + Ge mixture", "suspected multi-phase", `chemistry_hint=["Si","C"]`
  for two distinct elemental phases, etc.). With NO chemistry hypothesis
  (blind mixture, in-situ frame), run `identify_mixture` — it discovers
  the phase candidates itself and ends with the same joint MILP. With a
  chemistry hypothesis, use the **list-of-lists**
  chemistry form to get separate candidate lists per phase:
  `chemistry=[["Si"], ["Ge"]]` — NOT `chemistry=["Si", "Ge"]` which would
  ask the DB for Si-Ge *binary compounds* instead of Si and Ge
  separately. For step 2 use **`score_xrd_match_multiphase` from the
  start** (not the per-candidate `score_xrd_match_robust` with
  `algorithm='mip'`). The multi-phase tool accepts the full list of
  candidate phase patterns and solves one joint MILP across them, with
  per-phase activation binaries that let the solver leave a phase out
  entirely. Output includes per-phase coverage and matched-peak lists
  — strictly more informative than running per-candidate MIPs and
  comparing them, which loses the cross-phase assignment constraints.
  The per-candidate `score_xrd_match_robust(algorithm='mip')` remains
  available as a fallback when only one phase's identity matters and
  the per-phase decomposition isn't needed.

- **Post-fit pattern (no chemistry hypothesis)** — no hint at all. This
  is the blind / unknown-sample case. **Fingerprint first**: run
  `extract_peaks` then `search_match_pattern` against the reference
  library (with `calibrate_zero` first when an internal standard is
  present) — a convincing figure of merit identifies the phase in one
  deterministic call; confirm by simulating the hit and refining. Only
  when no library match is convincing take the **indexing route**
  (possible new phase): run
  `extract_peaks` (feed the indexer ALL confident peaks — lower
  `prominence_frac` if it returns fewer than ~10), then `index_pattern`
  to recover candidate unit cells from the peak positions alone.
  **Arbitrate the cells with `validate_cell_lebail` before searching**:
  Le-Bail-fit each plausible candidate (cheap, structure-free) and keep
  the SMALLEST cell that fits the whole profile — this kills wrong cells
  and subcells immediately and turns alias families into a single
  answer, instead of burning database searches on a bad cell. Then
  search by the CELL, not by guessed elements: a `search_structures`
  call may omit `chemistry` entirely and pass the indexed cell's
  `lattice_param_ranges` as the query key (served by COD and local-CIF;
  MP requires chemistry). For a NON-cubic cell pass the
  permutation-invariant `volume` range rather than per-axis a/b/c
  ranges — a database entry's axis convention may be a permutation of
  the indexed cell's, and per-axis windows would false-reject it. The
  cell-only hit list intentionally includes chemically different phases
  sharing the cell (e.g. a Si-cell query also returns solid Ar); the
  simulate+score loop discriminates them — intensities carry the
  chemistry. Add a chemistry list-of-lists to the same call only as a
  *refinement* when composition hints exist. **Never loop over
  single-chemistry `search_structures` calls** — one call handles
  multiple hypotheses. Indexing caveats (see the tool docs): solutions
  come in families (×1/2 subcells, ×√2/×√3 supercells), so when the top
  cell finds no DB match, try the other candidates and volume-related
  multiples; the simulate+score loop — not M20 — is the arbiter of a
  candidate cell. Finally, whatever phase wins scoring, its cell should
  agree with the indexed cell — a disagreement means one of the two is
  wrong.

**Recognizing the multi-phase trigger.** Any of these phrasings in
`system_info` / notes mean "use `score_xrd_match_multiphase`":
"mixture", "multi-phase", "two-phase", "binary mixture" (as opposed to
"binary compound"), "co-existing phases", or `chemistry_hint`
containing two elements with NO compound name (e.g. `["Si", "Ge"]`
with note "suspected mixture" — the elements are distinct phases, not
a compound). Blind (no chemistry): `identify_mixture` does the whole
loop — candidate discovery, subtraction, joint confirmation — in one
call. **When the objective asks HOW MUCH — quantitative fractions, "what
proportion", or "screen which of these N candidates are present and in what
amount" — quantification is a DISTINCT planned step (the scorer coverages /
RIR proxies are only screening estimates). This decomposition step IS a fit
— it is the one exception to the pattern-MATCHING-not-fitting framing above,
which governs the identification stages only. Route it by cost:**
- **FIRST `quantify_phases_nnls`** over the candidate CIF shortlist — a
  fixed-library non-negative decomposition that fits ALL candidates against
  the observed pattern at once, so absent candidates fall to ~0 weight and the
  present ones get fractions in a single fast solve. This is the right first
  pass for a many-candidate screen or per-frame series quantification, and it
  returns `reliable=False` when a phase is missing / off-database (it does not
  fabricate fractions). Plan it the moment the task supplies several candidate
  structures and asks which are present / in what amount.
- **THEN escalate to `refine_rietveld_multiphase`** for rigorous weight
  fractions with esds, on the confirmed small (2–3) phase set NNLS screened in.
- **Never force single-phase Rietveld on a mixture** — it collapses all
  intensity into one phase and reports a fake 100 %. Choose each phase's structure by scoring candidate
simulations against THAT phase's own matched peaks (within-phase relative
intensities are fraction-independent) — never against the whole mixture,
where a weak-scattering phase's candidates all look bad — and REQUIRE the
entry's space group to match the named phase: a named mineral means its
standard polymorph (corundum = R-3c α-Al2O3, zincite = P6_3mc, fluorite =
Fm-3m), and a right-composition wrong-polymorph entry (θ- vs α-Al2O3) is
the dense-line impostor in miniature — its line forest covers the phase's
few peaks while its own strong lines are absent from the data. And treat
Rietveld red flags as a bad STARTING STRUCTURE, not a dead end: put a
bounded swap-retry loop INSIDE the script — if `converged` is False, a
refined cell departs from the named phase's known lattice, or a phase the
scorer confirmed refines to ~0/100% weight, substitute the next-best
entry for the offending phase and call the refinement again (very old
database entries with idealized coordinates and no displacement
parameters make bad starting models). A retry loop
needs a POOL: keep several space-group-consistent entries per phase
(tolerate space-group setting variants when filtering), and when a
phase's pool shrinks to one entry, WIDEN the search (raise top_n, relax
filters) before refining — a pool of one cannot be swapped. When in doubt, run `score_xrd_match_multiphase` with a
single-candidate list — it gracefully reduces to the single-phase MIP
under that input (with one phase always active) and the joint solver's
output format is the same.

**Candidate count — tight first pass, widen on failure.** The cost is the
per-candidate *simulation*, so keep the FIRST pass cheap: `search_structures(
query={"top_n": N, ...})` with N in [5, 10]. That already identifies common
single-polymorph phases.

But if that pass FAILS (best `figure_of_merit` below the accept threshold /
all candidates marginal-or-reject), the answer is often a phase that the tight
retrieval simply did not return — a *polymorph-rich* chemistry (e.g. Ti-O has
many TiO2 polymorphs plus Magnéli suboxides; the right one can be candidate #20,
not #5). On such a re-plan you SHOULD widen the retrieval — `top_n` up to ~30
(and/or relax symmetry/lattice filters, or try an alternate chemistry
hypothesis) — and re-run. This widening is *expected and allowed* on a failed
pass; do not stay capped at 10 while re-planning a search that returned no
confident match. (For an in-situ *series*, keep it tight per frame — the
establishing frame can widen once, then lock the identified phase.)

**Wavelength selection.** Default CuKa unless the experiment metadata
says otherwise. MoKa is common for high-2θ work. A wavelength mismatch
gives uniformly bad correlations / FOMs for all candidates with a
characteristic shift in peak positions — suspect that first when every
candidate scores reject.

Call `resolve_wavelength(system_info)` once near the top of the
analysis script and pass its return value to every
`simulate_xrd_pattern` call instead of hard-coding `wavelength='CuKa'`.
The resolver reads structured `experiment.wavelength` / `source` /
`x_ray_source` fields first, then falls back to a free-text scan for
canonical source names. When metadata is silent it returns the default
`'CuKa'`, so the call is safe even on patterns with no metadata at all.

**Narrowing the candidate list.** The `search_structures` `query` dict
accepts three optional filters that often pay for themselves:

- `z_range: (int, int)` — number of sites per unit cell. Useful when
  the user has a rough atom-count expectation (`(1, 8)` for simple
  binaries; `(8, 50)` for typical oxides).
- `density_range: (float, float)` — g/cm³. Narrows by physical density
  when the sample's bulk density is known from independent measurement.
- `anonymous_formula: str` — stoichiometry template, e.g. `'AB2'` for
  rutile/anatase-type, `'ABC3'` for perovskites. Pymatgen's
  `Composition.anonymized_formula` is the matching convention.

All three are optional and respected by Materials Project and the
local CIF backend; COD ignores them.

**Profile fitting is a DOWNSTREAM follow-up, not part of identification.**
Crystallite size / strain (per-peak pseudo-Voigt → Scherrer / Williamson-Hall)
is the `curve_fitting/xrd_profile` skill's job, run as a **separate step after
the phase is identified** — never an in-ID fit. Identification needs only peak
*positions and relative intensities* (the light `extract_peaks`), which the
scorers consume directly. Do **not** call `fit_profile`, fit pseudo-Voigt peaks,
or compute per-peak R² inside the identification script: it adds nothing the
match needs and pulls the run into the curve-fitting pipeline (an R²-shaped
deliverable the `figure_of_merit` gate then rejects, triggering avoidable
refinement iterations). When line broadening matters for the match on
nanocrystalline data, **widen the scorer's `fwhm` (0.3-0.5°)** rather than
fitting each peak.

## analysis

**CRITICAL: structure-matching workflow.** The per-item analysis script
must follow this exact sequence:

1. Load experimental 2-theta + intensity arrays.
2. Call `search_structures` with `top_n` 5-10 (first pass). If the run is a
   re-plan after a failed/low-FoM pass, widen `top_n` up to ~30 and/or relax
   filters (see "Candidate count — tight first pass, widen on failure").
3. For each candidate, call `simulate_xrd_pattern` and `score_xrd_match_fast`.
4. Filter to candidates with `verdict in {'accept', 'marginal'}`.
5. Call `extract_peaks` once on the experimental pattern.
6. For each surviving candidate, call `score_xrd_match_robust` (default
   `algorithm='hanawalt'`).
7. Emit a `MATCH_RESULTS_JSON: {...}` line collecting the ranked match
   list, and a `FIT_RESULTS_JSON: {...}` line with verdict + best-match
   metadata. The `search_structures` tool already prints its own
   `DB_MATCHES_JSON:` marker; the framework's stdout parser lifts it
   into `fit_results['db_matches']` automatically.

**Don't profile-fit for identification.** Two common over-builds to avoid:
- The **fast tier needs no peak extraction** — it cross-correlates the
  *continuous* (background-subtracted) pattern directly. Subtract the background,
  then correlate; do not extract or fit peaks before Step 3.
- For the **robust tier**, use the **light** `extract_peaks` (positions +
  relative intensities). Do **NOT** fit a pseudo-Voigt profile to every peak —
  that is the `xrd_profile` skill's specialized job (crystallite size / strain)
  and is unnecessary for identification, which only needs positions + relative
  intensities. FWHMs from `extract_peaks` are optional refinement, not a goal.
- **Emit `figure_of_merit`, never `r_squared`.** The ID gate scores by
  `figure_of_merit`; there is no curve fit here, so there is no R² to report.
  For the visualization, overlay the best-match **simulated** pattern (broadened
  sticks) on the experimental data — do not `curve_fit` a profile for the plot
  or compute an R² for it.
- **Plot legends must name what is actually drawn.** An identification
  overlay is NOT a fit — label it `"Simulated <formula> (match overlay)"`,
  and label its difference trace `"Data − overlay"`, never `"Fit"` /
  `"Residuals"`. Reserve the word *fit* for genuine refinement output
  (Rietveld/Le Bail `y_calc`, labeled e.g. `"Rietveld fit (Rwp=…)"`).
  Never emit legend entries for curves that are not drawn (no template
  `"Component 1"` / `"Background"` leftovers), and title the figure by
  what it shows (`"Data and match overlay"`, not `"Data and Fit"`).
  This is a scientific-honesty rule, not cosmetics: a mismatched
  overlay labeled "Fit" reads as a failed refinement — and at a phase
  transition the overlay SHOULD mismatch. In series mode this applies
  per frame: each frame's legend carries the phase(s) actually overlaid
  on THAT frame.
- **Overlay rendering must not misrepresent match quality.** Broaden the
  simulated sticks to the DATA's observed peak width (estimate the
  narrowest resolved experimental peak; `fwhm='auto'` does this) — an
  over-broadened overlay smears a positionally good match into an
  apparent mismatch. Draw a combined SUM curve ONLY when TWO OR MORE
  phases are overlaid — with a single phase the overlay already IS the
  sum, so a separate "Sum overlay" just draws an identical line on top of
  the component, hiding it and leaving a legend entry for an invisible
  curve; draw ONE line. Likewise never plot "Raw" as its own curve when
  it coincides with "Data" — every legend entry must correspond to a
  curve that is actually visible and distinguishable in the plot. Pick
  the y-scale by dynamic range — log
  when peaks span more than ~1.5 decades, otherwise linear — and keep it
  consistent across all frames of one series (a minority phase that is
  invisible on a linear axis is a plotting failure, not evidence of
  absence).
- **Plot only metrics this skill defines.** Trend panels and dashboards
  carry `figure_of_merit`, per-phase `coverage` (a fraction — CAP at 1;
  if a computed value exceeds 1, the aggregation is wrong: never sum
  coverages of duplicate entries of the same phase), shares,
  `lattice_scale`, zero-shift. NEVER resurrect a forbidden metric in a
  dashboard (an R² panel on a matching-mode series plots noise and
  invites misreading).
- **Mark saturation and quantization — do not plot them as physics.** A
  fitted value pinned at its search bound (e.g. zero-shift at ±0.4°, the
  scorer's default window) is a SATURATED estimate: draw the bound as a
  dashed line and flag pinned points instead of connecting them as data.
  Grid-quantized knobs (lattice_scale steps of 0.002) produce staircase
  trends — say so in the caption/text before fitting a slope through the
  steps.

**Complete two-tier template** — adapt for the active wavelength and
chemistry hypothesis:

```python
import json
import numpy as np

from scilink.skills.structure_matching.xrd.search_structures import search_structures
from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
from scilink.skills.structure_matching.xrd.score_match_fast import score_xrd_match_fast
from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust
from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks

# ---- Step 1: Load experimental pattern ----
data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
exp_2theta, exp_intensity = data[:, 0], data[:, 1]

# ---- Step 2: Query DB ----
hits = search_structures(
    query={"chemistry": CHEMISTRY_HINT, "top_n": 5},
    output_dir="./candidates",
)

# ---- Step 3: Fast tier — broad ranking ----
WAVELENGTH = "CuKa"
two_theta_range = (float(exp_2theta.min()), float(exp_2theta.max()))
fast_results = []
for cand in hits["candidates"]:
    sim = simulate_xrd_pattern(
        cand["structure_path"], wavelength=WAVELENGTH,
        two_theta_range=two_theta_range,
    )
    fast = score_xrd_match_fast(
        exp_two_theta=exp_2theta.tolist(),
        exp_intensity=exp_intensity.tolist(),
        sim_two_theta=sim["two_theta"],
        sim_intensity=sim["intensities"],
    )
    fast_results.append({**cand, "fast": fast, "sim_peaks": sim})

# ---- Step 4: Filter to plausible candidates ----
plausible = [r for r in fast_results if r["fast"]["verdict"] in {"accept", "marginal"}]
if not plausible:
    plausible = sorted(fast_results, key=lambda r: -r["fast"]["correlation"])[:3]

# ---- Step 5: Extract experimental peaks once ----
exp_peaks = extract_peaks(exp_2theta.tolist(), exp_intensity.tolist())

# ---- Step 6: Robust tier — confident identification ----
final = []
for r in plausible:
    robust = score_xrd_match_robust(
        sim_two_theta=r["sim_peaks"]["two_theta"],
        sim_intensity=r["sim_peaks"]["intensities"],
        exp_peaks=exp_peaks,
        algorithm="hanawalt",  # change to 'mip' for multi-phase / fitted parameters
    )
    final.append({
        "id": r["id"], "source": r["source"], "formula": r["formula"],
        "space_group": r["space_group"],
        "correlation_fast": r["fast"]["correlation"],
        "figure_of_merit": robust["figure_of_merit"],
        "verdict": robust["verdict"],
        "fitted_shift_fast": r["fast"]["fitted_shift"],
        "fitted_scale_fast": r["fast"]["fitted_scale"],
    })

final.sort(key=lambda r: -r["figure_of_merit"])
best = final[0] if final else None

# ---- Step 7: Emit ranked match list + fit results ----
print("MATCH_RESULTS_JSON: " + json.dumps(final))
print("FIT_RESULTS_JSON: " + json.dumps({
    "best_match": best,
    "candidates_considered": len(final),
    # The interpretation stage reads ONLY 'parameters' and 'fit_quality'
    # from this payload — the phase identity MUST be inside 'parameters'
    # or the final report cannot name the phase it identified.
    "parameters": {
        "identified_phase": best["formula"] if best else None,
        "space_group": best["space_group"] if best else None,
        "database_id": best["id"] if best else None,
        "runner_up": final[1]["formula"] if len(final) > 1 else None,
        "runner_up_fom": final[1]["figure_of_merit"] if len(final) > 1 else None,
    },
    "fit_quality": {
        "figure_of_merit": best["figure_of_merit"] if best else None,
        "verdict": best["verdict"] if best else "no_candidates",
    },
}))
```

**The identification IS the deliverable — emit the identity into
`parameters`.** Downstream interpretation reads `parameters` and
`fit_quality` from `FIT_RESULTS_JSON`; content anywhere else (including
`best_match`, which feeds the report tables) is invisible to it. So
`parameters.identified_phase` (+ `space_group`, `database_id`) is
MANDATORY in every identification emit, and the analysis text must NAME
the phase. A run that reports only a figure of merit has discarded its
own answer.

**Multi-phase emit (REQUIRED when using `score_xrd_match_multiphase`).** The
multi-phase scorer returns ONE result with `active_phases` (not a ranked
per-candidate list), plus `figure_of_merit` (= 1 − cost) and `verdict`. The
quality gate reads `fit_quality.figure_of_merit`, so you MUST surface the
multi-phase `figure_of_merit` there or the run is hard-rejected as "metric
missing" regardless of how good the match is:

```python
mp = score_xrd_match_multiphase(exp_peaks=exp_peaks, candidates=candidates)
print("FIT_RESULTS_JSON: " + json.dumps({
    "active_phases": mp["active_phases"],          # each: id, formula, coverage,
                                                   # matched_peaks, lattice_scale
    "unmatched_exp": mp["unmatched_exp"],          # peaks no phase explains
    "parameters": {                                # interpretation reads THIS
        "identified_phases": [p["formula"] for p in mp["active_phases"]],
        "phase_coverages": {p["formula"]: p["coverage"]
                            for p in mp["active_phases"]},
    },
    "fit_quality": {
        "figure_of_merit": mp["figure_of_merit"],  # gate reads THIS
        "verdict": mp["verdict"],
        "cost": mp["cost"],
    },
}))
```

**Background handling.** Both scorers default to subtracting the
experimental minimum as a flat offset. For patterns with significant
continuous background (amorphous halo, fluorescence), call
`fit_background(two_theta, intensity, method='snip')` from
`scilink.skills.curve_fitting.xrd_profile.background` to estimate the
continuous background, subtract it, and pass `background="none"` to
the scorers. SNIP handles smooth amorphous floors without imposing a
polynomial shape; use `method='polynomial'` only when a polynomial is
genuinely the right model.

**Wavelength consistency.** Pass the same `wavelength` to every
`simulate_xrd_pattern` call. Pull from experiment metadata when present;
otherwise default CuKa.

**Peak broadening.** The fast tier's `fwhm` defaults to `'auto'`, which
estimates the experimental peak width and broadens the simulation to match
(floored at 0.15°) — so nanocrystalline / low-resolution patterns are handled
without manual tuning. Pass a number only to force an exact width. The robust
tier matches on peak *positions* within `tol_deg` (default 0.3°); widen
`tol_deg` to ~0.5° for very broad peaks whose centers are poorly defined.

**When the robust tier disagrees with the fast tier.** Trust the robust
tier — it factors out background, scale, and intensity-ratio effects
that the fast tier folds into the correlation. The fast tier is a
triage step; the robust tier is the identification.

**In-situ / series — profile fitting and identification are two passes,
reconciled.** The richest in-situ analysis runs BOTH: profile fitting (the
`xrd_profile` skill — `fit_pattern` per frame) for the database-independent
structural evolution (peak positions → thermal expansion, widths →
crystallite size/strain, areas → phase-fraction proxy, and the transition
from peak appearance/shift), and identification (this skill) for the phase
names. They answer different questions and are gated differently, so they
are two separate `analyze` passes over the same frames, joined afterward by
`reconcile_series_phases` — which labels the peak-evolution trends with the
identified phases and cross-checks the transition the two methods find.
Prefer this coupled workflow whenever the phases might not all be in a
database: profile fitting keeps working where identification hits the "not
in any database" wall (organics, novel synthesis products), so the trends
stay complete and the reconciliation reports honestly which regimes it
could and could not name. Agreement between the two transitions is strong
corroboration; divergence is a flag to investigate. Identification alone
(the establishing-frames workflow below) is the right lighter pass when you
only need "which phases, how much" and the phases are known.

**Quantifying "how much" — the fraction ladder.** Once `identify_mixture` (or a
search-match shortlist) has given you the candidate phases, route the
quantification by cost, and NEVER force a single-phase fit on a mixture (it
collapses all intensity into one phase and reports a fake 100 %):
1. `quantify_phases_nnls` — the FIRST-choice fast quantifier. Fits the whole
   pattern as a non-negative sum of ALL candidate patterns at once, so absent
   candidates fall to ~0 weight and the present ones get fractions in one solve.
   Use it for a fast screen over MANY candidates, for per-frame series
   quantification (Rietveld per frame is too slow), or any "roughly how much of
   each" answer. It returns `reliable=False` when a phase is missing — honest.
   This is the right tool the moment the task says *fractions / how much / screen
   which of these are present* over more than one candidate. CALL THE TOOL — do
   NOT hand-roll `scipy.optimize.nnls`; the tool builds the design matrix
   (simulated + broadened patterns), handles background, returns weight-fraction
   estimates, and does the missing-phase abstention you would otherwise skip:
   ```python
   from scilink.skills.structure_matching.xrd.quantify_nnls import quantify_phases_nnls
   q = quantify_phases_nnls(two_theta, intensity, candidate_cif_paths,
                            wavelength=lam, fwhm_deg=0.15, background='snip')
   # q['phases']   -> [{'structure_path', 'intensity_fraction', 'weight_fraction_est'}, ...]
   # q['reliable'] -> False when the candidates can't explain the pattern
   #                  (a phase missing / off-database) -> report that, DON'T fake fractions.
   ```
2. `refine_rietveld_multiphase` — escalate to this only for RIGOROUS weight
   fractions with esds on a small CONFIRMED phase set (typically 2–3 phases NNLS
   already screened in). Do not run it as the first pass over a long candidate
   list, and do not run single-phase Rietveld to "quantify" a mixture.

**In-situ / series — identify at ESTABLISHING frames, then track.** A
series is not N independent identifications. Identify each phase where it
is PUREST, then score every frame against that fixed endmember set. For a
simple transformation ramp (dehydration, calcination, a polymorphic
transition — the common shape) the establishing frames are simply the
first and last frames; when the shape differs, choose accordingly:
a crystallization series establishes on the first frame with Bragg peaks
(the amorphous start has nothing to identify); an incomplete or multi-step
run (A → B → C) may need a mid-series frame — start with the frames you
CAN identify, track, and let the residual alerts point at what is missing;
spectator phases present in every frame (holder, crucible, additives,
internal standard) mean even the "pure" frames are mixtures — establish
with `identify_mixture` and include the spectators as endmembers. When an
alert fires (unexplained intensity concentrated in some frames), identify
THAT frame (`identify_mixture` on its residual-heavy peak list), ADD the
discovered phase to the endmember set, and re-run the tracking — the
endmember set converges in one or two passes. Do **not** re-identify per
frame: besides being slow, a mid-transition frame actively resists
identification — coverage splits between phases and overlapping peaks pull
the extracted centroids, so the true phase's FOM sinks while dense-line
degenerates (severe for organics at low angle) float up. When a phase is
in NO database (common for organic/molecular phases), use its establishing
frame's extracted peak list as an EMPIRICAL endmember — identification of
that phase can wait; tracking cannot. Two practical cautions for in-situ
lab data: crop the low-angle air-scatter upturn (2θ ≲ 5°) before peak
extraction, and expect a residual bump on the first frames AFTER a
transition (a freshly formed phase is cooler/less relaxed than the
endpoint reference — lattice-scale drift, not a new phase).

**Tracking, two execution shapes.** When the whole series is available to
one script, call `track_phase_series` — the deterministic driver (shares,
transition window, alerts in one call). When the agent's per-frame series
loop runs each frame in its own working directory, use the lock-file
pattern below — the anchor frame locks the phase set, later frames score
against it.

**Lock the phase set, not the search.** For a
time/temperature series (operando, a ramp), identify ONCE on the establishing
(anchor) frame, then score every later frame against that LOCKED phase set — do
**not** re-search the database per frame (slow, and the candidate ranking can
flicker frame-to-frame). The agent reuses the anchor's script VERBATIM in a
per-frame working directory, and the anchor always runs first, so make the script
**self-locking** via a shared file one directory up (the per-frame dirs are
siblings under a common parent):

1. Read this frame's pattern from the canonical `data.npy` in the working dir.
2. Resolve a shared lock path: `lock = Path.cwd().parent / "xrd_locked_phases.json"`.
3. **Anchor frame (lock absent):** run the normal search → simulate → score
   identification; once the phase(s) are confirmed, SAVE the matched phase
   references to `lock` — for each phase its simulated `two_theta` + `intensities`
   (the `sim_*` keys `score_xrd_match_multiphase` expects), plus `formula`, `id`,
   and any fitted `lattice_scale`. This is the "lock the identified phase" step.
4. **Later frames (lock present):** SKIP the search entirely; load the locked
   phase patterns and run `score_xrd_match_multiphase(exp_peaks=<this frame's
   extracted peaks>, candidates=<locked phases>)` → per-frame per-phase `coverage`
   and `lattice_scale`.
5. Emit `FIT_RESULTS_JSON` with `figure_of_merit` plus, for THIS frame, each
   phase's `coverage` and `lattice_scale`. Aggregated across frames these trace
   the **phase-fraction evolution** (coverage rising / falling = a phase growing /
   disappearing) and **thermal expansion** (lattice_scale drift).

The anchor establishes the lock before any later frame reads it (later frames may
run in parallel and only READ the lock — no race). Locking the phase *set* — not
a frozen search, not frozen peak positions — is what generalises: each later frame
re-extracts its own peaks and re-fits per-phase lattice scale, so the method
follows peak shifts (thermal expansion) and intensity changes (phase fraction)
while keeping the *identity* decision fixed from the anchor. (Mirrors the
`xrd_profile` skill's "lock the method, not the values," adapted to identification.)

## interpretation

**Verdict thresholds.**

Fast tier (correlation, in [-1, 1]):
- correlation ≥ 0.85 → "accept"
- correlation ≥ 0.60 → "marginal"
- otherwise → "reject"

Robust tier — Hanawalt (figure-of-merit, in [0, 1]):
- FOM ≥ 0.70 → "accept"
- FOM ≥ 0.40 → "marginal"
- otherwise → "reject"

Robust tier — MIP (cost, in [0, 1], lower is better):
- cost ≤ 0.25 → "accept"
- cost ≤ 0.55 → "marginal"
- otherwise → "reject"

**Reporting style for synthesis.** Phrase identification in proportion
to the robust-tier margin:

- Best FOM > 0.85 and runner-up > 1.5× lower: declare identification
  with high confidence. Name the phase, space group, and (if MIP was
  used) the fitted zero-shift and lattice scale.
- Best FOM in [0.70, 0.85]: declare identification with caveats. List
  alternative candidates within 0.1 FOM of the best.
- Best FOM in [0.40, 0.70]: do not declare a single identification.
  List the top 3 candidates as "consistent with the experimental
  pattern" and recommend higher-resolution data or a complementary
  technique (EELS, EDS).
- All FOMs below 0.40: report no confident match. Suggest broadening
  the chemistry hypothesis, checking the experimental wavelength, or
  running a peak-fit-first workflow.

**What a good score cannot tell you.** A high FOM / low MIP cost
confirms the phase is present; it does NOT confirm purity, sample
preparation correctness, or absence of minor phases. Mention this
caveat when reporting identifications.

**Multi-phase samples.** Hanawalt is single-phase by construction. For
suspected multi-phase mixtures, switch to `score_xrd_match_multiphase`
and feed it the full candidate phase list — the joint MILP allocates
peaks across phases with per-phase activation binaries and reports
per-phase coverage. Increase `max_exp_peaks` (default 30) when the
mixture has many resolved peaks per phase. The tool does NOT compute
quantitative phase fractions (peak-area weighted Rietveld refinement
is the standard for that); the `coverage` field is a peak-count
proxy, not a phase fraction. Note this caveat in the report.

**`predicted_coverage` — reject over-predicting false matches.** Each active
phase reports `predicted_coverage`: the intensity-weighted fraction of *that
phase's own strong reflections* that are actually present in the data
(bidirectional matching). A real phase shows nearly all its strong peaks
(`predicted_coverage` ≈ 0.8–1.0). A **peak-rich wrong phase** (e.g. a Magnéli
suboxide standing in for TiO₂, or a telluride/alloy standing in for a pure
metal) explains a few experimental peaks by overlap but leaves most of its OWN
strong peaks unobserved (`predicted_coverage` ≈ 0.1–0.4). **Treat an active
phase with `predicted_coverage` below ~0.5 as a likely false match: do not
report or lock it — widen the search (more candidates / the correct polymorph)
to find a phase that explains those peaks with high predicted_coverage.** Low
`coverage` AND low `predicted_coverage` together = the scorer latched onto a
minority of correctly-positioned peaks while the phase is wrong. (A genuinely
textured sample can suppress some reflections and lower `predicted_coverage`
legitimately — weigh it against the residuals and the alternatives.)

## validation

**Per-candidate sanity checks.**

- The simulated pattern must cover the experimental 2-theta range. If
  `min(sim["two_theta"])` is more than 1° above `min(exp_2theta)`,
  expand the simulation range or note unobserved low-angle peaks.
- The number of simulated peaks should be reasonable for the
  candidate's symmetry (5-50 peaks typical in 10-90° 2θ; outliers
  indicate the simulation range was too narrow or too broad).
- Fast and robust tier verdicts should agree on direction. When the
  fast tier says "accept" but the robust tier says "reject", the
  correlation is being fooled by background or scale — re-run with a
  polynomial background subtraction.

**Cross-candidate sanity.** When 3+ candidates score "accept" in the
robust tier, the chemistry hypothesis is too broad — narrow it. When 0
candidates score above "reject", the chemistry hypothesis is wrong or
the wavelength is mismatched.

**Zero-shift sanity (MIP only).** A fitted zero-shift larger than 0.3°
typically indicates either a sample-displacement problem on the
instrument or a wavelength mismatch — not a true identification issue.
Flag it in the report.

**FWHM tuning.** If the best candidate scores FOM > 0.6 but visual
inspection (peak positions match) suggests it's the right phase, retry
the robust scorer with `tol_deg=0.5` (looser tolerance) or extract
peaks with a larger `min_distance_deg` to suppress noise peaks. Broad
peaks in real data require looser tolerance windows.

**Materials-Project authority.** When MP and a local CIF both return the
same (formula, space_group), `search_structures` keeps the MP entry and
drops the local one. Local CIFs without space-group metadata cannot
dedup against MP and appear as separate candidates — verify they aren't
duplicates of MP entries already in the candidate list.
