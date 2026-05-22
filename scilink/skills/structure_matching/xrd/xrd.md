---
description: XRD structure matching — query crystal-structure databases (Materials Project, local CIF), simulate kinematic patterns, score by cross-correlation (fast) and Hanawalt / MIP peak-matching (robust).
quality_gate:
  metric: figure_of_merit
  accept_threshold: 0.70
  hard_reject_threshold: 0.40
  direction: higher_is_better
live_tick:
  enabled: true
  tick_fn: scilink.skills.structure_matching.xrd.live_tick:xrd_live_tick
  data_type: spectrum_1d
  wavelength: CuKa
  trigger_overrides:
    heartbeat_sec: 45
    confidence_threshold: 0.7
---
# XRD Structure Matching Skill

## overview

Identify a crystalline phase from an experimental X-ray diffraction (XRD)
pattern by matching against database structures. The skill ships five
tools the analysis script chains together:

- `search_structures` — query Materials Project and / or a local CIF
  directory for candidate structures (chemistry + symmetry filters).
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

Install dependencies: `pip install scilink[structure-matching]` (pymatgen
with the XRD analysis module, mp-api, pulp).

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
  for two distinct elemental phases, etc.). Use the **list-of-lists**
  chemistry form to get separate candidate lists per phase:
  `chemistry=[["Si"], ["Ge"]]` — NOT `chemistry=["Si", "Ge"]` which would
  ask the DB for Si-Ge *binary compounds* instead of Si and Ge
  separately. For step 2 use **`algorithm='mip'` from the start** (not
  Hanawalt) so the assignment problem can match peaks across both
  phases simultaneously, and report identification per phase. The
  R-factor in this case is dominated by whichever phase the LLM picks
  first as "best"; per-phase FOMs are more informative than a single
  overall verdict.

- **Post-fit pattern (no chemistry hypothesis)** — no hint at all. Run
  `extract_peaks` first to estimate the dominant peak positions, infer
  an approximate lattice from the strongest peak via Bragg's law for a
  guessed crystal system, then make a **single** `search_structures`
  call with a list-of-lists chemistry hypothesis (e.g.
  `chemistry=[["Si"], ["C"], ["Ge"], ["Ti","O"]]`). The tool dispatches
  one DB query per hypothesis, dedupes, and merges results. **Never
  loop over single-chemistry `search_structures` calls** — that path
  has historically broken on consolidation, and the tool is built to
  handle multiple hypotheses in one invocation.

**Recognizing the multi-phase trigger.** Any of these phrasings in
`system_info` / notes mean "use the multi-phase mode": "mixture",
"multi-phase", "two-phase", "binary mixture" (as opposed to "binary
compound"), "co-existing phases", or `chemistry_hint` containing two
elements with NO compound name (e.g. `["Si", "Ge"]` with note
"suspected mixture" — the elements are distinct phases, not a
compound). When in doubt, run multi-phase MIP — it's strictly more
informative than Hanawalt for single-phase data too (just slower).

**Bounding the candidate count.** Always set `search_structures(query={
"top_n": N, ...})` with N in [3, 10]. More than 10 candidates per
spectrum bloats simulation cost without adding identification certainty
— if the answer isn't in the top 5, the chemistry hypothesis is usually
wrong, not the candidate count.

**Wavelength selection.** Default CuKa unless the experiment metadata
says otherwise. MoKa is common for high-2θ work. A wavelength mismatch
gives uniformly bad correlations / FOMs for all candidates with a
characteristic shift in peak positions — suspect that first when every
candidate scores reject.

## analysis

**CRITICAL: structure-matching workflow.** The per-item analysis script
must follow this exact sequence:

1. Load experimental 2-theta + intensity arrays.
2. Call `search_structures` once with `top_n` between 3 and 10.
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
    "fit_quality": {
        "figure_of_merit": best["figure_of_merit"] if best else None,
        "verdict": best["verdict"] if best else "no_candidates",
    },
}))
```

**Background handling.** Both scorers default to subtracting the
experimental minimum as a flat offset. For patterns with significant
continuous background (amorphous halo, fluorescence), fit and subtract a
low-order polynomial background BEFORE calling the scorers and pass
`background="none"`.

**Wavelength consistency.** Pass the same `wavelength` to every
`simulate_xrd_pattern` call. Pull from experiment metadata when present;
otherwise default CuKa.

**Peak broadening.** Both scorers use a Lorentzian with FWHM=0.15° by
default. For low-resolution or strongly broadened experimental patterns
(nanocrystalline samples), bump `fwhm` to 0.3-0.5° so peaks overlap as
they do in the data.

**When the robust tier disagrees with the fast tier.** Trust the robust
tier — it factors out background, scale, and intensity-ratio effects
that the fast tier folds into the correlation. The fast tier is a
triage step; the robust tier is the identification.

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
suspected multi-phase mixtures, switch to MIP and increase
`max_exp_peaks` so the assignment problem has enough peaks to allocate
across phases. v1 of this skill does not include quantitative phase
fraction analysis (Rietveld refinement is the next step) — note this in
the report.

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
