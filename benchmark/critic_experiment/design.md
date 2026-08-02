# Critic-validator experiment — design document

**Status:** draft, pre-implementation. Intended to be pressure-tested
*before* any code is written.

**Why this document exists.** The SciLink simulation stack is approaching
several architectural forks (standalone critic agents vs. pipeline-stage
hooks on a foundation agent; deterministic vs. LLM vs. multi-reviewer DAG
critique; per-skill RAG-grounded reviewers vs. prompt-only). The
multi-agent literature (Reflexion, Self-Refine, multi-agent debate,
MacNet) provides general evidence that reflective evaluators improve task
completion, but none of it measures the SciLink-specific setting where
**cheap deterministic verifiers already exist** (`pymatgen.Incar.check_params`,
`lmp -in <file> -echo log -nocite`, OUTCAR convergence parsers). That
shifts the cost-benefit and is the question our experiment answers.

The output is a result we can defend in the manuscript: not "we followed
the foundation-agent pattern" but "we measured X vs. Y on Z fixtures and
the data says A."

---

## 1. Research questions

**RQ1 — Mechanism (the load-bearing one).** For a critic role with a
strong engine-native deterministic verifier, does an LLM critique layer
add precision over deterministic-only? Does a multi-reviewer DAG
(MacNet-style) add precision over single-LLM?

**RQ2 — Shape.** Independently of mechanism, does packaging the critic
as a standalone agent class vs. a pipeline-stage hook inside a foundation
agent change accuracy, cost, or end-to-end success?

**RQ3 — Engine / verifier-shape interaction.** Does the answer to RQ1
depend on the engine? Pre-run INCAR validation uses pymatgen's static
analysis (`Incar.check_params`); LAMMPS pre-run validation uses the real
binary as a parse-only pass (`lmp -in <file> -echo log -nocite`). Both
are "strong" in different ways. v1 tests both engines directly so the
universal-vs-engine-specific question is answered by data, not deferred.

**RQ4 — Model-tier sensitivity.** Do the conclusions hold across
`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`? This is a
sensitivity check, not the primary analysis.

RQ1 and RQ3 are paper-load-bearing. RQ2 and RQ4 are supporting.

---

## 2. Pre-registered hypotheses

We write these down *before* running so the post-hoc story is honest.

**H1 (null, pre-registered):** For pre-run INCAR validation,
**det-only ≈ det+single-LLM** on F1. Justification: pymatgen's tag check
is comprehensive; "ISPN→ISPIN" is the canonical failure mode and pymatgen
catches it. The LLM's marginal value is on *physics* mistakes (ISPIN=1
when ISPIN=2 was needed), but those are rarer than tag typos.

  - **Falsifier:** det+single-LLM beats det-only by ≥5% F1 with bootstrap
    95% CI not crossing zero. Or det+single-LLM's repair rate
    (after-fix pymatgen pass-rate) ≥ det-only's by ≥10 percentage points.

**H2:** For roles with weak deterministic baselines (MD quality —
"did equilibration finish? is the RDF first peak reasonable?"), LLM
critique will help meaningfully. We don't run this experiment in v1, but
state the hypothesis to motivate the v2 extension.

**H3 (MacNet):** Multi-reviewer DAG (3 reviewers on orthogonal axes:
*syntax / physics / convergence-risk*, plus a synthesizer) beats
single-LLM by ≥5% F1, with the gap widening on harder fixtures
(LDAUL+MAGMOM-AFM systems vs. vanilla Si). Cost grows ~4×.

  - **Falsifier:** DAG ≈ single-LLM within noise, or DAG worse on F1.

**H4 (shape):** Standalone-agent vs. pipeline-hook is a wash on F1;
differs only on cost (standalone may invoke an extra LLM call to
*decide* whether to run the check).

  - **Falsifier:** F1 differs by ≥3% with CI not crossing zero.

If H1 is **not** falsified, we have a paper finding:
*"in domains with strong engine-native verifiers, LLM critique adds cost
without improving precision."* That result alone justifies the work.
If H3 holds and H1 is falsified, we have the inverse:
*"layered reflection scales the way the agent-collaboration literature
predicts even in scientific-software domains."* Either outcome ships.

---

## 3. Experimental design

### 3.1 Variants (cells of the matrix)

For the pre_run INCAR validator role, six implementation variants:

| ID  | Mechanism | Shape |
|-----|-----------|-------|
| A1  | Det only (pymatgen `Incar.check_params` + tag whitelist) | Standalone |
| A2  | Det only | Pipeline hook |
| B1  | Det + single-LLM (current `IncarValidatorAgent.validate_and_improve_incar`) | Standalone |
| B2  | Det + single-LLM | Pipeline hook |
| C1  | Det + multi-reviewer DAG (3 reviewers + synthesizer) | Standalone |
| C2  | Det + multi-reviewer DAG | Pipeline hook |

Plus a **no-critic control (X0):** `PeriodicDFTAgent.generate_inputs`
without any post-generation check. Measures the "what does the unaided
generator produce" baseline.

### 3.2 The multi-reviewer DAG (variant C)

Three parallel reviewers, each seeing the INCAR + system description but
with a different lens:

  - **Syntax reviewer.** "Are all tags real VASP tags? Any values
    syntactically malformed?" Backed by the pymatgen tag whitelist as
    additional context.
  - **Physics reviewer.** "Given the system description, are the
    parameter *choices* physically appropriate? (ISPIN for magnetic
    systems, +U for f-electrons, IDIPOL for asymmetric slabs.)" The
    current production literature-review prompt.
  - **Convergence-risk reviewer.** "Given these tags, will this calculation
    converge with reasonable wall-clock? Flag ENCUT/KPOINTS that will
    take days; flag IBRION/POTIM combinations that diverge."

A fourth **synthesizer** sees all three reviews and produces the final
issue list + suggested fixes. Synthesizer prompt explicitly resolves
contradictions (e.g., syntax-OK but physics-WRONG → physics wins).

Why this DAG shape: it picks three *orthogonal axes of failure*, not three
redundant reviewers. Redundant reviewers test ensemble-of-equals; this
tests whether failure-mode diversity (the MacNet "irregular topologies
beat regular ones" argument) shows up here.

### 3.3 Test article scope (v1)

**Primary:** pre_run validation across **two engines** — DFT (VASP
INCAR) and classical MD (LAMMPS input script). Same role, different
deterministic-verifier shapes. The cross-engine comparison directly
tests RQ3 (does the answer depend on engine / verifier shape).

  - **VASP INCAR pre_run.** Deterministic baseline:
    `pymatgen.Incar.check_params()` — static tag + value-range check.
    Production B variant exists today:
    `IncarValidatorAgent.validate_and_improve_incar`.
  - **LAMMPS input pre_run.** Deterministic baseline:
    `lmp -in <file> -echo log -nocite` (parse-only run; non-zero exit on
    syntax / fix / pair_style errors). Production B variant does **not**
    exist yet — needs authoring as part of v1. This is a known Phase 4
    gap on the broader plan; doing it inside the experiment unblocks
    LAMMPS coverage either way.

**Deferred to v2 (if v1 lands cleanly):**
  - VaspQualityAgent post-run (moderate det baseline)
  - MD quality critic (weak det baseline — most likely to show LLM value;
    independently tests RQ3 from the *weaker* end of the spectrum)
  - VaspUpdater runtime critic (weak det baseline = "retry as-is")

v1 is one critic role × 2 engines × 6 + 1 variants × N fixtures × M
repeats. Wider than originally scoped, but the cross-engine pair is
exactly the comparison that lets the paper say more than "we tested one
engine."

---

## 4. Test fixtures

### 4.1 Required ground-truth labels per fixture

Each INCAR fixture needs:
- **`true_issues`** — a list of (tag, severity, category) triples representing
  every real problem. `category ∈ {syntax_typo, malformed_value,
  wrong_choice_for_system, missing_required, redundant}`.
- **`generated_from_prompt`** — natural-language request that produced this
  INCAR. Needed because physics-correctness is prompt-relative.
- **`canonical_fix`** — the corrected INCAR (so we can measure "did the
  critic's suggested fix actually fix it" by comparing structure, not text).

### 4.2 Fixture authoring sources

**Per engine (VASP and LAMMPS), four sources:**

**Source 1 — extend the variability test set.**
  - *VASP*: extend `test_incar_variability.py` (3 prompts: Fe BCC, UO₂,
    Pt111+CO). Generate N=20 INCARs per prompt with unmodified
    `PeriodicDFTAgent`. Label by hand. Yields ~60 fixtures.
  - *LAMMPS*: parallel construction — pick 3 system prompts
    (water box, NaCl(aq), MoS₂ slab) and generate N=20 LAMMPS input
    scripts per prompt with `MDSimulationAgent.generate_simulation`.
    Hand-label. Yields ~60 fixtures.

**Source 2 — planted errors.** Author ~30 fixtures per engine by
single-mutation injection from known-good baselines.
  - *VASP*: 10 tag typos (`ISPN`, `ENCT`, `KSPCING`); 10 value errors
    (`ISPIN=1` for Fe; `ENCUT=200` too low; `LDAUL=2` wrong subshell);
    10 coupled errors (typo + value).
  - *LAMMPS*: 10 syntax typos (`pair_styl`, `velosity`, malformed `fix`
    arg counts); 10 physics-wrong (timestep too large for the ensemble,
    units mismatch with potential, `pair_style` wrong for the system's
    chemistry); 10 coupled errors.

**Source 3 — controls (no errors).** 20 known-good fixtures per engine
(verified by hand). False-positive rate measurement.

**Source 4 — real-world failures.**
  - *VASP*: mine `examples/breakage_benchmark_*/` for prior real
    failures with known correct fixes (~20 fixtures).
  - *LAMMPS*: mine prior LAMMPS run failures from
    `examples/mlip/` outputs and any preserved cluster-run logs
    (~10–20 fixtures; if fewer than 10 found, supplement from LAMMPS
    mailing-list archived bug reports — public corpus).

Held-out from all prompt tuning; critic-implementer never sees the
labels before the run.

**Totals per engine:** ~130 (VASP) + ~130 (LAMMPS) ≈ 260 labeled
fixtures.

### 4.3 Sample size justification

For pairwise F1 comparison with α=0.05, power=0.8, detecting a 5% F1
difference, we need ~60 paired fixtures (lower-bound, McNemar's test).
130 per engine gives us headroom for stratification by difficulty and
per-prompt breakouts, and lets us test cross-engine differences with the
same statistical power as within-engine ones. 5 repeats per
fixture-variant combination handle LLM nondeterminism.

Total LLM-task calls: 260 × 7 variants × 5 repeats = 9100
(only ~6500 hit the LLM, since 1 of the 7 variants is no-critic and
some det-only paths don't call LLM either). At sonnet pricing this is
roughly $30–80 depending on prompt size; at opus, ~$300–600. Sensitivity
sweep adds ~30% on top of those baseline numbers but only for top-2
variants.

### 4.4 Who labels the fixtures

**Critical:** the person labeling true_issues should NOT be the person
implementing the multi-reviewer DAG. Mitigation against fixture
overfitting:
  - Sarah authors labels for Sources 1+2+3 (physics expertise) across
    both engines.
  - Source 4 uses pre-existing labels from prior work — held-out from any
    optimization.
  - DAG reviewer prompts are written from the *production skill
    documentation* (`scilink/skills/periodic_dft/vasp/vasp.md` and
    `scilink/skills/molecular_dynamics/lammps/lammps.md`), not from the
    fixture set.

---

## 5. Scoring rubric

### 5.1 Per-fixture, per-variant primary metrics

For each (fixture, variant) pair:

  - **Recall** = (true issues caught) / (true issues present). On
    fixtures with multiple issues, partial credit.
  - **Precision** = (true issues caught) / (issues reported). False
    positives drag this down.
  - **F1** = harmonic mean.
  - **Repair rate.** Apply the variant's suggested fix. Does the resulting
    INCAR pass pymatgen + reach a canonical-correct physics tuple?
    Binary per fixture.
  - **Cost.** LLM input + output tokens, summed across all calls the
    variant made for this fixture.
  - **Latency.** Wall-clock seconds (median over the 5 repeats).

### 5.2 Per-variant aggregated metrics

  - Mean recall, precision, F1, repair rate — bootstrap 95% CI (10k
    resamples).
  - Cost per fixture (mean and median).
  - **Stability:** F1 standard deviation across the 5 repeats per fixture,
    averaged across fixtures. Lower = more reproducible.

### 5.3 Pairwise comparison

For each variant pair (A1 vs. B1, B1 vs. C1, A1 vs. A2, B1 vs. B2,
C1 vs. C2):
  - Paired difference in F1 per fixture.
  - Wilcoxon signed-rank test on the paired differences.
  - Bonferroni correction across the 5 pre-registered comparisons:
    significance threshold α/5 = 0.01.
  - **Effect size reported alongside p-value.** Cohen's d on F1 deltas.
    We care more about effect size than significance — a 1% F1 difference
    that is statistically significant on 4550 trials still doesn't change
    the engineering decision.

### 5.4 Cost-aware composite

The architectural decision balances accuracy and cost. Define:

  - **Accuracy-cost ratio:** F1 / log(tokens). Logged because the
    accuracy ceiling is 1.0 but token counts vary by 10×+ across variants.
  - Plot variants on (cost, F1) scatter; the Pareto frontier is the
    interesting set. Anything dominated is excluded regardless of
    significance.

---

## 6. Threats to validity (and mitigations)

| Threat | Mitigation |
|---|---|
| Fixture authoring biases toward typos the DAG implementer knows | Labels done by Sarah, not implementer; Source 4 held-out completely |
| LLM prompt tuning leaks the test set | Use production prompts (the ones in `IncarLiteratureAgent` / `INCAR_VALIDATION_INSTRUCTIONS`) for B; DAG prompts derived from VASP skill markdown, frozen before any benchmark run |
| Model choice flips the conclusion | Primary run on one model (proposal: `claude-sonnet-4-6` — cheapest credible); sensitivity sweep with opus + haiku on top 2 variants only |
| Reviewer-DAG voting threshold cherry-picked | Pre-register the synthesizer prompt (no `keep_if ≥2 of 3` post-hoc tuning) |
| Multiple-comparison data dredging | 5 comparisons pre-registered, Bonferroni applied, all results reported |
| Engine-native baseline strength | Use production `check_incar_syntax` for det layer; don't tune deterministic checker for the experiment |
| Repair-rate measurement gameable by ensemble of fixes | Apply only the highest-confidence single fix per issue; tied fixes broken by deterministic tie-break (lex order on tag name) |
| Cost-vs-accuracy trade is unfair if variants disagree on cost-quality knob | Each variant runs at its production default settings; no per-variant tuning |

---

## 7. Decision criteria (pre-registered)

What result drives what action. Written *before* the data, to avoid
post-hoc rationalization.

**Decision 1 (mechanism).** Based on F1 deltas between A, B, C at the
pipeline-hook shape (A2/B2/C2 — the hook is the upstream-mandated pattern,
so judge mechanism using it):

  - **det-only (A2) within 3% F1 of B2 (det+LLM):** ship det-only,
    drop LLM critique for pre_run INCAR. Save cost. Paper claim:
    "engine-native verifiers dominate for this role."
  - **B2 beats A2 by ≥5% F1, C2 ≈ B2:** ship det+single-LLM. Paper
    claim: "single-LLM critique helps; topology doesn't scale here."
  - **C2 beats B2 by ≥5% F1 AND cost(C2) ≤ 4×cost(B2):** ship DAG
    pattern. Paper claim: "multi-reviewer DAG transfers to scientific
    domains within cost bounds."
  - **C2 beats B2 by ≥5% F1 BUT cost > 4×:** ship single-LLM; flag DAG
    as a "max-accuracy mode" toggle.
  - **Mixed (A2 ≈ B2 but C2 > both):** ship DAG. The signal that
    layered reflection matters more than one-shot LLM is itself a paper
    finding.

**Decision 2 (shape).** Based on F1 deltas standalone vs. hook at the
winning mechanism:

  - **Difference within 3% F1 AND cost difference <20%:** pick the
    engineering-simpler pattern (probably pipeline hook, aligned with
    upstream foundation-agent direction).
  - **Standalone wins by ≥5% F1:** keep standalone. **This pushes back
    on Maxim's "no new subagents" mandate with data**, and the paper
    documents the disagreement honestly.
  - **Hook wins by ≥5% F1:** unambiguously migrate to hook.

**Decision 3 (scope expansion).** If v1 produces a clean decisive result
(any of Decision 1's branches except "mixed"), proceed to v2 with the
same protocol applied to MD quality critic (where H2 predicts a different
answer due to weaker det baseline). If v1 is muddy, stop and write up
"inconclusive — re-design needed" rather than expand scope.

---

## 8. Implementation plan

### 8.1 Code structure

```
benchmark/critic_experiment/
├── design.md                  ← this file
├── README.md                  ← quick-start + how to add variants
├── fixtures/
│   ├── incar/                 ← VASP fixtures
│   │   ├── from_generator/    ← Source 1 (60)
│   │   ├── planted/           ← Source 2 (30)
│   │   ├── controls/          ← Source 3 (20)
│   │   └── breakage/          ← Source 4 (~20 reused)
│   ├── lammps/                ← LAMMPS fixtures
│   │   ├── from_generator/    ← Source 1 (60)
│   │   ├── planted/           ← Source 2 (30)
│   │   ├── controls/          ← Source 3 (20)
│   │   └── breakage/          ← Source 4 (~10–20)
│   └── labels.jsonl           ← unified label store (engine-tagged)
├── variants/
│   ├── _base.py               ← shared scaffolding (apply-fix, score-fixture)
│   ├── _engine_protocol.py    ← per-engine plug-in (check_syntax, apply_fix, ...)
│   ├── engines/
│   │   ├── vasp.py            ← VASP-specific det baseline + B prompt + DAG prompts
│   │   └── lammps.py          ← LAMMPS-specific (lmp -nocite subprocess + B + DAG)
│   ├── a_det_only.py          ← mechanism-only; reads engine from fixture
│   ├── b_det_single_llm.py
│   ├── c_det_dag.py
│   └── x_no_critic.py
├── shapes/
│   ├── standalone.py
│   └── hook.py
├── run_experiment.py          ← drives matrix: fixtures × variants × shapes × repeats
├── analyze.py                 ← bootstrap CIs + Wilcoxon + cross-engine breakouts + plots
└── outputs/                   ← (gitignored) results + manifests
```

Variants are engine-agnostic at the top level; engine-specific behavior
(deterministic check, LLM prompts, fix application) plugs in through
`_engine_protocol.py`. This keeps the DAG and shape adapters from being
duplicated per engine.

Variant code is small (~50–150 lines each). Engine plug-ins are
~100–200 lines each. Most of the work is fixture labeling and the
analyzer's cross-engine breakouts.

### 8.2 Sequencing (rough days)

1. **Fixture authoring (4–5 days, mostly Sarah).** Both engines, all 4
   sources. VASP first (existing harness), LAMMPS second (parallel
   construction with `MDSimulationAgent.generate_simulation`).
2. **Engine plug-ins (2 days, me).** VASP plug-in wraps existing
   `check_incar_syntax` + `IncarValidatorAgent` prompt + DAG. LAMMPS
   plug-in implements `lmp -in -echo log -nocite` subprocess + a new
   LAMMPS-side B prompt + DAG. **This is the load-bearing work that
   doesn't exist on disk yet.**
3. **Variant implementation (2 days).** A/B/C/X variants on top of the
   engine plug-in protocol. Most logic is engine-agnostic.
4. **Shape wrappers (1.5 days).** Standalone vs. hook adapters. Hook
   needs minimal foundation-agent-pipeline stub on both
   `PeriodicDFTAgent` and `MDSimulationAgent`.
5. **`run_experiment.py` + manifest format (1 day).** Reuses
   `benchmark/_score.py` patterns.
6. **Analyzer (1.5 days).** Bootstrap + Wilcoxon + the (cost, F1) Pareto
   scatter for the paper figure + cross-engine breakouts for RQ3.
7. **Run (3–4 days wall-clock).** Primary on sonnet, then sensitivity on
   top 2 variants × {opus, haiku}. Most is LLM wait time.
8. **Write-up (2 days).** Paper-ready figure + table + decision narrative.

Total ~17–20 days of active work, **~4 weeks calendar**.

### 8.3 What's explicitly OUT of scope for v1

  - Non-pre_run critics (quality, runtime). Deferred to v2, conditional
    on v1 landing.
  - Engines beyond VASP + LAMMPS. Mol-DFT critic doesn't exist; QE skill
    landed but no pre_run critic for QE; both deferred to v2.
  - Sensitivity to the temperature / sampling parameters. Use production
    defaults.
  - Tuning the deterministic baselines (e.g., adding tag-whitelist
    augmentations for VASP; adding wrapper checks beyond
    `lmp -in -echo log -nocite` for LAMMPS). Use the production
    implementations.
  - Verifying that fixtures are physically realistic across the full
    range of usage. Stay in the prompt sets already covered by the
    variability harnesses (VASP: Fe / UO₂ / Pt111; LAMMPS: water /
    NaCl(aq) / MoS₂).

---

## 9. Paper integration

### 9.1 The figure(s)

**Primary:** two-panel (cost, F1) scatter — left panel VASP, right
panel LAMMPS — 6 variant points + X0 baseline per panel, Pareto
frontier highlighted, bootstrap 95% CI error bars. Side-by-side
comparison makes the cross-engine answer visible at a glance.

**Secondary:** cross-engine delta plot — for each variant, plot
(F1_VASP − F1_LAMMPS) with CIs. Bars whose CI cross zero indicate
engine-invariant behavior; bars away from zero are engine-specific.

### 9.2 The table

Per (variant × engine): F1 ± CI, precision, recall, repair rate, cost,
latency, stability. 14 rows × 7 columns (or grouped 7 rows with
columns × {VASP, LAMMPS}).

### 9.3 The claim

Five possible framings prewritten (more outcomes because the
cross-engine axis adds branches):

  - **If H1 holds on both engines (det dominates universally):**
    "Across both engines tested, deterministic engine-native verifiers
    match or exceed LLM-based critique. Multi-agent reflection patterns
    from the general-agent literature transfer with caveats — cheap
    domain-specific verifiers should be exhausted before LLM critics
    are added, regardless of engine."
  - **If H1 holds on VASP but fails on LAMMPS (verifier-shape
    interaction):** "The answer depends on the deterministic baseline's
    shape. Static analyzers (pymatgen) are competitive with LLM critics
    for INCAR validation; subprocess parse-only checks (`lmp -nocite`)
    leave room for LLM critique on LAMMPS scripts. The right
    architectural choice is engine-specific."
  - **If H3 holds on both (DAG wins universally):**
    "Multi-reviewer DAG critique transfers from general-agent
    benchmarks (MacNet et al.) to scientific software workflows across
    engines, with X% F1 gain at Y× cost over single-LLM critique."
  - **If H3 holds on the engine with the weaker deterministic baseline
    only:** "Multi-reviewer reflection helps most where engine-native
    verifiers leave gaps; the gain scales inversely with deterministic
    baseline coverage. This refines the MacNet finding to the
    scientific-software setting."
  - **If shape (RQ2) flips and standalone wins on either engine:**
    "Empirically, separate critic agent classes outperform same-logic
    pipeline-stage hooks by X% F1. We attribute this to [whatever the
    data shows]; the foundation-agent industry pattern may not be
    universally optimal in domains with structured verification stages."

### 9.4 Honest framing

  - State explicitly: pre_run only; quality and runtime critics
    deferred to v2.
  - State explicitly: this is one model family (Claude). Cross-family
    generalization is future work.
  - State explicitly: two engines tested; mol-DFT (no SciLink agent
    today) and QE (skill landed, no pre_run critic) are out of scope.
  - State explicitly: deterministic baselines are pymatgen's static
    analysis (VASP) and the LAMMPS binary's parse-only mode. Other
    engines' baselines may differ in strength and shape.

---

## 10. Confirmed parameters (sign-off 2026-05-29)

  - **Scope:** pre_run role across **two engines** (VASP INCAR + LAMMPS
    input), v1. The cross-engine axis tests RQ3 directly. Quality and
    runtime critics, mol-DFT, QE all deferred to v2.
  - **Fixture-label authoring split:** Sarah on Sources 1+2+3 (both
    engines); Source 4 held out from any prompt tuning.
  - **Primary model:** `claude-sonnet-4-6`. Sensitivity sweep on
    `claude-opus-4-7` + `claude-haiku-4-5` on top-2 variants only.
  - **Pre-registered hypotheses and decision criteria:** Sections 2 + 7
    of this document. Contracts once data lands.

Phase 2 of the broader plan = "execute this experiment." The rest of
the refactor phases are conditional on its outcome.
