# Shared QC Engine, Synthesis Re-entry, and Hyperspectral Expansion — Combined Design Note

Status: DRAFT for review — no code has been changed.
Scope: `scilink/agents/exp_agents/` (curve fitting, image analysis, hyperspectral).
Related: issue #322 (pluggable synthesis re-entry), issue #323 / `feature_conditioned_literature_plan.md`
(feature-conditioned interpretation literature), the foundation-agent definition in `CLAUDE.md`
("self-refinement is one shape", elements 1 and 5).

---

## 0. Prime directive

Two constraints govern everything below, in priority order:

1. **Curve-fitting and image-analysis behavior is FROZEN during extraction.** Same prompts
   (byte-identical), same LLM call count and order on default paths, same result-dict keys,
   same thresholds, same knobs. Every extraction phase must be provably a no-op for these
   two agents (§8 defines "provably").
2. **Hyperspectral gains capabilities it lacks today** — verification history / T=2 staging,
   literature plumbing, feature surfacing, and (via the shared base) automatic inheritance of
   the #322/#323 machinery. All HS changes are additive: new result keys, new optional
   `analyze()` params, no change to existing outputs.

Corollary: where CF and IA have *diverged* (rate-based escalation, plan best-of-N,
`parallel_workers`, fast-accept semantics), the extraction **represents both behaviors via
hooks** — it does not harmonize them. Harmonization is a separate, explicit, opt-in decision
per divergence, made after extraction (§9).

---

## 1. The architecture in one picture

All three agents implement the same loop — *generate code → execute → verify → escalate
structural freedom on failure → accept/salvage* — at the per-item level, and the same
*verify → critique → re-enter* loop again at the synthesis level. Layered:

```
Layer 3 — SYNTHESIS (once per run)
    trend codegen + cross-item interpretation over cached per-item records
    #322 re-entry targets this layer; #323 Channel B feeds it literature payloads
    HS's RunSelfReflection/ApplyReflectionUpdates pair is the existing prototype

Layer 2 — SERIES ORCHESTRATION (once per run, drives N items)
    scout → plan+lock at anchor → fan out → outlier flag → adaptive refit → consistency
    duplicated across CF/IA (structural twins); absent in HS

Layer 1 — PER-ITEM QC ENGINE (once per item)
    generate → execute → verify(+history) → anneal → gate → human/judge fallback
    near-copy duplicated CF/IA; same *shape* in HS's RunDynamicAnalysisController

Layer 0 — SHARED TYPES (the spine)
    QualityGate (unified) · VerificationRecord/quality_history · CritiquePayload
    per-item result record ("series_results entry")
```

The types in Layer 0 are what every other piece consumes:

| consumer | uses |
|---|---|
| Layer-1 engine | gate decisions, verdict schema, history record |
| Layer-2 outlier/refit/consistency | gate metric, result records, consistency→CritiquePayload |
| Layer-3 re-entry (#322) | cached result records + CritiquePayload (any producer) |
| #323 `refine_interpretation` | surfaced features from the result record; produces a literature CritiquePayload |
| T=2 staging / skill graduation | `quality_history.verification_iterations[].annealing_level` |
| best-of-N escalation gates | gate fast-accept margins, `_produced_at_level` |

---

## 2. Layer 0 — shared types

### 2.1 Unified `QualityGate` (extend `quality_gate.py`, no semantic change for curve)

Today: curve resolves a `QualityGate` (`quality_gate.py:33-195`) but its **driver** accept
checks are still raw floats (`best_r2 >= self.r2_threshold`,
`curve_fitting_controllers.py:4611,4732`); the gate object only frames the verifier and the
non-R² metric paths. Image has **no gate object** — a bare
`self.quality_threshold = 0.7` compared inline (`image_analysis_controllers.py:1934,2110,
3397,3648,3816,4397,4714`).

Changes:

- **New field `value_source: Literal["result","verdict"] = "result"`.**
  `"result"` = extract the metric from the item's `fit_quality` dict (curve, deterministic).
  `"verdict"` = the metric is assigned by the LLM verifier's returned JSON (image
  `quality_score`). `extract()` gains an optional `verdict` argument consulted only when
  `value_source == "verdict"`.
- **New module constant** `IMAGE_SCORE_DEFAULT = QualityGate(metric="quality_score",
  accept_threshold=0.7, hard_reject_threshold=0.7, direction="higher_is_better",
  physical_review=True, best_value=1.0, value_source="verdict")`.
  `hard_reject == accept` ⇒ empty soft band, which matches image's current plain-threshold
  semantics exactly.
- **Route existing raw comparisons through `gate.is_accept()` / `clears_by_fast_margin()`**
  in both drivers. Equivalence is arithmetic (`>=` on the same floats) and is pinned by
  tests, including the mutation path where curve's human feedback `adjust_threshold`
  action rewrites the threshold mid-run (`curve_fitting_controllers.py:4652`) — the gate a
  driver consults must observe that mutation (`with_accept_threshold()` re-derivation, not a
  frozen snapshot taken at pipeline build).
- **Composite accept rule for HS** (used only by HS, later phase): a small
  `SetAcceptRule(min_pass_fraction=0.5, required_names=[...])` helper evaluated over per-map
  verdicts — models HS's `SUCCESS_THRESHOLD` + `required_outputs` gate
  (`hyperspectral_controllers.py:1975,2358-2404`) without touching the scalar gate.

Explicitly **not** in the gate: curve's Option B (`best_ever_rejected` forcing a judge even
when R² clears, `:4611`) — that is verifier-arbitration policy, stays a curve engine-policy
flag. Also not in the gate: image's CO_PILOT accept-path approval (`:3826-3841`).

### 2.2 `VerificationRecord` — one builder for `quality_history`

Today: `_build_quality_history` twins (curve `:6072`, image `:4698`) and
`build_verification_prompt_with_history` twins (curve `:588`, image `:178`) are ~90%
isomorphic; keys differ (`final_r2`/`r_squared`/`model` vs `final_score`/`score`/
`result_type`). HS records **nothing** across attempts — which is exactly why HS is locked
out of T=2 staging (`_maybe_stage_t2_solutions` keys on
`verification_iterations[].annealing_level`), `starting_annealing_level` re-runs, and
best-of-N struggle gates.

Change: one shared builder in a new `exp_agents/_verification_record.py`:

```python
METRIC_KEYMAP = {
    "curve_fitting":  {"final": "final_r2",   "iter": "r_squared", "extra_iter": "model"},
    "image_analysis": {"final": "final_score", "iter": "score",    "extra_iter": "result_type"},
    "hyperspectral":  {"final": "final_verdict", "iter": "passed_fraction", "extra_iter": None},
}

def build_quality_history(iterations, *, threshold, approved, script_errors,
                          judge_reasoning, keymap, extras=None) -> dict: ...
def build_verification_prompt_history(iterations, *, keymap) -> list: ...
```

**Backcompat rule:** for CF/IA the builder must emit **exactly today's dicts** — same keys,
same nesting, same optional fields (verified by golden fixtures, §7). Curve-only extras
(`alternative_models[]`, `approved_by="verifier"` overwrite at `:4599`) and image-only extras
(`score_explanation`) ride in `extras`. A typed dataclass wrapper can come later; phase 1 is
dict-shaped on purpose.

Known asymmetry preserved (not "fixed"): curve does **not** build a `quality_history` on the
locked-script reuse path while image does (`image:3435` vs curve reuse path `:4082-4142`).
Documented, left as-is.

### 2.3 `CritiquePayload` — the one re-entry currency

New small type (dataclass in `exp_agents/_critique.py`), the payload #322 asks for, also
used internally by the engine and layer 2:

```python
@dataclass(frozen=True)
class CritiquePayload:
    source: Literal["verifier", "human", "consistency", "literature", "orchestrator"]
    critique: str                      # free-text critique / guidance
    hints: dict | None = None          # structured priors (expected model, param bounds,
                                       #   known transition, required outputs, ...)
    target: Literal["synthesis"] | int = "synthesis"   # synthesis stage or unit index
    provenance: dict | None = None     # e.g. literature files, vote counts, refit peer info
```

Producers and consumers, current and planned:

| producer | consumer | exists today as |
|---|---|---|
| verifier | layer-1 refit prompt | `refine_from_issues` (curve `:4409`) / verification feedback (image `:3735`) |
| human | layer-1 poor-result fallback | `_get_human_feedback_for_poor_fit/_quality` |
| consistency pass | layer-2 re-refit | peer-evidence prompt in `AdaptiveRefitController` |
| HS reflection critic | layer-3 editor | `reflection_result` → `ApplyReflectionUpdatesController` |
| human (#322 primary) | layer-3 re-entry | — (new) |
| literature (#323) | layer-3 re-entry | — (new) |
| literature prior (future, #322 §3) | layer-1 unit refit | — (future; lands in `_build_refit_state`) |

During extraction the existing producers are *wrapped* into this type at the call boundary;
the strings that reach prompts are unchanged.

### 2.4 Per-item result record

No new class in phase 1 — the `series_results` entry dicts stay as they are (keys per
modality; see the stage map in §3). Two **additive** normalizations, needed by #323:

- image: hoist `extracted_features` of the anchor/series into a top-level result field
  (currently nested per `series_results` entry) — additive key on the compiled result;
- hyperspectral: lift `custom_analysis_metadata_list` (`hyperspectral_controllers.py:1447`,
  written at `:2523`) into the compiled result as `extracted_features` — additive key.

---

## 3. Layer 1 — `CodegenQCEngine` (CF/IA extraction; HS adoption later)

### 3.1 What is pure orchestration (moves into the engine)

Per the stage-by-stage mapping (drivers: curve `_fit_with_quality_control` `:4034-4765`,
image `_execute_and_verify` `:3322-4021`):

- MAX_ATTEMPTS loop: generate → canonical-input guard → conformance → sandbox execute →
  correct (curve `:3095-3170` / image `:2491-2585`)
- verification `for/else` loop shell + history accumulation + final-verify-on-exhaustion
- annealing state machine (patience, iteration floor, escalate-into-hot script drop,
  `_CONSTRAINT_ANNEALING_SCHEDULE` / `_SKILL_STRICTNESS_SCHEDULE`)
- best/high-water tracking, locked-script reuse fast-path skeleton, anchor gating
- human-feedback gating triple (`enable_human_feedback and _is_anchor and not
  _suppress_human_feedback`), judge-fallback shell
- `quality_history` construction, `_stamp_hot_deviation`

### 3.2 What stays behind hooks (modality-specific, verbatim moves)

| hook | curve implementation | image implementation |
|---|---|---|
| `pre_verify_diagnostics` | residual diagnostics + zoom panels + fit.npy realign + upward R² recompute (`:3196-3256`) | — (none) |
| `extra_escalation_trigger` | rate-based trigger (`:4483`) | — (patience+floor only) |
| `promotion_policy` | 3-way strict-improve / `R2_FLOOR` reject / physics-deferral (`:4446-4476`) | plain high-water on score (`:3546-3550`) |
| `verifier_bypass` | `physical_review=False` skill-workflow bypass (`:3486-3554`); `max_verification_iterations<=0` shortcut (`:4173`) | — |
| `script_postprocess` | — | `_sanitize_script` (`:2538`) |
| `initial_score_policy` | trust numeric R² immediately | provisional 0.0 until verifier scores |
| `operand_staging` | extra data columns (`:3087`) | — |
| `human_actions` | `adjust_threshold` + `retry` | `retry` only |
| misc | — | `_TOOL_CONSTRAINT_SCHEDULE`, `DUMP_ITER_VIZ`, `null_decline` handling, CO_PILOT accept approval |

### 3.3 Interface sketch

```python
class EscalationPolicy:
    schedule: tuple[str, ...]            # constraint prose per level
    strictness_schedule: tuple[str, ...] # skill-recipe strictness per level
    patience: int = 2
    floor_fn: Callable[[int, int], int]  # (iteration, n_levels) -> min level
    rate_trigger: Callable | None = None # curve-only add-on; None = disabled

class CodegenQCEngine:
    def __init__(self, *, generate_fn, execute_fn, verify_fn, correct_fn,
                 conformance_fn, gate: QualityGate, escalation: EscalationPolicy,
                 keymap: dict, hooks: EngineHooks, human_feedback=None,
                 judge_fn=None, logger=None): ...
    def run_item(self, item_ctx: dict, state: dict) -> dict   # today's series_results entry
```

`execute_fn` is a callable — subprocess `ScriptExecutor` for CF/IA, **in-process
`exec()`+`ExecutionTimeout` for HS** (deliberate perf choice for 100MB cubes,
`hyperspectral_controllers.py:1965-1972`; the engine never assumes a backend).

### 3.4 Extraction method: move, don't rewrite

Each stage is lifted **verbatim** from one of the two copies with the state keys and prompt
templates parameterized; where the copies differ, the difference becomes a hook whose two
implementations are the two existing bodies, moved unchanged. The two controllers become
thin adapters that build their engine with today's exact prompts/schedules/thresholds.
Golden tests (§7) pin equivalence.

---

## 4. Layer 3 — synthesis re-entry (#322) and literature mounting (#323)

### 4.1 Mechanism: generalize the HS reflection pair

`RunSelfReflectionController` + `ApplyReflectionUpdatesController`
(`hyperspectral_controllers.py:2712-2845`) already implement critique→re-enter at the
synthesis stage and read only `result_json` + `analysis_images` — they are modality-agnostic
today. Plan:

1. Move both to `base_controllers.py` unchanged (HS pipeline imports move; zero behavior
   change for HS — pinned by its existing pipeline tests).
2. Generalize the editor into `SynthesisReEntryController`: same body, but the critique
   comes from a `CritiquePayload` instead of only `state["reflection_result"]`. The
   reflection critic becomes one producer (`source="verifier"`); a human prompt is a second
   (`source="human"`, #322's primary target); #323's literature result is a third.
3. **Re-entry over cached records** (#322's cost win): a re-entry entry point that loads the
   persisted per-item results (`series_fit_results.json`, curve `:2490`; image
   `saved_arrays`/features), injects the payload, and re-runs only the layer-3 stage
   (trend codegen + interpretation). Tier A (text+features in, revised text out) first —
   per #323's settled decision that Tier A is the load-bearing, generalizable tier; Tier B
   full re-synthesis deferred.
4. Storage: append-only `interpretation_revisions` on the analysis record (never overwrite
   `detailed_analysis`) — adopt #323's storage decision for **all** producers, human
   included, so revisions from any source are auditable and reversible.

**CF/IA freeze compliance:** the re-entry controller is *not inserted* into the CF/IA
default pipelines. It is reachable only via (a) an explicit orchestrator tool call
(`refine_interpretation`, or a human "revise the interpretation: …" request), or (b) an
opt-in constructor flag for the autonomous critic (default OFF for CF/IA). HS keeps its
current default (reflection ON in the synthesis pipeline), unchanged.

### 4.2 #323 mounting

Per the issue's settled decisions (D1–D3) and build order (curve → hyperspectral → image):

- Orchestrator-level `refine_interpretation(analysis_id, focus=None)` tool: read record →
  per-modality "features → query" micro-call → literature search (existing
  `FittingModelLiteratureAgent` backend) → build `CritiquePayload(source="literature",
  provenance={files})` → call the layer-3 re-entry → append revision.
- The **one integration rule this note adds**: #323's "revise interpretation" step MUST go
  through the §4.1 re-entry controller — no fourth parallel "revise the text" implementation
  (three already exist: HS editor, `_refine_analysis_with_feedback` base_agent.py:1413, the
  feedback mixin).
- Prereqs satisfied by §2.4 feature surfacing. ID-mode gating (D2) is #323's own scope,
  unaffected here.

---

## 5. Layer 2 — series orchestration (extraction deferred; design constraint now)

Extraction of the generic series layer (scout → lock → fan-out → outlier → adaptive refit →
consistency → trend) happens **after** the engine, since it is expressed against the engine
interface. Two things are decided now so earlier phases don't paint us into a corner:

- The consistency pass emits its peer-evidence as `CritiquePayload(source="consistency",
  target=<unit>)` — same currency as everything else; `_build_refit_state`
  (curve `:6311` / image `:5811`) is the single injection point, which also pre-plumbs
  #322's future "human/literature × single-unit refit" cells.
- CF↔IA layer-2 divergences (`parallel_workers` curve-only; plan best-of-N image-only)
  are **preserved** through extraction and offered cross-modality only as later opt-ins.

---

## 6. Hyperspectral expansion (the additive half)

Ordered by value/risk; each item is independently shippable and purely additive.

**HS-1. Verification record + T=2 staging (highest leverage).**
Inside `RunDynamicAnalysisController`'s existing retry loop, record per-attempt entries via
the §2.2 builder (annealing level = the `_codegen_retry_feedback` failure stage 0/1/2,
`hyperspectral_controllers.py:266-310`; issues = QC critiques; approved = task_success).
Attach `quality_history` to the compiled result (additive key). Then add
`_maybe_stage_t2_solutions` to `HyperspectralAnalysisAgent` mirroring the curve/image hooks
(gate: approved + reached hottest retry stage + novel). **This brings hyperspectral into the
skill-memory flywheel (staging → review-gated upgrade/consolidate) for the first time.**
No behavior change to the loop itself.

**HS-2. Literature plumbing.**
- Fix the silent drop: add `literature_file: Optional[str] = None` to
  `HyperspectralAnalysisAgent.analyze()` (today swallowed by `**kwargs`,
  `hyperspectral_analysis_agent.py:189`) and thread `literature_context` into the planning /
  interpretation prompts the way CF/IA do.
- Merge the near-identical CF/IA `LiteratureSearchController` (curve `:1362`, image `:1837`)
  into one shared class (Tier-1 dedup) and wire HS to it → HS gets Channel A.
- Surface `extracted_features` (§2.4) → HS becomes #323 Channel-B capable ("zero → full",
  #323's biggest delta).

**HS-3. Engine adoption, outer loop only.**
Port `RunDynamicAnalysisController`'s per-target attempt loop onto `CodegenQCEngine` with
`execute_fn = in-process exec`, `verify_fn = voted combined review`, `escalation =`
3-stage feedback ladder as an `EscalationPolicy`. The **per-map inner loop stays native**
(per-map verdicts + `SetAcceptRule` fraction/required-outputs gate) — the granularity
mismatch is real and we do not force it. Salvage judge + `degradation_notes` stay exactly
as they are (they are HS's accept/fallback stage).

**HS-4 (deferred, needs a user use case). Series mode.**
In-situ datacube series via the generic layer 2: locked config = decomposition params +
analysis targets locked at the anchor cube; non-anchor cheap pass skips decomposition
re-selection. Explicitly out of scope until demanded; the layer-2 extraction keeps the door
open (#322's HS scope note).

**Cross-pollination back to CF/IA (opt-in only, never default):**
- voted majority-to-reject verification (`SANITY_VOTES` wrapper,
  `hyperspectral_controllers.py:2556-2670`) as an optional engine verifier policy;
- salvage/degradation honesty (`status="partial"` + `degradation_notes`,
  `hyperspectral_analysis_agent.py:364-373`) as an optional result-status policy.
Both change observable behavior, so they ship OFF for CF/IA and are enabled only by an
explicit later decision.

---

## 7. Operating regimes — post-analysis vs. real-time in-situ

The engine must eventually serve two regimes: **post-analysis** (today's focus — thorough,
LLM-heavy QC) and **real-time in-situ** (per-frame analysis during a measurement, where the
budget is effectively *zero LLM calls on the happy path*). Building the regimes is phase-6
work, but three design constraints are adopted **now** so earlier phases don't scatter the
knobs:

### 7.1 `QCProfile` — a named bundle, not loose kwargs

All per-stage toggles are gathered into one profile object from the start (phases 1/4)
rather than accumulating as independent parameters:

```python
@dataclass(frozen=True)
class QCProfile:
    name: str                              # "thorough" | "realtime" | custom
    max_verification_iterations: int
    check_plan_conformance: bool = True
    best_of_n_eligible: bool = True
    human_feedback: bool = True
    literature: bool = True
    escalation_enabled: bool = True        # annealing ladder on/off
    voted_verification: bool = False
```

Presets: **`THOROUGH`** = exactly today's defaults (the engine default — behavior freeze
holds); **`REALTIME`** = verification 0–1, conformance off, best-of-N off, human feedback
off, literature off, no annealing. Curve's existing `max_verification_iterations=0`
"fast/in-situ" bypass (`curve_fitting_agent.py:459-464`) is the seed — today curve-only and
ad hoc; under the profile it becomes uniform across modalities.

### 7.2 The real-time loop shape: lock once, execute per frame, gate as drift detector

A lighter profile alone is not what makes in-situ fast — even one verification pass is a
multimodal LLM call per frame. The real-time pattern is the existing lock-and-apply
machinery re-organized from batch to streaming:

1. **Anchor thoroughly, once** (pre-run or on frame 1): full engine invocation under
   `THOROUGH`; plan → verify → lock the script.
2. **Per frame: execute the locked script only** — deterministic code execution, no LLM
   (demonstrated by the in-situ XRD `fit_pattern` work at ~0.8 s/frame).
3. **Per frame: drift check via the gate** — `gate.is_accept(metric)` on the locked
   script's output is arithmetic, not an LLM call. The unified `QualityGate` doubles as the
   online drift detector.
4. **On gate breach: escalate** — a real engine invocation (the adaptive-refit path) under
   the `REALTIME` profile, or queue the frame for post-hoc treatment.

Design constraint on the Layer-2 extraction (§5): the per-item loop must not assume items
arrive as a completed list. Concretely, outlier detection — today a post-hoc σ pass over
the whole series — needs an incremental per-frame variant; nothing else in layer 2 is
inherently batch.

### 7.3 Regime provenance in the record — "cheap now, rigorous later"

The per-item `quality_history` gains an additive `produced_under_profile` field (same
honesty pattern as HS's `degradation_notes`). This makes the two regimes *phases of one
workflow*: frames analyzed under `REALTIME` (plus gate-breach flags) are exactly the set a
post-experiment sweep re-runs under `THOROUGH`, after which §4's re-entry re-synthesizes
the trend over the upgraded records — with full provenance of which results got which
treatment.

Caveats: the profile controls **LLM cost, not numerics cost** — for heavy modalities
(hyperspectral per-pixel fits on large cubes) "real-time" may mean per-N-frames or reduced
resolution regardless of QC toggles. And per the prime directive, `THOROUGH` is the default
everywhere; `REALTIME` is opt-in and changes no existing path.

---

## 8. No-regression strategy (how "provably a no-op" is enforced)

Layered, per the lessons already logged on this codebase (verify the saved script, not the
result dict; prove no-op empirically; codegen-contract changes must cover every prompt
variant — here the rule is stronger: **no prompt changes at all** in extraction phases).

1. **Phase 0 golden harness (built before any refactor).** With a mocked model (canned
   deterministic responses) run both agents over fixture data covering: single + series,
   anchor + non-anchor, reuse path, verification failure → annealing escalation → hot,
   human-feedback off, judge fallback, best-of-N (n=2, escalation on/off). Capture as
   goldens: (a) every prompt string sent to the model, in order; (b) the full compiled
   result dict; (c) `quality_history`; (d) the saved scripts. Extraction phases must
   reproduce all four byte-identically (allowing only whitespace-insensitive comparison if
   a template refactor demands it — flag any such case in review).
2. **Arithmetic-equivalence unit tests** for gate routing: for a grid of values around each
   threshold, `gate.is_accept(v) == (v >= t)` etc., including post-`adjust_threshold`
   mutation.
3. **Existing offline suites** (multiskill 53, consolidation 29, column-handling, etc.)
   green untouched.
4. **Live smoke matrix** after each phase that touches a default path: one real run per
   modality on bedrock (the standard live-validation practice for this repo), diffing
   result structure + reading the saved scripts and the report, not just statuses.
5. **Cost parity check:** count LLM calls in the golden runs; extraction phases must not
   add or remove calls on CF/IA default paths.
6. **Deliberate-asymmetry ledger** (from §2–§5: reuse-path history, rate trigger, plan
   best-of-N, parallel workers, fast-accept semantics, Option B, CO_PILOT accept approval).
   Reviewers check the ledger against the diff: nothing on it may silently change.

---

## 9. Phasing

| phase | content | CF/IA risk | HS gain | gates |
|---|---|---|---|---|
| 0 | golden harness + fixtures | none (tests only) | — | goldens committed |
| 1 | Layer-0 types: gate `value_source` + IA gate instance + driver routing; shared `quality_history`/history-prompt builders; `CritiquePayload`; knobs gathered into `QCProfile` (§7.1, `THOROUGH` default) | no-op (goldens + arithmetic tests) | — | §8.1–.3, .5 |
| 2 | HS-1 (record + T=2 staging), HS-2 (literature plumbing, shared lit controller), §2.4 feature surfacing | none (CF/IA untouched except lit-controller move, golden-pinned) | history, staging, Channel A, #323-ready | HS live smoke; additive-keys check |
| 3 | Layer-3 re-entry: move reflection pair to base; `SynthesisReEntryController` + human/verifier producers; `refine_interpretation` (#323) mounts, curve first | none by construction (not in default pipelines; opt-in flags OFF) | inherits re-entry + Channel B | HS pipeline tests; new-path live tests |
| 4 | Layer-1 engine extraction (CF/IA adapters) | the big one — goldens + full live matrix | — | §8 all |
| 5 | HS-3 engine adoption (outer loop) | none | unified loop, voted verifier formalized | HS goldens + live |
| 6 (deferred) | Layer-2 generic series; HS-4 series; cross-pollination opt-ins; operating regimes (§7: `REALTIME` preset, streaming layer 2, profile provenance) | explicit opt-in decisions | series mode; in-situ mode | per-feature |

Phases 2 and 3 do not depend on phase 4 — HS expansion and #322/#323 land on the types
alone, so the highest-risk extraction (phase 4) can be scheduled independently (or even
deferred) without blocking the user-visible wins.

---

## 10. Open questions for review

1. **Phase-4 appetite.** Phases 0–3 deliver the HS expansion and both issues with CF/IA
   frozen. Phase 4 (engine extraction) is where near-copy duplication is actually retired —
   but it is also the only phase with real regression exposure. Proceed after 3, or park
   until the next time a change must touch both copies (the CLAUDE.md "abstract when it
   hurts" trigger has fired once already: the fast-accept divergence)?
2. **Gate routing depth in phase 1.** Route only the driver accept checks (minimal), or
   also the outlier-detection and fast-accept sites in the same pass? Minimal keeps the
   diff reviewable; full keeps the "one gate" invariant honest sooner.
3. **`quality_history` on curve's reuse path** — leave the asymmetry (this note's default)
   or fix it as a deliberate small behavior change with its own flag?
4. **HS annealing-level mapping for T=2 staging (HS-1):** treating retry stage 2
   ("abandon method family") as "hot" makes HS staging fire on method-changing successes —
   is that the intended novelty bar, or should HS staging also require `required_outputs`
   satisfaction?
5. **Autonomous synthesis critic for CF/IA** (reflection producer, default OFF): expose the
   flag at `analyze()` level, orchestrator level, or both?
6. **Naming**: `CodegenQCEngine` / `SynthesisReEntryController` / `CritiquePayload` — fine,
   or align with foundation-agent vocabulary (e.g. `ExtensibilityLoop` per CLAUDE.md
   element 5)?
7. **Regime exposure (§7):** where is the profile selected — `analyze(profile="realtime")`,
   an orchestrator tool argument, skill frontmatter, or all three (with the usual
   user-request-wins precedence)? And does the streaming drift-check loop live in layer 2
   or as a thin dedicated runner above it?
