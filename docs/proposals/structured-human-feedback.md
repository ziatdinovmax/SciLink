# Proposal: human feedback shows the decision, not the console

Status: plan, drafted 2026-09-26. Stage 1 (the contract, both renderers,
the kind tables, the curve fitting plan gate) and stage 2 (the other plan
gates: image analysis plan, hyperspectral preprocessing targets, series
regime plan, planning-mode plan review) and stage 4 (the pickers and
comparisons: best-of-N join on curves and images, series consensus,
consistency keep-or-revert, plan candidates, plan reopen) are implemented;
the user-guided keep-or-revert gates are out with the result gates they
follow. Stage 3 was
written and then dropped: the reachability audit below showed that none of
the fit and result gates fires under the orchestrator's defaults, which
matches the expectation that beyond plan approval and the best-of-N choice
there is no human review of analysis results. Checked live on
Bedrock Opus 4.8: the curve fitting plan gate (browser and shell, a
Raman-like spectrum), the image analysis plan gate (browser, the
polycrystalline grains demo) and the planning plan review (shell, a small
annealing experiment); the two hyperspectral gates are unit-tested against
their printers. Enter approves on both surfaces. Scope rule: this work improves how the
existing gates are shown; it adds no gate and re-enables none. The
`analysis_review` gate (`SimpleFeedbackCollector`) was judged impractical
and is not wired into any pipeline, so it is not in the table below.

## The problem

The web UI's feedback surface looks like a terminal because it *is* the
terminal. Every agent gate (25 `request_human_feedback` calls) holds a
structured object at ask time — a plan dict, a fit result with a review PNG,
a candidate list with per-candidate judge comments — and
prints it, then asks. The web channel (`server/hitl_channel.py`) captures
stdout, and the presenter (`server/presenter.py`) regex-parses the captured
text back into widgets. Its own docstring calls this "regex-over-captured-
stdout by necessity". Whatever the regexes miss lands in `context_display`,
which `FeedbackPanel.tsx` renders as a monospace pre-wrap box and the
terminal shell prints inside a panel.

Two facts make this cheap to fix:

- `FeedbackRequest` already has `context` and `origin`. One gate fills
  `context` (the hyperspectral regime plan, with rendered text); a few put
  routing data in `origin`, and the two that carry decision data there
  (`auto_repair`, the reopen `reason`) already render as a proper callout.
  The path from gate to widget exists; it just carries text.
- The 25 gates are six decision shapes. Counted by `kind`: review a plan
  (6, counting `approve_or_revise`), review a result or fit (6, incl.
  `review_metrics`), keep or revert (5), pick a candidate (5), confirm (2),
  describe missing metadata (1).

## Design

**Two independent things travel with a question: what is under review, and
what the decision is.** Today both are inferred from prose. After this
change both are declared by the gate.

### 1. `subject`: what is under review, as blocks

A new optional field on `FeedbackRequest`:

```python
subject: Optional[Dict[str, Any]] = None   # JSON-serializable
```

with the shape `{"title": str, "blocks": [Block, ...]}`. A fixed block
vocabulary, small on purpose (same principle as the skill section
vocabulary: authors write against named shapes, renderers know what to do
with each):

| block | payload | renders as |
|---|---|---|
| `text` | `{"markdown": str}` | markdown body |
| `fields` | `{"items": [{"label", "value", "unit"?, "flag"?}]}` | label/value grid; `flag` (`ok`/`warn`/`bad`) colours the value (R² against threshold) |
| `chips` | `{"label": str, "items": [str]}` | parameters / features to extract |
| `steps` | `{"label": str, "items": [str]}` | numbered pipeline / strategy steps |
| `table` | `{"columns": [str], "rows": [[...]], "caption"?}` | fitted parameters, scalarizer columns, per-regime rows |
| `figure` | `{"path": str, "caption"?}` | image via `/files`; path relative to the session dir |
| `candidates` | `{"items": [{"idx", "name", "metric"?, "value"?, "approved"?, "figure"?, "judge_comment"?}], "pick": int, "reasoning"?, "caveats"?} ` | one card per candidate, pick highlighted |
| `compare` | `{"left": {"label", "blocks"}, "right": {"label", "blocks"}}` | two columns (old vs new fit, approved plan vs revision) |
| `notice` | `{"title", "lines", "tone"}` | the callout the plan gates already have |

Rules:

- Every block may carry a `label`: the section name as the printout has
  it, emoji included ("🔍 Observations"). Both renderers put labeled blocks
  in one aligned label column; a short list is plain text, not chips.
- A gate keeps printing exactly what it prints today. Printing is the
  console surface, the verbose log and the audit trail; nothing here
  removes it. The web and the shell stop *depending* on it.
- `subject` is built by a small pure function next to the gate
  (`_plan_subject(state)`, `_fit_subject(fit_result, r2, png)`), unit-tested
  without an LLM. Where a `_display_plan` exists, the subject builder and
  the printer read the same dict; over time the printer can render from the
  subject so the two cannot drift.
- Paths in `figure` are absolute at the gate; the presenter relativises
  them to the session dir as it does for preview images today.
- The feedback log (`hitl._append_record`) and `pending_question.json`
  record `subject` too, so a run that died while blocked says what it was
  showing, and a resumed session can re-present it.

### 2. The decision comes from `kind`, not from the prompt text

The presenter chooses the widget from `kind` (with `options` and
`origin.stage` for the variants), never from the prompt or the context:

| `kind` | widget | replies |
|---|---|---|
| `review_plan`, `approve_or_revise`, `review_fit`, `review_result`, `review_metrics`, `dataset_description` | text + accept | free text / `""` |
| `keep_or_revert` | two buttons, optional text | `keep` / `""` / text (reopen gate) |
| `bestofn_select`, `plan_candidate_select`, `consensus_select` | candidate picker | digit / `""` |
| `confirm` | confirm / cancel | `y` / `no` |

Labels move to a table in `scilink/ui/vocabulary.py` keyed by
`(kind, stage)` with a per-kind default, so the shell and the web say the
same words (the existing sync test covers it). The reply contracts are
unchanged, so no gate's parsing of the answer changes.

### 3. Presenter: prefer the subject, fall back to the regexes

`present_question` gains two branches. If `hreq.subject` is set, the payload
carries `subject` (paths relativised) and the widget from the kind table;
`context_display` still travels but the front-ends show only the blocks (they
are the printed text; showing both was tried and dropped). If `subject` is absent, the
current path runs unchanged. That keeps every unconverted gate and every
existing presenter test working during the conversion and lets the gates
convert one at a time.

The payload keeps its current keys (`widget`, `labels`, `prompt`,
`context_display`, `preview_images`, `candidate_captions`, `code_files`,
`notice`, `candidates`, `judge_pick`, `fanout`) so the shell and the MCP
server keep working while they are updated; `subject` is additive.

### 4. Renderers: one per surface, one block vocabulary

- **Web.** `FeedbackPanel.tsx` becomes a thin decision panel: a
  `SubjectBlocks` component renders the blocks, the widget row renders the
  decision. The action row is sticky at the bottom of the panel and the
  blocks scroll inside it, so a long plan never pushes the buttons below the
  fold (the reason the auto-repair notice was moved out of the printed text).
  Nothing is clamped: a reviewer reads the whole plan before deciding.
- **Shell.** `cli/shell/channel.py`'s `_show_context` renders blocks with
  rich (`Table`, `Panel`, `Markdown`, figure paths) and prints the console
  text only when there is no subject. The Ctrl+O overflow stays for long
  console text.
- **MCP.** `_pending_questions` exposes `subject` beside `prompt` so an
  agent client can show or reason over the structured question.

### 5. Gate-by-gate conversion

| gate (file:stage) | kind | subject blocks |
|---|---|---|
| curve `fitting_plan` | review_plan | fields (mode, model), text (observations, approach), steps (strategy), chips (parameters); series: table of regimes + text transitions |
| image `analysis_plan` | review_plan | same shape with pipeline steps, features chips, quality criteria, expected outputs |
| base `preprocess_plan` (targets) | review_plan | table (index, target, value, description) |
| hyperspectral `series_regime_plan` | review_plan | table of regimes, text rationale, fields (change-point summary); replaces the `context=rendered` text |
| exp `iteration_feedback` | review_plan | text (current result summary), fields |
| planning `plan_review` | approve_or_revise | text (plan summary as the critic sees it: `summarize_plan_for_critic`), chips (equipment), steps, notice (auto-repair, caveats) |
| planning `missing_metadata` | dataset_description | fields (file, columns seen) |
| curve/image `consistency_result` | keep_or_revert | compare (both fits, with the consistency numbers) |
| planning `plan_reopen` | keep_or_revert | notice (reason), compare (approved vs revision, changed fields only) |
| curve/image `bestofn_join` | bestofn_select | candidates (metric, gate, iterations, figure, judge comment, reasoning) |
| planning `plan_candidates` | plan_candidate_select | candidates (name, judge per-candidate comment, caveats on the pick, report path) |
| curve/image `series_consensus` | consensus_select | candidates (model, spectra/images it covered, R²) |
| meta `fanout_confirm` | confirm | fields (verdict, join axis), steps (branches), text (rationale, autonomy note, soft-cap warning) |

Fan-out branch questions need nothing: `QueueChannel.serve_pending` copies
the request with `replace`, so `subject` travels with the branch label.
The regime gate loops up to three rounds and rebuilds its subject each
round.

## Staging

Each stage is one PR, lands with the regex path intact, and is checked
live on both surfaces (the pexpect shell harness and the Chrome test from
the web UI live-test notes).

1. **Contract and renderers, no gate changes.** `subject` on
   `FeedbackRequest` and in the feedback log; block vocabulary documented
   in `hitl.py`; kind→widget and label tables in `vocabulary.py`
   (regenerate `vocabulary.ts`); presenter prefers `subject`;
   `SubjectBlocks` in the web UI with the sticky action row and the
   console disclosure; rich block renderer in the shell; MCP exposes
   `subject`. Tests: presenter unit tests per block and per kind, a
   fixture of presented questions (one per shape) that pins the Python
   side, `test_vocabulary_sync` extended to the label table.
2. **Plan gates** (curve, image, preprocessing targets, hyperspectral
   regimes, iteration feedback). These are the gates a scientist sits at
   longest. Subject builders unit-tested against sample `state` dicts.
3. ~~Fit and result gates~~ — dropped, see the audit: none fires under
   the defaults.
4. **Candidate and compare gates** (best-of-N ×2, plan candidates,
   consensus ×2, consistency ×2, plan reopen). Done; `parse_bestofn_review`
   and `parse_plan_candidate_review` retire with the regex path in stage 6.
   Two contract points learned here: the keep-or-revert primary button sends
   the gate's own first `options` word ("keep", "consensus"), never a fixed
   "keep"; and a picker whose gate also takes a typed reply (a model name,
   'more') says so on its candidates block (`free_text`), which becomes a
   box under the picker on both surfaces. Live checks of every reply path
   (2026-09-27, real browser against a launcher-only hook that parks
   synthetic gates, plus one real three-candidate run) found and fixed:
   Enter on a picker sending the judge's pick whatever card was selected;
   the floating figure inset covering the Accept button (it now hides
   while a question is pending); Enter in a secondary box inserting a
   newline instead of sending.
5. **Planning plan review, metadata, fan-out, image confirm.** Retires
   `parse_fanout_confirm`.
6. **Retire the regex path.** `clean_context` and the classifier go;
   `context_display` remains only as the console disclosure. Delete the
   ported Streamlit parsers and update the presenter tests to the fixture.

Stages 2 to 5 can be reordered by whatever gets exercised live first.

## Reachability audit (2026-09-27)

Every gate traced to its callers and the condition at the call site, under
the orchestrator's defaults (three candidates with escalation for image and
curve, thorough profile, human feedback on). Only the first group gets a
subject.

**Fires under the defaults**

| gate | when |
|---|---|
| curve fitting plan, image analysis plan | every co-pilot run (seen live) |
| planning plan review, plan candidates picker, plan reopen | every co-pilot planning run (review seen live); the picker with best-of-N on; reopen only on an agent-initiated revision |
| planning code review (same function, printed text) | co-pilot, when the plan carries implementation code |
| hyperspectral preprocessing targets | once per cube at depth 0, hyperspectral pipeline only |
| series regime plan | hyperspectral series in co-pilot or autopilot, not on a locked replay |
| best-of-N join picker (curve, image) | when attempt 0 is weak and the run escalates (seen live) |
| series consensus picker, consistency keep-or-revert (curve, image) | series adaptive refit, when refits disagree or a refit is worse |
| fan-out confirm | meta fan-out, always |
| missing-metadata description | planning, a data file without metadata the session did not write |

**Cannot fire under the defaults.** Attempt 0 of a best-of-N run is a
candidate job, and every candidate job sets `_suppress_human_feedback`, so
these fire only when the candidate count resolves to one (an explicit
`n_candidates=1`, the quick or extract profile, direct agent use): image
result review, image poor quality, curve poor fit, and the two user-guided
keep-or-revert gates that follow them. Judged dead in practice; no subject.

**Never fires.** No caller, or switched off: analysis review, first-spectrum
fit review, scalarizer confirmation, the mixin's iteration feedback, and the
image tier-2 approval (needs `analysis_depth="auto"`; the orchestrator
passes "basic"). No subject.

## Out of scope

- Changing any reply contract or any gate's control flow.
- Removing the printed output. It stays as the console, the verbose log
  and the record.
- The Streamlit UI (`scilink/ui/app.py`): superseded by the React UI and
  not updated for `subject`; it keeps working on the printed text.
- Editing a plan in place in the browser (structured edits instead of free
  text). Worth doing after the plan is shown as a card, not before.

## Risks and answers

- **Drift between print and subject.** Both read the same dict; the
  subject builder is the unit under test, and where practical the printer
  is rewritten to render from the subject (a `render_blocks_text` helper),
  so there is one source.
- **Payload size.** Subjects are small (a plan, a table of parameters).
  Code files already travel inline; figures travel as paths.
- **Two renderers to maintain.** They share one vocabulary and one fixture,
  the same arrangement as the narration reader and vocabulary today.
- **A gate that forgets a block** degrades to what it shows today: the
  console text is still there behind the disclosure, and the widget is
  still right because it comes from `kind`.
