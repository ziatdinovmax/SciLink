# One reading layer across the orchestrator modes

**Status:** proposal, nothing implemented. Symptom tracked in #397.
**Motivation:** every mode reads files, each does it differently, and the
differences are accidents rather than decisions.

## Two live failures, one cause

**Planning cannot open a PDF.** A delegation was given a funded proposal to
fold into a portfolio. `read_file` opened it as text:

```
read_file     -> '%PDF-1.7\n%\xef\xbf\xbd...1082 0 obj'
extract_text  -> 70,987 chars, 29 pages, clean
```

The agent correctly diagnosed the garbling and proceeded from the caller's
summary instead — so the run degraded quietly, grounded on a paraphrase
rather than on the document. The extractor it needed is already in the repo
and already used by two other modes.

**Meta and analysis cannot reach the end of one.** Their readers truncate
from the head at 200k chars with no `tail`, no `search`, and a bare
`truncated: true`. That is the same shape that made an agent call planning's
`read_file` five times on a 35 KB white paper before giving up (fixed there
in #394).

The capability gap runs in **both** directions, which is the tell that this
is a layering problem rather than three separate bugs.

## Current state

| mode | tool | extraction | windowing |
|---|---|---|---|
| planning | `read_file` | text/json/csv only | head + tail + search (lines) |
| meta | `view_document` | `extract_text` (pdf/docx/ocr) | head only (chars, per document) |
| analysis | `read_document` | `extract_text` (pdf/docx/ocr) | head only (chars, **combined**) |
| simulation | — | — | — |

Extraction is already shared (`scilink/parsers/extract.py`). Windowing is
not — and windowing is the half that was just improved, in exactly one
place. Caps differ in units (lines vs chars) and in scope (per document vs
combined), and the result dicts use different field names for the same
facts.

## What to unify, and what not to

**Not one tool.** These are two genuinely different jobs:

- *inspect a session artifact* — a plan, a log, a document we just wrote.
  Line-oriented, high call frequency, "did this end correctly".
- *ingest user source material* — a methods paper, a protocol PDF.
  Format extraction, char-oriented, feeds grounding.

Collapsing them forces one set of semantics onto both and makes the tool
description harder to route on — the failure mode this branch has spent its
time fixing. Keep the three tools and their names.

**Share the middle.**

```
        extract_text()      <- already shared
              |
          window()          <- the missing piece
        /     |      \
read_file  view_document  read_document
```

`window()` owns: head / tail / search, the truncation notice that names the
way out, and one reporting vocabulary (`mode`, `total`, `shown`,
`truncated`). Nothing else moves.

## Phases

**Phase 0 — planning reads documents (small, independently shippable).**
Route document extensions in `read_file` through `extract_text` instead of
opening them as text. Fixes the observed failure on its own; everything
below can wait. ~10 lines plus tests.

**Phase 1 — extract `window()` into `scilink/utils/`.** Lift the logic
already written for `read_file` (#394) verbatim; `read_file` becomes its
first caller. Lands as a provable no-op: the existing read_file tests pass
unchanged.

**Phase 2 — adopt it in `view_document` and `read_document`.** They gain
`tail` and `search` and the actionable notice. Their caps stay where they
are; only the modes and the reporting change.

**Phase 3 — reconcile the caps.** Make `read_document`'s cap per document
rather than combined (today a fourth file can silently push the first
three's tail out) and settle on one unit. This is the only phase that
changes existing behaviour, so it lands alone and gets a live check.

**Phase 4 — decide simulation deliberately.** It has `analyze_output` for
engine logs and no generic reader. That may be correct rather than a gap;
either way, record the decision instead of leaving it an accident.

## Stays mode-specific

- `read_document` persisting a literature file for `run_analysis` to ground
  on — that is analysis-mode grounding semantics, not reading. But when the
  persisted artifact is truncated it should say so **in the file**, since it
  outlives the conversation.
- `view_document`'s OCR fallback and tabular preview.
- Path resolution and sandboxing rules, which differ by mode for reasons
  that are still good.

## Risks

- **The stdout lines are a UI contract.** The Streamlit parser reads them;
  changing the printed shape needs both sides moved together.
- **`read_document`'s side effect** means a windowing change alters a
  persisted grounding artifact, not just a tool result. Phase 2 must not
  change what gets written, only what gets returned.
- **Three call sites, one util** is the classic moment to over-abstract.
  `window()` should stay a function over `(text_or_lines, mode, n, pattern)`
  with no knowledge of tools, sessions, or paths.

## Not in scope

- No change to `extract_text` itself.
- No new orchestrator mode, no base class (see CLAUDE.md — the trigger here
  is a diverging *fix*, which is why the shared piece is a util and not a
  parent class).

## Estimate

Phase 0 is under an hour. Phases 1-2 about a day. Phase 3 half a day plus a
live check because it changes behaviour. Phase 4 is a decision, not work.
