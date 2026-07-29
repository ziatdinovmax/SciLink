# One reading layer across the orchestrator modes

**Status:** proposal. Nothing implemented. Tracked as #397.
**Written for a fresh session** — it assumes no context beyond the repo.
**Scope note:** the branch this was written on (`fix-planning-kb-scope-and-tool-args`,
PR #394) is planning-mode only. Phase 0 below is planning-mode and *could*
land there; Phases 1-4 touch meta and analysis and want their own branch.

## The problem in one paragraph

Every orchestrator mode reads files, each does it differently, and the
differences are accidents rather than decisions. Text **extraction** is
already shared (`scilink/parsers/extract.py`); **windowing** — how a file too
large to return whole is navigated and how that is reported — is not. So
each mode is missing the half another mode has.

## Two observed failures

**Planning cannot open a PDF.** A delegation was handed a 29-page funded
proposal to fold into a portfolio. `read_file` opened it as text:

```
read_file     -> '%PDF-1.7\n%<bytes>...1082 0 obj'
extract_text  -> 70,987 chars, 29 pages, clean
```

The agent diagnosed the garbling correctly and proceeded from the caller's
summary instead, so the run degraded quietly — grounded on a paraphrase
rather than on the document. Reproduce with
`meta_session_20260727_195954/uploads/Narrative_PNNL_Zhang_87627.pdf`.

**Meta and analysis cannot reach the end of one.** Both truncate from the
head with no `tail`, no `search`, and a bare `truncated: true`. That is the
same shape that made an agent call planning's `read_file` five times on a
35 KB white paper before giving up — fixed there in PR #394, not elsewhere.

## Current state

| mode | tool | where | extraction | windowing |
|---|---|---|---|---|
| planning | `read_file` | `planning_agents/orchestrator_tools.py:4656` | text/json/csv only (opens everything else as text at `:4738`) | head + tail + search, line-based (`:4742-4799`) |
| meta | `view_document` | `meta_agent/meta_orchestrator_tools.py:1502` | `extract_text` | head only, 200k chars **per document** (`:1500`) |
| analysis | `read_document` | `exp_agents/analysis_orchestrator_tools.py:5500` | `extract_text` | head only, 200k chars **combined** (`:51`) |
| simulation | — | — | — | — |

`extract_text` (`parsers/extract.py:16`) already handles
`.pdf .docx .md .txt .json .yaml .yml .csv .xls .xlsx`, with OCR fallback for
scanned PDFs and adaptive tabular preview.

## Measured blast radius

**These tools have no code callers.** All three are closures registered into
a tool map; nothing in the codebase imports or invokes them. Their callers
are language models, so the risk surface is tool descriptions and prompt
text, not a call graph.

- source files touched: 1 (phase 0), 2 (phase 1), 2 (phase 2)
- tests referencing any of the three: 4 — `test_read_file_modes.py`,
  `test_planning_campaigns.py`, `test_meta_tool_dispatch_guards.py`,
  `test_knowledge_query.py`
- the planning system prompt names `read_file` in 4 places
  (`planning_orchestrator.py`) and must stay truthful
- **the one real coupling:** `read_document` persists its combined text as a
  literature file (`analysis_orchestrator_tools.py:5545`) that
  `run_analysis` grounds on; `literature_file` has 49 references across
  `exp_agents`. Most consume the path, not the content — but this is where a
  windowing change stops being cosmetic.

NOT a risk, contrary to an earlier draft of this document: the Streamlit UI
does **not** parse these tools' stdout. It gates on the candidate-card and
feedback markers only. Verified by grep; stated here so nobody plans around
a phantom hazard.

## What to unify, and what not to

**Not one tool.** Two genuinely different jobs:

- *inspect a session artifact* — a plan, a log, a document we just wrote.
  Line-oriented, high call frequency, "did this end correctly".
- *ingest user source material* — a methods paper, a protocol PDF. Format
  extraction, char-oriented, feeds grounding.

Collapsing them forces one set of semantics on both and makes the tool
description harder to route on, which is the failure mode PR #394 spent its
time fixing. Keep three tools and their names.

**Share the middle.**

```
        extract_text()      <- already shared
              |
          window()          <- the missing piece
        /     |      \
read_file  view_document  read_document
```

`window()` owns head / tail / search, the truncation notice that names the
way out, and one reporting vocabulary (`mode`, `total`, `shown`,
`truncated`). It should be a pure function over
`(text_or_lines, mode, n, pattern)` with no knowledge of tools, sessions or
paths — three call sites is exactly the moment to over-abstract.

## Phases

**Phase 0 — planning reads documents.** In `read_file`, route the extensions
`extract_text` supports through it instead of opening them as text
(`orchestrator_tools.py:4738`). Fixes the observed PDF failure on its own.
Planning-only, so it can land on PR #394. Under an hour with tests.
*Verify:* the PNNL proposal above returns prose, not `%PDF-1.7`.

**Phase 1 — extract `window()` into `scilink/utils/`.** Lift the logic
already written for `read_file` in PR #394 (`:4742-4799`) verbatim;
`read_file` becomes its first caller. Lands as a provable no-op —
`test_read_file_modes.py` passes unchanged. ~½ day.

**Phase 2 — adopt it in `view_document` and `read_document`.** They gain
`tail`, `search` and the actionable notice. Caps stay where they are; only
modes and reporting change. **Must not change what `read_document` writes**,
only what it returns. ~½ day.

**Phase 3 — reconcile the caps.** Make `read_document`'s cap per document
rather than combined (today a fourth file can silently push the first
three's tail out of the saved artifact, with nothing attributing the loss),
and settle on one unit. The only phase that changes observable behaviour —
lands alone, with a live check. ½ day.

**Phase 4 — decide simulation deliberately.** It has `analyze_output` for
engine logs and no generic reader. That may be correct rather than a gap.
Record the decision either way. Not work.

## Stays mode-specific

- `read_document` persisting a literature file — analysis-mode grounding
  semantics, not reading. If the persisted artifact is truncated it should
  say so **in the file**, since it outlives the conversation.
- `view_document`'s OCR fallback and tabular preview.
- Path resolution and sandboxing, which differ by mode for reasons that
  still hold (planning sandboxes writes to the current delegation).

## Not in scope

- No change to `extract_text`.
- No new orchestrator mode and no base class. Per CLAUDE.md the trigger here
  is a diverging *fix*, which is why the shared piece is a util rather than
  a parent class.

## Prior art in this repo

PR #394 added `tail`, `search` and the actionable truncation notice to
`read_file`, with tests in `tests/test_read_file_modes.py` and a live probe
confirming an agent uses `tail` unprompted (one call, 10 s, on a 45 KB
document). Phases 1-2 are that work generalised; read those tests first.
