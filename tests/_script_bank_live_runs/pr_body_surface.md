Follow-up to #347 (stacked on its branch): the script bank's **user surface** and its **bridge to skill graduation** — items 1 and 5 from that PR's deferred list.

## What a user gets

**See and manage what the agent remembers.** Bank records were previously inspectable only as JSON files under `~/.scilink/script_bank/`. Now:

- `scilink memory bank` lists banked scripts per domain with their cross-session stats — how many sessions each script has succeeded in, how often it was retrieved, its quality metric. Records proven across ≥3 sessions are starred as graduation candidates (`--proven-only` filters to them). `bank-show` prints a full record including the script; `bank-prune` deletes one.
- The UI memory panel gains a matching **Script bank** section beside the staged-knowledge section: per-domain expanders, usage badges, a record viewer, and promote/delete actions.

**An evidence-based path from episodic memory to skills.** `scilink memory bank-promote <domain>/<id>` (or the UI button) copies a proven record into the distill-staging buffer, where the *existing* review-gated ceremony applies unchanged — upgrade it into an existing skill, or consolidate it with others of its technique into a new one. This completes the memory story from issue #346: the bank gives immediate breadth (every success, zero ceremony), graduation gives curated depth, and "succeeded repeatedly across sessions" replaces "climbed to hot once" as the promotion evidence. The bank record itself is kept (episodic history is not consumed) and marked so surfaces show the promotion and repeats refuse.

Smoke-tested against the real store produced by #347's live-validation runs (11 curve + 3 hyperspectral records): the listing stars the two genuinely proven records (4 and 3 sessions), and promoting one staged it under a derived technique label, immediately visible to `scilink memory staged` with its metric and provenance.

---

## Technical details (for AI reviewers)

- `scilink/skills/_shared/_script_bank.py`: `proven_n()` (default 3, `$SCILINK_BANK_PROVEN_N`, floor 2), `record_label(rec)` (model_type → analysis_type → analysis_target fallback chain), `bank_summary(domain)` (display rows sorted by `(-n_successes, -n_retrievals, id)`), `promote_to_staging(domain, rid, technique=None, *, root, staging_root)` → stages `{provenance: "bank_proven", model, deviation_from_plan: <cross-session evidence>, plan_summary, measurement_context, working_script, session, bank_id, r_squared|quality_score}` via the existing `_staging.stage_solution`; refuses when `promoted_to_staging` is already set or the script is empty; deterministic technique label = snake-cased `record_label` cut at a word boundary ≤48 chars (no LLM — the staging vocabulary accepts any label, and the existing LLM labeler remains available upstream for T=2 staging).
- `scilink/cli/memory.py`: four flat subcommands (`bank`, `bank-show`, `bank-promote`, `bank-prune`) mirroring the staged-solutions command shapes, including the `_warn_memory_off` nudge on promote and `--yes` on prune. Module docstring updated.
- `scilink/ui/components/skills.py`: `_render_bank_section()` + `_render_bank_record()` appended after the staged section; thin rendering over the module helpers (no logic in Streamlit code); promote button disabled once promoted; delete via the per-record `···` popover.
- Tests: +5 in `tests/test_script_bank.py` (36 total) — summary shape/ordering/proven flag, `SCILINK_BANK_PROVEN_N` override, full promote round-trip asserting the staged record's provenance/metric/evidence and the refuse-on-repeat, explicit-technique and missing-record errors. Goldens pass unregenerated (no agent-path changes in this PR).
- Performance (second commit): Streamlit renders popover/expander content eagerly on every rerun — the cause of the panel slowdown with many memory records. All heavy content (skill markdown, staged/bank record views incl. `st.code` script highlighting) now loads only behind an explicit "Load full content" tick; long lists paginate (15 + Show all); oversized scripts truncate with a CLI pointer. Headless `AppTest` regression tests (`tests/test_memory_panel.py`): a 70-record store renders in ~0.08 s with zero eager code blocks. This also fixes the pre-existing eager renders in the provisional-skills and staged-knowledge sections.
- Known limits: promotion metadata maps `r_squared`/generic metrics onto the staging record's existing fields.
