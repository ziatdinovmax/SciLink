# Self-evolving agent demo — annealing + memory + skills

A two-run story on the 1D EELS core-loss spectrum (`../eels_identification_demo/`)
showing how SciLink's curve-fitting agent **gets better at its job by remembering
what worked** — the loop that makes an agent self-evolving.

```
 seed:  a THIN base EELS skill (curve_fitting/eels) — states the goal, no recipe yet

                 ┌────────────────────────── Run A ──────────────────────────┐
 thin skill ──▶  T=0 reject ─▶ T=1 reject ─▶ T=2 HOT: abandon plan, regenerate ─▶ success
 + wrong plan        (constraint annealing climbs as verification keeps failing)         │
                                                                                          ▼
                                                       STAGE raw T=2 solution (buffer)
                                                                                          │
                                       upgrade@1: MERGE the staged solution INTO the skill
                                       EELS skill  v1 (thin) ──────────▶ v2 (enriched)
                                                                                          │
                 ┌────────────────────────── Run B ──────────────────────────┐           ▼
 same spectrum ──▶  locks the right multi-edge model from the start ──▶ success at T=0
 (enriched skill)   no escalation needed — the agent learned
```

The T=2 win is first **staged** (not turned straight into a skill); the demo then
**upgrades** the existing EELS skill from that single staged solution (N=1 merge).
A run with no matching skill would instead accumulate staged solutions and
**consolidate** N of them into a new skill — see `scilink memory staged`.

- **Annealing** lets the agent escape a bad plan within a single run (T=0→T=1→T=2).
- **Memory** persists the novel pipeline outside the session (`~/.scilink`, survives `pip` upgrades).
- **Skills** make that memory reusable — and here the learning **graduates into an
  existing skill**: the thin `curve_fitting/eels` skill is *enriched* (v1→v2) with
  the recipe annealing discovered, so the next run on a similar problem succeeds
  immediately. The agent's durable competence compounds.

> Note: there is no built-in 1-D EELS curve-fitting skill in SciLink today, so the
> demo **seeds a thin base `curve_fitting/eels` skill** (`base_eels_skill.md`) into
> its own memory store, then shows it being graduated/enriched. Nothing is written
> into the shipped `scilink/skills/` tree, and the committed auto-distill behavior
> (which creates provisional `auto_*` skills) is unchanged — the merge-into-existing
> step is performed by the demo harness using the same graduation helper's update branch.

## Run

```bash
export AWS_BEARER_TOKEN_BEDROCK=...   # or any LiteLLM-supported provider creds
export AWS_REGION_NAME=us-east-1
python examples/self_evolving_demo/run_demo.py
```

Each run makes several LLM calls and **executes LLM-generated curve-fit code**
(`UNSAFE_EXECUTION_OK=true` is set by the script). Run A deliberately escalates to
T=2, so it takes longer than a normal fit. Use `--skip-run-b` to record only the
escalation+distill half, or `--max-verification N` to give it more room to climb.

## What it records (under `recording/`)

| File | Use for the slide |
|---|---|
| `manifest.json` | Full structured record: both runs' inputs, per-iteration annealing level + R², the verifier's **outcome + suggestion summarized at each T**, final models, the EELS skill v1/v2, and the before/after comparison |
| `self_evolution.png` | 2-panel slide figure: **①** the annealing climb annotated with the verifier's outcome+suggestion at each T and the hot-regeneration point; **②** the memory→skill-update arc (EELS v1 → distilled T=2 recipe → EELS v2) and Run-B payoff |
| `artifacts/run_a_fit.png`, `run_b_fit.png` | The fitted curves before/after learning |
| `artifacts/run_a_*.py`, `run_b_*.py` | The actual scripts the agent wrote |
| `artifacts/eels_v1.md`, `eels_v2.md` | The EELS skill before and after graduation (the diff *is* the learning) |

The demo's memory store lives in `.scilink_memory/` (override with `--home`);
`recording/` and `.scilink_memory/` are git-ignored.
