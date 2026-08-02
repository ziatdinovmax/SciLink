# Critic-validator A/B experiment

Pre-registered experiment to decide critic-validator architecture
(mechanism × shape) before the broader refactor. Design doc:
[`design.md`](design.md). Memory pointer:
`project_critic_experiment.md`.

## Layout

```
critic_experiment/
├── design.md                  experiment design (pressure-tested before code)
├── README.md                  (this file)
├── _schema.py                 FixtureLabel + Issue dataclasses + JSONL I/O
├── prompts/
│   ├── vasp.py                3 VASP prompts (Fe BCC, UO₂, Pt111+CO)
│   └── lammps.py              3 LAMMPS prompts (water, LiPF₆/EC, NaCl(aq))
├── generate_source1.py        run agents N times → stub label rows
├── fixtures/                  (committed once labeled)
│   ├── incar/from_generator/  Source 1 outputs (VASP)
│   ├── incar/planted/         Source 2 — hand-mutated baselines
│   ├── incar/controls/        Source 3 — known-good (false-positive set)
│   ├── incar/breakage/        Source 4 — real-world failures
│   ├── lammps/from_generator/ Source 1 (LAMMPS)
│   ├── lammps/planted/        Source 2 (LAMMPS)
│   ├── lammps/controls/       Source 3 (LAMMPS)
│   ├── lammps/breakage/       Source 4 (LAMMPS)
│   └── labels.jsonl           unified, engine-tagged ground truth
├── variants/                  (TODO — built after fixtures lock)
├── shapes/                    (TODO)
├── run_experiment.py          (TODO)
├── analyze.py                 (TODO)
└── outputs/                   (gitignored)
```

## Quick start — fixture authoring workflow

### Source 1 — generator-derived (~60 fixtures per engine)

Run the production agents N=20 times against each of the 3 prompts;
stub label rows get appended for hand-filling.

```bash
# VASP — requires SCILINK_API_KEY in env
python -m benchmark.critic_experiment.generate_source1 --engine vasp

# LAMMPS
python -m benchmark.critic_experiment.generate_source1 --engine lammps

# both, smaller smoke set
python -m benchmark.critic_experiment.generate_source1 \
    --engine both --n-trials 2
```

Each generated file lands under `fixtures/<engine>/from_generator/` and
one row per fixture is appended to `fixtures/labels.jsonl` with
`true_issues=[]`. **The labeling pass — filling in `true_issues` — is
done by hand by someone with physics expertise, not the variant
implementer.** This is the experiment's main defense against fixture
overfitting (design doc Section 6).

### Source 2 — planted (30 per engine)

Author by mutating known-good baselines with single targeted edits
(typo, value error, coupled). Fixtures + labels go in together. No
generator script — these are intentional, author-controlled inputs.

  - 10 syntax typos (VASP: ISPN, ENCT, KSPCING; LAMMPS: pair_styl,
    velosity, malformed fix arg-counts)
  - 10 value errors (VASP: ISPIN=1 for Fe, ENCUT=200, LDAUL=2 wrong
    subshell; LAMMPS: timestep too large, units mismatch with potential,
    wrong pair_style for chemistry)
  - 10 coupled errors (typo + value in same file)

### Source 3 — controls (20 per engine)

Hand-verified known-good fixtures. `true_issues=[]` for all. Used to
measure false-positive rate — a critic that's too aggressive will flag
issues here.

### Source 4 — real-world failures (held out)

  - VASP: mine `examples/breakage_benchmark_*/` for prior real failures
    with known correct fixes (~20).
  - LAMMPS: mine `examples/mlip/` outputs or preserved cluster logs;
    supplement from LAMMPS mailing-list archived bug reports if fewer
    than 10 (~10–20 total).

Source 4 is held out from any prompt tuning. Use as out-of-distribution
sanity check.

## Schema

Each row in `labels.jsonl` is one [`FixtureLabel`](_schema.py):

```python
@dataclass
class FixtureLabel:
    id: str                              # <engine>_<prompt>_<NNN>
    engine: Literal["vasp", "lammps"]
    source: Literal["from_generator", "planted", "controls", "breakage"]
    prompt_id: str                       # references prompts/<engine>.py
    prompt_text: str                     # frozen at fixture-creation
    system_name: str                     # benchmark.systems entry
    fixture_path: str                    # relative to critic_experiment/
    true_issues: List[Issue]             # filled by labeling pass
    canonical_fix_path: Optional[str]
    labeled_by: Optional[str]
    labeled_at: Optional[str]            # ISO date
    notes: str
```

Each `Issue`:

```python
@dataclass
class Issue:
    locator: str                         # "tag:ISPN" / "command:pair_style"
    severity: "error" | "warning" | "info"
    category: "syntax_typo" | "malformed_value" | "wrong_choice_for_system"
            | "missing_required" | "redundant"
    message: str
    fix: Optional[Dict[str, Any]]        # {"tag": "ISPIN", "value": "2"}
```

## What's blocked on what

  - **Variants** (A/B/C/X × standalone/hook shapes) — blocked on
    fixtures locking. Started after Source 1+2+3 land for both engines.
  - **Run + analyze** — blocked on variants.
  - **Source 4** — can be added late; doesn't block variant
    implementation (it's held-out anyway).

## Status — 2026-05-29

  - Design doc finalized, pre-registered hypotheses signed off.
  - Scaffolding (this directory) created.
  - Next: Sarah runs `generate_source1.py` for both engines; hand-labels
    Sources 1+2+3 over ~4-5 days.
  - In parallel: variant scaffolding + engine plug-in stubs.
