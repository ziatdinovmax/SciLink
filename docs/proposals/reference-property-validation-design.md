# Reference-Property Validation — design + generalization benchmark

Status: proposal / design notes. Motivated by the UC2 aqueous Zn-electrolyte
study, but the capability and its benchmark are deliberately system-agnostic.

## Motivation

A SciLink-managed composition series (1.0 M Zn(OTf)₂ in water/ethyl-isopropyl-
sulfone, EIS mole fraction 0.03→0.15) produced a **clean, converged, wrong-
direction** density trend: density fell as EIS was added, where experiment
rises. The absolute values were also ~7–9% low. The error is not a workflow
bug — the series built correctly and ran to equilibrium — it is a force-field
limitation: OpenFF Sage gives the sulfone too large a molar volume, so model-
EIS lightens the mixture instead of densifying it.

The question the use case must answer: **can SciLink reach the right answer
without the experimental data?** UC2 is a *predictive* demonstration, so
catching the error by comparing to the measured density trend is off the table
— that is validation against the answer, not prediction.

## The capability

**Principle (no use-case tokens):** before trusting a novel prediction,
validate the model against independently-known reference quantities of simpler
or constituent cases. You do not report a DFT formation energy from a setup
that cannot reproduce the known lattice constant; you do not report a mixture
property from a force field that cannot reproduce its pure components.

This is a **per-modality validation stage** in the simulate self-refinement
loop (the same slot where `StructureValidatorAgent` / `IncarValidatorAgent`
live inside Generate). It fits the foundation-agent shape: a baseline prompt
per modality, with the *specific* references — which constituents, which
properties, what known values — resolved at runtime from the system, the
active skill, and the LLM's general knowledge. Nothing about density or
sulfones is hardcoded.

Per modality:
- **Classical MD** → pure-component / bulk properties (density, and where the
  goal needs them, transport / structure) before the mixture prediction.
- **DFT** → relaxed cell / known structural or reference quantity before the
  novel energetic.
- **MLIP** → known material properties (lattice, elastic, phonon stability)
  before production.

### Where it does NOT live

- **Not the meta orchestrator.** In the *predictive* framing there is no
  experimental result to bridge to, so the catch is not cross-mode. (If the
  experimental data *were* present, cross-checking sim-trend vs measured-trend
  would be a legitimate meta act via its existing delegate-and-bridge pattern
  — but that is validation, not prediction, and out of scope for the use case.)
- **Not the per-run `RunCritic`.** A single member's 1.12 g/cm³ is not *per se*
  unphysical; there is no single-run signal. The catch is a
  reference-comparison the per-run critic structurally should not carry.

### The check is reasoning-first, not lookup-first

The right question is not "does this match a stored value?" but "**is this
consistent with physics and with what we know about this system?**" The engine
of the check is the LLM reasoning that through. Known reference values, skill
knowledge, and physical laws are all inputs it reasons *with* — not a separate,
superior channel it falls back to.

Two ways the same sulfone error is catchable make the point:
- **Anchored:** compare the simulated pure-sulfone density to a known value and
  see it is ~20% low. `diag2_pure_eis_density` is a hand-run instance; the
  capability makes it an autonomous stage. Strongest when a reference exists —
  it has a ground truth and names the culprit.
- **Reasoned, no stored value needed:** sulfones are dense liquids, so adding
  one should raise the mixture density; the run says it falls, so something is
  wrong. More general — it still works where no convenient reference exists.

So the reasoned channel is **not** demoted to advisory. It is the general
engine; anchoring makes specific judgments quantitative and points at the
culprit. The real hazard is the *other* direction: reasoning can flag a
surprising-but-**correct** result. The answer is not to hold the reasoning
back — it is to make the check *careful*: reasoning proposes, and we confirm it
really is wrong (a cheap targeted check, or the human) before anything is
changed. The benchmark's silent-on-surprising-but-correct cases exist to
calibrate exactly this.

### The fix side (general, built on the existing loop)

Catching a problem is half of "SciLink does it, not the scientist." The other
half is fixing the right part. SciLink already has the loop shape for this
(generate → run → branch on outcome → fix → rerun); today it can fix input
files on an engine error and refine on a shaky run. Two things are missing and
this feature adds them:

1. The physics/consistency check above, so a run that *finishes cleanly but is
   physically wrong* no longer passes.
2. A **reparameterization** fixer, so when the diagnosis points at the force
   field (not the deck) the loop can swap/repair it — the first new fixer of
   its kind.

The intent is general: the loop should route *any* quality problem to a fixer
that can adjust *whichever* part is wrong — structure, force field, or input
files — with the human confirming the change (consistent with the "max quality
with human feedback" use-case framing). Day one is a handful of concrete
problem→fixer pairs (physics-check → FF-fix, end to end on the electrolyte),
structured so more pairs bolt on.

### Extensibility (the load-bearing requirement)

New coverage must be a drop-in, never core surgery — the same bar SciLink holds
for engines and skills:
- **New check** = add domain expectation to a **skill bundle** (what properties
  matter, what known values/behaviors to reason against). The reasoning step is
  one general mechanism; the domain content lives in skills.
- **New fixer** = **register** a fixer keyed to a problem type. The loop asks
  "what is wrong and who handles it?" rather than hardcoding a fixed branch set.

This is not automatic — the seams must be designed so a new check/fixer really
is a drop-in, or it degrades into special cases. Getting that structure right
up front is part of the work, not just making the electrolyte pass.

### Anti-overfitting discipline

The tempting wrong build is "a checker that knows sulfones are dense and
watches density trends" — overfit twice (property + chemistry). Rules:
- No `density` / `sulfone` (or any UC2 token) in the stage's prompts or code.
- References resolved at runtime, per system; the stage *discovers* what to
  validate from the system + goal + skill, not from a baked-in rule.
- Encode the principle in one sentence, not a list of examples from the trace.

## The generalization benchmark

This is **benchmarking the capability** — systematic, tiered, scored —
distinct from UC2, which is a max-quality, human-in-the-loop *use case*. It
lives as its own suite in SciLink-Benchmarks, alongside the critic A/B and
skill-graduation studies.

**Two-dimensional grid.** Difficulty × expected verdict. Every tier carries
both *should-flag* (a real miscalibration) and *should-stay-silent* (a good
model, or an acceptable-error case), because a validator that flags everything
passes sensitivity and fails specificity.

**Scored on three axes:** sensitivity (catches the bad models), specificity
(does not cry wolf on good models / acceptable error / surprising-but-correct
results), localization (names the right quantity / component).

Difficulty is grounded in *what makes a validation call hard*: how well-known
the reference is, how large/unambiguous the error, whether the culprit must be
isolated, whether the failure is in a non-obvious property, and false-positive
traps.

### Enumerated cases (~23; 3–4 flag + 3–4 silent per tier + boundary)

| Tier | ID | Modality | Case | Reference property | Expect |
|---|---|---|---|---|---|
| Easy | E-F1 | MD | TIP3P water | self-diffusion (~2.4× high) | flag |
| Easy | E-F2 | DFT | Ge under PBE | band gap (metallic vs 0.67 eV) | flag |
| Easy | E-F3 | DFT | graphite, PBE no-dispersion | interlayer spacing (~4.3 vs 3.35 Å) | flag |
| Easy | E-F4 | MLIP | foundation MLIP, clearly OOD chemistry | equilibrium volume | flag |
| Easy | E-S1 | MD | TIP4P/2005 water | density + diffusion (both right) | silent |
| Easy | E-S2 | DFT | Si under PBE | lattice constant (~0.7%, acceptable) | silent |
| Easy | E-S3 | MLIP | MACE-MP0 / CHGNet bulk Cu | lattice + bulk modulus | silent |
| Easy | E-S4 | MD | OPLS/GAFF ethanol or benzene | density (well-reproduced) | silent |
| Med | M-F1 | MD | binary organic mix, ONE component's FF off ~10–15% | density + localization | flag |
| Med | M-F2 | MD | non-polarizable ionic liquid | viscosity (2–5× high; transport not structure) | flag |
| Med | M-F3 | DFT | vdW-layered (MoS₂/hBN), PBE no-dispersion | interlayer spacing (moderate) | flag |
| Med | M-F4 | MLIP | poorly-covered material | bulk modulus (~20–30% off) | flag |
| Med | M-S1 | MD | well-parameterized multi-solvent mixture | density (genuinely right) | silent |
| Med | M-S2 | DFT | lattice constant 1% off | lattice constant (acceptable, tolerance) | silent |
| Med | M-S3 | MLIP | moderately-covered material, ~2–3% off | lattice + modulus (acceptable) | silent |
| Med | M-S4 | MD | FF with modest documented-OK density error (~3%) | density (acceptable) | silent |
| Hard | H-F1 | MD | model with correct density, wrong derived property | dielectric / transport (must check right observable) | flag |
| Hard | H-F2 | DFT | correct lattice, inverted relative polymorph stability | relative phase stability | flag |
| Hard | H-F3 | MLIP | soft-phonon instability invisible in static lattice | phonon / elastic stability | flag |
| Hard | H-S1 | MD | water density maximum at 4 °C | non-monotonic density(T) — real | silent |
| Hard | H-S2 | DFT/MLIP | negative thermal expansion (e.g. ZrW₂O₈) | thermal-expansion sign — real | silent |
| Hard | H-S3 | MD | counterintuitive-but-correct coordination result | structure — real | silent |
| Hard | H-B1 | MD | emergent mixture error, ALL constituents validate | mixture property, constituents clean | **fail-to-catch (boundary)** |

**The boundary case (H-B1)** is deliberate: every pure-component check passes,
yet the composed system is wrong (e.g. a combining-rule / cross-interaction or
missing-polarization effect that only appears in the mixture). Reference-
property validation *should* come up empty here. Reporting the regime where
the capability cannot substitute for human/experimental judgment is worth as
much as reporting where it works — it defines when human-in-the-loop stays
required.

## Open questions

- **Always-on vs triggered.** Lean: always-on pure-component validation for
  molecular mixtures ("validate before you predict" shouldn't be conditional);
  a triggered variant risks overfitting the trigger.
- **Fuzzy ground truth.** Whether to add cases where the reference itself is
  uncertain, to stress-test the validator's honesty when it cannot be sure.
- **Breadth.** First pass ~23 cases as above; widen toward the structure-gen
  suite's scale if the capability holds up.

## Relationship to UC2

UC2 is the headline predictive demonstration; this benchmark justifies the
capability UC2 relies on. The intended UC2 arc: run the series → the FF-
validation stage flags the under-dense sulfone from pure-component checks →
swap the sulfone model, rerun → correct increasing trend → *only then* reveal
the experimental spreadsheet as confirmation the prediction was right.
