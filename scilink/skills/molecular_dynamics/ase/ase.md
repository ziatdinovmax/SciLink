---
description: ASE MD / relaxation runner for a deployed MLIP potential — executes a generated Python driver script (run_md.py / run_relax.py) for systems (e.g. metals) that have no classical force field. The universal MLIP execution path, not a general-purpose classical MD engine.
outputs:
  trajectory: [traj]
  thermo_log: [thermo.log]
inputs:
  deck: ["run_md.py", "run_relax.py"]
detect:
  python_modules: [ase]
  guidance: |
    ASE here is a Python driver, not a standalone MD engine binary. It is the
    universal fallback for EXECUTING a deployed MLIP potential when no classical
    engine fits (issue #429) — a Pt/metal system, for example, which OpenFF and
    classical force fields cannot parameterize. It is deliberately NOT offered
    as a general user-available MD engine in routing: the router injects "ase"
    only for an MLIP-only audience (no classical MD engine installed), and the
    pipeline re-points to this bundle when potential selection chooses an MLIP
    after routing. Everyday molecular systems still route to lammps / gromacs.
health:
  # Gross physical-sanity band for a finished ASE MD run — a tripwire for a
  # non-physical result (e.g. a thermostat that ran away), NOT an accuracy
  # check. The ASE MD driver logs temperature + energies only (there is no
  # density / pressure / volume column — see _ase_runner.py), so only
  # `temperature` is declared here. Keep the bounds loose: only a genuinely
  # broken run should trip them. Accuracy is judged downstream against a
  # reference, never here. Read by the `read_health_observable` hook.
  - observable: temperature   # K
    min: 1.0                  # a dead / frozen run; a real MD setpoint is >> 1 K
    max: 50000.0              # a thermostat blow-up; far above any physical setpoint
---

## Overview

ASE (the Atomic Simulation Environment) is used here as a backend-agnostic
driver for a *deployed MLIP potential*. SciLink's ASE runner
(`scilink/agents/sim_agents/_ase_runner.py`) emits a single, self-contained
Python script that reads a structure, attaches the potential's ASE calculator,
and runs either molecular dynamics or a geometry + cell relaxation. One template
covers every MLIP backend (MACE, CHGNet, NequIP, …); the only backend-specific
parts are data on the deployed potential's calculator spec.

Key characteristics:
- Input: a generated Python script (`run_md.py` for MD, `run_relax.py` for
  relaxation). There is no separate engine input deck.
- Launched as `<python> <script>` — no external engine binary. The run uses the
  same Python environment SciLink runs in, so the MLIP backend the script
  imports is the one that is installed.
- MD output: an ASE trajectory (`traj`) plus a thermo log (`thermo.log`) with a
  `# step time(ps) PE(eV) KE(eV) T(K)` header — temperature and energies only.
- Relaxation output: a relaxed structure (`relaxed.xyz`) plus `relax_result.json`
  carrying `lattice_constant_A`, `final_energy_eV`, and `converged`.

## When this engine is used

This is the execution surface for the MLIP path. The router does not pick "ase"
for an ordinary system — it picks a scale (`molecular_dynamics`) and a classical
engine (lammps / gromacs) where one is available. The potential-selection step,
which runs *after* routing once the structure and species are known, decides
whether the interatomic potential is a classical force field or an MLIP. When it
chooses an MLIP, input generation runs the ASE path and the pipeline re-points
execution (the health gate, the RunCritic grounding, and the launch command) at
this bundle. The router only injects "ase" directly for an MLIP-only audience
(e.g. a `mace-torch` install with no classical MD engine on the machine).

## Health gate

The `health:` block above declares a single, loose temperature band. It is a
gross physical-sanity tripwire — a run whose temperature settled near zero (a
dead/frozen system) or exploded to tens of thousands of kelvin (a thermostat
instability) is non-physical regardless of the research goal. It is NOT an
accuracy check: whether a 300 K setpoint equilibrated to 298 vs 305 K is a
question answered downstream against a reference, never here. Because the ASE MD
driver logs no density / pressure / volume column, the reader serves only
temperature (and the energies); a request for an unlogged observable returns
`None` so the gate skips it rather than fabricating a verdict.
