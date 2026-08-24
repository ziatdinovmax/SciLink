---
description: GROMACS classical molecular dynamics — .mdp run-control files for biomolecular and soft-matter systems (proteins, membranes, solvated small molecules) and condensed phases, with the OPLS-AA / AMBER / CHARMM / GROMOS force fields, PME electrostatics, and thermostats/barostats for NVT/NPT ensembles.
outputs:
  trajectory: [xtc, trr]
  thermo_log: [md.log, ener.edr]
detect:
  binaries: [gmx, gmx_mpi, gmx_d, mdrun, mdrun_mpi]
  env_vars: [GMXBIN, GMXDATA, GMXLIB]
  python_modules: []
  guidance: |
    GROMACS ships one driver binary, `gmx` (double-precision `gmx_d`,
    MPI build `gmx_mpi`), with sub-commands: `gmx grompp` preprocesses
    an .mdp + topology (.top) + coordinates (.gro/.pdb) into a run
    input (.tpr), and `gmx mdrun -deffnm md` runs it. Older/threaded
    builds expose a standalone `mdrun`. On HPC the binary appears after
    `module load gromacs/<version>`, which also sets $GMXBIN/$GMXDATA.
    Detection should treat `gmx`/`gmx_mpi`/`mdrun` on $PATH, or any of
    these env vars, as a positive hit. GROMACS has no importable Python
    runtime surface, so python_modules is empty.
---
# GROMACS Classical Molecular Dynamics Skill

## Overview

Classical atomistic dynamics with GROMACS, the de-facto engine for biomolecular
and soft-matter MD: solvated proteins, nucleic acids, lipid membranes, and
solvated small molecules, as well as bulk liquids. The primary artifact this
skill generates is the **`.mdp` run-control file** — the run parameters
(integrator, ensemble, thermostat/barostat, cutoffs, electrostatics, output
cadence). The topology (`.top`/`.itp`) and coordinates (`.gro`/`.pdb`) are the
inputs GROMACS consumes alongside it; `gmx grompp` assembles all three into a
`.tpr`, and `gmx mdrun` executes it. GROMACS is the natural MD engine where the
force field is a biomolecular/soft-matter one (OPLS-AA, AMBER, CHARMM, GROMOS)
and PME electrostatics with explicit solvent are expected.

**Scope:** this bundle covers **single-shot** and **staged** (min → NVT → NPT →
production) generation. **Fan-out parameter sweeps** (one independent run per
value of a swept variable) are **not yet supported** — the bundle ships no
`expand_parameter_sweep`, so a sweep request degrades to a single run (the agent
logs a warning). Express a multi-value study as an explicit staged campaign for
now; a sweep expander is the follow-up.

## Planning

**Ensemble & phases:** a production run is staged — energy minimization
(`integrator = steep`), then NVT equilibration (temperature coupling on, position
restraints optional), then NPT equilibration (add pressure coupling), then the
NPT/NVT production run. Each phase is its own `.mdp`; later phases continue from
the previous checkpoint (`gmx mdrun -cpi`).

**Thermostat:** `tcoupl = v-rescale` (velocity-rescaling; the robust,
canonical-sampling default) for equilibration and production. `nose-hoover` for
a rigorously correct production ensemble once equilibrated. Set `tc-grps` to
couple solute and solvent separately (e.g. `Protein Non-Protein` or `System`),
`ref-t` to the target temperature per group, and `tau-t` (0.1 ps typical).

**Barostat (NPT only):** `pcoupl = c-rescale` (or `parrinello-rahman` for
production), `ref-p = 1.0` bar, `tau-p = 2.0`, and a `compressibility` (4.5e-5
bar⁻¹ for water). Do NOT set pressure coupling for an NVT run.

**Timestep:** `dt = 0.002` ps (2 fs) with `constraints = h-bonds` (LINCS);
`dt = 0.001` (1 fs) if bonds are unconstrained. `nsteps` sets the run length
(25 ns = 12,500,000 steps at 2 fs).

**Electrostatics & cutoffs:** `coulombtype = PME` for explicit-solvent systems,
`rcoulomb = 1.0`, `rvdw = 1.0` nm, `cutoff-scheme = Verlet`. PME is expected for
biomolecular/solvated work; reaction-field only for special cases.

**Force field:** the topology is written for a specific force field (OPLS-AA,
AMBER, CHARMM36, GROMOS). The `.mdp` must be consistent with it — CHARMM needs
`vdw-modifier = force-switch` with `rvdw-switch = 1.0`, `rvdw = 1.2`; OPLS/AMBER
use plain cutoffs at 1.0 nm.

## Implementation

The generated artifact is a complete `.mdp`. Skeleton for an NPT production run:

    ; run control
    integrator      = md
    dt              = 0.002
    nsteps          = 12500000        ; 25 ns
    ; output control
    nstxout-compressed = 5000         ; trajectory every 10 ps
    nstenergy       = 5000
    nstlog          = 5000
    ; neighbor searching / cutoffs
    cutoff-scheme   = Verlet
    rlist           = 1.0
    rvdw            = 1.0
    rcoulomb        = 1.0
    ; electrostatics
    coulombtype     = PME
    fourierspacing  = 0.12
    ; temperature coupling
    tcoupl          = v-rescale
    tc-grps         = System
    tau-t           = 0.1
    ref-t           = 300
    ; pressure coupling (NPT)
    pcoupl          = c-rescale
    pcoupltype      = isotropic
    tau-p           = 2.0
    ref-p           = 1.0
    compressibility = 4.5e-5
    ; bonds
    constraints     = h-bonds
    constraint-algorithm = lincs

**Energy minimization** deck (a distinct phase):

    integrator      = steep
    emtol           = 1000.0
    emstep          = 0.01
    nsteps          = 50000
    cutoff-scheme   = Verlet
    coulombtype     = PME
    rvdw            = 1.0
    rcoulomb        = 1.0

**Per-phase decks:** emit one `.mdp` per phase (min → NVT → NPT → production).
NVT drops the `pcoupl` block; the production deck uses `parrinello-rahman` /
`nose-hoover` once the system is equilibrated. `gen-vel = yes` with `gen-temp`
seeds velocities on the first NVT phase only.

**Execution:** `gmx grompp -f md.mdp -c conf.gro -p topol.top -o md.tpr` then
`gmx mdrun -deffnm md`.

## Interpretation

Read a finished run against the request, not just whether `mdrun` exited.

- **Energy/temperature/pressure** (`.edr`, via `gmx energy`): temperature must
  sit at `ref-t` and pressure average near `ref-p` for an NPT run; a large drift
  means the system never equilibrated.
- **Density** (NPT): should plateau at the physical value for the fluid; a
  falling or unphysical density signals a barostat or topology problem.
- **Constraint/energy warnings** (`md.log`): "LINCS warnings", "1-4 interaction
  not within cutoff", or blowing-up energies indicate a bad start structure or
  too-large a timestep — minimize/equilibrate more, not a longer production run.
- **Trajectory** (`.xtc`/`.trr`): confirm the cadence recorded the observable
  the goal needs (e.g. frequent stress/velocity output for transport
  properties).

## Validation

**Pre-run checks (before `grompp`):**

- `integrator` is set (`md`, `md-vv`, or `steep`/`cg` for minimization), and
  `dt` and `nsteps` are present and physical (`dt` ≤ 0.002 with `constraints =
  h-bonds`, ≤ 0.001 without).
- Temperature coupling is well-formed: `tcoupl` names a real thermostat, and the
  number of `ref-t` and `tau-t` values matches the number of `tc-grps`.
- Pressure coupling is present for an NPT request and ABSENT for NVT; when
  present, `ref-p`, `tau-p`, and `compressibility` are set.
- `cutoff-scheme = Verlet`, and `rvdw`/`rcoulomb` are set; `coulombtype = PME`
  for an explicit-solvent/biomolecular system.
- The cutoffs are consistent with the force field (CHARMM force-switch at
  1.2 nm; OPLS/AMBER plain at 1.0 nm).
- `constraints` and `constraint-algorithm` are consistent with `dt`.
- For thermochemistry/transport, the output cadence (`nstxout-compressed`,
  `nstenergy`) records the quantity the goal requires.
