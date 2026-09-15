---
description: NWChem molecular quantum chemistry — Gaussian-basis DFT, HF, and post-HF (MP2/CCSD(T)/TCE) on finite molecules and ions; geometry optimization, analytic frequencies/thermochemistry, and COSMO/SMD implicit solvation. Charge/multiplicity-aware; handles charged ion pairs that periodic codes cannot.
detect:
  binaries: [nwchem]
  env_vars: [NWCHEM_TOP, NWCHEM_EXECUTABLE, NWCHEM_BASIS_LIBRARY]
  python_modules: []
  guidance: |
    NWChem runs as the `nwchem` binary, typically `nwchem job.nw` (often
    MPI-launched: `mpirun -np N nwchem job.nw`). On HPC clusters it is
    usually only on $PATH after `module load nwchem/<version>`; the basis
    and ECP libraries are located via $NWCHEM_BASIS_LIBRARY (set by the
    module). Detection should treat the `nwchem` binary on $PATH, or any
    of these env vars, as a positive hit. Unlike PySCF (a python module),
    NWChem is an external executable, so python_modules is empty.
---
# NWChem Molecular Quantum Chemistry Skill

## overview

Gaussian-basis-set molecular electronic structure with NWChem, for finite
molecules, ions, and small clusters — NOT periodic solids. This skill covers
deck generation for DFT, Hartree–Fock, and post-HF correlated methods
(MP2, CCSD(T) via the TCE module) together with geometry optimization,
analytic vibrational frequencies and thermochemistry, and implicit solvation
(COSMO / SMD). The target is decks that are physically correct,
charge/multiplicity-consistent, and aligned with standard computational
chemistry practice. NWChem is the right tool where periodic planewave codes
are not: charged species (e.g. carbamate / ammonium ion pairs), solution-phase
free energies, and analytic-Hessian thermochemistry.

## planning

**Method selection (the method is a task-line choice, not a separate code):**
- DFT for geometries, conformers, and routine thermochemistry. Hybrid
  functionals (B3LYP, PBE0) or range-separated/dispersion-corrected
  (ωB97X-D, M06-2X) for main-group thermochemistry of amines/carbamates.
  Always add an empirical dispersion correction (`disp vdw 3` for D3) with
  functionals that lack it.
- HF only as a reference or a starting point for post-HF.
- Post-HF (MP2, then CCSD(T) via TCE) for benchmark reaction energies on a
  few key species once DFT geometries are in hand. CCSD(T)/CBS on the
  optimized geometry is the gold-standard ΔH check; it is expensive, so
  reserve it for the rate-determining comparison.

**Basis set:** def2 family is a sound default — def2-SVP for optimization
and frequencies, def2-TZVP (or def2-TZVPD for anions) for final single-point
energies. Diffuse functions matter for anions (carbamate) and for solvation;
prefer the "D"/aug variants there. Confirm the basis covers every element in
the structure.

**Charge & multiplicity:** set both explicitly from the chemistry. Neutral
closed-shell amine → charge 0, multiplicity 1. Carbamate anion → charge -1;
ammonium cation → charge +1; the neutral ion pair as a whole → charge 0.
Odd electron count requires an open-shell treatment (ODFT/UHF, multiplicity 2).

**Solvation:** for the liquid-phase energetics that actually matter here, add
COSMO or SMD. Gas-phase numbers will misrank ion-pair stability. Use the
solvent dielectric of the working fluid (or water as a reference).

**Workflow staging:** optimize → frequencies → high-level single point, as
three sequential phases sharing a run directory. Frequencies must be computed
at the same level/basis as the optimization, and confirm a true minimum
before trusting thermochemistry.

## implementation

An NWChem deck is a single text file (conventionally `job.nw`) built from
blocks plus one or more `task` lines that execute them in order.

**Deck skeleton:**

    title "amine CO2 carbamate"
    charge 0
    geometry units angstrom
      <element>  x  y  z
      ...
    end
    basis
      * library def2-svp
    end
    dft
      xc b3lyp
      disp vdw 3
      mult 1
    end
    task dft optimize
    task dft freq
    task dft energy

**Implicit solvation (COSMO) block** — append before the task lines for
solution-phase work:

    cosmo
      solvent water
    end

(For SMD, use the `cosmo` block with `do_cosmo_smd .true.`.) COSMO must be
active for the energy/thermo tasks whose solution-phase values you report.

**Charged ion pair example** (carbamate anion):

    charge -1
    ...
    dft
      xc m06-2x
      mult 1
    end
    cosmo
      solvent water
    end
    task dft optimize
    task dft freq

**Post-HF single point** (CCSD(T) via TCE on a DFT geometry):

    basis
      * library def2-tzvp
    end
    tce
      ccsd(t)
    end
    task tce energy

**Per-phase decks:** emit one deck per phase (optimize, freq, single-point)
when staging, each reading the previous phase's geometry, so the refinement
loop can run, assess, and fix each phase independently.

## interpretation

Read a finished run against the calculation's intent, not just exit status.
The deterministic snapshot (`nwchem_output.snapshot_run`, via cclib) surfaces
the fields below; judge them against the request.

- **SCF convergence** (cclib `scfenergies`): the SCF energy must be present
  and converged for any result to be meaningful. A missing/last-iteration
  energy with no convergence flag means the SCF did not finish.
- **Geometry convergence** (cclib `optdone`): for an optimization this must be
  True; if False the geometry is still moving and the energy is not a minimum.
- **Imaginary frequencies** (cclib `vibfreqs`, negative values): a genuine
  minimum has zero; one imaginary mode is a transition state; several usually
  mean a bad geometry or too-loose optimization. Thermochemistry is only valid
  at a true minimum.
- **Thermochemistry** (cclib `enthalpy`, `freeenergy`, in Hartree): use these
  for ΔH / ΔG of reaction. Reaction ΔH = Σ products − Σ reactants; for the
  carbamate equilibrium include amine, CO2, carbamate, ammonium, and ion pair.
- **Basis/method warnings:** watch for "basis not found for element", linear
  dependence in the basis (drop redundant diffuse functions), or SCF stalling
  on anions (add diffuse functions, tighten/raise iterations).

When a result contradicts the chemistry (e.g. a "minimum" with imaginary
modes, or gas-phase ranking that flips a known liquid/solid trend), distrust
the inputs — charge/multiplicity, missing solvation, basis — before the number.

## validation

**Pre-run checks (before submitting a deck):**

- A `basis` block is present and the chosen basis set exists for every element
  in the geometry (def2 families cover H–Rn; verify for any heavy/unusual
  atoms and add ECPs where required).
- `charge` is set and consistent with the species; spin `mult` is consistent
  with the electron count (even electrons → singlet unless deliberately
  open-shell; odd electrons → multiplicity 2 and an unrestricted method).
- The method block matches the task line: `dft` block ↔ `task dft ...`,
  `tce`/`mp2` block ↔ `task tce/mp2 ...`. A `task dft` with no `dft` block, or
  a post-HF task with no correlated method block, will fail.
- For solution-phase requests, a well-formed `cosmo` block is present and the
  solvent is named; the thermo/energy tasks run with it active.
- For thermochemistry, a `task ... freq` follows the optimization at the same
  level/basis, and the optimization task precedes it.
- Geometry units are declared and coordinates are sane (no overlapping atoms,
  reasonable bond lengths) — NWChem will run a garbage geometry to a garbage
  number.
