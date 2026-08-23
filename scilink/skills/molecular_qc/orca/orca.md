---
description: ORCA molecular quantum chemistry — Gaussian-basis DFT, HF, and post-HF (MP2, DLPNO-CCSD(T)) on finite molecules and ions; geometry optimization, analytic frequencies/thermochemistry, and CPCM/SMD implicit solvation. Charge/multiplicity-aware; RI/RIJCOSX-accelerated; handles charged ion pairs that periodic codes cannot.
detect:
  binaries: [orca]
  env_vars: []
  python_modules: []
  guidance: |
    ORCA runs as the `orca` binary, invoked `orca orca.inp > orca.out`. It
    MUST be called by its FULL PATH (e.g. `/opt/orca/orca orca.inp`) so ORCA
    can locate its own sub-executables (orca_scf, orca_gtoint, ...) in the
    same directory; a bare `orca` on $PATH works only if the whole install
    dir is on $PATH. Parallelism is internal — set `%pal nprocs N end` in the
    deck and still launch the single `orca` command (do NOT wrap it in
    `mpirun`; ORCA spawns MPI itself and needs its bundled OpenMPI on
    $LD_LIBRARY_PATH). On HPC it is usually available after
    `module load orca/<version>`. Detection should treat the `orca` binary
    (on $PATH or a known install dir) as a positive hit. ORCA has no standard
    environment variables and is an external executable, so env_vars and
    python_modules are empty.
---
# ORCA Molecular Quantum Chemistry Skill

## overview

Gaussian-basis-set molecular electronic structure with ORCA, for finite
molecules, ions, and small clusters — NOT periodic solids. This skill covers
deck generation for DFT, Hartree–Fock, and post-HF correlated methods
(MP2, and DLPNO-CCSD(T) — ORCA's near-linear-scaling coupled cluster) together
with geometry optimization, analytic vibrational frequencies and
thermochemistry, and implicit solvation (CPCM / SMD). The target is decks that
are physically correct, charge/multiplicity-consistent, and aligned with
standard computational-chemistry practice. ORCA is the right tool where periodic
planewave codes are not: charged species (e.g. carbamate / ammonium ion pairs),
solution-phase free energies, and analytic-Hessian thermochemistry — and its
DLPNO-CCSD(T) makes gold-standard energetics affordable on sizeable molecules.

## planning

**Method selection (the method is a keyword-line choice, not a separate code):**
- DFT for geometries, conformers, and routine thermochemistry. Hybrid
  functionals (B3LYP, PBE0) or range-separated/dispersion-corrected
  (ωB97X-D3, M06-2X) for main-group thermochemistry of amines/carbamates.
  Add an empirical dispersion correction (`D3BJ` or `D4`) with functionals that
  lack it — put it directly on the `!` line (`! B3LYP D3BJ ...`).
- HF only as a reference or a starting point for post-HF.
- Post-HF: MP2, then `DLPNO-CCSD(T)` for benchmark reaction energies on a few
  key species once DFT geometries are in hand. DLPNO-CCSD(T)/CBS on the
  optimized geometry is the gold-standard ΔH check and is far cheaper than
  canonical CCSD(T); reserve it for the rate-determining comparison and pair it
  with a correlation-fitting aux basis (see basis).

**Basis set:** the def2 family is a sound default — def2-SVP for optimization
and frequencies, def2-TZVP (or def2-TZVPD for anions) for final single-point
energies. Diffuse functions matter for anions (carbamate) and for solvation;
prefer the "D"/ma- variants there. ORCA leans on the resolution of the identity:
add `def2/J` as the Coulomb-fitting auxiliary basis for RI-J / RIJCOSX (standard
with hybrids: `! RIJCOSX def2/J`), and a `/C` correlation-fitting basis
(e.g. `def2-TZVP/C`) for MP2 / DLPNO. Confirm the basis covers every element.

**Charge & multiplicity:** set both explicitly on the coordinate line
(`* xyz <charge> <mult>`), from the chemistry. Neutral closed-shell amine →
`* xyz 0 1`. Carbamate anion → charge -1; ammonium cation → charge +1; the
neutral ion pair as a whole → charge 0. An odd electron count requires an
open-shell treatment (`! UKS`/`! UHF`, multiplicity 2).

**Solvation:** for the liquid-phase energetics that actually matter here, add
CPCM or SMD. Gas-phase numbers will misrank ion-pair stability. Use the solvent
of the working fluid (or water as a reference): `! CPCM(water)`, or SMD via a
`%cpcm smd true SMDsolvent "water" end` block.

**Workflow staging:** optimize → frequencies → high-level single point. ORCA can
combine `! Opt Freq` in one run; when staging as separate phases, each reads the
previous phase's geometry (e.g. `* xyzfile <charge> <mult> prev.xyz`).
Frequencies must be computed at the same level/basis as the optimization, and a
true minimum (no imaginary modes) confirmed before trusting thermochemistry.

## implementation

An ORCA deck is a single text file (conventionally `orca.inp`): one or more
`!` "simple-keyword" lines, optional `%block ... end` sections for detailed
settings, and a coordinate block.

**Deck skeleton:**

    ! B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF Opt Freq
    %maxcore 3000
    %pal nprocs 8 end
    * xyz 0 1
      <element>  x  y  z
      ...
    *

- The `!` line carries method, dispersion, basis, RI/aux basis, SCF/opt
  tightness, and run types (`Opt`, `Freq`/`NumFreq`; a bare deck with none is a
  single-point energy).
- `%maxcore` is memory **per core** in MB; `%pal nprocs N end` sets core count.

**Implicit solvation** — add to the `!` line for CPCM, or a block for SMD:

    ! CPCM(water)
    # ...or SMD:
    %cpcm
      smd true
      SMDsolvent "water"
    end

Solvation must be active for the energy/thermo phases whose solution-phase
values you report.

**Charged ion pair example** (carbamate anion, solution-phase optimization):

    ! M06-2X D3ZERO def2-TZVPD RIJCOSX def2/J CPCM(water) TightSCF Opt Freq
    %pal nprocs 8 end
    * xyz -1 1
      ...
    *

**Post-HF single point** (DLPNO-CCSD(T) on a DFT geometry):

    ! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C RIJCOSX def2/J TightSCF
    %pal nprocs 8 end
    * xyzfile 0 1 opt.xyz

**Reading a previous phase's geometry:** `* xyzfile <charge> <mult> prev.xyz`
points at the `.xyz` ORCA writes after an optimization, so per-phase decks chain
cleanly and the refinement loop can run, assess, and fix each phase
independently.

## interpretation

Read a finished run against the calculation's intent, not just exit status.
The deterministic snapshot (`orca_output.snapshot_run`, via cclib) surfaces the
fields below; judge them against the request. ORCA prints
`****ORCA TERMINATED NORMALLY****` on a clean finish — its absence means the run
crashed even if partial output parses.

- **SCF convergence** (cclib `scfenergies`): the SCF energy must be present and
  converged for any result to be meaningful. `SCF NOT CONVERGED AFTER ...` in
  the log means it did not finish.
- **Geometry convergence** (cclib `optdone`): for an optimization this must be
  True; if False the geometry is still moving and the energy is not a minimum.
- **Imaginary frequencies** (cclib `vibfreqs`, negative values): a genuine
  minimum has zero; one imaginary mode is a transition state; several usually
  mean a bad geometry or too-loose optimization. Thermochemistry is only valid
  at a true minimum.
- **Thermochemistry** (cclib `enthalpy`, `freeenergy`, in Hartree): use these
  for ΔH / ΔG of reaction. Reaction ΔH = Σ products − Σ reactants; for the
  carbamate equilibrium include amine, CO2, carbamate, ammonium, and ion pair.
- **Basis/method warnings:** watch for a missing basis or auxiliary basis
  (`could not find the basis`), an inconsistent charge/multiplicity
  (`... multiplicity ... is odd`), or an unknown keyword
  (`UNRECOGNIZED OR DUPLICATED KEYWORD`).

When a result contradicts the chemistry (a "minimum" with imaginary modes, or a
gas-phase ranking that flips a known liquid/solid trend), distrust the inputs —
charge/multiplicity, missing solvation, basis/aux basis — before the number.

## validation

**Pre-run checks (before submitting a deck):**

- The `!` line names a method **and** a basis, and the basis set exists for
  every element in the geometry (def2 families cover H–Rn; verify for any
  heavy/unusual atoms and add ECPs / a covering basis where required).
- When RI/RIJCOSX or a correlated method is requested, the matching auxiliary
  basis is present: `def2/J` for RI-J / RIJCOSX Coulomb fitting, a `/C` basis
  (e.g. `def2-TZVP/C`) for MP2 / DLPNO correlation fitting.
- The coordinate line sets `<charge>` and `<mult>` consistently with the
  species: even electron count → singlet unless deliberately open-shell; odd
  electrons → multiplicity 2 with an unrestricted method (`UKS`/`UHF`).
- The requested run type matches the intent: `Opt` for a geometry, a `Freq`
  after the optimization (same level/basis) for thermochemistry, neither for a
  single point.
- For solution-phase requests, a CPCM keyword or SMD block is present and the
  solvent is named; the thermo/energy phases run with it active.
- The coordinate block is well-formed: `* xyz <charge> <mult>` ... closed with a
  lone `*` (or `* xyzfile <charge> <mult> file.xyz`), and coordinates are sane
  (no overlapping atoms, reasonable bond lengths) — ORCA will run a garbage
  geometry to a garbage number.
- Resource blocks are reasonable: `%maxcore` is per-core memory in MB (not
  total), and `%pal nprocs N end` matches the cores the job will actually get.
