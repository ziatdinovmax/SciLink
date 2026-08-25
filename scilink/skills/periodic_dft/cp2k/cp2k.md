---
description: CP2K (Quickstep) DFT input generation — nested &SECTION/&END input for the Gaussian and plane-waves (GPW/GAPW) method, covering the &GLOBAL run type, &FORCE_EVAL/&DFT electronic structure (&QS, &MGRID cutoff, &SCF, &XC functional + dispersion, &KPOINTS, DFT+U, Fermi smearing), and the &SUBSYS structure (&CELL, &COORD, per-element &KIND with GTH basis + pseudopotential), for metals, semiconductors, correlated oxides, slabs, and molecules.
detect:
  binaries: [cp2k, cp2k.psmp, cp2k.popt, cp2k.ssmp, cp2k.sopt]
  env_vars: [CP2K_DATA_DIR]
  python_modules: []
  guidance: |
    CP2K's Quickstep engine is the `cp2k` binary, usually built with a
    parallelism suffix: `cp2k.psmp` (MPI+OpenMP), `cp2k.popt` (MPI),
    `cp2k.ssmp` (OpenMP), `cp2k.sopt` (serial). On HPC clusters it is
    typically on $PATH only after `module load cp2k/<version>`, which also
    sets $CP2K_DATA_DIR pointing at the basis-set and pseudopotential
    libraries (BASIS_MOLOPT, GTH_POTENTIALS). Detection treats any `cp2k[.*]`
    binary on $PATH, or $CP2K_DATA_DIR, as a positive hit.
---
# CP2K Input Generation Skill

## overview

Density Functional Theory with CP2K's Quickstep module, which uses the Gaussian
and plane-waves (GPW) method --- Gaussian basis functions for the orbitals and an
auxiliary plane-wave grid for the density. This skill builds a single CP2K input
file: the nested `&SECTION ... &END SECTION` blocks covering `&GLOBAL` (the run
type), `&FORCE_EVAL/&DFT` (electronic structure: `&QS`, `&MGRID`, `&SCF`, `&XC`,
`&KPOINTS`), and `&SUBSYS` (the structure: `&CELL`, `&COORD`, and one `&KIND` per
element carrying its Gaussian basis set and GTH pseudopotential). It covers metals,
semiconductors, insulators, correlated oxides, surface slabs, and molecules. The
goal is input that is physically correct and consistent with standard CP2K
practice. **Energies and `CUTOFF` are in Rydberg; `ELECTRONIC_TEMPERATURE` is in
Kelvin.**

## non-negotiable constraints (decide these FIRST)

CP2K couples the SCF solver, the Brillouin-zone sampling, and the functional in
ways that are hard errors, not preferences. A deck that violates any of these
**aborts immediately** — resolve them before writing anything else, and do not let
the generic "a periodic solid takes a k-mesh" instinct override rules 2–3:

1. **`&OT` ⟺ Gamma; `&DIAGONALIZATION` ⟺ k-points.** Orbital transformation
   **cannot** run with a `&KPOINTS` mesh (`OT not possible with kpoint
   calculations`). Pick exactly one path per run.
2. **DFT+U does NOT support k-points.** A `+U` run **must be Gamma-point** — omit
   `&KPOINTS` entirely (build a supercell for sampling) and use `&OT`. A `&KPOINTS`
   mesh on a `+U` deck aborts (`Method not implemented for k-points`).
3. **Hybrid / exact-exchange functionals do NOT support k-points and require
   `&OT`.** Run them **Gamma-point + `&OT`, no ADMM** for a small cell. A
   `&DIAGONALIZATION` hybrid, or ADMM MO-purification without OT, aborts.
4. **`POTENTIAL GTH-PBE-q<N>` must use a valence that exists for the element**
   (table under *planning*) — a guessed `q` aborts (`atomic potential ... not
   found`).

**Method decision, once and for all:**

| system | k-points | SCF solver | smearing |
|---|---|---|---|
| metal (plain GGA) | `&KPOINTS` mesh | `&DIAGONALIZATION` | `&SMEAR` + `ADDED_MOS` |
| semiconductor/insulator (plain GGA) | `&KPOINTS` mesh | `&DIAGONALIZATION` | none |
| **DFT+U (any)** | **none — Gamma** | **`&OT`** | none |
| **hybrid (any)** | **none — Gamma** | **`&OT`** | none |
| molecule / large cell | none — Gamma | `&OT` | none |

## planning

**Method:** `METHOD Quickstep` with GPW is the default. Use GAPW (`&QS METHOD
GAPW`) only when all-electron accuracy or hard pseudopotentials demand it. A CP2K
calculation always needs, per element, a Gaussian **basis set** and a matching
**GTH pseudopotential** consistent with the functional.

**Run type (`&GLOBAL RUN_TYPE`):** `ENERGY` for a single point, `ENERGY_FORCE` when
forces are wanted, `GEO_OPT` to relax atoms, `CELL_OPT` to relax the cell, `MD` for
dynamics. A relaxation additionally needs a `&MOTION/&GEO_OPT` (or `&CELL_OPT`)
block.

**Functional (`&XC`):** `&XC_FUNCTIONAL PBE` (or `PBESOL`, `BLYP`) for GGA;
`&XC_FUNCTIONAL PBE0` / `HSE06` for hybrids, which require an `&HF` block
(exact-exchange `FRACTION`, an `&INTERACTION_POTENTIAL`, and for HSE06 a screening
`OMEGA` and `POTENTIAL_TYPE SHORTRANGE`). SCAN and other meta-GGAs go through
`&XC_FUNCTIONAL`. Add dispersion with a `&VDW_POTENTIAL` block (e.g. `&PAIR_POTENTIAL
TYPE DFTD3`) for functionals that lack it.

**Spin and magnetism:** for any open-shell or magnetic system set `&DFT LSD` (or
`UKS T`) to make the calculation spin-polarized. Seed the magnetic state with a
`MAGNETIZATION` on the relevant `&KIND` (unpaired electrons per atom of that
species), or a `MULTIPLICITY` in `&FORCE_EVAL/&DFT`. An antiferromagnet needs two
`&KIND` entries for the same element (e.g. `Ni_up`/`Ni_down`) with opposite
`MAGNETIZATION`, referenced from `&COORD`.

**Hubbard U (DFT+U):** for correlated 3d/4f oxides, enable `&DFT PLUS_U_METHOD
MULLIKEN` and add a `&DFT_PLUS_U` block inside the correlated element's `&KIND`
with the angular momentum `L` (2 for d, 3 for f) and `U_MINUS_J` in eV.

**Smearing (metals):** a metal needs electronic smearing --- a `&SCF/&SMEAR` block
with `METHOD FERMI_DIRAC` and `ELECTRONIC_TEMPERATURE` (~300 K), together with
extra unoccupied states (`ADDED_MOS`) and, usually, `&MIXING METHOD
BROYDEN_MIXING`. Insulators and semiconductors take no smearing.

**k-points:** periodic solids need a `&KPOINTS` block (`SCHEME MONKHORST-PACK k k
k`); a metal needs a dense mesh, a semiconductor a moderate one. Molecules and
very large cells run Gamma-only (omit `&KPOINTS`). **Critical CP2K limitation:**
the k-point code does NOT support `DFT+U` or hybrid/exact-exchange functionals
(both abort with `Method not implemented for k-points`). For a **DFT+U** or
**hybrid** calculation, do NOT emit a `&KPOINTS` block — run **Gamma-point on a
supercell** built large enough to sample the physics (CP2K's GPW cost is nearly
linear in cell size, so a Gamma supercell is the intended route, not a k-mesh on
the primitive cell).

**SCF method — OT and k-points are mutually exclusive; pick one path.** This is
the single most common CP2K setup error. `&OT` (orbital transformation) **cannot**
run with a `&KPOINTS` mesh (`OT not possible with kpoint calculations`), and
hybrids/exact exchange **require** `&OT`. So the SCF method is decided by whether
the run is k-sampled or Gamma-point:

- **k-sampled run** (any `&KPOINTS` mesh — a plain-GGA metal or semiconductor):
  use **`&DIAGONALIZATION`**, never `&OT`. Metals add `&SMEAR` + `ADDED_MOS`;
  semiconductors/insulators do not.
- **Gamma-point run** (no `&KPOINTS` — a hybrid, a DFT+U system, a molecule, or a
  large cell): use **`&OT`** (`MINIMIZER DIIS`, `PRECONDITIONER
  FULL_SINGLE_INVERSE`, with an `&OUTER_SCF`), never `&DIAGONALIZATION` with
  smearing.

A **hybrid** therefore must be **Gamma-point + OT** (a `&DIAGONALIZATION` hybrid,
or ADMM with default MO-based purification without OT, aborts: `ADMM: MO-based
purification requires OT`); keep it **OT, Gamma, no ADMM** unless the cell is
large. A **DFT+U** run is likewise Gamma-point + OT.

**Basis and pseudopotential:** each `&KIND` names a `BASIS_SET` (e.g.
`DZVP-MOLOPT-SR-GTH`) and a `POTENTIAL` (e.g. `GTH-PBE-q4`) consistent with the
functional; the `&DFT` block points at the library files via `BASIS_SET_FILE_NAME`
and `POTENTIAL_FILE_NAME`. The `q<N>` suffix is the valence electron count and
**must match an entry that actually exists in `GTH_POTENTIALS` for that element**
— a wrong `q` aborts the run (`atomic potential <GTH-PBE-qN> ... not found`), so
do NOT guess it. Use the standard GTH-PBE valences:

- main group, full valence: H q1, Li q3, B q3, C q4, N q5, O q6, F q7, Na q9,
  Mg q10, Al q3, Si q4, P q5, S q6, Cl q7, K q9, Ca q10, Ga q13, Ge q4, As q5,
  Se q6, Br q7;
- transition metals, semi-core: Sc q11, Ti q12, V q13, Cr q14, Mn q15, Fe q16,
  Co q17, Ni q18, Cu q11, Zn q12, Zr q12, Nb q13, Mo q14, Ru q16, Rh q17,
  Pd q18, Ag q11, Cd q12, Pt q18, Au q11.

## implementation

A CP2K input is nested `&SECTION`/`&END` blocks. Skeleton for a single-point PBE
energy of a semiconductor:

    &GLOBAL
      PROJECT germanium
      RUN_TYPE ENERGY
      PRINT_LEVEL MEDIUM
    &END GLOBAL
    &FORCE_EVAL
      METHOD Quickstep
      &DFT
        BASIS_SET_FILE_NAME BASIS_MOLOPT
        POTENTIAL_FILE_NAME GTH_POTENTIALS
        &MGRID
          CUTOFF 400
          REL_CUTOFF 50
        &END MGRID
        &QS
          EPS_DEFAULT 1.0E-10
        &END QS
        &SCF
          SCF_GUESS ATOMIC
          EPS_SCF 1.0E-6
          MAX_SCF 50
          &DIAGONALIZATION      # k-sampled run -> diagonalization, NOT &OT
            ALGORITHM STANDARD
          &END DIAGONALIZATION
        &END SCF
        &XC
          &XC_FUNCTIONAL PBE
          &END XC_FUNCTIONAL
        &END XC
        &KPOINTS
          SCHEME MONKHORST-PACK 4 4 4
        &END KPOINTS
      &END DFT
      &SUBSYS
        &CELL
          A  5.658 0.000 0.000
          B  0.000 5.658 0.000
          C  0.000 0.000 5.658
        &END CELL
        &COORD
          Ge 0.0 0.0 0.0
          Ge 0.25 0.25 0.25
        &END COORD
        &KIND Ge
          BASIS_SET DZVP-MOLOPT-SR-GTH
          POTENTIAL GTH-PBE-q4
        &END KIND
      &END SUBSYS
    &END FORCE_EVAL

**Ferromagnetic metal** --- add spin, a moment, smearing, and extra states:

    &DFT
      LSD
      ...
      &SCF
        ADDED_MOS 20
        &SMEAR
          METHOD FERMI_DIRAC
          ELECTRONIC_TEMPERATURE 300
        &END SMEAR
        &MIXING
          METHOD BROYDEN_MIXING
          ALPHA 0.4
        &END MIXING
      &END SCF
      ...
    &KIND Co
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q17
      MAGNETIZATION 1.6
    &END KIND

**Antiferromagnet with Hubbard U** (e.g. MnO) --- split the correlated element and
add DFT+U. Run this **Gamma-point on a supercell** (no `&KPOINTS`) — CP2K's DFT+U
does not run with a k-mesh:

    &DFT
      LSD
      PLUS_U_METHOD MULLIKEN
      ...
    &KIND Mn_up
      ELEMENT Mn
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q15
      MAGNETIZATION 5.0
      &DFT_PLUS_U
        L 2
        U_MINUS_J [eV] 4.0
      &END DFT_PLUS_U
    &END KIND
    &KIND Mn_down
      ELEMENT Mn
      ...
      MAGNETIZATION -5.0
      &DFT_PLUS_U
        L 2
        U_MINUS_J [eV] 4.0
      &END DFT_PLUS_U
    &END KIND

**Geometry relaxation** --- set the run type and add a motion block:

    &GLOBAL
      RUN_TYPE GEO_OPT
    &END GLOBAL
    ...
    &MOTION
      &GEO_OPT
        OPTIMIZER BFGS
        MAX_ITER 200
        MAX_FORCE 1.0E-4
      &END GEO_OPT
    &END MOTION

**Hybrid functional (HSE06)** --- replace the `&XC_FUNCTIONAL` block with a hybrid
and an `&HF` block. Because it is a hybrid: use **OT SCF** (an `&OT` block, not
`&DIAGONALIZATION`), run **Gamma-point** (no `&KPOINTS`), and do **not** add ADMM
for a small cell — the `&SCF` becomes:

    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 30
      &OT
        MINIMIZER DIIS
        PRECONDITIONER FULL_SINGLE_INVERSE
      &END OT
      &OUTER_SCF
        MAX_SCF 10
        EPS_SCF 1.0E-6
      &END OUTER_SCF
    &END SCF

and the exchange block is:

    &XC
      &XC_FUNCTIONAL
        &PBE
          SCALE_X 0.0
          SCALE_C 1.0
        &END PBE
        &XWPBE
          SCALE_X -0.25
          SCALE_X0 1.0
          OMEGA 0.11
        &END XWPBE
      &END XC_FUNCTIONAL
      &HF
        FRACTION 0.25
        &SCREENING
          EPS_SCHWARZ 1.0E-6
        &END SCREENING
        &INTERACTION_POTENTIAL
          POTENTIAL_TYPE SHORTRANGE
          OMEGA 0.11
        &END INTERACTION_POTENTIAL
      &END HF
    &END XC

## interpretation

Read a finished run against the request, not just whether CP2K exited.

- **SCF convergence:** the `*** SCF run converged ***` banner must appear; a run
  that hit `MAX_SCF` without it is not converged --- tighten `&MIXING`, add
  `ADDED_MOS`/smearing for a metal, or lower `ALPHA`.
- **Total energy** is printed as `ENERGY| Total FORCE_EVAL ... [a.u.]`; convert from
  Hartree as needed.
- **Geometry optimization** must report `GEOMETRY OPTIMIZATION COMPLETED`; otherwise
  the structure is not relaxed.
- **Magnetization:** for a spin-polarized run confirm the integrated spin moment is
  the intended one (a nonmagnetic solution to a magnetic request means `LSD`/the
  initial `MAGNETIZATION` was missing).
- **Common failures:** a missing basis/potential for an element (`KIND` mismatch),
  an `&MGRID CUTOFF` too low (grid-dependent energy), or a metal without smearing
  (SCF oscillates and will not converge).

## validation

**Pre-run checks (before submitting):**

- Every element in `&COORD` has a `&KIND` with both a `BASIS_SET` and a `POTENTIAL`,
  and the potential is consistent with the functional (a `GTH-PBE` potential with a
  PBE calculation).
- `&GLOBAL RUN_TYPE` matches the intent, and a `GEO_OPT`/`CELL_OPT` run has the
  matching `&MOTION` block.
- Spin is enabled (`LSD`/`UKS`) whenever the system is magnetic or open-shell, and
  a magnetic request seeds `MAGNETIZATION` (or `MULTIPLICITY`); an antiferromagnet
  uses opposite-sign moments on split `&KIND`s.
- A DFT+U request has `PLUS_U_METHOD` set and a `&DFT_PLUS_U` block (with `L` and
  `U_MINUS_J`) on the correlated element's `&KIND`.
- A metal has a `&SCF/&SMEAR` block with `ADDED_MOS` and uses `&DIAGONALIZATION`;
  an insulator/semiconductor has no smearing and uses `&OT`.
- A hybrid or DFT+U run uses `&OT` and is **Gamma-point (no `&KPOINTS`)**; a plain
  GGA solid has a `&KPOINTS` mesh. (Gamma-only is otherwise reserved for molecules
  or very large cells.)
- Every `&KIND`'s `POTENTIAL GTH-PBE-q<N>` uses a valence that exists for that
  element (see the table above) — a guessed `q` aborts the run.
- Every opened `&SECTION` is closed by a matching `&END`, and `&MGRID CUTOFF` /
  `&SCF EPS_SCF` are present and sane.
