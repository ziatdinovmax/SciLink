---
description: Quantum ESPRESSO (pw.x) DFT input generation — namelist/card construction (&CONTROL/&SYSTEM/&ELECTRONS/&IONS/&CELL; ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, CELL_PARAMETERS), plane-wave cutoffs, smearing, spin polarization, and pseudopotential conventions for metals, semiconductors, slabs, and molecules.
detect:
  binaries: [pw.x]
  env_vars: []
  python_modules: []
  guidance: |
    Quantum ESPRESSO's plane-wave SCF / relaxation engine is pw.x. On
    HPC clusters it is frequently available only after `module load
    quantum-espresso/<version>` (or `qe/<version>`), so a bare $PATH
    check can miss it. Pseudopotentials are located via the `pseudo_dir`
    input variable or the $ESPRESSO_PSEUDO environment variable.
    Detection relies on pw.x being on $PATH (which `module load
    quantum-espresso` provides). QE env vars like $ESPRESSO_PSEUDO point at a
    pseudopotential directory, not the pw.x binary, so they are not used as a
    detection hit.
---
# Quantum ESPRESSO Input Generation Skill

## overview

Density Functional Theory (DFT) calculations with the plane-wave self-consistent
field code `pw.x` from Quantum ESPRESSO (QE). This skill covers construction of a
single `pw.x` input file — the Fortran namelists (`&CONTROL`, `&SYSTEM`,
`&ELECTRONS`, and `&IONS`/`&CELL` when relaxing) and the structural cards
(`ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, `K_POINTS`, `CELL_PARAMETERS`) — for metals,
semiconductors, insulators, surface slabs, and isolated molecules. The goal is input
that is physically correct, computationally efficient, and consistent with standard
QE practice. **All energies, cutoffs, `degauss`, and `conv_thr` are in Rydberg.**

## planning

**Functional and pseudopotentials:** Unlike VASP, QE does NOT have a functional tag —
the exchange-correlation functional is carried by the pseudopotentials. Use a single,
self-consistent pseudopotential library for ALL species:
- **SSSP** (PBE) Efficiency or Precision is the recommended default — it ships a
  validated per-element list with recommended `ecutwfc`/`ecutrho`.
- PSlibrary, GBRV (ultrasoft), or ONCV / PseudoDojo (norm-conserving) are alternatives.
List each species' `.UPF` file in `ATOMIC_SPECIES` and point `pseudo_dir` at the
directory (or `$ESPRESSO_PSEUDO`). NEVER mix functionals or libraries across species —
it produces meaningless energies. (A functional override via `input_dft` in `&SYSTEM`
exists but should be avoided in favor of matching pseudopotentials.)

**Plane-wave cutoffs (`ecutwfc` / `ecutrho`):** `ecutwfc` is the wavefunction cutoff,
`ecutrho` the charge-density cutoff (both Ry). The ratio depends on pseudopotential type:
- Norm-conserving (ONCV): `ecutrho = 4 * ecutwfc`.
- Ultrasoft / PAW: `ecutrho = 8-12 * ecutwfc`.
Always use the per-element cutoffs recommended by your library (SSSP lists them) and take
the MAXIMUM over all species. Safe starting defaults for PBE ultrasoft/PAW:
`ecutwfc = 50 Ry`, `ecutrho = 400-600 Ry`. Hydrogen, first-row elements (O, N, F), and
3d transition metals need higher `ecutwfc` (60-90 Ry). Under-converged cutoffs are the
single most common cause of wrong energies and forces.

**System identification:** before choosing parameters, identify the system:
1. Metal, semiconductor, insulator, or molecule?
2. Bulk, surface slab, or isolated molecule/cluster?
3. Magnetic elements (Fe, Co, Ni, Mn, Cr)?
4. Hydrogen / first-row elements (need higher `ecutwfc`)?
5. Calculation type: `scf`, `relax`, `vc-relax`, `nscf`, or `bands`?

**Smearing (`occupations`):** dictated primarily by system type:
- **Metals / metallic slabs:** `occupations='smearing'`, `smearing='mv'`
  (Marzari-Vanderbilt cold) or `'mp'` (Methfessel-Paxton), `degauss=0.01-0.02`.
  Required — without smearing a metal will not converge.
- **Semiconductors / insulators (relaxation):** `occupations='smearing'`,
  `smearing='gaussian'`, `degauss=0.005-0.01` is robust; `occupations='fixed'` is
  acceptable when there is a clear gap.
- **Insulators (accurate static DOS/energies):** `occupations='tetrahedra'` (or
  `'tetrahedra_opt'`) on a Gamma-centered mesh, via an `nscf` step.
- **Molecules in a box:** Gamma point only, small `degauss` with `'gaussian'`, or
  `occupations='fixed'`.
`degauss` is in Ry and MUST be > 0 whenever `occupations='smearing'`.

**Spin polarization:** systems containing Fe, Co, Ni, Mn, or Cr MUST set `nspin=2`
and `starting_magnetization(i)` for the magnetic species. QE's
`starting_magnetization` is a fraction of the valence (~ +1.0 = strongly polarized).
Typical starting values: Fe/Mn ~ 0.6-1.0, Co ~ 0.5, Ni ~ 0.3, Cr ~ ±0.5. Omitting
`nspin=2` for magnetic systems gives the wrong magnetic ground state and energies. For
DFT+U, add `lda_plus_u=.true.` and `Hubbard_U(i)` (or the `HUBBARD` card in QE >= 7.1).

## implementation

**Emit a single `pw.x` input file named `pw.in`.** Order: namelists `&CONTROL`,
`&SYSTEM`, `&ELECTRONS` (then `&IONS` for relax/md, `&CELL` for vc-relax), followed by
the cards `ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, `K_POINTS`, and `CELL_PARAMETERS`.

**Geometry:** use `ibrav=0` with an explicit `CELL_PARAMETERS {angstrom}` block — the
most robust choice for arbitrary cells coming from a POSCAR/CIF. Set `nat` (total atom
count) and `ntyp` (number of distinct species). Use `ATOMIC_POSITIONS {crystal}`
(fractional) or `{angstrom}`.

**`&CONTROL` essentials:**
- `calculation = 'scf' | 'relax' | 'vc-relax' | 'nscf' | 'bands'`
- `prefix`, `outdir = './out'`, `pseudo_dir = './pseudo'` (or `$ESPRESSO_PSEUDO`)
- `tprnfor = .true.`, `tstress = .true.` (print forces and stress)
- For relax/vc-relax: `nstep = 100-200`, `forc_conv_thr = 1.0d-3` (Ry/Bohr),
  `etot_conv_thr = 1.0d-4` (Ry)

**`&ELECTRONS`:**
- `conv_thr = 1.0d-6` for production (`1.0d-8` for accurate forces/phonons).
- `mixing_beta = 0.7` default; lower to `0.2-0.3` for metals or hard-to-converge SCF.
- `mixing_mode = 'plain'` default; `'local-TF'` helps metals, inhomogeneous, or charged
  cells.
- `electron_maxstep = 100-200`.

**`&IONS` (relax/vc-relax/md):** `ion_dynamics='bfgs'` for relaxations.
**`&CELL` (vc-relax only):** `cell_dynamics='bfgs'`, `press=0.0`; constrain with
`cell_dofree` (e.g. `'2Dxy'` to relax only in-plane for a slab).

**`K_POINTS`:**
- **Bulk:** `K_POINTS automatic` with a Monkhorst-Pack mesh of ~0.15-0.25 Ang^-1 spacing
  (≈ 8-16^3 on a cubic 3-4 Ang cell; denser for metals/small cells, sparser for large ones);
  Gamma-centered shift `0 0 0` for hexagonal cells and metals: `nk1 nk2 nk3 0 0 0`.
- **Slab:** same in-plane density, `1` in the vacuum direction, e.g.
  `nk1 nk2 1 0 0 0`; set `assume_isolated='2D'` (or use `dipfield`/`edir`/`emaxpos`/
  `eopreg` for an asymmetric slab dipole correction).
- **Molecule in a box:** `K_POINTS gamma`; set `assume_isolated='mt'`
  (Martyna-Tuckerman) for charged or strongly dipolar molecules.

**Selective dynamics:** freeze atoms with `if_pos` flags (`0`=fixed, `1`=free) appended
after the coordinates in `ATOMIC_POSITIONS`, e.g. `Zn 0.0 0.0 0.0 0 0 0`. For slabs,
fix the bottom layers to mimic bulk.

**Template — bulk vc-relax (metal):**

  &CONTROL
    calculation   = 'vc-relax'
    prefix        = 'mat'
    outdir        = './out'
    pseudo_dir    = './pseudo'
    tprnfor       = .true.
    tstress       = .true.
    forc_conv_thr = 1.0d-3
  /
  &SYSTEM
    ibrav       = 0
    nat         = 4
    ntyp        = 1
    ecutwfc     = 50
    ecutrho     = 400
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.015
  /
  &ELECTRONS
    conv_thr     = 1.0d-6
    mixing_beta  = 0.3
  /
  &IONS
    ion_dynamics = 'bfgs'
  /
  &CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
  /
  ATOMIC_SPECIES
    Cu  63.546  Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
  ATOMIC_POSITIONS crystal
    Cu  0.0  0.0  0.0
    Cu  0.5  0.5  0.0
    Cu  0.5  0.0  0.5
    Cu  0.0  0.5  0.5
  K_POINTS automatic
    12 12 12 0 0 0
  CELL_PARAMETERS angstrom
    3.61 0.00 0.00
    0.00 3.61 0.00
    0.00 0.00 3.61

**Template — surface slab relax (semiconductor, bottom layers fixed):** as above but
`calculation='relax'`, drop `&CELL`, `smearing='gaussian'` with `degauss=0.007`,
`K_POINTS automatic` = `nk1 nk2 1 0 0 0`, add `assume_isolated='2D'` in `&SYSTEM`, and
append `0 0 0` to the bottom-layer rows in `ATOMIC_POSITIONS`.

**Template — molecule in a box:** `calculation='relax'`, large cubic
`CELL_PARAMETERS`, `K_POINTS gamma`, `occupations='smearing'` with
`smearing='gaussian'` and small `degauss=0.002` (or `occupations='fixed'`),
`assume_isolated='mt'` if charged (`tot_charge` set accordingly).

## validation

**Quality checks for the generated `pw.in`:**
- Namelist syntax: each namelist opens with `&NAME` and closes with a single `/` on its
  own line. Cards (`ATOMIC_SPECIES`, etc.) take NO `/`.
- `nat` equals the number of `ATOMIC_POSITIONS` rows; `ntyp` equals the number of
  distinct `ATOMIC_SPECIES` entries.
- Every species used in `ATOMIC_POSITIONS` appears in `ATOMIC_SPECIES` with a `.UPF`
  file, and all UPFs are from the SAME functional/library.
- `ibrav=0` REQUIRES a `CELL_PARAMETERS` card (with units). If `ibrav != 0`,
  `CELL_PARAMETERS` must be absent.
- `ecutrho` is present and `>= 4*ecutwfc` (norm-conserving) or `~8-12*ecutwfc`
  (ultrasoft/PAW).
- `occupations='smearing'` REQUIRES `degauss > 0` and a `smearing` type. Metals must use
  smearing (`mv`/`mp`), never `'fixed'`.
- `nspin=2` with `starting_magnetization` is present for any system containing
  Fe, Co, Ni, Mn, or Cr.
- `&IONS` is present iff `calculation` is `relax`/`vc-relax`/`md`; `&CELL` is present iff
  `vc-relax`. A `'scf'` calculation must NOT contain `&IONS` or `&CELL`.
- Slabs: `K_POINTS` is `1` in the vacuum direction. Molecules: `K_POINTS gamma`.
- Units are stated and consistent on `ATOMIC_POSITIONS` and `CELL_PARAMETERS`.

**Common errors and their input fixes:**
- "file (pseudo) not found" / "reading pseudopotential": `pseudo_dir` wrong, or the
  `ATOMIC_SPECIES` UPF filename does not match. Fix the path/filename.
- "convergence NOT achieved after N iterations": lower `mixing_beta` to 0.2-0.3,
  increase `electron_maxstep`, try `mixing_mode='local-TF'` (metals).
- "the system is metallic, specify occupations": add `occupations='smearing'`,
  a `smearing` type, and `degauss`.
- "ibrav .ne. 0 but CELL_PARAMETERS given" or "ibrav==0 but no CELL_PARAMETERS": use
  `ibrav=0` together with a `CELL_PARAMETERS` card.
- "degauss is zero" with `occupations='smearing'`: set `degauss > 0`.
- "too few bands" / "not enough empty states": increase `nbnd`.
- "S matrix not positive definite" / charge sloshing: increase `ecutrho`, reduce
  `mixing_beta`.
- Charged or strongly dipolar molecule artifacts: set `assume_isolated='mt'` and
  `tot_charge`.
