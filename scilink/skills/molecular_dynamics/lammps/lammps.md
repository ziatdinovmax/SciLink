---
description: LAMMPS classical molecular dynamics — input scripts for materials, biomolecular, and reactive systems with pair styles spanning EAM, Tersoff, ReaxFF, OPLS-AA, AMBER, and MLIP backends.
outputs:
  trajectory: [lammpstrj, dcd, xtc, trr, nc]
  thermo_log: [log.lammps, thermo.log]
detect:
  binaries: [lmp, lmp_mpi, lmp_serial, lmp_kokkos_cuda_mpi, lammps]
  env_vars: [LAMMPS_HOME, LAMMPS_DIR]
  python_modules: [lammps]
  guidance: |
    LAMMPS binaries follow the pattern lmp[_<variant>] where variant
    is a build-time suffix (e.g. lmp_serial, lmp_mpi, lmp_kokkos_cuda).
    The plain `lmp` is the conda-forge / pip default. The `lammps`
    Python module (PYTHON package binding) is also a valid runtime
    surface — if importable, the engine is usable even without a
    standalone binary on $PATH.
---

## Overview

LAMMPS is a general-purpose classical molecular dynamics engine for materials
science, chemistry, and biology. It uses a single input script with a companion
data file. Supports bulk crystals, surfaces, interfaces, liquids, polymers, and
biomolecules via a wide range of force fields and boundary conditions.

Key characteristics:
- Input: text script (`run.lammps`) + data file (`system.data`)
- Units are force-field-dependent (metal vs real vs lj)
- atom_style is system-dependent (atomic vs charge vs full)
- Many-body potentials (EAM, Tersoff) use external potential files
- Pairwise potentials (LJ, Buckingham) use pair_coeff commands
- Execution: `mpirun -np N lmp -in run.lammps`


## Planning

### Unit system, atom style, and force field selection

| System Type               | units | atom_style | pair_style              | Potential Source     |
|---------------------------|-------|------------|-------------------------|----------------------|
| Metals, alloys            | metal | atomic     | eam/alloy or meam/c     | External .eam.alloy  |
| Semiconductors (Si,C,SiC) | metal | atomic     | tersoff or sw or airebo | External .tersoff/.sw |
| Oxides, ionic crystals    | real  | charge     | buck/coul/long          | pair_coeff + kspace   |
| Oxides (reactive/COMB)    | metal | charge     | comb3 or reaxff         | External ffield       |
| Biomolecular (AMBER)      | real  | full       | lj/charmm/coul/long 10 12 | Embedded in data file |
| Biomolecular (CHARMM)     | real  | full       | lj/charmm/coul/long 10 12 | Embedded in data file |
| Reactive chemistry        | real  | charge     | reaxff                  | External ffield.reax  |
| Coarse-grained / LJ fluid | lj   | atomic     | lj/cut                  | pair_coeff commands   |
| Interfaces (hybrid)       | metal | atomic/charge | hybrid/overlay       | Mixed sources         |

### Thermostat/barostat damping by unit system

| units | timestep     | Tdamp   | Pdamp    |
|-------|-------------|---------|----------|
| metal | 0.001 ps    | 0.1 ps  | 1.0 ps   |
| real  | 1.0-2.0 fs  | 100 fs  | 1000 fs  |
| lj    | 0.005 τ     | 1.0 τ   | 10.0 τ   |

CRITICAL: These values are NOT interchangeable across unit systems.

### Boundary conditions

| Geometry              | boundary | Notes                                   |
|-----------------------|----------|-----------------------------------------|
| Bulk 3D periodic      | p p p    | Crystals, liquids, solutions            |
| Surface slab (z-free) | p p s    | Add 15-30 Å vacuum; no z-barostat      |
| Wire (y,z free)       | p s s    | 1D periodic                             |
| Nanoparticle          | s s s    | Non-periodic; shrink-wrapped            |

### Equilibration protocols

Crystalline solids:
1. `minimize 1.0e-6 1.0e-8 10000 100000`
2. NPT (aniso) at target T,P for 50-200 ps → lattice parameter convergence
3. Production: NPT or NVT or NVE

Surface / slab:
1. Minimize (freeze bottom layers with `fix setforce 0 0 0`)
2. NVT for 50-100 ps (no z-barostat; barostat only periodic dims)
3. Production: NVT

Liquids and solutions:
1. `minimize 1.0e-4 1.0e-6 5000 50000`
2. NPT at target T,P for 0.5-2 ns → density convergence
3. Production: NPT or NVT

### Staging integrators (multi-phase runs)

Each stage uses exactly ONE time integrator on a group. To move between phases —
e.g. NPT density equilibration → NVT production — `unfix` the previous integrator
BEFORE defining the next, and give them distinct fix IDs:

```
fix   eq all npt temp {temperature} {temperature} {Tdamp} iso {pressure} {pressure} {Pdamp}
run   {equil_steps}
unfix eq
fix   prod all nvt temp {temperature} {temperature} {Tdamp}
# … production output (e.g. the stress log a viscosity goal needs) …
run   {prod_steps}
```

A single script legitimately contains both an npt and an nvt fix, but only ONE
may be active on a group at a time — leaving both active is the fix nvt/npt
conflict in Forbidden patterns.

### Technique-specific requirements

Transport coefficients (viscosity, self-diffusion) — equilibrium Green-Kubo/Einstein:
- The deck's job is to EMIT the raw time series; the coefficient is computed AFTER
  the run by the matching `simulation_analysis` skill (e.g. `viscosity_greenkubo`).
  Do NOT compute the transport coefficient in the input — no in-deck stress-
  autocorrelation integral, no `variable visc`/`${visc}`, no `fix ave/correlate`
  for the coefficient itself.
- **Shear viscosity:** on the production run, log the full pressure tensor densely
  — add `pxy pxz pyz vol` to `thermo_style custom` with a tight `thermo` interval
  (every few fs), or write them via `fix ave/time {N} 1 {N} v_pxy v_pxz v_pyz file
  stress.dat`. The `viscosity_greenkubo` skill reads that log (it also needs T and
  vol) and does the Green-Kubo integral. It converges slowly for viscous liquids →
  the production run must be long (tens of ns) with stress sampled every few fs.
- **Self-diffusion:** dump the unwrapped coordinate trajectory at a regular
  interval; the diffusion analysis skill computes the MSD/Einstein slope.

Tensile deformation:
- `fix deform x erate R` + thermostat on transverse dims
- Output: `variable strain equal (lx-v_Lx0)/v_Lx0`, stress from pressure tensor

Thermal conductivity (NEMD):
- Heat source/sink via `fix heat` on groups at opposite ends
- Output: temperature profile via `compute chunk/atom` + `fix ave/chunk`
- Needs 1-5 ns for steady-state gradient

Umbrella sampling:
- `fix spring/couple` or `fix colvars` for biasing potential
- One simulation per window; output CV for WHAM analysis

ReaxFF:
- MUST include `fix qeq/reaxff` for charge equilibration
- Timestep 0.25-1.0 fs (smaller than standard)


## Analysis

### System type detection from data file
- atom_style atomic + no bonds → metal, semiconductor, or atomic solid
- atom_style charge + no bonds → oxide, ionic crystal
- atom_style full + bonds/angles → biomolecular, polymer, organic
- Large vacuum gap in one dimension → surface or thin film
- Pair Coeffs section present → self-contained (AMBER pipeline or user-prepared)
- No Pair Coeffs → needs potential file or pair_coeff commands in script

### Element heuristics
- Single metal (Fe,Cu,Al,Ni,W,Au,Ti...) → bulk metal or alloy
- Metal + O → metal oxide
- Si → semiconductor; Si+O → silica; Si+C → silicon carbide
- C alone → diamond/graphene/CNT/amorphous carbon
- C+H+N+O with bonds → organic or biomolecular
- Na+Cl or similar halides → ionic crystal or salt solution

## Interpretation

### Key diagnostic checks
- Temperature fluctuates around target: OK (±5-10% for small systems)
- NVE energy drift < 1e-5 eV/atom/ps (metal) or < 0.01 kcal/mol/atom/ns (real): OK
- Lattice parameters match experiment within 1-3%: potential is reasonable
- Continuous lattice drift in NPT: wrong potential or phase transition

### Common errors
- "Lost atoms": overlapping atoms (minimize first), timestep too large, or wrong units
- "All pair coeffs not set": missing pair_coeff; for EAM, all elements must be listed
- "Pair style requires KSpace": using coul/long without `kspace_style pppm`
- "Cannot open potential file": file not in working directory or wrong path
- Temperature immediately 1e6+ K: unit mismatch between potential and script
- Pressure ~1e6 at start: structure not minimized
- "Out of range atoms - cannot compute PPPM": an atom/ion left the long-range
  grid *mid-run* (a fast species, or after an NPT volume change) — a
  deck-fixable dynamics problem, not a structure one. Reduce the timestep
  (e.g. 2→1 fs for flexible molecular liquids), tighten `neigh_modify every 1
  delay 0 check yes` with a larger neighbor skin, and soften an over-aggressive
  barostat (longer Pdamp).
- Non-finite (nan/inf) or huge-*negative* energy at step 0 that persists even
  though the deck already minimizes: overlapping / near-zero-distance atoms from
  a bad initial pack (often opposite charges on top of each other). Minimization
  CANNOT fix this — it collapses the overlap further. The structure must be
  regenerated / repacked; do not patch the deck.


## Validation

### Command ordering (required)
1. units, dimension, boundary, atom_style
2. read_data or read_restart
3. pair_style, bond_style, angle_style (if applicable)
4. pair_coeff (or potential file reference)
5. kspace_style, special_bonds (if needed)
6. neighbor, neigh_modify
7. group, region definitions
8. fix commands (shake, thermostat, barostat, deform)
9. compute, variable definitions
10. thermo, thermo_style, dump, restart
11. timestep
12. minimize (if needed)
13. run

**Fully-typed force-field data files invert steps 2–4.** When the data file
already contains inline coefficients (the data-file analysis reports
`Coefficients in data file: Yes` — e.g. an OpenFF Interchange export with `Pair
Coeffs` / `Bond Coeffs` / `Angle Coeffs` / `Dihedral Coeffs` sections), declare
every needed `*_style` (`pair_style`, plus `bond_style` / `angle_style` /
`dihedral_style` for the bonded sections present) **before** `read_data`, and do
**not** emit `pair_coeff` / `bond_coeff`. LAMMPS reads the coefficients from the
data file at `read_data` time; parsing a `Pair Coeffs` section with no prior
`pair_style` aborts with "Must define pair_style before Pair Coeffs". Only the
coefficient-parsing styles move before `read_data` — put `kspace_style` (e.g.
`pppm 1e-4`, needed for a `coul/long` pair_style) **after** `read_data`. A
typed Interchange data file carries an `xy xz yz` line, so the box is triclinic
and PPPM aborts with "Must redefine kspace_style after changing to triclinic
box" if `kspace_style` is declared first.

### Forbidden patterns
- pair_coeff before pair_style
- read_data before pair_style when the data file has inline Pair Coeffs (a
  fully-typed FF data file) — declare the coefficient styles first
- kspace_style before read_data — declare kspace AFTER read_data (PPPM aborts on
  a triclinic data file, which any OpenFF Interchange export is)
- kspace_style with EAM/Tersoff/SW (no Coulomb)
- kspace_style missing with coul/long or buck/coul/long
- bond_style or fix shake with atom_style atomic
- NPT barostat on a non-periodic dimension
- fix nvt and fix npt on the same group simultaneously
- fix shake defined after run
- Unresolved variables: ${temp}, {pressure}, etc.
- Missing run command (script does nothing)
- ReaxFF without fix qeq/reaxff

### Parameter sanity (by unit system)

| Check         | metal          | real              | lj            |
|---------------|----------------|-------------------|---------------|
| timestep      | 0.0005-0.005   | 0.5-4.0           | 0.001-0.01    |
| Tdamp         | 0.05-0.5       | 50-500            | 0.5-5.0       |
| Pdamp         | 0.5-5.0        | 500-5000          | 5.0-50.0      |
| temperature   | 1-5000 K       | 1-5000 K          | 0.1-10.0      |


## Implementation

### Metal (EAM, bulk)
``` 
units metal
atom_style atomic
boundary p p p
read_data {data_filename}
pair_style eam/alloy
pair_coeff * * {potential_file} {element_list}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 100
thermo_style custom step temp press pe ke etotal vol lx ly lz
minimize 1.0e-6 1.0e-8 10000 100000
velocity all create {temperature} {seed} dist gaussian
fix 1 all npt temp {temperature} {temperature} 0.1 aniso {pressure} {pressure} 1.0
timestep 0.001
dump 1 all custom 1000 trajectory.dump id type x y z
restart 10000 checkpoint.restart1 checkpoint.restart2
run {total_steps}
write_data final.data
```

### Semiconductor / covalent (Tersoff)
```
units metal
atom_style atomic
boundary p p p
read_data {data_filename}
pair_style tersoff
pair_coeff * * {potential_file} {element_list}
(same dynamics block as metal, with appropriate groups for surfaces)
```

### Oxide / ionic (Buckingham + Coulomb)
```
units real
atom_style charge
boundary p p p
read_data {data_filename}
pair_style buck/coul/long 10.0
pair_coeff 1 1 0.0 1.0 0.0
pair_coeff 1 2 {A} {rho} {C}
pair_coeff 2 2 0.0 1.0 0.0
kspace_style pppm 1.0e-5
(dynamics with timestep 1.0, Tdamp 100.0, Pdamp 1000.0)
```

### Biomolecular / typed force-field data file (AMBER / GAFF / OpenFF — coefficients in the data file)
The data file carries Pair/Bond/Angle/Dihedral Coeffs, so **every `*_style` is
declared BEFORE `read_data`** and **`kspace_style` AFTER it** (see Command
ordering). `pair_modify` follows `pair_style`. Do **not** issue
`pair_coeff`/`bond_coeff` — `read_data` supplies them.
```
units real
atom_style full
boundary p p p
# --- force-field styles FIRST (coeffs are read from the data file) ---
pair_style lj/cut/coul/long 10.0          # GAFF/OpenFF; lj/charmm/coul/long 10.0 12.0 for CHARMM
pair_modify mix arithmetic tail yes        # Lorentz-Berthelot; MUST follow pair_style
bond_style harmonic
angle_style harmonic
dihedral_style fourier                     # match the data file (OpenFF/Interchange: fourier)
# improper_style cvff                      # only if the data file has impropers (improper types > 0)
special_bonds amber                        # AMBER/GAFF/OpenFF 1-4 scaling
read_data {data_filename}
# --- kspace AFTER read_data (PPPM binds to the box; required for a triclinic data file) ---
kspace_style pppm 1.0e-4
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
fix SHAKE all shake 1.0e-5 100 0 m 1.008   # constrains X-H bonds (rigid water); enables 2 fs
(dynamics with timestep 2.0, Tdamp 100.0, Pdamp 1000.0)
```

### ReaxFF (reactive)
```
units real
atom_style charge
boundary p p p
read_data {data_filename}
pair_style reaxff NULL safezone 3.0 mincap 150
pair_coeff * * {reaxff_potential} {element_list}
fix QEQ all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
(dynamics with timestep 0.25-1.0, Tdamp 100.0)
```

### Selecting atoms by chemical species (multi-component systems)
When a `group` or `compute` must target a specific chemical species — not just
an element — resolve atom types to their parent molecule first. Atom type
numbers are assigned per system and are NOT stable across runs, so never
hardcode a type integer or assume the same number means the same species in
another data file; re-derive the mapping for every system.

The same element often belongs to several distinct species (e.g. oxygen in a
solvent, in a hydroxyl group, and in a carbonyl), so selecting by element or
mass alone conflates them:
- The components manifest (`components.json`) lists each molecular species
  (name, SMILES, count) in the order they were packed into the box.
- In the typed data file (OpenFF/Interchange, `atom_style full`) the Atoms
  section carries a molecule-ID per atom, and molecules appear in that packing
  order — so each molecule-ID maps to one species. A force field that types by
  chemical environment (OpenFF) gives the same element *different* types in
  different species, so each atom type usually resolves to exactly one species.
- Resolve type → parent molecule from the manifest plus the data file, then
  build the `group`/`compute` from the resolved type(s). A pair analysis
  targeting one species' atom (say a specific solvent's oxygen) must select that
  species' type, which is distinct from the same element in another molecule.
