---
description: CHGNet (Crystal Hamiltonian Graph Neural Network) — universal pretrained ML interatomic potential, MPtrj-trained, predicts magnetic moments, ASE-only (no LAMMPS pair_style).
detect:
  binaries: []
  env_vars: []
  python_modules: [chgnet]
  guidance: |
    CHGNet is distributed as a pure-Python package (`pip install chgnet`).
    It does NOT provide a LAMMPS pair_style — MD runs go through ASE's
    Langevin / NPT / Verlet integrators with `chgnet.model.dynamics
    .CHGNetCalculator` attached as the ASE calculator. A successful
    `import chgnet` means the backend is usable.
deploy:
  import_line: "from chgnet.model.dynamics import CHGNetCalculator"
  construct_expr: "CHGNetCalculator(use_device=DEVICE)"
  device_env_var: CHGNET_DEVICE
  devices: [cuda, cpu, mps]
  default_model: chgnet
  model_file: none
  notes: "CHGNet universal MPtrj-pretrained model. ASE-only: no LAMMPS pair_style; the MD agent must use the ASE runner."
  trained_file:
    construct_expr: "CHGNetCalculator(model=__import__('chgnet.model', fromlist=['CHGNet']).CHGNet.from_file({model_file!r}), use_device=DEVICE)"
    notes: "Trained CHGNet model ({name}). ASE-only: no LAMMPS pair_style."
pretrained:
  - name: chgnet
    domain: universal-inorganic
    description: 'MPtrj-pretrained (~1.5M DFT relaxations). Predicts energies, forces, stresses and per-atom magnetic moments (unique among universal MLIPs). Fast. ASE only: no LAMMPS pair_style. Not for phonons, NEB barriers or elastic constants.'
    elements: {z: [1, 94], exclude: [Po, At, Rn, Fr, Ra]}
---
# CHGNet (Crystal Hamiltonian Graph Neural Network)

## overview

CHGNet is a universal ML interatomic potential pretrained on the MPtrj
dataset. It predicts energies, forces, stresses, and per-atom magnetic
moments; the last is unique among universal MLIPs and makes CHGNet the
natural choice for magnetic systems.

CHGNet is **ASE-only**. There is no LAMMPS `pair_style chgnet`. MD runs use
the `runner="ase"` path through `MLIPAgent` (the LAMMPS path raises
`NotImplementedError` with an actionable message). For systems where LAMMPS
throughput is required, prefer MACE via `pair_style mace`.

## planning

### Element coverage

Check this FIRST; a system with any element outside this list
cannot be modelled:
- `chgnet` (MPtrj): Z = 1–94 (H to Pu) except Po, At, Rn, Fr, Ra (89 elements).
  Noble gases are nominally in but nearly absent from the training data (Ne 7, Ar 13, He 80 frames): treat them as
  uncovered. Actinides Ac–Pu are sparse. Z > 94 is outside the embedding and errors.
  (Source: chgnet `model/encoders.py` max_num_elements=94; MPtrj element counts.)

### When to pick CHGNet

- **System contains magnetic elements (Fe, Co, Ni, Mn, Cr, …).** CHGNet
  predicts per-atom magnetic moments alongside energies and forces, enabling
  spin-state characterization without a separate DFT calculation.
- **Speed matters more than top-line force accuracy.** CHGNet is markedly
  faster per atom than equivariant MLIPs on the same hardware; use it for
  rapid equilibration or large-cell compositional screens.

When to not use CHGNet:

- Accurate forces for NEB barriers, phonon calculations, or elastic constants.
- LAMMPS-side production runs (large system, MPI, long timescale).
- Systems where another MLIP has been benchmarked against DFT for the target property.

## implementation

Standard ASE-based MD with the CHGNet calculator:

```python
from ase.io.lammpsdata import read_lammps_data
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from chgnet.model.dynamics import CHGNetCalculator

atoms = read_lammps_data("system.data", style="atomic", sort_by_id=True)
atoms.calc = CHGNetCalculator()      # auto-loads the pretrained model

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

`CHGNetCalculator()` accepts a `use_device="cuda"` kwarg; let
`mlip_tools.generate_ase_script` inject this from the agent's `device`
parameter.

## interpretation

Trajectories are standard ASE `.traj` files. Use `ase.io.read` or
`ase.io.iread` with any standard ASE-compatible analysis tool. Per-atom
magnetic moments are accessible via `atoms.get_magnetic_moments()` provided
the calculator was still attached when the snapshot was written.

Common failure modes:

- **Silent OOM on GPU**: CHGNet on a 24 GB card handles up to ~3000 atoms in
  single precision. Larger systems fall back to fp32 or CPU; watch for silent
  slow-downs rather than hard errors.
- **Spurious magnetic ordering**: predicted spins can lock into the wrong state
  for ambiguous magnetic systems; seed with `initial_magmoms` if the ground
  state is known.
- **Out-of-distribution drift**: rare-earth or heavy elements may be poorly
  covered; cross-check with MACE-mp-0 or run a short DFT benchmark.

## validation

CHGNet-specific accuracy thresholds (per model card, evaluated against the
MPtrj test split):

- Energy MAE: ~30 meV/atom (in-distribution)
- Force MAE: ~80 meV/Å (in-distribution)
- Magnetic moment MAE: ~0.3 μB/atom (transition metals)

Out-of-distribution warning signs:
- Forces > 5 eV/Å on equilibrium-looking structures → system likely outside
  MPtrj coverage; consider MACE-mp-0 or collect DFT reference data
- Energy drift > 0.1 eV/atom/ps in NVE → potential unstable for this
  chemistry; tighten the timestep or switch backends

For magnetic systems, validate predicted moments against known DFT values for
at least one representative structure before trusting long trajectories.
