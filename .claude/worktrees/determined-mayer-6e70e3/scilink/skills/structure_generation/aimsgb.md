## Overview

The `aimsgb` library constructs **grain boundary (GB) structures** — bicrystals
with a coincident-site-lattice (CSL) interface — for periodic DFT and MD
simulations. Use it for any request involving a grain boundary, bicrystal,
twist boundary, tilt boundary, or sigma-value parametrization.

Install: `pip install aimsgb`. Already available if scilink's [sim] extras
are installed.

When to reach for this skill instead of plain ASE:
- Request mentions Σ values (sigma 5, sigma 7, etc.), CSL theory
- Twist or tilt boundary specifications
- Coincident-site-lattice rotation axis + plane parametrization
- Bicrystal models for interfacial DFT studies

For non-GB structures (defects, surfaces, simple supercells), do NOT use
aimsgb — fall back to ASE / pymatgen primitives.

## Planning

A grain boundary in aimsgb is parametrized by **four** quantities the user
must (directly or implicitly) supply:

1. **Rotation axis** `[u, v, w]` — Miller indices of the axis around which
   grain B is rotated relative to grain A. Common choices:
   - FCC: `[001]`, `[110]`, `[111]`
   - BCC: `[001]`, `[110]`, `[111]`
   - HCP: typically `[0001]` for c-axis rotations

2. **Sigma (Σ)** — odd integer giving the area ratio between the CSL unit
   cell and the underlying lattice. Lower Σ → smaller (cheaper) GB
   supercell. Common low-Σ values for cubic systems:
   - Σ3, Σ5, Σ7, Σ9, Σ11, Σ13 — these dominate experimental literature.
   - **Σ MUST BE ODD.** aimsgb auto-corrects even values with a warning
     by dividing out factors of 2; pass odd values directly to avoid surprise.

3. **GB plane** `[h, k, l]` — Miller index of the boundary plane.
   Together with the axis, this determines the boundary type:
   - **Axis parallel to plane** (e.g., axis `[001]`, plane `[001]`-family)
     → **twist GB** (rotation about the surface normal)
   - **Axis perpendicular to plane** (e.g., axis `[001]`, plane `[100]`)
     → **tilt GB** (rotation about an in-plane direction)

4. **Initial structure** — the bulk crystal, supplied as a `Grain` object
   (a subclass of pymatgen's `Structure`). NOT a plain `pymatgen.Structure` —
   passing one raises `ValueError`. Construct via `Grain.from_struct(...)`
   or `Grain.from_mp_id("mp-...")`.

If the user gives only an axis and a sigma (no plane), use
`GBInformation(axis, max_sigma)` to enumerate valid plane choices for that
axis/sigma pair, then pick the lowest-index plane.

## Implementation

### Minimal usage pattern

```python
from aimsgb import Grain, GrainBoundary
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

# 1. Build the initial Grain (a pymatgen Structure subclass).
#    Either fetch from Materials Project:
grain = Grain.from_mp_id("mp-13")  # bcc Fe; needs MP_API_KEY env var
#    or load from an existing structure:
# from pymatgen.core import Structure
# struct = Structure.from_file("Fe.cif")
# grain = Grain.from_struct(struct)

# 2. Construct the grain boundary.
gb = GrainBoundary(
    axis=[0, 0, 1],    # rotation axis
    sigma=5,           # Σ (must be odd)
    plane=[3, 1, 0],   # GB plane
    initial_struct=grain,
    uc_a=1,            # number of unit cells of grain A (default 1)
    uc_b=1,            # number of unit cells of grain B (default 1)
)

# 3. Build the actual atomic structure.
gb_struct = gb.build_gb(
    vacuum=0.0,         # set >0 for slab GB (free surfaces above and below)
    add_if_dist=0.0,    # extra spacing at the interface (Å); 0 keeps default
    to_primitive=True,  # reduce final cell to primitive when possible
)

# 4. Convert pymatgen Structure → ASE Atoms and save as POSCAR.
atoms = AseAtomsAdaptor.get_atoms(gb_struct)
write("POSCAR", atoms, format="vasp", sort=True)

print(f"STRUCTURE_SAVED:POSCAR")  # required confirmation line for scilink
```

### Choosing the plane when the user only specifies axis + sigma

```python
from aimsgb import GBInformation

info = GBInformation(axis=[0, 0, 1], max_sigma=5, specific=True)
# info is a dict-like: keys are sigma values, values describe valid planes.
# Inspect with str(info) or iterate to pick the simplest plane for `sigma`.
print(info)
```

### Twist boundary (axis ‖ plane)

```python
# [001] twist boundary — rotation around z, plane normal also z.
gb = GrainBoundary(
    axis=[0, 0, 1], sigma=5, plane=[0, 0, 1],
    initial_struct=grain,
)
```

### Tilt boundary (axis ⊥ plane)

```python
# [001] axis with plane in the xz-family → tilt about the in-plane direction.
gb = GrainBoundary(
    axis=[0, 0, 1], sigma=5, plane=[3, 1, 0],
    initial_struct=grain,
)
```

### Asymmetric grain sizes

Use `uc_a` / `uc_b` to make grain A and grain B different thicknesses (e.g.,
for studying GB segregation gradients).

```python
gb = GrainBoundary(
    axis=[0, 0, 1], sigma=5, plane=[3, 1, 0],
    initial_struct=grain, uc_a=2, uc_b=1,  # grain A is twice as thick
)
```

### Adding vacuum (slab GB for surface-sensitive studies)

```python
gb_struct = gb.build_gb(vacuum=15.0)  # 15 Å of vacuum on top and bottom
```

### Stacking arbitrary grains (no CSL constraint)

If the user wants to stack two arbitrary slabs (NOT a CSL bicrystal), use
`Grain.stack_grains` directly instead of `GrainBoundary`:

```python
from aimsgb import Grain
gb_struct = Grain.stack_grains(
    grain_a, grain_b, vacuum=0.0, gap=0.0, direction=2,
)
```

This is escape-hatch territory; prefer `GrainBoundary` when CSL theory applies.

## Validation

Sanity checks to run after construction (the script can print these so the
user / validator can verify):

1. **Stoichiometry preserved.** `len(gb_struct.composition)` and the
   element ratios should match the bulk grain. CSL construction sometimes
   trims atoms at the interface — verify the count makes sense.

2. **Cell dimensions reasonable.** A Σ5 boundary on a cubic 4 Å lattice
   should yield in-plane lattice vectors of order √5 × 4 ≈ 9 Å. If the
   resulting cell is enormous (>100 Å in any direction), Σ may be too high
   for the requested axis/plane combination.

3. **Sigma is odd in the final structure.** Print `gb.sigma` and confirm
   it's the value the user requested (or note if aimsgb reduced it).

4. **No atomic clashes much smaller than 0.5 Å.** Some clashes (<1 Å) are
   normal at unrelaxed GB interfaces and DFT relaxation will resolve them.
   Sub-0.5 Å clashes usually indicate a construction error (wrong plane
   choice for the axis, or `add_if_dist=0` with a poorly-matched interface).

5. **Periodic-image GB distance.** The structure has TWO interfaces (one in
   the cell, one across the periodic boundary). Make sure both are physically
   reasonable distances apart — if the cell is too thin along the GB normal,
   neighboring boundaries interact through the periodic image.

## Common pitfalls

- **Even sigma**: passing `sigma=4` triggers a warning and aimsgb divides by 2
  to get the largest odd factor (`sigma=1`). Always pass odd values.
- **`Structure` instead of `Grain`**: `GrainBoundary(initial_struct=Structure(...))`
  raises `ValueError`. Wrap with `Grain.from_struct(struct)`.
- **Non-cubic crystals**: aimsgb converts to the conventional cell internally.
  For hex/trigonal systems, ensure the axis is expressed in conventional
  Miller indices (e.g., `[0001]`-style for hcp), not Bravais-lattice indices.
- **Plane perpendicular to axis vs parallel**: confusing this swaps tilt and
  twist. Plane parallel to axis → twist; plane perpendicular → tilt.
