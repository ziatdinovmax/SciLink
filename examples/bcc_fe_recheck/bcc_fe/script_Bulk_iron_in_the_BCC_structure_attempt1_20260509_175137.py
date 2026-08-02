import os
import numpy as np
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.calculators.vasp import Vasp

# Fetch BCC Fe from Materials Project (mp-13)
with MPRester(os.getenv("MP_API_KEY")) as mpr:
    structure = mpr.get_structure_by_material_id("mp-13")

# Convert to ASE Atoms
atoms = AseAtomsAdaptor.get_atoms(structure)

# Print basic info
print(f"Formula: {atoms.get_chemical_formula()}")
print(f"Number of atoms: {len(atoms)}")
print(f"Cell:\n{atoms.cell}")
print(f"Positions:\n{atoms.get_positions()}")

# Set initial magnetic moments (~5 muB on each Fe atom for ferromagnetic setup)
initial_magmoms = [5.0] * len(atoms)
atoms.set_initial_magnetic_moments(initial_magmoms)
print(f"Initial magnetic moments: {atoms.get_initial_magnetic_moments()}")

# Save the structure in POSCAR format
write('POSCAR', atoms, format='vasp')
print("STRUCTURE_SAVED:POSCAR")

# Print VASP INCAR recommendations for the user
print("\n--- Recommended VASP INCAR settings for spin-polarized PBE SCF ---")
print("ISPIN = 2          # Spin-polarized calculation")
print("MAGMOM = 5.0       # Initial magnetic moment per Fe atom")
print("GGA = PE            # PBE functional")
print("ENCUT = 520         # Plane-wave cutoff (eV)")
print("ISMEAR = 1          # Methfessel-Paxton smearing (metals)")
print("SIGMA = 0.1         # Smearing width (eV)")
print("EDIFF = 1E-6        # SCF convergence criterion")
print("IBRION = -1         # No ionic relaxation (static SCF)")
print("NSW = 0             # No ionic steps")
print("LORBIT = 11         # Write projected DOS and magnetic moments")
