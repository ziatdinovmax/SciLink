import os
import numpy as np
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.units import Rydberg

# Fetch the diamond cubic Si structure from Materials Project (mp-149)
with MPRester(os.getenv("MP_API_KEY")) as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")

# Convert to ASE Atoms (this gives the 2-atom primitive cell for Fd-3m Si)
atoms = AseAtomsAdaptor.get_atoms(structure)

print(f"Number of atoms: {len(atoms)}")
print(f"Chemical formula: {atoms.get_chemical_formula()}")
print(f"Cell:\n{atoms.cell[:]}")
print(f"Positions:\n{atoms.get_positions()}")

# Save as POSCAR
write('POSCAR', atoms, format='vasp')
print("STRUCTURE_SAVED:POSCAR")

# Set up Quantum ESPRESSO PBE SCF calculation
pseudopotentials = {'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'}

input_data = {
    'control': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'prefix': 'si_bulk',
        'outdir': './tmp',
        'pseudo_dir': './',
        'tprnfor': True,
        'tstress': True,
    },
    'system': {
        'ecutwfc': 50.0,
        'ecutrho': 400.0,
        'occupations': 'smearing',
        'smearing': 'cold',
        'degauss': 0.01,
    },
    'electrons': {
        'conv_thr': 1.0e-8,
        'mixing_beta': 0.7,
    },
}

kpts = (8, 8, 8)

# Determine pw.x command from environment or default
pw_command = os.environ.get('ASE_ESPRESSO_COMMAND', 'pw.x')
# If ASE_ESPRESSO_COMMAND contains the full template like 'pw.x -in PREFIX.pwi > PREFIX.pwo'
# we need just the binary for the profile
if ' ' in pw_command:
    pw_binary = pw_command.split()[0]
else:
    pw_binary = pw_command

try:
    # Try newer ASE API with EspressoProfile
    profile = EspressoProfile(argv=[pw_binary])
    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
        koffset=(0, 0, 0),
    )
except Exception:
    # Fallback: set the environment variable and use older API
    if 'ASE_ESPRESSO_COMMAND' not in os.environ:
        os.environ['ASE_ESPRESSO_COMMAND'] = 'pw.x -in PREFIX.pwi > PREFIX.pwo'
    calc = Espresso(
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
        koffset=(0, 0, 0),
    )

atoms.calc = calc

# Run SCF
energy = atoms.get_potential_energy()
print(f"\nPBE SCF total energy: {energy:.6f} eV")
print(f"Energy per atom: {energy / len(atoms):.6f} eV/atom")
