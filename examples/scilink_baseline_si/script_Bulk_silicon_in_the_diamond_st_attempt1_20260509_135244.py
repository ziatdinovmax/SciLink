import os
import numpy as np
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.calculators.espresso import Espresso, EspressoProfile

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

pseudo_dir = os.environ.get('PSEUDO_DIR', './')

input_data = {
    'control': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'prefix': 'si_bulk',
        'outdir': './tmp',
        'pseudo_dir': pseudo_dir,
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

import shutil
import subprocess

# Check if pw.x is available
pw_command = os.environ.get('ASE_ESPRESSO_COMMAND', None)
pw_executable = shutil.which('pw.x')

if pw_executable is None and pw_command is None:
    print("\nWARNING: pw.x not found in PATH. Cannot run Quantum ESPRESSO SCF calculation.")
    print("To run the SCF calculation, install Quantum ESPRESSO and ensure pw.x is in your PATH.")
    print("Skipping SCF energy calculation.")
else:
    try:
        # Try the modern ASE EspressoProfile API
        if pw_executable:
            profile = EspressoProfile(command=pw_executable, pseudo_dir=pseudo_dir)
        else:
            # Extract binary from command string
            cmd = pw_command.split()[0] if pw_command else 'pw.x'
            profile = EspressoProfile(command=cmd, pseudo_dir=pseudo_dir)
        
        calc = Espresso(
            profile=profile,
            pseudopotentials=pseudopotentials,
            input_data=input_data,
            kpts=kpts,
            koffset=(0, 0, 0),
        )
    except TypeError:
        try:
            if pw_executable:
                profile = EspressoProfile(argv=[pw_executable])
            else:
                cmd = pw_command.split()[0] if pw_command else 'pw.x'
                profile = EspressoProfile(argv=[cmd])
            calc = Espresso(
                profile=profile,
                pseudopotentials=pseudopotentials,
                input_data=input_data,
                kpts=kpts,
                koffset=(0, 0, 0),
            )
        except Exception:
            # Fallback for older ASE versions
            if 'ASE_ESPRESSO_COMMAND' not in os.environ:
                binary = pw_executable if pw_executable else 'pw.x'
                os.environ['ASE_ESPRESSO_COMMAND'] = f'{binary} -in PREFIX.pwi > PREFIX.pwo'
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
