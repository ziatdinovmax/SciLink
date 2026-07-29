# tests/test_lammps_skill/conftest.py
"""
Shared fixtures for LAMMPS skill tests.

Generates minimal-but-valid LAMMPS data files and scripts covering
every system category and atom_style that lammps_tools must handle.
"""

import pytest
from pathlib import Path

# ─── Data File Fixtures ──────────────────────────────────────────────
# Each is the smallest file that exercises a specific parser branch.

BULK_CU_DATA = """\
LAMMPS data file for bulk Cu (atom_style atomic)

4 atoms
1 atom types

0.0 3.615 xlo xhi
0.0 3.615 ylo yhi
0.0 3.615 zlo zhi

Masses

1 63.546 # Cu

Atoms # atomic

1 1 0.000 0.000 0.000
2 1 1.808 1.808 0.000
3 1 1.808 0.000 1.808
4 1 0.000 1.808 1.808
"""

NACL_DATA = """\
LAMMPS data file for NaCl (atom_style charge)

8 atoms
2 atom types

0.0 5.64 xlo xhi
0.0 5.64 ylo yhi
0.0 5.64 zlo zhi

Masses

1 22.990 # Na
2 35.453 # Cl

Atoms # charge

1 1  1.0 0.00 0.00 0.00
2 2 -1.0 2.82 0.00 0.00
3 1  1.0 2.82 2.82 0.00
4 2 -1.0 0.00 2.82 0.00
5 1  1.0 0.00 0.00 2.82
6 2 -1.0 2.82 0.00 2.82
7 1  1.0 2.82 2.82 2.82
8 2 -1.0 0.00 2.82 2.82
"""

WATER_FULL_DATA = """\
LAMMPS data file for water (atom_style full)

6 atoms
2 bonds
1 angles
2 atom types
1 bond types
1 angle types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.999 # O
2  1.008 # H

Pair Coeffs

1 0.1553 3.166
2 0.0    0.0

Bond Coeffs

1 600.0 1.0

Angle Coeffs

1 75.0 109.47

Atoms # full

1 1 1 -0.8476 2.0 2.0 2.0
2 1 2  0.4238 2.8 2.5 2.0
3 1 2  0.4238 1.2 2.5 2.0
4 2 1 -0.8476 6.0 6.0 6.0
5 2 2  0.4238 6.8 6.5 6.0
6 2 2  0.4238 5.2 6.5 6.0

Bonds

1 1 1 2
2 1 1 3

Angles

1 1 2 1 3
"""

SI_BULK_DATA = """\
LAMMPS data file for Si (atom_style atomic)

8 atoms
1 atom types

0.0 5.431 xlo xhi
0.0 5.431 ylo yhi
0.0 5.431 zlo zhi

Masses

1 28.086 # Si

Atoms # atomic

1 1 0.000 0.000 0.000
2 1 2.716 2.716 0.000
3 1 2.716 0.000 2.716
4 1 0.000 2.716 2.716
5 1 1.358 1.358 1.358
6 1 4.073 4.073 1.358
7 1 4.073 1.358 4.073
8 1 1.358 4.073 4.073
"""

CU_SLAB_DATA = """\
LAMMPS data file for Cu slab with vacuum (atom_style atomic)

4 atoms
1 atom types

0.0 3.615 xlo xhi
0.0 3.615 ylo yhi
0.0 50.00 zlo zhi

Masses

1 63.546 # Cu

Atoms # atomic

1 1 0.000 0.000 10.000
2 1 1.808 1.808 10.000
3 1 1.808 0.000 11.808
4 1 0.000 1.808 11.808
"""

MGO_DATA = """\
LAMMPS data file for MgO (atom_style charge)

8 atoms
2 atom types

0.0 4.212 xlo xhi
0.0 4.212 ylo yhi
0.0 4.212 zlo zhi

Masses

1 24.305 # Mg
2 15.999 # O

Atoms # charge

1 1  2.0 0.000 0.000 0.000
2 2 -2.0 2.106 0.000 0.000
3 1  2.0 2.106 2.106 0.000
4 2 -2.0 0.000 2.106 0.000
5 1  2.0 0.000 0.000 2.106
6 2 -2.0 2.106 0.000 2.106
7 1  2.0 2.106 2.106 2.106
8 2 -2.0 0.000 2.106 2.106
"""

BIOMOLECULAR_DATA = """\
LAMMPS data file for alanine dipeptide (atom_style full, AMBER)

10 atoms
9 bonds
3 atom types
2 bond types
0 angle types
0 dihedral types
0 improper types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 12.011 # C
2 14.007 # N
3  1.008 # H

Pair Coeffs

1 0.0860 3.3997
2 0.1700 3.2500
3 0.0157 2.6495

Bond Coeffs

1 317.0 1.522
2 337.0 1.449

Atoms # full

1  1 1 0.5973 5.0 5.0 5.0
2  1 2 -0.4157 6.5 5.0 5.0
3  1 3 0.2719 5.0 6.0 5.0
4  1 3 0.2719 5.0 5.0 6.0
5  1 1 0.5973 8.0 5.0 5.0
6  1 2 -0.4157 9.5 5.0 5.0
7  1 3 0.2719 8.0 6.0 5.0
8  1 3 0.2719 8.0 5.0 6.0
9  1 3 0.2719 6.5 6.0 5.0
10 1 3 0.2719 9.5 6.0 5.0

Bonds

1 1 1 2
2 2 2 3
3 2 2 4
4 1 2 5
5 2 5 6
6 2 5 7
7 2 5 8
8 2 6 9
9 2 6 10
"""


# ─── Script Fixtures ─────────────────────────────────────────────────

VALID_METAL_SCRIPT = """\
units metal
atom_style atomic
boundary p p p

read_data cu_bulk.data

pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 100
thermo_style custom step temp press pe ke etotal vol lx ly lz

minimize 1.0e-6 1.0e-8 10000 100000

velocity all create 300.0 12345 dist gaussian
fix 1 all npt temp 300.0 300.0 0.1 aniso 0.0 0.0 1.0
timestep 0.001

dump 1 all custom 1000 trajectory.dump id type x y z
run 100000
write_data final.data
"""

VALID_BIOMOLECULAR_SCRIPT = """\
units real
atom_style full
boundary p p p

read_data system.data

pair_style lj/charmm/coul/long 10.0 12.0
pair_modify mix arithmetic
bond_style harmonic
angle_style harmonic
dihedral_style fourier
improper_style cvff
special_bonds amber
kspace_style pppm 1.0e-5

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

fix SHAKE all shake 1.0e-5 100 0 m 1.008

thermo 1000
thermo_style custom step temp press pe ke etotal density vol

minimize 1.0e-4 1.0e-6 5000 50000

velocity all create 300.0 12345 dist gaussian
fix NPT all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
timestep 2.0

dump 1 all dcd 5000 trajectory.dcd
restart 50000 checkpoint.1 checkpoint.2
run 500000
write_data final.data
"""

VALID_IONIC_SCRIPT = """\
units real
atom_style charge
boundary p p p

read_data nacl.data

pair_style buck/coul/long 10.0
pair_coeff 1 1 0.0 1.0 0.0
pair_coeff 1 2 1227.2 0.3066 0.0
pair_coeff 2 2 3400.0 0.3200 0.0
kspace_style pppm 1.0e-5

thermo 500
minimize 1.0e-4 1.0e-6 5000 50000

velocity all create 300.0 54321
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
timestep 1.0

run 200000
"""

VALID_SLAB_SCRIPT = """\
units metal
atom_style atomic
boundary p p s

read_data cu_slab.data

pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu

group bottom id 1 2
group mobile id 3 4

fix freeze bottom setforce 0.0 0.0 0.0

thermo 100
minimize 1.0e-6 1.0e-8 10000 100000

velocity mobile create 300.0 12345
fix 1 mobile nvt temp 300.0 300.0 0.1
timestep 0.001

run 50000
"""

# ── Error scripts (each has exactly one known problem) ──

ERROR_KSPACE_WITH_EAM = """\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
kspace_style pppm 1.0e-5
timestep 0.001
run 1000
"""

ERROR_NO_UNITS = """\
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
kspace_style pppm 1.0e-5
timestep 2.0
run 1000
"""

ERROR_COEFF_BEFORE_STYLE = """\
units real
atom_style charge
boundary p p p
read_data nacl.data
pair_coeff 1 1 0.0 1.0 0.0
pair_style buck/coul/long 10.0
kspace_style pppm 1.0e-5
timestep 1.0
run 1000
"""

ERROR_REAXFF_NO_QEQ = """\
units real
atom_style charge
boundary p p p
read_data system.data
pair_style reaxff NULL
pair_coeff * * ffield.reax C H O
timestep 0.25
run 10000
"""

ERROR_COUL_NO_KSPACE = """\
units real
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
timestep 2.0
fix 1 all nvt temp 300.0 300.0 100.0
run 1000
"""

ERROR_BOND_WITH_ATOMIC = """\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
bond_style harmonic
timestep 0.001
run 1000
"""

ERROR_NVT_NPT_SAME_GROUP = """\
units real
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
kspace_style pppm 1.0e-5
fix 1 all nvt temp 300.0 300.0 100.0
fix 2 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
timestep 2.0
run 1000
"""

ERROR_NO_RUN = """\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
timestep 0.001
"""

ERROR_UNRESOLVED_VARS = """\
units real
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
kspace_style pppm 1.0e-5
fix 1 all nvt temp ${temperature} ${temperature} 100.0
timestep 2.0
run 1000
"""

ERROR_METAL_TIMESTEP_REAL_UNITS = """\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
timestep 2.0
run 1000
"""

WARNING_SLAB_NPT_ISO = """\
units metal
atom_style atomic
boundary p p s
read_data cu_slab.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
timestep 0.001
run 1000
"""

WARNING_NO_SHAKE_2FS = """\
units real
atom_style full
boundary p p p
read_data system.data
pair_style lj/cut/coul/long 12.0
kspace_style pppm 1.0e-5
fix 1 all nvt temp 300.0 300.0 100.0
timestep 2.0
run 1000
"""

WARNING_METAL_TDAMP_REAL_UNITS = """\
units metal
atom_style atomic
boundary p p p
read_data cu_bulk.data
pair_style eam/alloy
pair_coeff * * Cu_mishin1.eam.alloy Cu
fix 1 all nvt temp 300.0 300.0 100.0
timestep 0.001
run 1000
"""


VALID_CREATE_ATOMS_SCRIPT = """\
units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
timestep 0.005
run 250
"""


# ─── Pytest Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def fixtures_dir(tmp_path):
    """Create a temporary directory with all fixture files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()

    # Write data files
    data_files = {
        "cu_bulk.data": BULK_CU_DATA,
        "nacl.data": NACL_DATA,
        "water_spc.data": WATER_FULL_DATA,
        "si_bulk.data": SI_BULK_DATA,
        "cu_slab.data": CU_SLAB_DATA,
        "mgo.data": MGO_DATA,
        "protein.data": BIOMOLECULAR_DATA,
    }
    for name, content in data_files.items():
        (data_dir / name).write_text(content)

    # Write valid scripts
    valid_scripts = {
        "valid_metal.lammps": VALID_METAL_SCRIPT,
        "valid_bio.lammps": VALID_BIOMOLECULAR_SCRIPT,
        "valid_ionic.lammps": VALID_IONIC_SCRIPT,
        "valid_slab.lammps": VALID_SLAB_SCRIPT,
        "valid_create_atoms.lammps": VALID_CREATE_ATOMS_SCRIPT,
    }
    for name, content in valid_scripts.items():
        (script_dir / name).write_text(content)

    # Write error scripts
    error_scripts = {
        "err_kspace_eam.lammps": ERROR_KSPACE_WITH_EAM,
        "err_no_units.lammps": ERROR_NO_UNITS,
        "err_coeff_before_style.lammps": ERROR_COEFF_BEFORE_STYLE,
        "err_reaxff_no_qeq.lammps": ERROR_REAXFF_NO_QEQ,
        "err_coul_no_kspace.lammps": ERROR_COUL_NO_KSPACE,
        "err_bond_atomic.lammps": ERROR_BOND_WITH_ATOMIC,
        "err_nvt_npt_same.lammps": ERROR_NVT_NPT_SAME_GROUP,
        "err_no_run.lammps": ERROR_NO_RUN,
        "err_unresolved.lammps": ERROR_UNRESOLVED_VARS,
        "err_metal_timestep.lammps": ERROR_METAL_TIMESTEP_REAL_UNITS,
    }
    for name, content in error_scripts.items():
        (script_dir / name).write_text(content)

    # Write warning scripts
    warning_scripts = {
        "warn_slab_npt.lammps": WARNING_SLAB_NPT_ISO,
        "warn_no_shake.lammps": WARNING_NO_SHAKE_2FS,
        "warn_metal_tdamp.lammps": WARNING_METAL_TDAMP_REAL_UNITS,
    }
    for name, content in warning_scripts.items():
        (script_dir / name).write_text(content)

    # Create dummy potential file for valid_metal to find
    (script_dir / "Cu_mishin1.eam.alloy").write_text("# dummy potential\n")

    return tmp_path


@pytest.fixture
def data_dir(fixtures_dir):
    return fixtures_dir / "data"


@pytest.fixture
def script_dir(fixtures_dir):
    return fixtures_dir / "scripts"
