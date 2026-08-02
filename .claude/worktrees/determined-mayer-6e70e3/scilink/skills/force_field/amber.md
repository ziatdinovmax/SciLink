## Overview

The AMBER (Assisted Model Building with Energy Refinement) family of force
fields is the standard choice for biomolecular simulations of proteins, nucleic
acids, lipids, carbohydrates, and small organic molecules. For LAMMPS
simulations, the parameterization pipeline uses **AmberTools** (antechamber,
tleap, parmchk2) to build topology and assign charges, then **ParmEd** to
convert to a LAMMPS data file that embeds all coefficients and charges.

### When to use AMBER

- Proteins (any size): ff14SB, ff19SB
- DNA: OL15 or BSC1
- RNA: OL3
- Lipid membranes: Lipid21 (or Lipid17)
- Carbohydrates / glycans: GLYCAM06j-1
- Small organic molecules, ligands, solvents: GAFF2 (preferred) or GAFF
- Solvated systems with explicit water: TIP3P, SPC/E, OPC, TIP4P-Ew
- Monovalent ions (Na+, K+, Cl-): Joung-Cheatham (auto-loaded with water model)
- Divalent ions (Mg2+, Ca2+, Zn2+): 12-6 LJ sets from Aqvist or Li-Merz

### When NOT to use AMBER

- CHARMM-ecosystem molecules (use CGenFF / CHARMM36m instead)
- OPLS-parameterized systems
- Coarse-grained models (Martini, SIRAH)
- Reactive chemistry (use ReaxFF)
- Bulk metals or inorganic crystals without organic components

### Key advantage for LAMMPS

ParmEd writes a self-contained LAMMPS data file that includes Masses, Pair
Coeffs, Bond Coeffs, Angle Coeffs, Dihedral Coeffs, Improper Coeffs, Atoms
(with partial charges), and full bonded topology. No separate parameter include
file is needed -- the read_data command loads everything.


## Planning

### Force field variant selection

| System Component   | Recommended FF   | leaprc File                    | Notes                           |
|--------------------|------------------|--------------------------------|---------------------------------|
| Proteins           | ff19SB           | leaprc.protein.ff19SB          | Best with OPC water             |
| Proteins (legacy)  | ff14SB           | leaprc.protein.ff14SB          | Best with TIP3P water           |
| DNA                | OL15             | leaprc.DNA.OL15                | Current standard                |
| RNA                | OL3              | leaprc.RNA.OL3                 | Current standard                |
| Lipids             | Lipid21          | leaprc.lipid21                 | Successor to Lipid17            |
| Carbohydrates      | GLYCAM06j-1      | leaprc.GLYCAM_06j-1            | Standard AMBER glycan FF        |
| Small molecules    | GAFF2            | leaprc.gaff2                   | Improved torsions over GAFF     |
| Small mol (legacy) | GAFF             | leaprc.gaff                    | Larger validation set           |

### Water model selection

| Water Model | leaprc                   | Best Paired With | 3-site? | Notes                    |
|-------------|--------------------------|------------------|---------|--------------------------|
| TIP3P       | leaprc.water.tip3p       | ff14SB           | Yes     | Fast, standard           |
| SPC/E       | leaprc.water.spce        | ff14SB           | Yes     | Better bulk water props  |
| OPC         | leaprc.water.opc         | ff19SB           | Yes     | Best accuracy with ff19SB|
| OPC3        | leaprc.water.opc3        | ff19SB           | Yes     | 3-point, good accuracy   |
| TIP4P-Ew   | leaprc.water.tip4pew     | ff14SB           | No      | 4-site, needs virtual    |

Decision rule: Use OPC with ff19SB. Use TIP3P with ff14SB. Avoid TIP4P
in LAMMPS unless specifically needed (virtual site handling adds complexity).

### Charge method selection for small molecules

| Method    | antechamber flag | Accuracy | Speed            | When to Use                      |
|-----------|------------------|----------|------------------|----------------------------------|
| AM1-BCC   | -c bcc           | Good     | Seconds-minutes  | Default for most molecules       |
| RESP      | -c resp          | Best     | Minutes-hours    | Publication-quality studies       |
| Gasteiger | -c gas           | Low      | Instant          | Quick screening, debugging       |
| Mulliken  | -c mul           | Low      | Needs QM         | Rarely used in practice          |

Decision rule: Use AM1-BCC (bcc) unless the research goal explicitly
demands RESP-quality charges or the molecule is very large (>200 atoms, where
bcc may be slow -- fall back to gas for initial tests).

### Pipeline decision flowchart

1. Does the system contain non-standard residues (ligands, cofactors, novel molecules)?
   - YES: Run antechamber + parmchk2 on each non-standard residue first
   - NO: Skip directly to tleap

2. Does the system need explicit solvent?
   - YES: Use solvatebox in tleap with appropriate water box
   - NO: Skip solvation (gas-phase or implicit solvent)

3. Does the system need to be neutralized?
   - YES: Use addIonsRand SYS Na+ 0 Cl- 0 for neutralization
   - Additional salt? Add extra ion pairs for target concentration

4. Are there disulfide bonds?
   - YES: Add bond SYS.X.SG SYS.Y.SG commands in tleap

5. Are there non-standard protonation states?
   - YES: Rename HIS to HID/HIE/HIP, ASP to ASH, GLU to GLH in PDB before tleap


## Analysis

### System composition analysis for AMBER

When analyzing a molecular system for AMBER parameterization, identify:

1. Standard biomolecular components (handled directly by tleap):
   - Standard amino acid residues (ALA, ARG, ASN, ... VAL)
   - Standard nucleotides (DA, DT, DG, DC, A, U, G, C)
   - Common lipids with Lipid21 residue names
   - Water (WAT, HOH, TIP3, TIP4)
   - Monovalent ions (Na+, K+, Cl-, Li+, Rb+, Cs+, F-, Br-, I-)

2. Non-standard residues requiring antechamber:
   - Drug molecules, inhibitors, substrates
   - Cofactors not in the AMBER library (some like HEM have AMBER params)
   - Modified amino acids or nucleotides
   - Organic solvents (if not in standard library)
   - Any residue that tleap reports as unknown

3. Special structural features:
   - Disulfide bonds (CYS-CYS with S-gamma distance < 2.5 A)
   - Metal coordination sites (may need bonded or nonbonded model)
   - Covalent modifications (phosphorylation, glycosylation, etc.)
   - Non-standard protonation (HID/HIE/HIP, ASH, GLH, CYM, LYN)

4. System size considerations:
   - < 50,000 atoms: Standard pipeline, AM1-BCC charges fine
   - 50,000-500,000 atoms: Standard pipeline, may want to pre-minimize in AMBER
   - > 500,000 atoms: Consider whether full atomistic AMBER is needed

### Element-to-component mapping heuristics

- C, H, N, O, S in protein-like ratios: protein
- P + ribose sugars: nucleic acid
- Long hydrocarbon chains + phospholipid head: lipid
- C, H, O in sugar-like ratios: carbohydrate
- Isolated small organic residue (< 100 atoms): small molecule, needs GAFF2


## Interpretation

### Interpreting antechamber output

- Successful run: Produces MOL2 file with GAFF2 atom types and AM1-BCC charges.
  Verify charges sum to the expected net charge (within +/-0.001 e).
- sqm failure: The semi-empirical QM step failed. Common causes:
  - Incorrect net charge specified
  - Unusual bonding pattern that AM1 cannot handle
  - Very large molecule (>200 heavy atoms) -- try -c gas first
- Missing atom types: Some atoms could not be typed. Check the molecule
  structure for unusual valences or radicals.

### Interpreting parmchk2 output

- No ATTN markers: All parameters found in the GAFF/GAFF2 database. Good.
- ATTN markers present: Parameters were estimated by analogy. These should be
  reviewed manually for accuracy. The more ATTN markers, the less reliable the
  parameterization. Consider QM-based parameterization for critical parameters.

### Interpreting tleap output

- "Added X missing atoms": tleap added hydrogens or completed residues. Normal.
- "Could not find unit": A residue in the PDB has no matching force field entry.
  Must parameterize with antechamber first, or rename to the correct AMBER name.
- "close contact": Atoms are too close. May need energy minimization.
- "WARNING: The unperturbed charge is not zero": System is not neutral.
  Add addIonsRand to neutralize, or verify this is intentional.

### Interpreting ParmEd conversion

- File size check: A solvated protein system (~50K atoms) should produce a
  data file of roughly 5-20 MB. If the file is suspiciously small, conversion
  may have failed.
- All-zero charges: ParmEd failed to transfer charges. Verify the prmtop
  contains charges with cpptraj or by inspecting the prmtop file directly.
- Missing coefficient sections: ParmEd version may be too old. Requires
  ParmEd >= 3.4 for correct LAMMPS output.


## Validation

### Required checks after parameterization

1. Charge neutrality: Total system charge should be 0.0 (or the intended
   net charge). Sum all charges in the Atoms section of the LAMMPS data file.
   Tolerance: |total| < 0.01 e.

2. Molecular charge integrity: Each molecule type should have the expected
   charge. Water: 0.0. Na+: +1.0. Cl-: -1.0. Protein: typically 0 after
   neutralization. Verify per-molecule charge sums.

3. Section completeness: The LAMMPS data file must contain:
   - Masses (all atom types)
   - Pair Coeffs (LJ epsilon and sigma for each type)
   - Bond Coeffs (if bonds > 0)
   - Angle Coeffs (if angles > 0)
   - Dihedral Coeffs (if dihedrals > 0)
   - Improper Coeffs (if impropers > 0)
   - Atoms section with correct format for atom_style full

4. Parameter sanity ranges (AMBER, real units):
   - LJ epsilon: 0.0-0.5 kcal/mol (most atoms 0.01-0.25)
   - LJ sigma: 1.0-4.5 A (H ~1.0, C ~3.4, O ~3.1)
   - Bond K: 100-1000 kcal/mol/A^2
   - Bond r0: 0.9-2.5 A
   - Angle K: 10-200 kcal/mol/rad^2
   - Angle theta0: 90-180 degrees
   - Charges: |q| typically < 1.5 e for organic atoms

5. No zero-epsilon non-hydrogen atoms: In AMBER, only lone-pair virtual
   sites should have epsilon=0. If a real atom (C, N, O, S) has epsilon=0,
   something went wrong.

6. ATTN parameter review: If parmchk2 produced ATTN-marked parameters,
   flag them as low-confidence. For production simulations, these should be
   validated against QM calculations or experimental data.

### LAMMPS compatibility checks

- pair_style lj/charmm/coul/long 10.0 12.0 or lj/cut/coul/long 12.0
- special_bonds amber (sets 1-4 scaling: scee=1/1.2, scnb=1/2.0)
- bond_style harmonic
- angle_style harmonic
- dihedral_style fourier (AMBER uses Fourier series for dihedrals)
- improper_style cvff (AMBER improper torsions)
- kspace_style pppm 1.0e-5
- units real
- atom_style full


## Implementation

### Full pipeline pseudocode

INPUT: pdb_file, research_goal, small_molecule_info (optional)

1. CHECK TOOLS
   Verify: antechamber, tleap, parmchk2, parmed all available

2. ANALYZE SYSTEM
   Detect: proteins, DNA, RNA, lipids, carbs, small molecules, ions, water
   Identify: non-standard residues needing antechamber

3. CLEAN PDB (optional)
   pdb4amber -i input.pdb -o cleaned.pdb --dry --nohyd

4. PARAMETERIZE SMALL MOLECULES (for each non-standard residue)
   antechamber -i ligand.pdb -fi pdb -o ligand.mol2 -fo mol2 -c bcc -at gaff2 -nc CHARGE -pf y
   parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s 2

5. BUILD TOPOLOGY (write and run tleap script)
   source leaprc.gaff2              (if small molecules)
   source leaprc.protein.ff19SB     (if proteins)
   source leaprc.DNA.OL15           (if DNA)
   source leaprc.RNA.OL3            (if RNA)
   source leaprc.lipid21            (if lipids)
   source leaprc.GLYCAM_06j-1       (if carbohydrates)
   source leaprc.water.opc          (water model)
   LIG = loadmol2 ligand.mol2       (for each small molecule)
   loadamberparams ligand.frcmod
   SYS = loadpdb cleaned.pdb
   solvatebox SYS OPCBOX 12.0       (if solvating)
   addIonsRand SYS Na+ 0 Cl- 0      (if neutralizing)
   check SYS
   saveamberparm SYS system.prmtop system.inpcrd

6. CONVERT TO LAMMPS
   ParmEd: load prmtop + inpcrd, save as system.data

7. VALIDATE
   Check: sections present, charges sum correctly, parameters in sane ranges

8. GENERATE LAMMPS INPUT HEADER
   units real
   atom_style full
   pair_style lj/charmm/coul/long 10.0 12.0
   pair_modify mix arithmetic
   kspace_style pppm 1.0e-5
   special_bonds amber
   bond_style harmonic
   angle_style harmonic
   dihedral_style fourier
   improper_style cvff
   read_data system.data

### LAMMPS pair_style notes for AMBER

The standard AMBER to LAMMPS workflow uses lj/charmm/coul/long because it
supports the inner/outer cutoff switching that AMBER expects. An acceptable
alternative is lj/cut/coul/long 12.0 with pair_modify tail yes.

The special_bonds amber command sets:
- 1-2 interactions: excluded (factor 0.0)
- 1-3 interactions: excluded (factor 0.0)
- 1-4 interactions: LJ scaled by 1/2.0 = 0.5, Coulomb scaled by 1/1.2 = 0.8333

### Common antechamber commands

Standard small molecule (neutral):
antechamber -i mol.pdb -fi pdb -o mol.mol2 -fo mol2 -c bcc -at gaff2 -nc 0 -pf y

Charged ligand (e.g., -1 charge):
antechamber -i mol.pdb -fi pdb -o mol.mol2 -fo mol2 -c bcc -at gaff2 -nc -1 -pf y

Fast charges for testing:
antechamber -i mol.pdb -fi pdb -o mol.mol2 -fo mol2 -c gas -at gaff2 -nc 0 -pf y

From SDF input:
antechamber -i mol.sdf -fi sdf -o mol.mol2 -fo mol2 -c bcc -at gaff2 -nc 0 -pf y

### ParmEd conversion (Python)

import parmed as pmd
system = pmd.load_file("system.prmtop", xyz="system.inpcrd")
system.save("system.data", overwrite=True)
