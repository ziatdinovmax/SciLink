import os
import subprocess
import tempfile
import logging
from typing import Optional, Dict, List, Union, Tuple


class VMDLAMMPSConverter:
    """
    A class for converting PDB files to LAMMPS data files using VMD.
    
    This class uses VMD's Topo Tools plugin to create LAMMPS data files with
    full molecular topology information (bonds, angles, etc.).
    
    Includes post-processing to fix molecule ID assignments that VMD/topotools
    gets wrong when multiple chains have overlapping residue numbers.
    """
    
    def __init__(self,
                 vmd_path: Optional[str] = None,
                 working_dir: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the converter.
        
        Args:
            vmd_path: Path to the VMD executable. If None, tries to find it in PATH.
            working_dir: Directory to store temporary files and output. If None, uses a temp directory.
            log_level: Logging level (default: logging.INFO).
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Find VMD executable
        self.vmd_path = vmd_path or self._find_vmd()
        if not self.vmd_path:
            raise ValueError("VMD executable not found. Please provide the path to VMD.")
            
        # Setup working directory
        if working_dir:
            self.working_dir = working_dir
            os.makedirs(working_dir, exist_ok=True)
            self.temp_dir = None
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.working_dir = self.temp_dir.name
            
        self.logger.info(f"Initialized VMD-LAMMPS converter. Working directory: {self.working_dir}")
        
    def __del__(self):
        """Clean up temporary directory if created."""
        if self.temp_dir:
            self.temp_dir.cleanup()
            
    def _find_vmd(self) -> Optional[str]:
        """
        Try to find the VMD executable in the system PATH.
        
        Returns:
            Path to VMD executable if found, None otherwise.
        """
        vmd_names = ["vmd", "vmd.exe"]
        
        for name in vmd_names:
            try:
                path = subprocess.check_output(
                    ["which", name],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                if path:
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
        # Check common installation directories
        common_paths = [
            "/usr/local/bin/vmd",
            "/opt/vmd/bin/vmd",
            "C:\\Program Files\\VMD\\vmd.exe",
            "C:\\Program Files (x86)\\VMD\\vmd.exe",
            "/Applications/VMD 1.9.4a57-arm64-Rev12.app/Contents/MacOS/vmd",
        ]
        
        for path in common_paths:
            if os.path.isfile(path):
                return path
                
        return None
        
    def convert(self,
                pdb_file: str,
                output_file: Optional[str] = None,
                box_dimensions: Optional[Union[float, Tuple[float, float, float]]] = None,
                options: Optional[Dict[str, Union[str, bool, int]]] = None) -> str:
        """
        Convert a PDB file to a LAMMPS data file using VMD.
        
        Args:
            pdb_file: Path to the input PDB file.
            output_file: Path for the output LAMMPS data file. If None, generates a name.
            box_dimensions: Either a single float for cubic box or tuple of (x, y, z).
            options: Dictionary of conversion options:
                - autobonds: Whether to automatically generate bonds (default: True)
                - retypebonds: Whether to retype bonds (default: True)
                - guessangles: Whether to guess angles (default: True)
                - guess_dihedrals: Whether to guess dihedrals (default: False)
                - guess_impropers: Whether to guess impropers (default: False)
                - style: LAMMPS data file style (default: 'full')
                - center_system: Center molecules in the box (default: True)
                - fix_molecule_ids: Fix molecule IDs using PDB info (default: True)
                
        Returns:
            Path to the generated LAMMPS data file.
        """
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
            
        # Set default options
        default_options = {
            "autobonds": True,
            "retypebonds": True,
            "guessangles": True,
            "guess_dihedrals": False,
            "guess_impropers": False,
            "style": "full",
            "center_system": True,
            "fix_molecule_ids": True,  # Enable post-processing fix by default
        }
        
        if options:
            default_options.update(options)
        options = default_options
        
        # Handle box dimensions
        if box_dimensions is None:
            import re
            filename = os.path.basename(pdb_file)
            match = re.search(r'(\d+)A', filename)
            if match:
                box_dimensions = float(match.group(1))
                self.logger.info(f"Extracted box size from filename: {box_dimensions}Å")
            else:
                box_dimensions = 40.0
                self.logger.warning(
                    f"Box dimensions not provided. Using default: {box_dimensions}Å cubic box"
                )
            
        # Generate output filename if not provided
        if not output_file:
            base_name = os.path.splitext(os.path.basename(pdb_file))[0]
            output_file = os.path.join(self.working_dir, f"{base_name}.data")
            
        # Create and run VMD script
        script_path = self._create_vmd_script(pdb_file, output_file, options, box_dimensions)
        
        self.logger.info(f"Converting {pdb_file} to LAMMPS data file...")
        self._run_vmd_script(script_path)
        
        if not os.path.isfile(output_file):
            raise RuntimeError(f"Failed to generate LAMMPS data file: {output_file}")
        
        # Post-process to fix molecule IDs
        if options.get("fix_molecule_ids", True):
            self.logger.info("Post-processing: Fixing molecule IDs from PDB...")
            self.fix_molecule_ids_from_pdb(output_file, pdb_file)
            
        self.logger.info(f"Successfully created LAMMPS data file: {output_file}")
        return output_file
        
    def _create_vmd_script(self,
                          pdb_file: str,
                          output_file: str,
                          options: Dict[str, Union[str, bool, int]],
                          box_dimensions: Union[float, Tuple[float, float, float]]) -> str:
        """
        Create VMD/TCL script for conversion.
        
        Note: Even though we set molecule IDs here, topotools may override them.
        The fix_molecule_ids_from_pdb() post-processing step ensures correct IDs.
        """
        script_path = os.path.join(self.working_dir, "convert_to_lammps.tcl")
        
        # Handle box dimensions
        if isinstance(box_dimensions, (int, float)):
            box_x = box_y = box_z = float(box_dimensions)
        elif isinstance(box_dimensions, (list, tuple)) and len(box_dimensions) == 3:
            box_x, box_y, box_z = [float(dim) for dim in box_dimensions]
        else:
            raise ValueError(f"Invalid box dimensions: {box_dimensions}")
        
        autobonds = "yes" if options.get("autobonds", True) else "no"
        
        script_content = f'''# VMD script to convert PDB to LAMMPS data file
# Generated by VMDLAMMPSConverter

package require topotools
package require pbctools

# Load PDB file
mol new "{os.path.abspath(pdb_file)}" autobonds {autobonds} waitfor all
puts "Loaded PDB file"

# Set periodic box dimensions
pbc set {{{box_x} {box_y} {box_z}}} -all
puts "Set box dimensions to {box_x} x {box_y} x {box_z} Angstroms"

'''
        # Center system if requested
        if options.get("center_system", True):
            script_content += f'''# Center system in box
set all [atomselect top all]
set center [measure center $all]
$all moveby [vecscale -1.0 $center]
$all moveby [list {box_x/2} {box_y/2} {box_z/2}]
$all delete
puts "Centered system in box"

'''

        # Atom type consolidation
        script_content += '''# Consolidate atom types by element
topo retypeatoms element
puts "Consolidated atom types by element"

'''
        
        # Topology operations
        if options.get("retypebonds", True):
            script_content += '''topo retypebonds
puts "Retyped bonds"
'''
        
        if options.get("guessangles", True):
            script_content += '''topo guessangles
puts "Guessed angles"
'''
        
        if options.get("guess_dihedrals", False):
            script_content += '''topo guessdihedrals
puts "Guessed dihedrals"
'''
        
        if options.get("guess_impropers", False):
            script_content += '''topo guessimpropers
puts "Guessed impropers"
'''

        # Write output
        script_content += f'''
# Write LAMMPS data file
topo writelammpsdata "{os.path.abspath(output_file)}" {options.get("style", "full")}
puts "Wrote LAMMPS data file"

# Summary
set all [atomselect top all]
puts ""
puts "=== Conversion Summary ==="
puts "Total atoms: [$all num]"
puts "Atom types: [llength [lsort -unique [$all get type]]]"
$all delete

exit
'''

        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def _run_vmd_script(self, script_path: str) -> None:
        """Run a VMD script."""
        cmd = [self.vmd_path, "-dispdev", "text", "-e", script_path]
        
        self.logger.debug(f"Running VMD command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.debug(result.stdout)
            
            if result.stderr:
                self.logger.warning(f"VMD warnings:\n{result.stderr}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VMD execution failed: {e}")
            self.logger.error(f"VMD stderr: {e.stderr}")
            raise RuntimeError("VMD execution failed. See log for details.")
    
    def fix_molecule_ids_from_pdb(self, data_file: str, pdb_file: str) -> str:
        """
        Fix molecule IDs in LAMMPS data file using PDB chain+resid information.
        
        VMD/topotools assigns molecule IDs based on resid alone, which causes
        atoms from different chains with the same resid to be grouped together.
        This method corrects that by matching atoms by their ORDER in the file,
        since VMD preserves atom ordering from the PDB.
        
        Args:
            data_file: Path to LAMMPS data file to fix (modified in place)
            pdb_file: Path to original PDB file with correct residue info
            
        Returns:
            Path to the fixed data file
        """
        self.logger.info(f"Fixing molecule IDs using PDB chain+resid information")
        
        # Step 1: Parse PDB to get atom order -> (chain, resid) mapping
        # VMD preserves atom order, so we can match by index
        pdb_atoms = []  # List of {"chain": ..., "resid": ..., "element": ...}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        atom_name = line[12:16].strip()
                        chain = line[21].strip() or "A"
                        resid = int(line[22:26].strip())
                        resname = line[17:20].strip()
                        element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                        
                        pdb_atoms.append({
                            "chain": chain,
                            "resid": resid,
                            "resname": resname,
                            "element": element,
                            "atom_name": atom_name
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse PDB line: {line.strip()}")
                        continue
        
        self.logger.info(f"Parsed {len(pdb_atoms)} atoms from PDB")
        
        # Step 2: Create globally unique molecule ID mapping from chain+resid
        unique_molecules = {}  # (chain, resid) -> unique_mol_id
        mol_counter = 0
        
        # Process in order to maintain consistent numbering
        for atom in pdb_atoms:
            key = (atom["chain"], atom["resid"])
            if key not in unique_molecules:
                mol_counter += 1
                unique_molecules[key] = mol_counter
        
        self.logger.info(f"Created {mol_counter} unique molecule IDs")
        
        # Count molecules by chain for logging
        chain_mol_counts = {}
        for (chain, resid) in unique_molecules.keys():
            chain_mol_counts[chain] = chain_mol_counts.get(chain, 0) + 1
        self.logger.info(f"Molecules by chain: {chain_mol_counts}")
        
        # Step 3: Read data file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Step 4: Find and fix atom lines
        new_lines = []
        in_atoms = False
        atom_index = 0  # Track which PDB atom we're on
        fixed_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Detect Atoms section
            if stripped.startswith("Atoms"):
                in_atoms = True
                new_lines.append(line)
                continue
            
            # Detect end of Atoms section
            if in_atoms and stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]:
                in_atoms = False
                new_lines.append(line)
                continue
            
            # Skip empty lines and comments in Atoms section
            if in_atoms and (not stripped or stripped.startswith("#")):
                new_lines.append(line)
                continue
            
            # Process atom lines
            if in_atoms and stripped:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = int(parts[0])
                        old_mol_id = int(parts[1])
                        atom_type = int(parts[2])
                        charge = parts[3]
                        x, y, z = parts[4], parts[5], parts[6]
                        
                        # Get correct molecule ID from PDB atom at same index
                        if atom_index < len(pdb_atoms):
                            pdb_atom = pdb_atoms[atom_index]
                            chain = pdb_atom["chain"]
                            resid = pdb_atom["resid"]
                            new_mol_id = unique_molecules[(chain, resid)]
                            fixed_count += 1
                        else:
                            self.logger.warning(f"Atom {atom_id} has no corresponding PDB atom")
                            new_mol_id = old_mol_id
                        
                        atom_index += 1
                        
                        # Preserve comment if present
                        comment = ""
                        if "#" in line:
                            comment = " #" + line.split("#", 1)[1].rstrip("\n")
                        
                        # Reconstruct line with fixed molecule ID
                        new_line = (
                            f"{atom_id:>8} {new_mol_id:>8} {atom_type:>4} {charge:>12} "
                            f"{x:>14} {y:>14} {z:>14}{comment}\n"
                        )
                        new_lines.append(new_line)
                        continue
                        
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse atom line: {stripped}")
            
            # Keep all other lines unchanged
            new_lines.append(line)
        
        self.logger.info(f"Fixed molecule IDs for {fixed_count} atoms (matched by atom order)")
        
        if atom_index != len(pdb_atoms):
            self.logger.warning(
                f"Atom count mismatch: data file has {atom_index} atoms, "
                f"PDB has {len(pdb_atoms)} atoms"
            )
        
        # Step 5: Write fixed data file
        with open(data_file, 'w') as f:
            f.writelines(new_lines)
        
        self.logger.info(f"Wrote fixed data file: {data_file}")
        
        return data_file

    def calculate_box_from_pdb(self, pdb_file: str, padding: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate appropriate box dimensions from PDB file atom positions.
        
        Args:
            pdb_file: Path to the PDB file.
            padding: Extra space to add around the system in Ångstroms.
            
        Returns:
            Tuple of (x_size, y_size, z_size) for box dimensions.
        """
        script_path = os.path.join(self.working_dir, "calc_box_size.tcl")
        output_path = os.path.join(self.working_dir, "box_size.txt")
        
        script_content = f'''package require topotools
mol new "{os.path.abspath(pdb_file)}" waitfor all
set all [atomselect top all]
set minmax [measure minmax $all]
set min_coords [lindex $minmax 0]
set max_coords [lindex $minmax 1]
set padding {padding}
set xsize [expr [lindex $max_coords 0] - [lindex $min_coords 0] + 2*$padding]
set ysize [expr [lindex $max_coords 1] - [lindex $min_coords 1] + 2*$padding]
set zsize [expr [lindex $max_coords 2] - [lindex $min_coords 2] + 2*$padding]
set outfile [open "{output_path}" w]
puts $outfile "$xsize $ysize $zsize"
close $outfile
$all delete
exit
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        self._run_vmd_script(script_path)
        
        with open(output_path, 'r') as f:
            box_dims = f.read().strip().split()
            x_size, y_size, z_size = map(float, box_dims)
            
        self.logger.info(f"Calculated box dimensions: {x_size:.2f} x {y_size:.2f} x {z_size:.2f} Å")
        return (x_size, y_size, z_size)
    
    def add_box_to_pdb(self, 
                       input_pdb: str, 
                       box_dimensions: Union[float, Tuple[float, float, float]]) -> str:
        """
        Add box dimensions to a PDB file by adding a CRYST1 record.
        
        Args:
            input_pdb: Path to the input PDB file.
            box_dimensions: Box dimensions as float (cubic) or tuple (x, y, z).
            
        Returns:
            Path to the modified PDB file.
        """
        if isinstance(box_dimensions, (int, float)):
            box_x = box_y = box_z = float(box_dimensions)
        elif isinstance(box_dimensions, (list, tuple)) and len(box_dimensions) == 3:
            box_x, box_y, box_z = [float(dim) for dim in box_dimensions]
        else:
            raise ValueError("Box dimensions must be float or tuple of 3 floats.")
        
        output_pdb = os.path.join(
            self.working_dir, 
            f"{os.path.splitext(os.path.basename(input_pdb))[0]}_with_box.pdb"
        )
        
        cryst_line = f"CRYST1{box_x:9.3f}{box_y:9.3f}{box_z:9.3f}  90.00  90.00  90.00 P 1           1\n"
        
        with open(input_pdb, 'r') as f_in:
            content = f_in.readlines()
            
        with open(output_pdb, 'w') as f_out:
            f_out.write(cryst_line)
            f_out.writelines(content)
            
        self.logger.info(f"Created PDB with box dimensions: {output_pdb}")
        return output_pdb
    
    def verify_molecule_assignments(self, data_file: str) -> Dict[str, any]:
        """
        Verify molecule assignments in a LAMMPS data file.
        
        Useful for debugging to see what molecules were detected.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary with molecule statistics
        """
        molecules = {}  # mol_id -> list of (atom_id, atom_type)
        type_elements = {}  # type_id -> element name
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Parse Masses section for type -> element mapping
        in_masses = False
        for line in lines:
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not stripped or stripped.startswith("Atoms") or stripped.startswith("Pair"):
                    break
                if stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2 and "#" in stripped:
                    try:
                        type_id = int(parts[0])
                        comment = stripped.split("#")[1].strip()
                        element = comment.split()[0] if comment else "?"
                        type_elements[type_id] = element
                    except (ValueError, IndexError):
                        pass
        
        # Parse Atoms section
        in_atoms = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]:
                    break
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        atom_id = int(parts[0])
                        mol_id = int(parts[1])
                        atom_type = int(parts[2])
                        
                        if mol_id not in molecules:
                            molecules[mol_id] = []
                        molecules[mol_id].append((atom_id, atom_type))
                    except (ValueError, IndexError):
                        pass
        
        # Analyze molecule compositions
        compositions = {}  # formula -> count
        for mol_id, atoms in molecules.items():
            elements = [type_elements.get(t, "?") for _, t in atoms]
            elem_counts = {}
            for e in elements:
                elem_counts[e] = elem_counts.get(e, 0) + 1
            
            # Build formula (C first, H second, then alphabetical)
            formula_parts = []
            for el in ["C", "H"]:
                if el in elem_counts:
                    cnt = elem_counts[el]
                    formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
            for el in sorted(elem_counts.keys()):
                if el not in ["C", "H"]:
                    cnt = elem_counts[el]
                    formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
            
            formula = "".join(formula_parts)
            compositions[formula] = compositions.get(formula, 0) + 1
        
        result = {
            "total_molecules": len(molecules),
            "type_elements": type_elements,
            "compositions": compositions
        }
        
        self.logger.info(f"Molecule verification:")
        self.logger.info(f"  Total molecules: {result['total_molecules']}")
        self.logger.info(f"  Type elements: {result['type_elements']}")
        self.logger.info(f"  Compositions:")
        for formula, count in sorted(compositions.items(), key=lambda x: -x[1]):
            self.logger.info(f"    {formula}: {count}")
        
        return result
