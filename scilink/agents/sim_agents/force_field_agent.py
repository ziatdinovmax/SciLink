import os
import re
import logging
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
from MDAnalysis import Universe
import numpy as np

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from ._deprecation import normalize_params


class ForceFieldAgent:
    """
    AI-driven agent for optimal force field selection and parameter acquisition.
    
    This agent leverages LLMs to analyze molecular systems and research goals, then:
    1. Selects the most appropriate force field based on system composition and research objectives
    2. Determines the best method to obtain parameters (database, manual, QM, etc.)
    3. Executes the chosen parameterization strategy
    4. Validates parameters for scientific rigor
    
    The agent works in conjunction with other simulation agents in the pipeline.
    """
    
    def __init__(self, 
                 working_dir: str, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the ForceFieldAgent.
        
        Args:
            working_dir: Directory for output files and intermediate calculations
            api_key: API key for the LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="ForceFieldAgent"
        )
        
        # Initialize model using wrapper structure
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy.")

            self.logger.info(f"ForceFieldAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"ForceFieldAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None
            
        # Force field databases and methods
        self.ff_databases = {
            "water": ["SPC/E", "TIP3P", "TIP4P", "OPC", "TIP5P"],
            "proteins": ["AMBER ff14SB", "AMBER ff19SB", "CHARMM36m", "OPLS-AA/M"],
            "lipids": ["CHARMM36", "Slipids", "Lipid17", "Martini"],
            "carbohydrates": ["GLYCAM06", "CHARMM36-carb", "GROMOS 56A(CARBO)"],
            "small_molecules": ["GAFF", "GAFF2", "CGenFF", "OpenFF"],
            "ions": ["Joung-Cheatham", "CHARMM", "Åqvist", "Li-Merz"],
            "polymers": ["PCFF", "COMPASS", "OPLS-AA"],
            "metals": ["EAM", "COMB", "ReaxFF"],
            "interfaces": ["INTERFACE-AMBER", "CHARMM-METAL"]
        }
        
        # Parameter acquisition methods
        self.param_methods = {
            "database": {
                "description": "Direct parameter extraction from established databases",
                "tools": ["ParmEd", "AmberTools", "CGenFF", "MATCH"],
                "strengths": ["Fast", "Reliable for standard molecules"]
            },
            "analogy": {
                "description": "Parameters by chemical analogy to known molecules",
                "tools": ["ffTK", "LigParGen", "CGenFF"],
                "strengths": ["Good balance of speed and accuracy"]
            },
            "quantum": {
                "description": "Ab initio parameterization from QM calculations",
                "tools": ["ffTK", "ForceBalance", "poltype2"],
                "strengths": ["Highest accuracy", "Novel molecules"]
            }
        }

        self._mass_to_element = self._build_mass_lookup()
    
    
    def _build_mass_lookup(self) -> Dict[str, float]:
        """Build a dictionary mapping element symbols to their atomic masses."""
        from ase.data import atomic_masses, chemical_symbols
        import math
    
        lookup = {}
        for i, symbol in enumerate(chemical_symbols):
            if i == 0:  # Skip dummy
                continue
            mass = atomic_masses[i]
            if mass is not None and mass > 0:
                try:
                    if not math.isnan(mass):
                        lookup[symbol] = mass
                except (TypeError, ValueError):
                    pass
    
        return lookup
    
    
    def _guess_element_from_mass(self, mass: float, tolerance: float = 0.5) -> str:
        """
        Guess element from atomic mass using ASE's periodic table data.
    
        Args:
            mass: Atomic mass to identify
            tolerance: Maximum allowed difference from reference mass
    
        Returns:
            Element symbol or 'X' if unknown
        """
        best_match = "X"
        best_diff = float('inf')
    
        for symbol, ref_mass in self._mass_to_element.items():
            diff = abs(mass - ref_mass)
    
            if diff < tolerance:
                return symbol
    
            if diff < best_diff:
                best_diff = diff
                best_match = symbol
    
        if best_diff > tolerance:
            self.logger.warning(
                f"Could not identify element for mass {mass:.4f}. "
                f"Closest: {best_match} (diff: {best_diff:.4f})"
            )
    
        return best_match

    def _generate_json(self, prompt: str) -> dict:
        """Generate JSON response from LLM."""
        self.logger.info("Sending request to LLM...")
        
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            
            raw_text = response.text.strip()
            
            # Handle markdown code blocks if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw_text = "\n".join(json_lines)
            
            # Try to find JSON object if there's extra text
            if not raw_text.startswith("{"):
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]
            
            return json.loads(raw_text)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise

    def _generate_text(self, prompt: str) -> str:
        """Generate text response from LLM."""
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise
        
    def select_force_field(self, 
                         pdb_file: str, 
                         research_goal: str,
                         system_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze system composition and research goal to select optimal force field.
        
        Args:
            pdb_file: Path to the PDB file containing the molecular system
            research_goal: Research objective in natural language
            system_description: Optional description of the system
            
        Returns:
            Dictionary with force field selection and parameterization information
        """
        self.logger.info(f"Analyzing system in {pdb_file} for force field selection")
        
        # Analyze the system composition
        system_info = self._analyze_system_composition(pdb_file)
        
        if not system_description:
            system_description = self._generate_system_description(system_info)
            
        self.logger.info(f"System description: {system_description}")
        
        # Select force field using LLM
        force_field_selection = self._select_optimal_force_field(
            system_info=system_info,
            research_goal=research_goal,
            system_description=system_description
        )
        
        # Determine parameter acquisition method
        param_method = self._determine_parameter_method(
            system_info=system_info,
            force_field=force_field_selection["force_field"],
            research_goal=research_goal
        )
        
        # Log the decisions
        self.logger.info(f"Selected force field: {force_field_selection['force_field']}")
        self.logger.info(f"Parameter acquisition method: {param_method['method']}")
        
        # Create the result dictionary
        result = {
            "system_info": system_info,
            "system_description": system_description,
            "force_field": force_field_selection,
            "parameter_method": param_method,
            "working_dir": self.working_dir
        }
        
        # Save selection info to file
        selection_file = os.path.join(self.working_dir, "force_field_selection.json")
        with open(selection_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
    
    def acquire_parameters(self, 
                           selection_info: Dict[str, Any], 
                           data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire force field parameters using the selected method.
        
        Args:
            selection_info: Force field selection info from select_force_field()
            data_file: Optional existing LAMMPS data file to enhance
            
        Returns:
            Dictionary with parameter files and information
        """
        method = selection_info["parameter_method"]["method"]
        force_field_info = selection_info["force_field"]
        force_field = force_field_info.get("force_field", "OPLS-AA")
        system_info = selection_info["system_info"]
        
        self.logger.info(f"Acquiring parameters via {method} method for {force_field}")
        
        # For now, override quantum/analogy to database
        # These methods require external tools and manual intervention
        if method in ["quantum", "analogy"]:
            self.logger.warning(
                f"Method '{method}' requires external tools. "
                f"Falling back to 'database' method for automated pipeline."
            )
            method = "database"
        
        # Initialize parameters structure
        params = {
            "source": method,
            "force_field": force_field,
            "force_field_info": force_field_info,
            "system_info": system_info,
            "atom_types": {},
            "parameter_files": {},
            "lammps_settings": {
                "pair_style": force_field_info.get("lammps_pair_style", "lj/cut/coul/long 10.0"),
                "bond_style": force_field_info.get("lammps_bond_style", "harmonic"),
                "angle_style": force_field_info.get("lammps_angle_style", "harmonic"),
                "dihedral_style": force_field_info.get("lammps_dihedral_style", "opls"),
                "kspace_style": "pppm 1.0e-4",
                "special_bonds": "lj/coul 0.0 0.0 0.5",
            }
        }
        
        # If data_file is provided, extract atom types from it
        if data_file and os.path.exists(data_file):
            self.logger.info(f"Extracting atom types from data file: {data_file}")
            try:
                atom_types = self._extract_atom_types_from_data(data_file)
                params["atom_types"] = atom_types
                self.logger.info(f"Found {len(atom_types)} atom types in data file")
            except Exception as e:
                self.logger.warning(f"Could not extract atom types from data file: {e}")
        
        # Generate LJ parameters for atom types
        self.logger.info("Generating force field parameters...")
        try:
            generated_params = self._generate_parameters_with_llm(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
            params.update(generated_params)
        except Exception as e:
            self.logger.error(f"Parameter generation failed: {e}")
            params["generation_error"] = str(e)
        
        # Validate parameters
        self.logger.info("Validating parameters...")
        validation = self._validate_parameters(params, system_info)
        params["validation"] = validation
        
        if validation.get("errors"):
            self.logger.warning(f"Parameter validation errors: {validation['errors']}")
        if validation.get("warnings"):
            self.logger.info(f"Parameter validation warnings: {validation['warnings']}")
        
        # Generate summary
        params["summary"] = self._generate_parameter_summary(params, selection_info)
        
        # Save parameter info
        param_file = os.path.join(self.working_dir, "parameter_info.json")
        try:
            with open(param_file, 'w') as f:
                serializable_params = {
                    k: v for k, v in params.items() 
                    if k not in ["raw_data", "quantum_results"]
                }
                json.dump(serializable_params, f, indent=2)
            self.logger.info(f"Saved parameter info to {param_file}")
        except Exception as e:
            self.logger.warning(f"Could not save parameter info: {e}")
        
        return params

    def generate_lammps_parameters(self, 
                               parameter_info: Dict[str, Any], 
                               data_file: str) -> Dict[str, str]:
        """
        Generate LAMMPS parameter files based on the acquired parameters.
        
        Uses the actual data file to ensure type consistency.
        
        Args:
            parameter_info: Parameter info from acquire_parameters()
            data_file: Path to LAMMPS data file to enhance
            
        Returns:
            Dictionary with paths to parameter files
        """
        self.logger.info(f"Generating LAMMPS parameter files for {data_file}")
        
        # Use the data-file-aware method to ensure type consistency
        param_content = self._generate_lammps_parameters_from_data(
            data_file=data_file,
            force_field_info={"force_field": parameter_info.get("force_field", {})}
        )
        
        # Write parameter files
        files = {}
        
        # Main parameter file
        param_file = os.path.join(self.working_dir, "ff_params.lammps")
        with open(param_file, 'w') as f:
            f.write(param_content["main"])
        files["main"] = param_file
        
        # Additional files if needed
        if "additional" in param_content:
            for name, content in param_content["additional"].items():
                if content:  # Only write non-empty content
                    add_file = os.path.join(self.working_dir, f"{name}.lammps")
                    with open(add_file, 'w') as f:
                        f.write(content)
                    files[name] = add_file
                
        self.logger.info(f"Generated {len(files)} parameter files")
        return files

    # ================================
    # PRIVATE METHODS
    # ================================
    
    def _analyze_system_composition(self, pdb_file: str) -> Dict[str, Any]:
        """
        Analyze the molecular system's composition from a PDB file.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary with system information
        """
        self.logger.info(f"Analyzing composition of {pdb_file}")
        
        try:
            # Load the PDB using MDAnalysis
            u = Universe(pdb_file)
            
            # Count atoms by element
            elements = {}
            for atom in u.atoms:
                # Get element from atom name (first character, or first two if second is lowercase)
                if len(atom.name) > 1 and atom.name[1].islower():
                    element = atom.name[:2]
                else:
                    element = atom.name[0]
                    
                elements[element] = elements.get(element, 0) + 1
                
            # Identify molecular components
            has_water = self._detect_water(u, elements)
            has_proteins = self._detect_proteins(u)
            has_lipids = self._detect_lipids(u)
            has_nucleic_acids = self._detect_nucleic_acids(u)
            has_ions = self._detect_ions(elements)
            has_small_molecules = self._detect_small_molecules(u, elements)
            has_metals = self._detect_metals(elements)
            has_carbohydrates = self._detect_carbohydrates(u)
            
            # Identify special system characteristics
            is_interface = self._detect_interface(u)
            is_gas_phase = self._detect_gas_phase(u)
            
            # Calculate basic system dimensions
            box_dimensions = u.dimensions[:3] if hasattr(u, 'dimensions') and u.dimensions is not None else [0, 0, 0]
            
            system_info = {
                "filename": os.path.basename(pdb_file),
                "n_atoms": len(u.atoms),
                "n_residues": len(u.residues),
                "n_molecules": len(u.segments),
                "elements": elements,
                "composition": {
                    "water": has_water,
                    "proteins": has_proteins,
                    "lipids": has_lipids,
                    "nucleic_acids": has_nucleic_acids, 
                    "ions": has_ions,
                    "small_molecules": has_small_molecules,
                    "metals": has_metals,
                    "carbohydrates": has_carbohydrates
                },
                "system_type": {
                    "interface": is_interface,
                    "gas_phase": is_gas_phase
                },
                "box_dimensions": box_dimensions.tolist() if isinstance(box_dimensions, np.ndarray) else box_dimensions,
            }
            
            # Add residue details if available
            if has_proteins or has_nucleic_acids or has_small_molecules:
                residue_names = [res.resname for res in u.residues]
                residue_counts = {}
                for name in residue_names:
                    residue_counts[name] = residue_counts.get(name, 0) + 1
                system_info["residue_counts"] = residue_counts
                
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing PDB file: {e}")
            # Return minimal info if analysis fails
            return {
                "filename": os.path.basename(pdb_file),
                "n_atoms": 0,
                "elements": {},
                "composition": {
                    "water": False,
                    "proteins": False,
                    "lipids": False,
                    "nucleic_acids": False,
                    "ions": False,
                    "small_molecules": False,
                    "metals": False,
                    "carbohydrates": False
                }
            }
    
    def _detect_water(self, universe, elements):
        """Detect if system contains water molecules."""
        # Check for standard water residue names
        water_residues = ['WAT', 'HOH', 'H2O', 'SOL', 'TIP', 'SPC']
        for res in universe.residues:
            if any(water in res.resname for water in water_residues):
                return True
                
        # Check element ratio - water has 2:1 H:O ratio
        # Only useful for simple systems
        if 'H' in elements and 'O' in elements:
            h_atoms = elements.get('H', 0)
            o_atoms = elements.get('O', 0)
            if h_atoms > 0 and o_atoms > 0 and (h_atoms / o_atoms) > 1.5:
                return True
                
        return False
    
    def _detect_proteins(self, universe):
        """Detect if system contains protein molecules."""
        # Standard amino acid residues
        amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                      'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                      'TYR', 'VAL']
        
        for res in universe.residues:
            if res.resname in amino_acids:
                return True
        return False
    
    def _detect_lipids(self, universe):
        """Detect if system contains lipid molecules."""
        # Common lipid residue names
        lipid_residues = ['POPC', 'POPE', 'DPPC', 'DOPC', 'DMPC', 'CHOL', 'CHL', 
                         'DLPE', 'DLPC', 'DSPC', 'DAPC', 'DOPE', 'POPG', 'DPPG']
        
        for res in universe.residues:
            if res.resname in lipid_residues:
                return True
                
        # Alternative detection: large residues with many carbons and some phosphorus
        return False
    
    def _detect_nucleic_acids(self, universe):
        """Detect if system contains DNA or RNA."""
        # Nucleotide residue names
        nucleotides = ['ADE', 'THY', 'GUA', 'CYT', 'URA', 'A', 'T', 'G', 'C', 'U',
                      'DA', 'DT', 'DG', 'DC', 'DU', 'AMP', 'GMP', 'CMP', 'TMP', 'UMP']
        
        for res in universe.residues:
            if res.resname in nucleotides:
                return True
        return False
    
    def _detect_ions(self, elements):
        """Detect if system contains common ions."""
        ion_elements = ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Li']
        return any(ion in elements for ion in ion_elements)
    
    def _detect_small_molecules(self, universe, elements):
        """Detect if system contains small organic molecules."""
        if 'C' in elements and elements['C'] > 0:
            # Filter out known biomolecules
            if not self._detect_proteins(universe) and not self._detect_lipids(universe) and not self._detect_nucleic_acids(universe):
                return True
        return False
    
    def _detect_metals(self, elements):
        """Detect if system contains metal atoms."""
        metal_elements = ['Fe', 'Zn', 'Cu', 'Ni', 'Co', 'Mn', 'Mg', 'Ca', 'Na', 'K', 
                         'Al', 'Ti', 'V', 'Cr', 'Pd', 'Pt', 'Au', 'Ag', 'Hg']
        return any(metal in elements for metal in metal_elements)
    
    def _detect_carbohydrates(self, universe):
        """Detect if system contains carbohydrates."""
        # Common carbohydrate/sugar residue names
        carb_residues = ['GLC', 'GAL', 'MAN', 'FUC', 'XYL', 'NAG', 'SIA', 'RIB', 
                        'AGLC', 'BGLC', 'GLCA', 'GLCN']
        
        for res in universe.residues:
            if res.resname in carb_residues:
                return True
        return False
    
    def _detect_interface(self, universe):
        """Detect if system represents an interface."""
        # Simple heuristic: check if there are empty regions in z dimension
        try:
            z_coords = universe.atoms.positions[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            z_range = z_max - z_min
            
            # Divide into bins along z
            n_bins = 20
            hist, edges = np.histogram(z_coords, bins=n_bins)
            
            # Look for empty regions in the middle (interface)
            # Ignore top and bottom 15% which might be naturally empty
            middle_bins = hist[int(n_bins*0.15):int(n_bins*0.85)]
            if min(middle_bins) < max(middle_bins) * 0.1:  # If some middle bins are nearly empty
                return True
                
            return False
        except:
            return False
    
    def _detect_gas_phase(self, universe):
        """Detect if system is in gas phase (no bulk solvent)."""
        # Heuristic: if no water and low density, likely gas phase
        has_water = self._detect_water(universe, {})
        
        if not has_water and hasattr(universe, 'dimensions') and universe.dimensions is not None:
            try:
                # Calculate rough density in g/cm^3
                volume = universe.dimensions[0] * universe.dimensions[1] * universe.dimensions[2] / 1000  # A^3 to nm^3
                n_atoms = len(universe.atoms)
                # Rough average: 12 g/mol per atom (mainly C)
                density = (n_atoms * 12) / (volume * 0.6022)  # g/cm^3
                
                # Gas phase typically has very low density
                if density < 0.1:  # g/cm^3
                    return True
            except:
                pass
                
        return False
    
    def _generate_system_description(self, system_info: Dict[str, Any]) -> str:
        """
        Generate a human-readable description of the system.
        
        Args:
            system_info: System information from _analyze_system_composition
            
        Returns:
            Human-readable description
        """
        description_parts = []
        
        # Add information about major components
        comp = system_info["composition"]
        
        if comp["water"]:
            description_parts.append("water")
            
        if comp["ions"]:
            ion_elements = [e for e in system_info.get("elements", {}) if e in ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe']]
            if ion_elements:
                description_parts.append("+".join(ion_elements) + " ions")
            else:
                description_parts.append("ions")
                
        if comp["proteins"]:
            n_residues = system_info.get("n_residues", 0)
            if n_residues > 300:
                description_parts.append("large protein")
            else:
                description_parts.append("protein")
                
        if comp["lipids"]:
            description_parts.append("lipids")
            
        if comp["nucleic_acids"]:
            description_parts.append("nucleic acids")
            
        if comp["carbohydrates"]:
            description_parts.append("carbohydrates")
            
        if comp["small_molecules"]:
            description_parts.append("organic molecules")
            
        if comp["metals"]:
            description_parts.append("metals")
            
        # Add system type
        system_type = system_info["system_type"]
        if system_type["interface"]:
            description_parts.append("interface")
            
        if system_type["gas_phase"]:
            description_parts.append("gas phase")
            
        # Combine into description
        if description_parts:
            description = " with ".join(description_parts)
        else:
            description = "molecular system"
            
        return f"{description} ({system_info['n_atoms']} atoms)"
    
    def _select_optimal_force_field(self,
                                    system_info: Dict[str, Any],
                                    research_goal: str,
                                    system_description: str) -> Dict[str, Any]:
        """Select optimal force field using LLM."""
        
        prompt = f"""
    You are an expert in molecular dynamics force field selection for LAMMPS simulations.
    
    SYSTEM INFORMATION:
    - Total atoms: {system_info.get('n_atoms', 'Unknown')}
    - Elements present: {system_info.get('elements', {})}
    - Composition: {system_info.get('composition', {})}
    - System description: {system_description}
    
    RESEARCH GOAL:
    {research_goal}
    
    TARGET MD SOFTWARE: LAMMPS
    
    Select the optimal force field for this LAMMPS simulation. Consider:
    1. Scientific accuracy for the research goals
    2. **LAMMPS compatibility** - the force field must be implementable with standard LAMMPS pair_styles
    3. Parameter availability for all species
    4. Computational efficiency
    
    IMPORTANT LAMMPS CONSTRAINTS:
    - Standard pair_styles: lj/cut/coul/long, lj/charmm/coul/long, buck/coul/long
    - For metal ions, consider: simple point charge models (Joung-Cheatham, Aqvist) 
      which work with standard LJ, rather than 12-6-4 models requiring special implementation
    - Water models: TIP3P, TIP4P, SPC/E are well-supported
    - For organic molecules: OPLS-AA, GAFF, CHARMM are well-supported
    
    Return a JSON object with:
    {{
        "force_field": "<primary force field name>",
        "compatible_water_model": "<water model>",
        "ion_parameters": "<ion parameter set>",
        "justification": "<scientific reasoning>",
        "lammps_pair_style": "<recommended LAMMPS pair_style>",
        "alternatives": ["<alternative 1>", "<alternative 2>"],
        "cautions": "<important considerations>",
        "parameter_availability": "high|medium|low"
    }}
    """
        
        return self._generate_json(prompt)

    def _determine_parameter_method(self, 
                                system_info: Dict[str, Any],
                                force_field: str,
                                research_goal: str) -> Dict[str, Any]:
        """
        Determine the best method to obtain force field parameters.
        
        Args:
            system_info: System information from _analyze_system_composition
            force_field: Selected force field name
            research_goal: Research objective in natural language
            
        Returns:
            Dictionary with parameter acquisition method and details
        """
        self.logger.info(f"Determining parameter acquisition method for {force_field}")
        
        # Format system composition for prompt
        composition = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in composition.items()])
        
        # Format parameter method info for prompt
        method_info = ""
        for method, details in self.param_methods.items():
            tools = ", ".join(details["tools"])
            strengths = ", ".join(details["strengths"])
            method_info += f"- {method.upper()}: {details['description']}\n  Tools: {tools}\n  Strengths: {strengths}\n\n"
        
        # Create the prompt for the LLM
        prompt = f"""
        As an expert in molecular dynamics parameterization, determine the best method to obtain force field parameters.
        
        SELECTED FORCE FIELD: {force_field}
        
        SYSTEM COMPOSITION:
        {comp_str}
        
        RESEARCH GOAL: "{research_goal}"
        
        PARAMETER ACQUISITION METHODS:
        {method_info}
        
        Based on this information, what is the best method to obtain parameters for this system?
        Consider these factors:
        1. Parameter availability for the selected force field
        2. Presence of non-standard molecules requiring custom parameterization
        3. Accuracy requirements implied by the research goal
        4. Computational resources and time constraints
        5. Availability of QM-level parameterization tools if needed
        
        Provide your response as JSON with this structure:
        {{
            "method": "database|analogy|quantum",
            "justification": "Detailed scientific explanation for this choice",
            "recommended_tools": ["Tool1", "Tool2"],
            "estimated_effort": "low|medium|high",
            "specific_approaches": [
                "Detailed step-by-step approaches to obtain parameters"
            ]
        }}
        Include only the JSON response with no additional text.
        """
        
        try:
            param_method = self._generate_json(prompt)
            
            # Ensure all expected fields are present
            param_method.setdefault("method", "database")
            param_method.setdefault("recommended_tools", [])
            param_method.setdefault("estimated_effort", "medium")
            param_method.setdefault("specific_approaches", [])
            
            return param_method
            
        except Exception as e:
            self.logger.error(f"Error determining parameter method: {e}")
            # Fallback to database method
            return {
                "method": "database",
                "justification": "Default selection due to LLM analysis failure.",
                "recommended_tools": ["ParmEd", "AmberTools"],
                "estimated_effort": "medium",
                "specific_approaches": ["Use standard force field databases"]
            }
    
    def _acquire_parameters_from_database(self, 
                                       force_field: str,
                                       system_info: Dict[str, Any],
                                       data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters from established force field databases.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} from databases")
        
        # Determine which database files to use based on force field
        ff_files = self._determine_force_field_files(force_field, system_info)
        
        # Extract parameter data from database files
        parameters = {
            "source": "database",
            "force_field": force_field,
            "parameter_files": ff_files,
            "atom_types": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {},
            "impropers": {},
            "nonbonded": {},
        }
        
        # If data_file is provided, extract atom types from it
        if data_file:
            atom_types = self._extract_atom_types_from_data(data_file)
            parameters["atom_types"] = atom_types
            
        # Generate parameter data using LLM
        parameters.update(self._generate_parameters_with_llm(force_field, system_info, data_file))
        
        # Log the acquired parameters
        self.logger.info(f"Acquired parameters for {len(parameters['atom_types'])} atom types")
        
        return parameters
    
    def _acquire_parameters_by_analogy(self,
                                    force_field: str,
                                    system_info: Dict[str, Any],
                                    data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters by chemical analogy to known molecules.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} by chemical analogy")
        
        # Start with database parameters as a baseline
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "analogy"
        
        # Identify molecules needing parameterization by analogy
        unique_molecules = self._extract_unique_molecules(system_info)
        
        # For each unique molecule, find analogies and parameters
        analogy_params = {}
        for molecule in unique_molecules:
            if molecule not in parameters["atom_types"]:
                analogy = self._find_molecular_analogy(molecule, force_field)
                if analogy:
                    analogy_params[molecule] = analogy
        
        parameters["analogies"] = analogy_params
        
        # Use LLM to fill in missing parameters based on analogies
        parameters.update(self._enhance_parameters_with_llm(parameters, "analogy"))
        
        return parameters
    
    def _acquire_parameters_from_quantum(self,
                                      force_field: str,
                                      system_info: Dict[str, Any],
                                      data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire parameters from quantum mechanical calculations.
        
        Args:
            force_field: Selected force field name
            system_info: System information from _analyze_system_composition
            data_file: Optional existing LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Acquiring parameters for {force_field} via quantum calculations")
        
        # Start with database parameters for standard components
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "quantum"
        
        # Identify molecules needing QM parameterization
        unique_molecules = self._extract_unique_molecules(system_info)
        standard_molecules = self._identify_standard_molecules(unique_molecules, force_field)
        
        # Molecules needing QM treatment = unique - standard
        qm_needed = [m for m in unique_molecules if m not in standard_molecules]
        
        if not qm_needed:
            self.logger.info("No molecules need QM parameterization, using database parameters")
            return parameters
            
        # In a real implementation, we would run QM calculations here
        # For this agent, we'll simulate QM parameters using the LLM
        
        # Enhance parameters with LLM to simulate QM-derived parameters
        parameters["qm_molecules"] = qm_needed
        parameters.update(self._enhance_parameters_with_llm(parameters, "quantum"))
        
        return parameters
        
    def _determine_force_field_files(self, force_field: str, system_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine which force field database files are needed.
        
        Args:
            force_field: Selected force field name
            system_info: System information
            
        Returns:
            Dictionary mapping parameter types to file paths
        """
        # This would typically map to real force field files
        # Here we'll return placeholders that would be resolved in a real implementation
        
        ff_files = {}
        comp = system_info["composition"]
        
        if "AMBER" in force_field:
            ff_base = "amber"
            if comp["proteins"]:
                ff_files["proteins"] = f"{ff_base}/ff14SB.dat"
            if comp["water"]:
                ff_files["water"] = f"{ff_base}/tip3p.dat"
            if comp["ions"]:
                ff_files["ions"] = f"{ff_base}/ions.dat"
            if comp["nucleic_acids"]:
                ff_files["nucleic_acids"] = f"{ff_base}/DNA.OL15.dat"
            if comp["small_molecules"]:
                ff_files["small_molecules"] = f"{ff_base}/gaff.dat"
                
        elif "CHARMM" in force_field:
            ff_base = "charmm"
            if comp["proteins"]:
                ff_files["proteins"] = f"{ff_base}/prot.prm"
            if comp["water"]:
                ff_files["water"] = f"{ff_base}/water.prm"
            if comp["lipids"]:
                ff_files["lipids"] = f"{ff_base}/lipid.prm"
                
        elif "OPLS" in force_field:
            ff_base = "opls"
            ff_files["main"] = f"{ff_base}/oplsaa.prm"
            
        else:
            # Generic case
            ff_files["main"] = f"generic/{force_field.lower().replace(' ', '_')}.dat"
            
        return ff_files
    
    def _extract_atom_types_from_data(self, data_file: str) -> Dict[str, Any]:
        """
        Extract atom types from a LAMMPS data file.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary mapping atom types to properties
        """
        atom_types = {}
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
            # Find the "Masses" section
            in_masses = False
            for line in lines:
                line = line.strip()
                
                if "Masses" in line:
                    in_masses = True
                    continue
                elif in_masses and line.startswith("#"):
                    continue
                elif in_masses and not line:  # Empty line ends section
                    in_masses = False
                elif in_masses:
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        # Guess element from mass
                        element = self._guess_element_from_mass(mass)
                        atom_types[atom_type] = {
                            "mass": mass,
                            "element": element
                        }
            
            # Look for atom types in "Pair Coeffs" section
            in_pair_coeffs = False
            for line in lines:
                line = line.strip()
                
                if "Pair Coeffs" in line:
                    in_pair_coeffs = True
                    continue
                elif in_pair_coeffs and line.startswith("#"):
                    continue
                elif in_pair_coeffs and not line:  # Empty line ends section
                    in_pair_coeffs = False
                elif in_pair_coeffs:
                    parts = line.split()
                    if len(parts) >= 3:
                        atom_type = int(parts[0])
                        epsilon = float(parts[1])
                        sigma = float(parts[2])
                        if atom_type in atom_types:
                            atom_types[atom_type].update({
                                "epsilon": epsilon,
                                "sigma": sigma
                            })
                        
        except Exception as e:
            self.logger.error(f"Error extracting atom types from data file: {e}")
            
        return atom_types
        
    def _extract_unique_molecules(self, system_info: Dict[str, Any]) -> List[str]:
        """
        Extract unique molecules from system information.
        
        Args:
            system_info: System information from _analyze_system_composition
            
        Returns:
            List of unique molecule names
        """
        unique_molecules = []
        
        # If residue_counts exists, use those as the molecules
        if "residue_counts" in system_info:
            unique_molecules = list(system_info["residue_counts"].keys())
        else:
            # Try to infer molecules from composition
            comp = system_info["composition"]
            if comp["water"]:
                unique_molecules.append("HOH")
            if comp["ions"]:
                elements = system_info.get("elements", {})
                for ion in ["Na", "K", "Cl", "Ca", "Mg"]:
                    if ion in elements:
                        unique_molecules.append(ion)
                        
        return unique_molecules
    
    def _identify_standard_molecules(self, molecules: List[str], force_field: str) -> List[str]:
        """
        Identify which molecules are standard for a given force field.
        
        Args:
            molecules: List of molecule names
            force_field: Force field name
            
        Returns:
            List of standard molecule names
        """
        # Common standard molecules across force fields
        standard_molecules = ["HOH", "WAT", "TIP3", "SOL", "Na", "K", "Cl", "Ca", "Mg"]
        
        # Standard amino acids
        amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                      "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
                      "TYR", "VAL"]
        
        # Standard nucleotides
        nucleotides = ["ADE", "THY", "GUA", "CYT", "URA", "A", "T", "G", "C", "U",
                      "DA", "DT", "DG", "DC", "DU"]
        
        if "AMBER" in force_field or "CHARMM" in force_field:
            standard_molecules.extend(amino_acids)
            standard_molecules.extend(nucleotides)
            
        return [m for m in molecules if m in standard_molecules]
    
    def _find_molecular_analogy(self, molecule: str, force_field: str) -> Dict[str, Any]:
        """
        Find analogous molecules for parameterization by analogy.
        
        Args:
            molecule: Molecule name
            force_field: Force field name
            
        Returns:
            Dictionary with analogy information
        """
        # This would typically involve a database search or structural comparison
        # Here we'll simulate the result
        
        # Example return structure
        return {
            "similar_to": "similar molecule name",
            "similarity": 0.85,  # 0-1 score
            "modifications_needed": ["replace methyl with ethyl"],
            "parameter_adjustments": ["increase C-C bond length by 0.02 Å"]
        }
    
    def _generate_parameters_with_llm(self, 
                                   force_field: str, 
                                   system_info: Dict[str, Any],
                                   data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate force field parameters using LLM.
        
        Args:
            force_field: Selected force field name
            system_info: System information
            data_file: Optional LAMMPS data file
            
        Returns:
            Dictionary with parameter information
        """
        self.logger.info(f"Generating parameters for {force_field} using LLM")
        
        # Extract elements and composition for the prompt
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("elements", {}).items()])
        comp = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in comp.items()])
        
        # Create prompt for LLM
        prompt = f"""
        As an expert in molecular dynamics force fields, provide appropriate LAMMPS parameters for this system.
        
        FORCE FIELD: {force_field}
        
        SYSTEM ELEMENTS: {elements_str}
        
        SYSTEM COMPOSITION:
        {comp_str}
        
        You need to generate scientifically accurate force field parameters for LAMMPS based on the {force_field} force field.
        For each parameter type, provide values compatible with LAMMPS syntax and the selected force field.
        
        Please provide parameters in the following JSON format:
        {{
            "atom_types": {{
                "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}},
                "2": {{"name": "H", "mass": 1.008, "description": "Water hydrogen", "charge": 0.4238, "epsilon": 0.0, "sigma": 0.0}}
            }},
            "bonds": {{
                "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
            }},
            "angles": {{
                "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
            }},
            "dihedrals": {{
                "1": {{"type": "periodic", "atoms": ["X", "X", "X", "X"], "k": 0.0, "d": 1, "n": 1, "description": "Example dihedral"}}
            }},
            "nonbonded_terms": {{
                "mixing_rule": "geometric for epsilon, arithmetic for sigma",
                "cutoff": 10.0
            }}
        }}
        
        Include only parameters relevant to the system composition. The parameters should be scientifically accurate for the {force_field} force field.
        Include only the JSON response with no additional text.
        """
        
        try:
            parameters = self._generate_json(prompt)
            
            # Ensure all parameter categories exist
            for category in ["atom_types", "bonds", "angles", "dihedrals", "nonbonded_terms"]:
                if category not in parameters:
                    parameters[category] = {}
                    
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error generating parameters with LLM: {e}")
            # Return empty parameters
            return {
                "atom_types": {},
                "bonds": {},
                "angles": {},
                "dihedrals": {},
                "nonbonded_terms": {
                    "mixing_rule": "geometric for epsilon, arithmetic for sigma",
                    "cutoff": 10.0
                }
            }
    
    def _enhance_parameters_with_llm(self, 
                                  parameters: Dict[str, Any], 
                                  method: str) -> Dict[str, Any]:
        """
        Enhance parameters using LLM based on the specified method.
        
        Args:
            parameters: Existing parameters
            method: Method being used (analogy or quantum)
            
        Returns:
            Dictionary with enhanced parameter information
        """
        force_field = parameters.get("force_field", "Unknown")
        
        # Create prompt for LLM based on method
        if method == "analogy":
            # Extract analogies for the prompt
            analogies_str = ""
            for molecule, analogy in parameters.get("analogies", {}).items():
                similar_to = analogy.get("similar_to", "unknown")
                similarity = analogy.get("similarity", 0)
                modifications = ", ".join(analogy.get("modifications_needed", []))
                analogies_str += f"- {molecule}: similar to {similar_to} (similarity: {similarity})\n  Modifications: {modifications}\n"
                
            prompt = f"""
            As an expert in molecular force field parameterization by analogy, enhance these parameters.
            
            FORCE FIELD: {force_field}
            
            CURRENT PARAMETERS:
            {json.dumps(parameters.get('atom_types', {}), indent=2)}
            
            MOLECULAR ANALOGIES:
            {analogies_str}
            
            Based on the molecular analogies provided, enhance the parameters to account for the chemical differences.
            Adjust parameters like bond lengths, angles, charges, and non-bonded terms based on chemical intuition and the force field paradigm.
            
            Please provide enhanced parameters in this JSON format:
            {{
                "atom_types": {{
                    "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}
                }},
                "bonds": {{
                    "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
                }},
                "angles": {{
                    "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
                }}
            }}
            Include only the JSON response with no additional text.
            """
            
        elif method == "quantum":
            # Extract QM molecules for the prompt
            qm_molecules = ", ".join(parameters.get("qm_molecules", []))
            
            prompt = f"""
            As an expert in quantum-derived force field parameterization, enhance these parameters.
            
            FORCE FIELD: {force_field}
            
            CURRENT PARAMETERS:
            {json.dumps(parameters.get('atom_types', {}), indent=2)}
            
            MOLECULES NEEDING QM PARAMETERIZATION: {qm_molecules}
            
            Based on your expertise in quantum chemistry and force field development, provide enhanced parameters 
            that would typically be derived from quantum mechanical calculations. Focus on accurate charges, bond, angle,
            and dihedral parameters that reflect the electronic structure of the molecules.
            
            Please provide quantum-derived parameters in this JSON format:
            {{
                "atom_types": {{
                    "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}
                }},
                "bonds": {{
                    "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
                }},
                "angles": {{
                    "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
                }}
            }}
            Include only the JSON response with no additional text.
            """
            
        else:
            # Unknown method
            return parameters
            
        try:
            enhanced_params = self._generate_json(prompt)
            
            # Merge enhanced parameters with original parameters
            for category in ["atom_types", "bonds", "angles", "dihedrals"]:
                if category in enhanced_params:
                    # Add new parameters
                    for key, value in enhanced_params[category].items():
                        if key not in parameters.get(category, {}):
                            if category not in parameters:
                                parameters[category] = {}
                            parameters[category][key] = value
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error enhancing parameters with LLM: {e}")
            return parameters
    
    def _validate_parameters(self, parameters: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for scientific rigor.
        
        Args:
            parameters: Parameter information
            system_info: System information
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "quality_metrics": {}
        }
        
        # Check for missing parameters
        missing_atom_types = []
        elements = system_info.get("elements", {})
        
        # Check if parameters exist for all elements
        for element in elements:
            found = False
            for atom_type in parameters.get("atom_types", {}).values():
                if atom_type.get("name", "") == element or atom_type.get("element", "") == element:
                    found = True
                    break
                    
            if not found:
                missing_atom_types.append(element)
                validation["warnings"].append(f"Missing parameters for element {element}")
                
        if missing_atom_types:
            validation["passed"] = False
            
        # Check for reasonable parameter values
        for atom_type, props in parameters.get("atom_types", {}).items():
            # Check mass
            mass = props.get("mass", 0)
            if mass <= 0:
                validation["errors"].append(f"Invalid mass for atom type {atom_type}: {mass}")
                validation["passed"] = False
                
            # Check charges - should be reasonable values
            charge = props.get("charge", 0)
            if abs(charge) > 2.0:
                validation["warnings"].append(f"Unusual charge for atom type {atom_type}: {charge}")
                
            # Check LJ parameters
            epsilon = props.get("epsilon", 0)
            sigma = props.get("sigma", 0)
            if epsilon < 0:
                validation["errors"].append(f"Invalid epsilon for atom type {atom_type}: {epsilon}")
                validation["passed"] = False
            if sigma <= 0:
                validation["errors"].append(f"Invalid sigma for atom type {atom_type}: {sigma}")
                validation["passed"] = False
                
        # Calculate quality metrics
        param_source = parameters.get("source", "unknown")
        
        # Source-based quality score (0-100)
        source_quality = {
            "database": 70,  # Good baseline
            "analogy": 80,   # Better than database
            "quantum": 95    # Best quality
        }.get(param_source, 50)
        
        # Adjust based on coverage
        coverage = 100 - (len(missing_atom_types) / max(1, len(elements)) * 100)
        
        # Compute combined score
        quality_score = (source_quality * 0.7) + (coverage * 0.3)
        
        validation["quality_metrics"] = {
            "overall_score": int(quality_score),
            "parameter_source": param_source,
            "coverage": int(coverage),
            "missing_elements": missing_atom_types
        }
        
        return validation
    
    def _parse_data_file(self, data_file: str) -> Dict[str, Any]:
        """
        Parse a LAMMPS data file to understand what parameters are needed.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary with data file information
        """
        info = {
            "atom_types": 0,
            "bond_types": 0,
            "angle_types": 0,
            "dihedral_types": 0,
            "improper_types": 0,
            "atoms": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "masses": {},
            "box": []
        }
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for header information
                if "atoms" in line and not "types" in line:
                    info["atoms"] = int(line.split()[0])
                elif "bonds" in line and not "types" in line:
                    info["bonds"] = int(line.split()[0])
                elif "angles" in line and not "types" in line:
                    info["angles"] = int(line.split()[0])
                elif "dihedrals" in line and not "types" in line:
                    info["dihedrals"] = int(line.split()[0])
                elif "impropers" in line and not "types" in line:
                    info["impropers"] = int(line.split()[0])
                elif "atom types" in line:
                    info["atom_types"] = int(line.split()[0])
                elif "bond types" in line:
                    info["bond_types"] = int(line.split()[0])
                elif "angle types" in line:
                    info["angle_types"] = int(line.split()[0])
                elif "dihedral types" in line:
                    info["dihedral_types"] = int(line.split()[0])
                elif "improper types" in line:
                    info["improper_types"] = int(line.split()[0])
                elif "xlo xhi" in line:
                    parts = line.split()
                    info["box"].append((float(parts[0]), float(parts[1])))
                    
            # Extract masses
            in_masses = False
            for line in lines:
                line = line.strip()
                
                if "Masses" in line:
                    in_masses = True
                    continue
                elif in_masses and not line:  # Empty line
                    in_masses = False
                elif in_masses and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        info["masses"][atom_type] = mass
                        
        except Exception as e:
            self.logger.error(f"Error parsing data file: {e}")
            
        return info

    def _generate_lammps_parameters_from_data(self,
                                               data_file: str,
                                               force_field_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LAMMPS parameter file content that EXACTLY matches the data file.
        
        This reads the actual atom types from the data file and generates
        parameters only for those types, ensuring consistency.
        
        Args:
            data_file: Path to LAMMPS data file
            force_field_info: Force field selection info
            
        Returns:
            Dictionary with parameter file content
        """
        self.logger.info(f"Generating parameters matched to data file: {data_file}")
        
        # Step 1: Parse the data file to get EXACT atom types
        data_info = self._parse_data_file_for_charges(data_file)
        
        n_atom_types = len(data_info["masses"])
        n_bond_types = 0
        n_angle_types = 0
        n_dihedral_types = 0
        n_improper_types = 0
        
        # Count bond/angle/dihedral/improper types from data file header
        with open(data_file, 'r') as f:
            for line in f:
                line_lower = line.lower()
                if "bond types" in line_lower:
                    n_bond_types = int(line.split()[0])
                elif "angle types" in line_lower:
                    n_angle_types = int(line.split()[0])
                elif "dihedral types" in line_lower:
                    n_dihedral_types = int(line.split()[0])
                elif "improper types" in line_lower:
                    n_improper_types = int(line.split()[0])
        
        # Build type info string for LLM
        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            mass = data_info["masses"][type_id]
            element = data_info["type_elements"].get(type_id, "X")
            name = data_info["type_names"].get(type_id, element)
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: element={element}, mass={mass:.4f}, count={count}")
        
        type_info_str = "\n".join(type_info)
        
        # Get force field name
        force_field = force_field_info.get("force_field", {})
        if isinstance(force_field, dict):
            ff_name = force_field.get("force_field", "OPLS-AA")
            water_model = force_field.get("compatible_water_model", "TIP3P")
        else:
            ff_name = str(force_field) if force_field else "OPLS-AA"
            water_model = "TIP3P"
        
        self.logger.info(f"Data file has {n_atom_types} atom types, {n_bond_types} bond types, "
                         f"{n_angle_types} angle types, {n_dihedral_types} dihedral types")
        
        prompt = f"""
    Generate a LAMMPS force field parameter file for this molecular system.
    
    CRITICAL CONSTRAINTS - READ CAREFULLY:
    1. The data file has EXACTLY {n_atom_types} atom types (numbered 1 to {n_atom_types})
    2. DO NOT define parameters for atom types outside the range 1-{n_atom_types}
    3. DO NOT include 'mass' commands - masses are already defined in the data file
    4. DO NOT include 'units' or 'atom_style' commands - these are set elsewhere
    
    ATOM TYPES IN THE DATA FILE:
    {type_info_str}
    
    TOPOLOGY FROM DATA FILE:
    - Atom types: {n_atom_types}
    - Bond types: {n_bond_types}
    - Angle types: {n_angle_types}
    - Dihedral types: {n_dihedral_types}
    - Improper types: {n_improper_types}
    
    FORCE FIELD: {ff_name}
    WATER MODEL (if applicable): {water_model}
    
    GENERATE PARAMETERS IN THIS EXACT ORDER:
    
    1. Pair style and settings:
       pair_style lj/cut/coul/long 10.0
       pair_modify mix geometric
       kspace_style pppm 1.0e-4
       special_bonds lj/coul 0.0 0.0 0.5
    
    2. Bond/angle/dihedral styles (ONLY if corresponding types > 0):
       bond_style harmonic          # only if {n_bond_types} > 0
       angle_style harmonic         # only if {n_angle_types} > 0
       dihedral_style opls          # only if {n_dihedral_types} > 0
       improper_style harmonic      # only if {n_improper_types} > 0
    
    3. Pair coefficients for EACH atom type (self-interactions):
       Format: pair_coeff I I epsilon sigma
       - Use appropriate Lennard-Jones parameters for each element
       - epsilon in kcal/mol, sigma in Angstroms
       - Base parameters on the {ff_name} force field
    
    4. Bond coefficients (if {n_bond_types} > 0):
       Format: bond_coeff N K r0
       - K in kcal/mol/Å², r0 in Angstroms
    
    5. Angle coefficients (if {n_angle_types} > 0):
       Format: angle_coeff N K theta0
       - K in kcal/mol/rad², theta0 in degrees
    
    6. Dihedral coefficients (if {n_dihedral_types} > 0):
       Format for OPLS: dihedral_coeff N K1 K2 K3 K4
       - K values in kcal/mol
    
    IMPORTANT REMINDERS:
    - Generate pair_coeff for types 1 through {n_atom_types} ONLY
    - Generate bond_coeff for types 1 through {n_bond_types} ONLY
    - Generate angle_coeff for types 1 through {n_angle_types} ONLY
    - Generate dihedral_coeff for types 1 through {n_dihedral_types} ONLY
    - Do NOT generate any *_coeff commands if that type count is 0
    
    Output ONLY valid LAMMPS commands. No markdown formatting, no explanations, no code blocks.
    """
        
        response = self._generate_text(prompt)
        
        # Clean up response - remove any markdown formatting
        param_content = response.strip()
        if param_content.startswith("```"):
            lines = param_content.split("\n")
            param_content = "\n".join(
                line for line in lines 
                if not line.strip().startswith("```")
            )
        
        # Validate and fix any out-of-range type references
        param_content = self._validate_and_fix_param_types(
            param_content, 
            n_atom_types, 
            n_bond_types, 
            n_angle_types, 
            n_dihedral_types,
            n_improper_types
        )
        
        # Add header comment
        header = f"""# Force field parameters
    # Generated by ForceFieldAgent
    # Force field: {ff_name}
    # Water model: {water_model}
    #
    # Data file type counts:
    #   Atom types: {n_atom_types}
    #   Bond types: {n_bond_types}
    #   Angle types: {n_angle_types}
    #   Dihedral types: {n_dihedral_types}
    #   Improper types: {n_improper_types}
    
    """
        
        return {
            "main": header + param_content,
            "additional": {}
        }            
    
    def _validate_and_fix_param_types(self,
                                       param_content: str,
                                       n_atom_types: int,
                                       n_bond_types: int,
                                       n_angle_types: int,
                                       n_dihedral_types: int,
                                       n_improper_types: int = 0) -> str:
        """
        Validate parameter file and remove any out-of-range type references.
        Also removes mass commands since masses are in the data file.
        
        Args:
            param_content: Raw parameter file content
            n_atom_types: Number of atom types in data file
            n_bond_types: Number of bond types in data file
            n_angle_types: Number of angle types in data file
            n_dihedral_types: Number of dihedral types in data file
            n_improper_types: Number of improper types in data file
            
        Returns:
            Cleaned parameter file content
        """
        lines = param_content.split('\n')
        valid_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                valid_lines.append(line)
                continue
            
            # Remove mass commands - masses are in data file
            if stripped.startswith('mass ') or stripped.startswith('mass\t'):
                self.logger.warning(f"Removing mass command (already in data file): {stripped}")
                continue
            
            # Remove units command - set elsewhere
            if stripped.startswith('units '):
                self.logger.warning(f"Removing units command (set in main script): {stripped}")
                continue
            
            # Remove atom_style command - set elsewhere
            if stripped.startswith('atom_style '):
                self.logger.warning(f"Removing atom_style command (set in main script): {stripped}")
                continue
            
            # Check pair_coeff lines
            if stripped.startswith('pair_coeff'):
                match = re.match(r'pair_coeff\s+(\d+)\s+(\d+)', stripped)
                if match:
                    type1, type2 = int(match.group(1)), int(match.group(2))
                    if type1 > n_atom_types or type2 > n_atom_types:
                        self.logger.warning(f"Removing invalid pair_coeff (types {type1},{type2} > {n_atom_types}): {stripped}")
                        continue
                # Handle wildcard pair_coeff
                elif re.match(r'pair_coeff\s+\*\s+\*', stripped):
                    valid_lines.append(line)
                    continue
                valid_lines.append(line)
                continue
            
            # Check bond_coeff lines
            if stripped.startswith('bond_coeff'):
                if n_bond_types == 0:
                    self.logger.warning(f"Removing bond_coeff (no bond types in data): {stripped}")
                    continue
                match = re.match(r'bond_coeff\s+(\d+)', stripped)
                if match:
                    type_id = int(match.group(1))
                    if type_id > n_bond_types:
                        self.logger.warning(f"Removing invalid bond_coeff (type {type_id} > {n_bond_types}): {stripped}")
                        continue
                valid_lines.append(line)
                continue
            
            # Check angle_coeff lines
            if stripped.startswith('angle_coeff'):
                if n_angle_types == 0:
                    self.logger.warning(f"Removing angle_coeff (no angle types in data): {stripped}")
                    continue
                match = re.match(r'angle_coeff\s+(\d+)', stripped)
                if match:
                    type_id = int(match.group(1))
                    if type_id > n_angle_types:
                        self.logger.warning(f"Removing invalid angle_coeff (type {type_id} > {n_angle_types}): {stripped}")
                        continue
                valid_lines.append(line)
                continue
            
            # Check dihedral_coeff lines
            if stripped.startswith('dihedral_coeff'):
                if n_dihedral_types == 0:
                    self.logger.warning(f"Removing dihedral_coeff (no dihedral types in data): {stripped}")
                    continue
                match = re.match(r'dihedral_coeff\s+(\d+)', stripped)
                if match:
                    type_id = int(match.group(1))
                    if type_id > n_dihedral_types:
                        self.logger.warning(f"Removing invalid dihedral_coeff (type {type_id} > {n_dihedral_types}): {stripped}")
                        continue
                valid_lines.append(line)
                continue
            
            # Check improper_coeff lines
            if stripped.startswith('improper_coeff'):
                if n_improper_types == 0:
                    self.logger.warning(f"Removing improper_coeff (no improper types in data): {stripped}")
                    continue
                match = re.match(r'improper_coeff\s+(\d+)', stripped)
                if match:
                    type_id = int(match.group(1))
                    if type_id > n_improper_types:
                        self.logger.warning(f"Removing invalid improper_coeff (type {type_id} > {n_improper_types}): {stripped}")
                        continue
                valid_lines.append(line)
                continue
            
            # Remove style commands if no corresponding types
            if stripped.startswith('bond_style') and n_bond_types == 0:
                self.logger.warning(f"Removing bond_style (no bond types in data): {stripped}")
                continue
            
            if stripped.startswith('angle_style') and n_angle_types == 0:
                self.logger.warning(f"Removing angle_style (no angle types in data): {stripped}")
                continue
            
            if stripped.startswith('dihedral_style') and n_dihedral_types == 0:
                self.logger.warning(f"Removing dihedral_style (no dihedral types in data): {stripped}")
                continue
            
            if stripped.startswith('improper_style') and n_improper_types == 0:
                self.logger.warning(f"Removing improper_style (no improper types in data): {stripped}")
                continue
            
            # Keep all other valid lines
            valid_lines.append(line)
        
        return '\n'.join(valid_lines)

    def _generate_lammps_parameters(self, 
                                data_file_info: Dict[str, Any],
                                parameter_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LAMMPS parameter file content based on acquired parameters.
        
        Args:
            data_file_info: Information from parsing the data file
            parameter_info: Parameter information from acquire_parameters()
            
        Returns:
            Dictionary with parameter file content
        """
        self.logger.info("Generating LAMMPS parameter file content")
        
        # Use LLM to generate scientifically accurate LAMMPS parameters
        force_field = parameter_info.get("force_field", "Unknown")
        parameters = parameter_info.get("atom_types", {})
        validation = parameter_info.get("validation", {})
        
        # Format parameter info for the prompt
        param_str = json.dumps(parameter_info, indent=2)
        data_file_str = json.dumps(data_file_info, indent=2)
        
        # Create prompt for LLM
        prompt = f"""
        As an expert in molecular dynamics force fields, generate LAMMPS parameter files for this system.
        
        FORCE FIELD: {force_field}
        
        DATA FILE INFORMATION:
        {data_file_str}
        
        PARAMETER INFORMATION:
        {param_str}
        
        IMPORTANT: Do NOT include 'units' commands in the parameter file, as these will be set in the main input script.
        
        Based on this information, generate a complete and scientifically accurate LAMMPS parameter file.
        The file should include all parameters needed for a LAMMPS simulation with this force field.
        
        IMPORTANT GUIDELINES:
        1. The file must ONLY contain force field parameters - NO simulation setup commands (e.g., units command).
        2. DO include comments explaining the parameters
        3. Focus solely on defining force field parameters, not running a simulation
       
        4. PAIR STYLE GUIDELINES:
           - For TIP4P water: Use 'hybrid/overlay' pair style as shown in the TIP4P information section
           - For TIP3P/SPC/E: Use standard 'lj/cut/coul/long' pair style
           
        5. PAIR COEFFICIENTS GUIDELINES:
           - For standard pair styles: pair_coeff 1 1 0.16275 3.16435  # O-O
           - For hybrid pair styles (TIP4P): pair_coeff 1 1 lj/cut/coul/long 0.16275 3.16435  # O-O
           - ALWAYS include the sub-style name in pair_coeff commands when using hybrid pair styles
           
        6. Include proper mass, bond_coeff, and angle_coeff commands.
        7. Include special_bonds settings appropriate for this force field.
        8. Include kspace_style command for long-range electrostatics. 

        Include ONLY these parameter-related commands:
        1. pair_coeff commands for atom interactions
        2. bond_coeff commands for bond parameters
        3. angle_coeff commands for angle parameters
        4. dihedral_coeff commands (if needed)
        5. improper_coeff commands (if needed)
        6. special_bonds settings (allowed, as it defines force field behavior)
        7. pair_style, bond_style, angle_style (allowed, as they define force field styles)
        8. mass commands (allowed, as they define atom masses)
        9. set type commands ONLY if they set force field parameters
        10. kspace_style (allowed, as it defines long-range interaction handling) 

        Format the output as a JSON object with the main parameter file content and any additional files needed:
        {{
            "main": "# LAMMPS parameters for {force_field}\\n\\npair_style lj/cut/coul/long 10.0\\n...",
            "additional": {{
                "water_params": "# Water model parameters\\n\\n...",
                "other_file": "# Other parameters\\n\\n..."
            }}
        }}
        
        Make sure the LAMMPS syntax is correct and all parameters are scientifically accurate for the {force_field} force field.
        Include only the JSON response with no additional text.
        """
        
        try:
            param_files = self._generate_json(prompt)
            
            # Ensure main content exists
            if "main" not in param_files:
                param_files["main"] = self._generate_fallback_parameters(data_file_info, parameter_info)
                
            # Ensure additional exists
            if "additional" not in param_files:
                param_files["additional"] = {}
                
            return param_files
            
        except Exception as e:
            self.logger.error(f"Error generating LAMMPS parameters: {e}")
            # Generate fallback parameters
            return {
                "main": self._generate_fallback_parameters(data_file_info, parameter_info),
                "additional": {}
            }
    
    def _generate_fallback_parameters(self, 
                                   data_file_info: Dict[str, Any],
                                   parameter_info: Dict[str, Any]) -> str:
        """
        Generate fallback parameters if LLM generation fails.
        
        Args:
            data_file_info: Information from parsing the data file
            parameter_info: Parameter information from acquire_parameters()
            
        Returns:
            Parameter file content as string
        """
        force_field = parameter_info.get("force_field", "Unknown")
        
        # Build basic parameter file content
        lines = [
            f"# LAMMPS parameters for {force_field}",
            "# Generated by ForceFieldAgent (fallback generator)",
            "",
            "# General force field settings",
            "pair_style lj/cut/coul/long 10.0",
            "bond_style harmonic",
            "angle_style harmonic",
            ""
        ]
        
        # Add dihedral style if needed
        if data_file_info.get("dihedral_types", 0) > 0:
            lines.append("dihedral_style harmonic")
            
        # Add improper style if needed
        if data_file_info.get("improper_types", 0) > 0:
            lines.append("improper_style harmonic")
            
        lines.append("")
        
        # Add pair coefficients
        atom_types = data_file_info.get("atom_types", 0)
        if atom_types > 0:
            lines.append("# Pair coefficients")
            for i in range(1, atom_types + 1):
                mass = data_file_info.get("masses", {}).get(i, 12.0)  # Default to carbon mass
                element = self._guess_element_from_mass(mass)
                
                # Use reasonable defaults based on element
                if element == "O":
                    epsilon, sigma = 0.1553, 3.166
                elif element == "H":
                    epsilon, sigma = 0.0, 0.0
                elif element == "C":
                    epsilon, sigma = 0.1094, 3.4
                elif element == "N":
                    epsilon, sigma = 0.1700, 3.25
                elif element in ["Na", "K"]:
                    epsilon, sigma = 0.1, 2.8
                elif element in ["Cl", "Br"]:
                    epsilon, sigma = 0.1, 4.4
                else:
                    epsilon, sigma = 0.1, 3.0
                    
                lines.append(f"pair_coeff {i} {i} {epsilon} {sigma}  # {element}")
                
            lines.append("")
            
        # Add bond coefficients
        bond_types = data_file_info.get("bond_types", 0)
        if bond_types > 0:
            lines.append("# Bond coefficients")
            for i in range(1, bond_types + 1):
                lines.append(f"bond_coeff {i} 450.0 1.0  # Generic bond")
            lines.append("")
            
        # Add angle coefficients
        angle_types = data_file_info.get("angle_types", 0)
        if angle_types > 0:
            lines.append("# Angle coefficients")
            for i in range(1, angle_types + 1):
                lines.append(f"angle_coeff {i} 55.0 109.47  # Generic angle")
            lines.append("")
            
        # Add special bonds
        lines.append("# Special bonds settings")
        if "AMBER" in force_field or "CHARMM" in force_field:
            lines.append("special_bonds lj/coul 0.0 0.0 0.5")
        else:
            lines.append("special_bonds lj/coul 0.0 0.0 0.5  # Generic setting")
            
        lines.append("")
        lines.append("# Long-range electrostatics")
        lines.append("kspace_style pppm 1.0e-5")
        
        return "\n".join(lines)
    
    def _generate_parameter_summary(self, 
                                 params: Dict[str, Any], 
                                 selection_info: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the parameters.
        
        Args:
            params: Parameter information
            selection_info: Force field selection information
            
        Returns:
            Markdown summary text
        """
        force_field = selection_info.get("force_field", {}).get("force_field", "Unknown")
        water_model = selection_info.get("force_field", {}).get("compatible_water_model", "Unknown")
        justification = selection_info.get("force_field", {}).get("justification", "")
        
        method = selection_info.get("parameter_method", {}).get("method", "unknown")
        method_desc = {
            "database": "Direct extraction from established databases",
            "analogy": "Parameters by chemical analogy to known molecules",
            "quantum": "Ab initio parameterization from quantum calculations"
        }.get(method, "Unknown method")
        
        # Get atom type counts
        atom_types_count = len(params.get("atom_types", {}))
        
        # Get validation info
        validation = params.get("validation", {})
        quality_score = validation.get("quality_metrics", {}).get("overall_score", 0)
        passed = validation.get("passed", False)
        warnings = validation.get("warnings", [])
        
        # Create the summary text
        lines = [
            f"# Force Field Parameters Summary",
            "",
            f"## Selection Details",
            f"- **Force Field**: {force_field}",
            f"- **Water Model**: {water_model}",
            f"- **Parameter Source**: {method} ({method_desc})",
            f"- **Quality Score**: {quality_score}/100",
            "",
            f"## Justification",
            f"{justification}",
            "",
            f"## Parameter Statistics",
            f"- **Atom Types**: {atom_types_count}",
            f"- **Bond Types**: {len(params.get('bonds', {}))}",
            f"- **Angle Types**: {len(params.get('angles', {}))}",
            f"- **Dihedral Types**: {len(params.get('dihedrals', {}))}",
            "",
        ]
        
        # Add validation section
        lines.extend([
            f"## Validation",
            f"- **Passed**: {'Yes' if passed else 'No'}",
            ""
        ])
        
        if warnings:
            lines.append("### Warnings")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
            
        return "\n".join(lines)

    def _fix_lammps_syntax(self, lammps_text: str) -> str:
        """
        Fix common LAMMPS syntax issues in generated parameters.
        
        Args:
            lammps_text: LAMMPS parameter text
            
        Returns:
            Corrected LAMMPS parameter text
        """
        lines = lammps_text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix missing spaces after commas
            line = re.sub(r',(?=\S)', ', ', line)
            
            # Fix incorrect comment syntax
            line = re.sub(r'(//.+)$', r'#\1', line)
            
            # Fix invalid syntax in pair_coeff commands
            if line.strip().startswith("pair_coeff") and re.search(r'pair_coeff\s+\*\s+\*\s+[\d\.]+\s*$', line):
                line = line + " # Missing sigma parameter"
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)

    def assign_charges_to_data_file(self,
                                     data_file: str,
                                     parameter_info: Optional[Dict[str, Any]] = None,
                                     pdb_file: Optional[str] = None,
                                     research_goal: Optional[str] = None,
                                     output_file: Optional[str] = None) -> str:
        """
        Assign partial charges to a LAMMPS data file based on force field parameters.
        
        This method analyzes the data file structure, determines appropriate charges
        using the LLM based on atom types and force field, and writes the charges
        back to the data file.
        
        Args:
            data_file: Path to LAMMPS data file (with 0.0 charges)
            parameter_info: Optional pre-computed parameter info from acquire_parameters()
            pdb_file: Optional PDB file for additional context
            research_goal: Optional research goal for context
            output_file: Output path (if None, creates _charged.data version)
            
        Returns:
            Path to the data file with charges assigned
        """
        self.logger.info(f"Assigning charges to {data_file}")
        
        if output_file is None:
            base, ext = os.path.splitext(data_file)
            output_file = f"{base}_charged{ext}"
        
        # Step 1: Parse the data file to understand its structure
        data_info = self._parse_data_file_for_charges(data_file)
        
        # Generate charge assignments
        if parameter_info and parameter_info.get("atom_types"):
            # Use FF parameter mapping approach
            charge_assignments = self._generate_charge_assignments(
                data_info=data_info,
                parameter_info=parameter_info,  # Pass this!
                pdb_file=pdb_file,
                research_goal=research_goal
            )
        else:
            # Fallback to direct LLM generation
            charge_assignments = self._generate_charges_with_llm(
                data_info=data_info,
                pdb_file=pdb_file,
                research_goal=research_goal
            )

        # Step 3: Validate charge assignments
        validation = self._validate_charge_assignments(charge_assignments, data_info)
        if not validation["valid"]:
            self.logger.warning(f"Charge validation warnings: {validation['warnings']}")
        
        # Step 4: Write the data file with charges
        self._write_data_file_with_charges(
            input_file=data_file,
            output_file=output_file,
            charge_assignments=charge_assignments,
            data_info=data_info
        )
        
        # Step 5: Save charge assignment info
        charge_info_file = os.path.join(self.working_dir, "charge_assignments.json")
        with open(charge_info_file, 'w') as f:
            json.dump({
                "charge_assignments": charge_assignments,
                "validation": validation,
                "input_file": data_file,
                "output_file": output_file
            }, f, indent=2)
        
        self.logger.info(f"Charges assigned successfully: {output_file}")
        return output_file
    
    
    def _parse_data_file_for_charges(self, data_file: str) -> Dict[str, Any]:
        """
        Parse a LAMMPS data file to extract detailed information needed for charge assignment.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Dictionary with detailed data file information
        """
        info = {
            "n_atoms": 0,
            "n_atom_types": 0,
            "masses": {},           # type_id -> mass
            "type_names": {},       # type_id -> guessed name from comment or mass
            "type_elements": {},    # type_id -> element symbol
            "type_counts": {},      # type_id -> count of atoms
            "molecules": {},        # mol_id -> list of (atom_id, type_id)
            "atom_style": "full",   # Detected atom style
            "box": None,
            "header_lines": [],
            "atoms_section_start": 0,
            "atoms_section_end": 0,
        }
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # First pass: parse header to get counts
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Parse atom/type counts from header
            parts = stripped.split()
            if len(parts) >= 2:
                if parts[1] == "atoms":
                    info["n_atoms"] = int(parts[0])
                elif parts[1] == "atom" and len(parts) >= 3 and parts[2] == "types":
                    info["n_atom_types"] = int(parts[0])
                elif "atom types" in stripped:
                    info["n_atom_types"] = int(parts[0])
            
            # Parse box dimensions
            if "xlo xhi" in stripped:
                info["box"] = {"xlo": float(parts[0]), "xhi": float(parts[1])}
        
        self.logger.info(f"Data file header: {info['n_atoms']} atoms, {info['n_atom_types']} atom types")
        
        # Second pass: find section locations
        section_lines = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped in ["Masses", "Atoms", "Velocities", "Bonds", "Angles", 
                           "Dihedrals", "Impropers", "Pair Coeffs", "Bond Coeffs",
                           "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"]:
                section_lines[stripped] = i
            # Handle "Atoms # full" format
            elif stripped.startswith("Atoms"):
                section_lines["Atoms"] = i
                if "#" in stripped:
                    style = stripped.split("#")[1].strip()
                    info["atom_style"] = style
        
        # Parse Masses section
        if "Masses" in section_lines:
            start = section_lines["Masses"] + 1
            for i in range(start, len(lines)):
                line = lines[i].strip()
                
                # Stop at empty line or next section
                if not line:
                    continue
                if line in section_lines or line.startswith("Atoms") or line.startswith("Pair") or line.startswith("Bond"):
                    break
                if line.startswith("#"):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        type_id = int(parts[0])
                        mass = float(parts[1])
                        
                        # Sanity check: type_id should be <= n_atom_types
                        if type_id > info["n_atom_types"]:
                            self.logger.warning(f"Skipping invalid type_id {type_id} > {info['n_atom_types']}")
                            continue
                        
                        info["masses"][type_id] = mass
                        
                        # Extract element name from comment if present
                        if "#" in line:
                            comment = line.split("#")[1].strip()
                            type_name = comment.split()[0] if comment.split() else None
                            if type_name:
                                info["type_names"][type_id] = type_name
                                info["type_elements"][type_id] = type_name  # In this case, comment IS the element
                        else:
                            # Guess element from mass
                            element = self._guess_element_from_mass(mass)
                            info["type_elements"][type_id] = element
                            info["type_names"][type_id] = element
                        
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Could not parse mass line: {line} - {e}")
                        continue
        
        self.logger.info(f"Parsed {len(info['masses'])} masses: {info['masses']}")
        self.logger.info(f"Element mapping: {info['type_elements']}")
        
        # Parse Atoms section
        if "Atoms" in section_lines:
            start = section_lines["Atoms"] + 1
            
            # Skip blank/comment lines after "Atoms" header
            while start < len(lines) and (not lines[start].strip() or lines[start].strip().startswith("#")):
                start += 1
            
            info["atoms_section_start"] = start
            
            # Initialize type counts
            for type_id in info["masses"].keys():
                info["type_counts"][type_id] = 0
            
            for i in range(start, len(lines)):
                line = lines[i].strip()
                
                # Stop at empty line (end of section) or next section header
                if not line:
                    # Could be end of section or just blank line - check next few lines
                    blank_count = 0
                    for j in range(i, min(i + 3, len(lines))):
                        if not lines[j].strip():
                            blank_count += 1
                        elif lines[j].strip() in section_lines or lines[j].strip().startswith(("Velocities", "Bonds", "Angles")):
                            info["atoms_section_end"] = i
                            break
                    if blank_count >= 2:
                        info["atoms_section_end"] = i
                        break
                    continue
                
                if line.startswith("#"):
                    continue
                
                # Check if we've hit another section
                if line in section_lines or line.startswith(("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers")):
                    info["atoms_section_end"] = i
                    break
                
                # Parse atom line: atom_id mol_id atom_type charge x y z [flags]
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = int(parts[0])
                        mol_id = int(parts[1])
                        atom_type = int(parts[2])
                        # parts[3] is charge
                        # parts[4:7] are x, y, z
                        
                        # CRITICAL VALIDATION: atom_type must be in valid range!
                        if atom_type < 1 or atom_type > info["n_atom_types"]:
                            self.logger.warning(f"Invalid atom type {atom_type} for atom {atom_id} "
                                              f"(valid range: 1-{info['n_atom_types']})")
                            continue
                        
                        # Count this atom type
                        info["type_counts"][atom_type] = info["type_counts"].get(atom_type, 0) + 1
                        
                        # Track molecules
                        if mol_id not in info["molecules"]:
                            info["molecules"][mol_id] = []
                        info["molecules"][mol_id].append((atom_id, atom_type))
                        
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Could not parse atom line {i}: {line} - {e}")
                        continue
            else:
                info["atoms_section_end"] = len(lines)
        
        # Final validation
        total_counted = sum(info["type_counts"].values())
        if total_counted != info["n_atoms"]:
            self.logger.warning(f"Atom count mismatch: header says {info['n_atoms']}, counted {total_counted}")
        
        self.logger.info(f"Atom type counts: {info['type_counts']}")
        self.logger.info(f"Number of molecules: {len(info['molecules'])}")
        
        return info
    
    
    def _get_default_charge_for_element(self, element: str) -> float:
        """
        Get a reasonable default charge for an element when LLM fails.
        
        Uses common oxidation states and typical force field values.
        For truly unknown elements, returns 0.0.
        """
        from ase.data import atomic_numbers
        
        # Check if it's a known element
        if element not in atomic_numbers:
            self.logger.warning(f"Unknown element symbol: {element}")
            return 0.0
        
        # Alkali metals: +1
        alkali = ["Li", "Na", "K", "Rb", "Cs"]
        if element in alkali:
            return 1.0
        
        # Alkaline earth metals: +2
        alkaline_earth = ["Be", "Mg", "Ca", "Sr", "Ba"]
        if element in alkaline_earth:
            return 2.0
        
        # Common transition metals (default oxidation states)
        transition_metals = {
            "Zn": 2.0, "Cu": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0,
            "Mn": 2.0, "Cr": 3.0, "Ti": 4.0, "Ag": 1.0, "Au": 1.0,
            "Pt": 2.0, "Pd": 2.0, "Cd": 2.0, "Hg": 2.0
        }
        if element in transition_metals:
            return transition_metals[element]
        
        # Halogens: -1 (as anions) or small negative (in molecules)
        halogens = ["F", "Cl", "Br", "I"]
        if element in halogens:
            return -0.2  # Small negative, conservative for molecular context
        
        # Oxygen: typically negative
        if element == "O":
            return -0.4  # Conservative default
        
        # Hydrogen: typically positive in most contexts
        if element == "H":
            return 0.3  # Conservative default
        
        # Nitrogen: context dependent, default to small negative
        if element == "N":
            return -0.3
        
        # Sulfur: context dependent
        if element == "S":
            return 0.0
        
        # Carbon and other main group elements: neutral default
        return 0.0
    
    
    def _get_default_charges(self, data_info: Dict[str, Any]) -> Dict[int, float]:
        """
        Get default charges for all atom types when LLM fails completely.
        """
        charges = {}
        for type_id, element in data_info["type_elements"].items():
            charges[type_id] = self._get_default_charge_for_element(element)
        
        self.logger.warning(f"Using fallback default charges: {charges}")
        return charges

    def _analyze_molecular_compositions(self, data_info: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Analyze molecules in the system to identify distinct molecule types.
        
        Args:
            data_info: Parsed data file information
            
        Returns:
            Dictionary of molecule types with their compositions and counts
        """
        # Group molecules by their atom type composition
        mol_signatures = {}
        
        for mol_id, atoms in data_info["molecules"].items():
            # Create a signature based on sorted atom types
            type_counts = {}
            for atom_id, atom_type in atoms:
                type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
            
            # Convert to tuple for hashing
            signature = tuple(sorted(type_counts.items()))
            
            if signature not in mol_signatures:
                mol_signatures[signature] = {
                    "count": 0,
                    "type_counts": type_counts,
                    "example_mol_id": mol_id
                }
            mol_signatures[signature]["count"] += 1
        
        # Convert to more readable format
        mol_types = {}
        for i, (signature, data) in enumerate(mol_signatures.items()):
            # Try to identify molecule type
            mol_name = self._identify_molecule_type(data["type_counts"], data_info)
            
            composition_str = ", ".join([
                f"{data_info['type_elements'].get(t, '?')}{c}" 
                for t, c in sorted(data["type_counts"].items())
            ])
            
            mol_types[mol_name] = {
                "count": data["count"],
                "composition": composition_str,
                "type_counts": data["type_counts"]
            }
        
        return mol_types
    
    
    def _identify_molecule_type(self, type_counts: Dict[int, int], data_info: Dict[str, Any]) -> str:
        """
        Try to identify what type of molecule based on its composition.
        
        Uses common molecular patterns as heuristics, falls back to formula-based naming.
        This is for identification/labeling only - does NOT determine charges.
        
        Args:
            type_counts: Dictionary of atom_type -> count
            data_info: Parsed data file information
            
        Returns:
            Molecule type name (string identifier)
        """
        # Get element composition
        element_counts = {}
        for type_id, count in type_counts.items():
            element = data_info["type_elements"].get(type_id, "X")
            element_counts[element] = element_counts.get(element, 0) + count
        
        n_atoms = sum(element_counts.values())
        
        # === Common molecule patterns (heuristics for naming, not charges) ===
        
        # Monatomic species (ions)
        if n_atoms == 1:
            element = list(element_counts.keys())[0]
            return f"{element.lower()}_ion"
        
        # Water: H2O (3 atoms, 2H + 1O)
        if (n_atoms == 3 and 
            element_counts.get("H", 0) == 2 and 
            element_counts.get("O", 0) == 1):
            return "water"
        
        # Hydroxide: OH (2 atoms)
        if (n_atoms == 2 and 
            element_counts.get("H", 0) == 1 and 
            element_counts.get("O", 0) == 1):
            return "hydroxide"
        
        # Hydronium: H3O (4 atoms)
        if (n_atoms == 4 and 
            element_counts.get("H", 0) == 3 and 
            element_counts.get("O", 0) == 1):
            return "hydronium"
        
        # Ammonia: NH3
        if (n_atoms == 4 and 
            element_counts.get("N", 0) == 1 and 
            element_counts.get("H", 0) == 3):
            return "ammonia"
        
        # Ammonium: NH4
        if (n_atoms == 5 and 
            element_counts.get("N", 0) == 1 and 
            element_counts.get("H", 0) == 4):
            return "ammonium"
        
        # Methane: CH4
        if (n_atoms == 5 and 
            element_counts.get("C", 0) == 1 and 
            element_counts.get("H", 0) == 4):
            return "methane"
        
        # Methanol: CH4O (CH3OH)
        if (n_atoms == 6 and 
            element_counts.get("C", 0) == 1 and 
            element_counts.get("H", 0) == 4 and
            element_counts.get("O", 0) == 1):
            return "methanol"
        
        # Ethanol: C2H6O
        if (n_atoms == 9 and 
            element_counts.get("C", 0) == 2 and 
            element_counts.get("H", 0) == 6 and
            element_counts.get("O", 0) == 1):
            return "ethanol"
        
        # Carbon dioxide: CO2
        if (n_atoms == 3 and 
            element_counts.get("C", 0) == 1 and 
            element_counts.get("O", 0) == 2):
            return "carbon_dioxide"
        
        # Carbonate: CO3
        if (n_atoms == 4 and 
            element_counts.get("C", 0) == 1 and 
            element_counts.get("O", 0) == 3):
            return "carbonate"
        
        # Bicarbonate: HCO3
        if (n_atoms == 5 and 
            element_counts.get("H", 0) == 1 and
            element_counts.get("C", 0) == 1 and 
            element_counts.get("O", 0) == 3):
            return "bicarbonate"
        
        # Nitrate: NO3
        if (n_atoms == 4 and 
            element_counts.get("N", 0) == 1 and 
            element_counts.get("O", 0) == 3):
            return "nitrate"
        
        # Sulfate: SO4
        if (n_atoms == 5 and 
            element_counts.get("S", 0) == 1 and 
            element_counts.get("O", 0) == 4 and
            element_counts.get("C", 0) == 0):
            return "sulfate"
        
        # Bisulfate: HSO4
        if (n_atoms == 6 and 
            element_counts.get("H", 0) == 1 and
            element_counts.get("S", 0) == 1 and 
            element_counts.get("O", 0) == 4):
            return "bisulfate"
        
        # Phosphate: PO4
        if (n_atoms == 5 and 
            element_counts.get("P", 0) == 1 and 
            element_counts.get("O", 0) == 4):
            return "phosphate"
        
        # Perchlorate: ClO4
        if (n_atoms == 5 and 
            element_counts.get("Cl", 0) == 1 and 
            element_counts.get("O", 0) == 4):
            return "perchlorate"
        
        # === Pattern-based identification for larger molecules ===
        
        # Fluorosulfonate-type (contains C, F, S, O) - e.g., triflate, mesylate variants
        if (element_counts.get("S", 0) >= 1 and 
            element_counts.get("O", 0) >= 3 and
            element_counts.get("F", 0) >= 1):
            return "fluorosulfonate"
        
        # Sulfonate-type (contains S and O, possibly C, no F)
        if (element_counts.get("S", 0) >= 1 and 
            element_counts.get("O", 0) >= 3 and
            element_counts.get("F", 0) == 0 and
            element_counts.get("C", 0) >= 1):
            return "sulfonate"
        
        # Carboxylate-type (COO group)
        if (element_counts.get("C", 0) >= 1 and 
            element_counts.get("O", 0) >= 2 and
            element_counts.get("S", 0) == 0 and
            element_counts.get("N", 0) == 0 and
            element_counts.get("P", 0) == 0):
            # Check if it's likely a carboxylic acid/carboxylate
            c_count = element_counts.get("C", 0)
            o_count = element_counts.get("O", 0)
            if o_count == 2 or (o_count >= 2 and o_count <= c_count + 1):
                return "carboxylate"
        
        # Amine-type (contains N and H, possibly C)
        if (element_counts.get("N", 0) >= 1 and 
            element_counts.get("H", 0) >= 1 and
            element_counts.get("O", 0) == 0):
            return "amine"
        
        # Imidazolium-type (ring with N)
        if (element_counts.get("N", 0) >= 2 and 
            element_counts.get("C", 0) >= 3 and
            element_counts.get("H", 0) >= 4):
            return "imidazolium"
        
        # === Generic fallback: formula-based name ===
        
        # Build formula string with standard ordering: C, H, then alphabetical
        formula_parts = []
        
        # Carbon first (if present)
        if "C" in element_counts:
            cnt = element_counts["C"]
            formula_parts.append(f"C{cnt}" if cnt > 1 else "C")
        
        # Hydrogen second (if present)
        if "H" in element_counts:
            cnt = element_counts["H"]
            formula_parts.append(f"H{cnt}" if cnt > 1 else "H")
        
        # Remaining elements alphabetically
        for el in sorted(element_counts.keys()):
            if el not in ["C", "H"]:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        
        formula = "".join(formula_parts)
        return f"molecule_{formula}"    

    def _map_data_types_to_ff_types(self,
                                     data_info: Dict[str, Any],
                                     parameter_info: Dict[str, Any],
                                     research_goal: Optional[str] = None) -> Dict[int, str]:
        """
        Use LLM to map data file atom type IDs to force field atom type names.
        """
        self.logger.info("Mapping data file types to force field atom types via LLM")
        
        type_to_element = data_info["type_elements"]
        type_to_name = data_info.get("type_names", {})
        
        # Get molecular context for each type
        mol_compositions = self._analyze_molecular_compositions(data_info)
        type_to_context = self._build_type_to_context_map(mol_compositions)
        
        # Build data file type descriptions
        data_type_lines = []
        for type_id in sorted(data_info["masses"].keys()):
            element = type_to_element.get(type_id, "?")
            mass = data_info["masses"][type_id]
            count = data_info["type_counts"].get(type_id, 0)
            contexts = type_to_context.get(type_id, ["unknown"])
            name = type_to_name.get(type_id, element)
            
            data_type_lines.append(
                f"  DATA_TYPE_{type_id}: element={element}, mass={mass:.3f}, "
                f"count={count}, appears_in={contexts}, label='{name}'"
            )
        data_types_str = "\n".join(data_type_lines)
        
        # Build force field atom type descriptions with clear names
        ff_atom_types = parameter_info.get("atom_types", {})
        ff_type_lines = []
        ff_type_names = list(ff_atom_types.keys())  # Get actual type names
        
        for ff_name in ff_type_names:
            params = ff_atom_types[ff_name]
            mass = params.get("mass", 0)
            charge = params.get("charge", 0)
            desc = params.get("description", "")
            ff_type_lines.append(
                f"  FF_TYPE '{ff_name}': mass={mass:.3f}, charge={charge:+.4f}, description='{desc}'"
            )
        ff_types_str = "\n".join(ff_type_lines)
        
        # Build molecule descriptions
        mol_desc_lines = []
        for mol_name, mol_data in mol_compositions.items():
            element_counts = {}
            for type_id, count in mol_data["type_counts"].items():
                element = type_to_element.get(type_id, "X")
                element_counts[element] = element_counts.get(element, 0) + count
            
            formula = self._build_molecular_formula(element_counts)
            type_list = [f"DATA_TYPE_{t}" for t in sorted(mol_data["type_counts"].keys())]
            
            mol_desc_lines.append(
                f"  {mol_name}: {mol_data['count']} molecules, formula={formula}, "
                f"uses: [{', '.join(type_list)}]"
            )
        mol_desc_str = "\n".join(mol_desc_lines)
        
        type_ids = sorted(data_info["masses"].keys())
        
        # Make the list of available FF type names very clear
        ff_names_list = ", ".join([f"'{name}'" for name in ff_type_names])
        
        prompt = f"""
    You are mapping atom types from a LAMMPS data file to force field parameter names.
    
    DATA FILE ATOM TYPES (these need to be mapped):
    {data_types_str}
    
    AVAILABLE FORCE FIELD TYPES (use these EXACT names in your response):
    {ff_types_str}
    
    MOLECULAR CONTEXT:
    {mol_desc_str}
    
    YOUR TASK:
    For each DATA_TYPE, select the best matching FF_TYPE based on:
    1. Element must match
    2. Molecular context (water O ≠ sulfonate O)
    3. Description should match the chemical environment
    
    AVAILABLE FF_TYPE NAMES: {ff_names_list}
    
    IMPORTANT: 
    - Return the FF_TYPE NAME (like 'OW', 'HW', 'CT', etc.), NOT a number
    - Each data type maps to exactly one FF type name
    
    Return a JSON object mapping data file type ID to FF type NAME:
    {{
        "1": "<ff_type_name>",
        "2": "<ff_type_name>",
        ...
    }}
    
    Example response format:
    {{
        "1": "CT",
        "2": "F",
        "3": "HW",
        "4": "OW",
        "5": "S",
        "6": "Zn2+",
        "7": "OS"
    }}
    
    Data file type IDs to map: {', '.join(str(t) for t in type_ids)}
    """
    
        try:
            response = self._generate_json(prompt)
            self.logger.info(f"LLM type mapping response: {response}")
            
            # Validate and build mapping
            type_mapping = {}
            valid_ff_types = set(ff_atom_types.keys())
            
            for key, ff_type in response.items():
                try:
                    type_id = int(key)
                    
                    if type_id not in data_info["masses"]:
                        self.logger.warning(f"Ignoring mapping for non-existent type {type_id}")
                        continue
                    
                    # Handle case where LLM returned a number instead of name
                    if ff_type.isdigit() or (isinstance(ff_type, str) and ff_type.lstrip('-').isdigit()):
                        self.logger.warning(
                            f"LLM returned number '{ff_type}' instead of FF type name for type {type_id}"
                        )
                        # Try to use as index into ff_type_names
                        try:
                            idx = int(ff_type) - 1  # Assuming 1-based
                            if 0 <= idx < len(ff_type_names):
                                ff_type = ff_type_names[idx]
                                self.logger.info(f"  Converted to: {ff_type}")
                            else:
                                element = type_to_element.get(type_id, "?")
                                ff_type = self._find_closest_ff_type(element, ff_atom_types)
                        except (ValueError, IndexError):
                            element = type_to_element.get(type_id, "?")
                            ff_type = self._find_closest_ff_type(element, ff_atom_types)
                    
                    # Validate FF type exists
                    if ff_type not in valid_ff_types:
                        self.logger.warning(
                            f"FF type '{ff_type}' not in parameters for type {type_id}, "
                            f"finding closest match"
                        )
                        element = type_to_element.get(type_id, "?")
                        ff_type = self._find_closest_ff_type(element, ff_atom_types)
                    
                    type_mapping[type_id] = ff_type
                    self.logger.info(
                        f"Mapped: Type {type_id} ({type_to_element.get(type_id, '?')}) → '{ff_type}'"
                    )
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not parse mapping for {key}: {e}")
            
            # Fill in any missing mappings
            missing = set(data_info["masses"].keys()) - set(type_mapping.keys())
            if missing:
                self.logger.warning(f"Missing mappings for types: {missing}")
                for type_id in missing:
                    element = type_to_element.get(type_id, "?")
                    ff_type = self._find_closest_ff_type(element, ff_atom_types)
                    type_mapping[type_id] = ff_type
                    self.logger.info(f"Fallback mapping: Type {type_id} ({element}) → '{ff_type}'")
            
            return type_mapping
            
        except Exception as e:
            self.logger.error(f"Error in LLM type mapping: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return self._fallback_type_mapping(data_info, parameter_info)

    def _find_closest_ff_type(self, 
                              element: str, 
                              ff_atom_types: Dict[str, Any],
                              context: Optional[str] = None) -> str:
        """
        Find closest matching FF type for an element, optionally considering context.
        
        Args:
            element: Element symbol (e.g., 'O', 'H', 'C')
            ff_atom_types: Dictionary of FF atom type parameters
            context: Optional molecular context (e.g., 'water', 'fluorosulfonate')
            
        Returns:
            FF type name
        """
        candidates = []
        
        for ff_name, params in ff_atom_types.items():
            desc = params.get("description", "").lower()
            mass = params.get("mass", 0)
            
            # Check if element matches (by mass or name)
            element_lower = element.lower()
            ff_name_lower = ff_name.lower()
            
            # Element in name check
            element_match = (
                element_lower in ff_name_lower or 
                element_lower in desc or
                ff_name_lower.startswith(element_lower)
            )
            
            # Mass check
            try:
                from ase.data import atomic_masses, atomic_numbers
                target_mass = atomic_masses[atomic_numbers[element]]
                mass_match = abs(mass - target_mass) < 1.0
            except:
                mass_match = False
            
            if element_match or mass_match:
                # Score by context match
                score = 0
                if context:
                    context_lower = context.lower()
                    if "water" in context_lower:
                        if "water" in desc or ff_name_lower in ["ow", "hw", "o_w", "h_w", "oh2"]:
                            score += 10
                    elif "sulfon" in context_lower or "triflate" in context_lower:
                        if "sulfon" in desc or "sulfate" in desc or ff_name_lower in ["os", "ot", "st", "s"]:
                            score += 10
                        if "water" in desc:
                            score -= 5  # Penalize water types for sulfonate context
                
                candidates.append((ff_name, score))
        
        if candidates:
            # Sort by score (highest first), then by name
            candidates.sort(key=lambda x: (-x[1], x[0]))
            return candidates[0][0]
        
        # Last resort: return first available type
        if ff_atom_types:
            return list(ff_atom_types.keys())[0]
        return element   
    
    def _fallback_type_mapping(self,
                               data_info: Dict[str, Any],
                               parameter_info: Dict[str, Any]) -> Dict[int, str]:
        """Simple fallback type mapping based on element and mass."""
        self.logger.warning("Using fallback type mapping (element-based)")
    
        type_to_element = data_info["type_elements"]
        ff_atom_types = parameter_info.get("atom_types", {})
    
        # Group FF types by element (inferred from mass or name)
        from ase.data import atomic_masses, atomic_numbers, chemical_symbols
    
        ff_by_element = {}
        for ff_name, params in ff_atom_types.items():
            mass = params.get("mass", 0)
            # Guess element from mass
            element = None
            for i, sym in enumerate(chemical_symbols[1:], 1):
                if abs(atomic_masses[i] - mass) < 0.5:
                    element = sym
                    break
            if element:
                if element not in ff_by_element:
                    ff_by_element[element] = []
                ff_by_element[element].append(ff_name)
    
        # Map each data type to first matching FF type
        type_mapping = {}
        for type_id, element in type_to_element.items():
            if element in ff_by_element and ff_by_element[element]:
                type_mapping[type_id] = ff_by_element[element][0]
            else:
                type_mapping[type_id] = element  # Use element name as fallback
    
        return type_mapping

    def _generate_charges_with_llm(self,
                                      data_info: Dict[str, Any],
                                      parameter_info: Optional[Dict[str, Any]] = None,
                                      pdb_file: Optional[str] = None,
                                      research_goal: Optional[str] = None) -> Dict[int, float]:
        """
        Generate charge assignments by mapping data file types to force field types.
        
        If parameter_info is provided with atom_types, uses LLM to map data file
        types to FF types and extracts charges from the FF parameters.
        
        Otherwise, falls back to direct LLM charge generation.
        """
        self.logger.info("Generating charge assignments")
        
        # If we have force field parameters, use type mapping approach
        ff_atom_types = parameter_info.get("atom_types", {}) if parameter_info else {}
        
        if ff_atom_types:
            self.logger.info("Using force field parameter mapping approach")
            
            # Step 1: Map data file types to FF types
            type_mapping = self._map_data_types_to_ff_types(
                data_info=data_info,
                parameter_info=parameter_info,
                research_goal=research_goal
            )
            
            self.logger.info(f"Type mapping: {type_mapping}")
            
            # Step 2: Extract charges from FF parameters using the mapping
            charge_assignments = {}
            type_to_element = data_info["type_elements"]
            
            for type_id, ff_type in type_mapping.items():
                if ff_type in ff_atom_types:
                    charge = ff_atom_types[ff_type].get("charge", 0.0)
                    charge_assignments[type_id] = charge
                    self.logger.info(
                        f"Type {type_id} ({type_to_element.get(type_id, '?')}) → "
                        f"{ff_type} → charge {charge:+.4f}"
                    )
                else:
                    self.logger.warning(
                        f"FF type '{ff_type}' not found in parameters for type {type_id}"
                    )
                    charge_assignments[type_id] = 0.0
            
            # Step 3: Validate molecule charges
            mol_compositions = self._analyze_molecular_compositions(data_info)
            self._log_molecular_charge_validation(
                charge_assignments, mol_compositions, type_to_element
            )
            
            # Step 4: Check if any molecules have incorrect total charges
            # If so, the FF parameters themselves may be inconsistent
            validation_issues = self._check_molecule_charge_totals(
                charge_assignments, mol_compositions, type_to_element
            )
            
            if validation_issues:
                self.logger.warning(
                    f"Molecule charge validation issues detected: {validation_issues}"
                )
                # Optionally: fall back to LLM-based charge generation
                # Or: try to fix by adjusting charges
            
            return charge_assignments
        
        else:
            # Fall back to direct LLM charge generation (original approach)
            self.logger.info("No FF parameters available, using direct LLM charge generation")
            return self._generate_charges_with_llm(
                data_info=data_info,
                pdb_file=pdb_file,
                research_goal=research_goal
            )

    def _build_type_to_context_map(self, mol_compositions: Dict[str, Any]) -> Dict[int, List[str]]:
        """Build mapping of type_id -> list of molecule names it appears in."""
        type_to_context = {}
        for mol_name, mol_data in mol_compositions.items():
            for type_id in mol_data["type_counts"].keys():
                if type_id not in type_to_context:
                    type_to_context[type_id] = []
                if mol_name not in type_to_context[type_id]:
                    type_to_context[type_id].append(mol_name)
        return type_to_context
    
    
    def _build_molecular_formula(self, element_counts: Dict[str, int]) -> str:
        """Build molecular formula string from element counts."""
        formula_parts = []
        for el in ["C", "H"]:
            if el in element_counts:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        for el in sorted(element_counts.keys()):
            if el not in ["C", "H"]:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        return "".join(formula_parts)
    
    
    def _validate_and_fix_molecule_charges(self,
                                            charge_assignments: Dict[int, float],
                                            mol_compositions: Dict[str, Any],
                                            expected_mol_charges: Dict[str, float],
                                            type_to_element: Dict[int, str],
                                            type_to_context: Dict[int, List[str]]) -> Dict[int, float]:
        """Validate and fix molecule charges if they don't sum correctly."""
        self.logger.info("Validating and fixing molecule charges...")
        
        for mol_name, mol_data in mol_compositions.items():
            expected = expected_mol_charges.get(mol_name, 0.0)
            
            current = sum(
                charge_assignments.get(t, 0.0) * c
                for t, c in mol_data["type_counts"].items()
            )
            
            error = current - expected
            
            if abs(error) < 0.01:
                self.logger.info(f"  {mol_name}: {current:.4f} ✓")
                continue
            
            self.logger.warning(f"  {mol_name}: {current:.4f} (expected {expected}), error={error:.4f}")
            
            # Find types exclusive to this molecule for adjustment
            exclusive_types = [
                t for t in mol_data["type_counts"]
                if len(type_to_context.get(t, [])) == 1 and type_to_context.get(t, [None])[0] == mol_name
            ]
            
            if not exclusive_types:
                # Use most abundant type in molecule
                exclusive_types = [max(mol_data["type_counts"].items(), key=lambda x: x[1])[0]]
            
            total_atoms = sum(mol_data["type_counts"].get(t, 0) for t in exclusive_types)
            
            if total_atoms > 0:
                correction = -error / total_atoms
                for type_id in exclusive_types:
                    old = charge_assignments.get(type_id, 0.0)
                    charge_assignments[type_id] = old + correction
                    self.logger.info(f"    Adjusted type {type_id}: {old:.4f} → {old + correction:.4f}")
        
        return charge_assignments

    def _generate_charge_assignments(self,
                                      data_info: Dict[str, Any],
                                      parameter_info: Optional[Dict[str, Any]] = None,
                                      pdb_file: Optional[str] = None,
                                      research_goal: Optional[str] = None) -> Dict[int, float]:
        """Generate charge assignments."""
        self.logger.info("Generating charge assignments")
        
        ff_atom_types = {}
        if parameter_info:
            ff_atom_types = parameter_info.get("atom_types", {})
        
        if ff_atom_types:
            self.logger.info("Using force field parameter mapping approach")
            
            # Check if atom_types is keyed by name or by number
            # If by number, build a name-based lookup
            first_key = next(iter(ff_atom_types.keys()), "")
            if first_key.isdigit():
                # Parameter file uses numeric keys with 'name' field
                # Build name -> params lookup
                ff_by_name = {}
                for key, params in ff_atom_types.items():
                    name = params.get("name", key)
                    ff_by_name[name] = params
                self.logger.info(f"FF types by name: {list(ff_by_name.keys())}")
            else:
                # Parameter file already uses names as keys
                ff_by_name = ff_atom_types
            
            # Map data types to FF type names
            type_mapping = self._map_data_types_to_ff_types(
                data_info=data_info,
                parameter_info={"atom_types": ff_by_name},  # Pass name-keyed version
                research_goal=research_goal
            )
            
            self.logger.info(f"Type mapping: {type_mapping}")
            
            # Extract charges
            charge_assignments = {}
            type_to_element = data_info["type_elements"]
            
            for type_id, ff_type in type_mapping.items():
                if ff_type in ff_by_name:
                    charge = ff_by_name[ff_type].get("charge", 0.0)
                    charge_assignments[type_id] = charge
                    self.logger.info(
                        f"Type {type_id} ({type_to_element.get(type_id, '?')}) → "
                        f"'{ff_type}' → charge {charge:+.4f}"
                    )
                else:
                    self.logger.warning(f"FF type '{ff_type}' not found for type {type_id}")
                    charge_assignments[type_id] = 0.0
            
            # Validate and fix
            mol_compositions = self._analyze_molecular_compositions(data_info)
            validation_issues = self._check_molecule_charge_totals(
                charge_assignments, mol_compositions, type_to_element
            )
            
            if validation_issues:
                self.logger.warning(f"Molecule charge issues: {validation_issues}")
                expected_mol_charges = self._determine_expected_molecule_charges(
                    mol_compositions, data_info
                )
                type_to_context = self._build_type_to_context_map(mol_compositions)
                
                charge_assignments = self._validate_and_fix_molecule_charges(
                    charge_assignments=charge_assignments,
                    mol_compositions=mol_compositions,
                    expected_mol_charges=expected_mol_charges,
                    type_to_element=type_to_element,
                    type_to_context=type_to_context
                )
            
            self._log_molecular_charge_validation(
                charge_assignments, mol_compositions, type_to_element
            )
            
            return charge_assignments
    
        else:
            self.logger.info("No FF parameters, using direct LLM generation")
            return self._generate_charges_direct_llm(
                data_info=data_info,
                pdb_file=pdb_file,
                research_goal=research_goal
            )

    def _check_molecule_charge_totals(self,
                                       charge_assignments: Dict[int, float],
                                       mol_compositions: Dict[str, Any],
                                       type_to_element: Dict[int, str]) -> List[str]:
        """Check if molecule charges sum to reasonable values."""
        issues = []
        
        expected_charges = self._determine_expected_molecule_charges(
            mol_compositions, {"type_elements": type_to_element}
        )
        
        for mol_name, mol_data in mol_compositions.items():
            total_charge = sum(
                charge_assignments.get(type_id, 0.0) * count
                for type_id, count in mol_data["type_counts"].items()
            )
            
            expected = expected_charges.get(mol_name, 0.0)
            
            if abs(total_charge - expected) > 0.05:
                issues.append(
                    f"{mol_name}: got {total_charge:.3f}, expected {expected:.1f}"
                )
        
        return issues
    
    
    def _determine_expected_molecule_charges(self,
                                              mol_compositions: Dict[str, Any],
                                              data_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine expected formal charges for each molecule type.
        Uses chemical heuristics based on molecular formula.
        """
        expected_charges = {}
        type_to_element = data_info.get("type_elements", {})
        
        for mol_name, mol_data in mol_compositions.items():
            element_counts = {}
            for type_id, count in mol_data["type_counts"].items():
                element = type_to_element.get(type_id, "X")
                element_counts[element] = element_counts.get(element, 0) + count
            
            charge = self._infer_formal_charge_from_composition(mol_name, element_counts)
            expected_charges[mol_name] = charge
        
        return expected_charges
    
    
    def _infer_formal_charge_from_composition(self,
                                               mol_name: str,
                                               element_counts: Dict[str, int]) -> float:
        """
        Infer formal charge from molecular composition using chemical heuristics.
        """
        # Water: H2O → 0
        if element_counts == {"H": 2, "O": 1}:
            return 0.0
        
        # Monatomic ions
        if len(element_counts) == 1 and list(element_counts.values())[0] == 1:
            element = list(element_counts.keys())[0]
            monatomic = {
                "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
                "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2,
                "Fe": 2, "Cu": 2, "Ni": 2, "Co": 2,
                "Al": 3, "Fe3": 3,
                "F": -1, "Cl": -1, "Br": -1, "I": -1,
            }
            return float(monatomic.get(element, 0))
        
        # Sulfonate (RSO3): charge -1
        if "S" in element_counts and element_counts.get("O", 0) == 3:
            return -1.0
        
        # Sulfate (SO4): charge -2
        if element_counts.get("S", 0) == 1 and element_counts.get("O", 0) == 4:
            if "C" not in element_counts:
                return -2.0
        
        # Nitrate (NO3): charge -1
        if element_counts == {"N": 1, "O": 3}:
            return -1.0
        
        # Phosphate (PO4): charge -3
        if element_counts == {"P": 1, "O": 4}:
            return -3.0
        
        # Hydroxide (OH): charge -1
        if element_counts == {"O": 1, "H": 1}:
            return -1.0
        
        # Ammonium (NH4): charge +1
        if element_counts == {"N": 1, "H": 4}:
            return 1.0
        
        # Default: neutral
        return 0.0    
    
    def _generate_missing_charges(self,
                                   missing_types: set,
                                   data_info: Dict[str, Any],
                                   type_to_context: Dict[int, List[str]],
                                   existing_charges: Dict[int, float]) -> Dict[int, float]:
        """
        Generate charges for types that were missing from the initial LLM response.
        """
        self.logger.info(f"Generating charges for missing types: {missing_types}")
        
        type_to_element = data_info["type_elements"]
        
        # Build info about missing types
        missing_info = []
        for type_id in sorted(missing_types):
            element = type_to_element.get(type_id, "X")
            mass = data_info["masses"].get(type_id, 0)
            contexts = type_to_context.get(type_id, ["unknown"])
            missing_info.append(
                f"  Type {type_id}: element={element}, mass={mass:.3f}, appears_in=[{', '.join(contexts)}]"
            )
        
        missing_info_str = "\n".join(missing_info)
        missing_ids_str = ", ".join(str(t) for t in sorted(missing_types))
        
        # Show existing charges for context
        existing_str = ", ".join(f"{t}:{c:.4f}" for t, c in sorted(existing_charges.items()))
        
        prompt = f"""
    You are assigning partial charges to atom types in a molecular simulation.
    
    The following types still need charges:
    {missing_info_str}
    
    Already assigned charges (for context): {existing_str}
    
    Assign appropriate partial charges based on:
    1. The element type
    2. The molecular context (appears_in field)
    3. Consistency with already assigned charges for similar elements
    
    Return ONLY a JSON object:
    {{
        {', '.join(f'"{t}": <charge>' for t in sorted(missing_types))}
    }}
    """
    
        try:
            response = self._generate_json(prompt)
            
            charges = {}
            for key, value in response.items():
                try:
                    type_id = int(key)
                    if type_id in missing_types:
                        charge = float(value) if not isinstance(value, dict) else float(value.get("charge", 0.0))
                        charges[type_id] = charge
                except (ValueError, TypeError):
                    continue
            
            # Fill any still-missing with defaults
            still_missing = missing_types - set(charges.keys())
            for type_id in still_missing:
                element = type_to_element.get(type_id, "X")
                charges[type_id] = self._get_default_charge_for_element(element)
                self.logger.warning(f"Using default charge for type {type_id} ({element})")
            
            return charges
            
        except Exception as e:
            self.logger.error(f"Error generating missing charges: {e}")
            # Return defaults for all missing
            charges = {}
            for type_id in missing_types:
                element = type_to_element.get(type_id, "X")
                charges[type_id] = self._get_default_charge_for_element(element)
            return charges
    
    
    def _generate_charges_fallback(self,
                                    data_info: Dict[str, Any],
                                    type_to_context: Dict[int, List[str]]) -> Dict[int, float]:
        """
        Fallback charge generation with a simpler, more direct prompt.
        """
        self.logger.info("Using fallback charge generation")
        
        type_to_element = data_info["type_elements"]
        mol_compositions = self._analyze_molecular_compositions(data_info)
        
        # Build a simple type list
        type_lines = []
        for type_id in sorted(data_info["masses"].keys()):
            element = type_to_element.get(type_id, "X")
            contexts = type_to_context.get(type_id, ["unknown"])
            type_lines.append(f"Type {type_id}: {element} in {contexts}")
        
        type_list_str = "\n".join(type_lines)
        type_ids = sorted(data_info["masses"].keys())
        
        prompt = f"""
    Assign partial charges for these atom types in a molecular dynamics simulation:
    
    {type_list_str}
    
    Rules:
    - Use standard force field values (TIP3P for water, OPLS-AA or similar for organics)
    - Neutral molecules should sum to charge 0
    - Ions should sum to their formal charge
    - Different types of the same element in different contexts need different charges
    
    Return JSON mapping type ID to charge:
    {{
        {', '.join(f'"{t}": <charge>' for t in type_ids)}
    }}
    """
    
        try:
            response = self._generate_json(prompt)
            
            charges = {}
            for key, value in response.items():
                try:
                    type_id = int(key)
                    if type_id in data_info["masses"]:
                        charge = float(value) if not isinstance(value, dict) else float(value.get("charge", 0.0))
                        charges[type_id] = charge
                except (ValueError, TypeError):
                    continue
            
            # Fill missing with defaults
            for type_id in data_info["masses"].keys():
                if type_id not in charges:
                    element = type_to_element.get(type_id, "X")
                    charges[type_id] = self._get_default_charge_for_element(element)
            
            return charges
            
        except Exception as e:
            self.logger.error(f"Fallback charge generation failed: {e}")
            # Ultimate fallback: use element-based defaults
            return self._get_default_charges(data_info)
    
    
    def _log_molecular_charge_validation(self,
                                          charges: Dict[int, float],
                                          mol_compositions: Dict[str, Dict],
                                          type_to_element: Dict[int, str]) -> None:
        """
        Log validation of molecular charges (for debugging, not enforcing).
        """
        self.logger.info("Validating molecular charges:")
        
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = 0.0
            charge_breakdown = []
            
            for type_id, count in mol_data["type_counts"].items():
                charge = charges.get(type_id, 0.0)
                contribution = charge * count
                mol_charge += contribution
                element = type_to_element.get(type_id, "?")
                charge_breakdown.append(f"{element}(type{type_id}):{charge:.3f}×{count}={contribution:.3f}")
            
            self.logger.info(
                f"  {mol_name}: total_charge={mol_charge:.4f} "
                f"[{', '.join(charge_breakdown)}]"
            )
    
    
    def _extract_charges_from_parameters(self,
                                          parameter_info: Dict[str, Any],
                                          data_info: Dict[str, Any]) -> Dict[int, float]:
        """
        Extract charge assignments from existing parameter info.
        
        If the data file has more types than parameter_info (due to type splitting),
        fall back to LLM generation for all charges to ensure context-awareness.
        """
        self.logger.info("Attempting to extract charges from parameter_info")
        
        # Check if types were split (data file has more types than parameter_info)
        n_param_types = len(parameter_info.get("atom_types", {}))
        n_data_types = len(data_info["masses"])
        
        if n_data_types > n_param_types:
            self.logger.warning(
                f"Data file has {n_data_types} types but parameter_info has {n_param_types}. "
                f"Types were likely split by molecular context. Using LLM for context-aware charges."
            )
            return self._generate_charge_assignments(data_info)
        
        if n_param_types == 0:
            self.logger.warning("parameter_info has no atom_types, using LLM")
            return self._generate_charge_assignments(data_info)
        
        # Build element -> charge lookup from parameter_info
        element_charges = {}
        atom_types = parameter_info.get("atom_types", {})
        
        self.logger.info(f"parameter_info has {n_param_types} atom types")
        
        for type_id_str, type_info in atom_types.items():
            if not isinstance(type_info, dict):
                continue
            
            element = type_info.get("name") or type_info.get("element")
            charge = type_info.get("charge")
            
            if element is not None and charge is not None:
                try:
                    charge = float(charge)
                    element_normalized = self._normalize_element_name(element)
                    
                    if element_normalized not in element_charges:
                        element_charges[element_normalized] = charge
                        self.logger.info(f"  {element} -> {element_normalized}: charge={charge}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not parse charge for {element}: {e}")
        
        if not element_charges:
            self.logger.warning("No valid charges found in parameter_info, using LLM")
            return self._generate_charge_assignments(data_info)
        
        self.logger.info(f"Element charges from parameter_info: {element_charges}")
        
        # Map element charges to data file type IDs
        charges = {}
        data_type_to_element = data_info["type_elements"]
        
        for type_id, element in data_type_to_element.items():
            element_normalized = self._normalize_element_name(element)
            
            if element_normalized in element_charges:
                charges[type_id] = element_charges[element_normalized]
                self.logger.info(f"Type {type_id} ({element}): charge={charges[type_id]}")
            else:
                self.logger.warning(f"No charge for element {element_normalized} (type {type_id})")
        
        # Fill in missing types
        missing_types = set(data_info["masses"].keys()) - set(charges.keys())
        
        if missing_types:
            self.logger.warning(f"Missing charges for {len(missing_types)} types, using LLM")
            llm_charges = self._generate_charge_assignments(data_info)
            
            for type_id in missing_types:
                charges[type_id] = llm_charges.get(
                    type_id, 
                    self._get_default_charge_for_element(data_type_to_element.get(type_id, "X"))
                )
        
        self.logger.info(f"Final charge assignments: {charges}")
        return charges

    def _normalize_element_name(self, element: str) -> str:
        """
        Normalize element names to handle variations.
        """
        if not element:
            return "X"
        
        # Strip whitespace and convert to string
        element = str(element).strip()
        
        # If it's already a standard 1-2 char symbol, just capitalize
        if len(element) <= 2 and element.isalpha():
            return element.capitalize()
        
        # Handle description-style names like "Water oxygen" or "CF3 carbon"
        # Extract just the element part
        element_lower = element.lower()
        
        # Check if any standard element name is in the string
        element_names = {
            "hydrogen": "H", "carbon": "C", "nitrogen": "N", "oxygen": "O",
            "fluorine": "F", "sulfur": "S", "chlorine": "Cl", "sodium": "Na",
            "potassium": "K", "calcium": "Ca", "magnesium": "Mg", "zinc": "Zn",
            "iron": "Fe", "copper": "Cu", "phosphorus": "P", "bromine": "Br",
        }
        
        for name, symbol in element_names.items():
            if name in element_lower:
                return symbol
        
        # Common variations mapping
        variations = {
            # Water oxygens
            "ow": "O", "oh": "O", "o_w": "O", "o_water": "O", "owater": "O",
            "ot": "O", "os": "O", "o": "O",
            # Water hydrogens  
            "hw": "H", "ho": "H", "h_w": "H", "h_water": "H", "hwater": "H",
            "ht": "H", "hs": "H", "h": "H",
            # Carbon types
            "ct": "C", "ca": "C", "c3": "C", "c2": "C", "cx": "C", "c": "C",
            # Sulfur
            "s": "S", "sh": "S", "ss": "S",
            # Fluorine
            "f": "F", "f1": "F",
            # Zinc
            "zn": "Zn", "zn2+": "Zn", "zn+2": "Zn",
            # Other common variations
            "na": "Na", "na+": "Na", "cl": "Cl", "cl-": "Cl",
            "fe": "Fe", "ca_ion": "Ca", "mg_ion": "Mg", "zn_ion": "Zn",
        }
        
        # Check variations
        if element_lower in variations:
            return variations[element_lower]
        
        # Check if first word is a variation
        first_word = element_lower.split()[0] if element_lower.split() else element_lower
        if first_word in variations:
            return variations[first_word]
        
        # Last resort: return first 1-2 characters
        if len(element) >= 2 and element[1].islower():
            return element[:2].capitalize()
        else:
            return element[0].upper()

    def _validate_charge_assignments(self,
                                      charge_assignments: Dict[int, float],
                                      data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that charge assignments are physically reasonable.
        
        Uses general chemistry principles, not system-specific expectations.
        
        Args:
            charge_assignments: Dictionary of type_id -> charge
            data_info: Parsed data file information
            
        Returns:
            Validation result dictionary
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "total_charge": 0.0,
            "molecule_charges": {}
        }
        
        # Check that all atom types have charges
        for type_id in data_info["masses"].keys():
            if type_id not in charge_assignments:
                validation["errors"].append(f"Missing charge for atom type {type_id}")
                validation["valid"] = False
        
        # Calculate total system charge
        total_charge = 0.0
        for type_id, count in data_info["type_counts"].items():
            charge = charge_assignments.get(type_id, 0.0)
            total_charge += charge * count
        
        validation["total_charge"] = total_charge
        
        # Check if total charge is reasonable (should be near integer)
        rounded_total = round(total_charge)
        if abs(total_charge - rounded_total) > 0.1:
            validation["warnings"].append(
                f"Total system charge ({total_charge:.4f}) is not close to integer {rounded_total}"
            )
        
        # Calculate charge per molecule type
        mol_compositions = self._analyze_molecular_compositions(data_info)
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = sum(
                charge_assignments.get(type_id, 0.0) * count
                for type_id, count in mol_data["type_counts"].items()
            )
            validation["molecule_charges"][mol_name] = mol_charge
            
            # Check if molecule charge is close to an integer
            # Most molecules should have integer formal charges (0, ±1, ±2, etc.)
            rounded_mol_charge = round(mol_charge)
            if abs(mol_charge - rounded_mol_charge) > 0.15:
                validation["warnings"].append(
                    f"{mol_name} charge ({mol_charge:.4f}) deviates significantly from integer {rounded_mol_charge}"
                )
        
        # Element-specific sanity checks based on general chemistry
        # These are soft warnings, not hard failures
        for type_id, charge in charge_assignments.items():
            element = data_info["type_elements"].get(type_id, "?")
            
            # Very large charges are suspicious for any element
            if abs(charge) > 4.0:
                validation["warnings"].append(
                    f"Unusually large charge magnitude for type {type_id} ({element}): {charge:.4f}"
                )
            
            # Hydrogen is almost always positive or near-zero in molecular systems
            if element == "H" and charge < -0.5:
                validation["warnings"].append(
                    f"Unusually negative hydrogen (type {type_id}): {charge:.4f}"
                )
            
            # Oxygen is typically negative in most contexts
            if element == "O" and charge > 1.0:
                validation["warnings"].append(
                    f"Unusually positive oxygen (type {type_id}): {charge:.4f}"
                )
            
            # Fluorine is the most electronegative element - rarely positive
            if element == "F" and charge > 0.3:
                validation["warnings"].append(
                    f"Positive charge on fluorine (type {type_id}): {charge:.4f}"
                )
            
            # Halogens (Cl, Br, I) are typically negative as ions or slightly negative in molecules
            if element in ["Cl", "Br", "I"] and charge > 0.5:
                validation["warnings"].append(
                    f"Unusually positive halogen {element} (type {type_id}): {charge:.4f}"
                )
            
            # Alkali metals should be positive
            if element in ["Li", "Na", "K", "Rb", "Cs"] and charge < 0.5:
                validation["warnings"].append(
                    f"Alkali metal {element} (type {type_id}) has low charge: {charge:.4f}"
                )
            
            # Alkaline earth and transition metals typically positive
            if element in ["Mg", "Ca", "Zn", "Fe", "Cu", "Ni", "Co", "Mn"] and charge < 0:
                validation["warnings"].append(
                    f"Negative charge on metal {element} (type {type_id}): {charge:.4f}"
                )
        
        return validation

    def _write_data_file_with_charges(self,
                                       input_file: str,
                                       output_file: str,
                                       charge_assignments: Dict[int, float],
                                       data_info: Dict[str, Any]) -> None:
        """
        Write a new data file with charges assigned.
        
        Args:
            input_file: Path to input data file
            output_file: Path to output data file
            charge_assignments: Dictionary of type_id -> charge
            data_info: Parsed data file information
        """
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        in_atoms_section = False
        charges_written = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Detect Atoms section
            if stripped.startswith("Atoms"):
                in_atoms_section = True
                new_lines.append(line)
                continue
            
            # Detect end of Atoms section (next section header or significant gap)
            if in_atoms_section:
                if stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", 
                              "Impropers", "Pair Coeffs", "Bond Coeffs",
                              "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"]:
                    in_atoms_section = False
                    new_lines.append(line)
                    continue
                
                # Skip empty lines and comments in atoms section
                if not stripped or stripped.startswith("#"):
                    new_lines.append(line)
                    continue
                
                # Parse and modify atom line
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = parts[0]
                        mol_id = parts[1]
                        atom_type = int(parts[2])
                        x, y, z = parts[4], parts[5], parts[6]
                        
                        # Get new charge
                        new_charge = charge_assignments.get(atom_type, 0.0)
                        
                        # Preserve comment if present
                        comment = ""
                        if "#" in line:
                            comment = " #" + line.split("#", 1)[1].rstrip("\n")
                        
                        # Reconstruct line
                        new_line = (
                            f"{atom_id:>8} {mol_id:>8} {atom_type:>4} {new_charge:>12.6f} "
                            f"{x:>14} {y:>14} {z:>14}{comment}\n"
                        )
                        new_lines.append(new_line)
                        charges_written += 1
                        continue
                        
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse atom line {i}: {e}")
                        new_lines.append(line)
                        continue
            
            # Keep all other lines unchanged
            new_lines.append(line)
        
        # Write output file
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        
        self.logger.info(f"Wrote data file with charges: {output_file} ({charges_written} atoms)")
        
        if charges_written == 0:
            self.logger.error("WARNING: No charges were written! Check Atoms section detection.")

    def complete_parameterization(self,
                                   pdb_file: str,
                                   data_file: str,
                                   research_goal: str,
                                   system_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete parameterization workflow: select force field, acquire parameters,
        assign charges to data file, and generate LAMMPS parameter files.
        
        Args:
            pdb_file: Path to PDB file (for analysis)
            data_file: Path to LAMMPS data file (to add charges to)
            research_goal: Research objective
            system_description: Optional system description
            
        Returns:
            Dictionary with all parameterization results and file paths
        """
        self.logger.info("="*60)
        self.logger.info("COMPLETE PARAMETERIZATION WORKFLOW")
        self.logger.info("="*60)
        
        results = {
            "status": "success",
            "input_files": {
                "pdb_file": pdb_file,
                "data_file": data_file
            },
            "output_files": {},
            "force_field": None,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Select force field
            self.logger.info("\n[Step 1/4] Selecting force field...")
            selection_info = self.select_force_field(
                pdb_file=pdb_file,
                research_goal=research_goal,
                system_description=system_description
            )
            results["force_field"] = selection_info["force_field"]
            results["output_files"]["selection_info"] = os.path.join(
                self.working_dir, "force_field_selection.json"
            )
            
            # Step 2: Acquire parameters
            self.logger.info("\n[Step 2/4] Acquiring force field parameters...")
            parameter_info = self.acquire_parameters(
                selection_info=selection_info,
                data_file=data_file
            )
            results["output_files"]["parameter_info"] = os.path.join(
                self.working_dir, "parameter_info.json"
            )
            
            # Step 2.5: Split atom types if same element appears in different molecules
            self.logger.info("\n[Step 2.5/4] Checking for atom types needing context-based splitting...")
            data_file = self.split_atom_types_by_molecule_context(data_file)
            
            # Step 3: Assign charges to data file
            self.logger.info("\n[Step 3/4] Assigning charges to data file...")
            charged_data_file = self.assign_charges_to_data_file(
                data_file=data_file,
                parameter_info=parameter_info,
                pdb_file=pdb_file,
                research_goal=research_goal
            )
            results["output_files"]["charged_data_file"] = charged_data_file
            results["output_files"]["charge_assignments"] = os.path.join(
                self.working_dir, "charge_assignments.json"
            )
            
            # Step 4: Generate LAMMPS parameter files
            self.logger.info("\n[Step 4/4] Generating LAMMPS parameter files...")
            param_files = self.generate_lammps_parameters(
                parameter_info=parameter_info,
                data_file=charged_data_file
            )
            results["output_files"]["parameter_files"] = param_files
            
            # Collect any warnings from validation
            charge_info_file = os.path.join(self.working_dir, "charge_assignments.json")
            if os.path.exists(charge_info_file):
                with open(charge_info_file, 'r') as f:
                    charge_info = json.load(f)
                    validation = charge_info.get("validation", {})
                    results["warnings"].extend(validation.get("warnings", []))
            
            self.logger.info("\n" + "="*60)
            self.logger.info("PARAMETERIZATION COMPLETE")
            self.logger.info("="*60)
            self.logger.info(f"Force field: {results['force_field'].get('force_field', 'Unknown')}")
            self.logger.info(f"Charged data file: {charged_data_file}")
            self.logger.info(f"Parameter files: {list(param_files.keys())}")
            
        except Exception as e:
            self.logger.error(f"Parameterization failed: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
            import traceback
            results["traceback"] = traceback.format_exc()
        
        return results

    def split_atom_types_by_molecule_context(self,
                                              data_file: str,
                                              output_file: Optional[str] = None) -> str:
        """
        Split atom types that appear in different molecular contexts.
        """
        self.logger.info(f"Analyzing atom types for context-based splitting: {data_file}")
        
        if output_file is None:
            output_file = data_file
        
        # Parse the data file
        data_info = self._parse_data_file_for_charges(data_file)
        mol_compositions = self._analyze_molecular_compositions(data_info)
        
        # Find elements that appear in multiple molecule types
        element_to_molecules = {}
        for mol_name, mol_data in mol_compositions.items():
            for type_id in mol_data["type_counts"].keys():
                element = data_info["type_elements"].get(type_id, "X")
                if element not in element_to_molecules:
                    element_to_molecules[element] = set()
                element_to_molecules[element].add(mol_name)
        
        # Identify elements needing splitting
        elements_to_split = {
            el: mols for el, mols in element_to_molecules.items()
            if len(mols) > 1
        }
        
        if not elements_to_split:
            self.logger.info("No atom types need splitting")
            return data_file
        
        self.logger.info(f"Elements in multiple contexts: {elements_to_split}")
        
        # Read original file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Determine which types need splitting and build mapping
        old_n_types = data_info["n_atom_types"]
        new_type_counter = old_n_types
        
        type_split_map = {}  # old_type -> {mol_name: new_type}
        new_type_info = {}   # new_type -> {"mass": ..., "element": ..., "context": ...}
        
        for element, mol_names in elements_to_split.items():
            # Find original type for this element
            original_type = None
            for type_id, el in data_info["type_elements"].items():
                if el == element and type_id <= old_n_types:
                    original_type = type_id
                    break
            
            if original_type is None:
                continue
            
            # Keep original type for the most common molecule
            mol_list = sorted(mol_names, key=lambda m: -mol_compositions[m]["count"])
            
            type_split_map[original_type] = {}
            
            for i, mol_name in enumerate(mol_list):
                if i == 0:
                    type_split_map[original_type][mol_name] = original_type
                else:
                    new_type_counter += 1
                    type_split_map[original_type][mol_name] = new_type_counter
                    new_type_info[new_type_counter] = {
                        "mass": data_info["masses"][original_type],
                        "element": element,
                        "context": mol_name
                    }
                    self.logger.info(
                        f"New type {new_type_counter} for {element} in {mol_name}"
                    )
        
        if not new_type_info:
            self.logger.info("No new types needed after analysis")
            return data_file
        
        # Build mol_id -> mol_name mapping
        mol_id_to_name = {}
        for mol_name, mol_data in mol_compositions.items():
            for mol_id, atoms in data_info["molecules"].items():
                type_counts = {}
                for atom_id, atom_type in atoms:
                    type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
                if type_counts == mol_data["type_counts"]:
                    mol_id_to_name[mol_id] = mol_name
        
        # Rewrite the file
        new_lines = []
        in_masses = False
        in_atoms = False
        masses_collected = []  # Collect all mass lines
        header_updated = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Update atom types count in header
            if not header_updated and "atom types" in stripped.lower():
                parts = stripped.split()
                try:
                    old_count = int(parts[0])
                    new_count = new_type_counter
                    new_lines.append(f" {new_count} atom types\n")
                    header_updated = True
                    i += 1
                    continue
                except (ValueError, IndexError):
                    pass
            
            # Detect Masses section
            if stripped == "Masses":
                in_masses = True
                new_lines.append(line)
                i += 1
                continue
            
            # Collect mass lines
            if in_masses:
                # Check if we've hit the next section or Atoms
                if stripped.startswith("Atoms") or stripped.startswith("Pair"):
                    # We've reached the end of masses section
                    # Write all original masses in order, then new ones
                    
                    # Sort collected masses by type ID
                    masses_collected.sort(key=lambda x: x[0])
                    
                    # Write blank line after "Masses" header
                    new_lines.append("\n")
                    
                    # Write original masses
                    for type_id, mass_line in masses_collected:
                        new_lines.append(mass_line)
                    
                    # Write new masses
                    for new_type in sorted(new_type_info.keys()):
                        info = new_type_info[new_type]
                        new_lines.append(
                            f" {new_type} {info['mass']:.6f} # {info['element']} ({info['context']})\n"
                        )
                    
                    # Write blank line before next section
                    new_lines.append("\n")
                    
                    # Done with masses
                    in_masses = False
                    
                    # Now write the current line (Atoms or Pair header)
                    new_lines.append(line)
                    i += 1
                    continue
                
                # Skip blank lines in masses section
                if not stripped:
                    i += 1
                    continue
                
                # Skip comment lines
                if stripped.startswith("#"):
                    i += 1
                    continue
                
                # Parse and collect mass line
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        type_id = int(parts[0])
                        # Keep the original line formatting
                        masses_collected.append((type_id, line))
                    except ValueError:
                        new_lines.append(line)
                
                i += 1
                continue
            
            # Detect Atoms section
            if stripped.startswith("Atoms"):
                in_atoms = True
                new_lines.append(line)
                i += 1
                continue
            
            # Detect end of Atoms section
            if in_atoms and stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]:
                in_atoms = False
                new_lines.append(line)
                i += 1
                continue
            
            # Modify atom lines to use new types where needed
            if in_atoms and stripped and not stripped.startswith("#"):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = int(parts[0])
                        mol_id = int(parts[1])
                        old_type = int(parts[2])
                        charge = parts[3]
                        x, y, z = parts[4], parts[5], parts[6]
                        
                        # Preserve comment
                        comment = ""
                        if "#" in line:
                            comment = " #" + line.split("#", 1)[1].rstrip("\n")
                        
                        # Check if this type needs remapping
                        new_type = old_type
                        if old_type in type_split_map:
                            mol_name = mol_id_to_name.get(mol_id)
                            if mol_name and mol_name in type_split_map[old_type]:
                                new_type = type_split_map[old_type][mol_name]
                        
                        new_line = (
                            f"{atom_id:>8} {mol_id:>8} {new_type:>4} {charge:>12} "
                            f"{x:>14} {y:>14} {z:>14}{comment}\n"
                        )
                        new_lines.append(new_line)
                        i += 1
                        continue
                        
                    except (ValueError, IndexError):
                        pass
            
            # Keep all other lines unchanged
            new_lines.append(line)
            i += 1
        
        # Write output
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        
        self.logger.info(f"Split {len(new_type_info)} atom types, total now: {new_type_counter}")
        self.logger.info(f"Wrote: {output_file}")
        
        return output_file


    def diagnose_and_fix_force_field(self,
                                      quality_result: Dict[str, Any],
                                      research_goal: str,
                                      data_file: str,
                                      ff_params_path: str,
                                      stage: str) -> Dict[str, Any]:
        """
        Diagnose and fix force field parameters based on quality analysis.
        
        Args:
            quality_result: Quality check output from LAMMPSAnalysisAgent
            research_goal: Research objective
            data_file: Path to current LAMMPS data file
            ff_params_path: Path to ff_params.lammps
            stage: Current simulation stage name
            
        Returns:
            Dictionary with:
                - ff_modified: bool
                - charges_modified: bool
                - diagnosis: str
                - ff_backup: path to backup (if modified)
                - charge_backup: path to backup (if modified)
                - details: dict with full analysis
        """
        self.logger.info(f"Diagnosing quality issues for stage: {stage}")
        
        result = {
            "ff_modified": False,
            "charges_modified": False,
            "diagnosis": "",
            "details": {}
        }
        
        issues = quality_result.get("issues", [])
        recommendations = quality_result.get("recommendations", [])
        metrics = quality_result.get("quality_metrics", {})
        
        # Classify what needs fixing
        issue_text = " ".join([
            i.get("description", "").lower() for i in issues
        ])
        rec_text = " ".join([
            (r.get("description", "") if isinstance(r, dict) else str(r)).lower()
            for r in recommendations
        ])
        all_text = issue_text + " " + rec_text
        
        needs_ff = any(kw in all_text for kw in [
            "density", "volume", "coordination", "rdf", "radial",
            "solvation", "energy", "lj", "lennard", "sigma", "epsilon",
            "mixing rule", "pair_modify"
        ])
        
        needs_charges = any(kw in all_text for kw in [
            "charge", "electrostatic", "coulomb", "neutral", "dipole",
            "density", "structure"
        ])
        
        # Fix force field parameters
        if needs_ff and os.path.exists(ff_params_path):
            self.logger.info("Attempting force field parameter correction...")
            ff_fixed, ff_info = self._diagnose_ff_params(
                quality_result=quality_result,
                research_goal=research_goal,
                data_file=data_file,
                ff_params_path=ff_params_path,
                stage=stage
            )
            result["ff_modified"] = ff_fixed
            result["details"]["force_field"] = ff_info
            if ff_fixed:
                result["ff_backup"] = ff_info.get("backup")
        
        # Fix charges
        if needs_charges:
            self.logger.info("Attempting charge correction...")
            charge_fixed, charge_info = self._diagnose_charges(
                quality_result=quality_result,
                research_goal=research_goal,
                data_file=data_file,
                stage=stage
            )
            result["charges_modified"] = charge_fixed
            result["details"]["charges"] = charge_info
            if charge_fixed:
                result["charge_backup"] = charge_info.get("backup")
        
        # Build diagnosis summary
        diagnosis_parts = []
        if result["ff_modified"]:
            diagnosis_parts.append(f"FF parameters adjusted: {result['details']['force_field'].get('summary', '')}")
        if result["charges_modified"]:
            diagnosis_parts.append(f"Charges adjusted: {result['details']['charges'].get('summary', '')}")
        if not diagnosis_parts:
            diagnosis_parts.append("No parameter changes needed - issue may be in simulation settings")
        
        result["diagnosis"] = "; ".join(diagnosis_parts)
        
        return result
    
    
    def _diagnose_ff_params(self,
                             quality_result: Dict[str, Any],
                             research_goal: str,
                             data_file: str,
                             ff_params_path: str,
                             stage: str) -> Tuple[bool, Dict[str, Any]]:
        """Diagnose and fix force field parameters based on quality metrics."""
        
        with open(ff_params_path, 'r') as f:
            current_ff = f.read()
        
        data_info = self._parse_data_file_for_charges(data_file)
        
        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            element = data_info["type_elements"].get(type_id, "?")
            mass = data_info["masses"][type_id]
            name = data_info.get("type_names", {}).get(type_id, element)
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: {name} ({element}), mass={mass:.3f}, count={count}")
        type_info_str = "\n".join(type_info)
        
        # Get thermo data from log
        log_file = os.path.join(self.working_dir, "log.lammps")
        thermo_excerpt = ""
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            thermo_lines = [l for l in log_lines if l.strip() and
                           l.strip()[0].isdigit() and len(l.split()) >= 5]
            if thermo_lines:
                thermo_excerpt = "".join(thermo_lines[-20:])
        
        issues_str = "\n".join([
            f"- [{i.get('severity', '?')}] {i.get('description', '')}"
            for i in quality_result.get("issues", [])
        ])
        
        metrics_str = json.dumps(quality_result.get("quality_metrics", {}), indent=2, default=str)
        
        prompt = f"""
    You are an expert molecular dynamics force field specialist. A LAMMPS simulation 
    completed but quality analysis found physics problems.
    
    RESEARCH GOAL: {research_goal}
    STAGE: {stage}
    
    QUALITY ISSUES:
    {issues_str}
    
    QUALITY METRICS:
    {metrics_str}
    
    ATOM TYPES IN DATA FILE:
    {type_info_str}
    
    CURRENT FORCE FIELD (ff_params.lammps):
    {current_ff}
    
    RECENT THERMO OUTPUT:
    {thermo_excerpt[-1500:]}
    
    Analyze the quality issues and determine if force field parameters need adjustment.
    
    Common fixes:
    - Low density: pair_modify mix should be geometric for OPLS-AA, check sigma values
    - Wrong coordination: adjust epsilon/sigma for relevant pairs
    - High energy: check for missing or incorrect coefficients
    - Wrong mixing rule is a very common error
    
    Return JSON:
    {{
        "diagnosis": "What is wrong and why",
        "changes_needed": true/false,
        "changes": [
            {{
                "what": "Description",
                "why": "Physical justification",
                "original_line": "original line",
                "corrected_line": "corrected line"
            }}
        ],
        "corrected_ff_params": "Complete corrected ff_params.lammps content or null"
    }}
    """
        
        try:
            result = self._generate_json(prompt)
            
            diagnosis = result.get("diagnosis", "No diagnosis")
            self.logger.info(f"FF diagnosis: {diagnosis[:200]}")
            
            if not result.get("changes_needed", False):
                return False, {"diagnosis": diagnosis, "changes_needed": False}
            
            corrected_ff = result.get("corrected_ff_params")
            if corrected_ff and corrected_ff.strip() != current_ff.strip():
                import shutil
                backup_path = ff_params_path + f".before_quality_{stage}"
                shutil.copy2(ff_params_path, backup_path)
                
                with open(ff_params_path, 'w') as f:
                    f.write(corrected_ff)
                
                changes = result.get("changes", [])
                return True, {
                    "summary": f"{len(changes)} parameter changes: {diagnosis[:80]}",
                    "diagnosis": diagnosis,
                    "changes": changes,
                    "backup": backup_path
                }
            
            return False, {"diagnosis": diagnosis, "no_actual_changes": True}
            
        except Exception as e:
            self.logger.error(f"FF diagnosis failed: {e}")
            return False, {"error": str(e)}    
    
    def _diagnose_charges(self,
                           quality_result: Dict[str, Any],
                           research_goal: str,
                           data_file: str,
                           stage: str) -> Tuple[bool, Dict[str, Any]]:
        """Diagnose and fix charge assignments based on quality metrics."""
        
        data_info = self._parse_data_file_for_charges(data_file)
        mol_compositions = self._analyze_molecular_compositions(data_info)
        
        # Read charge_assignments.json if it exists
        charge_file = os.path.join(self.working_dir, "charge_assignments.json")
        current_charges = {}
        validation = {}
        if os.path.exists(charge_file):
            with open(charge_file, 'r') as f:
                charge_data = json.load(f)
                current_charges = charge_data.get("charge_assignments", {})
                validation = charge_data.get("validation", {})
        
        if not current_charges:
            return False, {"error": "No charge assignments found"}
        
        # Build info strings OUTSIDE f-string
        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            element = data_info["type_elements"].get(type_id, "?")
            name = data_info.get("type_names", {}).get(type_id, element)
            charge = current_charges.get(str(type_id), "?")
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: {name} ({element}), charge={charge}, count={count}")
        type_info_str = "\n".join(type_info)
        
        mol_info = []
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = sum(
                float(current_charges.get(str(tid), 0)) * cnt
                for tid, cnt in mol_data["type_counts"].items()
            )
            mol_info.append(f"  {mol_name}: {mol_data['count']} molecules, charge={mol_charge:.4f}")
        mol_info_str = "\n".join(mol_info)
        
        issues_str = "\n".join([
            f"- [{i.get('severity', '?')}] {i.get('description', '')}"
            for i in quality_result.get("issues", [])
        ])
        
        prompt = f"""
    You are an expert in molecular dynamics charge parameterization. A simulation has
    quality issues that may be related to partial charges.
    
    RESEARCH GOAL: {research_goal}
    
    QUALITY ISSUES:
    {issues_str}
    
    ATOM TYPES AND CURRENT CHARGES:
    {type_info_str}
    
    MOLECULAR CHARGES:
    {mol_info_str}
    
    Determine if charges need adjustment. Key rules:
    - Water molecules must sum to exactly 0.0
    - Monatomic ions have their formal charge
    - Polyatomic anions sum to their formal charge
    - Total system should be neutral
    
    Return JSON:
    {{
        "diagnosis": "Analysis of charge issues",
        "changes_needed": true/false,
        "corrected_charges": {{
            "type_id_string": new_charge_float
        }}
    }}
    """
        
        try:
            result = self._generate_json(prompt)
            
            diagnosis = result.get("diagnosis", "No diagnosis")
            self.logger.info(f"Charge diagnosis: {diagnosis[:200]}")
            
            if not result.get("changes_needed", False):
                return False, {"diagnosis": diagnosis}
            
            corrected_charges = result.get("corrected_charges")
            if not corrected_charges:
                return False, {"diagnosis": diagnosis, "no_corrections": True}
            
            # Convert string keys to the format _write_data_file_with_charges expects
            int_charges = {}
            for k, v in corrected_charges.items():
                try:
                    int_charges[int(k)] = float(v)
                except (ValueError, TypeError):
                    continue
            
            # Backup original
            import shutil
            backup_path = data_file + f".before_charge_fix_{stage}"
            shutil.copy2(data_file, backup_path)
            
            # Use the existing method that writes charges
            self._write_data_file_with_charges(
                input_file=data_file,
                output_file=data_file,  # Overwrite in place
                charge_assignments=int_charges,
                data_info=data_info
            )
            
            # Update charge_assignments.json
            new_validation = self._validate_charge_assignments(int_charges, data_info)
            with open(charge_file, 'w') as f:
                json.dump({
                    "charge_assignments": {str(k): v for k, v in int_charges.items()},
                    "validation": new_validation,
                    "quality_fix_stage": stage,
                    "previous_charges": current_charges
                }, f, indent=2)
            
            return True, {
                "summary": f"Updated {len(int_charges)} types, total_charge={new_validation.get('total_charge', 0):.4f}",
                "diagnosis": diagnosis,
                "backup": backup_path
            }
            
        except Exception as e:
            self.logger.error(f"Charge diagnosis failed: {e}")
            return False, {"error": str(e)}
