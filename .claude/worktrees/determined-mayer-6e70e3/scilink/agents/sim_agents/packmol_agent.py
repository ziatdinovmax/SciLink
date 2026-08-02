import os
import json
import logging
import shutil
import subprocess
import re
from typing import Dict, Any, List, Optional

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from .instruct import (
    MOLECULE_EXTRACTION_TEMPLATE,
    SMILES_GENERATION_TEMPLATE,
    PACKMOL_SCRIPT_GENERATION_TEMPLATE
)
from ._deprecation import normalize_params

from ase.build import molecule
from ase.collections import g2
from ase.io import write
from ase.data.pubchem import pubchem_atoms_search

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class PackmolGeneratorAgent:
    """
    General-purpose PACKMOL agent that can handle arbitrary molecular systems by:
    1. Using LLM to identify and parse molecules from descriptions
    2. Systematically searching multiple databases for ANY molecule
    3. Generating SMILES for unknown molecules using LLM
    4. Providing detailed, educational prompts for PACKMOL generation
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: str = "gemini-2.5-flash-preview-05-20", 
                 working_dir: str = "packmol_run",
                 base_url: Optional[str] = None,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the PACKMOL generator agent.
        
        Args:
            api_key: API key for the LLM provider
            model_name: Model name to use
            working_dir: Working directory for output files
            base_url: Optional base URL for internal proxy
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        self.working_dir = working_dir
        
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
            source="PackmolGeneratorAgent"
        )
        
        # Initialize model using wrapper structure
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy.")

            self.logger.info(f"PackmolGeneratorAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"PackmolGeneratorAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.model_name = model_name
        self.generation_config = None
        
        # Initialize available molecule sources
        self.ase_molecules = set(g2.names)
        self.logger.info(f"Available sources: ASE ({len(self.ase_molecules)} molecules), PubChem, RDKit")

        if not shutil.which("packmol"):
            raise FileNotFoundError("PACKMOL executable not found in PATH")

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

    def _extract_molecules_with_llm(self, description: str) -> List[Dict[str, Any]]:
        """Use LLM to intelligently extract molecules and their properties from description"""
        
        extraction_prompt = MOLECULE_EXTRACTION_TEMPLATE.format(description=description)

        try:
            analysis = self._generate_json(extraction_prompt)
            self.logger.info(f"LLM extracted molecules: {[mol['identifier'] for mol in analysis['molecules']]}")
            return analysis['molecules']
            
        except Exception as e:
            self.logger.error(f"LLM molecule extraction failed: {e}")
            # Fallback to simple pattern matching
            return self._simple_molecule_extraction(description)

    def _simple_molecule_extraction(self, description: str) -> List[Dict[str, Any]]:
        """Fallback: simple pattern matching for molecule extraction"""
        molecules = []
        
        # Extract obvious chemical formulas
        formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'
        formulas = re.findall(formula_pattern, description)
        
        for formula in formulas:
            if len(formula) > 1:  # Skip single letters
                molecules.append({
                    "identifier": formula,
                    "formula": formula,
                    "alternative_names": [formula.lower()],
                    "smiles": None,
                    "estimated_count": "unknown"
                })
        
        # Extract common molecule names
        common_names = ['water', 'benzene', 'methanol', 'ethanol', 'acetone', 'dmso']
        for name in common_names:
            if name in description.lower():
                molecules.append({
                    "identifier": name,
                    "formula": None,
                    "alternative_names": [name],
                    "smiles": None,
                    "estimated_count": "unknown"
                })
        
        return molecules

    def _build_molecule_systematically(self, mol_info: Dict[str, Any]) -> Optional[str]:
        """Systematically try all available methods to build a molecule"""
        identifier = mol_info['identifier']
        safe_name = re.sub(r'[^\w\-_]', '_', identifier.lower())
        
        self.logger.info(f"Building molecule: {identifier}")
        
        # Strategy 1: Try ASE if we have a formula
        if mol_info.get('formula') and mol_info['formula'] in self.ase_molecules:
            filepath = self._try_ase(mol_info['formula'], safe_name)
            if filepath:
                return filepath
        
        # Strategy 2: Try PubChem with all possible names
        search_terms = [identifier]
        if mol_info.get('alternative_names'):
            search_terms.extend(mol_info['alternative_names'])
        if mol_info.get('formula'):
            search_terms.append(mol_info['formula'])
            
        for term in search_terms:
            filepath = self._try_pubchem(term, safe_name)
            if filepath:
                return filepath
        
        # Strategy 3: Try RDKit with SMILES
        if mol_info.get('smiles') and HAS_RDKIT:
            filepath = self._try_rdkit_smiles(mol_info['smiles'], safe_name)
            if filepath:
                return filepath
        
        # Strategy 4: Ask LLM for SMILES if we don't have it
        if not mol_info.get('smiles'):
            smiles = self._get_smiles_from_llm(identifier)
            if smiles and HAS_RDKIT:
                filepath = self._try_rdkit_smiles(smiles, safe_name)
                if filepath:
                    return filepath
        
        self.logger.warning(f"Failed to build {identifier} with all available methods")
        return None

    def _try_ase(self, formula: str, safe_name: str) -> Optional[str]:
        """Try building with ASE"""
        try:
            mol = molecule(formula)
            filepath = os.path.join(self.working_dir, "components", f"{safe_name}.pdb")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            write(filepath, mol)
            self.logger.info(f"✓ Built {formula} via ASE")
            return filepath
        except Exception as e:
            self.logger.debug(f"ASE failed for {formula}: {e}")
            return None

    def _try_pubchem(self, search_term: str, safe_name: str) -> Optional[str]:
        """Try building with PubChem"""
        try:
            # Clean the search term
            clean_term = search_term.strip()
            
            atoms = pubchem_atoms_search(name=clean_term)
            if atoms is None:
                return None
                
            filepath = os.path.join(self.working_dir, "components", f"{safe_name}.pdb")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            write(filepath, atoms)
            self.logger.info(f"✓ Built {search_term} via PubChem")
            return filepath
            
        except Exception as e:
            self.logger.debug(f"PubChem failed for {search_term}: {e}")
            return None

    def _try_rdkit_smiles(self, smiles: str, safe_name: str) -> Optional[str]:
        """Try building with RDKit from SMILES"""
        try:
            # Handle ionic compounds by trying to build without charges first
            clean_smiles = smiles
            
            # For ionic compounds like [Zn+2].[O-]S(=O)(=O)C(F)(F)F.[O-]S(=O)(=O)C(F)(F)F
            # Try building just the organic part if the full ionic SMILES fails
            if '[' in smiles and '+' in smiles:
                # Try to extract non-metal parts
                parts = smiles.split('.')
                organic_parts = [part for part in parts if not any(metal in part for metal in ['Zn', 'Li', 'Na', 'K', 'Mg', 'Ca', '[', '+', '-'])]
                if organic_parts:
                    clean_smiles = '.'.join(organic_parts)
                    self.logger.info(f"Trying organic part only: {clean_smiles}")
            
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is None:
                return None
                
            mol = Chem.AddHs(mol)
            
            # Try embedding with multiple attempts
            success = False
            for attempt in range(3):
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42 + attempt)
                    success = True
                    break
                except Exception as e:
                    self.logger.debug(f"Embedding attempt {attempt + 1} failed: {e}")
                    continue
            
            if not success:
                self.logger.debug(f"Could not generate 3D coordinates for {smiles}")
                return None
            
            # Try geometry optimization (skip if it fails)
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception as e:
                self.logger.debug(f"Geometry optimization failed, using unoptimized structure: {e}")
            
            # Convert to ASE
            conf = mol.GetConformer()
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
                
            from ase import Atoms
            atoms = Atoms(symbols=symbols, positions=positions)
            
            filepath = os.path.join(self.working_dir, "components", f"{safe_name}.pdb")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            write(filepath, atoms)
            
            self.logger.info(f"✓ Built from SMILES: {clean_smiles} ({len(atoms)} atoms)")
            return filepath
            
        except Exception as e:
            self.logger.debug(f"RDKit failed for SMILES {smiles}: {e}")
            return None

    def _get_smiles_from_llm(self, molecule_identifier: str) -> Optional[str]:
        """Ask LLM to provide SMILES for unknown molecules"""
        
        smiles_prompt = SMILES_GENERATION_TEMPLATE.format(molecule_identifier=molecule_identifier)

        try:
            smiles = self._generate_text(smiles_prompt).strip()
            
            if smiles == "UNKNOWN" or len(smiles) > 200:  # Sanity check
                return None
                
            self.logger.info(f"LLM provided SMILES for {molecule_identifier}: {smiles}")
            return smiles
            
        except Exception as e:
            self.logger.debug(f"LLM SMILES generation failed for {molecule_identifier}: {e}")
            return None

    def _create_comprehensive_packmol_prompt(self, description: str, built_molecules: Dict[str, str]) -> str:
        """Create detailed PACKMOL prompt with educational content"""
        
        molecule_list = "\n".join([f"- {name}: components/{name}.pdb" for name in built_molecules.keys()])
        
        return PACKMOL_SCRIPT_GENERATION_TEMPLATE.format(
            description=description,
            molecule_list=molecule_list
        )

    def generate_structure(self, description: str) -> Dict[str, Any]:
        """Generate arbitrary molecular structure using general methods"""
        try:
            os.makedirs(self.working_dir, exist_ok=True)
            
            # 1. Extract molecules from description using LLM
            molecule_list = self._extract_molecules_with_llm(description)
            
            if not molecule_list:
                return {
                    "status": "error",
                    "message": "Could not identify any molecules in the description. Please specify molecules more clearly."
                }
            
            # 2. Build all molecules systematically
            built_molecules = {}
            failed_molecules = []
            
            for mol_info in molecule_list:
                filepath = self._build_molecule_systematically(mol_info)
                if filepath:
                    safe_name = re.sub(r'[^\w\-_]', '_', mol_info['identifier'].lower())
                    built_molecules[safe_name] = filepath
                else:
                    failed_molecules.append(mol_info['identifier'])
            
            if not built_molecules:
                available_methods = ["ASE g2 database", "PubChem", "LLM-generated SMILES"]
                if HAS_RDKIT:
                    available_methods.append("RDKit")
                    
                return {
                    "status": "error",
                    "message": f"Could not build any molecules. Failed: {failed_molecules}. Available methods: {', '.join(available_methods)}. Try providing chemical formulas or SMILES strings."
                }
            
            # Log results
            self.logger.info(f"Successfully built: {list(built_molecules.keys())}")
            if failed_molecules:
                self.logger.warning(f"Failed to build: {failed_molecules}")
            
            # 3. Generate PACKMOL script
            prompt = self._create_comprehensive_packmol_prompt(description, built_molecules)
            
            try:
                llm_response = self._generate_json(prompt)
                
            except Exception as e:
                return {"status": "error", "message": f"LLM script generation failed: {str(e)}"}
            
            # 4. Execute PACKMOL
            try:
                output_filename = llm_response["output_filename"]
                script_content = llm_response["packmol_script"].replace("\\n", "\n")
                final_script = script_content.format(output_filename=output_filename)
                
                script_path = os.path.join(self.working_dir, "input.inp")
                with open(script_path, "w") as f:
                    f.write(final_script)
                
                self.logger.info(f"PACKMOL script written to: {script_path}")
                self.logger.info(f"Script content:\n{final_script}")
                
                # Execute PACKMOL with proper file input method
                command = f"packmol < {os.path.basename(script_path)}"
                self.logger.info(f"Running command: {command}")
                
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=self.working_dir, timeout=300
                )
                
                if result.returncode != 0:
                    return {
                        "status": "error",
                        "message": f"PACKMOL execution failed:\nReturn code: {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}\nScript content:\n{final_script}"
                    }
                
                output_path = os.path.join(self.working_dir, output_filename)
                if not os.path.exists(output_path):
                    return {
                        "status": "error", 
                        "message": f"PACKMOL completed but output file missing: {output_filename}\nStdout: {result.stdout}\nWorking directory contents: {os.listdir(self.working_dir)}"
                    }
                
                return {
                    "status": "success",
                    "output_file": os.path.abspath(output_path),
                    "built_molecules": list(built_molecules.keys()),
                    "failed_molecules": failed_molecules,
                    "message": f"Successfully generated molecular system with {len(built_molecules)} molecule types"
                }
                
            except Exception as e:
                return {"status": "error", "message": f"PACKMOL execution error: {str(e)}"}
                
        except Exception as e:
            self.logger.exception("Critical error in structure generation")
            return {"status": "error", "message": f"Critical error: {str(e)}"}

    def debug_molecule(self, molecule_identifier: str) -> Dict[str, Any]:
        """Debug method to test building a specific molecule with all strategies"""
        self.logger.info(f"🔍 Debug building molecule: {molecule_identifier}")
        
        # Extract molecule info
        mol_info = {
            "identifier": molecule_identifier,
            "formula": None,
            "alternative_names": [molecule_identifier.lower()],
            "smiles": None,
            "estimated_count": "debug"
        }
        
        results = {
            "molecule": molecule_identifier,
            "strategies_tried": [],
            "success": False,
            "filepath": None,
            "errors": []
        }
        
        safe_name = re.sub(r'[^\w\-_]', '_', molecule_identifier.lower())
        
        # Try ASE
        if molecule_identifier in self.ase_molecules:
            try:
                filepath = self._try_ase(molecule_identifier, safe_name)
                results["strategies_tried"].append({"method": "ASE", "success": bool(filepath)})
                if filepath:
                    results["success"] = True
                    results["filepath"] = filepath
                    return results
            except Exception as e:
                results["errors"].append(f"ASE failed: {e}")
        
        # Try PubChem
        try:
            filepath = self._try_pubchem(molecule_identifier, safe_name)
            results["strategies_tried"].append({"method": "PubChem", "success": bool(filepath)})
            if filepath:
                results["success"] = True
                results["filepath"] = filepath
                return results
        except Exception as e:
            results["errors"].append(f"PubChem failed: {e}")
        
        # Try LLM SMILES generation + RDKit
        if HAS_RDKIT:
            try:
                smiles = self._get_smiles_from_llm(molecule_identifier)
                if smiles:
                    filepath = self._try_rdkit_smiles(smiles, safe_name)
                    results["strategies_tried"].append({
                        "method": "LLM_SMILES + RDKit", 
                        "success": bool(filepath),
                        "smiles": smiles
                    })
                    if filepath:
                        results["success"] = True
                        results["filepath"] = filepath
                        return results
                else:
                    results["strategies_tried"].append({"method": "LLM_SMILES", "success": False, "note": "No SMILES provided"})
            except Exception as e:
                results["errors"].append(f"LLM SMILES + RDKit failed: {e}")
        
        return results