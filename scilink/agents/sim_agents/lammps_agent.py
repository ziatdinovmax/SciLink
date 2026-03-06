import os
import re
import shutil
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from ase import io
from ase.io.lammpsdata import read_lammps_data

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .instruct import LAMMPS_INPUT_GENERATION_TEMPLATE
from ._deprecation import normalize_params


class LAMMPSSimulationAgent:
    def __init__(self, 
                 working_dir: str, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the LAMMPS simulation agent.
        
        Args:
            working_dir: Directory for output files
            api_key: API key for the LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(exist_ok=True, parents=True)
        
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
            source="LAMMPSSimulationAgent"
        )
        
        # Initialize model using wrapper structure
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy")
            
            self.logger.info(f"Using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"Using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None

    def _integrate_force_field_files(self, script_text: str, force_field_files: Dict[str, str]) -> str:
        """
        Integrate force field parameter files into the LAMMPS script.
        
        Handles cases where files may already be in the working directory.
        """
        if not force_field_files:
            return script_text
    
        lines = script_text.split('\n')
    
        # Find the position to insert force field parameters (after read_data)
        insert_pos = 0
        for i, line in enumerate(lines):
            if "read_data" in line:
                insert_pos = i + 1
                break
    
        # Fallback: find first non-comment, non-empty line
        if insert_pos == 0:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    insert_pos = i
                    break
    
        insertion = ["", "# Include force field parameters"]
        included_files = []
    
        def safe_copy_file(source: str, dest_dir: Path, preferred_name: Optional[str] = None) -> Optional[str]:
            """
            Safely copy a file to destination directory.
            Returns the filename to use in the include statement, or None if failed.
            """
            if not source or not os.path.exists(source):
                self.logger.warning(f"Source file does not exist: {source}")
                return None
            
            source_path = Path(source).resolve()
            
            # Determine destination filename
            if preferred_name:
                dest_name = preferred_name if preferred_name.endswith('.lammps') else f"{preferred_name}.lammps"
            else:
                dest_name = source_path.name
            
            dest_path = (dest_dir / dest_name).resolve()
            
            # Check if source and destination are the same file
            try:
                if source_path.samefile(dest_path):
                    self.logger.info(f"File already in working directory: {dest_name}")
                    return dest_name
            except (OSError, FileNotFoundError):
                # samefile can fail if dest doesn't exist yet, which is fine
                pass
            
            # Check if destination already exists with same content
            if dest_path.exists():
                try:
                    if source_path.read_bytes() == dest_path.read_bytes():
                        self.logger.info(f"Identical file already exists: {dest_name}")
                        return dest_name
                except Exception:
                    pass
                
                # Destination exists but is different - create unique name
                base = dest_path.stem
                suffix = dest_path.suffix or '.lammps'
                counter = 1
                while dest_path.exists():
                    dest_name = f"{base}_{counter}{suffix}"
                    dest_path = dest_dir / dest_name
                    counter += 1
            
            # Copy the file
            try:
                shutil.copy2(str(source_path), str(dest_path))
                self.logger.info(f"Copied {source_path.name} to {dest_path}")
                return dest_name
            except shutil.SameFileError:
                # This shouldn't happen after our checks, but handle it anyway
                self.logger.info(f"File already in place: {dest_name}")
                return dest_path.name
            except Exception as e:
                self.logger.error(f"Failed to copy {source}: {e}")
                return None
    
        # Process main force field file
        if "main" in force_field_files:
            main_source = force_field_files["main"]
            filename = safe_copy_file(main_source, self.working_dir)
            if filename:
                insertion.append(f"include {filename}")
                included_files.append(filename)
    
        # Process additional force field files
        additional = force_field_files.get("additional", {})
        if isinstance(additional, dict):
            for name, path in additional.items():
                if path:
                    filename = safe_copy_file(path, self.working_dir, preferred_name=name)
                    if filename and filename not in included_files:
                        insertion.append(f"include {filename}")
                        included_files.append(filename)
        elif isinstance(additional, str):
            # Handle case where additional is a single file path string
            filename = safe_copy_file(additional, self.working_dir)
            if filename and filename not in included_files:
                insertion.append(f"include {filename}")
                included_files.append(filename)
    
        # Only add insertion if we actually have files to include
        if len(insertion) > 2:  # More than just the header comments
            insertion.append("")
            updated_lines = lines[:insert_pos] + insertion + lines[insert_pos:]
            return '\n'.join(updated_lines)
        else:
            self.logger.warning("No force field files were successfully integrated")
            return script_text

    def generate_simulation(self,
                              data_file: str,
                              research_goal: str,
                              system_description: Optional[str] = None,
                              temperature: float = 300.0,
                              pressure: float = 1.0,
                              force_field_files: Optional[Dict[str, str]] = None,
                              **kwargs) -> Dict[str, Any]:
            """
            Generate LAMMPS simulation(s) based on a research goal.
    
            Args:
                data_file: Path to LAMMPS data file
                research_goal: Research objective in natural language
                system_description: Description of the molecular system (optional)
                temperature: Default temperature in K
                pressure: Default pressure in atm
                force_field_files: Dictionary with paths to force field parameter files
                **kwargs: Additional parameters
    
            Returns:
                Dictionary with generated simulation info
            """
            # Copy the data file to the working directory
            local_data_file = self._copy_data_file(data_file)
    
            # Analyze the system
            system_info = self.analyze_system(data_file)
    
            # If system description is not provided, generate one
            if not system_description:
                system_description = self._generate_system_description(system_info)
    
            # Determine simulation parameters
            simulation_params = self._determine_simulation_parameters(
                research_goal=research_goal,
                system_info=system_info,
                temperature=temperature,
                pressure=pressure,
                **kwargs
            )
    
            # Generate LAMMPS script
            script_text = self._generate_script(
                data_filename=os.path.basename(local_data_file),
                research_goal=research_goal,
                system_description=system_description,
                system_info=system_info,
                **simulation_params
            )
    
            # Add force field parameters
            if force_field_files:
                self.logger.info("Integrating provided force field files")
                script_text = self._integrate_force_field_files(script_text, force_field_files)
            else:
                self.logger.info("Generating basic force field parameters")
                script_text = self._ensure_force_field_parameters(script_text, system_info)
                        # Save the script
            script_path = self.working_dir / "run.lammps"
            with open(script_path, 'w') as f:
                f.write(script_text)
    
            # Create README
            readme_path = self._generate_readme(
                research_goal=research_goal,
                system_description=system_description,
                system_info=system_info,
                simulation_params=simulation_params,
                script_path=str(script_path)
            )
    
            return {
                "script_path": str(script_path),
                "readme_path": readme_path,
                "data_path": str(local_data_file),
                "system_info": system_info,
                "simulation_parameters": simulation_params
            }

    def generate_staged_simulation(self,
                                   data_file: str,
                                   research_goal: str,
                                   system_description: Optional[str] = None,
                                   force_field_files: Optional[Dict[str, str]] = None,
                                   **kwargs) -> Dict[str, Any]:
        """
        Generate simulation broken into checkpointed stages for quality monitoring.
        
        This generates separate scripts for:
        - Minimization
        - Equilibration (may be multiple phases)
        - Production
        
        Each stage writes restart files and can be quality-checked independently.
        
        Args:
            data_file: Path to LAMMPS data file
            research_goal: Research objective
            system_description: System description (optional)
            force_field_files: Force field parameter files (optional)
            **kwargs: Additional simulation parameters
            
        Returns:
            Dictionary with:
                - All fields from generate_simulation()
                - "staged_scripts": Dict mapping stage name to script path
                - "stages": List of stage names in execution order
        """
        self.logger.info("Generating staged simulation with checkpoints")
        
        # First generate full simulation (reuse existing logic)
        full_sim_info = self.generate_simulation(
            data_file=data_file,
            research_goal=research_goal,
            system_description=system_description,
            force_field_files=force_field_files,
            **kwargs
        )
        
        # Read the generated full script
        full_script = Path(full_sim_info["script_path"]).read_text()
        
        # Split into stages using LLM
        self.logger.info("Splitting simulation into stages for quality checks")
        stages = self._split_into_stages(
            script_content=full_script,
            simulation_params=full_sim_info["simulation_parameters"],
            system_info=full_sim_info["system_info"],
            data_filename=os.path.basename(full_sim_info["data_path"])
        )
        
        # Write individual stage scripts
        stage_scripts = {}
        for stage_name, stage_content in stages.items():
            stage_path = self.working_dir / f"run_{stage_name}.lammps"
            with open(stage_path, 'w') as f:
                f.write(stage_content)
            stage_scripts[stage_name] = str(stage_path)
            self.logger.info(f"  ✓ Generated stage: {stage_name} -> {stage_path.name}")
        
        # Add to return dict
        full_sim_info["staged_scripts"] = stage_scripts
        full_sim_info["stages"] = list(stages.keys())
        full_sim_info["is_staged"] = True
        
        return full_sim_info


    # ============================================================================
    # HELPER METHODS FOR LLM CALLS
    # ============================================================================
    
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from LLM.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Parsed JSON response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            raise ValueError(f"Could not parse JSON from LLM response: {e}")
        except Exception as e:
            self.logger.error(f"Error generating JSON: {e}")
            raise
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text response from LLM.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise
    
   
    def _determine_simulation_parameters(self, 
                                       research_goal: str, 
                                       system_info: Dict[str, Any],
                                       temperature: float = 300.0,
                                       pressure: float = 1.0,
                                       **kwargs) -> Dict[str, Any]:
        """Determine simulation parameters based on research goal."""
        self.logger.info(f"Determining simulation parameters for research goal: {research_goal}")
        
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("element_counts", {}).items()])
        
        prompt = f"""
Analyze this research goal for a molecular dynamics simulation and recommend parameters.

RESEARCH GOAL: "{research_goal}"

SYSTEM INFORMATION:
- Elements: {elements_str}
- Total atoms: {system_info.get('atom_count', 0)}
- Contains water: {'Yes' if system_info.get('has_water', False) else 'No'}
- Contains ions: {'Yes' if system_info.get('has_ions', False) else 'No'}
- Contains organic: {'Yes' if system_info.get('has_organic', False) else 'No'}

DETERMINE:
1. Does this require multiple simulations (e.g., umbrella sampling windows, temperature series)?
2. If yes, how many simulations and what varies between them?
3. What specific technique is needed?
4. What ensemble, temperature, pressure, timestep?
5. What simulation time per run?
6. What specific LAMMPS commands/fixes are needed?
7. What outputs are needed for analysis?

Respond with JSON:
{{
    "requires_multiple_simulations": true/false,
    "simulation_technique": "umbrella_sampling" or "steered_md" or "standard_md" etc.,
    "number_of_simulations": 1 or 25 etc.,
    "variable_parameter": "distance" or "temperature" or null,
    "variable_values": [2.5, 3.0, 3.5, ...] or null,
    
    "ensemble": "NPT",
    "temperature": 300.0,
    "pressure": 1.0,
    "timestep": 2.0,
    "simulation_time": 2.0,
    "equilibration_time": 0.2,
    "production_time": 1.8,
    
    "special_fixes": [{{"command": "fix spring/couple", "description": "...", "parameters": {{}}}}],
    "required_outputs": ["distance_trajectory", "energy", "restart"],
    "analysis_method": "WHAM" or "direct" etc.,
    "methodology_description": "Brief explanation"
}}
"""
        
        try:
            params = self._generate_json(prompt)  # ✅ Use helper method
            
            # Add defaults
            params.setdefault("requires_multiple_simulations", False)
            params.setdefault("number_of_simulations", 1)
            params.setdefault("ensemble", "NPT")
            params.setdefault("temperature", temperature)
            params.setdefault("pressure", pressure)
            params.setdefault("timestep", 2.0)
            params.setdefault("simulation_time", 2.0)
            
            # Override with explicit kwargs
            for key, value in kwargs.items():
                params[key] = value
            
            self.logger.info(f"Simulation type: {params.get('simulation_technique', 'standard_md')}")
            if params.get("requires_multiple_simulations"):
                self.logger.info(f"Multiple simulations detected: {params.get('number_of_simulations')} runs")
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error determining simulation parameters: {e}")
            return {
                "requires_multiple_simulations": False,
                "number_of_simulations": 1,
                "simulation_technique": "standard_md",
                "ensemble": "NPT",
                "temperature": temperature,
                "pressure": pressure,
                "timestep": 2.0,
                "simulation_time": 2.0,
                "equilibration_time": 0.5,
                "production_time": 1.5,
                "required_outputs": ["energy", "trajectory"]
            }
        
    def _split_into_stages(self,
                          script_content: str,
                          simulation_params: Dict[str, Any],
                          system_info: Dict[str, Any],
                          data_filename: str) -> Dict[str, str]:
        """Split LAMMPS script into stages at natural checkpoints."""
        
        prompt = f"""
    You are a LAMMPS expert. Split this simulation into stages for quality checking.
    
    FULL LAMMPS SCRIPT:
    {script_content}
    
    DATA FILE: {data_filename}
    
    SIMULATION INFO:
    - Ensemble: {simulation_params.get('ensemble', 'NPT')}
    - Temperature: {simulation_params.get('temperature', 300)} K
    - Pressure: {simulation_params.get('pressure', 1.0)} atm
    - Simulation time: {simulation_params.get('simulation_time', 2)} ns
    
    TASK:
    Split this script into 2-4 stages at natural breakpoints:
    1. **equilibration_npt** - NPT equilibration to reach target T and P
    2. **production** - Production run for data collection
    
    Each stage MUST be a complete, standalone, runnable LAMMPS script that includes:
    1. units real
    2. atom_style full
    3. boundary p p p
    4. read_data {data_filename} (for first stage) OR read_restart (for later stages)
    5. All necessary pair_style, bond_style, etc. OR include ff_params.lammps
    6. Proper fix commands for the ensemble
    7. Output commands (thermo, dump, etc.)
    8. run command
    9. write_restart command at the end
    
    CRITICAL REQUIREMENTS:
    - Each script must start with: units real
    - Do NOT use undefined variables like ${{t}} or ${{run_id}} - use actual values
    - Temperature should be {simulation_params.get('temperature', 300)} K
    - Use actual numbers, not placeholders
    - Each stage must be independently executable
    
    Return JSON where keys are stage names and values are complete LAMMPS scripts.
    Example format:
    {{
        "equilibration_npt": "units real\\natom_style full\\n...",
        "production": "units real\\natom_style full\\n..."
    }}
    
    Return ONLY valid JSON with complete scripts. No markdown, no explanations.
    """
        
        try:
            stages = self._generate_json(prompt)
            
            if not stages or not isinstance(stages, dict):
                raise ValueError("Invalid stage splitting result")
            
            # Validate and fix each stage script
            validated_stages = {}
            for stage_name, stage_script in stages.items():
                if not isinstance(stage_script, str):
                    self.logger.warning(f"Stage {stage_name} is not a string, skipping")
                    continue
                
                # Clean and validate the script
                cleaned_script = self._validate_and_fix_stage_script(
                    stage_script, 
                    stage_name, 
                    data_filename, 
                    simulation_params
                )
                
                if cleaned_script:
                    validated_stages[stage_name] = cleaned_script
                else:
                    self.logger.warning(f"Stage {stage_name} failed validation, skipping")
            
            if not validated_stages:
                raise ValueError("No valid stages produced")
            
            self.logger.info(f"Split into {len(validated_stages)} stages: {list(validated_stages.keys())}")
            return validated_stages
            
        except Exception as e:
            self.logger.error(f"Error splitting into stages: {e}")
            self.logger.warning("Falling back to single-stage simulation")
            # Return the original script as a single stage, but validate it first
            validated = self._validate_and_fix_stage_script(
                script_content, "production", data_filename, simulation_params
            )
            return {"production": validated or script_content}


    def _validate_and_fix_stage_script(self,
                                        script: str,
                                        stage_name: str,
                                        data_filename: str,
                                        params: Dict[str, Any]) -> Optional[str]:
        """
        Validate and fix a LAMMPS stage script.
        
        Returns the fixed script, or None if unfixable.
        """
        if not script or not script.strip():
            return None
        
        lines = script.strip().split('\n')
        fixed_lines = []
        
        # Track what we've seen
        has_units = False
        has_atom_style = False
        has_boundary = False
        has_read_data = False
        has_read_restart = False
        
        # Get actual values for substitution
        temperature = params.get('temperature', 300.0)
        pressure = params.get('pressure', 1.0)
        timestep = params.get('timestep', 2.0)
        
        for line in lines:
            original_line = line
            stripped = line.strip()
            
            # Skip empty lines and comments (keep them)
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Check for and fix broken template lines
            # Pattern: starts with { or contains unresolved ${...} or {...}
            if stripped.startswith('{') and not stripped.startswith('{#'):
                # This is a broken template line like "{run_id}: Temperature..."
                self.logger.warning(f"Removing broken template line: {stripped[:50]}...")
                continue
            
            # Fix unresolved variables
            line = line.replace('${t}', str(temperature))
            line = line.replace('${T}', str(temperature))
            line = line.replace('${temp}', str(temperature))
            line = line.replace('${temperature}', str(temperature))
            line = line.replace('{t}', str(temperature))
            line = line.replace('{temperature}', str(temperature))
            
            line = line.replace('${p}', str(pressure))
            line = line.replace('${P}', str(pressure))
            line = line.replace('${press}', str(pressure))
            line = line.replace('${pressure}', str(pressure))
            line = line.replace('{p}', str(pressure))
            line = line.replace('{pressure}', str(pressure))
            
            line = line.replace('${run_id}', '1')
            line = line.replace('{run_id}', '1')
            line = line.replace('${dt}', str(timestep))
            line = line.replace('{dt}', str(timestep))
            
            line = line.replace('{data_file}', data_filename)
            line = line.replace('${data_file}', data_filename)
            line = line.replace('<your_data_filename>', data_filename)
            line = line.replace('your_data_file.data', data_filename)
            line = line.replace('system.data', data_filename)
            
            # Check for unmatched quotes (common LLM error)
            stripped_fixed = line.strip()
            if stripped_fixed.count('"') % 2 != 0:
                # Odd number of quotes - try to fix
                if stripped_fixed.endswith('"') and not stripped_fixed.startswith('print'):
                    # Looks like a broken print statement
                    self.logger.warning(f"Fixing broken quote line: {stripped_fixed[:50]}...")
                    line = f'print "{stripped_fixed.rstrip(chr(34))}"'
                elif stripped_fixed.startswith('"'):
                    # Missing closing quote
                    line = line.rstrip() + '"'
            
            # Track what commands we see
            first_word = stripped_fixed.split()[0] if stripped_fixed.split() else ''
            if first_word == 'units':
                has_units = True
            elif first_word == 'atom_style':
                has_atom_style = True
            elif first_word == 'boundary':
                has_boundary = True
            elif first_word == 'read_data':
                has_read_data = True
            elif first_word == 'read_restart':
                has_read_restart = True
            
            fixed_lines.append(line)
        
        # Prepend missing initialization if needed
        init_lines = []
        
        if not has_units:
            init_lines.append("units real")
        if not has_atom_style:
            init_lines.append("atom_style full")
        if not has_boundary:
            init_lines.append("boundary p p p")
        
        # Add read_data if this is the first stage and it's missing
        if not has_read_data and not has_read_restart:
            if stage_name in ['minimization', 'equilibration', 'equilibration_nvt', 'equilibration_npt'] or 'equil' in stage_name.lower():
                init_lines.append("")
                init_lines.append(f"read_data {data_filename}")
        
        if init_lines:
            # Find where to insert (after any initial comments)
            insert_pos = 0
            for i, line in enumerate(fixed_lines):
                if line.strip() and not line.strip().startswith('#'):
                    insert_pos = i
                    break
            
            # Add header comment
            init_lines.insert(0, f"# Stage: {stage_name}")
            init_lines.insert(1, f"# Auto-fixed by LAMMPSSimulationAgent")
            init_lines.append("")
            
            fixed_lines = fixed_lines[:insert_pos] + init_lines + fixed_lines[insert_pos:]
        
        result = '\n'.join(fixed_lines)
        
        # Final validation - check for any remaining broken patterns
        if re.search(r'\{[a-z_]+\}', result, re.IGNORECASE):
            remaining = re.findall(r'\{[a-z_]+\}', result, re.IGNORECASE)
            self.logger.warning(f"Script still has unresolved templates after fixing: {remaining}")
        
        return result

    def analyze_system(self, data_file: str) -> Dict[str, Any]:
        """Analyze a LAMMPS data file using ASE to identify its components."""
        self.logger.info(f"Analyzing system from {data_file}")
        try:
            atoms = read_lammps_data(data_file, style="full", units="real")
            
            element_counts = {}
            for symbol in atoms.get_chemical_symbols():
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
                
            has_water = ('O' in element_counts and 'H' in element_counts and 
                        element_counts.get('H', 0) >= 2 * element_counts.get('O', 0))
            has_ions = any(x in element_counts for x in ['Na', 'Cl', 'K', 'Ca', 'Mg'])
            has_organic = 'C' in element_counts
            
            bond_types, angle_types = self._extract_bond_angle_types(data_file)
            
            system_info = {
                "atom_count": len(atoms),
                "elements": list(element_counts.keys()),
                "element_counts": element_counts,
                "box_dimensions": atoms.get_cell().diagonal().tolist(),
                "has_water": has_water,
                "has_ions": has_ions,
                "has_organic": has_organic,
                "bond_types": bond_types,
                "angle_types": angle_types
            }
            
            self.logger.info(f"System analysis complete: {system_info}")
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing data file: {e}")
            return {
                "atom_count": 0,
                "elements": [],
                "element_counts": {},
                "has_water": False,
                "has_ions": False,
                "has_organic": False,
                "bond_types": 0,
                "angle_types": 0
            }
    
    def _extract_bond_angle_types(self, data_file: str) -> tuple:
        """Extract bond and angle type information from a LAMMPS data file."""
        bond_types = 0
        angle_types = 0
        
        try:
            with open(data_file, 'r') as f:
                content = f.read()
                
                match = re.search(r'(\d+)\s+bond\s+types', content)
                if match:
                    bond_types = int(match.group(1))
                    
                match = re.search(r'(\d+)\s+angle\s+types', content)
                if match:
                    angle_types = int(match.group(1))
                    
        except Exception as e:
            self.logger.error(f"Error extracting bond/angle types: {e}")
            
        return bond_types, angle_types
    
    def _copy_data_file(self, source_path: str) -> Path:
        """Copy the data file to the working directory."""
        dest_filename = "system.data"
        dest_path = self.working_dir / dest_filename
        
        shutil.copy2(source_path, dest_path)
        self.logger.info(f"Copied data file from {source_path} to {dest_path}")
        
        return dest_path
    
    def _generate_system_description(self, system_info: Dict[str, Any]) -> str:
        """Generate a system description based on analysis."""
        elements = system_info.get("elements", [])
        atom_count = system_info.get("atom_count", 0)
        
        description_parts = []
        
        if system_info.get("has_water", False):
            description_parts.append("water")
        
        if system_info.get("has_ions", False):
            ions = [e for e in elements if e in ["Na", "K", "Cl", "Ca", "Mg"]]
            if ions:
                description_parts.append("+".join(ions) + " ions")
            else:
                description_parts.append("ions")
        
        if system_info.get("has_organic", False) and "C" in elements:
            description_parts.append("organic molecules")
        
        if not description_parts:
            description_parts.append("molecular system")
        
        description = " with ".join(description_parts)
        return f"{description} ({atom_count} atoms)"
    
    def _determine_simulation_parameters(self, 
                                       research_goal: str, 
                                       system_info: Dict[str, Any],
                                       temperature: float = 300.0,
                                       pressure: float = 1.0,
                                       **kwargs) -> Dict[str, Any]:
        """
        Determine simulation parameters based on the research goal and system info.
        Detects if multiple simulations (windows) are needed.
        """
        self.logger.info(f"Determining simulation parameters for research goal: {research_goal}")
        
        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("element_counts", {}).items()])
        
        prompt = f"""
Analyze this research goal for a molecular dynamics simulation and recommend parameters.

RESEARCH GOAL: "{research_goal}"

SYSTEM INFORMATION:
- Elements: {elements_str}
- Total atoms: {system_info.get('atom_count', 0)}
- Contains water: {'Yes' if system_info.get('has_water', False) else 'No'}
- Contains ions: {'Yes' if system_info.get('has_ions', False) else 'No'}
- Contains organic: {'Yes' if system_info.get('has_organic', False) else 'No'}

DETERMINE:
1. Does this require multiple simulations (e.g., umbrella sampling windows, temperature series)?
2. If yes, how many simulations and what varies between them?
3. What specific technique is needed (umbrella sampling, steered MD, T-REMD, etc.)?
4. What ensemble, temperature, pressure, timestep?
5. What simulation time per run?
6. What specific LAMMPS commands/fixes are needed?
7. What outputs are needed for analysis?

Respond with JSON:
{{
    "requires_multiple_simulations": true/false,
    "simulation_technique": "umbrella_sampling" or "steered_md" or "standard_md" etc.,
    "number_of_simulations": 1 or 25 etc.,
    "variable_parameter": "distance" or "temperature" or null,
    "variable_values": [2.5, 3.0, 3.5, ...] or null,
    
    "ensemble": "NPT",
    "temperature": 300.0,
    "pressure": 1.0,
    "timestep": 2.0,
    "simulation_time": 2.0,
    "equilibration_time": 0.2,
    "production_time": 1.8,
    
    "special_fixes": [
        {{"command": "fix spring/couple", "description": "Harmonic restraint", "parameters": {{"force_constant": 40.0}}}}
    ],
    
    "required_outputs": ["distance_trajectory", "energy", "restart"],
    "analysis_method": "WHAM" or "direct" etc.,
    
    "methodology_description": "Brief explanation of how to achieve the research goal"
}}
"""
        
        try:
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            params = json.loads(response.text)
            
            # Add defaults
            params.setdefault("requires_multiple_simulations", False)
            params.setdefault("number_of_simulations", 1)
            params.setdefault("ensemble", "NPT")
            params.setdefault("temperature", temperature)
            params.setdefault("pressure", pressure)
            params.setdefault("timestep", 2.0)
            params.setdefault("simulation_time", 2.0)
            
            # Override with explicit kwargs if provided
            for key, value in kwargs.items():
                params[key] = value
            
            self.logger.info(f"Simulation type: {params.get('simulation_technique', 'standard_md')}")
            if params.get("requires_multiple_simulations"):
                self.logger.info(f"Multiple simulations detected: {params.get('number_of_simulations')} runs")
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error determining simulation parameters: {e}")
            return {
                "requires_multiple_simulations": False,
                "number_of_simulations": 1,
                "simulation_technique": "standard_md",
                "ensemble": "NPT",
                "temperature": temperature,
                "pressure": pressure,
                "timestep": 2.0,
                "simulation_time": 2.0,
                "equilibration_time": 0.5,
                "production_time": 1.5,
                "required_outputs": ["energy", "trajectory"]
            }
    
    def _parse_data_file_types(self, data_file: str) -> str:
        """
        Parse data file to extract type information for script generation.
        
        Returns a formatted string describing all types in the data file,
        suitable for including in LLM prompts.
        """
        type_info = []
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
            
            # Parse comment headers
            sections = {"# Pair Coeffs": "ATOM", "# Bond Coeffs": "BOND", 
                        "# Angle Coeffs": "ANGLE", "# Dihedral Coeffs": "DIHEDRAL"}
            current_section = None
            
            for line in lines:
                stripped = line.strip()
                
                if stripped in sections:
                    current_section = sections[stripped]
                    type_info.append(f"\n{current_section} TYPES:")
                    continue
                elif stripped == "#":
                    continue
                elif not stripped.startswith("#"):
                    current_section = None
                    continue
                
                if current_section and stripped.startswith("#"):
                    parts = stripped[1:].strip().split()
                    if len(parts) >= 2:
                        try:
                            type_id = int(parts[0])
                            name = " ".join(parts[1:])
                            type_info.append(f"  {current_section.lower()}_type {type_id} = {name}")
                        except ValueError:
                            pass
            
            # Parse header counts
            type_info.insert(0, "DATA FILE TOPOLOGY:")
            for line in lines[:20]:
                stripped = line.strip()
                parts = stripped.split()
                if len(parts) >= 2:
                    for keyword in ["atoms", "bonds", "angles", "dihedrals", "impropers",
                                   "atom types", "bond types", "angle types", "dihedral types"]:
                        if keyword in stripped.lower():
                            type_info.insert(1, f"  {stripped}")
                            break
            
            # Parse Masses with element names
            in_masses = False
            type_info.append("\nMASS-ELEMENT MAPPING:")
            for line in lines:
                stripped = line.strip()
                if stripped == "Masses":
                    in_masses = True
                    continue
                if in_masses:
                    if not stripped or stripped.startswith("Atoms"):
                        break
                    if stripped.startswith("#"):
                        continue
                    parts = stripped.split()
                    if len(parts) >= 2 and "#" in stripped:
                        try:
                            type_id = int(parts[0])
                            mass = float(parts[1])
                            comment = stripped.split("#")[1].strip()
                            type_info.append(f"  atom_type {type_id} = {comment} (mass {mass:.3f})")
                        except (ValueError, IndexError):
                            pass
        
        except Exception as e:
            type_info.append(f"  (Could not parse: {e})")
        
        return "\n".join(type_info)
    
    def _generate_script(self, **kwargs) -> str:
        """Generate LAMMPS script(s) based on simulation parameters."""
        system_info = kwargs.get("system_info", {})
        data_filename = kwargs.get("data_filename", "system.data")
        research_goal = kwargs.get("research_goal", "")
        system_description = kwargs.get("system_description", "")
    
        # Parse data file for correct type info
        data_file_path = self.working_dir / data_filename
        if data_file_path.exists():
            data_type_info = self._parse_data_file_types(str(data_file_path))
        else:
            data_type_info = "DATA FILE NOT FOUND - use placeholder type numbers"
    
        requires_multiple = kwargs.get("requires_multiple_simulations", False)
        num_sims = kwargs.get("number_of_simulations", 1)
        sim_technique = kwargs.get("simulation_technique", "standard_md")
    
        ensemble = kwargs.get("ensemble", "NPT")
        temperature = kwargs.get("temperature", 300.0)
        pressure = kwargs.get("pressure", 1.0)
        timestep = kwargs.get("timestep", 2.0)
        simulation_time = kwargs.get("simulation_time", 2.0)
    
        equil_time = kwargs.get("equilibration_time", simulation_time * 0.25)
        prod_time = kwargs.get("production_time", simulation_time * 0.75)
        equil_steps = int((equil_time * 1e6) / timestep)
        prod_steps = int((prod_time * 1e6) / timestep)
    
        if requires_multiple and num_sims > 1:
            script_prompt = self._build_multi_simulation_prompt(
                data_type_info=data_type_info,
                research_goal=research_goal,
                system_description=system_description,
                system_info=system_info,
                data_filename=data_filename,
                num_simulations=num_sims,
                technique=sim_technique,
                variable_param=kwargs.get("variable_parameter"),
                variable_values=kwargs.get("variable_values"),
                special_fixes=kwargs.get("special_fixes", []),
                methodology=kwargs.get("methodology_description", ""),
                temperature=temperature,
                pressure=pressure,
                ensemble=ensemble,
                timestep=timestep,
                simulation_time=simulation_time,
                equilibration_time=equil_time,
                production_time=prod_time,
                analysis_method=kwargs.get("analysis_method", "WHAM")
            )
        else:
            script_prompt = self._build_single_simulation_prompt(
                data_type_info=data_type_info,
                research_goal=research_goal,
                system_description=system_description,
                system_info=system_info,
                data_filename=data_filename,
                temperature=temperature,
                pressure=pressure,
                ensemble=ensemble,
                timestep=timestep,
                simulation_time=simulation_time,
                equilibration_time=equil_time,
                production_time=prod_time,
                equil_steps=equil_steps,
                prod_steps=prod_steps,
                properties_to_calculate=kwargs.get("properties_to_calculate", []),
                required_outputs=kwargs.get("required_outputs", []),
                output_commands=self._generate_output_commands(
                    kwargs.get("required_outputs", []),
                    kwargs.get("properties_to_calculate", []),
                    system_info
                )
            )
    
        response = self.model.generate_content(script_prompt)
        script_text = response.text
        script_text = self._clean_script(script_text)
    
        return script_text

    def _build_multi_simulation_prompt(self,
                                       research_goal: str,
                                       system_description: str,
                                       system_info: Dict[str, Any],
                                       data_filename: str,
                                       num_simulations: int,
                                       technique: str,
                                       variable_param: Optional[str],
                                       variable_values: Optional[List[float]],
                                       special_fixes: List[Dict[str, Any]],
                                       methodology: str,
                                       temperature: float,
                                       pressure: float,
                                       ensemble: str,
                                       timestep: float,
                                       simulation_time: float,
                                       equilibration_time: float,
                                       production_time: float,
                                       analysis_method: str = "WHAM",
                                       data_type_info: str = "") -> str:
        """Build prompt for multiple simulations (umbrella sampling, etc.)"""
        
        if variable_values:
            values_str = ", ".join([f"{v:.2f}" for v in variable_values[:5]]) + f"... ({len(variable_values)} total)"
        else:
            values_str = "To be determined"
        
        fixes_str = "\n".join([f"  - {fix.get('command', 'N/A')}: {fix.get('description', '')}" 
                              for fix in special_fixes]) if special_fixes else "  - Standard MD fixes"
        
        prompt = f"""
Generate a LAMMPS script that implements multiple related simulations to achieve this research goal:

RESEARCH GOAL: {research_goal}

APPROACH: {methodology}

MULTI-SIMULATION SETUP:
- Technique: {technique}
- Number of simulations: {num_simulations}
- Variable parameter: {variable_param}
- Values: {values_str}

SYSTEM:
- Data file: {data_filename}
- Atoms: {system_info.get('atom_count', 0)}
- Components: {"water, " if system_info.get('has_water') else ""}{"ions, " if system_info.get('has_ions') else ""}{"organic" if system_info.get('has_organic') else ""}

REQUIRED SPECIAL COMMANDS/FIXES:
{fixes_str}

SIMULATION PARAMETERS (per run):
- Temperature: {temperature} K
- Pressure: {pressure} atm
- Ensemble: {ensemble}
- Timestep: {timestep} fs
- Equilibration: {equilibration_time} ns
- Production: {production_time} ns

IMPLEMENTATION OPTIONS:
1. Generate a master script with LAMMPS variable loops to run all {num_simulations} simulations sequentially
2. Or generate a template that can be run {num_simulations} times with different parameters
3. Whichever is more appropriate for {technique}

CRITICAL REQUIREMENTS:
- For umbrella sampling: use "fix spring/couple" or "fix colvars" to apply biasing potential
- Save collective variable data to separate files: colvar_window_${{i}}.dat or similar
- Write restart files every 10000-50000 steps: restart.*.${{i}}.prod
- Include comments explaining the biasing/sampling methodology
- Ensure outputs are suitable for {analysis_method} analysis

OUTPUT:
Generate a complete, executable LAMMPS script (or set of instructions for running multiple scripts).
Include detailed comments explaining the multi-simulation setup.

Return ONLY the LAMMPS input content without markdown formatting.
"""
        
        return prompt
    
    def _build_single_simulation_prompt(self,
                                       research_goal: str,
                                       system_description: str,
                                       system_info: Dict[str, Any],
                                       data_filename: str,
                                       temperature: float,
                                       pressure: float,
                                       ensemble: str,
                                       timestep: float,
                                       simulation_time: float,
                                       equilibration_time: float,
                                       production_time: float,
                                       equil_steps: int,
                                       prod_steps: int,
                                       properties_to_calculate: List[str],
                                       required_outputs: List[str],
                                       output_commands: str,
                                       data_type_info: str = "") -> str:
        """Build prompt for standard single simulation."""
        
        element_info_str = "\n  - ".join([f"{e}: {c}" for e, c in 
                                         system_info.get("element_counts", {}).items()])
        
        return LAMMPS_INPUT_GENERATION_TEMPLATE.format(
            research_goal=research_goal,
            system_description=system_description,
            element_info_str=element_info_str,
            atom_count=system_info.get("atom_count", 0),
            box_dimensions=system_info.get("box_dimensions", [40, 40, 40]),
            bond_types=system_info.get("bond_types", 0),
            angle_types=system_info.get("angle_types", 0),
            data_type_info=data_type_info,
            has_water="Yes" if system_info.get("has_water") else "No",
            has_ions="Yes" if system_info.get("has_ions") else "No",
            has_organic="Yes" if system_info.get("has_organic") else "No",
            properties_to_calculate_str=", ".join(properties_to_calculate),
            required_outputs_str=", ".join(required_outputs),
            temperature=temperature,
            pressure=pressure,
            ensemble=ensemble,
            timestep=timestep,
            simulation_time=simulation_time,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
            data_filename=data_filename,
            output_commands=output_commands
        )

    def _generate_output_commands(self, 
                                required_outputs: List[str], 
                                properties_to_calculate: List[str], 
                                system_info: Dict[str, Any]) -> str:
        """Generate LAMMPS output command instructions."""
        instructions = []
        
        instructions.append("Include regular thermodynamic output (temperature, pressure, energy, etc.)")
        
        if "trajectory" in required_outputs:
            instructions.append("Output trajectory in DCD or XYZ format at appropriate intervals")
        
        if "density" in required_outputs or "density" in properties_to_calculate:
            instructions.append("Calculate and output system density")
        
        if "rdf" in required_outputs or any(p in properties_to_calculate for p in ["rdf", "radial distribution", "pair correlation"]):
            atom_pairs = []
            if system_info.get("has_water", False):
                atom_pairs.append("O-O")
            if system_info.get("has_ions", False):
                if "Na" in system_info.get("elements", []) and "Cl" in system_info.get("elements", []):
                    atom_pairs.append("Na-Cl")
                    atom_pairs.append("Na-O")
                    atom_pairs.append("Cl-O")
            
            if atom_pairs:
                instructions.append(f"Calculate radial distribution functions for atom pairs: {', '.join(atom_pairs)}")
            else:
                instructions.append("Calculate radial distribution functions for relevant atom pairs")
        
        if "msd" in required_outputs or "diffusion" in required_outputs or any(p in properties_to_calculate for p in ["diffusion", "mobility", "msd"]):
            if system_info.get("has_ions", False):
                instructions.append("Calculate mean squared displacement (MSD) separately for each ion type")
            else:
                instructions.append("Calculate mean squared displacement (MSD) for appropriate atom types")
        
        if "viscosity" in required_outputs or "viscosity" in properties_to_calculate:
            instructions.append("Calculate viscosity using Green-Kubo formalism with pressure tensor autocorrelation")
        
        if "dielectric" in required_outputs or any(p in properties_to_calculate for p in ["dielectric", "polarization"]):
            instructions.append("Track system dipole moment for dielectric constant calculation")
        
        return "\n".join(instructions)
    
    def _clean_script(self, script_text: str) -> str:
        """Remove markdown formatting and other unwanted elements from the script."""
        script_text = re.sub(r'```(?:lammps|bash)?', '', script_text)
        script_text = script_text.replace('```', '')
        script_text = script_text.strip()
        
        if not script_text.startswith(('#', 'units', 'echo', 'log', 'atom_style')):
            script_text = f"# LAMMPS script for: {script_text.split(os.linesep)[0]}\n\n" + script_text
        
        self.logger.info("Cleaned script output of markdown formatting")
        return script_text
    
    def _ensure_force_field_parameters(self, script: str, system_info: Dict[str, Any]) -> str:
        """Ensure the script has all necessary force field parameters."""
        lines = script.split('\n')
        
        has_bond_style = any("bond_style" in line.lower() for line in lines)
        has_bond_coeffs = any("bond_coeff" in line.lower() for line in lines)
        has_angle_style = any("angle_style" in line.lower() for line in lines)
        has_angle_coeffs = any("angle_coeff" in line.lower() for line in lines)
        
        if not (has_bond_style and has_bond_coeffs and has_angle_style and has_angle_coeffs):
            self.logger.warning("Adding missing force field parameters to the script")
            
            insert_idx = 0
            for i, line in enumerate(lines):
                if "read_data" in line:
                    insert_idx = i + 1
                    break
            
            ff_params = self._generate_force_field_parameters(system_info)
            
            lines.insert(insert_idx, "\n# Force field parameters added by LAMMPSSimulationAgent")
            lines.insert(insert_idx + 1, ff_params)
            lines.insert(insert_idx + 2, "")
            
        return '\n'.join(lines)
    
    def _generate_force_field_parameters(self, system_info: Dict[str, Any]) -> str:
        """Generate force field parameters based on system analysis."""
        params = []
        
        params.append("# Basic force field styles")
        params.append("pair_style lj/cut/coul/long 10.0")
        params.append("bond_style harmonic")
        params.append("angle_style harmonic")
        params.append("special_bonds lj/coul 0.0 0.0 0.5")
        params.append("kspace_style pppm 0.0001")
        params.append("")
        
        bond_types = system_info.get("bond_types", 0)
        if bond_types > 0:
            params.append("# Bond coefficients")
            for i in range(1, bond_types + 1):
                params.append(f"bond_coeff {i} 450.0 1.0  # Generic bond")
            params.append("")
        
        angle_types = system_info.get("angle_types", 0)
        if angle_types > 0:
            params.append("# Angle coefficients")
            for i in range(1, angle_types + 1):
                params.append(f"angle_coeff {i} 55.0 109.47  # Generic angle")
            params.append("")
        
        params.append("# Pair coefficients")
        element_types = {}
        type_idx = 1
        
        for element in system_info.get("elements", []):
            element_types[element] = type_idx
            type_idx += 1
            
        if element_types:
            for el, idx in element_types.items():
                if el == "O":
                    params.append(f"pair_coeff {idx} {idx} 0.1553 3.166  # Oxygen")
                elif el == "H":
                    params.append(f"pair_coeff {idx} {idx} 0.0 0.0  # Hydrogen")
                elif el == "Na":
                    params.append(f"pair_coeff {idx} {idx} 0.0115 2.275  # Sodium")
                elif el == "Cl":
                    params.append(f"pair_coeff {idx} {idx} 0.1 4.417  # Chloride")
                else:
                    params.append(f"pair_coeff {idx} {idx} 0.1 3.0  # Generic {el}")
        else:
            params.append("pair_coeff * * 0.0 0.0")
            params.append("# For water O-O")
            params.append("pair_coeff 1 1 0.1553 3.166")
            
        return "\n".join(params)
    
    def _generate_readme(self, **kwargs) -> str:
        """Generate a README file with analysis instructions based on the research goal."""
        research_goal = kwargs.get("research_goal", "")
        system_description = kwargs.get("system_description", "")
        system_info = kwargs.get("system_info", {})
        simulation_params = kwargs.get("simulation_params", {})
        
        readme_path = self.working_dir / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write(f"# Molecular Dynamics Simulation: {system_description}\n\n")
            f.write(f"## Research Goal\n{research_goal}\n\n")
            
            f.write("## System Composition\n")
            for element, count in system_info.get("element_counts", {}).items():
                f.write(f"- {element}: {count} atoms\n")
            f.write(f"- Total atoms: {system_info.get('atom_count', 'Unknown')}\n\n")
            
            f.write("## Simulation Parameters\n")
            properties = simulation_params.get("properties_to_calculate", [])
            if properties:
                f.write(f"- Properties to calculate: {', '.join(properties)}\n")
            f.write(f"- Temperature: {simulation_params.get('temperature', 300.0)} K\n")
            f.write(f"- Pressure: {simulation_params.get('pressure', 1.0)} atm\n")
            f.write(f"- Ensemble: {simulation_params.get('ensemble', 'NPT')}\n")
            f.write(f"- Timestep: {simulation_params.get('timestep', 2.0)} fs\n")
            f.write(f"- Total simulation time: {simulation_params.get('simulation_time', 2.0)} ns\n\n")
            
            f.write("## How to Run\n")
            f.write("```bash\n")
            f.write(f"cd {self.working_dir}\n")
            f.write(f"lmp -in {os.path.basename(kwargs.get('script_path', 'run.lammps'))}\n")
            f.write("```\n\n")
            
            f.write("## Analysis Instructions\n")
            
            analysis_steps = self._generate_analysis_instructions(
                research_goal=research_goal,
                properties=properties,
                required_outputs=simulation_params.get("required_outputs", []),
                system_info=system_info
            )
            
            for i, step in enumerate(analysis_steps, 1):
                f.write(f"{i}. {step}\n")
        
        return str(readme_path)
    
    def _generate_analysis_instructions(self, 
                                      research_goal: str, 
                                      properties: List[str], 
                                      required_outputs: List[str],
                                      system_info: Dict[str, Any]) -> List[str]:
        """Generate step-by-step analysis instructions based on the research goal."""
        instructions = [
            "Verify system equilibration by checking energy, temperature, and pressure over time",
            "Analyze trajectory files using visualization tools like VMD or OVITO"
        ]
        
        if "density" in properties or "density" in required_outputs:
            instructions.append("Calculate average density from the production phase")
            instructions.append("Compare density with experimental values")
        
        if any(p in properties + required_outputs for p in ["diffusion", "msd", "mobility"]):
            instructions.append("Plot mean squared displacement (MSD) vs time")
            instructions.append("Calculate diffusion coefficients using the Einstein relation: D = MSD/(6t)")
            if system_info.get("has_ions", False):
                instructions.append("Compare diffusion coefficients of different ion types")
        
        if any(p in properties + required_outputs for p in ["rdf", "structure", "radial"]):
            instructions.append("Plot radial distribution functions to analyze molecular structure")
            instructions.append("Identify coordination shells from RDF peaks")
            if system_info.get("has_water", False) and system_info.get("has_ions", False):
                instructions.append("Calculate hydration numbers of ions by integrating the first peak of ion-water RDFs")
        
        if any(p in properties + required_outputs for p in ["viscosity"]):
            instructions.append("Calculate viscosity from the Green-Kubo integral of pressure tensor autocorrelation")
            instructions.append("Compare calculated viscosity with experimental values")
        
        if any(p in properties + required_outputs for p in ["dielectric", "polarization"]):
            instructions.append("Calculate dielectric constant from dipole moment fluctuations")
        
        lower_goal = research_goal.lower()
        
        if "compare" in lower_goal or "different" in lower_goal:
            instructions.append("Compare results across different simulation conditions or systems")
        
        if "temperature" in lower_goal and "effect" in lower_goal:
            instructions.append("Plot the calculated properties as a function of temperature to identify trends")
        
        if "pressure" in lower_goal and "effect" in lower_goal:
            instructions.append("Plot the calculated properties as a function of pressure to identify trends")
        
        if "concentration" in lower_goal:
            instructions.append("Analyze how properties change with concentration")
        
        return instructions
