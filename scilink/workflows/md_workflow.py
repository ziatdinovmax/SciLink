# scilink/workflows/lammps_workflow.py
import os
import re
import sys
import logging
import shutil
import json
from io import StringIO
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from ..auth import get_api_key, APIKeyNotFoundError
from ..agents.sim_agents.lammps_utils import VMDLAMMPSConverter
from ..agents.sim_agents.force_field_agent import ForceFieldAgent
from ..agents.sim_agents.lammps_agent import LAMMPSSimulationAgent
from ..agents.sim_agents.lammps_updater import LAMMPSUpdater

class LAMMPSWorkflow:
    """
    Orchestrates a complete LAMMPS molecular dynamics workflow.
    
    This workflow takes a PDB structure file and a research goal as input,
    generates LAMMPS data files using VMD, selects appropriate force field parameters,
    and creates a complete LAMMPS input script. It also supports error detection and 
    automatic correction with restart capabilities.
    """
    
    def __init__(self,
                 api_key: str = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 vmd_path: str = None,
                 output_dir: str = "lammps_workflow_output",
                 max_refinement_cycles: int = 10,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initializes the LAMMPS workflow and its constituent agents.
        
        Args:
            api_key (str, optional): API key for the LLM provider
                Defaults to auto-discovery from environment variables.
            model_name (str, optional): Model name to use
            base_url: Optional base URL for internal proxy
            vmd_path (str, optional): Path to VMD executable. Required for 
                PDB to LAMMPS data conversion.
            output_dir (str, optional): Directory to save all generated files.
            max_refinement_cycles (int, optional): Maximum number of refinement
                attempts for error correction.
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        # Auto-discover API keys
        if api_key is None:
            api_key = get_api_key('google')
            if not api_key:
                raise APIKeyNotFoundError('google')
                
        # Setup logging
        self.log_capture = StringIO()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(name)s: %(message)s',
            force=True,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.StreamHandler(self.log_capture)
            ]
        )
        self.logger = logging.getLogger(__name__)

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

        # Store configuration
        self.api_key = api_key
        self.vmd_path = vmd_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.max_refinement_cycles = max_refinement_cycles
        
        # Create output directories
        self.base_dir = output_dir
        self.ff_dir = os.path.join(self.base_dir, "force_field")
        self.sim_dir = os.path.join(self.base_dir, "simulation")
        os.makedirs(self.ff_dir, exist_ok=True)
        os.makedirs(self.sim_dir, exist_ok=True)
        
        # Initialize agents
        self.converter = VMDLAMMPSConverter(
            vmd_path=vmd_path,
            working_dir=self.base_dir
        ) if vmd_path else None
        
        self.ff_agent = ForceFieldAgent(
            api_key=api_key,
            working_dir=self.ff_dir,
        )
        
        self.lammps_agent = LAMMPSSimulationAgent(
            api_key=api_key,
            working_dir=self.sim_dir,
            model_name=model_name
        )
        
        self.lammps_updater = LAMMPSUpdater(
            api_key=api_key,
            model_name=model_name
        )
    
    def run_complete_workflow(self, 
                             pdb_file: str, 
                             research_goal: str,
                             box_dimensions: float = 40.0,
                             simulation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete LAMMPS workflow from PDB to final input script.
        
        Args:
            pdb_file: Path to input PDB structure file
            research_goal: Research objective in natural language
            box_dimensions: Size of simulation box in Angstroms
            simulation_params: Optional override parameters for the simulation
            
        Returns:
            Dictionary with workflow results
        """
        workflow_result = {
            "research_goal": research_goal,
            "input_structure": pdb_file,
            "steps_completed": [],
            "final_status": "started"
        }
        
        print(f"\n🚀 LAMMPS Workflow Starting")
        print(f"{'='*60}")
        print(f"📝 Goal: {research_goal}")
        print(f"📋 Input: {pdb_file}")
        print(f"📁 Output: {self.output_dir}/")
        print(f"{'='*60}")
        
        # Step 1: Structure Conversion
        print(f"\n🔄 WORKFLOW STEP 1: Convert PDB to LAMMPS Data Format")
        print(f"{'─'*50}")
        
        if self.converter is None:
            print(f"❌ VMD path not provided, cannot perform conversion")
            workflow_result["final_status"] = "failed_structure_conversion"
            return workflow_result
        
        structure_result = self._convert_structure(pdb_file, box_dimensions)
        workflow_result["structure_conversion"] = structure_result
        
        if structure_result["status"] != "success":
            print(f"❌ Structure conversion failed: {structure_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_structure_conversion"
            return workflow_result
            
        workflow_result["steps_completed"].append("structure_conversion")
        data_file = structure_result["data_file"]
        print(f"✅ LAMMPS data file generated: {os.path.basename(data_file)}")
        
        # Step 2: Force Field Selection and Parameters
        print(f"\n🧪 WORKFLOW STEP 2: Force Field Selection and Parameter Generation")
        print(f"{'─'*50}")
        
        ff_result = self._generate_force_field_params(pdb_file, data_file, research_goal)
        workflow_result["force_field_generation"] = ff_result
        
        if ff_result["status"] != "success":
            print(f"❌ Force field generation failed: {ff_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_force_field_generation"
            return workflow_result
            
        workflow_result["steps_completed"].append("force_field_generation")
        param_files = ff_result["param_files"]
        print(f"✅ Force field parameters generated: {', '.join(list(param_files.keys()))}")
        
        # Step 3: LAMMPS Simulation Script Generation
        print(f"\n⚙️ WORKFLOW STEP 3: LAMMPS Script Generation")
        print(f"{'─'*50}")
        
        script_result = self._generate_lammps_script(
            data_file=data_file, 
            research_goal=research_goal, 
            param_files=param_files,
            simulation_params=simulation_params
        )
        workflow_result["script_generation"] = script_result
        
        if script_result["status"] != "success":
            print(f"❌ LAMMPS script generation failed: {script_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_script_generation"
            return workflow_result
            
        workflow_result["steps_completed"].append("script_generation")
        script_path = script_result["script_path"]
        print(f"✅ LAMMPS script generated: {os.path.basename(script_path)}")
        print(f"📋 Simulation type: {script_result.get('summary', 'N/A')}")
        
        # Final output
        workflow_result["final_status"] = "success"
        workflow_result["output_directory"] = self.output_dir
        
        # Create final files manifest
        final_manifest = self._create_final_files_manifest(workflow_result)
        workflow_result["final_manifest"] = final_manifest
        
        # Save complete log
        self._save_workflow_log()
        
        # Final summary
        self._print_final_summary(workflow_result)
        
        return workflow_result
    
    def refine_from_errors(self, 
                          research_goal: str,
                          error_log_path: str,
                          cycle: int = 1) -> Dict[str, Any]:
        """
        Refine the LAMMPS script based on error logs from a previous run.
        
        Args:
            research_goal: Original research goal
            error_log_path: Path to LAMMPS log file with errors
            cycle: Current refinement cycle number
        
        Returns:
            Dictionary with refinement results
        """
        print(f"\n🔄 Refinement Cycle {cycle}/{self.max_refinement_cycles}")
        print(f"{'─'*50}")
        
        script_path = os.path.join(self.sim_dir, "run.lammps")
        ff_path = os.path.join(self.ff_dir, "ff_params.lammps")
        data_path = os.path.join(self.sim_dir, "system.data")
        
        if not os.path.exists(script_path):
            print(f"❌ Cannot find script to refine at {script_path}")
            return {
                "status": "error",
                "message": f"Script not found: {script_path}",
                "cycle": cycle
            }
            
        if not os.path.exists(error_log_path):
            print(f"❌ Error log not found at {error_log_path}")
            return {
                "status": "error",
                "message": f"Error log not found: {error_log_path}",
                "cycle": cycle
            }
            
        # Make backup of current script
        backup_path = f"{script_path}.bak{cycle}"
        shutil.copy2(script_path, backup_path)
        print(f"📁 Script backup created: {os.path.basename(backup_path)}")
        
        # Perform refinement
        print(f"🔍 Analyzing errors and refining script...")
        corrected_script, analysis = self.lammps_updater.refine_inputs(
            input_path=script_path,
            research_goal=research_goal,
            ff_path=ff_path,
            data_path=data_path,
            lammps_log=error_log_path
        )
        
        # Save corrected script and analysis
        with open(script_path, 'w') as f:
            f.write(corrected_script)
            
        analysis_path = os.path.join(self.sim_dir, f"error_analysis_{cycle}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        issue_count = len(analysis.get("issues", []))
        print(f"✅ Script refined to address {issue_count} issue(s)")
        
        # Check if restart is being used
        using_restart = analysis.get("should_restart", False)
        if using_restart:
            restart_file = os.path.basename(analysis.get("restart_file", "unknown"))
            print(f"🔄 Using restart file: {restart_file}")
        
        return {
            "status": "success",
            "script_path": script_path,
            "analysis_path": analysis_path,
            "issues_fixed": issue_count,
            "using_restart": using_restart,
            "restart_file": analysis.get("restart_file", None) if using_restart else None,
            "cycle": cycle
        }
    
    def iterative_refinement(self, 
                           research_goal: str,
                           run_lammps_command: str,
                           max_cycles: int = None) -> Dict[str, Any]:
        """
        Iteratively refine LAMMPS script by running, detecting errors, and fixing.
        """
        if max_cycles is None:
            max_cycles = self.max_refinement_cycles
            
        import subprocess
        import time
        
        script_path = os.path.join(self.sim_dir, "run.lammps")
        script_filename = os.path.basename(script_path)  # Just "run.lammps"
        log_path = os.path.join(self.sim_dir, "log.lammps")
        
        results = {
            "status": "started",
            "cycles": [],
            "final_cycle": 0,
            "success": False
        }
        
        for cycle in range(1, max_cycles + 1):
            # Update final_cycle for this iteration
            results["final_cycle"] = cycle
            
            print(f"\n🔄 REFINEMENT CYCLE {cycle}/{max_cycles}")
            print(f"{'─'*50}")
            
            # Make backup of current script
            backup_path = f"{script_path}.bak{cycle}"
            shutil.copy2(script_path, backup_path)
            print(f"📁 Script backup created: {os.path.basename(backup_path)}")
            
            # If log file exists from previous run, save a backup
            if os.path.exists(log_path):
                log_backup = f"{log_path}.bak{cycle}"
                shutil.copy2(log_path, log_backup)
                print(f"📄 Log backup created: {os.path.basename(log_backup)}")
            
            # Run LAMMPS with proper path handling
            print(f"▶️ Running LAMMPS simulation...")
            full_command = f"{run_lammps_command} {script_filename}"
            print(f"   $ {full_command} (in {self.sim_dir})")
            
            try:
                process = subprocess.run(
                    full_command, 
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.sim_dir
                )
                
                # Write stdout and stderr to files
                stdout_path = os.path.join(self.sim_dir, f"stdout_{cycle}.txt")
                stderr_path = os.path.join(self.sim_dir, f"stderr_{cycle}.txt")
                
                with open(stdout_path, "w") as f:
                    f.write(process.stdout)
                with open(stderr_path, "w") as f:
                    f.write(process.stderr)
                
                # Check for success
                run_successful = process.returncode == 0
                
                # Look for LAMMPS errors in the log file (primary source of error info)
                lammps_errors = []
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                        # Extract LAMMPS errors from log file
                        error_matches = re.findall(r"ERROR:.*", log_content)
                        if error_matches:
                            lammps_errors.extend(error_matches)
                
                # If we didn't find errors in the log, check stdout as fallback
                if not lammps_errors:
                    error_matches = re.findall(r"ERROR:.*", process.stdout)
                    if error_matches:
                        lammps_errors.extend(error_matches)
                        # Append stdout errors to log file if we found errors there but not in the log
                        if os.path.exists(log_path):
                            with open(log_path, 'a') as f:
                                f.write("\n=== ERRORS FROM STDOUT ===\n")
                                for line in process.stdout.splitlines():
                                    if "ERROR" in line:
                                        f.write(f"{line}\n")
                
                # Check for system errors in stderr
                system_errors = []
                if "ERROR" in process.stderr:
                    system_errors = [line.strip() for line in process.stderr.splitlines() if "ERROR" in line]
                    # If there are system errors and no LAMMPS log was created, create one
                    if system_errors and not os.path.exists(log_path):
                        with open(log_path, "w") as f:
                            f.write("=== SYSTEM ERRORS ===\n")
                            f.write(process.stderr)

                # Check for misc errors in stdout
                misc_errors = []
                if "ERROR" in process.stdout:
                    misc_errors = [line.strip() for line in process.stdout.splitlines() if "ERROR" in line]
                    # If there are misc errors and no LAMMPS log was created, create one
                    if misc_errors and not os.path.exists(log_path):
                        with open(log_path, "w") as f:
                            f.write("=== OTHER ERRORS ===\n")
                            f.write(process.stdout)

                # Log errors to console
                if lammps_errors:
                    print(f"⚠️ LAMMPS reported errors:")
                    for error in lammps_errors[:3]:  # Show first few errors
                        print(f"   {error.strip()}")
                    if len(lammps_errors) > 3:
                        print(f"   ...and {len(lammps_errors) - 3} more errors")
                
                if system_errors:
                    print(f"⚠️ System reported errors:")
                    for error in system_errors[:3]:  # Show first few errors
                        print(f"   {error.strip()}")
                
                # Determine if simulation was successful
                if run_successful and not lammps_errors and not system_errors:
                    print(f"✅ Simulation completed successfully!")
                    results["status"] = "success"
                    results["success"] = True
                    break
                else:
                    print(f"⚠️ Simulation encountered issues (return code: {process.returncode})")
                
                # Wait for any asynchronous file operations to complete
                time.sleep(1)
                
                # Refine the script
                refine_result = self.refine_from_errors(
                    research_goal=research_goal,
                    error_log_path=log_path,
                    cycle=cycle
                )
                
                results["cycles"].append({
                    "cycle": cycle,
                    "return_code": process.returncode,
                    "lammps_errors": lammps_errors,
                    "system_errors": system_errors,
                    "refinement": refine_result
                })
                
                if refine_result["status"] != "success":
                    print(f"❌ Failed to refine script: {refine_result.get('message', 'Unknown error')}")
                    results["status"] = "failed_refinement"
                    break
                    
            except Exception as e:
                print(f"❌ Error running simulation: {e}")
                results["cycles"].append({
                    "cycle": cycle,
                    "error": str(e)
                })
                results["status"] = "error"
                break
        
        # Final summary
        if results.get("success", False):
            print(f"\n✅ Simulation successfully completed after {results['final_cycle']} refinement cycles")
        else:
            print(f"\n⚠️ Failed to achieve successful simulation after {results['final_cycle']} refinement cycles")
                
        return results
 
    def _convert_structure(self, pdb_file: str, box_dimensions: float) -> Dict[str, Any]:
        """Convert PDB to LAMMPS data format using VMD."""
        try:
            # Add box dimensions to PDB
            pdb_with_box = self.converter.add_box_to_pdb(
                input_pdb=pdb_file,
                box_dimensions=box_dimensions
            )
            
            # Convert to LAMMPS data format
            options = {
                "autobonds": True,
                "retypebonds": True,
                "guessangles": True,
                "guess_dihedrals": True,
                "guess_impropers": False,
                "style": "full",
                "atom_style": "full",
                "center_system": True
            }
            
            data_file = self.converter.convert(
                pdb_file=pdb_with_box,
                output_file="system.data",
                box_dimensions=box_dimensions,
                options=options
            )
            
            return {
                "status": "success",
                "data_file": data_file,
                "pdb_with_box": pdb_with_box,
                "box_dimensions": box_dimensions
            }
        except Exception as e:
            self.logger.error(f"Structure conversion failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Structure conversion failed: {e}"
            }
    
    def _generate_force_field_params(self, pdb_file: str, data_file: str, research_goal: str) -> Dict[str, Any]:
        """Select force field and generate parameters."""
        try:
            # Select the best force field
            ff_selection = self.ff_agent.select_force_field(
                pdb_file=pdb_file,
                research_goal=research_goal
            )
            
            # Acquire parameters
            ff_params = self.ff_agent.acquire_parameters(
                selection_info=ff_selection,
                data_file=data_file
            )
            
            # Generate LAMMPS parameter files
            param_files = self.ff_agent.generate_lammps_parameters(
                parameter_info=ff_params,
                data_file=data_file
            )
            
            return {
                "status": "success",
                "force_field_selection": ff_selection,
                "param_files": param_files,
                "parameter_info": ff_params
            }
        except Exception as e:
            self.logger.error(f"Force field generation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Force field generation failed: {e}"
            }
    
    def _generate_lammps_script(self, 
                             data_file: str, 
                             research_goal: str, 
                             param_files: Dict[str, str],
                             simulation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate LAMMPS simulation script."""
        try:
            # Default parameters
            params = {}
            if simulation_params:
                params.update(simulation_params)
                
            # Generate simulation script
            sim_result = self.lammps_agent.generate_simulation(
                data_file=data_file,
                research_goal=research_goal,
                force_field_files=param_files,
                **params
            )
            
            # Extract relevant information
            sim_params = sim_result.get("simulation_parameters", {})
            summary = f"{sim_params.get('ensemble', 'N/A')} simulation of {sim_params.get('simulation_time', 0.0)} ns"
            
            return {
                "status": "success",
                "script_path": sim_result.get("script_path"),
                "readme_path": sim_result.get("readme_path"),
                "simulation_parameters": sim_params,
                "summary": summary
            }
        except Exception as e:
            self.logger.error(f"LAMMPS script generation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"LAMMPS script generation failed: {e}"
            }
    
    def _create_final_files_manifest(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a manifest of final output files."""
        manifest = {
            "workflow_status": workflow_result["final_status"],
            "research_goal": workflow_result["research_goal"],
            "output_directory": self.output_dir,
            "final_files": {},
            "ready_to_run": False
        }
        
        try:
            # Add structure files
            if "structure_conversion" in workflow_result and workflow_result["structure_conversion"]["status"] == "success":
                data_file = workflow_result["structure_conversion"]["data_file"]
                manifest["final_files"]["data_file"] = os.path.basename(data_file)
                
            # Add force field files
            if "force_field_generation" in workflow_result and workflow_result["force_field_generation"]["status"] == "success":
                param_files = workflow_result["force_field_generation"]["param_files"]
                manifest["final_files"]["force_field_files"] = {
                    k: os.path.basename(v) for k, v in param_files.items()
                }
                
            # Add script files
            if "script_generation" in workflow_result and workflow_result["script_generation"]["status"] == "success":
                script_path = workflow_result["script_generation"]["script_path"]
                readme_path = workflow_result["script_generation"].get("readme_path")
                manifest["final_files"]["script"] = os.path.basename(script_path)
                if readme_path:
                    manifest["final_files"]["readme"] = os.path.basename(readme_path)
                    
            # Check if we have all essential components
            if (
                "data_file" in manifest["final_files"] and
                "force_field_files" in manifest["final_files"] and
                "script" in manifest["final_files"]
            ):
                manifest["ready_to_run"] = True
                
            # Save the manifest
            manifest_path = os.path.join(self.output_dir, "final_files_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to create manifest: {e}", exc_info=True)
            
        return manifest
    
    def _print_final_summary(self, workflow_result: Dict[str, Any]) -> None:
        """Print a clean summary of the workflow results."""
        print(f"\n🎉 LAMMPS Workflow Complete!")
        print(f"{'='*60}")
        
        status = workflow_result.get('final_status')
        steps = workflow_result.get('steps_completed', [])
        
        print(f"📋 Status: {status}")
        print(f"✅ Steps: {' → '.join(steps)}")
        print(f"📁 Output: {self.output_dir}/")
        
        # Structure conversion details
        if "structure_conversion" in workflow_result and workflow_result["structure_conversion"]["status"] == "success":
            data_file = os.path.basename(workflow_result["structure_conversion"]["data_file"])
            dimensions = workflow_result["structure_conversion"].get("box_dimensions", "N/A")
            print(f"🔄 Structure: {data_file} (Box: {dimensions}Å)")
            
        # Force field details
        if "force_field_generation" in workflow_result and workflow_result["force_field_generation"]["status"] == "success":
            ff_selection = workflow_result["force_field_generation"].get("force_field_selection", {})
            ff_name = ff_selection.get("selected_force_field", "Unknown")
            print(f"🧪 Force Field: {ff_name}")
            
        # Script details
        if "script_generation" in workflow_result and workflow_result["script_generation"]["status"] == "success":
            summary = workflow_result["script_generation"].get("summary", "N/A")
            print(f"⚙️ LAMMPS: {summary}")
            
        # Ready to run files
        if "final_manifest" in workflow_result and workflow_result["final_manifest"].get("ready_to_run", False):
            print(f"\n📄 Ready to Run:")
            final_files = workflow_result["final_manifest"]["final_files"]
            print(f"    • {final_files.get('data_file', 'system.data')}")
            print(f"    • {final_files.get('script', 'run.lammps')}")
            if "force_field_files" in final_files:
                for name, file in final_files["force_field_files"].items():
                    print(f"    • {file}")
                    
        print(f"{'='*60}")
    
    def _save_workflow_log(self) -> str:
        """Save captured logs to a file."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = os.path.join(self.output_dir, "workflow_log.txt")
            
            with open(log_path, 'w') as f:
                f.write(f"LAMMPS Workflow Complete Log\n")
                f.write(f"{'='*30}\n\n")
                f.write(log_content)
                
            print(f"📝 Complete workflow log saved: {log_path}")
            return log_path
        except Exception as e:
            print(f"Warning: Could not save workflow log: {e}")
            return ""

