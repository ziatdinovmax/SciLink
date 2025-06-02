import os
import sys
import logging
from io import StringIO
import json
from typing import Optional, Dict, Any

# Import all the agents
from sim_agents.structure_agent import StructureGenerator
from sim_agents.val_agent import StructureValidatorAgent
from sim_agents.vasp_agent import VaspInputAgent
from sim_agents.val_agent import IncarValidatorAgent


class DFTWorkflow:
    """
    Workflow for generating DFT input structures: 
    User request → Structure → Validation → VASP inputs → Literature validation → Improvements -> Final VASP inputs
    """
    
    def __init__(self, google_api_key: str, futurehouse_api_key: str = None,
                 generator_model: str = "gemini-2.5-pro-preview-05-06", 
                 validator_model: str = "gemini-2.5-pro-preview-05-06",
                 output_dir: str = "vasp_workflow_output",
                 max_refinement_cycles: int = 2,
                 mp_api_key: str = None):
        """
        Initialize the complete DFT workflow.
        
        Args:
            google_api_key: Google API key for Gemini models
            futurehouse_api_key: FutureHouse API key for literature validation (optional)
            generator_model: Model for structure generation
            validator_model: Model for structure validation  
            output_dir: Directory to save all outputs
            max_refinement_cycles: Max structure refinement cycles
        """

        # Setup logging
        self.log_capture = StringIO()
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(name)s: %(message)s', 
            force=True, 
            handlers=[
                logging.StreamHandler(sys.stdout),  # Display in console
                logging.StreamHandler(self.log_capture)  # Capture to string
            ]
        )

        self.logger = logging.getLogger(__name__)

        self.google_api_key = google_api_key
        self.futurehouse_api_key = futurehouse_api_key
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles
        
        
        # Initialize agents
        self.structure_generator = StructureGenerator(
            api_key=google_api_key,
            model_name=generator_model,
            generated_script_dir=output_dir,
            mp_api_key=mp_api_key
        )
        
        self.structure_validator = StructureValidatorAgent(
            api_key=google_api_key,
            model_name=validator_model
        )
        
        self.vasp_agent = VaspInputAgent(
            api_key=google_api_key,
            model_name=generator_model
        )
        
        if futurehouse_api_key:
            self.incar_validator = IncarValidatorAgent(
                api_key=google_api_key,
                futurehouse_api_key=futurehouse_api_key
            )
        else:
            self.incar_validator = None
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_complete_workflow(self, user_request: str) -> Dict[str, Any]:
        """
        Run the complete workflow from user request to final VASP inputs.
        
        Args:
            user_request: User's structure/calculation request
            
        Returns:
            Dictionary with all workflow results
        """
        workflow_result = {
            "user_request": user_request,
            "steps_completed": [],
            "final_status": "started"
        }
        
        self.logger.info(f"Starting complete DFT workflow for: {user_request}")
        
        # Step 1: Structure Generation and Validation
        structure_result = self._generate_and_validate_structure(user_request)
        workflow_result["structure_generation"] = structure_result
        
        if structure_result["status"] != "success":
            workflow_result["final_status"] = "failed_structure_generation"
            return workflow_result
        
        workflow_result["steps_completed"].append("structure_generation")
        structure_path = structure_result["final_structure_path"]
        
        # Step 2: VASP Input Generation
        vasp_result = self._generate_vasp_inputs(structure_path, user_request)
        workflow_result["vasp_generation"] = vasp_result
        
        if vasp_result["status"] != "success":
            workflow_result["final_status"] = "failed_vasp_generation"
            return workflow_result
            
        workflow_result["steps_completed"].append("vasp_generation")
        
        # Step 3: Literature Validation and Improvements (optional)
        if self.incar_validator:
            improvement_result = self._validate_and_improve_incar(
                vasp_result, structure_path, user_request
            )
            workflow_result["incar_improvement"] = improvement_result
            workflow_result["steps_completed"].append("incar_improvement")
        else:
            self.logger.info("Skipping literature validation - no FutureHouse API key provided")
            workflow_result["incar_improvement"] = {"status": "skipped", "message": "No FutureHouse API key"}
        
        workflow_result["final_status"] = "success"
        workflow_result["output_directory"] = self.output_dir
        
        # Create final files manifest
        final_manifest = self._create_final_files_manifest(workflow_result)
        workflow_result["final_manifest"] = final_manifest
        
        # Save complete log
        self._save_workflow_log()
        
        self.logger.info(f"Complete DFT workflow finished successfully: {self.output_dir}")
        return workflow_result
    
    def _generate_and_validate_structure(self, user_request: str) -> Dict[str, Any]:
        """Generate and validate atomic structure."""
        
        self.logger.info("Step 1: Structure generation and validation")
        
        previous_script_content = None
        validator_feedback = None
        
        for cycle in range(self.max_refinement_cycles + 1):
            self.logger.info(f"Structure cycle {cycle + 1}/{self.max_refinement_cycles + 1}")
            
            # Generate structure

            gen_result = self.structure_generator.generate_script(
                original_user_request=user_request + ". Save the structure in POSCAR format.",
                attempt_number_overall=cycle + 1,
                is_refinement_from_validation=(cycle > 0),
                previous_script_content=previous_script_content if cycle > 0 else None,
                validator_feedback=validator_feedback if cycle > 0 else None
            )
            
            if gen_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Structure generation failed: {gen_result.get('message')}",
                    "cycle": cycle + 1
                }
            
            structure_file = gen_result["output_file"] # absolute path
            script_content = gen_result["final_script_content"]
            previous_script_content = script_content  # Store for next cycle

            full_structure_path = structure_file
            
            # Validate structure
            val_result = self.structure_validator.validate_structure_and_script(
                structure_file_path=full_structure_path,
                generating_script_content=script_content,
                original_request=user_request
            )
            
            validator_feedback = val_result  # Store for next cycle
            
            if val_result["status"] == "success":
                return {
                    "status": "success",
                    "final_structure_path": full_structure_path,
                    "final_script_path": gen_result["final_script_path"],
                    "cycles_used": cycle + 1,
                    "validation_result": val_result
                }
            elif cycle < self.max_refinement_cycles:
                self.logger.warning(f"Validation issues found, refining... {val_result.get('all_identified_issues')}")
                continue
            else:
                # Max cycles reached, proceed anyway
                self.logger.warning("Max refinement cycles reached, proceeding with current structure")
                return {
                    "status": "success",
                    "final_structure_path": full_structure_path,
                    "final_script_path": gen_result["final_script_path"], 
                    "cycles_used": cycle + 1,
                    "validation_result": val_result,
                    "warning": "Structure may have validation issues"
                }
        
        return {"status": "error", "message": "Structure generation loop failed"}
    
    def _generate_vasp_inputs(self, structure_path: str, user_request: str) -> Dict[str, Any]:
        """Generate VASP INCAR and KPOINTS files."""
        
        self.logger.info("Step 2: VASP input generation")
        
        # Generate initial VASP inputs
        vasp_result = self.vasp_agent.generate_vasp_inputs(
            poscar_path=structure_path,
            original_request=user_request
        )
        
        if vasp_result["status"] != "success":
            return vasp_result
        
        # Save initial VASP files
        saved_files = self.vasp_agent.save_inputs(vasp_result, self.output_dir)
        vasp_result["saved_files"] = saved_files
        
        return vasp_result
    
    def _validate_and_improve_incar(self, vasp_result: Dict[str, Any], 
                                   structure_path: str, user_request: str) -> Dict[str, Any]:
        """Validate INCAR against literature and apply improvements."""
        
        self.logger.info("Step 3: Literature validation and INCAR improvement")
        
        # Validate against literature
        validation_result = self.incar_validator.validate_and_improve_incar(
            incar_content=vasp_result["incar"],
            system_description=user_request
        )
        
        if validation_result["status"] != "success":
            return validation_result
        
        # Check if improvements are needed
        if validation_result["validation_status"] == "needs_adjustment":
            self.logger.info("Literature suggests improvements, applying them...")
            
            # Apply improvements
            improvement_result = self.vasp_agent.apply_improvements(
                original_incar=vasp_result["incar"],
                validation_result=validation_result,
                poscar_path=structure_path,
                original_request=user_request,
                output_dir=self.output_dir
            )
            
            validation_result["improvement_application"] = improvement_result
            
            if improvement_result["status"] == "success":
                self.logger.info(f"Successfully applied {improvement_result['adjustments_count']} improvements")
            
        else:
            self.logger.info("No improvements needed - INCAR parameters look good!")
            validation_result["improvement_application"] = {
                "status": "no_changes",
                "message": "No improvements needed"
            }
        
        # Save validation report
        self.incar_validator.save_validation_report(validation_result, self.output_dir)
        
        return validation_result
    
    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the workflow results."""
        
        summary = f"DFT Workflow Summary\n{'='*20}\n"
        summary += f"Request: {workflow_result['user_request']}\n"
        summary += f"Status: {workflow_result['final_status']}\n"
        summary += f"Steps completed: {', '.join(workflow_result['steps_completed'])}\n"
        summary += f"Output directory: {workflow_result.get('output_directory', 'N/A')}\n\n"
        
        # Structure generation summary
        if "structure_generation" in workflow_result:
            struct_result = workflow_result["structure_generation"]
            if struct_result["status"] == "success":
                struct_file = os.path.basename(struct_result['final_structure_path'])
                summary += f"✓ Final Structure: {struct_file}\n"
                summary += f"  Refinement cycles: {struct_result['cycles_used']}\n"
                summary += f"  Location: {workflow_result.get('output_directory', '.')}/\n"
        
        # VASP generation summary  
        if "vasp_generation" in workflow_result:
            vasp_result = workflow_result["vasp_generation"]
            if vasp_result["status"] == "success":
                summary += f"✓ VASP Input Files:\n"
                
                # Determine which INCAR to highlight
                if ("incar_improvement" in workflow_result and 
                    workflow_result["incar_improvement"].get("improvement_application", {}).get("status") == "success"):
                    summary += f"  - INCAR_improved (literature-validated) ⭐\n"
                    summary += f"  - INCAR (original)\n"
                else:
                    summary += f"  - INCAR\n"
                
                summary += f"  - KPOINTS\n"
                summary += f"  Calculation: {vasp_result['summary']}\n"
        
        # Improvement summary
        if "incar_improvement" in workflow_result:
            imp_result = workflow_result["incar_improvement"]
            if imp_result["status"] == "success":
                if imp_result["validation_status"] == "needs_adjustment":
                    adj_count = len(imp_result.get("suggested_adjustments", []))
                    summary += f"✓ Literature improvements: {adj_count} adjustments applied\n"
                else:
                    summary += f"✓ Literature validation: No improvements needed\n"
        
        # Add final files section
        if "final_manifest" in workflow_result:
            manifest = workflow_result["final_manifest"]
            if manifest.get("ready_for_vasp"):
                summary += f"\n📋 FINAL FILES FOR VASP:\n"
                final_files = manifest["final_files"]
                summary += f"  Structure: {final_files.get('structure', 'N/A')}\n"
                summary += f"  INCAR: {final_files.get('incar', 'N/A')}\n"
                summary += f"  KPOINTS: {final_files.get('kpoints', 'N/A')}\n"
                summary += f"  Directory: {manifest['output_directory']}/\n"
        
        return summary

    def _save_workflow_log(self) -> str:
        """Save all captured logs to a file."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = os.path.join(self.output_dir, "workflow_log.txt")
            
            with open(log_path, 'w') as f:
                f.write(f"DFT Workflow Complete Log\n")
                f.write(f"{'='*30}\n\n")
                f.write(log_content)
            
            print(f"📝 Complete workflow log saved: {log_path}")
            return log_path
            
        except Exception as e:
            print(f"Warning: Could not save workflow log: {e}")
            return ""
        
    def _create_final_files_manifest(self, workflow_result: Dict[str, Any]) -> Dict[str, str]:
        """Create a JSON manifest of final files."""
        
        manifest = {
            "workflow_status": workflow_result["final_status"],
            "user_request": workflow_result["user_request"],
            "output_directory": self.output_dir,
            "final_files": {},
            "ready_for_vasp": False
        }
        
        # Determine final structure file
        if ("structure_generation" in workflow_result and 
            workflow_result["structure_generation"]["status"] == "success"):
            structure_path = workflow_result["structure_generation"]["final_structure_path"]
            manifest["final_files"]["structure"] = os.path.basename(structure_path)
        
        # Determine final VASP input files
        if ("vasp_generation" in workflow_result and 
            workflow_result["vasp_generation"]["status"] == "success"):
            
            # Check if improved INCAR exists
            if ("incar_improvement" in workflow_result and 
                workflow_result["incar_improvement"].get("improvement_application", {}).get("status") == "success"):
                manifest["final_files"]["incar"] = "INCAR_improved"
                manifest["literature_validated"] = True
            else:
                manifest["final_files"]["incar"] = "INCAR"
                manifest["literature_validated"] = False
                
            manifest["final_files"]["kpoints"] = "KPOINTS"
            
            # Mark as ready if we have all required files
            if all(key in manifest["final_files"] for key in ["structure", "incar", "kpoints"]):
                manifest["ready_for_vasp"] = True
        
        # Save manifest
        try:
            manifest_path = os.path.join(self.output_dir, "final_files_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            self.logger.info(f"Final files manifest saved: {manifest_path}")
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
            
        return manifest
