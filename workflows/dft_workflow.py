import os
import logging
from typing import Optional, Dict, Any

# Import all the agents
from sim_agents.structure_agent import StructureGenerator
from sim_agents.val_agent import StructureValidatorAgent
from sim_agents.vasp_agent import VaspInputAgent
from sim_agents.val_agent import IncarValidatorAgent


class DFTWorkflow:
    """
    Complete DFT inputs preparation workflow:
    User request → Structure → Validation → VASP inputs → Literature validation → Improvements -> Final VASP inputs
    """
    
    def __init__(self, google_api_key: str, futurehouse_api_key: str = None,
                 generator_model: str = "gemini-2.5-pro-preview-05-06", 
                 validator_model: str = "gemini-2.5-pro-preview-05-06",
                 output_dir: str = "vasp_workflow_output",
                 max_refinement_cycles: int = 2):
        """
        Initialize the complete VASP workflow.
        
        Args:
            google_api_key: Google API key for Gemini models
            futurehouse_api_key: FutureHouse API key for literature validation (optional)
            generator_model: Model for structure generation
            validator_model: Model for structure validation  
            output_dir: Directory to save all outputs
            max_refinement_cycles: Max structure refinement cycles
        """
        self.google_api_key = google_api_key
        self.futurehouse_api_key = futurehouse_api_key
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.structure_generator = StructureGenerator(
            api_key=google_api_key,
            model_name=generator_model,
            generated_script_dir=output_dir
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
        
        self.logger.info(f"Starting complete VASP workflow for: {user_request}")
        
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
        
        self.logger.info(f"Complete VASP workflow finished successfully: {self.output_dir}")
        return workflow_result
    
    def _generate_and_validate_structure(self, user_request: str) -> Dict[str, Any]:
        """Generate and validate atomic structure."""
        
        self.logger.info("Step 1: Structure generation and validation")
        
        for cycle in range(self.max_refinement_cycles + 1):
            self.logger.info(f"Structure cycle {cycle + 1}/{self.max_refinement_cycles + 1}")
            
            # Generate structure
            gen_result = self.structure_generator.generate_script(
                original_user_request=user_request,
                attempt_number_overall=cycle + 1,
                is_refinement_from_validation=(cycle > 0)
            )
            
            if gen_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Structure generation failed: {gen_result.get('message')}",
                    "cycle": cycle + 1
                }
            
            structure_file = gen_result["output_file"]
            script_content = gen_result["final_script_content"]
            
            # Validate structure
            val_result = self.structure_validator.validate_structure_and_script(
                structure_file_path=structure_file,
                generating_script_content=script_content,
                original_request=user_request
            )
            
            if val_result["status"] == "success":
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
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
                    "final_structure_path": structure_file,
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
        
        summary = f"VASP Workflow Summary\n{'='*20}\n"
        summary += f"Request: {workflow_result['user_request']}\n"
        summary += f"Status: {workflow_result['final_status']}\n"
        summary += f"Steps completed: {', '.join(workflow_result['steps_completed'])}\n"
        summary += f"Output directory: {workflow_result.get('output_directory', 'N/A')}\n\n"
        
        # Structure generation summary
        if "structure_generation" in workflow_result:
            struct_result = workflow_result["structure_generation"]
            if struct_result["status"] == "success":
                summary += f"✓ Structure: {struct_result['final_structure_path']}\n"
                summary += f"  Refinement cycles: {struct_result['cycles_used']}\n"
        
        # VASP generation summary  
        if "vasp_generation" in workflow_result:
            vasp_result = workflow_result["vasp_generation"]
            if vasp_result["status"] == "success":
                summary += f"✓ VASP files: INCAR, KPOINTS generated\n"
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
        
        return summary
