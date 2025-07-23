import os
import sys
import json
import logging
from pathlib import Path
from io import StringIO
from typing import Optional, Dict, Any

from ..auth import get_api_key, APIKeyNotFoundError
from ..agents.sim_agents.structure_agent import StructureGenerator
from ..agents.sim_agents.val_agent import StructureValidatorAgent, IncarValidatorAgent
from ..agents.sim_agents.vasp_agent import VaspInputAgent
from ..agents.sim_agents.vasp_error_updater_agent import VaspErrorUpdaterAgent

from ..agents.sim_agents.atomate2_agent import Atomate2InputAgent
from pymatgen.io.vasp.inputs import Poscar


class DFTWorkflow:
    """
    Complete DFT workflow: User request → Structure → Validation → VASP inputs
    """
    
    def __init__(self, 
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 mp_api_key: str = None,
                 generator_model: str = "gemini-2.5-pro-preview-06-05",
                 validator_model: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "dft_workflow_output",
                 max_refinement_cycles: int = 4,
                 script_timeout: int = 180):
        
        # Auto-discover API keys
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if futurehouse_api_key is None:
            futurehouse_api_key = get_api_key('futurehouse')
            # Optional for DFT workflow
        
        if mp_api_key is None:
            mp_api_key = get_api_key('materials_project')
            # Optional - will warn in agents if needed
        
        # Setup logging with better formatting
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

        self.google_api_key = google_api_key
        self.futurehouse_api_key = futurehouse_api_key
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles
        
        # Initialize agents with simplified parameters
        self.structure_generator = StructureGenerator(
            api_key=google_api_key,
            model_name=generator_model,
            executor_timeout=script_timeout,
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

        # error_log based INCAR/KPOINTS refinement
        self.vasp_error_updater = VaspErrorUpdaterAgent(
            api_key=google_api_key,
            model_name=generator_model
        )
                     
        # Atomate2 agent for full VASP input deck
        self.atomate2_agent = Atomate2InputAgent(
            incar_settings=None,
            kpoints_settings=None,
            potcar_settings=None
        )
        
        if futurehouse_api_key:
            self.incar_validator = IncarValidatorAgent(
                api_key=google_api_key,
                futurehouse_api_key=futurehouse_api_key
            )
        else:
            self.incar_validator = None
            print("ℹ️  Literature validation disabled (no FutureHouse API key)")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_complete_workflow(self, user_request: str, log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete workflow from user request to final VASP inputs.
        """
        workflow_result = {
            "user_request": user_request,
            "steps_completed": [],
            "final_status": "started"
        }
        
        print(f"\n🚀 DFT Workflow Starting")
        print(f"{'='*60}")
        print(f"📝 Request: {user_request}")
        print(f"📁 Output:  {self.output_dir}/")
        print(f"{'='*60}")
        
        # Step 1: Structure Generation and Validation
        print(f"\n🏗️  WORKFLOW STEP 1: Structure Generation & Validation")
        print(f"{'─'*50}")
        
        structure_result = self._generate_and_validate_structure(user_request)
        workflow_result["structure_generation"] = structure_result
        
        if structure_result["status"] != "success":
            print(f"❌ Structure generation failed: {structure_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_structure_generation"
            return workflow_result
        
#        workflow_result["steps_completed"].append("structure_generation")
#        structure_path = structure_result["final_structure_path"]
#        
#        print(f"✅ Structure generated: {os.path.basename(structure_path)}")
        workflow_result["steps_completed"].append("structure_generation")
        structure_path = structure_result["final_structure_path"]

        # ─── Rename the ASE‑script POSCAR so Atomate2’s POSCAR won't overwrite it ───
        if os.path.basename(structure_path) == "POSCAR":
            preserved = os.path.join(self.output_dir, "POSCAR_structure")
            os.replace(structure_path, preserved)
            structure_path = preserved
            print("ℹ️  Renamed initial POSCAR → POSCAR_structure")

        print(f"✅ Structure generated: {os.path.basename(structure_path)}")
        if structure_result.get("warning"):
            print(f"⚠️  {structure_result['warning']}")
        
        # Step 2: VASP Input Generation
        print(f"\n⚛️  WORKFLOW STEP 2: VASP Input Generation")
        print(f"{'─'*50}")
        
        vasp_result = self._generate_vasp_inputs(structure_path, user_request)
        workflow_result["vasp_generation"] = vasp_result
        
        if vasp_result["status"] != "success":
            print(f"❌ VASP generation failed: {vasp_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_vasp_generation"
            return workflow_result
            
        workflow_result["steps_completed"].append("vasp_generation")
#        print(f"✅ VASP inputs generated: INCAR, KPOINTS")
        print("✅ VASP inputs generated via Atomate2: POSCAR, INCAR, KPOINTS")
        print(f"📋 Calculation type: {vasp_result.get('summary', 'N/A')}")

        # ─── Step 3: Error‑based INCAR/KPOINTS refinement (if you passed a log) ───
        if log_path:
            print("\n🔄 WORKFLOW STEP 3: Refining INCAR/KPOINTS from VASP log")
            ref = self.refine_from_log(user_request, log_path)
            workflow_result["error_refinement"] = ref
            workflow_result["steps_completed"].append("error_refinement")
        else:
            print("ℹ️  skipping error‑based refinement")
        
        # Step 4: Literature Validation and Improvements (optional)
        if self.incar_validator:
            print(f"\n📚  WORKFLOW STEP 4: Literature Validation")
            print(f"{'─'*50}")
            
            improvement_result = self._validate_and_improve_incar(
                vasp_result, structure_path, user_request
            )
            workflow_result["incar_improvement"] = improvement_result
            workflow_result["steps_completed"].append("incar_improvement")
        else:
            workflow_result["incar_improvement"] = {"status": "skipped", "message": "No FutureHouse API key"}
        
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

    def refine_from_log(self, original_request: str, log_path: str) -> Dict[str, Any]:
        """
        Given a VASP stdout/stderr log file, iteratively refine INCAR/KPOINTS
        in self.output_dir using VaspErrorUpdaterAgent.
        """
        outdir    = Path(self.output_dir)
        poscar_f  = outdir / "POSCAR"
        incar_f   = outdir / "INCAR"
        kpoints_f = outdir / "KPOINTS"

        log_text = Path(log_path).read_text()
        old_incar   = incar_f.read_text()
        old_kpoints = kpoints_f.read_text()

        plan = self.vasp_error_updater.refine_inputs(
            poscar_path=str(poscar_f),
            incar_path=str(incar_f),
            kpoints_path=str(kpoints_f),
            vasp_log=log_text,
            original_request=original_request
        )
        print("Plan:", plan)

        if plan.get("status") == "success":
            # INCAR backup & overwrite
            new_incar = plan.get("suggested_incar", "")
            if new_incar and new_incar != old_incar:
                ver = 0
                while (incar_f.with_suffix(f"{incar_f.suffix}.v{ver}")).exists():
                    ver += 1
                incar_f.rename(incar_f.with_suffix(f"{incar_f.suffix}.v{ver}"))
                incar_f.write_text(new_incar)
                print(f"   • INCAR updated → backed up as INCAR{incar_f.suffix}.v{ver}")

            # KPOINTS backup & overwrite
            new_kp = plan.get("suggested_kpoints", "")
            if new_kp and new_kp != old_kpoints:
                ver = 0
                while (kpoints_f.with_suffix(f"{kpoints_f.suffix}.v{ver}")).exists():
                    ver += 1
                kpoints_f.rename(kpoints_f.with_suffix(f"{kpoints_f.suffix}.v{ver}"))
                kpoints_f.write_text(new_kp)
                print(f"   • KPOINTS updated → backed up as KPOINTS{kpoints_f.suffix}.v{ver}")
        else:
            print("⚠️  Refinement failed:", plan.get("message"))

        return {
            "final_incar":   str(incar_f),
            "final_kpoints": str(kpoints_f),
            "status":        plan.get("status"),
            "message":       plan.get("message", "")
        }
    
    def _generate_and_validate_structure(self, user_request: str) -> Dict[str, Any]:
        """Generate and validate atomic structure with improved output formatting."""
        
        previous_script_content = None
        validator_feedback = None
        
        for cycle in range(self.max_refinement_cycles + 1):
            cycle_num = cycle + 1
            total_cycles = self.max_refinement_cycles + 1
            
            if cycle == 0:
                print(f"🔨 Generating structure (attempt {cycle_num}/{total_cycles})")
            else:
                print(f"🔄 Refining structure (attempt {cycle_num}/{total_cycles})")
                print(f"   Addressing: {len(validator_feedback.get('all_identified_issues', []))} validation issues")
            
            # Generate structure
            gen_result = self.structure_generator.generate_script(
                original_user_request=user_request + ". Save the structure in POSCAR format.",
                attempt_number_overall=cycle_num,
                is_refinement_from_validation=(cycle > 0),
                previous_script_content=previous_script_content if cycle > 0 else None,
                validator_feedback=validator_feedback if cycle > 0 else None
            )
            
            if gen_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Structure generation failed on cycle {cycle_num}: {gen_result.get('message')}",
                    "cycle": cycle_num
                }
            
            structure_file = gen_result["output_file"]
            script_content = gen_result["final_script_content"]
            previous_script_content = script_content
            
            print(f"   ✅ Structure file: {os.path.basename(structure_file)}")
            print(f"   🐍 Script: {os.path.basename(gen_result['final_script_path'])}")
            
            # Validate structure
            print(f"🔍 Validating structure...")
            val_result = self.structure_validator.validate_structure_and_script(
                structure_file_path=structure_file,
                generating_script_content=script_content,
                original_request=user_request
            )
            
            validator_feedback = val_result
            
            # Format validation results nicely
            self._print_validation_results(val_result, cycle_num)
            
            if val_result["status"] == "success":
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
                    "final_script_path": gen_result["final_script_path"],
                    "cycles_used": cycle_num,
                    "validation_result": val_result
                }
            elif cycle < self.max_refinement_cycles:
                print(f"🔄 Issues found, attempting refinement...")
                continue
            else:
                print(f"⚠️  Max refinement cycles reached, proceeding with current structure")
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
                    "final_script_path": gen_result["final_script_path"], 
                    "cycles_used": cycle_num,
                    "validation_result": val_result,
                    "warning": "Structure may have validation issues"
                }
        
        return {"status": "error", "message": "Structure generation loop failed"}
    
    def _print_validation_results(self, val_result: Dict[str, Any], cycle_num: int):
        """Print validation results in a user-friendly format."""
        
        if val_result["status"] == "success":
            print(f"   ✅ Validation passed")
            return
        
        # Format issues nicely
        issues = val_result.get("all_identified_issues", [])
        hints = val_result.get("script_modification_hints", [])
        assessment = val_result.get("overall_assessment", "No assessment provided")
        
        print(f"   ⚠️  Validation found {len(issues)} issue(s):")
        print(f"\n   📋 Overall Assessment:")
        print(f"      {assessment}")
        
        if issues:
            print(f"\n   🔍 Specific Issues:")
            for i, issue in enumerate(issues, 1):
                print(f"      {i}. {issue}")
        
        if hints:
            print(f"\n   💡 Suggested Improvements:")
            for i, hint in enumerate(hints, 1):
                print(f"      {i}. {hint}")
        
        print()  # Add spacing
    
    def _generate_vasp_inputs(self, structure_path: str, user_request: str) -> Dict[str, Any]:
        """Generate VASP INCAR and KPOINTS files."""
        
#        print(f"📝 Generating VASP input files...")
#        
#        # Generate initial VASP inputs
#        vasp_result = self.vasp_agent.generate_vasp_inputs(
#            poscar_path=structure_path,
#            original_request=user_request
#        )
#        
#        if vasp_result["status"] != "success":
#            return vasp_result
#        
#        # Save initial VASP files
#        saved_files = self.vasp_agent.save_inputs(vasp_result, self.output_dir)
#        vasp_result["saved_files"] = saved_files
#        
#        return vasp_result

        # Try Atomate2 for POSCAR/INCAR/KPOINTS (no POTCAR)
        print("📝 Generating POSCAR, INCAR, KPOINTS via Atomate2…")
        structure = Poscar.from_file(structure_path).structure
        try:
            # Try Atomate2 first
            self.atomate2_agent.generate(structure, self.output_dir)
            saved = [str(Path(self.output_dir) / f) for f in ("POSCAR", "INCAR", "KPOINTS")]
            return {"status": "success", "saved_files": saved}
        except Exception as e:
            # Any error in Atomate2 → fall back to LLM
            print(f"⚠️ Atomate2 failed ({e}); falling back to LLM…")
            vasp_res = self.vasp_agent.generate_vasp_inputs(
                poscar_path=structure_path,
                original_request=user_request
            )
            if vasp_res.get("status") != "success":
                return vasp_res
            saved = self.vasp_agent.save_inputs(vasp_res, self.output_dir)
            # include the already‑generated POSCAR
            return {"status": "success", "saved_files": [structure_path] + saved}
    
    def _validate_and_improve_incar(self, vasp_result: Dict[str, Any], 
                                   structure_path: str, user_request: str) -> Dict[str, Any]:
        """Validate INCAR against literature and apply improvements."""
        
        print(f"📖 Validating INCAR parameters against literature...")
        
        # Validate against literature
        validation_result = self.incar_validator.validate_and_improve_incar(
            incar_content=vasp_result["incar"],
            system_description=user_request
        )
        
        if validation_result["status"] != "success":
            print(f"❌ Literature validation failed: {validation_result.get('message')}")
            return validation_result
        
        # Check if improvements are needed
        if validation_result["validation_status"] == "needs_adjustment":
            adjustments = validation_result.get("suggested_adjustments", [])
            print(f"💡 Literature suggests {len(adjustments)} improvement(s):")
            
            for i, adj in enumerate(adjustments, 1):
                param = adj.get("parameter", "Unknown")
                current = adj.get("current_value", "N/A")
                suggested = adj.get("suggested_value", "N/A")
                reason = adj.get("reason", "No reason provided")
                print(f"   {i}. {param}: {current} → {suggested}")
                print(f"      Reason: {reason}")
            
            print(f"\n🔧 Applying improvements...")
            
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
                print(f"   ✅ Applied {improvement_result['adjustments_count']} improvements")
                print(f"   📄 Saved as: INCAR_improved")
            else:
                print(f"   ❌ Failed to apply improvements: {improvement_result.get('message')}")
            
        else:
            print(f"✅ INCAR parameters look good - no improvements needed")
            validation_result["improvement_application"] = {
                "status": "no_changes",
                "message": "No improvements needed"
            }
        
        # Save validation report
        self.incar_validator.save_validation_report(validation_result, self.output_dir)
        
        return validation_result
    
    def _print_final_summary(self, workflow_result: Dict[str, Any]):
        """Print a clean final summary."""
        
        print(f"\n🎉 DFT Workflow Complete!")
        print(f"{'='*60}")
        
        # Basic info
        status = workflow_result.get('final_status')
        steps = workflow_result.get('steps_completed', [])
        
        print(f"📋 Status: {status}")
        print(f"✅ Steps: {' → '.join(steps)}")
        print(f"📁 Output: {self.output_dir}/")
        
        # Structure info
        if "structure_generation" in workflow_result:
            struct_result = workflow_result["structure_generation"]
            if struct_result["status"] == "success":
                cycles = struct_result.get('cycles_used', 1)
                structure_file = os.path.basename(struct_result['final_structure_path'])
                print(f"🏗️  Structure: {structure_file} (refined {cycles} cycle{'s' if cycles > 1 else ''})")
        
        # VASP info
        if "vasp_generation" in workflow_result:
            vasp_result = workflow_result["vasp_generation"]
            if vasp_result["status"] == "success":
                calc_type = vasp_result.get('summary', 'DFT calculation')
                print(f"⚛️  VASP: {calc_type}")
        
        # Literature validation info
        if "incar_improvement" in workflow_result:
            imp_result = workflow_result["incar_improvement"]
            if imp_result["status"] == "success":
                if imp_result["validation_status"] == "needs_adjustment":
                    adj_count = len(imp_result.get("suggested_adjustments", []))
                    print(f"📚 Literature: {adj_count} parameter improvement{'s' if adj_count > 1 else ''} applied")
                else:
                    print(f"📚 Literature: Parameters validated, no changes needed")
        
        # Final files
        print(f"\n📄 Ready for VASP:")
        manifest = workflow_result.get("final_manifest", {})
        if manifest.get("ready_for_vasp"):
            files = manifest["final_files"]
            structure_file = files.get('structure', 'POSCAR')
            incar_file = files.get('incar', 'INCAR')
            kpoints_file = files.get('kpoints', 'KPOINTS')
            
            print(f"   • {structure_file}")
            print(f"   • {incar_file}{' ⭐ (literature-optimized)' if manifest.get('literature_validated') else ''}")
            print(f"   • {kpoints_file}")
        
        print(f"{'='*60}")
    
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
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
            
        return manifest
