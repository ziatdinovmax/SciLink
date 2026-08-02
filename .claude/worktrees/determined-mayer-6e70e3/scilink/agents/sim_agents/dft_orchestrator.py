# scilink/agents/sim_agents/dft_orchestrator.py

import os
import sys
import logging
import json
from io import StringIO
from typing import Optional, Dict, Any
from pathlib import Path

from ase.io import read as ase_read

from ...auth import get_api_key, APIKeyNotFoundError
from ._deprecation import normalize_params
from .structure_agent import StructureGenerator
from .val_agent import StructureValidatorAgent, IncarValidatorAgent
from .vasp_agent import VaspInputAgent
from .vasp_updater import VaspUpdater
# Atomate2Input is imported lazily in the "atomate2" branch so users picking
# vasp_generator_method="llm" don't need pymatgen/atomate2 installed.


class DFTOrchestrator:
    """
    Orchestrates a complete Density Functional Theory (DFT) input-generation pipeline.

    Composes the sim-agents stack (StructureGenerator, StructureValidatorAgent,
    VaspInputAgent / Atomate2Input, VaspUpdater, IncarValidatorAgent) to turn a
    high-level natural-language request into a validated atomic structure plus a
    complete set of VASP input files (POSCAR, INCAR, KPOINTS) ready for calculation.

    Includes an iterative refinement loop where the validator provides feedback to
    the structure generator, enabling self-correction. Does not run VASP itself.
    """

    def __init__(self,
                 api_key: str = None,
                 base_url: Optional[str] = None,
                 futurehouse_api_key: str = None,
                 mp_api_key: str = None,
                 generator_model: str = "gemini-3-pro-preview",
                 validator_model: str = "gemini-3-pro-preview",
                 output_dir: str = "dft_workflow_output",
                 max_refinement_cycles: int = 4,
                 script_timeout: int = 300,
                 vasp_generator_method: str = "llm",
                 # Deprecated aliases
                 google_api_key: str = None,
                 local_model: str = None):
        """
        Initialize the DFT orchestrator and its constituent agents.

        Args:
            api_key: API key for the LLM provider. Auto-discovered from the
                environment when both api_key and base_url are None.
            base_url: Optional base URL for an OpenAI-compatible internal proxy.
            futurehouse_api_key: FutureHouse API key for literature validation.
                Auto-discovered when None.
            mp_api_key: Materials Project API key for structure lookups.
                Auto-discovered when None.
            generator_model: Model name for structure / VASP-input generation.
            validator_model: Model name for structure validation.
            output_dir: Directory to save all generated files.
            max_refinement_cycles: Maximum validator-guided correction cycles.
            script_timeout: Timeout (seconds) for executing AI-generated ASE scripts.
            vasp_generator_method: "atomate2" (rule-based, recommended for production)
                or "llm" (AI-driven, more flexible but less predictable).
            google_api_key: Deprecated, use api_key instead.
            local_model: Deprecated, use base_url instead.
        """

        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="DFTOrchestrator",
        )

        # Auto-discover API keys
        if api_key is None and base_url is None:
            api_key = get_api_key('google')
            if not api_key:
                raise APIKeyNotFoundError('google')

        if futurehouse_api_key is None:
            futurehouse_api_key = get_api_key('futurehouse')

        if mp_api_key is None:
            mp_api_key = get_api_key('materials_project')

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

        self.api_key = api_key
        self.base_url = base_url
        self.futurehouse_api_key = futurehouse_api_key
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles
        self.vasp_generator_method = vasp_generator_method

        # Initialize agents
        self.structure_generator = StructureGenerator(
            api_key=api_key,
            base_url=base_url,
            model_name=generator_model,
            executor_timeout=script_timeout,
            generated_script_dir=output_dir,
            mp_api_key=mp_api_key,
        )

        self.structure_validator = StructureValidatorAgent(
            api_key=api_key,
            base_url=base_url,
            model_name=validator_model,
        )

        # Instantiate the correct VASP agent based on the chosen method.
        if self.vasp_generator_method == "llm":
            print("ℹ️  VASP Generator: 'llm' (default). Using AI to generate VASP inputs.")
            self.vasp_agent = VaspInputAgent(
                api_key=api_key,
                base_url=base_url,
                model_name=generator_model,
            )
        elif self.vasp_generator_method == "atomate2":
            print("ℹ️  VASP Generator: 'atomate2'. Using pymatgen/atomate2 for reliable inputs.")
            try:
                from .atomate2_utils import Atomate2Input
            except ImportError as e:
                raise ImportError(
                    "vasp_generator_method='atomate2' requires the [sim] extras "
                    "(pymatgen, atomate2). Install with: pip install 'scilink[sim]'. "
                    f"Original error: {e}"
                )
            self.vasp_agent = Atomate2Input()
        else:
            raise ValueError(f"Invalid vasp_generator_method: '{self.vasp_generator_method}'. "
                             f"Choose 'llm' or 'atomate2'.")

        # error_log based INCAR/KPOINTS refinement
        self.vasp_error_updater = VaspUpdater(
            api_key=api_key,
            base_url=base_url,
            model_name=generator_model,
        )

        if futurehouse_api_key:
            self.incar_validator = IncarValidatorAgent(
                api_key=api_key,
                base_url=base_url,
                futurehouse_api_key=futurehouse_api_key,
            )
        else:
            self.incar_validator = None
            print("ℹ️  Literature validation disabled (no FutureHouse API key)")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def run_complete_workflow(self, user_request: str) -> Dict[str, Any]:
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
        print(f"⚙️  VASP Input Method: '{self.vasp_generator_method}' is active.")
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

        workflow_result["steps_completed"].append("structure_generation")
        structure_path = structure_result["final_structure_path"]

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
        print(f"✅ VASP inputs generated: INCAR, KPOINTS, POSCAR")
        print(f"📋 Calculation type: {vasp_result.get('summary', 'N/A')}")

        if self.incar_validator and self.vasp_generator_method == "llm":
            print(f"\n📚  WORKFLOW STEP 3: Literature Validation")
            print(f"{'─'*50}")

            improvement_result = self._validate_and_improve_incar(
                vasp_result, structure_path, user_request
            )
            workflow_result["incar_improvement"] = improvement_result
            workflow_result["steps_completed"].append("incar_improvement")
        else:
            if self.vasp_generator_method == "atomate2":
                msg = "Skipped, Atomate2 uses expert-defined parameters."
            else:
                msg = "Skipped, no FutureHouse API key."
            print(f"\n📚  WORKFLOW STEP 3: Literature Validation")
            print(f"{'─'*50}")
            print(f"   {msg}")
            workflow_result["incar_improvement"] = {"status": "skipped", "message": msg}

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
        in self.output_dir using VaspUpdater.
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
            "message":       plan.get("message", ""),
            "explanation":    plan.get("explanation", {})
        }

    def _generate_and_validate_structure(self, user_request: str) -> Dict[str, Any]:
        """Generate and validate atomic structure with improved output formatting.

        Includes a circuit-breaker that exits early when refinement stops making
        progress (issue count not strictly decreasing for 2 consecutive cycles,
        or generator returned an unchanged script).
        """

        previous_script_content = None
        previous_structure_file = None
        previous_final_script_path = None
        validator_feedback = None
        attempt_history: list = []   # full per-cycle log: {script, issues, hints}

        for cycle in range(self.max_refinement_cycles + 1):
            cycle_num = cycle + 1
            total_cycles = self.max_refinement_cycles + 1

            if cycle == 0:
                print(f"🔨 Generating structure (attempt {cycle_num}/{total_cycles})")
            else:
                print(f"🔄 Refining structure (attempt {cycle_num}/{total_cycles})")
                print(f"    Addressing: {len(validator_feedback.get('all_identified_issues', []))} validation issues")

            gen_result = self.structure_generator.generate_script(
                original_user_request=user_request + ". Save the structure in POSCAR format.",
                attempt_number_overall=cycle_num,
                is_refinement_from_validation=(cycle > 0),
                previous_script_content=previous_script_content if cycle > 0 else None,
                validator_feedback=validator_feedback if cycle > 0 else None,
                attempt_history=attempt_history if cycle > 0 else None,
            )

            if gen_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Structure generation failed on cycle {cycle_num}: {gen_result.get('message')}",
                    "cycle": cycle_num
                }

            structure_file = gen_result["output_file"]
            script_content = gen_result["final_script_content"]

            # CIRCUIT-BREAKER 1: generator returned an unchanged script.
            # Treat as "the model has nothing more to fix" and accept current state.
            if cycle > 0 and script_content == previous_script_content:
                print(f"🛑 Generator returned an unchanged script — "
                      f"accepting current structure (cycle {cycle_num}).")
                return {
                    "status": "success",
                    "final_structure_path": previous_structure_file or structure_file,
                    "final_script_path": previous_final_script_path or gen_result["final_script_path"],
                    "cycles_used": cycle_num,
                    "validation_result": validator_feedback,
                    "warning": "Refinement stopped: generator made no further changes.",
                }

            previous_script_content = script_content
            previous_structure_file = structure_file
            previous_final_script_path = gen_result["final_script_path"]

            print(f"    ✅ Structure file: {os.path.basename(structure_file)}")
            print(f"    🐍 Script: {os.path.basename(gen_result['final_script_path'])}")

            print(f"🔍 Validating structure...")
            val_result = self.structure_validator.validate_structure_and_script(
                structure_file_path=structure_file,
                generating_script_content=script_content,
                original_request=user_request
            )

            validator_feedback = val_result
            self._print_validation_results(val_result, cycle_num)

            # Record this cycle in the history for the next iteration's prompt
            attempt_history.append({
                "script": script_content,
                "issues": list(val_result.get("all_identified_issues", []) or []),
                "hints": list(val_result.get("script_modification_hints", []) or []),
            })

            if val_result["status"] == "success":
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
                    "final_script_path": gen_result["final_script_path"],
                    "cycles_used": cycle_num,
                    "validation_result": val_result
                }

            # CIRCUIT-BREAKER 2: issue count failed to strictly decrease over
            # the last 2 consecutive cycles. Two distinct sub-cases:
            #   - PLATEAU (n2 == n1 == n0): same cosmetic complaints repeating
            #   - DIVERGENCE (n0 > n2): refinement is making the structure
            #     worse — each cycle introduces new issues without resolving
            #     old ones. Surfaced as a louder warning so callers know the
            #     final structure has substantial unresolved problems.
            # In both cases we accept the current structure (continuing to
            # refine wouldn't help), but the warning text differs.
            if len(attempt_history) >= 3:
                n_now = len(attempt_history[-1]["issues"])
                n_prev = len(attempt_history[-2]["issues"])
                n_prev2 = len(attempt_history[-3]["issues"])
                if n_now >= n_prev and n_prev >= n_prev2:
                    if n_now > n_prev2:
                        print(f"🛑 Validator complaints diverging "
                              f"({n_prev2} → {n_prev} → {n_now}); refinement is "
                              f"making the structure worse, not better. Accepting "
                              f"current structure but flagging unresolved issues.")
                        warning = (
                            f"Refinement stopped: validator complaints "
                            f"diverging ({n_prev2} → {n_prev} → {n_now}). "
                            f"Structure may have substantial unresolved issues; "
                            f"review the validation feedback before relying on "
                            f"the VASP inputs."
                        )
                    else:
                        print(f"🛑 Issue count plateaued over 2 cycles "
                              f"({n_prev2} → {n_prev} → {n_now}); validator "
                              f"complaints appear cosmetic. Accepting current "
                              f"structure.")
                        warning = (
                            "Refinement stopped: issue count plateaued "
                            "(likely cosmetic)."
                        )
                    return {
                        "status": "success",
                        "final_structure_path": structure_file,
                        "final_script_path": gen_result["final_script_path"],
                        "cycles_used": cycle_num,
                        "validation_result": val_result,
                        "warning": warning,
                    }

            if cycle < self.max_refinement_cycles:
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
            print(f"    ✅ Validation passed")
            return

        issues = val_result.get("all_identified_issues", [])
        hints = val_result.get("script_modification_hints", [])
        assessment = val_result.get("overall_assessment", "No assessment provided")

        print(f"    ⚠️  Validation found {len(issues)} issue(s):")
        print(f"\n    📋 Overall Assessment:")
        print(f"       {assessment}")

        if issues:
            print(f"\n    🔍 Specific Issues:")
            for i, issue in enumerate(issues, 1):
                print(f"       {i}. {issue}")

        if hints:
            print(f"\n    💡 Suggested Improvements:")
            for i, hint in enumerate(hints, 1):
                print(f"       {i}. {hint}")
        print()

    def _generate_vasp_inputs(self, structure_path: str, user_request: str) -> Dict[str, Any]:
        """Generate VASP INCAR and KPOINTS files using the selected method."""
        print(f"📝 Generating VASP input files using '{self.vasp_generator_method}' method...")

        if self.vasp_generator_method == "llm":
            vasp_result = self.vasp_agent.generate_vasp_inputs(
                poscar_path=structure_path,
                original_request=user_request
            )
            if vasp_result.get("status") == "success":
                self.vasp_agent.save_inputs(vasp_result, self.output_dir)
            return vasp_result

        elif self.vasp_generator_method == "atomate2":
            try:
                structure_obj = ase_read(structure_path)
                # The generate method now handles file writing internally
                self.vasp_agent.generate(
                    structure=structure_obj,
                    output_dir=self.output_dir
                )
                return {
                    "status": "success",
                    "summary": "Standard relaxation set from atomate2/pymatgen",
                    "incar": (Path(self.output_dir) / "INCAR").read_text()
                }
            except Exception as e:
                self.logger.error(f"Atomate2 input generation failed: {e}", exc_info=True)
                return {"status": "error", "message": f"Atomate2 generation failed: {e}"}

        return {"status": "error", "message": "Invalid VASP generator method."}

    def _validate_and_improve_incar(self, vasp_result: Dict[str, Any],
                                    structure_path: str, user_request: str) -> Dict[str, Any]:
        """Validate INCAR against literature and apply improvements."""

        if self.vasp_generator_method != "llm":
            msg = "Literature validation is only applicable for the 'llm' generator."
            self.logger.info(msg)
            return {"status": "skipped", "message": msg}

        print(f"📖 Validating INCAR parameters against literature...")
        validation_result = self.incar_validator.validate_and_improve_incar(
            incar_content=vasp_result["incar"],
            system_description=user_request
        )
        return validation_result

    def _print_final_summary(self, workflow_result: Dict[str, Any]):
        """Print a clean final summary."""
        print(f"\n🎉 DFT Workflow Complete!")
        print(f"{'='*60}")
        status = workflow_result.get('final_status')
        steps = workflow_result.get('steps_completed', [])
        print(f"📋 Status: {status}")
        print(f"✅ Steps: {' → '.join(steps)}")
        print(f"📁 Output: {self.output_dir}/")

        if "structure_generation" in workflow_result:
            struct_result = workflow_result["structure_generation"]
            if struct_result["status"] == "success":
                cycles = struct_result.get('cycles_used', 1)
                structure_file = os.path.basename(struct_result['final_structure_path'])
                print(f"🏗️  Structure: {structure_file} (refined {cycles} cycle{'s' if cycles > 1 else ''})")

        if "vasp_generation" in workflow_result:
            vasp_result = workflow_result["vasp_generation"]
            if vasp_result["status"] == "success":
                calc_type = vasp_result.get('summary', 'DFT calculation')
                print(f"⚛️  VASP: {calc_type}")

        if "incar_improvement" in workflow_result:
            imp_result = workflow_result["incar_improvement"]
            if imp_result["status"] == "success":
                if imp_result["validation_status"] == "needs_adjustment":
                    adj_count = len(imp_result.get("suggested_adjustments", []))
                    print(f"📚 Literature: {adj_count} parameter improvement{'s' if adj_count > 1 else ''} applied")
                else:
                    print(f"📚 Literature: Parameters validated, no changes needed")

        print(f"\n📄 Ready for VASP:")
        manifest = workflow_result.get("final_manifest", {})
        if manifest.get("ready_for_vasp"):
            files = manifest["final_files"]
            structure_file = files.get('structure', 'POSCAR')
            incar_file = files.get('incar', 'INCAR')
            kpoints_file = files.get('kpoints', 'KPOINTS')
            print(f"    • {structure_file}")
            print(f"    • {incar_file}{' ⭐ (literature-optimized)' if manifest.get('literature_validated') else ''}")
            print(f"    • {kpoints_file}")
        print(f"{'='*60}")

    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the workflow results."""
        summary = f"DFT Workflow Summary\n{'='*20}\n"
        summary += f"Request: {workflow_result['user_request']}\n"
        summary += f"Status: {workflow_result['final_status']}\n"
        summary += f"Steps completed: {', '.join(workflow_result['steps_completed'])}\n"
        summary += f"Output directory: {workflow_result.get('output_dir', 'N/A')}\n\n"
        if "structure_generation" in workflow_result:
            struct_result = workflow_result["structure_generation"]
            if struct_result["status"] == "success":
                struct_file = os.path.basename(struct_result['final_structure_path'])
                summary += f"✓ Final Structure: {struct_file}\n"
                summary += f"  Refinement cycles: {struct_result['cycles_used']}\n"
                summary += f"  Location: {workflow_result.get('output_dir', '.')}/\n"
        if "vasp_generation" in workflow_result:
            vasp_result = workflow_result["vasp_generation"]
            if vasp_result["status"] == "success":
                summary += f"✓ VASP Input Files:\n"
                if ("incar_improvement" in workflow_result and
                        workflow_result["incar_improvement"].get("improvement_application", {}).get("status") == "success"):
                    summary += f"  - INCAR_improved (literature-validated) ⭐\n"
                    summary += f"  - INCAR (original)\n"
                else:
                    summary += f"  - INCAR\n"
                summary += f"  - KPOINTS\n"
                summary += f"  Calculation: {vasp_result['summary']}\n"
        if "incar_improvement" in workflow_result:
            imp_result = workflow_result["incar_improvement"]
            if imp_result["status"] == "success":
                if imp_result["validation_status"] == "needs_adjustment":
                    adj_count = len(imp_result.get("suggested_adjustments", []))
                    summary += f"✓ Literature improvements: {adj_count} adjustments applied\n"
                else:
                    summary += f"✓ Literature validation: No improvements needed\n"
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
        if ("structure_generation" in workflow_result and
                workflow_result["structure_generation"]["status"] == "success"):
            structure_path = workflow_result["structure_generation"]["final_structure_path"]
            manifest["final_files"]["structure"] = os.path.basename(structure_path)
        if ("vasp_generation" in workflow_result and
                workflow_result["vasp_generation"]["status"] == "success"):
            if ("incar_improvement" in workflow_result and
                    workflow_result["incar_improvement"].get("improvement_application", {}).get("status") == "success"):
                manifest["final_files"]["incar"] = "INCAR_improved"
                manifest["literature_validated"] = True
            else:
                manifest["final_files"]["incar"] = "INCAR"
                manifest["literature_validated"] = False
            manifest["final_files"]["kpoints"] = "KPOINTS"
            if all(key in manifest["final_files"] for key in ["structure", "incar", "kpoints"]):
                manifest["ready_for_vasp"] = True
        try:
            manifest_path = os.path.join(self.output_dir, "final_files_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
        return manifest
