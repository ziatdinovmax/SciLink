"""
Tool registry for the SimulationOrchestratorAgent.

Mirrors the shape of AnalysisOrchestratorTools — each tool is a closure
registered via _register_tool with an OpenAI-format JSONSchema. Tools are
dispatched from the chat loop's manual tool-call handler.

Each tool wraps a piece of the existing sim_agents stack
(StructureGenerator, StructureValidatorAgent, VaspInputAgent, etc.) and
records a structure-centric session record in
`orch.generated_structures` so subsequent tools can find prior work.

Tools are constructed fresh per call (StructureGenerator's per-call
`generated_script_dir` makes caching awkward, and construction is fast).
"""

import glob
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class SimulationOrchestratorTools:
    """Tool registry + dispatch for SimulationOrchestratorAgent.

    Each tool is registered as a closure so it can capture a reference
    to the parent orchestrator (and therefore its session state).
    """

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent
                SimulationOrchestratorAgent.
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        # Lazily-initialized StructureGenerator reused across generate_structure
        # / refine_structure calls. Its MaterialsProjectHelper instance (and
        # therefore the MP record cache) survives between calls — so iterating
        # on variants of the same material doesn't re-fetch the same MP query
        # over and over.
        self._sg = None

        self._register_all_tools()

    def _get_structure_generator(self, workdir: str):
        """Return a session-shared StructureGenerator, with its
        ``generated_script_dir`` set to the per-call workdir.

        Lazy-initializes on first call. Reuses the same instance — and its
        MP-helper cache, model wrapper, and script executor — across all
        `generate_structure` / `refine_structure` calls in the session.
        """
        from .structure_agent import StructureGenerator
        if self._sg is None:
            self._sg = StructureGenerator(
                api_key=self.orch.api_key,
                base_url=self.orch.base_url,
                model_name=self.orch.model_name,
                generated_script_dir=str(workdir),
                mp_api_key=self.orch.mp_api_key,
            )
        else:
            self._sg.generated_script_dir = str(workdir)
            # ScriptExecutor's working_dir is per-call (passed to execute_script),
            # so no mutation needed there. The model wrapper, MP helper, and
            # cached MP records all stay live across calls.
        return self._sg

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_all_tools(self) -> None:
        """Register all tools with OpenAI format. Called once from __init__."""

        # =====================================================================
        # 0. SESSION STATUS  (low-cost diagnostic)
        # =====================================================================
        def session_status() -> str:
            structures = self.orch.generated_structures or []
            params = self.orch.default_calc_params or {}
            return json.dumps({
                "status": "ok",
                "session_dir": str(self.orch.base_dir),
                "structures_generated": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "poscar_path": s.get("poscar_path"),
                        "incar_path": s.get("incar_path"),
                    } for s in structures
                ],
                "default_calc_params": params,
                "simulation_mode": self.orch.simulation_mode.value,
            })

        self._register_tool(
            func=session_status,
            name="session_status",
            description=(
                "Report the current simulation session state — structures "
                "generated so far, sticky calculation parameters, output "
                "directory. Free to call; useful when you need to remember "
                "what's already been built before deciding the next step."
            ),
            parameters={},
            required=[],
        )

        # =====================================================================
        # 1. GENERATE STRUCTURE  (build → validate → refine, internal)
        # =====================================================================
        def generate_structure(description: str, skill: str = None,
                               validate_and_refine: bool = True,
                               max_refinement_cycles: int = 3,
                               based_on_slug: str = None) -> str:
            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            skill_content = self._load_skill_content(skill) if skill else None

            # If the user is asking for a variant of a previously-built
            # structure, fetch the prior script so the LLM can apply a
            # minimal delta instead of rewriting from scratch. Skipped
            # silently if the slug isn't found (caller intent ambiguous;
            # better to fall through to initial-build than refuse).
            prior_script = None
            if based_on_slug:
                prior = next(
                    (s for s in (self.orch.generated_structures or [])
                     if s.get("slug") == based_on_slug),
                    None,
                )
                if prior is None:
                    return json.dumps({
                        "status": "error",
                        "message": (
                            f"based_on_slug='{based_on_slug}' not found in "
                            f"this session. Call list_generated_structures to "
                            f"see available slugs, or omit based_on_slug to "
                            f"build from scratch."
                        ),
                    })
                prior_script = prior.get("script_content")
                if not prior_script:
                    self.logger.warning(
                        f"based_on_slug='{based_on_slug}' found but has no "
                        "script_content; falling through to initial build."
                    )

            try:
                sg = self._get_structure_generator(str(workdir))
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureGenerator: {e}",
                })

            # Append POSCAR-format request so downstream VASP tools can read it.
            request = description
            if "poscar" not in request.lower():
                request = request + ". Save the structure in POSCAR format."

            # Cycle 1: initial generation OR modification of prior script.
            result = sg.generate_script(
                original_user_request=request,
                attempt_number_overall=1,
                is_refinement_from_validation=False,
                skill_content=skill_content,
                prior_script_to_modify=prior_script,
            )
            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or result.get("last_error") or "Unknown failure",
                    "last_attempted_script_path": result.get("last_attempted_script_path"),
                })

            record = {
                "slug": slug,
                "description": description,
                "structure_dir": str(workdir),
                "poscar_path": result["output_file"],
                "script_path": result["final_script_path"],
                "script_content": result["final_script_content"],
                "skill": skill,
                "based_on_slug": based_on_slug,
                "incar_path": None,
                "kpoints_path": None,
                "vasp_summary": None,
                "validation": None,
                "created_at": datetime.now().isoformat(),
            }
            self.orch.generated_structures.append(record)

            # Optionally chain validate + refine internally so the user's
            # chat doesn't pause between generate and validate in co-pilot
            # mode (mirrors how analyze mode's run_analysis does build +
            # validate + refine inside a single tool call).
            cycles_used = 1
            warning = None
            if validate_and_refine:
                cycles_used, warning = self._validate_refine_loop(
                    record=record,
                    sg=sg,
                    original_request=request,
                    skill_content=skill_content,
                    max_cycles=max_refinement_cycles,
                )

            return json.dumps({
                "status": "success",
                "slug": slug,
                "structure_dir": str(workdir),
                "poscar_path": record["poscar_path"],
                "script_path": record["script_path"],
                "n_atoms": self._count_atoms(record["poscar_path"]),
                "skill_used": skill,
                "validation": {
                    "status": (record.get("validation") or {}).get("status"),
                    "issue_count": len(
                        (record.get("validation") or {}).get("all_identified_issues", []) or []
                    ),
                    "overall_assessment": (record.get("validation") or {}).get("overall_assessment", ""),
                } if record.get("validation") else None,
                "refinement_cycles_used": cycles_used,
                "warning": warning,
                "next_steps": (
                    "Generate VASP inputs with generate_vasp_inputs(...) "
                    "for the desired calculation type, or build a related "
                    "structure variant via another generate_structure call."
                    if not warning
                    else "Review the warning before proceeding to VASP inputs."
                ),
            })

        self._register_tool(
            func=generate_structure,
            name="generate_structure",
            description=(
                "Build an atomic structure from a natural-language description "
                "(e.g., 'rutile TiO2 with one O vacancy', 'graphene/MoS2 "
                "heterostructure'). By default also runs validation + "
                "refinement internally — same shape as analyze mode's "
                "`run_analysis`: one tool call returns a structure that has "
                "already been reviewed and improved if needed.\n\n"
                "Returns POSCAR + the structure record's session slug. Does "
                "NOT produce VASP inputs — call `generate_vasp_inputs` for "
                "those, or `run_complete_dft_workflow` for the full pipeline "
                "(structure + inputs together).\n\n"
                "Set `skill='aimsgb'` for grain boundaries / bicrystals / "
                "coincident-site-lattice constructions to load curated "
                "library guidance. Skip the `skill` parameter for plain "
                "ASE / pymatgen workflows.\n\n"
                "Use `validate_and_refine=False` only when the user has "
                "explicitly asked for a single-shot build with no "
                "validation (rare). The standalone `validate_structure` and "
                "`refine_structure` tools remain available for re-validating "
                "after a manual edit or external modification."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the structure to "
                        "build. Be specific about polymorph (e.g., "
                        "'rutile TiO2', 'wurtzite GaN'), supercell size, "
                        "defects, and other modifications. Materials Project "
                        "lookup is automatic when MP_API_KEY is configured."
                    ),
                },
                "skill": {
                    "type": "string",
                    "description": (
                        "Optional name of a built-in structure-generation "
                        "skill to load as additional library guidance. "
                        "Currently available: 'aimsgb' (grain boundaries, "
                        "bicrystals, Σ-value parametrized interfaces). Omit "
                        "for plain ASE / pymatgen workflows."
                    ),
                },
                "validate_and_refine": {
                    "type": "boolean",
                    "description": (
                        "Whether to run validation + refinement internally "
                        "after the initial build (default: true). Set false "
                        "only when the user explicitly wants a one-shot "
                        "build with no review."
                    ),
                },
                "max_refinement_cycles": {
                    "type": "integer",
                    "description": (
                        "Cap on validator-driven refinement cycles when "
                        "validate_and_refine=true (default: 3)."
                    ),
                },
                "based_on_slug": {
                    "type": "string",
                    "description": (
                        "Optional slug of a structure already built in this "
                        "session. When set, the script generator applies "
                        "the request as a minimal delta to that structure's "
                        "prior script instead of rewriting from scratch. "
                        "Available slugs from list_generated_structures."
                    ),
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 2. VALIDATE STRUCTURE
        # =====================================================================
        def validate_structure(poscar_path: str, original_request: str) -> str:
            from .val_agent import StructureValidatorAgent

            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            script_content = self._find_script_content(poscar_path)
            if not script_content:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Could not locate the generating script next to the "
                        "POSCAR. Validation requires the original script for "
                        "context. Re-run generate_structure if needed."
                    ),
                })

            try:
                validator = StructureValidatorAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureValidatorAgent: {e}",
                })

            val_result = validator.validate_structure_and_script(
                structure_file_path=poscar_path,
                generating_script_content=script_content,
                original_request=original_request,
            )

            # Attach to the matching session record (if any)
            record = self._find_structure_record(poscar_path)
            if record is not None:
                record["validation"] = val_result

            return json.dumps({
                "status": val_result.get("status", "unknown"),
                "overall_assessment": val_result.get("overall_assessment", ""),
                "all_identified_issues": val_result.get("all_identified_issues", []),
                "script_modification_hints": val_result.get("script_modification_hints", []),
                "poscar_path": poscar_path,
            })

        self._register_tool(
            func=validate_structure,
            name="validate_structure",
            description=(
                "Run a multimodal review of a previously generated structure "
                "(POSCAR + generating script + axis-view images). Returns "
                "overall_assessment, identified issues, and script-modification "
                "hints. Status is 'success' when no issues remain, "
                "'needs_correction' when refinement is warranted. Use after "
                "generate_structure to verify the geometry before producing "
                "VASP inputs."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR file to validate.",
                },
                "original_request": {
                    "type": "string",
                    "description": (
                        "The original natural-language request the structure "
                        "was built for — used to check that the result "
                        "matches what was asked for."
                    ),
                },
            },
            required=["poscar_path", "original_request"],
        )

        # =====================================================================
        # 5. GENERATE VASP INPUTS
        # =====================================================================
        def generate_vasp_inputs(poscar_path: str, request: str,
                                 method: str = "llm") -> str:
            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            structure_dir = Path(poscar_path).parent

            try:
                if method == "llm":
                    from .vasp_agent import VaspInputAgent
                    agent = VaspInputAgent(
                        api_key=self.orch.api_key,
                        base_url=self.orch.base_url,
                        model_name=self.orch.model_name,
                    )
                    vasp_result = agent.generate_vasp_inputs(
                        poscar_path=poscar_path,
                        original_request=request,
                    )
                    if vasp_result.get("status") != "success":
                        return json.dumps({
                            "status": "error",
                            "message": vasp_result.get("message") or "VASP input generation failed",
                        })
                    saved = agent.save_inputs(vasp_result, str(structure_dir))
                    if "error" in saved:
                        return json.dumps({"status": "error", "message": saved["error"]})
                    summary = vasp_result.get("summary", "")

                elif method == "atomate2":
                    try:
                        from .atomate2_utils import Atomate2Input
                    except ImportError as e:
                        return json.dumps({
                            "status": "error",
                            "message": (
                                "method='atomate2' requires the [sim] extras "
                                "(pymatgen, atomate2). Install with: "
                                "pip install 'scilink[sim]'. "
                                f"Original error: {e}"
                            ),
                        })
                    from ase.io import read as ase_read
                    structure_obj = ase_read(poscar_path)
                    Atomate2Input().generate(
                        structure=structure_obj,
                        output_dir=str(structure_dir),
                    )
                    summary = "Standard relaxation set from atomate2/pymatgen"

                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Invalid method '{method}'. Choose 'llm' or 'atomate2'.",
                    })

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"VASP input generation failed: {e}",
                })

            incar_path = structure_dir / "INCAR"
            kpoints_path = structure_dir / "KPOINTS"

            record = self._find_structure_record(poscar_path)
            if record is not None:
                record["incar_path"] = str(incar_path)
                record["kpoints_path"] = str(kpoints_path)
                record["vasp_summary"] = summary

            return json.dumps({
                "status": "success",
                "incar_path": str(incar_path),
                "kpoints_path": str(kpoints_path),
                "summary": summary,
                "method": method,
                "structure_dir": str(structure_dir),
            })

        self._register_tool(
            func=generate_vasp_inputs,
            name="generate_vasp_inputs",
            description=(
                "Generate VASP INCAR and KPOINTS files for a given structure "
                "tailored to the scientific objective in `request`. Saves "
                "INCAR + KPOINTS alongside the POSCAR. method='llm' (default) "
                "uses an LLM to derive parameters; method='atomate2' uses "
                "pymatgen/atomate2's MPRelaxSet (deterministic, requires the "
                "[sim] extras)."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR the inputs should match.",
                },
                "request": {
                    "type": "string",
                    "description": (
                        "Scientific objective / calculation type description "
                        "(e.g., 'static SCF for band structure', "
                        "'relaxation with vdW corrections for an interface'). "
                        "Drives INCAR parameter choices."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["llm", "atomate2"],
                    "description": (
                        "'llm' (default): AI-driven, more flexible. "
                        "'atomate2': rule-based, deterministic, requires the "
                        "[sim] extras."
                    ),
                },
            },
            required=["poscar_path", "request"],
        )

        # =====================================================================
        # 10. RUN COMPLETE DFT WORKFLOW (one-shot shortcut)
        # =====================================================================
        def run_complete_dft_workflow(description: str,
                                      max_refinement_cycles: int = 4,
                                      vasp_generator_method: str = "llm") -> str:
            from .dft_orchestrator import DFTOrchestrator

            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                wf = DFTOrchestrator(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                    mp_api_key=self.orch.mp_api_key,
                    generator_model=self.orch.model_name,
                    validator_model=self.orch.model_name,
                    output_dir=str(workdir),
                    max_refinement_cycles=max_refinement_cycles,
                    vasp_generator_method=vasp_generator_method,
                )
                result = wf.run_complete_workflow(description)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"DFT workflow failed: {e}",
                })

            final_status = result.get("final_status")
            structure_gen = result.get("structure_generation", {}) or {}
            structure_warning = structure_gen.get("warning")
            cycles_used = structure_gen.get("cycles_used")
            val_result = structure_gen.get("validation_result", {}) or {}
            outstanding_issues = val_result.get("all_identified_issues", []) or []

            poscar_path = workdir / "POSCAR"
            incar_path = workdir / "INCAR"
            kpoints_path = workdir / "KPOINTS"

            # Record in session state (only if structure exists)
            if poscar_path.exists():
                record = {
                    "slug": slug,
                    "description": description,
                    "structure_dir": str(workdir),
                    "poscar_path": str(poscar_path),
                    "script_path": structure_gen.get("final_script_path"),
                    "script_content": None,  # not surfaced by run_complete_workflow
                    "incar_path": str(incar_path) if incar_path.exists() else None,
                    "kpoints_path": str(kpoints_path) if kpoints_path.exists() else None,
                    "vasp_summary": result.get("vasp_generation", {}).get("summary"),
                    "validation": val_result,
                    "created_at": datetime.now().isoformat(),
                }
                self.orch.generated_structures.append(record)

            return json.dumps({
                "status": final_status if final_status else "error",
                "ready_for_vasp": final_status == "success",
                "output_directory": str(workdir),
                "manifest_path": str(workdir / "final_files_manifest.json"),
                "structure_warning": structure_warning,
                "structure_refinement_cycles": cycles_used,
                "structure_outstanding_issues_count": len(outstanding_issues),
                "structure_outstanding_issues": outstanding_issues[:10],
            })

        self._register_tool(
            func=run_complete_dft_workflow,
            name="run_complete_dft_workflow",
            description=(
                "Run the full DFT input pipeline as a one-shot: structure "
                "generation → validation → refinement → VASP inputs (with "
                "optional literature validation when FUTUREHOUSE_API_KEY is "
                "set). Use when the user just wants 'a complete DFT setup' "
                "without iterating on each step. For iterative work "
                "(build → check → refine → inputs), use the granular tools "
                "(generate_structure, validate_structure, refine_structure, "
                "generate_vasp_inputs) instead."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": "Natural-language description of the structure to build and prep.",
                },
                "max_refinement_cycles": {
                    "type": "integer",
                    "description": "Maximum validator-guided refinement cycles (default: 4).",
                },
                "vasp_generator_method": {
                    "type": "string",
                    "enum": ["llm", "atomate2"],
                    "description": (
                        "How to produce INCAR/KPOINTS. 'llm' (default) is "
                        "more flexible; 'atomate2' is rule-based and faster."
                    ),
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 3. REFINE STRUCTURE
        # =====================================================================
        def refine_structure(poscar_path: str, original_request: str) -> str:
            record = self._find_structure_record(poscar_path)
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Refinement requires a structure that was generated "
                        "in this session (so the validator feedback and "
                        "prior script are available). No record found for: "
                        f"{poscar_path}. Generate the structure first via "
                        "generate_structure, then validate, then refine."
                    ),
                })

            validator_feedback = record.get("validation")
            if not validator_feedback or validator_feedback.get("status") == "success":
                return json.dumps({
                    "status": "no_changes_needed",
                    "message": (
                        "No refinement-worthy validator feedback on record. "
                        "Run validate_structure first; if it returns "
                        "'success', the structure is already a fine starting "
                        "point and no refinement is needed."
                    ),
                })

            prior_script = record.get("script_content") or self._find_script_content(poscar_path)
            if not prior_script:
                return json.dumps({
                    "status": "error",
                    "message": "Could not locate the prior script for refinement.",
                })

            workdir = Path(record["structure_dir"])
            try:
                sg = self._get_structure_generator(str(workdir))
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructureGenerator: {e}",
                })

            request = original_request
            if "poscar" not in request.lower():
                request = request + ". Save the structure in POSCAR format."

            # Re-apply the same skill (if any) the original generation used,
            # so the refinement prompt has the same library guidance available.
            skill_content = self._load_skill_content(record.get("skill")) if record.get("skill") else None

            result = sg.generate_script(
                original_user_request=request,
                attempt_number_overall=2,  # refinement cycle
                is_refinement_from_validation=True,
                previous_script_content=prior_script,
                validator_feedback=validator_feedback,
                skill_content=skill_content,
            )

            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or result.get("last_error") or "Refinement failed",
                })

            new_poscar = result["output_file"]
            new_script_path = result["final_script_path"]
            new_script_content = result["final_script_content"]
            n_atoms = self._count_atoms(new_poscar)

            # Update the record in place rather than appending — refinement
            # produces a successor of the same logical structure.
            record["poscar_path"] = new_poscar
            record["script_path"] = new_script_path
            record["script_content"] = new_script_content
            record["validation"] = None  # invalidate prior validation
            record["incar_path"] = None  # invalidate prior inputs (geometry changed)
            record["kpoints_path"] = None
            record["vasp_summary"] = None

            return json.dumps({
                "status": "success",
                "slug": record["slug"],
                "poscar_path": new_poscar,
                "script_path": new_script_path,
                "n_atoms": n_atoms,
                "next_steps": (
                    "Optionally call validate_structure again to confirm the "
                    "refinement addressed the prior issues; then proceed to "
                    "generate_vasp_inputs."
                ),
            })

        self._register_tool(
            func=refine_structure,
            name="refine_structure",
            description=(
                "Re-generate a structure that this session already built, "
                "incorporating feedback from a prior validate_structure call. "
                "Updates the structure in place — the slug, directory, and "
                "session record are preserved; the POSCAR + script are "
                "replaced. Prior INCAR/KPOINTS (if any) are invalidated since "
                "the geometry changed. Requires the structure to have been "
                "validated in this session — run validate_structure first."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR to refine.",
                },
                "original_request": {
                    "type": "string",
                    "description": (
                        "The original natural-language request the structure "
                        "was built for. Refinement uses this together with "
                        "the validator's feedback."
                    ),
                },
            },
            required=["poscar_path", "original_request"],
        )

        # =====================================================================
        # 4. VIEW STRUCTURE
        # =====================================================================
        def view_structure(poscar_path: str) -> str:
            from .utils import generate_structure_views

            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })

            try:
                image_paths = generate_structure_views(poscar_path)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to render structure views: {e}",
                })

            if not image_paths:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Structure rendering produced no output (ASE may be "
                        "missing or the file may be unparseable)."
                    ),
                })

            return json.dumps({
                "status": "success",
                "image_paths": image_paths,
                "note": (
                    "PNG renders along the X, Y, and Z axes have been written "
                    "next to the POSCAR. The user can open them; the model "
                    "cannot view image bytes through this text-only tool "
                    "interface."
                ),
            })

        self._register_tool(
            func=view_structure,
            name="view_structure",
            description=(
                "Render axis-view PNG images (along X, Y, Z) of a structure "
                "for visual inspection. Saves images alongside the POSCAR. "
                "Useful when a user wants to eyeball the geometry before "
                "running calculations; the images are surfaced to the user, "
                "not to the model itself."
            ),
            parameters={
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR to render.",
                },
            },
            required=["poscar_path"],
        )

        # =====================================================================
        # 6. VALIDATE INCAR (literature-grounded)
        # =====================================================================
        def validate_incar(incar_path: str, system_description: str) -> str:
            if not Path(incar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"INCAR not found: {incar_path}",
                })

            if not self.orch.futurehouse_api_key:
                return json.dumps({
                    "status": "skipped",
                    "message": (
                        "Literature validation requires a FutureHouse API "
                        "key. Set FUTUREHOUSE_API_KEY in the environment "
                        "or pass futurehouse_api_key when constructing the "
                        "orchestrator."
                    ),
                })

            try:
                from .val_agent import IncarValidatorAgent
                validator = IncarValidatorAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct IncarValidatorAgent: {e}",
                })

            incar_content = Path(incar_path).read_text()
            result = validator.validate_and_improve_incar(
                incar_content=incar_content,
                system_description=system_description,
            )

            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Literature validation failed",
                })

            return json.dumps({
                "status": "success",
                "validation_status": result.get("validation_status"),
                "overall_assessment": result.get("overall_assessment"),
                "suggested_adjustments": result.get("suggested_adjustments", []),
                "literature_review": result.get("literature_review", "")[:2000],
                "incar_path": incar_path,
            })

        self._register_tool(
            func=validate_incar,
            name="validate_incar",
            description=(
                "Run a literature-grounded review of an INCAR file: pulls "
                "papers via FutureHouse, asks an LLM whether the chosen "
                "parameters are consistent with established practice for "
                "the system in question, returns suggested adjustments. "
                "Returns 'skipped' status when no FutureHouse API key is "
                "configured. Pair with apply_incar_improvements to write "
                "the suggested INCAR to disk."
            ),
            parameters={
                "incar_path": {
                    "type": "string",
                    "description": "Absolute path to the INCAR to validate.",
                },
                "system_description": {
                    "type": "string",
                    "description": (
                        "What system the INCAR is for and what the calculation "
                        "is supposed to compute (used as context for the "
                        "literature review)."
                    ),
                },
            },
            required=["incar_path", "system_description"],
        )

        # =====================================================================
        # 7. APPLY INCAR IMPROVEMENTS
        # =====================================================================
        def apply_incar_improvements(incar_path: str, poscar_path: str,
                                     original_request: str,
                                     suggested_adjustments: list,
                                     overall_assessment: str = "") -> str:
            if not Path(incar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"INCAR not found: {incar_path}",
                })
            if not Path(poscar_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {poscar_path}",
                })
            if not suggested_adjustments:
                return json.dumps({
                    "status": "no_changes",
                    "message": "No adjustments provided — nothing to apply.",
                })

            try:
                from .vasp_agent import VaspInputAgent
                agent = VaspInputAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct VaspInputAgent: {e}",
                })

            original_incar = Path(incar_path).read_text()
            output_dir = str(Path(incar_path).parent)

            result = agent.apply_improvements(
                original_incar=original_incar,
                validation_result={
                    "validation_status": "needs_adjustment",
                    "suggested_adjustments": suggested_adjustments,
                    "overall_assessment": overall_assessment,
                },
                poscar_path=poscar_path,
                original_request=original_request,
                output_dir=output_dir,
            )

            if result.get("status") not in ("success", "no_changes"):
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Apply-improvements failed",
                })

            return json.dumps({
                "status": result.get("status"),
                "improvements_applied": result.get("improvements_applied", False),
                "adjustments_count": result.get("adjustments_count", 0),
                "improved_incar_path": result.get("improved_incar_path"),
            })

        self._register_tool(
            func=apply_incar_improvements,
            name="apply_incar_improvements",
            description=(
                "Apply a list of literature-validated INCAR adjustments to an "
                "existing INCAR, writing the result as INCAR_improved next to "
                "the original. Pair with validate_incar — pass its "
                "suggested_adjustments through directly."
            ),
            parameters={
                "incar_path": {
                    "type": "string",
                    "description": "Absolute path to the original INCAR.",
                },
                "poscar_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR (provides system context).",
                },
                "original_request": {
                    "type": "string",
                    "description": "The original calculation-type request.",
                },
                "suggested_adjustments": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of adjustment dicts in the shape returned by "
                        "validate_incar (each with parameter/current_value/"
                        "suggested_value/reason)."
                    ),
                },
                "overall_assessment": {
                    "type": "string",
                    "description": "Brief literature-review summary (passed verbatim).",
                },
            },
            required=["incar_path", "poscar_path", "original_request", "suggested_adjustments"],
        )

        # =====================================================================
        # 11. LIST GENERATED STRUCTURES
        # =====================================================================
        def list_generated_structures() -> str:
            structures = self.orch.generated_structures or []
            return json.dumps({
                "status": "ok",
                "count": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "structure_dir": s.get("structure_dir"),
                        "poscar_path": s.get("poscar_path"),
                        "incar_path": s.get("incar_path"),
                        "kpoints_path": s.get("kpoints_path"),
                        "has_validation": s.get("validation") is not None,
                        "created_at": s.get("created_at"),
                    } for s in structures
                ],
            })

        self._register_tool(
            func=list_generated_structures,
            name="list_generated_structures",
            description=(
                "List all structures generated in this session with their "
                "paths and current state (whether VASP inputs exist, whether "
                "validation has been run). Use to remember what's been built "
                "before deciding next steps."
            ),
            parameters={},
            required=[],
        )

        # =====================================================================
        # 8. ANALYZE VASP OUTPUT (post-run)
        # =====================================================================
        def analyze_vasp_output(output_dir: str) -> str:
            from .post_run_analysis import analyze_run_directory

            summary = analyze_run_directory(output_dir)
            # Trim verbose subdicts before sending back — keep it scannable
            if isinstance(summary, dict):
                vr = summary.get("vasprun") or {}
                if "incar_snapshot" in vr and len(vr["incar_snapshot"]) > 30:
                    vr["incar_snapshot"] = dict(list(vr["incar_snapshot"].items())[:30])
                    vr["incar_snapshot_truncated"] = True
            return json.dumps(summary)

        self._register_tool(
            func=analyze_vasp_output,
            name="analyze_vasp_output",
            description=(
                "Read VASP output files (vasprun.xml + OUTCAR + stdout/"
                "stderr logs) from a completed or failed run and return a "
                "structured summary: convergence status (converged / "
                "not_converged / failed / unknown), final energy, ionic "
                "step count, max force on last step, snapshot of effective "
                "INCAR settings, and a list of human-readable hints for "
                "any known VASP error patterns matched in the logs. Use "
                "after the user runs VASP and points you at the run "
                "directory."
            ),
            parameters={
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Absolute path to the VASP run directory containing "
                        "vasprun.xml / OUTCAR / log files."
                    ),
                },
            },
            required=["output_dir"],
        )

        # =====================================================================
        # 9. SUGGEST INCAR FIXES (from VASP error log)
        # =====================================================================
        def suggest_incar_fixes(log_path: str, original_request: str) -> str:
            from .vasp_updater import VaspUpdater

            log_file = Path(log_path)
            if not log_file.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Log file not found: {log_path}",
                })

            run_dir = log_file.parent
            poscar = run_dir / "POSCAR"
            incar = run_dir / "INCAR"
            kpoints = run_dir / "KPOINTS"
            for required in [poscar, incar, kpoints]:
                if not required.exists():
                    return json.dumps({
                        "status": "error",
                        "message": (
                            f"Expected {required.name} alongside the log at "
                            f"{run_dir}, not found. suggest_incar_fixes "
                            "needs the original POSCAR, INCAR, and KPOINTS "
                            "in the same directory as the log."
                        ),
                    })

            try:
                updater = VaspUpdater(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct VaspUpdater: {e}",
                })

            log_text = log_file.read_text(errors="replace")
            try:
                plan = updater.refine_inputs(
                    poscar_path=str(poscar),
                    incar_path=str(incar),
                    kpoints_path=str(kpoints),
                    vasp_log=log_text,
                    original_request=original_request,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"VaspUpdater failed: {e}",
                })

            if plan.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": plan.get("message") or "VaspUpdater did not produce a plan",
                })

            return json.dumps({
                "status": "success",
                "suggested_incar": plan.get("suggested_incar", ""),
                "suggested_kpoints": plan.get("suggested_kpoints", ""),
                "explanation": plan.get("explanation", ""),
                "note": (
                    "Suggestions returned as text; not applied to disk. "
                    "If the user wants to use them, write them to disk "
                    "manually or re-generate via generate_vasp_inputs with "
                    "an updated request."
                ),
            })

        self._register_tool(
            func=suggest_incar_fixes,
            name="suggest_incar_fixes",
            description=(
                "When a VASP run failed and the user has the log, ask the "
                "VaspUpdater to read the log + the original INCAR/KPOINTS/"
                "POSCAR and propose revised inputs that would address the "
                "error. Returns suggested INCAR + KPOINTS as text plus an "
                "explanation. Does NOT write to disk; the user decides "
                "whether to apply the suggestions."
            ),
            parameters={
                "log_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the VASP stdout/stderr log file. "
                        "POSCAR / INCAR / KPOINTS must live in the same "
                        "directory."
                    ),
                },
                "original_request": {
                    "type": "string",
                    "description": (
                        "What the calculation was supposed to do — used as "
                        "context for the fix suggestions."
                    ),
                },
            },
            required=["log_path", "original_request"],
        )

        # =====================================================================
        # 12. SUBMIT VASP JOB
        # =====================================================================
        def submit_vasp_job(
            structure_slug: str,
            remote_dir: str,
            job_name: str = "vasp",
            partition: str = "",
            n_nodes: int = 1,
            n_tasks: int = 16,
            time_limit: str = "04:00:00",
            vasp_command: str = "srun vasp_std",
            modules: str = "",
            extra_directives: str = "",
        ) -> str:
            conn = self.orch.hpc_connection
            sched = self.orch.hpc_scheduler
            if conn is None or sched is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No HPC connection active. Construct "
                        "SimulationOrchestratorAgent with hpc_connection= "
                        "and hpc_scheduler= to enable job submission."
                    ),
                })
            if not conn.is_connected:
                return json.dumps({
                    "status": "error",
                    "message": "HPC connection is not active. Reconnect and retry.",
                })

            record = next(
                (s for s in (self.orch.generated_structures or [])
                 if s.get("slug") == structure_slug),
                None,
            )
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Structure '{structure_slug}' not found in this "
                        "session. Call list_generated_structures to see "
                        "available slugs."
                    ),
                })

            poscar = record.get("poscar_path")
            incar = record.get("incar_path")
            kpoints = record.get("kpoints_path")
            missing = [
                n for n, p in [("POSCAR", poscar), ("INCAR", incar), ("KPOINTS", kpoints)]
                if not p or not Path(p).exists()
            ]
            if missing:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Missing local files before upload: {missing}. "
                        "Run generate_vasp_inputs first."
                    ),
                })

            try:
                conn.mkdir_p(remote_dir)
                for local_path, remote_name in [
                    (poscar, "POSCAR"),
                    (incar, "INCAR"),
                    (kpoints, "KPOINTS"),
                ]:
                    conn.upload(local_path, f"{remote_dir}/{remote_name}")

                script_content = self._generate_job_script(
                    sched=sched,
                    job_name=job_name,
                    n_nodes=n_nodes,
                    n_tasks=n_tasks,
                    time_limit=time_limit,
                    partition=partition,
                    vasp_command=vasp_command,
                    modules=modules,
                    extra_directives=extra_directives,
                )
                local_script = Path(record["structure_dir"]) / "job.sh"
                local_script.write_text(script_content)
                remote_script = f"{remote_dir}/job.sh"
                conn.upload(str(local_script), remote_script)

                job_id = sched.submit(remote_script, work_dir=remote_dir)
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

            record["hpc_job_id"] = job_id
            record["hpc_remote_dir"] = remote_dir
            record["hpc_results_dir"] = None

            return json.dumps({
                "status": "success",
                "job_id": job_id,
                "scheduler": sched.name,
                "remote_dir": remote_dir,
                "next_steps": (
                    f"Monitor with get_job_status('{job_id}'). "
                    "When status is Completed, call "
                    f"download_vasp_results('{job_id}') to retrieve outputs."
                ),
            })

        self._register_tool(
            func=submit_vasp_job,
            name="submit_vasp_job",
            description=(
                "Upload VASP input files (POSCAR, INCAR, KPOINTS) to a remote "
                "HPC cluster and submit a job via the active scheduler "
                "(SLURM / PBS / LSF). Requires hpc_connection and hpc_scheduler "
                "to be set on the orchestrator. Call generate_vasp_inputs first "
                "to ensure local inputs exist."
            ),
            parameters={
                "structure_slug": {
                    "type": "string",
                    "description": "Session slug of the structure to submit (from list_generated_structures).",
                },
                "remote_dir": {
                    "type": "string",
                    "description": "Absolute remote path for the job working directory (e.g. /scratch/user/run1).",
                },
                "job_name": {
                    "type": "string",
                    "description": "Scheduler job name (default: 'vasp').",
                },
                "partition": {
                    "type": "string",
                    "description": "Scheduler partition / queue. Omit to use the cluster default.",
                },
                "n_nodes": {
                    "type": "integer",
                    "description": "Number of nodes (default: 1).",
                },
                "n_tasks": {
                    "type": "integer",
                    "description": "Number of MPI tasks / CPUs (default: 16).",
                },
                "time_limit": {
                    "type": "string",
                    "description": "Wall-time limit in HH:MM:SS (default: '04:00:00').",
                },
                "vasp_command": {
                    "type": "string",
                    "description": (
                        "Full run command including MPI launcher "
                        "(e.g. 'srun vasp_std', 'mpirun -np 16 vasp_std'). "
                        "Written verbatim into the job script. Default: 'srun vasp_std'."
                    ),
                },
                "modules": {
                    "type": "string",
                    "description": (
                        "Shell commands to load the VASP environment, written "
                        "verbatim into the job script "
                        "(e.g. 'module load vasp/6.3.2 intel/2023'). Omit if "
                        "the user's .bashrc already loads VASP."
                    ),
                },
                "extra_directives": {
                    "type": "string",
                    "description": (
                        "Additional scheduler directives to inject into the "
                        "job script header, one per line "
                        "(e.g. '#SBATCH --gres=gpu:1')."
                    ),
                },
            },
            required=["structure_slug", "remote_dir"],
        )

        # =====================================================================
        # 13. GET JOB STATUS
        # =====================================================================
        def get_job_status(job_id: str) -> str:
            conn = self.orch.hpc_connection
            sched = self.orch.hpc_scheduler
            if conn is None or sched is None:
                return json.dumps({
                    "status": "error",
                    "message": "No HPC connection active.",
                })
            if not conn.is_connected:
                return json.dumps({
                    "status": "error",
                    "message": "HPC connection is not active. Reconnect and retry.",
                })
            try:
                job = sched.status(job_id)
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

            return json.dumps({
                "status": "success",
                "job_id": job.job_id,
                "job_status": job.status.value,
                "is_terminal": job.status.is_terminal,
                "raw_status": job.raw_status,
                "partition": job.partition,
                "nodes": job.nodes,
                "ntasks": job.ntasks,
                "time_limit": job.time_limit,
                "time_used": job.time_used,
                "work_dir": job.work_dir,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "exit_code": job.exit_code,
                "node_list": job.node_list,
                "next_steps": (
                    f"download_vasp_results('{job_id}') to retrieve outputs."
                    if job.status.is_terminal and job.status.value == "Completed"
                    else (
                        f"Call get_job_status('{job_id}') again to check progress."
                        if not job.status.is_terminal
                        else "Job ended in a non-success state. Use analyze_vasp_output or suggest_incar_fixes."
                    )
                ),
            })

        self._register_tool(
            func=get_job_status,
            name="get_job_status",
            description=(
                "Poll the HPC scheduler for the current status of a submitted "
                "job. Returns job_status (Pending / Running / Completed / "
                "Failed / Cancelled / Timeout), time used, and whether the "
                "job has reached a terminal state. Use after submit_vasp_job "
                "to check progress."
            ),
            parameters={
                "job_id": {
                    "type": "string",
                    "description": "Scheduler job ID returned by submit_vasp_job.",
                },
            },
            required=["job_id"],
        )

        # =====================================================================
        # 14. DOWNLOAD VASP RESULTS
        # =====================================================================
        def download_vasp_results(job_id: str, local_dir: str = "") -> str:
            conn = self.orch.hpc_connection
            if conn is None:
                return json.dumps({
                    "status": "error",
                    "message": "No HPC connection active.",
                })
            if not conn.is_connected:
                return json.dumps({
                    "status": "error",
                    "message": "HPC connection is not active. Reconnect and retry.",
                })

            record = self._find_structure_by_job_id(job_id)
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"No session record found for job_id='{job_id}'. "
                        "Only jobs submitted via submit_vasp_job in this "
                        "session can be downloaded automatically."
                    ),
                })

            remote_dir = record.get("hpc_remote_dir")
            if not remote_dir:
                return json.dumps({
                    "status": "error",
                    "message": "No remote_dir recorded for this job.",
                })

            dest = Path(local_dir) if local_dir else Path(record["structure_dir"]) / "hpc_results"
            dest.mkdir(parents=True, exist_ok=True)

            target_files = [
                "vasprun.xml", "OUTCAR", "CONTCAR", "OSZICAR",
                "EIGENVAL", "DOSCAR", "vasp.stdout", "vasp.stderr",
            ]

            downloaded, skipped = [], []
            for fname in target_files:
                remote_path = f"{remote_dir}/{fname}"
                local_path = dest / fname
                try:
                    conn.download(remote_path, str(local_path))
                    downloaded.append(fname)
                except Exception:
                    skipped.append(fname)

            if not downloaded:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"No output files found in {remote_dir}. "
                        "The job may not have produced output yet."
                    ),
                })

            record["hpc_results_dir"] = str(dest)
            return json.dumps({
                "status": "success",
                "local_dir": str(dest),
                "downloaded": downloaded,
                "skipped": skipped,
                "next_steps": (
                    f"analyze_vasp_output('{dest}') to parse results, or "
                    f"generate_final_report('{record['slug']}') for a full summary."
                ),
            })

        self._register_tool(
            func=download_vasp_results,
            name="download_vasp_results",
            description=(
                "Download VASP output files (vasprun.xml, OUTCAR, CONTCAR, "
                "OSZICAR, etc.) from the remote HPC directory to a local "
                "directory. Skips files that don't exist (e.g. DOSCAR for "
                "a plain relaxation). Call after get_job_status confirms "
                "the job is Completed."
            ),
            parameters={
                "job_id": {
                    "type": "string",
                    "description": "Scheduler job ID returned by submit_vasp_job.",
                },
                "local_dir": {
                    "type": "string",
                    "description": (
                        "Local directory to save results. Defaults to "
                        "<structure_dir>/hpc_results/ when omitted."
                    ),
                },
            },
            required=["job_id"],
        )

        # =====================================================================
        # 15. GENERATE FINAL REPORT
        # =====================================================================
        def generate_final_report(
            structure_slug: str,
            output_path: str = "",
        ) -> str:
            record = next(
                (s for s in (self.orch.generated_structures or [])
                 if s.get("slug") == structure_slug),
                None,
            )
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Structure '{structure_slug}' not found. "
                        "Call list_generated_structures to see available slugs."
                    ),
                })

            # Run post-run analysis if results are available
            vasp_analysis = None
            results_dir = record.get("hpc_results_dir") or record.get("structure_dir")
            if results_dir and (Path(results_dir) / "vasprun.xml").exists():
                try:
                    from .post_run_analysis import analyze_run_directory
                    vasp_analysis = analyze_run_directory(results_dir)
                except Exception as e:
                    vasp_analysis = {"error": str(e)}

            lines = [
                "# VASP DFT Simulation Report",
                f"\n## Structure: {record.get('description', structure_slug)}",
                f"- **Slug:** `{structure_slug}`",
                f"- **Created:** {record.get('created_at', 'unknown')}",
                f"- **POSCAR:** `{record.get('poscar_path', 'N/A')}`",
            ]

            n_atoms = self._count_atoms(record.get("poscar_path", ""))
            if n_atoms is not None:
                lines.append(f"- **Atoms:** {n_atoms}")

            val = record.get("validation") or {}
            if val:
                lines += [
                    "\n## Structure Validation",
                    f"- **Status:** {val.get('status', 'unknown')}",
                    f"- **Assessment:** {val.get('overall_assessment', '')}",
                ]
                issues = val.get("all_identified_issues") or []
                if issues:
                    lines.append("- **Issues:**")
                    for iss in issues:
                        lines.append(f"  - {iss}")

            if record.get("incar_path") and Path(record["incar_path"]).exists():
                lines += [
                    "\n## VASP Inputs",
                    f"- **INCAR:** `{record['incar_path']}`",
                    f"- **KPOINTS:** `{record.get('kpoints_path', 'N/A')}`",
                ]
                if record.get("vasp_summary"):
                    lines.append(f"- **Summary:** {record['vasp_summary']}")

            if record.get("hpc_job_id"):
                sched_name = (
                    self.orch.hpc_scheduler.name
                    if self.orch.hpc_scheduler else "unknown"
                )
                lines += [
                    "\n## HPC Job",
                    f"- **Scheduler:** {sched_name}",
                    f"- **Job ID:** {record['hpc_job_id']}",
                    f"- **Remote dir:** `{record.get('hpc_remote_dir', 'N/A')}`",
                    f"- **Local results:** `{record.get('hpc_results_dir', 'N/A')}`",
                ]

            if vasp_analysis:
                lines.append("\n## Calculation Results")
                if "error" in vasp_analysis:
                    lines.append(f"- **Parse error:** {vasp_analysis['error']}")
                else:
                    vr = vasp_analysis.get("vasprun") or {}
                    oc = vasp_analysis.get("outcar") or {}
                    lines += [
                        f"- **Convergence:** {vasp_analysis.get('convergence_status', 'unknown')}",
                        f"- **Final energy:** {vr.get('final_energy_eV', 'N/A')} eV",
                        f"- **Ionic steps:** {vr.get('n_ionic_steps', 'N/A')}",
                        f"- **Max force (last step):** {oc.get('max_force_eV_per_A', 'N/A')} eV/Å",
                    ]
                    hints = vasp_analysis.get("error_hints") or []
                    if hints:
                        lines.append("- **Error hints:**")
                        for h in hints:
                            lines.append(f"  - {h}")

            report_text = "\n".join(lines) + "\n"
            out_path = (
                Path(output_path) if output_path
                else Path(record["structure_dir"]) / "final_report.md"
            )
            try:
                out_path.write_text(report_text)
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Failed to write report: {e}"})

            return json.dumps({
                "status": "success",
                "report_path": str(out_path),
                "report": report_text,
            })

        self._register_tool(
            func=generate_final_report,
            name="generate_final_report",
            description=(
                "Generate a Markdown summary report for a completed simulation "
                "workflow: structure description, validation outcome, VASP input "
                "settings, HPC job info, and parsed results (energy, convergence, "
                "error hints). Saves to <structure_dir>/final_report.md by default. "
                "Call after download_vasp_results to include calculation outcomes."
            ),
            parameters={
                "structure_slug": {
                    "type": "string",
                    "description": "Session slug of the structure to report on.",
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Full local path for the report file. Defaults to "
                        "<structure_dir>/final_report.md when omitted."
                    ),
                },
            },
            required=["structure_slug"],
        )

        # ↓↓↓ CLI flesh-out (step 5), run_task (step 6), tests (step 7).

    # ------------------------------------------------------------------
    # Helpers used by tool closures
    # ------------------------------------------------------------------

    @staticmethod
    def _print_validation_results(val_result: Dict[str, Any], cycle_num: int) -> None:
        """Mirror DFTOrchestrator._print_validation_results so the simulate
        orchestrator's chat shows the same structured assessment / issues /
        improvements block users are used to from analyze→DFT runs."""
        if val_result.get("status") == "success":
            print(f"    ✅ Validation passed (cycle {cycle_num})")
            return

        issues = val_result.get("all_identified_issues", []) or []
        hints = val_result.get("script_modification_hints", []) or []
        assessment = val_result.get("overall_assessment", "No assessment provided")

        print(f"    ⚠️  Validation (cycle {cycle_num}) found {len(issues)} issue(s):")
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

    def _validate_refine_loop(self, record: Dict[str, Any], sg,
                              original_request: str,
                              skill_content: Optional[str],
                              max_cycles: int) -> tuple:
        """Run validate → refine → validate up to ``max_cycles`` times.

        Updates ``record`` in place: after each cycle, ``poscar_path`` /
        ``script_path`` / ``script_content`` are replaced with the latest
        attempt and ``validation`` holds the latest validator output.

        Mirrors DFTOrchestrator._generate_and_validate_structure's
        circuit-breakers (unchanged-script and plateau-vs-divergence) AND
        its progress-reporting format so users see the same "📋 Assessment
        / 🔍 Issues / 💡 Improvements" block they're used to from
        analyze-mode DFT runs.

        Returns: (cycles_used, warning_or_None).
        """
        from .val_agent import StructureValidatorAgent

        try:
            validator = StructureValidatorAgent(
                api_key=self.orch.api_key,
                base_url=self.orch.base_url,
                model_name=self.orch.model_name,
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to construct validator: {e}. Skipping validate+refine."
            )
            return 1, f"Validation skipped (validator construction failed: {e})"

        attempt_history: list = []
        prev_script = record.get("script_content")

        for cycle in range(1, max_cycles + 1):
            print(f"🔍 Validating structure (cycle {cycle}/{max_cycles})...")
            val = validator.validate_structure_and_script(
                structure_file_path=record["poscar_path"],
                generating_script_content=record["script_content"],
                original_request=original_request,
            )
            record["validation"] = val
            attempt_history.append({
                "script": record["script_content"],
                "issues": list(val.get("all_identified_issues", []) or []),
                "hints": list(val.get("script_modification_hints", []) or []),
            })

            self._print_validation_results(val, cycle)

            if val.get("status") == "success":
                return cycle, None

            # Plateau vs divergence circuit-breaker (mirrors DFTOrchestrator).
            if len(attempt_history) >= 3:
                n_now = len(attempt_history[-1]["issues"])
                n_prev = len(attempt_history[-2]["issues"])
                n_prev2 = len(attempt_history[-3]["issues"])
                if n_now >= n_prev and n_prev >= n_prev2:
                    if n_now > n_prev2:
                        msg = (
                            f"Refinement stopped: validator complaints "
                            f"diverging ({n_prev2} → {n_prev} → {n_now}). "
                            f"Structure may have substantial unresolved "
                            f"issues; review before proceeding to VASP."
                        )
                        print(f"🛑 {msg}")
                        return cycle, msg
                    msg = (
                        "Refinement stopped: issue count plateaued "
                        "(likely cosmetic)."
                    )
                    print(f"🛑 {msg}")
                    return cycle, msg

            if cycle >= max_cycles:
                msg = (
                    f"Max refinement cycles ({max_cycles}) reached; "
                    f"structure has unresolved validation issues."
                )
                print(f"⚠️  {msg}")
                return cycle, msg

            # Refine.
            n_issues = len(val.get("all_identified_issues", []) or [])
            print(f"🔄 Refining structure (cycle {cycle + 1}/{max_cycles}) — "
                  f"addressing {n_issues} issue(s)")
            refine_result = sg.generate_script(
                original_user_request=original_request,
                attempt_number_overall=cycle + 1,
                is_refinement_from_validation=True,
                previous_script_content=record["script_content"],
                validator_feedback=val,
                attempt_history=attempt_history,
                skill_content=skill_content,
            )
            if refine_result.get("status") != "success":
                # Surface the actual underlying error (often a truncated
                # Python traceback from the failed inner-retry loop), and
                # make it explicit that we're keeping the prior good state
                # so the user knows the cycle-N structure is still usable.
                last_err = (
                    refine_result.get("last_error")
                    or refine_result.get("message")
                    or "(no detail captured)"
                )
                err_snippet = str(last_err).strip()
                if len(err_snippet) > 800:
                    err_snippet = err_snippet[:800] + "\n   [... truncated ...]"
                print(f"❌ Refinement attempt for cycle {cycle + 1} failed.")
                print(f"   Underlying error:\n   {err_snippet.replace(chr(10), chr(10) + '   ')}")
                print(f"   Keeping the structure from cycle {cycle} "
                      f"(POSCAR: {record['poscar_path']}).")
                msg = (
                    f"Refinement failed on cycle {cycle + 1} (kept the "
                    f"structure from cycle {cycle}, which IS usable as a "
                    f"DFT starting geometry — the failure was in the "
                    f"validator-driven rewrite, not in the original build)."
                )
                return cycle, msg

            new_script = refine_result["final_script_content"]
            if new_script == prev_script:
                msg = "Refinement stopped: generator made no further changes."
                print(f"🛑 {msg}")
                return cycle, msg

            record["poscar_path"] = refine_result["output_file"]
            record["script_path"] = refine_result["final_script_path"]
            record["script_content"] = new_script
            prev_script = new_script

        return max_cycles, None

    def _load_skill_content(self, skill_name: str) -> Optional[str]:
        """Resolve a skill name to its content, formatted as a single block.

        Resolution order:
          1. Built-in skills under scilink/skills/structure_generation/
          2. User-registered skills via orchestrator.register_skill()

        Returns None on any failure (fail-closed; the structure-gen prompt
        falls through to the generic template). Doesn't raise.
        """
        if not skill_name:
            return None
        try:
            from scilink.skills.loader import load_skill
            parsed = load_skill(skill_name, domain="structure_generation")
        except FileNotFoundError:
            # Fall back to user-registered skills (orchestrator-level)
            user_skills = getattr(self.orch, "_custom_skills", {}) or {}
            path = user_skills.get(skill_name)
            if not path:
                self.logger.warning(
                    f"Skill '{skill_name}' not found in built-ins or "
                    "user-registered skills. Proceeding without skill content."
                )
                return None
            try:
                from scilink.skills.loader import load_skill as _load
                parsed = _load(path)
            except Exception as e:
                self.logger.warning(f"Failed to load user skill '{skill_name}': {e}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to load skill '{skill_name}': {e}")
            return None

        # Concatenate the populated sections in a stable order.
        section_order = ["overview", "planning", "implementation",
                         "validation", "interpretation", "analysis"]
        chunks = []
        for sec in section_order:
            body = (parsed.get(sec) or "").strip()
            if body:
                chunks.append(f"### {sec.capitalize()}\n\n{body}")
        if not chunks:
            return None
        header = f"# Skill: {parsed.get('name') or skill_name}"
        return header + "\n\n" + "\n\n".join(chunks)

    def _make_slug(self, description: str) -> str:
        """Build a unique short slug from a description for use as a
        directory name. Always increments the orchestrator's structure
        counter so concurrent calls with the same description don't
        collide."""
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", description)[:40].strip("_") or "structure"
        self.orch._structure_counter += 1
        return f"{safe}_{self.orch._structure_counter:03d}"

    @staticmethod
    def _count_atoms(poscar_path: str) -> Optional[int]:
        """Best-effort atom count via ASE; returns None on parse failure."""
        try:
            from ase.io import read as ase_read
            atoms = ase_read(poscar_path)
            return len(atoms)
        except Exception:
            return None

    def _find_script_content(self, poscar_path: str) -> Optional[str]:
        """Find the generating script for a POSCAR.

        First check the orchestrator's session records (cheap, exact). If
        not found, fall back to globbing `script_*.py` in the POSCAR's
        directory and reading the most recent one — this lets validation
        work even when the LLM passes around paths without going through
        the session record path.
        """
        record = self._find_structure_record(poscar_path)
        if record and record.get("script_content"):
            return record["script_content"]

        poscar_dir = Path(poscar_path).parent
        candidates = sorted(
            poscar_dir.glob("script_*.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        try:
            return candidates[0].read_text(encoding="utf-8")
        except Exception:
            return None

    def _find_structure_record(self, poscar_path: str) -> Optional[Dict[str, Any]]:
        """Find the session record matching a POSCAR path, by string match."""
        target = str(Path(poscar_path).resolve())
        for record in self.orch.generated_structures or []:
            try:
                if str(Path(record.get("poscar_path", "")).resolve()) == target:
                    return record
            except Exception:
                continue
        return None

    def _find_structure_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Find the session record that was submitted as the given HPC job."""
        for record in self.orch.generated_structures or []:
            if record.get("hpc_job_id") == job_id:
                return record
        return None

    @staticmethod
    def _generate_job_script(
        sched,
        job_name: str,
        n_nodes: int,
        n_tasks: int,
        time_limit: str,
        partition: str,
        vasp_command: str,
        modules: str,
        extra_directives: str,
    ) -> str:
        """Generate a scheduler job script for VASP."""
        sname = getattr(sched, "name", "SLURM").upper()
        lines = ["#!/bin/bash"]

        if sname == "PBS":
            lines += [
                f"#PBS -N {job_name}",
                f"#PBS -l nodes={n_nodes}:ppn={n_tasks}",
                f"#PBS -l walltime={time_limit}",
                f"#PBS -o vasp.stdout",
                f"#PBS -e vasp.stderr",
            ]
            if partition:
                lines.append(f"#PBS -q {partition}")
            if extra_directives:
                lines.append(extra_directives)
            lines.append("\ncd $PBS_O_WORKDIR")
        elif sname == "LSF":
            lines += [
                f"#BSUB -J {job_name}",
                f"#BSUB -n {n_tasks}",
                f"#BSUB -W {time_limit}",
                f"#BSUB -o vasp.stdout",
                f"#BSUB -e vasp.stderr",
            ]
            if partition:
                lines.append(f"#BSUB -q {partition}")
            if extra_directives:
                lines.append(extra_directives)
        else:  # SLURM (default)
            lines += [
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --nodes={n_nodes}",
                f"#SBATCH --ntasks={n_tasks}",
                f"#SBATCH --time={time_limit}",
                f"#SBATCH --output=vasp.stdout",
                f"#SBATCH --error=vasp.stderr",
            ]
            if partition:
                lines.append(f"#SBATCH --partition={partition}")
            if extra_directives:
                lines.append(extra_directives)

        if modules:
            lines += ["", modules]

        lines += ["", vasp_command, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Registration + dispatch primitives (mirror analyze-mode shapes)
    # ------------------------------------------------------------------

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None,
    ) -> None:
        """Register a tool in OpenAI format."""
        self.functions_map[name] = func
        self.openai_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        })

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given arguments. Always returns a
        JSON string the chat loop can hand back to the LLM."""
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found",
            })
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })
