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

import pandas as pd

# read_document character cap — longer documents are truncated (~50k tokens).
_READ_DOC_MAX_CHARS = 200_000


def _extract_document_text(path: Path, ocr_model: Any = None) -> Dict[str, Any]:
    """Extract plain text from a PDF / DOCX / Markdown / text file.

    Thin wrapper over the shared ``scilink.parsers.extract_text`` (which adds
    table-aware PDF extraction); it applies read_document's character cap.
    When ``ocr_model`` is supplied, scanned/sparse PDF pages are transcribed
    via the vision-OCR fallback. Returns a dict with ``text`` plus metadata
    (page/paragraph count, ``n_chars``, ``truncated``, ``n_ocr_pages``).
    Raises ValueError for an unsupported extension; reader errors propagate
    to the caller.
    """
    from ...parsers import extract_text

    info = extract_text(path, ocr_model=ocr_model)
    text = info.get("text", "")
    info["truncated"] = len(text) > _READ_DOC_MAX_CHARS
    if info["truncated"]:
        text = text[:_READ_DOC_MAX_CHARS]
        info["text"] = text
        info["n_chars"] = len(text)
    return info


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

        # Lazily-initialized StructurePipeline reused across
        # generate_structure / refine_structure calls. It owns the shared
        # StructureGenerator (so the MaterialsProjectHelper cache survives
        # between calls), the validator, and the generate→validate→refine loop —
        # the tools delegate to it rather than reimplementing that loop.
        self._so = None

        self._register_all_tools()

    def _get_structure_pipeline(self, workdir: str):
        """Return a session-shared StructurePipeline, lazy-initialized on
        first call. Reuses the same instance — its StructureGenerator (and the
        MP-helper cache), validator, model wrapper, and script executor — across
        all generate_structure / refine_structure calls in the session. The
        per-call output directory is passed through generate_and_validate.
        """
        from .structure_pipeline import StructurePipeline
        if self._so is None:
            self._so = StructurePipeline(
                api_key=self.orch.api_key,
                base_url=self.orch.base_url,
                generator_model=self.orch.model_name,
                validator_model=self.orch.model_name,
                mp_api_key=self.orch.mp_api_key,
                output_dir=str(workdir),
            )
        return self._so

    def _get_structure_generator(self, workdir: str):
        """Return the session-shared StructureGenerator (owned by the shared
        StructurePipeline), with its ``generated_script_dir`` set to the
        per-call workdir. Used by refine_structure's single-step rewrite.
        """
        so = self._get_structure_pipeline(str(workdir))
        so.structure_generator.generated_script_dir = str(workdir)
        return so.structure_generator

    # ------------------------------------------------------------------
    # Engine-neutral critic helpers
    # ------------------------------------------------------------------

    def _resolve_engine(self, software: Optional[str]) -> "tuple[Optional[str], Optional[str]]":
        """Resolve ``(skill, domain)`` for a critic call.

        With an explicit ``software`` override, the scale (domain) is derived
        from where that engine's skill bundle lives — not from routing or a
        hardcoded default — so e.g. ``software="lammps"`` resolves to
        ``("lammps", "molecular_dynamics")`` and ``software="vasp"`` to
        ``("vasp", "periodic_dft")`` regardless of routing. The search is
        scoped to the known simulation scales (the pipeline's scale
        registry). When ``software`` is omitted, both come from the
        orchestrator's routing decision.
        """
        if not software:
            return self.orch.active_skill_and_domain()

        # Derive the scale from the engine's bundle location across the
        # known simulation scales — no hardcoded engine→scale bias.
        from ...skills.loader import list_all_skills
        from .simulation_pipeline import _DEFAULT_ENGINE
        all_skills = list_all_skills()
        for scale in _DEFAULT_ENGINE:
            if software in all_skills.get(scale, []):
                return software, scale
        # Unknown engine for the known scales: keep the name, take the scale
        # from routing if any (still no hardcoded default).
        _engine, routed_scale = self.orch.active_skill_and_domain()
        return software, routed_scale

    def _get_input_validator(self):
        """Construct an InputValidator with the session's credentials.

        Forwards the orchestrator's FutureHouse key so literature-grounded
        review is available when one is configured (degrades gracefully
        when absent)."""
        from .critics import InputValidator
        return InputValidator(
            api_key=self.orch.api_key,
            base_url=self.orch.base_url,
            model_name=self.orch.model_name,
            futurehouse_api_key=getattr(self.orch, "futurehouse_api_key", None),
        )

    def _get_run_critic(self):
        """Construct a RunCritic with the session's LLM credentials."""
        from .critics import RunCritic
        return RunCritic(
            api_key=self.orch.api_key,
            base_url=self.orch.base_url,
            model_name=self.orch.model_name,
        )

    def _record_input_files(self, record: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Return ``{filename: contents}`` for a structure record's inputs.

        Reads from the record's generic ``input_files`` map (filename →
        path) when present, falling back to the legacy ``incar_path`` /
        ``kpoints_path`` fields so records created before the generic map
        still resolve. Missing or unreadable files are skipped.
        """
        if not record:
            return {}
        paths: Dict[str, str] = {}
        generic = record.get("input_files")
        if isinstance(generic, dict) and generic:
            paths = dict(generic)
        else:
            for fname, key in (("INCAR", "incar_path"), ("KPOINTS", "kpoints_path")):
                p = record.get(key)
                if p:
                    paths[fname] = p
        contents: Dict[str, str] = {}
        for fname, p in paths.items():
            try:
                if p and Path(p).exists():
                    contents[fname] = Path(p).read_text()
            except Exception:
                continue
        return contents

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
            engine, scale = self.orch.active_skill_and_domain()
            routing = self.orch.routing_decision or {}
            return json.dumps({
                "status": "ok",
                "session_dir": str(self.orch.base_dir),
                "structures_generated": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "structure_path": s.get("structure_path"),
                        "input_files": s.get("input_files") or {},
                    } for s in structures
                ],
                "default_calc_params": params,
                "simulation_mode": self.orch.simulation_mode.value,
                "routing": {
                    "scale": scale,
                    "engine": engine,
                    "routed": bool(scale and engine),
                    "reasoning": routing.get("reasoning"),
                },
            })

        self._register_tool(
            func=session_status,
            name="session_status",
            description=(
                "Report the current simulation session state — structures "
                "generated so far, sticky calculation parameters, the "
                "active routing decision (which scale and engine are in "
                "use), and the output directory. Free to call; useful "
                "when you need to remember what's already been built "
                "before deciding the next step."
            ),
            parameters={},
            required=[],
        )

        # =====================================================================
        # 0b. ROUTE SIMULATION  (pick scale + engine for the user's goal)
        # =====================================================================
        def route_simulation(user_goal: str,
                             system_description: str = None) -> str:
            """Pick (scale, engine) for the user's simulation goal.

            Builds the candidate set from the agent-supports ∩
            user-available intersection (via skill-bundle discovery
            and AvailableSoftware probes), then asks the LLM to choose
            among them based on the goal's physics.
            """
            from .simulation_router import SimulationRouter
            router = SimulationRouter(model=self.orch.model)
            decision = router.route(
                user_goal=user_goal,
                system_description=system_description,
            )
            # Stash on the orchestrator so subsequent tool calls / the
            # chat loop can see what was picked, without re-routing on
            # every turn.
            self.orch.routing_decision = decision
            return json.dumps(decision, indent=2)

        self._register_tool(
            func=route_simulation,
            name="route_simulation",
            description=(
                "Pick (scale, engine) for the user's simulation goal. "
                "Returns JSON {scale, engine, reasoning, alternatives, "
                "candidates_considered}. CALL THIS EARLY in the "
                "conversation, before generating structures or inputs, "
                "so subsequent tool calls target the right engine. The "
                "decision intersects three things: (1) which scale agents "
                "are loaded, (2) which engines the user has installed "
                "(per their `available_software.yaml`), (3) the LLM's "
                "judgment on which scale fits the user's physics goal. "
                "Supported dispatch paths: periodic_dft (VASP, QE) via "
                "generate_dft_inputs; molecular_dynamics (LAMMPS) via "
                "generate_dft_inputs; machine_learning_potentials (MACE, "
                "CHGNet, DeePMD, UMA, orb) via run_mlip_simulation. "
                "For EXAFS workflows use run_exafs_workflow directly — "
                "it handles routing, MLIP deployment, and FEFF setup."
            ),
            parameters={
                "user_goal": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what the user "
                        "wants to simulate (e.g. 'Relax a Cu(111) slab "
                        "and report a stable lattice constant')."
                    ),
                },
                "system_description": {
                    "type": "string",
                    "description": (
                        "Optional brief description of the system "
                        "(e.g. 'metallic surface, 16 atoms, includes "
                        "CO adsorbate'). Helps the router pick the "
                        "right scale; omit if not yet known."
                    ),
                },
            },
            required=["user_goal"],
        )

        # =====================================================================
        # 0c. PLAN STRUCTURE  (structure_class + simulation_scale + constraints)
        # =====================================================================
        def plan_structure(description: str, system_description: str = None) -> str:
            """Decide HOW to build a structure: structure_class + simulation_scale
            + the cross-term constraints (size / periodicity / solvation / charge).
            Returns a StructureSpec as JSON."""
            from .structure_planner import StructurePlanner
            spec = StructurePlanner(model=self.orch.model).plan(
                description, system_description=system_description)
            self.orch.structure_plan = spec  # stash for subsequent generate_structure
            return json.dumps(spec.to_dict(), indent=2)

        self._register_tool(
            func=plan_structure,
            name="plan_structure",
            description=(
                "Decide how to build an atomic structure from a free-text request, "
                "along TWO axes plus the constraints from their interaction: "
                "structure_class (crystal / molecular / condensed / biomolecular — the "
                "kind of structure) and simulation_scale (periodic_dft / molecular_dft / "
                "molecular_dynamics / machine_learning_potentials — what it's for), and "
                "derives size / periodicity / solvation / charge constraints (e.g. MD -> "
                "large + explicit solvent; molecular DFT -> isolated + implicit). Returns "
                "a StructureSpec JSON. Call before generate_structure when the build "
                "approach isn't obvious: pass the returned structure_class to "
                "generate_structure, and feed its size/periodicity/solvation into the "
                "generate_structure `constraints` argument."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the structure / system "
                        "(e.g. 'a solvated lysozyme system', 'band structure of rutile TiO2')."
                    ),
                },
                "system_description": {
                    "type": "string",
                    "description": "Optional extra context about the system; omit if unknown.",
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 1. GENERATE STRUCTURE  (build → validate → refine, internal)
        # =====================================================================
        def generate_structure(description: str, skill=None,
                               structure_class: str = "crystal",
                               constraints: str = None,
                               validate_and_refine: bool = True,
                               max_refinement_cycles: int = 3,
                               based_on_slug: str = None) -> str:
            # ``skill`` accepts str | list[str] | None — multi-skill support
            # via _load_skill_content. Single string and single-element list
            # behave identically.
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
                so = self._get_structure_pipeline(str(workdir))
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct StructurePipeline: {e}",
                })

            # Delegate the whole generate → validate → refine loop to the
            # StructurePipeline (single source of truth). structure_class
            # defaults to "crystal": simulate mode is periodic-DFT-centric today,
            # so crystal is the sensible default *class* (it supplies the
            # class-specific validation rubric). A user-supplied `skill` (rendered
            # into skill_content above) overrides the crystal *generation* skill;
            # the crystal validation rubric still applies. The orchestrator
            # appends the POSCAR-format instruction. (When the StructurePlanner
            # lands it will set structure_class per request.)
            result = so.generate_and_validate(
                description,
                structure_class=structure_class,
                skill_content=skill_content,
                constraints=constraints,
                prior_script=prior_script,
                validate=validate_and_refine,
                max_cycles=max_refinement_cycles,
                output_dir=str(workdir),
            )
            if result.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Structure generation failed",
                })

            val = result.get("validation_result")
            record = {
                "slug": slug,
                "description": description,
                "structure_dir": str(workdir),
                "structure_path": result["final_structure_path"],
                "script_path": result["final_script_path"],
                "script_content": result.get("final_script_content"),
                "skill": skill,
                "based_on_slug": based_on_slug,
                "input_files": {},
                "summary": None,
                "validation": val,
                "created_at": datetime.now().isoformat(),
            }
            self.orch.generated_structures.append(record)

            return json.dumps({
                "status": "success",
                "slug": slug,
                "structure_dir": str(workdir),
                "structure_path": record["structure_path"],
                "script_path": record["script_path"],
                "n_atoms": self._count_atoms(record["structure_path"]),
                "skill_used": skill,
                "validation": {
                    "status": (val or {}).get("status"),
                    "issue_count": len(
                        (val or {}).get("all_identified_issues", []) or []
                    ),
                    "overall_assessment": (val or {}).get("overall_assessment", ""),
                } if val else None,
                "refinement_cycles_used": result.get("cycles_used", 1),
                "warning": result.get("warning"),
                "next_steps": (
                    "Generate VASP inputs with generate_dft_inputs(...) "
                    "for the desired calculation type, or build a related "
                    "structure variant via another generate_structure call."
                    if not result.get("warning")
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
                "NOT produce VASP inputs — call `generate_dft_inputs` for "
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
                    # Single skill name or list of names — multi-skill
                    # support via _load_skill_content. Schema permits both
                    # via JSON Schema ``oneOf``.
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Optional name (or list of names) of built-in "
                        "structure-generation skills to load as additional "
                        "library guidance. Currently available: 'aimsgb' "
                        "(grain boundaries, bicrystals, Σ-value parametrized "
                        "interfaces). Omit for plain ASE / pymatgen "
                        "workflows; pass a list to combine multiple skills."
                    ),
                },
                "structure_class": {
                    "type": "string",
                    "description": (
                        "Structure archetype that selects the build skill + output "
                        "format + validation rubric: 'crystal' (default; periodic "
                        "solids/slabs/defects → POSCAR), 'molecular' (isolated "
                        "molecules → xyz), 'condensed' (solvated/liquid boxes → "
                        "POSCAR), 'biomolecular' (proteins/nucleic acids → pdb). "
                        "Use the structure_class returned by plan_structure."
                    ),
                },
                "constraints": {
                    "type": "string",
                    "description": (
                        "Optional build-constraints block to honor (target size / "
                        "periodicity / solvation / charge), typically the "
                        "size+periodicity+solvation from a plan_structure result."
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
        def validate_structure(structure_path: str, original_request: str) -> str:
            from .val_agent import StructureValidatorAgent

            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {structure_path}",
                })

            script_content = self._find_script_content(structure_path)
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
                structure_file_path=structure_path,
                generating_script_content=script_content,
                original_request=original_request,
            )

            # Attach to the matching session record (if any)
            record = self._find_structure_record(structure_path)
            if record is not None:
                record["validation"] = val_result

            return json.dumps({
                "status": val_result.get("status", "unknown"),
                "overall_assessment": val_result.get("overall_assessment", ""),
                "all_identified_issues": val_result.get("all_identified_issues", []),
                "script_modification_hints": val_result.get("script_modification_hints", []),
                "structure_path": structure_path,
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
                "structure_path": {
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
            required=["structure_path", "original_request"],
        )

        # =====================================================================
        # 5. GENERATE VASP INPUTS
        # =====================================================================
        def generate_dft_inputs(structure_path: str, request: str,
                                software: str = None, method: str = "llm") -> str:
            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Structure file not found: {structure_path}",
                })

            skill, domain = self._resolve_engine(software)
            if not skill:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No engine selected. Call route_simulation first, or "
                        "pass `software` explicitly (e.g. 'vasp')."
                    ),
                })

            structure_dir = Path(structure_path).parent
            try:
                from .simulation_pipeline import _generate_inputs
                gen = _generate_inputs(
                    scale=domain, software=skill, method=method,
                    structure_file=structure_path, request=request,
                    output_dir=str(structure_dir),
                    api_key=self.orch.api_key, base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"DFT input generation failed: {e}",
                })

            if gen.get("status") not in (None, "success"):
                return json.dumps({
                    "status": "error",
                    "message": gen.get("message") or "DFT input generation failed",
                })

            # Generic input-files record: filename -> path of the saved inputs.
            file_paths = {
                fn: str(structure_dir / fn)
                for fn in (gen.get("input_files") or {})
                if (structure_dir / fn).exists()
            }
            summary = gen.get("summary", "")
            record = self._find_structure_record(structure_path)
            if record is not None:
                record["input_files"] = file_paths
                record["summary"] = summary

            return json.dumps({
                "status": "success",
                "engine": skill,
                "input_files": file_paths,
                "summary": summary,
                "method": method,
                "structure_dir": str(structure_dir),
            })

        self._register_tool(
            func=generate_dft_inputs,
            name="generate_dft_inputs",
            description=(
                "Generate periodic-DFT input files for a structure, tailored "
                "to the scientific objective in `request`, and save them "
                "alongside the structure. Engine-neutral: `software` selects "
                "the engine (e.g. 'vasp', 'qe'), defaulting to the routing "
                "decision. method='llm' (default) derives parameters with an "
                "LLM; a named method (e.g. 'atomate2' for VASP) uses a "
                "deterministic generation backend from the engine's skill "
                "bundle (requires the [sim] extras). Returns a generic "
                "input_files map."
            ),
            parameters={
                "structure_path": {
                    "type": "string",
                    "description": "Absolute path to the structure the inputs should match.",
                },
                "request": {
                    "type": "string",
                    "description": (
                        "Scientific objective / calculation type "
                        "(e.g., 'static SCF for band structure', "
                        "'relaxation with vdW corrections'). Drives parameters."
                    ),
                },
                "software": {
                    "type": "string",
                    "description": (
                        "Optional engine override (e.g. 'vasp', 'qe'). "
                        "Defaults to the engine chosen by route_simulation."
                    ),
                },
                "method": {
                    "type": "string",
                    "description": (
                        "'llm' (default): AI-driven generation. A named "
                        "method (e.g. 'atomate2') uses a deterministic "
                        "backend from the engine's skill bundle."
                    ),
                },
            },
            required=["structure_path", "request"],
        )

        # =====================================================================
        # 10. RUN COMPLETE DFT WORKFLOW (one-shot shortcut)
        # =====================================================================
        def run_complete_dft_workflow(description: str,
                                      max_refinement_cycles: int = 4,
                                      vasp_generator_method: str = "llm") -> str:
            from .simulation_pipeline import run_complete_workflow

            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_complete_workflow(
                    description,
                    scale="periodic_dft",
                    software="vasp",
                    method=vasp_generator_method,
                    output_dir=str(workdir),
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                    mp_api_key=self.orch.mp_api_key,
                    max_refinement_cycles=max_refinement_cycles,
                )
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

            # Engine-neutral: take the structure path from the result and
            # build the input-files map from the generated inputs. The defensive
            # fallback uses the engine-neutral default filename (generation emits
            # extended XYZ), not a VASP POSCAR.
            structure_path = Path(
                structure_gen.get("final_structure_path") or (workdir / "structure.extxyz")
            )
            input_generation = result.get("input_generation", {}) or {}
            input_files = {
                fn: str(workdir / fn)
                for fn in (input_generation.get("input_files") or {})
                if (workdir / fn).exists()
            }

            # Record in session state (only if structure exists)
            if structure_path.exists():
                record = {
                    "slug": slug,
                    "description": description,
                    "structure_dir": str(workdir),
                    "structure_path": str(structure_path),
                    "script_path": structure_gen.get("final_script_path"),
                    "script_content": None,  # not surfaced by run_complete_workflow
                    "input_files": input_files,
                    "summary": input_generation.get("summary"),
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
                "generate_dft_inputs) instead."
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
        # 10b. RUN SIMULATION (engine-neutral one-shot, WITH execution)
        # =====================================================================
        def run_simulation(description: str, run_command: str = None,
                           scale: str = None, software: str = None,
                           structure_class: str = None,
                           max_refinement_cycles: int = 4,
                           max_run_cycles: int = 3,
                           run_timeout: int = 3600) -> str:
            from .simulation_pipeline import run_complete_workflow

            # 1. Route (scale, engine) if not supplied — reuse a prior routing
            #    decision so we don't re-route every call. Engine-neutral: any
            #    engine the router picks flows through run_complete_workflow.
            if not (scale and software):
                decision = getattr(self.orch, "routing_decision", None)
                if not decision:
                    from .simulation_router import SimulationRouter
                    decision = SimulationRouter(model=self.orch.model).route(
                        user_goal=description)
                    self.orch.routing_decision = decision
                scale = scale or decision.get("scale")
                software = software or decision.get("engine")

            # 2. run_command: user-provided > the engine skill's declared default
            #    (`default_run_command`) > None. The engine's own launch command
            #    lives in its skill bundle — never hardcoded here.
            rc_source = "user" if run_command else None
            if run_command is None and software:
                try:
                    from ...skills._shared._registry import get_tool_function
                    get_rc = get_tool_function("default_run_command",
                                               active_skills=[software])
                    run_command = get_rc()
                    rc_source = "skill" if run_command else None
                except Exception:
                    run_command = None

            # 3. Executor: local when a run_command is available and no HPC
            #    connection is attached. (HPC-submission via hpc_connection is a
            #    follow-up; with no executor the workflow still generates +
            #    validates inputs, it just does not run them.)
            executor = None
            if run_command and getattr(self.orch, "hpc_connection", None) is None:
                from .refinement import LocalExecutor
                executor = LocalExecutor(timeout=run_timeout)

            slug = self._make_slug(description)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_complete_workflow(
                    description,
                    scale=scale, software=software,
                    structure_class=structure_class,
                    output_dir=str(workdir),
                    api_key=self.orch.api_key, base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                    mp_api_key=self.orch.mp_api_key,
                    max_refinement_cycles=max_refinement_cycles,
                    max_run_cycles=max_run_cycles,
                    executor=executor, run_command=run_command,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error", "scale": scale, "engine": software,
                    "message": f"Simulation workflow failed: {e}",
                })

            final_status = result.get("final_status")
            structure_gen = result.get("structure_generation", {}) or {}
            structure_path = Path(
                structure_gen.get("final_structure_path")
                or (workdir / "structure.extxyz")
            )
            if structure_path.exists():
                self.orch.generated_structures.append({
                    "slug": slug, "description": description,
                    "structure_dir": str(workdir),
                    "structure_path": str(structure_path),
                    "scale": scale, "engine": software,
                    "created_at": datetime.now().isoformat(),
                })

            refinement = result.get("refinement") or {}
            executed = executor is not None
            return json.dumps({
                "status": final_status if final_status else "error",
                "scale": scale, "engine": software,
                "executed": executed,
                "run_command_source": rc_source,
                "refinement_status": refinement.get("status"),
                "output_directory": str(workdir),
                "note": (
                    None if executed else
                    "No run_command available (no engine binary on PATH and none "
                    "supplied) — generated and validated inputs only, did NOT "
                    "run. Pass run_command to execute."
                ),
            })

        self._register_tool(
            func=run_simulation,
            name="run_simulation",
            description=(
                "Engine-neutral one-shot: build + validate the structure, "
                "generate engine inputs, and RUN + refine the simulation — for "
                "any scale/engine (periodic DFT, classical MD, MLIP-MD). Routes "
                "(scale, engine) from the description when not given. Executes "
                "locally via the engine's own run command (from its skill "
                "bundle); pass `run_command` to override or if the skill's binary "
                "isn't found. Use this when the user wants the simulation "
                "actually RUN, not just its inputs prepared. Returns JSON "
                "(status, scale, engine, executed, refinement_status, "
                "output_directory)."
            ),
            parameters={
                "description": {
                    "type": "string",
                    "description": "Natural-language system + goal to simulate.",
                },
                "run_command": {
                    "type": "string",
                    "description": (
                        "Optional run-command template with a `{script}` "
                        "placeholder (e.g. 'lmp -in {script}'). Overrides the "
                        "engine skill's default; supply this if the default "
                        "binary is missing or fails."
                    ),
                },
                "scale": {
                    "type": "string",
                    "description": ("Optional scale override "
                                    "('periodic_dft' | 'molecular_dynamics'); "
                                    "routed from the description if omitted."),
                },
                "software": {
                    "type": "string",
                    "description": ("Optional engine override "
                                    "('vasp' | 'lammps'); routed if omitted."),
                },
                "structure_class": {
                    "type": "string",
                    "description": (
                        "Optional structure-class override ('crystal' | "
                        "'molecular' | 'condensed' | 'biomolecular'). Omit and it "
                        "is derived from the scale, where the molecular_dynamics "
                        "default 'condensed' means a liquid / solution / solvated "
                        "box. MD also covers crystalline and biomolecular systems, "
                        "and the derived default is wrong for those — so SET IT "
                        "EXPLICITLY from the task: 'crystal' for a crystalline "
                        "solid, a melt-from-crystal, or a slab interface (e.g. "
                        "melting Cu, Li diffusion in LiCoO2, water on a TiO2 slab); "
                        "'biomolecular' for a protein. periodic_dft derives "
                        "'crystal', molecular_qc derives 'molecular'."
                    ),
                },
                "max_run_cycles": {
                    "type": "integer",
                    "description": "Max run → assess → fix cycles per phase (default 3).",
                },
            },
            required=["description"],
        )

        # =====================================================================
        # 3. REFINE STRUCTURE
        # =====================================================================
        def refine_structure(structure_path: str, original_request: str) -> str:
            record = self._find_structure_record(structure_path)
            if record is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "Refinement requires a structure that was generated "
                        "in this session (so the validator feedback and "
                        "prior script are available). No record found for: "
                        f"{structure_path}. Generate the structure first via "
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

            prior_script = record.get("script_content") or self._find_script_content(structure_path)
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

            new_structure_path = result["output_file"]
            new_script_path = result["final_script_path"]
            new_script_content = result["final_script_content"]
            n_atoms = self._count_atoms(new_structure_path)

            # Update the record in place rather than appending — refinement
            # produces a successor of the same logical structure.
            record["structure_path"] = new_structure_path
            record["script_path"] = new_script_path
            record["script_content"] = new_script_content
            record["validation"] = None  # invalidate prior validation
            record["input_files"] = {}    # invalidate prior inputs (geometry changed)
            record["summary"] = None

            return json.dumps({
                "status": "success",
                "slug": record["slug"],
                "structure_path": new_structure_path,
                "script_path": new_script_path,
                "n_atoms": n_atoms,
                "next_steps": (
                    "Optionally call validate_structure again to confirm the "
                    "refinement addressed the prior issues; then proceed to "
                    "generate_dft_inputs."
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
                "structure_path": {
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
            required=["structure_path", "original_request"],
        )

        # =====================================================================
        # 4. VIEW STRUCTURE
        # =====================================================================
        def view_structure(structure_path: str) -> str:
            from .utils import generate_structure_views

            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"POSCAR not found: {structure_path}",
                })

            try:
                image_paths = generate_structure_views(structure_path)
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
                "structure_path": {
                    "type": "string",
                    "description": "Absolute path to the POSCAR to render.",
                },
            },
            required=["structure_path"],
        )

        # =====================================================================
        # 6. VALIDATE INPUTS (engine-neutral, pre-run)
        # =====================================================================
        def validate_inputs(structure_path: str, system_description: str,
                            software: str = None) -> str:
            skill, domain = self._resolve_engine(software)
            if not skill:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No engine selected. Call route_simulation first, or "
                        "pass `software` explicitly (e.g. 'vasp')."
                    ),
                })

            record = self._find_structure_record(structure_path)
            input_files = self._record_input_files(record)
            if not input_files:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No input files found for this structure. Run "
                        "generate_dft_inputs (or generate_md_inputs) first."
                    ),
                })

            try:
                validator = self._get_input_validator()
                report = validator.validate(
                    input_files=input_files,
                    system_description=system_description,
                    skill=skill,
                    domain=domain,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Input validation failed: {e}",
                })

            if record is not None:
                record["input_validation"] = report
            report.setdefault("status", "success")
            report["engine"] = skill
            return json.dumps(report, default=str)

        self._register_tool(
            func=validate_inputs,
            name="validate_inputs",
            description=(
                "Pre-run review of the generated input files for a structure, "
                "engine-neutral. Routes to the InputValidator critic, which "
                "combines the active engine skill's validation guidance, the "
                "engine's deterministic syntax check, and — when a FutureHouse "
                "key is configured — a literature-grounded review, returning "
                "suggested adjustments. The engine is taken from the active "
                "routing decision unless `software` is given. Use after "
                "generating inputs and before submitting a run."
            ),
            parameters={
                "structure_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the structure's POSCAR, used to "
                        "locate its generated input files in the session."
                    ),
                },
                "system_description": {
                    "type": "string",
                    "description": (
                        "What system the inputs are for and what the "
                        "calculation should compute — context for judging "
                        "whether the parameter choices are appropriate."
                    ),
                },
                "software": {
                    "type": "string",
                    "description": (
                        "Optional engine override (e.g. 'vasp'). Defaults to "
                        "the engine chosen by route_simulation."
                    ),
                },
            },
            required=["structure_path", "system_description"],
        )

        # =====================================================================
        # 7. APPLY INCAR IMPROVEMENTS
        # =====================================================================
        def apply_input_adjustments(structure_path: str,
                                    original_request: str,
                                    suggested_adjustments: list,
                                    software: str = None,
                                    overall_assessment: str = "") -> str:
            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Structure file not found: {structure_path}",
                })
            if not suggested_adjustments:
                return json.dumps({
                    "status": "no_changes",
                    "message": "No adjustments provided — nothing to apply.",
                })

            skill, domain = self._resolve_engine(software)
            if not skill:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No engine selected. Call route_simulation first, or "
                        "pass `software` explicitly (e.g. 'vasp')."
                    ),
                })

            record = self._find_structure_record(structure_path)
            original_inputs = self._record_input_files(record)
            if not original_inputs:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No generated inputs to adjust. Run generate_dft_inputs "
                        "(or generate_md_inputs) first."
                    ),
                })

            # The engine-neutral apply lives on the periodic-DFT foundation
            # agent (software-agnostic across vasp/qe). Other scales gain
            # their own apply when their foundation agent implements it.
            if domain != "periodic_dft":
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"apply_input_adjustments is not yet available for "
                        f"scale '{domain}'."
                    ),
                })

            try:
                from .periodic_dft_agent import PeriodicDFTAgent
                agent = PeriodicDFTAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to construct generator agent: {e}",
                })

            output_dir = str(Path(structure_path).parent)
            result = agent.apply_improvements(
                original_inputs=original_inputs,
                validation_result={
                    "validation_status": "needs_adjustment",
                    "suggested_adjustments": suggested_adjustments,
                    "overall_assessment": overall_assessment,
                },
                structure_file=structure_path,
                request=original_request,
                output_dir=output_dir,
                software=skill,
            )

            if result.get("status") not in ("success", "no_changes"):
                return json.dumps({
                    "status": "error",
                    "message": result.get("message") or "Apply-adjustments failed",
                })

            improved = result.get("improved_paths") or {}
            if record is not None and improved:
                record["input_files"] = dict(improved)

            return json.dumps({
                "status": result.get("status"),
                "engine": skill,
                "improved_paths": improved,
            })

        self._register_tool(
            func=apply_input_adjustments,
            name="apply_input_adjustments",
            description=(
                "Apply a list of validated parameter adjustments to a "
                "structure's generated inputs, writing improved files next to "
                "the originals and updating the session record. Engine-neutral: "
                "`software` selects the engine (defaults to the routing "
                "decision). Pair with validate_inputs — pass its "
                "suggested_adjustments through directly."
            ),
            parameters={
                "structure_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the structure whose generated inputs "
                        "should be adjusted (provides system context)."
                    ),
                },
                "original_request": {
                    "type": "string",
                    "description": "The original calculation-type request.",
                },
                "suggested_adjustments": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Adjustment dicts in the shape validate_inputs returns "
                        "(each with file/key/current/suggested/reason)."
                    ),
                },
                "software": {
                    "type": "string",
                    "description": (
                        "Optional engine override (e.g. 'vasp', 'qe'). "
                        "Defaults to the engine chosen by route_simulation."
                    ),
                },
                "overall_assessment": {
                    "type": "string",
                    "description": "Brief validation summary (passed verbatim).",
                },
            },
            required=["structure_path", "original_request", "suggested_adjustments"],
        )

        # =====================================================================
        # 10b. EDIT / RENAME FILE — surgical, byte-exact, no LLM regeneration
        # =====================================================================
        # Editable input/deck formats: "" covers extensionless VASP inputs
        # (INCAR/POSCAR/KPOINTS/POTCAR); the rest are LAMMPS/QE decks and job
        # scripts. On top of the shared text defaults.
        _SIM_EDIT_SUFFIXES = {
            "", ".lammps", ".data", ".in", ".param", ".mod", ".inc",
            ".sbatch", ".pwi", ".lmp",
        }

        def edit_file(path: str, old_text: str = None, new_text: str = None,
                      replace_all: bool = False, edits: list = None) -> str:
            """Mechanical in-place edit of a generated input/deck file:
            replace exact text snippets, one old/new pair or a batched `edits`
            list applied atomically. For retuning several coupled parameters,
            use apply_input_adjustments instead."""
            print(f"  ⚡ Tool: Editing file '{path}'...")
            try:
                from ...utils.file_edit import (apply_surgical_edits,
                                                 DEFAULT_EDITABLE_SUFFIXES)
                if not edits:
                    if old_text is None or new_text is None:
                        return json.dumps({"status": "error", "message": (
                            "Provide old_text+new_text or a non-empty edits "
                            "list.")})
                    edits = [{"old_text": old_text, "new_text": new_text,
                              "replace_all": replace_all}]
                root = Path(self.orch.base_dir).resolve()
                rp = Path(path)
                if not rp.is_absolute():
                    rp = root / rp
                rp = rp.resolve()
                out = apply_surgical_edits(
                    rp, edits, root=root, backup_dir=rp.parent,
                    allowed_suffixes=DEFAULT_EDITABLE_SUFFIXES | _SIM_EDIT_SUFFIXES,
                    not_found_message=(
                        "the snippet must match the file byte for byte. "
                        "For periodic_dft, apply_input_adjustments retunes "
                        "tags without needing an exact match and is the "
                        "route to take when a verbatim guess misses."),
                    too_large_message=(
                        "Edit too large for edit_file. For a broader input "
                        "change — retuning several coupled parameters, or "
                        "regenerating a deck — use apply_input_adjustments. For "
                        "a verbatim insertion, split the text at unique "
                        "boundaries into snippets under the cap and pass "
                        "them TOGETHER as one `edits` list in a single call."),
                )
                if out["status"] == "success":
                    n = out.get("n_edits", 1)
                    print(f"    ✏️  Edited in place"
                          f"{f' ({n} edits)' if n > 1 else ''}: {rp.name}")
                return json.dumps(out)
            except Exception as e:
                logging.error(f"edit_file failed: {e}", exc_info=True)
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=edit_file,
            name="edit_file",
            description=(
                "Surgically edit a generated input or deck file IN PLACE by "
                "replacing exact text snippets — one INCAR tag, a timestep, a "
                "thermostat damping, a pair_style cutoff. The canonical use is "
                "'change ENCUT to 520 in this INCAR' without regenerating the "
                "whole input. Copy old_text VERBATIM from the file; each "
                "snippet is capped at 2000 characters. Several "
                "changes to ONE file go in a single call as the `edits` list "
                "(atomic, in order). Retuning several coupled parameters at "
                "once, or a change that needs the input rebuilt, is "
                "apply_input_adjustments' job, not this. A pre-edit backup is "
                "kept automatically."
            ),
            parameters={
                "path": {"type": "string", "description": (
                    "Path of the file to edit — absolute, or relative to the "
                    "session directory. Must be inside the session.")},
                "old_text": {"type": "string", "description": (
                    "Single-edit form: the exact snippet to replace, copied "
                    "VERBATIM including whitespace. Must match exactly one "
                    "place unless replace_all. Omit when passing `edits`.")},
                "new_text": {"type": "string", "description": (
                    "Single-edit form: the replacement text. Omit with "
                    "`edits`.")},
                "replace_all": {"type": "boolean", "description": (
                    "Single-edit form: replace every occurrence instead of "
                    "requiring a unique match. Default false.")},
                "edits": {
                    "type": "array",
                    "items": {"type": "object", "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean"}},
                        "required": ["old_text", "new_text"]},
                    "description": (
                        "Batched form: ALL changes for this file in one call, "
                        "applied in order; all-or-nothing — if one edit fails "
                        "to match, nothing is applied.")},
            },
            required=["path"],
        )

        def rename_file(path: str, new_name: str, copy: bool = False) -> str:
            """Byte-exact rename (or copy) of a generated file within its own
            directory — the content never passes through the model."""
            print(f"  ⚡ Tool: {'Copying' if copy else 'Renaming'} "
                  f"'{path}' → '{new_name}'...")
            try:
                from ...utils.file_edit import rename_or_copy_file
                root = Path(self.orch.base_dir).resolve()
                rp = Path(path)
                if not rp.is_absolute():
                    rp = root / rp
                rp = rp.resolve()
                safe = Path(new_name).name
                if not safe:
                    return json.dumps({"status": "error",
                                       "message": "Invalid new_name."})
                dest = (rp.parent / safe).resolve()
                if dest == rp:
                    return json.dumps({"status": "error", "message": (
                        f"'{safe}' is already this file's name — nothing to "
                        f"do.")})
                out = rename_or_copy_file(rp, dest, root=root, copy=copy)
                if out["status"] == "success":
                    print(f"    📛 {'Copied' if copy else 'Renamed'}: "
                          f"{Path(out['path']).name}")
                return json.dumps(out)
            except Exception as e:
                logging.error(f"rename_file failed: {e}", exc_info=True)
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=rename_file,
            name="rename_file",
            description=(
                "Rename or copy a generated file BYTE-EXACTLY within its own "
                "directory — the content never passes through the model. Use "
                "for a byte-exact rename (copy=false) or an exact duplicate "
                "(copy=true); never reconstruct a file by regenerating it to "
                "rename it, which risks perturbing the content."
            ),
            parameters={
                "path": {"type": "string", "description": (
                    "Path of the file to rename — absolute or relative to the "
                    "session directory.")},
                "new_name": {"type": "string", "description": (
                    "The new bare filename (kept in the file's own "
                    "directory).")},
                "copy": {"type": "boolean", "description": (
                    "Copy instead of rename (default false).")},
            },
            required=["path", "new_name"],
        )

        # =====================================================================
        # 10c. SAVE / APPEND / READ FILE — generic session-dir file I/O
        # =====================================================================
        # These mirror the generic file tools the analysis/planning
        # orchestrators expose (save_file, append_file, read_file,
        # read_document): the simulation agent can persist a report or a
        # scratch script, read back a log/deck/config, or extract text from a
        # provided document — the same non-domain-specific surface, without
        # touching the domain tools (generate_structure, run_simulation, …).
        def save_file(filename: str, content: str, subfolder: str = "") -> str:
            """Save text content (reports, notes, small scripts) to a NEW file
            in the session directory. Use edit_file to change an existing
            file; use append_file to grow a large file in chunks."""
            print(f"  ⚡ Tool: Saving file '{filename}'...")

            # Sanitise: strip path separators from filename to prevent traversal.
            safe_name = Path(filename).name
            if not safe_name:
                return json.dumps({
                    "status": "error",
                    "message": "Invalid filename.",
                })

            target_dir = Path(self.orch.base_dir)
            if subfolder:
                safe_sub = Path(subfolder).name
                target_dir = target_dir / safe_sub
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / safe_name

            try:
                dest.write_text(content, encoding="utf-8")
                print(f"    💾 Saved: {dest}")
                return json.dumps({
                    "status": "success",
                    "path": str(dest),
                    "size_bytes": dest.stat().st_size,
                })
            except Exception as e:
                logging.error(f"save_file failed: {e}")
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                })

        self._register_tool(
            func=save_file,
            name="save_file",
            description=(
                "Save text content (a report, summary, notes, or a small "
                "script you have already composed) to a NEW file in the "
                "session directory. To change an EXISTING file, use edit_file "
                "for a snippet swap or rename_file to change its name — those "
                "keep the content byte-safe. Large content may not survive the "
                "trip as a single tool-call argument — for anything long "
                "(roughly >100 lines), save the first chunk with save_file and "
                "the rest with append_file. Do NOT use this to hand-write "
                "simulation inputs or structures — generate_structure, "
                "generate_dft_inputs, and run_simulation build those."
            ),
            parameters={
                "filename": {
                    "type": "string",
                    "description": (
                        "Name of the file to create, e.g. 'run_notes.md', "
                        "'energies.csv', or 'analysis.py'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file.",
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Optional subfolder within the session directory, "
                        "e.g. 'reports' or 'scripts'. Created if it doesn't "
                        "exist."
                    ),
                },
            },
            required=["filename", "content"],
        )

        def append_file(filename: str, content: str, subfolder: str = "") -> str:
            """Append text content to a file in the session directory (created
            if it doesn't exist). Companion to save_file for chunked writes of
            large content."""
            print(f"  ⚡ Tool: Appending to file '{filename}'...")

            safe_name = Path(filename).name
            if not safe_name:
                return json.dumps({
                    "status": "error",
                    "message": "Invalid filename.",
                })

            target_dir = Path(self.orch.base_dir)
            if subfolder:
                safe_sub = Path(subfolder).name
                target_dir = target_dir / safe_sub
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / safe_name

            try:
                with open(dest, "a", encoding="utf-8") as f:
                    f.write(content)
                print(f"    💾 Appended: {dest}")
                return json.dumps({
                    "status": "success",
                    "path": str(dest),
                    "size_bytes": dest.stat().st_size,
                })
            except Exception as e:
                logging.error(f"append_file failed: {e}")
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                })

        self._register_tool(
            func=append_file,
            name="append_file",
            description=(
                "Append text content to a file in the session directory "
                "(created if it doesn't exist). Use together with save_file "
                "to write large files in chunks — save_file for the first "
                "chunk, then append_file for each subsequent chunk — keeping "
                "every chunk small enough to pass reliably as a tool argument."
            ),
            parameters={
                "filename": {
                    "type": "string",
                    "description": "Name of the file to append to.",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to append to the file.",
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Optional subfolder within the session directory, "
                        "e.g. 'reports' or 'scripts'. Created if it doesn't "
                        "exist."
                    ),
                },
            },
            required=["filename", "content"],
        )

        # Some files are read for their whole content, never their head (a
        # report's sections sit in sequence, so a head-only read hides all but
        # the first). These get the whole body up to a cap; past it the read
        # truncates but emits the section outline so offset= reaches the rest.
        _FULL_READ_STEMS = ("report", "summary", "literature_search")
        _FULL_READ_MAX_CHARS = 250_000

        def read_file(file_path: str, max_lines: int = 200,
                      tail: bool = False, search: str = None,
                      offset: int = None) -> str:
            """Read and return the contents of a text/JSON/CSV/log file — an
            INCAR, a POSCAR, a LAMMPS deck, an OUTCAR/log, a job script, a
            generated report — without triggering any simulation. PDF/DOCX are
            extracted to text. Reads from the top by default; use offset to
            jump to the middle, tail to read the end, or search to find a
            pattern."""
            print(f"  ⚡ Tool: Reading file '{file_path}'...")

            # Resolve path: absolute as-is, else relative to the session dir.
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(self.orch.base_dir) / path
            if not path.is_file():
                return json.dumps({
                    "status": "error",
                    "message": f"Not a file: {file_path}",
                })

            try:
                ext = path.suffix.lower()

                # Size guard — skip for Excel/CSV since we cap at 100 rows × 40
                # cols. Documents get more headroom: extraction is page-based
                # and a figure-heavy PDF is megabytes of images, not of text.
                if ext not in ('.xlsx', '.xls', '.csv'):
                    size_mb = path.stat().st_size / (1024 * 1024)
                    cap_mb = 25 if ext in ('.pdf', '.docx') else 5
                    if size_mb > cap_mb:
                        return json.dumps({
                            "status": "error",
                            "message": f"File too large ({size_mb:.1f} MB).",
                        })

                if ext == ".json":
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    content = json.dumps(data, indent=2)
                    return json.dumps({
                        "status": "success",
                        "file_path": str(path),
                        "content": content,
                    })

                if ext in ('.xlsx', '.xls', '.csv'):
                    MAX_PREVIEW_ROWS = 100
                    MAX_PREVIEW_COLS = 40
                    MAX_PREVIEW_CHARS = 30000
                    if ext == '.csv':
                        df_preview = pd.read_csv(path, nrows=MAX_PREVIEW_ROWS)
                        with open(path) as _f:
                            total_rows = sum(1 for _ in _f) - 1
                    else:
                        df_preview = pd.read_excel(path, nrows=MAX_PREVIEW_ROWS)
                        try:
                            import openpyxl
                            _wb = openpyxl.load_workbook(path, read_only=True)
                            total_rows = _wb.active.max_row - 1
                            _wb.close()
                        except Exception:
                            total_rows = len(df_preview)
                    total_cols = len(df_preview.columns)
                    display_df = df_preview.iloc[:, :MAX_PREVIEW_COLS]
                    preview_text = display_df.to_string()
                    if len(preview_text) > MAX_PREVIEW_CHARS and len(display_df) > 5:
                        ratio = MAX_PREVIEW_CHARS / len(preview_text)
                        fewer_rows = max(5, int(len(display_df) * ratio))
                        display_df = display_df.iloc[:fewer_rows]
                        preview_text = display_df.to_string()
                        if len(preview_text) > MAX_PREVIEW_CHARS:
                            preview_text = preview_text[:MAX_PREVIEW_CHARS] + "\n... (truncated)"
                    shown_rows = len(display_df)
                    shown_cols = len(display_df.columns)
                    trunc_parts = []
                    if shown_rows < total_rows:
                        trunc_parts.append(f"first {shown_rows} rows")
                    if shown_cols < total_cols:
                        trunc_parts.append(f"first {shown_cols} columns")
                    trunc = f" (showing {', '.join(trunc_parts)})" if trunc_parts else ""
                    content = f"Shape: {total_rows} rows × {total_cols} columns{trunc}\n\n{preview_text}"
                    return json.dumps({
                        "status": "success",
                        "file_path": str(path),
                        "content": content,
                    })

                doc_meta = {}
                if ext in ('.pdf', '.docx'):
                    # Opened as text a PDF returns its compressed byte streams;
                    # route through the shared extractor instead (table-aware,
                    # OCR fallback via the session model).
                    from ...parsers.extract import extract_text
                    info = extract_text(
                        str(path), ocr_model=getattr(self.orch, "model", None))
                    raw = info.get("text") or ""
                    if not raw.strip():
                        return json.dumps({
                            "status": "error",
                            "message": (
                                f"No extractable text in {path.name} "
                                "(empty or image-only document)."),
                        })
                    lines = raw.splitlines(keepends=True)
                    doc_meta = {k: info[k] for k in
                                ("n_pages", "n_ocr_pages", "n_paragraphs")
                                if info.get(k) is not None}
                    doc_meta["extracted"] = ext.lstrip(".")
                else:
                    with open(path, 'r', encoding='utf-8',
                              errors='replace') as f:
                        lines = f.readlines()
                total = len(lines)

                if search:
                    # The real question behind most repeat reads is "is X in
                    # here, and where" — a search, not a read. Answer it
                    # directly, cheaply, however long the file is.
                    try:
                        rx = re.compile(search, re.I)
                    except re.error as e:
                        return json.dumps({
                            "status": "error",
                            "message": f"Invalid search pattern: {e}"})
                    hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
                    CAP = 40
                    shown, out = hits[:CAP], []
                    for i in shown:
                        lo, hi = max(0, i - 1), min(total, i + 2)
                        out.append(f"@@ line {i + 1}\n"
                                   + "".join(lines[lo:hi]).rstrip("\n"))
                    body = "\n\n".join(out) if out else "(no matches)"
                    note = (f"{len(hits)} matching line(s) in {total} total"
                            + (f"; showing the first {CAP}" if len(hits) > CAP
                               else ""))
                    return json.dumps({
                        "status": "success",
                        "file_path": str(path),
                        "mode": "search",
                        "pattern": search,
                        "matches": len(hits),
                        "match_lines": [i + 1 for i in shown],
                        "total_lines": total,
                        "content": f"{note}\n\n{body}",
                        **doc_meta,
                    })

                # A truncated read must say what it is missing and where, or the
                # agent cannot tell a short file from a short read.
                def _outline():
                    heads = [(i + 1, ln.strip()) for i, ln in
                             enumerate(lines) if ln.startswith('#')]
                    if len(heads) < 2:
                        return ""
                    return "\nSections: " + " · ".join(
                        f"{h.lstrip('# ')[:44]} @ line {n}"
                        for n, h in heads[:12]) + (
                        " …" if len(heads) > 12 else "")

                # Files read for their whole content, not their head.
                whole = (any(s in path.name.lower()
                             for s in _FULL_READ_STEMS)
                         and offset is None and not tail
                         and len("".join(lines)) <= _FULL_READ_MAX_CHARS)

                truncated = True
                if whole or total <= max_lines:
                    first, last, content = 1, total, "".join(lines)
                    truncated = False
                elif offset is not None:
                    start = max(0, offset - 1)
                    shown = lines[start:start + max_lines]
                    first, last = start + 1, min(total, start + max_lines)
                    content = "".join(shown) + (
                        f"\n... (showing lines {first}-{last} of {total}."
                        f"{_outline()})")
                elif tail:
                    shown = lines[-max_lines:]
                    first, last = total - max_lines + 1, total
                    more = (f"... ({first - 1} earlier lines not shown; "
                            f"omit tail to read from the top)")
                    content = more + "\n" + "".join(shown)
                else:
                    shown = lines[:max_lines]
                    first, last = 1, max_lines
                    content = "".join(shown) + (
                        f"\n... ({total - max_lines} more lines not "
                        f"shown — this is a TRUNCATED READ, not the whole "
                        f"file. Jump to any part with offset=<line>; read "
                        f"the END with tail=true; find something with "
                        f"search='<pattern>'; or raise max_lines."
                        f"{_outline()})")

                return json.dumps({
                    "status": "success",
                    "file_path": str(path),
                    "mode": "tail" if tail else "head",
                    "total_lines": total,
                    "shown_lines": f"{first}-{last}",
                    "truncated": truncated,
                    "content": content,
                    **doc_meta,
                })

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to read file: {e}",
                })

        self._register_tool(
            func=read_file,
            name="read_file",
            description=(
                "Read a text, JSON, CSV, or log file — an INCAR, POSCAR, "
                "KPOINTS, a LAMMPS deck, an OUTCAR / vasprun / MD log, a job "
                "script, or a report generated this session. PDF and Word "
                "documents are extracted to text automatically (tables "
                "preserved, scanned pages OCR'd). Reads from the TOP by "
                "default. For a long file do not read it repeatedly hoping to "
                "see more — you will get the same lines back. A truncated read "
                "lists the section headings and their line numbers: use "
                "offset=<line> to jump to one, search='<pattern>' to find "
                "where something is (and whether it is there at all), or "
                "tail=true to read the END (e.g. whether a run log ends in "
                "success). Path is absolute or relative to the session "
                "directory."
            ),
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (default: 200).",
                },
                "tail": {
                    "type": "boolean",
                    "description": (
                        "Read the LAST max_lines lines instead of the first. "
                        "Use to check how a long log or file ENDS — that a run "
                        "converged, that a file was not truncated."
                    ),
                },
                "search": {
                    "type": "string",
                    "description": (
                        "Case-insensitive regex. Returns every matching line "
                        "with its line number and one line of context either "
                        "side, plus the total match count — instead of the "
                        "file body. The right tool for 'did this run raise an "
                        "error', 'which lines set ENCUT', 'is NELM in here'. "
                        "Far cheaper than reading a long file, and it answers "
                        "presence/absence definitively."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "1-based line to start reading from — the way to read "
                        "the MIDDLE of a file. A truncated read lists the "
                        "section headings with their line numbers; pass one "
                        "here to jump straight there. Do NOT re-read from the "
                        "top hoping to see more — you will get the same lines "
                        "back."
                    ),
                },
            },
            required=["file_path"],
        )

        def read_document(paths) -> str:
            """Read one or more PDF / DOCX / MD / TXT documents and return the
            combined extracted text — a methods paper, a protocol, prior notes
            the user provided."""
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                return json.dumps({
                    "status": "error",
                    "message": "No document path provided.",
                })
            print(f"  📄 Tool: Reading {len(paths)} document(s)...")
            docs, errors = [], []
            for p in paths:
                dp = Path(p)
                if not dp.is_absolute():
                    dp = Path(self.orch.base_dir) / dp
                if not dp.is_file():
                    errors.append(f"Not a file: {p}")
                    continue
                try:
                    docs.append((dp, _extract_document_text(
                        dp, ocr_model=getattr(self.orch, "model", None))))
                except ValueError as e:
                    errors.append(str(e))
                except Exception as e:
                    logging.error(f"read_document failed for {p}: {e}")
                    errors.append(f"Could not read {dp.name}: {e}")
            if not docs:
                return json.dumps({
                    "status": "error",
                    "message": "No documents could be read.",
                    "errors": errors,
                })
            combined = "\n\n---\n\n".join(
                f"## {dp.name}\n\n{info['text']}" for dp, info in docs
            )
            combined_truncated = len(combined) > _READ_DOC_MAX_CHARS
            if combined_truncated:
                combined = combined[:_READ_DOC_MAX_CHARS]
            n_ocr = sum(info.get("n_ocr_pages", 0) for _, info in docs)
            return json.dumps({
                "status": "success",
                "n_documents": len(docs),
                "n_ocr_pages": n_ocr,
                "ocr_note": (
                    f"{n_ocr} scanned page(s) had no text layer and were "
                    "transcribed by vision-OCR — verify any figures/numerics."
                ) if n_ocr else None,
                "documents": [
                    {"name": dp.name,
                     **{k: v for k, v in info.items() if k != "text"}}
                    for dp, info in docs
                ],
                "errors": errors or None,
                "combined_truncated": combined_truncated,
                "text": combined,
            })

        self._register_tool(
            func=read_document,
            name="read_document",
            description=(
                "Read one or more documents the user provided — PDF, DOCX, "
                "Markdown, or text files (a methods paper, a computational "
                "protocol, a prior report, notes). Returns the combined "
                "extracted text straight into context: tables are preserved "
                "and scanned pages are OCR'd. For a handful of documents you "
                "want to read in full. To inspect a single local file (a deck, "
                "a log, a config) prefer read_file, which supports "
                "offset/tail/search over long files. Pass absolute paths (or "
                "paths relative to the session directory)."
            ),
            parameters={
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Path(s) to the document(s) to read (.pdf, .docx, .md, "
                        "or .txt). Multiple documents are combined into one "
                        "text block."
                    ),
                },
            },
            required=["paths"],
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
                        "structure_path": s.get("structure_path"),
                        "input_files": s.get("input_files") or {},
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
        # 8. ANALYZE OUTPUT (engine-neutral, post-run)
        # =====================================================================
        def analyze_output(output_dir: str, research_goal: str,
                           software: str = None, fixes_mode: str = "auto") -> str:
            skill, domain = self._resolve_engine(software)
            if not skill:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No engine selected. Call route_simulation first, or "
                        "pass `software` explicitly (e.g. 'vasp')."
                    ),
                })

            try:
                critic = self._get_run_critic()
                report = critic.assess(
                    output_dir=output_dir,
                    research_goal=research_goal,
                    skill=skill,
                    domain=domain,
                    fixes_mode=fixes_mode,
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Run analysis failed: {e}",
                })

            report.setdefault("status", "success")
            report["engine"] = skill
            return json.dumps(report, default=str)

        self._register_tool(
            func=analyze_output,
            name="analyze_output",
            description=(
                "Post-run review of a finished calculation directory, "
                "engine-neutral. Routes to the RunCritic, which reads the "
                "engine's output files and the active skill's interpretation "
                "guidance to return a verdict (good / warning / poor / "
                "needs_fixes), the run status, reasoning, and — when the run "
                "failed or the result is unsatisfactory — proposed patched "
                "input files. Handles both failed and successful runs in one "
                "call. The engine is taken from the active routing decision "
                "unless `software` is given. Use after the user runs the "
                "calculation and points you at the run directory."
            ),
            parameters={
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Absolute path to the finished run's output "
                        "directory (engine-specific contents, e.g. "
                        "vasprun.xml / OUTCAR / logs for VASP)."
                    ),
                },
                "research_goal": {
                    "type": "string",
                    "description": (
                        "What the calculation was meant to compute — drives "
                        "whether the result is sufficient for the intent and "
                        "what fixes to suggest."
                    ),
                },
                "software": {
                    "type": "string",
                    "description": (
                        "Optional engine override (e.g. 'vasp'). Defaults to "
                        "the engine chosen by route_simulation."
                    ),
                },
                "fixes_mode": {
                    "type": "string",
                    "enum": ["auto", "always", "skip"],
                    "description": (
                        "When to propose patched inputs: 'auto' (default; "
                        "only on failure or a poor verdict), 'always' "
                        "(whenever below 'good'), or 'skip' (verdict only)."
                    ),
                },
            },
            required=["output_dir", "research_goal"],
        )

        # =====================================================================
        # 12. SUBMIT VASP JOB
        # =====================================================================
        def submit_simulation_job(
            structure_slug: str,
            remote_dir: str,
            run_command: str,
            job_name: str = "sim",
            partition: str = "",
            n_nodes: int = 1,
            n_tasks: int = 16,
            time_limit: str = "04:00:00",
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

            structure_file = record.get("structure_path")
            input_files = record.get("input_files") or {}
            if not structure_file or not Path(structure_file).exists():
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No local structure file to upload. Run "
                        "generate_structure first."
                    ),
                })
            if not input_files:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "No generated input files to upload. Run "
                        "generate_dft_inputs (or generate_md_inputs) first."
                    ),
                })
            missing = [fn for fn, p in input_files.items()
                       if not p or not Path(p).exists()]
            if missing:
                return json.dumps({
                    "status": "error",
                    "message": f"Missing local input files before upload: {missing}.",
                })

            try:
                conn.mkdir_p(remote_dir)
                # Upload the structure under its own filename + every input file.
                conn.upload(structure_file,
                            f"{remote_dir}/{Path(structure_file).name}")
                for fname, local_path in input_files.items():
                    conn.upload(local_path, f"{remote_dir}/{fname}")

                script_content = self._generate_job_script(
                    sched=sched,
                    job_name=job_name,
                    n_nodes=n_nodes,
                    n_tasks=n_tasks,
                    time_limit=time_limit,
                    partition=partition,
                    run_command=run_command,
                    modules=modules,
                    extra_directives=extra_directives,
                )
                local_script = Path(record["structure_dir"]) / "job.sh"
                local_script.write_text(script_content, encoding="utf-8")
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
                    f"download_job_results('{job_id}') to retrieve outputs."
                ),
            })

        self._register_tool(
            func=submit_simulation_job,
            name="submit_simulation_job",
            description=(
                "Upload a structure and its generated input files to a remote "
                "HPC cluster and submit a job via the active scheduler "
                "(SLURM / PBS / LSF). Engine-neutral: uploads whatever inputs "
                "were generated for the structure plus the structure file. "
                "Requires hpc_connection and hpc_scheduler on the orchestrator, "
                "and that inputs were generated first (generate_dft_inputs / "
                "generate_md_inputs). The engine's run command is supplied "
                "via `run_command`."
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
                "run_command": {
                    "type": "string",
                    "description": (
                        "Full engine run command including MPI launcher, "
                        "written verbatim into the job script (e.g. "
                        "'srun vasp_std', 'mpirun -np 16 lmp -in run.lammps'). "
                        "Engine-specific — there is no default."
                    ),
                },
                "job_name": {
                    "type": "string",
                    "description": "Scheduler job name (default: 'sim').",
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
                "modules": {
                    "type": "string",
                    "description": (
                        "Shell commands to load the engine environment, written "
                        "verbatim into the job script (e.g. "
                        "'module load vasp/6.3.2 intel/2023'). Omit if the "
                        "user's .bashrc already loads it."
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
            required=["structure_slug", "remote_dir", "run_command"],
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
                    f"download_job_results('{job_id}') to retrieve outputs."
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
                "job has reached a terminal state. Use after submit_simulation_job "
                "to check progress."
            ),
            parameters={
                "job_id": {
                    "type": "string",
                    "description": "Scheduler job ID returned by submit_simulation_job.",
                },
            },
            required=["job_id"],
        )

        # =====================================================================
        # 14. DOWNLOAD VASP RESULTS
        # =====================================================================
        def download_job_results(job_id: str, local_dir: str = "") -> str:
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
                        "Only jobs submitted via submit_simulation_job in this "
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
            func=download_job_results,
            name="download_job_results",
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
                    "description": "Scheduler job ID returned by submit_simulation_job.",
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

            # Run post-run analysis if results are available — engine-neutral
            # via the active skill's snapshot_run tool. Only include the
            # analysis section when the snapshot reflects a real run.
            run_snapshot = None
            results_dir = record.get("hpc_results_dir") or record.get("structure_dir")
            skill, _domain = self._resolve_engine(None)
            if results_dir and skill:
                try:
                    from ...skills._shared._registry import get_tool_function
                    snap = get_tool_function("snapshot_run", active_skills=[skill])
                    snapshot = snap(results_dir)
                    if (snapshot.get("status") == "ok"
                            and (snapshot.get("files_found")
                                 or snapshot.get("convergence_status", "unknown") != "unknown")):
                        run_snapshot = snapshot
                except LookupError:
                    run_snapshot = None
                except Exception as e:
                    run_snapshot = {"error": str(e)}

            lines = [
                "# Simulation Report",
                f"\n## Structure: {record.get('description', structure_slug)}",
                f"- **Slug:** `{structure_slug}`",
                f"- **Created:** {record.get('created_at', 'unknown')}",
                f"- **Structure:** `{record.get('structure_path', 'N/A')}`",
            ]

            n_atoms = self._count_atoms(record.get("structure_path", ""))
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

            input_files = record.get("input_files") or {}
            if input_files:
                lines.append("\n## Generated Inputs")
                for fname, fpath in input_files.items():
                    lines.append(f"- **{fname}:** `{fpath}`")
                if record.get("summary"):
                    lines.append(f"- **Summary:** {record['summary']}")

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

            if run_snapshot:
                lines.append("\n## Calculation Results")
                if "error" in run_snapshot:
                    lines.append(f"- **Parse error:** {run_snapshot['error']}")
                else:
                    vr = run_snapshot.get("vasprun") or {}
                    lines += [
                        f"- **Convergence:** {run_snapshot.get('convergence_status', 'unknown')}",
                        f"- **Final energy:** {vr.get('final_energy', 'N/A')} eV",
                        f"- **Ionic steps:** {vr.get('n_ionic_steps', 'N/A')}",
                        f"- **Max force (last step):** {vr.get('max_force_eV_per_A', 'N/A')} eV/Å",
                    ]
                    hints = run_snapshot.get("log_error_hints") or []
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
                out_path.write_text(report_text, encoding="utf-8")
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
                "Call after download_job_results to include calculation outcomes."
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

        # =====================================================================
        # 16. RUN MLIP SIMULATION
        # =====================================================================
        def run_mlip_simulation(
            structure_path: str,
            research_goal: str,
            backend: str = None,
            model_name: str = None,
            task: str = "md",
            temperature: float = 300.0,
            n_steps: int = 10000,
            device: str = "cpu",
        ) -> str:
            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Structure file not found: {structure_path}",
                })

            slug = self._make_slug(research_goal)
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            try:
                from .mlip_agent import MLIPAgent
                from ase.io import read as _ase_read
                atoms = _ase_read(structure_path)
                elements = sorted(set(atoms.get_chemical_symbols()))
                system_info = {
                    "elements": {e: None for e in elements},
                    "n_atoms": len(atoms),
                }
                agent = MLIPAgent(
                    working_dir=str(workdir),
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
                sim_params = {
                    "task": task,
                    "temperature": temperature,
                    "n_steps": n_steps,
                    "device": device,
                }
                result = agent.deploy_pretrained(
                    system_info=system_info,
                    research_goal=research_goal,
                    structure_file=structure_path,
                    backend=backend,
                    model_name=model_name,
                    simulation_params=sim_params,
                    runner="ase",
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"MLIP simulation failed: {e}",
                })

            run_path = result.get("run_path") or result.get("script_path")
            record = {
                "slug": slug,
                "description": research_goal,
                "structure_dir": str(workdir),
                "structure_path": structure_path,
                "script_path": run_path,
                "script_content": None,
                "input_files": {Path(run_path).name: run_path} if run_path else {},
                "summary": (
                    f"MLIP {result.get('backend', '')} {result.get('model_name', '')} "
                    f"{task} at {temperature} K, {n_steps} steps"
                ),
                "validation": None,
                "created_at": datetime.now().isoformat(),
                "mlip_backend": result.get("backend"),
                "mlip_model": result.get("model_name"),
            }
            self.orch.generated_structures.append(record)

            return json.dumps({
                "status": "success",
                "slug": slug,
                "backend": result.get("backend"),
                "model_name": result.get("model_name"),
                "run_script": run_path,
                "workdir": str(workdir),
                "task": task,
                "next_steps": (
                    f"Run the script: python {run_path}\n"
                    "Then call analyze_output(workdir, research_goal) to assess "
                    "the trajectory, or run_exafs_workflow(trajectory_path, ...) "
                    "to compute EXAFS spectra from the trajectory."
                ),
            })

        self._register_tool(
            func=run_mlip_simulation,
            name="run_mlip_simulation",
            description=(
                "Deploy an MLIP (machine-learning interatomic potential) and "
                "generate a run script for relaxation or NVT molecular dynamics. "
                "Selects the best pretrained foundation model for the system "
                "(MACE, CHGNet, orb, UMA, …) unless `backend` is specified. "
                "Writes an ASE run script to a session-managed directory. "
                "Returns the script path — the user runs it, then calls "
                "analyze_output or run_exafs_workflow on the resulting trajectory. "
                "For EXAFS simulations, use run_exafs_workflow which calls this "
                "step internally."
            ),
            parameters={
                "structure_path": {
                    "type": "string",
                    "description": "Absolute path to the input structure (CIF, POSCAR, extxyz, etc.).",
                },
                "research_goal": {
                    "type": "string",
                    "description": (
                        "What the simulation should achieve "
                        "(e.g. 'NVT MD at 300 K for EXAFS thermal sampling of Fe2O3')."
                    ),
                },
                "backend": {
                    "type": "string",
                    "description": (
                        "Optional MLIP backend override: 'mace', 'chgnet', 'deepmd', "
                        "'uma', 'orb'. When omitted, the best available backend for "
                        "the system is selected automatically."
                    ),
                },
                "model_name": {
                    "type": "string",
                    "description": (
                        "Optional foundation-model identifier within the backend "
                        "(e.g. 'mace-omat-0', 'mace-off23'). Defaults to the "
                        "backend's recommended foundation model."
                    ),
                },
                "task": {
                    "type": "string",
                    "enum": ["md", "relax"],
                    "description": "'md' for NVT molecular dynamics (default), 'relax' for cell + geometry optimization.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Simulation temperature in Kelvin (default: 300).",
                },
                "n_steps": {
                    "type": "integer",
                    "description": "Number of MD or optimization steps (default: 10000).",
                },
                "device": {
                    "type": "string",
                    "description": "'cpu' (default) or 'cuda'. Use 'cuda' when a GPU is available.",
                },
            },
            required=["structure_path", "research_goal"],
        )

        # =====================================================================
        # 17. RUN EXAFS WORKFLOW
        # =====================================================================
        def run_exafs_workflow(
            structure_path: str,
            absorber_element: str,
            edge: str = "K",
            rmax: float = 6.0,
            temperature: float = 300.0,
            n_md_steps: int = 83500,
            md_step_size: int = 250,
            backend: str = None,
            device: str = "cpu",
            feff_binary: str = None,
        ) -> str:
            if not Path(structure_path).exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Structure file not found: {structure_path}",
                })

            slug = self._make_slug(f"exafs_{absorber_element}_{Path(structure_path).stem}")
            workdir = self.orch.structures_dir / slug
            workdir.mkdir(parents=True, exist_ok=True)

            # ── Step 1: MLIP deployment + run script generation ───────────
            try:
                from .mlip_agent import MLIPAgent
                from ase.io import read as _ase_read
                atoms = _ase_read(structure_path)
                elements = sorted(set(atoms.get_chemical_symbols()))
                system_info = {
                    "elements": {e: None for e in elements},
                    "n_atoms": len(atoms),
                }
                agent = MLIPAgent(
                    working_dir=str(workdir),
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
                research_goal = (
                    f"NVT MD at {temperature} K for EXAFS thermal sampling of "
                    f"{absorber_element} {edge}-edge in {Path(structure_path).stem}"
                )
                sim_params = {
                    "task": "md",
                    "temperature": temperature,
                    "n_steps": n_md_steps,
                    "device": device,
                }
                mlip_result = agent.deploy_pretrained(
                    system_info=system_info,
                    research_goal=research_goal,
                    structure_file=structure_path,
                    backend=backend,
                    simulation_params=sim_params,
                    runner="ase",
                )
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "stage": "mlip_deploy",
                    "message": f"MLIP deployment failed: {e}",
                })

            run_script = mlip_result.get("run_path") or mlip_result.get("script_path")

            # ── Step 2: locate absorber index ─────────────────────────────
            try:
                from ase.io import read as _ase_read2
                struct = _ase_read2(structure_path)
                absorber_indices = [
                    i for i, sym in enumerate(struct.get_chemical_symbols())
                    if sym == absorber_element
                ]
                if not absorber_indices:
                    return json.dumps({
                        "status": "error",
                        "stage": "absorber_search",
                        "message": (
                            f"Element '{absorber_element}' not found in structure. "
                            f"Elements present: {sorted(set(struct.get_chemical_symbols()))}"
                        ),
                    })
                target_atom = absorber_indices[0]
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "stage": "absorber_search",
                    "message": f"Failed to parse structure for absorber: {e}",
                })

            # ── Step 3: detect FEFF binary ────────────────────────────────
            import os as _os
            import shutil as _shutil
            _FEFF_NAMES = ("feff.x", "feff9", "feff8", "feff7", "feff8l", "feff6l", "feff")
            feff_exe = feff_binary
            if not feff_exe:
                explicit = _os.environ.get("FEFF_BIN", "").strip()
                if explicit and _os.path.isfile(explicit):
                    feff_exe = explicit
                elif _os.environ.get("FEFF_DIR", "").strip():
                    feff_dir = _os.environ["FEFF_DIR"].strip()
                    for _name in _FEFF_NAMES:
                        _c = _os.path.join(feff_dir, _name)
                        if _os.path.isfile(_c):
                            feff_exe = _c
                            break
                if not feff_exe:
                    for _name in _FEFF_NAMES:
                        _found = _shutil.which(_name)
                        if _found:
                            feff_exe = _found
                            break
                if not feff_exe:
                    _site = "/share/feff/feff90_binaries/feff.x"
                    if _os.path.isfile(_site):
                        feff_exe = _site

            # ── Step 4: HOLE card from edge ───────────────────────────────
            _HOLE_MAP = {"K": 1, "L1": 2, "L2": 3, "L3": 4, "M1": 5}
            hole = _HOLE_MAP.get(edge.upper(), 1)

            # Assemble the plan output ─────────────────────────────────────
            record = {
                "slug": slug,
                "description": f"EXAFS {absorber_element} {edge}-edge in {Path(structure_path).stem}",
                "structure_dir": str(workdir),
                "structure_path": structure_path,
                "script_path": run_script,
                "script_content": None,
                "input_files": {Path(run_script).name: run_script} if run_script else {},
                "summary": (
                    f"EXAFS workflow: {absorber_element} {edge}-edge, "
                    f"MLIP={mlip_result.get('backend')}/{mlip_result.get('model_name')}, "
                    f"T={temperature} K, rmax={rmax} Å"
                ),
                "validation": None,
                "created_at": datetime.now().isoformat(),
                "exafs": {
                    "absorber_element": absorber_element,
                    "absorber_index": target_atom,
                    "edge": edge,
                    "hole": hole,
                    "rmax": rmax,
                    "temperature": temperature,
                    "n_md_steps": n_md_steps,
                    "md_step_size": md_step_size,
                    "feff_binary": feff_exe,
                    "mlip_backend": mlip_result.get("backend"),
                    "mlip_model": mlip_result.get("model_name"),
                },
            }
            self.orch.generated_structures.append(record)

            feff_note = (
                f"FEFF binary located: {feff_exe}"
                if feff_exe
                else (
                    "WARNING: No FEFF binary found. Download a free version "
                    "(FEFF6-lite or FEFF8-lite) from "
                    "https://feff.phys.washington.edu/feffproject-feff-download.html "
                    "and set $FEFF_BIN before running Stage 3."
                )
            )

            return json.dumps({
                "status": "success",
                "slug": slug,
                "workdir": str(workdir),
                "mlip_backend": mlip_result.get("backend"),
                "mlip_model": mlip_result.get("model_name"),
                "run_script": run_script,
                "absorber_element": absorber_element,
                "absorber_index": target_atom,
                "edge": edge,
                "hole": hole,
                "rmax": rmax,
                "feff_binary": feff_exe,
                "feff_note": feff_note,
                "next_steps": (
                    f"1. Run the MD script to generate a trajectory:\n"
                    f"   python {run_script}\n"
                    f"2. Generate FEFF inputs from the trajectory:\n"
                    f"   from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
                    f"import generate_feff_inputs_from_trajectory\n"
                    f"   result = generate_feff_inputs_from_trajectory(\n"
                    f"       trajectory_path='md_trajectory.xyz',\n"
                    f"       target_atom={target_atom}, hole={hole},\n"
                    f"       rmax={rmax}, scf='6.0 0 30 0.2 1',\n"
                    f"       step_size={md_step_size})\n"
                    f"3. Run FEFF batch jobs on the generated inputs.\n"
                    f"4. Average chi(k):\n"
                    f"   from scilink.skills.exafs_simulation.exafs_workflow.feff_tools "
                    f"import average_chi\n"
                    f"   average_chi(result['output_dir'], 'exafs_output')\n"
                    + (f"\n{feff_note}" if not feff_exe else "")
                ),
            })

        self._register_tool(
            func=run_exafs_workflow,
            name="run_exafs_workflow",
            description=(
                "Set up a complete EXAFS simulation workflow: deploys an MLIP "
                "for the structure, generates a run script for NVT MD thermal "
                "sampling, detects any installed FEFF binary, and returns a "
                "step-by-step plan with the correct parameters (absorber index, "
                "HOLE card, rmax, step_size) pre-filled. The user runs the MD "
                "script to produce a trajectory, then calls "
                "generate_feff_inputs_from_trajectory and average_chi (provided "
                "as TOOL_SPEC helpers in the exafs_simulation skill). "
                "Co-loads the exafs_simulation skill automatically — no need to "
                "call route_simulation first for EXAFS workflows."
            ),
            parameters={
                "structure_path": {
                    "type": "string",
                    "description": "Absolute path to the input crystal structure (CIF, POSCAR, extxyz, etc.).",
                },
                "absorber_element": {
                    "type": "string",
                    "description": (
                        "Chemical symbol of the X-ray absorbing element "
                        "(e.g. 'Fe', 'Cu', 'Ni'). First atom of this species is used."
                    ),
                },
                "edge": {
                    "type": "string",
                    "description": (
                        "X-ray absorption edge: 'K' (default), 'L1', 'L2', 'L3', 'M1'. "
                        "Use K for 3d metals, L3 for 4d/5d metals."
                    ),
                },
                "rmax": {
                    "type": "number",
                    "description": (
                        "FEFF RMAX path cutoff in Angstroms (default: 6.0). "
                        "Use 6–8 Å for 2–4 coordination shells."
                    ),
                },
                "temperature": {
                    "type": "number",
                    "description": "MD temperature in Kelvin (default: 300). Match experimental conditions.",
                },
                "n_md_steps": {
                    "type": "integer",
                    "description": "Total number of MD timesteps (default: 83500 ≈ 20 ps at 0.24 fs/step).",
                },
                "md_step_size": {
                    "type": "integer",
                    "description": "Sample one FEFF input per this many MD frames (default: 250).",
                },
                "backend": {
                    "type": "string",
                    "description": (
                        "Optional MLIP backend override: 'mace', 'chgnet', 'deepmd', 'uma', 'orb'. "
                        "When omitted, the best available backend for the system is selected."
                    ),
                },
                "device": {
                    "type": "string",
                    "description": "'cpu' (default) or 'cuda'.",
                },
                "feff_binary": {
                    "type": "string",
                    "description": (
                        "Optional explicit path to a FEFF binary. When omitted, "
                        "the tool searches $FEFF_BIN, $FEFF_DIR, $PATH, and the "
                        "site default /share/feff/feff90_binaries/feff.x."
                    ),
                },
            },
            required=["structure_path", "absorber_element"],
        )

        # ↓↓↓ CLI flesh-out (step 5), run_task (step 6), tests (step 7).

    # ------------------------------------------------------------------
    # Helpers used by tool closures
    # ------------------------------------------------------------------

    def _load_skill_content(self, skill) -> Optional[str]:
        """Resolve one or more skill names to their content as a single block.

        ``skill`` accepts a single name/path string, a list of names/paths,
        or ``None``. With multiple skills, each is rendered as its own
        ``# Skill: <name>`` section and the results are concatenated.

        Resolution order per skill:
          1. Built-in skills under scilink/skills/structure_generation/<name>/
          2. User-registered skills via orchestrator.register_skill()

        Returns None on any failure (fail-closed; the structure-gen prompt
        falls through to the generic template). Skills that fail to resolve
        are logged as warnings and dropped — others still render.
        """
        if not skill:
            return None
        names: list[str] = [skill] if isinstance(skill, str) else list(skill)
        names = [n for n in names if n]
        if not names:
            return None

        rendered = [block for n in names if (block := self._render_one_skill(n))]
        if not rendered:
            return None
        return "\n\n".join(rendered)

    def _render_one_skill(self, skill_name: str) -> Optional[str]:
        """Resolve and render a single skill; helper for ``_load_skill_content``."""
        try:
            from scilink.skills.loader import load_skill
            parsed = load_skill(skill_name, domain="structure_generation")
        except FileNotFoundError:
            user_skills = getattr(self.orch, "_custom_skills", {}) or {}
            path = user_skills.get(skill_name)
            if not path:
                self.logger.warning(
                    f"Skill '{skill_name}' not found in built-ins or "
                    "user-registered skills. Skipping this skill."
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

        # Concatenate populated canonical sections, then any non-canonical
        # ``extras`` so author-written content (e.g. "Common pitfalls")
        # isn't silently dropped.
        section_order = ["overview", "planning", "implementation",
                         "validation", "interpretation", "analysis"]
        chunks = []
        for sec in section_order:
            body = (parsed.get(sec) or "").strip()
            if body:
                chunks.append(f"### {sec.capitalize()}\n\n{body}")
        for heading, body in (parsed.get("extras") or {}).items():
            body = (body or "").strip()
            if body:
                chunks.append(f"### {heading.capitalize()}\n\n{body}")
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
    def _count_atoms(structure_path: str) -> Optional[int]:
        """Best-effort atom count via ASE; returns None on parse failure."""
        try:
            from ase.io import read as ase_read
            atoms = ase_read(structure_path)
            return len(atoms)
        except Exception:
            return None

    def _find_script_content(self, structure_path: str) -> Optional[str]:
        """Find the generating script for a POSCAR.

        First check the orchestrator's session records (cheap, exact). If
        not found, fall back to globbing `script_*.py` in the POSCAR's
        directory and reading the most recent one — this lets validation
        work even when the LLM passes around paths without going through
        the session record path.
        """
        record = self._find_structure_record(structure_path)
        if record and record.get("script_content"):
            return record["script_content"]

        poscar_dir = Path(structure_path).parent
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

    def _find_structure_record(self, structure_path: str) -> Optional[Dict[str, Any]]:
        """Find the session record matching a POSCAR path, by string match."""
        target = str(Path(structure_path).resolve())
        for record in self.orch.generated_structures or []:
            try:
                if str(Path(record.get("structure_path", "")).resolve()) == target:
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
        run_command: str,
        modules: str,
        extra_directives: str,
    ) -> str:
        """Generate a scheduler job script for an arbitrary engine run command."""
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

        lines += ["", run_command, ""]
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
