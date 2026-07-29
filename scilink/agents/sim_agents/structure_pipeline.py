# scilink/agents/sim_agents/structure_pipeline.py

import os
import sys
import logging
from io import StringIO
from typing import Optional, Dict, Any
from pathlib import Path

from ...auth import (
    get_api_key,
    require_vendor_credentials,
)
from .structure_agent import StructureGenerator
from .val_agent import StructureValidatorAgent


class StructurePipeline:
    """
    Engine-agnostic, structure-class-aware orchestrator for atomic-structure
    generation.

    Owns the structure generate → validate → refine loop: it composes a
    ``StructureGenerator`` and a ``StructureValidatorAgent``, loads the
    ``structure_generation/<structure_class>`` skill bundle to steer the build,
    and iterates on validator feedback until the structure passes (or a
    circuit-breaker fires). It does NOT generate any engine inputs — that is the
    job of a downstream simulation pipeline (e.g. ``run_complete_workflow``),
    which composes this class for its structure step. The chat
    ``SimulationOrchestratorAgent`` tools delegate here too, so the loop lives in
    exactly one place.

    The control flow here is deterministic Python; the LLM is used only within
    bounded steps (generation, validation), each returning a structured result.
    """

    def __init__(self,
                 api_key: str = None,
                 base_url: Optional[str] = None,
                 mp_api_key: str = None,
                 generator_model: str = "gemini-3-pro-preview",
                 validator_model: str = "gemini-3-pro-preview",
                 output_dir: str = "structure_output",
                 max_refinement_cycles: int = 4,
                 script_timeout: int = 300):
        """
        Initialize the structure orchestrator and its generator/validator agents.

        Args:
            api_key: API key for the LLM provider. When both ``api_key`` and
                ``base_url`` are None, LiteLLM auto-discovers the vendor env
                var (``ANTHROPIC_API_KEY`` etc.); if none is available,
                ``require_vendor_credentials`` raises with actionable guidance.
                ``SCILINK_API_KEY`` is for internal-proxy use (pair with
                ``base_url``).
            base_url: Optional base URL for an OpenAI-compatible internal proxy.
            mp_api_key: Materials Project API key for structure lookups
                (auto-discovered when None).
            generator_model: Model name for structure generation.
            validator_model: Model name for structure validation.
            output_dir: Directory to save generated structures, scripts, and the
                run log.
            max_refinement_cycles: Maximum validator-guided correction cycles.
            script_timeout: Timeout (seconds) for executing generated ASE scripts.
        """
        if api_key is None and base_url is None:
            require_vendor_credentials(generator_model)

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
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles

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

        os.makedirs(output_dir, exist_ok=True)

    def build_structure(self, user_request: str,
                        structure_class: str,
                        **kwargs) -> Dict[str, Any]:
        """
        Generate and validate an atomic structure from a natural-language request.

        Runs the structure generate → validate → refine loop, then stops (no
        engine inputs are produced). Useful for structure-generation benchmarking
        and for engine-agnostic workflows where inputs are generated separately
        (e.g. via ``PeriodicDFTAgent`` for VASP, Quantum ESPRESSO, etc.).

        Parameters
        ----------
        user_request : str
            Natural-language description of the structure to build.
        structure_class : str
            **Required** — the structure archetype whose generation skill + validation
            rubric + output format guide the build (``crystal`` / ``molecular`` /
            ``condensed`` / ``biomolecular``). No default on this standalone entry point,
            so a caller building e.g. a molecule must say so rather than silently getting
            the crystal rubric. (The DFT path keeps a ``"crystal"`` default in
            ``run_complete_workflow``, where it's contextually correct.)
        **kwargs
            Forwarded to :meth:`generate_and_validate` (``skill_content``,
            ``prior_script``, ``validate``, ``max_cycles``, ``output_dir``).

        Returns the structure result dict:
            status                : "success" or "error"
            final_structure_path  : path to the generated structure (on success)
            final_script_path     : path to the script that built it
            final_script_content  : source of the script that built it
            cycles_used           : number of generate/validate cycles taken
            validation_result     : validator feedback (issues, hints, ...)
            warning               : present if refinement stopped early
            message / cycle       : present on error
        """
        print(f"\n🏗️  Structure Generation & Validation  (class: {structure_class})")
        print(f"{'='*60}")
        print(f"📝 Request: {user_request}")
        print(f"📁 Output:  {self.output_dir}/")
        print(f"{'='*60}")

        result = self.generate_and_validate(user_request, structure_class=structure_class, **kwargs)

        if result.get("status") == "success":
            print(f"✅ Structure generated: "
                  f"{os.path.basename(result['final_structure_path'])} "
                  f"({result.get('cycles_used', '?')} cycle(s))")
            if result.get("warning"):
                print(f"⚠️  {result['warning']}")
        else:
            print(f"❌ Structure generation failed: {result.get('message', 'Unknown error')}")

        self._save_workflow_log()
        return result

    def _load_structure_skill(self, structure_class: str) -> Optional[str]:
        """Load the structure-generation skill bundle for ``structure_class`` and
        assemble its generation-facing guidance (overview / planning /
        implementation sections) into a single text block to inject into the
        generator prompt. Returns None (generic generation) when no bundle exists.
        """
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(structure_class, domain="structure_generation")
        except FileNotFoundError:
            self.logger.info(
                f"No structure_generation skill for structure_class="
                f"'{structure_class}'; using generic generation."
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"Failed to load structure_generation skill '{structure_class}': {e}"
            )
            return None

        parts = []
        for section in ("overview", "planning", "implementation"):
            body = (parsed.get(section) or "").strip()
            if body:
                parts.append(f"## {section}\n{body}")
        if not parts:
            return None
        self.logger.info(
            f"Loaded structure_generation skill: {parsed.get('name', structure_class)}"
        )
        return "\n\n".join(parts)

    def _load_validation_rubric(self, structure_class: str) -> Optional[str]:
        """Load the ``## Validation`` section of the
        ``structure_generation/<structure_class>`` skill to steer the validator
        with class-specific acceptance criteria. The validator's base prompt is
        class-neutral, so this rubric is what makes validation scale-aware (e.g. a
        molecule is not judged by periodic-crystal supercell/vacuum criteria).
        Returns None when no bundle / no validation section exists.
        """
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(structure_class, domain="structure_generation")
        except FileNotFoundError:
            return None
        except Exception as e:
            self.logger.warning(
                f"Failed to load validation rubric for '{structure_class}': {e}"
            )
            return None
        body = (parsed.get("validation") or "").strip()
        if not body:
            return None
        self.logger.info(
            f"Loaded validation rubric for structure_class: {parsed.get('name', structure_class)}"
        )
        return body

    def _load_output_format(self, structure_class: str) -> Optional[str]:
        """Read the ``output_format`` frontmatter key of the
        ``structure_generation/<structure_class>`` skill (e.g. POSCAR / xyz / pdb).
        Returns None when no bundle / no key is set.
        """
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(structure_class, domain="structure_generation")
        except Exception:
            return None
        fmt = (parsed.get("meta") or {}).get("output_format")
        return str(fmt) if fmt else None

    def generate_and_validate(self, user_request: str,
                              structure_class: Optional[str],
                              *,
                              skill_content: Optional[str] = None,
                              validation_rubric: Optional[str] = None,
                              output_format: Optional[str] = None,
                              constraints: Optional[str] = None,
                              prior_script: Optional[str] = None,
                              validate: bool = True,
                              max_cycles: Optional[int] = None,
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate and validate an atomic structure (the core refine loop).

        This is the single home for the structure generate → validate → refine
        loop; both the ``run_complete_workflow`` pipeline and the chat
        ``SimulationOrchestratorAgent`` tools delegate here rather than
        reimplementing it.

        Parameters
        ----------
        user_request : str
            Natural-language description of the structure to build.
        structure_class : str or None
            **Required (no default)** — the structure archetype; selects the
            ``structure_generation/<class>`` skill bundle + validation rubric +
            output format when ``skill_content`` is not supplied. No silent default,
            so a non-crystal build can't accidentally get the crystal rubric. Pass
            ``None`` explicitly to skip auto-loading a class skill (e.g. when the
            caller fully controls ``skill_content``).
        skill_content : str, optional
            Pre-rendered skill guidance injected verbatim. When given it is used
            as-is and ``structure_class`` is not consulted for skill loading — this
            lets callers compose multiple / user-registered skills themselves.
        validation_rubric : str, optional
            Pre-rendered class-specific validation criteria injected into the
            validator. Defaults to the ``## Validation`` section of the
            ``structure_class`` skill when not supplied (and ``structure_class``
            is set).
        output_format : str, optional
            Structure file format to request (``extxyz`` / ``xyz`` / ``pdb`` / ...).
            Defaults to the ``output_format`` frontmatter of the ``structure_class``
            skill, else the engine-neutral ``extxyz`` (portable coordinates; the
            engine-native input is written downstream by the engine step).
        constraints : str, optional
            A build-constraints block appended to the generation request (e.g. from
            ``StructureSpec.as_constraints_text()`` — target size / periodicity /
            solvation / charge), so the planner's cross-axis reasoning shapes the build.
        prior_script : str, optional
            A previously generated script to modify (variant builds); applied to
            the initial generation only.
        validate : bool
            When False, generate once and return without validation / refinement.
        max_cycles : int, optional
            Override for ``max_refinement_cycles`` for this call.
        output_dir : str, optional
            Per-call output directory for the structure / script (defaults to the
            orchestrator's ``output_dir``).

        Includes circuit-breakers that exit early when refinement stops making
        progress (issue count not strictly decreasing for 2 consecutive cycles,
        or generator returned an unchanged script).
        """
        if output_dir is not None:
            self.output_dir = output_dir
            self.structure_generator.generated_script_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        if skill_content is None and structure_class is not None:
            skill_content = self._load_structure_skill(structure_class)

        if validation_rubric is None and structure_class is not None:
            validation_rubric = self._load_validation_rubric(structure_class)

        if output_format is None and structure_class is not None:
            output_format = self._load_output_format(structure_class)
        # Engine-neutral default: structure generation emits portable coordinates
        # (extended XYZ carries cell + PBC); the engine-native input is written
        # downstream by the engine step. Per-class skills may override via their
        # output_format frontmatter (xyz for isolated molecules, pdb for biomolecules).
        output_format = output_format or "extxyz"
        request = f"{user_request}. Save the structure in {output_format} format."
        if constraints:
            request += "\n\n" + constraints

        max_cycles = self.max_refinement_cycles if max_cycles is None else max_cycles

        previous_script_content = None
        previous_structure_file = None
        previous_final_script_path = None
        validator_feedback = None
        attempt_history: list = []   # full per-cycle log: {script, issues, hints}

        for cycle in range(max_cycles + 1):
            cycle_num = cycle + 1
            total_cycles = max_cycles + 1

            if cycle == 0:
                print(f"🔨 Generating structure (attempt {cycle_num}/{total_cycles})")
            else:
                print(f"🔄 Refining structure (attempt {cycle_num}/{total_cycles})")
                print(f"    Addressing: {len(validator_feedback.get('all_identified_issues', []))} validation issues")

            gen_result = self.structure_generator.generate_script(
                original_user_request=request,
                attempt_number_overall=cycle_num,
                is_refinement_from_validation=(cycle > 0),
                previous_script_content=previous_script_content if cycle > 0 else None,
                validator_feedback=validator_feedback if cycle > 0 else None,
                attempt_history=attempt_history if cycle > 0 else None,
                skill_content=skill_content,
                prior_script_to_modify=prior_script if cycle == 0 else None,
            )

            if gen_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Structure generation failed on cycle {cycle_num}: {gen_result.get('message')}",
                    "cycle": cycle_num
                }

            structure_file = gen_result["output_file"]
            script_content = gen_result["final_script_content"]

            # Single-shot mode: caller asked for a build with no validation.
            if not validate:
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
                    "final_script_path": gen_result["final_script_path"],
                    "final_script_content": script_content,
                    "cycles_used": cycle_num,
                    "validation_result": None,
                }

            # CIRCUIT-BREAKER 1: generator returned an unchanged script.
            # Treat as "the model has nothing more to fix" and accept current state.
            if cycle > 0 and script_content == previous_script_content:
                print(f"🛑 Generator returned an unchanged script — "
                      f"accepting current structure (cycle {cycle_num}).")
                return {
                    "status": "success",
                    "final_structure_path": previous_structure_file or structure_file,
                    "final_script_path": previous_final_script_path or gen_result["final_script_path"],
                    "final_script_content": script_content,
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
                original_request=user_request,
                validation_rubric=validation_rubric,
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
                    "final_script_content": script_content,
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
                            f"the structure."
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
                        "final_script_content": script_content,
                        "cycles_used": cycle_num,
                        "validation_result": val_result,
                        "warning": warning,
                    }

            if cycle < max_cycles:
                print(f"🔄 Issues found, attempting refinement...")
                continue
            else:
                print(f"⚠️  Max refinement cycles reached, proceeding with current structure")
                return {
                    "status": "success",
                    "final_structure_path": structure_file,
                    "final_script_path": gen_result["final_script_path"],
                    "final_script_content": script_content,
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

    def _save_workflow_log(self) -> str:
        """Save all captured logs to a file."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = os.path.join(self.output_dir, "workflow_log.txt")
            with open(log_path, 'w') as f:
                f.write(f"SciLink Workflow Log\n")
                f.write(f"{'='*30}\n\n")
                f.write(log_content)
            print(f"📝 Complete workflow log saved: {log_path}")
            return log_path
        except Exception as e:
            print(f"Warning: Could not save workflow log: {e}")
            return ""
