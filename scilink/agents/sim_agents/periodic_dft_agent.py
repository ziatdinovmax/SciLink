# scilink/agents/sim_agents/periodic_dft_agent.py
"""
Periodic-DFT simulation agent. Handles planewave / pseudopotential
DFT codes (VASP, QE, ABINIT, CP2K, ...) via skill bundles.

Software-specific behavior (INCAR vs &control namelist vs ABINIT input,
PAW pseudopotentials vs norm-conserving, etc.) lives in the per-engine
skill bundles at scilink/skills/periodic_dft/<engine>/<engine>.md and
the sibling tools modules. The agent class is scale-aware (periodic
DFT) and software-agnostic.

Today VASP is the only fully wired engine. Adding QE / ABINIT / CP2K
is a sibling skill bundle drop-in.
"""

import os
import re
import json
import logging
from typing import Optional
from ...auth import (
    get_internal_proxy_key,
    require_vendor_credentials,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params


class PeriodicDFTAgent:
    """Periodic / pseudopotential DFT agent.

    Scale-aware (periodic DFT), software-agnostic. Engine-specific
    behavior lives in skill bundles at
    ``scilink/skills/periodic_dft/<engine>/<engine>.md``.

    Currently supports VASP via the ``vasp`` skill bundle; QE / ABINIT /
    CP2K are extension points (drop in a sibling bundle).
    """

    SKILL_DOMAIN = "periodic_dft"

    @classmethod
    def supported_software(cls) -> list:
        """
        Auto-discover engine names this agent can currently handle.

        Returns every skill bundle name found for the agent's
        ``SKILL_DOMAIN`` across both built-in skills and any user-
        provided roots from ``$SCILINK_SKILLS_PATH``. A user dropping
        in their own ``periodic_dft/cp2k/cp2k.md`` will see ``cp2k``
        appear in the list with no source-code changes.

        Re-evaluated on each call so adding bundles or env-var entries
        mid-process takes effect immediately. Used by the orchestrator's
        routing layer to decide which engines are reachable.
        """
        from ...skills.loader import list_skills
        return list_skills(domain=cls.SKILL_DOMAIN)

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):
        """
        Initialize PeriodicDFTAgent.

        Parameters
        ----------
        api_key : str, optional
            API key for the LLM provider.
        model_name : str, optional
            Model name to use.
        base_url : str, optional
            Base URL for internal proxy.

        Software-specific project-config / deterministic-enforcement
        layers (e.g. ``VASPProjectConfig``) are kept on the per-software
        subclasses (e.g. ``VaspInputAgent``) so this base class stays
        software-agnostic.
        """
        self.logger = logging.getLogger(__name__)
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="PeriodicDFTAgent"
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            if api_key is None:
                require_vendor_credentials(model_name)
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None

    # Backward-compat: previous skill name was "vasp_input_generation" living
    # under the old skills/vasp/ tree. After the periodic-DFT refactor it's
    # just "vasp" under skills/periodic_dft/.
    _LEGACY_SKILL_ALIASES = {"vasp_input_generation": "vasp"}

    def _load_skill(self, skill: str) -> dict:
        """
        Load a periodic-DFT skill bundle (default: ``vasp``).

        Parameters
        ----------
        skill : str
            Skill name (resolved from ``scilink/skills/periodic_dft/``)
            or path to a .md file.

        Returns
        -------
        dict with skill_name and skill_sections, or empty skill state
        on failure.
        """
        resolved = self._LEGACY_SKILL_ALIASES.get(skill, skill)
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(resolved, domain=self.SKILL_DOMAIN)
            self.logger.info(
                f"Loaded {self.SKILL_DOMAIN} skill: {parsed.get('name', resolved)}"
            )
            return {
                "skill_name": parsed.get("name", resolved),
                "skill_sections": parsed,
            }
        except FileNotFoundError:
            self.logger.warning(
                f"Skill '{resolved}' not found under '{self.SKILL_DOMAIN}' — "
                f"proceeding without domain skill"
            )
            return {"skill_name": None, "skill_sections": None}
        except Exception as e:
            self.logger.warning(f"Failed to load skill '{resolved}': {e}")
            return {"skill_name": None, "skill_sections": None}

    def _build_prompt(self, structure_content: str, request: str,
                      software: str,
                      skill_sections: Optional[dict] = None,
                      native_structure_file: Optional[str] = None) -> str:
        """
        Build the full prompt, injecting skill content if available.

        Scaffold is software-agnostic — engine-specific guidance comes
        entirely from the loaded skill bundle's planning / implementation
        / validation sections. The ``software`` argument is interpolated
        into the scaffold so the LLM knows which engine's canonical
        filenames to emit (e.g. INCAR + KPOINTS for VASP, qe.in for QE).

        Parameters
        ----------
        structure_content : str
            Raw text of the structure file (e.g. POSCAR for VASP,
            <name>.xyz / CIF for others). Passed verbatim to the LLM.
        request : str
            User's natural-language description of the calculation.
        software : str
            Which engine's inputs to generate (lowercased keyword the
            skill bundle was indexed by, e.g. ``"vasp"``, ``"qe"``).
        skill_sections : dict, optional
            Parsed skill sections from ``load_skill()``. Each present
            section is prepended as its own labelled block.
        """
        skill_parts = []
        if skill_sections:
            for section_name in ("planning", "implementation", "validation"):
                content = skill_sections.get(section_name)
                if content:
                    skill_parts.append(
                        f"## {section_name.title()}\n{content}"
                    )

        base = (
            f"You are an expert in periodic / pseudopotential DFT "
            f"calculations using {software.upper()}.\n\n"
            f"Your task is to generate the input file(s) needed for "
            f"the requested calculation, following the guidance above.\n\n"
            f"**Structure file content** (use as-is, do not modify):\n"
            f"```\n{structure_content}\n```\n\n"
            f"**User request:**\n{request}\n\n"
            f"**Output format:** Return ONLY a JSON object with this "
            f"structure:\n"
            f"```json\n"
            f"{{\n"
            f'  "input_files": {{\n'
            f'    "<filename>": "<full file content>"\n'
            f"    ...\n"
            f"  }},\n"
            f'  "notes": "<any caveats, assumptions, or recommendations>"\n'
            f"}}\n"
            f"```\n\n"
            f"Pick the canonical filenames the {software.upper()} engine "
            f"expects (e.g. INCAR + KPOINTS for VASP, qe.in for QE, "
            f"<name>.abi for ABINIT). Filenames are case-sensitive."
        )
        if native_structure_file:
            base += (
                f"\n\nThe structure/coordinate file (`{native_structure_file}`) is "
                f"written automatically from the structure above — do NOT emit "
                f"`{native_structure_file}` yourself; generate only the remaining "
                f"input files (calculation settings, k-points, etc.). The structure "
                f"above is provided so you can choose those settings appropriately."
            )
        if skill_parts:
            return "\n\n".join(skill_parts) + "\n\n---\n\n" + base
        return base

    def _parse_response(self, response_text: str) -> dict:
        """
        Robustly parse LLM response, handling common formatting issues.
        """
        text = response_text.strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")

    @staticmethod
    def _native_structure_spec(skill_state) -> tuple:
        """(filename, ASE format) for the engine's separate coordinate file, as
        declared in the skill frontmatter, or (None, None). Engines that embed
        coordinates in their main input (e.g. QE) declare neither."""
        meta = ((skill_state or {}).get("skill_sections") or {}).get("meta") or {}
        return meta.get("structure_file"), meta.get("structure_format")

    def _inject_native_structure(self, result: dict, structure_file: str,
                                 skill_state) -> None:
        """Write the engine-native coordinate file deterministically from the
        generated structure (via ASE), overwriting any LLM-authored copy.

        Structure generation emits portable coordinates (extended XYZ); the
        engine's coordinate input (e.g. a VASP POSCAR) is produced here, exactly,
        rather than relying on the LLM to transcribe positions. No-op for engines
        whose skill declares no separate coordinate file (e.g. QE embeds the
        geometry in its main input)."""
        fname, fmt = self._native_structure_spec(skill_state)
        if not (fname and fmt) or not isinstance(result.get("input_files"), dict):
            return
        try:
            import io
            from ase.io import read as _ase_read, write as _ase_write
            atoms = _ase_read(structure_file)
            buf = io.StringIO()
            _ase_write(buf, atoms, format=fmt)
            result["input_files"][fname] = buf.getvalue()
            self.logger.info(
                f"Wrote deterministic {fname} ({fmt}) from "
                f"{os.path.basename(structure_file)}"
            )
        except Exception as e:
            self.logger.warning(f"Could not write deterministic {fname}: {e}")

    def generate_inputs(self, structure_file: str,
                        request: str,
                        software: str = "vasp",
                        skill: Optional[str] = None) -> dict:
        """
        Generate input files for a periodic-DFT calculation.

        Scale-aware, software-agnostic entry point. The ``software``
        argument names the engine (which must have a matching skill
        bundle at ``scilink/skills/periodic_dft/<software>/``); the
        skill bundle's content drives all engine-specific behavior.

        Parameters
        ----------
        structure_file : str
            Path to the structure file (POSCAR / CIF / XYZ / engine-
            specific). Read as raw text and passed verbatim to the LLM
            (the skill bundle tells the LLM how to interpret it).
        request : str
            Natural-language description of the calculation.
        software : str, optional
            Which engine's inputs to generate. Default ``"vasp"``.
        skill : str, optional
            Skill bundle name to load. Defaults to ``software`` (so
            ``software="vasp"`` loads the ``vasp`` skill bundle). Pass
            an explicit name (or full path to a .md file) to override.
            Set to None to skip skill loading.

        Returns
        -------
        dict with keys:
            status      : "success" or "error"
            software    : echo of the software argument
            input_files : {filename: content} dict for every input file
                          the agent generated (e.g. {"INCAR": "...",
                          "KPOINTS": "..."} for VASP)
            notes       : optional caveats / recommendations from the LLM
            message     : present on error only
        """
        try:
            with open(structure_file, 'r') as f:
                structure_content = f.read()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read structure file: {e}",
            }

        skill_sections = None
        skill_state = None
        chosen_skill = skill if skill is not None else software
        if chosen_skill:
            skill_state = self._load_skill(chosen_skill)
            skill_sections = skill_state.get("skill_sections")

        native_file, _ = self._native_structure_spec(skill_state)
        prompt = self._build_prompt(
            structure_content=structure_content,
            request=request,
            software=software,
            skill_sections=skill_sections,
            native_structure_file=native_file,
        )

        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = self._parse_response(response.text)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Generation failed: {e}",
            }

        # Backward-compat: if the LLM returns the legacy VASP-flat shape
        # (top-level "incar" / "kpoints"), collapse it into input_files.
        if "input_files" not in result:
            legacy_files = {}
            for legacy_key, canonical_name in (
                ("incar", "INCAR"),
                ("kpoints", "KPOINTS"),
                ("poscar", "POSCAR"),
            ):
                if legacy_key in result:
                    legacy_files[canonical_name] = result.pop(legacy_key)
            if legacy_files:
                result["input_files"] = legacy_files

        if not isinstance(result.get("input_files"), dict):
            return {
                "status": "error",
                "message": (
                    "LLM response did not include an 'input_files' object"
                ),
                "raw_result": result,
            }

        # ── pre-submit syntax pass ────────────────────────────────────
        # Engine-neutral: ask the active engine's skill bundle for its
        # syntax checker/fixer via the registry. The agent never names an
        # engine or imports a bundle by path — an engine with no such
        # tools simply skips this advisory pass. See CLAUDE.md
        # "Engine-neutral contracts" and the skill-bundle callables pattern.
        try:
            from ...skills._shared._registry import get_tool_function
            try:
                checker = get_tool_function(
                    "check_input_syntax", active_skills=[software]
                )
            except LookupError:
                checker = None
            if checker is not None:
                issues = checker(input_files=result["input_files"])
                if issues:
                    applied: list = []
                    try:
                        fixer = get_tool_function(
                            "apply_input_syntax_fixes", active_skills=[software]
                        )
                    except LookupError:
                        fixer = None
                    if fixer is not None:
                        fixed_files, applied = fixer(
                            input_files=result["input_files"], issues=issues
                        )
                        result["input_files"] = fixed_files
                        for a in applied:
                            self.logger.info(
                                "pre-submit syntax fix: %s → %s",
                                a.get("renamed_from"), a.get("renamed_to"),
                            )
                    result["syntax_check"] = {
                        "issues": issues,
                        "applied_fixes": applied,
                    }
        except Exception as exc:
            # Syntax check is advisory — never block generation on it.
            self.logger.warning("input syntax check failed: %s", exc)

        # Deterministic engine-native coordinate file (exact coordinates).
        self._inject_native_structure(result, structure_file, skill_state)

        result["status"] = "success"
        result["software"] = software
        return result

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """
        Save every entry in ``result["input_files"]`` to ``output_dir``
        using the dict key as the literal filename. Software-agnostic.

        Returns a dict mapping each filename to the absolute path it
        was written to, plus ``"error"`` on failure.
        """
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}

        input_files = result.get("input_files")
        if not isinstance(input_files, dict) or not input_files:
            return {"error": "No input_files to save"}

        os.makedirs(output_dir, exist_ok=True)
        saved: dict = {}
        try:
            for filename, content in input_files.items():
                path = os.path.join(output_dir, filename)
                with open(path, 'w') as f:
                    f.write(content)
                saved[filename] = path
            return saved
        except Exception as e:
            return {"error": f"Save failed: {e}"}

    def apply_improvements(self, original_inputs: dict, validation_result: dict,
                           structure_file: str, request: str,
                           output_dir: str = ".",
                           software: str = "vasp",
                           skill: Optional[str] = None) -> dict:
        """
        Regenerate input files incorporating validation-suggested fixes.

        Software-agnostic. ``original_inputs`` is the same shape as
        ``generate_inputs()`` returns: ``{filename: content}``. The
        validation_result's ``suggested_adjustments`` are passed to the
        LLM verbatim — adjustment vocabulary (parameter names, etc.)
        is engine-specific and lives in the skill bundle.

        Parameters
        ----------
        original_inputs : dict
            Mapping {filename: content} of the inputs to improve.
        validation_result : dict
            Validation result. Expected keys: ``validation_status``
            (``"needs_adjustment"`` triggers regeneration), and
            ``suggested_adjustments`` (a list of {parameter, current,
            suggested, reason} entries).
        structure_file : str
            Path to the structure file (re-read so the LLM has full
            context for the regeneration).
        request : str
            Original natural-language request.
        output_dir : str
            Directory where improved files are written.
        software : str
            Engine name (default ``"vasp"``).
        skill : str, optional
            Skill name. Defaults to ``software``.
        """
        if validation_result.get("validation_status") != "needs_adjustment":
            return {
                "status": "no_changes",
                "message": "No improvements needed",
            }

        adjustments = validation_result.get("suggested_adjustments", [])
        if not adjustments:
            return {"status": "error", "message": "No adjustments available"}

        try:
            with open(structure_file, 'r') as f:
                structure_content = f.read()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read structure file: {e}",
            }

        improvement_lines = [
            "IMPROVEMENT INSTRUCTIONS:",
            "Modify the provided input file(s) according to these "
            "literature-validated suggestions:",
            "",
        ]
        for adj in adjustments:
            improvement_lines.append(
                f"• {adj.get('parameter')}: "
                f"{adj.get('current_value')} → {adj.get('suggested_value')}"
            )
            improvement_lines.append(
                f"  Reason: {adj.get('reason')}"
            )
            improvement_lines.append("")
        improvement_lines.append(
            f"Literature assessment: "
            f"{validation_result.get('overall_assessment', '')}"
        )
        improvement_lines.append(
            "Regenerate the input file(s) with these changes applied."
        )
        improvement_instructions = "\n".join(improvement_lines)

        skill_sections = None
        skill_state = None
        chosen_skill = skill if skill is not None else software
        if chosen_skill:
            skill_state = self._load_skill(chosen_skill)
            skill_sections = skill_state.get("skill_sections")

        native_file, _ = self._native_structure_spec(skill_state)
        base_prompt = self._build_prompt(
            structure_content=structure_content,
            request=request,
            software=software,
            skill_sections=skill_sections,
            native_structure_file=native_file,
        )

        original_block = "\n\n".join(
            f"## ORIGINAL {filename} TO IMPROVE:\n{content}"
            for filename, content in original_inputs.items()
        )

        prompt = (
            f"{base_prompt}\n\n"
            f"{original_block}\n\n"
            f"{improvement_instructions}"
        )

        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = self._parse_response(response.text)
        except Exception as e:
            self.logger.error(f"Failed to generate improved inputs: {e}")
            return {
                "status": "error",
                "message": f"Failed to generate improved inputs: {e}",
            }

        # Same legacy-shape rescue as generate_inputs
        if "input_files" not in result:
            legacy_files = {}
            for legacy_key, canonical in (
                ("incar", "INCAR"), ("kpoints", "KPOINTS"),
                ("poscar", "POSCAR"),
            ):
                if legacy_key in result:
                    legacy_files[canonical] = result.pop(legacy_key)
            if legacy_files:
                result["input_files"] = legacy_files

        if not isinstance(result.get("input_files"), dict):
            return {
                "status": "error",
                "message": "No input_files in LLM response",
                "raw_result": result,
            }

        # Keep the engine-native coordinate file exact (overwrite any LLM copy)
        # before writing the improved set to disk.
        self._inject_native_structure(result, structure_file, skill_state)

        os.makedirs(output_dir, exist_ok=True)
        improved_paths = {}
        for filename, content in result["input_files"].items():
            path = os.path.join(output_dir, f"{filename}_improved")
            with open(path, 'w') as f:
                f.write(content)
            improved_paths[filename] = path

        result.update({
            "status": "success",
            "software": software,
            "improvements_applied": True,
            "adjustments_count": len(adjustments),
            "improved_paths": improved_paths,
        })

        self.logger.info(
            f"Generated improved {software} inputs with "
            f"{len(adjustments)} literature-based corrections"
        )
        return result
