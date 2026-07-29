# scilink/agents/sim_agents/molecular_qc_agent.py
"""
Molecular quantum-chemistry agent. Handles finite-system (non-periodic)
electronic-structure codes (NWChem, PySCF, ORCA, Gaussian, ...) via skill
bundles.

The discriminator from PeriodicDFTAgent is *finite / molecular* — no
periodic boundary conditions — NOT the method or the basis. This agent is
therefore both **method-agnostic** (DFT, HF, MP2, CCSD(T), CASSCF are a
task-line/keyword choice inside the same deck) and **basis-agnostic**
(Gaussian for NWChem/PySCF/ORCA; numerical-orbital, Slater, or real-space
for other engines). Engine-specific behavior — deck syntax, basis-set
paradigm, solvation keywords (COSMO/SMD), analytic-frequency
thermochemistry, post-HF task lines — lives in the per-engine skill
bundles at scilink/skills/molecular_qc/<engine>/<engine>.md and the
sibling tools modules.

Today NWChem is the first fully wired engine. Adding PySCF / ORCA /
Gaussian is a sibling skill bundle drop-in — no change to this class.
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


class MolecularQCAgent:
    """Molecular (finite-system) quantum-chemistry agent.

    Scale-aware (molecular electronic structure), software-agnostic,
    method-agnostic, basis-agnostic. Engine-specific behavior lives in
    skill bundles at ``scilink/skills/molecular_qc/<engine>/<engine>.md``.

    Currently supports NWChem via the ``nwchem`` skill bundle; PySCF /
    ORCA / Gaussian are extension points (drop in a sibling bundle).
    """

    SKILL_DOMAIN = "molecular_qc"

    @classmethod
    def supported_software(cls) -> list:
        """
        Auto-discover engine names this agent can currently handle.

        Returns every skill bundle name found for the agent's
        ``SKILL_DOMAIN`` across both built-in skills and any user-
        provided roots from ``$SCILINK_SKILLS_PATH``. A user dropping
        in their own ``molecular_qc/orca/orca.md`` will see ``orca``
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
        Initialize MolecularQCAgent.

        Parameters
        ----------
        api_key : str, optional
            API key for the LLM provider.
        model_name : str, optional
            Model name to use.
        base_url : str, optional
            Base URL for internal proxy.

        Mirrors PeriodicDFTAgent's credential plumbing exactly so both
        scale agents resolve LLM access identically.
        """
        self.logger = logging.getLogger(__name__)
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="MolecularQCAgent"
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

    def _load_skill(self, skill: str) -> dict:
        """
        Load a molecular-QC skill bundle (default: ``nwchem``).

        Parameters
        ----------
        skill : str
            Skill name (resolved from ``scilink/skills/molecular_qc/``)
            or path to a .md file.

        Returns
        -------
        dict with skill_name and skill_sections, or empty skill state
        on failure.
        """
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(skill, domain=self.SKILL_DOMAIN)
            self.logger.info(
                f"Loaded {self.SKILL_DOMAIN} skill: {parsed.get('name', skill)}"
            )
            return {
                "skill_name": parsed.get("name", skill),
                "skill_sections": parsed,
            }
        except FileNotFoundError:
            self.logger.warning(
                f"Skill '{skill}' not found under '{self.SKILL_DOMAIN}' — "
                f"proceeding without domain skill"
            )
            return {"skill_name": None, "skill_sections": None}
        except Exception as e:
            self.logger.warning(f"Failed to load skill '{skill}': {e}")
            return {"skill_name": None, "skill_sections": None}

    def _build_prompt(self, structure_content: str, request: str,
                      software: str,
                      skill_sections: Optional[dict] = None) -> str:
        """
        Build the full prompt, injecting skill content if available.

        Scaffold is software-, method-, and basis-agnostic — engine
        specifics (deck syntax, basis paradigm, solvation keywords, post-HF
        task lines, method selection) come entirely from the loaded skill
        bundle's planning / implementation / validation sections. The
        ``software`` argument is interpolated so the LLM emits the right
        engine's canonical filenames (e.g. a single ``job.nw`` deck for
        NWChem, a ``run.py`` script for PySCF, an ``orca.inp`` for ORCA).
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
            f"You are an expert in molecular (finite-system) "
            f"electronic-structure calculations using {software.upper()}.\n\n"
            f"Your task is to generate the input file(s) needed for the "
            f"requested calculation, following the guidance above. The system "
            f"is a finite molecule, ion, or small cluster (NOT a periodic "
            f"solid). Set charge and spin multiplicity explicitly and "
            f"consistently with the structure. Choose the method (DFT "
            f"functional or wavefunction method such as HF/MP2/CCSD(T)) and "
            f"basis appropriate to the request, and include an implicit "
            f"solvation model and/or analytic frequencies when the request "
            f"calls for solution-phase energetics or thermochemistry.\n\n"
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
            f"expects (e.g. a single ``job.nw`` deck for NWChem, ``run.py`` "
            f"for PySCF, ``orca.inp`` for ORCA). Filenames are case-sensitive."
        )
        if skill_parts:
            return "\n\n".join(skill_parts) + "\n\n---\n\n" + base
        return base

    def _parse_response(self, response_text: str) -> dict:
        """Robustly parse LLM response, handling common formatting issues."""
        text = response_text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")

    def generate_inputs(self, structure_file: str,
                        request: str,
                        software: str = "nwchem",
                        skill: Optional[str] = None) -> dict:
        """
        Generate input files for a molecular QC calculation.

        Scale-aware, software-agnostic entry point. The ``software``
        argument names the engine (which must have a matching skill
        bundle at ``scilink/skills/molecular_qc/<software>/``); the skill
        bundle's content drives all engine-specific behavior.

        Parameters
        ----------
        structure_file : str
            Path to the structure file (XYZ / engine-specific). Read as
            raw text and passed verbatim to the LLM (the skill bundle
            tells the LLM how to interpret it).
        request : str
            Natural-language description of the calculation.
        software : str, optional
            Which engine's inputs to generate. Default ``"nwchem"``.
        skill : str, optional
            Skill bundle name to load. Defaults to ``software``. Pass an
            explicit name (or full path to a .md file) to override; set
            to None to skip skill loading.

        Returns
        -------
        dict with keys:
            status      : "success" or "error"
            software    : echo of the software argument
            input_files : {filename: content} for every input file
            entry_file  : the deck/script the refinement loop runs (set
                          when a single input file is produced)
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
        chosen_skill = skill if skill is not None else software
        if chosen_skill:
            skill_state = self._load_skill(chosen_skill)
            skill_sections = skill_state.get("skill_sections")

        prompt = self._build_prompt(
            structure_content=structure_content,
            request=request,
            software=software,
            skill_sections=skill_sections,
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

        if not isinstance(result.get("input_files"), dict) or not result["input_files"]:
            return {
                "status": "error",
                "message": "LLM response did not include an 'input_files' object",
                "raw_result": result,
            }

        # ── pre-submit syntax pass (engine-neutral, advisory) ─────────
        # Ask the active engine's skill bundle for its syntax checker/fixer
        # via the registry. The agent never names an engine or imports a
        # bundle by path — an engine with no such tools simply skips this.
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
                    result["syntax_check"] = {
                        "issues": issues,
                        "applied_fixes": applied,
                    }
        except Exception as exc:
            self.logger.warning("input syntax check failed: %s", exc)

        # Record the entry deck/script so the refinement loop knows what to
        # run. Single-file engines (NWChem deck, PySCF script) are the common
        # case; multi-file callers can override downstream.
        if "entry_file" not in result and len(result["input_files"]) == 1:
            result["entry_file"] = next(iter(result["input_files"]))

        result["status"] = "success"
        result["software"] = software
        return result

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """
        Save every entry in ``result["input_files"]`` to ``output_dir``
        using the dict key as the literal filename. Software-agnostic.

        Returns a dict mapping each filename to the absolute path it was
        written to, plus ``"error"`` on failure.
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
