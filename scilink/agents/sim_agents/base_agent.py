# agents/sim_agents/base_agent.py
"""
Base class for all simulation agents.

Provides shared infrastructure:
  - Working directory and logging
  - LLM model initialization
  - Skill loading and context injection
  - JSON/text generation helpers
  - Validation pattern (tools then LLM fallback)
  - Iterative refinement (refine, fix_error)
  - Output cleaning

Scale-specific subclasses (MDSimulationAgent, ElectronicStructureAgent, etc.)
implement the domain-specific pipeline: analyze, plan, generate.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
    require_vendor_credentials,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...skills.loader import load_skill, list_skills
from ._deprecation import normalize_params


class SimulationAgent(ABC):
    """
    Abstract base for all simulation agents.

    Subclasses must define:
        SKILL_DOMAIN:  str         -- e.g., "md_simulation", "dft", "fea"
        EXTENSION_MAP: dict        -- file extensions to skill names
        TOOL_REGISTRY: dict        -- skill names to tool modules
        analyze_system()           -- parse structure/mesh into system info
        plan_simulation()          -- research goal to parameter plan
        generate_simulation()      -- full pipeline
    """

    SKILL_DOMAIN: str = ""
    EXTENSION_MAP: Dict[str, str] = {}
    TOOL_REGISTRY: Dict[str, Any] = {}

    def __init__(
        self,
        working_dir: str,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        skill: Optional[str] = None,
        local_model: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(exist_ok=True, parents=True)

        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source=self.__class__.__name__,
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                raise ValueError("API key required for internal proxy.")
            self.logger.info(f"Using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url
            )
        else:
            # Public / LiteLLM — delegate model→provider→env-var resolution
            # to LiteLLM (works for any model LiteLLM supports; raises a
            # message naming the missing vendor env var if not).
            if api_key is None:
                require_vendor_credentials(model_name)
            self.logger.info(f"Using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name, api_key=api_key
            )

        self.skill_name: Optional[str] = None
        self.skill_sections: Optional[Dict[str, str]] = None
        self.tools_module = None

        try:
            self._available_skills = list_skills(domain=self.SKILL_DOMAIN)
        except Exception:
            self._available_skills = []

        if skill:
            self._load_skill(skill)

    # ================================================================
    # SKILL MANAGEMENT
    # ================================================================

    def _load_skill(self, skill: str) -> bool:
        try:
            parsed = load_skill(skill, domain=self.SKILL_DOMAIN)
            self.skill_name = parsed["name"]
            self.skill_sections = parsed
            self.logger.info(f"Loaded skill: {self.skill_name}")

            if self.skill_name in self.TOOL_REGISTRY:
                self.tools_module = self.TOOL_REGISTRY[self.skill_name]
                self.logger.info(f"Tools available: {self.skill_name}")
            else:
                self.tools_module = None
                self.logger.info(f"No tools for '{self.skill_name}'")
            return True
        except FileNotFoundError:
            self.logger.warning(
                f"Skill '{skill}' not found in domain '{self.SKILL_DOMAIN}'. "
                f"Available: {self._available_skills}"
            )
            return False

    def _auto_select_skill(self, input_file: str) -> bool:
        if self.skill_sections is not None:
            return True

        ext = Path(input_file).suffix.lower()
        for skill_name, extensions in self.EXTENSION_MAP.items():
            if ext in extensions and skill_name in self._available_skills:
                return self._load_skill(skill_name)

        self.logger.warning(
            f"No skill matched for '{ext}'. Available: {self._available_skills}"
        )
        return False

    def _get_skill_context(
        self,
        section: Optional[str] = None,
        include_all: bool = False,
    ) -> str:
        if not self.skill_sections:
            return ""

        if include_all:
            parts = [f"=== Domain Knowledge: {self.skill_name} ==="]
            for key in (
                "overview", "planning", "analysis",
                "interpretation", "validation", "implementation",
            ):
                content = self.skill_sections.get(key, "")
                if content:
                    parts.append(f"\n--- {key.upper()} ---")
                    parts.append(content)
            return "\n".join(parts)

        if section:
            content = self.skill_sections.get(section, "")
            if content:
                return (
                    f"=== Domain Knowledge ({self.skill_name} -- {section}) ===\n"
                    f"{content}"
                )
            return ""

        overview = self.skill_sections.get("overview", "")
        if overview:
            return f"=== Domain Knowledge: {self.skill_name} ===\n{overview}"
        return ""

    # ================================================================
    # LLM HELPERS
    # ================================================================

    def _generate_json(self, prompt: str, _attempts: int = 3) -> Dict[str, Any]:
        """Generate a JSON object from the model, robust to a bad completion.

        An occasional empty or non-JSON response — a proxy hiccup, a
        content-filter stop, a stray prose wrapper — otherwise raises on the
        first try and aborts the whole generation, which is expensive mid-workflow
        (it burns the caller's retry budget without ever reaching the real work).
        So retry a few times, nudging explicitly for JSON on the retries, and
        salvage an embedded object before giving up.
        """
        attempts = max(1, _attempts)
        last_err: Any = None
        for i in range(attempts):
            p = prompt if i == 0 else (
                prompt + "\n\nRespond with ONLY a single valid JSON object — "
                "no prose, no code fences.")
            response = self.model.generate_content(
                p,
                generation_config={"response_mime_type": "application/json"},
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    last_err = e
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except json.JSONDecodeError as e2:
                            last_err = e2
            else:
                last_err = ValueError("empty model response")
            self.logger.warning(
                "JSON parse attempt %d/%d failed (%s); retrying",
                i + 1, attempts, last_err)
        self.logger.error(f"JSON parse failed after {attempts} attempts: {last_err}")
        raise ValueError(f"Could not parse JSON after {attempts} attempts: {last_err}")

    def _generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate(
        self,
        script_path: str,
        system_info: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.tools_module and hasattr(self.tools_module, "validate_script"):
            result = self.tools_module.validate_script(script_path, system_info)
            for e in result.get("errors", []):
                self.logger.error(f"Validation: {e}")
            for w in result.get("warnings", []):
                self.logger.warning(f"Validation: {w}")
            return result
        return self._llm_validate(script_path, plan)

    def _llm_validate(
        self, script_path: str, plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        content = Path(script_path).read_text()
        rules = self._get_skill_context(section="validation")
        if not rules:
            return {
                "valid": True,
                "errors": [],
                "warnings": ["No validation skill loaded"],
            }

        prompt = (
            "Validate this simulation input against the rules below.\n\n"
            "RULES:\n"
            f"{rules}\n\n"
            "INPUT FILE:\n"
            f"{content}\n\n"
            'Return JSON: {"valid": bool, "errors": [...], "warnings": [...]}'
        )
        try:
            return self._generate_json(prompt)
        except Exception as e:
            return {"valid": True, "errors": [], "warnings": [str(e)]}

    def _attempt_fix(
        self,
        script: str,
        errors: List[str],
        plan: Dict[str, Any],
    ) -> str:
        interpretation = self._get_skill_context(section="interpretation")
        implementation = self._get_skill_context(section="implementation")

        prompt = (
            "Fix these errors in the simulation input file.\n\n"
            "ERRORS:\n"
            f"{json.dumps(errors, indent=2)}\n\n"
            "SCRIPT:\n"
            f"{script}\n\n"
            "KNOWN ERRORS AND FIXES:\n"
            f"{interpretation}\n\n"
            "CORRECT SYNTAX:\n"
            f"{implementation}\n\n"
            "Fix ONLY the errors. Mark with: # FIXED: <description>\n"
            "Return ONLY the corrected file. No markdown."
        )
        try:
            fixed = self._generate_text(prompt)
            return self._clean_output(fixed)
        except Exception:
            return script

    # ================================================================
    # CLEANING
    # ================================================================

    def _clean_output(self, text: str) -> str:
        if self.tools_module and hasattr(self.tools_module, "clean_script"):
            return self.tools_module.clean_script(text)
        text = re.sub(r'`{3}(?:\w+)?', '', text)
        return text.replace('`' * 3, '').strip()

    # ================================================================
    # ITERATIVE REFINEMENT
    # ================================================================

    def refine(self, feedback: str) -> Dict[str, Any]:
        script_path = self._find_current_script()
        current = script_path.read_text()
        implementation = self._get_skill_context(section="implementation")

        prompt = (
            "Modify the simulation input per user request.\n\n"
            f'REQUEST: "{feedback}"\n\n'
            "CURRENT FILE:\n"
            f"{current}\n\n"
            "REFERENCE:\n"
            f"{implementation}\n\n"
            "Apply changes. Preserve everything else.\n"
            "Mark with: # CHANGED: <description>\n"
            "Return ONLY the file. No markdown."
        )
        updated = self._clean_output(self._generate_text(prompt))
        script_path.write_text(updated, encoding="utf-8")
        self.logger.info(f"Refined: {script_path}")
        return {
            "script_path": str(script_path),
            "refinement_type": "modification",
        }

    def fix_error(
        self, error_message: str, log_output: Optional[str] = None
    ) -> Dict[str, Any]:
        script_path = self._find_current_script()
        current = script_path.read_text()
        interpretation = self._get_skill_context(section="interpretation")
        implementation = self._get_skill_context(section="implementation")

        log_block = ""
        if log_output:
            log_block = f"\nFULL LOG (tail):\n{log_output[-3000:]}"

        prompt = (
            "Fix this runtime error.\n\n"
            f"ERROR: {error_message}\n"
            f"{log_block}\n\n"
            "SCRIPT:\n"
            f"{current}\n\n"
            "KNOWN ERRORS:\n"
            f"{interpretation}\n\n"
            "CORRECT SYNTAX:\n"
            f"{implementation}\n\n"
            "Fix ONLY what is needed. Mark: # FIXED: <description>\n"
            "Return ONLY the file. No markdown."
        )
        fixed = self._clean_output(self._generate_text(prompt))
        script_path.write_text(fixed, encoding="utf-8")
        self.logger.info(f"Fixed: {script_path}")
        return {
            "script_path": str(script_path),
            "refinement_type": "error_fix",
        }

    def _find_current_script(self) -> Path:
        for pattern in [
            "run.*", "*.lammps", "*.in", "INCAR", "*.inp", "*.py",
        ]:
            matches = sorted(self.working_dir.glob(pattern))
            if matches:
                return matches[-1]
        raise FileNotFoundError(f"No script in {self.working_dir}")

    # ================================================================
    # ABSTRACT METHODS
    # ================================================================

    @abstractmethod
    def analyze_system(self, input_file: str) -> Dict[str, Any]:
        """Parse structure/mesh file into system information dict."""
        raise NotImplementedError

    @abstractmethod
    def plan_simulation(
        self, research_goal: str, system_info: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Determine simulation parameters from research goal and system info."""
        raise NotImplementedError

    @abstractmethod
    def generate_simulation(
        self, structure_file: str, research_goal: str, **kwargs
    ) -> Dict[str, Any]:
        """Full pipeline: analyze, plan, generate, validate."""
        raise NotImplementedError
