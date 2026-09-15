"""
VASP-specific pre-run input validator.

Two-layer pre-run validation for VASP INCAR files:

  1. Engine-native syntax check  (``check_incar_syntax``)
     Wraps pymatgen's ``Incar.check_params()`` to catch typoed /
     non-conforming INCAR tags (e.g. ``ISPN = 2`` instead of ``ISPIN``).
     No LLM call.  Returns structured issues with high/low-confidence
     rename suggestions derived from difflib against the canonical
     VASP tag list.

  2. LLM / literature check       (``IncarValidatorAgent.validate_and_improve_incar``)
     Asks a FutureHouse literature agent + an LLM to judge whether the
     parameter choices are physically appropriate for the system.
     Pre-existing behavior — moved here from val_agent.py as part of
     the 2026-05-17 split.

Why two layers:  VASP accepts unknown INCAR keys silently — it just
emits a one-line OUTCAR warning and runs with that key ignored.  An
``ISPN`` typo can therefore disable spin polarisation on Fe and produce
a physics-wrong result that converges by every other metric.  The
syntax layer catches this before submission; the LLM layer catches
"valid tag, wrong value for this system".

Engine-neutral consumers (orchestrators, benchmark harness, future
meta-agent) call ``check_incar_syntax`` / ``apply_incar_syntax_fixes``
and never import pymatgen directly.  pymatgen import is lazy so a
machine without it returns an empty issue list rather than crashing.

The shape ``check_syntax(content) -> List[issue]`` + optional
``validate_and_improve()`` is the engine-neutral contract.  Future
``lammps_input_validator.py`` and ``gromacs_input_validator.py``
should mirror it; see CLAUDE.md "Engine-neutral contracts".
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
    require_vendor_credentials,
)
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ..lit_agents.literature_agent import IncarLiteratureAgent
from ._deprecation import normalize_params
from .instruct import INCAR_VALIDATION_INSTRUCTIONS

# The deterministic INCAR syntax helpers live in the VASP skill bundle
# (scilink/skills/periodic_dft/vasp/vasp_syntax.py) — the single source of
# truth, discovered via the skill registry. Re-exported here so existing
# imports and IncarValidatorAgent.check_syntax keep resolving unchanged.
from ...skills.periodic_dft.vasp.vasp_syntax import (  # noqa: F401
    apply_incar_syntax_fixes,
    check_incar_syntax,
)


class IncarValidatorAgent:
    """Agent that validates and suggests improvements to VASP INCAR files.

    Two surfaces:

      * ``check_syntax(incar_content)`` — engine-native pymatgen check.
        No LLM, no FutureHouse dependency.  Returns the same list shape
        as the module-level ``check_incar_syntax`` and is provided as a
        convenience so callers that already hold a validator instance
        don't need a separate import.

      * ``validate_and_improve_incar(incar_content, system_description)``
        — LLM-driven literature review of parameter choices.  Requires
        an LLM model + FutureHouse API key (existing pre-PR behavior).

    The two surfaces are independent: a caller that only needs the
    syntax pass can use the module function and skip the LLM init
    entirely.
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 futurehouse_api_key: str = None,
                 max_wait_time: int = 500,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):

        self.logger = logging.getLogger(__name__)

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="IncarValidatorAgent"
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
            # Public / LiteLLM — delegate model→provider→env-var resolution
            # to LiteLLM (works for any model LiteLLM supports; raises a
            # message naming the missing vendor env var if not).
            if api_key is None:
                require_vendor_credentials(model_name)
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None

        self.literature_agent = IncarLiteratureAgent(
            api_key=futurehouse_api_key,
            max_wait_time=max_wait_time
        )

    def check_syntax(self, incar_content: str) -> List[Dict[str, Any]]:
        """Instance-method convenience wrapper around ``check_incar_syntax``."""
        return check_incar_syntax(incar_content)

    def validate_and_improve_incar(self, incar_content: str,
                                   system_description: str) -> dict:
        """Validate INCAR parameters and suggest improvements based on literature."""

        self.logger.info("Getting literature review of INCAR parameters...")
        lit_result = self.literature_agent.validate_incar(
            incar_content, system_description
        )

        if lit_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Literature review failed: {lit_result.get('message')}",
                "validation_status": "unknown"
            }

        self.logger.info("Analyzing literature review for potential improvements...")

        prompt = f"""{INCAR_VALIDATION_INSTRUCTIONS}

## ORIGINAL INCAR:
{incar_content}

## SYSTEM DESCRIPTION:
{system_description}

## LITERATURE REVIEW:
{lit_result['response']}

Analyze the literature review and suggest specific parameter adjustments if needed."""

        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = json.loads(response.text)
            result.update({
                "status": "success",
                "literature_review": lit_result['response'],
                "literature_task_id": lit_result.get('task_id')
            })
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing literature review: {e}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}",
                "literature_review": lit_result['response']
            }

    def save_validation_report(self, validation_result: dict,
                               output_dir: str = ".") -> dict:
        """Save validation report and revised INCAR if needed."""
        if validation_result.get("status") != "success":
            return {"error": "Validation was not successful"}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        try:
            report_path = os.path.join(output_dir, "incar_validation_report.json")
            with open(report_path, 'w', encoding="utf-8") as f:
                json.dump(validation_result, f, indent=2, default=str)
            saved_files["validation_report"] = report_path

            if (validation_result.get("validation_status") == "needs_adjustment" and
                    validation_result.get("revised_incar")):

                revised_path = os.path.join(output_dir, "INCAR_revised")
                with open(revised_path, 'w', encoding="utf-8") as f:
                    f.write(validation_result["revised_incar"])
                saved_files["revised_incar"] = revised_path

                summary_path = os.path.join(output_dir, "incar_adjustments.txt")
                with open(summary_path, 'w', encoding="utf-8") as f:
                    f.write("INCAR Parameter Adjustments\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Overall Assessment: {validation_result.get('overall_assessment', 'N/A')}\n\n")

                    adjustments = validation_result.get("suggested_adjustments", [])
                    if adjustments:
                        f.write("Suggested Changes:\n")
                        for adj in adjustments:
                            f.write(f"\n• {adj.get('parameter')}:\n")
                            f.write(f"  Current: {adj.get('current_value')}\n")
                            f.write(f"  Suggested: {adj.get('suggested_value')}\n")
                            f.write(f"  Reason: {adj.get('reason')}\n")
                    else:
                        f.write("No specific adjustments suggested.\n")

                saved_files["adjustment_summary"] = summary_path

            self.logger.info(f"Validation report saved: {saved_files}")
            return saved_files

        except Exception as e:
            self.logger.error(f"Error saving validation files: {e}")
            return {"error": f"Save failed: {str(e)}"}
