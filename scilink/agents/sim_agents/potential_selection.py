"""Potential-family selection for molecular-dynamics tasks.

Once the router has decided a task is molecular dynamics and the structure
exists, this step decides where the interatomic potential comes from — a
classical force field or a machine-learning potential (MLIP) — using the
``potential_selection`` skill as guidance and an LLM as the judge. It is the
layer *above* "which force field" / "which MLIP": it picks the family, and the
``force_field`` / ``machine_learning_potentials`` skills pick the member.

This is deliberately made after the structure is built, so the species are
known and force-field coverage is a checkable fact rather than a guess at
routing time. See ``docs/proposals/mlip-as-potential-source.md`` (issue #429).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ...auth import get_internal_proxy_key, require_vendor_credentials
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...skills.loader import load_skill
from ._deprecation import normalize_params

logger = logging.getLogger(__name__)

_FAMILIES = ("force_field", "mlip")
# When the LLM cannot be reached or returns garbage, default to the MLIP:
# a universal pretrained potential degrades gracefully across chemistries,
# whereas a classical force field with an uncovered species fails silently.
_FALLBACK_FAMILY = "mlip"


def _build_model(model_name: str, api_key: Optional[str], base_url: Optional[str]):
    """Construct an LLM client the same way the foundation agents do."""
    api_key, base_url = normalize_params(
        api_key=api_key, base_url=base_url, source="PotentialSelection",
    )
    if base_url:
        if api_key is None:
            api_key = get_internal_proxy_key()
        return OpenAIAsGenerativeModel(
            model=model_name, api_key=api_key, base_url=base_url,
        )
    if api_key is None:
        require_vendor_credentials(model_name)
    return LiteLLMGenerativeModel(model=model_name, api_key=api_key)


def _species_from_structure(structure_file: str):
    """(sorted element symbols, n_atoms) from a structure file, best-effort."""
    try:
        from ase.io import read as _ase_read
        atoms = _ase_read(structure_file)
        return sorted(set(atoms.get_chemical_symbols())), len(atoms)
    except Exception as exc:  # unreadable / exotic format — reason from the goal
        logger.debug("Could not read species from %s: %r", structure_file, exc)
        return [], 0


def _parse_decision(text: str) -> Dict[str, Any]:
    """Parse the LLM response, tolerating code fences / preamble."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse potential-selection JSON: {text[:200]}")


def _build_prompt(guidance: str, species, n_atoms: int, research_goal: str) -> str:
    species_line = (
        ", ".join(species) + f"  ({n_atoms} atoms)" if species
        else "(could not read species from the structure — infer from the goal)"
    )
    return (
        "You choose the interatomic-potential FAMILY for a molecular-dynamics "
        "run: a classical force field, or a machine-learning potential (MLIP). "
        "Decide from the system's chemistry and the research goal, using the "
        "guidance below. You are NOT choosing a specific force field or MLIP — "
        "only the family.\n\n"
        "GUIDANCE:\n"
        f"{guidance}\n\n"
        "SYSTEM SPECIES:\n"
        f"{species_line}\n\n"
        "RESEARCH GOAL:\n"
        f"{research_goal}\n\n"
        "OUTPUT FORMAT: Return ONLY a JSON object, no prose:\n"
        "{\n"
        '  "family": "force_field" | "mlip",\n'
        '  "reasoning": "<one or two sentences naming the species or goal '
        'feature that drove the choice>"\n'
        "}"
    )


def select_potential_family(
    *,
    structure_file: str,
    research_goal: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    force_field_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Decide classical-FF vs MLIP for a molecular-dynamics task.

    Returns ``{"family": "force_field"|"mlip", "reasoning": str,
    "source": "forced"|"llm"|"fallback"}``. A caller that supplied
    ``force_field_files`` has already chosen classical, so no LLM call is made.
    """
    if force_field_files:
        return {"family": "force_field",
                "reasoning": "caller supplied force-field files",
                "source": "forced"}

    species, n_atoms = _species_from_structure(structure_file)
    try:
        skill = load_skill("potential_selection", domain="potential_selection")
        guidance = "\n\n".join(
            s for s in (skill.get("overview", ""), skill.get("planning", ""))
            if s.strip()
        )
    except Exception as exc:
        logger.warning("potential_selection skill unavailable (%r)", exc)
        guidance = ""

    prompt = _build_prompt(guidance, species, n_atoms, research_goal)
    try:
        model = _build_model(model_name, api_key, base_url)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or str(response)
        decision = _parse_decision(text)
    except Exception as exc:
        logger.warning(
            "potential selection failed (%r); defaulting to %s",
            exc, _FALLBACK_FAMILY,
        )
        return {"family": _FALLBACK_FAMILY,
                "reasoning": f"selection failed, defaulted to "
                             f"{_FALLBACK_FAMILY}: {exc!r}",
                "source": "fallback"}

    family = decision.get("family")
    if family not in _FAMILIES:
        logger.warning(
            "potential selection returned invalid family %r; defaulting to %s",
            family, _FALLBACK_FAMILY,
        )
        return {"family": _FALLBACK_FAMILY,
                "reasoning": f"invalid family {family!r}, defaulted to "
                             f"{_FALLBACK_FAMILY}",
                "source": "fallback"}

    return {"family": family,
            "reasoning": decision.get("reasoning", ""),
            "source": "llm"}
