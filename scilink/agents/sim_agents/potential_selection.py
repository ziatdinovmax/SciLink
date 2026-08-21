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


def _available_families(available_software) -> set:
    """Potential families that can actually run, from installed software.

    Classical FF runs through a classical MD engine; an MLIP runs through ASE
    or its own backend. Returns the empty set when availability can't be probed
    (the choice is then left unconstrained). See issue #429.
    """
    try:
        from ...utils.available_software import AvailableSoftware
        sw = available_software or AvailableSoftware.auto()
        fams = set()
        if sw.list_available(domain="molecular_dynamics"):
            fams.add("force_field")
        if sw.list_available(domain="machine_learning_potentials"):
            fams.add("mlip")
        return fams
    except Exception as exc:
        logger.debug("could not determine available potential families: %r", exc)
        return set()


def select_potential_family(
    *,
    structure_file: str,
    research_goal: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    force_field_files: Optional[Dict[str, str]] = None,
    available_software: Any = None,
) -> Dict[str, Any]:
    """Decide classical-FF vs MLIP for a molecular-dynamics task.

    Returns ``{"family": "force_field"|"mlip", "reasoning": str, "source": ...}``.
    The family is constrained to what is installed (``available_software``, or
    ``AvailableSoftware.auto()`` when None): a caller that supplied
    ``force_field_files`` has already chosen classical; when only one family is
    installed the decision is forced with no LLM call; otherwise the model
    chooses and its pick (and the fallback) are clamped to installed families,
    so an MLIP is never selected without an MLIP backend present (issue #429).
    """
    if force_field_files:
        return {"family": "force_field",
                "reasoning": "caller supplied force-field files",
                "source": "forced"}

    # Availability gate: a family that isn't installed can't run.
    available = _available_families(available_software)
    if len(available) == 1:
        fam = next(iter(available))
        logger.info("only the %s potential family is installed; selecting it "
                    "without an LLM call", fam)
        return {"family": fam,
                "reasoning": f"only the {fam} family is installed",
                "source": "forced-availability"}

    def _fallback_family(reason: str) -> Dict[str, Any]:
        # Prefer an installed family; the MLIP degrades most gracefully.
        fam = "mlip" if (not available or "mlip" in available) else "force_field"
        return {"family": fam, "reasoning": reason, "source": "fallback"}

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
        logger.warning("potential selection failed (%r); falling back", exc)
        return _fallback_family(f"selection failed: {exc!r}")

    family = decision.get("family")
    if family not in _FAMILIES:
        logger.warning("potential selection returned invalid family %r; "
                       "falling back", family)
        return _fallback_family(f"invalid family {family!r}")

    if available and family not in available:
        fam = next(iter(available))
        logger.info("model chose %s but only %s is installed; using %s",
                    family, sorted(available), fam)
        return {"family": fam,
                "reasoning": f"model chose {family}, constrained to installed "
                             f"{fam}: {decision.get('reasoning', '')}",
                "source": "constrained"}

    return {"family": family,
            "reasoning": decision.get("reasoning", ""),
            "source": "llm"}
