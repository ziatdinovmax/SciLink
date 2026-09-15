"""
Structure planning layer (the two-axis reasoning step).

Given a free-text request, the StructurePlanner decides — in ONE bounded LLM call —
both axes that govern how a structure should be built, and the constraints that
follow from their interaction:

  (i)  structure_class   — crystal / molecular / condensed / biomolecular
                           (the HOW-to-build axis; selects a structure_generation skill)
  (ii) simulation_scale  — periodic_dft / molecular_dft / molecular_dynamics /
                           machine_learning_potentials (the existing SKILL_DOMAIN axis,
                           shared with SimulationRouter)

and derives the structure-relevant constraints conditioned on (class × scale):
size_target, periodicity, solvation, charge/spin. These cross-terms are real — e.g.
classical MD handles larger boxes with EXPLICIT solvent, MLIPs run smaller cells than
classical FF, molecular DFT prefers IMPLICIT solvent — so neither axis alone is enough.

It emits a :class:`StructureSpec` (typed, serializable, overridable), which
:meth:`StructurePipeline.build_structure` consumes (``structure_class`` today;
constraint injection is a follow-up).

Relationship to ``SimulationRouter``: the router picks ``(scale, engine)`` from the
user's *installed* software (engine-availability-gated) for dispatch. The planner is
about STRUCTURE building and does NOT require any engine installed — it reasons over
the same scale vocabulary (it reuses ``DEFAULT_SCALE_DESCRIPTIONS``) but adds the
structure_class + constraints. The two are complementary tools, not duplicates: the
chat orchestrator can call ``route_simulation`` for engine dispatch and
``plan_structure`` for the build spec.

The control flow is deterministic Python; the LLM is used only for the bounded
classification/derivation step, returning a structured result.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ...auth import (
    require_vendor_credentials,
)
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .simulation_router import DEFAULT_SCALE_DESCRIPTIONS

_logger = logging.getLogger(__name__)


# The canonical structure-generation classes (skills under
# structure_generation/<class>/). Other skills in that domain (e.g. aimsgb) are
# *techniques within* a class, not top-level classes, so they're excluded here.
STRUCTURE_CLASSES = ["crystal", "molecular", "condensed", "biomolecular"]


# Structure-relevant defaults per simulation scale — the conditioning policy the
# planner reasons over (the LLM adapts these to the specific request). This is the
# declarative home for the cross-axis rules: MD -> large + explicit solvent,
# MLIP -> medium (smaller than classical FF), molecular_dft -> isolated + implicit
# solvent, periodic_dft -> small periodic cells.
SCALE_POLICY: Dict[str, Dict[str, str]] = {
    "periodic_dft": {
        "size_target": "small — ~10–200 atoms (up to ~500 with effort); keep cells minimal (expensive)",
        "periodicity": "3D periodic; slab + vacuum (≥12 Å) for surfaces",
        "solvation": "none, or implicit (continuum) — no explicit solvent molecules",
    },
    "molecular_dft": {
        "size_target": "single molecule / small cluster — ~1–100 atoms",
        "periodicity": "none (isolated, non-periodic)",
        "solvation": "implicit (continuum) at the DFT step — do NOT add explicit solvent unless asked",
    },
    "molecular_dynamics": {
        "size_target": "large — ~10^3–10^6 atoms",
        "periodicity": "periodic box",
        "solvation": "explicit solvent + neutralizing counter-ions (force field needed for all atoms)",
    },
    "machine_learning_potentials": {
        "size_target": "medium — ~10^2–10^4 atoms (smaller than classical FF for speed)",
        "periodicity": "periodic box or cluster",
        "solvation": "explicit or none, as the chemistry requires",
    },
}


@dataclass
class StructureSpec:
    """The structure-build contract emitted by :class:`StructurePlanner`.

    On a planning failure (LLM error or out-of-vocabulary axes), ``structure_class``
    and ``simulation_scale`` are ``None`` and ``error`` is set — the planner does NOT
    silently fall back to a default class. Programmatic callers should check
    :attr:`is_valid` (or call :meth:`raise_if_error`) before using the spec.
    """
    structure_class: Optional[str] = None      # crystal | molecular | condensed | biomolecular (None on error)
    simulation_scale: Optional[str] = None     # periodic_dft | molecular_dft | molecular_dynamics | machine_learning_potentials
    engine: Optional[str] = None               # vasp | qe | pyscf | lammps | mace | ...
    size_target: Optional[str] = None
    periodicity: Optional[str] = None
    solvation: Optional[str] = None
    charge_spin: Optional[str] = None
    reasoning: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """True when planning succeeded (no error and both axes resolved)."""
        return (self.error is None
                and self.structure_class is not None
                and self.simulation_scale is not None)

    def raise_if_error(self) -> "StructureSpec":
        """Raise ValueError if planning failed; else return self (for chaining)."""
        if not self.is_valid:
            raise ValueError(self.error or "StructurePlanner returned an invalid StructureSpec")
        return self

    def as_constraints_text(self) -> str:
        """Render the build constraints as a prompt block for the structure generator
        (empty string when none are set). This is how the planner's cross-axis
        reasoning (size / periodicity / solvation / charge) actually shapes generation."""
        fields = [("target size", self.size_target), ("periodicity", self.periodicity),
                  ("solvation", self.solvation), ("charge/spin", self.charge_spin)]
        lines = [f"- {label}: {val}" for label, val in fields if val]
        if not lines:
            return ""
        return "Build constraints (from the structure plan — honor these):\n" + "\n".join(lines)


class StructurePlanner:
    """LLM planner: free-text request -> :class:`StructureSpec` (two-axis + constraints).

    Construction (standalone)::

        planner = StructurePlanner(model_name="claude-opus-4-6")
        spec = planner.plan("a solvated lysozyme system for MD")
        spec.structure_class  # -> "biomolecular"

    Or reuse an existing model wrapper (e.g. from the chat orchestrator)::

        planner = StructurePlanner(model=orchestrator.model)
    """

    def __init__(self,
                 model: Any = None,
                 model_name: str = "claude-opus-4-6",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 available_software: Any = None,
                 scale_policy: Optional[Dict[str, Dict[str, str]]] = None,
                 generation_config: Optional[dict] = None):
        if model is not None:
            self.model = model
        else:
            if api_key is None and base_url is None:
                require_vendor_credentials(model_name)
            self.model = LiteLLMGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url
            )
        self.available_software = available_software
        self.scale_policy = dict(scale_policy if scale_policy is not None else SCALE_POLICY)
        self.generation_config = generation_config

    # ── public ────────────────────────────────────────────────────────

    def plan(self, request: str, system_description: Optional[str] = None) -> StructureSpec:
        """Decide structure_class, simulation_scale, engine, and constraints for ``request``.

        On failure the returned spec has ``error`` set and ``structure_class`` /
        ``simulation_scale`` = ``None`` (no silent default). Check ``spec.is_valid``
        or call ``spec.raise_if_error()`` before using it.
        """
        prompt = self._build_prompt(request, system_description)
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            text = getattr(response, "text", None) or str(response)
            decision = self._parse_response(text)
        except Exception as exc:
            return StructureSpec(error=f"Planner LLM call/parse failed: {exc!r}")

        sclass = decision.get("structure_class")
        scale = decision.get("simulation_scale")
        if sclass not in STRUCTURE_CLASSES or scale not in self.scale_policy:
            return StructureSpec(
                structure_class=sclass if sclass in STRUCTURE_CLASSES else None,
                simulation_scale=scale if scale in self.scale_policy else None,
                error=(f"LLM returned out-of-vocabulary axis "
                       f"(structure_class={sclass!r}, simulation_scale={scale!r}); "
                       f"check the request or the prompt."),
                reasoning=decision.get("reasoning", ""),
            )

        return StructureSpec(
            structure_class=sclass,
            simulation_scale=scale,
            engine=decision.get("engine"),
            size_target=decision.get("size_target"),
            periodicity=decision.get("periodicity"),
            solvation=decision.get("solvation"),
            charge_spin=decision.get("charge_spin"),
            reasoning=decision.get("reasoning", ""),
            alternatives=decision.get("alternatives", []) or [],
        )

    # ── internals ─────────────────────────────────────────────────────

    def _class_options(self) -> Dict[str, str]:
        """{class: one-line description} pulled from each class skill's frontmatter."""
        out: Dict[str, str] = {}
        try:
            from ...skills.loader import load_skill
        except Exception:
            return {c: c for c in STRUCTURE_CLASSES}
        for c in STRUCTURE_CLASSES:
            try:
                desc = (load_skill(c, domain="structure_generation").get("meta") or {}).get("description")
            except Exception:
                desc = None
            out[c] = desc or c
        return out

    def _build_prompt(self, request: str, system_description: Optional[str]) -> str:
        classes_block = "\n".join(
            f"  - {name}: {desc}" for name, desc in self._class_options().items()
        )
        scales_block = "\n".join(
            f"  - {scale}: {DEFAULT_SCALE_DESCRIPTIONS.get(scale, 'no description')}\n"
            f"      structure defaults → size: {pol.get('size_target','?')}; "
            f"periodicity: {pol.get('periodicity','?')}; solvation: {pol.get('solvation','?')}"
            for scale, pol in self.scale_policy.items()
        )
        avail = ""
        if self.available_software is not None:
            try:
                avail_map = {
                    s: self.available_software.list_available(domain=s)
                    for s in self.scale_policy
                }
                avail = ("\nINSTALLED ENGINES (prefer these for `engine` when a scale matches):\n"
                         + "\n".join(f"  - {s}: {e}" for s, e in avail_map.items() if e) + "\n")
            except Exception:
                avail = ""
        sys_desc = system_description or "(none provided)"

        return (
            "You are SciLink's structure-planning assistant. From the user's request, decide how an "
            "INITIAL atomic structure should be built, along TWO axes — (1) the STRUCTURE CLASS (what kind "
            "of structure to build) and (2) the SIMULATION SCALE (what method/engine it is for) — plus the "
            "constraints that follow from their interaction. Reason about the physics; the constraints are "
            "cross-terms of these two axes (e.g. classical MD uses larger boxes with EXPLICIT solvent; MLIPs "
            "use smaller cells than classical force fields; molecular DFT uses IMPLICIT solvent and no periodic "
            "cell; periodic DFT keeps cells small).\n\n"
            f"USER REQUEST:\n{request}\n\n"
            f"SYSTEM DESCRIPTION:\n{sys_desc}\n\n"
            "AXIS 1 — STRUCTURE CLASS (how to build; pick exactly one):\n"
            f"{classes_block}\n\n"
            "AXIS 2 — SIMULATION SCALE (what it's for; pick exactly one) with structure defaults:\n"
            f"{scales_block}\n"
            f"{avail}\n"
            "DECISION CRITERIA:\n"
            "  1. structure_class from the physical TYPE of the system (periodic solid → crystal; isolated "
            "molecule/cluster → molecular; many-molecule liquid/solution/amorphous box → condensed; "
            "protein/nucleic-acid/membrane → biomolecular).\n"
            "  2. simulation_scale from the goal/method implied by the request (and installed engines if listed).\n"
            "  3. Derive size_target / periodicity / solvation / charge_spin from the chosen scale's defaults, "
            "adapted to the specific request (honor explicit numbers/conditions the user gave).\n"
            "  4. engine: a conventional engine for that scale (prefer an installed one if listed); null if unsure.\n\n"
            "OUTPUT FORMAT — return ONLY a JSON object, no prose:\n"
            "{\n"
            '  "structure_class": "<one of: crystal | molecular | condensed | biomolecular>",\n'
            '  "simulation_scale": "<one of: periodic_dft | molecular_dft | molecular_dynamics | machine_learning_potentials>",\n'
            '  "engine": "<engine name or null>",\n'
            '  "size_target": "<short phrase>",\n'
            '  "periodicity": "<3d | slab | none | box>",\n'
            '  "solvation": "<none | implicit | explicit>",\n'
            '  "charge_spin": "<short note or null>",\n'
            '  "reasoning": "<1-2 sentences>",\n'
            '  "alternatives": [{"structure_class": "...", "simulation_scale": "...", "tradeoff": "<one line>"}]\n'
            "}"
        )

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """Parse the LLM JSON, tolerating code fences / preamble."""
        text = (text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse planner JSON: {text[:300]}")
