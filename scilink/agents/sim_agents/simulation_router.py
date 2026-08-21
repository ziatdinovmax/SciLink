"""
Simulation routing layer.

Given a user's research goal and (optional) system description, picks
the (scale, engine) combination that best fits — out of the
intersection of:

    agent_supports × user_available × scale_appropriateness_for_goal

The first comes from each scale agent's ``supported_software()``
classmethod (scale-aware skill-bundle discovery), the second from
``AvailableSoftware.auto()`` (frontmatter probes + YAML cache), and
the third from an LLM call that interprets the user's goal against
each scale's physical regime.

Designed to be used either:
  - **Standalone** — ``SimulationRouter(model).route(goal, system)``
  - **As an orchestrator tool** — wrapped in a tool spec the
    SimulationOrchestratorAgent exposes to its chat LLM (future commit)

The router does NOT dispatch — it just decides. The orchestrator owns
the dispatch (constructing the chosen agent, running the workflow).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ...utils.available_software import AvailableSoftware

_logger = logging.getLogger(__name__)


# Scale descriptions — physical regime + when to pick each. Used as
# LLM context when routing. Keyed by SKILL_DOMAIN. Hardcoded today;
# easy to refactor later to a SCALE_DESCRIPTION class var on each
# scale agent or to a meta-skill if scopes grow.
DEFAULT_SCALE_DESCRIPTIONS: Dict[str, str] = {
    "periodic_dft": (
        "Planewave / pseudopotential DFT for periodic systems. "
        "Best for: bulk crystals, slab surfaces, metals, oxides, "
        "supercells with defects, transition states (NEB), electronic "
        "structure of solids. ~10-200 atoms typical; up to ~500 with "
        "effort. Engines: VASP, QE, ABINIT, CP2K."
    ),
    "molecular_qc": (
        "Molecular / finite-system electronic structure for isolated "
        "molecules, ions, and small clusters. Covers DFT and wavefunction "
        "methods (HF, MP2, CCSD(T), CASSCF) — method and basis are engine "
        "knobs, not separate scales. Best for: molecular geometry, reaction "
        "energetics/thermochemistry, implicit solvation, vibrational and "
        "excited-state analysis, charged ion pairs. NOT for periodic "
        "systems. Engines: NWChem, PySCF, ORCA, Gaussian. (May not be "
        "installed.)"
    ),
    "molecular_dynamics": (
        "Molecular dynamics — ensemble sampling of atomic motion for "
        "structure, transport, thermodynamics, and phase behavior at "
        "classical-MD cost. Covers biomolecules, polymers, and liquids "
        "as well as inorganic solids, metals, interfaces, and reactive "
        "or novel materials. The interatomic potential — a classical "
        "force field OR a machine-learning potential (MACE, CHGNet, "
        "NequIP, ...) — is selected AFTER routing, once the structure "
        "and its species are known, so whether a good classical force "
        "field exists for the chemistry does NOT affect this routing "
        "choice. Use for any dynamics / sampling goal (diffusion, "
        "viscosity, melting point, RDFs, free energies), regardless of "
        "chemistry. Engines: LAMMPS, GROMACS, OpenMM, AMBER."
    ),
}
# NOTE: ``machine_learning_potentials`` is deliberately NOT a scale. An MLIP
# is a *potential source*, not a physical regime — the classical-FF-vs-MLIP
# choice is made downstream by the potential-selection step once the species
# are known (see docs/proposals/mlip-as-potential-source.md, issue #429).


def discover_scale_agents() -> Dict[str, Dict[str, Any]]:
    """Walk the available scale agents and report what they support.

    Each known agent class declares a SKILL_DOMAIN class var and a
    ``supported_software()`` classmethod (auto-discovered from the
    skill bundles for its domain). Missing agents are tolerated — the
    router handles whichever scales are importable.

    Returns:
        ``{skill_domain: {"agent_class": <class>, "supported": [...]}}``
        for each scale agent that imports cleanly.
    """
    found: Dict[str, Dict[str, Any]] = {}

    try:
        from .periodic_dft_agent import PeriodicDFTAgent
        found[PeriodicDFTAgent.SKILL_DOMAIN] = {
            "agent_class": PeriodicDFTAgent,
            "supported": PeriodicDFTAgent.supported_software(),
        }
    except ImportError as exc:
        _logger.debug("PeriodicDFTAgent not importable: %s", exc)

    try:
        from .molecular_qc_agent import MolecularQCAgent
        found[MolecularQCAgent.SKILL_DOMAIN] = {
            "agent_class": MolecularQCAgent,
            "supported": MolecularQCAgent.supported_software(),
        }
    except ImportError as exc:
        _logger.debug("MolecularQCAgent not importable: %s", exc)

    try:
        from .md_simulation_agent import MDSimulationAgent
        # Today MDSimulationAgent reports its software via TOOL_REGISTRY
        # (runtime-keyed by successfully-imported tools modules). We
        # also union with skill-bundle discovery so a markdown-only
        # bundle counts even if its sibling .py wasn't importable.
        from ...skills.loader import list_skills
        supported = sorted(
            set(getattr(MDSimulationAgent, "TOOL_REGISTRY", {}).keys())
            | set(list_skills(domain="molecular_dynamics"))
        )
        found["molecular_dynamics"] = {
            "agent_class": MDSimulationAgent,
            "supported": supported,
        }
    except ImportError as exc:
        _logger.debug("MDSimulationAgent not importable: %s", exc)

    # NB: MLIPAgent is intentionally NOT registered as a scale. An MLIP is a
    # potential source selected downstream of ``molecular_dynamics`` routing,
    # not a physical regime the router chooses among (issue #429).

    return found


class SimulationRouter:
    """LLM-driven router picking (scale, engine) for a user's goal.

    Construction:

        from scilink.agents.sim_agents.simulation_router import SimulationRouter
        router = SimulationRouter(model=my_llm_client)
        decision = router.route(
            user_goal="Equilibrate Cu(111) + CO at 300 K and check binding",
            system_description="metallic surface slab, ~50 atoms",
        )

    The returned decision is a dict:

        {
            "scale":       "periodic_dft",
            "engine":      "vasp",
            "reasoning":   "Metallic Cu(111) with CO adsorbate is a "
                           "classic catalysis system; VASP is...",
            "alternatives": [
                {"scale": "molecular_dynamics",
                 "engine": "lammps",
                 "tradeoff": "Faster sampling; an MLIP would be selected "
                             "downstream since Cu/CO lacks a good classical FF"}
            ],
            "candidates_considered": {"periodic_dft": ["vasp"], ...},
        }

    On failure (no candidates, LLM error, or LLM picks invalid combo)
    the dict has ``scale``/``engine`` set to ``None`` plus an
    ``error`` key.
    """

    def __init__(self,
                 model: Any,
                 available_software: Optional[AvailableSoftware] = None,
                 scale_descriptions: Optional[Dict[str, str]] = None,
                 generation_config: Optional[dict] = None):
        self.model = model
        self.available_software = (
            available_software
            if available_software is not None
            else AvailableSoftware.auto()
        )
        self.scale_descriptions = dict(
            scale_descriptions if scale_descriptions is not None
            else DEFAULT_SCALE_DESCRIPTIONS
        )
        self.generation_config = generation_config

    # ── candidate computation ─────────────────────────────────────

    def candidate_engines(self) -> Dict[str, List[str]]:
        """Engines available to both the agent and the user, per scale.

        Returns:
            ``{scale: [engine, ...]}`` for scales with at least one
            engine in the intersection. Scales with no available
            engines are omitted.
        """
        scale_agents = discover_scale_agents()
        mlip_backends = self.available_software.list_available(
            domain="machine_learning_potentials")
        out: Dict[str, List[str]] = {}
        for scale, info in scale_agents.items():
            agent_supports = set(info.get("supported", []))
            user_has = set(self.available_software.list_available(domain=scale))
            intersection = sorted(agent_supports & user_has)
            if scale == "molecular_dynamics" and not intersection and mlip_backends:
                # The MD scale can be served by an MLIP through the ASE runner,
                # which needs no classical MD engine — so keep it routable for
                # the MLIP-only audience (e.g. pip ``mace-torch``, no LAMMPS).
                # The potential-selection step picks the MLIP family and the
                # pipeline runs it via ASE (issue #429, availability gate).
                intersection = ["ase"]
            if intersection:
                out[scale] = intersection
        return out

    # ── public route ──────────────────────────────────────────────

    def route(self,
              user_goal: str,
              system_description: Optional[str] = None) -> Dict[str, Any]:
        """Pick (scale, engine) for the user's goal."""
        candidates = self.candidate_engines()
        if not candidates:
            return {
                "scale": None,
                "engine": None,
                "error": (
                    "No engines available. Either no skill bundles match "
                    "the user's installed software, or AvailableSoftware "
                    "needs to be (re)detected. Try "
                    "`AvailableSoftware.refresh()`."
                ),
                "candidates_considered": {},
            }

        prompt = self._build_prompt(user_goal, system_description, candidates)
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config,
            )
            text = getattr(response, "text", None) or str(response)
            decision = self._parse_response(text)
        except Exception as exc:
            return {
                "scale": None,
                "engine": None,
                "error": f"Routing LLM call failed: {exc!r}",
                "candidates_considered": candidates,
            }

        scale = decision.get("scale")
        engine = decision.get("engine")
        if scale not in candidates or engine not in candidates.get(scale, []):
            return {
                "scale": None,
                "engine": None,
                "error": (
                    f"LLM picked scale={scale!r}, engine={engine!r}, "
                    f"but that combo is not in the candidate set "
                    f"{candidates}. Verify the prompt and re-run."
                ),
                "raw_decision": decision,
                "candidates_considered": candidates,
            }

        decision.setdefault("alternatives", [])
        decision["candidates_considered"] = candidates
        return decision

    # ── internals ─────────────────────────────────────────────────

    def _build_prompt(self,
                      user_goal: str,
                      system_description: Optional[str],
                      candidates: Dict[str, List[str]]) -> str:
        candidates_block = "\n".join(
            f"  - {scale}: {engines}"
            for scale, engines in sorted(candidates.items())
        )
        scale_descs_block = "\n".join(
            f"  - {scale}: "
            f"{self.scale_descriptions.get(scale, 'no description available')}"
            for scale in sorted(candidates)
        )
        sys_desc = (
            system_description
            if system_description else "(no explicit system description provided)"
        )

        return (
            "You are a simulation routing assistant for SciLink. Given "
            "the user's research goal and (optional) system description, "
            "pick the best (scale, engine) combination from the available "
            "candidates. Do not invent engines that are not listed.\n\n"
            "USER GOAL:\n"
            f"{user_goal}\n\n"
            "SYSTEM DESCRIPTION:\n"
            f"{sys_desc}\n\n"
            "AVAILABLE CANDIDATES (scale → engines you may pick from):\n"
            f"{candidates_block}\n\n"
            "SCALE GUIDANCE (when each is appropriate):\n"
            f"{scale_descs_block}\n\n"
            "DECISION CRITERIA:\n"
            "  1. Match the scale to the physics of the system (size, "
            "periodicity, bonding type, required accuracy).\n"
            "  2. Within the chosen scale, pick the engine that's "
            "conventional for this kind of calculation among the "
            "available options.\n"
            "  3. If multiple scales fit, prefer the one that gives the "
            "best accuracy/speed tradeoff for the stated goal.\n\n"
            "OUTPUT FORMAT: Return ONLY a JSON object, no prose:\n"
            "{\n"
            '  "scale":   "<one of the scale keys above>",\n'
            '  "engine":  "<one of the engines listed for that scale>",\n'
            '  "reasoning": "<one or two sentences explaining the choice>",\n'
            '  "alternatives": [\n'
            '    {"scale": "...", "engine": "...", "tradeoff": "<one line>"}\n'
            '  ]\n'
            "}"
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse the LLM response, tolerating code fences / preamble."""
        text = text.strip()
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

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse routing JSON: {text[:300]}")
