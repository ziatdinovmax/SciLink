"""The produce side of the observable-requirements contract.

``parameters = f(system, observables)`` needs the observable set *authored*
somewhere. This module derives it: from the research goal (plus the system and
the active skill's planning guidance), an LLM step emits the typed, engine-
neutral ``Requirement`` list the Generate stage consumes and the pre-run gate
checks. The observable -> signal/cadence knowledge is LLM-reasoned but guided by
skill content, and the deterministic coverage checkers backstop it — the same
prose -> skill -> deterministic reliability ladder used elsewhere.

Reuses the critic base for LLM-client construction (proxy vs LiteLLM, key
handling) and ``_load_skill_section`` / ``_generate_json``.
"""

from typing import Any, Dict, List, Optional

from .critics import _CriticBase
from .contradictions import CHECK_KINDS, Requirement


class ObservableRequirementsDeriver(_CriticBase):
    """Derives the typed observable-requirements contract from a research goal.

    Reasons about which observables a goal requires and, for each, the raw
    signal that must be logged and how densely — emitting engine-neutral
    ``Requirement`` objects. Guided by the active skill's ``planning`` section
    (where the observable -> signal/cadence recipes live); engine-specific
    realization of each requirement is the checkers' / skill tools' job, not
    this step's.
    """

    SKILL_SECTION = "planning"
    BASELINE_PROMPT_TEMPLATE = ""

    def derive(
        self,
        research_goal: str,
        *,
        system_description: str = "",
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Requirement]:
        """Return the observable-requirements contract for a research goal.

        Args:
            research_goal: What the run must compute.
            system_description: Optional free-text system description.
            skill: Skill bundle whose ``planning`` section guides the
                observable -> signal/cadence mapping.
            domain: Skill subdirectory (required when ``skill`` is given).

        Returns:
            A list of ``Requirement`` objects (possibly empty). Malformed or
            unknown-kind entries are dropped rather than raised — a derivation
            failure degrades to "no declared observables", never blocks a run.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        skill_context = self._load_skill_section(skill, domain or "")
        prompt = self._build_prompt(research_goal, system_description, skill_context)
        try:
            report = self._generate_json(prompt)
        except Exception as exc:  # derivation is best-effort, never fatal
            self.logger.warning("Observable derivation failed: %s", exc)
            return []
        return self._parse(report)

    def _build_prompt(self, research_goal: str, system_description: str,
                      skill_context: str) -> str:
        return (
            "You are configuring a simulation. Determine the OBSERVABLES that "
            "must be computed to satisfy the research goal, and for each, the "
            "raw signal the run must log and how densely it must be sampled so "
            "the observable is recoverable.\n\n"
            f"RESEARCH GOAL: {research_goal}\n"
            f"SYSTEM: {system_description or '(not provided)'}\n\n"
            f"{skill_context or '(no engine skill guidance loaded)'}\n\n"
            "Emit one entry per requirement an observable imposes:\n"
            "- check_kind 'signal_present': a raw signal that must be logged. "
            "params: {\"signal\": <name, e.g. 'stress', 'trajectory', 'energy', "
            "'temperature', 'density'>}.\n"
            "- check_kind 'cadence': ONLY when dense sampling is needed to "
            "extract the property (e.g. a transport/correlation quantity). "
            "params: {\"signal\": <name>, \"max_interval_steps\": <int>}.\n"
            "Include only observables the goal actually requires — do not invent "
            "them — and omit cadence when default logging suffices.\n"
            "Return JSON: {\"observables\": [{\"observable\": <str>, "
            "\"check_kind\": <str>, \"params\": {...}}, ...]}."
        )

    def _parse(self, report: Dict[str, Any]) -> List[Requirement]:
        out: List[Requirement] = []
        for item in (report.get("observables") or []):
            if not isinstance(item, dict):
                continue
            observable = item.get("observable")
            check_kind = item.get("check_kind")
            if not observable or check_kind not in CHECK_KINDS:
                continue
            params = item.get("params")
            out.append(Requirement(
                observable=str(observable),
                check_kind=check_kind,
                params=params if isinstance(params, dict) else {},
            ))
        return out
