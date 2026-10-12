"""System-aware derivation of physical-sanity ``health:`` bands (issue #783).

The #782 health gate reads its plausible bands from STATIC skill frontmatter
(e.g. density ``0.02–30`` for any LAMMPS run). Those universal bands catch only
the grossest pathologies: a box that half-vaporized to 0.1 g/cm³ sails straight
through ``0.02–30`` even though it is plainly broken for liquid water. This
module makes the bands SYSTEM-AWARE — derived from the actual system — while the
frontmatter stays as the fallback (resolved in ``refinement._resolve_health``).

The design constraint is AVOIDING ACCURACY-CREEP. The model is asked only for
the EXPECTED single value ``e`` of each sanity observable for this system; the
WIDTH of the band is owned by CODE (the multiplicative clamp below), never by
the model. That is structural: the model is literally not given a way to hand
back a narrow band, so it cannot quietly turn a gross-sanity tripwire into an
accuracy check. The reasoning/section reuse mirrors
:class:`~scilink.agents.sim_agents.critics.ReferencePropertyCritic` (same
``validation`` skill section, where the expected/reference-value knowledge
lives), but this deriver never calls it: that critic judges MEASURED values
through a live-measurement seam that is not yet implemented, and it returns
verdicts, not bounds.

Fail-open is MANDATORY: any failure (no skill, LLM error, unparseable output)
yields ``[]`` so the gate silently falls back to the frontmatter bands and a run
is never blocked by a derivation hiccup.
"""

import math
from typing import Any, Dict, List, Optional

from .critics import _CriticBase
from .health import _as_number


# The fixed slack: the model supplies the CENTER e, CODE owns the WIDTH.
#
#     min = e / K      max = e * K
#
# K=5 gives water (e≈1.0) a band of 0.2–5.0: it catches a half-vaporized box at
# 0.1 g/cm³ (which the universal frontmatter 0.02–30 misses) yet still lets a
# merely-inaccurate 0.85 pass — a gross-sanity tripwire, not an accuracy check.
# It is multiplicative so one constant works across density, pressure, lattice
# constant, modulus, etc. (each with its own scale). This is deliberately NOT a
# "superset of the frontmatter" rule: that would forbid the useful tightening of
# 0.02–30 down to 0.2–5.0 that is the whole point of making bands system-aware.
_SLACK_K = 5.0


def _clamp_to_slack_band(e_min: float, e_max: float) -> Optional[Dict[str, float]]:
    """Widen a (min, max) pair so the band is never tighter than K-slack.

    The normal path passes a single expected value as ``e_min == e_max == e`` and
    gets ``e/K .. e*K``. If a caller ever supplies an explicit band, it is only
    ever WIDENED (never narrowed) around its geometric center ``g = sqrt(min*max)``:
    ``final_min = min(e_min, g/K)``, ``final_max = max(e_max, g*K)``. Code can
    only widen; a model-narrowed band can never tighten the gate.

    Returns the raw ``{"min", "max"}`` spec, or ``None`` when the center/bounds
    are non-positive or non-finite (the observable is then dropped and the gate
    falls back to the frontmatter band).
    """
    if e_min is None or e_max is None:
        return None
    if not (math.isfinite(e_min) and math.isfinite(e_max)):
        return None
    if e_min <= 0 or e_max <= 0:
        return None
    lo, hi = (e_min, e_max) if e_min <= e_max else (e_max, e_min)
    g = math.sqrt(lo * hi)
    final_min = min(lo, g / _SLACK_K)
    final_max = max(hi, g * _SLACK_K)
    if not (math.isfinite(final_min) and math.isfinite(final_max)):
        return None
    return {"min": final_min, "max": final_max}


def clamp_center_to_band(observable: str, expected) -> Optional[Dict[str, Any]]:
    """Turn an expected CENTER value ``e`` into a widened raw ``health:`` spec.

    ``expected`` is parsed with :func:`health._as_number`, so the same coercion
    rules (numeric strings yes, booleans/None no) apply here as in the gate.
    Returns ``{"observable", "min", "max"}`` or ``None`` when ``e`` is
    non-positive / non-finite / unparseable (drop it, fall back to frontmatter).
    """
    if not observable:
        return None
    e = _as_number(expected)
    if e is None or not math.isfinite(e) or e <= 0:
        return None
    # Compute the K-slack band directly (not via the geometric-mean helper):
    # for a single center the band is simply [e/K, e*K], which stays correct
    # and underflow-safe down to the smallest positive float, whereas
    # ``sqrt(e*e)`` underflows to 0.0 for e <~ 1e-154 and would collapse the
    # band to [0.0, e]. The geometric-mean path is only needed when a caller
    # supplies a genuine (min, max) band.
    lo, hi = e / _SLACK_K, e * _SLACK_K
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return {"observable": str(observable), "min": lo, "max": hi}


class HealthBoundsDeriver(_CriticBase):
    """Derives system-aware physical-sanity ``health:`` bands from the system.

    For each gross-sanity observable the skill declares an interest in, the model
    is asked for the EXPECTED/TYPICAL single value for THIS system (a number, not
    a band); CODE widens it to a K-slack band via :func:`clamp_center_to_band`.
    Guided by the active skill's ``validation`` section — the same section
    :class:`~scilink.agents.sim_agents.critics.ReferencePropertyCritic` reads for
    known reference behaviour.

    Best-effort: ``derive`` returns raw spec dicts already widened by the clamp,
    or ``[]`` on ANY failure (fail-open), so the gate falls back to frontmatter.
    """

    SKILL_SECTION = "validation"
    BASELINE_PROMPT_TEMPLATE = ""

    def derive(
        self,
        research_goal: str,
        *,
        system_description: str = "",
        components: Optional[List[Dict[str, Any]]] = None,
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return system-aware ``health:`` band spec dicts for this system.

        Args:
            research_goal: What the run computes (context for the judge).
            system_description: Optional free-text system description.
            components: Optional packed-box components (name/smiles/count) that
                anchor the model's reasoning about the expected values.
            skill: Skill bundle whose ``validation`` section supplies known
                reference behaviour (same section as ``ReferencePropertyCritic``).
            domain: Skill subdirectory (required when ``skill`` is given).

        Returns:
            A list of raw ``{"observable", "min", "max"}`` spec dicts, each band
            already widened by the K-slack clamp, or ``[]`` on any failure.
            ``parse_health_specs`` validates these identically to frontmatter.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        try:
            skill_context = self._load_skill_section(skill, domain or "")
            prompt = self._build_prompt(
                research_goal, system_description, components, skill_context)
            report = self._generate_json(prompt)
            return self._parse(report)
        except Exception as exc:  # derivation is best-effort, never fatal
            self.logger.warning("Health-bounds derivation failed: %s", exc)
            return []

    def _build_prompt(self, research_goal: str, system_description: str,
                      components: Optional[List[Dict[str, Any]]],
                      skill_context: str) -> str:
        if components:
            comp_lines = "\n".join(
                f"- {c.get('name', '?')}"
                + (f" [{c.get('smiles')}]" if c.get("smiles") else "")
                + (f" x{c.get('count')}" if c.get("count") is not None else "")
                for c in components if isinstance(c, dict)
            )
            comp_block = f"=== Components ===\n{comp_lines}\n\n"
        else:
            comp_block = ""
        return (
            "You are setting GROSS PHYSICAL-SANITY expectations for a simulation "
            "before it runs. Independently of any target result, state the "
            "EXPECTED / TYPICAL value of each sanity observable for THIS system, "
            "reasoning from the well-established behaviour of its constituents.\n\n"
            f"{skill_context or '(no engine skill guidance loaded)'}\n\n"
            f"=== System ===\n{system_description or '(not provided)'}\n\n"
            f"{comp_block}"
            f"=== Research goal (context only) ===\n{research_goal}\n\n"
            "For each sanity observable whose plausible scale you can state for "
            "this system (e.g. 'density' for a condensed-phase box), give a "
            "SINGLE expected value `e` — one positive number in the observable's "
            "usual units, NOT a range and NOT an uncertainty. You are giving the "
            "CENTER only; the code owns the tolerance. Omit any observable you "
            "cannot ground for this system.\n"
            "Return JSON: {\"bounds\": [{\"observable\": <str>, "
            "\"expected\": <number>}, ...]}."
        )

    def _parse(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in (report.get("bounds") or []):
            if not isinstance(item, dict):
                continue
            spec = clamp_center_to_band(item.get("observable"),
                                        item.get("expected"))
            if spec is not None:
                out.append(spec)
        return out
