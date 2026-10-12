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


# The default slack: the model supplies the CENTER e, CODE owns the WIDTH.
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
# The factor is a CALLER-level knob (``health_slack_factor``, threaded from the
# pipeline) defaulting to this constant — the model still never sets the width.
_SLACK_K = 5.0


def clamp_center_to_band(
    observable: str, expected, slack_factor: float = _SLACK_K,
) -> Optional[Dict[str, Any]]:
    """Turn an expected CENTER value ``e`` into a K-slack raw ``health:`` spec.

    ``expected`` is parsed with :func:`health._as_number`, so the same coercion
    rules (numeric strings yes, booleans/None no) apply here as in the gate. The
    band is computed directly as ``[e / K, e * K]`` with ``K = slack_factor``;
    the direct form stays underflow-safe down to the smallest positive float
    (a ``sqrt(e*e)`` center would underflow to 0.0 for e <~ 1e-154 and collapse
    the band to ``[0.0, e]``). Returns ``{"observable", "min", "max"}`` or
    ``None`` when ``e`` is non-positive / non-finite / unparseable, or when
    ``slack_factor`` is not a finite number > 1 (drop it, fall back to
    frontmatter).
    """
    if not observable:
        return None
    e = _as_number(expected)
    if e is None or not math.isfinite(e) or e <= 0:
        return None
    k = _as_number(slack_factor)
    if k is None or not math.isfinite(k) or k <= 1:
        return None
    lo, hi = e / k, e * k
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
        input_files: Optional[Dict[str, str]] = None,
        slack_factor: float = _SLACK_K,
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
            input_files: Optional generated deck (``{filename: contents}``) so the
                model reasons about the EXACT run it is grounding — the goal AND
                the deck, as #783 asks. Truncated in the prompt.
            slack_factor: Multiplicative K owning the band WIDTH (min = e/K,
                max = e*K). Caller-level knob; the model never sets the width.

        Returns:
            A list of raw ``{"observable", "min", "max", "source"}`` spec dicts,
            each band widened by the K-slack clamp and each carrying
            ``source="derived"``; ``[]`` on any failure. ``parse_health_specs``
            validates these identically to frontmatter.

        A derived center is kept only when it is physically consistent with the
        frontmatter fallback band (if one exists for that observable): a center
        OUTSIDE the frontmatter ``[min, max]`` is almost always a units slip
        (e.g. water handed back as 997 kg/m³ instead of ~1 g/cm³), which would
        turn the gate into a tripwire that fires on every healthy run. Such a
        band is dropped so the frontmatter fallback stands.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        try:
            # The frontmatter health block tells us WHICH observables to ground,
            # their units, and the fallback band each derived center is sanity-
            # checked against (the same load path the gate uses).
            fm_bands = self._frontmatter_bands(skill, domain)
            skill_context = self._load_skill_section(skill, domain or "")
            prompt = self._build_prompt(
                research_goal, system_description, components, skill_context,
                fm_bands, input_files)
            report = self._generate_json(prompt)
            return self._parse(report, fm_bands, slack_factor)
        except Exception as exc:  # derivation is best-effort, never fatal
            self.logger.warning("Health-bounds derivation failed: %s", exc)
            return []

    @staticmethod
    def _frontmatter_bands(skill: Optional[str], domain: Optional[str]):
        """Parsed frontmatter ``health:`` bands for the active engine, by name.

        Uses the same ``_health_specs`` -> ``parse_health_specs`` path as the
        gate, so the deriver sees exactly the observables (and units, and
        fallback bands) the gate will judge against. Best-effort: ``{}`` on any
        failure keeps derivation fail-open.
        """
        from .refinement import _health_specs
        from .health import parse_health_specs
        try:
            bands = parse_health_specs(_health_specs(skill, domain))
        except Exception:
            return {}
        return {b.observable: b for b in bands}

    def _build_prompt(self, research_goal: str, system_description: str,
                      components: Optional[List[Dict[str, Any]]],
                      skill_context: str, fm_bands, input_files) -> str:
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
        if fm_bands:
            obs_lines = "\n".join(
                f"- {b.observable}"
                + (f" (units: {b.units})" if b.units else " (units: unspecified)")
                for b in fm_bands.values()
            )
            obs_block = (
                "=== Sanity observables to ground (give the expected value in "
                "the stated units) ===\n" + obs_lines + "\n\n")
        else:
            obs_block = ""
        if input_files:
            deck_block = "=== Generated input deck ===\n" + _format_deck(
                input_files) + "\n\n"
        else:
            deck_block = ""
        return (
            "You are setting GROSS PHYSICAL-SANITY expectations for a simulation "
            "before it runs. Independently of any target result, state the "
            "EXPECTED / TYPICAL value of each sanity observable for THIS system, "
            "reasoning from the well-established behaviour of its constituents.\n\n"
            f"{skill_context or '(no engine skill guidance loaded)'}\n\n"
            f"=== System ===\n{system_description or '(not provided)'}\n\n"
            f"{comp_block}"
            f"{obs_block}"
            f"{deck_block}"
            f"=== Research goal (context only) ===\n{research_goal}\n\n"
            "For each sanity observable whose plausible scale you can state for "
            "this system (e.g. 'density' for a condensed-phase box), give a "
            "SINGLE expected value `e` — one positive number IN THE STATED UNITS "
            "for that observable, NOT a range and NOT an uncertainty. You are "
            "giving the CENTER only; the code owns the tolerance. Omit any "
            "observable you cannot ground for this system.\n"
            "Return JSON: {\"bounds\": [{\"observable\": <str>, "
            "\"expected\": <number>}, ...]}."
        )

    def _parse(self, report: Dict[str, Any], fm_bands,
               slack_factor: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in (report.get("bounds") or []):
            if not isinstance(item, dict):
                continue
            observable = item.get("observable")
            expected = item.get("expected")
            spec = clamp_center_to_band(observable, expected, slack_factor)
            if spec is None:
                continue
            # Units-slip guard: a derived CENTER that falls outside the
            # frontmatter fallback band for the same observable is almost
            # certainly a wrong-units value (e.g. 997 kg/m^3 for a density the
            # gate reads in g/cm^3). Dropping it keeps the frontmatter band as
            # the fallback rather than deriving a band that trips every run.
            fm = fm_bands.get(spec["observable"]) if fm_bands else None
            if fm is not None and not _center_in_band(_as_number(expected), fm):
                self.logger.warning(
                    "dropping derived health band for %r: center %r is outside "
                    "the frontmatter fallback [%s, %s] (likely a units slip); "
                    "keeping the frontmatter band",
                    spec["observable"], expected, fm.minimum, fm.maximum)
                continue
            spec["source"] = "derived"
            out.append(spec)
        return out


def _center_in_band(e: Optional[float], band) -> bool:
    """Whether a derived center lies within a (possibly one-sided) band."""
    if e is None or not math.isfinite(e):
        return False
    if band.minimum is not None and e < band.minimum:
        return False
    if band.maximum is not None and e > band.maximum:
        return False
    return True


def _format_deck(input_files: Dict[str, str], limit: int = 4000) -> str:
    """Render a deck mapping as labelled, truncated sections for the prompt."""
    chunks = []
    for name, content in (input_files or {}).items():
        body = (content or "").rstrip()
        if len(body) > limit:
            body = body[:limit] + "\n[truncated]"
        chunks.append(f"--- {name} ---\n{body}")
    return "\n\n".join(chunks)
