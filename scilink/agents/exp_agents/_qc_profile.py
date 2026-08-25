"""QCProfile — a named bundle of the QC-loop toggles (operating regimes).

The analysis agents run in two regimes: **post-analysis** (thorough, LLM-heavy
QC — today's behavior and the default) and **real-time in-situ** (per-frame
analysis during a measurement, where the happy path must spend zero LLM
calls). Rather than letting the per-stage toggles accumulate as loose kwargs,
they are gathered here into one named object from the start.

The profile controls **LLM cost, not numerics cost** — it decides which QC
stages run, not how expensive the executed analysis script is. The real-time
loop shape itself (lock once → execute per frame → gate as drift detector →
escalate on breach) is layer-2 work; see ``analysis_qc_unification_plan.md``
§7. Until the engine consumes profiles (phase 4+), this type is the canonical
*naming* of the knobs; agents keep their existing keyword arguments and
:meth:`QCProfile.from_agent_kwargs` maps them onto a profile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class QCProfile:
    """A named bundle of QC-loop stage toggles.

    Attributes:
        name: Profile name ("thorough", "realtime", or custom).
        max_verification_iterations: Budget for the LLM verification loop.
            0 bypasses verification entirely.
        check_plan_conformance: Run the plan-conformance LLM check after each
            fresh script generation.
        best_of_n_eligible: Whether best-of-N candidate fan-out may run.
        human_feedback: Whether interactive human-feedback checkpoints fire.
        literature: Whether in-run literature search runs.
        escalation_enabled: Whether the constraint-annealing ladder may
            escalate (False pins the loop at its starting level).
        voted_verification: Run N independent verifier judgments with a
            majority-to-reject policy instead of a single judgment
            (hyperspectral's sanity-vote pattern; opt-in for other
            modalities).
    """

    name: str
    max_verification_iterations: int = 7
    check_plan_conformance: bool = True
    best_of_n_eligible: bool = True
    human_feedback: bool = True
    literature: bool = True
    escalation_enabled: bool = True
    voted_verification: bool = False

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("QCProfile.name must be non-empty")
        if self.max_verification_iterations < 0:
            raise ValueError("QCProfile.max_verification_iterations must be >= 0")

    def with_overrides(self, **overrides: Any) -> "QCProfile":
        """A copy with specific fields overridden (name kept unless given)."""
        return replace(self, **overrides)

    @classmethod
    def from_agent_kwargs(
        cls,
        base: Optional["QCProfile"] = None,
        *,
        max_verification_iterations: Optional[int] = None,
        enable_human_feedback: Optional[bool] = None,
        use_literature: Optional[bool] = None,
        n_candidates: Optional[int] = None,
    ) -> "QCProfile":
        """Map the agents' existing constructor/analyze kwargs onto a profile.

        This is the bridge that keeps the public agent surface unchanged
        while the engine consumes a single profile object internally.
        ``None`` leaves the base profile's value in place.
        """
        profile = base or THOROUGH
        updates: Dict[str, Any] = {}
        if max_verification_iterations is not None:
            updates["max_verification_iterations"] = int(max_verification_iterations)
        if enable_human_feedback is not None:
            updates["human_feedback"] = bool(enable_human_feedback)
        if use_literature is not None:
            updates["literature"] = bool(use_literature)
        if n_candidates is not None:
            updates["best_of_n_eligible"] = int(n_candidates) > 1
        return profile.with_overrides(**updates) if updates else profile


#: Today's defaults — the engine default; the behavior freeze holds.
THOROUGH = QCProfile(name="thorough")

#: The in-situ preset: at most one verification pass, no conformance check,
#: no fan-out, no interactive pauses, no in-run literature, no annealing.
#: The zero-LLM-call happy path comes from the lock-once/execute-per-frame
#: loop shape (layer 2), not from this profile alone — see the module doc.
REALTIME = QCProfile(
    name="realtime",
    max_verification_iterations=1,
    check_plan_conformance=False,
    best_of_n_eligible=False,
    human_feedback=False,
    literature=False,
    escalation_enabled=False,
    voted_verification=False,
)

_PRESETS = {p.name: p for p in (THOROUGH, REALTIME)}


def resolve_profile(value: Any) -> QCProfile:
    """Coerce a profile argument: QCProfile | preset name | None → QCProfile.

    ``None`` resolves to :data:`THOROUGH` (the behavior-preserving default).
    """
    if value is None:
        return THOROUGH
    if isinstance(value, QCProfile):
        return value
    if isinstance(value, str):
        try:
            return _PRESETS[value.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unknown QC profile {value!r}; known presets: "
                f"{sorted(_PRESETS)}"
            ) from None
    raise TypeError(
        f"Expected QCProfile, preset name, or None; got {type(value).__name__}"
    )
