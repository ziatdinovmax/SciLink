"""QCProfile — a named bundle of the QC-loop toggles (operating regimes).

The analysis agents run in two regimes: **post-analysis** (thorough, LLM-heavy
QC — today's behavior and the default) and **real-time in-situ** (per-frame
analysis during a measurement, where the happy path must spend zero LLM
calls). Rather than letting the per-stage toggles accumulate as loose kwargs,
they are gathered here into one named object from the start.

Between those two sit the **fit-for-purpose** presets. How good a result has
to be is decided by whoever consumes it, not by the agent: a Bayesian-
optimization objective needs one trustworthy number, a quick look needs a
readable answer, and neither needs seven verification rounds, a refit of every
flagged unit, a trend script and a literature search. ``quick`` and
``extract`` start cold (no anchor, unlike ``realtime``) and cut LLM judgement
— never the deterministic gates, which cost nothing and are what keeps a
cheap number from being a biased one.

A profile covers the stages AROUND the QC loop as well as the loop itself,
because that is where much of a run's wall-clock goes (measured live on an
easy single spectrum: synthesis alone was a third of the run).

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


SYNTHESIS_LEVELS = ("full", "light", "none")
VERIFICATION_MODES = ("strict", "purpose")

#: Appended to the LLM verifier's prompt under ``verification="purpose"``.
#: One principle; the purpose text is filled from the run's own objective.
PURPOSE_VERIFICATION_TMPL = (
    "\n\n## Fit for purpose\nThis is a reduced-depth analysis whose result is "
    "needed for: {purpose}. Reject it only for a defect that would materially "
    "change those requested quantities; record any lesser imperfection under "
    "issues and accept. An imperfection the purpose does not depend on is not "
    "grounds for another refinement round.")


def verification_addendum(state: Optional[dict]) -> Optional[str]:
    """The fit-for-purpose clause when the run verifies under ``purpose``.

    The purpose is the caller's objective when there is one, else the
    quantities the locked plan set out to extract — so the verifier is always
    told WHAT must be right, never just "be lenient".
    """
    state = state or {}
    if state.get("_verification_mode") != "purpose":
        return None
    purpose = (state.get("analysis_objective") or "").strip()
    if not purpose:
        cfg = (state.get("locked_fitting_config")
               or state.get("locked_analysis_config") or {})
        wanted = cfg.get("parameters_to_extract") or cfg.get("features_to_extract")
        if isinstance(wanted, (list, tuple)) and wanted:
            purpose = "the extracted " + ", ".join(str(w) for w in wanted)
    return PURPOSE_VERIFICATION_TMPL.format(
        purpose=purpose or "the quantities the analysis plan set out to extract")

#: Appended to a synthesis prompt under ``synthesis="light"`` where synthesis
#: is a single call (curve, image). Measured live: an LLM call's latency
#: follows the length of what it WRITES, and synthesis was a third of an easy
#: curve run — so "light" has to mean a shorter answer, not just the same
#: call under another name. One principle, no list of things to omit.
LIGHT_SYNTHESIS_ADDENDUM = (
    "\n\n## Brevity\nThis is a quick-look analysis: the reader wants the result, "
    "not the full argument. Keep `detailed_analysis` to one short paragraph and "
    "report at most three scientific claims, the ones the numbers support most "
    "directly. Keep every required field of the output schema.")


def stamp_profile(state: Optional[dict], series_results: Any) -> None:
    """Mark every item produced under a reduced-depth profile, at the point
    the results are published — so EVERY file written afterwards
    (``series_fit_results.json`` as well as ``analysis_results.json``) carries
    it, and a later sweep can pick those items for a thorough pass. Observed
    live: stamping at result-compilation time came after the store stage had
    already written the per-item file. Thorough runs are left untouched.
    """
    name = (state or {}).get("_qc_profile")
    if name in (None, "thorough"):
        return
    for r in series_results or []:
        if isinstance(r, dict):
            r.setdefault("quality_history", {}).setdefault(
                "produced_under_profile", name)


def synthesis_addendum(state: Optional[dict]) -> Optional[str]:
    """The brevity addendum when the run's synthesis level is ``light``."""
    if (state or {}).get("_synthesis_level") == "light":
        return LIGHT_SYNTHESIS_ADDENDUM
    return None


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
        plan_validation: Run the LLM check of the plan against the data
            before executing it. A loaded technique skill's mandatory rules
            are still validated when this is off — only the redundant
            skill-free sanity pass is skipped.
        adaptive_refit: Re-analyse flagged series units with a full
            anchor-grade loop each.
        trend: Generate and run the series trend script.
        synthesis: ``"full"`` | ``"light"`` | ``"none"``. ``light`` drops the
            critic / editor pair where a modality has one (hyperspectral) and
            equals ``full`` where synthesis is a single call (curve, image);
            ``none`` skips the narrative entirely — the result is its
            numbers.
        tier2: Whether the image agent may run its second, deep pass.
        verification: ``"strict"`` | ``"purpose"``. How the LLM verifier
            decides to REJECT. ``strict`` is today's behavior: any defect it
            can name is grounds for another refinement round. ``purpose``
            tells it what the result is for and asks it to reject only for a
            defect that would materially change the requested quantities —
            lesser imperfections are recorded as issues, not as rejections.
            Measured live, this — not the iteration cap — is what "picky"
            costs: a particle count needed two refinement rounds under either
            cap, each a 40–150 s multimodal call. The deterministic gates are
            untouched by this field.
        time_budget_s: Soft wall-clock budget for the run (None = unbounded).
    """

    name: str
    max_verification_iterations: int = 7
    check_plan_conformance: bool = True
    best_of_n_eligible: bool = True
    human_feedback: bool = True
    literature: bool = True
    escalation_enabled: bool = True
    voted_verification: bool = False
    plan_validation: bool = True
    adaptive_refit: bool = True
    trend: bool = True
    synthesis: str = "full"
    tier2: bool = True
    verification: str = "strict"
    time_budget_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("QCProfile.name must be non-empty")
        if self.max_verification_iterations < 0:
            raise ValueError("QCProfile.max_verification_iterations must be >= 0")
        if self.synthesis not in SYNTHESIS_LEVELS:
            raise ValueError(
                f"QCProfile.synthesis must be one of {SYNTHESIS_LEVELS}; "
                f"got {self.synthesis!r}")
        if self.verification not in VERIFICATION_MODES:
            raise ValueError(
                f"QCProfile.verification must be one of {VERIFICATION_MODES}; "
                f"got {self.verification!r}")
        if self.time_budget_s is not None and self.time_budget_s <= 0:
            raise ValueError("QCProfile.time_budget_s must be > 0 (or None)")

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
    plan_validation=False,
    adaptive_refit=False,
    trend=False,
    synthesis="none",
    tier2=False,
)

#: A quick look for a person: a readable answer, fast. Two verification
#: passes instead of seven, none of the redundant LLM checks, no literature,
#: no refit of flagged units, no trend script, no image tier 2. Human gates
#: are left to the autonomy mode — depth does not decide who gets asked.
QUICK = QCProfile(
    name="quick",
    max_verification_iterations=2,
    check_plan_conformance=False,
    best_of_n_eligible=False,
    literature=False,
    escalation_enabled=False,
    plan_validation=False,
    adaptive_refit=False,
    trend=False,
    synthesis="light",
    tier2=False,
    verification="purpose",
)

#: Numbers for a machine: ``quick`` with no narrative at all. The consumer is
#: an optimizer or a feature table, which reads ``extracted_features`` /
#: fit parameters and never the prose.
EXTRACT = QUICK.with_overrides(name="extract", synthesis="none")

_PRESETS = {p.name: p for p in (THOROUGH, QUICK, EXTRACT, REALTIME)}


def resolve_profile(value: Any) -> QCProfile:
    """Coerce a profile argument → QCProfile.

    Accepts a QCProfile, a preset name, ``None`` (→ :data:`THOROUGH`, the
    behavior-preserving default), or a dict ``{"base": <preset>, **fields}``
    — a preset with specific fields overridden, which is how a caller asks
    for e.g. quick-but-keep-the-trend without a new named preset.
    """
    if value is None:
        return THOROUGH
    if isinstance(value, QCProfile):
        return value
    if isinstance(value, dict):
        fields = dict(value)
        base = resolve_profile(fields.pop("base", fields.pop("name", None)))
        known = set(QCProfile.__dataclass_fields__) - {"name"}
        unknown = sorted(set(fields) - known)
        if unknown:
            raise ValueError(
                f"unknown QC profile field(s) {unknown}; known: {sorted(known)}")
        return base.with_overrides(**fields) if fields else base
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
