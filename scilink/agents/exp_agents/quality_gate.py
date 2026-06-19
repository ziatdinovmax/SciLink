"""Configurable framework-level quality gate for analyze-mode agents.

Historically the only gate was R² ≥ 0.95 (a curve-fitting default that was
hardcoded across :mod:`curve_fitting_controllers`). For workflow-style
skills like ``structure_matching/xrd`` the natural quality metric is a
figure-of-merit or a normalized cost, not R² — so the gate is now a
data-class that the agent resolves from four sources, in priority order:

1. Explicit ``quality_gate=`` keyword on :meth:`CurveFittingAgent.analyze`.
2. Explicit ``quality_gate=`` on :meth:`CurveFittingAgent.__init__`.
3. Skill frontmatter ``quality_gate:`` block (parsed by
   :func:`scilink.skills.loader.load_skill`).
4. Framework default — :data:`R_SQUARED_DEFAULT` (R² ≥ 0.95; preserves
   pre-existing behavior for every curve-fit skill that does not
   declare its own gate).

A legacy ``r2_threshold=`` keyword still works as a shortcut that
constructs :class:`QualityGate` with ``metric='r_squared'``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


# How to interpret the threshold:
#   higher_is_better → accept if value >= accept_threshold; hard-reject if value < hard_reject_threshold
#   lower_is_better  → accept if value <= accept_threshold; hard-reject if value > hard_reject_threshold
_DIRECTIONS = {"higher_is_better", "lower_is_better"}


@dataclass(frozen=True)
class QualityGate:
    """Framework-level accept / reject criterion for a single fit result.

    The gate reads ``fit_result["fit_quality"][metric]`` from the script's
    ``FIT_RESULTS_JSON`` output and applies threshold + direction logic.
    A missing or non-numeric value is treated as hard-reject — same
    behavior as the original R² gate when ``fit_quality`` was missing.
    """

    metric: str = "r_squared"
    accept_threshold: float = 0.95
    hard_reject_threshold: float = 0.90
    direction: str = "higher_is_better"
    # Whether the LLM physics verifier runs on top of the numeric gate.
    # True for goodness-of-fit metrics (r_squared, peak_region_r2, BIC, …) — the
    # verifier inspects the plot/residuals/parameters for physics problems a
    # number can't catch. False for *workflow-scoring* gates (e.g. xrd's
    # figure_of_merit), where the skill's own scoring tool IS the verification
    # and the goodness-of-fit-shaped prompt does not apply.
    physical_review: bool = True
    # The metric's theoretical best-possible value: the CEILING for
    # higher_is_better (e.g. 1.0 for R²) or the FLOOR for lower_is_better
    # (e.g. 0.0 for χ²/RMSE). None when the metric is unbounded on its
    # "better" side (e.g. raw BIC). Used only to cap the escalation
    # fast-accept margin so a near-optimal accept_threshold can't demand an
    # unreachable value (reproduces the old R² `(1-accept)/2` cap).
    best_value: Optional[float] = None

    @property
    def label(self) -> str:
        """Human label for the metric in prompts/messages."""
        return "R²" if self.metric == "r_squared" else self.metric

    @property
    def accept_cmp(self) -> str:
        return "≥" if self.direction == "higher_is_better" else "≤"

    @property
    def reject_cmp(self) -> str:
        return "<" if self.direction == "higher_is_better" else ">"

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"QualityGate.direction must be one of {sorted(_DIRECTIONS)}; "
                f"got {self.direction!r}"
            )
        if self.direction == "higher_is_better":
            if self.hard_reject_threshold > self.accept_threshold:
                raise ValueError(
                    "higher_is_better: hard_reject_threshold "
                    f"({self.hard_reject_threshold}) must be ≤ accept_threshold "
                    f"({self.accept_threshold})"
                )
        else:
            if self.hard_reject_threshold < self.accept_threshold:
                raise ValueError(
                    "lower_is_better: hard_reject_threshold "
                    f"({self.hard_reject_threshold}) must be ≥ accept_threshold "
                    f"({self.accept_threshold})"
                )

    # -- evaluation -------------------------------------------------------

    def extract(self, fit_quality: Optional[dict]) -> Optional[float]:
        """Read this gate's metric value from a ``fit_quality`` dict.

        Returns ``None`` when missing, non-numeric, or NaN — callers
        treat ``None`` as a hard reject.
        """
        if not isinstance(fit_quality, dict):
            return None
        v = fit_quality.get(self.metric)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return f

    def is_accept(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        if self.direction == "higher_is_better":
            return value >= self.accept_threshold
        return value <= self.accept_threshold

    def is_hard_reject(self, value: Optional[float]) -> bool:
        if value is None:
            return True
        if self.direction == "higher_is_better":
            return value < self.hard_reject_threshold
        return value > self.hard_reject_threshold

    def is_soft_band(self, value: Optional[float]) -> bool:
        """Between hard-reject and accept — verifier gets the final call."""
        if value is None:
            return False
        return (not self.is_accept(value)) and (not self.is_hard_reject(value))

    # -- escalation fast-accept (metric-agnostic) -------------------------

    def fast_accept_margin(self, fraction: float) -> float:
        """Escalation fast-accept margin (in metric units, >= 0).

        A fraction of the soft band (``|accept - hard_reject|``), capped at
        half the room to ``best_value`` so the fast-accept point never passes
        the metric's optimum. For the default R² gate (band 0.05, best 1.0)
        and ``fraction=0.4`` this is 0.02 — identical to the old absolute
        ``ESCALATION_R2_MARGIN`` — and the cap reproduces the old
        ``(1 - accept)/2`` shrink as accept_threshold approaches 1.0.
        """
        band = abs(self.accept_threshold - self.hard_reject_threshold)
        margin = max(0.0, fraction) * band
        if self.best_value is not None:
            room = abs(self.best_value - self.accept_threshold)
            margin = min(margin, room / 2.0)
        return margin

    def clears_by_fast_margin(
        self, value: Optional[float], fraction: float
    ) -> bool:
        """True if ``value`` clears ``accept_threshold`` by the fast margin in
        the gate's direction — the escalation fast-accept quality test."""
        if value is None:
            return False
        margin = self.fast_accept_margin(fraction)
        if self.direction == "higher_is_better":
            return value >= self.accept_threshold + margin
        return value <= self.accept_threshold - margin

    # -- display helpers --------------------------------------------------

    def describe(self) -> str:
        """One-line human-readable summary for logs and prompts."""
        cmp = "≥" if self.direction == "higher_is_better" else "≤"
        return (
            f"{self.metric} {cmp} {self.accept_threshold:.3f} accepts; "
            f"hard-reject at {self.metric} "
            f"{'<' if self.direction == 'higher_is_better' else '>'} "
            f"{self.hard_reject_threshold:.3f}"
        )

    def short_label(self) -> str:
        """Just the metric name, used in log lines like '✅ Fit approved (...)'."""
        return self.metric.replace("_", " ").upper() if self.metric == "r_squared" else self.metric

    # -- builders ---------------------------------------------------------

    def with_accept_threshold(self, value: float) -> "QualityGate":
        """Return a copy with a different accept_threshold; preserves direction.

        Adjusts hard_reject_threshold proportionally to keep the soft band
        width relative to the original gate.
        """
        soft_width = abs(self.accept_threshold - self.hard_reject_threshold)
        if self.direction == "higher_is_better":
            return replace(self, accept_threshold=value,
                           hard_reject_threshold=value - soft_width)
        return replace(self, accept_threshold=value,
                       hard_reject_threshold=value + soft_width)


# The framework-level default — preserves the pre-existing R² ≥ 0.95
# behavior for every skill / call that doesn't override.
R_SQUARED_DEFAULT = QualityGate(
    metric="r_squared",
    accept_threshold=0.95,
    hard_reject_threshold=0.90,
    direction="higher_is_better",
    best_value=1.0,
)

# Metrics with a known optimum on their "better" side, used to default
# ``best_value`` when a skill's frontmatter gate omits it.
_KNOWN_BEST_VALUE = {"r_squared": 1.0, "peak_region_r2": 1.0}


def resolve_gate(
    *,
    call_override: Optional[Any] = None,
    agent_default: Optional[Any] = None,
    skill_meta: Optional[dict] = None,
    user_threshold: Optional[float] = None,
    legacy_threshold: Optional[float] = None,
    logger: Optional[Any] = None,
) -> QualityGate:
    """Resolve the effective gate from all sources.

    Priority (highest first):
      1. ``call_override`` — explicit ``analyze(quality_gate=...)`` arg (a full
         gate, possibly with a non-R² metric — deliberate, wins over everything).
      2. ``agent_default`` — ``CurveFittingAgent(quality_gate=...)`` arg.
      3. ``user_threshold`` — an experienced user's explicit numeric R² override
         (e.g. ``analyze(r2_threshold=...)`` from the UI/chat). Wins over a skill
         gate, but ONLY for the threshold value. **Metric guard:** if the active
         skill scores by a non-``r_squared`` metric, a bare R² number must not
         silently replace that metric — the override is ignored (skill gate kept)
         and a warning is logged. Changing the metric requires an explicit
         ``quality_gate`` (layer 1), not a threshold number.
      4. ``skill_meta['quality_gate']`` — frontmatter block from the active skill
         (or the first loaded skill in a multi-skill set).
      5. ``legacy_threshold`` — the agent constructor's ``r2_threshold`` default;
         builds a ``QualityGate(metric='r_squared', accept_threshold=value)``.
      6. :data:`R_SQUARED_DEFAULT`.

    Each layer can be either a :class:`QualityGate` instance or a plain
    dict (matching the dataclass fields). Dicts are coerced via
    :func:`from_mapping`. ``None`` at any layer falls through. ``logger`` (if
    provided) receives a warning when a user R² override conflicts with, or is
    blocked by, a skill gate.
    """
    for source in (call_override, agent_default):
        gate = _coerce(source)
        if gate is not None:
            return gate

    skill_gate = _coerce((skill_meta or {}).get("quality_gate")) if skill_meta else None

    # Experienced-user numeric R² override (layer 3): wins over the skill gate,
    # but only as a threshold — never silently swaps a skill's non-R² metric.
    if user_threshold is not None:
        if skill_gate is not None and skill_gate.metric != "r_squared":
            if logger is not None:
                logger.warning(
                    f"   R² override {float(user_threshold):.3f} ignored: the active "
                    f"skill scores by '{skill_gate.metric}', not R². Keeping the "
                    f"skill's quality gate. (To change the metric, pass an explicit "
                    f"quality_gate; to change the threshold, configure it on the skill.)"
                )
            return skill_gate
        if (logger is not None and skill_gate is not None
                and skill_gate.metric == "r_squared"
                and abs(skill_gate.accept_threshold - float(user_threshold)) > 1e-9):
            logger.warning(
                f"   Using user R² override {float(user_threshold):.3f} "
                f"(active skill recommends {skill_gate.accept_threshold:.3f})."
            )
        return R_SQUARED_DEFAULT.with_accept_threshold(float(user_threshold))

    if skill_gate is not None:
        return skill_gate
    if legacy_threshold is not None:
        return R_SQUARED_DEFAULT.with_accept_threshold(float(legacy_threshold))
    return R_SQUARED_DEFAULT


def from_mapping(data: dict) -> QualityGate:
    """Construct a QualityGate from a frontmatter-style mapping."""
    metric = str(data.get("metric", R_SQUARED_DEFAULT.metric))
    # Default best_value for known bounded GOF metrics when not declared, so
    # the fast-accept ceiling cap applies even to skill-authored R² gates.
    bv = data.get("best_value", _KNOWN_BEST_VALUE.get(metric))
    return QualityGate(
        metric=metric,
        accept_threshold=float(data.get("accept_threshold", R_SQUARED_DEFAULT.accept_threshold)),
        hard_reject_threshold=float(data.get(
            "hard_reject_threshold", R_SQUARED_DEFAULT.hard_reject_threshold
        )),
        direction=str(data.get("direction", R_SQUARED_DEFAULT.direction)),
        physical_review=bool(data.get("physical_review", R_SQUARED_DEFAULT.physical_review)),
        best_value=(float(bv) if bv is not None else None),
    )


def _coerce(value: Any) -> Optional[QualityGate]:
    if value is None:
        return None
    if isinstance(value, QualityGate):
        return value
    if isinstance(value, dict):
        return from_mapping(value)
    raise TypeError(
        f"Expected QualityGate, dict, or None; got {type(value).__name__}"
    )
