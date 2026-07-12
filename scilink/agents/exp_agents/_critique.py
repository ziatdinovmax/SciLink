"""CritiquePayload — the one re-entry currency of the analysis QC stack.

Every "something reviewed a result and wants a stage re-run" interaction in
the analysis agents carries the same information: who produced the critique,
the critique text, optional structured hints/priors, and which stage or unit
it targets. Historically each producer had its own ad-hoc shape (verifier
``issues_found`` + ``recommended_action``, human feedback strings, the
consistency pass's peer-evidence prompt, the hyperspectral reflection
critique). This type names that currency once so producers are
interchangeable consumers of one re-entry mechanism:

- **verifier** — the in-loop LLM verifier (per-unit refit; exists today)
- **human** — a person's critique (per-unit poor-result feedback today;
  synthesis re-entry per issue #322)
- **consistency** — the series consistency pass's peer evidence (exists today)
- **literature** — a feature-conditioned literature result (issue #323)
- **orchestrator** — a programmatic caller (meta agent / run_task)

Wiring happens in the re-entry phase of the QC unification (issue #327,
``analysis_qc_unification_plan.md`` §2.3, §4); existing producers are wrapped
at their call boundaries without changing the strings that reach prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

#: Valid critique producers.
CRITIQUE_SOURCES = (
    "verifier",
    "human",
    "consistency",
    "literature",
    "orchestrator",
)

#: Target sentinel for the synthesis/interpretation stage (vs. a unit index).
SYNTHESIS = "synthesis"


@dataclass(frozen=True)
class CritiquePayload:
    """A critique/context payload for re-entering a QC stage.

    Attributes:
        source: Which producer generated this critique (see
            :data:`CRITIQUE_SOURCES`).
        critique: Free-text critique / guidance. This is what reaches the
            prompt of the re-entered stage.
        hints: Optional structured priors — e.g. an expected model, parameter
            bounds, a known transition, required outputs. Consumers that
            don't understand a hint ignore it.
        target: ``"synthesis"`` to re-enter the interpretation/synthesis
            stage over cached per-unit results, or an integer unit index to
            re-enter a single unit's fit/analysis.
        provenance: Optional audit trail — literature files consulted, vote
            counts, the peer-consensus evidence, etc. Never injected into
            prompts; stored with the revision record.
    """

    source: str
    critique: str
    hints: Optional[Dict[str, Any]] = None
    target: Union[str, int] = SYNTHESIS
    provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.source not in CRITIQUE_SOURCES:
            raise ValueError(
                f"CritiquePayload.source must be one of {CRITIQUE_SOURCES}; "
                f"got {self.source!r}"
            )
        if not isinstance(self.critique, str) or not self.critique.strip():
            raise ValueError("CritiquePayload.critique must be non-empty text")
        if not (self.target == SYNTHESIS or isinstance(self.target, int)):
            raise ValueError(
                f"CritiquePayload.target must be 'synthesis' or a unit index; "
                f"got {self.target!r}"
            )

    @property
    def targets_synthesis(self) -> bool:
        return self.target == SYNTHESIS
