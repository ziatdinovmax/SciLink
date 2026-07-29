"""Engine-neutral contradiction framework: verify declared requirements against
produced artifacts at a workflow boundary.

A *requirement* is a typed, engine-neutral statement of something the workflow
must be able to produce: a named observable, a ``check_kind`` drawn from a small
fixed vocabulary, and params. A *checker* registered for each ``check_kind``
verifies the requirement against the current artifacts and returns a
``Contradiction`` when it cannot be met. Engine-specific realizations (reading a
LAMMPS data file, splitting atom types) are delegated to conventional
skill-bundle tools resolved by name — never hardcoded here — so a new engine is
one skill bundle, not a change to this module.

The ``check_kind`` vocabulary (see
``docs/proposals/observable-requirements-contract.md``):

- ``signal_present`` — the raw signal the observable needs is captured
- ``cadence`` — it is sampled finely / long enough
- ``selection_realizable`` — required atom selections are distinctly expressible
- ``empirical_adequacy`` — (post-run) the property actually converged
- ``cross_run_consistency`` — a series uses consistent definitions

Only kinds with real tenants are implemented; the rest are declared slots so the
vocabulary is visible without being built speculatively. The dispatch is
fail-open: a checker that errors never blocks a run.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Requirement:
    """An engine-neutral statement of something the workflow must produce.

    Attributes:
        observable: Human/plan-facing name, e.g. ``"Zn solvation RDF"``.
        check_kind: One of the fixed vocabulary above.
        params: Kind-specific payload (e.g. the atom selections that must be
            distinctly expressible for ``selection_realizable``).
    """

    observable: str
    check_kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Contradiction:
    """A requirement a stage cannot satisfy, surfaced at its boundary."""

    requirement: Requirement
    check_kind: str
    message: str
    resolvable: bool = False
    # {"tool": <conventional skill-tool name>, "kwargs": {...}} — how to fix it,
    # if a resolution exists. The tool is resolved against the active skills.
    resolution: Optional[Dict[str, Any]] = None


# check_kind -> checker(requirement, artifacts, *, active_skills) -> Contradiction | None
_CHECKERS: Dict[str, Callable[..., Optional[Contradiction]]] = {}

# Full vocabulary, so unimplemented kinds are visible as declared slots rather
# than silent gaps.
CHECK_KINDS = (
    "signal_present",
    "cadence",
    "selection_realizable",
    "empirical_adequacy",
    "cross_run_consistency",
)


def register_checker(check_kind: str):
    """Register a checker for a ``check_kind``. Kind must be in ``CHECK_KINDS``."""
    if check_kind not in CHECK_KINDS:
        raise ValueError(
            f"unknown check_kind {check_kind!r}; extend CHECK_KINDS deliberately "
            f"(known: {CHECK_KINDS})"
        )

    def deco(fn):
        _CHECKERS[check_kind] = fn
        return fn

    return deco


def implemented_kinds() -> List[str]:
    """The ``check_kind`` values that currently have a registered checker."""
    return sorted(_CHECKERS)


def check_requirements(
    requirements: List[Requirement],
    artifacts: Dict[str, Any],
    active_skills: Optional[List[str]] = None,
) -> List[Contradiction]:
    """Dispatch each requirement to its ``check_kind`` checker; collect contradictions.

    Args:
        requirements: The declared contract for this boundary.
        artifacts: The produced artifacts to check against (e.g. ``data_file``,
            ``components_json``, ``deck``) — checker-dependent.
        active_skills: Skills whose bundles provide engine-specific realizations.

    Returns:
        One ``Contradiction`` per unmet requirement (empty if all satisfiable).
        A requirement whose ``check_kind`` has no registered checker is skipped
        (declared-but-unimplemented slot); a checker that raises is skipped
        (fail-open) — neither is reported as a contradiction.
    """
    out: List[Contradiction] = []
    for req in requirements:
        checker = _CHECKERS.get(req.check_kind)
        if checker is None:
            logger.debug(
                "no checker for kind %r (declared slot); skipping %r",
                req.check_kind, req.observable,
            )
            continue
        try:
            result = checker(req, artifacts, active_skills=active_skills)
        except Exception as e:  # fail-open: a checker error never blocks a run
            logger.warning(
                "contradiction checker %r failed on %r: %s",
                req.check_kind, req.observable, e,
            )
            result = None
        if result is not None:
            out.append(result)
    return out


@register_checker("selection_realizable")
def _check_selection_realizable(
    req: Requirement, artifacts: Dict[str, Any], *, active_skills=None,
) -> Optional[Contradiction]:
    """Required atom selections must each be *distinctly* expressible in the input.

    Engine-neutral: the selection→identifier mapping is delegated to the engine's
    conventional ``map_selections_to_types`` tool (a method on the ``engine_tools``
    module the caller supplies in ``artifacts``, resolved exactly like the MD
    agent's ``_TOOL_REGISTRY``). Two (or more) selections that resolve to the
    *same* identifier set cannot be told apart downstream (e.g. a species-resolved
    RDF where two species share one atom type) — that is the contradiction,
    resolvable by the engine's ``split_shared_types`` tool where one exists.
    """
    selections = list(req.params.get("selections") or [])
    if len(selections) < 2:
        return None
    engine_tools = artifacts.get("engine_tools")
    if engine_tools is None or not hasattr(engine_tools, "map_selections_to_types"):
        return None  # engine has no realization of this check kind

    mapping = engine_tools.map_selections_to_types(
        data_file=artifacts.get("data_file"),
        components_json=artifacts.get("components_json"),
        selections=selections,
    )

    by_ids: Dict[tuple, List[str]] = defaultdict(list)
    for sel in selections:
        ids = tuple(sorted(mapping.get(sel) or ()))
        if ids:
            by_ids[ids].append(sel)
    collisions = [sels for sels in by_ids.values() if len(sels) > 1]
    if not collisions:
        return None

    merged = "; ".join(" & ".join(sels) for sels in collisions)
    return Contradiction(
        requirement=req,
        check_kind="selection_realizable",
        message=(
            f"required selections collapse to a single identifier and cannot be "
            f"distinguished as authored: {merged}"
        ),
        resolvable=True,
        resolution={
            "tool": "split_shared_types",
            "kwargs": {
                "data_file": artifacts.get("data_file"),
                "components_json": artifacts.get("components_json"),
                "collisions": collisions,
            },
        },
    )
