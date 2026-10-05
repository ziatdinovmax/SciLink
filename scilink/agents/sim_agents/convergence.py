"""Numeric parameter-convergence comparator (engine-neutral).

A static calculation's accuracy depends on numerical parameters (VASP ENCUT and
k-points, QE ecutwfc/ecutrho, a QC basis-set ladder). The convention is to run
the calculation at increasing settings and take the cheapest setting past which
the observable stops changing within a tolerance. This module holds the pure
decision — "given an observable measured along a ladder of settings, where does
it plateau?" — with no engine, LLM, or I/O. The engine skills declare the ladder,
the observable, and the tolerance; the sweep driver runs the ladder and feeds the
results here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class ConvergenceResult:
    """Outcome of assessing one parameter ladder.

    Attributes:
        converged: True when a demonstrated plateau exists — the observable at
            some setting and every higher setting agree with the most-accurate
            (highest) setting within ``tolerance``, and at least two settings
            take part (a single point cannot demonstrate a plateau).
        setting: The cheapest setting that is converged (the one to adopt), or
            None when not converged.
        value: The observable at the most-accurate (highest) setting — the best
            available estimate — or None when there is nothing to report.
        deltas: ``(setting, |value - best_value|)`` for each setting, in input
            order; the evidence behind the verdict.
        reason: Short human-readable explanation.
    """

    converged: bool
    setting: Any
    value: Optional[float]
    deltas: List[Tuple[Any, float]]
    reason: str


def converged_setting(
    observations: Sequence[Tuple[Any, Optional[float]]],
    tolerance: float,
) -> ConvergenceResult:
    """Find where an observable plateaus along a ladder of settings.

    Args:
        observations: ``(setting, value)`` pairs ordered from least to most
            accurate/expensive (e.g. ascending ENCUT). ``value`` is the
            convergence observable (energy/atom, lattice constant, gap, …).
            A ``None`` value marks a setting whose run failed or produced no
            reading; such settings cannot anchor a plateau.
        tolerance: Maximum absolute difference (in the observable's units) that
            counts as "unchanged". Must be non-negative.

    Returns:
        A :class:`ConvergenceResult`. The adopted setting is the *cheapest* one
        from which the observable no longer moves beyond ``tolerance`` relative
        to the most-accurate setting, so downstream work runs at the cheapest
        trustworthy setting while the reported value is the most accurate.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    finite = [(s, v) for s, v in observations if v is not None]
    if not finite:
        return ConvergenceResult(False, None, None, [], "no readable observations")
    if len(finite) == 1:
        # One point cannot demonstrate a plateau, but it is the best estimate.
        s, v = finite[0]
        return ConvergenceResult(
            False, None, v, [(s, 0.0)],
            "only one setting produced a value — cannot demonstrate a plateau")

    best_value = finite[-1][1]  # highest (most accurate) setting
    deltas = [(s, abs(v - best_value)) for s, v in finite]

    # The cheapest setting from which every higher setting (inclusive) is within
    # tolerance of the best value. Walk from cheapest up; the first setting whose
    # own delta AND all deltas above it are within tolerance is the plateau start.
    converged_idx: Optional[int] = None
    for i in range(len(finite)):
        if all(d <= tolerance for _, d in deltas[i:]):
            converged_idx = i
            break

    if converged_idx is None or converged_idx >= len(finite) - 1:
        # Either nothing is within tolerance of the top, or only the top setting
        # agrees with itself — the ladder has not shown a plateau. Report the
        # best value so the caller can widen the ladder, but do not claim
        # convergence.
        return ConvergenceResult(
            False, None, best_value, deltas,
            "observable still changing at the top of the ladder — extend it")

    setting = finite[converged_idx][0]
    return ConvergenceResult(
        True, setting, best_value, deltas,
        f"converged at {setting}: observable within {tolerance} of the "
        f"most-accurate setting from here up")


@dataclass
class SweepResult:
    """Outcome of running one parameter ladder to convergence.

    Attributes:
        convergence: The comparator verdict over the ladder.
        observations: ``(setting, value)`` for each ladder rung, in order
            (``value`` is None where the run failed or produced no reading).
        run_dirs: ``setting -> run directory`` for every rung that ran.
        param_name: The parameter that was swept (for reporting).
    """

    convergence: ConvergenceResult
    observations: List[Tuple[Any, Optional[float]]]
    run_dirs: Dict[Any, str]
    param_name: str


def run_convergence_sweep(
    *,
    ladder: Sequence[Any],
    build_member: Callable[[Any], Dict[str, str]],
    run_ladder: Callable[[Dict[Any, Dict[str, str]]], Dict[Any, str]],
    read_observable: Callable[[Optional[str]], Optional[float]],
    tolerance: float,
    param_name: str = "parameter",
) -> SweepResult:
    """Run a parameter ladder as a batch fan-out and assess convergence.

    Engine-neutral. The three callbacks carry all engine-specific knowledge, so
    this driver has no dependency on any engine, agent, or executor and is
    exercised with fakes in tests:

    - ``build_member(setting)`` returns the input-file map for that rung (the
      base deck with the swept parameter set to ``setting`` — the engine skill's
      param-setter).
    - ``run_ladder({setting: inputs})`` executes every rung (in production, one
      fan-out campaign through the existing executor path) and returns
      ``{setting: run_dir}``. Rungs it omits are treated as failed.
    - ``read_observable(run_dir)`` extracts the convergence observable from a
      finished rung, or returns None if it cannot (the engine's output parser).

    Args:
        ladder: Settings from least to most accurate/expensive.
        tolerance: Passed to :func:`converged_setting`.
        param_name: Name of the swept parameter, for the result.

    Returns:
        A :class:`SweepResult` pairing the comparator verdict with the raw
        observations and run directories.
    """
    members = {setting: build_member(setting) for setting in ladder}
    run_dirs = run_ladder(members)
    observations: List[Tuple[Any, Optional[float]]] = []
    for setting in ladder:
        run_dir = run_dirs.get(setting)
        value = read_observable(run_dir) if run_dir is not None else None
        observations.append((setting, value))
    convergence = converged_setting(observations, tolerance)
    return SweepResult(
        convergence=convergence,
        observations=observations,
        run_dirs=run_dirs,
        param_name=param_name,
    )


@dataclass
class ParameterConvergence:
    """The result of converging a set of parameters for one calculation.

    Attributes:
        final_inputs: The deck with every converged parameter adopted (a
            parameter that did not converge is left at its base value).
        sweeps: One :class:`SweepResult` per parameter, in the order swept.
        all_converged: True only if every parameter showed a plateau.
    """

    final_inputs: Dict[str, str]
    sweeps: List[SweepResult]
    all_converged: bool


def converge_parameters(
    *,
    base_inputs: Dict[str, str],
    specs: Sequence[dict],
    set_param: Callable[[Dict[str, str], str, Any], Dict[str, str]],
    read_observable: Callable[[Optional[str], str], Optional[float]],
    run_ladder: Callable[[str, Dict[Any, Dict[str, str]]], Dict[Any, str]],
    tolerance_default: float = 0.0,
) -> ParameterConvergence:
    """Converge several numerical parameters in declared order (adopt-as-you-go).

    Each spec (from the skill's ``convergence:`` frontmatter) is swept with
    :func:`run_convergence_sweep`; when a parameter converges its value is
    written into the working deck before the next parameter is swept, so later
    ladders sit on the earlier converged settings — the standard protocol.

    Engine-neutral. ``set_param`` / ``read_observable`` are the engine skill's
    registry hooks; ``run_ladder(param, {setting: inputs})`` executes one ladder
    (namespaced by ``param`` so run dirs don't collide) and returns
    ``{setting: run_dir}``.

    Args:
        base_inputs: The generated base deck.
        specs: Convergence specs, each ``{parameter, ladder, observable,
            tolerance?}``.
        set_param, read_observable, run_ladder: Injected engine/execution hooks.
        tolerance_default: Tolerance for a spec that omits one.

    Returns:
        A :class:`ParameterConvergence`.
    """
    working = dict(base_inputs)
    sweeps: List[SweepResult] = []
    for spec in specs:
        param = spec["parameter"]
        ladder = spec["ladder"]
        observable = spec["observable"]
        tolerance = spec.get("tolerance", tolerance_default)

        sweep = run_convergence_sweep(
            ladder=ladder,
            build_member=lambda v, w=working, p=param: set_param(w, p, v),
            run_ladder=lambda members, p=param: run_ladder(p, members),
            read_observable=lambda d, o=observable: read_observable(d, o),
            tolerance=tolerance,
            param_name=param,
        )
        sweeps.append(sweep)
        if sweep.convergence.converged:
            working = set_param(working, param, sweep.convergence.setting)

    return ParameterConvergence(
        final_inputs=working,
        sweeps=sweeps,
        all_converged=all(s.convergence.converged for s in sweeps),
    )
