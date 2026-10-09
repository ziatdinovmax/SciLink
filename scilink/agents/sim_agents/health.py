"""Engine-neutral physical-sanity gate for a finished run.

A completed process is not the same as a physically valid run: a barostat can
drive a simulation box to a near-vacuum, an SCF can settle on a nonsensical
energy, a relaxation can explode — and the engine still exits 0. Nothing
downstream catches that, because the autonomy's self-correction (method
escalation, reparameterization) only fires on a *converged observable compared
to a reference*, which a pathological run never reaches.

This module is the tripwire ahead of that path. It is pure: given the
observables a run produced and the plausible bands a skill declares, it returns
the bands that were violated. It knows no engine and does no I/O — the per-engine
hook that *reads* an observable from a run directory, and the ``health:`` block
that *declares* the bands, live in the skill bundle and are resolved by name
through the registry (mirroring the ``convergence:`` machinery in
:mod:`scilink.agents.sim_agents.convergence`).

The bands are **gross physical-sanity bounds, not accuracy checks**: a liquid
whose density came out near zero is broken; whether its density is 1.00 vs 1.05
g/cm³ is an accuracy question answered downstream against a reference, never
here. Keep the declared bands loose enough that only a genuinely non-physical
run trips them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HealthBand:
    """A plausible range for one observable.

    Either bound may be ``None`` (one-sided). ``observable`` is the name the
    per-engine reader hook understands (e.g. ``"density"``).
    """

    observable: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass(frozen=True)
class HealthViolation:
    """A reading that fell outside its band (or was non-finite)."""

    observable: str
    value: float
    minimum: Optional[float]
    maximum: Optional[float]

    @property
    def reason(self) -> str:
        if not math.isfinite(self.value):
            return f"{self.observable}={self.value} (non-finite)"
        if self.minimum is not None and self.value < self.minimum:
            return f"{self.observable}={self.value:g} < min {self.minimum:g}"
        if self.maximum is not None and self.value > self.maximum:
            return f"{self.observable}={self.value:g} > max {self.maximum:g}"
        return f"{self.observable}={self.value:g} out of band"


def parse_health_specs(specs) -> List[HealthBand]:
    """Coerce raw frontmatter specs (a list of dicts) into :class:`HealthBand`.

    Tolerant by design: a spec missing an ``observable`` name, or with no bound
    at all, is skipped rather than raising, so a malformed ``health:`` block
    never breaks a run. Accepts ``min``/``minimum`` and ``max``/``maximum``.
    """
    bands: List[HealthBand] = []
    if not isinstance(specs, list):
        return bands
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("observable")
        if not name:
            continue
        lo = spec.get("min", spec.get("minimum"))
        hi = spec.get("max", spec.get("maximum"))
        lo = float(lo) if isinstance(lo, (int, float)) else None
        hi = float(hi) if isinstance(hi, (int, float)) else None
        if lo is None and hi is None:
            continue
        bands.append(HealthBand(observable=str(name), minimum=lo, maximum=hi))
    return bands


def evaluate_health(
    observations: Dict[str, Optional[float]],
    specs,
) -> List[HealthViolation]:
    """Return the bands a run violated.

    ``observations`` maps observable name -> value (or ``None`` when the reader
    could not extract it). A ``None`` value is *skipped*, not treated as a
    violation: the gate judges only what it can actually read, and an engine
    that declares no bands (empty ``specs``) is simply ungated. A non-finite
    value (NaN/inf) is always a violation — it is never a valid physical state.

    No side effects, no I/O, no engine knowledge: pure and unit-testable.
    """
    bands = parse_health_specs(specs)
    violations: List[HealthViolation] = []
    for band in bands:
        value = observations.get(band.observable)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            violations.append(HealthViolation(
                band.observable, value, band.minimum, band.maximum))
            continue
        if band.minimum is not None and value < band.minimum:
            violations.append(HealthViolation(
                band.observable, value, band.minimum, band.maximum))
        elif band.maximum is not None and value > band.maximum:
            violations.append(HealthViolation(
                band.observable, value, band.minimum, band.maximum))
    return violations


def observable_names(specs) -> List[str]:
    """The distinct observable names a ``health:`` block declares."""
    seen: List[str] = []
    for band in parse_health_specs(specs):
        if band.observable not in seen:
            seen.append(band.observable)
    return seen
