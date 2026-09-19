"""The instrument side of a live loop — and the point where a simulator is
swapped for the real thing.

An :class:`Instrument` is anything that (a) declares the acquisition parameters
its controller accepts and (b) returns a frame for a given set of them::

    class MySpectrometer(Instrument):
        name = "my_raman"
        system_info = {"technique": "Raman spectroscopy", ...}
        schema = InstrumentSchema.from_dict({"integration_s": {"low": 0.1, "high": 60}})
        defaults = {"integration_s": 1.0}

        def acquire(self, params):
            x, y = vendor_api.measure(integration=params["integration_s"])
            return Frame(x=x, y=y, params=params)

The simulators in :mod:`scilink.live.simulators` implement exactly this
interface, so a scientist can get a feel for the loop on a simulated experiment
and then replace ``InSituRaman()`` with ``MySpectrometer()`` — nothing else
changes. :func:`run_experiment` is the driver both go through:

    acquire(params) -> loop.step(frame) -> (maybe) apply the recommendation

SciLink never actuates: the only place parameters change is this driver, and
only according to the ``apply`` policy the caller chose.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .recommend import InstrumentSchema

APPLY_POLICIES = ("never", "approved", "valid")


@dataclass
class Frame:
    """One acquisition: a 1D curve plus what produced it."""
    x: Any
    y: Any
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Simulators only — the ground truth behind this frame. A real instrument
    #: leaves it empty; nothing in the loop ever reads it.
    truth: Dict[str, Any] = field(default_factory=dict)
    x_label: str = "x"
    y_label: str = "y"

    def save(self, directory: str, index: int, stem: str = "frame") -> str:
        """Write ``<stem>_<index>.csv`` and a same-stem ``.json`` sidecar (the
        acquisition parameters and metadata); return the CSV path."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stem}_{index:06d}.csv"
        np.savetxt(path, np.column_stack([np.asarray(self.x, float), np.asarray(self.y, float)]),
                   delimiter=",", header=f"{self.x_label},{self.y_label}", comments="")
        sidecar = {"params": self.params, "meta": self.meta}
        if self.truth:
            sidecar["truth"] = self.truth
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, default=str))
        return str(path)


class Instrument:
    """The interface a live loop's data source implements. Subclass it for a
    real instrument; see the module docstring."""

    #: Short identifier.
    name: str = "instrument"
    #: Measurement metadata handed to every analysis (technique, sample, axes).
    system_info: Dict[str, Any] = {}
    #: The acquisition parameters the controller accepts, with their limits.
    schema: Optional[InstrumentSchema] = None
    #: Parameters to start from.
    defaults: Dict[str, Any] = {}
    #: Suggested pinned outputs for this kind of measurement — name: definition.
    outputs: Dict[str, str] = {}
    #: Suggested targets, in plain words.
    targets: List[str] = []

    def acquire(self, params: Dict[str, Any]) -> Frame:  # pragma: no cover - interface
        raise NotImplementedError

    def check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """``defaults`` overlaid with ``params``, refused if the schema objects —
        the instrument-side guard, independent of the loop's own validation."""
        merged = {**self.defaults, **(params or {})}
        if self.schema is not None:
            problems = self.schema.validate(merged)
            if problems:
                raise ValueError(f"{self.name}: refusing to acquire — " + "; ".join(problems))
        return merged


def run_experiment(instrument: Instrument, loop: Any, n_frames: int, *,
                   apply: str = "approved", params: Optional[Dict[str, Any]] = None,
                   interval_s: float = 0.0,
                   on_frame: Optional[Callable[[Frame, Dict[str, Any]], None]] = None,
                   stop: Optional[Callable[[], bool]] = None,
                   operator: Optional[Callable[[Dict[str, Any], Dict[str, Any]],
                                               Optional[Dict[str, Any]]]] = None
                   ) -> List[Dict[str, Any]]:
    """Acquire → analyse → (maybe) apply the recommendation, ``n_frames`` times.

    ``apply`` is the only thing that ever changes the acquisition parameters:

    - ``"never"``    — recommendations are recorded, parameters never change;
    - ``"approved"`` — applied only when the loop did not mark them
      ``requires_approval`` (i.e. a closed loop and a valid parameter set);
    - ``"valid"``    — every valid parameter recommendation is applied (the
      caller is the operator pressing "accept").

    ``operator(current_params, record)`` is the person at the controls: called
    after every frame, it may return parameters to use from the next frame on —
    an accepted recommendation, or a manual change. They go through the same
    instrument-side check as everything else.

    The loop must already be armed (``setup()``). Returns the frame records,
    each with the simulator's ``truth`` attached when there is one.
    """
    if apply not in APPLY_POLICIES:
        raise ValueError(f"apply must be one of {APPLY_POLICIES}")
    current = instrument.check(params or {})
    incoming = Path(loop.output_dir) / "incoming"
    records: List[Dict[str, Any]] = []
    applied_from: Optional[int] = None
    for i in range(1, n_frames + 1):
        if stop is not None and stop():
            break
        frame = instrument.acquire(dict(current))
        path = frame.save(str(incoming), i)
        record = loop.step(path, params=dict(current))
        if frame.truth:
            record = {**record, "truth": frame.truth}
        records.append(record)
        if on_frame is not None:
            on_frame(frame, record)
        rec = record.get("recommendation") or {}
        fresh = rec.get("based_on_step") != applied_from
        if (apply != "never" and rec.get("valid") and rec.get("params") and fresh
                and (apply == "valid" or not rec.get("requires_approval"))):
            current = instrument.check({**current, **rec["params"]})
            applied_from = rec.get("based_on_step")
        if operator is not None:
            chosen = operator(dict(current), record)
            if chosen:
                current = instrument.check({**current, **chosen})
        if interval_s:
            time.sleep(interval_s)
    return records
