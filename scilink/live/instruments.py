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
only according to the ``apply`` policy the caller chose. The same holds for
holding the experiment: ``pause()`` / ``resume()`` are called here only, and only
when the caller asked for decision points (``pause_on``) and supplied someone to
decide (``on_pause``).

An instrument has an identity (``describe()``): runs record which instrument
they served, so what is learned can be kept per instrument, not per session.
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
    """One acquisition plus what produced it: a 1D curve, or — with ``cube`` — a
    datacube (H, W, channels), where ``x`` / ``y`` are its spectral axis and mean
    spectrum (what a person looks at, and what the change signal reads)."""
    x: Any
    y: Any
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Simulators only — the ground truth behind this frame. A real instrument
    #: leaves it empty; nothing in the loop ever reads it.
    truth: Dict[str, Any] = field(default_factory=dict)
    x_label: str = "x"
    y_label: str = "y"
    #: A datacube frame (H, W, channels). ``None`` for a curve.
    cube: Any = None
    #: An image frame (H, W); ``x`` / ``y`` are then its radial power spectrum.
    image: Any = None

    def save(self, directory: str, index: int, stem: str = "frame") -> str:
        """Write ``<stem>_<index>.csv`` (a curve) or ``.npy`` (a datacube) and a
        same-stem ``.json`` sidecar (the acquisition parameters and metadata);
        return the data file's path."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.cube is not None or self.image is not None:
            path = d / f"{stem}_{index:06d}.npy"
            np.save(path, np.asarray(self.cube if self.cube is not None else self.image,
                                     dtype=np.float32))
        else:
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
    #: What a frame IS: ``"curve"`` (a 1D spectrum), ``"hyperspectral"`` (a
    #: datacube) or ``"image"``. A loop given this instrument follows it accordingly.
    modality: str = "curve"

    def acquire(self, params: Dict[str, Any]) -> Frame:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def id(self) -> str:
        """A stable identifier of THIS instrument (default: its ``name``). Runs
        are stamped with it, so that what is learned at an instrument — recipes,
        states already seen, history — can belong to the instrument rather than
        to whichever session happened to drive it."""
        return str(getattr(self, "instrument_id", None) or self.name)

    def describe(self) -> Dict[str, Any]:
        info = self.system_info if isinstance(self.system_info, dict) else {}
        return {"id": self.id, "name": self.name, "technique": info.get("technique"),
                "kind": type(self).__name__, "can_pause": self.can_pause,
                "modality": self.modality}

    #: True when ``pause()`` really holds the experiment (not just the acquisition).
    can_pause: bool = False

    def pause(self) -> None:
        """Hold the experiment where holding is possible: blank the beam, hold
        the temperature ramp, park the stage. Default: nothing — then a pause
        only stops acquiring, and a sample that evolves on its own keeps
        evolving. Called by :func:`run_experiment`, only under a ``pause_on``
        policy the caller chose."""

    def resume(self) -> None:
        """Undo ``pause()``."""

    def check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """``defaults`` overlaid with ``params``, refused if the schema objects —
        the instrument-side guard, independent of the loop's own validation."""
        merged = {**self.defaults, **(params or {})}
        if self.schema is not None:
            problems = self.schema.validate(merged)
            if problems:
                raise ValueError(f"{self.name}: refusing to acquire — " + "; ".join(problems))
        return merged


class EndOfData(Exception):
    """The data source has nothing more to give (a replay ran out of files)."""


_CURVE_SUFFIXES = (".csv", ".txt", ".xy", ".dat", ".tsv", ".npy")


def read_curve(path: str):
    """``(x, y, x_label, y_label)`` from a recorded 1D measurement: a two-column
    text file (comma / tab / whitespace separated, optional header line) or a
    ``.npy`` of shape (N, 2), (2, N) or (N,) — the last with the index as x."""
    p = Path(path)
    x_label, y_label = "x", "y"
    if p.suffix.lower() == ".npy":
        a = np.asarray(np.load(p), dtype=float)
    else:
        lines = [ln for ln in p.read_text(errors="replace").splitlines()
                 if ln.strip() and not ln.lstrip().startswith(("#", "%", ";"))]
        if not lines:
            raise ValueError(f"{p.name}: no data")
        delim = "," if "," in lines[-1] else ("\t" if "\t" in lines[-1] else None)

        def _row(ln):
            return [float(v) for v in ln.replace(";", " ").split(delim)[:2]]
        try:
            _row(lines[0])
        except ValueError:                      # a header line: keep its names
            names = [n.strip() for n in lines[0].split(delim)]
            if len(names) >= 2:
                x_label, y_label = names[0] or "x", names[1] or "y"
            lines = lines[1:]
        a = np.asarray([_row(ln) for ln in lines], dtype=float)
    if a.ndim == 1:
        a = np.column_stack([np.arange(a.size, dtype=float), a])
    elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] < 3:
        raise ValueError(f"{p.name}: expected two columns (x, y), got shape {a.shape}")
    return a[:, 0], a[:, 1], x_label, y_label


def _natural_key(path: Path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


class ReplayInstrument(Instrument):
    """Recorded measurements served one per ``acquire()``, in file order — real
    data through the live loop, before there is a live instrument.

        inst = ReplayInstrument("runs/2026-03-anneal/", system_info={
            "technique": "Raman spectroscopy", "sample": "carbon film, annealed in situ",
            "x_axis": "Raman shift (cm^-1)", "y_axis": "intensity (counts)"},
            outputs={"g_position": "position of the G band"})

    ``source`` is a directory (every two-column ``.csv/.txt/.xy/.dat/.tsv/.npy``
    in it, sorted naturally so ``scan_2`` precedes ``scan_10``) or an explicit
    list of files. A same-stem ``.json`` sidecar, when present, supplies the
    frame's recorded ``params`` and ``meta``. There is no ``schema``: recorded
    data cannot be steered, so recommendations made on a replay are advice about
    a run that already happened. ``acquire()`` raises :class:`EndOfData` after
    the last file, which ends :func:`run_experiment` cleanly.
    """

    def __init__(self, source: Any, *, system_info: Optional[Dict[str, Any]] = None,
                 outputs: Optional[Dict[str, str]] = None, targets: Optional[List[str]] = None,
                 name: str = "replay", modality: Optional[str] = None) -> None:
        from .modality import CUBE_SUFFIXES, IMAGE_SUFFIXES
        suffixes = tuple(dict.fromkeys(_CURVE_SUFFIXES + CUBE_SUFFIXES + IMAGE_SUFFIXES))
        if isinstance(source, (str, Path)):
            root = Path(source).expanduser()
            if not root.is_dir():
                raise ValueError(f"replay source {str(root)!r} is not a directory")
            files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
        else:
            files = [Path(p).expanduser() for p in source]
        self.files: List[Path] = sorted(files, key=_natural_key)
        if not self.files:
            raise ValueError(f"replay source has no measurement files ({', '.join(suffixes)})")
        # Recorded datacubes (a folder of .npy / HDF5 spectrum images) replay as
        # cubes. Told, or read off the first file: 3D is a datacube.
        self.modality = modality or self._sniff(self.files[0])
        if self.modality == "hyperspectral":
            self.files = [p for p in self.files if p.suffix.lower() in CUBE_SUFFIXES]
        elif self.modality == "image":
            self.files = [p for p in self.files if p.suffix.lower() in IMAGE_SUFFIXES]
        self.name = name
        self.system_info = dict(system_info or {})
        # Recorded data is not an instrument, and two folders are not the same
        # one: what is remembered is kept under the instrument the metadata names
        # (system_info["instrument"]), else under the folder the data came from.
        self.instrument_id = (str(self.system_info.get("instrument") or "").strip()
                              or f"{name}-{self.files[0].parent.name or 'files'}")
        self.outputs = dict(outputs or {})
        self.targets = list(targets or [])
        self.events = [{"frame": 1, "what": f"{len(self.files)} recorded measurements, "
                                            f"{self.files[0].name} … {self.files[-1].name}"}]
        self._next = 0

    def __len__(self) -> int:
        return len(self.files)

    @property
    def remaining(self) -> int:
        return len(self.files) - self._next

    def acquire(self, params: Dict[str, Any]) -> Frame:
        if self._next >= len(self.files):
            raise EndOfData(f"all {len(self.files)} recorded measurements have been replayed")
        path = self.files[self._next]
        self._next += 1
        cube = image = None
        if self.modality == "image":
            from .modality import ImageModality, load_image
            image = load_image(path)
            x, y = ImageModality().read_signal(str(path), self.system_info)
            x_label, y_label = "spatial_frequency_cycles_per_px", "log10_power"
        elif self.modality == "hyperspectral":
            from .modality import HyperspectralModality, load_cube
            cube = load_cube(path)
            x, y = HyperspectralModality().read_signal(str(path), self.system_info)
            x_label, y_label = "channel", "mean_intensity"
        else:
            x, y, x_label, y_label = read_curve(str(path))
        recorded: Dict[str, Any] = {}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                recorded = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                recorded = {}
        if not isinstance(recorded, dict):
            recorded = {}
        rec_params = recorded.get("params") if isinstance(recorded.get("params"), dict) else {}
        meta = recorded.get("meta") if isinstance(recorded.get("meta"), dict) else {
            k: v for k, v in recorded.items() if k not in ("params", "truth")}
        return Frame(x=x, y=y, params={**rec_params, **(params or {})},
                     meta={**meta, "source_file": path.name, "index": self._next},
                     x_label=x_label, y_label=y_label, cube=cube, image=image)

    @staticmethod
    def _sniff(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in (".h5", ".hdf5", ".nxs"):
            return "hyperspectral"
        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            return "image"
        if suffix == ".npy":
            try:
                shape = np.load(path, mmap_mode="r").shape
            except Exception:  # noqa: BLE001
                return "curve"
            if len(shape) == 3:
                return "hyperspectral"
            # (N, 2) / (2, N) / (N,) is a curve; anything wider both ways is an image
            return "image" if len(shape) == 2 and min(shape) >= 16 else "curve"
        return "curve"


def run_experiment(instrument: Instrument, loop: Any, n_frames: int, *,
                   apply: str = "approved", params: Optional[Dict[str, Any]] = None,
                   interval_s: float = 0.0,
                   on_frame: Optional[Callable[[Frame, Dict[str, Any]], None]] = None,
                   stop: Optional[Callable[[], bool]] = None,
                   operator: Optional[Callable[[Dict[str, Any], Dict[str, Any]],
                                               Optional[Dict[str, Any]]]] = None,
                   pause_on: Any = (),
                   on_pause: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None
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

    ``pause_on`` makes the experiment WAIT when something worth a decision
    happens — ``"novelty"`` (the data changed for good: the loop's ``novelty``
    event, with how much and where) and/or ``"breach"`` (a run of frames the
    recipe fails on). Where an experiment can be held, this is what turns a
    discovery from a log entry into a decision point: the frames stop, the
    instrument is asked to hold (``instrument.pause()``), and the slow work —
    a thorough analysis, a literature check, a person looking — happens while
    the sample is still in the state that looked new. ``on_pause(event, record)``
    blocks until that decision exists and returns ``"resume"``, ``"stop"``, or
    a dict of acquisition parameters to resume with (checked like any other).
    It is required with ``pause_on``: a pause nobody can end is a hang.

    The caller owns the loop: call ``loop.close()`` when the stream ends (or use
    the loop as a context manager) so a re-anchor still running in the
    background is stopped. A rebuild takes one to three minutes, so a recording
    replayed with ``interval_s=0`` usually ends before one lands; give it a
    realistic ``interval_s`` to see a rebuild adopted.
    """
    if apply not in APPLY_POLICIES:
        raise ValueError(f"apply must be one of {APPLY_POLICIES}")
    pause_on = {pause_on} if isinstance(pause_on, str) else set(pause_on or ())
    if pause_on - {"novelty", "breach"}:
        raise ValueError("pause_on accepts 'novelty' and 'breach'")
    if pause_on and on_pause is None:
        raise ValueError("pause_on needs on_pause: a pause nobody can end is a hang")
    was_breaching = False
    current = instrument.check(params or {})
    incoming = Path(loop.output_dir) / "incoming"
    records: List[Dict[str, Any]] = []
    applied_from: Optional[int] = None
    for i in range(1, n_frames + 1):
        if stop is not None and stop():
            break
        try:
            frame = instrument.acquire(dict(current))
        except EndOfData:
            break
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
        # A decision point, where the caller asked for one.
        reason = None
        if "novelty" in pause_on and record.get("novelty"):
            reason = {"why": "novelty", **record["novelty"]}
        elif "breach" in pause_on and record.get("needs_escalation") and not was_breaching:
            reason = {"why": "breach", "step": record.get("step"), "flags": record.get("flags")}
        was_breaching = bool(record.get("needs_escalation"))
        if reason is not None:
            note = getattr(loop, "record_event", None) or (lambda *a, **k: None)
            t_pause = time.time()
            note("paused", why=reason["why"], held=bool(instrument.can_pause))
            instrument.pause()
            try:
                decision = on_pause(reason, record)
            finally:
                instrument.resume()
            changed = decision if isinstance(decision, dict) and decision else None
            note("resumed", waited_s=round(time.time() - t_pause, 1), params=changed,
                 decision="stop" if decision == "stop" else "change" if changed else "resume")
            if decision == "stop":
                break
            if changed:
                current = instrument.check({**current, **changed})
        if operator is not None:
            chosen = operator(dict(current), record)
            if chosen:
                current = instrument.check({**current, **chosen})
        if interval_s:
            time.sleep(interval_s)
    return records
