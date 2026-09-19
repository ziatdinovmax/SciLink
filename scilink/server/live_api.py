"""Backend of the web UI's Live tab: run a live measurement loop inside a
session, and let the page watch it.

A live run is a session-level background job, not a chat turn: it outlives
turns, and the page only OBSERVES it — everything it shows is read back from the
loop's own ``loop_log.jsonl``, the same file a status command or a later sweep
would read. The instrument is one of the simulated experiments
(:mod:`scilink.live.simulators`) or the user's own
:class:`~scilink.live.instruments.Instrument`, named as ``package.module:Class``
— the swap the simulators exist to rehearse.

One run per session at a time. The run's directory is
``<session_dir>/live/run_NNN/``.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_RUNS: Dict[str, "LiveRun"] = {}
_LOCK = threading.Lock()


class LiveError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def _instrument_info(inst: Any) -> Dict[str, Any]:
    return {
        "name": inst.name,
        "technique": (inst.system_info or {}).get("technique"),
        "sample": (inst.system_info or {}).get("sample"),
        "x_axis": (inst.system_info or {}).get("x_axis"),
        "y_axis": (inst.system_info or {}).get("y_axis"),
        "about": " ".join((inst.__class__.__doc__ or "").replace("``", "").split()),
        "schema": inst.schema.to_dict() if inst.schema is not None else {},
        "defaults": dict(inst.defaults or {}),
        "outputs": dict(inst.outputs or {}),
        "targets": list(inst.targets or []),
        "events": list(getattr(inst, "events", []) or []),
    }


def list_simulators() -> List[Dict[str, Any]]:
    from scilink.live.simulators import SIMULATORS
    return [_instrument_info(cls()) for cls in SIMULATORS.values()]


def _replay_instrument(config: Dict[str, Any]) -> Any:
    """Recorded measurements from a folder on this machine, as an instrument."""
    from scilink.live.instruments import ReplayInstrument
    folder = str(config.get("replay_dir") or "").strip()
    if not folder:
        raise LiveError(400, "Replay needs the folder that holds the recorded measurements.")
    info = {k: str(v).strip() for k, v in (config.get("system_info") or {}).items()
            if str(v or "").strip()}
    if not info.get("technique"):
        raise LiveError(400, "Replay needs at least the measurement technique — the analysis "
                             "is only as good as what it is told about the data.")
    outputs = {str(k).strip(): str(v).strip() for k, v in (config.get("outputs") or {}).items()
               if str(k).strip() and str(v or "").strip()}
    try:
        return ReplayInstrument(folder, system_info=info, outputs=outputs,
                                targets=[str(t).strip() for t in (config.get("targets") or [])
                                         if str(t).strip()])
    except ValueError as e:
        raise LiveError(400, str(e))


def _make_instrument(spec: str, seed: int, allow_custom: bool = True,
                     config: Optional[Dict[str, Any]] = None) -> Any:
    """A simulator by name, recorded data (``replay``), or the user's own
    ``package.module:Class``.

    Importing a module runs its top-level code, so a custom instrument is a
    local-machine feature (``allow_custom``); and the class is checked to BE an
    Instrument before it is called — never call an arbitrary named attribute."""
    from scilink.live.instruments import Instrument
    from scilink.live.simulators import SIMULATORS
    if spec in SIMULATORS:
        return SIMULATORS[spec](seed=seed)
    if spec == "replay":
        if not allow_custom:
            raise LiveError(403, "Replaying a folder is only available when SciLink runs on "
                                 "your own machine.")
        return _replay_instrument(config or {})
    if ":" not in spec:
        raise LiveError(400, f"Unknown instrument {spec!r}. Use one of {sorted(SIMULATORS)} "
                             "or your own as 'package.module:ClassName'.")
    if not allow_custom:
        raise LiveError(403, "Custom instruments are only available when SciLink runs on "
                             "your own machine; this server offers the simulated experiments.")
    module_name, _, cls_name = spec.partition(":")
    try:
        cls = getattr(importlib.import_module(module_name), cls_name)
    except Exception as e:  # noqa: BLE001
        raise LiveError(400, f"Could not load instrument {spec!r}: {type(e).__name__}: {e}")
    if not (isinstance(cls, type) and issubclass(cls, Instrument)):
        raise LiveError(400, f"{spec!r} is not a scilink.live.Instrument subclass.")
    try:
        return cls()
    except Exception as e:  # noqa: BLE001
        raise LiveError(400, f"Could not construct {spec!r}: {type(e).__name__}: {e}")


class LiveRun:
    """One background live run: arm the loop, then acquire → step → recommend."""

    def __init__(self, session_id: str, session_dir: str, agent: Any,
                 config: Dict[str, Any], allow_custom: bool = True) -> None:
        self.session_id = session_id
        self.config = config
        self.state = "arming"
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.instrument = _make_instrument(str(config.get("instrument") or ""),
                                           int(config.get("seed") or 0), allow_custom, config)
        if hasattr(self.instrument, "remaining"):       # a recording is finite; one file is the reference
            config["n_frames"] = max(1, min(int(config.get("n_frames") or 10 ** 6),
                                            len(self.instrument) - 1))
        root = Path(session_dir) / "live"
        root.mkdir(parents=True, exist_ok=True)
        n = len([p for p in root.glob("run_*") if p.is_dir()]) + 1
        self.run_dir = root / f"run_{n:03d}"
        self._stop = threading.Event()
        self._operator_params: Optional[Dict[str, Any]] = None
        self._oplock = threading.Lock()
        self.current_params: Dict[str, Any] = dict(self.instrument.defaults or {})
        #: Simulators only: the ground truth per frame, shown beside the estimate.
        self._truth: Dict[int, Dict[str, Any]] = {}
        self.loop: Any = None
        self._agent = agent
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"scilink-live-{session_id}")
        self._thread.start()

    # ------------------------------------------------------------------ run
    def _credentials(self) -> Dict[str, Any]:
        a = self._agent
        return {"model_name": getattr(a, "model_name", None) or "claude-opus-4-6",
                "api_key": getattr(a, "api_key", None),
                "base_url": getattr(a, "base_url", None)}

    def _recommender(self, creds: Dict[str, Any]):
        from scilink.live import GPRecommender, LLMRecommender
        kind = str(self.config.get("recommender") or "none")
        inst = self.instrument
        if kind == "none" or inst.schema is None:
            return None
        if kind == "gp":
            key = self.config.get("objective_key")
            if not key:
                raise LiveError(400, "The GP recommender needs an objective output to optimize.")
            return GPRecommender(inst.schema, key,
                                 direction=self.config.get("direction") or "maximize")
        if kind == "llm":
            objective = str(self.config.get("objective") or "").strip()
            if not objective:
                raise LiveError(400, "The LLM recommender needs a goal in words.")
            model = getattr(self.loop._agent_factory(str(self.run_dir / "model")), "model", None)
            keys = list(inst.outputs) + ["fit_r_squared"]
            return LLMRecommender(model, inst.schema, objective,
                                  output=self.config.get("llm_output") or "params",
                                  every=int(self.config.get("every") or 5), feature_keys=keys)
        raise LiveError(400, f"Unknown recommender {kind!r}.")

    def _run(self) -> None:
        from scilink.live import MeasurementLoop, run_experiment
        cfg, inst = self.config, self.instrument
        try:
            creds = self._credentials()
            outputs = dict(inst.outputs or {}) if cfg.get("pin_outputs", True) else {}
            self.loop = MeasurementLoop(
                str(self.run_dir / "loop"), system_info=inst.system_info,
                targets=list(inst.targets or []), outputs=outputs, schema=inst.schema,
                objective_key=(cfg.get("objective_key") or None) if outputs else None,
                auto_escalate=bool(cfg.get("auto_escalate", True)),
                breach_patience=int(cfg.get("breach_patience") or 3),
                closed_loop=(cfg.get("apply") == "valid"),
                frame_deadline_s=float(cfg.get("frame_deadline_s") or 10.0), **creds)
            reference = inst.acquire({}).save(str(self.run_dir / "reference"), 0, stem="reference")
            self.loop.setup(reference=reference,
                            profile=(cfg.get("reference_profile") or None))
            self.loop.recommender = self._recommender(creds)
            self.state = "running"
            run_experiment(
                inst, self.loop, int(cfg.get("n_frames") or 60),
                apply=str(cfg.get("apply") or "never"),
                interval_s=float(cfg.get("interval_s") or 2.0),
                stop=self._stop.is_set, operator=self._operator,
                on_frame=self._on_frame)
            self.state = "stopped" if self._stop.is_set() else "done"
        except LiveError as e:
            self.state, self.error = "error", e.message
        except Exception as e:  # noqa: BLE001 - reported to the page
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            (self.run_dir).mkdir(parents=True, exist_ok=True)
            (self.run_dir / "error.txt").write_text(traceback.format_exc())

    def _on_frame(self, frame: Any, record: Dict[str, Any]) -> None:
        self.current_params = dict(record.get("params") or {})
        if frame.truth:
            self._truth[int(record["step"])] = {
                k: v for k, v in frame.truth.items() if isinstance(v, (int, float))}

    def _operator(self, current: Dict[str, Any], record: Dict[str, Any]):
        with self._oplock:
            chosen, self._operator_params = self._operator_params, None
        return chosen

    # -------------------------------------------------------------- control
    def stop(self) -> None:
        self._stop.set()

    def set_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parameters the operator wants from the next frame on — an accepted
        recommendation or a manual change. Checked against the schema here so a
        bad value is refused at the button, not inside the run."""
        if self.instrument.schema is not None:
            problems = self.instrument.schema.validate(params)
            if problems:
                raise LiveError(400, "; ".join(problems))
        with self._oplock:
            self._operator_params = {**(self._operator_params or {}), **params}
        return {"queued": params}

    # ---------------------------------------------------------------- view
    def _events(self) -> List[Dict[str, Any]]:
        if self.loop is None:
            return []
        try:
            return self.loop.read_log()
        except Exception:  # noqa: BLE001
            return []

    def snapshot(self, tail: int = 400) -> Dict[str, Any]:
        events = self._events()
        frames = [e for e in events if e.get("event") == "frame"]
        other = [e for e in events if e.get("event") != "frame"]
        latest = frames[-1] if frames else None
        recs = [e for e in events if e.get("event") == "recommendation"]
        latest_rec = (latest or {}).get("recommendation") or (recs[-1] if recs else None)
        status: Dict[str, Any] = {}
        if self.loop is not None and self.loop.recipe is not None:
            try:
                status = self.loop.status()
            except Exception:  # noqa: BLE001
                status = {}
        return {
            "state": self.state, "error": self.error,
            "run_dir": str(self.run_dir), "elapsed_s": round(time.time() - self.started_at, 1),
            "config": self.config, "instrument": _instrument_info(self.instrument),
            "status": status, "current_params": self.current_params,
            "n_frames_total": int(self.config.get("n_frames") or 60),
            "frames": [{
                "step": f["step"], "features": f.get("features") or {},
                "flags": f.get("flags") or [], "latency_s": f.get("latency_s"),
                "recipe_id": f.get("recipe_id"), "params": f.get("params") or {},
                "gate": f.get("gate") or {}, "needs_escalation": f.get("needs_escalation"),
                "escalation": f.get("escalation"),
                "truth": self._truth.get(int(f["step"]), {}),
            } for f in frames[-tail:]],
            "events": [{k: v for k, v in e.items() if k not in ("reference_features",)}
                       for e in other[-60:]],
            "recommendation": latest_rec,
            # Before the first frame, the reference itself: something real to
            # look at during the minutes the loop is being armed.
            "latest": self._curve(latest) if latest else self._reference_curve(),
        }

    def _reference_curve(self) -> Optional[Dict[str, Any]]:
        ref = self.run_dir / "reference" / "reference_000000.csv"
        return self._curve({"step": 0, "data": str(ref)}) if ref.exists() else None

    @staticmethod
    def _curve(frame: Dict[str, Any], max_points: int = 600) -> Optional[Dict[str, Any]]:
        try:
            import numpy as np
            data = np.loadtxt(frame["data"], delimiter=",", skiprows=1)
            step = max(1, len(data) // max_points)
            return {"step": frame["step"], "x": [round(float(v), 5) for v in data[::step, 0]],
                    "y": [round(float(v), 5) for v in data[::step, 1]]}
        except Exception:  # noqa: BLE001
            return None


# ──────────────────────────────────────────────────────────────
# module API used by app.py
# ──────────────────────────────────────────────────────────────

def start(session: Any, config: Dict[str, Any], allow_custom: bool = True) -> Dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(session.id)
        if run is not None and run.state in ("arming", "running"):
            raise LiveError(409, "A live run is already active in this session — stop it first.")
        run = LiveRun(session.id, session.session_dir, session.agent, dict(config or {}),
                      allow_custom=allow_custom)
        _RUNS[session.id] = run
    return run.snapshot()


def snapshot(session: Any) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None:
        return {"state": "idle", "simulators": list_simulators()}
    return run.snapshot()


def stop(session: Any) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None:
        raise LiveError(404, "No live run in this session.")
    run.stop()
    return {"state": "stopping"}


def set_params(session: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None or run.state != "running":
        raise LiveError(409, "No running live loop to send parameters to.")
    return run.set_params(dict(params or {}))


def clear(session: Any) -> Dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(session.id)
        if run is not None and run.state in ("arming", "running"):
            raise LiveError(409, "Stop the run before starting a new one.")
        _RUNS.pop(session.id, None)
    return {"state": "idle", "simulators": list_simulators()}
