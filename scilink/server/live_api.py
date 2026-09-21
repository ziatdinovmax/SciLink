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


def _plain(text: Any) -> str:
    """Display copy for the page: one paragraph, no em dashes or semicolons."""
    t = " ".join(str(text or "").replace("``", "").split())
    return t.replace(" — ", ", ").replace("; ", ", ")


def _instrument_info(inst: Any) -> Dict[str, Any]:
    schema = inst.schema.to_dict() if inst.schema is not None else {}
    for spec in schema.values():
        if spec.get("description"):
            spec["description"] = _plain(spec["description"])
    return {
        "id": getattr(inst, "id", inst.name),
        "name": inst.name,
        "can_pause": bool(getattr(inst, "can_pause", False)),
        "technique": (inst.system_info or {}).get("technique"),
        "sample": (inst.system_info or {}).get("sample"),
        "x_axis": (inst.system_info or {}).get("x_axis"),
        "y_axis": (inst.system_info or {}).get("y_axis"),
        "about": _plain(inst.__class__.__doc__),
        "simulated": inst.__class__.__module__.endswith(".simulators"),
        "modality": getattr(inst, "modality", "curve"),
        "schema": schema,
        "defaults": dict(inst.defaults or {}),
        "outputs": {k: _plain(v) for k, v in (inst.outputs or {}).items()},
        "targets": list(inst.targets or []),
        "events": [{**e, "what": _plain(e.get("what"))}
                   for e in (getattr(inst, "events", []) or [])],
        "held": [_plain(h) for h in (getattr(inst, "held", None) or [])],
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
        raise LiveError(400, "Replay needs the measurement technique. The analysis is only as "
                             "good as what it is told about the data.")
    outputs = {str(k).strip(): str(v).strip() for k, v in (config.get("outputs") or {}).items()
               if str(k).strip() and str(v or "").strip()}
    try:
        return ReplayInstrument(folder, system_info=info, outputs=outputs,
                                targets=[str(t).strip() for t in (config.get("targets") or [])
                                         if str(t).strip()])
    except ValueError as e:
        raise LiveError(400, str(e))


def list_mcp_servers(agent: Any) -> List[Dict[str, Any]]:
    """MCP servers connected in this session (the MCP tab) and their tools — any
    of which may be an instrument's ``acquire``."""
    out = []
    for name, conn in (getattr(agent, "_mcp_connections", None) or {}).items():
        tools = [(t.get("function") or {}).get("name") for t in (getattr(conn, "tool_schemas", None) or [])]
        out.append({"name": name, "tools": [t for t in tools if t]})
    return out


def _described(config: Dict[str, Any]):
    """What the form says about the measurement: system_info, outputs, targets."""
    info = {k: str(v).strip() for k, v in (config.get("system_info") or {}).items()
            if str(v or "").strip()}
    outputs = {str(k).strip(): str(v).strip() for k, v in (config.get("outputs") or {}).items()
               if str(k).strip() and str(v or "").strip()}
    targets = [str(t).strip() for t in (config.get("targets") or []) if str(t).strip()]
    return info, outputs, targets


def _mcp_instrument(config: Dict[str, Any], agent: Any) -> Any:
    """An instrument behind an MCP server already connected in the MCP tab. The
    server is the driver; nothing is imported or executed here."""
    from scilink.live.mcp_instrument import MCPInstrument
    server = str(config.get("mcp_server") or "").strip()
    conn = (getattr(agent, "_mcp_connections", None) or {}).get(server)
    if conn is None:
        raise LiveError(400, f"No MCP server named {server!r} is connected in this session. "
                             "Connect it in the MCP tab first.")
    info, outputs, targets = _described(config)
    try:
        inst = MCPInstrument(conn, tool=str(config.get("mcp_tool") or "acquire"),
                             system_info=info or None, outputs=outputs or None,
                             targets=targets or None)
    except ValueError as e:
        raise LiveError(400, str(e))
    if not (inst.system_info or {}).get("technique"):
        raise LiveError(400, "Say which measurement technique this is. The server did not "
                             "describe itself and the analysis needs to know.")
    return inst


def _make_instrument(spec: str, seed: int, allow_custom: bool = True,
                     config: Optional[Dict[str, Any]] = None, agent: Any = None) -> Any:
    """A simulator by name, recorded data (``replay``), or the user's own
    ``package.module:Class``.

    Importing a module runs its top-level code, so a custom instrument is a
    local-machine feature (``allow_custom``); and the class is checked to BE an
    Instrument before it is called — never call an arbitrary named attribute."""
    from scilink.live.instruments import Instrument
    from scilink.live.simulators import SIMULATORS
    if spec in SIMULATORS:
        return SIMULATORS[spec](seed=seed)
    if spec == "mcp":
        return _mcp_instrument(config or {}, agent)
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
                             "your own machine. This server offers the simulated experiments.")
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


def list_reference_analyses(session_dir: str, limit: int = 40) -> List[Dict[str, Any]]:
    """Curve-fit runs in this session that a live loop can adopt as its
    reference (a saved script plus its results): chat analyses, and the
    references of earlier live runs. Newest first, paths relative to the session."""
    root = Path(session_dir)
    found = []
    for marker in root.rglob("series_fit_results.json"):
        d = marker.parent
        rel = d.relative_to(root)
        if len(rel.parts) > 8 or "frames" in rel.parts or "portability" in rel.parts \
                or "pinning" in rel.parts or "escalations" in rel.parts:
            continue
        if not any((d / "scripts").glob("*.py")):
            continue
        model = ""
        try:
            data = json.loads(marker.read_text())
            model = str((data.get("locked_config") or {}).get("physical_model")
                        or ((data.get("results") or [{}])[0]).get("model_type") or "")
        except (OSError, ValueError, AttributeError, IndexError):
            pass
        found.append({"path": str(rel), "name": d.name if d.name != "reference" else str(rel),
                      "model": model[:160], "modified": marker.stat().st_mtime,
                      "from_live_run": "live" in rel.parts,
                      "has_data": (d / "spectrum_0000" / "data.npy").exists()})
    found.sort(key=lambda r: -r["modified"])
    return found[:limit]


def _anchor_reference_csv(anchor: Path, dest: Path) -> Optional[str]:
    """The data a past analysis was run on, as a CSV the loop can check pinned
    outputs (and portability) against. None when the run kept no arrays."""
    try:
        import numpy as np
        data = np.asarray(np.load(anchor / "spectrum_0000" / "data.npy"), dtype=float)
        if data.ndim != 2:
            return None
        if data.shape[0] == 2 and data.shape[1] != 2:
            data = data.T
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "reference_000000.csv"
        np.savetxt(path, data[:, :2], delimiter=",", header="x,y", comments="")
        return str(path)
    except (OSError, ValueError):
        return None


class _LogTail:
    """Incremental reader of ``loop_log.jsonl``. An open-ended run can last for
    hours; re-reading the whole log on every poll would grow without bound, so
    this keeps a file offset, running counters and the last ``keep`` frames."""

    def __init__(self, keep: int = 600) -> None:
        self.keep, self.offset = keep, 0
        self.frames: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.n_frames = self.n_clean = self.llm_calls = self.reanchors = self.audits = 0
        self.last_audit: Optional[Dict[str, Any]] = None
        self.flag_counts: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.max_latency = 0.0

    def read(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind("\n")
        if end < 0:
            return
        self.offset += len(chunk[:end + 1].encode("utf-8"))
        for line in chunk[:end].splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "frame":
                self.events = (self.events + [e])[-80:]
                self.reanchors += e.get("event") == "reanchor"
                if e.get("event") == "audit":
                    self.audits += 1
                    self.last_audit = e
                continue
            self.n_frames += 1
            self.n_clean += not e.get("flags")
            self.llm_calls += int(e.get("llm_calls") or 0)
            for f in e.get("flags") or []:
                self.flag_counts[f] = self.flag_counts.get(f, 0) + 1
            lat = float(e.get("latency_s") or 0.0)
            self.max_latency = max(self.max_latency, lat)
            self.latencies = (self.latencies + [lat])[-200:]
            self.frames = (self.frames + [e])[-self.keep:]

    def status(self) -> Dict[str, Any]:
        lat = sorted(self.latencies) or [0.0]
        return {"frames": self.n_frames, "clean_frames": self.n_clean,
                "flag_counts": dict(self.flag_counts), "llm_calls_in_frames": self.llm_calls,
                "latency_s": {"median": lat[len(lat) // 2], "max": self.max_latency},
                "reanchors": self.reanchors, "audits": self.audits}


class LiveRun:
    """One background live run: arm the loop, then acquire → step → recommend."""

    def __init__(self, session_id: str, session_dir: str, agent: Any,
                 config: Dict[str, Any], allow_custom: bool = True) -> None:
        self.session_id = session_id
        self.config = config
        self.state = "arming"
        self.error: Optional[str] = None
        self.note: Optional[str] = None
        self.started_at = time.time()
        self.instrument = _make_instrument(str(config.get("instrument") or ""),
                                           int(config.get("seed") or 0), allow_custom, config,
                                           agent)
        # Frames to collect. None = open-ended, until Stop — the normal case
        # at an instrument. A recording is finite whatever was asked.
        n = config.get("n_frames")
        self.n_frames: Optional[int] = int(n) if n not in (None, "", 0, "0") else None
        self.from_analysis = str(config.get("reference_source") or "first_frame") == "analysis"
        if hasattr(self.instrument, "remaining"):
            n_ref = max(1, min(int(config.get("reference_frames") or 1), 25))
            left = len(self.instrument) - (0 if self.from_analysis else n_ref)   # some files are the reference
            self.n_frames = max(1, min(self.n_frames or left, left))
        self._tail = _LogTail()
        self._session_dir = Path(session_dir).resolve()
        root = Path(session_dir) / "live"
        root.mkdir(parents=True, exist_ok=True)
        n = len([p for p in root.glob("run_*") if p.is_dir()]) + 1
        self.run_dir = root / f"run_{n:03d}"
        self._stop = threading.Event()
        # A decision point: the run waits here until the person answers.
        self.pause_on = [w for w in (config.get("pause_on") or []) if w in ("novelty", "breach")]
        self.paused: Optional[Dict[str, Any]] = None
        self._decision: Any = None
        self._decided = threading.Event()
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
            # What the user wants the analysis to know — context, not a constraint.
            # It reaches the reference analysis, rebuilds, audits and the recommender.
            info = dict(inst.system_info or {})
            if str(cfg.get("notes") or "").strip():
                info["notes_from_the_user"] = str(cfg["notes"]).strip()[:2000]
            self.loop = MeasurementLoop(
                str(self.run_dir / "loop"), system_info=info, instrument=inst,
                on_change=str(cfg.get("on_change") or "report"),
                targets=list(inst.targets or []), outputs=outputs, schema=inst.schema,
                objective_key=(cfg.get("objective_key") or None) if outputs else None,
                auto_escalate=bool(cfg.get("auto_escalate", True)),
                breach_patience=int(cfg.get("breach_patience") or 3),
                reanchor_frames=int(cfg.get("reanchor_frames") or 5),
                audit_every=(int(cfg["audit_every"]) if cfg.get("audit_every") else None),
                closed_loop=(cfg.get("apply") == "valid"),
                # How long a frame may take before it is flagged slow. It belongs to
                # the instrument's cadence, so the page sets it; absent = 10 s (the
                # simulators' pace), an explicit null = no deadline.
                frame_deadline_s=((float(cfg["frame_deadline_s"]) if cfg["frame_deadline_s"] else None)
                                  if "frame_deadline_s" in cfg else 10.0), **creds)
            if self.from_analysis:
                anchor = (self._session_dir / str(cfg.get("reference_analysis") or "")).resolve()
                if not (anchor.is_dir() and anchor.is_relative_to(self._session_dir)
                        and (anchor / "series_fit_results.json").exists()):
                    raise LiveError(400, "That analysis is not a curve-fit run in this session.")
                ref_csv = _anchor_reference_csv(anchor, self.run_dir / "reference")
                if outputs and ref_csv is None:
                    # Named outputs are checked on the data the analysis saw;
                    # without it the recipe's own names are reported instead.
                    self.loop.outputs, self.note = {}, (
                        "This analysis kept no data arrays, so outputs are reported "
                        "under the recipe's own names.")
                self.loop.setup(anchor=str(anchor), reference_data=ref_csv)
            else:
                # The first frame, or the first few: several frames let the plan
                # see what moves before the recipe is locked on the last of them.
                n_ref = max(1, min(int(cfg.get("reference_frames") or 1), 25))
                refs = []
                for i in range(n_ref):
                    if self._stop.is_set():
                        break
                    refs.append(inst.acquire({}).save(str(self.run_dir / "reference"), i,
                                                      stem="reference"))
                    if i < n_ref - 1 and float(cfg.get("interval_s") or 0) > 0:
                        time.sleep(float(cfg.get("interval_s")))
                if not refs:
                    raise LiveError(400, "Stopped before a reference was acquired.")
                self.loop.setup(reference=refs if len(refs) > 1 else refs[0],
                                profile=(cfg.get("reference_profile") or None))
            self.loop.recommender = self._recommender(creds)
            self.state = "running"
            run_experiment(
                inst, self.loop, self.n_frames if self.n_frames is not None else 10 ** 9,
                apply=str(cfg.get("apply") or "never"),
                interval_s=float(cfg.get("interval_s") or 2.0),
                stop=self._stop.is_set, operator=self._operator,
                on_frame=self._on_frame,
                pause_on=self.pause_on, on_pause=self._on_pause if self.pause_on else None)
            self.state = "stopped" if self._stop.is_set() else "done"
        except LiveError as e:
            self.state, self.error = "error", e.message
        except Exception as e:  # noqa: BLE001 - reported to the page
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            (self.run_dir).mkdir(parents=True, exist_ok=True)
            (self.run_dir / "error.txt").write_text(traceback.format_exc())
        finally:
            if self.loop is not None:
                try:
                    self.loop.close()
                except Exception:  # noqa: BLE001
                    pass

    def _on_frame(self, frame: Any, record: Dict[str, Any]) -> None:
        self.current_params = dict(record.get("params") or {})
        if frame.truth:
            self._truth[int(record["step"])] = {
                k: v for k, v in frame.truth.items() if isinstance(v, (int, float))}
            for old_step in [k for k in self._truth if k < int(record["step"]) - 1000]:
                del self._truth[old_step]

    def _on_pause(self, event: Dict[str, Any], record: Dict[str, Any]) -> Any:
        """Wait for the person. Stop ends the wait; an optional time limit
        resumes unchanged, for an experiment that cannot be held for long."""
        limit = float(self.config.get("pause_timeout_s") or 0) or None
        self._decision = None
        self._decided.clear()
        self.paused = {**event, "at": time.time(), "experiment_held": bool(self.instrument.can_pause),
                       "timeout_s": limit}
        self.state = "paused"
        # The recommender was asked because the data changed; with no next frame
        # to collect its answer, fetch it here so the person deciding sees it.
        threading.Thread(target=self._recommend_while_paused, daemon=True,
                         name=f"scilink-live-pause-{self.session_id}").start()
        try:
            while not self._decided.wait(0.25):
                if self._stop.is_set():
                    return "stop"
                if limit and time.time() - self.paused["at"] > limit:
                    self.note = "Nobody answered the pause in time. The run went on unchanged."
                    return "resume"
            return self._decision or "resume"
        finally:
            self.paused = None
            self.state = "running"

    def _recommend_while_paused(self) -> None:
        try:
            self.loop.recommend_now()
        except Exception:  # noqa: BLE001 - a recommendation never fails a run
            pass

    def decide(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The person's answer to a pause: resume, resume with changed parameters, stop."""
        if self.paused is None:
            raise LiveError(409, "The run is not paused.")
        if action not in ("resume", "stop"):
            raise LiveError(400, "action is 'resume' or 'stop'.")
        if params and self.instrument.schema is not None:
            problems = self.instrument.schema.validate(params)
            if problems:
                raise LiveError(400, "; ".join(problems))
        self._decision = "stop" if action == "stop" else (dict(params) if params else "resume")
        if action == "stop":
            self._stop.set()
        self._decided.set()
        return {"decision": action, "params": params or None}

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
    def snapshot(self, tail: int = 400) -> Dict[str, Any]:
        if self.loop is not None:
            self._tail.read(self.loop.log_path)
        frames, other = self._tail.frames, self._tail.events
        latest = frames[-1] if frames else None
        recs = [e for e in other if e.get("event") == "recommendation"]
        latest_rec = (latest or {}).get("recommendation") or (recs[-1] if recs else None)
        if recs and latest_rec is not recs[-1] and (
                int(recs[-1].get("based_on_step") or -1) > int((latest_rec or {}).get("based_on_step") or -1)):
            latest_rec = recs[-1]            # one that arrived with no frame after it (a pause)
        status: Dict[str, Any] = self._tail.status()
        if self.loop is not None and self.loop.recipe is not None:
            meta = getattr(self.loop, "_escalation_meta", None) or {}
            status.update({"recipe": self.loop.recipe, "escalating": bool(self.loop.escalating),
                           "background": meta.get("mode") if self.loop.escalating else None,
                           "drift_fraction_bar": self.loop.drift_fraction})
        return {
            "state": self.state, "error": self.error, "note": self.note,
            "paused": self.paused, "pause_on": self.pause_on,
            "run_dir": str(self.run_dir), "elapsed_s": round(time.time() - self.started_at, 1),
            "config": self.config, "instrument": _instrument_info(self.instrument),
            "status": status, "current_params": self.current_params,
            "n_frames_total": self.n_frames,
            "output_keys": self._output_keys(latest),
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
            "last_audit": self._tail.last_audit,
            "novelties": [self._novelty_view(e) for e in other if e.get("event") == "novelty"][-5:],
            # Before the first frame, the reference itself: something real to
            # look at during the minutes the loop is being armed.
            "latest": self._curve(latest) if latest else self._reference_curve(),
        }

    def _novelty_view(self, e: Dict[str, Any]) -> Dict[str, Any]:
        """A novelty event for the page, with the session-relative path of the
        frame to hand to Chat."""
        try:
            rel = str(Path(str(e.get("data"))).resolve().relative_to(self._session_dir))
        except Exception:  # noqa: BLE001
            rel = str(e.get("data") or "")
        return {k: e.get(k) for k in ("step", "since_step", "fraction", "from_reference", "where",
                                      "recipe_fits", "window_share", "region")} | {
            # Shown relative; handed to Chat absolute — observed: the chat agent
            # could not resolve a session-relative path and asked for the full one.
            "frame_path": rel, "frame_abs_path": str(e.get("data") or "")}

    def _output_keys(self, latest: Optional[Dict[str, Any]], cap: int = 6) -> List[str]:
        """What to trace: the outputs the user named, else the recipe's own
        quantities (uncertainties and fit statistics left out)."""
        if self.loop is not None and self.loop.outputs:
            return list(self.loop.outputs)[:cap]
        feats = (latest or {}).get("features") or {}
        return [k for k in feats if not k.startswith("fit_")
                and not k.endswith(("_err", "_error", "_stderr", "_std"))][:cap]

    def _reference_curve(self) -> Optional[Dict[str, Any]]:
        refs = sorted(p for p in (self.run_dir / "reference").glob("reference_*")
                      if p.suffix.lower() != ".json")
        return self._curve({"step": 0, "data": str(refs[-1])}) if refs else None

    def _curve(self, frame: Dict[str, Any], max_points: int = 600) -> Optional[Dict[str, Any]]:
        """The frame's data and, when the analysis left one, the fitted model on
        the same x — the live analysis result, not just its input. A datacube
        is shown as its mean spectrum (what the change signal reads)."""
        try:
            import numpy as np
            if getattr(self.instrument, "modality", "curve") == "hyperspectral":
                from scilink.live.modality import HyperspectralModality
                x, y = HyperspectralModality().read_signal(
                    str(frame["data"]), getattr(self.instrument, "system_info", None))
                data = np.column_stack([x, y])
            else:
                data = np.loadtxt(frame["data"], delimiter=",", skiprows=1)
            step = max(1, len(data) // max_points)
            out = {"step": frame["step"], "x": [round(float(v), 5) for v in data[::step, 0]],
                   "y": [round(float(v), 5) for v in data[::step, 1]]}
        except Exception:  # noqa: BLE001
            return None
        try:
            fit = np.asarray(np.load(Path(str(frame.get("frame_dir"))) / "spectrum_0000" / "fit.npy"),
                             dtype=float)
            if fit.ndim == 2:
                fit = fit[1] if (fit.shape[0] == 2 and fit.shape[1] != 2) else fit[:, -1]
            if fit.size == len(data):
                out["fit"] = [None if not np.isfinite(v) else round(float(v), 5)
                              for v in fit.ravel()[::step]]
        except Exception:  # noqa: BLE001 - a frame without a stored fit shows its data only
            pass
        return out


# ──────────────────────────────────────────────────────────────
# module API used by app.py
# ──────────────────────────────────────────────────────────────

def start(session: Any, config: Dict[str, Any], allow_custom: bool = True) -> Dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(session.id)
        if run is not None and run.state in ("arming", "running", "paused"):
            raise LiveError(409, "A live run is already active in this session. Stop it first.")
        run = LiveRun(session.id, session.session_dir, session.agent, dict(config or {}),
                      allow_custom=allow_custom)
        _RUNS[session.id] = run
    return run.snapshot()


def _idle(session: Any) -> Dict[str, Any]:
    return {"state": "idle", "simulators": list_simulators(),
            "mcp_servers": list_mcp_servers(session.agent),
            "analyses": list_reference_analyses(session.session_dir)}


def snapshot(session: Any) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None:
        return _idle(session)
    return run.snapshot()


def stop(session: Any) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None:
        raise LiveError(404, "No live run in this session.")
    run.stop()
    return {"state": "stopping"}


def set_params(session: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None or run.state not in ("running", "paused"):
        raise LiveError(409, "No running live loop to send parameters to.")
    return run.set_params(dict(params or {}))


def decide(session: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    run = _RUNS.get(session.id)
    if run is None:
        raise LiveError(404, "No live run in this session.")
    return run.decide(str((body or {}).get("action") or "resume"), (body or {}).get("params") or None)


def clear(session: Any) -> Dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(session.id)
        if run is not None and run.state in ("arming", "running", "paused"):
            raise LiveError(409, "Stop the run before starting a new one.")
        _RUNS.pop(session.id, None)
    return _idle(session)
