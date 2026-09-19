"""MeasurementLoop — a live analysis loop for 1D spectra.

The shape is the one ``_qc_profile.py`` names and the series code already
runs: **lock once → execute per frame → gate as drift detector → flag on
breach**. A reference is analysed properly once (thorough, human gates and
all — or found in the script bank, or an existing run directory is adopted);
its approved script becomes the locked recipe; every frame after that replays
the recipe through the curve agent's ``realtime`` profile, which spends no LLM
call on the happy path.

Two clocks. ``step()`` is the fast one: synchronous, deterministic, no network
— it must work on an instrument PC with no LLM endpoint reachable. Anything
that needs a model (the reference analysis, a re-anchor) belongs to the slow
clock and never runs inside ``step()``; a frame the recipe cannot handle is
returned *flagged*, and ``needs_escalation`` tells the caller the recipe has
stopped describing the data.

Escalation is that slow clock at work. ``escalate(frame)`` re-anchors on the
breaching frame in a spawned process while ``step()`` keeps answering (flagged)
with the old recipe; the new recipe is adopted between frames. The re-anchor
is one ordinary analysis at a fit-for-purpose depth, which already IS the
cost ladder: the script bank is auditioned first (a regime seen before costs
no model call), then a banked script is edit-adapted (one call), and only
then is code written fresh.

The loop recommends, it never actuates: ``step()`` returns numbers, flags and
(when a recommender is attached) suggested next parameters. Bounds and safe
limits belong to the caller.

``loop_log.jsonl`` is the contract. One JSON record per event, append-only:
the UI, a status command, a later thorough sweep of the flagged frames and the
feature table all read that one file. ``loop_state.json`` holds what is needed
to resume after a crash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOOP_LOG_NAME = "loop_log.jsonl"
LOOP_STATE_NAME = "loop_state.json"
SCHEMA_VERSION = 1

#: Flags a frame record can carry. A flag never stops the loop.
FLAG_FIT_FAILED = "fit_failed"            # the recipe did not produce a result
FLAG_GATE_POOR = "gate_poor"              # below the deterministic fit gate
FLAG_DRIFT = "drift_suspected"            # the data no longer looks like the reference
FLAG_DEADLINE = "deadline_missed"         # slower than frame_deadline_s
FLAG_LLM_USED = "llm_used"                # the zero-LLM path was left (fallback codegen)
FLAG_OUT_OF_RANGE = "out_of_reference_range"  # a target left its plausible range

#: Flags that say "the recipe has stopped describing the data".
_BREACH_FLAGS = (FLAG_FIT_FAILED, FLAG_GATE_POOR, FLAG_DRIFT)


class LoopNotReady(RuntimeError):
    """``step()`` / ``amend()`` before ``setup()`` locked a recipe."""


from ._reanchor import reanchor as _reanchor_worker  # noqa: E402


class _ProcessEscalation:
    """A re-anchor running in its own interpreter; ``poll()`` never blocks.

    Launched as ``python -m scilink.live._reanchor`` with the spec on stdin —
    NOT ``multiprocessing`` spawn. A spawned child re-imports the caller's
    ``__main__``, so a driving script without an ``if __name__ == "__main__"``
    guard re-runs itself inside the worker (observed live: the child re-ran the
    whole scenario, including deleting the run directory). A ``-m`` subprocess
    never sees the caller's main module, and stdin keeps the spec — which may
    carry a credential — off the disk and off the command line.
    """

    def __init__(self, spec: Dict[str, Any]) -> None:
        import subprocess
        import sys
        self.out_dir = Path(spec["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.out_dir / "worker.out", "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "scilink.live._reanchor"],
            stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT)
        self._proc.stdin.write(json.dumps(spec, default=str).encode("utf-8"))
        self._proc.stdin.close()

    def poll(self) -> Optional[Dict[str, Any]]:
        result = self.out_dir / "result.json"
        if result.exists():
            try:
                return json.loads(result.read_text())
            except ValueError:
                return None                  # mid-rename; next poll
        code = self._proc.poll()
        if code is not None:
            return {"status": "error",
                    "error": f"escalation worker exited (code {code}) without a result; "
                             f"see {self.out_dir / 'worker.out'}"}
        return None

    def wait(self, timeout: Optional[float] = None) -> None:
        self._proc.wait(timeout)


class _InlineEscalation:
    """The same re-anchor, run in this process (``background=False``)."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        self.out_dir = Path(spec["out_dir"])
        _reanchor_worker(spec)

    def poll(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads((self.out_dir / "result.json").read_text())
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _numeric_features(result: Dict[str, Any], lift: bool = True) -> Dict[str, float]:
    """Flat numeric features of one frame: every scalar fit parameter, plus the
    fit-quality scalars under a ``fit_`` prefix. Pinned outputs (see
    :mod:`scilink.live.pinning`) are also exposed under their plain declared
    names unless ``lift`` is off."""
    from ..agents.exp_agents.feature_table import _flatten_scalars
    out: Dict[str, float] = {}
    for prefix, block in (("", result.get("fitting_parameters")),
                          ("fit_", result.get("fit_quality"))):
        for k, v in _flatten_scalars(block or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v != v or v in (float("inf"), float("-inf")):   # NaN / inf
                continue
            out[f"{prefix}{k}"] = float(v)
    if lift:
        from .pinning import lift_outputs
        out = lift_outputs(out)
    return out


class MeasurementLoop:
    """A live analysis loop over 1D spectra.

    Args:
        output_dir: Loop directory (log, state, per-frame run directories).
        model_name / api_key / base_url: For the slow clock only (the
            reference analysis). ``step()`` never calls a model.
        system_info: Measurement metadata passed to every analysis.
        targets: The quantities the consumer needs, in plain words — handed to
            the reference analysis so its verifier judges those.
        outputs: Pinned outputs — ``{name: definition in plain words}``, e.g.
            ``{"peak1_height": "height of the first peak above the baseline"}``.
            A generated script names its parameters as it pleases (seen live:
            ``peak_1_amplitude`` was the height in one script, the area in the
            next), so every recipe this loop locks — the first, and each
            re-anchor — is extended to report exactly these names. See
            :mod:`scilink.live.pinning`. Costs one model call per recipe, on
            the slow clock.
        objective_key: Exact name of the feature a recommender optimizes — a
            pinned output name when ``outputs`` is used; validated against the
            recipe's features at ``setup()``.
        frame_deadline_s: Per-frame latency budget. A slower frame is flagged
            (``deadline_missed``), not interrupted.
        breach_patience: Consecutive breaching frames (failed / below the gate
            / drifting) before ``needs_escalation`` is raised. 1 escalates on
            the first; the default tolerates a single glitch frame.
        range_widen: A gated feature is ``out_of_reference_range`` when it
            leaves the running [min, max] of accepted frames widened by this
            many spans (the hyperspectral replay gate's rule). ``None``
            disables the gate.
        range_warmup: Clean frames the range gate only LEARNS from before it
            judges anything. One reference value gives no scale — observed
            live, a gate armed on the reference alone flagged every frame on a
            nuisance parameter (a pseudo-Voigt mixing fraction, 0.035 vs
            0.028) and, never having learned, then flagged the real drift too.
        range_adopt_after: Consecutive frames that are out of range while the
            fit gate is good and no drift is suspected before the new level is
            adopted: the data says the value is real, so the range moves to it
            (logged as ``range_adopted``) instead of flagging forever.
        noise_gate_tolerance: The fit and drift gates are calibrated on the
            reference run, in units of its own noise (see ``live/gates.py``): a
            frame below the agent's R² bar is still accepted when its residual
            scatter is within this factor of the reference's, and the drift bar
            drops to what the reference's fingerprint does under its own noise.
            Calibration only ever relaxes the constant bars. ``None`` keeps the
            constant bars (R² 0.95, similarity 0.92) — right only for clean data.
            The default rests on limited evidence: one locked recipe replayed
            over 225 real low-loss EELS spectra (untouched frames stayed within
            1.1× the reference; at 1.25 an injected feature 60 % / 30 % / 15 % of
            the peak height was caught 100 % / 47 % / 2 % of the time, at 1.5
            only 80 % / 29 % / 2 %) plus the four simulators (≤ 1.1×). LOWER it
            to catch weaker changes, RAISE it if clean frames are flagged
            ``gate_poor``; re-derive it when other real data are available.
        check_portability: At ``setup()``, replay the recipe on the reference
            with its signal scaled ×3 and ×0.35 and record whether the fit
            quality survives — a recipe with a bound read off the reference does
            not travel to stronger or weaker frames. A third replay moves the x
            axis by 3 % of its span and reports, without judging, whether the
            recipe follows its features or fixes their positions. Three zero-LLM
            replays in all.
        gate_keys: Features the range gate watches. Default: the pinned
            ``outputs`` when declared — the quantities the user named, not the
            recipe's nuisance parameters (observed live: an ill-determined
            pseudo-Voigt mixing fraction wandering over ten decades flagged
            clean XRD frames); else ``objective_key`` when set; else every
            feature that is not a fit uncertainty (``*_err`` and the like never
            are — they scatter by construction).
            With a ``recommender`` attached and no explicit ``gate_keys`` the
            range gate is OFF: it assumes fixed acquisition conditions, and a
            recommender's job is to change them.
        reanchor_frames: How many of the most recent frames a re-anchor is
            planned from. With several, the plan sees the change happening — what
            fades, what grows — instead of one snapshot of it. Observed live
            (XRD through a phase transition): a recipe rebuilt from one
            mid-transition frame was stale when adopted and a second rebuild
            followed at once. The recipe is locked on the newest frame; the bank
            is still asked about that frame first. 1 = the breaching frame only.
        auto_escalate: Start a background re-anchor by itself the moment
            ``needs_escalation`` is raised (on the frame that raised it). Off
            by default — a re-anchor calls a model, and whether the loop may
            is the caller's decision.
        escalation_profile: Depth of a re-anchor (default ``extract``: bank
            first, then edit-adapt, then fresh code, two verification passes).
        recommender: What proposes the next measurement — see
            :mod:`scilink.live.recommend`. BO is one option among several: a
            rule table, a surrogate optimizer, or a language model writing the
            next acquisition parameters as JSON (or a revised protocol). It
            sees every frame that produced features, WITH its flags — a noisy
            or drifting frame is often exactly what a recommendation is for
            (observed live: hiding flagged frames starved a GP exploring focus,
            where defocus reads as drift, and an LLM asked to fix low SNR,
            where every frame is below the fit gate). It can never fail a
            frame. A ``fast``-clock
            recommender runs inside ``step()``; a ``slow`` one (an LLM) runs
            off the fast path and ``step()`` returns its most recent finished
            recommendation, stamped with the step it was based on.
        schema: The instrument controller's acquisition parameters
            (:class:`~scilink.live.recommend.InstrumentSchema`). The LOOP
            validates every recommendation against it — an unknown name or an
            out-of-bounds value is refused, never clamped. The caller owns the
            instrument and its safe limits.
        closed_loop: ``False`` (default, advisory): every recommendation is
            marked ``requires_approval``. ``True``: valid parameter
            recommendations are not. A protocol always requires approval, and
            nothing is ever actuated by SciLink either way.
        agent_factory: ``f(output_dir) -> agent`` (tests; defaults to the
            curve-fitting agent).
    """

    def __init__(self, output_dir: str, *,
                 model_name: str = "claude-opus-4-6",
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 system_info: Any = None,
                 targets: Optional[List[str]] = None,
                 outputs: Optional[Dict[str, str]] = None,
                 objective_key: Optional[str] = None,
                 frame_deadline_s: Optional[float] = None,
                 breach_patience: int = 2,
                 range_widen: Optional[float] = 1.0,
                 range_warmup: int = 5,
                 range_adopt_after: int = 3,
                 gate_keys: Optional[List[str]] = None,
                 noise_gate_tolerance: Optional[float] = 1.25,
                 check_portability: bool = True,
                 auto_escalate: bool = False,
                 reanchor_frames: int = 5,
                 escalation_profile: str = "extract",
                 escalation_runner: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 recommender: Any = None,
                 schema: Any = None,
                 closed_loop: bool = False,
                 agent_factory: Optional[Callable[[str], Any]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name, self.api_key, self.base_url = model_name, api_key, base_url
        self.system_info = system_info
        self.targets = [str(t) for t in (targets or []) if str(t).strip()]
        self.outputs = {str(k): str(v) for k, v in (outputs or {}).items()}
        self.objective_key = objective_key
        self.frame_deadline_s = frame_deadline_s
        self.breach_patience = max(1, int(breach_patience))
        self.range_widen = range_widen
        self.range_warmup = max(0, int(range_warmup))
        self.range_adopt_after = max(1, int(range_adopt_after))
        self.gate_keys = list(gate_keys) if gate_keys else None
        self.noise_gate_tolerance = noise_gate_tolerance
        self.check_portability = check_portability
        self._calibration: Dict[str, Any] = {}
        self._n_learned = 0
        self._out_of_range_streak: List[Dict[str, float]] = []
        self.auto_escalate = bool(auto_escalate)
        self.reanchor_frames = max(1, int(reanchor_frames or 1))
        self._recent_frames: List[str] = []
        # The drift signal compares each frame with ONE reference frame. After a
        # window-planned re-anchor that frame is stale by construction (see
        # ``_rebase_drift``); the loop then holds the reference itself.
        self._drift_reference: Optional[Dict[str, Any]] = None
        self._rebase: Optional[List[Dict[str, Any]]] = None
        self.escalation_profile = escalation_profile
        self._escalation_runner = escalation_runner
        self._escalation: Any = None
        self._escalation_meta: Dict[str, Any] = {}
        self._n_escalations = 0
        self.recommender = recommender
        self.schema = schema
        self.closed_loop = bool(closed_loop)
        self._n_observed = 0
        self._last_recommendation: Optional[Dict[str, Any]] = None
        from .recommend import SlowSlot
        self._slot = SlowSlot(logger=logger)
        self._agent_factory = agent_factory or self._default_agent
        self.logger = logger or logging.getLogger("MeasurementLoop")

        self.anchor_dir: Optional[Path] = None
        self.recipe: Optional[Dict[str, Any]] = None
        self._edits: List[Dict[str, Any]] = []
        self._step = 0
        self._consecutive_breaches = 0
        self._ranges: Dict[str, List[float]] = {}
        self._reference_features: Dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    def _default_agent(self, output_dir: str):
        from ..agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        return CurveFittingAgent(api_key=self.api_key, model_name=self.model_name,
                                 base_url=self.base_url, output_dir=output_dir,
                                 enable_human_feedback=self._human_feedback)

    _human_feedback = False

    def setup(self, reference: Optional[Any] = None, *,
              anchor: Optional[str] = None,
              reference_data: Optional[str] = None,
              profile: Optional[str] = None,
              script_edits: Optional[List[Dict[str, Any]]] = None,
              enable_human_feedback: bool = False) -> Dict[str, Any]:
        """Lock the recipe. The slow clock — this may call a model; ``step()``
        never does.

        Exactly one source:

        - ``anchor``: the directory of an existing curve-fit run (typically a
          thorough analysis of reference data done earlier, in chat or
          standalone). Adopted as-is: no model call.
        - ``reference``: a data file. It is analysed now under ``profile``
          (default thorough; ``extract`` makes it bank-first, so a known
          system arms in seconds) and that run becomes the anchor.
          ``enable_human_feedback`` keeps the analysis's plan / result gates
          for a caller attached to a person.
          A LIST of files (the first frames of the stream, in order) is
          analysed as a series instead: the plan is made with all of them in
          view — what moves, what appears, what is noise — which a single
          frame cannot show, and the recipe is locked on the LAST of them, the
          state the stream continues from. Trend analysis and synthesis are
          skipped; the series is a means here, not a result.

        ``script_edits`` (exact old/new snippet pairs) are applied to the
        anchor's script on every frame — tomorrow's loop from yesterday's
        recipe with one knob changed.

        With pinned ``outputs`` the recipe is extended here to report them and
        checked on the reference data; an adopted ``anchor`` then needs
        ``reference_data`` — the file that run analysed — to be checked on.
        """
        if bool(reference) == bool(anchor):
            raise ValueError("setup() takes exactly one of `reference` (a data "
                             "file to analyse now) or `anchor` (a prior run directory).")
        t0 = time.perf_counter()
        source, ref_result = "anchor", None
        refs = ([str(reference)] if isinstance(reference, (str, Path))
                else [str(r) for r in (reference or [])])
        series_info: Optional[Dict[str, Any]] = None
        if refs:
            self._human_feedback = bool(enable_human_feedback)
            run_dir = self.output_dir / "reference"
            agent = self._agent_factory(str(run_dir))
            # The plan is told the recipe will be replayed on frames it has not
            # seen; the structural half of that rule is the portability check.
            kwargs: Dict[str, Any] = {"system_info": self.system_info, "stream_reference": True}
            if profile:
                kwargs["profile"] = profile
            if self.targets:
                kwargs["targets"] = self.targets
            if len(refs) > 1:
                kwargs["series_metadata"] = {"variable": "frame", "values": list(range(len(refs)))}
                # The series is a means here: its plan, made with every frame
                # in view, is what is wanted. No trend, no synthesis, and no
                # per-frame re-analysis — observed on real EELS frames, the R²
                # bar flagged 7 of 8 noise-limited frames and each was
                # re-analysed by a model for about two minutes.
                kwargs["profile"] = {"base": profile or "thorough", "trend": False,
                                     "synthesis": "none", "adaptive_refit": False}
            ref_result = agent.analyze(refs if len(refs) > 1 else refs[0], **kwargs)
            self._human_feedback = False
            if (ref_result or {}).get("status") != "success":
                raise RuntimeError(
                    "setup(): the reference analysis did not succeed — "
                    f"{(ref_result or {}).get('error')}")
            anchor = (ref_result.get("output_directory") or str(run_dir))
            source = ("bank" if ref_result.get("cold_start")
                      else f"reference:{profile or 'thorough'}")
            if len(refs) > 1:
                anchor, series_info = self._single_frame_anchor(
                    Path(anchor), self.output_dir / "reference_anchor", refs)
                series_info["llm_calls"] = (ref_result.get("stage_timings") or {}).get("llm_calls")
                reference, ref_result = series_info["data"], None
                source += f":{len(refs)} frames"
            else:
                reference = refs[0]

        script, anchor_dir = self._anchor_script(anchor)
        if script is None:
            raise ValueError(
                f"setup(): {anchor} holds no reusable curve-fit run (expected "
                "series_fit_results.json and a saved script under scripts/).")
        self.anchor_dir = anchor_dir
        self._edits = []
        if script_edits:
            self._validate_edits(script, list(script_edits))
            self._edits = list(script_edits)
        self.recipe = self._recipe_record(script, source)
        self._calibrate()

        if ref_result is not None:
            self._reference_features = _numeric_features(ref_result)
        else:
            self._reference_features = self._features_from_anchor(anchor_dir)
        pinned = None
        if self.outputs:
            ref_data = reference or reference_data
            if not ref_data:
                raise ValueError(
                    "pinned `outputs` are checked on the reference data: pass "
                    "reference_data=<the file the anchor run analysed>.")
            pinned = self._pin(script, str(ref_data), self.output_dir / "pinning")
            self.recipe = self._recipe_record(script, source)
        if self.objective_key and self.objective_key not in self._reference_features:
            raise ValueError(
                f"objective_key {self.objective_key!r} is not a feature this recipe "
                f"produces. Available: {sorted(self._reference_features)}")
        self._ranges = {k: [v, v] for k, v in self._reference_features.items()}
        self._n_learned, self._out_of_range_streak = 0, []
        portability = self._check_portability(reference or reference_data)

        record = {
            "event": "setup", "recipe": self.recipe,
            "anchor_dir": str(self.anchor_dir), "source": source,
            "seconds": round(time.perf_counter() - t0, 3),
            "reference_features": self._reference_features,
            "gate_calibration": self._calibration,
            **({"portability": portability} if portability else {}),
            **({"reference_frames": {k: v for k, v in series_info.items() if k != "data"}}
               if series_info else {}),
            "targets": self.targets, "objective_key": self.objective_key,
            "frame_deadline_s": self.frame_deadline_s,
        }
        if pinned is not None:
            record["pinned_outputs"] = {"definitions": self.outputs,
                                        "rationale": pinned["rationale"],
                                        "attempts": pinned["attempts"]}
        if ref_result is not None:
            st = ref_result.get("stage_timings") or {}
            record["llm_calls"] = st.get("llm_calls")
        elif series_info:
            record["llm_calls"] = series_info.get("llm_calls")
        self._append(record)
        self._save_state()
        self.logger.info(
            f"🔒 Loop armed: recipe {self.recipe['id']} from {source} "
            f"({record['seconds']:.1f}s); {len(self._reference_features)} features.")
        return record

    def _pin(self, script: str, reference_data: str, work_dir: Path) -> Dict[str, Any]:
        """Extend the locked recipe to report the pinned outputs (one model
        call, slow clock), verified on the reference data. The pin edits join
        the recipe's edit list after the caller's own."""
        from .pinning import agent_replay, pin_outputs
        model = getattr(self._agent_factory(str(work_dir / "model")), "model", None)
        if model is None:
            raise RuntimeError("pinning needs a model: the agent factory's agent has none")
        result = pin_outputs(
            script=(self._validate_edits(script, self._edits) if self._edits else script),
            outputs=self.outputs, model=model, logger=self.logger,
            replay=agent_replay(self._agent_factory, str(self.anchor_dir), reference_data,
                                self.system_info, str(work_dir), base_edits=self._edits))
        self._edits = list(self._edits) + result["edits"]
        self._reference_features = result["features"]
        return result

    @staticmethod
    def _anchor_script(anchor: str):
        from ..agents.exp_agents.controllers.curve_fitting_controllers import (
            _load_prior_curve_fit_state)
        anchor_dir, _summary, script, _label = _load_prior_curve_fit_state(anchor)
        return (script, anchor_dir) if anchor_dir is not None and script else (None, None)

    @staticmethod
    def _single_frame_anchor(series_dir: Path, dest: Path, files: List[str]):
        """A series run's LAST successful frame, laid out as a single-spectrum
        anchor (its own script, results and arrays) — so replay, pinning, gate
        calibration and the portability check all read it like any reference.
        A series keeps one script per spectrum and, with regimes, more than one
        model; the stream continues from the last frame, so that is the recipe
        to lock. Returns ``(anchor_dir, info)``."""
        data = json.loads((series_dir / "series_fit_results.json").read_text())
        ok = [r for r in (data.get("results") or [])
              if isinstance(r, dict) and r.get("success") and r.get("script")]
        if not ok:
            raise RuntimeError("setup(): no frame of the reference series was fitted successfully")
        last = ok[-1]
        idx = int(last.get("index", len(files) - 1))
        (dest / "scripts").mkdir(parents=True, exist_ok=True)
        (dest / "scripts" / "fitting_script.py").write_text(last["script"], encoding="utf-8")
        config = dict(data.get("locked_config") or {})
        if last.get("model_type"):
            config["physical_model"] = last["model_type"]     # the last regime's, when there were several
        regimes = ((data.get("series_analysis_plan") or {}).get("regimes") or [])
        single = {**data, "total_spectra": 1, "successful": 1, "is_single_spectrum": True,
                  "locked_config": config, "series_analysis_plan": None,
                  "results": [{**last, "index": 0}],
                  "derived_from": {"series_dir": str(series_dir), "index": idx}}
        (dest / "series_fit_results.json").write_text(json.dumps(single, indent=1, default=str))
        (dest / "analysis_results.json").write_text(json.dumps({
            "status": "success", "model_type": last.get("model_type"),
            "fitting_parameters": last.get("parameters") or {},
            "fit_quality": last.get("fit_quality") or {}}, indent=1, default=str))
        arrays = series_dir / f"spectrum_{idx:04d}"
        if arrays.is_dir():
            shutil.copytree(arrays, dest / "spectrum_0000", dirs_exist_ok=True)
        info = {"n": len(files), "fitted": len(ok), "anchored_on": idx,
                "regimes": len(regimes) or 1, "model": str(last.get("model_type") or "")[:200],
                "data": files[min(idx, len(files) - 1)]}
        return str(dest), info

    @staticmethod
    def _features_from_anchor(anchor_dir: Path) -> Dict[str, float]:
        try:
            rj = json.loads((anchor_dir / "analysis_results.json").read_text())
            return _numeric_features(rj)
        except Exception:  # noqa: BLE001 - an adopted run may predate the file
            return {}

    @staticmethod
    def _validate_edits(script: str, edits: List[Dict[str, Any]]) -> str:
        from ..utils.file_edit import apply_snippet_edits
        res = apply_snippet_edits(script, edits)
        if res["status"] != "success":
            raise ValueError(f"script_edits do not apply to the locked recipe: "
                             f"{res['message']}")
        return res["text"]

    def _recipe_record(self, script: str, source: str) -> Dict[str, Any]:
        effective = self._validate_edits(script, self._edits) if self._edits else script
        digest = hashlib.sha1(effective.encode("utf-8")).hexdigest()[:12]
        rec = {"id": digest, "source": source, "n_edits": len(self._edits)}
        if self.outputs:
            rec["pinned_outputs"] = sorted(self.outputs)
        return rec

    # ------------------------------------------------------------------- step
    def step(self, data_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyse one frame with the locked recipe. The fast clock: no model
        call on the happy path, never raises for a bad frame — it flags it."""
        if self.recipe is None or self.anchor_dir is None:
            raise LoopNotReady("call setup() before step()")
        # Between frames is the only moment a finished re-anchor is adopted.
        self._poll_escalation()
        self._step += 1
        idx = self._step
        t0 = time.perf_counter()
        frame_dir = self.output_dir / "frames" / f"frame_{idx:06d}"
        flags: List[str] = []
        result: Dict[str, Any] = {}
        error = None
        try:
            agent = self._agent_factory(str(frame_dir))
            kwargs: Dict[str, Any] = dict(
                system_info=self.system_info,
                prior_analysis_paths=[str(self.anchor_dir)],
                reuse_locked_script=True, profile="realtime",
                # The fast clock never calls a model: a recipe that cannot run
                # on this frame fails the frame (flagged, escalated off-path)
                # instead of being repaired or re-derived in-frame. Observed
                # live: without this, a three-peak recipe meeting two-peak
                # data cost three frames ~40 s and an LLM call each.
                strict_replay=True)
            if self._edits:
                kwargs["script_edits"] = list(self._edits)
            result = agent.analyze(data_path, **kwargs) or {}
        except Exception as e:  # noqa: BLE001 - one frame must not kill the loop
            error = f"{type(e).__name__}: {e}"
            self.logger.exception(f"frame {idx} raised")
        latency = time.perf_counter() - t0

        features = _numeric_features(result) if result.get("status") == "success" else {}
        validity = result.get("reuse_validity") or {}
        gate_extra: Dict[str, Any] = {}
        if error or result.get("status") != "success" or not features:
            flags.append(FLAG_FIT_FAILED)
            if not error and result.get("error"):
                error = json.dumps(result.get("error"), default=str)[:300]
        else:
            gate_extra = self._judge(validity, str(frame_dir), str(data_path))
            if gate_extra.pop("poor"):
                flags.append(FLAG_GATE_POOR)
            if gate_extra.pop("drift"):
                flags.append(FLAG_DRIFT)
        llm_calls = (result.get("stage_timings") or {}).get("llm_calls")
        if llm_calls:
            flags.append(FLAG_LLM_USED)
        if self.frame_deadline_s and latency > self.frame_deadline_s:
            flags.append(FLAG_DEADLINE)
        # The range gate judges only a frame the fit gate and the drift signal
        # already accept — it is a plausibility check on good fits, not a
        # second opinion on bad ones.
        out_of_range = {} if flags else self._out_of_range(features)
        adopted = None
        if out_of_range:
            self._out_of_range_streak.append(features)
            if len(self._out_of_range_streak) >= self.range_adopt_after:
                # Good fit, no drift, and it keeps saying the same thing: the
                # value is real. Move the range to it rather than flag forever.
                for f in self._out_of_range_streak:
                    self._widen_ranges(f)
                adopted = sorted(out_of_range)
                self._out_of_range_streak, out_of_range = [], {}
            else:
                flags.append(FLAG_OUT_OF_RANGE)
        elif not flags:
            self._out_of_range_streak = []

        breach = any(f in _BREACH_FLAGS for f in flags)
        self._consecutive_breaches = self._consecutive_breaches + 1 if breach else 0
        needs_escalation = self._consecutive_breaches >= self.breach_patience
        clean = not flags or flags == [FLAG_DEADLINE]
        if clean:
            self._widen_ranges(features)
            self._n_learned += 1

        record: Dict[str, Any] = {
            "event": "frame", "step": idx, "data": str(data_path),
            "params": params or {}, "features": features,
            "gate": {**{k: validity.get(k) for k in
                        ("verdict", "r_squared", "threshold", "drift",
                         "fingerprint_similarity") if k in validity}, **gate_extra},
            "flags": flags, "needs_escalation": needs_escalation,
            "consecutive_breaches": self._consecutive_breaches,
            "latency_s": round(latency, 3), "llm_calls": llm_calls or 0,
            "recipe_id": self.recipe["id"], "frame_dir": str(frame_dir),
        }
        if out_of_range:
            record["out_of_range"] = out_of_range
        if adopted:
            record["range_adopted"] = adopted
        if error:
            record["error"] = error
        if self.objective_key:
            record["objective"] = features.get(self.objective_key)
        if FLAG_FIT_FAILED not in flags:       # a dead frame teaches a plan nothing
            self._recent_frames = (self._recent_frames + [str(data_path)])[-25:]
        record["recommendation"] = self._recommend(params, features, flags, idx)
        if self._escalation is not None:
            record["escalation"] = "running"
        self._append(record)
        self._save_state()
        if needs_escalation and self.auto_escalate and self._escalation is None:
            try:
                self.escalate(data_path)
                record["escalation"] = "started"
            except Exception as e:  # noqa: BLE001 - never fails the frame
                self.logger.warning(f"auto-escalation could not start: {e}")
        return record

    # -------------------------------------------------------------- escalation
    def escalate(self, data_path: str, *, profile: Optional[str] = None,
                 background: bool = True, frames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Re-anchor on ``data_path`` (normally the frame that breached). The
        slow clock: this may call a model, which is why ``step()`` never calls
        it unless ``auto_escalate`` was asked for.

        The plan is made from a window of recent frames ending in ``data_path``
        — ``frames`` when given, else the last ``reanchor_frames`` the loop has
        seen — and the recipe is locked on ``data_path`` itself.

        In the background (default) the re-anchor runs in a spawned process
        and ``step()`` keeps answering with the old recipe, flagged; the new
        recipe is adopted at the start of the first ``step()`` after it
        finishes. ``background=False`` runs it here and adopts it before
        returning. One escalation at a time.
        """
        if self.recipe is None:
            raise LoopNotReady("call setup() before escalate()")
        if self._escalation is not None:
            raise RuntimeError("an escalation is already running")
        self._n_escalations += 1
        out_dir = self.output_dir / "escalations" / f"escalation_{self._n_escalations:03d}"
        analyze_kwargs: Dict[str, Any] = {"system_info": self.system_info,
                                          "profile": profile or self.escalation_profile,
                                          "stream_reference": True}
        if self.targets:
            analyze_kwargs["targets"] = self.targets
        # Never audition the recipe that just breached: that is circular.
        try:
            from ..skills._shared._script_bank import script_hash
            script, _ = self._anchor_script(str(self.anchor_dir))
            if script:
                effective = self._validate_edits(script, self._edits) if self._edits else script
                analyze_kwargs["bank_exclude"] = sorted({script_hash(script),
                                                         script_hash(effective)})
        except Exception:  # noqa: BLE001 - exclusion is a safeguard, not a requirement
            pass
        window = [str(f) for f in (frames if frames is not None
                                   else self._recent_frames[-self.reanchor_frames:])]
        window = [f for f in window if f != str(data_path) and Path(f).is_file()]
        window = window + [str(data_path)]
        if frames is None:
            window = window[-self.reanchor_frames:]
        spec = {
            "out_dir": str(out_dir),
            "data_path": window if len(window) > 1 else str(data_path),
            # In memory only — a spec is never written to disk (it may carry
            # a credential).
            "agent_kwargs": {"api_key": self.api_key, "model_name": self.model_name,
                             "base_url": self.base_url},
            "analyze_kwargs": analyze_kwargs,
            "sandbox_approved": self._sandbox_approved(),
            # A new recipe must report the SAME pinned names; the worker pins
            # it (a model call — slow clock) on the frame it re-anchored on.
            "pin_outputs": dict(self.outputs) or None,
            "system_info": self.system_info,
        }
        self._escalation_meta = {"index": self._n_escalations, "data": str(data_path),
                                 "window": len(window),
                                 "profile": analyze_kwargs["profile"],
                                 "from_recipe": self.recipe["id"],
                                 "started_step": self._step, "t0": time.perf_counter()}
        self._append({"event": "escalation_started", "step": self._step,
                      **{k: v for k, v in self._escalation_meta.items() if k != "t0"}})
        runner = self._escalation_runner or (
            _ProcessEscalation if background else _InlineEscalation)
        self._escalation = runner(spec)
        if not background:
            return self._poll_escalation() or {"event": "escalation_started"}
        return {"event": "escalation_started", **{
            k: v for k, v in self._escalation_meta.items() if k != "t0"}}

    @staticmethod
    def _sandbox_approved() -> bool:
        import os
        try:
            from .. import executors
            if getattr(executors, "_GLOBAL_SANDBOX_APPROVED", False):
                return True
        except Exception:  # noqa: BLE001
            pass
        return os.environ.get("UNSAFE_EXECUTION_OK", "").strip().lower() in ("1", "true", "yes")

    @property
    def escalating(self) -> bool:
        return self._escalation is not None

    def _poll_escalation(self) -> Optional[Dict[str, Any]]:
        """Adopt a finished re-anchor (or record its failure). Never blocks."""
        if self._escalation is None:
            return None
        try:
            result = self._escalation.poll()
        except Exception as e:  # noqa: BLE001
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        if result is None:
            return None
        meta, self._escalation = self._escalation_meta, None
        base = {"step": self._step, "index": meta.get("index"),
                "from_recipe": meta.get("from_recipe"),
                "frames_answered_meanwhile": self._step - int(meta.get("started_step") or 0),
                "seconds": result.get("seconds"), "llm_calls": result.get("llm_calls") or 0,
                **({"window": result["window"]} if result.get("window") else {})}
        script, anchor_dir = (None, None)
        if result.get("status") == "success" and result.get("output_directory"):
            script, anchor_dir = self._anchor_script(result["output_directory"])
        if script is None:
            record = {"event": "escalation_failed", **base,
                      "error": str(result.get("error") or "the re-anchor produced no reusable run")[:300]}
            self._append(record)
            self._save_state()
            return record
        if self.outputs and not result.get("pin_edits"):
            record = {"event": "escalation_failed", **base,
                      "error": ("the new recipe could not be extended to report the pinned "
                                f"outputs: {result.get('pin_error')}")[:300]}
            self._append(record)
            self._save_state()
            return record
        # Adopt. Amendments were written against the OLD script and do not
        # carry; the plausible ranges belong to the old regime and re-learn.
        # The pin edits were produced for THIS script by the worker.
        self.anchor_dir, self._edits = anchor_dir, list(result.get("pin_edits") or [])
        source = "bank" if result.get("cold_start") else f"reanchor:{meta.get('profile')}"
        self.recipe = self._recipe_record(script, source)
        self._calibrate()
        self._drift_reference = None
        self._rebase = [] if result.get("window") else None
        self._reference_features = (result.get("pin_features")
                                    or self._features_from_anchor(anchor_dir))
        self._ranges = {k: [v, v] for k, v in self._reference_features.items()}
        self._n_learned, self._out_of_range_streak = 0, []
        self._consecutive_breaches = 0
        if self.objective_key and self.objective_key not in self._reference_features:
            self.logger.warning(
                f"the new recipe does not produce objective_key {self.objective_key!r}; "
                f"available: {sorted(self._reference_features)}")
        record = {"event": "reanchor", **base, "recipe": self.recipe, "source": source,
                  "anchor_dir": str(anchor_dir), "gate_calibration": self._calibration,
                  "objective_key_present": (self.objective_key in self._reference_features
                                            if self.objective_key else None)}
        self._append(record)
        self._save_state()
        self.logger.info(f"🔁 Re-anchored: recipe {self.recipe['id']} from {source} "
                         f"({record['seconds']}s, {record['llm_calls']} LLM call(s)).")
        return record

    def _check_portability(self, reference_data: Optional[str]) -> Dict[str, Any]:
        """Replay the recipe on the reference with its signal scaled up and
        down (zero-LLM). Advisory: the verdict is recorded and logged."""
        if not self.check_portability or not reference_data or self.anchor_dir is None:
            return {}
        try:
            from .portability import (agent_replay_r2, check_portability, describe,
                                      describe_positions)
            work = self.output_dir / "portability"
            report = check_portability(
                agent_replay_r2(self._agent_factory, str(self.anchor_dir), self.system_info,
                                str(work), edits=self._edits),
                str(reference_data), self._reference_features.get("fit_r_squared"), str(work))
        except Exception as e:  # noqa: BLE001 - a check, never a dependency
            self.logger.warning(f"portability check skipped: {e}")
            return {}
        if report:
            report["summary"] = describe(report)
            report["positions_summary"] = describe_positions(report)
            (self.logger.info if report["portable"] else self.logger.warning)(
                f"   portability: {report['summary']} {report['positions_summary']}".rstrip())
        return report

    def _calibrate(self) -> None:
        """Ask the current anchor run what its noise does to the gates."""
        self._calibration = {}
        if self.noise_gate_tolerance is None or self.anchor_dir is None:
            return
        try:
            from .gates import calibrate
            self._calibration = calibrate(str(self.anchor_dir))
        except Exception as e:  # noqa: BLE001 - calibration is an improvement, never a dependency
            self.logger.warning(f"gate calibration skipped: {e}")
        if self._calibration:
            self.logger.info(f"   gates calibrated on the reference: {self._calibration}")

    @staticmethod
    def _fingerprint(data_path: str) -> Optional[Dict[str, Any]]:
        try:
            from ..skills._shared import _script_bank
            from .instruments import read_curve
            x, y, _, _ = read_curve(data_path)
            return _script_bank.curve_fingerprint(x, y)
        except Exception:  # noqa: BLE001 - no fingerprint, no opinion
            return None

    def _rebase_drift(self, poor: bool, drift: bool, data_path: str,
                      floor: float) -> Optional[bool]:
        """After a window-planned re-anchor: is the drift reference just stale?

        The recipe was planned across a change and locked on the window's last
        frame; by the time it is adopted the stream has moved further along the
        same change. Observed live (XRD through a phase transition): the new
        recipe fitted every frame at R² 0.99 and the loop still rebuilt it at
        once, because pure phase B no longer looked like the mid-transition
        anchor frame. So: if the first frames under the new recipe all pass the
        fit gate and look like EACH OTHER, the stream is stable and the
        reference moves to them. A poor fit, or a frame that matches the anchor
        after all, cancels it and the ordinary rules apply. Returns the frame's
        drift verdict when it took the decision, else None."""
        if self._rebase is None:
            return None
        if poor or not drift:
            self._rebase = None
            return None
        fp = self._fingerprint(data_path)
        if fp is None:
            self._rebase = None
            return None
        from ..skills._shared import _script_bank
        if any(_script_bank._curve_similarity(fp, other) < floor for other in self._rebase):
            self._rebase = None                  # still moving: not a stable new state
            return None
        self._rebase.append(fp)
        if len(self._rebase) < max(1, self.breach_patience - 1):
            return None                          # flagged for now; not yet a breach run
        self._drift_reference, self._rebase = self._rebase[0], None
        self._append({"event": "drift_rebased", "step": self._step,
                      "recipe_id": self.recipe["id"],
                      "why": "first frames under the new recipe fit and agree with each other"})
        return False

    def _judge(self, validity: Dict[str, Any], frame_dir: str,
               data_path: Optional[str] = None) -> Dict[str, Any]:
        """The frame's fit and drift verdicts, with the agent's constant bars
        relaxed to what the reference run achieved under its own noise. Returns
        ``poor`` / ``drift`` plus whatever it measured, for the record."""
        poor = validity.get("verdict") not in (None, "good")
        drift = validity.get("drift") == "suspected"
        out: Dict[str, Any] = {}
        cal = self._calibration
        if cal and poor and cal.get("residual_excess"):
            from .gates import residual_excess
            excess = residual_excess(frame_dir)
            if excess is not None:
                out["residual_excess"] = round(excess, 3)
                out["reference_excess"] = cal["residual_excess"]
                if excess <= self.noise_gate_tolerance * cal["residual_excess"]:
                    poor, out["accepted_in_noise_units"] = False, True
        sim = validity.get("fingerprint_similarity")
        floor = float(cal.get("drift_floor") if cal.get("drift_floor") is not None else 0.92)
        if self._drift_reference is not None and data_path:
            fp = self._fingerprint(data_path)     # the loop holds the reference (see _rebase_drift)
            if fp is not None:
                from ..skills._shared import _script_bank
                sim = round(float(_script_bank._curve_similarity(fp, self._drift_reference)), 3)
                out["fingerprint_similarity_rebased"] = sim
                drift = sim < floor
        if cal and drift and cal.get("drift_floor") is not None and sim is not None:
            out["drift_floor"] = cal["drift_floor"]
            if float(sim) >= cal["drift_floor"]:
                drift = False
        if data_path:
            decided = self._rebase_drift(poor, drift, data_path, floor)
            if decided is not None:
                drift = decided
        return {"poor": poor, "drift": drift, **out}

    _UNCERTAINTY_SUFFIXES = ("_err", "_error", "_stderr", "_std", "_unc", "_uncertainty", "_sigma_err")

    def _gated(self, key: str) -> bool:
        if self.gate_keys is not None:
            return key in self.gate_keys
        if self.recommender is not None:
            return False        # conditions are being changed on purpose
        if self.outputs:
            return key in self.outputs     # what the user asked for, by name
        if self.objective_key:
            return key == self.objective_key
        return not (key.startswith("fit_") or key.endswith(self._UNCERTAINTY_SUFFIXES))

    def _out_of_range(self, features: Dict[str, float]) -> Dict[str, Any]:
        if (self.range_widen is None or not features
                or self._n_learned < self.range_warmup):
            return {}
        out = {}
        for k, v in features.items():
            if not self._gated(k) or k not in self._ranges:
                continue
            lo, hi = self._ranges[k]
            span = max(hi - lo, 0.05 * max(abs(lo), abs(hi), 1e-12))
            if v < lo - self.range_widen * span or v > hi + self.range_widen * span:
                out[k] = {"value": v, "range": [lo, hi]}
        return out

    def _widen_ranges(self, features: Dict[str, float]) -> None:
        for k, v in features.items():
            lo, hi = self._ranges.get(k, [v, v])
            self._ranges[k] = [min(lo, v), max(hi, v)]

    def _recommend(self, params, features, flags: List[str], idx: int):
        """The recommendation this frame's record carries. Never raises, never
        waits: a slow recommender's call runs off the fast path."""
        rec = self.recommender
        if rec is None:
            return None
        from .recommend import finalize
        source = getattr(rec, "name", rec.__class__.__name__)
        if getattr(rec, "context", True) is None and self.system_info:
            rec.context = dict(self.system_info)   # a model must know what is being measured
        try:
            observed = bool(features) and params is not None
            if observed:
                try:
                    rec.observe(params, features, step=idx, flags=list(flags))
                except TypeError:              # a duck-typed observe(params, features)
                    rec.observe(params, features)
                self._n_observed += 1
            if getattr(rec, "clock", "fast") != "slow":
                return finalize(rec.suggest(), self.schema, source=source,
                                based_on_step=idx, closed_loop=self.closed_loop)
            # Slow clock: collect a finished call, maybe start the next one.
            done = self._slot.take()
            if done is not None:
                self._last_recommendation = self._adopt_recommendation(
                    finalize(done["raw"], self.schema, source=source,
                             based_on_step=done["based_on_step"],
                             closed_loop=self.closed_loop))
            every = max(1, int(getattr(rec, "every", 1)))
            if observed and self._n_observed % every == 0:
                self._slot.start(rec, idx)
            if self._last_recommendation is None:
                return {"pending": True} if self._slot.busy else None
            return {**self._last_recommendation, "pending": self._slot.busy}
        except Exception as e:  # noqa: BLE001 - a recommender never fails a frame
            self.logger.warning(f"recommender failed: {e}")
            return finalize({"params": None, "problems": [f"{type(e).__name__}: {e}"]},
                            self.schema, source=source, based_on_step=idx,
                            closed_loop=self.closed_loop)

    def _adopt_recommendation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Log a slow recommender's result; write a protocol out as a file —
        an artifact for a person, never something SciLink runs."""
        if rec.get("kind") == "protocol" and rec.get("protocol"):
            d = self.output_dir / "protocols"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"protocol_step_{int(rec['based_on_step']):06d}.txt"
            path.write_text(str(rec["protocol"]), encoding="utf-8")
            rec = {**rec, "protocol_path": str(path)}
        self._append({"event": "recommendation", "step": self._step, **rec})
        return rec

    # ------------------------------------------------------------------ amend
    def amend(self, edits: List[Dict[str, Any]], note: Optional[str] = None) -> Dict[str, Any]:
        """Change a knob of the locked recipe without re-anchoring.

        ``edits`` are exact old/new snippet pairs against the recipe AS IT RUNS
        NOW (the anchor's script with earlier amendments applied). They are
        validated here, atomically — a list that does not apply changes
        nothing. The recipe id changes, so the log shows where the method did.
        """
        if self.recipe is None or self.anchor_dir is None:
            raise LoopNotReady("call setup() before amend()")
        script, _ = self._anchor_script(str(self.anchor_dir))
        current = self._validate_edits(script, self._edits) if self._edits else script
        self._validate_edits(current, list(edits))          # raises if they do not apply
        previous = self.recipe["id"]
        self._edits = self._edits + list(edits)
        self.recipe = self._recipe_record(script, self.recipe["source"])
        self._consecutive_breaches = 0
        record = {"event": "amend", "step": self._step, "previous_recipe_id": previous,
                  "recipe": self.recipe, "edits": list(edits), "note": note}
        self._append(record)
        self._save_state()
        return record

    # ----------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        frames = [r for r in self.read_log() if r.get("event") == "frame"]
        lat = sorted(r["latency_s"] for r in frames) or [0.0]
        counts: Dict[str, int] = {}
        for r in frames:
            for f in r.get("flags") or []:
                counts[f] = counts.get(f, 0) + 1
        return {
            "armed": self.recipe is not None,
            "recipe": self.recipe, "anchor_dir": str(self.anchor_dir) if self.anchor_dir else None,
            "frames": len(frames),
            "clean_frames": sum(1 for r in frames if not r.get("flags")),
            "flag_counts": counts,
            "llm_calls_in_frames": sum(int(r.get("llm_calls") or 0) for r in frames),
            "latency_s": {"median": lat[len(lat) // 2], "max": lat[-1]},
            "needs_escalation": bool(frames and frames[-1].get("needs_escalation")),
            "escalating": self.escalating,
            "reanchors": sum(1 for r in self.read_log() if r.get("event") == "reanchor"),
        }

    # ------------------------------------------------------------ persistence
    @property
    def log_path(self) -> Path:
        return self.output_dir / LOOP_LOG_NAME

    def read_log(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                     # a torn line from a killed run
        return out

    def _append(self, record: Dict[str, Any]) -> None:
        rec = {"v": SCHEMA_VERSION, "timestamp": _now(), **record}
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def _save_state(self) -> None:
        state = {
            "v": SCHEMA_VERSION, "anchor_dir": str(self.anchor_dir) if self.anchor_dir else None,
            "recipe": self.recipe, "edits": self._edits, "step": self._step,
            "consecutive_breaches": self._consecutive_breaches,
            "ranges": self._ranges, "reference_features": self._reference_features,
            "n_learned": self._n_learned, "range_warmup": self.range_warmup,
            "range_adopt_after": self.range_adopt_after, "gate_keys": self.gate_keys,
            "targets": self.targets, "outputs": self.outputs,
            "objective_key": self.objective_key,
            "frame_deadline_s": self.frame_deadline_s,
            "breach_patience": self.breach_patience, "range_widen": self.range_widen,
            "system_info": self.system_info,
            "gate_calibration": self._calibration,
            "noise_gate_tolerance": self.noise_gate_tolerance,
            "reanchor_frames": self.reanchor_frames, "recent_frames": self._recent_frames,
            "drift_reference": self._drift_reference, "drift_rebase": self._rebase,
        }
        tmp = self.output_dir / (LOOP_STATE_NAME + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.output_dir / LOOP_STATE_NAME)

    @classmethod
    def resume(cls, output_dir: str, **kwargs: Any) -> "MeasurementLoop":
        """Re-arm a loop from ``loop_state.json`` after a crash or restart —
        no model call; the step counter continues where the log stopped."""
        state = json.loads((Path(output_dir) / LOOP_STATE_NAME).read_text())
        for k in ("targets", "outputs", "objective_key", "frame_deadline_s", "breach_patience",
                  "range_widen", "range_warmup", "range_adopt_after", "gate_keys",
                  "system_info"):
            kwargs.setdefault(k, state.get(k))
        if "reanchor_frames" in state:
            kwargs.setdefault("reanchor_frames", state["reanchor_frames"])
        if "noise_gate_tolerance" in state:      # absent in older states: keep the default
            kwargs.setdefault("noise_gate_tolerance", state["noise_gate_tolerance"])
        loop = cls(output_dir, **kwargs)
        loop.anchor_dir = Path(state["anchor_dir"]) if state.get("anchor_dir") else None
        loop.recipe = state.get("recipe")
        loop._edits = state.get("edits") or []
        loop._step = int(state.get("step") or 0)
        loop._consecutive_breaches = int(state.get("consecutive_breaches") or 0)
        loop._ranges = {k: list(v) for k, v in (state.get("ranges") or {}).items()}
        loop._reference_features = state.get("reference_features") or {}
        loop._n_learned = int(state.get("n_learned") or 0)
        loop._calibration = state.get("gate_calibration") or {}
        loop._recent_frames = [str(f) for f in (state.get("recent_frames") or [])]
        loop._drift_reference = state.get("drift_reference")
        loop._rebase = state.get("drift_rebase")
        loop._append({"event": "resume", "step": loop._step, "recipe": loop.recipe})
        return loop
