"""What a live loop needs to know about the KIND of data it follows.

The loop itself is modality-neutral: two clocks, the flags, breach patience, the
change signal, novelty, audits, pausing, the recommender and the log do not care
whether a frame is a spectrum or a datacube. What differs is small and lives
here:

- which analysis agent locks the recipe and replays it, and with what arguments;
- how a result becomes the flat ``{name: number}`` features the loop tracks;
- what "the recipe still fits this frame" means (the agent's own verdict);
- which 1D curve stands for a frame in the change signal (``live/drift.py``
  works on any curve: a spectrum is itself, a datacube is its mean spectrum);
- what can be done to a locked recipe (pinned outputs, snippet edits, the
  portability check and a multi-frame reference are curve mechanisms today).

``CurveModality`` is the loop as it was. ``HyperspectralModality`` follows a
stream of datacubes: the reference cube is analysed in full once, its approved
per-pixel script(s) are replayed on every later cube with no model call
(``strict_replay``) and judged by the agent's deterministic replay gate against
the reference's own map statistics, and a rebuild or an audit re-derives the
code for the SAME targets and output names (``locked_targets``), so the tracked
quantities keep their names by construction and nothing needs pinning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CurveModality:
    """1D curves through ``CurveFittingAgent`` — the loop's original behaviour."""

    name = "curve"
    #: What the loop may do to a locked recipe of this kind.
    pinning = True
    script_edits = True
    portability = True
    series_reference = True
    window_reanchor = True
    #: Result statuses whose numbers may be read (the verdict still judges them).
    usable_status = ("success",)
    #: True where names are only ASKED for (not pinned, not fixed by construction):
    #: a rebuilt recipe that does not report a tracked output is refused.
    require_outputs_after_rebuild = False
    #: What the slow clock does by default when the data changed and the recipe
    #: still fits (the loop's ``on_change``).
    default_on_change = "report"
    #: True where one disagreeing audit must not win by itself.
    audit_needs_second_opinion = False

    # ------------------------------------------------------------------ agent
    def make_agent(self, loop: Any, output_dir: str) -> Any:
        from ..agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        return CurveFittingAgent(api_key=loop.api_key, model_name=loop.model_name,
                                 base_url=loop.base_url, output_dir=output_dir,
                                 enable_human_feedback=loop._human_feedback)

    def reference_kwargs(self, loop: Any, refs: List[str], profile: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(refs), "stream_reference": True}
        if profile:
            kwargs["profile"] = profile
        if loop.targets:
            kwargs["targets"] = loop.targets
        if len(refs) > 1:
            kwargs["series_metadata"] = {"variable": "frame", "values": list(range(len(refs)))}
            # The series is a means here: its plan, made with every frame
            # in view, is what is wanted. No trend, no synthesis, and no
            # per-frame re-analysis — observed on real EELS frames, the R²
            # bar flagged 7 of 8 noise-limited frames and each was
            # re-analysed by a model for about two minutes.
            kwargs["profile"] = {"base": profile or "thorough", "trend": False,
                                 "synthesis": "none", "adaptive_refit": False}
        return kwargs

    def replay_kwargs(self, loop: Any, data_path: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            system_info=loop._with_sidecar(data_path),
            prior_analysis_paths=[str(loop.anchor_dir)],
            reuse_locked_script=True, profile="realtime",
            # The fast clock never calls a model: a recipe that cannot run
            # on this frame fails the frame (flagged, escalated off-path)
            # instead of being repaired or re-derived in-frame. Observed
            # live: without this, a three-peak recipe meeting two-peak
            # data cost three frames ~40 s and an LLM call each.
            strict_replay=True)
        if loop._edits:
            kwargs["script_edits"] = list(loop._edits)
        return kwargs

    def escalation_kwargs(self, loop: Any, data_path: str, profile: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(data_path),
                                  "profile": profile, "stream_reference": True}
        if loop.targets:
            kwargs["targets"] = loop.targets
        return kwargs

    # --------------------------------------------------------------- results
    def features(self, result: Dict[str, Any]) -> Dict[str, float]:
        from .measurement_loop import _numeric_features
        return _numeric_features(result)

    def validity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result.get("reuse_validity") or {}

    # ---------------------------------------------------------------- anchor
    def anchor_script(self, anchor: str) -> Tuple[Optional[str], Optional[Path]]:
        from ..agents.exp_agents.controllers.curve_fitting_controllers import (
            _load_prior_curve_fit_state)
        anchor_dir, _summary, script, _label = _load_prior_curve_fit_state(anchor)
        return (script, anchor_dir) if anchor_dir is not None and script else (None, None)

    no_anchor_message = ("holds no reusable curve-fit run (expected series_fit_results.json "
                         "and a saved script under scripts/).")

    def why_nothing_to_lock(self, result: Dict[str, Any]) -> str:
        return "."

    def series_anchor(self, series_dir: Path, dest: Path, files: List[str]):
        """From a reference analysed as a SERIES (several first frames), the run to
        lock: ``(anchor_dir, info)``. A curve series keeps one script per
        spectrum, so its last fitted frame is laid out as a single-spectrum run."""
        from .measurement_loop import MeasurementLoop
        return MeasurementLoop._single_frame_anchor(series_dir, dest, files)

    def features_from_anchor(self, anchor_dir: Path) -> Dict[str, float]:
        try:
            return self.features(json.loads((Path(anchor_dir) / "analysis_results.json").read_text()))
        except Exception:  # noqa: BLE001 - an adopted run may predate the file
            return {}

    def anchor_state(self, anchor_dir: Path, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """What a replay needs from the anchor besides its script. Nothing, for curves."""
        return {}

    def bake_edits(self, anchor_dir: Path, edits: List[Dict[str, Any]], dest: Path) -> Optional[Path]:
        """Snippet edits to a locked recipe. A curve replay applies them on every
        frame (``script_edits``), so nothing is written: ``None``."""
        return None

    def check_portability(self, loop: Any, reference_data: str) -> Dict[str, Any]:
        """Does the locked recipe depend on the reference's signal level (and, for a
        curve, on its positions)? Replays only, zero model calls."""
        from .portability import (agent_replay_r2, check_portability, describe,
                                  describe_positions)
        work = loop.output_dir / "portability"
        report = check_portability(
            agent_replay_r2(loop._agent_factory, str(loop.anchor_dir), loop.system_info,
                            str(work), edits=loop._edits),
            str(reference_data), loop._reference_features.get("fit_r_squared"), str(work))
        if report:
            report["summary"] = describe(report)
            report["positions_summary"] = describe_positions(report)
        return report

    def calibrate(self, anchor_dir: Path) -> Dict[str, Any]:
        from .gates import calibrate
        return calibrate(str(anchor_dir))

    def residual_excess(self, frame_dir: str) -> Optional[float]:
        from .gates import residual_excess
        return residual_excess(frame_dir)

    # ---------------------------------------------------------------- signal
    def read_signal(self, data_path: str, system_info: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """The 1D curve that stands for this frame in the change signal."""
        from .instruments import read_curve
        x, y, _, _ = read_curve(str(data_path))
        return x, y

    def read_signals(self, data_path: str, system_info: Any = None) -> Dict[str, Tuple[Any, Any]]:
        """The named curves the change signal watches for this frame. One, for a curve."""
        return {"signal": self.read_signal(data_path, system_info)}

    def annotate_where(self, where: List[Dict[str, Any]], data_path: str,
                       system_info: Any = None) -> List[Dict[str, Any]]:
        """Say a located change in the terms a person uses for this kind of data.
        A spectrum's axis already is that; an image's is a spatial frequency."""
        return where


class HyperspectralModality(CurveModality):
    """Datacubes (H, W, channels) through ``HyperspectralAnalysisAgent``."""

    name = "hyperspectral"
    pinning = False            # names are fixed by locked_targets, by construction
    script_edits = True        # baked into a copy of the anchor (bake_edits), not applied per frame
    portability = True         # by outputs: each stays put or scales with the counts
    series_reference = True    # several first cubes: planned as a series, locked on the last one's regime
    window_reanchor = False
    usable_status = ("success", "partial")     # some maps passed the gate: tracked, and flagged poor

    def make_agent(self, loop: Any, output_dir: str) -> Any:
        from ..agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
        return HyperspectralAnalysisAgent(api_key=loop.api_key, model_name=loop.model_name,
                                          base_url=loop.base_url, output_dir=output_dir,
                                          enable_human_feedback=loop._human_feedback)

    @staticmethod
    def _objective(loop: Any) -> Optional[str]:
        """What the stream is followed FOR, in the analysis's own terms: the
        targets, and the named quantities as maps to report under those names."""
        parts = []
        if loop.targets:
            parts.append("Quantities of interest: " + "; ".join(str(t) for t in loop.targets) + ".")
        if loop.outputs:
            parts.append("Report each of these as a per-pixel map under exactly this name: "
                         + "; ".join(f"{k} ({v})" for k, v in loop.outputs.items()) + ".")
        if parts:
            parts.append("The same analysis will be repeated on later datacubes of the same "
                         "experiment, so it must not depend on values read off this one.")
        return " ".join(parts) or None

    def reference_kwargs(self, loop: Any, refs: List[str], profile: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(refs[-1])}
        if profile:
            kwargs["profile"] = profile
        objective = self._objective(loop)
        if objective:
            kwargs["objective"] = objective
        if len(refs) > 1:
            # Several first cubes are analysed as a series: every cube's mean
            # spectrum is scouted and the plan is made with the variation in
            # view (one noisy or unrepresentative cube does not decide the
            # method), the first cube of each regime is analysed in full and the
            # rest replay it. The series is a means: no trend, no synthesis, no
            # per-cube refits.
            kwargs["series_metadata"] = {"variable": "frame", "values": list(range(len(refs)))}
            kwargs["profile"] = {"base": profile or "thorough", "trend": False,
                                 "synthesis": "none", "adaptive_refit": False}
        return kwargs

    def series_anchor(self, series_dir: Path, dest: Path, files: List[str]):
        """The recipe of the regime the LAST reference cube belongs to: the state
        the stream continues from. The series already keeps each regime's anchor
        as a complete single-cube run, so nothing is laid out again."""
        data = json.loads((Path(series_dir) / "series_analysis_results.json").read_text())
        rows = [r for r in (data.get("results") or []) if isinstance(r, dict) and r.get("success")]
        regimes = ((data.get("locked_config") or {}).get("regimes") or {})
        if not rows or not regimes:
            raise RuntimeError("setup(): the reference series locked no recipe "
                               "(no cube of it was analysed successfully)")
        last = rows[-1]
        lock = regimes.get(last.get("regime")) or list(regimes.values())[-1]
        info = {"n": len(files), "fitted": len(rows), "anchored_on": lock.get("anchor_index"),
                "regimes": len(regimes), "model": "; ".join(
                    str(t.get("target") or "")[:120] for t in (lock.get("targets") or []))[:200],
                "data": files[min(int(last.get("index", len(files) - 1)), len(files) - 1)]}
        return str(lock["anchor_output_dir"]), info

    def replay_kwargs(self, loop: Any, data_path: str) -> Dict[str, Any]:
        state = loop._modality_state or {}
        return dict(system_info=loop._with_sidecar(data_path),
                    prior_analysis_paths=[str(loop.anchor_dir)],
                    reuse_locked_script=True, strict_replay=True,
                    replay_reference=state.get("replay_reference") or {})

    def escalation_kwargs(self, loop: Any, data_path: str, profile: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(data_path), "profile": profile}
        targets = (loop._modality_state or {}).get("locked_targets")
        if targets:
            # Fresh code for the SAME targets and output names: what is tracked
            # keeps its names whatever method the rebuild arrives at.
            kwargs["locked_targets"] = targets
        else:
            objective = self._objective(loop)
            if objective:
                kwargs["objective"] = objective
        return kwargs

    # --------------------------------------------------------------- results
    def features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Per-map MEANS and global scalars. A map's min and max are the
        extremes of a noisy field, not quantities to track or to gate on."""
        from ..agents.exp_agents.controllers.hyperspectral_series import flatten_feature_records
        records = result.get("feature_records") or result.get("extracted_features")
        if isinstance(records, dict):                      # an adopted run's flat file
            flat = {k: v for k, v in records.items() if isinstance(v, (int, float))}
        else:
            flat = flatten_feature_records(records)
        out = {}
        for key, value in flat.items():
            parts = key.split("_")
            if "min" in parts or "max" in parts:
                continue
            out[key] = float(value)
        return out

    def validity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        reuse = result.get("script_reuse") or {}
        withheld = [str(n) for d in (result.get("degraded_outputs") or []) if isinstance(d, dict)
                    for n in (d.get("missing_required") or [])]
        degraded = bool(result.get("degraded_outputs"))
        good = (result.get("status") == "success" and reuse.get("verbatim", True)
                and not reuse.get("scope_degraded") and not degraded)
        out: Dict[str, Any] = {"verdict": "good" if good else "poor",
                               "status": result.get("status")}
        if withheld:
            out["withheld"] = withheld
        return out

    # ---------------------------------------------------------------- anchor
    @staticmethod
    def _records_file(anchor: str) -> Optional[Path]:
        root = Path(str(anchor))
        if root.is_file():
            return root if root.name == "dynamic_analysis_records.json" else None
        for pattern in ("dynamic_analysis_records.json", "*/dynamic_analysis_records.json",
                        "results/*/dynamic_analysis_records.json"):
            found = sorted(root.glob(pattern))
            if found:
                return found[-1]
        return None

    @classmethod
    def _approved(cls, anchor: str) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
        path = cls._records_file(anchor)
        if path is None:
            return [], None
        try:
            records = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return [], None
        ok = [r for r in records if isinstance(r, dict) and r.get("task_success") and r.get("script")]
        return ok, path.parent

    def anchor_script(self, anchor: str) -> Tuple[Optional[str], Optional[Path]]:
        ok, anchor_dir = self._approved(anchor)
        if not ok:
            return None, None
        return "\n\n".join(str(r["script"]) for r in ok), anchor_dir

    def why_nothing_to_lock(self, result: Dict[str, Any]) -> str:
        missing = [str(n) for d in (result.get("degraded_outputs") or []) if isinstance(d, dict)
                   for n in (d.get("missing_required") or [])]
        return (f": the required output(s) {missing} never passed verification." if missing
                else ": no analysis task was approved.")

    no_anchor_message = ("holds no reusable hyperspectral run (expected a "
                         "dynamic_analysis_records.json with an approved script).")

    def anchor_state(self, anchor_dir: Path, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The replay gate's reference (the anchor's own map statistics) and the
        targets a rebuild must keep."""
        ok, _ = self._approved(str(anchor_dir))
        records = (result or {}).get("extracted_features")
        if not isinstance(records, list):
            try:
                records = json.loads((Path(anchor_dir) / "analysis_results.json").read_text()
                                     ).get("feature_records") or []
            except Exception:  # noqa: BLE001
                records = []
        reference = {}
        for m in records:
            if isinstance(m, dict) and m.get("name") and isinstance(m.get("stats"), dict):
                reference[str(m["name"])] = {
                    # In a stream the tracked value is expected to move: the replay
                    # gate judges method health, the loop's range gate plausibility.
                    "values_may_move": True,
                    **{k: v for k, v in m["stats"].items() if isinstance(v, (int, float))},
                    **({"coverage": m["coverage"]} if isinstance(m.get("coverage"), (int, float)) else {})}
        targets = [{"target": r.get("target"),
                    "required_outputs": list(r.get("required_outputs") or [])} for r in ok]
        required = {n for t in targets for n in t["required_outputs"]}
        extra = [n for n in reference if n not in required]
        if targets and extra:
            targets[0]["extra_outputs"] = extra
        return {"replay_reference": reference, "locked_targets": targets}

    def bake_edits(self, anchor_dir: Path, edits: List[Dict[str, Any]], dest: Path) -> Optional[Path]:
        """A cube replay runs the approved script(s) of a run directory verbatim,
        so an amendment is a COPY of that run with the edited script: the same
        records, targets and map statistics, one knob changed. Each edit must
        apply, exactly once, to exactly one of the approved scripts; a list that
        does not apply writes nothing. The copy says what it was amended from."""
        from ..utils.file_edit import apply_snippet_edits
        path = self._records_file(str(anchor_dir))
        if path is None:
            raise ValueError(f"{anchor_dir} holds no dynamic_analysis_records.json to amend")
        records = json.loads(path.read_text())
        for edit in edits:
            old = str(edit.get("old_text") or "")
            hits = [r for r in records if isinstance(r, dict) and r.get("task_success")
                    and old and str(r.get("script") or "").count(old) == 1]
            if len(hits) != 1:
                raise ValueError("script_edits do not apply to the locked recipe: "
                                 f"{old[:80]!r} is found in {len(hits)} of its scripts (exactly one needed)")
            res = apply_snippet_edits(hits[0]["script"], [edit])
            if res["status"] != "success":
                raise ValueError(f"script_edits do not apply to the locked recipe: {res['message']}")
            hits[0]["script"] = res["text"]
            hits[0]["amended"] = True
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "dynamic_analysis_records.json").write_text(json.dumps(records, indent=1, default=str))
        results = path.parent / "analysis_results.json"
        if results.is_file():
            (dest / "analysis_results.json").write_text(results.read_text())
        (dest / "amended_from.json").write_text(json.dumps(
            {"anchor_dir": str(path.parent), "edits": edits}, indent=1, default=str))
        return dest

    def check_portability(self, loop: Any, reference_data: str) -> Dict[str, Any]:
        from .portability import check_cube_portability, describe_cube
        work = loop.output_dir / "portability"

        def replay(path: str, tag: str) -> Optional[Dict[str, float]]:
            agent = loop._agent_factory(str(work / tag))
            res = agent.analyze(path, **self.replay_kwargs(loop, path)) or {}
            good = res.get("status") == "success" and self.validity(res).get("verdict") == "good"
            return self.features(res) if good else None
        report = check_cube_portability(replay, str(reference_data), str(work),
                                        tracked=list(loop.outputs) or None)
        if report:
            report["summary"] = describe_cube(report)
            report["positions_summary"] = ""
        return report

    def calibrate(self, anchor_dir: Path) -> Dict[str, Any]:
        return {}                # the replay gate is already relative to the anchor's own maps

    def residual_excess(self, frame_dir: str) -> Optional[float]:
        return None

    # ---------------------------------------------------------------- signal
    #: The field is also watched in regions, on a small pyramid of grids: a
    #: change confined to part of the field is diluted in the mean spectrum by
    #: the area it covers, and a feature that straddles the blocks of one grid
    #: sits inside a block of another (measured: a 3 x 3-pixel patch on a 14 x 14
    #: field was seen by the 3 x 3 grid 8 times of 8 and by the 4 x 4 grid 0 of
    #: 8). LOWER ``MIN_REGION_PIXELS`` or add a level for smaller features on a
    #: clean instrument, RAISE it (or use ``GRIDS = ()``) for very noisy cubes.
    GRIDS = (2, 3, 4)
    MIN_REGION_PIXELS = 9
    _ROWS = {2: ("upper", "lower"), 3: ("upper", "middle", "lower")}
    _COLS = {2: ("left", "right"), 3: ("left", "centre", "right")}

    @classmethod
    def region_name(cls, r: int, i: int, j: int) -> str:
        # Unique across grids: the 3 x 3 corners are not the 2 x 2 quadrants.
        if r in cls._ROWS:
            row, col = cls._ROWS[r][i], cls._COLS[r][j]
            where = "centre" if (row, col) == ("middle", "centre") else f"{row} {col}"
            return f"{where} {'quarter' if r == 2 else 'ninth'}"
        return f"row {i + 1}, column {j + 1} of a {r} x {r} grid"

    def read_signals(self, data_path: str, system_info: Any = None) -> Dict[str, Tuple[Any, Any]]:
        """The whole field's mean spectrum first, then each region's."""
        cube = load_cube(data_path)
        x = self._axis(data_path, cube.shape[-1], system_info)
        out = {"whole field": (x, np.nanmean(cube.reshape(-1, cube.shape[-1]), axis=0))}
        h, w = cube.shape[0], cube.shape[1]
        for r in self.GRIDS:
            if (h // r) * (w // r) < self.MIN_REGION_PIXELS or h < r or w < r:
                continue
            for i in range(r):
                for j in range(r):
                    block = cube[i * h // r:(i + 1) * h // r, j * w // r:(j + 1) * w // r]
                    out[self.region_name(r, i, j)] = (
                        x, np.nanmean(block.reshape(-1, cube.shape[-1]), axis=0))
        return out

    def read_signal(self, data_path: str, system_info: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """The cube's mean spectrum, on its physical axis when the metadata
        gives one (so a located change reads in eV or nm, not in channels)."""
        cube = load_cube(data_path)
        y = np.nanmean(cube.reshape(-1, cube.shape[-1]), axis=0)
        return self._axis(data_path, y.size, system_info), y

    @staticmethod
    def _axis(data_path: str, n: int, system_info: Any = None) -> np.ndarray:
        y = np.empty(n)
        x = np.arange(y.size, dtype=float)
        info = system_info if isinstance(system_info, dict) else {}
        try:
            side = json.loads(Path(str(data_path)).with_suffix(".json").read_text())
            meta = side.get("meta") if isinstance(side.get("meta"), dict) else side
            info = {**(meta if isinstance(meta, dict) else {}), **info}
        except Exception:  # noqa: BLE001 - no sidecar
            pass
        try:
            from ..skills.hyperspectral.eels.eels import create_axis
            axis, _label, has_axis = create_axis(y.size, info, axis_index=2)
            if has_axis and len(axis) == y.size:
                x = np.asarray(axis, dtype=float)
        except Exception:  # noqa: BLE001 - channels are a fine axis for a change signal
            pass
        return x


def load_cube(data_path: Any) -> np.ndarray:
    """A datacube as (H, W, channels). ``.npy`` or HDF5, like the agent's loader."""
    path = Path(str(data_path))
    if path.suffix.lower() == ".npy":
        cube = np.load(path)
    elif path.suffix.lower() in (".h5", ".hdf5", ".nxs"):
        from ..utils.hdf5_utils import load_hdf5_signal
        cube = np.asarray(load_hdf5_signal(str(path)))
    else:
        raise ValueError(f"{path.name}: a datacube is a .npy or HDF5 file")
    cube = np.asarray(cube, dtype=float)
    if cube.ndim == 2:
        cube = cube[None, ...]
    if cube.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D datacube, got shape {cube.shape}")
    return cube


class ImageModality(CurveModality):
    """2D images through ``ImageAnalysisAgent``.

    The reference image is analysed in full once; every later image is answered by
    a STRICT replay of its approved script (``analyze(strict_replay=True)``: no
    skill selection, planning, vision review, repair, tier 2, synthesis or report;
    zero model calls) and judged on evidence alone (``_replay_feature_gate``: the
    script still reports every quantity it reported on the reference, finite, and
    still finds something). That gate sees method HEALTH, not correctness: an image
    analysis has no R², so a segmentation that runs and is wrong is caught only by
    an audit (``audit_every``) or by the change signal. Say so to whoever relies on it.

    The change signal reads the image's radially averaged power spectrum, of the
    whole field and of each quarter (a change confined to part of it is not
    diluted), after a robust normalisation so gain and offset do not register. It
    answers to feature size, periodicity, focus and noise, and it is
    translation-invariant, so a field of view that drifts is not a change. The
    representation was chosen by measurement (log power on linear bins from
    k = 0.01): on tiles of a real HAADF image it raised no false novelty and found
    a defocus blur, a 5 % lattice change, an amorphous patch and doubled noise,
    where linear-power variants missed the blur and the noise and misfired on a
    slow coarsening. An intensity histogram beside it changed nothing and is not
    used."""

    name = "image"
    pinning = False            # names are asked for in the objective and checked after a rebuild
    script_edits = True        # applied per frame by the agent, like a curve
    portability = True         # by outputs, under a change of intensity level (.npy images)
    series_reference = True
    window_reanchor = False
    require_outputs_after_rebuild = True
    # The replay verdict of an image sees whether the method still runs, not whether
    # it is still right (live: a recipe went on counting 60 of 117 particles with a
    # "good" gate). So a change that "still fits" is checked by default, and a single
    # quick image analysis is not trusted over the recipe without a second one.
    default_on_change = "audit"
    audit_needs_second_opinion = True

    def make_agent(self, loop: Any, output_dir: str) -> Any:
        from ..agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        return ImageAnalysisAgent(api_key=loop.api_key, model_name=loop.model_name,
                                  base_url=loop.base_url, output_dir=output_dir,
                                  enable_human_feedback=loop._human_feedback)

    @staticmethod
    def _objective(loop: Any, names: Optional[List[str]] = None) -> Optional[str]:
        parts = []
        if loop.targets:
            parts.append("Quantities of interest: " + "; ".join(str(t) for t in loop.targets) + ".")
        if names:
            parts.append("Report these quantities in extracted_features under EXACTLY these names "
                         "(they are tracked across images by name): " + ", ".join(names) + ".")
        elif loop.outputs:
            parts.append("Report each of these in extracted_features under exactly this name: "
                         + "; ".join(f"{k} ({v})" for k, v in loop.outputs.items()) + ".")
        if parts:
            parts.append("The same script will be run unchanged on later images of the same "
                         "experiment, so it must not depend on values read off this one "
                         "(thresholds and sizes relative to the image's own statistics).")
        return " ".join(parts) or None

    def reference_kwargs(self, loop: Any, refs: List[str], profile: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(refs[-1])}
        if profile:
            kwargs["profile"] = profile
        if loop.targets:
            kwargs["targets"] = loop.targets
        objective = self._objective(loop)
        if objective:
            kwargs["objective"] = objective
        if len(refs) > 1:
            kwargs["series_metadata"] = {"variable": "frame", "values": list(range(len(refs)))}
            kwargs["profile"] = {"base": profile or "thorough", "trend": False,
                                 "synthesis": "none", "adaptive_refit": False}
        return kwargs

    def replay_kwargs(self, loop: Any, data_path: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            system_info=loop._with_sidecar(data_path),
            prior_analysis_paths=[str(loop.anchor_dir)],
            reuse_locked_script=True, strict_replay=True,
            replay_reference=(loop._modality_state or {}).get("replay_reference") or None)
        if loop._edits:
            kwargs["script_edits"] = list(loop._edits)
        return kwargs

    def escalation_kwargs(self, loop: Any, data_path: str, profile: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"system_info": loop._with_sidecar(data_path), "profile": profile}
        if loop.targets:
            kwargs["targets"] = loop.targets
        # What is tracked keeps its names: asked for here, checked at adoption.
        objective = self._objective(loop, names=list(loop.outputs) or None)
        if objective:
            kwargs["objective"] = objective
        return kwargs

    def features(self, result: Dict[str, Any]) -> Dict[str, float]:
        import math
        feats = result.get("extracted_features") or {}
        return {str(k): float(v) for k, v in feats.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}

    def validity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result.get("reuse_validity") or {}

    def anchor_script(self, anchor: str) -> Tuple[Optional[str], Optional[Path]]:
        from ..agents.exp_agents.controllers.image_analysis_controllers import (
            _first_prior_image_script, _load_prior_state)
        script, _label = _first_prior_image_script({"prior_analysis_paths": [str(anchor)]})
        anchor_dir, _data = _load_prior_state(str(anchor))
        return (script, anchor_dir) if script and anchor_dir is not None else (None, None)

    no_anchor_message = ("holds no reusable image-analysis run (expected analysis_results.json "
                         "and a saved script under scripts/).")

    def anchor_state(self, anchor_dir: Path, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        reference = self.features(result) if result else self.features_from_anchor(anchor_dir)
        return {"replay_reference": reference}

    def series_anchor(self, series_dir: Path, dest: Path, files: List[str]):
        """A series keeps one script per image; the stream continues from the LAST
        one analysed successfully, so that is laid out as a single-image run."""
        data = json.loads((Path(series_dir) / "series_analysis_results.json").read_text())
        ok = [r for r in (data.get("results") or []) if isinstance(r, dict) and r.get("success")]
        scripts = sorted((Path(series_dir) / "scripts").glob("*.py"))
        if not ok or not scripts:
            raise RuntimeError("setup(): no image of the reference series was analysed successfully")
        last = ok[-1]
        idx = int(last.get("index", len(files) - 1))
        stem = Path(str(last.get("name") or files[min(idx, len(files) - 1)])).stem
        script = next((p for p in scripts if p.stem == stem), scripts[-1])
        dest = Path(dest)
        (dest / "scripts").mkdir(parents=True, exist_ok=True)
        (dest / "scripts" / "analysis_script.py").write_text(script.read_text(), encoding="utf-8")
        (dest / "analysis_results.json").write_text(json.dumps({
            "status": "success", "analysis_type": last.get("analysis_type"),
            "extracted_features": last.get("extracted_features") or {},
            "derived_from": {"series_dir": str(series_dir), "index": idx}}, indent=1, default=str))
        info = {"n": len(files), "fitted": len(ok), "anchored_on": idx,
                "regimes": len(((data.get("series_analysis_plan") or {}).get("regimes") or [])) or 1,
                "model": str(last.get("analysis_type") or "")[:200],
                "data": files[min(idx, len(files) - 1)]}
        return str(dest), info

    def check_portability(self, loop: Any, reference_data: str) -> Dict[str, Any]:
        from .portability import check_cube_portability, describe_cube
        if Path(str(reference_data)).suffix.lower() != ".npy":
            return {}                    # an 8-bit image cannot be rescaled without clipping
        work = loop.output_dir / "portability"

        def replay(path: str, tag: str) -> Optional[Dict[str, float]]:
            agent = loop._agent_factory(str(work / tag))
            res = agent.analyze(path, **self.replay_kwargs(loop, path)) or {}
            good = res.get("status") == "success" and self.validity(res).get("verdict") == "good"
            return self.features(res) if good else None
        report = check_cube_portability(replay, str(reference_data), str(work),
                                        tracked=list(loop.outputs) or None, load=load_image)
        if report:
            report["kind"] = "image"
            report["summary"] = describe_cube(report)
            report["positions_summary"] = ""
        return report

    def calibrate(self, anchor_dir: Path) -> Dict[str, Any]:
        return {}

    def residual_excess(self, frame_dir: str) -> Optional[float]:
        return None

    def annotate_where(self, where: List[Dict[str, Any]], data_path: str,
                       system_info: Any = None) -> List[Dict[str, Any]]:
        """The change is located on a spatial-frequency axis (cycles per pixel).
        What a person reads is the LENGTH SCALE: one period, in nm when the pixel
        size is known. Added beside the frequencies, never instead of them."""
        nm_per_px = None
        try:
            from ..skills._shared.image_analysis_tools import resolve_pixel_size_nm
            info = system_info if isinstance(system_info, dict) else {}
            try:
                side = json.loads(Path(str(data_path)).with_suffix(".json").read_text())
                meta = side.get("meta") if isinstance(side.get("meta"), dict) else side
                info = {**(meta if isinstance(meta, dict) else {}), **info}
            except Exception:  # noqa: BLE001 - no sidecar
                pass
            px = resolve_pixel_size_nm(info, load_image(data_path).shape)
            nm_per_px = float(px["x"]) if px else None
        except Exception:  # noqa: BLE001 - pixels are a fine unit
            nm_per_px = None
        unit, k = ("nm", nm_per_px) if nm_per_px else ("px", 1.0)
        out = []
        for w in where:
            w = dict(w)
            if w.get("kind") != "window" and all(isinstance(w.get(f), (int, float)) and w[f] > 0
                                                 for f in ("x_from", "x_to", "x_peak")):
                w["length_from"], w["length_to"] = round(k / w["x_to"], 4), round(k / w["x_from"], 4)
                w["length_peak"], w["length_units"] = round(k / w["x_peak"], 4), unit
            out.append(w)
        return out

    # ---------------------------------------------------------------- signal
    N_K = 96

    def read_signal(self, data_path: str, system_info: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        return radial_power_spectrum(normalised(load_image(data_path)), self.N_K)

    def read_signals(self, data_path: str, system_info: Any = None) -> Dict[str, Tuple[Any, Any]]:
        z = normalised(load_image(data_path))
        out = {"whole field": radial_power_spectrum(z, self.N_K)}
        h, w = z.shape
        if min(h, w) >= 64:
            for i, row in enumerate(("upper", "lower")):
                for j, col in enumerate(("left", "right")):
                    block = z[i * h // 2:(i + 1) * h // 2, j * w // 2:(j + 1) * w // 2]
                    out[f"{row} {col} quarter"] = radial_power_spectrum(block, self.N_K)
        return out


IMAGE_SUFFIXES = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_image(data_path: Any) -> np.ndarray:
    """A 2D greyscale image as floats (``.npy`` or a common image file)."""
    path = Path(str(data_path))
    if path.suffix.lower() == ".npy":
        a = np.load(path)
    else:
        from ..skills._shared.image_analysis_tools import load_image_data
        a = np.asarray(load_image_data(str(path)))
    a = np.asarray(a, dtype=float)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D image, got shape {a.shape}")
    return a


def normalised(image: np.ndarray) -> np.ndarray:
    """Robust z-scores: detector gain and offset do not register as a change."""
    med = float(np.median(image))
    mad = 1.4826 * float(np.median(np.abs(image - med))) or float(image.std()) or 1.0
    return (image - med) / mad


def radial_power_spectrum(z: np.ndarray, n_bins: int = 96,
                          k_min: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """log10 of the radially averaged power against spatial frequency (cycles per
    pixel, ``k_min`` to 0.5). Translation-invariant, so a field of view that
    drifts is not a change; what moves it is feature size, periodicity, focus,
    noise. The lowest frequencies are left out: a few samples each, and they
    carry the uneven background and whatever drifted into the field."""
    h, w = z.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2((z - z.mean()) * window))) ** 2 / (h * w)
    ky, kx = np.indices(power.shape)
    radius = np.hypot((ky - h // 2) / h, (kx - w // 2) / w)
    edges = np.linspace(k_min, 0.5, n_bins + 1)
    idx = np.digitize(radius.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    total = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=n_bins)
    count = np.maximum(np.bincount(idx[ok], minlength=n_bins), 1)
    return 0.5 * (edges[1:] + edges[:-1]), np.log10(total / count + 1e-12)


_MODALITIES = {"curve": CurveModality, "hyperspectral": HyperspectralModality,
               "image": ImageModality}
CUBE_SUFFIXES = (".npy", ".h5", ".hdf5", ".nxs")


def resolve_modality(modality: Any) -> CurveModality:
    """``"curve"`` (default), ``"hyperspectral"``, or an instance."""
    if modality is None:
        return CurveModality()
    if isinstance(modality, str):
        try:
            return _MODALITIES[modality]()
        except KeyError:
            raise ValueError(f"unknown modality {modality!r}; one of {sorted(_MODALITIES)}") from None
    return modality
