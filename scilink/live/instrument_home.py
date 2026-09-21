"""What an instrument has learned, kept across runs.

A live loop locks a recipe for one run. The instrument outlives the run: the
same technique on the same kind of sample comes back tomorrow, and the recipe
that took minutes and a handful of model calls to build is still right. An
``InstrumentHome`` is the instrument's own store, keyed on the identity the
instrument declares (``Instrument.describe()["id"]``), not on a chat session:

    ~/.scilink/instruments/<instrument id>/
        instrument.json            who this is, first and last seen
        recipes/<recipe id>/
            recipe.json            modality, technique, tracked outputs, provenance, use counts
            anchor/                the run the recipe was locked from (what a replay needs)
        runs.jsonl                 one line per finished run

A loop opened with ``remember=True`` does three things with it. At ``setup`` it
first tries the instrument's known recipes on the reference data by strict replay
(no model call, seconds): one that fits, and reports what is tracked, arms the
loop. Otherwise the reference is analysed as usual and the new recipe is kept. A
rebuild also tries them before building anything. And ``close`` writes the run's
summary. Nothing here imports the server, chat or an orchestrator.

Opt-in on purpose: a recipe verified on one sample is a hypothesis about the
next, which is why a recalled recipe is replayed and judged before it is used,
and why the change signal and the audits run on it like on any other.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Never copied into the store: heavy per-run artifacts a replay does not need.
_SKIP_DIRS = ("dcnn_trained", "__pycache__", "_candidates")
_MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_RECIPES = 12


def instruments_root(root: Optional[str] = None) -> Path:
    if root:
        return Path(root).expanduser()
    base = os.environ.get("SCILINK_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".scilink") / "instruments"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "instrument"


class InstrumentHome:
    def __init__(self, instrument: Any, root: Optional[str] = None) -> None:
        info = instrument if isinstance(instrument, dict) else (
            instrument.describe() if hasattr(instrument, "describe") else {"id": str(instrument)})
        if not info.get("id"):
            raise ValueError("an instrument home needs the instrument's id (Instrument.describe()['id'])")
        self.info = dict(info)
        self.dir = instruments_root(root) / _safe(info["id"])
        (self.dir / "recipes").mkdir(parents=True, exist_ok=True)
        self._touch()

    def _touch(self) -> None:
        path = self.dir / "instrument.json"
        try:
            known = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            known = {"first_seen": _now()}
        known.update({**self.info, "last_seen": _now()})
        path.write_text(json.dumps(known, indent=1, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- recipes
    def recipes(self, modality: Optional[str] = None, technique: Optional[str] = None,
                outputs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Known recipes that could serve this measurement, most recently used
        first: the same modality, the same technique when both say one, and every
        tracked output among what the recipe reports."""
        found = []
        for meta_path in (self.dir / "recipes").glob("*/recipe.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001 - a torn record is skipped
                continue
            anchor = meta_path.parent / "anchor"
            if not anchor.is_dir():
                continue
            if modality and meta.get("modality") != modality:
                continue
            if technique and meta.get("technique") and _norm(meta["technique"]) != _norm(technique):
                continue
            if outputs and not set(outputs) <= set(meta.get("reports") or []):
                continue
            found.append({**meta, "anchor_dir": str(anchor)})
        found.sort(key=lambda m: str(m.get("last_used") or m.get("created") or ""), reverse=True)
        return found

    def save_recipe(self, loop: Any, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Keep the loop's current recipe. Idempotent per recipe id."""
        if loop.recipe is None or loop.anchor_dir is None:
            return None
        rid = str(loop.recipe["id"])
        dest = self.dir / "recipes" / rid
        meta_path = dest / "recipe.json"
        if not meta_path.is_file():
            if (dest / "anchor").exists():
                shutil.rmtree(dest / "anchor", ignore_errors=True)
            _copy_run(Path(loop.anchor_dir), dest / "anchor")
        info = loop.system_info if isinstance(loop.system_info, dict) else {}
        meta = {"recipe_id": rid, "modality": loop.modality.name,
                "technique": info.get("technique") or (loop.instrument or {}).get("technique"),
                "sample": info.get("sample"), "outputs": dict(loop.outputs or {}),
                "targets": list(loop.targets or []), "edits": list(loop._edits or []),
                "reports": sorted(loop._reference_features or {}),
                "source": source or loop.recipe.get("source"), "created": _now(), "uses": 0}
        if meta_path.is_file():
            try:
                old = json.loads(meta_path.read_text())
                meta.update({"created": old.get("created", meta["created"]), "uses": int(old.get("uses") or 0)})
            except Exception:  # noqa: BLE001
                pass
        meta["last_used"] = _now()
        meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        self._trim()
        return meta

    def used(self, recipe_id: str) -> None:
        path = self.dir / "recipes" / str(recipe_id) / "recipe.json"
        try:
            meta = json.loads(path.read_text())
            meta["uses"], meta["last_used"] = int(meta.get("uses") or 0) + 1, _now()
            path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _trim(self) -> None:
        metas = self.recipes()
        for meta in metas[MAX_RECIPES:]:
            shutil.rmtree(Path(meta["anchor_dir"]).parent, ignore_errors=True)

    # ------------------------------------------------------------------- runs
    def record_run(self, loop: Any) -> Dict[str, Any]:
        status = loop.status()
        log = loop.read_log()
        novelties = [{k: e.get(k) for k in ("step", "since_step", "onset", "fraction", "region", "where")}
                     for e in log if e.get("event") == "novelty"]
        frames = [e for e in log if e.get("event") == "frame"]
        params: Dict[str, List[float]] = {}
        for f in frames:
            for k, v in (f.get("params") or {}).items():
                if isinstance(v, (int, float)):
                    lo, hi = params.get(k, [v, v])
                    params[k] = [min(lo, v), max(hi, v)]
        summary = {"when": _now(), "run_dir": str(loop.output_dir), "frames": len(frames),
                   "clean_frames": sum(1 for f in frames if not f.get("flags")),
                   "recipes": sorted({f.get("recipe_id") for f in frames if f.get("recipe_id")}),
                   "reanchors": status.get("reanchors"), "audits": status.get("audits"),
                   "novelties": novelties, "params": params,
                   "tracked": {k: [f["features"][k] for f in frames[-1:] if k in (f.get("features") or {})]
                               for k in (loop.outputs or {})}}
        with open(self.dir / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")
        return summary

    def runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            lines = (self.dir / "runs.jsonl").read_text().splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def _copy_run(src: Path, dest: Path) -> None:
    """The run directory a recipe was locked from, without what a replay never
    reads (model weights, candidate attempts, anything very large)."""
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        target = dest / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and path.stat().st_size <= _MAX_FILE_BYTES:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
