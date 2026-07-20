"""Flatten a completed analysis run into a tabular feature file.

One row per analyzed unit (a spectrum, an image, …); columns = the unit's
experimental conditions (from its per-unit sidecar JSON) + extracted scalar
features. The CSV feeds the planning Scalarizer / Bayesian optimization via
the meta-agent — see ``analysis_bo_feature_table_plan.md``.

The numeric path is LLM-free: this reads the structured result files the
analysis pipeline already persisted and writes a deterministic flatten.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _flatten_scalars(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively collect scalar leaves of a nested dict as flat columns.

    Lists / arrays / maps are skipped — only scalars belong in a feature row.
    """
    flat: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return flat
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_scalars(value, name + "_"))
        elif isinstance(value, (int, float, str)):  # bool is an int subclass
            flat[name] = value
    return flat


def _sidecar_conditions(data_path: Optional[str]) -> Dict[str, Any]:
    """Per-unit experimental conditions from the data file's sidecar JSON
    (``spec_5K.csv`` -> ``spec_5K.json``). Empty dict if absent / unreadable."""
    if not data_path:
        return {}
    sidecar = Path(data_path).with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        cond = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001 - a bad sidecar must not break the run
        return {}
    if not isinstance(cond, dict):
        return {}
    return {k: v for k, v in cond.items() if isinstance(v, (int, float, str))}


def _series_conditions(series_meta: Any, index: Any,
                       data_path: Optional[str]) -> Dict[str, Any]:
    """Per-unit conditions from the run's top-level ``series_metadata`` — the
    control variable(s) when conditions are supplied as a manifest / series list
    rather than per-file sidecar JSONs. Each block is
    ``{"variable", "values", "unit"}`` where ``values`` is either a list aligned
    to the spectrum ``index`` or a ``{filename-or-stem: value}`` map; the primary
    block is joined together with any ``secondary_variables`` (grid designs).
    Returns ``{}`` on anything unexpected — a malformed block must not break the
    run, and sidecar conditions (read separately) take precedence over these."""
    if not isinstance(series_meta, dict):
        return {}
    blocks = [series_meta]
    sec = series_meta.get("secondary_variables")
    if isinstance(sec, list):
        blocks.extend(sec)
    name = Path(data_path).name if data_path else None
    stem = Path(data_path).stem if data_path else None
    out: Dict[str, Any] = {}
    for b in blocks:
        if not isinstance(b, dict):
            continue
        var, vals = b.get("variable"), b.get("values")
        if not isinstance(var, str) or vals is None:
            continue
        val = None
        if isinstance(vals, dict):
            val = vals.get(name, vals.get(stem))
        elif isinstance(vals, (list, tuple)):
            if isinstance(index, int) and 0 <= index < len(vals):
                val = vals[index]
        if isinstance(val, (int, float, str)):
            out[var] = val
    return out


def _curve_fit_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """One row per spectrum from a curve-fitting run's series_fit_results.json."""
    sfr = output_dir / "series_fit_results.json"
    if not sfr.is_file():
        return []
    try:
        data = json.loads(sfr.read_text())
    except Exception:  # noqa: BLE001
        return []
    series_meta = data.get("series_metadata")
    rows: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        if not isinstance(r, dict) or not r.get("success"):
            continue
        row: Dict[str, Any] = {"unit": r.get("name") or f"index_{r.get('index')}"}
        # A per-file sidecar is the authoritative, complete per-unit condition
        # record; fall back to the coarser series_metadata ONLY for units without
        # one. Using it as a strict fallback (not an additive layer) avoids
        # double-counting the same control variable under different names — e.g.
        # sidecar 'temperature_C' alongside a series 'temperature' column.
        row.update(_sidecar_conditions(r.get("data_path"))
                   or _series_conditions(series_meta, r.get("index"), r.get("data_path")))
        row.update(_flatten_scalars(r.get("parameters")))
        row.update(_flatten_scalars(r.get("fit_quality"), "fit_"))
        rows.append(row)
    return rows


def _image_series_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """One row per image from an image-analysis run's
    series_analysis_results.json (the image-series analog of a curve-fitting
    run's series_fit_results.json; a single image is a series of one)."""
    sar = output_dir / "series_analysis_results.json"
    if not sar.is_file():
        return []
    try:
        data = json.loads(sar.read_text())
    except Exception:  # noqa: BLE001
        return []
    series_meta = data.get("series_metadata")
    rows: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        if not isinstance(r, dict) or not r.get("success"):
            continue
        row: Dict[str, Any] = {"unit": r.get("name") or f"index_{r.get('index')}"}
        # Sidecar is authoritative per unit; series_metadata is a strict fallback
        # for units lacking one (see _curve_fit_rows for the rationale).
        row.update(_sidecar_conditions(r.get("data_path"))
                   or _series_conditions(series_meta, r.get("index"), r.get("data_path")))
        row.update(_flatten_scalars(r.get("extracted_features")))
        row.update(_flatten_scalars(r.get("quality_metrics"), "quality_"))
        rows.append(row)
    return rows


def _extracted_feature_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """Generic fallback: one row from an agent that records a top-level
    ``extracted_features`` dict in analysis_results.json."""
    ar = output_dir / "analysis_results.json"
    if not ar.is_file():
        return []
    try:
        data = json.loads(ar.read_text())
    except Exception:  # noqa: BLE001
        return []
    feats = data.get("extracted_features")
    if not isinstance(feats, dict) or not feats:
        return []
    row: Dict[str, Any] = {"unit": output_dir.name}
    row.update(_flatten_scalars(feats))
    return [row]


def write_feature_table(output_dir) -> Optional[str]:
    """Write ``<output_dir>/features.csv`` — a flat per-unit feature table
    derived from the run's structured result files.

    Returns the absolute path, or ``None`` if no adapter applies or the run
    produced no scalar features. Never raises — a failure here must not break
    the analysis.
    """
    try:
        output_dir = Path(output_dir)
        # Per-unit series adapters first (curve-fitting spectra, then image
        # series — both one row per unit with conditions merged from sidecars);
        # fall back to the generic top-level ``extracted_features`` dict.
        rows = (
            _curve_fit_rows(output_dir)
            or _image_series_rows(output_dir)
            or _extracted_feature_rows(output_dir)
        )
        if not rows:
            return None
        columns: List[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        dest = output_dir / "features.csv"
        with open(dest, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(dest.resolve())
    except Exception as e:  # noqa: BLE001 - never break the analysis on this
        logger.warning(f"feature table emit failed: {e}")
        return None
