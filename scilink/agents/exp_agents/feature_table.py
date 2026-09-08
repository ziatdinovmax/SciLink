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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Trailing name tokens that carry no identity: ``metric`` and ``metric_value``
# are the same quantity. Kept deliberately small — statistics suffixes (mean,
# std, err) and single-letter fit-parameter names are NOT variants.
_VARIANT_SUFFIX_TOKENS = ("value", "val")


def _variant_key(name: str) -> str:
    """Column name reduced to its identity: lower-cased tokens with trailing
    ``value`` / ``val`` stripped, so ``Metric_value`` and ``metric`` collide."""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]
    while toks and toks[-1] in _VARIANT_SUFFIX_TOKENS:
        toks.pop()
    return "_".join(toks)


def _present(value: Any) -> bool:
    return value is not None and value != "" and not (
        isinstance(value, float) and value != value)


def merge_variant_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse per-unit key-name variants of one quantity into one column
    (issue #534).

    Per-unit scripts are generated independently, so one unit may report a
    derived scalar as ``metric`` while the rest report ``metric_value``. Left
    alone, the flatten writes two sibling columns, each partially populated,
    and a BO keyed on one of them silently drops the other units. Two names
    are merged only when they reduce to the same :func:`_variant_key` AND no
    unit populates both (complementary sets) — a unit carrying both is
    evidence they are distinct quantities and they are left untouched. The
    survivor is the name populated by the most units (first seen on a tie),
    so a target name a caller already chose stays valid. Mutates ``rows`` in
    place; returns the merge notes (``column``, ``merged_from``, ``units``).
    """
    order: List[str] = []
    for row in rows:
        for key in row:
            if key not in order:
                order.append(key)
    groups: Dict[str, List[str]] = {}
    for key in order:
        groups.setdefault(_variant_key(key), []).append(key)
    notes: List[Dict[str, Any]] = []
    for names in groups.values():
        if len(names) < 2:
            continue
        # Complementary: no unit carries more than one of the variants.
        if any(sum(_present(row.get(n)) for n in names) > 1 for row in rows):
            continue
        counts = {n: sum(_present(row.get(n)) for row in rows) for n in names}
        canonical = max(names, key=lambda n: (counts[n], -names.index(n)))
        moved_units: List[str] = []
        for row in rows:
            for n in names:
                if n == canonical or n not in row:
                    continue
                val = row.pop(n)
                if _present(val):
                    row[canonical] = val
                    moved_units.append(str(row.get("unit", "?")))
        merged_from = [n for n in names if n != canonical]
        notes.append({"column": canonical, "merged_from": merged_from,
                      "units": moved_units})
        logger.warning(
            f"feature table: merged variant column(s) {merged_from} into "
            f"'{canonical}' for unit(s) {moved_units} — the same quantity was "
            f"reported under different names (#534)")
    return notes


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


_WARN_MAX_UNITS = 6  # unit names listed per warning before eliding


def _feature_table_warnings(header: List[str], rows: List[List[str]]
                            ) -> List[str]:
    """Human-readable hazards a consumer keying a BO on this table must see
    (issue #534): a column empty for SOME units (those units would be
    silently excluded from an optimization keyed on it), and a pair of
    partially populated columns whose populated unit sets are complementary
    (the signature of one quantity reported under two names)."""
    n = len(rows)
    if n == 0 or not header:
        return []
    unit_idx = next((i for i, c in enumerate(header)
                     if str(c).lower() == "unit"), None)

    def _unit(r_i: int) -> str:
        r = rows[r_i]
        if unit_idx is not None and unit_idx < len(r) and r[unit_idx] != "":
            return r[unit_idx]
        return f"row {r_i + 1}"

    def _elide(names: List[str]) -> str:
        shown = ", ".join(names[:_WARN_MAX_UNITS])
        extra = len(names) - _WARN_MAX_UNITS
        return shown + (f", … (+{extra})" if extra > 0 else "")

    populated: Dict[int, set] = {}
    for i in range(len(header)):
        populated[i] = {r_i for r_i, r in enumerate(rows)
                        if i < len(r) and r[i] != "" and r[i].lower() != "nan"}
    partial = [i for i in range(len(header)) if 0 < len(populated[i]) < n]
    warnings: List[str] = []
    for i in partial:
        holes = sorted(set(range(n)) - populated[i])
        warnings.append(
            f"Column '{header[i]}' is empty for {len(holes)} of {n} units "
            f"({_elide([_unit(h) for h in holes])}); an optimization keyed "
            f"on it would exclude those units.")
    for a_pos, i in enumerate(partial):
        for j in partial[a_pos + 1:]:
            pi, pj = populated[i], populated[j]
            if pi & pj or len(pi | pj) != n:
                continue
            warnings.append(
                f"Columns '{header[i]}' ({len(pi)} units) and '{header[j]}' "
                f"({len(pj)} units) are populated for complementary unit sets "
                f"— likely the same quantity reported under two names. Keying "
                f"a BO on either one drops the other's units; pick one name "
                f"and re-report the other units under it.")
    return warnings


def describe_feature_table(path) -> Optional[Dict[str, Any]]:
    """Schema summary of a feature CSV for callers that cannot open the file
    (a remote MCP client, an LLM deciding inputs/targets): column names, row
    count, per-column missing counts (only columns with any missing), and
    ``warnings`` — the partial-population / split-column hazards a BO
    handoff must not proceed past silently (#534). Never raises."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return None
            missing = [0] * len(header)
            rows: List[List[str]] = []
            for row in reader:
                rows.append(row)
                for i, v in enumerate(row[:len(header)]):
                    if v == "" or v.lower() == "nan":
                        missing[i] += 1
                if len(row) < len(header):
                    for i in range(len(row), len(header)):
                        missing[i] += 1
        return {
            "columns": header,
            "n_rows": len(rows),
            "missing": {c: m for c, m in zip(header, missing) if m},
            "warnings": _feature_table_warnings(header, rows),
        }
    except Exception:  # noqa: BLE001 - descriptive only
        return None


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
        # One quantity under two per-unit names would otherwise become two
        # half-empty sibling columns (#534).
        merge_variant_columns(rows)
        columns: List[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        dest = output_dir / "features.csv"
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(dest.resolve())
    except Exception as e:  # noqa: BLE001 - never break the analysis on this
        logger.warning(f"feature table emit failed: {e}")
        return None
