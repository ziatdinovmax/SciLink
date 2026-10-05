"""One analysis over several measurements: the series' other shape (#754).

A series is analysed per unit: the first unit's recipe is locked and replayed
on every later one. Some methods cannot be split that way, because their
result is defined only over the set of measurements, not for any one of them.
Split per unit, each script holds one measurement and the method is
undefined. Observed live: every unit script then constructed the measurements
it lacked from an assumed parameter, measured its own assumption back, and the
result passed every gate.

So the series planner (curve, image and hyperspectral alike) declares the
shape with its plan: ``"analysis_shape": "per_unit"`` (the default, the series
mode as it was) or ``"joint"``. A joint plan is run as ONE analysis whose
inputs are all the units, staged here with a manifest the generated script
reads. This module is the modality-neutral half: the planner rule, reading the
declaration, staging, and the prompt text. Each agent supplies only how one
unit becomes an array (or a path, for a unit too large to copy).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence  # noqa: F401

import numpy as np

PER_UNIT = "per_unit"
JOINT = "joint"
SHAPES = (PER_UNIT, JOINT)
MANIFEST_NAME = "joint_units.json"

#: The series planners' rule, one principle (appended to each agent's series
#: planning prompt). The JSON field it names is read by ``analysis_shape_of``.
PLANNER_RULE = (
    '- `"analysis_shape"` (top-level field, default `"per_unit"`): `"per_unit"` when each '
    'measurement can be analysed on its own and then compared; `"joint"` when the requested '
    "method's result is defined only over several measurements together, not for any one of "
    "them, so they must be analysed in ONE analysis. A joint analysis is one analysis over all "
    "the measurements; do not plan regimes for it."
)


def analysis_shape_of(plan: Any) -> str:
    """The shape a planner declared: ``joint`` only when it said so, else
    ``per_unit`` (a missing, misspelled or malformed field keeps the series
    mode as it was)."""
    if not isinstance(plan, dict):
        return PER_UNIT
    raw = plan.get("analysis_shape")
    if raw is None and isinstance(plan.get("series_analysis_plan"), dict):
        raw = plan["series_analysis_plan"].get("analysis_shape")
    norm = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    return JOINT if norm == JOINT else PER_UNIT


def stage_joint_units(units: Sequence[Dict[str, Any]], out_dir: Path | str, *,
                      to_array: Optional[Callable[[str], Any]] = None) -> Path:
    """Write the joint analysis's inputs and their manifest into ``out_dir``.

    ``units``: one dict per measurement, in series order: ``path`` (the
    source file), optional ``label`` and ``control_value`` (the series'
    control variable for that unit). ``to_array`` (the agent's loader) turns
    a source file into the array its scripts take; the array is written as
    ``joint_unit_<k>.npy`` and that copy is what the script reads. Without a
    loader, or when it returns None, the source path itself is listed.
    Returns the manifest's path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, u in enumerate(units):
        src = str(u.get("path") or "")
        staged = None
        if to_array is not None and src:
            try:
                arr = to_array(src)
            except Exception:  # noqa: BLE001 - an unloadable unit is listed by its source
                arr = None
            if arr is not None:
                staged = out / f"joint_unit_{k:04d}.npy"
                np.save(staged, np.asarray(arr))
        row = {"index": k, "path": str(staged or src), "source": src,
               "label": u.get("label") or Path(src).stem}
        if u.get("control_value") is not None:
            row["control_value"] = u["control_value"]
        if staged is not None:
            row["shape"] = list(np.load(staged, mmap_mode="r").shape)
        rows.append(row)
    manifest = out / MANIFEST_NAME
    manifest.write_text(json.dumps({"units": rows}, indent=1, default=str), encoding="utf-8")
    return manifest


def read_manifest(path: Path | str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    units = data.get("units") if isinstance(data, dict) else None
    return [u for u in units if isinstance(u, dict)] if isinstance(units, list) else []


def joint_units_block(manifest: Path | str, *, control_name: Optional[str] = None) -> str:
    """The code-generation prompt's description of a joint analysis's inputs."""
    units = read_manifest(manifest)
    if not units:
        return ""
    ctl = f" ({control_name})" if control_name else ""
    lines = []
    for u in units:
        val = f", {control_name or 'control'} = {u['control_value']}" if "control_value" in u else ""
        shp = f", shape {tuple(u['shape'])}" if u.get("shape") else ""
        lines.append(f"- unit {u['index']}: `{u['path']}` ({u['label']}{val}{shp})")
    return (
        f"\n**JOINT ANALYSIS: {len(units)} measurements, analysed together in this one script.**\n"
        f"The method needs all of them at once, so the script must load EVERY file listed below "
        f"(the manifest `{manifest}` lists the same, with each unit's control value{ctl}). The "
        f"primary data file is unit 0, one of these measurements. Every file is a measurement, "
        f"not a reference.\n" + "\n".join(lines) + "\n"
    )


def joint_units_planning_text(manifest: Path | str, *, control_name: Optional[str] = None) -> str:
    """The planning prompt's note: the analysis is joint, over these units."""
    units = read_manifest(manifest)
    if not units:
        return ""
    vals = [u.get("control_value") for u in units if "control_value" in u]
    over = (f" over {control_name or 'the control variable'} = {vals}" if vals else "")
    return (f"\n## Joint analysis\nThis is ONE analysis of {len(units)} measurements{over}, "
            f"planned as a joint method (the measurements are its inputs, not repeats of one "
            f"fit). Plan the method over all of them; the script receives every file.\n")


def joint_contract_note(manifest: Path | str, *, control_name: Optional[str] = None) -> str:
    """For the checks and repairs of a joint run's script (conformance,
    correction): reading every listed measurement IS the contract. Without
    it, a conformance pass read the extra files as a breach of the
    data-loading rules and the correction cut the script back to one unit."""
    block = joint_units_block(manifest, control_name=control_name)
    if not block:
        return ""
    return ("\n**Joint analysis contract:** this script is required to read every measurement listed "
            "below, by the paths given (absolute paths, outside the working directory). Reading them "
            "is the plan, not a deviation from the data-loading rules; a correction keeps reading all "
            "of them.\n" + block)


def with_joint_note(prompt: str, state: dict) -> str:
    """``prompt`` with a joint run's contract note before its response footer."""
    if not state.get("joint_manifest"):
        return prompt
    note = joint_contract_note(state["joint_manifest"], control_name=(state.get("joint_control") or None))
    if not note:
        return prompt
    marker = "**Response:**"
    return prompt.replace(marker, note + "\n" + marker, 1) if marker in prompt else prompt + note


#: Why a joint run's script is not replayed (#757 review): it reads the
#: measurements of its own run by their paths, so on new data it would rest on
#: the old ones, and a replay gate has no reason to object.
REPLAY_REFUSAL = ("a joint analysis's script reads the measurements of its own run by their paths; "
                  "replayed on new data it would rest on those old measurements, so it is not replayed "
                  "(a replay over a new set of measurements is not designed yet)")


def _shape_in(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data.get("analysis_shape") if isinstance(data, dict) else None


def replay_refusal(path: Any) -> Optional[str]:
    """``REPLAY_REFUSAL`` when ``path`` (a run folder, a script or records file
    inside one, or a board copy with its ``<stem>.recipe.json`` sidecar)
    belongs to a joint run, else None. The one check every reuse reader
    consults, whatever the agent."""
    if path is None:
        return None
    p = Path(str(path))
    candidates = []
    if p.is_dir():
        candidates.append(p / "analysis_results.json")
    else:
        candidates.append(p.with_name(f"{p.stem}.recipe.json"))       # a board copy's sidecar
        for up in (p.parent, p.parent.parent):                         # the run it sits in
            candidates.append(up / "analysis_results.json")
    return REPLAY_REFUSAL if any(_shape_in(c) == JOINT for c in candidates) else None


def shape_text(shape: Any, n_units: int) -> str:
    """The plan gate's line on the series' shape (console)."""
    if shape == JOINT:
        return (f"🔗 Analysis shape: JOINT, one analysis over all {n_units} measurements together. "
                f"No per-measurement rows, no trend, no regimes. Type feedback to analyse each "
                f"measurement separately instead.")
    return f"🔗 Analysis shape: per unit, one analysis per measurement ({n_units}), then the trend."


def shape_blocks(shape: Any, n_units: int) -> list:
    """The plan gate's subject blocks for the series' shape: a fields row, and
    a notice when it is joint, so Enter accepts what was shown (#757)."""
    from ...hitl import subject_block
    joint = shape == JOINT
    blocks = [subject_block("fields", label="🔗 Analysis shape", items=[{
        "label": "shape",
        "value": (f"joint: one analysis over all {n_units} measurements" if joint
                  else f"per unit: one analysis per measurement ({n_units})"),
        **({"flag": "warn"} if joint else {})}])]
    if joint:
        blocks.append(subject_block("notice", title="One analysis over every measurement", tone="warn", lines=[
            f"The series of {n_units} becomes ONE analysis whose inputs are all the measurements.",
            "No per-measurement rows, no trend, no regimes.",
            "Type feedback to analyse each measurement separately instead."]))
    return blocks


def redirect_warning(n_units: int) -> str:
    """The joint run's warning on its result, for a headless caller or the meta."""
    return (f"Run as ONE joint analysis over all {n_units} measurements (the planner declared "
            f"analysis_shape: joint), not as a series: no per-measurement rows, no trend.")
