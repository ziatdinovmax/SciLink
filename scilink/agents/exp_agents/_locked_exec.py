"""Shared locked-script execution for the series-based foundation agents.

Both the curve-fitting and image-analysis agents lock one LLM-generated script
after planning and reuse it across every item in a series. The reuse used to be
done by **regex-rewriting the generated source** per item (the data path, the
visualization filename, and a `glob.glob('*.npy')` neutralization) — fragile,
because a path the regex didn't anticipate silently made the script read the
wrong item.

This module replaces that with a simple, robust contract: run the locked script
**verbatim** in a per-item working directory where the primary data is staged
under a canonical name (``data.npy``) and the script writes a canonical
``visualization.png``. No source rewriting, no cross-item glob hazard (each cwd
holds only that item's files). The codegen prompts tell the model to read
``data.npy`` from the working directory and save ``visualization.png``; a guard
(:func:`script_uses_canonical_input`) rejects a script that ignores the contract,
so a non-conforming script fails the caller's existing verify/retry loop rather
than silently mis-reading.

Marker parsing stays with each caller (curve-fitting and image analysis use
different markers and post-processing), so this helper only owns the part that
was actually duplicated and fragile.
"""

from pathlib import Path

import numpy as np

DATA_NAME = "data.npy"
VIZ_NAME = "visualization.png"


def script_uses_canonical_input(script: str, data_name: str = DATA_NAME) -> bool:
    """True if the generated script references the canonical input filename.

    Cheap guard against a script that hardcodes some other path instead of
    reading the staged ``data.npy`` from the working directory."""
    return bool(script) and data_name in script


def stage_and_run(executor, script, primary_array, item_dir, *,
                  data_name: str = DATA_NAME, viz_name: str = VIZ_NAME,
                  aux: dict = None) -> dict:
    """Stage ``primary_array`` as ``data_name`` in ``item_dir`` and run ``script``
    VERBATIM there (working_dir=item_dir), then collect the canonical viz.

    ``aux`` (optional) maps extra canonical filenames -> arrays to stage alongside
    the primary (e.g. weights). Returns a dict with the raw executor result, its
    stdout/status, and the located visualization path/bytes — the caller parses
    its own stdout marker and assembles its result dict.
    """
    item_dir = Path(item_dir)
    item_dir.mkdir(parents=True, exist_ok=True)
    np.save(item_dir / data_name, primary_array)
    for name, arr in (aux or {}).items():
        np.save(item_dir / name, arr)

    viz = item_dir / viz_name
    viz.unlink(missing_ok=True)   # clear any stale viz before the run

    exec_res = executor.execute_script(script, working_dir=str(item_dir))

    has_viz = viz.exists()
    return {
        "exec": exec_res,
        "status": exec_res.get("status"),
        "stdout": exec_res.get("stdout", "") or "",
        "stderr": exec_res.get("stderr", "") or "",
        "visualization_path": str(viz) if has_viz else None,
        "visualization_bytes": viz.read_bytes() if has_viz else None,
        "item_dir": str(item_dir),
    }
