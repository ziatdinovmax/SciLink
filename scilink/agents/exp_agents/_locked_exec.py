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

import json
import logging
import os
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)

DATA_NAME = "data.npy"
VIZ_NAME = "visualization.png"
META_NAME = "metadata.json"
# Best-of-N anchor attempts run in per-attempt subdirs under this directory
# (inside the per-image working dir); winner files are promoted up, losers
# stay for audit. Consumers that walk output trees skip it by this name.
CANDIDATES_DIR_NAME = "_candidates"


def atomic_np_save(path, arr) -> None:
    """np.save with atomic publication: unique sibling temp + os.replace.

    Concurrent writers of identical content (best-of-N attempts staging the
    same auxiliary operand) become race-free: each writes its own temp file
    and the rename is atomic, so readers never see a torn file. The temp name
    ends in ``.npy`` so np.save appends nothing.
    """
    import threading

    path = Path(path)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}_{threading.get_ident()}.tmp.npy"
    )
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def script_uses_canonical_input(script: str, data_name: str = DATA_NAME) -> bool:
    """True if the generated script references the canonical input filename.

    Cheap guard against a script that hardcodes some other path instead of
    reading the staged ``data.npy`` from the working directory."""
    return bool(script) and data_name in script


def stage_and_run(executor, script, primary_array, item_dir, *,
                  data_name: str = DATA_NAME, viz_name: str = VIZ_NAME,
                  aux: dict = None, metadata: dict = None,
                  meta_name: str = META_NAME,
                  timeout: int | None = None) -> dict:
    """Stage ``primary_array`` as ``data_name`` in ``item_dir`` and run ``script``
    VERBATIM there (working_dir=item_dir), then collect the canonical viz.

    ``aux`` (optional) maps extra canonical filenames -> arrays to stage alongside
    the primary (e.g. weights). Returns a dict with the raw executor result, its
    stdout/status, and the located visualization path/bytes — the caller parses
    its own stdout marker and assembles its result dict.

    ``metadata`` (optional) — when a non-empty dict is given, it is written to
    ``item_dir / meta_name`` (default ``metadata.json``) so the generated script
    can read authoritative calibration (FOV, data_range, …) from a file instead
    of guessing. Default ``None`` is a no-op: no file is written and existing
    single/series behavior is unchanged.
    """
    item_dir = Path(item_dir)
    item_dir.mkdir(parents=True, exist_ok=True)
    np.save(item_dir / data_name, primary_array)
    for name, arr in (aux or {}).items():
        np.save(item_dir / name, arr)
    if metadata:
        try:
            with open(item_dir / meta_name, "w", encoding="utf-8") as _fh:
                json.dump(metadata, _fh, indent=2, default=str)
        except Exception:
            pass   # sidecar is best-effort; never block execution on it

    viz = item_dir / viz_name
    viz.unlink(missing_ok=True)   # clear any stale viz before the run

    exec_res = executor.execute_script(script, working_dir=str(item_dir),
                                       timeout=timeout)

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


def is_timeout_error(err) -> bool:
    """True when an attempt error is the executor's timeout message."""
    return "timed out" in str(err or "").lower()


def trailing_timeout_failures(attempts: list) -> int:
    """Count consecutive TRAILING attempts that failed on execution timeout.

    Used to tell a refit/rewrite that the recent failures were budget
    failures, not quality failures — a different-but-equally-slow approach
    would fail the same way.
    """
    n = 0
    for a in reversed(attempts or []):
        r = a.get("result") or {}
        if r.get("success") is False and is_timeout_error(r.get("error")):
            n += 1
        else:
            break
    return n


def should_escalate_timeout_model(base_script, attempt: int,
                                  max_attempts: int,
                                  consecutive_timeouts: int) -> bool:
    """Last-resort model escalation predicate (shared by curve + image).

    Fires ONLY on the final TWO corrections of a fresh fit (base_script
    None — locked-reuse items must never restructure) whose recent failures
    are >= 2 consecutive timeouts. Two chances, not one: a restructured
    script sometimes carries an introduced bug, and a crash resets the
    consecutive-timeout counter, so the very next correction naturally takes
    the plain debug path and can fix it (observed live)."""
    return (base_script is None and attempt >= max_attempts - 1
            and consecutive_timeouts >= 2)


# Adaptive-timeout policy for the per-item attempt loops. A subprocess that
# times out is rarely "broken code" — usually the fit just needs longer.
# Re-running the SAME script with 2× the timeout, up to a hard cap, avoids
# burning LLM tokens on a correction pass "fixing" code that wasn't wrong.
TIMEOUT_ESCALATIONS = 2
TIMEOUT_GROWTH = 2.0
TIMEOUT_HARD_CAP_S = 1800


def stage_and_run_adaptive(executor, script, primary_array, item_dir, *,
                           aux: dict = None, metadata: dict = None,
                           logger=None) -> dict:
    """`stage_and_run` with adaptive timeout escalation.

    Starts at the executor's configured timeout; on a timeout, retries the
    SAME script with a doubled timeout, up to ``TIMEOUT_ESCALATIONS`` retries
    bounded by ``TIMEOUT_HARD_CAP_S``. Non-timeout failures are returned
    as-is so the caller's script-correction loop handles them. The final
    (still timed-out) result is returned unchanged, so the correction LLM
    sees the standard "timed out" message.
    """
    log = logger or _LOG
    current = int(executor.timeout)
    run = None
    for esc in range(TIMEOUT_ESCALATIONS + 1):
        run = stage_and_run(executor, script, primary_array, item_dir,
                            aux=aux, metadata=metadata, timeout=current)
        if run["status"] == "success":
            return run
        msg = (run["exec"].get("message") or "").lower()
        if "timed out" not in msg:
            return run  # genuine script error — the correction loop's job
        next_timeout = min(int(current * TIMEOUT_GROWTH), TIMEOUT_HARD_CAP_S)
        if next_timeout <= current or esc >= TIMEOUT_ESCALATIONS:
            log.warning(
                f"    ⏱  Timed out at {current}s; escalation budget exhausted "
                f"({TIMEOUT_ESCALATIONS} retries, {TIMEOUT_HARD_CAP_S}s cap) "
                f"— handing the timeout to the correction loop."
            )
            return run
        log.warning(
            f"    ⏱  Script timed out at {current}s — retrying same script "
            f"with {next_timeout}s (escalation {esc + 1}/{TIMEOUT_ESCALATIONS})"
        )
        current = next_timeout
    return run
