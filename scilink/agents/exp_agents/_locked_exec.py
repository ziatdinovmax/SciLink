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
from typing import Optional

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


def escalate_timeouts(attempt, *, base_timeout: int, timed_out, logger=None,
                      remaining_s=None, escalations: Optional[int] = None):
    """The timeout policy every analysis agent runs generated code under
    (#699): a script that is merely SLOW gets more time before anyone calls
    it broken. ``attempt(timeout_s)`` runs the same script once under that
    limit and returns whatever the caller's runner returns; ``timed_out(out)``
    says whether that outcome was a timeout. On a timeout the same script is
    run again with ``TIMEOUT_GROWTH`` times the limit, up to
    ``TIMEOUT_ESCALATIONS`` retries bounded by ``TIMEOUT_HARD_CAP_S``; any
    other outcome is returned at once (a genuine error is the correction
    loop's job). The final, still-timed-out outcome is returned unchanged so
    the correction loop sees the standard message. ``remaining_s`` (a
    callable giving the seconds left on the run's deadline and the loop's
    budget, or a number) clamps the RETRIES to what is left — the first
    limit is never clamped, a script that started finishes — and ``escalations`` (default ``TIMEOUT_ESCALATIONS``; 0 for a
    strict replay, which must fail fast) bounds the retries. Returns
    ``(out, timeout_s)`` — the outcome and the limit it was produced under,
    which a locked replay of the script may start from."""
    log = logger or _LOG
    budget = TIMEOUT_ESCALATIONS if escalations is None else max(0, int(escalations))

    def left():
        # the seconds left as they are (the message says them), and as a
        # limit (never under 1 s: a 0 s limit would be no run at all)
        r = remaining_s() if callable(remaining_s) else remaining_s
        return (None, None) if r is None else (max(0, int(r)), max(1, int(r)))
    # the FIRST limit is never clamped: the run budget is soft — a script that
    # started finishes (RunBudget's promise) — only the retries are bounded
    current = int(base_timeout)
    out = None
    for esc in range(budget + 1):
        out = attempt(current)
        if not timed_out(out):
            return out, current
        grown = min(int(current * TIMEOUT_GROWTH), TIMEOUT_HARD_CAP_S)
        r_said, r = left()
        next_timeout = grown if r is None else min(grown, r)
        if next_timeout <= current or esc >= budget:
            if esc >= budget:
                why = (f"escalation budget exhausted ({budget} retr{'y' if budget == 1 else 'ies'})" if budget
                       else "no escalation on this clock")
            elif r is not None and next_timeout >= r:
                why = f"the time left ({r_said}s, the run's deadline or the loop's budget) allows no longer retry"
            else:
                why = f"the {TIMEOUT_HARD_CAP_S}s cap allows no longer retry"
            log.warning(f"    ⏱  Timed out at {current}s; {why} — handing the timeout to the correction loop.")
            return out, current
        log.warning(
            f"    ⏱  Script timed out at {current}s — retrying same script "
            f"with {next_timeout}s (escalation {esc + 1}/{budget})"
        )
        current = next_timeout
    return out, current


def stage_and_run_adaptive(executor, script, primary_array, item_dir, *,
                           aux: dict = None, metadata: dict = None,
                           logger=None) -> dict:
    """`stage_and_run` under ``escalate_timeouts``: starts at the executor's
    configured timeout; a timed-out script is retried with a doubled limit,
    up to the policy's cap; any other failure is returned as-is for the
    caller's script-correction loop."""
    def attempt(timeout_s: int) -> dict:
        return stage_and_run(executor, script, primary_array, item_dir,
                             aux=aux, metadata=metadata, timeout=timeout_s)

    def timed_out(run: dict) -> bool:
        return run["status"] != "success" and "timed out" in (run["exec"].get("message") or "").lower()
    run, _ = escalate_timeouts(attempt, base_timeout=int(executor.timeout), timed_out=timed_out, logger=logger)
    return run
