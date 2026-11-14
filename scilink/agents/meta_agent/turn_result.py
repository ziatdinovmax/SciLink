"""What one meta turn did, as the ``run_task`` contract's shape.

The meta has no ``run_task`` — it is the top-level caller — so a
programmatic caller of ``chat()`` (headless ``-p``) had only the reply text
to go by, and called a turn a success whenever there was one: a turn whose
every delegation failed still exited 0. The delegation ledger knows better.
This reads the entries the turn worked on and gives the same fields the
specialists' ``run_task`` returns, with a status derived from them.
"""

from __future__ import annotations

from typing import Any, Dict, List

# A delegation that produced a result. Anything else ("error", "failed",
# "interrupted", "cancelled", a "running" left by an aborted turn) did not.
_PRODUCED = ("success", "partial")


def _unique(items) -> List[Any]:
    seen, out = set(), []
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def turn_status(reply: str, entries: List[Dict[str, Any]], *, chat_error=None,
                hit_iteration_cap: bool = False) -> str:
    """``success`` · ``partial`` · ``error`` for one meta turn.

    ``error``: the turn itself failed (the chat loop raised, hit its
    iteration cap, or returned nothing), or it delegated and no delegation
    produced a result. ``partial``: some delegations produced a result and
    some did not — the reply may well account for it (a retry that worked),
    which the ledger cannot tell, so it is not called an error. ``success``:
    every delegation produced a result, or the turn needed none.
    """
    if chat_error or hit_iteration_cap or not (reply or "").strip():
        return "error"
    failed = [e for e in entries if str(e.get("status", "")).lower() not in _PRODUCED]
    if not failed:
        return "success"
    return "partial" if len(failed) < len(entries) else "error"


def ledger_snapshot(agent) -> Dict[Any, Any]:
    """The ledger as it stands before a turn: ``{index: completed_at}``.

    Taken AFTER the sweep ``chat()`` itself starts with: a restored session
    can hold a delegation an earlier, stopped session left ``running``, and
    the sweep closes it at the start of the next turn — it was then charged
    to that turn as a failed delegation. Sweeping first makes it part of the
    "before". Idempotent, so ``chat()`` sweeping again finds nothing.
    """
    sweep = getattr(agent, "_sweep_interrupted_delegations", None)
    if callable(sweep):
        sweep()
    return {e.get("index"): e.get("completed_at")
            for e in (getattr(agent, "_delegation_ledger", None) or []) if isinstance(e, dict)}


def turn_entries(ledger: List[Dict[str, Any]], before: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """The ledger entries one turn worked on: the ones it added, and the
    earlier ones it closed again (a resumed fan-out reopens its entries in
    place, so position says nothing). Read by CHANGE against the snapshot —
    never by comparing timestamps, which misattributes entries on a host
    whose clock is behind the one that wrote the ledger."""
    return [e for e in ledger if isinstance(e, dict)
            and (e.get("index") not in before or e.get("completed_at") != before[e.get("index")])]


def turn_result(agent, task: str, reply: str,
                before: Dict[Any, Any] = None) -> Dict[str, Any]:
    """The structured result of one turn; ``before`` is the
    :func:`ledger_snapshot` taken before ``chat()`` was called (none: every
    entry is the turn's)."""
    ledger = list(getattr(agent, "_delegation_ledger", None) or [])
    entries = turn_entries(ledger, before or {})
    chat_error = getattr(agent, "_last_chat_error", None)
    cap = bool(getattr(agent, "_last_chat_hit_iter_cap", False))
    status = turn_status(reply, entries, chat_error=chat_error, hit_iteration_cap=cap)

    from ..exp_agents._verification_record import did_nothing, merge_verification
    failed = [e for e in entries if str(e.get("status", "")).lower() not in _PRODUCED]
    warnings = [f"Delegation #{e.get('index')} ({e.get('mode')}) did not complete "
                f"(status={e.get('status')})" + (f": {e['error']}" if e.get("error") else ".")
                for e in failed]
    warnings += [w for e in entries for w in (e.get("warnings") or [])]
    # A delegation that ran no analysis is a success by its status (the
    # specialist replied) and has nothing to show: said here, since neither
    # the status nor the files say it.
    warnings += [f"Delegation #{e.get('index')} (analysis) ran no analysis."
                 for e in entries if e not in failed and e.get("mode") == "analysis"
                 and did_nothing(e.get("verification"))]

    result: Dict[str, Any] = {
        "status": status,
        "task": task,
        "summary": reply or "",
        # A delegation that produced nothing has no results to list: what it
        # left on disk is not one.
        "files_produced": _unique(f for e in entries if e not in failed
                                  for f in (e.get("files_produced") or [])),
        "feature_tables": _unique(f for e in entries if e not in failed
                                  for f in (e.get("feature_tables") or [])),
        "key_findings": _unique(k for e in entries if e not in failed
                                for k in (e.get("key_findings") or [])),
        "suggested_followups": _unique(s for e in entries
                                       for s in (e.get("suggested_followups") or [])),
        "warnings": _unique(warnings),
        "delegations": len(entries),
        "delegation_results": [
            {"index": e.get("index"), "mode": e.get("mode"), "label": e.get("label") or "",
             "status": e.get("status"), "error": e.get("error"),
             "analysis_ids": e.get("analysis_ids") or [],
             "verification": e.get("verification")}
            for e in entries],
    }
    verification = merge_verification((e.get("index"), e.get("verification")) for e in entries)
    if verification is not None:
        result["verification"] = verification
    if status == "error":
        result["error"] = (chat_error or (reply if cap else None)
                           or "; ".join(str(e.get("error")) for e in failed if e.get("error"))
                           or ("The turn produced no reply." if not (reply or "").strip()
                               else f"No delegation produced a result ({len(failed)} failed)."))
    return result
