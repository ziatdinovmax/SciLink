"""What one meta turn did, as the ``run_task`` contract's shape.

The meta has no ``run_task`` — it is the top-level caller — so a
programmatic caller of ``chat()`` (headless ``-p``) had only the reply text
to go by, and called a turn a success whenever there was one: a turn whose
every delegation failed still exited 0. The delegation ledger knows better.
This reads the entries the turn added and gives the same fields the
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


def turn_result(agent, task: str, reply: str, since: int = 0) -> Dict[str, Any]:
    """The structured result of the turn that added ledger entries after
    position ``since`` (the ledger's length before ``chat()`` was called)."""
    ledger = list(getattr(agent, "_delegation_ledger", None) or [])
    entries = [e for e in ledger[since:] if isinstance(e, dict)]
    chat_error = getattr(agent, "_last_chat_error", None)
    cap = bool(getattr(agent, "_last_chat_hit_iter_cap", False))
    status = turn_status(reply, entries, chat_error=chat_error, hit_iteration_cap=cap)

    failed = [e for e in entries if str(e.get("status", "")).lower() not in _PRODUCED]
    warnings = [f"Delegation #{e.get('index')} ({e.get('mode')}) did not complete "
                f"(status={e.get('status')})" + (f": {e['error']}" if e.get("error") else ".")
                for e in failed]
    warnings += [w for e in entries for w in (e.get("warnings") or [])]

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
    from ..exp_agents._verification_record import merge_verification
    verification = merge_verification(e.get("verification") for e in entries)
    if verification is not None:
        result["verification"] = verification
    if status == "error":
        result["error"] = (chat_error or (reply if cap else None)
                           or "; ".join(str(e.get("error")) for e in failed if e.get("error"))
                           or "The turn produced no reply.")
    return result
