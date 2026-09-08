"""Meta-session delegation ledger, shaped for the web UI's Delegations tab.

The meta orchestrator keeps ``_delegation_ledger`` in memory (restored from
the checkpoint on resume). This module reads it — never writes — and emits
a compact, JSON-safe view: one row per delegation with the fields the tab
renders (specialist, label, status, context-flow edges, summary, findings,
files relative to the session, warnings, timings) plus the per-specialist
sub-agent annotation the Streamlit telemetry tab shows.

``ledger_signature`` is the cheap change detector the turn watcher polls
every fs tick; the full view is built only when it changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SUMMARY_MAX = 1200
_TASK_MAX = 400
_FINDINGS_MAX = 12
_FILES_MAX = 40
_WARNINGS_MAX = 12


def ledger_of(agent: Any) -> List[Dict[str, Any]]:
    """The live ledger list, or [] for a non-meta agent."""
    ledger = getattr(agent, "_delegation_ledger", None)
    return list(ledger) if isinstance(ledger, list) else []


def ledger_signature(agent: Any) -> Tuple:
    """Changes whenever a delegation is opened, closed, or its status /
    summary changes — cheap enough to compute on every watcher tick."""
    return tuple(
        (e.get("index"), e.get("status"), e.get("completed_at"),
         len(e.get("summary") or ""), len(e.get("files_produced") or []))
        for e in ledger_of(agent))


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _rel(path: Any, session_dir: Optional[str]) -> str:
    """Session-relative path when inside the session (so the tab can open it
    in the Files explorer), else the path as recorded."""
    p = str(path or "")
    if not session_dir or not p:
        return p
    try:
        return str(Path(p).resolve().relative_to(Path(session_dir).resolve()))
    except (ValueError, OSError):
        return p


def _sub_agents(session_dir: Optional[str]) -> Dict[str, List[str]]:
    """Which worker agents each specialist used, from the ``*_state.json``
    files the workers persist (same source as the telemetry reader, but
    reading only the ``agent_type`` field)."""
    out: Dict[str, List[str]] = {}
    if not session_dir:
        return out
    base = Path(session_dir)
    if not base.is_dir():
        return out
    labels = {"curve_fitting": "Curve Fitting", "bo": "Bayesian Optimization",
              "scalarizer": "Scalarizer"}
    try:
        for sp in sorted(base.rglob("*_state.json")):
            try:
                data = json.loads(sp.read_text())
            except Exception:  # noqa: BLE001 - one bad file must not hide the rest
                continue
            if not isinstance(data, dict) or not data.get("action_history"):
                continue
            parts = set(sp.parts)
            specialist = ("analysis" if "analysis" in parts
                          else "planning" if "planning" in parts
                          else "simulation" if "simulation" in parts
                          else "other")
            raw = data.get("agent_type")
            stem = sp.stem[:-6] if sp.stem.endswith("_state") else sp.stem
            name = (labels.get(str(raw).strip().lower(), str(raw).strip())
                    if isinstance(raw, str) and raw.strip()
                    else labels.get(stem, stem.replace("_", " ").title()))
            names = out.setdefault(specialist, [])
            if name not in names:
                names.append(name)
    except OSError:
        pass
    return out


def delegation_view(agent: Any, session_dir: Optional[str] = None
                    ) -> Dict[str, Any]:
    """The Delegations-tab payload: ``{"delegations": [...], "sub_agents":
    {...}}``. Never raises — a malformed entry degrades to what it has."""
    rows: List[Dict[str, Any]] = []
    for e in ledger_of(agent):
        try:
            fan = e.get("fanout")
            rows.append({
                "index": e.get("index"),
                "mode": e.get("mode") or "?",
                "label": (e.get("label") or "").strip()
                or _clip(e.get("task"), 60),
                "task": _clip(e.get("task"), _TASK_MAX),
                "status": e.get("status") or "running",
                "context_from": [int(x) for x in (e.get("context_from") or [])
                                 if str(x).isdigit()],
                "informed_by": list(e.get("informed_by") or []),
                "fanout": bool(fan),
                "fanout_group": (fan.get("id") or fan.get("group")
                                 if isinstance(fan, dict) else None),
                "labels": list(e.get("labels") or []),   # fusion inputs
                "timestamp": e.get("timestamp"),
                "completed_at": e.get("completed_at"),
                "summary": _clip(e.get("summary"), _SUMMARY_MAX),
                "key_findings": [str(k) for k in
                                 (e.get("key_findings") or [])[:_FINDINGS_MAX]],
                "files_produced": [_rel(p, session_dir) for p in
                                   (e.get("files_produced") or [])[:_FILES_MAX]],
                "n_feature_tables": len(e.get("feature_tables") or []),
                "warnings": [str(w) for w in
                             (e.get("warnings") or [])[:_WARNINGS_MAX]],
                "error": (str(e.get("error")) if e.get("error") else None),
                "timed_out": bool(e.get("timed_out")),
                "resumed": bool(e.get("resumed_from_interruption")),
            })
        except Exception:  # noqa: BLE001
            rows.append({"index": e.get("index"), "mode": "?", "label": "?",
                         "status": e.get("status") or "?",
                         "context_from": [], "files_produced": [],
                         "key_findings": [], "warnings": []})
    return {"delegations": rows, "sub_agents": _sub_agents(session_dir)}
