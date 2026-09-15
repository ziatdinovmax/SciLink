"""Read-only telemetry snapshot of a meta (Explore) session.

Aggregates state the agents already persist — the meta's delegation ledger,
the specialist orchestrators' counters, each worker agent's ``action_history``
(the uniform ``_log_action`` audit trail, with full input / result / rationale
per action), and the analysis sub-agents' ``analysis_results.json`` reasoning
— into one plain dict for the Telemetry UI tab.

The reader is LLM-free, stdlib-only, and never raises: a malformed file
degrades that one entry rather than breaking the tab.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DETAILED_MAX = 2000  # cap on the analysis detailed_analysis reasoning text

# Friendly names for the known worker state files; anything else falls back to
# a title-cased form of the file stem (so a new agent still shows up sanely).
_AGENT_LABELS = {
    "curve_fitting": "Curve Fitting",
    "bo": "Bayesian Optimization",
    "scalarizer": "Scalarizer",
}


def _agent_label(agent_type: Optional[str], stem: str) -> str:
    """Human-readable agent name from ``agent_type`` or the state-file stem."""
    if isinstance(agent_type, str) and agent_type.strip():
        key = agent_type.strip().lower()
        return _AGENT_LABELS.get(key, agent_type.strip())
    base = stem[:-6] if stem.endswith("_state") else stem  # "bo_state" -> "bo"
    return _AGENT_LABELS.get(base, base.replace("_", " ").title())


def _action_status(result: Any) -> str:
    """Best-effort outcome label for one action_history entry's ``result``.

    BO / Scalarizer results carry an explicit ``status``; curve-fitting results
    do not — for those a completed action is reported as ``ok``.
    """
    if isinstance(result, dict):
        status = result.get("status")
        if isinstance(status, str) and status:
            return status
        if result.get("error"):
            return "error"
        return "ok"
    return "ok" if result else "-"


def _specialist_of(path: Path) -> str:
    """Which specialist subtree (``analysis`` / ``planning``) a file sits in."""
    parts = set(path.parts)
    if "analysis" in parts:
        return "analysis"
    if "planning" in parts:
        return "planning"
    return "other"


def _worker_telemetry(state_path: Path) -> Optional[Dict[str, Any]]:
    """One agent row from a ``*_state.json`` file, or None if it has no
    ``action_history`` (so non-agent ``*_state.json`` files are skipped).

    Each action carries the full ``input`` / ``result`` / ``rationale`` /
    ``feedback`` so the UI can render a detailed per-action breakdown.
    """
    try:
        data = json.loads(state_path.read_text())
    except Exception:  # noqa: BLE001 - a bad state file must not break the tab
        return None
    if not isinstance(data, dict):
        return None
    history = data.get("action_history")
    if not isinstance(history, list) or not history:
        return None

    actions: List[Dict[str, Any]] = []
    by_type: Dict[str, int] = {}
    outcomes = {"success": 0, "error": 0, "other": 0}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or "?")
        status = _action_status(entry.get("result"))
        by_type[action] = by_type.get(action, 0) + 1
        if status in ("success", "ok"):
            outcomes["success"] += 1
        elif status == "error":
            outcomes["error"] += 1
        else:
            outcomes["other"] += 1
        actions.append({
            "timestamp": entry.get("timestamp"),
            "action": action,
            "status": status,
            "rationale": entry.get("rationale"),
            "input": entry.get("input"),
            "result": entry.get("result"),
            "feedback": entry.get("feedback"),
        })

    stamps = [a["timestamp"] for a in actions if a["timestamp"]]
    return {
        "specialist": _specialist_of(state_path),
        "name": _agent_label(data.get("agent_type"), state_path.stem),
        "status": data.get("status"),
        "action_count": len(actions),
        "actions_by_type": by_type,
        "outcomes": outcomes,
        "first_timestamp": min(stamps) if stamps else None,
        "last_timestamp": max(stamps) if stamps else None,
        "actions": actions,
        "source_file": str(state_path),
    }


def _analysis_reports(base_dir: Path) -> List[Dict[str, Any]]:
    """Per-analysis scientific reasoning from each ``analysis_results.json`` —
    the detailed_analysis narrative and the extracted scientific claims."""
    reports: List[Dict[str, Any]] = []
    adir = base_dir / "analysis"
    if not adir.is_dir():
        return reports
    for ar in sorted(adir.rglob("analysis_results.json")):
        try:
            data = json.loads(ar.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict):
            continue
        claims: List[Dict[str, Any]] = []
        for c in data.get("scientific_claims") or []:
            if isinstance(c, dict):
                claims.append({"claim": c.get("claim"),
                               "impact": c.get("scientific_impact")})
            elif c:
                claims.append({"claim": str(c), "impact": None})
        detailed = str(data.get("detailed_analysis") or "")
        if len(detailed) > _DETAILED_MAX:
            detailed = detailed[:_DETAILED_MAX] + "…"
        reports.append({
            "analysis_id": ar.parent.name,
            "status": data.get("status"),
            "detailed_analysis": detailed,
            "claims": claims,
            "output_dir": str(ar.parent),
            "report_file": str(ar),
        })
    return reports


def _maybe_json(value: Any) -> Any:
    """Parse a JSON string into a dict/list; leave anything else as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value
    return value


def _extract_tool_calls(messages: list) -> List[Dict[str, Any]]:
    """Ordered tool calls in one chat history, each matched to its result.

    Handles the OpenAI-style ``tool_calls`` on assistant messages
    (``function.name`` / ``function.arguments`` JSON string), with a light
    fallback for a flattened ``{name, args}`` shape. Each entry carries the
    parsed ``args``, the parsed tool ``result`` (matched by ``tool_call_id``),
    and a ``status`` derived from that result — ``pending`` when the call has
    no result yet (a tool still running, or a turn in flight).
    """
    # First pass: tool_call_id -> result payload, from role:"tool" messages.
    results: Dict[str, Any] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid:
                results[tcid] = _maybe_json(m.get("content"))

    seq: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = fn.get("name")
            if not name:
                continue
            args = _maybe_json(fn.get("arguments", fn.get("args")))
            has_result = tc.get("id") in results
            result = results.get(tc.get("id"))
            seq.append({
                "tool": name,
                "args": args if isinstance(args, dict) else {"_value": args},
                "result": result,
                "status": _action_status(result) if has_result else "pending",
            })
    return seq


def _tool_sequence(base_dir: Path, meta_agent: Any) -> Dict[str, Dict[str, Any]]:
    """Per-agent ordered tool-call sequence.

    Prefers each agent's live in-memory ``messages`` — which grows per tool
    iteration mid-turn, so the sequence updates in real time — and falls back
    to the persisted ``chat_history.json`` (written only at end of turn) when
    no live agent object is available (e.g. a child not yet re-created after a
    resume). Pure read: a snapshot copy guards against the background chat
    thread mutating ``messages`` concurrently.

    Each layer's entry carries its ``calls`` and the ``source`` file path (the
    meta's at the session root, each specialist's in its sub-dir).
    """
    children = getattr(meta_agent, "_children", {}) or {}
    live = {"meta": meta_agent, "analysis": children.get("analysis"),
            "planning": children.get("planning")}
    out: Dict[str, Dict[str, Any]] = {}
    for layer, sub in (("meta", ""), ("analysis", "analysis"),
                        ("planning", "planning")):
        path = (base_dir / sub / "chat_history.json" if sub
                else base_dir / "chat_history.json")
        calls: Optional[List[Dict[str, Any]]] = None

        obj = live.get(layer)
        msgs = getattr(obj, "messages", None) if obj is not None else None
        if isinstance(msgs, list):
            try:
                calls = _extract_tool_calls(list(msgs))  # snapshot, then read
            except Exception:  # noqa: BLE001 - concurrent mutation; next poll
                calls = None

        if calls is None and path.is_file():  # fall back to the persisted file
            try:
                messages = json.loads(path.read_text())
                if isinstance(messages, list):
                    calls = _extract_tool_calls(messages)
            except Exception:  # noqa: BLE001
                calls = None

        if calls:
            out[layer] = {"calls": calls, "source": str(path)}
    return out


def _csv_row_count(path: Any) -> int:
    """Row count of a CSV, or 0 if absent / unreadable."""
    try:
        import pandas as pd
        if path and Path(path).exists():
            return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _specialists(meta_agent: Any) -> Dict[str, Any]:
    """Per-specialist counters — the same fields the meta's
    ``_session_state_summary`` reads, defensively via ``getattr``."""
    children = getattr(meta_agent, "_children", {}) or {}
    out: Dict[str, Any] = {}
    for name in ("analysis", "planning"):
        child = children.get(name)
        if child is None:
            out[name] = {"instantiated": False}
            continue
        info: Dict[str, Any] = {
            "instantiated": True,
            "message_count": getattr(child, "message_count", 0),
        }
        if name == "analysis":
            info["analyses_run"] = len(
                getattr(child, "analysis_results", []) or [])
        else:
            info["optimization_targets"] = list(
                getattr(child, "expected_target_columns", []) or [])
            info["bo_data_points"] = _csv_row_count(
                getattr(child, "bo_data_path", None))
        out[name] = info
    return out


def collect_session_telemetry(meta_agent: Any) -> Dict[str, Any]:
    """Read-only telemetry snapshot of a meta session. Never raises.

    Works for a live session (ledger in memory) and a resumed one (ledger
    restored from the checkpoint; worker state files re-read from disk).
    """
    try:
        meta_mode = getattr(meta_agent, "meta_mode", None)
        meta_mode_str = getattr(meta_mode, "value", None) or str(meta_mode)
        # No configured base_dir -> scan nothing (never fall back to cwd, which
        # would rglob the whole tree).
        base_dir_raw = getattr(meta_agent, "base_dir", None)
        ledger = list(getattr(meta_agent, "_delegation_ledger", []) or [])

        delegations = [{
            "index": e.get("index"),
            "mode": e.get("mode"),
            "label": (e.get("label") or "").strip()
            or " ".join(str(e.get("task") or "").split())[:60],
            "status": e.get("status"),
            "context_from": e.get("context_from") or [],
            "timestamp": e.get("timestamp"),
            "completed_at": e.get("completed_at"),
            "files": len(e.get("files_produced") or []),
            "feature_tables": len(e.get("feature_tables") or []),
            "warnings": len(e.get("warnings") or []),
        } for e in ledger]

        agents: List[Dict[str, Any]] = []
        analysis_reports: List[Dict[str, Any]] = []
        tool_sequence: Dict[str, List[Dict[str, Any]]] = {}
        if base_dir_raw and Path(base_dir_raw).is_dir():
            base = Path(base_dir_raw)
            for sp in sorted(base.rglob("*_state.json")):
                row = _worker_telemetry(sp)
                if row:
                    agents.append(row)
            analysis_reports = _analysis_reports(base)
            tool_sequence = _tool_sequence(base, meta_agent)

        # Which sub-agents each specialist (mode) ended up using — for the
        # dependency graph's per-mode sub-agent annotation.
        sub_agents: Dict[str, List[str]] = {}
        for a in agents:
            names = sub_agents.setdefault(a["specialist"], [])
            if a["name"] not in names:
                names.append(a["name"])

        return {
            "meta": {
                "meta_mode": meta_mode_str,
                "session_dir": str(base_dir_raw or ""),
                "delegations_total": len(ledger),
                "delegations": delegations,
            },
            "specialists": _specialists(meta_agent),
            "agents": agents,
            "sub_agents": sub_agents,
            "analysis_reports": analysis_reports,
            "tool_sequence": tool_sequence,
        }
    except Exception as e:  # noqa: BLE001 - telemetry must never break the UI
        logger.warning(f"telemetry collection failed: {e}")
        return {"meta": {}, "specialists": {}, "agents": [], "sub_agents": {},
                "analysis_reports": [], "tool_sequence": {}, "error": str(e)}
