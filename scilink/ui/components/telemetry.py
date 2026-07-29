"""Telemetry tab — live snapshot of meta + specialist + worker-agent state.

Explore (meta) mode only. A compact, status-colored dependency graph of the
delegation ledger (meta -> delegations, annotated with the sub-agent each mode
selected, plus the `context_from` provenance edges) sits on top; the delegation
ledger and a per-agent tool-call sequence back it underneath. Refreshes on the
sidebar delegation tree's cadence (2s while a chat task runs).
"""

from pathlib import Path

import streamlit as st

from scilink.agents.meta_agent.telemetry import collect_session_telemetry

# Same palette as the sidebar delegation tree, for consistency.
_STATUS_FILL = {
    "success": "#3fb950",   # green
    "error": "#f85149",     # red
    "running": "#d29922",   # amber
}
_DEFAULT_FILL = "#8893a5"   # grey — unknown / not-yet-finished


def render_telemetry_tab() -> None:
    """Render the Explore-mode Telemetry tab."""
    task = st.session_state.get("chat_task")
    _interval = "2s" if (task is not None
                         and getattr(task, "is_running", False)) else None

    @st.fragment(run_every=_interval)
    def _panel() -> None:
        agent = st.session_state.get("agent")
        if agent is None or not hasattr(agent, "_delegation_ledger"):
            st.info("Telemetry is available once an Explore session is running.")
            return

        tel = collect_session_telemetry(agent)
        meta = tel.get("meta", {})
        agents = tel.get("agents", [])
        delegations = meta.get("delegations", [])

        st.markdown(
            f"**{str(meta.get('meta_mode', '—')).title()}**  ·  "
            f"{meta.get('delegations_total', 0)} delegations  ·  "
            f"{len(agents)} worker agents"
        )
        st.caption(f"Session: {meta.get('session_dir', '—')}")

        if not delegations:
            st.info("No delegations yet — describe a goal and the meta "
                    "routes it to a specialist.")
            return

        # ── Dependency graph ─────────────────────────────────────────
        st.graphviz_chart(
            _delegation_graph_dot(meta, tel.get("sub_agents", {})),
            width="content",
        )
        st.caption("Grey edge = meta dispatched the delegation.  "
                   "Blue edge = a delegation's result fed the next as context.")

        # ── Delegation ledger ────────────────────────────────────────
        st.subheader("Delegation ledger")
        _ledger_table(delegations)

        # ── Tool sequence ────────────────────────────────────────────
        st.subheader("Tool sequence")
        st.caption("Every tool call each agent's LLM made, in order.")
        _tool_sequence_section(tel.get("tool_sequence", {}))

    _panel()


# ── dependency graph ─────────────────────────────────────────────────

def _dot_escape(text) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _truncate(text, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _delegation_graph_dot(meta: dict, sub_agents: dict) -> str:
    """DOT string: meta-agent root -> delegation nodes (colored by status,
    annotated with the sub-agent(s) the mode used), with dispatch edges and
    `context_from` provenance edges. Kept compact via tight spacing + a size
    cap so it does not dominate the tab."""
    dels = meta.get("delegations", [])
    out = [
        "digraph telemetry {",
        '  rankdir=TB; bgcolor="transparent"; pad=0.15;',
        '  size="7,4.5"; ratio=compress; ranksep=0.32; nodesep=0.22;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", '
        'fontsize=9, margin="0.11,0.05", color="#30363d", fontcolor="white"];',
        '  edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
        f'  meta [label="Meta-agent ({_dot_escape(str(meta.get("meta_mode", "")).title())})", '
        'fillcolor="#30363d"];',
    ]
    for d in dels:
        idx = d.get("index")
        fill = _STATUS_FILL.get(d.get("status"), _DEFAULT_FILL)
        subs = sub_agents.get(d.get("mode"), [])
        sub_line = ("\\n↳ " + _dot_escape(_truncate(", ".join(subs), 32))
                    if subs else "")
        label = (f'#{idx} · {_dot_escape(d.get("mode", "?"))}\\n'
                 f'{_dot_escape(_truncate(d.get("label", ""), 24))}{sub_line}')
        out.append(f'  d{idx} [label="{label}", fillcolor="{fill}"];')
    for d in dels:                                   # dispatch edges
        out.append(f'  meta -> d{d.get("index")} [color="#8893a5"];')
    for d in dels:                                   # context-provenance edges
        for src in d.get("context_from", []):
            out.append(f'  d{src} -> d{d.get("index")} '
                       f'[color="#58a6ff", penwidth=2.0, label="ctx"];')
    out.append("}")
    return "\n".join(out)


# ── tables ───────────────────────────────────────────────────────────

def _short_time(ts) -> str:
    """ISO timestamp -> HH:MM:SS (keeps the table narrow)."""
    if not ts:
        return ""
    s = str(ts)
    return s.split("T", 1)[1][:8] if "T" in s else s


def _ledger_table(delegations: list) -> None:
    import pandas as pd

    rows = []
    for d in delegations:
        cf = d.get("context_from") or []
        rows.append({
            "#": d.get("index"),
            "specialist": d.get("mode"),
            "task": d.get("label"),
            "status": d.get("status"),
            "context from": ", ".join(f"#{n}" for n in cf) if cf else "",
            "files": d.get("files", 0),
            "feature tables": d.get("feature_tables", 0),
            "warnings": d.get("warnings", 0),
            "started": _short_time(d.get("timestamp")),
            "completed": _short_time(d.get("completed_at")),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── tool sequence ────────────────────────────────────────────────────

_LAYER_LABELS = [
    ("meta", "Meta-agent"),
    ("analysis", "Analysis specialist"),
    ("planning", "Planning specialist"),
]


def _type_name(v) -> str:
    """Short type label for one value — no content, just the type."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    return type(v).__name__


def _type_summary(obj, limit: int = 220) -> str:
    """Type signature of a value — for a dict, its keys mapped to value types;
    otherwise the bare type. Shows the shape without the actual content."""
    if obj is None:
        return "—"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        s = "{" + ", ".join(f"{k}: {_type_name(v)}"
                             for k, v in obj.items()) + "}"
    elif isinstance(obj, list):
        s = f"list[{len(obj)}]"
    else:
        s = _type_name(obj)
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _tool_sequence_section(sequence: dict) -> None:
    """Per-agent ordered tool calls — input/output types and whether each tool
    worked, derived from its result."""
    import pandas as pd

    shown = False
    for key, label in _LAYER_LABELS:
        entry = sequence.get(key) or {}
        calls = entry.get("calls") or []
        if not calls:
            continue
        shown = True
        st.markdown(f"**{label}** — {len(calls)} tool call(s)")
        rows = [{
            "#": i,
            "tool": c.get("tool"),
            "input": _type_summary(c.get("args")),
            "output": _type_summary(c.get("result")),
            "status": c.get("status", "—"),
        } for i, c in enumerate(calls, 1)]
        st.dataframe(pd.DataFrame(rows), width="stretch",
                     hide_index=True)
        src = entry.get("source")
        if src and Path(src).is_file():
            try:
                history = Path(src).read_text()
            except Exception:  # noqa: BLE001
                history = None
            if history:
                st.download_button(
                    "Download full history",
                    data=history,
                    file_name=f"{key}_chat_history.json",
                    mime="application/json",
                    key=f"telemetry_dl_{key}",
                )
    if not shown:
        st.caption("No tool calls recorded yet.")


