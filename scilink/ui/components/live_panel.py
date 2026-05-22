"""Streamlit panel for the live-monitoring mode.

Renders one of two views depending on session state:

  Setup view  — when no live session is active. Shows a form for
                data-source path, skill picker (filtered to skills
                whose frontmatter declares a ``live_tick.tick_fn``),
                tick interval, and trigger toggles. "Start" button
                instantiates an ``AnalysisOrchestratorAgent`` and a
                :class:`LiveSession` and stashes both in
                ``st.session_state``.

  Dashboard   — when a live session is running. Layout: latest data
                plot + metric time-series chart on the left, decision
                feed on the right, controls (Interpret Now, Stop) at
                the bottom. Auto-refreshes via
                ``st.fragment(run_every="2s")`` — same primitive
                the existing chat-task polling uses
                (``app.py:539``).
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_live_panel() -> None:
    """Top-level renderer; called from ``app.py`` when ``app_mode == 'live'``."""
    st.header("Live Monitoring")
    st.caption(
        "Real-time agent interpretation of an in-progress measurement. "
        "The fast tick loop runs every couple of seconds; the LLM is "
        "invoked only when an interesting event fires."
    )

    if st.session_state.get("live_session") is None:
        _render_setup()
    else:
        _render_dashboard()


# ---------------------------------------------------------------------------
# Setup view
# ---------------------------------------------------------------------------


def _resolve_sidebar_config() -> tuple[str, str, bool]:
    """Read model / API key / consent from the standard sidebar fields.

    Matches the keys the sidebar writes: cfg_model_preset / cfg_model_custom
    (sidebar.py:213-222) and cfg_api_key / cfg_consent (lines 223, 267).
    """
    preset = st.session_state.get("cfg_model_preset", "")
    if preset == "Custom":
        model = st.session_state.get("cfg_model_custom", "") or ""
    else:
        model = preset or "claude-opus-4-6"
    api_key = st.session_state.get("cfg_api_key") or ""
    consent = bool(st.session_state.get("cfg_consent"))
    return model, api_key, consent


def _render_setup() -> None:
    # Sidebar prerequisites — same as analyze / plan / simulate modes:
    # CONSENT is the only hard gate at the UI level (sidebar.py:343).
    # The API key resolves through the standard chain at dispatch time:
    # sidebar input → ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY.
    model, api_key, consent = _resolve_sidebar_config()

    env_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    # Banner — only consent missing is strictly blocking; key missing is a soft warning
    if not consent:
        st.warning(
            "Before starting a live session, please check the "
            "**code-execution consent** box in the sidebar."
        )
    elif not api_key and not env_key:
        st.info(
            "No API key in the sidebar and no LLM env var set. You can "
            "still configure a session below, but starting it will fail "
            "until a key is provided."
        )
    else:
        key_source = "sidebar" if api_key else "environment"
        st.success(
            f"Sidebar config ready · model **{model}** · API key from "
            f"**{key_source}** · consent given."
        )

    # Skills with a live_tick block — filter to those that resolve to an importable
    # tick_fn at session-start time so the user only sees options that will work.
    skill_options = _discover_live_enabled_skills()

    with st.form("live_session_setup"):
        st.subheader("Live session — measurement setup")
        st.caption(
            "Configure the data source + skill + triggers below. Model and "
            "API key are read from the sidebar fields."
        )

        data_path = st.text_input(
            "Data file path",
            value="",
            placeholder="/path/to/in-progress-scan.csv",
            help="Path to a file your instrument writes to as it scans. "
                 "SciLink polls this file every tick.",
        )

        col1, col2 = st.columns(2)
        with col1:
            if skill_options:
                skill_label = st.selectbox(
                    "Tick skill",
                    options=[s["label"] for s in skill_options],
                    help="Skill whose live_tick function drives the metric.",
                )
                selected_skill = next(
                    s for s in skill_options if s["label"] == skill_label
                )
            else:
                st.warning(
                    "No skills with a `live_tick:` block found. The skill's "
                    "frontmatter needs `live_tick.tick_fn` set to a "
                    "module:function path."
                )
                selected_skill = None
        with col2:
            tick_interval = st.number_input(
                "Tick interval (seconds)",
                min_value=0.5, max_value=60.0, value=2.0, step=0.5,
                help="How often to poll the data source and run the tick function.",
            )

        st.markdown("**Source type**")
        source_kind = st.radio(
            "Data source", options=["mtime_poll", "append_only"],
            label_visibility="collapsed", horizontal=True,
            help=(
                "`mtime_poll` rereads the whole file each time it changes "
                "(rewriting instruments). `append_only` returns only the "
                "newly-appended bytes (incremental writers)."
            ),
        )

        with st.expander("Triggers (default: verdict change + new feature + reversal + heartbeat)"):
            t_verdict = st.checkbox("Verdict change", value=True)
            t_new_feature = st.checkbox("New feature appears", value=True)
            t_reversal = st.checkbox("Confidence reversal", value=True)
            t_heartbeat = st.checkbox("Heartbeat", value=True)
            heartbeat_sec = st.number_input(
                "Heartbeat interval (s)", min_value=10, max_value=600, value=60,
                disabled=not t_heartbeat,
            )
            t_threshold = st.checkbox("Confidence threshold cross", value=False)
            threshold_value = st.number_input(
                "Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                disabled=not t_threshold,
            )

        # Match the standard sidebar's pattern (sidebar.py:343): consent
        # is the only strict UI-level gate. Missing API key surfaces as an
        # error at dispatch time after env-var fallback fails.
        start = st.form_submit_button(
            "Start Live Session",
            type="primary",
            disabled=not consent,
            help=(
                "Check the code-execution consent box in the sidebar first."
                if not consent else None
            ),
        )

    if not start:
        return

    # Validation
    if not consent:
        st.error(
            "Code-execution consent required — check the sidebar checkbox first."
        )
        return
    # Resolve the actual key at dispatch time (sidebar input takes precedence
    # over env vars, matching sidebar.py:663's resolution order).
    resolved_key = api_key or env_key
    if not resolved_key:
        st.error(
            "No API key — enter one in the sidebar or set "
            "ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY in the "
            "environment."
        )
        return
    if not data_path:
        st.error("Data file path is required.")
        return
    if not Path(data_path).exists():
        st.warning(
            f"Path does not exist yet: {data_path}. The session will start "
            "and the tick loop will pick up the file once your instrument "
            "creates it."
        )
    if selected_skill is None:
        st.error("No live-tick-enabled skill available to drive the session.")
        return

    _spin_up_live_session(
        model=model,
        api_key=resolved_key,
        data_path=data_path,
        source_kind=source_kind,
        skill_name=selected_skill["name"],
        skill_domain=selected_skill["domain"],
        tick_interval_sec=float(tick_interval),
        triggers={
            "verdict_change": t_verdict,
            "new_feature": t_new_feature,
            "reversal": t_reversal,
            "heartbeat": t_heartbeat,
            "heartbeat_sec": float(heartbeat_sec) if t_heartbeat else None,
            "threshold": float(threshold_value) if t_threshold else None,
        },
    )


def _discover_live_enabled_skills() -> list[dict]:
    """Walk all available skills; return those whose frontmatter declares
    a non-empty `live_tick.tick_fn`. Caller still resolves the dotted
    path at start time so we fail fast on bad paths."""
    from scilink.skills.loader import list_all_skills, load_skill

    out: list[dict] = []
    for domain, names in list_all_skills().items():
        for name in names:
            try:
                parsed = load_skill(name, domain=domain)
            except Exception:
                continue
            block = parsed.get("meta", {}).get("live_tick") or {}
            if not isinstance(block, dict):
                continue
            if not block.get("enabled", True):
                continue
            if not block.get("tick_fn"):
                continue
            out.append({
                "name": name,
                "domain": domain,
                "label": f"{domain}/{name}",
                "description": parsed.get("meta", {}).get("description", ""),
            })
    out.sort(key=lambda s: s["label"])
    return out


def _spin_up_live_session(
    *,
    model: str,
    api_key: str,
    data_path: str,
    source_kind: str,
    skill_name: str,
    skill_domain: str,
    tick_interval_sec: float,
    triggers: dict,
) -> None:
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )
    from scilink.agents.exp_agents.live_data_sources import (
        AppendOnlyFileSource,
        MtimePollFileSource,
    )
    from scilink.agents.exp_agents.live_session import LiveSession
    from scilink.agents.exp_agents.live_triggers import (
        ConfidenceReversalTrigger,
        HeartbeatTrigger,
        ManualTrigger,
        NewFeatureTrigger,
        ThresholdCrossTrigger,
        TriggerPolicy,
        VerdictChangeTrigger,
    )
    from scilink.skills.loader import load_skill, resolve_tick_fn
    from scilink.ui.config import SESSION_DIR_PREFIXES

    # Session dir parallels the analyze/plan/simulate pattern.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{SESSION_DIR_PREFIXES['live']}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Build the agent. live mode is autonomous — the LLM doesn't pause for
    # mid-call human feedback during a tick-driven interpretation.
    try:
        orch = AnalysisOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=api_key,
            model_name=model,
            analysis_mode=AnalysisMode.AUTONOMOUS,
        )
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return

    # Resolve the skill's tick function (fails fast on broken dotted path).
    try:
        skill = load_skill(skill_name, domain=skill_domain)
        tick_fn = resolve_tick_fn(skill.get("meta", {}))
    except Exception as e:
        st.error(f"Failed to load skill {skill_domain}/{skill_name}: {e}")
        return
    if tick_fn is None:
        st.error(
            f"Skill {skill_domain}/{skill_name} does not declare a live_tick.tick_fn. "
            "Pick another skill."
        )
        return

    # Data source
    if source_kind == "append_only":
        source = AppendOnlyFileSource(data_path)
    else:
        source = MtimePollFileSource(data_path)

    # Trigger policy from the UI choices
    trigger_list = []
    if triggers["verdict_change"]:
        trigger_list.append(VerdictChangeTrigger())
    if triggers["new_feature"]:
        trigger_list.append(NewFeatureTrigger())
    if triggers["reversal"]:
        trigger_list.append(ConfidenceReversalTrigger())
    if triggers["threshold"] is not None:
        trigger_list.append(
            ThresholdCrossTrigger(threshold=triggers["threshold"], direction="above")
        )
    if triggers["heartbeat"]:
        trigger_list.append(HeartbeatTrigger(interval_sec=triggers["heartbeat_sec"]))
    trigger_list.append(ManualTrigger())  # always-on so "Interpret Now" works
    policy = TriggerPolicy(triggers=trigger_list)

    session = LiveSession(
        orchestrator=orch,
        data_source=source,
        tick_fn=tick_fn,
        tick_interval_sec=tick_interval_sec,
        trigger_policy=policy,
        history_path=session_dir / "live_ticks.jsonl",
        skill_state=skill,
        on_llm_response=lambda ev, result: _record_llm_response(ev, result),
    )
    session.start()

    st.session_state.live_orch = orch
    st.session_state.live_session = session
    st.session_state.live_session_dir = str(session_dir)
    st.session_state.live_data_path = data_path
    st.session_state.live_skill_label = f"{skill_domain}/{skill_name}"
    st.session_state.live_session_feed = []  # list of {"timestamp", "event", "summary"}
    st.rerun()


def _record_llm_response(event, result: dict) -> None:
    """Callback called from the LiveSession dispatcher thread when run_task returns.

    Pushes a structured entry into st.session_state.live_session_feed for the
    dashboard fragment to render. The fragment owns rendering; this thread
    only mutates the list.
    """
    feed = st.session_state.get("live_session_feed")
    if feed is None:
        return  # session not fully wired yet
    feed.append({
        "timestamp": time.time(),
        "event_name": event.name,
        "event_details": event.details,
        "status": result.get("status", "unknown"),
        "summary": result.get("summary", ""),
    })


# ---------------------------------------------------------------------------
# Dashboard view
# ---------------------------------------------------------------------------


def _render_dashboard() -> None:
    session = st.session_state.live_session
    session_dir = st.session_state.get("live_session_dir", "?")
    skill_label = st.session_state.get("live_skill_label", "?")
    data_path = st.session_state.get("live_data_path", "?")

    # Header bar
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown(
            f"**Session active** · skill `{skill_label}` · data `{data_path}` "
            f"· interval {session.tick_interval_sec:.1f}s"
        )
    with col_h2:
        if st.button("Stop Session", type="primary"):
            session.stop(timeout=3.0)
            st.session_state.live_session = None
            st.session_state.live_orch = None
            st.rerun()

    @st.fragment(run_every="2s")
    def _live_view() -> None:
        latest = session.latest()
        hist = session.history()

        # ── Top-level status pills ────────────────────────────────
        verdict = latest.verdict if latest else "unknown"
        metric_value = latest.primary_metric if latest else 0.0
        metric_name = latest.metric_name if latest else "metric"
        verdict_color = {
            "accept": "🟢", "marginal": "🟡", "reject": "🔴",
        }.get(verdict, "⚪")
        st.markdown(
            f"### {verdict_color} **{verdict.upper()}**  ·  "
            f"{metric_name}: **{metric_value:.3f}**  ·  "
            f"LLM: {'⏳ busy' if session.llm_busy else '✅ idle'}  ·  "
            f"ticks: {len(hist)}"
        )

        col_left, col_right = st.columns([3, 2])

        # ── Left: data + metric chart ─────────────────────────────
        with col_left:
            if hist:
                metrics_series = [t.primary_metric for t in hist]
                st.line_chart(metrics_series, height=200, use_container_width=True)
                st.caption(f"{metric_name} over time ({len(hist)} ticks)")
            else:
                st.info("Waiting for first tick — data source has not produced new data yet.")

            # Show detected features for the latest tick, if any
            if latest and latest.detected_features:
                with st.expander(f"Detected features ({len(latest.detected_features)})"):
                    st.json(latest.detected_features[:30])

        # ── Right: decision feed ──────────────────────────────────
        with col_right:
            feed = st.session_state.get("live_session_feed", []) or []
            st.markdown("**Decision feed**")
            if not feed:
                st.caption("No LLM interpretations yet. Triggers will fire as events occur, "
                           "or click _Interpret Now_ below to request one manually.")
            else:
                # Most recent first
                for entry in list(reversed(feed))[:10]:
                    ts = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
                    name = entry["event_name"]
                    status = entry["status"]
                    summary = entry["summary"]
                    icon = "✅" if status == "success" else "⚠️"
                    with st.container(border=True):
                        st.markdown(f"{icon} **{ts}** · `{name}`")
                        if summary:
                            st.write(summary[:1500])
                            if len(summary) > 1500:
                                st.caption(f"({len(summary) - 1500} more characters truncated)")

        # ── Controls ──────────────────────────────────────────────
        if st.button("Interpret Now", help="Manually request an LLM interpretation"):
            session.force_interpretation()
            st.toast("Manual interpretation requested.")

        st.caption(f"Session dir: `{session_dir}` · JSONL: `live_ticks.jsonl`")

    _live_view()


__all__ = ["render_live_panel"]
