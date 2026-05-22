"""Streamlit panel for the live-monitoring mode.

Renders one of two views depending on session state:

  Setup view  — when no live session is active. The user enters a
                data path and describes the experiment in natural
                language; an LLM parse extracts the structured config
                (skill, source kind, reading interval, triggers) for
                the user to review before starting the session. The
                model + API key + consent come from the standard
                sidebar fields (no panel-level prereqs banner — that
                matches how analyze / plan / simulate work).

  Dashboard   — when a live session is running. Layout: latest data
                plot + metric time-series chart on the left, decision
                feed on the right, controls (Interpret Now, Stop) at
                the bottom. Auto-refreshes via
                ``st.fragment(run_every="2s")`` — same primitive
                the existing chat-task polling uses
                (``app.py:539``).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_live_panel() -> None:
    """Top-level renderer; called from ``app.py`` when ``app_mode == 'live'``."""
    st.header("Live Monitoring")
    st.caption(
        "Real-time agent interpretation of an in-progress measurement. "
        "The fast reading loop runs every couple of seconds; the LLM is "
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


_PARSE_SYSTEM_PROMPT = """You extract structured live-monitoring configuration from a scientist's plain-language description of an experiment.

You will be given a short description of an in-progress measurement
(XRD, Raman, EELS, generic spectroscopy, etc.). Return ONE JSON object
matching this schema — no prose, no markdown fences:

{{
  "skill_label": "<domain/name>",            // see AVAILABLE_SKILLS below
  "source_kind": "mtime_poll" | "append_only" | "directory_watch",
  "source_pattern": "*.csv",                  // glob, only used when source_kind == directory_watch
  "source_strategy": "newest" | "unseen",     // only used when source_kind == directory_watch
  "source_sort_by": "mtime" | "name",         // only used when source_kind == directory_watch
  "reading_interval_sec": 2.0,                // float, seconds between readings
  "triggers": {{
    "verdict_change":  true | false,          // fire on verdict transitions
    "new_feature":     true | false,          // fire when a new peak / feature appears
    "reversal":        true | false,          // fire on confidence reversal
    "heartbeat_sec":   60.0 | 0,              // 0 disables the periodic heartbeat
    "threshold":       null | <float 0..1>     // null = no threshold-cross trigger
  }},
  "chemistry_hint": null | ["Si","O"] | [["Si"],["Ge"]],   // optional; for structure_matching skills
  "summary": "one sentence describing what you extracted"
}}

Source-kind disambiguation rules:
  - The instrument **rewrites a single file** as the scan progresses  → mtime_poll
  - The instrument **appends to a single file** (one new row per step) → append_only
  - The instrument writes **one file per measurement into a folder**   → directory_watch
  If the user says "a folder of files" or names a pattern like
  "scan_XXXX.csv", choose directory_watch.

Reading-interval defaults to 2.0 s unless the user specifies otherwise.

Trigger defaults (when the user doesn't say): verdict_change=true,
new_feature=true, reversal=true, heartbeat_sec=60.0, threshold=null.

If the user explicitly disables a trigger ("no heartbeat", "skip the
verdict-change trigger") set it to false / 0 accordingly. If they
mention a confidence threshold ("alert me when FOM > 0.7"), set
threshold to that float.

Pick the skill_label whose description best matches the experiment.
If nothing matches, use "diagnostics/live_passthrough".

AVAILABLE_SKILLS:
{available_skills}
"""


def _parse_description(
    description: str, model: str, api_key: str,
    skill_options: list[dict],
) -> Optional[dict]:
    """Call the LLM to convert a natural-language description into a
    structured config dict. Returns None on failure (caller surfaces error)."""
    try:
        import litellm
    except ImportError:
        st.error("LiteLLM not available; cannot parse description.")
        return None

    from scilink.wrappers.litellm_wrapper import _normalize_model_name

    skills_blob = "\n".join(
        f"  - {s['label']}: {s['description'][:200]}"
        for s in skill_options
    ) or "  (no live-reading-enabled skills found)"
    system = _PARSE_SYSTEM_PROMPT.format(available_skills=skills_blob)

    try:
        resp = litellm.completion(
            model=_normalize_model_name(model),
            api_key=api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": description.strip()},
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None

    try:
        text = resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Unexpected LLM response shape: {e}")
        return None

    return _extract_json(text)


def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of the LLM response (tolerates surrounding text)."""
    # Strip common code-fence wrappers
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last <= first:
            st.error("No JSON object found in LLM response.")
            with st.expander("Raw LLM response"):
                st.code(text[:2000])
            return None
        candidate = text[first : last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM JSON: {e}")
        with st.expander("Raw LLM response"):
            st.code(candidate[:2000])
        return None


def _render_parsed_preview(parsed: dict) -> None:
    """Render the parsed config as a readable preview."""
    st.markdown("**Parsed configuration**")
    summary = parsed.get("summary", "")
    if summary:
        st.caption(summary)

    triggers = parsed.get("triggers", {}) or {}
    trigger_list = []
    if triggers.get("verdict_change", True):
        trigger_list.append("verdict change")
    if triggers.get("new_feature", True):
        trigger_list.append("new feature")
    if triggers.get("reversal", True):
        trigger_list.append("confidence reversal")
    if triggers.get("heartbeat_sec"):
        trigger_list.append(f"heartbeat every {triggers['heartbeat_sec']:.0f}s")
    if triggers.get("threshold") is not None:
        trigger_list.append(f"threshold cross at {triggers['threshold']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Skill:**  `{parsed.get('skill_label', '?')}`")
        st.write(f"**Source:**  `{parsed.get('source_kind', '?')}`")
        if parsed.get("source_kind") == "directory_watch":
            st.write(
                f"&nbsp;&nbsp;&nbsp;pattern `{parsed.get('source_pattern', '*.txt')}`, "
                f"strategy `{parsed.get('source_strategy', 'newest')}`, "
                f"sort `{parsed.get('source_sort_by', 'mtime')}`",
                unsafe_allow_html=True,
            )
        chem = parsed.get("chemistry_hint")
        if chem:
            st.write(f"**Chemistry hint:**  `{chem}`")
    with col2:
        st.write(f"**Reading interval:**  `{parsed.get('reading_interval_sec', 2.0)} s`")
        st.write(f"**Triggers:**  {', '.join(trigger_list) or '_none_'}")


def _render_setup() -> None:
    # Sidebar handles model / API key / consent — same as analyze / plan /
    # simulate. No prereqs banner here; the sidebar IS the visible status,
    # and the Start button gates on consent (sidebar.py:343 pattern).
    model, api_key, consent = _resolve_sidebar_config()
    env_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_key = api_key or env_key

    skill_options = _discover_live_enabled_skills()
    if not skill_options:
        st.warning(
            "No skills with a `live_reading:` block found. The skill's "
            "frontmatter needs `live_reading.reading_fn` set to a "
            "module:function path."
        )
        return

    st.subheader("Start a live session")

    data_path = st.text_input(
        "Data file or directory path",
        key="live_data_path_input",
        placeholder="/path/to/in-progress-scan.csv   OR   /path/to/scan_directory/",
        help=(
            "Single file (instrument rewrites or appends as it scans) "
            "OR directory (instrument writes one file per measurement step)."
        ),
    )

    description = st.text_area(
        "Describe your experiment",
        key="live_description",
        height=150,
        placeholder=(
            "Examples:\n"
            "• I'm running an XRD scan on a Si sample. The instrument writes "
            "scan_XXXX.csv files into a folder, one per measurement. Alert me "
            "when the verdict changes or when a new peak appears.\n\n"
            "• Single CSV file gets rewritten as the scan progresses. "
            "Check every 3 seconds. Heartbeat every 30s. No threshold trigger."
        ),
        help=(
            "Plain language. The LLM extracts the source kind, reading "
            "interval, triggers, and any chemistry hint, then shows you the "
            "result for review before starting."
        ),
    )

    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        parse_clicked = st.button(
            "Parse description",
            disabled=not (description and description.strip() and resolved_key),
            help=(
                "Enter a description and ensure an API key is set in the sidebar."
                if not (description and description.strip() and resolved_key)
                else None
            ),
        )
    if parse_clicked:
        with st.spinner("Parsing description…"):
            parsed = _parse_description(
                description.strip(), model, resolved_key, skill_options,
            )
        if parsed is not None:
            st.session_state.live_parsed_config = parsed

    parsed = st.session_state.get("live_parsed_config")
    if parsed:
        st.divider()
        _render_parsed_preview(parsed)

        start_disabled = not consent or not data_path
        start_help = None
        if not consent:
            start_help = "Check the code-execution consent box in the sidebar first."
        elif not data_path:
            start_help = "Enter a data file or directory path above."
        if st.button("Start Live Session", type="primary",
                     disabled=start_disabled, help=start_help):
            _start_from_parsed(
                parsed=parsed, data_path=data_path,
                model=model, api_key=resolved_key, skill_options=skill_options,
            )


def _start_from_parsed(*, parsed: dict, data_path: str, model: str,
                        api_key: str, skill_options: list[dict]) -> None:
    # Resolve skill label → (domain, name)
    skill_label = parsed.get("skill_label", "")
    selected = next((s for s in skill_options if s["label"] == skill_label), None)
    if selected is None:
        # Fallback to diagnostics if the parser hallucinated a skill
        selected = next(
            (s for s in skill_options if s["label"] == "diagnostics/live_passthrough"),
            None,
        )
    if selected is None:
        st.error(f"Skill {skill_label!r} not available and no fallback found.")
        return

    if not Path(data_path).exists():
        st.warning(
            f"Path does not exist yet: {data_path}. The session will start "
            "and the reading loop will pick up the file once your instrument "
            "creates it."
        )

    source_kind = parsed.get("source_kind", "mtime_poll")
    source_kwargs = (
        {
            "pattern": parsed.get("source_pattern", "*.txt"),
            "strategy": parsed.get("source_strategy", "newest"),
            "sort_by": parsed.get("source_sort_by", "mtime"),
        }
        if source_kind == "directory_watch" else {}
    )
    triggers_in = parsed.get("triggers", {}) or {}
    triggers = {
        "verdict_change": bool(triggers_in.get("verdict_change", True)),
        "new_feature":    bool(triggers_in.get("new_feature", True)),
        "reversal":       bool(triggers_in.get("reversal", True)),
        "heartbeat":      bool(triggers_in.get("heartbeat_sec", 60.0)),
        "heartbeat_sec":  float(triggers_in.get("heartbeat_sec") or 60.0)
                          if triggers_in.get("heartbeat_sec") else None,
        "threshold":      (
            float(triggers_in["threshold"])
            if triggers_in.get("threshold") is not None else None
        ),
    }

    # Stash the chemistry hint where the tick fn reads it
    st.session_state.live_chemistry_hint = parsed.get("chemistry_hint")

    _spin_up_live_session(
        model=model,
        api_key=api_key,
        data_path=data_path,
        source_kind=source_kind,
        source_kwargs=source_kwargs,
        skill_name=selected["name"],
        skill_domain=selected["domain"],
        reading_interval_sec=float(parsed.get("reading_interval_sec", 2.0)),
        triggers=triggers,
    )


def _discover_live_enabled_skills() -> list[dict]:
    """Walk all available skills; return those whose frontmatter declares
    a non-empty `live_reading.reading_fn`. Caller still resolves the dotted
    path at start time so we fail fast on bad paths."""
    from scilink.skills.loader import list_all_skills, load_skill

    out: list[dict] = []
    for domain, names in list_all_skills().items():
        for name in names:
            try:
                parsed = load_skill(name, domain=domain)
            except Exception:
                continue
            block = parsed.get("meta", {}).get("live_reading") or {}
            if not isinstance(block, dict):
                continue
            if not block.get("enabled", True):
                continue
            if not block.get("reading_fn"):
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
    source_kwargs: Optional[dict] = None,
    skill_name: str,
    skill_domain: str,
    reading_interval_sec: float,
    triggers: dict,
) -> None:
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )
    from scilink.agents.exp_agents.live_data_sources import (
        AppendOnlyFileSource,
        DirectoryWatchSource,
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
    from scilink.skills.loader import load_skill, resolve_reading_fn
    from scilink.ui.config import SESSION_DIR_PREFIXES

    # Session dir parallels the analyze/plan/simulate pattern.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{SESSION_DIR_PREFIXES['live']}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Build the agent. live mode is autonomous — the LLM doesn't pause for
    # mid-call human feedback during a reading-driven interpretation.
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

    # Resolve the skill's reading function (fails fast on broken dotted path).
    try:
        skill = load_skill(skill_name, domain=skill_domain)
        reading_fn = resolve_reading_fn(skill.get("meta", {}))
    except Exception as e:
        st.error(f"Failed to load skill {skill_domain}/{skill_name}: {e}")
        return
    if reading_fn is None:
        st.error(
            f"Skill {skill_domain}/{skill_name} does not declare a live_reading.reading_fn. "
            "Pick another skill."
        )
        return

    # Data source
    if source_kind == "directory_watch":
        source = DirectoryWatchSource(
            data_path,
            pattern=(source_kwargs or {}).get("pattern", "*.txt"),
            strategy=(source_kwargs or {}).get("strategy", "newest"),
            sort_by=(source_kwargs or {}).get("sort_by", "mtime"),
        )
    elif source_kind == "append_only":
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
        reading_fn=reading_fn,
        reading_interval_sec=reading_interval_sec,
        trigger_policy=policy,
        history_path=session_dir / "live_readings.jsonl",
        skill_state=skill,
        on_llm_response=lambda ev, result: _record_llm_response(ev, result),
    )
    # Thread the parsed chemistry hint (if any) into the session state the
    # tick / reading function reads from. The structure_matching/xrd
    # reading_fn uses session_state["chemistry_hint"] to scope its DB query.
    chem_hint = st.session_state.get("live_chemistry_hint")
    if chem_hint is not None:
        session._session_state["chemistry_hint"] = chem_hint
    session._session_state["candidates_dir"] = str(session_dir / "candidates")
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
            f"· interval {session.reading_interval_sec:.1f}s"
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
            f"readings: {len(hist)}"
        )

        col_left, col_right = st.columns([3, 2])

        # ── Left: data + metric chart ─────────────────────────────
        with col_left:
            if hist:
                metrics_series = [t.primary_metric for t in hist]
                st.line_chart(metrics_series, height=200, use_container_width=True)
                st.caption(f"{metric_name} over time ({len(hist)} readings)")
            else:
                st.info("Waiting for first reading — data source has not produced new data yet.")

            # Show detected features for the latest reading, if any
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

        st.caption(f"Session dir: `{session_dir}` · JSONL: `live_readings.jsonl`")

    _live_view()


__all__ = ["render_live_panel"]
