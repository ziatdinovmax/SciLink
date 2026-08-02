"""Sidebar component: configuration, file upload, agent status."""

import base64
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from ..config import (
    EMBEDDING_MODEL_OPTIONS,
    MODEL_OPTIONS,
    SESSION_DIR_PREFIXES,
    SUPPORTED_CODE_EXTENSIONS,
    SUPPORTED_DATA_EXTENSIONS,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    SUPPORTED_METADATA_EXTENSIONS,
    SUPPORTED_PLANNING_DATA_EXTENSIONS,
    extra_data_extensions_for,
)


_LOGO_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOGO_DARK = _LOGO_DIR / "scilink_logo_v3_dark.svg"
_LOGO_LIGHT = _LOGO_DIR / "scilink_logo_v3_light.svg"

def _render_hpc_connection() -> None:
    """Compact HPC connection controls for the sidebar."""
    try:
        import paramiko  # noqa: F401
    except ImportError:
        with st.expander("🖥️ HPC Connection"):
            st.caption("Install `paramiko` to enable HPC connectivity.")
        return

    from scilink.hpc.connection import HPCConnection, HPCProfile
    from scilink.hpc.scheduler import detect_scheduler

    conn = st.session_state.get("hpc_connection")
    is_connected = conn is not None and conn.is_connected

    label = (
        f"🟢 {conn.profile.username}@{conn.profile.hostname}"
        if is_connected
        else "🖥️ HPC Connection"
    )

    with st.expander(label, expanded=False):
        if is_connected:
            sched = st.session_state.get("hpc_scheduler")
            if sched:
                st.caption(f"Scheduler: **{sched.name}**")
            tracked = st.session_state.get("hpc_tracked_jobs", {})
            from scilink.hpc.scheduler import JobStatus
            n_active = sum(
                1 for j in tracked.values()
                if j.status in (JobStatus.RUNNING, JobStatus.PENDING)
            )
            if n_active:
                st.caption(f"Active jobs: **{n_active}**")
            if st.button(
                "Disconnect",
                key="sidebar_hpc_disconnect",
                use_container_width=True,
            ):
                conn.disconnect()
                st.session_state.hpc_connection = None
                st.session_state.hpc_scheduler = None
                st.rerun()
            return

        # ── Not connected — login form ────────────────────
        saved: list = st.session_state.get("hpc_saved_profiles", [])

        prefill = None
        if saved:
            names = ["New connection…"] + [p.name for p in saved]
            sel = st.selectbox("Profile", names, key="sidebar_hpc_profile_sel")
            if sel != "New connection…":
                prefill = next(p for p in saved if p.name == sel)

        hostname = st.text_input(
            "Host",
            value=prefill.hostname if prefill else "",
            placeholder="login.cluster.edu",
            key="sidebar_hpc_host",
        )
        username = st.text_input(
            "User",
            value=prefill.username if prefill else "",
            key="sidebar_hpc_user",
        )
        port = st.number_input(
            "Port", value=prefill.port if prefill else 22,
            min_value=1, max_value=65535, key="sidebar_hpc_port",
        )
        auth = st.radio(
            "Auth",
            ["SSH Key", "Password"],
            horizontal=True,
            key="sidebar_hpc_auth",
        )
        password = key_path = key_pass = proxy = ""
        if auth == "Password":
            password = st.text_input("Password", type="password", key="sidebar_hpc_pass")
        else:
            key_path = st.text_input(
                "Key path (blank → default)",
                value=prefill.key_path if prefill else "",
                key="sidebar_hpc_keypath",
            )
            key_pass = st.text_input(
                "Key passphrase",
                type="password",
                key="sidebar_hpc_keypass",
            )
        proxy = st.text_input(
            "ProxyJump (optional)",
            value=prefill.proxy_jump if prefill else "",
            placeholder="user@bastion",
            key="sidebar_hpc_proxy",
        )

        save_chk = st.checkbox("Remember profile", key="sidebar_hpc_save")

        if st.button(
            "Connect",
            disabled=not (hostname and username),
            use_container_width=True,
            key="sidebar_hpc_connect",
        ):
            profile = HPCProfile(
                name=hostname.split(".")[0] if hostname else "cluster",
                hostname=hostname,
                username=username,
                port=int(port),
                auth_method="key" if auth == "SSH Key" else "password",
                key_path=key_path,
                proxy_jump=proxy,
            )
            try:
                c = HPCConnection(profile)
                c.connect(password=password, key_passphrase=key_pass)
                sched = detect_scheduler(c)
                home = c.home_dir()

                st.session_state.hpc_connection = c
                st.session_state.hpc_scheduler = sched
                st.session_state.hpc_remote_cwd = home

                if save_chk:
                    profiles = [p for p in saved if p.name != profile.name]
                    profiles.append(profile)
                    st.session_state.hpc_saved_profiles = profiles

                # Wire HPC into SimulationOrchestratorAgent if already running
                _wire_hpc_to_agent(c, sched)

                st.rerun()
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

def _wire_hpc_to_agent(conn=None, sched=None) -> None:
    """
    Attach HPC connection and scheduler to the SimulationOrchestratorAgent
    if one is already running in this session.

    Safe to call speculatively — does nothing if the agent isn't a
    SimulationOrchestratorAgent or doesn't have the expected attributes.
    """
    agent = st.session_state.get("agent")
    if agent is None:
        return
    try:
        from scilink.agents.sim_agents.simulation_orchestrator import (
            SimulationOrchestratorAgent,
        )
        if not isinstance(agent, SimulationOrchestratorAgent):
            return
    except ImportError:
        return

    if conn is not None:
        agent.hpc_connection = conn
    if sched is not None:
        agent.hpc_scheduler = sched


def render_sidebar() -> None:
    with st.sidebar:
        # ── Theme toggle ──────────────────────────────────
        _is_dark = st.session_state.get("theme_mode", "dark") == "dark"
        st.markdown('<span class="theme-toggle-anchor"></span>', unsafe_allow_html=True)

        def _toggle_theme():
            cur = st.session_state.get("theme_mode", "dark")
            st.session_state.theme_mode = "light" if cur == "dark" else "dark"

        _tcol, _ = st.columns([0.2, 0.8])
        with _tcol:
            st.button(
                "\u2600\ufe0f" if _is_dark else "\U0001f319",
                key="theme_toggle",
                on_click=_toggle_theme,
            )

        _logo = _LOGO_DARK if _is_dark else _LOGO_LIGHT
        if st.session_state.agent_initialized and _logo.exists():
            _b64 = base64.b64encode(_logo.read_bytes()).decode()
            if _is_dark:
                st.markdown(
                    '<style>'
                    '@keyframes logo-spin{to{transform:rotate(360deg)}}'
                    '.logo-glow-sm{position:relative;padding:2px;border-radius:10px;'
                    'overflow:hidden;width:140px;margin:0 auto}'
                    '.logo-glow-sm::before{content:"";position:absolute;'
                    'top:-40%;left:-40%;width:180%;height:180%;'
                    'background:conic-gradient('
                    'transparent 0deg,transparent 270deg,#3A4556 300deg,'
                    '#82B1FF 330deg,#FFF 345deg,#82B1FF 355deg,transparent 360deg);'
                    'animation:logo-spin 4s linear infinite;z-index:0}'
                    '.logo-glow-sm>img{position:relative;z-index:1;border-radius:8px;'
                    'display:block;width:100%}'
                    '</style>'
                    f'<div class="logo-glow-sm">'
                    f'<img src="data:image/svg+xml;base64,{_b64}"/>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="width:140px;margin:0 auto">'
                    f'<img src="data:image/svg+xml;base64,{_b64}" '
                    f'style="border-radius:8px;display:block;width:100%"/>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.title("SciLink")
        _locked = st.session_state.agent_initialized
        _is_simulate = st.session_state.app_mode == "simulate"
        preset = st.selectbox(
            "Model",
            MODEL_OPTIONS + ["Custom"],
            key="cfg_model_preset",
            disabled=_locked,
        )
        if preset == "Custom":
            model = st.text_input("Model name", key="cfg_model_custom", disabled=_locked)
        else:
            model = preset
        api_key = st.text_input("API key", type="password", key="cfg_api_key", disabled=_locked)
        base_url = st.text_input("Base URL (optional)", key="cfg_base_url", disabled=_locked)
        fh_api_key = st.text_input("FutureHouse API key (optional)", type="password", key="cfg_fh_api_key", disabled=_locked)
        mp_api_key = st.text_input("Materials Project API key (optional)", type="password", key="cfg_mp_api_key", disabled=_locked)

        _render_hpc_connection()

        # Planning mode: embedding model
        if st.session_state.app_mode == "plan":
            st.selectbox(
                "Embedding model",
                EMBEDDING_MODEL_OPTIONS + ["Custom"],
                key="cfg_embedding_preset",
                disabled=_locked,
            )
            if st.session_state.get("cfg_embedding_preset") == "Custom":
                st.text_input("Embedding model name", key="cfg_embedding_custom", disabled=_locked)
            st.text_input(
                "Embedding API key (optional)",
                type="password",
                key="cfg_embedding_api_key",
                disabled=_locked,
            )
            st.caption("\u2139\ufe0f Leave blank to use the main API key")

        mode = st.selectbox(
            "Autonomy mode",
            ["co-pilot", "supervised", "autonomous"],
            key="cfg_mode",
            disabled=_locked,
        )
        consent = st.checkbox(
            "I understand that the agent will execute generated Python code on my machine",
            key="cfg_consent",
            disabled=_locked,
        )
        if not consent:
            with st.expander("What does the agent execute?"):
                from scilink.executors import LLM_EXECUTION_DESCRIPTION
                st.text(LLM_EXECUTION_DESCRIPTION)
                st.warning(
                    "**Recommendation:** Run this app inside a Docker container "
                    "or a VM to sandbox generated code execution. "
                    "Example: `docker run -p 8501:8501 scilink-ui`"
                )
        # ── Resume past session ────────────────────────────
        if not _locked:
            _resume_mode = st.session_state.app_mode or "analyze"
            past = _discover_resumable_sessions(_resume_mode)
            if past:
                with st.expander("Resume past session", expanded=False):
                    labels = []
                    for sess in past:
                        lbl = sess["label"]
                        details = []
                        s = sess["summary"]
                        if "analysis_count" in s:
                            details.append(f"{s['analysis_count']} analyses")
                        if s.get("data_file"):
                            details.append(s["data_file"])
                        if s.get("message_count"):
                            details.append(f"{s['message_count']} messages")
                        if not sess["has_checkpoint"]:
                            details.append("no checkpoint")
                        if details:
                            lbl += f"  ({', '.join(details)})"
                        labels.append(lbl)

                    selected_idx = st.selectbox(
                        "Select session",
                        range(len(labels)),
                        format_func=lambda i: labels[i],
                        key="resume_session_selector",
                    )
                    selected = past[selected_idx]

                    if not selected["has_checkpoint"]:
                        st.warning(
                            "No checkpoint found. Agent state will not be "
                            "restored, but chat history will be loaded."
                        )

                    if st.button(
                        "Resume Session",
                        disabled=not consent,
                        use_container_width=True,
                        key="resume_session_btn",
                    ):
                        _r_embed_preset = st.session_state.get("cfg_embedding_preset", "")
                        _r_embed_model = (
                            st.session_state.get("cfg_embedding_custom", "")
                            if _r_embed_preset == "Custom" else _r_embed_preset
                        ) or None
                        _r_embed_api_key = st.session_state.get("cfg_embedding_api_key", "") or None
                        st.session_state._pending_resume = {
                            "session_dir": str(selected["path"]),
                            "model": model,
                            "api_key": api_key,
                            "base_url": base_url,
                            "mode": mode,
                            "fh_api_key": fh_api_key,
                            "mp_api_key": mp_api_key,
                            "embedding_model": _r_embed_model,
                            "embedding_api_key": _r_embed_api_key,
                        }

        col1, col2 = st.columns(2)

        with col1:
            start_disabled = st.session_state.agent_initialized or not consent
            if st.button("Start Session", disabled=start_disabled, width="stretch"):
                # Stash config and let the main app handle the heavy init
                # outside the sidebar context so the spinner is visible.
                _embed_preset = st.session_state.get("cfg_embedding_preset", "")
                _embed_model = (
                    st.session_state.get("cfg_embedding_custom", "")
                    if _embed_preset == "Custom" else _embed_preset
                ) or None
                _embed_api_key = st.session_state.get("cfg_embedding_api_key", "") or None
                st.session_state._pending_init = {
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "mode": mode,
                    "fh_api_key": fh_api_key,
                    "mp_api_key": mp_api_key,
                    "embedding_model": _embed_model,
                    "embedding_api_key": _embed_api_key,
                }

        with col2:
            if st.button("Reset Session", disabled=not st.session_state.agent_initialized,
                         width="stretch"):
                _reset_session()

        # ── File upload (mode-specific) ──────────────────────────
        if st.session_state.agent_initialized:
            st.divider()
            app_mode = st.session_state.app_mode

            if app_mode == "analyze":
                _render_analyze_sidebar_uploads()
            elif app_mode == "plan":
                _render_planning_sidebar_uploads()

            # ── Agent status (mode-specific) ─────────────────────
            st.divider()
            st.subheader("Agent Status")
            if app_mode == "analyze":
                _render_analyze_status()
            elif app_mode == "plan":
                _render_planning_status()

        # ── Vibes ──────────────────────────────────────────
        st.divider()
        st.radio(
            "Vibe",
            ["Professional", "Positivity boost", "Space nerd"],
            key="vibe_theme",
            horizontal=True,
        )
        vibe = st.session_state.get("vibe_theme", "Professional")
        if vibe == "Positivity boost":
            st.slider("\U0001f49c Hearts", min_value=0, max_value=50, value=7, key="vibe_hearts")
            st.slider("\u2795 Pluses", min_value=0, max_value=50, value=7, key="vibe_pluses")
        elif vibe == "Space nerd":
            st.slider("\U0001f680 Rockets", min_value=0, max_value=50, value=7, key="vibe_rockets")
            st.slider("\U0001f6f8 UFOs", min_value=0, max_value=10, value=1, key="vibe_ufos")

        # ── Quit button (always visible at bottom) ────────────
        st.divider()
        if st.button("Quit App", use_container_width=True):
            # Stop any running agent thread
            task = st.session_state.get("chat_task")
            if task and task.is_running:
                task.stopped = True
                if task.live_capture:
                    task.live_capture.request_stop()
                if task.feedback_request is not None:
                    task.feedback_request.response = ""
                    task.feedback_request.event.set()
            # Inject JS to replace the page with a goodbye message,
            # then kill the server after a short delay.
            import streamlit.components.v1 as components
            components.html(
                '<script>'
                'window.parent.document.body.innerHTML = '
                '\'<div style="display:flex;align-items:center;justify-content:center;'
                'height:100vh;font-family:sans-serif;color:#888;background:#0e1117;">'
                '<h2>Server stopped. You can close this window.</h2></div>\';'
                '</script>',
                height=0,
            )
            import time, signal
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)


def _render_analyze_sidebar_uploads() -> None:
    """Sidebar upload widgets for analyze mode."""
    st.subheader("Upload Files")

    extra_exts = extra_data_extensions_for(st.session_state.get("agent"))
    data_exts = SUPPORTED_DATA_EXTENSIONS + extra_exts

    data_files = st.file_uploader(
        "Data file(s)",
        type=[e.lstrip(".") for e in data_exts],
        key="uploader_data",
        accept_multiple_files=True,
    )
    if extra_exts:
        st.caption("Vendor formats enabled via SciFiReaders MCP")
    if data_files:
        if len(data_files) == 1:
            save_upload(data_files[0], "data")
        else:
            save_upload_batch(data_files, "data")

    meta_files = st.file_uploader(
        "Metadata file(s)",
        type=[e.lstrip(".") for e in SUPPORTED_METADATA_EXTENSIONS],
        key="uploader_meta",
        accept_multiple_files=True,
    )
    if meta_files:
        if len(meta_files) == 1:
            save_upload(meta_files[0], "metadata")
        else:
            save_metadata_to_series(meta_files)


def _render_planning_sidebar_uploads() -> None:
    """Compact sidebar upload widgets for plan mode."""
    st.subheader("Upload Files")

    knowledge_files = st.file_uploader(
        "Knowledge",
        type=[e.lstrip(".") for e in SUPPORTED_KNOWLEDGE_EXTENSIONS],
        key="sidebar_uploader_knowledge",
        accept_multiple_files=True,
    )
    if knowledge_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(knowledge_files, "knowledge")

    code_files = st.file_uploader(
        "Code",
        type=[e.lstrip(".") for e in SUPPORTED_CODE_EXTENSIONS],
        key="sidebar_uploader_code",
        accept_multiple_files=True,
    )
    if code_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(code_files, "code")

    data_files = st.file_uploader(
        "Data",
        type=[e.lstrip(".") for e in SUPPORTED_PLANNING_DATA_EXTENSIONS],
        key="sidebar_uploader_planning_data",
        accept_multiple_files=True,
    )
    if data_files:
        from .chat_uploads import save_planning_uploads
        save_planning_uploads(data_files, "data")


def _render_analyze_status() -> None:
    """Show analysis-specific agent status metrics."""
    agent = st.session_state.agent
    if agent.selected_agent_id is not None:
        entry = agent._agent_registry.get(agent.selected_agent_id, {})
        agent_label = entry.get("name", f"Agent {agent.selected_agent_id}")
    else:
        agent_label = "None"

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.metric("Selected Agent", agent_label)
    with row1_c2:
        st.metric("Analyses", len(agent.analysis_results))

    data_path = agent.current_data_path
    if data_path:
        st.metric("Data File", Path(data_path).name)
    else:
        st.metric("Data File", "None")

    if agent.current_data_type:
        st.metric("Data Type", agent.current_data_type)


def _render_planning_status() -> None:
    """Show planning-specific agent status metrics."""
    objective = st.session_state.get("planning_objective", "")
    n_messages = len(st.session_state.chat_messages)
    mode = st.session_state.agent_config.get("mode", "supervised")

    if objective:
        st.metric("Objective", objective[:40] + ("..." if len(objective) > 40 else ""))
    st.metric("Messages", n_messages)
    st.metric("Autonomy", mode.replace("-", " ").title())


# ── helpers ──────────────────────────────────────────────────────

def start_session(model: str, api_key: str, base_url: str, mode: str, fh_api_key: str = "", mp_api_key: str = "", embedding_model: str = None, embedding_api_key: str = None) -> None:
    """Initialize the agent and session directory.

    Dispatches to the appropriate agent based on ``st.session_state.app_mode``.
    Designed to be called from the **main** content area (not inside
    ``with st.sidebar``) so that ``st.spinner`` is visible to the user.
    """
    import scilink
    import scilink.executors as executors

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key and not base_url:
        st.sidebar.error("Provide an API key or set an environment variable (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).")
        return

    # Optional MP key: register so DFTOrchestrator's auto-discovery picks it up
    # later when the analysis orchestrator dispatches to structure generation.
    if mp_api_key:
        scilink.set_api_key('materials_project', mp_api_key)

    app_mode = st.session_state.app_mode or "analyze"
    prefix = SESSION_DIR_PREFIXES.get(app_mode, "session")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{prefix}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Auto-approve sandbox for the Streamlit session (user checked consent box)
    executors._GLOBAL_SANDBOX_APPROVED = True

    try:
        if app_mode == "plan":
            agent = _init_planning_agent(
                session_dir, resolved_key, model, base_url, mode, fh_api_key,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
            )
        else:
            agent = _init_analysis_agent(
                session_dir, resolved_key, model, base_url, mode, fh_api_key,
            )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return

    st.session_state.agent = agent
    st.session_state.agent_initialized = True
    st.session_state.session_dir = str(session_dir)
    st.session_state.agent_config = {
        "model": model,
        "mode": mode,
    }
    st.session_state.chat_messages = []
    st.session_state.known_images = set()
    st.rerun()


def _init_analysis_agent(session_dir, api_key, model, base_url, mode, fh_api_key):
    """Create an AnalysisOrchestratorAgent."""
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )

    mode_map = {
        "co-pilot": AnalysisMode.CO_PILOT,
        "supervised": AnalysisMode.SUPERVISED,
        "autonomous": AnalysisMode.AUTONOMOUS,
    }
    return AnalysisOrchestratorAgent(
        base_dir=str(session_dir),
        api_key=api_key,
        model_name=model,
        base_url=base_url or None,
        analysis_mode=mode_map[mode],
        futurehouse_api_key=fh_api_key or None,
    )


def _init_planning_agent(session_dir, api_key, model, base_url, mode, fh_api_key,
                         embedding_model=None, embedding_api_key=None):
    """Create a PlanningOrchestratorAgent."""
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel,
        PlanningOrchestratorAgent,
    )

    mode_map = {
        "co-pilot": AutonomyLevel.CO_PILOT,
        "supervised": AutonomyLevel.SUPERVISED,
        "autonomous": AutonomyLevel.AUTONOMOUS,
    }
    objective = st.session_state.get("planning_objective", "").strip() or "Undefined Research Goal"

    # Shared knowledge directory (persists across sessions)
    knowledge_dir = Path.cwd() / "kb_storage"
    knowledge_dir.mkdir(exist_ok=True)
    # Per-session directories
    code_dir = session_dir / "code"
    data_dir = session_dir / "data"
    code_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    kwargs = {}
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    if embedding_api_key:
        kwargs["embedding_api_key"] = embedding_api_key

    return PlanningOrchestratorAgent(
        objective=objective,
        base_dir=str(session_dir),
        api_key=api_key,
        model_name=model,
        base_url=base_url or None,
        autonomy_level=mode_map[mode],
        futurehouse_api_key=fh_api_key or None,
        knowledge_dir=str(knowledge_dir),
        code_dir=str(code_dir),
        data_dir=str(data_dir),
        **kwargs,
    )


def _discover_resumable_sessions(app_mode: str) -> list:
    """Find past session directories that can be resumed."""
    import json as _json

    parent = Path.cwd()
    prefix = SESSION_DIR_PREFIXES.get(app_mode, "analysis_session")
    sessions = sorted(parent.glob(f"{prefix}_*"), key=lambda p: p.name, reverse=True)

    result = []
    for s in sessions:
        if not s.is_dir():
            continue
        has_checkpoint = (s / "checkpoint.json").exists()
        has_chat = (s / "chat_history.json").exists()
        if not has_checkpoint and not has_chat:
            continue

        # Build human-readable label from timestamp in dir name
        parts = s.name.removeprefix(f"{prefix}_")
        try:
            label = f"{parts[:4]}-{parts[4:6]}-{parts[6:8]} {parts[9:11]}:{parts[11:13]}:{parts[13:15]}"
        except (IndexError, ValueError):
            label = s.name

        summary: dict = {}
        if has_checkpoint:
            try:
                ckpt = _json.loads((s / "checkpoint.json").read_text())
                summary["analysis_count"] = len(ckpt.get("analysis_results", []))
                dp = ckpt.get("current_data_path")
                if dp:
                    summary["data_file"] = Path(dp).name
            except Exception:
                pass
        if has_chat and "analysis_count" not in summary:
            try:
                hist = _json.loads((s / "chat_history.json").read_text())
                summary["message_count"] = sum(1 for m in hist if m.get("role") == "user")
            except Exception:
                pass

        result.append({
            "path": s,
            "label": label,
            "has_checkpoint": has_checkpoint,
            "has_chat_history": has_chat,
            "summary": summary,
        })
    return result


def _convert_chat_history_for_display(history: list) -> list:
    """Convert orchestrator chat_history.json to Streamlit chat_messages format.

    Keeps only user/assistant messages with text content; drops tool calls,
    tool results, and system messages.
    """
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def resume_session(
    session_dir: str,
    model: str,
    api_key: str,
    base_url: str,
    mode: str,
    fh_api_key: str = "",
    mp_api_key: str = "",
    embedding_model: str = None,
    embedding_api_key: str = None,
) -> None:
    """Restore an agent from a past session directory and re-populate the UI."""
    import json as _json
    import scilink
    import scilink.executors as executors

    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not resolved_key and not base_url:
        st.sidebar.error(
            "Provide an API key or set an environment variable "
            "(GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)."
        )
        return

    # Optional MP key registration (see start_session for rationale).
    if mp_api_key:
        scilink.set_api_key('materials_project', mp_api_key)

    session_path = Path(session_dir)
    executors._GLOBAL_SANDBOX_APPROVED = True

    app_mode = st.session_state.app_mode or "analyze"

    try:
        if app_mode == "plan":
            from scilink.agents.planning_agents.planning_orchestrator import (
                AutonomyLevel,
                PlanningOrchestratorAgent,
            )
            plan_mode_map = {
                "co-pilot": AutonomyLevel.CO_PILOT,
                "supervised": AutonomyLevel.SUPERVISED,
                "autonomous": AutonomyLevel.AUTONOMOUS,
            }
            kwargs = {}
            if embedding_model:
                kwargs["embedding_model"] = embedding_model
            if embedding_api_key:
                kwargs["embedding_api_key"] = embedding_api_key
            agent = PlanningOrchestratorAgent.restore_from_checkpoint(
                base_dir=str(session_path),
                api_key=resolved_key,
                model_name=model,
                base_url=base_url or None,
                autonomy_level=plan_mode_map[mode],
                futurehouse_api_key=fh_api_key or None,
                **kwargs,
            )
        else:
            from scilink.agents.exp_agents.analysis_orchestrator import (
                AnalysisMode,
                AnalysisOrchestratorAgent,
            )
            analysis_mode_map = {
                "co-pilot": AnalysisMode.CO_PILOT,
                "supervised": AnalysisMode.SUPERVISED,
                "autonomous": AnalysisMode.AUTONOMOUS,
            }
            agent = AnalysisOrchestratorAgent.restore_from_checkpoint(
                base_dir=str(session_path),
                api_key=resolved_key,
                model_name=model,
                base_url=base_url or None,
                analysis_mode=analysis_mode_map[mode],
                futurehouse_api_key=fh_api_key or None,
            )
    except Exception as exc:
        st.error(f"Failed to restore session: {exc}")
        return

    # Load chat history for Streamlit display
    display_messages: list = []
    chat_path = session_path / "chat_history.json"
    if chat_path.exists():
        try:
            raw = _json.loads(chat_path.read_text())
            display_messages = _convert_chat_history_for_display(raw)
        except Exception:
            pass

    # Pre-populate known images so they don't re-appear as "new"
    known: set = set()
    for ext in (".png", ".jpg", ".jpeg"):
        for p in session_path.rglob(f"*{ext}"):
            known.add(str(p))

    st.session_state.agent = agent
    st.session_state.agent_initialized = True
    st.session_state.session_dir = str(session_path)
    st.session_state.agent_config = {"model": model, "mode": mode}
    st.session_state.chat_messages = display_messages
    st.session_state.known_images = known
    st.rerun()


def _reset_session() -> None:
    # Stop the agent thread if it's still running
    task = st.session_state.get("chat_task")
    if task and task.is_running and task.feedback_request is not None:
        task.feedback_request.response = ""
        task.feedback_request.event.set()

    # Disconnect MCP servers to avoid orphaned subprocesses
    agent = st.session_state.get("agent")
    if agent is not None:
        for name in list(getattr(agent, "_mcp_connections", {})):
            try:
                agent.disconnect_mcp_server(name)
            except Exception:
                pass

    # Preserve mode and configuration across reset
    _keep = {
        "app_mode": st.session_state.get("app_mode"),
        "theme_mode": st.session_state.get("theme_mode", "dark"),
        "cfg_model_preset": st.session_state.get("cfg_model_preset"),
        "cfg_model_custom": st.session_state.get("cfg_model_custom"),
        "cfg_api_key": st.session_state.get("cfg_api_key"),
        "cfg_base_url": st.session_state.get("cfg_base_url"),
        "cfg_fh_api_key": st.session_state.get("cfg_fh_api_key"),
        "cfg_mp_api_key": st.session_state.get("cfg_mp_api_key"),
        "cfg_mode": st.session_state.get("cfg_mode"),
        "cfg_embedding_preset": st.session_state.get("cfg_embedding_preset"),
        "cfg_embedding_custom": st.session_state.get("cfg_embedding_custom"),
        "cfg_embedding_api_key": st.session_state.get("cfg_embedding_api_key"),
    }

    # Clear all state keys — both app state and widget keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Restore preserved keys
    for key, val in _keep.items():
        if val is not None:
            st.session_state[key] = val

    # Use query params to force a clean page load, which fully resets
    # widget keys that would otherwise re-initialize with stale values
    st.query_params["reset"] = "1"
    st.rerun()


def save_upload(uploaded_file, kind: str, auto_dispatch: bool = True) -> None:
    """Save an uploaded file and optionally queue an auto-examine/load."""
    session_dir = Path(st.session_state.session_dir)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    dest = upload_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    dest_str = str(dest)

    # Track the path so the sidebar shows the latest upload
    if kind == "data":
        st.session_state.uploaded_data_path = dest_str
    else:
        st.session_state.uploaded_metadata_path = dest_str

    # Only queue an auto-message the first time we see this file
    upload_key = (kind, dest_str)
    if upload_key not in st.session_state._processed_uploads:
        st.session_state._processed_uploads.add(upload_key)
        if auto_dispatch:
            if kind == "data":
                st.session_state.pending_auto_examine = dest_str
            else:
                st.session_state.pending_auto_load_metadata = dest_str
        st.sidebar.success(f"Uploaded {kind} file: {uploaded_file.name}")


_GLOBAL_META_NAMES = {"metadata.json", "meta.json", "info.json", "experiment.json"}


def save_metadata_to_series(uploaded_files: list, auto_dispatch: bool = True) -> None:
    """Save multiple metadata files into the series directory.

    Sidecar JSONs (per-file metadata) are placed alongside data files
    so that run_analysis can detect them via stem-matching.  If a
    global metadata file is found among the uploads, it is registered
    as uploaded_metadata_path for the orchestrator to load.
    """
    session_dir = Path(st.session_state.session_dir)
    series_dir = session_dir / "uploads" / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    global_meta_path = None
    for f in uploaded_files:
        dest = series_dir / f.name
        dest.write_bytes(f.getvalue())
        if f.name.lower() in _GLOBAL_META_NAMES:
            global_meta_path = str(dest)

    if global_meta_path:
        st.session_state.uploaded_metadata_path = global_meta_path

    st.session_state.uploaded_sidecar_metadata = True

    upload_key = ("metadata_batch", str(series_dir))
    if upload_key not in st.session_state._processed_uploads:
        st.session_state._processed_uploads.add(upload_key)
        st.sidebar.success(
            f"Uploaded {len(uploaded_files)} metadata file(s) to series/"
        )


def save_upload_batch(uploaded_files: list, kind: str, auto_dispatch: bool = True) -> None:
    """Save multiple uploaded files into a subdirectory for series/batch analysis."""
    session_dir = Path(st.session_state.session_dir)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    # Create a subdirectory for the series
    series_dir = upload_dir / "series"
    series_dir.mkdir(exist_ok=True)

    for f in uploaded_files:
        dest = series_dir / f.name
        dest.write_bytes(f.getvalue())

    series_dir_str = str(series_dir)
    st.session_state.uploaded_data_path = series_dir_str

    upload_key = (kind, series_dir_str)
    if upload_key not in st.session_state._processed_uploads:
        st.session_state._processed_uploads.add(upload_key)
        if auto_dispatch:
            st.session_state.pending_auto_examine = series_dir_str
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files to series/")

def _start_simulate_session(
    model: str,
    api_key: str,
    base_url: str,
    mode: str,
    fh_api_key: str,
) -> None:
    """Lightweight session init for simulate mode (no heavy agent)."""
    resolved_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not resolved_key and not base_url:
        st.sidebar.error("Provide an API key or set an environment variable.")
        return

    # Set API key in environment so the simulation agent can find it
    if api_key:
        _m = model.lower()
        if any(x in _m for x in ("gpt", "o1", "o3", "o4")):
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif any(x in _m for x in ("claude", "sonnet", "haiku", "opus")):
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif any(x in _m for x in ("gemini", "gemma")):
            os.environ.setdefault("GOOGLE_API_KEY", api_key)

    prefix = SESSION_DIR_PREFIXES.get("simulate", "simulation_session")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{prefix}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    st.session_state.agent_initialized = True
    st.session_state.session_dir = str(session_dir)
    st.session_state.agent_config = {
        "model": model,
        "api_key": resolved_key,
        "base_url": base_url or "",
        "mode": mode,
        "fh_api_key": fh_api_key or "",
    }
    st.session_state.chat_messages = []
    st.session_state.known_images = set()

    # If HPC was already connected before session start, wire it in now
    _conn = st.session_state.get("hpc_connection")
    _sched = st.session_state.get("hpc_scheduler")
    if _conn and _conn.is_connected:
        _wire_hpc_to_agent(_conn, _sched)

    st.rerun()


def _render_simulate_sidebar_status() -> None:
    """Show HPC and simulation agent status in the sidebar."""
    conn = st.session_state.get("hpc_connection")
    if conn and conn.is_connected:
        st.markdown(f"🟢 **{conn.profile.hostname}**")
        sched = st.session_state.get("hpc_scheduler")
        if sched:
            st.caption(f"Scheduler: {sched.name}")
    else:
        st.caption("🔴 No HPC connection")
        st.caption("Connect via the HPC section above, or work offline.")


def _render_simulate_status() -> None:
    """Show simulate-mode agent status metrics."""
    agent = st.session_state.get("hpc_sim_agent")
    result = st.session_state.get("hpc_gen_result")
    conn = st.session_state.get("hpc_connection")

    r1, r2 = st.columns(2)
    with r1:
        if agent and agent.skill_name:
            st.metric("Engine", agent.skill_name)
        else:
            st.metric("Engine", "—")
    with r2:
        connected = conn is not None and conn.is_connected
        st.metric("HPC", "Connected" if connected else "Offline")

    if result:
        sys_info = result.get("system_info", {})
        st.metric("Last system", sys_info.get("system_category", "—"))
        st.metric("Atoms", f"{sys_info.get('atom_count', '—'):,}")

    tracked = st.session_state.get("hpc_tracked_jobs", {})
    if tracked:
        from scilink.hpc.scheduler import JobStatus
        n_run = sum(1 for j in tracked.values() if j.status == JobStatus.RUNNING)
        n_done = sum(1 for j in tracked.values() if j.status.is_terminal)
        st.metric("Jobs", f"{n_run} running · {n_done} completed")
