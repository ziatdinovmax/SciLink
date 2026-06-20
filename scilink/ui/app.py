"""SciLink Streamlit UI — single-page app."""

import base64
import builtins
import re
import threading
from pathlib import Path

import streamlit as st

from scilink.ui.state import init_session_state, ChatTask, FeedbackRequest
from scilink.ui.components.sidebar import render_sidebar, start_session
from scilink.ui.components.chat_uploads import render_pre_chat_uploads
from scilink.ui.components.file_viewer import render_file_preview
from scilink.ui.components.tools_agents import render_tools_agents_tab
from scilink.ui.components.skills import render_skills_tab
from scilink.ui._features import simulate_enabled
from scilink.ui.output_capture import AgentStoppedError, OutputCapture
from scilink.ui.theme import inject_theme
from scilink.ui.config import AVATAR_USER, AVATAR_AGENT, APP_MODES, SESSION_DIR_PREFIXES

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove terminal ANSI styling. The agent emits it for the console; in the
    HTML log pane / st.code it would otherwise render as literal escape codes."""
    return _ANSI_RE.sub("", text or "")


def _log_to_html(text: str) -> str:
    """ANSI-strip a captured log and HTML-escape it. The agent's 💭 reasoning
    lines render dim+italic (a muted aside); the 🤖 answer header renders
    bold+bright (the deliverable) — the HTML equivalents of the terminal styling.
    💭 is the reserved reasoning marker; a meta-delegated specialist tags its
    reasoning as `💭 🧪` (analysis) / `💭 📋` (planning) — the source emoji rides
    INSIDE the 💭 line, so it stays distinguishable without making 🧪/📋 (heavily
    used as plan/report/step headers elsewhere) trigger thought-styling."""
    import html as _html

    out, in_thought = [], False
    for line in _strip_ansi(text).split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("🤖"):
            # Final-answer header — emphasized, and it ends any reasoning block.
            in_thought = False
            out.append(f'<span style="color:#7fdfff;font-weight:bold">{_html.escape(line)}</span>')
            continue
        if stripped.startswith("💭"):
            in_thought = True
        elif in_thought and not line.startswith("     "):
            in_thought = False
        esc = _html.escape(line)
        if in_thought:
            esc = f'<span style="color:#6ec1e4;font-style:italic">{esc}</span>'
        out.append(esc)
    return "\n".join(out)


def _escape_tildes(text: str) -> str:
    """Escape tildes outside LaTeX ($...$, $$...$$) to prevent Markdown strikethrough."""
    # Split on LaTeX delimiters, preserving them
    parts = re.split(r"(\$\$[\s\S]*?\$\$|\$[^$]+?\$)", text)
    for i, part in enumerate(parts):
        # Odd indices are LaTeX blocks — leave them untouched
        if i % 2 == 0:
            parts[i] = part.replace("~", "\\~")
    return "".join(parts)


def _render_fanout_confirm(ctx: str) -> None:
    """Clean confirmation panel for the meta fan-out launch — replaces the raw
    monospace stdout dump with readable markdown (verdict, join axis, the
    branch list; the long rationale tucked into an expander)."""
    def _g(pat):
        m = re.search(pat, ctx or "")
        return re.sub(r"\s{2,}", " ", m.group(1).strip()) if m else None
    verdict = _g(r"Complementarity verdict\s*:\s*(.+)")
    join = _g(r"Join axis\s*:\s*(.+)")
    rationale = _g(r"Rationale\s*:\s*(.+)")
    branches = [re.sub(r"\s{2,}", " ", b.strip())
                for b in re.findall(r"•\s*(.+)", ctx or "")]
    st.markdown("#### 🔀 Launch parallel multi-dataset analysis?")
    if verdict:
        st.markdown(f"**Complementarity:** {verdict}")
    if join:
        st.markdown(f"**Join axis:** {join}")
    if branches:
        st.markdown("**Branches** — run concurrently, each seeing the others "
                    "as auxiliary:\n" + "\n".join(f"- {b}" for b in branches))
    if rationale:
        st.markdown(f"**Why:** {rationale}")
    st.caption("Branches run autonomously — no per-branch approval pauses.")


_LOGO_DIR = Path(__file__).resolve().parent / "assets"
_LOGO_DARK = _LOGO_DIR / "scilink_logo_v3_dark.svg"
_LOGO_LIGHT = _LOGO_DIR / "scilink_logo_v3_light.svg"

st.set_page_config(page_title="SciLink", layout="wide")

# Handle reset: clear the query param so the next rerun is clean
if st.query_params.get("reset"):
    st.query_params.clear()

inject_theme()
init_session_state()
render_sidebar()


# ── helpers ──────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _find_new_images(summary_only: bool = False) -> list[str]:
    """Return image paths in the session dir not yet shown in chat.

    Skips temporary review files (e.g. first_spectrum_fit_review.png)
    to avoid showing the same fit plot twice during the feedback step.

    For scalarizer debug plots (debug_*.png), only shows the first,
    middle, and last files to give a representative spread without
    flooding the chat.

    If *summary_only* is True, only returns summary grid images
    (files containing ``Summary_Grid`` in their name). This is used
    during the human-feedback step so the user sees only the final
    NMF/PCA grid rather than every per-component plot.
    """
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    # Directories that contain user uploads, not agent output
    _upload_dirs = {"uploads", "knowledge", "code", "data"}
    new = []
    debug_plots = []  # collect scalarizer debug plots separately
    for ext in IMAGE_EXTENSIONS:
        for p in Path(session_dir).rglob(f"*{ext}"):
            if "review" in p.stem:
                continue
            # In bo_artifacts/, show only step dashboards (acq is embedded in dashboard)
            if p.parent.name == "bo_artifacts" and not p.stem.startswith("step_"):
                continue
            # Skip user-uploaded files
            if _upload_dirs & {part for part in p.relative_to(session_dir).parts[:-1]}:
                continue
            s = str(p)
            if s not in st.session_state.known_images:
                # Collect scalarizer debug plots for subsampling
                if p.stem.startswith("debug_"):
                    debug_plots.append(s)
                else:
                    st.session_state.known_images.add(s)
                    new.append(s)

    # Handle scalarizer debug plots based on autonomy level
    if debug_plots:
        for s in debug_plots:
            st.session_state.known_images.add(s)

        # In co-pilot mode, show sample fits inline (first, middle, last)
        # In autopilot/autonomous, skip — user can find them in File Explorer
        agent = st.session_state.get("agent")
        is_copilot = (
            agent is not None
            and hasattr(agent, "autonomy_level")
            and agent.autonomy_level.value == "co_pilot"
        )
        if is_copilot:
            import re
            def _natural_sort_key(s):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
            debug_plots.sort(key=_natural_sort_key)
            selected = [debug_plots[0]]
            if len(debug_plots) > 2:
                selected.append(debug_plots[len(debug_plots) // 2])
            if len(debug_plots) > 1:
                selected.append(debug_plots[-1])
            # Sample fits before BO dashboards (chronological order)
            new[0:0] = selected

    if summary_only:
        new = [p for p in new if "Summary_Grid" in Path(p).stem]
    return new


def _find_feedback_preview_images() -> list[str]:
    """Return preview images specifically meant for the feedback step.

    Finds curve-fitting review images (``*review*``) and hyperspectral
    Summary_Grid images (``*Summary_Grid*``).  The search is scoped to
    the most recently modified ``analysis_*`` directory so that images
    from earlier analyses in the same session are not shown.
    """
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    # Scope to the most recently modified analysis directory so we
    # don't pick up stale Summary_Grid / review images from previous runs.
    search_root = Path(session_dir)
    results_dir = search_root / "results"
    if results_dir.exists():
        analysis_dirs = sorted(
            [d for d in results_dir.iterdir()
             if d.is_dir() and d.name.startswith("analysis_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if analysis_dirs:
            search_root = analysis_dirs[0]
    previews = []
    for ext in IMAGE_EXTENSIONS:
        for p in search_root.rglob(f"*{ext}"):
            if "review" in p.stem or "Summary_Grid" in p.stem:
                previews.append(str(p))
    # Also check scalarizer_outputs for debug plots (planning agents)
    scalarizer_dir = Path(session_dir) / "scalarizer_outputs"
    if scalarizer_dir.exists():
        for ext in IMAGE_EXTENSIONS:
            for p in scalarizer_dir.glob(f"debug_*{ext}"):
                s = str(p)
                if s not in previews:
                    previews.append(s)
    return previews


def _find_code_review_files() -> list[tuple[str, str]]:
    """Return (filename, content) pairs for Python files awaiting review.

    Checks both ``temp_code_review/`` (initial generation) and
    ``temp_code_review_iter/`` (refinement iterations), returning
    files from whichever directory was modified most recently.
    """
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    candidates = [
        Path(session_dir) / "temp_code_review",
        Path(session_dir) / "temp_code_review_iter",
    ]
    # Pick the most recently modified directory that exists
    existing = [d for d in candidates if d.is_dir()]
    if not existing:
        return []
    review_dir = max(existing, key=lambda d: d.stat().st_mtime)
    files = []
    for p in sorted(review_dir.glob("*.py")):
        try:
            files.append((p.name, p.read_text(encoding="utf-8")))
        except Exception:
            files.append((p.name, "(could not read file)"))
    return files


def _find_new_html_reports() -> list[str]:
    """Return HTML report paths in the session dir not yet shown."""
    session_dir = st.session_state.session_dir
    if session_dir is None:
        return []
    new = []
    for p in Path(session_dir).rglob("*.html"):
        s = str(p)
        if s not in st.session_state.known_images:  # reuse the same set
            st.session_state.known_images.add(s)
            new.append(s)
    return new


def _parse_bestofn_review(context: str, prompt: str):
    """Parse the best-of-N candidate review from captured stdout.

    Returns ``(candidates, default_idx)`` where ``candidates`` is a list of
    ``{"idx": int, "label": str}`` for the MOST RECENT review block and
    ``default_idx`` is the judge's pick. Returns ``None`` when the buffer is
    not a best-of-N review — the modal then falls back to the generic text
    box, so any parse miss degrades gracefully (no regression). Works for both
    curve (``R²=``) and image (``score=``) reviews; the agent-side
    ``_get_bestofn_join_approval`` contract is unchanged (the chosen index is
    sent as the same bare digit it already parses).
    """
    # Gate on the CURRENT input() prompt — only the best-of-N choice prompt
    # contains "accept candidate". This prevents a stale review block left in
    # the accumulated session buffer from hijacking a later, unrelated feedback
    # prompt (plan/code/scalarizer/etc.).
    if not prompt or "accept candidate" not in prompt:
        return None
    if not context or "BEST-OF-N CANDIDATES" not in context:
        return None
    import re
    # Only the latest review block (the buffer accumulates the whole session).
    block = context[context.rfind("BEST-OF-N CANDIDATES"):]
    cands: dict[int, str] = {}
    pick = None
    for m in re.finditer(
        r"Candidate (\d+):\s*([^=]+)=([0-9.eE+\-]+),\s*approved=(\w+),"
        r"\s*iterations=(\d+)(.*)", block):
        idx = int(m.group(1))
        metric, value = m.group(2).strip(), m.group(3)
        approved = m.group(4).lower() == "true"
        iters = m.group(5)
        mark = "✓ approved" if approved else "✗ below gate"
        cands[idx] = f"Candidate {idx} — {metric}={value} · {mark} · {iters} iter"
        if "judge pick" in m.group(6).lower():
            pick = idx
    if not cands:
        return None
    if pick is None:
        pm = re.search(r"accept candidate (\d+)", prompt or "")
        pick = int(pm.group(1)) if pm else min(cands)
    ordered = [{"idx": i, "label": cands[i]} for i in sorted(cands)]
    return ordered, pick


def _run_agent_chat(task: ChatTask, agent, user_input: str) -> None:
    """Target for the background thread."""
    import logging

    original_input = builtins.input
    _metadata_cache: dict[str, str] = {}

    def _streamlit_input(prompt: str = "") -> str:
        if task.stopped:
            raise AgentStoppedError("Agent stopped by user")
        context = cap.getvalue()
        # Auto-reply for repeated metadata prompts (same file asked twice)
        if "Context" in prompt and "MISSING METADATA" in context:
            import re
            m = re.search(r"MISSING METADATA FOR:\s*(.+)", context)
            if m:
                fname = m.group(1).strip()
                if fname in _metadata_cache:
                    return _metadata_cache[fname]
        req = FeedbackRequest(prompt=prompt, context=context)
        task.feedback_request = req
        req.event.wait()
        task.feedback_request = None
        if task.stopped:
            raise AgentStoppedError("Agent stopped by user")
        response = req.response or ""
        # Cache metadata descriptions so repeated prompts auto-reply
        if "Context" in prompt and "MISSING METADATA" in context:
            import re
            m = re.search(r"MISSING METADATA FOR:\s*(.+)", context)
            if m:
                _metadata_cache[m.group(1).strip()] = response
        return response

    cap = OutputCapture()
    task.live_capture = cap

    # Route logging output through the capture buffer so that
    # logging.info() messages (used by the verification-correction
    # loop) appear in the live verbose panel alongside print() output.
    # A thread filter ensures each session only captures its own agent's
    # logs, preventing cross-talk when multiple analyses run concurrently.
    log_handler = logging.StreamHandler(cap._buffer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    _this_thread = threading.get_ident()
    # Best-of-N candidate attempts run in worker threads spawned by this
    # chat thread; effective_thread maps them back so their narration stays
    # in the verbose panel without admitting other sessions' threads. During
    # a CONCURRENT fan-out, keep_in_fanout_panel trims each candidate's
    # per-attempt detail to milestones (executing / verification / outcome) so
    # the panel stays readable; a lone attempt and the CLI keep full detail.
    from scilink.utils.log_context import effective_thread, keep_in_fanout_panel
    log_handler.addFilter(
        lambda record: (
            effective_thread(record.thread) == _this_thread
            and keep_in_fanout_panel(
                record.thread, record.getMessage(), record.levelno
            )
        )
    )
    root_logger = logging.getLogger()
    # Lower root to INFO so agent logger.info() messages (execution
    # details, verification steps, R² values) reach the handler.
    # Mute noisy third-party libraries to prevent flooding.
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    for _lib in ("urllib3", "httpx", "httpcore", "google", "openai",
                 "anthropic", "matplotlib", "PIL", "fsspec", "asyncio",
                 "grpc", "absl"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    root_logger.addHandler(log_handler)

    try:
        builtins.input = _streamlit_input
        with cap:
            result = agent.chat(user_input)
        if not task.stopped:
            task.result = result
            task.verbose_log = cap.getvalue()
    except AgentStoppedError:
        # Expected — user clicked Stop. Thread exits cleanly.
        pass
    except Exception as exc:
        if not task.stopped:
            task.error = str(exc)
            task.verbose_log = cap.getvalue()
    finally:
        builtins.input = original_input
        root_logger.removeHandler(log_handler)
        task.live_capture = None
        if not task.stopped:
            task.is_running = False


def _start_task(prompt: str) -> None:
    """Create a ChatTask, launch the agent thread, and rerun."""
    new_task = ChatTask(is_running=True)
    st.session_state.chat_task = new_task
    t = threading.Thread(
        target=_run_agent_chat,
        args=(new_task, st.session_state.agent, prompt),
        daemon=True,
    )
    new_task.thread = t
    t.start()
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# Welcome screen (before session is started)
# ══════════════════════════════════════════════════════════════════
if not st.session_state.agent_initialized:
    # Check if session init or resume was requested from the sidebar
    _pending = st.session_state.pop("_pending_init", None)
    _pending_resume = st.session_state.pop("_pending_resume", None)

    # ── Initializing: dim everything, show centered spinner ──
    if _pending is not None or _pending_resume is not None:
        st.markdown(
            """<style>
            section[data-testid="stSidebar"] { opacity: 0.3; pointer-events: none; }
            </style>""",
            unsafe_allow_html=True,
        )
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown(
                '<div style="display:flex;flex-direction:column;align-items:center;'
                'justify-content:center;min-height:60vh">'
                '<style>'
                '@keyframes init-pulse{'
                '0%,100%{opacity:0.4}50%{opacity:1}'
                '}'
                '</style>'
                '<div style="display:flex;gap:8px;margin-bottom:20px">'
                '<div style="width:10px;height:10px;border-radius:50%;'
                'background:#82B1FF;animation:init-pulse 1.4s ease-in-out infinite"></div>'
                '<div style="width:10px;height:10px;border-radius:50%;'
                'background:#82B1FF;animation:init-pulse 1.4s ease-in-out infinite 0.2s"></div>'
                '<div style="width:10px;height:10px;border-radius:50%;'
                'background:#82B1FF;animation:init-pulse 1.4s ease-in-out infinite 0.4s"></div>'
                '</div>'
                '<p style="color:#82B1FF;font-size:1.2em;font-weight:500;margin:0">'
                f'{"Restoring session..." if _pending_resume else "Initializing agent..."}</p>'
                '<p style="color:#6B7A8C;font-size:0.9em;margin-top:8px">'
                f'{"Loading checkpoint and chat history" if _pending_resume else "Setting up models and tools"}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        if _pending_resume is not None:
            from scilink.ui.components.sidebar import resume_session
            resume_session(**_pending_resume)
        else:
            start_session(**_pending)
        # start_session / resume_session call st.rerun() on success,
        # so we only reach here if initialization failed.
        st.stop()

    # ── Normal welcome screen ────────────────────────────────
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        # Mode selector — centered above the logo
        if st.session_state.app_mode is None:
            st.session_state.app_mode = "meta"
        # Fall back if a stale session_state still says "simulate" but the
        # [sim] extras are no longer installed.
        if st.session_state.app_mode == "simulate" and not simulate_enabled():
            st.session_state.app_mode = "analyze"
        _mode_map = {m["key"]: m for m in APP_MODES}
        # One button per mode; simulate is dropped when [sim] isn't installed.
        _modes = [m for m in APP_MODES
                  if m["key"] != "simulate" or simulate_enabled()]
        st.markdown('<div class="mode-selector-anchor"></div>', unsafe_allow_html=True)
        # One equal column per mode (no side padding). The buttons size to their
        # labels and the row centers/wraps via CSS, so they never truncate or
        # collide regardless of window width or OS display scaling.
        _cols = st.columns(len(_modes))
        for _m, _col in zip(_modes, _cols):
            with _col:
                _btype = ("primary" if st.session_state.app_mode == _m["key"]
                          else "secondary")
                if st.button(_m["label"], type=_btype, width="stretch",
                             key=f"mode_{_m['key']}"):
                    st.session_state.app_mode = _m["key"]
                    st.rerun()
        _cur_mode = _mode_map[st.session_state.app_mode]
        _cur_desc = _cur_mode["description"]
        if _cur_mode.get("beta"):
            _cur_desc = f"Mission Control · {_cur_desc}"
        st.markdown(
            f'<p style="text-align:center;color:#6B7A8C;font-size:0.85em;'
            f'margin-top:-4px;margin-bottom:12px">'
            f'{_cur_desc}</p>',
            unsafe_allow_html=True,
        )

        _is_dark = st.session_state.get("theme_mode", "dark") == "dark"
        _logo = _LOGO_DARK if _is_dark else _LOGO_LIGHT
        if _logo.exists():
            _b64 = base64.b64encode(_logo.read_bytes()).decode()
            if _is_dark:
                st.markdown(
                    '<style>'
                    '@keyframes logo-spin{to{transform:rotate(360deg)}}'
                    '.logo-glow{position:relative;padding:2px;border-radius:14px;overflow:hidden}'
                    '.logo-glow::before{content:"";position:absolute;'
                    'top:-40%;left:-40%;width:180%;height:180%;'
                    'background:conic-gradient('
                    'transparent 0deg,transparent 270deg,#3A4556 300deg,'
                    '#82B1FF 330deg,#FFF 345deg,#82B1FF 355deg,transparent 360deg);'
                    'animation:logo-spin 4s linear infinite;z-index:0}'
                    '.logo-glow>img{position:relative;z-index:1;border-radius:12px;'
                    'display:block;width:100%}'
                    '</style>'
                    f'<div class="logo-glow">'
                    f'<img src="data:image/svg+xml;base64,{_b64}"/>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<img src="data:image/svg+xml;base64,{_b64}" '
                    f'style="border-radius:12px;display:block;width:100%"/>',
                    unsafe_allow_html=True,
                )
        else:
            st.title("SciLink")
        st.markdown(
            '<p style="text-align:center;color:#9E9E9E;font-size:1.1em;'
            'margin-top:12px;margin-bottom:24px">'
            "LLM-powered agents for scientific research automation</p>",
            unsafe_allow_html=True,
        )
        # ── Simulate mode: pre-session connection status ─────────
        if st.session_state.app_mode == "simulate":
            st.markdown(
                '<p style="text-align:center;color:#6B7A8C;font-size:0.95em;'
                'margin-top:8px;margin-bottom:4px">'
                'Generate simulation inputs with AI agents, then optionally '
                'submit to HPC.'
                '</p>',
                unsafe_allow_html=True,
            )
            _conn = st.session_state.get("hpc_connection")
            if _conn and _conn.is_connected:
                st.markdown(
                    f'<p style="text-align:center;font-size:0.85em;margin-bottom:16px">'
                    f'🟢 Connected to <b>{_conn.profile.hostname}</b></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="text-align:center;color:#6B7A8C;font-size:0.85em;'
                    'margin-bottom:16px">'
                    '🔴 No HPC connection — connect via the sidebar before '
                    'starting, or work offline.'
                    '</p>',
                    unsafe_allow_html=True,
                )
            st.stop()
    st.stop()

# ══════════════════════════════════════════════════════════════════
# Active session — Chat + File Explorer tabs
# ══════════════════════════════════════════════════════════════════

if st.session_state.app_mode == "simulate" and simulate_enabled():
    # ── Simulate mode: no agent, just HPC UI ─────────────────
    from scilink.ui.components.simulations import render_simulations_tab
    sim_tab, terminal_note = st.tabs(["Simulations", "About"])
    with sim_tab:
        render_simulations_tab()
    with terminal_note:
        st.markdown(
            "### Simulation Mode\n\n"
            "You're running in **simulation-only** mode — no analysis or "
            "planning agent is active.\n\n"
            "Switch to **Analyze** or **Plan** from the sidebar to use "
            "the full agent framework alongside HPC simulations."
        )
else:
    # ── Analyze / Plan / Explore modes: full agent UI ────────
    # Tabs are built dynamically: Telemetry is appended only in Explore
    # (meta) mode, Simulations only when the simulate extra is enabled.
    _is_meta = st.session_state.app_mode == "meta"
    _tab_labels = ["Chat", "File Explorer", "Tools", "Skills"]
    if _is_meta:
        _tab_labels.append("Telemetry")
    if simulate_enabled():
        _tab_labels.append("Simulations")
    _tabs = st.tabs(_tab_labels)
    chat_tab, files_tab, tools_tab, skills_tab = _tabs[:4]
    _next = 4
    telemetry_tab = None
    if _is_meta:
        telemetry_tab, _next = _tabs[_next], _next + 1
    sim_tab = _tabs[_next] if simulate_enabled() else None

    # ── Chat tab ─────────────────────────────────────────────────────
    with chat_tab:
        # Show a prominent upload zone until the chat conversation starts
        if not st.session_state.chat_messages:
            render_pre_chat_uploads(_start_task)
    
        _avatars = {"user": AVATAR_USER, "assistant": AVATAR_AGENT}
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar=_avatars.get(msg["role"])):
                # Escape tildes outside LaTeX blocks to prevent Markdown strikethrough
                _content = _escape_tildes(msg["content"]) if msg["role"] == "assistant" else msg["content"]
                st.markdown(_content)
                for img_path in msg.get("images", []):
                    try:
                        _img_name = Path(img_path).stem
                        # Show a readable caption for scalarizer debug plots
                        if _img_name.startswith("debug_"):
                            _sample = _img_name[len("debug_"):]
                            st.image(img_path, caption=f"Sample fit: {_sample}")
                        else:
                            st.image(img_path)
                    except Exception:
                        st.caption(f"(image not found: {img_path})")
                for html_path in msg.get("html_reports", []):
                    p = Path(html_path)
                    if p.exists():
                        with st.expander(f"Report: {p.name}"):
                            st.iframe(
                                p.read_text(encoding="utf-8"),
                                height=600,
                            )
                        st.download_button(
                            f"Download {p.name}",
                            data=p.read_bytes(),
                            file_name=p.name,
                            mime="text/html",
                            key=f"dl_html_{html_path}",
                        )
                if msg.get("verbose"):
                    with st.expander("Verbose output"):
                        st.code(_strip_ansi(msg["verbose"]), language="text")
    
        # ── Agent monitoring fragment ─────────────────────────────────
        # Uses @st.fragment to rerun only this section (not the full page)
        # when the agent is working.  run_every="1s" polls the background
        # thread; scope="app" escalates to a full rerun when needed.
        task: ChatTask = st.session_state.chat_task
        _needs_polling = task.is_running and task.feedback_request is None
        _monitor_interval = "1s" if _needs_polling else None
    
        @st.fragment(run_every=_monitor_interval)
        def _agent_monitor():
            task: ChatTask = st.session_state.chat_task
    
            # ── 1. Completion — append result, full rerun to render it ──
            if not task.is_running and (task.result is not None or task.error is not None):
                content = task.result if task.result is not None else f"Error: {task.error}"
                # Strip markdown image tags with local file paths — images are
                # rendered separately via st.image() from _find_new_images()
                content = re.sub(r"!\[[^\]]*\]\([^)]+\)\n?", "", content).strip()
                new_images = _find_new_images()
                new_reports = _find_new_html_reports()
                # When an HTML report is present it already embeds the
                # relevant figures — skip showing raw images separately
                # to avoid duplicate clutter (matches curve fitting UX).
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": content,
                    "images": [] if new_reports else new_images,
                    "html_reports": new_reports,
                    "verbose": task.verbose_log or "",
                })
                st.session_state.chat_task = ChatTask()
                st.rerun(scope="app")
                return
    
            # ── 2. Feedback — render review UI, wait for user action ────
            if task.is_running and task.feedback_request is not None:
                req: FeedbackRequest = task.feedback_request
    
                # Cache preview images so fragment reruns don't lose them.
                # Only mark the preview images themselves as known (not ALL
                # images on disk) so that per-component plots generated
                # before this feedback step still appear on completion.
                #   • curve fitting  → *review*.png  (fit preview)
                #   • hyperspectral  → *Summary_Grid*.jpeg  (NMF/PCA grid)
                if "_feedback_preview_images" not in st.session_state:
                    previews = _find_feedback_preview_images()
                    st.session_state._feedback_preview_images = previews
                    for img in previews:
                        st.session_state.known_images.add(img)
                for img_path in st.session_state._feedback_preview_images:
                    # Caption best-of-N candidate fits with their number so the
                    # plot lines up with the radio selector below; other preview
                    # images (single fit, hyperspectral grid) render as before.
                    _bn = re.search(r"bestofn_candidate_(\d+)_review",
                                    Path(img_path).name)
                    if _bn:
                        st.image(img_path, caption=f"Candidate {int(_bn.group(1))}")
                    else:
                        st.image(img_path)
    
                # Show generated code files during code review
                _ctx_tail_early = (req.context or "")[-1500:]
                if "CODE REVIEW" in _ctx_tail_early or "Review files in" in _ctx_tail_early:
                    if "_code_review_files" not in st.session_state:
                        st.session_state._code_review_files = _find_code_review_files()
                    code_files = st.session_state._code_review_files
                    if code_files:
                        for fname, content in code_files:
                            with st.expander(f"📄 {fname}", expanded=len(code_files) == 1):
                                st.code(content, language="python")
    
                _is_fanout_confirm = ("parallel multi-dataset analysis"
                                      in (req.context or "").lower())
                if _is_fanout_confirm:
                    _render_fanout_confirm(req.context)
                elif req.context:
                    import html as _html

                    display_ctx = req.context
                    lines = display_ctx.split("\n")
                    # Find the last separator block (===…) followed by a
                    # non-empty content line — this is the start of the
                    # review section (e.g. fit result or plan summary).
                    start = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith("=" * 20) and i + 1 < len(lines) and lines[i + 1].strip():
                            start = i
                    if start:
                        display_ctx = "\n".join(lines[start:])
                    # Strip === separator lines
                    display_ctx = re.sub(r"^[=]{10,}\s*$", "", display_ctx, flags=re.MULTILINE)
                    # Strip whitespace-only lines (so they count as truly blank)
                    display_ctx = re.sub(r"^[ \t]+$", "", display_ctx, flags=re.MULTILINE)
                    # Collapse all runs of blank lines into a single newline
                    display_ctx = re.sub(r"\n{2,}", "\n", display_ctx).strip()
                    # Re-insert one blank line before each section header
                    # (lines starting with an emoji) for visual breathing room,
                    # but not between header and its body.
                    display_ctx = re.sub(
                        r"\n(?=[\U0001f300-\U0001fAFF\u2600-\u27BF\u2700-\u27BF])",
                        "\n\n",
                        display_ctx,
                    )
                    # Add an extra blank line after the top-level title
                    # (e.g. "📋 PROPOSED FITTING PLAN" or "📊 FIRST SPECTRUM FIT RESULT")
                    display_ctx = re.sub(
                        r"^(.+(?:PLAN|RESULT|REVIEW).*)$",
                        r"\1\n",
                        display_ctx,
                        count=1,
                        flags=re.MULTILINE,
                    )
                    escaped_ctx = _html.escape(display_ctx)
                    # Estimate height: ~20px per line, clamped to 150-400px
                    n_lines = escaped_ctx.count("\n") + 1
                    box_h = max(150, min(400, n_lines * 20 + 32))
                    st.iframe( 
                        f'<pre style="background:#1e1e1e;margin:0;'
                        f"border:1px solid #333;border-radius:6px;"
                        f"padding:12px 16px;font-family:monospace;"
                        f"font-size:13px;color:#e0e0e0;"
                        f"white-space:pre-wrap;word-wrap:break-word;"
                        f'overflow-y:auto;line-height:1.5">'
                        f"{escaped_ctx}</pre>",
                        height=box_h,
                    )
    
                # Adapt labels based on the type of input prompt.
                # Only check the tail of the context (last ~1500 chars) since
                # the buffer accumulates all stdout from the session.
                _ctx_tail = (req.context or "")[-1500:]
                _prompt = req.prompt or ""
                _is_keep_revert = "revert to original" in _ctx_tail.lower()
                # Best-of-N candidate review: parse the candidate list so we can
                # render a radio selector (scales to any N, unlike one button
                # per candidate). None -> falls through to the generic text box.
                _bestofn_data = _parse_bestofn_review(req.context or "", _prompt)
                # (_is_fanout_confirm computed above, before the context render)
                if "Context" in _prompt or "MISSING METADATA" in _ctx_tail:
                    _input_label = "Describe your data (optional):"
                    _submit_label = "Submit description"
                    _accept_label = "Skip (let agent guess)"
                elif "CODE REVIEW" in _ctx_tail or "Review files in" in _ctx_tail:
                    _input_label = "Your code feedback (optional):"
                    _submit_label = "Request changes"
                    _accept_label = "Approve code"
                elif "REQUESTING FEEDBACK" in _ctx_tail or "Review the plan" in _ctx_tail:
                    _input_label = "Your plan feedback (optional):"
                    _submit_label = "Request changes"
                    _accept_label = "Approve plan"
                elif "SCALARIZER REVIEW" in _ctx_tail:
                    _input_label = "Your extraction feedback (optional):"
                    _submit_label = "Request changes"
                    _accept_label = "Approve extraction"
                else:
                    _input_label = "Your feedback (optional):"
                    _submit_label = "Submit feedback"
                    _accept_label = "Accept as-is"
    
                # Keep/revert prompt: show two simple buttons, no text area
                if _is_keep_revert:
                    col_keep, col_revert = st.columns(2)
                    with col_keep:
                        if st.button("Keep user-guided fit", type="primary", width="stretch"):
                            req.response = "keep"
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                    with col_revert:
                        if st.button("Revert to original fit", type="primary", width="stretch"):
                            req.response = ""
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                # Fan-out launch confirmation: two clear buttons (Launch sends
                # "y", Cancel sends "no" — matching _confirm_fanout's parsing),
                # no text area (there is nothing to edit, only launch/abort).
                elif _is_fanout_confirm:
                    # Anchor so theme.py can force Cancel/Launch to identical
                    # box size (equal columns fix width; this fixes height too).
                    st.markdown(
                        '<span class="fanout-actions-anchor"></span>',
                        unsafe_allow_html=True,
                    )
                    col_cancel, col_go = st.columns(2)
                    with col_cancel:
                        if st.button("Cancel", type="secondary", width="stretch"):
                            req.response = "no"
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                    with col_go:
                        if st.button("🔀 Launch parallel analysis", type="primary",
                                     width="stretch"):
                            req.response = "y"
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                # Best-of-N review: a radio over the candidates (scales to any N)
                # + "Use selected" / "Accept judge's pick". Sends the chosen
                # index as the bare digit _get_bestofn_join_approval parses, or
                # "" to accept the pick — identical to the text-box contract.
                elif _bestofn_data:
                    _cands, _pick = _bestofn_data
                    _opts = [c["idx"] for c in _cands]
                    _labels = {c["idx"]: c["label"] for c in _cands}
                    _default = _opts.index(_pick) if _pick in _opts else 0
                    _choice = st.radio(
                        "Select the candidate to lock:",
                        _opts, index=_default,
                        format_func=lambda i: _labels.get(i, f"Candidate {i}"),
                        key="bestofn_choice",
                    )
                    # Pin both action buttons to equal height/shape (theme.py);
                    # the secondary "Accept" otherwise renders taller than the
                    # primary "Use selected".
                    st.markdown(
                        '<span class="bestofn-actions-anchor"></span>',
                        unsafe_allow_html=True,
                    )
                    col_use, col_pick = st.columns(2)
                    with col_use:
                        if st.button("Use selected", type="primary", width="stretch"):
                            req.response = str(_choice)
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                    with col_pick:
                        if st.button(f"Accept judge's pick (Candidate {_pick})",
                                     type="secondary", width="stretch"):
                            req.response = ""
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                else:
                    feedback = st.text_area(
                        _input_label,
                        key="feedback_input",
                    )
                    col_submit, col_accept = st.columns(2)
                    with col_submit:
                        if st.button(_submit_label, type="primary", disabled=not feedback.strip(),
                                     width="stretch"):
                            req.response = feedback.strip()
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                    with col_accept:
                        if st.button(_accept_label, type="primary", width="stretch"):
                            req.response = ""
                            req.event.set()
                            st.session_state.pop("_feedback_preview_images", None)
                            st.session_state.pop("_code_review_files", None)
                            st.rerun(scope="app")
                return
    
            # ── 3. Live monitoring — fragment auto-reruns, no blocking ──
            if task.is_running:
                _spin_col, _stop_col = st.columns([1, 0.07], vertical_alignment="center")
                with _spin_col:
                    _vibe = st.session_state.get("vibe_theme", "Professional")
                    _spinner_icons = {
                        "Professional": "\u2022",
                        "Positivity boost": "\U0001f43e",
                        "Space nerd": "\U0001f4e1",
                    }
                    _icon = _spinner_icons.get(_vibe, "\u2022")
                    _cls = "agent-spinner-heart" if _vibe != "Professional" else "agent-spinner-dot"
                    st.markdown(
                        '<div class="agent-spinner-container">'
                        f'  <span class="{_cls}">{_icon}</span>'
                        f'  <span class="{_cls}">{_icon}</span>'
                        f'  <span class="{_cls}">{_icon}</span>'
                        '  <span class="agent-spinner-label">Agent is working...</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                with _stop_col:
                    if st.button("■", type="secondary", key="stop_agent_btn",
                                 help="Stop agent"):
                        task.stopped = True
                        task.is_running = False
                        # Signal the agent thread to raise AgentStoppedError
                        # on its next print() call, then capture the log.
                        if task.live_capture:
                            task.live_capture.request_stop()
                            task.verbose_log = task.live_capture.getvalue()
                        else:
                            task.verbose_log = ""
                        task.live_capture = None
                        # Unblock any pending feedback wait
                        if task.feedback_request is not None:
                            task.feedback_request.response = ""
                            task.feedback_request.event.set()
                            task.feedback_request = None
                        _stop_label = (
                            "Planning stopped by user."
                            if st.session_state.app_mode == "plan"
                            else "Analysis stopped by user."
                        )
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": _stop_label,
                            "verbose": task.verbose_log,
                        })
                        st.session_state.chat_task = ChatTask()
                        st.rerun(scope="app")
                live = ""
                if task.live_capture is not None:
                    try:
                        live = task.live_capture.getvalue()
                    except Exception:
                        pass
                if live:
                    show = st.toggle("Show verbose output", key="live_verbose_toggle")
                    # Force-restyle the toggle track in light mode (BaseWeb
                    # uses inline styles that CSS cannot override).
                    if st.session_state.get("theme_mode", "dark") == "light":
                        st.iframe("""<script>
    (function(){
        var doc = window.parent.document;
        var OFF = '#90A4AE', ON = '#6200EE';
        function fix(){
            doc.querySelectorAll('label').forEach(function(lbl){
                var txt = lbl.textContent || '';
                if (txt.indexOf('verbose') < 0 && txt.indexOf('Verbose') < 0) return;
                lbl.querySelectorAll('div').forEach(function(d){
                    var w = d.getBoundingClientRect().width;
                    if (w > 28 && w < 80) {
                        var inp = lbl.querySelector('input');
                        var on = inp && inp.checked;
                        d.style.setProperty('background-color', on ? ON : OFF, 'important');
                    }
                });
            });
        }
        fix();
        new MutationObserver(fix).observe(doc.body,
            {childList:true, subtree:true, attributes:true, attributeFilter:['aria-checked','checked']});
    })();
    </script>""", height=1)
                    if show:
                        tail = "\n".join(live.split("\n")[-200:])
                        escaped = _log_to_html(tail)
                        st.iframe( 
                            f'<pre style="height:280px;overflow-y:auto;margin:0;'
                            f"background:#1e1e1e;padding:8px;border-radius:4px;"
                            f"border:1px solid #333;font-family:monospace;"
                            f"font-size:13px;white-space:pre-wrap;"
                            f'color:#e0e0e0" id="log">{escaped}</pre>'
                            f"<script>var e=document.getElementById('log');"
                            f"e.scrollTop=e.scrollHeight;</script>",
                            height=300,
                        )
    
        _agent_monitor()
    
    # ── Auto-dispatch uploads ────────────────────────────────────
    if not task.is_running:
        data_path = st.session_state.pop("pending_auto_examine", None)
        meta_path = st.session_state.pop("pending_auto_load_metadata", None)
        if data_path and meta_path:
            st.session_state["_upload_preamble"] = (
                f"I uploaded a data file at `{data_path}` "
                f"and a metadata file at `{meta_path}`. "
                f"Please examine the data and load the metadata."
            )
        elif data_path:
            st.session_state["_upload_preamble"] = (
                f"I uploaded a data file at `{data_path}`. "
                f"Please examine it."
            )
        elif meta_path:
            st.session_state["_upload_preamble"] = (
                f"I uploaded a metadata file at `{meta_path}`. "
                f"Please load it."
            )

    _has_conversation = bool(st.session_state.chat_messages)
    # Every mode has a dedicated start screen (goal field + a Start/Analyze
    # button) in render_pre_chat_uploads, so the bottom chat input is for
    # *continuing* a conversation and stays disabled until one exists.
    # (Previously plan/meta left it enabled at the start, duplicating the start
    # form and dropping any text typed into it.)
    _chat_disabled = task.is_running or not _has_conversation
    if _has_conversation:
        if st.session_state.app_mode == "meta":
            _chat_placeholder = "Message mission control..."
        elif st.session_state.app_mode == "plan":
            _chat_placeholder = "Message the planning agent..."
        else:
            _chat_placeholder = "Message the analysis agent..."
    elif st.session_state.app_mode == "analyze":
        _chat_placeholder = "Upload data and click Analyze to start"
    else:
        _chat_placeholder = "Enter your goal in the form above to start"
    # Render the chat input inside the Chat tab container. A top-level
    # st.chat_input is pinned to the bottom of the whole app and shows on
    # every tab; scoping it to chat_tab keeps it on the Chat tab only.
    with chat_tab:
        user_text = st.chat_input(_chat_placeholder, disabled=_chat_disabled)
    _upload_preamble = st.session_state.pop("_upload_preamble", None)

    if _upload_preamble and user_text:
        prompt = f"{_upload_preamble}\n\nAdditional context from user: {user_text}"
    elif _upload_preamble:
        prompt = _upload_preamble
    elif user_text:
        prompt = user_text
    else:
        prompt = None

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        _start_task(prompt)
    
    # ── File Explorer tab ────────────────────────────────────────────
    
    _FILE_TYPE_ICONS = {
        ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".tif": "🖼", ".tiff": "🖼",
        ".csv": "📊", ".tsv": "📊", ".xlsx": "📊",
        ".json": "📋", ".npy": "🔢",
        ".html": "📄", ".txt": "📝", ".md": "📝", ".log": "📝",
        ".py": "🐍",
    }
    
    
    def _discover_sessions(current_session_dir: str | None) -> list[tuple[str, Path]]:
        """Return (display_label, path) for session directories matching the current mode."""
        if current_session_dir is None:
            return []
        parent = Path(current_session_dir).parent
        app_mode = st.session_state.app_mode or "analyze"
        prefix = SESSION_DIR_PREFIXES.get(app_mode, "analysis_session")
        sessions = sorted(
            parent.glob(f"{prefix}_*"),
            key=lambda p: p.name,
            reverse=True,
        )
        current = Path(current_session_dir)
        result: list[tuple[str, Path]] = []
        for s in sessions:
            # Parse timestamp from directory name like <prefix>_20260223_201729
            parts = s.name.removeprefix(f"{prefix}_")
            try:
                label = f"{parts[:4]}-{parts[4:6]}-{parts[6:8]} {parts[9:11]}:{parts[11:13]}:{parts[13:15]}"
            except (IndexError, ValueError):
                label = s.name
            if s.resolve() == current.resolve():
                label += " (current)"
            result.append((label, s))
        return result
    
    
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    
    with files_tab:
        sessions = _discover_sessions(st.session_state.session_dir)
        if not sessions:
            st.info("Start a session first to browse output files.")
        else:
            labels = [s[0] for s in sessions]
            paths = [s[1] for s in sessions]
            selected_idx = st.selectbox(
                "Session",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key="session_selector",
            )
            session_path = paths[selected_idx]
            tree_col, preview_col = st.columns([1, 2])
    
            # Clear selected file when switching sessions
            sel = st.session_state.get("selected_preview_file")
            if sel and not str(sel).startswith(str(session_path)):
                st.session_state.selected_preview_file = None
    
            with tree_col:
                header_col, refresh_col = st.columns([3, 1])
                with header_col:
                    st.subheader("Session Files")
                with refresh_col:
                    # Streamlit only re-runs on user interaction, so agent
                    # output files written while the page is idle don't
                    # appear until the next click. This button forces a
                    # rerun, which re-evaluates the rglob below and shows
                    # any newly-created files.
                    if st.button("🔄 Refresh", key="files_refresh_btn",
                                 width="stretch",
                                 help="Re-scan the session directory for new files."):
                        st.rerun()
    
                all_files = list(session_path.rglob("*"))
                has_files = any(f.is_file() for f in all_files)
    
                if not has_files:
                    st.caption("No files found yet.")
                else:
                    def _render_dir(dir_path: Path, depth: int = 0) -> None:
                        """Render directory tree using expanders."""
                        items = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
                        dirs = [i for i in items if i.is_dir()]
                        files = [i for i in items if i.is_file()]
    
                        for d in dirs:
                            child_count = sum(1 for _ in d.rglob("*") if _.is_file())
                            if child_count == 0:
                                continue
                            with st.expander(f"📁 {d.name} ({child_count})", expanded=(depth == 0)):
                                _render_dir(d, depth + 1)
    
                        def _truncate_name(name: str, max_len: int = 25) -> str:
                            stem, suffix = Path(name).stem, Path(name).suffix
                            if len(name) <= max_len:
                                return name
                            keep = max_len - len(suffix) - 1
                            return stem[:keep] + "\u2026" + suffix
    
                        def _render_files(file_list: list[Path]) -> None:
                            for f in file_list:
                                icon = _FILE_TYPE_ICONS.get(f.suffix.lower(), "📄")
                                size = _format_size(f.stat().st_size)
                                is_sel = st.session_state.get("selected_preview_file") == str(f)
                                btn_type = "primary" if is_sel else "secondary"
                                display_name = _truncate_name(f.name)
                                if st.button(
                                    f"{icon}  {display_name}  ({size})",
                                    key=f"fbtn_{f.relative_to(session_path)}",
                                    width="stretch",
                                    type=btn_type,
                                    help=f.name,
                                ):
                                    st.session_state.selected_preview_file = str(f)
                                    st.rerun()
    
                        if files and depth == 0:
                            with st.expander(f"📄 Session ({len(files)})", expanded=True):
                                _render_files(files)
                        else:
                            _render_files(files)
    
                    _render_dir(session_path)
    
            selected_file = None
            sel_path = st.session_state.get("selected_preview_file")
            if sel_path:
                selected_file = Path(sel_path)
    
            with preview_col:
                if selected_file and selected_file.exists():
                    render_file_preview(selected_file)
                else:
                    st.caption("Select a file to preview.")
    
    # ── Tools tab ────────────────────────────────────────────────────
    with tools_tab:
        render_tools_agents_tab()
    
    # ── Skills tab ───────────────────────────────────────────────────
    with skills_tab:
        render_skills_tab()
    
    # ── Telemetry tab (Explore mode only) ────────────────────────────
    if telemetry_tab is not None:
        from scilink.ui.components.telemetry import render_telemetry_tab
        with telemetry_tab:
            render_telemetry_tab()

    # ── Simulations tab ──────────────────────────────────────────────
    if sim_tab is not None:
        from scilink.ui.components.simulations import render_simulations_tab
        with sim_tab:
            render_simulations_tab()
