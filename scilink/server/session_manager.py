"""Server-side session service: create / resume / discover / snapshot.

Ports the session-lifecycle logic of the Streamlit sidebar
(scilink/ui/components/sidebar.py — start_session :801, the three agent
factories :883-980, _discover_resumable_sessions :983,
_convert_chat_history_for_display :1032, _collect_restored_deliverables
:1047, resume_session :1079) with ``st.session_state`` replaced by explicit
parameters and a per-process registry keyed by session-dir name.

The orchestrators are consumed exactly as Streamlit consumes them; the one
behavioral divergence is the session root: Streamlit uses ``Path.cwd()``
implicitly, the server takes it as an explicit ``--session-root``
(defaulting to cwd) that also fences file serving.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scilink.providers import provider_for
from scilink.ui.config import SESSION_DIR_PREFIXES
from scilink.ui.session_meta import load_session_name, session_label

from .artifacts import ArtifactTracker
from .events import EventBuffer
from .runner import TurnState


class SessionError(Exception):
    """User-facing session-lifecycle failure (bad credentials, bad mode…)."""


@dataclass
class WebSession:
    id: str                         # session dir name
    session_dir: str                # absolute path
    mode: str                       # meta | analyze | plan
    model: str
    autonomy: str                   # co-pilot | autopilot | autonomous
    agent: Any
    events: EventBuffer = field(default_factory=EventBuffer)
    tracker: ArtifactTracker = None  # type: ignore[assignment]
    chat_messages: List[Dict[str, Any]] = field(default_factory=list)
    turn: Optional[TurnState] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.tracker is None:
            self.tracker = ArtifactTracker(self.session_dir)

    @property
    def status(self) -> str:
        turn = self.turn
        if turn is not None and turn.is_running:
            if turn.pending_question is not None:
                return "awaiting_input"
            return "running"
        return "idle"


def _resolve_credentials(model: str, api_key: str, base_url: str,
                         provider_fields: Dict[str, str]) -> Optional[str]:
    """Port of the start_session credential preamble (sidebar.py:811-829).

    Applies the provider spec (env exports for e.g. Bedrock), returns the
    resolved api_key for the agent constructor, and raises SessionError with
    the provider's own message when no credential path exists.
    """
    spec = provider_for(model)
    values = {f.name: provider_fields.get(f.name, f.default)
              for f in spec.fields}
    auth = spec.apply(pasted_key=api_key, values=values, base_url=base_url)
    os.environ.update({k: v for k, v in auth.env.items() if v})

    if auth.env:
        resolved_key = None
    else:
        resolved_key = (auth.api_key
                        or os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("ANTHROPIC_API_KEY"))

    if not (api_key or base_url or any(os.environ.get(e) for e in spec.cred_env)):
        raise SessionError(spec.cred_error)
    return resolved_key


# ── agent factories (ports of sidebar.py:883-980) ────────────────

def _init_analysis_agent(session_dir: Path, api_key, model, base_url,
                         autonomy, fh_api_key):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, AnalysisOrchestratorAgent)
    mode_map = {"co-pilot": AnalysisMode.CO_PILOT,
                "autopilot": AnalysisMode.AUTOPILOT,
                "autonomous": AnalysisMode.AUTONOMOUS}
    return AnalysisOrchestratorAgent(
        base_dir=str(session_dir), api_key=api_key, model_name=model,
        base_url=base_url or None, analysis_mode=mode_map[autonomy],
        futurehouse_api_key=fh_api_key or None)


def _init_planning_agent(session_dir: Path, api_key, model, base_url,
                         autonomy, fh_api_key, objective, session_root: Path,
                         embedding_model=None, embedding_api_key=None):
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel, PlanningOrchestratorAgent)
    mode_map = {"co-pilot": AutonomyLevel.CO_PILOT,
                "autopilot": AutonomyLevel.AUTOPILOT,
                "autonomous": AutonomyLevel.AUTONOMOUS}
    knowledge_dir = session_root / "kb_storage"   # shared across sessions
    knowledge_dir.mkdir(exist_ok=True)
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
        objective=(objective or "").strip() or "Undefined Research Goal",
        base_dir=str(session_dir), api_key=api_key, model_name=model,
        base_url=base_url or None, autonomy_level=mode_map[autonomy],
        futurehouse_api_key=fh_api_key or None,
        knowledge_dir=str(knowledge_dir), code_dir=str(code_dir),
        data_dir=str(data_dir), **kwargs)


def _init_meta_agent(session_dir: Path, api_key, model, base_url,
                     autonomy, fh_api_key,
                     embedding_model=None, embedding_api_key=None):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaMode, MetaOrchestratorAgent)
    mode_map = {"autopilot": MetaMode.AUTOPILOT,
                "autonomous": MetaMode.AUTONOMOUS}
    kwargs = {}
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    if embedding_api_key:
        kwargs["embedding_api_key"] = embedding_api_key
    return MetaOrchestratorAgent(
        base_dir=str(session_dir), api_key=api_key, model_name=model,
        base_url=base_url or None, meta_mode=mode_map[autonomy],
        futurehouse_api_key=fh_api_key or None, **kwargs)


# ── history / deliverable helpers (ports of sidebar.py:1032-1076) ─

def convert_chat_history_for_display(history: list) -> list:
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def collect_restored_deliverables(session_path: Path) -> tuple:
    """(md_paths, html_paths) — only the most recent marked deliverable."""
    from scilink.agents.planning_agents.user_interface import load_deliverables
    candidates = []
    for entry in load_deliverables(session_path):
        if not entry.get("deliverable"):
            continue
        p = Path(entry.get("path", ""))
        if p.exists() and p.suffix.lower() in (".md", ".html", ".htm"):
            candidates.append(p)
    if not candidates:
        return [], []
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    if latest.suffix.lower() == ".md":
        return [str(latest)], []
    return [], [str(latest)]


class SessionManager:
    def __init__(self, session_root: Path) -> None:
        self.session_root = session_root.resolve()
        self._sessions: Dict[str, WebSession] = {}
        self._lock = threading.Lock()

    # -- lookup ---------------------------------------------------------
    def get(self, session_id: str) -> Optional[WebSession]:
        return self._sessions.get(session_id)

    def list_live(self) -> List[Dict[str, Any]]:
        return [{
            "id": s.id, "mode": s.mode, "model": s.model,
            "status": s.status, "name": load_session_name(s.session_dir),
            "n_messages": len(s.chat_messages),
        } for s in self._sessions.values()]

    # -- discovery (port of sidebar.py:983-1029) ------------------------
    def discover_resumable(self, mode: str) -> List[Dict[str, Any]]:
        prefix = SESSION_DIR_PREFIXES.get(mode, "analysis_session")
        sessions = sorted(self.session_root.glob(f"{prefix}_*"),
                          key=lambda p: p.name, reverse=True)
        result = []
        for s in sessions:
            if not s.is_dir() or s.name in self._sessions:
                continue
            has_checkpoint = (s / "checkpoint.json").exists()
            has_chat = (s / "chat_history.json").exists()
            if not has_checkpoint and not has_chat:
                continue
            summary: Dict[str, Any] = {}
            if has_checkpoint:
                try:
                    ckpt = json.loads((s / "checkpoint.json").read_text())
                    summary["analysis_count"] = len(ckpt.get("analysis_results", []))
                    dp = ckpt.get("current_data_path")
                    if dp:
                        summary["data_file"] = Path(dp).name
                except Exception:
                    pass
            if has_chat and "analysis_count" not in summary:
                try:
                    hist = json.loads((s / "chat_history.json").read_text())
                    summary["message_count"] = sum(
                        1 for m in hist if m.get("role") == "user")
                except Exception:
                    pass
            result.append({
                "id": s.name,
                "label": session_label(s, prefix),
                "has_checkpoint": has_checkpoint,
                "has_chat_history": has_chat,
                "summary": summary,
            })
        return result

    # -- create ---------------------------------------------------------
    def create(self, *, mode: str, model: str, autonomy: str, api_key: str,
               base_url: str, provider_fields: Dict[str, str],
               fh_api_key: str, mp_api_key: str, objective: str = "",
               embedding_model: Optional[str] = None,
               embedding_api_key: Optional[str] = None) -> WebSession:
        if mode not in ("meta", "analyze", "plan"):
            raise SessionError(f"Unsupported mode: {mode!r}")
        if mode == "meta" and autonomy == "co-pilot":
            raise SessionError("Meta mode supports autopilot or autonomous only.")

        resolved_key = _resolve_credentials(model, api_key, base_url,
                                            provider_fields)
        if mp_api_key:
            import scilink
            scilink.set_api_key("materials_project", mp_api_key)

        prefix = SESSION_DIR_PREFIXES.get(mode, "session")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = (self.session_root / f"{prefix}_{ts}").resolve()
        session_dir.mkdir(parents=True, exist_ok=True)

        # Sandbox consent is validated at the endpoint; mirror the Streamlit
        # auto-approval here (sidebar.py:848).
        import scilink.executors as executors
        executors._GLOBAL_SANDBOX_APPROVED = True

        try:
            if mode == "meta":
                agent = _init_meta_agent(
                    session_dir, resolved_key, model, base_url, autonomy,
                    fh_api_key, embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key)
            elif mode == "plan":
                agent = _init_planning_agent(
                    session_dir, resolved_key, model, base_url, autonomy,
                    fh_api_key, objective, self.session_root,
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key)
            else:
                agent = _init_analysis_agent(
                    session_dir, resolved_key, model, base_url, autonomy,
                    fh_api_key)
        except Exception as exc:
            raise SessionError(f"Failed to initialize agent: {exc}") from exc

        session = WebSession(id=session_dir.name, session_dir=str(session_dir),
                             mode=mode, model=model, autonomy=autonomy,
                             agent=agent)
        with self._lock:
            self._sessions[session.id] = session
        return session

    # -- resume (port of sidebar.py:1079-1241) --------------------------
    def resume(self, *, resume_dir: str, mode: str, model: str, autonomy: str,
               api_key: str, base_url: str, provider_fields: Dict[str, str],
               fh_api_key: str, mp_api_key: str,
               embedding_model: Optional[str] = None,
               embedding_api_key: Optional[str] = None) -> WebSession:
        # resume_dir is a dir NAME under the session root, never a path.
        if "/" in resume_dir or "\\" in resume_dir or resume_dir in (".", ".."):
            raise SessionError("Invalid session directory name.")
        session_path = (self.session_root / resume_dir).resolve()
        if session_path.parent != self.session_root or not session_path.is_dir():
            raise SessionError(f"No such session: {resume_dir}")
        if resume_dir in self._sessions:
            return self._sessions[resume_dir]

        resolved_key = _resolve_credentials(model, api_key, base_url,
                                            provider_fields)
        if mp_api_key:
            import scilink
            scilink.set_api_key("materials_project", mp_api_key)

        import scilink.executors as executors
        executors._GLOBAL_SANDBOX_APPROVED = True

        kwargs = {}
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        if embedding_api_key:
            kwargs["embedding_api_key"] = embedding_api_key
        try:
            if mode == "meta":
                from scilink.agents.meta_agent.meta_orchestrator import (
                    MetaMode, MetaOrchestratorAgent)
                agent = MetaOrchestratorAgent.restore_from_checkpoint(
                    base_dir=str(session_path), api_key=resolved_key,
                    model_name=model, base_url=base_url or None,
                    meta_mode={"autopilot": MetaMode.AUTOPILOT,
                               "autonomous": MetaMode.AUTONOMOUS}[autonomy],
                    futurehouse_api_key=fh_api_key or None, **kwargs)
            elif mode == "plan":
                from scilink.agents.planning_agents.planning_orchestrator import (
                    AutonomyLevel, PlanningOrchestratorAgent)
                agent = PlanningOrchestratorAgent.restore_from_checkpoint(
                    base_dir=str(session_path), api_key=resolved_key,
                    model_name=model, base_url=base_url or None,
                    autonomy_level={"co-pilot": AutonomyLevel.CO_PILOT,
                                    "autopilot": AutonomyLevel.AUTOPILOT,
                                    "autonomous": AutonomyLevel.AUTONOMOUS}[autonomy],
                    futurehouse_api_key=fh_api_key or None, **kwargs)
            elif mode == "analyze":
                from scilink.agents.exp_agents.analysis_orchestrator import (
                    AnalysisMode, AnalysisOrchestratorAgent)
                agent = AnalysisOrchestratorAgent.restore_from_checkpoint(
                    base_dir=str(session_path), api_key=resolved_key,
                    model_name=model, base_url=base_url or None,
                    analysis_mode={"co-pilot": AnalysisMode.CO_PILOT,
                                   "autopilot": AnalysisMode.AUTOPILOT,
                                   "autonomous": AnalysisMode.AUTONOMOUS}[autonomy],
                    futurehouse_api_key=fh_api_key or None)
            else:
                raise SessionError(f"Unsupported mode: {mode!r}")
        except SessionError:
            raise
        except Exception as exc:
            raise SessionError(f"Failed to restore session: {exc}") from exc

        display_messages: List[Dict[str, Any]] = []
        chat_path = session_path / "chat_history.json"
        if chat_path.exists():
            try:
                raw = json.loads(chat_path.read_text())
                display_messages = convert_chat_history_for_display(raw)
            except Exception:
                pass

        # Re-embed the latest deliverable (chat attachments are not persisted).
        def _rel(p: str) -> str:
            try:
                return str(Path(p).resolve().relative_to(session_path))
            except ValueError:
                return p
        md_docs, html_docs = collect_restored_deliverables(session_path)
        if md_docs or html_docs:
            display_messages.append({
                "role": "assistant",
                "content": ("📎 Latest deliverable from this session "
                            "(earlier versions are in the Files tab):"),
                "md_reports": [{"path": _rel(p), "name": Path(p).name,
                                "title": Path(p).stem.replace("_", " ").title()}
                               for p in md_docs],
                "html_reports": [{"path": _rel(p), "name": Path(p).name}
                                 for p in html_docs],
            })

        session = WebSession(id=session_path.name,
                             session_dir=str(session_path), mode=mode,
                             model=model, autonomy=autonomy, agent=agent,
                             chat_messages=display_messages)
        session.tracker.mark_all_existing()
        with self._lock:
            self._sessions[session.id] = session
        return session

    # -- snapshot -------------------------------------------------------
    def snapshot(self, session: WebSession) -> Dict[str, Any]:
        pending = None
        turn = session.turn
        if turn is not None and turn.pending_question is not None:
            pending = turn.pending_question.presented
        # Cursor BEFORE the log read: a chunk emitted in between shows up in
        # both the snapshot log and the post-cursor stream (harmless brief
        # duplication) instead of being lost from a reattaching client.
        cursor = session.events.cursor
        live_log = ""
        if turn is not None and turn.is_running and turn.live_capture is not None:
            try:
                live_log = turn.live_capture.getvalue()
            except Exception:
                pass
        return {
            "id": session.id,
            "mode": session.mode,
            "model": session.model,
            "autonomy": session.autonomy,
            "status": session.status,
            "name": load_session_name(session.session_dir),
            "session_dir": session.session_dir,
            # Copy: the runner thread appends to this list mid-turn, and
            # FastAPI serializes the snapshot outside the session lock.
            "chat_messages": list(session.chat_messages),
            "pending_question": pending,
            # The narration captured so far this turn, so a reattaching
            # client can show the verbose panel mid-run.
            "live_log": live_log,
            # Start the SSE stream with ?after=<event_cursor> so events
            # already reflected in this snapshot are not replayed.
            "event_cursor": cursor,
        }
