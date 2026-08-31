"""FastAPI application factory for the SciLink web backend.

All endpoints under ``/api/v1``. Endpoints that construct/restore agents or
dispatch turns are sync ``def`` so FastAPI runs them in its threadpool —
the event loop (and every SSE heartbeat) stays live during multi-second
agent construction. The SSE generator is synchronous too (one threadpool
thread per open stream — fine for the local single-user posture).
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from scilink.providers import provider_for
from scilink.ui.config import (
    APP_MODES,
    EMBEDDING_MODEL_OPTIONS,
    MODEL_OPTIONS,
    resolve_prefill,
)
from scilink.ui.session_meta import save_session_name

from . import files as files_mod
from . import runner
from .schemas import (
    CreateSessionRequest,
    FeedbackResponseRequest,
    RenameSessionRequest,
    SendMessageRequest,
)
from .session_manager import SessionError, SessionManager, WebSession

# Consent text mirrored from the Streamlit sidebar checkbox (sidebar.py:382).
CONSENT_TEXT = ("I understand that the agent will execute generated "
                "Python code on my machine")


def create_app(session_root: Path, serve_frontend: bool = True) -> FastAPI:
    app = FastAPI(title="SciLink Web", docs_url="/api/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )
    manager = SessionManager(session_root)
    app.state.manager = manager

    def _session_or_404(session_id: str) -> WebSession:
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(404, f"No live session {session_id!r} — "
                                     "create or resume it first.")
        return session

    # ── config ───────────────────────────────────────────────────

    @app.get("/api/v1/config")
    def get_config(model: str = "", base_url: str = ""):
        """Static UI config + credential AVAILABILITY (never values)."""
        modes = [m for m in APP_MODES if m["key"] != "simulate"]
        model_q = model or MODEL_OPTIONS[0]
        prefill = resolve_prefill(model_q, existing_base_url=base_url)
        spec = provider_for(model_q)
        return {
            "modes": modes,
            "models": MODEL_OPTIONS,
            "embedding_models": EMBEDDING_MODEL_OPTIONS,
            "autonomy_options": {
                "meta": ["autopilot", "autonomous"],
                "analyze": ["co-pilot", "autopilot", "autonomous"],
                "plan": ["co-pilot", "autopilot", "autonomous"],
            },
            "consent_text": CONSENT_TEXT,
            "provider": {
                "name": spec.name,
                "key_label": spec.key_label,
                "fields": [{
                    "name": f.name, "label": f.label, "kind": f.kind,
                    "options": list(f.options), "default": f.default,
                    "help": f.help,
                } for f in spec.fields],
                "cred_error": spec.cred_error,
            },
            "credentials": {
                field: {"env_var": env, "is_set": bool(value)}
                for field, (value, env) in prefill.items()
            },
        }

    # ── sessions ─────────────────────────────────────────────────

    @app.get("/api/v1/sessions")
    def list_sessions(mode: str = "meta"):
        return {"live": manager.list_live(),
                "resumable": manager.discover_resumable(mode)}

    @app.post("/api/v1/sessions")
    def create_session(body: CreateSessionRequest):
        if not body.consent:
            raise HTTPException(400, "Consent to code execution is required "
                                     "to start a session.")
        try:
            if body.resume_dir:
                session = manager.resume(
                    resume_dir=body.resume_dir, mode=body.mode,
                    model=body.model, autonomy=body.autonomy,
                    api_key=body.api_key, base_url=body.base_url,
                    provider_fields=body.provider_fields,
                    fh_api_key=body.fh_api_key, mp_api_key=body.mp_api_key,
                    embedding_model=body.embedding_model,
                    embedding_api_key=body.embedding_api_key)
            else:
                session = manager.create(
                    mode=body.mode, model=body.model, autonomy=body.autonomy,
                    api_key=body.api_key, base_url=body.base_url,
                    provider_fields=body.provider_fields,
                    fh_api_key=body.fh_api_key, mp_api_key=body.mp_api_key,
                    objective=body.objective,
                    embedding_model=body.embedding_model,
                    embedding_api_key=body.embedding_api_key)
        except SessionError as exc:
            raise HTTPException(400, str(exc))
        return manager.snapshot(session)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str):
        return manager.snapshot(_session_or_404(session_id))

    @app.patch("/api/v1/sessions/{session_id}")
    def rename_session(session_id: str, body: RenameSessionRequest):
        session = _session_or_404(session_id)
        if not save_session_name(session.session_dir, body.name,
                                 named_by="user"):
            raise HTTPException(400, "Could not save the session name.")
        session.events.emit("session_named", {"name": body.name.strip()[:80]})
        return {"ok": True}

    # ── turns ────────────────────────────────────────────────────

    @app.post("/api/v1/sessions/{session_id}/messages", status_code=202)
    def send_message(session_id: str, body: SendMessageRequest):
        session = _session_or_404(session_id)
        content = body.content.strip()
        if not content:
            raise HTTPException(400, "Empty message.")
        with session.lock:
            if session.turn is not None and session.turn.is_running:
                raise HTTPException(409, "A turn is already running.")
            runner.start_turn(session, content)
        return {"status": "running"}

    @app.post("/api/v1/sessions/{session_id}/stop")
    def stop(session_id: str):
        session = _session_or_404(session_id)
        stopped = runner.request_stop(session)
        return {"stopped": stopped}

    @app.post("/api/v1/sessions/{session_id}/feedback")
    def feedback(session_id: str, body: FeedbackResponseRequest):
        session = _session_or_404(session_id)
        turn = session.turn
        pending = turn.pending_question if turn is not None else None
        if pending is None or pending.hreq.id != body.request_id:
            raise HTTPException(404, "No such pending question.")
        pending.response = body.response
        pending.event.set()
        return {"ok": True}

    # ── SSE ──────────────────────────────────────────────────────

    @app.get("/api/v1/sessions/{session_id}/events")
    def events(session_id: str, after: Optional[int] = None,
               last_event_id: Optional[str] = Header(default=None)):
        """SSE stream. ``Last-Event-ID`` (reconnects) wins over ``after``
        (initial attach from a snapshot's ``event_cursor``)."""
        session = _session_or_404(session_id)
        try:
            cursor = int(last_event_id) if last_event_id else after
        except ValueError:
            cursor = after
        return StreamingResponse(
            session.events.sse_stream(cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"})

    # ── files ────────────────────────────────────────────────────

    @app.post("/api/v1/sessions/{session_id}/uploads")
    def upload(session_id: str, category: str = Form(...),
               files: list[UploadFile] = File(...)):
        session = _session_or_404(session_id)
        try:
            payload = [(f.filename or "", f.file.read()) for f in files]
            return files_mod.save_uploads(session.session_dir, category,
                                          payload)
        except files_mod.UploadError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/v1/sessions/{session_id}/files")
    def get_file(session_id: str, path: str):
        session = _session_or_404(session_id)
        try:
            target = files_mod.resolve_safe(session.session_dir, path)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        if not target.is_file():
            raise HTTPException(404, f"No such file: {path}")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type)

    # ── static frontend (production) ─────────────────────────────

    if serve_frontend:
        dist = Path(__file__).resolve().parents[2] / "webui" / "dist"
        if dist.is_dir():
            from fastapi.staticfiles import StaticFiles

            class _SPAStaticFiles(StaticFiles):
                async def get_response(self, path, scope):
                    from starlette.exceptions import HTTPException as SHTTP
                    try:
                        return await super().get_response(path, scope)
                    except SHTTP as exc:
                        if exc.status_code == 404:
                            return await super().get_response("index.html", scope)
                        raise

            app.mount("/", _SPAStaticFiles(directory=str(dist), html=True),
                      name="webui")

    return app
