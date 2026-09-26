"""FastAPI application factory for the SciLink web backend.

All endpoints under ``/api/v1``. Endpoints that construct/restore agents or
dispatch turns are sync ``def`` so FastAPI runs them in its threadpool —
the event loop (and every SSE heartbeat) stays live during multi-second
agent construction. The SSE generator is synchronous too (one threadpool
thread per open stream — fine for the local single-user posture).
"""

from __future__ import annotations

import hmac
import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from scilink.providers import provider_for
from scilink.ui.config import (
    APP_MODES,
    EMBEDDING_MODEL_OPTIONS,
    MODEL_OPTIONS,
    resolve_prefill,
    resolve_embedding_prefill,
)
from scilink.ui import vocabulary as _vocab
from scilink.ui.session_meta import save_session_name

from . import files as files_mod
from . import runner
from .auth import _USERNAME_OK, COOKIE_NAME, DEFAULT_USER, AuthConfig, AuthMiddleware, user_root
from .schemas import (
    CreateSessionRequest,
    FeedbackResponseRequest,
    FolderCheckRequest,
    LoginRequest,
    MCPConnectRequest,
    MemoryApplyRequest,
    MemoryCheckRequest,
    MemoryConsolidateRequest,
    MemoryEditRequest,
    MemoryEnabledRequest,
    MemoryIdsRequest,
    MemoryUpgradeRequest,
    PlanDirsRequest,
    RenameSessionRequest,
    SendMessageRequest,
    DrainRequest,
)
from .session_manager import SessionError, SessionManager, WebSession

NO_BUNDLE_MESSAGE = (
    "SciLink web UI bundle not found.\n\n"
    "The API is up (see /api/docs), but the React bundle is built at release "
    "time and is not in git. From a repository checkout, build it once:\n\n"
    "    scripts/build_webui.sh        # or: cd webui && npm run build\n\n"
    "Release wheels (pip install scilink) ship the bundle already.\n")

# The code-execution consent sentence — shared with the terminal shell's
# startup confirmation through the vocabulary.
CONSENT_TEXT = _vocab.CONSENT_TEXT


def create_app(session_root: Path, serve_frontend: bool = True,
               auth: Optional[AuthConfig] = None,
               local_files: bool = True) -> FastAPI:
    """``auth=None`` is the local tool: no authentication, one implicit
    user whose sessions live in ``session_root``. With an ``AuthConfig``
    every ``/api/v1`` call must carry a token (bearer header or the login
    cookie); with per-user tokens each user gets ``<root>/users/<name>/``
    and an isolated live-session registry. ``local_files=False`` (the CLI
    sets it for a non-loopback bind) disables the endpoints that read
    arbitrary paths on the server's machine (pasted folders, plan dirs)
    — they only make sense when the browser and the server share a host."""
    from .cli import _headless_matplotlib
    _headless_matplotlib()
    app = FastAPI(title="SciLink Web", docs_url="/api/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
        expose_headers=["X-Preview-Kind"],
    )
    app.add_middleware(AuthMiddleware, auth=auth)
    app.state.auth = auth
    app.state.local_files = local_files
    # One SessionManager per user root (a single shared one when auth is
    # off or a single token is used) — the isolation boundary.
    managers: dict = {}
    session_root = session_root.resolve()

    def _user(request: Request) -> str:
        user = getattr(request.state, "user", None)
        return user or DEFAULT_USER

    confined = bool(auth is not None and auth.multi_user)

    def _mgr(request: Request) -> SessionManager:
        user = _user(request)
        mgr = managers.get(user)
        if mgr is None:
            mgr = SessionManager(user_root(session_root, auth, user),
                                 confined=confined, fenced=not local_files)
            managers[user] = mgr
        return mgr
    app.state.manager_for_user = lambda user: managers.get(user)
    # Single-user deployments (auth off, or one shared token) have exactly
    # one manager; expose it eagerly under the historical attribute so
    # tests and tooling that reach for `app.state.manager` keep working.
    # Multi-user servers have none to single out.
    if auth is None or not auth.multi_user:
        managers[DEFAULT_USER] = SessionManager(
            user_root(session_root, auth, DEFAULT_USER), fenced=not local_files)
        app.state.manager = managers[DEFAULT_USER]
    else:
        app.state.manager = None

    from .ops import Workload
    ops = Workload(session_root)
    app.state.ops = ops
    # Every LLM call in this process lands in one ledger: one server, one
    # workspace. Sessions are tagged on their turn threads (runner, live).
    from scilink import tracing
    from scilink.usage import ledger_for
    usage = ledger_for(session_root)
    app.state.usage = usage
    tracing.set_usage_sink(usage.record)
    tracing.set_session_resolver(lambda: ops.sole_active_session(managers))

    def _ops_allowed(request: Request, *, mutating: bool) -> None:
        """Status and drain are for the control plane: the ops token
        (``SCILINK_OPS_TOKEN``, sent as ``X-Ops-Token``) always works; a
        signed-in user works on a single-user server, and for status on a
        shared one. Drain on a shared server is the operator's alone, like
        /quit."""
        expected = os.environ.get("SCILINK_OPS_TOKEN")
        given = request.headers.get("x-ops-token", "")
        if expected and hmac.compare_digest(given, expected):
            return
        if auth is None:
            return
        user = auth.user_for_request(request)
        if user is None:
            raise HTTPException(401, "Not authenticated.")
        if mutating and auth.multi_user:
            raise HTTPException(403, "Draining a shared server needs the ops token.")

    @app.on_event("shutdown")
    def _mark_interrupted_turns() -> None:
        # A redeploy or a stop mid-turn: say so in each session on resume.
        for mgr in list(managers.values()):
            try:
                mgr.mark_interrupted()
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass

    def _session_or_404(request: Request, session_id: str) -> WebSession:
        session = _mgr(request).get(session_id)
        if session is None:
            raise HTTPException(404, f"No live session {session_id!r} — "
                                     "create or resume it first.")
        return session

    def _require_local_files() -> None:
        if not local_files:
            raise HTTPException(
                403, "Folders on the server's machine are not available on a "
                     "remote deployment — upload the folder instead.")

    # ── auth ─────────────────────────────────────────────────────

    @app.get("/api/v1/auth/me")
    def auth_me(request: Request):
        """Who am I: whether sign-in is required and, if signed in, who."""
        if auth is None:
            return {"auth_required": False, "user": DEFAULT_USER,
                    "multi_user": False, "local_files": local_files}
        return {"auth_required": True, "user": auth.user_for_request(request),
                "multi_user": auth.multi_user, "local_files": local_files}

    @app.post("/api/v1/auth/login")
    def auth_login(request: Request, body: LoginRequest):
        """Exchange an access token for the HttpOnly session cookie the
        browser needs (EventSource / <img> / downloads cannot send headers)."""
        if auth is None:
            return {"user": DEFAULT_USER}
        if auth.header:
            raise HTTPException(401, "This server is signed into through the proxy "
                                     "in front of it, not with a token.")
        cookie = auth.login(body.token)
        if cookie is None:
            raise HTTPException(401, "Invalid access token.")
        user = auth.user_for_cookie(cookie)
        resp = JSONResponse({"user": user})
        secure = (request.headers.get("x-forwarded-proto", request.url.scheme)
                  == "https")
        resp.set_cookie(COOKIE_NAME, cookie, httponly=True, samesite="lax",
                        secure=secure, path="/")
        return resp

    @app.post("/api/v1/auth/logout")
    def auth_logout(request: Request):
        if auth is not None:
            auth.logout(request.cookies.get(COOKIE_NAME))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE_NAME, path="/")
        return resp

    # ── config ───────────────────────────────────────────────────

    # ── ops: what a control plane asks ───────────────────────────

    @app.get("/api/v1/ops/health")
    def ops_health():
        """Up, version, workspace. Open: a load balancer cannot sign in."""
        return ops.health()

    @app.get("/api/v1/ops/status")
    def ops_status(request: Request):
        """Busy or idle, with the reasons, and for how long it has been
        idle as seen by whoever polls this."""
        _ops_allowed(request, mutating=False)
        return ops.status(managers)

    @app.post("/api/v1/ops/drain")
    def ops_drain(request: Request, body: DrainRequest):
        """Stop taking new turns, sessions and live runs (503 to those) so
        running work can finish; ``{"drain": false}`` reopens."""
        _ops_allowed(request, mutating=True)
        ops.draining = bool(body.drain)
        return ops.status(managers)

    @app.get("/api/v1/usage")
    def usage_summary(request: Request):
        """LLM usage in the current period: totals, per model, per session
        (best-effort), and the budget."""
        _ops_allowed(request, mutating=False)
        return usage.summary()

    @app.post("/api/v1/usage/period")
    def usage_new_period(request: Request):
        """Start a new billing period (the control plane, after it read the
        old one). The records stay on disk under a period marker."""
        _ops_allowed(request, mutating=True)
        return usage.new_period()

    @app.get("/api/v1/workspace")
    def workspace(request: Request):
        """The workspace manifest this server serves, if one is present."""
        return {"workspace": ops.workspace}

    @app.get("/api/v1/config")
    def get_config(request: Request, model: str = "", base_url: str = "",
                   embedding_model: str = "", embedding_base_url: str = ""):
        """Static UI config + credential AVAILABILITY (never values)."""
        modes = [m for m in APP_MODES if m["key"] != "simulate"]
        model_q = model or MODEL_OPTIONS[0]
        prefill = resolve_prefill(model_q, existing_base_url=base_url)
        spec = provider_for(model_q)
        # The embedding key follows the embedding model's vendor (preset or
        # custom name), like the Streamlit sidebar: availability only. With a
        # base URL every request, embeddings included, goes through the proxy
        # with the main key and the embedding key is ignored (the planning /
        # meta agents warn and drop it), so no vendor env var applies.
        # An embedding_base_url is the embeddings' own endpoint, used with the
        # embedding key (the main key when blank).
        if embedding_base_url:
            emb_value, emb_env, emb_endpoint = "", None, "embedding_base_url"
        elif base_url:
            emb_value, emb_env, emb_endpoint = "", None, "base_url"
        else:
            emb_value, emb_env = resolve_embedding_prefill(embedding_model)
            emb_endpoint = "vendor"
        return {
            "auth": {"required": auth is not None,
                     "user": _user(request) if auth is None
                     else auth.user_for_request(request),
                     "multi_user": bool(auth and auth.multi_user)},
            "local_files": local_files,
            "modes": modes,
            "models": MODEL_OPTIONS,
            "embedding_models": EMBEDDING_MODEL_OPTIONS,
            "autonomy_options": {
                k: _vocab.autonomy_options(k) for k in _vocab.mode_keys()},
            "consent_text": CONSENT_TEXT,
            # The words both chat surfaces share (labels, names, hints).
            "vocabulary": _vocab.as_json(),
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
            "embedding_credential": {"env_var": emb_env, "is_set": bool(emb_value),
                                     "proxied": emb_endpoint == "base_url",
                                     "endpoint": emb_endpoint},
        }

    # ── sessions ─────────────────────────────────────────────────

    @app.get("/api/v1/sessions")
    def list_sessions(request: Request, mode: str = "meta"):
        mgr = _mgr(request)
        return {"live": mgr.list_live(),
                "resumable": mgr.discover_resumable(mode)}

    @app.post("/api/v1/sessions")
    def create_session(request: Request, body: CreateSessionRequest):
        if not body.consent:
            raise HTTPException(400, "Consent to code execution is required "
                                     "to start a session.")
        ops.refuse_if_draining()
        ops.refuse_if_over_budget(usage)
        ops.touch()
        mgr = _mgr(request)
        try:
            if body.resume_dir:
                session = mgr.resume(
                    resume_dir=body.resume_dir, mode=body.mode,
                    model=body.model, autonomy=body.autonomy,
                    api_key=body.api_key, base_url=body.base_url,
                    provider_fields=body.provider_fields,
                    fh_api_key=body.fh_api_key, mp_api_key=body.mp_api_key,
                    embedding_model=body.embedding_model,
                    embedding_api_key=body.embedding_api_key,
                    embedding_base_url=body.embedding_base_url)
            else:
                session = mgr.create(
                    mode=body.mode, model=body.model, autonomy=body.autonomy,
                    api_key=body.api_key, base_url=body.base_url,
                    provider_fields=body.provider_fields,
                    fh_api_key=body.fh_api_key, mp_api_key=body.mp_api_key,
                    objective=body.objective,
                    embedding_model=body.embedding_model,
                    embedding_api_key=body.embedding_api_key,
                    embedding_base_url=body.embedding_base_url)
        except SessionError as exc:
            raise HTTPException(400, str(exc))
        return mgr.snapshot(session)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(request: Request, session_id: str):
        return _mgr(request).snapshot(_session_or_404(request, session_id))

    @app.delete("/api/v1/sessions/{session_id}")
    def reset_session(request: Request, session_id: str):
        """Reset: stop and drop the live session (its dir stays resumable)."""
        if not _mgr(request).remove(session_id):
            raise HTTPException(404, f"No live session {session_id!r}.")
        return {"ok": True}

    @app.post("/api/v1/quit")
    def quit_server(request: Request):
        """Port of the Streamlit Quit App button (sidebar.py:550): reply
        first, then terminate the server process. Refused on a shared
        multi-user server — one user must not shut everyone down."""
        if auth is not None and auth.multi_user:
            raise HTTPException(403, "Quit is disabled on a shared server.")
        import os
        import signal
        import threading

        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        return {"ok": True}

    @app.patch("/api/v1/sessions/{session_id}")
    def rename_session(request: Request, session_id: str, body: RenameSessionRequest):
        session = _session_or_404(request, session_id)
        if not save_session_name(session.session_dir, body.name,
                                 named_by="user"):
            raise HTTPException(400, "Could not save the session name.")
        session.events.emit("session_named", {"name": body.name.strip()[:80]})
        return {"ok": True}

    # ── turns ────────────────────────────────────────────────────

    @app.post("/api/v1/sessions/{session_id}/messages", status_code=202)
    def send_message(request: Request, session_id: str, body: SendMessageRequest):
        session = _session_or_404(request, session_id)
        content = body.content.strip()
        if not content:
            raise HTTPException(400, "Empty message.")
        ops.refuse_if_draining()
        ops.refuse_if_over_budget(usage)
        ops.touch()
        with session.lock:
            if session.turn is not None and session.turn.is_running:
                raise HTTPException(409, "A turn is already running.")
            runner.start_turn(session, content)
        return {"status": "running"}

    @app.post("/api/v1/sessions/{session_id}/stop")
    def stop(request: Request, session_id: str):
        session = _session_or_404(request, session_id)
        stopped = runner.request_stop(session)
        return {"stopped": stopped}

    @app.post("/api/v1/sessions/{session_id}/feedback")
    def feedback(request: Request, session_id: str, body: FeedbackResponseRequest):
        session = _session_or_404(request, session_id)
        turn = session.turn
        pending = turn.pending_question if turn is not None else None
        if pending is None or pending.hreq.id != body.request_id:
            raise HTTPException(404, "No such pending question.")
        pending.response = body.response
        pending.event.set()
        return {"ok": True}

    # ── SSE ──────────────────────────────────────────────────────

    @app.get("/api/v1/sessions/{session_id}/events")
    def events(request: Request, session_id: str, after: Optional[int] = None,
               last_event_id: Optional[str] = Header(default=None)):
        """SSE stream. ``Last-Event-ID`` (reconnects) wins over ``after``
        (initial attach from a snapshot's ``event_cursor``)."""
        session = _session_or_404(request, session_id)
        try:
            cursor = int(last_event_id) if last_event_id else after
        except ValueError:
            cursor = after
        return StreamingResponse(
            session.events.sse_stream(cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"})

    # ── pasted local folders (hero forms) ────────────────────────

    @app.post("/api/v1/sessions/{session_id}/folders")
    def check_folders(request: Request, session_id: str, body: FolderCheckRequest):
        """Validate pasted local folder paths and enumerate their tabular
        contents — the web twin of the Streamlit hero folder inputs
        (chat_uploads.py:176-266), which read the local filesystem directly.
        Local-tool posture: these are arbitrary machine paths, exactly as in
        Streamlit — refused (403) when the server is not on the browser's
        machine (``local_files=False``)."""
        _require_local_files()
        _session_or_404(request, session_id)
        _DATA_EXTS = {".csv", ".xlsx", ".tsv", ".txt"}

        def _nat(f: Path):
            import re
            return [int(c) if c.isdigit() else c.lower()
                    for c in re.split(r"(\d+)", f.name)]

        results = []
        for raw in body.paths[:20]:
            p = Path(raw.strip()).expanduser()
            entry: dict = {"path": str(p), "is_dir": p.is_dir(),
                           "data_files": [], "json_files": [],
                           "subdirs": []}
            if entry["is_dir"]:
                try:
                    files = [f for f in p.iterdir() if f.is_file()]
                    # Subfolders (name + file count) so the prompt can say
                    # the folder is nested — the agents' directory listings
                    # are one level deep, so the model must be told.
                    subdirs = sorted(
                        (d for d in p.iterdir()
                         if d.is_dir() and not d.name.startswith(".")),
                        key=_nat)[:50]
                    entry["subdirs"] = [{
                        "path": str(d),
                        "n_files": sum(1 for f in d.iterdir() if f.is_file()
                                       and not f.name.startswith(".")),
                    } for d in subdirs]
                    entry["data_files"] = [
                        str(f) for f in sorted(
                            (f for f in files if f.suffix.lower() in _DATA_EXTS),
                            key=_nat)][:200]
                    entry["json_files"] = [
                        str(f) for f in sorted(
                            (f for f in files if f.suffix.lower() == ".json"),
                            key=lambda f: f.name)][:200]
                except OSError:
                    pass
            results.append(entry)
        return {"results": results}

    @app.post("/api/v1/sessions/{session_id}/plan_dirs")
    def set_plan_dirs(request: Request, session_id: str, body: PlanDirsRequest):
        """Repoint the planning agent's resource dirs at pasted folders (port
        of chat_uploads.py:269-279): stable source paths let the KB reuse its
        FAISS indexes across sessions instead of rebuilding."""
        _require_local_files()
        session = _session_or_404(request, session_id)
        agent = session.agent
        applied = {}
        for attr, raw in (("knowledge_dir", body.knowledge),
                          ("code_dir", body.code), ("data_dir", body.data)):
            if not raw:
                continue
            if not hasattr(agent, attr):
                raise HTTPException(
                    400, f"This session's agent has no {attr} — folder "
                         "sources apply to plan mode.")
            p = Path(raw.strip()).expanduser()
            if not p.is_dir():
                raise HTTPException(400, f"Not a folder: {raw}")
            setattr(agent, attr, p)
            applied[attr] = str(p)
        return {"applied": applied}

    # ── files ────────────────────────────────────────────────────

    @app.post("/api/v1/sessions/{session_id}/uploads")
    def upload(request: Request, session_id: str, category: str = Form(...),
               files: list[UploadFile] = File(...),
               paths: str = Form("")):
        """Multipart upload. ``paths`` (optional) is a JSON list of relative
        paths aligned with ``files`` — a FOLDER upload, saved with its layout
        preserved. It travels as its own field rather than in the part
        filenames because browsers are free to strip directory components
        from a Content-Disposition filename."""
        session = _session_or_404(request, session_id)
        try:
            if paths:
                import json as _json
                try:
                    rels = _json.loads(paths)
                except ValueError:
                    raise files_mod.UploadError("`paths` must be a JSON list.")
                if not isinstance(rels, list) or len(rels) != len(files):
                    raise files_mod.UploadError(
                        "`paths` must list one relative path per file.")
                payload = [(str(r), f.file.read()) for r, f in zip(rels, files)]
                return files_mod.save_uploads(session.session_dir, category,
                                              payload, preserve_paths=True)
            payload = [(f.filename or "", f.file.read()) for f in files]
            return files_mod.save_uploads(session.session_dir, category,
                                          payload)
        except files_mod.UploadError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/v1/sessions/{session_id}/tree")
    def get_tree(request: Request, session_id: str):
        session = _session_or_404(request, session_id)
        from .tree import build_tree
        return build_tree(session.session_dir,
                          new_since=session.turn_started_at)

    # ── skills ───────────────────────────────────────────────────

    @app.get("/api/v1/sessions/{session_id}/skills")
    def get_skills(request: Request, session_id: str):
        """The skill catalog: built-in bundles by domain (with their
        one-line descriptions) and the custom skills registered on this
        session's agent."""
        from .skills_api import skill_catalog
        return skill_catalog(_session_or_404(request, session_id).agent)

    @app.get("/api/v1/sessions/{session_id}/skills/{domain}/{name}")
    def get_skill_markdown(request: Request, session_id: str, domain: str, name: str):
        """A skill's markdown — `domain` is a catalog domain, or `custom`."""

        from .skills_api import SkillError, skill_markdown
        session = _session_or_404(request, session_id)
        try:
            text = skill_markdown(session.agent, domain, name)
        except SkillError as exc:
            raise HTTPException(exc.status, str(exc))
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    @app.post("/api/v1/sessions/{session_id}/skills")
    def upload_skills(request: Request, session_id: str,
                      files: list[UploadFile] = File(...)):
        """Upload `.md` skill files: saved under the session's custom_skills/
        and registered with the agent for the rest of the session (the
        agents' selectors treat them like built-ins)."""
        from .skills_api import SkillError, register_uploaded_skills
        session = _session_or_404(request, session_id)
        payload = [(f.filename or "", f.file.read()) for f in files]
        try:
            return register_uploaded_skills(session.agent, session.session_dir, payload)
        except SkillError as exc:
            raise HTTPException(exc.status, str(exc))

    # ── persistent memory ────────────────────────────────────────
    # One store per server host ($SCILINK_HOME or ~/.scilink), shared by
    # every session and — on a multi-user server — every user: it is the
    # host's curated knowledge, the same thing `scilink memory` manages.

    from .memory_api import MemoryError

    def _mem(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except MemoryError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.get("/api/v1/memory")
    def memory_overview():
        """The switch, the pipeline strip, the bank, the inbox, the skills."""
        from .memory_api import memory_overview as _overview
        return _overview()

    @app.post("/api/v1/memory/enabled")
    def memory_set_enabled(body: MemoryEnabledRequest):
        from .memory_api import set_enabled
        return set_enabled(body.enabled)

    def _skill_ref(domain: str, name: str) -> None:
        """A skill is addressed by two names; neither may be a path."""
        for part in (domain, name):
            if not part or part in (".", "..") or set(part.lower()) - _USERNAME_OK:
                raise HTTPException(400, f"Bad skill reference {part!r}: letters, "
                                         "digits, '_', '-', '.' only")

    @app.get("/api/v1/memory/skills/{domain}/{name}")
    def memory_skill_text(domain: str, name: str):
        _skill_ref(domain, name)
        from .memory_api import skill_text
        return PlainTextResponse(_mem(skill_text, domain, name), media_type="text/markdown; charset=utf-8")

    @app.put("/api/v1/memory/skills/{domain}/{name}")
    def memory_skill_edit(domain: str, name: str, body: MemoryEditRequest):
        _skill_ref(domain, name)
        from .memory_api import skill_edit
        return _mem(skill_edit, domain, name, body.content)

    @app.post("/api/v1/memory/skills/{domain}/{name}/{action}")
    def memory_skill_action(domain: str, name: str, action: str):
        """promote · demote · prune · diff (a fork against its built-in) ·
        fork (copy a built-in into the store, shadowing it)."""
        _skill_ref(domain, name)
        from .memory_api import fork_builtin, skill_action
        if action == "fork":
            return _mem(fork_builtin, domain, name)
        return _mem(skill_action, domain, name, action)

    @app.get("/api/v1/memory/bank/{domain}/{rid}")
    def memory_bank_record(domain: str, rid: str):
        from .memory_api import bank_record
        return _mem(bank_record, domain, rid)

    @app.delete("/api/v1/memory/bank/{domain}/{rid}")
    def memory_bank_delete(domain: str, rid: str):
        from .memory_api import bank_delete
        return _mem(bank_delete, domain, rid)

    @app.post("/api/v1/memory/bank/{domain}/{rid}/nominate")
    def memory_bank_nominate(domain: str, rid: str):
        from .memory_api import bank_nominate
        return _mem(bank_nominate, domain, rid)

    @app.post("/api/v1/memory/bank/{domain}/nominate-group")
    def memory_bank_nominate_group(domain: str, body: MemoryIdsRequest):
        from .memory_api import bank_nominate_group
        return _mem(bank_nominate_group, domain, body.ids, body.technique)

    @app.get("/api/v1/memory/inbox/{domain}/{sid}")
    def memory_inbox_record(domain: str, sid: str):
        from .memory_api import inbox_record
        return _mem(inbox_record, domain, sid)

    @app.delete("/api/v1/memory/inbox/{domain}/{sid}")
    def memory_inbox_discard(domain: str, sid: str):
        from .memory_api import inbox_discard
        return _mem(inbox_discard, domain, sid)

    @app.post("/api/v1/memory/inbox/{domain}/targets")
    def memory_upgrade_targets(domain: str, body: MemoryIdsRequest):
        from .memory_api import upgrade_targets
        return {"targets": _mem(upgrade_targets, domain, body.ids)}

    @app.post("/api/v1/memory/inbox/{domain}/consolidate")
    def memory_consolidate(request: Request, domain: str, body: MemoryConsolidateRequest):
        """Start the distillation job (1-3 min); poll /memory/jobs/{id}."""
        from .memory_api import llm_call_for, start_consolidate
        session = _session_or_404(request, body.session_id)
        return _mem(start_consolidate, domain, body.ids, body.label,
                    _mem(llm_call_for, session.agent))

    @app.post("/api/v1/memory/inbox/{domain}/propose-upgrade")
    def memory_propose_upgrade(request: Request, domain: str, body: MemoryUpgradeRequest):
        """Start the upgrade-preview job (writes nothing); poll /memory/jobs/{id}."""
        from .memory_api import llm_call_for, start_propose_upgrade
        session = _session_or_404(request, body.session_id)
        return _mem(start_propose_upgrade, domain, body.ids, body.target_domain,
                    body.target_name, _mem(llm_call_for, session.agent))

    @app.post("/api/v1/memory/inbox/{domain}/apply-upgrade")
    def memory_apply_upgrade(domain: str, body: MemoryApplyRequest):
        from .memory_api import apply_upgrade
        return _mem(apply_upgrade, domain, body.ids, body.target_domain,
                    body.target_name, body.content, body.fork_builtin)

    @app.post("/api/v1/memory/check-upgrade")
    def memory_check_upgrade(body: MemoryCheckRequest):
        """Additivity warnings + diff for an edited proposal."""
        from .memory_api import upgrade_check
        return upgrade_check(body.existing, body.proposed)

    @app.get("/api/v1/memory/jobs/{job_id}")
    def memory_job(job_id: str):
        from .memory_api import job_status
        return _mem(job_status, job_id)

    # ── live measurement loop ────────────────────────────────────
    # A live run is a session-level background job that outlives chat turns;
    # the Live tab only observes it (everything shown is read back from the
    # loop's own loop_log.jsonl). See scilink/server/live_api.py.

    from .live_api import LiveError

    def _live(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except LiveError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.get("/api/v1/live/simulators")
    def live_simulators():
        from .live_api import list_simulators
        return {"simulators": list_simulators()}

    # What the instruments on this machine remember (scilink.live.instrument_home).
    # The store belongs to the machine, not to a session, so it has no session
    # in its path. On a shared multi-user server it is read-only: one user must
    # not delete what another user's runs rely on.
    @app.get("/api/v1/live/instruments")
    def live_instruments():
        from .live_api import list_instruments
        return {"instruments": list_instruments(),
                "can_forget": not (auth is not None and auth.multi_user)}

    @app.get("/api/v1/live/instruments/{instrument}")
    def live_instrument(instrument: str):
        from .live_api import instrument_memory
        return _live(instrument_memory, instrument)

    @app.delete("/api/v1/live/instruments/{instrument}/recipes/{recipe_id}")
    def live_forget_recipe(instrument: str, recipe_id: str):
        if auth is not None and auth.multi_user:
            raise HTTPException(403, "Forgetting a recipe is disabled on a shared server.")
        from .live_api import forget_recipe
        return _live(forget_recipe, instrument, recipe_id)

    @app.get("/api/v1/sessions/{session_id}/live")
    def live_snapshot(request: Request, session_id: str):
        from .live_api import snapshot
        return _live(snapshot, _session_or_404(request, session_id))

    @app.post("/api/v1/sessions/{session_id}/live/start")
    async def live_start(request: Request, session_id: str):
        from .live_api import start
        ops.refuse_if_draining()
        ops.refuse_if_over_budget(usage)
        ops.touch()
        body = await request.json()
        return _live(start, _session_or_404(request, session_id), body or {},
                     allow_custom=local_files)

    @app.post("/api/v1/sessions/{session_id}/live/stop")
    def live_stop(request: Request, session_id: str):
        from .live_api import stop
        return _live(stop, _session_or_404(request, session_id))

    @app.post("/api/v1/sessions/{session_id}/live/params")
    async def live_params(request: Request, session_id: str):
        """The operator's hand on the controls: an accepted recommendation or a
        manual change, used from the next frame on."""
        from .live_api import set_params
        body = await request.json()
        return _live(set_params, _session_or_404(request, session_id),
                     (body or {}).get("params") or {})

    @app.post("/api/v1/sessions/{session_id}/live/resume")
    async def live_resume(request: Request, session_id: str):
        """The answer to a pause: {"action": "resume" | "stop", "params": {...}}."""
        from .live_api import decide
        body = await request.json()
        return _live(decide, _session_or_404(request, session_id), body or {})

    @app.post("/api/v1/sessions/{session_id}/live/clear")
    def live_clear(request: Request, session_id: str):
        from .live_api import clear
        return _live(clear, _session_or_404(request, session_id))

    # ── tools / MCP ──────────────────────────────────────────────

    @app.get("/api/v1/sessions/{session_id}/tools")
    def get_tools(request: Request, session_id: str):
        """The connected MCP servers and the external tools they (or
        anything else) registered with the session's agent."""
        from .tools_api import tool_inventory
        return tool_inventory(_session_or_404(request, session_id).agent)

    @app.post("/api/v1/sessions/{session_id}/mcp")
    def connect_mcp_server(request: Request, session_id: str,
                           body: MCPConnectRequest):
        """Connect an MCP server (stdio command, SSE or streamable-HTTP
        URL) and register its tools with the agent. A stdio command runs
        on the server's machine — the same trust as the agents' own code
        execution, which every session already consents to."""
        from .tools_api import MCPError, connect_mcp, tool_inventory
        session = _session_or_404(request, session_id)
        try:
            n = connect_mcp(session.agent, name=body.name,
                            transport=body.transport, command=body.command,
                            url=body.url, headers=body.headers,
                            expand_env=local_files)
        except MCPError as exc:
            raise HTTPException(exc.status, str(exc))
        return {"registered": n, "inventory": tool_inventory(session.agent)}

    @app.delete("/api/v1/sessions/{session_id}/mcp/{server_name}")
    def disconnect_mcp_server(request: Request, session_id: str,
                              server_name: str):
        from .tools_api import MCPError, disconnect_mcp, tool_inventory
        session = _session_or_404(request, session_id)
        try:
            disconnect_mcp(session.agent, server_name)
        except MCPError as exc:
            raise HTTPException(exc.status, str(exc))
        return {"ok": True, "inventory": tool_inventory(session.agent)}

    @app.get("/api/v1/sessions/{session_id}/delegations")
    def get_delegations(request: Request, session_id: str):
        """The meta session's delegation ledger for the Delegations tab
        (same payload as the `delegations` SSE event). Empty for other
        modes rather than an error, so the client can call it blindly."""
        session = _session_or_404(request, session_id)
        from .delegations import delegation_view
        if session.mode != "meta":
            return {"delegations": [], "sub_agents": {}}
        return delegation_view(session.agent, session.session_dir)

    @app.get("/api/v1/sessions/{session_id}/telemetry")
    def get_telemetry(request: Request, session_id: str):
        """Full read-only telemetry snapshot (ledger, worker action
        histories, analysis reasoning, per-agent tool sequence) — the
        reader behind the Streamlit Telemetry tab, exposed for the web
        UI's future Telemetry view."""
        session = _session_or_404(request, session_id)
        from scilink.agents.meta_agent.telemetry import collect_session_telemetry
        return collect_session_telemetry(session.agent)

    @app.get("/api/v1/sessions/{session_id}/provenance")
    def get_provenance(request: Request, session_id: str):
        session = _session_or_404(request, session_id)
        from .tree import load_provenance
        return {"events": load_provenance(session.session_dir)}

    @app.get("/api/v1/sessions/{session_id}/thumb")
    def get_thumb(request: Request, session_id: str, path: str, size: int = 256,
                  cmap: str = "viridis"):
        session = _session_or_404(request, session_id)
        from fastapi.responses import Response

        from . import previews
        try:
            target = files_mod.resolve_safe(session.session_dir, path)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        if not target.is_file():
            raise HTTPException(404, f"No such file: {path}")
        try:
            png, kind = previews.render_thumbnail(target, size=size, cmap=cmap)
        except Exception as exc:
            raise HTTPException(422, f"Could not render {path}: {exc}")
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "no-cache",
                                 "X-Preview-Kind": kind})

    @app.get("/api/v1/sessions/{session_id}/table")
    def get_table(request: Request, session_id: str, path: str, limit: int = 500):
        session = _session_or_404(request, session_id)
        from . import previews
        try:
            target = files_mod.resolve_safe(session.session_dir, path)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        if not target.is_file():
            raise HTTPException(404, f"No such file: {path}")
        try:
            return previews.extract_table(target, limit=limit)
        except Exception as exc:
            raise HTTPException(422, f"Could not read table {path}: {exc}")

    @app.get("/api/v1/sessions/{session_id}/zip")
    def get_zip(request: Request, session_id: str, path: str = ""):
        session = _session_or_404(request, session_id)
        from fastapi.responses import Response

        from .tree import zip_directory
        try:
            blob = zip_directory(session.session_dir, path)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        except FileNotFoundError:
            raise HTTPException(404, f"No such directory: {path}")
        except ValueError as exc:
            raise HTTPException(413, str(exc))
        name = (Path(path).name or session.id) + ".zip"
        return Response(
            blob, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/v1/sessions/{session_id}/files")
    def get_file(request: Request, session_id: str, path: str):
        session = _session_or_404(request, session_id)
        try:
            target = files_mod.resolve_safe(session.session_dir, path)
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        if not target.is_file():
            raise HTTPException(404, f"No such file: {path}")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        # no-cache = revalidate, not "never store": a figure rewritten in place
        # (a refined visualization.png at the same path) must not be served
        # stale from the browser cache. FileResponse still emits ETag/
        # Last-Modified, so an unchanged file is a cheap 304.
        return FileResponse(target, media_type=media_type,
                            headers={"Cache-Control": "no-cache"})

    # ── static frontend (production) ─────────────────────────────

    if serve_frontend:
        # Repo checkout: webui/dist (freshest, from `npm run build`).
        # Installed package: the bundle shipped inside the wheel — refreshed
        # at release time via `npm run build:package` in webui/.
        candidates = [
            Path(__file__).resolve().parents[2] / "webui" / "dist",
            Path(__file__).resolve().parent / "static",
        ]
        dist = next((d for d in candidates if (d / "index.html").is_file()),
                    None)
        app.state.frontend_dir = str(dist) if dist is not None else None
        if dist is None:
            # The bundle is built at release time (release wheels carry it)
            # and is not in git — a checkout must build it once. Say so at
            # "/" instead of serving a bare 404 next to a working API.

            @app.get("/", include_in_schema=False)
            def _no_frontend():
                return PlainTextResponse(NO_BUNDLE_MESSAGE, status_code=503)
        if dist is not None:
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
