"""Skills tab: browse the skill catalog, view a skill's markdown, upload
custom skills for the session.

Thin over what exists: ``scilink.skills.loader`` lists and loads every
built-in / user-root skill bundle (48 files load in ~30 ms, so the catalog
is computed per request, no cache to invalidate), and every orchestrator
has ``register_skill(path)`` which makes an uploaded ``.md`` selectable by
the agents for the rest of the session (``_custom_skills``). Uploads land
in ``<session>/custom_skills/``, the Streamlit tab's convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_DESC_MAX = 400


def _clip(text: Any) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= _DESC_MAX else s[:_DESC_MAX - 1] + "…"


def skill_catalog(agent: Any) -> Dict[str, Any]:
    """``{"builtin": [{domain, label, skills: [{name, description}]}],
    "custom": [{name, path}], "skills_supported": bool}``."""
    from scilink.skills.loader import list_all_skills, load_skill, _resolve_skill_path, _SKILLS_DIR, graduated_skills_dir

    def _origin(domain: str, name: str) -> str:
        """``builtin`` (shipped), ``fork`` (a store copy shadowing a shipped
        skill), ``learned`` (graduated or distilled into the store)."""
        try:
            path = _resolve_skill_path(name, domain).resolve()
        except Exception:  # noqa: BLE001
            return "builtin"
        try:
            in_store = str(path).startswith(str(graduated_skills_dir().resolve()))
        except Exception:  # noqa: BLE001
            in_store = False
        if not in_store:
            return "builtin"
        return "fork" if (_SKILLS_DIR / domain / name / f"{name}.md").is_file() else "learned"

    builtin: List[Dict[str, Any]] = []
    try:
        domains = list_all_skills()
    except Exception:  # noqa: BLE001 - a broken user skill root must not hide the tab
        domains = {}
    for domain, names in domains.items():
        entries = []
        for name in names:
            desc, meta = "", {}
            try:
                meta = load_skill(name, domain).get("meta") or {}
                desc = _clip(meta.get("description"))
            except Exception:  # noqa: BLE001 - one bad skill file degrades to no blurb
                pass
            entries.append({"name": name, "description": desc,
                            "origin": _origin(domain, name),
                            "provisional": meta.get("provisional") is True})
        builtin.append({"domain": domain,
                        "label": domain.replace("_", " ").title(),
                        "skills": entries})
    custom = [{"name": n, "path": str(p)}
              for n, p in sorted((getattr(agent, "_custom_skills", None) or {}).items())]
    return {"builtin": builtin, "custom": custom,
            "skills_supported": callable(getattr(agent, "register_skill", None))}


class SkillError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def skill_markdown(agent: Any, domain: str, name: str) -> str:
    """The markdown of a built-in skill (``domain/name``) or, with domain
    ``custom``, of a skill registered on this session's agent."""
    if domain == "custom":
        path = (getattr(agent, "_custom_skills", None) or {}).get(name)
        if not path or not Path(path).is_file():
            raise SkillError(404, f"No custom skill named {name!r} in this session.")
        return Path(path).read_text(encoding="utf-8", errors="replace")
    from scilink.skills.loader import _resolve_skill_path
    try:
        return _resolve_skill_path(name, domain).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise SkillError(404, f"No skill {domain}/{name}: {exc}")


BUILDER_SECTIONS = ("overview", "planning", "implementation", "interpretation", "validation")
_NAME_RE = __import__("re").compile(r"^[a-z][a-z0-9_]{1,63}$")


def compose_skill(agent: Any, session_dir: str, *, name: str, domain: str, description: str,
                  technique: Sequence[str], sections: Dict[str, str], save: str) -> Dict[str, Any]:
    """The skill builder's one call: render a skill from its parts and, when
    asked, keep it.

    ``save="preview"`` renders only; ``"session"`` writes it under the
    session's ``custom_skills/`` and registers it with the agent (like an
    upload); ``"memory"`` writes an approved bundle into persistent memory
    (a human-authored skill needs no review) with ``provenance: authored``,
    refusing to overwrite and refusing while memory is off, because the
    store is inert then and the skill would not load.
    """
    from scilink.skills._shared._graduation import format_skill_as_markdown, merge_techniques

    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise SkillError(400, "The name is a slug: lowercase letters, digits and '_', starting with a letter (2–64 characters).")
    domain = (domain or "").strip()
    if not _NAME_RE.match(domain):
        raise SkillError(400, "The domain is a slug like curve_fitting or image_analysis.")
    description = " ".join(str(description or "").split())
    if not description:
        raise SkillError(400, "Give the skill a one-sentence description — the agents route on it.")
    body = {k: str((sections or {}).get(k) or "").strip() for k in BUILDER_SECTIONS}
    if not any(body.values()):
        raise SkillError(400, "Write at least one section.")
    data: Dict[str, Any] = {"description": description}
    techs = merge_techniques(list(technique or []))
    if techs:
        data["technique"] = techs
    data.update({k: v for k, v in body.items() if v})
    if save == "memory":
        data["provenance"] = "authored"
    markdown = format_skill_as_markdown(data)
    out: Dict[str, Any] = {"name": name, "domain": domain, "markdown": markdown, "saved": save}

    if save == "preview":
        return out
    if save == "session":
        if not callable(getattr(agent, "register_skill", None)):
            raise SkillError(400, "This session's agent does not take custom skills.")
        dest_dir = Path(session_dir) / "custom_skills"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.md"
        dest.write_text(markdown, encoding="utf-8")
        try:
            registered = str(agent.register_skill(str(dest)))
        except Exception as exc:  # noqa: BLE001
            raise SkillError(400, f"Saved {dest.name} but could not register it: {exc}")
        out.update({"path": str(dest), "registered": registered, "catalog": skill_catalog(agent)})
        return out
    if save == "memory":
        from scilink.skills.loader import graduated_skills_dir, memory_enabled
        if not memory_enabled():
            raise SkillError(400, "Persistent memory is off — turn it on (Memory tab) to save a skill into the store; "
                                  "or use it in this session only.")
        bundle = graduated_skills_dir() / domain / name
        md = bundle / f"{name}.md"
        if md.exists():
            raise SkillError(409, f"{domain}/{name} already exists in persistent memory — pick another name, "
                                  "or edit that skill on the Memory tab.")
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "__init__.py").touch()
        md.write_text(markdown, encoding="utf-8")
        out.update({"path": str(md), "catalog": skill_catalog(agent)})
        return out
    raise SkillError(400, "save must be one of preview, session, memory.")


def knowledge_bases() -> List[Dict[str, Any]]:
    """The named knowledge bases in the persistent store, for the builder's
    grounding menu."""
    try:
        from scilink.knowledge.kb_store import list_kbs
        return [{"name": k.get("name"), "embedding_model": k.get("embedding_model"),
                 "sources": [s if isinstance(s, str) else s.get("path") for s in (k.get("sources") or [])][:6]}
                for k in list_kbs()]
    except Exception:  # noqa: BLE001 - a broken store must not hide the builder
        return []


def draft_options(agent: Any) -> Dict[str, Any]:
    return {"knowledge_bases": knowledge_bases(),
            "literature_available": bool(getattr(agent, "futurehouse_api_key", None)
                                         or __import__("os").environ.get("FUTUREHOUSE_API_KEY"))}


def _kb_grounding(agent: Any, kb_ref: str, query: str) -> Tuple[str, Dict[str, Any]]:
    """Top chunks of a named knowledge base (or a KB directory) for the
    draft's query. Dense retrieval when the session has the KB's embedding
    provider, keyword retrieval otherwise; never raises."""
    from scilink.knowledge.kb_store import KB_BASE_NAME, resolve_knowledge_source
    from scilink.knowledge.knowledge_base import KnowledgeBase
    from scilink.knowledge.rag_engine import retrieve_context
    path, manifest = resolve_knowledge_source(kb_ref, strict=True)
    base_url = getattr(agent, "base_url", None)
    kb = KnowledgeBase(embedding_model=getattr(agent, "embedding_model", None)
                       or (manifest or {}).get("embedding_model"),
                       api_key=getattr(agent, "embedding_api_key", None),
                       base_url=base_url, use_litellm=not base_url)
    prefix = path / f"{KB_BASE_NAME}_docs"
    if not kb.load(str(prefix.with_suffix(".faiss")), str(prefix.with_suffix(".json")),
                   sources_path=str(prefix.with_suffix(".sources.json"))):
        return "", {"kb": str(kb_ref), "kb_chunks": 0, "warning": f"{kb_ref} has no built index."}
    text = retrieve_context(kb, query, top_k=8)
    n = text.count("Source: ") if text else 0
    return text, {"kb": str(kb_ref), "kb_chunks": n}


def _literature_grounding(agent: Any, query: str, max_wait: int = 900) -> Tuple[str, Dict[str, Any]]:
    """One FutureHouse literature answer for the draft's query; the call
    takes minutes, which is why the draft runs as a background job."""
    import os
    key = getattr(agent, "futurehouse_api_key", None) or os.environ.get("FUTUREHOUSE_API_KEY")
    if not key:
        return "", {"literature": "unavailable", "warning": "No FutureHouse key on this session."}
    from scilink.agents.lit_agents.literature_agent import LiteratureSearchAgent
    lit = LiteratureSearchAgent(api_key=key, max_wait_time=max_wait)
    res = lit._execute_crow_task(
        f"For a practitioner's method note on: {query}. Summarise the established analysis "
        "procedure, the model forms and parameter ranges commonly used, known pitfalls and how "
        "results are validated. Cite sources.", task_type="skill")
    if res.get("status") != "success":
        return "", {"literature": res.get("status", "error"), "warning": res.get("message")}
    return str(res.get("content") or ""), {"literature": "success", "sources": list(res.get("sources") or [])}


def _draft_plan(name, description, technique, sections, notes, fill):
    """What a draft would write and what it has to go on — checked before a
    job is started so a hopeless request is a 400, not a failed job."""
    current = {k: str((sections or {}).get(k) or "").strip() for k in BUILDER_SECTIONS}
    targets = [k for k in BUILDER_SECTIONS if fill == "all" or not current[k]]
    if not targets:
        raise SkillError(400, "Every section is already written — choose 'redraft all' to rewrite them.")
    query = " ".join(x for x in [description, ", ".join(technique or []), notes, (name or "").replace("_", " ")] if x).strip()
    if not query:
        raise SkillError(400, "Give the model something to go on: a description, a technique or notes.")
    return current, targets, query


def draft_skill(agent: Any, *, name: str, domain: str, description: str, technique: Sequence[str],
                sections: Dict[str, str], notes: str, kb: Optional[str], literature: bool,
                fill: str) -> Dict[str, Any]:
    """Draft the builder's sections with the session's model, grounded on a
    knowledge base and/or a literature search when asked. ``fill="empty"``
    writes only the sections the author left empty; ``"all"`` redrafts every
    section (the author's text still travels as authoritative context).
    Returns the proposed parts; nothing is saved."""
    import json as _json
    from scilink.agents.exp_agents.instruct import SKILL_DRAFT_INSTRUCTIONS
    from scilink.skills._shared._graduation import merge_techniques, parse_json_response
    from .memory_api import llm_call_for

    current, targets, query = _draft_plan(name, description, technique, sections, notes, fill)

    grounding_blocks: List[str] = []
    info: Dict[str, Any] = {"kb": None, "kb_chunks": 0, "literature": "not requested", "sources": []}
    warnings: List[str] = []
    if kb:
        try:
            text, meta = _kb_grounding(agent, kb, query)
        except Exception as exc:  # noqa: BLE001 - grounding degrades, never aborts
            text, meta = "", {"kb": kb, "kb_chunks": 0, "warning": f"Knowledge base not used: {exc}"}
        info.update({k: v for k, v in meta.items() if k != "warning"})
        if meta.get("warning"):
            warnings.append(meta["warning"])
        if text:
            grounding_blocks.append(f"[KB: {kb}]\n{text}")
    if literature:
        try:
            text, meta = _literature_grounding(agent, query)
        except Exception as exc:  # noqa: BLE001
            text, meta = "", {"literature": "error", "warning": f"Literature search not used: {exc}"}
        info["literature"] = meta.get("literature")
        info["sources"] = meta.get("sources") or []
        if meta.get("warning"):
            warnings.append(str(meta["warning"]))
        if text:
            grounding_blocks.append("[Literature search]\n" + text)
    grounding = "\n\n".join(f"[{i + 1}] {b}" for i, b in enumerate(grounding_blocks)) or "(none)"

    skill_json = _json.dumps({"name": name, "domain": domain, "description": description,
                              "technique": list(technique or []), **{k: v for k, v in current.items() if v}},
                             indent=2)
    prompt = SKILL_DRAFT_INSTRUCTIONS.format(
        skill_json=skill_json, notes=(notes or "").strip() or "(none)", grounding=grounding,
        targets=", ".join(targets), target_keys=_json.dumps(targets))
    call = llm_call_for(agent)
    raw = call(prompt)
    try:
        parsed = parse_json_response(raw)
    except ValueError:
        parsed = parse_json_response(call("Respond with ONLY a single JSON object — no prose before or "
                                          "after it.\n\n" + prompt))
    out_sections = {k: str(parsed.get(k) or "").strip() for k in targets if str(parsed.get(k) or "").strip()}
    if not out_sections:
        raise SkillError(502, "The model returned no section text.")
    result: Dict[str, Any] = {"sections": out_sections, "targets": targets, "grounding": info, "warnings": warnings}
    if not (description or "").strip() and str(parsed.get("description") or "").strip():
        result["description"] = " ".join(str(parsed["description"]).split())
    if not list(technique or []):
        techs = merge_techniques(parsed.get("technique"))
        if techs:
            result["technique"] = techs
    return result


def start_draft(agent: Any, **kw) -> Dict[str, Any]:
    """The draft as a background job (the literature search alone can take
    minutes); the result is read from ``GET /memory/jobs/{id}``."""
    from .memory_api import _start_job, llm_call_for
    _draft_plan(kw.get("name", ""), kw.get("description", ""), kw.get("technique"), kw.get("sections"),
                kw.get("notes", ""), kw.get("fill", "empty"))
    llm_call_for(agent)                      # a session without a model is a 400 now, not a failed job
    label = f"draft {kw.get('domain')}/{kw.get('name') or 'skill'}"
    return _start_job("draft", lambda: draft_skill(agent, **kw), label)


def register_uploaded_skills(agent: Any, session_dir: str,
                             files: Sequence[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Save ``.md`` uploads under ``<session>/custom_skills/`` and register
    each with the agent. Returns the registered names, per-file errors, and
    the refreshed catalog. A file that fails to register is still saved
    (the user can fix and re-upload) but reported."""
    if not callable(getattr(agent, "register_skill", None)):
        raise SkillError(400, "This session's agent does not take custom skills.")
    dest_dir = Path(session_dir) / "custom_skills"
    dest_dir.mkdir(parents=True, exist_ok=True)
    registered: List[str] = []
    errors: List[Dict[str, str]] = []
    for name, blob in files:
        base = Path(name).name
        if not base or base.startswith(".") or Path(base).suffix.lower() != ".md":
            errors.append({"file": name, "error": "A skill is a single .md file."})
            continue
        dest = dest_dir / base
        dest.write_bytes(blob)
        try:
            registered.append(str(agent.register_skill(str(dest))))
        except Exception as exc:  # noqa: BLE001 - report, keep going
            errors.append({"file": base, "error": str(exc)})
    return {"registered": registered, "errors": errors,
            "catalog": skill_catalog(agent)}
