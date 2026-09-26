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
from typing import Any, Dict, List, Sequence, Tuple

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
