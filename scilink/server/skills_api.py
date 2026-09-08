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
    from scilink.skills.loader import list_all_skills, load_skill

    builtin: List[Dict[str, Any]] = []
    try:
        domains = list_all_skills()
    except Exception:  # noqa: BLE001 - a broken user skill root must not hide the tab
        domains = {}
    for domain, names in domains.items():
        entries = []
        for name in names:
            desc = ""
            try:
                desc = _clip((load_skill(name, domain).get("meta") or {}).get("description"))
            except Exception:  # noqa: BLE001 - one bad skill file degrades to no blurb
                pass
            entries.append({"name": name, "description": desc})
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
