"""Shared infrastructure for on-the-fly skill graduation / distillation.

This is the package-neutral home for the graduation helpers. It lives
under ``scilink/skills/_shared/`` (not inside ``agents/sim_agents/``) so
the analysis, planning, meta, and CLI code paths can import it WITHOUT
pulling in ``ase`` (an optional dependency that ``scilink.agents.sim_agents``
hard-imports). The module itself is pure stdlib + lazy ``yaml``/loader —
no heavy imports. ``scilink/agents/sim_agents/skill_graduation.py`` is now
a thin re-export shim for backward compatibility.

Two-tier memory:
  - active_knowledge (in-memory, session-scoped): observations the
    agent records during a run. Cheap to add, easy to discard.
  - graduated skills (.md files on disk): durable rules the agent
    has crystallized. Loaded back into LLM context next session.

**Structured I/O contract.** The LLM is asked to return a JSON object
with named fields; this module deterministically formats that JSON
into a YAML+markdown skill file (using PyYAML for the frontmatter, so
quoting is always correct). The LLM never has to produce valid YAML.
This eliminates a class of LLM-format-bug failures we saw with an
earlier free-form-markdown contract.

Default storage: ~/.scilink/graduated_skills/<domain>/<name>/<name>.md
(see ``scilink.skills.loader.graduated_skills_dir`` — the single source
of truth, honoring ``$SCILINK_HOME``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..loader import graduated_skills_dir
from ._store_io import atomic_write_text, file_lock


def safe_path_component(value: str, *, fallback: str = "unknown") -> str:
    """Sanitize a string used as a single filesystem path component.

    `domain` / `skill_name` / `technique` are joined into paths under the
    persistent store. A value like ``"../../evil"`` would escape the store, so
    strip directory separators and parent refs, keeping only a flat, safe token.
    """
    base = Path(str(value)).name              # drops any dir part, incl. ".."
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return cleaned or fallback


_EPHEMERAL_WARNED = False


def warn_if_ephemeral_store() -> None:
    """Warn ONCE if the persistent store likely won't persist.

    SciLink is commonly run in Docker; inside a container the default home
    (``~/.scilink``) lives on the ephemeral container filesystem, so graduated/
    staged skills vanish when the container exits — silently defeating the whole
    "persistent memory" promise. We can't reliably detect a bind-mount, so we use
    a conservative heuristic: in a container AND ``$SCILINK_HOME`` is unset →
    advise mounting a volume. Advisory only; never blocks or relocates.
    """
    global _EPHEMERAL_WARNED
    if _EPHEMERAL_WARNED or os.environ.get("SCILINK_HOME"):
        return
    in_container = os.path.exists("/.dockerenv")
    if not in_container:
        try:
            with open("/proc/1/cgroup") as fh:
                in_container = "docker" in fh.read() or "kubepods" in fh.read()
        except OSError:
            in_container = False
    if in_container:
        _EPHEMERAL_WARNED = True
        from ..loader import scilink_home
        logging.warning(
            "Persistent memory at %s is on the container filesystem and will NOT "
            "survive container restarts. Mount a volume (e.g. "
            "`-v ~/.scilink:/home/scilinkuser/.scilink`) or set $SCILINK_HOME to a "
            "mounted path to keep graduated/distilled skills.",
            scilink_home(),
        )

# Backward-compatible module constant. Prefer calling
# ``graduated_skills_dir()`` (honors $SCILINK_HOME dynamically); this is
# kept so existing references to the name keep resolving.
GRADUATED_SKILLS_DIR = graduated_skills_dir()
WORD_COUNT_WARN_THRESHOLD = 8000

# Section keys in the structured skill JSON. Order matches what
# scilink/skills/loader.py knows about; rendered into markdown in this
# order so the file structure is consistent.
_SECTION_KEYS: tuple[str, ...] = (
    "overview",
    "planning",
    "analysis",
    "interpretation",
    "validation",
    "implementation",
)

# Extra frontmatter keys (beyond ``description``) that may be carried
# through from the structured JSON into the skill's YAML frontmatter.
# Fixed allowlist so the frontmatter can't accumulate arbitrary fields,
# and so description-only callers produce byte-identical output.
_EXTRA_META_KEYS: tuple[str, ...] = (
    "provisional",
    "provenance",
    "session",
    "r_squared",
    "quality_score",
    "n_examples",
)


# ──────────────────────────────────────────────────────────────
# In-session knowledge store
# ──────────────────────────────────────────────────────────────

class KnowledgeStore:
    """Session-scoped store of observations awaiting graduation.

    Attached to an agent instance; lives only as long as the agent.
    Intentionally simple — list of dicts with auto-assigned ids."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def record(self, observation: Dict[str, Any]) -> str:
        """Append an observation; return its assigned id."""
        entry_id = uuid.uuid4().hex[:8]
        entry = {"id": entry_id, **observation}
        self._entries.append(entry)
        self.logger.info(
            f"Recorded knowledge entry {entry_id}: "
            f"{str(observation.get('summary', ''))[:80]}"
        )
        return entry_id

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for e in self._entries:
            if e.get("id") == entry_id:
                return e
        return None

    def list(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def remove(self, entry_id: Optional[str] = None) -> int:
        """Remove a specific entry, or all when entry_id is None.
        Returns count removed."""
        if entry_id is None:
            n = len(self._entries)
            self._entries.clear()
            return n
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("id") != entry_id]
        return before - len(self._entries)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _format_knowledge(entry: Dict[str, Any]) -> str:
    """Format a knowledge entry as readable text for the LLM prompt."""
    lines = []
    for key, value in entry.items():
        if key == "id":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"**{pretty_key}:** {value}")
    return "\n".join(lines)


def parse_json_response(raw: str) -> Dict[str, Any]:
    """Tolerant JSON parse for an LLM response.

    Handles three common shapes:
      1. Pure JSON (the documented contract).
      2. JSON wrapped in ``` or ```json fences.
      3. JSON embedded in surrounding prose (matches the largest
         brace-balanced block).
    Raises ValueError if no JSON object can be located.
    """
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(
        f"No JSON object found in LLM response (first 200 chars): {raw[:200]!r}"
    )


def format_skill_as_markdown(data: Dict[str, Any]) -> str:
    """Deterministic JSON-dict → YAML+markdown skill file.

    Frontmatter is emitted via yaml.safe_dump so any value (including
    those starting with `{`, `"`, etc.) is correctly quoted. ``description``
    is always first; any keys in ``_EXTRA_META_KEYS`` that are present and
    non-null are carried through after it (used for provisional/auto-distill
    metadata). Sections follow in the canonical order; empty values are
    skipped.

    Note: a ``description``-only ``data`` produces byte-identical output
    to the legacy (description-only) formatter.
    """
    import yaml

    description = (str(data.get("description") or "")).strip()
    if not description:
        description = "auto-generated skill (LLM omitted description)"

    front: Dict[str, Any] = {"description": description}
    for key in _EXTRA_META_KEYS:
        if key in data and data[key] is not None:
            front[key] = data[key]

    frontmatter = yaml.safe_dump(
        front,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10_000,  # don't line-wrap the description
    ).strip()

    sections: List[str] = []
    for key in _SECTION_KEYS:
        content = (str(data.get(key) or "")).strip()
        if content:
            sections.append(f"## {key}\n\n{content}")

    body = "\n\n".join(sections)
    return f"---\n{frontmatter}\n---\n\n{body}\n"


def _read_skill_as_dict(skill_path: Path, *, domain: str) -> Dict[str, str]:
    """Load an existing skill file and return its data as a flat dict
    of {description, overview, planning, ...}. Uses the same loader the
    runtime uses, so frontmatter parsing is consistent."""
    from ..loader import load_skill

    parsed = load_skill(str(skill_path), domain=domain)
    data: Dict[str, str] = {
        "description": (parsed.get("meta") or {}).get("description", ""),
    }
    for key in _SECTION_KEYS:
        data[key] = parsed.get(key) or ""
    return data


# ──────────────────────────────────────────────────────────────
# Graduation
# ──────────────────────────────────────────────────────────────

def graduate_to_skill_file(
    *,
    knowledge_entry: Dict[str, Any],
    skill_name: str,
    domain: str,
    llm_call: Callable[[str], str],
    fresh_template: str,
    update_template: str,
    skills_root: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    append_sections: Optional[Dict[str, str]] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Graduate a knowledge entry into a skill .md file.

    Structured-I/O flow: the LLM returns a JSON object with named
    fields (description + canonical sections), and this function emits
    the YAML+markdown skill file deterministically. The LLM never has
    to produce valid YAML or markdown structure on its own.

    If a skill with this name already exists at
    ``<skills_root>/<domain>/<skill_name>/<skill_name>.md``, the LLM
    receives the existing skill as JSON via ``update_template`` and
    returns a merged JSON object. Otherwise ``fresh_template`` is used.

    Args:
        knowledge_entry: dict from `KnowledgeStore.get` or hand-built.
        skill_name: target skill name (used as both directory and .md
            filename).
        domain: skill domain (e.g. "vasp").
        llm_call: callable taking a prompt and returning the LLM's
            response text.
        fresh_template: prompt template for first-time graduation.
            Must accept {skill_name}, {domain}, {knowledge_text} keys
            and instruct the LLM to return a JSON object.
        update_template: prompt template for merging into an existing
            skill. Must accept {skill_name}, {existing_skill},
            {new_knowledge} keys and instruct the LLM to return a JSON
            object.
        skills_root: where graduated skills live. Defaults to the
            persistent store (~/.scilink/graduated_skills).
        extra_meta: optional frontmatter metadata (e.g.
            ``{"provisional": True, "provenance": "t2_autodistill"}``)
            merged into the parsed JSON before rendering. Only keys in
            ``_EXTRA_META_KEYS`` survive into the file.
        append_sections: optional ``{section: text}`` appended verbatim to
            the LLM-produced section content before rendering. Used to
            guarantee a piece of content lands in the file regardless of
            what the LLM returns (e.g. embedding the exact working script
            as a reference appendix in ``implementation``).

    Returns:
        dict with status, method ("created"/"updated"), skill_path,
        word_count, and a soft warning if the file is getting long.
    """
    skills_root = skills_root or graduated_skills_dir()
    warn_if_ephemeral_store()
    # Guard against path-traversal: domain/skill_name are filesystem components.
    domain = safe_path_component(domain, fallback="unknown_domain")
    skill_name = safe_path_component(skill_name, fallback="unnamed_skill")
    domain_dir = skills_root / domain
    skill_dir = domain_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / f"{skill_name}.md"

    knowledge_text = _format_knowledge(knowledge_entry)

    # The read-modify-write (read existing, LLM merge, write) must be one
    # critical section: two sessions graduating into the same skill would
    # otherwise both merge from the old content and the loser's work is
    # silently discarded. Locking the skill file serializes same-skill
    # graduations; graduation is rare, so this is fine.
    # ponytail: per-skill lock, split read/write locks if same-skill
    # throughput ever matters.
    with file_lock(skill_path):
        is_update = skill_path.exists()
        if is_update:
            existing_data = _read_skill_as_dict(skill_path, domain=domain)
            prompt = update_template.format(
                skill_name=skill_name,
                existing_skill=json.dumps(existing_data, indent=2),
                new_knowledge=knowledge_text,
            )
        else:
            prompt = fresh_template.format(
                skill_name=skill_name,
                domain=domain,
                knowledge_text=knowledge_text,
            )

        raw = llm_call(prompt)
        try:
            parsed = parse_json_response(raw)
        except ValueError:
            # The model answered in prose (or the JSON was truncated off the
            # end). One corrective retry with the contract restated up front.
            retry_prompt = (
                "Respond with ONLY a single JSON object — no prose before or "
                "after it.\n\n" + prompt
            )
            parsed = parse_json_response(llm_call(retry_prompt))
        if extra_meta:
            for key in _EXTRA_META_KEYS:
                if key in extra_meta and extra_meta[key] is not None:
                    parsed[key] = extra_meta[key]
        if append_sections:
            for key, text in append_sections.items():
                text = (text or "").strip()
                if not text:
                    continue
                existing = (str(parsed.get(key) or "")).strip()
                parsed[key] = f"{existing}\n\n{text}".strip() if existing else text
        skill_content = format_skill_as_markdown(parsed)

        # Build-only mode: return the proposed content WITHOUT writing it, so a
        # caller can show it for review (used by the upgrade propose/apply flow).
        if not write:
            return {
                "status": "success",
                "method": "updated" if is_update else "created",
                "skill_name": skill_name,
                "domain": domain,
                "skill_path": str(skill_path),
                "content": skill_content,
                "word_count": len(skill_content.split()),
            }

        # Loader-style bundles include __init__.py; harmless for path-loaded
        # skills, but keeps the layout consistent with built-ins.
        (skill_dir / "__init__.py").touch()
        atomic_write_text(skill_path, skill_content)

    word_count = len(skill_content.split())
    warning = None
    if word_count > WORD_COUNT_WARN_THRESHOLD:
        warning = (
            f"Graduated skill '{skill_name}' is now {word_count} words long; "
            f"consider running a manual consolidation pass."
        )

    return {
        "status": "success",
        "method": "updated" if is_update else "created",
        "skill_name": skill_name,
        "domain": domain,
        "skill_path": str(skill_path),
        "word_count": word_count,
        "warning": warning,
    }


# ──────────────────────────────────────────────────────────────
# Loading + prompt injection
# ──────────────────────────────────────────────────────────────

def load_graduated_skills(
    domain: str,
    skills_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return parsed skill dicts for every graduated skill in this domain.

    Each dict has the same shape as `scilink.skills.loader.load_skill`'s
    return value. Returns an empty list if no graduated skills exist.

    Loading uses the standard skill loader by passing the .md path
    directly (the loader supports path-as-skill — bypasses bundle
    lookup)."""
    skills_root = skills_root or graduated_skills_dir()
    domain_dir = skills_root / domain
    if not domain_dir.is_dir():
        return []

    from ..loader import load_skill

    skills: List[Dict[str, Any]] = []
    for entry in sorted(domain_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        md_path = entry / f"{entry.name}.md"
        if not md_path.exists():
            continue
        try:
            parsed = load_skill(str(md_path), domain=domain)
            skills.append(parsed)
        except Exception as exc:
            logging.warning(
                f"Could not load graduated skill at {md_path}: {exc}"
            )
    return skills


def format_graduated_skills_block(
    skills: List[Dict[str, Any]],
    *,
    sections: Optional[List[str]] = None,
) -> str:
    """Format graduated skills as a single markdown block for prompt injection.

    Each skill's description + selected sections rendered as nested
    markdown. Returns an empty string when `skills` is empty (so the
    caller can unconditionally concatenate this into a prompt).

    Args:
        skills: list returned by `load_graduated_skills`.
        sections: which canonical sections to include. Defaults to
            all six canonical sections so newly-graduated rules can't
            be hidden by the section filter — they get the prompt
            visibility they were graduated to provide.
    """
    if not skills:
        return ""
    sections = sections or list(_SECTION_KEYS)
    lines = ["", "## LEARNED RULES (graduated skills)", ""]
    for sk in skills:
        name = sk.get("name", "graduated")
        desc = (sk.get("meta") or {}).get("description", "")
        lines.append(f"### {name}")
        if desc:
            lines.append(f"_{desc}_")
        lines.append("")
        for section in sections:
            content = (sk.get(section) or "").strip()
            if content:
                lines.append(f"#### {section}")
                lines.append(content)
                lines.append("")
    return "\n".join(lines)
