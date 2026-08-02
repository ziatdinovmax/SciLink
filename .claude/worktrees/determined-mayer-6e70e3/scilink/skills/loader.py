"""
Skill loader for domain-specific knowledge files.

Skills are markdown files with structured sections (## headings) that provide
domain-specific guidance for LLM-driven analysis pipelines. They can be
built-in (shipped with the package) or user-provided (custom .md files).
"""

import re
from pathlib import Path
from typing import Dict

_SKILLS_DIR = Path(__file__).parent

_KNOWN_SECTIONS = {"overview", "planning", "analysis", "interpretation", "validation", "implementation"}


def list_skills(domain: str = "curve_fitting") -> list:
    """Return names of available built-in skills for a domain.

    Args:
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Sorted list of skill name strings (file stems).
    """
    skills_dir = _SKILLS_DIR / domain
    if not skills_dir.is_dir():
        return []
    return sorted(p.stem for p in skills_dir.glob("*.md"))


def list_all_skills() -> dict:
    """Auto-discover all built-in skill domains and their skills.

    Scans subdirectories of the skills package directory for ``.md`` files.

    Returns:
        Dict mapping domain names to sorted lists of skill names,
        e.g. ``{"curve_fitting": ["xps"], "hyperspectral": ["eels"]}``.
        Empty domains are omitted.
    """
    result = {}
    for sub in sorted(_SKILLS_DIR.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        names = sorted(p.stem for p in sub.glob("*.md"))
        if names:
            result[sub.name] = names
    return result


def load_skill(skill: str, domain: str = "curve_fitting") -> Dict[str, str]:
    """Load and parse a skill markdown file.

    Args:
        skill: Either a built-in skill name (e.g. "xps") which resolves to
            ``scilink/skills/{domain}/{name}.md``, or a path to a custom
            ``.md`` file.
        domain: Skill domain subdirectory (default: "curve_fitting").

    Returns:
        Dict with keys: name, overview, planning, analysis, interpretation,
        validation. Missing sections default to empty string.

    Raises:
        FileNotFoundError: If the skill file cannot be found.
        ValueError: If the file is empty or cannot be parsed.
    """
    path = _resolve_skill_path(skill, domain)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"Skill file is empty: {path}")

    sections = _parse_sections(text)
    sections["name"] = path.stem
    return sections


def _resolve_skill_path(skill: str, domain: str) -> Path:
    """Resolve a skill name or path to an actual file path."""
    candidate = Path(skill)
    if candidate.suffix.lower() == ".md":
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Skill file not found: {candidate}")

    built_in = _SKILLS_DIR / domain / f"{skill}.md"
    if built_in.exists():
        return built_in

    available_dir = _SKILLS_DIR / domain
    available = sorted(p.stem for p in available_dir.glob("*.md")) if available_dir.is_dir() else []
    raise FileNotFoundError(
        f"Skill '{skill}' not found. Available built-in skills for '{domain}': {available}"
    )


def _parse_sections(text: str) -> Dict[str, str]:
    """Parse markdown into sections keyed by ``## heading`` name."""
    sections = {k: "" for k in _KNOWN_SECTIONS}

    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)

    for part in parts[1:]:
        lines = part.split("\n", 1)
        heading = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""
        if heading in _KNOWN_SECTIONS:
            sections[heading] = body

    return sections
