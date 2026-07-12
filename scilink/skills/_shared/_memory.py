"""Persistent-memory backend API.

One package-neutral (``ase``-free) surface for listing, inspecting,
promoting, and pruning the skills in the persistent store
(``~/.scilink/graduated_skills/<domain>/<name>/<name>.md``). It is the
single backend reused by the ``scilink memory`` CLI, the UI memory panel,
and the meta-agent's review path — and the intended backend for a future
interactive skill builder.

A skill is "provisional" when its frontmatter carries ``provisional: true``
(auto-distilled from a T=2 curve-fit success). Provisional skills are kept
out of the auto-routing menu until promoted; ``promote_memory`` clears the
flag, after which they route normally.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..loader import graduated_skills_dir, load_skill
from ._graduation import safe_path_component

# Captures the frontmatter body with EXACTLY one trailing newline after the
# closing fence, so ``text[match.end():]`` preserves the rest of the file
# (including the blank line before the first section) byte-for-byte. The
# loader's own ``_split_frontmatter`` consumes that blank line, which we
# must not do when rewriting in place.
_FRONTMATTER_BLOCK_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _bundle_path(domain: str, name: str, *, root: Optional[Path] = None) -> Path:
    root = root or graduated_skills_dir()
    # domain/name are filesystem components — sanitize against path traversal.
    domain = safe_path_component(domain, fallback="unknown_domain")
    name = safe_path_component(name, fallback="unnamed_skill")
    return root / domain / name / f"{name}.md"


def _iter_bundles(root: Path):
    """Yield (domain, name, md_path) for every skill bundle under root."""
    if not root.is_dir():
        return
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith(("_", ".")):
            continue
        for skill_dir in sorted(domain_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith(("_", ".")):
                continue
            md = skill_dir / f"{skill_dir.name}.md"
            if md.exists():
                yield domain_dir.name, skill_dir.name, md


def list_memory(
    domain: Optional[str] = None,
    provisional: Optional[bool] = None,
    *,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List persisted skills.

    Args:
        domain: restrict to one domain (e.g. ``"curve_fitting"``).
        provisional: ``True`` → only provisional, ``False`` → only
            promoted, ``None`` → both.
        root: override the store root (mainly for tests).

    Returns:
        A list of dicts: ``{name, domain, path, provisional, provenance,
        session, r_squared, description}``.
    """
    root = root or graduated_skills_dir()
    out: List[Dict[str, Any]] = []
    for dom, name, md in _iter_bundles(root):
        if domain is not None and dom != domain:
            continue
        try:
            parsed = load_skill(str(md), domain=dom)
            meta = parsed.get("meta") or {}
        except Exception:
            meta = {}
        is_provisional = meta.get("provisional") is True
        if provisional is not None and is_provisional != provisional:
            continue
        out.append({
            "name": name,
            "domain": dom,
            "path": str(md),
            "provisional": is_provisional,
            "provenance": meta.get("provenance"),
            "session": meta.get("session"),
            "r_squared": meta.get("r_squared"),
            "description": meta.get("description", ""),
        })
    return out


def show_memory(domain: str, name: str, *, root: Optional[Path] = None) -> str:
    """Return the raw markdown of a persisted skill bundle."""
    md = _bundle_path(domain, name, root=root)
    if not md.exists():
        raise FileNotFoundError(f"No skill bundle: {domain}/{name}")
    return md.read_text()


def promote_memory(
    domain: str,
    name: str,
    to_domain: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote a provisional skill so it routes normally.

    Strips the ``provisional`` / ``provenance`` keys from the YAML
    frontmatter (leaving the section bodies byte-for-byte unchanged) so
    the skill re-enters the auto-routing menu. Optionally moves the bundle
    under a different (curated) domain.
    """
    root = root or graduated_skills_dir()
    md = _bundle_path(domain, name, root=root)
    if not md.exists():
        raise FileNotFoundError(f"No skill bundle: {domain}/{name}")

    import yaml

    text = md.read_text()
    match = _FRONTMATTER_BLOCK_RE.match(text)
    if not match:
        # No frontmatter to strip — nothing routes-blocking present.
        return {
            "status": "success",
            "name": name,
            "domain": domain,
            "path": str(md),
            "promoted": True,
        }
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)  # copy; preserves insertion order
    meta.pop("provisional", None)
    meta.pop("provenance", None)
    body = text[match.end():]  # preserved byte-for-byte

    if meta:
        frontmatter = yaml.safe_dump(
            meta,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=10_000,
        ).strip()
        new_text = f"---\n{frontmatter}\n---\n{body}"
    else:
        # No frontmatter left; drop the fence entirely.
        new_text = body.lstrip("\n")

    md.write_text(new_text)

    moved_to = None
    if to_domain and to_domain != domain:
        dest_dir = root / to_domain / name
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(md.parent), str(dest_dir))
        moved_to = str(dest_dir / f"{name}.md")

    return {
        "status": "success",
        "name": name,
        "domain": to_domain or domain,
        "path": moved_to or str(md),
        "promoted": True,
    }


def demote_memory(domain: str, name: str, *, root: Optional[Path] = None) -> Dict[str, Any]:
    """Demote a skill back to provisional — the reverse of ``promote_memory``.

    Sets ``provisional: true`` in the YAML frontmatter (section bodies stay
    byte-for-byte unchanged), taking the skill OUT of the auto-routing menu
    while keeping it explicitly loadable and reviewable. For a skill that
    turned out to mis-route or needs another look before it keeps
    auto-applying.
    """
    root = root or graduated_skills_dir()
    md = _bundle_path(domain, name, root=root)
    if not md.exists():
        raise FileNotFoundError(f"No skill bundle: {domain}/{name}")

    import yaml

    text = md.read_text()
    match = _FRONTMATTER_BLOCK_RE.match(text)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        body = text[match.end():]
    else:
        meta = {}
        body = text
    meta["provisional"] = True
    frontmatter = yaml.safe_dump(
        meta,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10_000,
    ).strip()
    md.write_text(f"---\n{frontmatter}\n---\n{body}")
    return {
        "status": "success",
        "name": name,
        "domain": domain,
        "path": str(md),
        "provisional": True,
    }


def prune_memory(domain: str, name: str, *, root: Optional[Path] = None) -> Dict[str, Any]:
    """Delete a persisted skill bundle directory."""
    root = root or graduated_skills_dir()
    md = _bundle_path(domain, name, root=root)
    skill_dir = md.parent
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"No skill bundle: {domain}/{name}")
    shutil.rmtree(skill_dir)
    return {"status": "success", "name": name, "domain": domain, "pruned": True}
