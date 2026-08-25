"""Shared agent-side skill relevance selector.

Lets the analysis foundation agents (image, curve-fitting, hyperspectral)
auto-select **zero or more** relevant skills by inspecting the actual data,
rather than relying solely on the orchestrator to pass a skill list.

The agent supplies only its modality-specific context (image bytes, curve
statistics, metadata); this module owns the catalog, the selection prompt,
parsing, and validation. It mirrors the catalog convention used by the
orchestrator's ``_build_skill_description`` — frontmatter ``description``
(falling back to the overview's first line) plus a ``[technique: …]`` tag,
with provisional (auto-distilled) skills excluded — so agent-side selection
technique-matches the same way the orchestrator does (see issue #251).

Selection is deliberately conservative: prefer zero or one skill unless
multiple are *clearly* warranted. The result is ranked most-relevant-first,
which downstream code relies on — the top-ranked skill drives single-recipe
codegen while the full set enriches the prose context at each stage.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


def build_skill_catalog(domain: str, *, include_provisional: bool = False,
                        custom_skills: Optional[dict] = None):
    """Build the selectable-skill catalog for ``domain``.

    Returns ``(catalog_lines, names)`` where ``catalog_lines`` is a list of
    human-/LLM-readable bullet strings and ``names`` is the set of valid skill
    names the selector is allowed to return. Mirrors the orchestrator's
    description convention (``analysis_orchestrator_tools._build_skill_description``).

    ``custom_skills`` is an optional ``{name: path}`` map of user-registered
    skills (UI upload / ``--skills``). They are folded into the catalog so the
    agent-side selector can auto-select them like built-ins (issue #256 fix #1),
    EXCEPT when a custom skill's frontmatter declares a *different* analysis
    modality — that explicit cross-modality skill is skipped (mirrors the
    loader's ``_MODALITY_DOMAINS`` guard). A custom skill with no declared
    domain is included; the selector's technique/modality reasoning rejects it
    if the data does not fit.
    """
    from ..loader import list_skills, load_skill, _MODALITY_DOMAINS

    catalog: List[str] = []
    names: set[str] = set()

    def _entry(parsed, name):
        meta = parsed.get("meta") or {}
        if meta.get("provisional") is True and not include_provisional:
            return False
        desc = meta.get("description")
        if not desc:
            desc = parsed.get("overview", "").split("\n")[0].strip()
        desc = desc.rstrip(".;,") if desc else desc
        techs = meta.get("technique")
        if isinstance(techs, str):
            techs = [techs]
        tech_tag = (
            f" [technique: {', '.join(map(str, techs))}]" if techs else ""
        )
        body = f" — {desc}" if desc else ""
        catalog.append(f"- **{name}**{tech_tag}{body}")
        names.add(name)
        return True

    for name in list_skills(domain=domain):
        try:
            parsed = load_skill(name, domain=domain)
        except Exception:
            continue
        _entry(parsed, name)

    # User-registered custom skills (session-scoped) — fold them in so the
    # agent can auto-select an uploaded skill, not just have the orchestrator
    # pass it authoritatively.
    for name, path in (custom_skills or {}).items():
        if name in names:
            continue
        try:
            parsed = load_skill(path, domain=domain)
        except Exception:
            continue
        declared = (parsed.get("meta") or {}).get("domain")
        if declared in _MODALITY_DOMAINS and declared != domain:
            continue  # explicitly authored for a different modality
        _entry(parsed, name)

    return catalog, names


_SELECTION_GUIDANCE = (
    "Select a skill only when its scope genuinely fits the data. When a skill "
    "declares a `[technique: …]` tag, REQUIRE that the data's measurement "
    "technique matches that tag — do NOT substitute the nearest-sounding "
    "technique skill (e.g. an XRD skill for Raman data). When a skill has no "
    "technique tag, match it on the data's imaging modality or the analysis "
    "task it describes (e.g. a touching/overlapping-objects skill for an image "
    "of densely packed grains). If nothing genuinely fits, return an empty "
    "list and the agent's baseline expertise will handle it. Prefer zero or "
    "one skill; return several ONLY when the data clearly spans complementary "
    "skills that each contribute. Order the list most-relevant first."
)

# Exclusive policy — for agents (e.g. curve-fitting) whose skills encode
# AUTHORITATIVE, mutually-exclusive technique rules. The data is produced by
# one measurement technique, and two technique skills would inject
# contradictory mandatory rules, so at most one may be selected.
_EXCLUSIVE_GUIDANCE = (
    "These skills encode authoritative, technique-specific rules and are "
    "mutually exclusive — the data is produced by ONE measurement technique. "
    "Select the SINGLE skill whose technique matches the data's technique "
    "(compare against each skill's `[technique: …]` tag). If no skill's "
    "technique matches, return an empty list and the agent's baseline handles "
    "it — do NOT substitute the nearest-sounding skill, and never select more "
    "than one."
)


def select_relevant_skills(
    *,
    model: Any,
    parse_fn: Callable,
    domain: str,
    context_parts: List[Any],
    generation_config: Any = None,
    safety_settings: Any = None,
    max_skills: int = 3,
    exclusive: bool = False,
    hint: Any = None,
    custom_skills: Optional[dict] = None,
    logger: Optional[Any] = None,
) -> List[str]:
    """Ask the model which domain skills (zero or more) fit the data.

    Args:
        model: LLM client exposing ``generate_content(contents, ...)``.
        parse_fn: the agent's response parser, returning ``(result_dict, _)``.
        domain: skill domain to draw the catalog from.
        context_parts: prompt parts describing the *actual* data — text
            (stats, metadata) and/or vision dicts (``{"mime_type", "data"}``).
        max_skills: hard cap on how many skills may be returned.
        exclusive: when True, the skills are treated as authoritative and
            mutually exclusive (curve-fitting techniques) — the prompt asks for
            the single best technique match and the result is capped to one.
            When False (default), multiple complementary skills may be
            returned (image / hyperspectral stages compose).
        hint: optional skill name(s) the orchestrator *suggests* may apply
            (from conversation/preview context the data may not show). A
            NON-BINDING prior — the agent confirms from the data, augments, or
            overrides. The agent has final authority.
        custom_skills: optional ``{name: path}`` of user-registered skills to
            fold into the catalog so they are auto-selectable. A returned
            custom name must be resolved to its path (via this same map) before
            loading, since it is not on the loader's search roots.

    Returns:
        A ranked (most-relevant-first), de-duplicated list of valid skill
        names, possibly empty. Never raises — failures log and return ``[]``.
    """
    catalog, valid = build_skill_catalog(domain, custom_skills=custom_skills)
    if not catalog:
        return []

    cap = 1 if exclusive else max_skills
    guidance = _EXCLUSIVE_GUIDANCE if exclusive else _SELECTION_GUIDANCE
    prompt: List[Any] = [
        "Decide which (if any) of these domain skills are relevant to the "
        "data described below.\n\n## Available Skills\n" + "\n".join(catalog),
        "\n## Data / Context\n",
    ]
    prompt.extend(context_parts)

    # Non-binding suggestion from the orchestrator (conversational/preview
    # context the agent can't see). Only surface hints that name real skills;
    # the agent still decides from the data above.
    hint_names = [hint] if isinstance(hint, str) else list(hint or [])
    hint_names = [h for h in hint_names if h in valid]
    if hint_names:
        prompt.append(
            "\n## Orchestrator suggestion (non-binding)\n"
            f"Based on the conversation/preview, these skills MAY apply: "
            f"{', '.join(hint_names)}. Treat this only as a prior — confirm it "
            "against the data above, add complementary skills if warranted, or "
            "disregard it if the data does not support it. You decide."
        )
    prompt.append(
        "\n" + guidance + "\n\n"
        'Respond with JSON: {"skills": ["<name>", ...]} listing the relevant '
        "skill names (an empty list if none apply)."
    )

    try:
        resp = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        result, _ = parse_fn(resp)
    except Exception as e:  # pragma: no cover - defensive
        if logger:
            logger.warning(f"  Skill selection failed: {e}")
        return []

    result = result or {}
    # Accept the list contract, but tolerate a singular {"skill": "x"} reply.
    raw = result.get("skills")
    if raw is None:
        one = result.get("skill")
        raw = [one] if one else []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    selected: List[str] = []
    seen: set[str] = set()
    for item in raw:
        name = item.strip() if isinstance(item, str) else None
        if name and name in valid and name not in seen:
            seen.add(name)
            selected.append(name)
        if len(selected) >= cap:
            break

    if logger:
        if selected:
            logger.info(f"  Auto-selected domain skill(s): {', '.join(selected)}")
        else:
            logger.info("  No skill auto-selected")
    return selected
