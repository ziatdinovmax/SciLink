"""Staging buffer for raw T=2 solutions, and the two distillation consumers.

The T=2 auto-distill hooks no longer write a skill per win. Instead they
**stage** the raw solution (the agent's knowledge entry + the verbatim working
script + an LLM-assigned technique label) here, and skills are produced later,
review-gated, two ways:

  upgrade@1  — a single staged solution enriches an EXISTING skill
               (``upgrade_skill_from_staged`` → ``graduate_to_skill_file`` update branch).
  new-skill@N — N staged solutions for one technique are consolidated into a
               NEW skill (``consolidate_technique``).

Staging lives at ``scilink_home()/distill_staging/<domain>/<id>.json`` — a sibling
of ``graduated_skills/`` so the skill loader never mistakes a staged record for a
skill. The module is package-neutral (``ase``-free): stdlib + the ``_graduation``
helper + the loader, same as ``_memory``.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..loader import scilink_home, load_skill, list_skills, graduated_skills_dir
from ._graduation import graduate_to_skill_file, safe_path_component, warn_if_ephemeral_store


# Friendly, jargon-free labels for record provenance (the stored values stay as
# t2_solution/error_fix/user_correction; these are for user-facing display only).
PROVENANCE_LABELS = {
    "t2_solution": "solved from scratch",
    "t2_hot_win": "solved from scratch (banked)",
    "bank_proven": "proven across sessions",
    "bank_nominated": "nominated from the bank",
    "error_fix": "error fix",
    "user_correction": "your feedback",
}


def metric_label(rec: Dict[str, Any]) -> str:
    """Agent-agnostic metric label for display: R² for fits, 'score' otherwise."""
    if rec.get("r_squared") is not None:
        return f"R² {rec['r_squared']}"
    if rec.get("quality_score") is not None:
        return f"score {rec['quality_score']}"
    return ""


def staging_dir() -> Path:
    """Root of the raw-solution staging buffer (honors ``$SCILINK_HOME``)."""
    return scilink_home() / "distill_staging"


# Advisory minimum number of staged solutions of a technique before consolidating
# them into a NEW skill. Below this, review surfaces accumulate rather than suggest
# graduation (a single example is usually too idiosyncratic to generalize). Upgrading
# an EXISTING skill is exempt — that is upgrade@1 by design. The function itself
# still consolidates whatever is present when explicitly called.
_DEFAULT_CONSOLIDATE_N = 3


def consolidate_min_n() -> int:
    """Readiness threshold for new-skill consolidation (``$SCILINK_CONSOLIDATE_N``)."""
    import os
    raw = os.environ.get("SCILINK_CONSOLIDATE_N", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_CONSOLIDATE_N


def _domain_dir(domain: str, *, root: Optional[Path] = None) -> Path:
    # domain is a filesystem component — sanitize to prevent path traversal.
    return (root or staging_dir()) / safe_path_component(domain, fallback="unknown_domain")


# ──────────────────────────────────────────────────────────────
# Buffer CRUD
# ──────────────────────────────────────────────────────────────

def stage_solution(
    domain: str,
    technique: str,
    record: Dict[str, Any],
    *,
    root: Optional[Path] = None,
) -> str:
    """Persist one raw T=2 solution; return its assigned id.

    ``record`` is the agent's ``knowledge_entry`` (model/deviation/metric/script,
    etc.); ``technique`` is the normalized grouping label. The stored JSON adds
    ``id``, ``domain``, ``technique``.
    """
    warn_if_ephemeral_store()
    d = _domain_dir(domain, root=root)
    d.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:8]
    payload = {"id": sid, "domain": domain, "technique": technique, **record}
    (d / f"{sid}.json").write_text(json.dumps(payload, indent=2, default=str))
    return sid


def list_staged(
    domain: Optional[str] = None,
    technique: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return staged solution records, optionally filtered by domain/technique."""
    base = root or staging_dir()
    out: List[Dict[str, Any]] = []
    if not base.is_dir():
        return out
    domains = [base / domain] if domain else [
        p for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_"))
    ]
    for dd in domains:
        if not dd.is_dir():
            continue
        for f in sorted(dd.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            if technique is not None and rec.get("technique") != technique:
                continue
            rec.setdefault("domain", dd.name)
            out.append(rec)
    return out


def get_staged(domain: str, sid: str, *, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    f = _domain_dir(domain, root=root) / f"{sid}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def relabel_staged(domain: str, sid: str, technique: str, *,
                   root: Optional[Path] = None) -> Dict[str, Any]:
    """Move a staged record to a different technique group.

    Technique labels are assigned at staging time (LLM or user), but the
    grouping they produce is what consolidation/upgrade operate on — so the
    user must be able to reassemble groups (e.g. pull a session's error
    lesson into the technique group its script was nominated under).
    The label is normalized the same way as assigned labels.
    """
    import re
    rec = get_staged(domain, sid, root=root)
    if rec is None:
        return {"status": "error", "message": f"No staged record {domain}/{sid}."}
    label = re.sub(r"[^a-z0-9]+", "_", str(technique).lower()).strip("_")[:48]
    if not label:
        return {"status": "error", "message": "Empty technique label."}
    rec["technique"] = label
    (_domain_dir(domain, root=root) / f"{sid}.json").write_text(
        json.dumps(rec, indent=2, default=str))
    return {"status": "success", "technique": label, "id": sid}


def remove_staged(domain: str, ids: List[str], *, root: Optional[Path] = None) -> int:
    """Delete staged records by id; return count removed."""
    d = _domain_dir(domain, root=root)
    n = 0
    for sid in ids:
        f = d / f"{sid}.json"
        if f.exists():
            f.unlink()
            n += 1
    return n


def group_by_technique(domain: str, *, root: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Group a domain's staged records by their technique label."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in list_staged(domain, root=root):
        groups.setdefault(rec.get("technique") or "unlabeled", []).append(rec)
    return groups


def assign_technique_label(
    domain: str,
    model: str,
    deviation: str,
    llm_call: Callable[[str], str],
    label_template: str,
    *,
    root: Optional[Path] = None,
) -> str:
    """LLM-assign a normalized snake_case technique label (reuse-or-create).

    Uses ``existing_techniques(domain)`` as the reuse vocabulary. Returns a
    sanitized label; falls back to ``"uncategorized"`` if the model yields
    nothing usable.
    """
    import re
    existing = existing_techniques(domain, root=root)
    prompt = label_template.format(
        model=model or "(unknown model)",
        deviation=deviation or "(no explicit note)",
        existing="\n".join(f"- {t}" for t in existing) or "(none)",
    )
    raw = (llm_call(prompt) or "").strip()
    first = raw.splitlines()[0] if raw else ""
    label = re.sub(r"[^a-z0-9]+", "_", first.lower()).strip("_")[:48]
    return label or "uncategorized"


def existing_techniques(domain: str, *, root: Optional[Path] = None) -> List[str]:
    """Vocabulary for the labeling prompt + target hints.

    Union of (a) technique labels already on staged records and (b) existing
    skill names in this domain (persistent + built-in, via the loader)."""
    labels = set(group_by_technique(domain, root=root).keys())
    try:
        labels.update(list_skills(domain))
    except Exception:
        pass
    return sorted(t for t in labels if t and t != "unlabeled")


# ──────────────────────────────────────────────────────────────
# Shared: build a reference block + knowledge text from a record
# ──────────────────────────────────────────────────────────────

def _script_of(rec: Dict[str, Any]) -> str:
    return (rec.get("working_script") or rec.get("script") or "").strip()


# How much of the working script to show the LLM as *input* (so it can extract
# the reusable recipe from real code). The script is NOT copied verbatim into the
# saved skill — doing so bloated reused skills (a ~1500-word one-shot script) and
# empirically degraded the next run by over-constraining it. Distill, don't dump.
_SCRIPT_PROMPT_CHARS = 4000


def _record_for_prompt(rec: Dict[str, Any]) -> Dict[str, Any]:
    """A record trimmed for the LLM prompt — drop bookkeeping; include a
    length-capped working script so the LLM can generalize from real code without
    the full script being echoed into the saved skill."""
    drop = {"id", "domain", "technique", "working_script", "script", "created_at"}
    out = {k: v for k, v in rec.items() if k not in drop}
    script = _script_of(rec)
    if script:
        out["working_script_excerpt"] = (
            script if len(script) <= _SCRIPT_PROMPT_CHARS
            else script[:_SCRIPT_PROMPT_CHARS] + "\n# … (truncated; full script in the staging record)"
        )
    return out


# ──────────────────────────────────────────────────────────────
# Feedback / error capture — the second staging source.
#
# Besides T=2 hot-annealing wins (``t2_solution`` provenance), the buffer also
# captures the two signals the old in-session distillation used — human feedback
# and recurring errors that were resolved during a run — so they persist across
# sessions and flow through the same review/upgrade/consolidate path. These are
# staged as records with provenance ``user_correction`` / ``error_fix``.
# ──────────────────────────────────────────────────────────────

def resolved_error_lessons(quality_history: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract ``{error, fix}`` lessons from a result's quality history.

    Pulls script errors that were diagnosed/fixed (curve-fitting stores the fix
    under ``diagnosis``, image analysis under ``fix``) and verification
    iterations where a fix was applied to named issues. Unresolved errors (no
    fix) are skipped — a lesson needs the resolution to be reusable.
    """
    qh = quality_history or {}
    lessons: List[Dict[str, str]] = []
    for se in qh.get("script_errors", []) or []:
        err = (se.get("error") or "").strip()
        fix = (se.get("fix") or se.get("diagnosis") or "").strip()
        if err and fix:
            lessons.append({"error": err[:400], "fix": fix[:400]})
    for it in qh.get("verification_iterations", []) or []:
        fix = (it.get("fix_applied") or "").strip()
        issues = [x.get("problem", "") for x in (it.get("issues") or []) if x.get("problem")]
        if fix and issues:
            lessons.append({"error": "; ".join(issues)[:400], "fix": fix[:400]})
    return lessons


def stage_feedback_and_errors(
    domain: str,
    *,
    results: List[Dict[str, Any]],
    feedback_texts: Optional[List[str]] = None,
    session: str,
    llm_call: Callable[[str], str],
    label_template: str,
    root: Optional[Path] = None,
) -> List[str]:
    """Stage human-feedback + resolved-error knowledge from a finished run.

    For each successful result that resolved errors along the way, stages an
    ``error_fix`` record; if the run gathered any human feedback (``feedback_texts``),
    stages one combined ``user_correction`` record. Each gets an LLM-assigned
    technique label so it accumulates alongside (and consolidates with) T=2
    solutions of the same technique. Returns the list of staged ids.
    """
    staged: List[str] = []
    for r in results or []:
        if not r.get("success"):
            continue
        lessons = resolved_error_lessons(r.get("quality_history"))
        if not lessons:
            continue
        model = r.get("model_type") or r.get("analysis_type") or "model"
        technique = assign_technique_label(
            domain, model, "recurring errors resolved during the run",
            llm_call, label_template, root=root)
        rec: Dict[str, Any] = {"provenance": "error_fix", "model": model,
                               "error_lessons": lessons, "session": session}
        r2 = (r.get("fit_quality") or {}).get("r_squared")
        if r2 is not None:
            rec["r_squared"] = round(float(r2), 4)
        staged.append(stage_solution(domain, technique, rec, root=root))

    texts = [str(t).strip() for t in (feedback_texts or []) if str(t).strip()]
    if texts:
        fb = "\n".join(dict.fromkeys(texts))  # dedupe, preserve order
        model = next((r.get("model_type") or r.get("analysis_type")
                      for r in (results or []) if r.get("model_type") or r.get("analysis_type")), "model")
        technique = assign_technique_label(
            domain, model, "user correction / domain expertise",
            llm_call, label_template, root=root)
        staged.append(stage_solution(domain, technique, {
            "provenance": "user_correction", "model": model,
            "user_feedback": fb, "session": session}, root=root))
    return staged


# ──────────────────────────────────────────────────────────────
# upgrade@1 — merge staged solution(s) into an EXISTING skill
#
# Upgrade mutates an existing (often already-active) skill in place, so it is
# split into propose → apply for a human-in-the-loop review of the merged
# content: ``propose_skill_upgrade`` builds the merged skill WITHOUT writing it
# (returns the proposed text + the current text for a diff); ``apply_skill_upgrade``
# backs up the current file and writes the approved content. ``upgrade_skill_from_staged``
# remains as the one-shot (propose+apply) for non-interactive callers.
# ──────────────────────────────────────────────────────────────

def _skill_md_path(target_domain: str, target_name: str,
                   skills_root: Optional[Path] = None) -> Path:
    n = safe_path_component(target_name)
    return ((skills_root or graduated_skills_dir())
            / safe_path_component(target_domain) / n / f"{n}.md")


def _missing_target_error(target_domain: str, target_name: str) -> Dict[str, Any]:
    return {"status": "error",
            "message": (f"Target skill '{target_domain}/{target_name}' does not exist — "
                        f"upgrade enriches an existing skill. Check the name, or use "
                        f"consolidate to create a new skill.")}


_FM_RE = __import__("re").compile(r"\A---\n(.*?)\n---\n", __import__("re").DOTALL)


def _preserve_structure(existing: str, proposed: str) -> str:
    """Structural additivity: reinstate what the JSON round-trip loses.

    The upgrade LLM works on a fixed-vocabulary JSON view of the skill, so
    its re-rendered markdown drops everything outside that vocabulary —
    observed live on a forked built-in: the ``technique:`` routing
    frontmatter (breaks skill auto-selection), the H1 title, the original
    heading casing, and an injected empty section. This deterministic pass
    restores the original frontmatter (the LLM may refresh only
    ``description``), the title, and heading case, and drops empty sections
    the original didn't have. Prose merging stays the LLM's job; structure
    is not up for negotiation.
    """
    import re
    import yaml

    m_old, m_new = _FM_RE.match(existing), _FM_RE.match(proposed)
    old_meta = yaml.safe_load(m_old.group(1)) if m_old else {}
    new_meta = yaml.safe_load(m_new.group(1)) if m_new else {}
    old_meta = old_meta if isinstance(old_meta, dict) else {}
    new_meta = new_meta if isinstance(new_meta, dict) else {}
    merged = dict(old_meta)
    if new_meta.get("description"):
        merged["description"] = new_meta["description"]

    body = proposed[m_new.end():] if m_new else proposed
    old_body = existing[m_old.end():] if m_old else existing

    # Restore original heading casing (case-insensitive match).
    old_heads = {h.lower(): h
                 for h in re.findall(r"^##\s+(.+?)\s*$", old_body, re.M)}
    body = re.sub(
        r"^(##\s+)(.+?)\s*$",
        lambda m: m.group(1) + old_heads.get(m.group(2).lower(), m.group(2)),
        body, flags=re.M)

    # Drop EMPTY sections the original didn't have (round-trip artifacts).
    parts = re.split(r"(?m)^(##\s+.+)$", body)
    rebuilt = [parts[0]]
    for i in range(1, len(parts), 2):
        head, section_body = parts[i], (parts[i + 1] if i + 1 < len(parts) else "")
        name = head.lstrip("# ").strip().lower()
        if not section_body.strip() and name not in old_heads:
            continue
        rebuilt.append(head + section_body)
    body = "".join(rebuilt) if len(rebuilt) > 1 else body

    # Reinstate whole sections the proposal dropped — additivity by
    # construction: an omitted section reappears verbatim from the original
    # (appended in original order), so an LLM omission can no longer delete
    # curated content. The reviewer still sees it via the diff.
    new_heads = {h.lower()
                 for h in re.findall(r"^##\s+(.+?)\s*$", body, re.M)}
    old_parts = re.split(r"(?m)^(##\s+.+)$", old_body)
    for i in range(1, len(old_parts), 2):
        head = old_parts[i]
        name = head.lstrip("# ").strip().lower()
        if name not in new_heads:
            section_body = old_parts[i + 1] if i + 1 < len(old_parts) else ""
            body = body.rstrip("\n") + "\n\n" + head + section_body

    # Restore a lost H1 title.
    m_title = re.match(r"\s*(#\s+[^#\n][^\n]*)", old_body)
    if m_title and not re.search(r"^#\s+[^#]", body, re.M):
        body = m_title.group(1) + "\n\n" + body.lstrip("\n")

    if merged:
        fm = yaml.safe_dump(merged, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=10_000).strip()
        return f"---\n{fm}\n---\n{body}"
    return body


def _regression_warnings(existing: str, proposed: str) -> List[str]:
    """Deterministic additivity check on an upgrade proposal.

    Upgrades must not regress a skill that already routes real analyses:
    flag dropped ``##`` sections and large content shrinkage so the human
    reviewer sees structural loss before approving. Advisory — the reviewer
    decides; a deliberate trim can still be applied.
    """
    import re
    old_secs = re.findall(r"^##\s+(.+?)\s*$", existing, re.M)
    new_secs = {s.lower() for s in re.findall(r"^##\s+(.+?)\s*$", proposed, re.M)}
    warns = [f"existing section '## {s}' is missing from the proposal"
             for s in old_secs if s.lower() not in new_secs]
    n_old, n_new = len(existing.split()), len(proposed.split())
    if n_old and n_new < 0.8 * n_old:
        warns.append(
            f"proposal is {100 - round(100 * n_new / n_old)}% shorter than the "
            f"current skill ({n_new} vs {n_old} words) — check that existing "
            f"guidance was not dropped")
    return warns


def propose_skill_upgrade(
    domain: str,
    staged_ids: List[str],
    *,
    target_domain: str,
    target_name: str,
    llm_call: Callable[[str], str],
    fresh_template: str,
    update_template: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the merged skill for review WITHOUT writing it.

    Returns ``proposed_content`` (the new skill text) and ``existing_content``
    (the current file) so a caller can show a diff and confirm. Nothing is
    written and no staged records are consumed until ``apply_skill_upgrade``.
    """
    recs = [r for r in (get_staged(domain, sid, root=root) for sid in staged_ids) if r]
    if not recs:
        return {"status": "error", "message": "No staged records found for the given ids."}

    target_md = _skill_md_path(target_domain, target_name, skills_root)
    if not target_md.exists():
        return _missing_target_error(target_domain, target_name)

    knowledge_entry = {"new_t2_solutions": [_record_for_prompt(r) for r in recs]}
    result = graduate_to_skill_file(
        knowledge_entry=knowledge_entry,
        skill_name=target_name,
        domain=target_domain,
        llm_call=llm_call,
        fresh_template=fresh_template,
        update_template=update_template,
        skills_root=skills_root,
        write=False,
    )
    if result.get("status") != "success":
        return result
    existing_content = target_md.read_text()
    proposed_content = _preserve_structure(existing_content, result["content"])
    return {
        "status": "success",
        "action": "proposed",
        "proposed_content": proposed_content,
        "existing_content": existing_content,
        "regression_warnings": _regression_warnings(
            existing_content, proposed_content),
        "skill_path": str(target_md),
        "target_domain": target_domain,
        "target_name": target_name,
        "staged_ids": [r["id"] for r in recs if r.get("id")],
        "word_count": result.get("word_count"),
    }


def apply_skill_upgrade(
    domain: str,
    staged_ids: List[str],
    *,
    target_domain: str,
    target_name: str,
    proposed_content: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a previously-proposed upgrade, backing up the current file first.

    The pre-upgrade skill is copied to ``<name>.md.bak`` (so an upgrade can be
    reverted), the approved ``proposed_content`` is written, and the consumed
    staged records are removed.
    """
    target_md = _skill_md_path(target_domain, target_name, skills_root)
    if not target_md.exists():
        return _missing_target_error(target_domain, target_name)
    # Back up the current version (single last-version undo; .bak is ignored by
    # the loader, which only discovers <name>.md).
    backup = target_md.with_name(target_md.name + ".bak")
    backup.write_text(target_md.read_text())
    (target_md.parent / "__init__.py").touch()
    target_md.write_text(proposed_content)
    removed = remove_staged(domain, staged_ids, root=root)
    return {
        "status": "success",
        "method": "updated",
        "skill_name": target_name,
        "domain": target_domain,
        "skill_path": str(target_md),
        "backup_path": str(backup),
        "consumed_staged": list(staged_ids),
        "n_consumed": removed,
        "word_count": len(proposed_content.split()),
    }


def upgrade_skill_from_staged(
    domain: str,
    staged_ids: List[str],
    *,
    target_domain: str,
    target_name: str,
    llm_call: Callable[[str], str],
    fresh_template: str,
    update_template: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """One-shot upgrade (propose + apply) for non-interactive callers.

    Interactive surfaces should call ``propose_skill_upgrade`` → show the diff →
    ``apply_skill_upgrade`` so a human reviews the merged content. This wrapper
    still backs up the pre-upgrade file (via ``apply_skill_upgrade``).
    """
    prop = propose_skill_upgrade(
        domain, staged_ids, target_domain=target_domain, target_name=target_name,
        llm_call=llm_call, fresh_template=fresh_template,
        update_template=update_template, root=root, skills_root=skills_root)
    if prop.get("status") != "success":
        return prop
    return apply_skill_upgrade(
        domain, prop["staged_ids"], target_domain=target_domain,
        target_name=target_name, proposed_content=prop["proposed_content"],
        root=root, skills_root=skills_root)


# ──────────────────────────────────────────────────────────────
# new-skill@N — consolidate N staged solutions into a NEW skill
# ──────────────────────────────────────────────────────────────

def consolidate_technique(
    domain: str,
    technique: str,
    *,
    llm_call: Callable[[str], str],
    consolidation_template: str,
    update_template: str,
    root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Distill all staged solutions for one technique into a single new skill.

    Builds a multi-example knowledge entry (each example's capped script included
    as input so the LLM generalizes from real code), calls ``graduate_to_skill_file``
    with the consolidation template (skill name ``auto_<technique>``), tags
    ``provenance=t2_consolidated`` + ``n_examples``, and removes the consumed staged
    records. The full verbatim scripts are NOT copied into the skill (kept concise
    and reusable; scripts remain in the staging records).
    """
    recs = list_staged(domain, technique, root=root)
    if not recs:
        return {"status": "error", "message": f"No staged solutions for {domain}/{technique}."}

    knowledge_entry = {
        "technique": technique,
        "n_examples": len(recs),
        "examples": [_record_for_prompt(r) for r in recs],
    }
    skill_name = f"auto_{technique}"
    result = graduate_to_skill_file(
        knowledge_entry=knowledge_entry,
        skill_name=skill_name,
        domain=domain,
        llm_call=llm_call,
        fresh_template=consolidation_template,
        update_template=update_template,
        skills_root=skills_root,
        extra_meta={
            "provisional": True,
            "provenance": "t2_consolidated",
            "n_examples": len(recs),
        },
    )
    if result.get("status") == "success":
        remove_staged(domain, [r["id"] for r in recs if r.get("id")], root=root)
        result["consumed_staged"] = [r.get("id") for r in recs]
        result["n_examples"] = len(recs)
    return result
