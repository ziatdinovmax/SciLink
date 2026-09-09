"""Persistent-memory surface for the web UI — the port of the Streamlit
memory panel (``scilink/ui/components/skills.py``) over the same backend
(``scilink.skills._shared._memory`` / ``_staging`` / ``_script_bank``,
also the backend of ``scilink memory``).

One store per server host (``$SCILINK_HOME`` or ``~/.scilink``): it is not
per session and not per user. The pipeline reads top to bottom in the
order knowledge flows — every success lands in the **script bank**
automatically; nominations, error lessons and user feedback wait in the
**review inbox**; reviewed knowledge is distilled (one large LLM call)
into **skills** that guide planning, provisional until approved.

Everything here is a plain function returning JSON-able dicts so the
endpoints stay thin; the two LLM-backed operations (consolidate into a new
skill, propose an upgrade to an existing one) take one to three minutes,
so they run as background **jobs** polled by id — a web request cannot
block that long the way the Streamlit spinner did.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONSOLIDATED_LABEL_RE = re.compile(r"[^a-z0-9]+")


class MemoryError(Exception):
    """A user-facing failure with an HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _mods():
    from scilink.skills import loader
    from scilink.skills._shared import _memory, _script_bank, _staging
    return loader, _memory, _script_bank, _staging


def normalize_label(label: str) -> str:
    return _CONSOLIDATED_LABEL_RE.sub("_", (label or "").lower()).strip("_")[:48]


# ── overview ─────────────────────────────────────────────────────────

def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one bad store must not break the panel
        logger.warning(f"memory panel: {fn.__name__ if hasattr(fn, '__name__') else fn} failed: {exc}")
        return default


def _inbox_row(rec: Dict[str, Any], _staging) -> Dict[str, Any]:
    prov = str(rec.get("provenance") or "")
    return {
        "id": rec.get("id"),
        "domain": rec.get("domain"),
        "technique": rec.get("technique") or "unlabeled",
        "provenance": prov,
        "provenance_label": _staging.PROVENANCE_LABELS.get(prov, prov or "?"),
        "metric": _safe(lambda: _staging.metric_label(rec), ""),
        "session": rec.get("session"),
        "bank_id": rec.get("bank_id"),
        "model": rec.get("model") or rec.get("planned_model"),
        "has_script": bool((rec.get("working_script") or rec.get("script") or "").strip()),
    }


def memory_overview() -> Dict[str, Any]:
    """Everything the panel shows at once: the switch, the pipeline strip,
    the bank by domain (with variant-group suggestions), the inbox by
    domain / technique, the skills. Never raises for a bad record."""
    loader, _memory, _script_bank, _staging = _mods()
    enabled = loader.memory_enabled()
    env = os.environ.get("SCILINK_MEMORY", "").strip()

    skills = _safe(_memory.list_memory, [])
    bank_rows = _safe(_script_bank.bank_summary, [])
    staged = _safe(_staging.list_staged, [])
    need = _safe(_staging.consolidate_min_n, 2)
    proven_n = _safe(_script_bank.proven_n, 2)

    bank: List[Dict[str, Any]] = []
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in bank_rows:
        by_domain.setdefault(r.get("domain") or "?", []).append(r)
    for domain, recs in sorted(by_domain.items()):
        groups = _safe(lambda d=domain: [g for g in _script_bank.find_variant_groups(d)
                                          if g.get("n_unpromoted", 0) > 0], [])
        bank.append({
            "domain": domain,
            "n_proven": sum(1 for r in recs if r.get("proven")),
            "records": [{
                "id": r.get("id"), "label": r.get("label"),
                "n_successes": r.get("n_successes"), "n_retrievals": r.get("n_retrievals"),
                "sessions": r.get("sessions") or [], "metric": r.get("metric"),
                "created_at": r.get("created_at"), "proven": bool(r.get("proven")),
                "promoted_to_staging": r.get("promoted_to_staging"),
            } for r in recs],
            "variant_groups": [{
                "ids": g.get("ids") or [], "min_similarity": g.get("min_similarity"),
                "suggested_technique": g.get("suggested_technique"),
                "n_unpromoted": g.get("n_unpromoted"),
            } for g in groups],
        })

    inbox_groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for rec in staged:
        if not isinstance(rec, dict):
            continue
        key = (rec.get("domain") or "?", rec.get("technique") or "unlabeled")
        inbox_groups.setdefault(key, []).append(rec)
    inbox: List[Dict[str, Any]] = []
    for (domain, technique), recs in sorted(inbox_groups.items()):
        # Same-session records filed under other labels (the lessons these
        # very analyses produced) are offered alongside the group.
        sessions = {r.get("session") for r in recs if r.get("session")}
        related = [o for o in staged if isinstance(o, dict)
                   and o.get("domain") == domain
                   and (o.get("technique") or "unlabeled") != technique
                   and o.get("session") in sessions]
        inbox.append({
            "domain": domain, "technique": technique,
            "ready": len(recs) >= need,
            "records": [_inbox_row(r, _staging) for r in recs],
            "related": [_inbox_row(r, _staging) for r in related],
        })

    n_ready = sum(1 for g in inbox if g["ready"])
    return {
        "enabled": enabled,
        "env_override": env or None,
        "home": _tilde(loader.scilink_home()),
        "consolidate_min_n": need,
        "proven_n": proven_n,
        "pipeline": {
            "bank_total": len(bank_rows),
            "bank_proven": sum(1 for r in bank_rows if r.get("proven")),
            "inbox_total": len(staged),
            "inbox_ready": n_ready,
            "skills_total": len(skills),
            "skills_provisional": sum(1 for s in skills if s.get("provisional")),
        },
        "bank": bank,
        "inbox": inbox,
        "skills": [{
            "name": s.get("name"), "domain": s.get("domain"), "path": s.get("path"),
            "provisional": bool(s.get("provisional")), "provenance": s.get("provenance"),
            "session": s.get("session"), "description": s.get("description") or "",
            "metric": _safe(lambda s=s: _staging.metric_label(s), ""),
            "shadows_builtin": _shadows_builtin(s.get("domain"), s.get("name")),
        } for s in skills],
    }


def _tilde(path) -> str:
    """``~/.scilink`` rather than the absolute home path, for display."""
    s = str(path)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home) else s


def _shadows_builtin(domain: Optional[str], name: Optional[str]) -> bool:
    try:
        from scilink.skills.loader import _SKILLS_DIR
        return bool(domain and name and (_SKILLS_DIR / domain / name / f"{name}.md").is_file())
    except Exception:  # noqa: BLE001
        return False


def set_enabled(enabled: bool) -> Dict[str, Any]:
    loader, *_ = _mods()
    loader.set_memory_enabled(bool(enabled))
    return {"enabled": loader.memory_enabled(),
            "env_override": os.environ.get("SCILINK_MEMORY", "").strip() or None}


# ── skills (stage 3) ─────────────────────────────────────────────────

def skill_text(domain: str, name: str) -> str:
    _, _memory, *_ = _mods()
    try:
        return _memory.show_memory(domain, name)
    except FileNotFoundError as exc:
        raise MemoryError(404, str(exc)) from exc


def skill_edit(domain: str, name: str, content: str) -> Dict[str, Any]:
    _, _memory, *_ = _mods()
    try:
        out = _memory.edit_memory(domain, name, content)
    except FileNotFoundError as exc:
        raise MemoryError(404, str(exc)) from exc
    if out.get("status") != "success":
        raise MemoryError(400, out.get("message") or "Save failed.")
    return out


def skill_action(domain: str, name: str, action: str) -> Dict[str, Any]:
    """promote (approve for routing) · demote (suspend) · prune (delete) ·
    diff (against the shipped built-in, for a fork)."""
    _, _memory, *_ = _mods()
    fn = {"promote": _memory.promote_memory, "demote": _memory.demote_memory,
          "prune": _memory.prune_memory, "diff": _memory.diff_builtin}.get(action)
    if fn is None:
        raise MemoryError(400, f"Unknown skill action {action!r}.")
    try:
        out = fn(domain, name)
    except FileNotFoundError as exc:
        raise MemoryError(404, str(exc)) from exc
    if isinstance(out, dict) and out.get("status") == "error":
        raise MemoryError(400, out.get("message") or f"{action} failed.")
    return out


def fork_builtin(domain: str, name: str) -> Dict[str, Any]:
    _, _memory, *_ = _mods()
    try:
        out = _memory.fork_builtin(domain, name)
    except FileNotFoundError as exc:
        raise MemoryError(404, str(exc)) from exc
    if out.get("status") != "success":
        raise MemoryError(409, out.get("message") or "Fork failed.")
    return out


# ── script bank (stage 1) ────────────────────────────────────────────

_BANK_HIDDEN = {"id", "domain", "script_hash", "working_script"}


def bank_record(domain: str, rid: str) -> Dict[str, Any]:
    _, _, _script_bank, _ = _mods()
    rec = _script_bank.get_record(domain, rid)
    if rec is None:
        raise MemoryError(404, f"No bank record {domain}/{rid}.")
    fields = {k: v for k, v in rec.items()
              if k not in _BANK_HIDDEN and v not in (None, "", [], {})}
    return {"id": rid, "domain": domain, "fields": fields,
            "script": rec.get("working_script") or ""}


def bank_nominate(domain: str, rid: str) -> Dict[str, Any]:
    _, _, _script_bank, _ = _mods()
    out = _script_bank.promote_to_staging(domain, rid)
    if out.get("status") != "success":
        raise MemoryError(409, out.get("message") or "Nomination failed.")
    return out


def bank_nominate_group(domain: str, ids: List[str], technique: Optional[str]) -> Dict[str, Any]:
    _, _, _script_bank, _ = _mods()
    out = _script_bank.promote_group_to_staging(domain, list(ids), technique=technique or None)
    if out.get("status") != "success":
        raise MemoryError(409, out.get("message") or "Group nomination failed.")
    return out


def bank_delete(domain: str, rid: str) -> Dict[str, Any]:
    _, _, _script_bank, _ = _mods()
    n = _script_bank.remove_records(domain, [rid])
    if not n:
        raise MemoryError(404, f"No bank record {domain}/{rid}.")
    return {"status": "success", "removed": n}


# ── review inbox (stage 2) ───────────────────────────────────────────

_STAGED_HIDDEN = {"id", "domain", "technique", "session", "working_script", "script"}


def inbox_record(domain: str, sid: str) -> Dict[str, Any]:
    _, _, _script_bank, _staging = _mods()
    rec = _staging.get_staged(domain, sid)
    if rec is None:
        raise MemoryError(404, f"No staged record {domain}/{sid}.")
    fields = {k: v for k, v in rec.items()
              if k not in _STAGED_HIDDEN and v not in (None, "", [], {})}
    bank_link = None
    if rec.get("bank_id"):
        brec = _safe(lambda: _script_bank.get_record(domain, rec["bank_id"]), None)
        n_succ = ((brec or {}).get("stats") or {}).get("n_successes")
        bank_link = {"bank_id": rec["bank_id"], "n_successes": n_succ}
    return {**_inbox_row(rec, _staging), "fields": fields,
            "script": rec.get("working_script") or rec.get("script") or "",
            "bank": bank_link}


def inbox_discard(domain: str, sid: str) -> Dict[str, Any]:
    _, _, _, _staging = _mods()
    n = _staging.remove_staged(domain, [sid])
    if not n:
        raise MemoryError(404, f"No staged record {domain}/{sid}.")
    return {"status": "success", "removed": n}


# Technique-match heuristic (ported from the Streamlit panel): does a
# record's context plausibly belong to a skill, judged on the skill's
# `technique:` routing tokens? Flags a mismatched upgrade BEFORE the
# expensive preview call.
_GENERIC_TECH_TOKENS = {"spectroscopy", "spectrometry", "spectrum", "spectra",
                        "microscopy", "micro", "imaging", "absorption",
                        "emission", "analysis", "scattering"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _skill_tokens(domain: str, name: str):
    tech_tokens, all_tokens = set(), set()
    try:
        loader, *_ = _mods()
        meta = (loader.load_skill(name, domain=domain) or {}).get("meta") or {}
        tech = meta.get("technique") or []
        if isinstance(tech, str):
            tech = [tech]
        for t in tech:
            tech_tokens |= {w for w in _TOKEN_RE.findall(str(t).lower())
                            if len(w) >= 3 and w not in _GENERIC_TECH_TOKENS}
        for s in [name, meta.get("description") or ""]:
            all_tokens |= {w for w in _TOKEN_RE.findall(str(s).lower()) if len(w) >= 3}
    except Exception:  # noqa: BLE001
        pass
    return tech_tokens, all_tokens | tech_tokens


def _record_tokens(rec: Dict[str, Any]) -> set:
    ctx = rec.get("measurement_context") or {}
    fields = (list(ctx.values()) if isinstance(ctx, dict) else []) + [
        rec.get("model"), rec.get("technique"), rec.get("planned_model"),
        rec.get("final_model_type"), rec.get("analysis_target")]
    toks: set = set()
    for v in fields:
        toks |= {w for w in _TOKEN_RE.findall(str(v or "").lower()) if len(w) >= 3}
    return toks


def _target_matches(recs: List[Dict[str, Any]], domain: str, name: str) -> Optional[bool]:
    tech_tokens, all_tokens = _skill_tokens(domain, name)
    votes = []
    for rec in recs:
        rt = _record_tokens(rec)
        if not rt:
            votes.append(None)
        elif tech_tokens:
            votes.append(bool(rt & tech_tokens))
        elif all_tokens:
            votes.append(bool(rt & all_tokens) or None)
        else:
            votes.append(None)
    if any(v is True for v in votes):
        return True
    if votes and all(v is False for v in votes):
        return False
    return None


def upgrade_targets(domain: str, ids: List[str]) -> List[Dict[str, Any]]:
    """Skills the selected records could upgrade — persistent ones in this
    domain, then built-ins (which fork on upgrade) — each with a
    true / false / null technique-match verdict, matches first."""
    loader, _memory, _, _staging = _mods()
    recs = [r for r in (_staging.get_staged(domain, i) for i in ids) if r]
    persistent = {s["name"] for s in _safe(lambda: _memory.list_memory(domain=domain), [])}
    builtin = [n for n in _safe(lambda: loader.list_skills(domain), []) if n not in persistent]
    out = []
    for n in sorted(persistent):
        out.append({"domain": domain, "name": n, "builtin": False,
                    "match": _target_matches(recs, domain, n)})
    for n in sorted(builtin):
        out.append({"domain": domain, "name": n, "builtin": True,
                    "match": _target_matches(recs, domain, n)})
    out.sort(key=lambda t: {True: 0, None: 1, False: 2}[t["match"]])
    return out


def upgrade_check(existing: str, proposed: str) -> Dict[str, Any]:
    """Additivity warnings + unified diff for a (possibly edited) proposal."""
    from scilink.skills._shared._staging import _regression_warnings
    diff = "\n".join(difflib.unified_diff(
        (existing or "").splitlines(), (proposed or "").splitlines(),
        fromfile="current", tofile="after upgrade", lineterm=""))
    return {"warnings": list(_regression_warnings(existing or "", proposed or "")),
            "diff": diff}


# ── LLM-backed jobs ──────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def llm_call_for(agent: Any) -> Callable[[str], str]:
    """The session's own model as a plain prompt → text callable (the same
    thing the Streamlit panel and `scilink memory` build)."""
    model = getattr(agent, "model", None)
    if model is None or not hasattr(model, "generate_content"):
        raise MemoryError(400, "This session's agent has no model to distill with.")

    def _call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)
    return _call


def _start_job(kind: str, fn: Callable[[], Dict[str, Any]], label: str) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "label": label, "status": "running",
           "result": None, "error": None}
    with _jobs_lock:
        _jobs[job_id] = job

    def _run():
        try:
            res = fn()
            if isinstance(res, dict) and res.get("status") == "error":
                job["status"] = "error"
                job["error"] = res.get("message") or f"{kind} failed."
                job["result"] = res
            else:
                job["status"] = "done"
                job["result"] = res
        except Exception as exc:  # noqa: BLE001 - LLM output is untrusted
            logger.exception(f"memory job {kind} failed")
            job["status"] = "error"
            job["error"] = str(exc)
    threading.Thread(target=_run, name=f"memory-{kind}-{job_id}", daemon=True).start()
    return {"job_id": job_id, "status": "running", "label": label}


def job_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise MemoryError(404, f"No memory job {job_id}.")
    return dict(job)


def start_consolidate(domain: str, ids: List[str], label: str,
                      llm_call: Callable[[str], str]) -> Dict[str, Any]:
    """Distill the selected records into a NEW provisional skill
    ``auto_<label>``. The records are relabeled under ``label`` first (the
    consolidation reads a whole technique group), so other records already
    carrying that label would be swept in — refused unless selected."""
    loader, _, _, _staging = _mods()
    if not loader.memory_enabled():
        raise MemoryError(400, "Persistent memory is off — turn it on to distill.")
    norm = normalize_label(label)
    if not norm:
        raise MemoryError(400, "Empty skill name.")
    need = _staging.consolidate_min_n()
    if len(ids) < need:
        raise MemoryError(400, f"Needs at least {need} selected records (one example "
                               "is too idiosyncratic to generalize).")
    swept = [r["id"] for r in _staging.list_staged(domain)
             if (r.get("technique") or "") == norm and r.get("id") not in ids]
    if swept:
        raise MemoryError(409, f"'{norm}' is also the label of {len(swept)} unselected "
                               f"record(s) ({', '.join(swept)}) — they would be included. "
                               "Select them or rename.")
    for rid in ids:
        if _staging.get_staged(domain, rid) is None:
            raise MemoryError(404, f"No staged record {domain}/{rid}.")
    from scilink.agents.exp_agents.instruct import (
        SKILL_UPDATE_INSTRUCTIONS, T2_CONSOLIDATION_INSTRUCTIONS)

    def _work():
        for rid in ids:
            _staging.relabel_staged(domain, rid, norm)
        res = _staging.consolidate_technique(
            domain, norm, llm_call=llm_call,
            consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
            update_template=SKILL_UPDATE_INSTRUCTIONS)
        if res.get("status") == "success":
            res = {**res, "skill_name": f"auto_{norm}", "domain": domain}
        return res
    return _start_job("consolidate", _work, f"{len(ids)} → auto_{norm}")


def start_propose_upgrade(domain: str, ids: List[str], target_domain: str,
                          target_name: str, llm_call: Callable[[str], str]) -> Dict[str, Any]:
    """Build the merged skill for review WITHOUT writing it. A built-in
    target is previewed against a temporary copy; applying forks it."""
    loader, _memory, _, _staging = _mods()
    if not loader.memory_enabled():
        raise MemoryError(400, "Persistent memory is off — turn it on to distill.")
    if not ids:
        raise MemoryError(400, "Select at least one record.")
    recs = [r for r in (_staging.get_staged(domain, i) for i in ids) if r]
    if len(recs) != len(ids):
        raise MemoryError(404, "One or more selected records no longer exist.")
    persistent = {s["name"] for s in _safe(lambda: _memory.list_memory(domain=target_domain), [])}
    builtin_target = target_name not in persistent
    from scilink.agents.exp_agents.instruct import (
        KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
    from scilink.skills.loader import _SKILLS_DIR
    match = _target_matches(recs, target_domain, target_name)

    def _work():
        skills_root = None
        tmp_root = None
        if builtin_target:
            src = _SKILLS_DIR / target_domain / target_name / f"{target_name}.md"
            if not src.is_file():
                return {"status": "error",
                        "message": f"No skill {target_domain}/{target_name} (persistent or built-in)."}
            tmp_root = Path(tempfile.mkdtemp(prefix="scilink_preview_"))
            (tmp_root / target_domain / target_name).mkdir(parents=True)
            shutil.copy(src, tmp_root / target_domain / target_name / f"{target_name}.md")
            skills_root = tmp_root
        try:
            prop = _staging.propose_skill_upgrade(
                domain, list(ids), target_domain=target_domain, target_name=target_name,
                llm_call=llm_call, fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                update_template=SKILL_UPDATE_INSTRUCTIONS, skills_root=skills_root)
        finally:
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)
        if prop.get("status") != "success":
            return prop
        check = upgrade_check(prop["existing_content"], prop["proposed_content"])
        warnings = check["warnings"] + list(prop.get("regression_warnings") or [])
        if match is False:
            warnings.append("technique mismatch: the selection's context does not match "
                            "this skill's routing — confirm this upgrade belongs here")
        return {
            "status": "success", "domain": domain, "staged_ids": list(ids),
            "target_domain": target_domain, "target_name": target_name,
            "builtin_target": builtin_target,
            "existing_content": prop["existing_content"],
            "proposed_content": prop["proposed_content"],
            "warnings": warnings, "diff": check["diff"],
        }
    return _start_job("upgrade", _work, f"{len(ids)} → {target_domain}/{target_name}")


def apply_upgrade(domain: str, ids: List[str], target_domain: str, target_name: str,
                  content: str, fork_first: bool) -> Dict[str, Any]:
    """Write a reviewed (possibly edited) proposal: fork the built-in first
    when the target was one, back up the current file, write, consume the
    records."""
    _, _memory, _, _staging = _mods()
    if not (content or "").strip():
        raise MemoryError(400, "Empty skill content.")
    if fork_first:
        out = _memory.fork_builtin(target_domain, target_name)
        if out.get("status") != "success" and "already forked" not in str(out.get("message")):
            raise MemoryError(409, out.get("message") or "Fork failed.")
    res = _staging.apply_skill_upgrade(
        domain, list(ids), target_domain=target_domain, target_name=target_name,
        proposed_content=content if content.endswith("\n") else content + "\n")
    if res.get("status") != "success":
        raise MemoryError(400, res.get("message") or "Apply failed.")
    return res
