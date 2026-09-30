"""Convert a parked ``hitl.FeedbackRequest`` into the payload the React
frontend and the terminal shell render.

A question is presented from what the gate declared: its ``subject`` (what
is under review, as blocks — ``scilink.hitl.SUBJECT_BLOCKS``) and its
``kind`` (the widget and its words, ``vocabulary.QUESTION_WIDGETS`` /
``QUESTION_LABELS``, a gate's own stage overriding the words). The captured
console text still travels as ``context_display``: a request without a
subject (a gate that has not declared one) is shown as its kind's widget
over that text, so nothing is ever sniffed from a prompt string. Figures and
reports in a subject are served relative to the session dir; a code-review
question carries the scripts read from the review folder.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scilink.hitl import SUBJECT_BLOCKS
from scilink.ui import vocabulary as V

_NOTICE_LINE_CHARS = 320      # one change, as shown in the gate's callout
_CANDIDATE_WIDGETS = ("bestofn", "plan_candidates")

logger = logging.getLogger(__name__)


def clean_context(context: str) -> str:
    """The captured console text for display: keep the last ``===``-delimited
    review section, strip separators, collapse blanks, re-add breathing room
    before emoji section headers."""
    display_ctx = context or ""
    lines = display_ctx.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 20) and i + 1 < len(lines) and lines[i + 1].strip():
            start = i
    if start:
        display_ctx = "\n".join(lines[start:])
    display_ctx = re.sub(r"^[=]{10,}\s*$", "", display_ctx, flags=re.MULTILINE)
    display_ctx = re.sub(r"^[ \t]+$", "", display_ctx, flags=re.MULTILINE)
    display_ctx = re.sub(r"\n{2,}", "\n", display_ctx).strip()
    display_ctx = re.sub(
        r"\n(?=[\U0001f300-\U0001fAFF\u2600-\u27BF])",
        "\n\n", display_ctx)
    display_ctx = re.sub(
        r"^(.+(?:PLAN|RESULT|REVIEW).*)$", r"\1\n",
        display_ctx, count=1, flags=re.MULTILINE)
    return display_ctx


# ── session-dir sweeps for the feedback surface ──────────────────

def find_code_review_files(session_dir: str) -> List[Tuple[str, str]]:
    """The scripts a code-review question is about, from the review folder."""
    if not session_dir:
        return []
    candidates = [Path(session_dir) / "temp_code_review",
                  Path(session_dir) / "temp_code_review_iter"]
    existing = [d for d in candidates if d.is_dir()]
    if not existing:
        return []
    review_dir = max(existing, key=lambda d: d.stat().st_mtime)
    files = []
    for p in sorted(review_dir.glob("*.py")):
        try:
            files.append((p.name, p.read_text(encoding="utf-8")))
        except Exception:
            files.append((p.name, "(could not read file)"))
    return files


# ── the presenter ────────────────────────────────────────────────

def _relpath(path: str, session_dir: str) -> Optional[str]:
    """``path`` relative to the session dir (what ``/files`` serves), or
    None when it lies outside it."""
    try:
        return str(Path(path).resolve().relative_to(Path(session_dir).resolve()))
    except (ValueError, OSError):
        return None


def present_subject(subject: Dict[str, Any], session_dir: str) -> Dict[str, Any]:
    """The subject as the front-ends receive it: blocks of an unknown type
    are dropped (logged, never fatal — the console text is still there),
    and every figure gets ``path`` relative to the session dir for the
    ``/files`` endpoint beside ``file``, the absolute path the shell
    prints (``path`` is None for a figure outside the session)."""
    def _figure(path: Any) -> Tuple[Optional[str], Optional[str]]:
        if not path:
            return None, None
        return _relpath(str(path), session_dir), str(path)

    def _blocks(blocks: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind not in SUBJECT_BLOCKS:
                logger.warning("subject block of unknown type %r dropped", kind)
                continue
            b = dict(block)
            if kind == "figure":
                b["path"], b["file"] = _figure(b.get("path"))
            elif kind == "candidates":
                items = []
                for c in b.get("items") or []:
                    c = dict(c)
                    if c.get("figure"):
                        c["figure"], c["figure_file"] = _figure(c["figure"])
                    if c.get("report"):
                        c["report"], c["report_file"] = _figure(c["report"])
                    items.append(c)
                b["items"] = items
            elif kind == "compare":
                for side in ("left", "right"):
                    part = dict(b.get(side) or {})
                    part["blocks"] = _blocks(part.get("blocks"))
                    b[side] = part
            out.append(b)
        return out

    return {"title": str(subject.get("title") or ""),
            "blocks": _blocks(subject.get("blocks"))}


def _candidate_rows(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The picker rows the shell and the legacy web widget read
    (``candidates`` / ``judge_pick``), derived from a candidates block."""
    rows = []
    for c in block.get("items") or []:
        idx = c.get("idx")
        parts = [f"{V.NAMES['candidate']} {idx}"]
        if c.get("name"):
            parts.append(str(c["name"]))
        if c.get("metric") is not None and c.get("value") is not None:
            parts.append(f"{c['metric']}={c['value']}")
        if c.get("approved") is not None:
            parts.append("✓ approved" if c["approved"] else "✗ below gate")
        rows.append({"idx": idx, "label": " — ".join(parts[:2]) + (
            " · " + " · ".join(parts[2:]) if len(parts) > 2 else "")})
    return rows


def _notice(hreq) -> Optional[Dict[str, Any]]:
    """What the decision is ABOUT, beside the buttons that make it: the
    change a revert would undo, or why an approved plan is being reopened.
    Read from the question, not the printed text: a real plan's caveats
    push the repair notice far outside the context tail."""
    if hreq.origin.get("stage") == "plan_review" and hreq.origin.get("auto_repair"):
        changes = [str(c) for c in hreq.origin["auto_repair"]]
        n = len(changes)
        return {
            "title": ("Auto-corrected before review"
                      + (f" ({n} changes)" if n > 1 else "")),
            # A note can quote a whole step. The callout is a reminder beside
            # the button; the full text is in the review above and the report.
            "lines": [c if len(c) <= _NOTICE_LINE_CHARS
                      else c[:_NOTICE_LINE_CHARS].rstrip() + " … (full text above)"
                      for c in changes]}
    if hreq.origin.get("stage") == "plan_reopen":
        return {"title": "The agent proposes to revise a plan you approved",
                "lines": [f"Reason given: {hreq.origin.get('reason') or 'none'}"]}
    return None


def question_asker(origin: Dict[str, Any]) -> str:
    """Who asks, for a question from a concurrent worker (a fan-out branch,
    a swarm item): the label and the subject it works on, so two workers'
    plan approvals do not look identical on screen. Empty for the session's
    own agent."""
    label = origin.get("branch_label")
    if not label:
        return ""
    kind = origin.get("worker_kind") or "branch"
    subject = origin.get("work_subject")
    return f"{kind}: {label}" + (f" · {subject}" if subject else "")


def present_question(hreq, context: str, session_dir: str) -> Dict[str, Any]:
    """Build the presented-question payload for one parked FeedbackRequest.

    ``hreq`` is a ``scilink.hitl.FeedbackRequest``; ``context`` is the
    captured stdout buffer at ask time. The widget and its words come from
    the kind (and the gate's stage), the body from the subject when the gate
    declared one, else from the console text. A picker kind without a
    candidates block is unusable as a picker and is presented as the text
    widget: its reply contract still takes the typed number.
    """
    ctx = context or ""
    ctx_tail = ctx[-1500:]
    stage = str(hreq.origin.get("stage") or "")
    code_files: List[Dict[str, str]] = []
    if (stage == "code_review" or "CODE REVIEW" in ctx_tail
            or "Review files in" in ctx_tail):
        code_files = [{"name": n, "content": c}
                      for n, c in find_code_review_files(session_dir)]
    widget = "code_review" if stage == "code_review" else V.question_widget(hreq.kind)
    labels = V.question_labels(hreq.kind, stage)
    payload: Dict[str, Any] = {
        "request_id": hreq.id,
        "kind": hreq.kind,
        "widget": widget,
        "labels": labels,
        "prompt": hreq.prompt or "",
        "options": list(hreq.options) if hreq.options else None,
        "context_display": clean_context(ctx),
        "code_files": code_files,
        "origin": dict(hreq.origin),
        "default": hreq.default,
    }
    asker = question_asker(hreq.origin)
    if asker:
        payload["asker"] = asker
    subject = present_subject(hreq.subject, session_dir) if hreq.subject else None
    if subject is not None:
        payload["subject"] = subject
    if widget in _CANDIDATE_WIDGETS:
        block = next((b for b in (subject or {}).get("blocks", [])
                      if b["type"] == "candidates"), None)
        rows = _candidate_rows(block) if block else []
        if rows:
            # ``pick`` None means no candidate is preferred (a consensus
            # question): the picker starts unselected and Enter keeps as-is.
            pick = block.get("pick")
            if pick is not None and pick not in [r["idx"] for r in rows]:
                pick = rows[0]["idx"]
            payload["candidates"] = rows
            payload["judge_pick"] = pick
            labels["accept"] = labels["accept"].format(pick=pick)
            if isinstance(block.get("free_text"), dict):
                labels["input"] = str(block["free_text"].get("input") or "")
                labels["submit"] = str(block["free_text"].get("submit") or "Send")
        else:
            logger.warning("%s question without a candidates block: presented "
                           "as text", hreq.kind)
            payload["widget"] = "generic"
            payload["labels"] = V.question_labels("free_text")
    if hreq.origin.get("auto_repair"):
        labels["revert_repair"] = V.REVERT_REPAIR_LABEL
    notice = _notice(hreq)
    if notice:
        payload["notice"] = notice
    return payload
