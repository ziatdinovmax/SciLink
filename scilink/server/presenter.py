"""Convert a parked ``hitl.FeedbackRequest`` into a structured payload the
React frontend and the terminal shell render without any prompt sniffing of
their own.

Two paths. A request that carries a ``subject`` (what is under review, as
blocks — ``scilink.hitl.SUBJECT_BLOCKS``) is presented from that and from
its ``kind`` (the widget and its words, ``vocabulary.QUESTION_WIDGETS`` /
``QUESTION_LABELS``); the captured console text still travels as
``context_display`` but the surfaces show only the blocks. A request without one
is presented from the captured console text: the classifiers and parsers
below are verbatim ports of the Streamlit widget chooser
(scilink/ui/app.py:1053-1327 and the module-level parse helpers), regex over
captured stdout by necessity; any parse miss degrades to ``widget:
"generic"`` exactly as the Streamlit UI falls back to its text box. Gates
move from the second path to the first one at a time.

Widget vocabulary (the ``widget`` field):
  generic | dataset_description | code_review | keep_revert | bestofn |
  plan_candidates | fanout_confirm
``dataset_description`` / ``code_review`` and the plan/extraction variants
share the generic textarea surface and differ only in ``labels``; they get
distinct widget names anyway so the frontend can attach extras (code files).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scilink.hitl import SUBJECT_BLOCKS
from scilink.ui import vocabulary as V

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_NOTICE_LINE_CHARS = 320      # one change, as shown in the gate's callout
_CANDIDATE_WIDGETS = ("bestofn", "plan_candidates")

logger = logging.getLogger(__name__)


# ── parsers (ported from scilink/ui/app.py) ──────────────────────

def parse_bestofn_review(context: str, prompt: str
                         ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Port of app.py:428 ``_parse_bestofn_review`` — unchanged logic."""
    if not prompt or "accept candidate" not in prompt:
        return None
    if not context or "BEST-OF-N CANDIDATES" not in context:
        return None
    block = context[context.rfind("BEST-OF-N CANDIDATES"):]
    cands: Dict[int, str] = {}
    pick = None
    for m in re.finditer(
        r"Candidate (\d+):\s*([^=]+)=([0-9.eE+\-]+),\s*approved=(\w+),"
        r"\s*iterations=(\d+)(.*)", block):
        idx = int(m.group(1))
        metric, value = m.group(2).strip(), m.group(3)
        approved = m.group(4).lower() == "true"
        iters = m.group(5)
        mark = "✓ approved" if approved else "✗ below gate"
        cands[idx] = f"Candidate {idx} — {metric}={value} · {mark} · {iters} iter"
        if "judge pick" in m.group(6).lower():
            pick = idx
    if not cands:
        return None
    if pick is None:
        pm = re.search(r"accept candidate (\d+)", prompt or "")
        pick = int(pm.group(1)) if pm else min(cands)
    ordered = [{"idx": i, "label": cands[i]} for i in sorted(cands)]
    return ordered, pick


def parse_plan_candidate_review(context: str, prompt: str
                                ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Port of app.py:473 ``_parse_plan_candidate_review`` — unchanged logic."""
    if not prompt or "accept plan candidate" not in prompt:
        return None
    if not context or "PLAN CANDIDATES" not in context:
        return None
    block = context[context.rfind("PLAN CANDIDATES"):]
    cands: Dict[int, str] = {}
    pick = None
    for m in re.finditer(r"── Candidate (\d+): (.+?) ──(.*)", block):
        idx = int(m.group(1))
        name = m.group(2).strip()
        if len(name) > 120:
            name = name[:117] + "…"
        cands[idx] = f"Candidate {idx} — {name}"
        if "judge pick" in m.group(3).lower():
            pick = idx
    if not cands:
        return None
    if pick is None:
        pm = re.search(r"accept plan candidate (\d+)", prompt)
        pick = int(pm.group(1)) if pm else min(cands)
    ordered = [{"idx": i, "label": cands[i]} for i in sorted(cands)]
    return ordered, pick


def parse_fanout_confirm(ctx: str) -> Dict[str, Any]:
    """Port of app.py:106 ``_render_fanout_confirm``'s extraction."""
    def _g(pat):
        m = re.search(pat, ctx or "")
        return re.sub(r"\s{2,}", " ", m.group(1).strip()) if m else None
    return {
        "verdict": _g(r"Complementarity verdict\s*:\s*(.+)"),
        "join_axis": _g(r"Join axis\s*:\s*(.+)"),
        "rationale": _g(r"Rationale\s*:\s*(.+)"),
        "branches": [re.sub(r"\s{2,}", " ", b.strip())
                     for b in re.findall(r"•\s*(.+)", ctx or "")],
    }


def clean_context(context: str) -> str:
    """Port of the context-box cleanup (app.py:1099-1132): keep the last
    ``===``-delimited review section, strip separators, collapse blanks,
    re-add breathing room before emoji section headers."""
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

def find_feedback_preview_images(session_dir: str) -> List[str]:
    """Port of app.py:232 ``_find_feedback_preview_images``."""
    if not session_dir:
        return []
    search_root = Path(session_dir)
    results_dir = search_root / "results"
    if results_dir.exists():
        analysis_dirs = sorted(
            [d for d in results_dir.iterdir()
             if d.is_dir() and d.name.startswith("analysis_")],
            key=lambda d: d.stat().st_mtime, reverse=True)
        if analysis_dirs:
            search_root = analysis_dirs[0]
    previews: List[str] = []
    for ext in _IMAGE_EXTENSIONS:
        for p in search_root.rglob(f"*{ext}"):
            if "review" in p.stem or "Summary_Grid" in p.stem:
                previews.append(str(p))
    scalarizer_dir = Path(session_dir) / "scalarizer_outputs"
    if scalarizer_dir.exists():
        for ext in _IMAGE_EXTENSIONS:
            for p in scalarizer_dir.glob(f"debug_*{ext}"):
                s = str(p)
                if s not in previews:
                    previews.append(s)
    return previews


def find_code_review_files(session_dir: str) -> List[Tuple[str, str]]:
    """Port of app.py:272 ``_find_code_review_files``."""
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


def _relpaths(paths: List[str], session_dir: str) -> List[str]:
    out = []
    for p in paths:
        rel = _relpath(p, session_dir)
        if rel is not None:
            out.append(rel)
    return out


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


def _present_from_subject(hreq, ctx: str, session_dir: str,
                          code_files: List[Dict[str, str]]) -> Dict[str, Any]:
    """The subject path: blocks from the gate, widget and words from the
    kind. No preview sweep — a gate with a subject declares its figures."""
    stage = str(hreq.origin.get("stage") or "")
    widget = "code_review" if stage == "code_review" else V.question_widget(hreq.kind)
    labels = V.question_labels(hreq.kind, stage)
    subject = present_subject(hreq.subject, session_dir)
    payload: Dict[str, Any] = {
        "request_id": hreq.id,
        "kind": hreq.kind,
        "widget": widget,
        "labels": labels,
        "prompt": hreq.prompt or "",
        "context_display": clean_context(ctx),
        "preview_images": [],
        "candidate_captions": {},
        "code_files": code_files,
        "origin": dict(hreq.origin),
        "options": list(hreq.options) if hreq.options else None,
        "default": hreq.default,
        "subject": subject,
    }
    if widget in _CANDIDATE_WIDGETS:
        block = next((b for b in subject["blocks"] if b["type"] == "candidates"), None)
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
            # A picker with nothing to pick from is unusable; the text
            # widget still takes the number the gate's reply contract reads.
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


def present_question(hreq, context: str, session_dir: str) -> Dict[str, Any]:
    """Build the PresentedQuestion payload for one parked FeedbackRequest.

    ``hreq`` is a ``scilink.hitl.FeedbackRequest``; ``context`` is the
    captured stdout buffer at ask time. Image paths come back relative to the
    session dir for the ``/files`` endpoint. The classifier order matches the
    Streamlit widget chooser exactly (keep_revert → fanout → bestofn →
    plan_candidates → labeled generic).
    """
    prompt = hreq.prompt or ""
    ctx = context or ""
    ctx_tail = ctx[-1500:]

    preview_images = find_feedback_preview_images(session_dir)
    candidate_captions = {}
    for img in preview_images:
        m = re.search(r"bestofn_candidate_(\d+)_review", Path(img).name)
        if m:
            candidate_captions[str(Path(img).name)] = f"Candidate {int(m.group(1))}"

    code_files: List[Dict[str, str]] = []
    if ("CODE REVIEW" in ctx_tail or "Review files in" in ctx_tail
            or hreq.origin.get("stage") == "code_review"):
        code_files = [{"name": n, "content": c}
                      for n, c in find_code_review_files(session_dir)]

    if hreq.subject:
        return _present_from_subject(hreq, ctx, session_dir, code_files)

    is_fanout = (hreq.origin.get("stage") == "fanout_confirm"
                 or "parallel multi-dataset analysis" in ctx.lower())
    is_keep_revert = (
        (hreq.kind == "keep_or_revert" and (hreq.options or [""])[0] == "keep")
        or "revert to original" in ctx_tail.lower())
    bestofn = parse_bestofn_review(ctx, prompt)
    plan_cands = parse_plan_candidate_review(ctx, prompt)

    # Label sets — port of app.py:1165-1185, words from the vocabulary.
    if (hreq.kind == "dataset_description"
            or "Context" in prompt or "MISSING METADATA" in ctx_tail):
        widget = "dataset_description"
        labels = V.question_labels("dataset_description")
    elif "CODE REVIEW" in ctx_tail or "Review files in" in ctx_tail:
        widget = "code_review"
        labels = V.question_labels("code_review")
    elif "REQUESTING FEEDBACK" in ctx_tail or "Review the plan" in ctx_tail:
        widget = "generic"
        labels = V.question_labels("review_plan")
        if hreq.origin.get("auto_repair"):
            # The plan on screen was repaired automatically; one click
            # restores it as authored (the console reply is "revert").
            labels["revert_repair"] = V.REVERT_REPAIR_LABEL
    elif hreq.kind == "review_metrics" or "SCALARIZER REVIEW" in ctx_tail:
        widget = "generic"
        labels = V.question_labels("review_metrics")
    else:
        widget = "generic"
        labels = V.question_labels("free_text")

    # Specialized surfaces override the labeled-generic classification, in
    # the same precedence order the Streamlit render branch uses.
    payload: Dict[str, Any] = {
        "request_id": hreq.id,
        "kind": hreq.kind,
        "widget": widget,
        "labels": labels,
        "prompt": prompt,
        "options": list(hreq.options) if hreq.options else None,
        "context_display": "" if is_fanout else clean_context(ctx),
        "preview_images": _relpaths(preview_images, session_dir),
        "candidate_captions": candidate_captions,
        "code_files": code_files,
        "origin": dict(hreq.origin),
        "default": hreq.default,
    }
    notice = _notice(hreq)
    if notice:
        payload["notice"] = notice
    if is_keep_revert:
        payload["widget"] = "keep_revert"
        # The reopen gate: same two-way widget, the plan gate's words — the
        # primary reply ("keep") adopts the agent's revision, the empty one
        # keeps the plan the human already approved; "input"/"submit" add
        # the third reply the console offers (adopt with changes, any text).
        payload["labels"] = V.question_labels(
            "keep_or_revert", str(hreq.origin.get("stage") or ""))
    elif is_fanout:
        payload["widget"] = "fanout_confirm"
        payload["fanout"] = parse_fanout_confirm(ctx)
        # Response contract of _confirm_fanout: "y" launches, "no" cancels.
        payload["labels"] = V.question_labels("confirm", "fanout_confirm")
    elif bestofn:
        cands, pick = bestofn
        payload["widget"] = "bestofn"
        payload["candidates"] = cands
        payload["judge_pick"] = pick
        payload["labels"] = V.question_labels("bestofn_select")
        payload["labels"]["accept"] = payload["labels"]["accept"].format(pick=pick)
    elif plan_cands:
        cands, pick = plan_cands
        payload["widget"] = "plan_candidates"
        payload["candidates"] = cands
        payload["judge_pick"] = pick
        payload["labels"] = V.question_labels("plan_candidate_select")
        payload["labels"]["accept"] = payload["labels"]["accept"].format(pick=pick)
    return payload
