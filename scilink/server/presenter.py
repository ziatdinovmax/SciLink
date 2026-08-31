"""Convert a parked ``hitl.FeedbackRequest`` into a structured payload the
React frontend renders without any prompt sniffing of its own.

The classifiers and parsers are verbatim ports of the Streamlit widget
chooser (scilink/ui/app.py:1053-1327 and the module-level parse helpers).
They are regex-over-captured-stdout by necessity; any parse miss degrades to
``widget: "generic"`` exactly as the Streamlit UI falls back to its text box.

Widget vocabulary (the ``widget`` field):
  generic | dataset_description | code_review | keep_revert | bestofn |
  plan_candidates | fanout_confirm
``dataset_description`` / ``code_review`` and the plan/extraction variants
share the generic textarea surface and differ only in ``labels``; they get
distinct widget names anyway so the frontend can attach extras (code files).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


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

def _relpaths(paths: List[str], session_dir: str) -> List[str]:
    out = []
    for p in paths:
        try:
            out.append(str(Path(p).resolve().relative_to(Path(session_dir).resolve())))
        except ValueError:
            continue
    return out


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
    if "CODE REVIEW" in ctx_tail or "Review files in" in ctx_tail:
        code_files = [{"name": n, "content": c}
                      for n, c in find_code_review_files(session_dir)]

    is_fanout = (hreq.origin.get("stage") == "fanout_confirm"
                 or "parallel multi-dataset analysis" in ctx.lower())
    is_keep_revert = (
        (hreq.kind == "keep_or_revert" and (hreq.options or [""])[0] == "keep")
        or "revert to original" in ctx_tail.lower())
    bestofn = parse_bestofn_review(ctx, prompt)
    plan_cands = parse_plan_candidate_review(ctx, prompt)

    # Label sets — port of app.py:1165-1185.
    if (hreq.kind == "dataset_description"
            or "Context" in prompt or "MISSING METADATA" in ctx_tail):
        widget = "dataset_description"
        labels = {"input": "Describe your data (optional):",
                  "submit": "Submit description",
                  "accept": "Skip (let agent guess)"}
    elif "CODE REVIEW" in ctx_tail or "Review files in" in ctx_tail:
        widget = "code_review"
        labels = {"input": "Your code feedback (optional):",
                  "submit": "Request changes", "accept": "Approve code"}
    elif "REQUESTING FEEDBACK" in ctx_tail or "Review the plan" in ctx_tail:
        widget = "generic"
        labels = {"input": "Your plan feedback (optional):",
                  "submit": "Request changes", "accept": "Approve plan"}
    elif hreq.kind == "review_metrics" or "SCALARIZER REVIEW" in ctx_tail:
        widget = "generic"
        labels = {"input": "Your extraction feedback (optional):",
                  "submit": "Request changes", "accept": "Approve extraction"}
    else:
        widget = "generic"
        labels = {"input": "Your feedback (optional):",
                  "submit": "Submit feedback", "accept": "Accept as-is"}

    # Specialized surfaces override the labeled-generic classification, in
    # the same precedence order the Streamlit render branch uses.
    payload: Dict[str, Any] = {
        "request_id": hreq.id,
        "kind": hreq.kind,
        "widget": widget,
        "labels": labels,
        "prompt": prompt,
        "context_display": "" if is_fanout else clean_context(ctx),
        "preview_images": _relpaths(preview_images, session_dir),
        "candidate_captions": candidate_captions,
        "code_files": code_files,
        "origin": dict(hreq.origin),
        "default": hreq.default,
    }
    if is_keep_revert:
        payload["widget"] = "keep_revert"
        payload["labels"] = {"keep": "Keep user-guided fit",
                             "revert": "Revert to original fit"}
    elif is_fanout:
        payload["widget"] = "fanout_confirm"
        payload["fanout"] = parse_fanout_confirm(ctx)
        # Response contract of _confirm_fanout: "y" launches, "no" cancels.
        payload["labels"] = {"confirm": "🔀 Launch parallel analysis",
                             "cancel": "Cancel"}
    elif bestofn:
        cands, pick = bestofn
        payload["widget"] = "bestofn"
        payload["candidates"] = cands
        payload["judge_pick"] = pick
        payload["labels"] = {
            "select": "Select the candidate to lock:",
            "use": "Use selected",
            "accept": f"Accept judge's pick (Candidate {pick})"}
    elif plan_cands:
        cands, pick = plan_cands
        payload["widget"] = "plan_candidates"
        payload["candidates"] = cands
        payload["judge_pick"] = pick
        payload["labels"] = {
            "select": "Select the plan candidate to proceed with:",
            "use": "Use selected plan",
            "accept": f"Accept judge's pick (Candidate {pick})"}
    return payload
