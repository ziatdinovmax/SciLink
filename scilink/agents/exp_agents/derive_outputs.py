"""Derive a secondary artifact from a completed run's outputs (#599).

The lightweight compute step between ``save_file`` (text the model already
holds) and ``run_analysis`` (a full, re-planned analysis): a short sandboxed
script that READS a prior run's saved files — arrays, tables, JSON,
figures — computes the requested product (reformat, subset, aggregate,
convert units, join, re-render, extract a scalar) and WRITES it under a
fresh output directory. No skill selection, no planning, no best-of-N.

Shape (mirrors data_preparation): inventory the artifacts → the model writes
(or the caller supplies) the script → static guard → sandbox run → the
script prints a ``DERIVE_RESULT_JSON:`` line naming its products → a
deterministic gate (every product exists under the output dir, is
non-empty, and was written by this run) → retry with the failure as
feedback. The approved script is kept as ``scripts/derive_script.py``.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .data_preparation import _DENIED_CALLS, _DENIED_IMPORTS, _llm_verdict, extract_script

RESULT_MARKER = "DERIVE_RESULT_JSON:"
_MAX_INVENTORY_FILES = 120
_MAX_INVENTORY_CHARS = 12000
_SKIP_DIRS = {"_scratch", "_candidates", "__pycache__", "norm"}


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------
def json_outline(obj: Any, depth: int = 4, max_keys: int = 10, max_chars: int = 700) -> str:
    """A compact STRUCTURAL outline of a JSON value — key names with value
    types, list lengths and the first element's shape — so the model can
    locate nested quantities (``results[i].parameters.peak_1.amplitude``)
    instead of guessing a layout from top-level keys alone."""
    def _o(v, d):
        if isinstance(v, dict):
            if d <= 0:
                return "{…}"
            items = list(v.items())
            body = ", ".join(f"{k}: {_o(val, d - 1)}" for k, val in items[:max_keys])
            return "{" + body + (f", …+{len(items) - max_keys}" if len(items) > max_keys else "") + "}"
        if isinstance(v, list):
            if not v:
                return "[]"
            return f"[{_o(v[0], d)} ×{len(v)}]"   # a list is transparent: its elements share its depth
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, (int, float)):
            return "null" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
        if v is None:
            return "null"
        if isinstance(v, str):
            return "str" if len(v) > 24 else repr(v)
        return type(v).__name__
    text = _o(obj, depth)
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _describe_file(path: Path) -> str:
    """One line: what a saved artifact holds (shape / columns / keys)."""
    suffix = path.suffix.lower()
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    info = f"{size} B"
    try:
        if suffix == ".npy":
            import numpy as np
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            info = f"npy shape={tuple(arr.shape)} dtype={arr.dtype}"
        elif suffix == ".npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                info = "npz keys=" + ", ".join(f"{k}{tuple(z[k].shape)}" for k in list(z.files)[:12])
        elif suffix in (".csv", ".tsv"):
            import pandas as pd
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=5)
            n_rows = sum(1 for _ in open(path, "rb")) - 1
            cols = list(df.columns)
            info = f"table rows={max(n_rows, 0)} columns={cols[:20]}" + (" …" if len(cols) > 20 else "")
            if len(df):
                info += " first_row=" + json.dumps({c: (None if pd.isna(x) else (round(float(x), 4) if isinstance(x, (int, float)) else str(x)[:20])) for c, x in df.iloc[0].items()})[:300]
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace")[:2_000_000])
            info = "json " + json_outline(data)
        elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            info = f"image {size} B"
        elif suffix in (".txt", ".md", ".log", ".py"):
            info = f"text {size} B"
    except Exception as e:  # noqa: BLE001 - the inventory never fails on one file
        info = f"{info} (unreadable: {type(e).__name__})"
    return info


def build_artifact_inventory(sources: list[Path], max_files: int = _MAX_INVENTORY_FILES) -> tuple[str, list[str]]:
    """``(text, files)``: a prompt-ready listing of every artifact under the
    sources (directories are walked; scratch / candidate folders skipped)
    and the absolute file list the script may read."""
    files: list[Path] = []
    for src in sources:
        src = Path(src)
        if src.is_file():
            files.append(src)
        elif src.is_dir():
            for p in sorted(src.rglob("*")):
                if p.is_file() and not p.name.startswith(".") and not (set(p.relative_to(src).parts[:-1]) & _SKIP_DIRS):
                    files.append(p)
    truncated = len(files) > max_files
    shown = files[:max_files]
    lines = []
    for src in sources:
        lines.append(f"SOURCE: {Path(src).resolve()}")
    for p in shown:
        try:
            rel = p.relative_to(next(s for s in sources if Path(s).is_dir() and p.is_relative_to(Path(s))))
        except (StopIteration, ValueError):
            rel = p.name
        lines.append(f"  {rel}  —  {_describe_file(p)}")
    if truncated:
        lines.append(f"  … {len(files) - max_files} more files not listed")
    text = "\n".join(lines)
    if len(text) > _MAX_INVENTORY_CHARS:
        text = text[:_MAX_INVENTORY_CHARS] + "\n  … (inventory truncated)"
    return text, [str(p.resolve()) for p in files]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
DERIVE_PROMPT = """You are deriving a SECONDARY ARTIFACT from a completed analysis run's saved outputs.
This is NOT a new analysis: do not re-analyze the raw data, do not re-plan, do not fit or
segment anything again. Read the files the run already produced, compute exactly the
product requested, and write it.

## Requested product
{task}

## Available artifacts (read these; paths are absolute at runtime via `_DERIVE["files"]`)
{inventory}

## Rules
- A runtime dict `_DERIVE` is defined before your code: `_DERIVE["files"]` is the list of
  absolute artifact paths above, `_DERIVE["sources"]` the source directories, and
  `_DERIVE["out_dir"]` the ONLY directory you may write to. Build every path from it —
  never hard-code a path.
- Libraries: numpy, pandas, scipy, json, matplotlib (Agg backend; save figures, never show).
  No subprocess, shell, network, or file deletion.
- Produce ONLY what was asked. Do not add unrequested files, columns, or analyses.
- Write each product under `_DERIVE["out_dir"]` with a descriptive file name.
- Finish by printing ONE line:
  `{marker}{{"products": [{{"path": "<abs path>", "description": "<one line>"}}], "summary": "<one or two sentences: what was derived and from which artifacts>"}}`
  Every product path must exist when the script ends.

Return only a single ```python code block.
"""


VERIFY_PROMPT = """You are checking a DERIVED PRODUCT against the request that asked for it.

## Requested product
{task}

## What the script reported
{summary}

## The products (previews)
{previews}

Judge ONLY whether the products deliver what was requested: the right content (values
present, not empty / null / NaN where the artifacts held the numbers), the requested
shape (rows, columns, keys, units, file type), and nothing extra. Do not judge style.
If a requested quantity genuinely does not exist in the run's artifacts, an honest
summary saying so PASSES.

Return JSON only: {{"verdict": "pass" | "fail", "reasons": ["..."], "required_fixes": ["..."]}}
"""


def _product_previews(products: list[dict], max_chars: int = 1400) -> str:
    """Head of each product for the verifier: CSV rows, JSON text, array
    keys / shapes / value ranges, image size."""
    parts = []
    for p in products:
        path = Path(p["path"]); suffix = path.suffix.lower()
        try:
            if suffix in (".csv", ".tsv", ".txt", ".md"):
                text = path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                body = "\n".join(lines[:12]) + (f"\n… ({len(lines)} lines)" if len(lines) > 12 else "")
            elif suffix == ".json":
                body = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            elif suffix == ".npz":
                import numpy as np
                with np.load(path, allow_pickle=False) as z:
                    body = "; ".join(f"{k}: shape={tuple(z[k].shape)} range=[{float(np.nanmin(z[k])):.4g}, {float(np.nanmax(z[k])):.4g}]"
                                     if z[k].size and np.issubdtype(z[k].dtype, np.number) else f"{k}: shape={tuple(z[k].shape)}"
                                     for k in list(z.files)[:12])
            elif suffix == ".npy":
                import numpy as np
                a = np.load(path, mmap_mode="r", allow_pickle=False)
                body = f"shape={tuple(a.shape)} dtype={a.dtype}" + (
                    f" range=[{float(np.nanmin(a)):.4g}, {float(np.nanmax(a)):.4g}] nan_fraction={float(np.isnan(a).mean()):.3f}"
                    if a.size and np.issubdtype(a.dtype, np.floating) else "")
            else:
                body = f"{path.stat().st_size} B"
        except Exception as e:  # noqa: BLE001
            body = f"(preview unavailable: {type(e).__name__})"
        parts.append(f"### {path.name} — {p.get('description', '')}\n{body[:max_chars]}")
    return "\n\n".join(parts)


def static_guard(script: str) -> Optional[str]:
    """Reject scripts that reach for process / OS escape hatches or ignore
    the runtime path dict."""
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return f"script has a syntax error: {e}"
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.split(".")[0] in _DENIED_IMPORTS:
                return f"script imports a forbidden module ('{n.split('.')[0]}')"
    for pat in _DENIED_CALLS:
        if re.search(rf"\b{re.escape(pat)}\s*\(", script):
            return f"script calls a forbidden function ('{pat}')"
    if "_DERIVE" not in script:
        return "the script must read its paths from the runtime `_DERIVE` dict"
    return None


def _header(paths: dict) -> str:
    return ("import os\nimport matplotlib\nmatplotlib.use('Agg')\n"
            "_DERIVE = " + json.dumps(paths) + "\n")


def parse_result_marker(stdout: str) -> tuple[Optional[dict], str]:
    lines = [ln for ln in (stdout or "").splitlines() if ln.startswith(RESULT_MARKER)]
    if not lines:
        return None, f"script did not print the {RESULT_MARKER} line"
    try:
        data = json.loads(lines[-1][len(RESULT_MARKER):])
    except json.JSONDecodeError as e:
        return None, f"{RESULT_MARKER} payload is not valid JSON: {e}"
    if not isinstance(data, dict):
        return None, f"{RESULT_MARKER} payload must be a JSON object"
    return data, ""


def check_products(result: dict, out_dir: Path, started_at: float) -> tuple[list[str], list[dict]]:
    """Deterministic gate: every product is a non-empty file under out_dir
    written by this run. Returns ``(problems, clean_products)``."""
    problems: list[str] = []
    clean: list[dict] = []
    products = result.get("products")
    if not isinstance(products, list) or not products:
        return ["the result names no products"], []
    out_res = out_dir.resolve()
    for i, p in enumerate(products):
        if not isinstance(p, dict) or not p.get("path"):
            problems.append(f"product {i} has no 'path'"); continue
        path = Path(str(p["path"]))
        if not path.is_absolute():
            path = out_dir / path
        rp = path.resolve()
        if not rp.is_relative_to(out_res):
            problems.append(f"product '{path.name}' is outside the output directory"); continue
        if not rp.is_file():
            problems.append(f"product '{path.name}' does not exist"); continue
        st = rp.stat()
        if st.st_size == 0:
            problems.append(f"product '{path.name}' is empty"); continue
        if st.st_mtime < started_at - 1:
            problems.append(f"product '{path.name}' predates this run"); continue
        clean.append({"path": str(rp), "description": str(p.get("description") or ""),
                      "size_bytes": st.st_size})
    return problems, clean


def _tail(s: str, n: int = 2500) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run_derivation(*, model, executor, sources: list[str], task: str, out_dir: Path,
                   scratch_dir: Path, code: Optional[str] = None,
                   logger: Optional[logging.Logger] = None, max_attempts: int = 3,
                   llm_verify: bool = True, parse_json=None) -> dict:
    """Inventory → (supplied or generated) script → guard → sandbox → gate →
    (optional) model check of the products against the task, retrying with
    the failure as feedback. Returns a dict with ``status`` ('success' |
    'error' | 'cancelled'), ``products``, ``summary``, ``script_path``,
    ``attempts``, ``verification`` and ``receipt``."""
    log = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir); scratch_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(exist_ok=True)
    inventory, files = build_artifact_inventory([Path(s) for s in sources])
    if not files:
        return {"status": "error", "message": "no readable artifacts under the given sources",
                "attempts": 0, "attempt_log": []}
    paths = {"files": files, "sources": [str(Path(s).resolve()) for s in sources],
             "out_dir": str(out_dir.resolve())}
    base_prompt = DERIVE_PROMPT.format(task=task, inventory=inventory, marker=RESULT_MARKER)
    feedback = ""
    attempts: list[dict] = []
    t0 = time.time()
    script: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        sys.stdout.write("")   # honour a cancelled stream between attempts
        if code and attempt == 1:
            script = code.strip()
            log.info(f"🧮 derive attempt {attempt}/{max_attempts}: running the supplied script")
        else:
            prompt = base_prompt
            if feedback:
                prompt += (f"\n\n## PREVIOUS ATTEMPT FAILED\n{feedback}\n"
                           + (f"\nThe script that failed:\n```python\n{script[:8000]}\n```\n" if script else "")
                           + "Fix it and return the whole corrected script.")
            log.info(f"🧮 derive attempt {attempt}/{max_attempts}: generating the script")
            try:
                raw = model.generate_content(prompt)
                text = getattr(raw, "raw_text", None) or (raw.text if hasattr(raw, "text") else str(raw))
            except Exception as e:  # noqa: BLE001
                feedback = f"code generation failed: {e}"; attempts.append({"attempt": attempt, "error": feedback}); continue
            script = extract_script(text) or (text.strip() if "_DERIVE" in text else None)
            if not script:
                feedback = "the model returned no python code block"; attempts.append({"attempt": attempt, "error": feedback}); continue
        guard = static_guard(script)
        if guard:
            feedback = guard; attempts.append({"attempt": attempt, "error": guard}); log.warning(f"🧮 rejected: {guard}"); continue
        spath = out_dir / "scripts" / f"derive_script_attempt{attempt}.py"
        spath.write_text(_header(paths) + script, encoding="utf-8")
        started = time.time()
        exec_res = executor.execute_script(_header(paths) + script, working_dir=str(scratch_dir))
        stdout = exec_res.get("stdout", "") or ""
        if exec_res.get("status") != "success":
            msg = exec_res.get("message", "") or ""
            if "stopped by the user" in msg:
                return {"status": "cancelled", "message": msg, "attempts": attempt, "attempt_log": attempts}
            feedback = f"script execution failed: {_tail(msg)}\nstdout tail:\n{_tail(stdout, 1200)}"
            attempts.append({"attempt": attempt, "error": "execution failed", "detail": _tail(msg, 600)})
            log.warning(f"🧮 execution failed: {(_tail(msg, 300).strip().splitlines() or ['?'])[-1]}"); continue
        result, err = parse_result_marker(stdout)
        if result is None:
            feedback = err + f"\nstdout tail:\n{_tail(stdout, 1200)}"; attempts.append({"attempt": attempt, "error": err}); continue
        problems, products = check_products(result, out_dir, started)
        if problems:
            feedback = "product checks failed: " + "; ".join(problems)
            attempts.append({"attempt": attempt, "error": "product checks", "detail": problems[:6]})
            log.warning(f"🧮 product checks failed: {feedback[:300]}"); continue
        summary = str(result.get("summary") or "")
        verdict = {"verdict": "pass", "reasons": []}
        if llm_verify:
            vp = VERIFY_PROMPT.format(task=task, summary=summary or "(none)",
                                      previews=_product_previews(products))
            # Two-vote gate (as in prepare_data): a "fail" stands only if an
            # independent second call agrees — a hallucinated failure would
            # otherwise cost a full regeneration.
            votes = []
            for _vote in range(2):
                try:
                    votes.append(_llm_verdict(model, vp, parse_json))
                except Exception as e:  # noqa: BLE001
                    log.warning(f"🧮 product verification skipped: {e}"); break
                if str(votes[-1].get("verdict", "pass")).lower() == "pass":
                    break
            if votes:
                verdict = votes[-1]
        if str(verdict.get("verdict", "pass")).lower() != "pass":
            fixes = "; ".join(map(str, verdict.get("required_fixes") or verdict.get("reasons") or []))
            feedback = ("the products were checked against the request and do not satisfy it: "
                        + fixes + f"\nproduct previews:\n{_product_previews(products, 600)}")
            attempts.append({"attempt": attempt, "error": "verification failed", "detail": fixes[:800]})
            log.warning(f"🧮 product verification failed: {fixes[:300]}"); continue
        final = out_dir / "scripts" / "derive_script.py"
        final.write_text(_header(paths) + script, encoding="utf-8")
        receipt = {"schema": "scilink_derivation_receipt_v1", "task": task,
                   "sources": paths["sources"], "inputs_seen": len(files),
                   "attempts": attempt, "script": str(final), "products": products,
                   "summary": summary, "supplied_code": bool(code),
                   "verification": verdict, "seconds": round(time.time() - t0, 1)}
        (out_dir / "derivation_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
        (out_dir / "analysis_results.json").write_text(json.dumps({
            "agent_type": "derivation", "status": "success", "task": task,
            "sources": paths["sources"], "products": products, "summary": summary,
            "detailed_analysis": summary}, indent=2, default=str))
        log.info(f"🧮 derived {len(products)} product(s) in {attempt} attempt(s)")
        return {"status": "success", "products": products, "summary": summary,
                "script_path": str(final), "attempts": attempt, "receipt": receipt,
                "verification": verdict, "attempt_log": attempts}
    last = attempts[-1] if attempts else {}
    return {"status": "error",
            "message": f"derivation failed after {max_attempts} attempt(s): "
                       f"{last.get('error', '?')}" + (f" — {last['detail']}" if last.get("detail") else ""),
            "attempts": max_attempts, "attempt_log": attempts}
