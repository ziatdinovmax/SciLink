"""Data-preparation stage: raw instrument container -> analysis-ready products.

The analysis agents take curves, images and spectral cubes. Some instruments
hand over something upstream of that — a temporal hologram stack, a raw
detector container with a reconstruction contract — that must be TRANSFORMED
(reconstructed, reduced, calibrated) before any curve/image agent may see it.
This module is the engine behind the orchestrator's ``prepare_data`` tool:

  detect_raw_instrument   sidecar/manifest/HDF5 markers -> "this needs preparation"
  build_prep_context      inventory of the container for the code-writing prompt
  run_preparation         plan -> generate script -> sandbox run -> verify
                          products (deterministic gate + optional skill-guided
                          LLM check) -> retry with feedback

Knowledge comes from ``scilink/skills/data_preparation/<skill>/`` bundles
(planning / implementation / validation sections) and their ``TOOL_SPEC``
helpers, rendered through the shared tool inventory. Products are ordinary
data files with same-stem JSON sidecars, so ``run_analysis``, fan-out and
fusion consume them unchanged.
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

DOMAIN = "data_preparation"
RESULT_MARKER = "PREP_RESULT_JSON:"
PRODUCT_KINDS = ("image", "curve", "hyperspectral", "stack", "table", "figure", "other")
MANIFEST_NAMES = ("reconstruction_manifest.json",)
_MARKER_KEYS = {
    "generic_image_routing_permitted": lambda v: v is False,
    "generic_run_analysis_permitted": lambda v: v is False,
    "generic_prepare_inputs_permitted": lambda v: v is False,
    "analysis_status": lambda v: isinstance(v, str) and "reconstruct" in v.lower(),
    "measurement_type": lambda v: isinstance(v, str) and any(
        k in v.lower() for k in ("hologram", "interferogram", "raw_detector", "sinogram")),
    "authoritative_boundary": lambda v: isinstance(v, str) and "raw" in v.lower(),
}
_HDF5_JSON_HINT_KEYS = ("analysis_contract_json", "processing_contracts_json",
                        "raw_first_boundary_json")
# Prep scripts may transform data (scipy/skimage/cv2 are legitimate) — the
# denylist only blocks process/OS escape hatches.
_DENIED_IMPORTS = ("subprocess", "shutil", "socket", "requests", "urllib", "ctypes")
_DENIED_CALLS = ("os.system", "os.remove", "os.unlink", "os.rmdir", "os.removedirs")
_MAX_INVENTORY_CHARS = 14000


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _read_json(path: Path, limit: int = 2_000_000) -> Optional[dict]:
    try:
        if path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _markers_in(obj: Any) -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            test = _MARKER_KEYS.get(k)
            if test and test(v):
                hits.append(f"{k}={v!r}"[:80])
    return hits


def _hdf5_markers(path: Path) -> list[str]:
    hits: list[str] = []
    try:
        import h5py
        with h5py.File(path, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.shape == () and \
                        any(name.endswith(k) for k in _HDF5_JSON_HINT_KEYS):
                    try:
                        v = obj[()]
                        v = v.decode("utf-8") if isinstance(v, bytes) else v
                        parsed = json.loads(v) if isinstance(v, str) else v
                        if isinstance(parsed, list) and parsed:
                            parsed = parsed[0]
                        for m in _markers_in(parsed):
                            hits.append(f"{name}: {m}")
                    except Exception:  # noqa: BLE001
                        pass
            h.visititems(visit)
            for k, v in h.attrs.items():
                for m in _markers_in({k: v}):
                    hits.append(f"root attr {m}")
    except Exception:  # noqa: BLE001
        pass
    return hits


def detect_raw_instrument(path: str | Path) -> Optional[dict]:
    """Return evidence that ``path`` is a raw instrument container that must be
    prepared before analysis, or ``None``.

    Evidence, in order: a same-stem JSON sidecar carrying routing-denial /
    reconstruction markers; a directory holding a reconstruction manifest or
    such sidecars; JSON contract datasets or root attributes inside an HDF5.
    """
    p = Path(path)
    evidence: list[str] = []
    manifest = None
    sidecar = None
    if p.is_dir():
        for name in MANIFEST_NAMES:
            m = p / name
            if m.is_file():
                manifest = m; evidence.append(f"manifest {name}")
                d = _read_json(m) or {}
                evidence += _markers_in(d)
        for j in sorted(p.glob("*.json"))[:200]:
            d = _read_json(j)
            hits = _markers_in(d) if d else []
            if hits:
                evidence.append(f"sidecar {j.name}: " + "; ".join(hits[:3]))
                sidecar = sidecar or j
        for sub in ("raw_hdf5", "raw"):
            sd = p / sub
            if sd.is_dir():
                for j in sorted(sd.glob("*.json"))[:200]:
                    d = _read_json(j)
                    hits = _markers_in(d) if d else []
                    if hits:
                        evidence.append(f"sidecar {sub}/{j.name}: " + "; ".join(hits[:3]))
                        sidecar = sidecar or j
                        break
    elif p.is_file():
        for cand in (p.with_suffix(".json"), p.parent / (p.name + ".json")):
            if cand.is_file() and cand != p:
                d = _read_json(cand)
                hits = _markers_in(d) if d else []
                if hits:
                    sidecar = cand; evidence.append(f"sidecar {cand.name}: " + "; ".join(hits[:3]))
                    break
        if p.suffix.lower() in (".h5", ".hdf5", ".nxs"):
            evidence += _hdf5_markers(p)[:4]
        for name in MANIFEST_NAMES:
            m = p.parent / name
            if m.is_file():
                manifest = m; evidence.append(f"manifest {name} beside the file")
    if not evidence:
        return None
    return {"data_type": "raw_instrument", "evidence": evidence[:8],
            "manifest": str(manifest) if manifest else None,
            "sidecar": str(sidecar) if sidecar else None}


# --------------------------------------------------------------------------
# Context inventory for the code-writing prompt
# --------------------------------------------------------------------------
def _hdf5_table(path: Path, max_rows: int = 40) -> str:
    rows = []
    try:
        import h5py
        with h5py.File(path, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and len(rows) < max_rows:
                    rows.append(f"  {name}  shape={tuple(obj.shape)} dtype={obj.dtype}")
            h.visititems(visit)
    except Exception as e:  # noqa: BLE001
        rows.append(f"  (could not open: {e})")
    return "\n".join(rows)


def build_prep_context(path: str | Path, detection: Optional[dict] = None,
                       max_chars: int = _MAX_INVENTORY_CHARS) -> str:
    """Human-readable inventory: file listing, manifest / sidecar JSON
    (truncated), HDF5 dataset tables. Absolute paths, so the generated script
    can reference bundle files through ``_PREP['input']``-relative joins."""
    p = Path(path).resolve()
    parts = [f"INPUT PATH: {p}  ({'directory' if p.is_dir() else 'file'})"]
    if detection:
        parts.append("RAW-INSTRUMENT EVIDENCE: " + "; ".join(detection.get("evidence", [])))
    files: list[Path] = []
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append(f)
            if len(files) >= 400:
                break
        parts.append("FILES (relative to the input directory, size in MB):")
        parts += [f"  {f.relative_to(p)}  {f.stat().st_size/1e6:.1f}" for f in files[:400]]
        if len(files) >= 400:
            parts.append("  ... (listing truncated)")
    else:
        files = [p]
    budget = max_chars - sum(len(x) for x in parts)
    # manifest first, then sidecars (smallest first), then HDF5 tables
    jsons = [f for f in files if f.suffix == ".json" and f.stat().st_size < 60_000]
    jsons.sort(key=lambda f: (0 if f.name in MANIFEST_NAMES else 1, f.stat().st_size))
    for j in jsons[:12]:
        txt = j.read_text(encoding="utf-8", errors="replace")
        if len(txt) > 3500:
            txt = txt[:3500] + " ...(truncated)"
        block = f"JSON {j.relative_to(p) if p.is_dir() else j.name}:\n{txt}"
        if budget - len(block) < 0:
            break
        parts.append(block); budget -= len(block)
    h5s = [f for f in files if f.suffix.lower() in (".h5", ".hdf5", ".nxs")]
    seen = 0
    for f in h5s:
        if seen >= 2 or budget < 800:
            break
        block = f"HDF5 {f.relative_to(p) if p.is_dir() else f.name} datasets:\n{_hdf5_table(f)}"
        parts.append(block); budget -= len(block); seen += 1
    if len(h5s) > seen:
        parts.append(f"(+{len(h5s) - seen} more HDF5 files with the same layout are listed above)")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Codegen prompt and guards
# --------------------------------------------------------------------------
PREP_PROMPT = """You are preparing RAW INSTRUMENT DATA for downstream scientific analysis.
This is a TRANSFORMING preparation step (reconstruction, reduction, calibration,
joining with a condition timeline) that turns a raw container into analysis-ready
products. It is NOT the scientific analysis itself: do not fit models, do not
interpret, do not draw conclusions. Produce clean inputs and honest QC.

## Task
{task}

## Input inventory
{context}

{skill_block}

{tool_inventory}

## Runtime contract (mandatory)
- A dict named `_PREP` is defined ABOVE your code. Use it for EVERY path:
    _PREP["input"]      # the file or directory given (absolute)
    _PREP["out_dir"]    # write every product here (absolute, exists)
    _PREP["scratch"]    # scratch directory for large intermediates (absolute, exists)
  Never hardcode absolute paths; build paths with os.path.join / pathlib from `_PREP`.
  Never write inside the input directory.
- Prefer the registered tools above when they fit; they encode the validated
  recipe. Combine them with numpy/scipy/skimage for glue and QC.
- Every product must be an analysis-ready file with a same-stem JSON sidecar:
  a 2-D `.npy` (image / map), a 3-D `.npy` (spectral cube, spectral axis last),
  a `.csv` with one header row for a curve or time series (x first column),
  or a `.png` figure for QC. The sidecar must state technique, units, axes,
  semantics, and interpretation limits. If a tool already writes sidecars, reuse them.
- Respect QC gates stated by the skill or the container (e.g. producer validation
  targets). If a gate fails, do NOT silently continue: report it in the result.
  `qc.passed` means: every gate that EXISTS for this data passed AND the skill's
  internal self-consistency checks passed. The ABSENCE of an external gate or
  contract (no validation targets, an auto-picked carrier, missing calibration)
  is a caveat to state in `qc.notes`, NOT a failure — do not set passed=false
  for a check you could not run.
- Product kinds: "image" (2-D array), "curve" (CSV), "hyperspectral" (3-D cube,
  spectral axis last), "stack" (3-D frame stack kept for provenance — NOT an
  analysis input), "table", "figure". Keep large intermediates as "stack" or
  under `receipts`, never as "image".
- Print progress sparingly. At the very end print ONE line exactly:
  {marker}{{"products": [{{"path": "<abs path>", "kind": "image|curve|hyperspectral|stack|table|figure",
  "sidecar": "<abs path or null>", "description": "<one line>", "group": "<optional series name>"}}, ...],
  "qc": {{"passed": true|false, "metrics": {{"<name>": <number>, ...}}, "notes": ["..."]}},
  "receipts": ["<abs path>", ...], "summary": "<2-4 sentences: what was produced and any caveat>"}}
  where `qc.metrics` holds ONLY numbers (gate values, coverage fractions, counts).

Respond with ONE ```python code block and nothing else.
"""

SKILL_BLOCK = """## Domain preparation skill: {name}
{description}

### Planning guidance
{planning}

### Implementation recipe (follow it; adapt parameters to this data)
{implementation}
"""

VERIFY_PROMPT = """You are verifying a DATA PREPARATION run (raw instrument container ->
analysis-ready products). Judge whether the products are trustworthy inputs for
downstream analysis, using the skill's validation rules below. Be strict about
QC gates and honest reporting; do not demand scientific interpretation.

## Task
{task}

## Skill validation rules
{validation}

## Script (generated)
```python
{script}
```

## Script stdout (tail)
{stdout}

## Reported result
{result}

## Deterministic checks (already enforced by the framework)
{checks}
(Every declared product exists under the output directory; a missing sidecar was
synthesized from the declared description — do not fail for sidecar presence.)

Framework contract the script must obey (do not ask for anything that violates it):
`qc.metrics` holds ONLY scalar numbers — per-item values (e.g. one coherence per
target frame) are separate keyed scalars, never lists; the receipt files hold the
full records. Judge the values that ARE reported against the rules; do not demand a
different reporting shape.

Reply with ONE JSON object: {{"verdict": "pass" | "fail", "reasons": ["..."],
"required_fixes": ["..."]}} — "fail" only for a concrete defect (a gate not met, a
product missing or mis-declared, a rule violated), never for style or shape.
"""


def static_guard(script: str) -> Optional[str]:
    """Reject scripts that reach for process / OS escape hatches."""
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return f"generated script has a syntax error: {e}"
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            root = n.split(".")[0]
            if root in _DENIED_IMPORTS:
                return f"generated script imports a forbidden module ('{root}')"
    for pat in _DENIED_CALLS:
        if re.search(rf"\b{re.escape(pat)}\s*\(", script):
            return f"generated script calls a forbidden function ('{pat}')"
    if "_PREP" not in script:
        return "the script must read every path from the runtime `_PREP` dict"
    return None


def extract_script(text: str) -> Optional[str]:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _prep_header(paths: dict) -> str:
    return "import os\n_PREP = " + json.dumps({k: str(v) for k, v in paths.items()}) + "\n"


def parse_result_marker(stdout: str) -> tuple[Optional[dict], str]:
    lines = [ln for ln in stdout.splitlines() if ln.startswith(RESULT_MARKER)]
    if not lines:
        return None, f"script did not print the {RESULT_MARKER} line"
    try:
        return json.loads(lines[-1][len(RESULT_MARKER):]), ""
    except json.JSONDecodeError as e:
        return None, f"{RESULT_MARKER} payload is not valid JSON: {e}"


def check_products(result: dict, out_dir: Path) -> tuple[list[str], list[dict]]:
    """Deterministic gate: products exist, live under out_dir, carry sidecars
    (a missing sidecar is synthesized from the description), kinds are known,
    qc.passed is true, qc.metrics numeric."""
    problems: list[str] = []
    products = result.get("products")
    if not isinstance(products, list) or not products:
        return ["no products declared"], []
    clean: list[dict] = []
    for i, pr in enumerate(products):
        if not isinstance(pr, dict) or not pr.get("path"):
            problems.append(f"product #{i} has no path"); continue
        path = Path(pr["path"])
        if not path.is_file():
            problems.append(f"product missing on disk: {path}"); continue
        try:
            path.resolve().relative_to(out_dir.resolve())
        except ValueError:
            problems.append(f"product written outside the output directory: {path}"); continue
        kind = pr.get("kind") or "other"
        if kind not in PRODUCT_KINDS:
            problems.append(f"product {path.name}: unknown kind '{kind}'"); continue
        if path.suffix.lower() == ".npy":
            try:
                import numpy as np
                arr = np.load(path, mmap_mode="r")
                if kind == "image" and arr.ndim != 2:
                    problems.append(f"{path.name}: kind image but ndim={arr.ndim}")
                if kind == "hyperspectral" and arr.ndim != 3:
                    problems.append(f"{path.name}: kind hyperspectral but ndim={arr.ndim}")
                if kind == "stack" and arr.ndim != 3:
                    problems.append(f"{path.name}: kind stack but ndim={arr.ndim}")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{path.name}: not loadable as .npy ({e})")
        side = Path(pr["sidecar"]) if pr.get("sidecar") else path.with_suffix(".json")
        if not side.is_file() and kind != "figure":
            try:
                side.write_text(json.dumps({"description": pr.get("description", ""),
                                            "kind": kind, "generated_by": "prepare_data (synthesized sidecar)"}, indent=2))
            except Exception:  # noqa: BLE001
                problems.append(f"{path.name}: sidecar missing and could not be written")
        clean.append({**pr, "path": str(path), "kind": kind,
                      "sidecar": str(side) if side.is_file() else None})
    qc = result.get("qc") or {}
    if not isinstance(qc, dict):
        problems.append("qc block missing")
    else:
        if qc.get("passed") is not True:
            problems.append("qc.passed is not true: " + "; ".join(map(str, qc.get("notes") or []))[:400])
        for k, v in (qc.get("metrics") or {}).items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                problems.append(f"qc.metrics['{k}'] is not a number")
    return problems, clean


def _llm_verdict(model, prompt: str, parse_json=None) -> dict:
    vraw = model.generate_content(prompt)
    vtext = getattr(vraw, "raw_text", None) or (vraw.text if hasattr(vraw, "text") else str(vraw))
    parsed = None
    if parse_json:
        try:
            parsed = parse_json(vtext)
            if isinstance(parsed, tuple):
                parsed = parsed[0]
        except Exception:  # noqa: BLE001
            parsed = None
    if not isinstance(parsed, dict):
        m = re.search(r"\{.*\}", vtext, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}
    return parsed if isinstance(parsed, dict) and parsed.get("verdict") else {"verdict": "pass", "reasons": []}


def _tail(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run_preparation(*, model, executor, data_path: str, task: str, out_dir: Path,
                    scratch_dir: Path, context: str, skill: Optional[dict] = None,
                    tool_inventory: str = "", logger: Optional[logging.Logger] = None,
                    max_attempts: int = 3, llm_verify: bool = True,
                    parse_json=None) -> dict:
    """Generate → guard → execute → verify, retrying with failure feedback.

    ``skill`` is a loaded skill dict (loader.load_skill) or None. Returns a
    dict with ``status`` ('success' | 'error'), ``products``, ``qc``,
    ``script_path``, ``attempts`` and a ``receipt``.
    """
    log = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir); scratch_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(exist_ok=True)
    skill_block = ""
    validation = ""
    if skill:
        meta = skill.get("meta") or {}
        skill_block = SKILL_BLOCK.format(
            name=skill.get("name", "?"), description=meta.get("description", ""),
            planning=skill.get("planning") or "(none)",
            implementation=skill.get("implementation") or skill.get("analysis") or "(none)")
        validation = skill.get("validation") or ""
    base_prompt = PREP_PROMPT.format(task=task, context=context, skill_block=skill_block,
                                     tool_inventory=tool_inventory, marker=RESULT_MARKER)
    paths = {"input": str(Path(data_path).resolve()), "out_dir": str(out_dir.resolve()),
             "scratch": str(scratch_dir.resolve())}
    feedback = ""
    attempts: list[dict] = []
    t0 = time.time()
    for attempt in range(1, max_attempts + 1):
        # A cancelled fan-out branch / UI stop raises on the next stdout write
        # (see meta_agent.fanout._ThreadStopStream); touch stdout so a long
        # preparation loop honours cancellation between attempts.
        sys.stdout.write("")
        prompt = base_prompt + (f"\n\n## PREVIOUS ATTEMPT FAILED\n{feedback}\nFix it and regenerate the whole script." if feedback else "")
        log.info(f"🧰 prepare_data attempt {attempt}/{max_attempts}: generating preparation script")
        try:
            raw = model.generate_content(prompt)
            # raw_text is the model's UNCLEANED output; the wrapper's .text is a
            # JSON-extracted view that slices a script from its first '{' (#238).
            text = getattr(raw, "raw_text", None) or (raw.text if hasattr(raw, "text") else str(raw))
        except Exception as e:  # noqa: BLE001
            feedback = f"code generation failed: {e}"; attempts.append({"attempt": attempt, "error": feedback}); continue
        script = extract_script(text)
        if not script:
            feedback = "the model returned no python code block"; attempts.append({"attempt": attempt, "error": feedback}); continue
        guard = static_guard(script)
        if guard:
            feedback = guard; attempts.append({"attempt": attempt, "error": feedback}); log.warning(f"🧰 rejected: {guard}"); continue
        spath = out_dir / "scripts" / f"prepare_script_attempt{attempt}.py"
        spath.write_text(_prep_header(paths) + script, encoding="utf-8")
        log.info(f"🧰 executing preparation script ({len(script)} chars)")
        exec_res = executor.execute_script(_prep_header(paths) + script, working_dir=str(scratch_dir))
        stdout = exec_res.get("stdout", "") or ""
        if exec_res.get("status") != "success":
            msg = exec_res.get("message", "") or ""
            if "stopped by the user" in msg:
                log.warning("🧰 preparation stopped (cancelled); not retrying")
                return {"status": "cancelled", "message": msg, "attempts": attempt, "attempt_log": attempts}
            feedback = f"script execution failed: {_tail(msg, 2500)}\nstdout tail:\n{_tail(stdout, 1500)}"
            attempts.append({"attempt": attempt, "error": "execution failed", "detail": _tail(msg, 800)})
            log.warning(f"🧰 execution failed: {_tail(msg, 400).strip().splitlines()[-1] if msg.strip() else 'no message'}"); continue
        result, err = parse_result_marker(stdout)
        if result is None:
            feedback = err + f"\nstdout tail:\n{_tail(stdout, 1500)}"; attempts.append({"attempt": attempt, "error": err}); continue
        problems, products = check_products(result, out_dir)
        checks = "all deterministic checks passed" if not problems else "; ".join(problems)
        if problems:
            feedback = "deterministic product checks failed: " + checks
            attempts.append({"attempt": attempt, "error": "product checks", "detail": problems[:6]})
            log.warning(f"🧰 product checks failed: {checks[:300]}"); continue
        verdict = {"verdict": "pass", "reasons": []}
        if llm_verify and validation:
            reported = dict(result, products=products)   # post-check state: sidecars synthesized, paths resolved
            vp = VERIFY_PROMPT.format(task=task, validation=validation, script=script[:12000],
                                      stdout=_tail(stdout, 2500), result=json.dumps(reported, indent=1)[:4000], checks=checks)
            # Two-vote gate: a "fail" is re-checked by an independent call and stands
            # only if both agree — a hallucinated failure costs a full regeneration.
            votes = []
            for _vote in range(2):
                try:
                    votes.append(_llm_verdict(model, vp, parse_json))
                except Exception as e:  # noqa: BLE001
                    log.warning(f"🧰 skill-guided verification skipped: {e}"); break
                if str(votes[-1].get("verdict", "pass")).lower() == "pass":
                    break
            if votes:
                verdict = votes[-1]
                if len(votes) == 2 and str(votes[-1].get("verdict")).lower() == "pass":
                    log.info("🧰 verification: first vote failed, second passed — accepting (single-vote failure discarded)")
        if str(verdict.get("verdict", "pass")).lower() != "pass":
            fixes = "; ".join(map(str, verdict.get("required_fixes") or verdict.get("reasons") or []))
            feedback = "skill-guided verification failed: " + fixes
            attempts.append({"attempt": attempt, "error": "verification failed", "detail": fixes[:800]})
            log.warning(f"🧰 verification failed: {fixes[:300]}"); continue
        final = out_dir / "scripts" / "prepare_script.py"
        final.write_text(_prep_header(paths) + script, encoding="utf-8")
        receipt = {"schema": "scilink_data_preparation_receipt_v1", "input": paths["input"],
                   "task": task, "skill": (skill or {}).get("name"), "attempts": attempt,
                   "script": str(final), "products": products, "qc": result.get("qc"),
                   "receipts": result.get("receipts") or [], "summary": result.get("summary", ""),
                   "verification": verdict, "seconds": round(time.time() - t0, 1)}
        (out_dir / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
        metrics = {k: v for k, v in ((result.get("qc") or {}).get("metrics") or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        (out_dir / "analysis_results.json").write_text(json.dumps({
            "agent_type": "data_preparation", "status": "success", "task": task,
            "skill": (skill or {}).get("name"), "products": products,
            "extracted_features": metrics, "summary": result.get("summary", "")}, indent=2, default=str))
        log.info(f"✅ prepare_data: {len(products)} product(s) after {attempt} attempt(s)")
        return {"status": "success", "products": products, "qc": result.get("qc"),
                "summary": result.get("summary", ""), "receipts": result.get("receipts") or [],
                "script_path": str(final), "receipt": str(out_dir / "preparation_receipt.json"),
                "attempts": attempt, "verification": verdict, "stdout_tail": _tail(stdout, 1200)}
    return {"status": "error", "message": f"no verified preparation after {max_attempts} attempt(s): {feedback[:600]}",
            "attempts": max_attempts, "attempt_log": attempts}
