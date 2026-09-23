"""Pinned outputs — names that mean the same thing under every recipe.

A generated fitting script names its parameters as it pleases, and the names
carry no guarantee: seen live, ``peak_1_amplitude`` was the peak HEIGHT in one
reference script and the AREA in the next. A live loop that re-anchors gets a
new script each time, so an objective keyed on a script's own parameter name
can silently start meaning something else.

Pinning fixes the names from the outside. The caller declares the outputs it
needs — ``{"peak1_height": "height of the first peak above the baseline", ...}``
— and every recipe the loop ever locks (the first one, and each re-anchor) is
extended to report exactly those names, by the definitions given.

How, and why this way:

- The model writes ONLY the few lines that compute the outputs from quantities
  the script already fitted. It does not return snippet edits: an edit list is
  all-or-nothing on exact whitespace (one mis-indented snippet discarded an
  otherwise clean adaptation in a live run). The block is spliced in
  mechanically, immediately before the script's own ``FIT_RESULTS_JSON`` print —
  a line found deterministically — and travels as ordinary ``script_edits``.
- Acceptance is deterministic, on the reference data: the extended script must
  run, every feature the recipe produced BEFORE must be unchanged (the block may
  add outputs, it may not move the fit), and every pinned output must be
  present and finite. A failure goes back to the model with the reason, up to
  ``max_attempts``.

What this does not verify is that a value matches its definition in words —
that stays the model's reading of the script, recorded with the recipe
(``definitions``) so a person can check it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

OUTPUTS_KEY = "outputs"

PIN_PROMPT = """You are extending a working curve-fitting script so that it reports a fixed set of NAMED OUTPUTS.

## The script (it works; do not change what it fits or how)
```python
{script}
```

## What it currently reports on this reference data
{features}

## Outputs to add — exact names, with their definitions
{outputs}

Write ONLY the Python lines that compute these outputs from quantities the script has ALREADY computed or fitted, and store them in the results dict that the script prints on its `FIT_RESULTS_JSON` line, under `parameters["{key}"]`:

    <results dict>["parameters"]["{key}"] = {{"<name>": <float>, ...}}

Rules:
- Your lines are inserted immediately BEFORE the `FIT_RESULTS_JSON` print, at that line's indentation. Use the variable names that exist at that point in the script.
- Every listed name must appear, as a plain finite float. If the script has an uncertainty for an output, also add "<name>_err".
- When the script already reports a quantity that meets a definition, report THAT value; derive something new only when nothing it reports does, and then follow the definition literally (a HEIGHT is the peak's maximum above the baseline, which for an area-normalised profile is not its amplitude parameter).
- Do not refit, do not re-read the data, do not modify existing entries, do not print anything.
{feedback}
Respond with ONE JSON object and nothing else:
{{"code": "<the python lines>", "rationale": "<one sentence per output: which fitted quantity it comes from>"}}"""


# The deterministic gate can say the named outputs exist and nothing else moved;
# it cannot say a value MEANS what its definition says. Observed live (in-situ
# Raman): the recipe already reported the D and G heights and their ratio (1.18,
# tracking the truth all run) while the added lines re-derived the ratio with the
# baseline subtracted twice — 1.74 on the reference, negative by the end — and
# every check passed. So a second, independent call reads the numbers.
REVIEW_PROMPT = """A curve-fitting script was extended to report NAMED OUTPUTS. Check the values it now reports for them on the reference data.

## Definitions
{outputs}

## What the script itself reports on this data (its own fitted and derived quantities)
{features}

## The values now reported for the named outputs
{pinned}

## The lines that compute them
```python
{code}
```

A named output is wrong when it contradicts its definition, or contradicts a quantity the script already reports that measures the same thing (or a simple function of reported quantities, such as a ratio or a difference). Work the arithmetic from the numbers above; do not judge style.

Respond with ONE JSON object and nothing else:
{{"ok": true | false, "problems": ["<output name>: <what it contradicts, with the numbers>", ...]}}"""


def review_pinned(model: Any, outputs: Dict[str, str], before: Dict[str, float],
                  after: Dict[str, float], code: str,
                  generation_config: Any = None) -> List[str]:
    """Problems an independent reading finds with the pinned VALUES; empty when
    none. A reviewer that cannot be parsed has no opinion (never blocks)."""
    from ..skills._shared._graduation import parse_json_response
    prefix = OUTPUTS_KEY + "_"
    prompt = REVIEW_PROMPT.format(
        outputs="\n".join(f"- `{n}`: {d}" for n, d in outputs.items()),
        features=json.dumps({k: round(v, 6) for k, v in sorted(before.items())
                             if not k.startswith(prefix)}, indent=1),
        pinned=json.dumps({k[len(prefix):]: round(v, 6) for k, v in sorted(after.items())
                           if k.startswith(prefix)}, indent=1),
        code=code)
    kwargs = {"generation_config": generation_config} if generation_config is not None else {}
    try:
        raw = model.generate_content(prompt, **kwargs)
        parsed = parse_json_response(raw.text if hasattr(raw, "text") else str(raw))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(parsed, dict) or parsed.get("ok") is not False:
        return []
    return [str(p)[:300] for p in (parsed.get("problems") or ["the reviewer rejected the values"])][:6]


def _print_lines(script: str) -> List[str]:
    """The distinct source lines that print the FIT_RESULTS_JSON marker."""
    seen: List[str] = []
    for line in script.splitlines():
        if "FIT_RESULTS_JSON" in line and re.search(r"\bprint\s*\(", line) and line not in seen:
            seen.append(line)
    return seen


def splice_edits(script: str, code: str) -> List[Dict[str, Any]]:
    """``script_edits`` that insert ``code`` before each FIT_RESULTS_JSON print,
    at that line's indentation. Raises ``ValueError`` when the script has no
    such line (it then does not follow the fitting contract at all)."""
    lines = _print_lines(script)
    if not lines:
        raise ValueError("the recipe has no FIT_RESULTS_JSON print line to extend")
    body = [ln for ln in str(code).replace("\t", "    ").splitlines()]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        raise ValueError("the model returned no code")
    # Remove the block's own common indentation, then apply the print line's.
    pad = min(len(ln) - len(ln.lstrip()) for ln in body if ln.strip())
    body = [ln[pad:] if ln.strip() else "" for ln in body]
    edits = []
    for line in lines:
        indent = line[:len(line) - len(line.lstrip())]
        block = "\n".join((indent + ln) if ln else "" for ln in body)
        edits.append({"old_text": line,
                      "new_text": f"{indent}# --- pinned outputs (scilink.live) ---\n{block}\n{line}",
                      "replace_all": script.count(line) > 1})
    return edits


def lift_outputs(features: Dict[str, float]) -> Dict[str, float]:
    """Pinned outputs appear flattened as ``outputs_<name>``; expose them under
    their plain names too (the names the caller declared)."""
    prefix = OUTPUTS_KEY + "_"
    out = dict(features)
    for k, v in features.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def check_pinned(before: Dict[str, float], after: Dict[str, float],
                 names: List[str], *, rel_tol: float = 1e-6) -> List[str]:
    """Why an extended recipe is not acceptable; empty when it is."""
    problems = []
    for name in names:
        v = after.get(f"{OUTPUTS_KEY}_{name}")
        if v is None:
            problems.append(f"output '{name}' is missing from parameters['{OUTPUTS_KEY}']")
        elif not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"output '{name}' is not a finite number ({v!r})")
    for k, v in before.items():
        if k.startswith(OUTPUTS_KEY + "_"):
            continue
        w = after.get(k)
        if w is None:
            problems.append(f"'{k}' disappeared — the added lines must not remove results")
        elif abs(w - v) > rel_tol * max(abs(v), abs(w), 1e-12) + 1e-12:
            problems.append(f"'{k}' changed ({v!r} -> {w!r}) — the added lines must not alter the fit")
    return problems


def propose_code(model: Any, script: str, outputs: Dict[str, str],
                 features: Dict[str, float], feedback: Optional[str] = None,
                 generation_config: Any = None) -> Dict[str, Any]:
    from ..skills._shared._graduation import parse_json_response
    prompt = PIN_PROMPT.format(
        script=script,
        features=json.dumps({k: round(v, 6) for k, v in sorted(features.items())
                             if not k.startswith(OUTPUTS_KEY + "_")}, indent=1),
        outputs="\n".join(f"- `{n}`: {d}" for n, d in outputs.items()),
        key=OUTPUTS_KEY,
        feedback=(f"\n## Your previous attempt was rejected\n{feedback}\nFix exactly that.\n"
                  if feedback else ""))
    kwargs = {"generation_config": generation_config} if generation_config is not None else {}
    raw = model.generate_content(prompt, **kwargs)
    parsed = parse_json_response(raw.text if hasattr(raw, "text") else str(raw))
    if not isinstance(parsed, dict) or not str(parsed.get("code") or "").strip():
        raise ValueError("the model did not return a JSON object with `code`")
    return parsed


def pin_outputs(*, script: str, outputs: Dict[str, str], model: Any,
                replay: Callable[[Optional[List[Dict[str, Any]]]], Dict[str, float]],
                max_attempts: int = 3, generation_config: Any = None,
                logger: Optional[logging.Logger] = None, review: bool = True) -> Dict[str, Any]:
    """Extend ``script`` to report ``outputs``; return the accepted
    ``{"edits", "features", "rationale", "attempts", "llm_calls", "review"}``.

    Two gates per attempt: the deterministic one (names present and finite,
    nothing else changed) and, with ``review``, an independent reading of the
    VALUES against the definitions and the script's own reported quantities.
    A reviewer objection is fed back like any other rejection. If the last
    attempt passes the deterministic gate and only the reviewer still objects,
    it is accepted and the objection travels in ``review`` — an argument between
    two model calls must not stop an experiment, but it must be visible.

    ``replay(edits)`` runs the locked recipe on the REFERENCE data with the
    given ``script_edits`` (``None`` = unedited) and returns its flat numeric
    features, or raises with the reason — the loop supplies it, so this module
    never touches an agent. Raises ``RuntimeError`` when no attempt is
    acceptable; the caller decides whether a loop may run unpinned.
    """
    log = logger or logging.getLogger("MeasurementLoop")
    names = list(outputs)
    if not names:
        raise ValueError("no outputs to pin")
    bad = [n for n in names if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n)]
    if bad:
        raise ValueError(f"output names must be identifiers: {bad}")
    before = replay(None)
    feedback, last, llm_calls = None, "no attempt made", 0

    def accepted(edits, after, proposal, attempt, objections):
        return {"edits": edits, "features": lift_outputs(after),
                "rationale": str(proposal.get("rationale") or "")[:600],
                "attempts": attempt, "llm_calls": llm_calls,
                "review": {"ok": not objections, "problems": objections}}

    for attempt in range(1, max_attempts + 1):
        objections: List[str] = []
        try:
            llm_calls += 1
            proposal = propose_code(model, script, outputs, before, feedback,
                                    generation_config=generation_config)
            edits = splice_edits(script, proposal["code"])
            after = replay(edits)
            problems = check_pinned(before, after, names)
            if not problems and review:
                llm_calls += 1
                objections = review_pinned(model, outputs, before, after, proposal["code"],
                                           generation_config=generation_config)
        except Exception as e:  # noqa: BLE001 - fed back to the model
            problems = [f"{type(e).__name__}: {e}"]
            after, proposal, edits = {}, {}, []
        if not problems and not objections:
            log.info(f"📌 Outputs pinned on attempt {attempt}: "
                     + ", ".join(f"{n}={after[OUTPUTS_KEY + '_' + n]:.5g}" for n in names))
            return accepted(edits, after, proposal, attempt, [])
        if not problems and attempt == max_attempts:
            log.warning("📌 Outputs pinned, but the value review still objects: "
                        + "; ".join(objections))
            return accepted(edits, after, proposal, attempt, objections)
        last = "; ".join(problems or [f"value review: {o}" for o in objections])[:900]
        feedback = last
        log.warning(f"📌 Pinning attempt {attempt}/{max_attempts} rejected: {last}")
    raise RuntimeError(f"could not pin outputs {names} after {max_attempts} attempt(s): {last}")


def agent_replay(agent_factory: Callable[[str], Any], anchor_dir: str, data_path: str,
                 system_info: Any, work_dir: str,
                 base_edits: Optional[List[Dict[str, Any]]] = None
                 ) -> Callable[[Optional[List[Dict[str, Any]]]], Dict[str, float]]:
    """A ``replay(edits)`` for :func:`pin_outputs`: the locked recipe on the
    reference data through the curve agent's strict zero-LLM replay, with
    ``base_edits`` (the caller's own amendments) always applied first."""
    from .measurement_loop import _numeric_features
    counter = [0]

    def replay(edits: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        counter[0] += 1
        agent = agent_factory(str(Path(work_dir) / f"pin_{counter[0]:02d}"))
        all_edits = list(base_edits or []) + list(edits or [])
        kwargs: Dict[str, Any] = dict(
            system_info=system_info, prior_analysis_paths=[str(anchor_dir)],
            reuse_locked_script=True, profile="realtime", strict_replay=True)
        if all_edits:
            kwargs["script_edits"] = all_edits
        res = agent.analyze(data_path, **kwargs) or {}
        if res.get("status") != "success":
            raise RuntimeError("the extended recipe did not run on the reference data: "
                               + json.dumps(res.get("error"), default=str)[:400])
        feats = _numeric_features(res, lift=False)
        if not feats:
            raise RuntimeError("the extended recipe produced no numeric results")
        return feats

    return replay
