"""LLM-driven analysis-script generation for live-monitoring sessions.

A live session's reading function is what processes each new spectrum /
image / data chunk and turns it into a structured ``LiveReadingResult``.
In the adaptive default mode, this function is **LLM-generated** at
session start (and may be re-generated mid-session when the supervisor
detects a deviation that requires a different analysis approach).

Three responsibilities:

  - :func:`generate_reading_script` — call the LLM with the operator's
    description + optional skill context; return Python source.
  - :func:`save_and_load_reading_script` — persist source to disk under
    the session directory and import it; return the callable.
  - :func:`validate_reading_script` — run the new function against a
    small sample of cached readings; reject if it raises or returns the
    wrong shape.

Validation is intentionally conservative: a buggy regeneration must
NOT replace a working script. The dispatcher catches validation
failure and keeps the previous reading_fn active.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from .live_data_sources import LatestData
from .live_types import LiveReadingResult

_logger = logging.getLogger(__name__)


_CODEGEN_SYSTEM_PROMPT = """You are writing the per-reading analysis function for a SciLink live-monitoring session.

The function is invoked many times during a live session (typically
every 2-5 seconds) as new data arrives from an in-progress experiment.
It must be fast (well under a second per call) and side-effect-free
beyond mutating session_state.

REQUIRED SIGNATURE (do not change it):

    def reading_fn(latest_data, session_state, skill_state):
        # latest_data is a LatestData dataclass with:
        #   .text         (str | None) — CSV/whitespace file contents
        #   .path         (Path | None)
        #   .timestamp    (float)
        #   .extras       (dict, source-specific)
        #
        # session_state (dict) — mutable; persist any caches here across
        #                        readings (e.g. pre-computed reference patterns)
        # skill_state   (dict) — frontmatter + sections of the active skill,
        #                        empty when no skill is selected
        #
        # Must return a LiveReadingResult dataclass (imported from
        # scilink.agents.exp_agents.live_types).
        ...

Available libraries: numpy as np, scipy.signal, scipy.optimize, scipy.stats.
Optional (try-import gracefully): pymatgen, sklearn.
Standard library is fully available.

LiveReadingResult fields (all positional / keyword arg construction works):
  timestamp        float
  primary_metric   float    (e.g. correlation, FOM, peak count)
  metric_name      str      (short identifier; appears in dashboards)
  verdict          str      ("accept" | "marginal" | "reject" | "unknown")
  detected_features list[dict]  (e.g. [{"position": 28.4, "intensity": 100}, ...])
  notes            str      (one-line human-readable summary)
  raw              dict     (extras for the supervisor LLM)

DESCRIPTION OF THE EXPERIMENT (from the operator):
{description}

ADDITIONAL CONTEXT (from the operator):
{additional_guidance}

SKILL CONTEXT (frontmatter description + analysis section, if any skill is selected):
{skill_context}

PRIOR ATTEMPTS (only present when this is a regeneration; see why and adapt):
{prior_context}

Return ONLY the Python source — no markdown fences, no surrounding prose.
Include all imports at the top. The framework will exec() the source
in a fresh module namespace and call the top-level ``reading_fn``.
"""


def generate_reading_script(
    *,
    description: str,
    additional_guidance: Optional[str],
    skill_context: Optional[str],
    model: str,
    api_key: str,
    prior_context: Optional[str] = None,
) -> Optional[str]:
    """Generate a reading_fn Python source via the LLM.

    Returns the source as a string, or None on any failure (LLM call,
    JSON parse, missing function). Caller treats None as "regeneration
    failed; keep the current reading_fn active."
    """
    try:
        import litellm
    except ImportError:
        _logger.error("litellm not installed; cannot generate reading script")
        return None
    try:
        from ...wrappers.litellm_wrapper import _normalize_model_name
    except ImportError:
        _normalize_model_name = lambda m: m  # noqa: E731

    # Use replace() not format() — the prompt contains literal `{"position": ...}`
    # examples that would be misinterpreted as format placeholders.
    system = (
        _CODEGEN_SYSTEM_PROMPT
        .replace("{description}", description.strip())
        .replace("{additional_guidance}",
                 (additional_guidance or "(none)").strip())
        .replace("{skill_context}",
                 (skill_context or "(no skill selected)").strip())
        .replace("{prior_context}",
                 (prior_context or "(first-time generation)").strip())
    )
    try:
        resp = litellm.completion(
            model=_normalize_model_name(model),
            api_key=api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Write the reading_fn now."},
            ],
            max_tokens=4000,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        _logger.error("LLM call failed in generate_reading_script: %s", e)
        return None

    # Strip code fences if the LLM wraps anyway
    fenced = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    if "def reading_fn" not in text:
        _logger.warning("Generated source has no `def reading_fn` — rejecting")
        return None
    return text


def save_and_load_reading_script(
    source: str, session_dir: Path, version: int,
) -> Optional[Callable]:
    """Persist source to ``session_dir/reading_script_vN.py`` and import.

    Returns the resolved ``reading_fn`` callable, or None if exec /
    lookup failed. Failures are logged with the traceback so the
    operator can see why the LLM's output didn't work.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    script_path = session_dir / f"reading_script_v{version}.py"
    script_path.write_text(source)

    module_name = f"_scilink_live_reading_v{version}_{int(time.time() * 1000)}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            _logger.error("Could not build module spec for %s", script_path)
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        _logger.error(
            "Failed to import generated script %s:\n%s",
            script_path, traceback.format_exc(),
        )
        return None

    fn = getattr(mod, "reading_fn", None)
    if not callable(fn):
        _logger.error("Generated script %s has no callable reading_fn", script_path)
        return None
    return fn


def validate_reading_script(
    reading_fn: Callable,
    sample_history: list[LatestData],
    skill_state: Optional[dict] = None,
) -> tuple[bool, str]:
    """Run the new ``reading_fn`` against a small sample of cached
    ``LatestData`` payloads. Must complete without raising and return
    a :class:`LiveReadingResult` each time.

    Returns ``(ok, reason)``. ``ok=False`` means the regeneration is
    rejected — the dispatcher keeps the previous reading_fn active and
    records the reason in JSONL for the operator to see.
    """
    if not sample_history:
        # Nothing to validate against; conservative pass — the first
        # actual reading after activation will surface any bug, but at
        # least we know the script parsed + imported.
        return True, "no sample history to validate against (script imported cleanly)"

    state: dict = {}
    for i, ld in enumerate(sample_history):
        try:
            result = reading_fn(ld, state, dict(skill_state or {}))
        except Exception as e:
            return False, (
                f"reading_fn raised on validation sample {i}: "
                f"{type(e).__name__}: {e}"
            )
        if not isinstance(result, LiveReadingResult):
            return False, (
                f"reading_fn returned {type(result).__name__} on sample {i}; "
                f"expected LiveReadingResult"
            )
        # Schema sanity: required fields present and right-ish types
        for field, expected_type in (
            ("primary_metric", (int, float)),
            ("metric_name", str),
            ("verdict", str),
        ):
            v = getattr(result, field, None)
            if not isinstance(v, expected_type):
                return False, (
                    f"reading_fn sample {i}: field {field} has type "
                    f"{type(v).__name__}; expected {expected_type}"
                )
        if result.verdict not in ("accept", "marginal", "reject", "unknown"):
            return False, (
                f"reading_fn sample {i}: verdict {result.verdict!r} not in "
                "{'accept','marginal','reject','unknown'}"
            )
    return True, "validated against {} sample readings".format(len(sample_history))


def generate_and_validate_with_retry(
    *,
    description: str,
    additional_guidance: Optional[str],
    skill_context: Optional[str],
    model: str,
    api_key: str,
    session_dir: Path,
    reference_data: list[LatestData],
    skill_state: Optional[dict] = None,
    max_attempts: int = 3,
    start_version: int = 1,
    initial_prior_context: Optional[str] = None,
) -> tuple[Optional[Callable], int, list[str]]:
    """Generate a reading_fn, validate against reference data, retry on failure.

    Mirrors analyze-mode's verification loop: the LLM gets the failure
    reason + its previous attempt as context for the next try. Returns
    ``(reading_fn, final_version, attempt_log)``. ``reading_fn`` is None
    when all attempts exhausted.

    The ``reference_data`` list is the gate: the script must process at
    least one of these payloads without raising AND produce a
    well-shaped :class:`LiveReadingResult`. Pass the operator's stored
    reference spectrum (or the first reading captured by the session)
    here. An empty list falls through to the conservative
    "imported cleanly" pass.

    Args:
        description, additional_guidance, skill_context: forwarded to
            :func:`generate_reading_script` on each attempt.
        session_dir: where to save attempted scripts (one per version).
        reference_data: payloads the script must successfully process.
        skill_state: passed to :func:`validate_reading_script`.
        max_attempts: cap on retries; default 3.
        start_version: first version number to use. Subsequent attempts
            increment. Useful when resuming or adapting (caller passes
            current_version + 1).
        initial_prior_context: optional context for the FIRST attempt
            (e.g. "the previously-active script was rejected because X
            — generate a fresh one avoiding that").

    Returns:
        ``(reading_fn, version, attempts)`` — ``reading_fn`` is None on
        full failure; ``version`` is the version of the script that
        passed (or the last attempted); ``attempts`` is a list of
        human-readable failure reasons (length up to max_attempts;
        empty when the first attempt validated).
    """
    attempts: list[str] = []
    prior = initial_prior_context

    for attempt_idx in range(max_attempts):
        version = start_version + attempt_idx
        source = generate_reading_script(
            description=description,
            additional_guidance=additional_guidance,
            skill_context=skill_context,
            model=model,
            api_key=api_key,
            prior_context=prior,
        )
        if source is None:
            attempts.append(
                f"attempt v{version}: LLM code-gen returned no source"
            )
            prior = "\n".join(attempts)
            continue

        fn = save_and_load_reading_script(source, session_dir, version)
        if fn is None:
            attempts.append(
                f"attempt v{version}: generated script failed to import "
                "(syntax error or missing reading_fn)"
            )
            prior = "\n".join(attempts) + "\n\nPrevious source was:\n" + source
            continue

        ok, reason = validate_reading_script(fn, reference_data, skill_state)
        if ok:
            return fn, version, attempts
        attempts.append(f"attempt v{version}: {reason}")
        prior = (
            "\n".join(attempts)
            + "\n\nPrevious source was:\n" + source
            + "\n\nFix the issue and produce a new full script."
        )

    return None, start_version + max_attempts - 1, attempts


def latest_data_from_file(path: str | Path) -> Optional[LatestData]:
    """Build a :class:`LatestData` from a single file on disk.

    Convenience for the upfront-validation path: the operator points at
    a stored reference spectrum, and we wrap it in the same
    ``LatestData`` shape the data sources produce so codegen +
    validation can run against it.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = None
    return LatestData(
        timestamp=time.time(),
        source_kind="reference_file",
        path=p,
        text=text,
    )


__all__ = [
    "generate_reading_script",
    "save_and_load_reading_script",
    "validate_reading_script",
    "generate_and_validate_with_retry",
    "latest_data_from_file",
]
