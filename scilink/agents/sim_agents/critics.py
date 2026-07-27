"""Engine-neutral critic agents for simulation work.

This module provides two foundation agents that review simulation inputs
and outputs without engine-specific code. Engine knowledge (VASP INCAR
conventions, LAMMPS pair_style rules, etc.) is supplied at call time by
the active skill bundle's markdown sections, so adding support for a new
engine requires only a new skill bundle, not changes here.

Public agents:

    InputValidator
        Pre-run reviewer. Given proposed input files and a system
        description, returns a structured report of suggested adjustments.
        Reads the active skill's ``validation`` section.

    RunCritic
        Post-run reviewer. Given a finished run directory and the user's
        research goal, returns a verdict on the result and (when relevant)
        a set of proposed input patches. Handles both failed and successful
        runs in one pass. Reads the active skill's ``interpretation``
        section.

The agents share a small base, :class:`_CriticBase`, that handles LLM
client construction (proxy and public paths), skill section loading,
and tolerant JSON parsing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...auth import (
    get_internal_proxy_key,
    require_vendor_credentials,
)
from ...skills.loader import list_skills, load_skill
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ._deprecation import normalize_params


# ──────────────────────────────────────────────────────────────────────────
# Shared base
# ──────────────────────────────────────────────────────────────────────────

class _CriticBase:
    """Base class providing LLM client construction and skill access.

    Subclasses declare two class attributes:

        SKILL_SECTION
            The markdown section name to load from the active skill on
            each call (e.g. ``"validation"``, ``"interpretation"``).

        BASELINE_PROMPT_TEMPLATE
            The engine-neutral prompt template, with named placeholders
            that the subclass's public method fills in.

    The skill domain (e.g. ``"periodic_dft"``, ``"molecular_dynamics"``)
    is a call-time argument rather than a class attribute, so one instance
    can serve calls against different engine families in the same session.
    """

    SKILL_SECTION: str = ""
    BASELINE_PROMPT_TEMPLATE: str = ""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        """Construct the critic agent and its underlying LLM client.

        Args:
            api_key: API key for the LLM provider. When ``base_url`` is
                set, this is the internal-proxy key (or read from
                ``SCILINK_API_KEY``). When ``base_url`` is unset, the
                key is forwarded to LiteLLM, which falls back to the
                vendor's conventional environment variable
                (``ANTHROPIC_API_KEY`` etc.) when ``api_key`` is None.
            model_name: Model identifier, in the form expected by the
                resolved provider (e.g. ``"claude-opus-4-6"``).
            base_url: Base URL for an OpenAI-compatible internal proxy.
                When provided, requests are routed through the proxy
                client; when ``None``, requests go through LiteLLM.
            futurehouse_api_key: Optional FutureHouse (Edison) API key
                enabling literature-grounded review. Falls back to the
                ``FUTUREHOUSE_API_KEY`` environment variable. When no key
                is available, literature grounding is skipped and the
                critic runs on baseline guidance and engine tools only.
            google_api_key: Deprecated. Use ``api_key`` instead.
            local_model: Deprecated. Use ``base_url`` instead.

        Raises:
            ValueError: If ``base_url`` is set and no API key can be
                resolved from arguments or environment.
        """
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        import os as _os
        self.futurehouse_api_key = (
            futurehouse_api_key or _os.environ.get("FUTUREHOUSE_API_KEY")
        )

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source=self.__class__.__name__,
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy. Set "
                    "SCILINK_API_KEY in the environment or pass api_key."
                )
            self.logger.info(f"Using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url
            )
        else:
            if api_key is None:
                require_vendor_credentials(model_name)
            self.logger.info(f"Using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name, api_key=api_key
            )

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    # ── skill access ──────────────────────────────────────────────────

    def _load_skill_section(
        self,
        skill: Optional[str],
        domain: str,
    ) -> str:
        """Load a skill bundle and return its ``SKILL_SECTION`` content.

        Args:
            skill: Skill bundle name within ``domain``, or ``None``.
            domain: Skill domain subdirectory the bundle lives under.

        Returns:
            The section content prefixed by a labelled header, or an
            empty string if no skill was requested, the bundle is not
            found, or the section is empty.
        """
        if not skill:
            return ""
        try:
            parsed = load_skill(skill, domain=domain)
        except FileNotFoundError:
            available = list_skills(domain=domain)
            self.logger.warning(
                f"Skill '{skill}' not found in '{domain}'. "
                f"Available: {available}. Falling back to baseline."
            )
            return ""
        section = parsed.get(self.SKILL_SECTION, "") or ""
        skill_name = parsed.get("name", skill)
        if not section.strip():
            self.logger.info(
                f"Skill '{skill_name}' has no '{self.SKILL_SECTION}' "
                f"section; using baseline only."
            )
            return ""
        return (
            f"=== Engine knowledge from skill '{skill_name}' "
            f"({self.SKILL_SECTION}) ===\n{section}"
        )

    # ── LLM helpers ───────────────────────────────────────────────────

    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM requesting JSON output and parse the response.

        Tolerates LLM responses that wrap JSON in code fences or
        surrounding prose by falling back to a brace-balanced extraction.

        Args:
            prompt: The complete prompt to send to the model.

        Returns:
            The parsed JSON object, or an error dict with
            ``status="error"`` and the raw response when parsing fails.
        """
        import re
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        self.logger.error(
            f"Could not parse LLM response as JSON. "
            f"First 400 chars: {text[:400]!r}"
        )
        return {
            "status": "error",
            "error": "Could not parse LLM response as JSON.",
            "raw_response": text[:2000],
        }


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _format_input_files(input_files: Dict[str, str]) -> str:
    """Render an input-files mapping as fenced markdown sections.

    Args:
        input_files: Mapping of filename to file content.

    Returns:
        A single string with each file rendered under a ``=== name ===``
        header. Individual file contents are truncated at 8000 characters
        with a trailing ``[truncated]`` marker.
    """
    chunks = []
    for name, content in input_files.items():
        body = (content or "").rstrip()
        if len(body) > 8000:
            body = body[:8000] + "\n... [truncated]"
        chunks.append(f"=== {name} ===\n{body}")
    return "\n\n".join(chunks)


_SNAPSHOT_TOOL_NAME = "snapshot_run"
_SYNTAX_TOOL_NAME = "check_input_syntax"


def _run_deterministic_syntax_check(
    input_files: Dict[str, str],
    skill: Optional[str],
) -> List[Dict[str, Any]]:
    """Run the active skill's deterministic pre-run syntax check, if any.

    Looks up a ``check_input_syntax`` callable in the active skill bundle
    and invokes it on the input files. The engine's tool selects whichever
    file it knows how to check and returns a list of issue dicts.

    Args:
        input_files: Mapping of input filename to file contents.
        skill: Name of the active skill bundle, or ``None``.

    Returns:
        The issue list from the engine's syntax check, or an empty list
        when no skill is active or the active skill registers no
        ``check_input_syntax`` tool. Never raises — a missing tool means
        the engine offers no deterministic syntax pass.
    """
    if not skill:
        return []
    from ...skills._shared._registry import get_tool_function
    try:
        checker = get_tool_function(
            _SYNTAX_TOOL_NAME, active_skills=[skill]
        )
    except LookupError:
        return []
    try:
        result = checker(input_files=input_files)
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Deterministic syntax check raised for skill '{skill}': {e}"
        )
        return []
    return result if isinstance(result, list) else []


def _has_runnable_content(text: str) -> bool:
    """Whether a proposed input file has any non-comment, non-blank line.

    Scoped to the engines on the executor path today (LAMMPS, VASP), whose
    comment markers are ``#`` and ``!``. This is NOT fully engine-neutral: a
    GROMACS ``.top`` comments with ``;`` and treats ``#include`` / ``#ifdef`` as
    directives (runnable, not comments), so this heuristic would misjudge one —
    handle those markers here before putting a GROMACS-style engine on the
    executor path. A whole-file fix that is only comments/blank is a no-op deck —
    running it does nothing — so it is not a real fix.
    """
    return any(
        s and not s.startswith(("#", "!"))
        for s in (line.strip() for line in text.splitlines())
    )


def _drop_vacuous_fix(report: Dict[str, Any]) -> None:
    """Reject a ``suggested_fixes`` set containing a vacuous (all-comment) file.

    The post-run critic can return an "explanation as comments" deck with no
    executable commands; running it wastes a refine cycle on a no-op (returncode
    0, no output) that then re-reads as ``needs_fixes``. ``suggested_fixes`` are
    whole-file replacements, so a file with no runnable line is not a usable fix
    — drop the whole proposal (the loop then stops cleanly rather than re-running
    nothing) and record why.
    """
    fixes = report.get("suggested_fixes")
    if not isinstance(fixes, dict) or not fixes:
        return
    vacuous = [name for name, text in fixes.items()
               if isinstance(text, str) and not _has_runnable_content(text)]
    if vacuous:
        report["suggested_fixes"] = None
        note = (
            f"Discarded the proposed fix: file(s) {vacuous} contain no "
            "executable commands (comments/blank only) — not a runnable input."
        )
        report["diagnostic_notes"] = (
            f"{report.get('diagnostic_notes') or ''} {note}".strip()
        )
        logging.getLogger(__name__).warning(note)


def _format_syntax_issues(issues: List[Dict[str, Any]]) -> str:
    """Render deterministic syntax issues as a prompt block.

    Args:
        issues: Issue dicts from a ``check_input_syntax`` call.

    Returns:
        A labelled markdown block listing each issue, or an empty string
        when there are no issues (so callers can concatenate freely).
    """
    if not issues:
        return ""
    lines = [
        "=== Deterministic syntax check (authoritative — already run) ===",
        "These tag-level issues were found by an engine-native syntax "
        "checker, not by you. Treat them as ground truth and fold them "
        "into your assessment; do not re-litigate tag spellings.",
    ]
    for it in issues:
        tag = it.get("tag")
        suggested = it.get("suggested")
        confidence = it.get("confidence", "")
        desc = it.get("description", "")
        lines.append(
            f"- tag={tag!r} suggested={suggested!r} "
            f"confidence={confidence}: {desc}"
        )
    return "\n".join(lines)


# Heuristics for the generic snapshot fallback.
_SNAPSHOT_TAIL_LINES = 80
_SNAPSHOT_MAX_FILE_BYTES = 256_000
_SNAPSHOT_MAX_FILES = 40
_SNAPSHOT_SNIFF_BYTES = 8192


def _looks_like_text(path: Path) -> bool:
    """Whether a file should be tailed into the generic snapshot.

    Engine-neutral: classifies by content, not by suffix, so any textual log
    is surfaced (``log.lammps``, ``OUTCAR``, ``stdout``) while binaries
    (restart files, trajectories, ``.h5``) are skipped. A file is text if it
    is under the size cap and its leading bytes contain no NUL byte.
    """
    try:
        if path.stat().st_size > _SNAPSHOT_MAX_FILE_BYTES:
            return False
        with open(path, "rb") as fh:
            return b"\x00" not in fh.read(_SNAPSHOT_SNIFF_BYTES)
    except OSError:
        return False


def _generic_snapshot(output_dir: str) -> Dict[str, Any]:
    """Build an engine-neutral snapshot of a run directory.

    Lists the directory's files and tails the text/log-like ones so a critic
    can read real output even when the active skill registers no engine
    ``snapshot_run`` parser (the prose-only case). The skill's
    ``interpretation`` prose tells the LLM what to look for in this content;
    this function only surfaces it. No engine names or filenames are assumed.

    Args:
        output_dir: Path to the run directory.

    Returns:
        A dict with a ``files`` listing (name + size) and a ``file_tails``
        mapping of filename to its last lines, or an ``error`` if the
        directory is unreadable.
    """
    out = Path(output_dir)
    if not out.exists():
        return {"error": f"output_dir does not exist: {output_dir}"}

    files: List[Dict[str, Any]] = []
    tails: Dict[str, str] = {}
    try:
        entries = sorted(p for p in out.iterdir() if p.is_file())
    except OSError as e:
        return {"error": f"could not list {output_dir}: {e}"}

    for path in entries[:_SNAPSHOT_MAX_FILES]:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files.append({"name": path.name, "size_bytes": size})
        try:
            if _looks_like_text(path):
                text = path.read_text(errors="replace")
                lines = text.splitlines()
                tails[path.name] = "\n".join(lines[-_SNAPSHOT_TAIL_LINES:])
        except OSError:
            continue

    return {
        "snapshot_kind": "generic",
        "note": (
            "No engine-specific output parser was registered for the active "
            "skill; this is a generic directory snapshot. Read the file "
            "tails below using the interpretation guidance for this engine."
        ),
        "files": files,
        "file_tails": tails,
    }


def _snapshot_run_outputs(output_dir: str, skill: Optional[str]) -> Dict[str, Any]:
    """Parse a finished run's output files into a structured snapshot.

    Looks up a ``snapshot_run`` callable in the active skill bundle via
    :func:`scilink.skills._shared._registry.get_tool_function`. The
    callable lives alongside its skill markdown (e.g.
    ``scilink/skills/periodic_dft/vasp/vasp_output.py``) and owns its own
    output shape; this function delegates without inspecting the result.

    When no skill is active or the active skill registers no
    ``snapshot_run`` callable, falls back to :func:`_generic_snapshot` — a
    file listing plus text/log tails — so the critic always receives real
    output to assess rather than only a note.

    Args:
        output_dir: Path to the directory containing run output files.
        skill: Name of the active skill bundle whose parser should be
            invoked.

    Returns:
        The skill parser's structured snapshot, or a generic directory
        snapshot when no parser is available.
    """
    if not skill:
        return _generic_snapshot(output_dir)
    from ...skills._shared._registry import get_tool_function
    try:
        parser = get_tool_function(
            _SNAPSHOT_TOOL_NAME, active_skills=[skill]
        )
    except LookupError:
        return _generic_snapshot(output_dir)
    return parser(output_dir)


# ──────────────────────────────────────────────────────────────────────────
# InputValidator
# ──────────────────────────────────────────────────────────────────────────

_INPUT_VALIDATOR_PROMPT = """\
You are a simulation input reviewer. Identify potential issues in the
proposed input files BEFORE the user commits compute resources to running
the calculation. Be concrete: flag what to change, why, and how serious it
is.

{skill_context}

{syntax_block}

{literature_block}

=== Proposed input files ===
{input_files}

=== System description (what the user is trying to compute) ===
{system_description}

Return a JSON object with these fields:
  status              "success" | "error"
  validation_status   "passes" | "needs_revision" | "fails"
  overall_assessment  2-3 sentence prose summary
  suggested_adjustments  list of objects, each:
      {{ "file": str,           // which input file
         "key": str,            // parameter / tag / line identifier
         "current": str,        // current value (or "missing")
         "suggested": str,      // proposed value
         "severity": "info" | "warning" | "error",
         "reason": str }}       // 1-2 sentences
  review_basis        prose: which engine conventions, system specifics,
                      or general best practice drove your call

If the inputs look correct, return validation_status="passes" with an
empty suggested_adjustments list — do not invent issues.
"""


class InputValidator(_CriticBase):
    """Pre-run reviewer for simulation input files.

    Reviews proposed input files against engine conventions and returns
    a structured report of suggested adjustments before the user submits
    the calculation. The engine-neutral baseline prompt frames the
    reasoning; engine-specific conventions are supplied by the active
    skill's ``validation`` section.

    Example:
        >>> validator = InputValidator(api_key=key, model_name=model)
        >>> result = validator.validate(
        ...     input_files={"INCAR": incar_text, "KPOINTS": kp_text},
        ...     system_description="Fe BCC, magnetic ground state at 0 K",
        ...     skill="vasp",
        ...     domain="periodic_dft",
        ... )
        >>> result["validation_status"]
        'needs_revision'
    """

    SKILL_SECTION = "validation"
    BASELINE_PROMPT_TEMPLATE = _INPUT_VALIDATOR_PROMPT

    def _literature_review(
        self,
        input_files: Dict[str, str],
        system_description: str,
        skill: Optional[str],
    ) -> str:
        """Return a literature-grounded review of the inputs, or ``""``.

        Runs only when a FutureHouse key is configured. Builds an
        engine-neutral query (the engine name is the active skill name)
        and returns the literature answer text for folding into the
        prompt and report. Failure-isolated: any error returns an empty
        string so literature trouble never blocks the review.

        Args:
            input_files: Mapping of input filename to contents.
            system_description: What the inputs are for.
            skill: Active engine skill name, used as the engine label.

        Returns:
            The literature review text, or an empty string when no key is
            configured or the search did not succeed.
        """
        if not self.futurehouse_api_key:
            return ""
        engine_label = (skill or "the engine").upper()
        try:
            from ..lit_agents.literature_agent import IncarLiteratureAgent
            agent = IncarLiteratureAgent(
                api_key=self.futurehouse_api_key,
                # 50-min ceiling, matching every other literature call site
                # (Edison CROW jobs routinely take 20-30 min; the class
                # default of 300s timed out on essentially every real query,
                # silently disabling INCAR literature validation).
                max_wait_time=3000,
            )
            result = agent.validate_inputs(
                input_files_text=_format_input_files(input_files),
                system_description=system_description,
                engine_label=engine_label,
            )
        except Exception as e:
            self.logger.warning(f"Literature review skipped: {e}")
            return ""
        if result.get("status") != "success":
            self.logger.info(
                f"Literature review unavailable: {result.get('message', result.get('status'))}"
            )
            return ""
        return (result.get("response") or "").strip()

    def validate(
        self,
        input_files: Dict[str, str],
        system_description: str,
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Review proposed input files and return a structured report.

        Args:
            input_files: Mapping of input filename to file contents.
                Filenames are engine-defined (e.g. ``INCAR`` and
                ``KPOINTS`` for VASP, ``input.lmp`` for LAMMPS). An
                empty mapping returns an error report.
            system_description: A natural-language description of the
                system being computed and the scientific objective, used
                to judge whether parameter choices are appropriate.
            skill: Name of the skill bundle within ``domain`` whose
                ``validation`` section should be loaded. When ``None``,
                the baseline prompt runs without engine-specific context.
            domain: Skill subdirectory the bundle lives under, e.g.
                ``"periodic_dft"`` or ``"molecular_dynamics"``. Required
                when ``skill`` is provided; ignored when ``skill`` is
                ``None``.

        Returns:
            A report dict with fields:

                status              ``"success"`` or ``"error"``
                validation_status   ``"passes"``, ``"needs_revision"``,
                                    or ``"fails"``
                overall_assessment  Prose summary
                suggested_adjustments
                                    List of adjustment dicts; each has
                                    ``file``, ``key``, ``current``,
                                    ``suggested``, ``severity``, ``reason``
                review_basis        Prose explanation of what guided
                                    the call

        Raises:
            ValueError: If ``skill`` is provided without ``domain``.
        """
        if not input_files:
            return {
                "status": "error",
                "error": "input_files is empty — nothing to validate.",
            }
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided. "
                "Pass the skill subdirectory (e.g. 'periodic_dft' or "
                "'molecular_dynamics') alongside the skill name."
            )

        skill_context = self._load_skill_section(skill, domain or "")

        # Run the engine's deterministic syntax check first and pass its
        # findings to the LLM as authoritative grounding, so the model
        # reasons about physics rather than re-checking tag spellings.
        syntax_issues = _run_deterministic_syntax_check(input_files, skill)

        # Ground the review in literature when a FutureHouse key is
        # configured; otherwise this is an empty string and the review
        # proceeds on baseline guidance + the syntax check.
        literature = self._literature_review(input_files, system_description, skill)
        literature_block = (
            f"=== Literature review (for grounding parameter choices) ===\n{literature}"
            if literature else ""
        )

        prompt = self.BASELINE_PROMPT_TEMPLATE.format(
            skill_context=skill_context or "(no engine skill loaded)",
            syntax_block=_format_syntax_issues(syntax_issues),
            literature_block=literature_block,
            input_files=_format_input_files(input_files),
            system_description=system_description,
        )
        report = self._generate_json(prompt)
        report.setdefault("status", "success")
        # Surface the deterministic findings + literature on the report
        # regardless of what the LLM did with them, so callers have the
        # ground truth and the source material.
        report["syntax_check"] = syntax_issues
        if literature:
            report["literature_review"] = literature
        return report


# ──────────────────────────────────────────────────────────────────────────
# RunCritic
# ──────────────────────────────────────────────────────────────────────────

_RUN_CRITIC_PROMPT = """\
You are a post-run simulation critic. The user has finished a calculation
and needs an assessment: did the run produce what they wanted, and if not,
what should they change? Handle both cases — a failed run (propose fixes)
and a successful run (give a verdict and sanity-check the physics).

{skill_context}

=== Output directory ===
{output_dir}

=== Output snapshot (parsed from the run directory) ===
{output_snapshot}

{input_files_block}

=== Research goal (what the user was trying to compute) ===
{research_goal}

{fixes_directive}

{observable_coverage}
Return a JSON object with these fields:
  status              "success" | "error"
  run_status          "succeeded" | "failed" | "incomplete"
  failure_class       "deck" | "structure" | "force_field" | null
                      The ROOT cause. Set when a run FAILED, or when a run
                      CONVERGED but the result is physically unsound (verdict
                      "poor"). null when the result is acceptable.
                      "structure" — caused by the initial atomic configuration
                      itself (a broken / overlapping pack), so NO change to the
                      run inputs can fix it and it must be regenerated; a tell is
                      a non-finite or absurd energy / pressure at the very first
                      step that persists even though the deck already minimizes.
                      "deck" — anything a corrected input file fixes (a setting,
                      a style, an unstable timestep, an atom leaving the grid
                      mid-run). "force_field" — the run completed but a computed
                      property contradicts the known physical behaviour of the
                      system or its components, and the cause is the model's
                      parameters, not the run settings, so no input-deck change
                      can fix it. When "structure" OR "force_field", set
                      suggested_fixes to null: the structure must be regenerated
                      / the force field re-parameterized, not patched by the deck.
  verdict             "good" | "warning" | "poor" | "needs_fixes"
                      good        — converged, physically sensible
                      warning     — converged but with concerns
                      poor        — converged but result is suspect or wrong
                      needs_fixes — did not converge or failed to run
                      Judge "poor" by reasoning about whether the computed
                      properties and their trends are consistent with the known
                      physical behaviour of this system and its components: a run
                      that completes cleanly but contradicts well-established
                      behaviour is "poor", not "good".
  reasoning           prose summary (3-6 sentences)
  suggested_fixes     {{ "filename": "complete_corrected_file", ... }} | null
                      Provide a non-null dict only when the verdict is
                      "poor" or "needs_fixes", run_status is "failed", OR
                      `missing_observables` is non-empty. Otherwise return null.
                      Each key MUST be the exact name of one of the input
                      files shown above — do not invent a new file, a
                      ".patch", or a "header to insert after read_data".
                      Each value MUST be the COMPLETE corrected file (the
                      whole input, ready to run as-is), never a diff,
                      snippet, or fragment to splice in.
  missing_observables list of {{ "property": ..., "required_output": ...,
                      "reason": ... }} | null
                      Populated ONLY when an observable-coverage check is
                      requested above. BLOCKING gaps only: a property the goal
                      requires whose data the run would not capture and could
                      not be reconstructed post-hoc from what the deck saves
                      (permanently lost). null/empty otherwise.
  advisory_observables same shape | null
                      NON-BLOCKING: goal-relevant properties that are absent
                      but recoverable in post-processing from what the deck
                      already saves, or optional/secondary. Informational —
                      do not gate on these. null/empty when none.
  recommendations     list of short strings — next steps the user should
                      consider (rerun with X, gather more data, etc.)
  diagnostic_notes    optional prose — specific log lines, energies,
                      forces, or convergence trends that informed the verdict
"""


class RunCritic(_CriticBase):
    """Post-run reviewer for finished simulation calculations.

    Inspects a finished run's output directory and returns a verdict
    on the result. When the run failed or the result is unsatisfactory,
    proposes patched input files that the user can resubmit. A single
    call covers both the convergence / runtime-error question and the
    physical-quality question.

    Engine-specific output parsers and error patterns are supplied by
    the active skill's ``interpretation`` section and the dispatched
    snapshot reader in :func:`_snapshot_run_outputs`.

    Example:
        >>> critic = RunCritic(api_key=key, model_name=model)
        >>> result = critic.assess(
        ...     output_dir="/path/to/run",
        ...     research_goal="Bulk Si lattice parameter from relaxation",
        ...     skill="vasp",
        ...     domain="periodic_dft",
        ... )
        >>> result["verdict"]
        'good'
    """

    SKILL_SECTION = "interpretation"
    BASELINE_PROMPT_TEMPLATE = _RUN_CRITIC_PROMPT

    def assess(
        self,
        output_dir: str,
        research_goal: str,
        skill: Optional[str] = None,
        domain: Optional[str] = None,
        fixes_mode: str = "auto",
        input_files: Optional[Dict[str, str]] = None,
        check_observables: bool = False,
    ) -> Dict[str, Any]:
        """Assess a finished run and return a verdict report.

        Args:
            output_dir: Path to the directory containing the finished
                run's output files. Contents are engine-specific
                (e.g. ``vasprun.xml``, ``OUTCAR`` for VASP;
                ``log.lammps`` for LAMMPS).
            research_goal: A natural-language description of what the
                user was trying to compute. Drives whether the result
                is sufficient for the intent.
            skill: Name of the skill bundle within ``domain`` whose
                ``interpretation`` section should be loaded. When
                ``None``, the baseline prompt runs without
                engine-specific context.
            domain: Skill subdirectory the bundle lives under, e.g.
                ``"periodic_dft"`` or ``"molecular_dynamics"``. Required
                when ``skill`` is provided; ignored when ``skill`` is
                ``None``. Also used to dispatch the output snapshot
                parser.
            fixes_mode: Controls when ``suggested_fixes`` may be
                populated:

                    ``"auto"`` (default)
                        Propose fixes only when ``run_status`` is
                        ``"failed"`` or ``verdict`` is ``"poor"`` /
                        ``"needs_fixes"``.
                    ``"always"``
                        Propose fixes whenever the verdict is below
                        ``"good"``.
                    ``"skip"``
                        Never propose fixes; ``suggested_fixes`` is
                        forced to ``None`` regardless of verdict.
            input_files: Optional mapping of input filename to contents.
                When provided, the files are shown to the critic so a
                proposed fix patches the real inputs by their real names
                with complete contents, rather than the critic guessing a
                filename and emitting a fragment.
            check_observables: When True, the critic additionally verifies
                that the run captures the data needed for the properties the
                research goal calls for. It separates BLOCKING gaps —
                required data that would be permanently lost (reported in
                ``missing_observables``, with a corrected deck in
                ``suggested_fixes`` when input_files are provided) — from
                recoverable-in-post-processing or optional ones (reported in
                ``advisory_observables``, non-blocking). Intended for the
                pre-run gate; defaults to False.

        Returns:
            A report dict with fields:

                status              ``"success"`` or ``"error"``
                run_status          ``"succeeded"``, ``"failed"``, or
                                    ``"incomplete"``
                verdict             ``"good"``, ``"warning"``, ``"poor"``,
                                    or ``"needs_fixes"``
                reasoning           Prose summary
                suggested_fixes     Mapping of filename to patched
                                    content, or ``None``
                recommendations     List of next-step strings
                diagnostic_notes    Optional prose on specific signals

        Raises:
            ValueError: If ``skill`` is provided without ``domain``.

        Notes:
            Missing output directories and parse failures surface as
            ``status="error"`` entries in the returned report rather
            than raised exceptions.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided. "
                "Pass the skill subdirectory (e.g. 'periodic_dft' or "
                "'molecular_dynamics') alongside the skill name."
            )
        out_path = Path(output_dir)
        if not out_path.exists():
            return {
                "status": "error",
                "error": f"output_dir does not exist: {output_dir}",
            }

        snapshot = _snapshot_run_outputs(str(out_path), skill)
        skill_context = self._load_skill_section(skill, domain or "")

        # When the caller supplies the input files, show them so a proposed
        # fix patches the real files by their real names with full contents —
        # rather than the critic guessing a filename and emitting a fragment.
        if input_files:
            input_files_block = (
                "=== Current input files (patch THESE — reuse the exact "
                "filenames, return the whole file) ===\n"
                + _format_input_files(input_files)
            )
        else:
            input_files_block = (
                "=== Current input files ===\n(not provided to this critic — "
                "if you must propose a fix, key it by the exact input filename "
                "referenced in the snapshot above and return the COMPLETE file)"
            )

        fixes_directive = {
            "auto": (
                "Propose fixes only when run_status is 'failed' or verdict "
                "is 'poor' or 'needs_fixes'."
            ),
            "always": (
                "Propose fixes whenever the verdict is below 'good'."
            ),
            "skip": (
                "Do NOT propose fixes — set suggested_fixes to null "
                "regardless of verdict."
            ),
        }.get(
            fixes_mode,
            "Propose fixes only when run_status is 'failed' or verdict "
            "is 'poor' or 'needs_fixes'.",
        )

        if check_observables:
            observable_coverage = (
                "=== Observable-coverage check (pre-run) ===\n"
                "Determine which physical properties the research goal requires. "
                "A gap is BLOCKING only when the raw data needed to compute the "
                "property is not being written at all and cannot be reconstructed "
                "after the run. Apply this test:\n"
                "- A saved trajectory of coordinates/velocities lets you recompute "
                "geometry- and motion-based properties in post-processing "
                "(structural, dynamical, correlation functions). Whenever any "
                "trajectory is dumped, these are NON-blocking — no matter how "
                "coarsely it is sampled.\n"
                "- A quantity that must be accumulated DURING the run from "
                "per-step forces, virial, or energy is NOT contained in a "
                "coordinate/velocity trajectory and cannot be rebuilt afterward. "
                "If the goal requires such a quantity and its log is absent, that "
                "is BLOCKING.\n"
                "- If no trajectory is saved at all, geometry/motion properties "
                "become blocking too (nothing to post-process).\n"
                "Resolution / sampling frequency is NEVER blocking.\n"
                "- `missing_observables` (BLOCKING): required properties whose "
                "raw data the run does not write at all. Only these justify "
                "regenerating the deck before submission.\n"
                "- `advisory_observables` (NON-BLOCKING): properties derivable in "
                "post-processing from what the deck saves (including if "
                "under-resolved), or optional/secondary. Report for awareness; do "
                "NOT block.\n"
                "Judge only what the goal actually requires; do not invent "
                "observables. When `missing_observables` is non-empty, also return "
                "`suggested_fixes`: the COMPLETE corrected deck(s) that add the "
                "missing output(s), keyed by the exact input filename."
            )
        else:
            observable_coverage = ""

        prompt = self.BASELINE_PROMPT_TEMPLATE.format(
            skill_context=skill_context or "(no engine skill loaded)",
            output_dir=str(out_path),
            output_snapshot=json.dumps(snapshot, indent=2, default=str)[:12000],
            input_files_block=input_files_block,
            research_goal=research_goal,
            fixes_directive=fixes_directive,
            observable_coverage=observable_coverage,
        )
        report = self._generate_json(prompt)
        report.setdefault("status", "success")
        if fixes_mode == "skip":
            report["suggested_fixes"] = None
        else:
            _drop_vacuous_fix(report)
        return report


# ──────────────────────────────────────────────────────────────────────────
# ReferencePropertyCritic
# ──────────────────────────────────────────────────────────────────────────

_REFERENCE_CRITIC_PROMPT = """\
You are validating a simulation's model BEFORE an expensive production run.
Below are reference properties measured for each pure component of the system,
computed with the SAME force field the production run will use. Independently of
any target result, judge each measured value against the KNOWN physical
behaviour of that component.

{skill_context}

=== System ===
{system_description}

=== Measured reference properties (this force field) ===
{measurements_block}

For EACH measured value, decide whether it is consistent with the
well-established behaviour of that substance. Reason from what is known about the
component; a stored reference value, when provided, is an anchor, not the only
basis. A value that clearly contradicts known behaviour means the underlying
model is miscalibrated, and any prediction built on it is untrustworthy — no
change to the run settings can fix that. If a value is only mildly off, or you
are unsure, treat it as consistent: this gate must not veto a sound model over a
surprising-but-plausible number. But judge whether the VALUE is correct, not
whether its error is surprising: a value that is clearly wrong is inconsistent
even when the error is a well-known or expected limitation of this class of model
— "expected" is not "acceptable", because the prediction built on it is still
untrustworthy. The benefit of the doubt is only for values genuinely close to
correct, or surprising but real — never for a large, confirmed error.

Return a JSON object:
  status           "success"
  per_measurement  list of {{ "component": ..., "property": ...,
                   "consistent": true|false, "reasoning": "<one sentence>" }} —
                   one entry per MEASURED value shown above (a component checked
                   on two properties gets two entries)
  verdict          "good" — every measured value is physically consistent
                   "poor" — at least one clearly contradicts known behaviour
  failure_class    when verdict is "poor", the miscalibrated model — name what
                   the system actually uses: "force_field" (classical MD),
                   "functional" (DFT), or "potential" (machine-learning
                   interatomic potential). null when verdict is "good".
  reasoning        short prose summary (which value, and why)
"""


class ReferencePropertyCritic(_CriticBase):
    """Reasons over pre-run reference-property measurements to decide whether
    the force field is trustworthy before the production run.

    Given each pure component's measured reference property (from the
    engine-neutral :func:`reference_validation.validate_component_properties`
    stage), it judges each value against the known behaviour of that substance
    and, when one clearly contradicts it, returns a ``poor`` verdict naming the
    miscalibrated model in the system's own terms — ``failure_class`` is
    ``"force_field"`` for classical MD, ``"functional"`` for DFT, or
    ``"potential"`` for a machine-learning interatomic potential. The MD value
    matches the post-run :class:`RunCritic`, so both feed one reparameterization
    fixer.

    Reasoning-first: the judgement rests on the model's knowledge of the
    components (a skill's ``validation`` section can supply known values as an
    anchor, but is not required), and it is deliberately conservative — a mildly
    off or merely surprising value is treated as consistent, so a sound model is
    never vetoed over an unexpected result.
    """

    SKILL_SECTION = "validation"
    BASELINE_PROMPT_TEMPLATE = _REFERENCE_CRITIC_PROMPT

    def assess(
        self,
        measurements: List[Dict[str, Any]],
        system_description: str = "",
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Judge measured reference properties; flag the force field if a value
        contradicts known behaviour.

        Args:
            measurements: The ``measurements`` list from
                ``validate_component_properties`` — entries with
                ``status="measured"`` carry ``value`` / ``units``; unmeasured
                ones are shown for context but not judged.
            system_description: What is being simulated (context for the judge).
            skill, domain: Optional skill bundle whose ``validation`` section
                supplies known reference behaviour as an anchor.

        Returns:
            ``{"status", "verdict", "failure_class", "per_measurement",
            "reasoning"}``. ``per_measurement`` has one entry per measured value
            (``{"component", "property", "consistent", "reasoning"}``);
            ``failure_class`` names the miscalibrated model in the system's own
            terms (``"force_field"`` / ``"functional"`` / ``"potential"``).
            Fails open to a non-blocking ``good`` verdict (no LLM call) when no
            component was measured.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        measured = [m for m in (measurements or [])
                    if m.get("status") == "measured"]
        if not measured:
            return {
                "status": "success",
                "verdict": "good",
                "failure_class": None,
                "per_measurement": [],
                "reasoning": "No reference properties were measured; nothing to "
                             "validate.",
            }

        lines = []
        for m in measurements:
            if m.get("status") == "measured":
                prop = f" — {m['property']}" if m.get("property") else ""
                unit = f" {m['units']}" if m.get("units") else ""
                smi = f" [{m['smiles']}]" if m.get("smiles") else ""
                lines.append(f"- {m['component']}{prop}{smi}: {m['value']}{unit}")
            else:
                lines.append(f"- {m.get('component')}: not measured "
                             f"({m.get('error', 'unknown')})")
        measurements_block = "\n".join(lines)

        skill_context = self._load_skill_section(skill, domain or "")
        prompt = self.BASELINE_PROMPT_TEMPLATE.format(
            skill_context=skill_context or "(no engine skill loaded)",
            system_description=system_description or "(not provided)",
            measurements_block=measurements_block,
        )
        report = self._generate_json(prompt)
        report.setdefault("status", "success")
        report.setdefault("verdict", "good")
        report.setdefault("failure_class", None)
        report.setdefault("per_measurement", [])
        return report


# ──────────────────────────────────────────────────────────────────────────
# ReferencePropertySelector
# ──────────────────────────────────────────────────────────────────────────

_REFERENCE_SELECTOR_PROMPT = """\
You are validating a simulation's force field BEFORE an expensive production
run. For EACH component of the system below, choose the SINGLE reference
property that best validates the force field for that component.

{skill_context}

=== System ===
{system_description}

=== Components ===
{components_block}

A good reference property is one that (a) has an independently KNOWN correct
value for that substance — from experiment or well-established data — so a
measured value can be judged, and (b) can be measured by a short standalone
simulation of the pure component. Choose the property that most sharply exposes
a miscalibration for THIS kind of substance. As a guide, not a rule: a molecular
liquid is usually best checked by its mass density; a crystalline solid by a
lattice constant; a small rigid molecule by a characteristic bond length or
angle. If a component has no independently-known reference property worth
checking, mark it not measurable and say why.

Return a JSON object:
  status       "success"
  selections   list of {{ "component": ..., "property": "<name>",
               "measurable": true|false, "rationale": "<one sentence>" }} — one
               entry per component; "property" may be null when "measurable" is
               false
"""


class ReferencePropertySelector(_CriticBase):
    """Chooses which reference property to validate for each component.

    Picks the property whose correct value is independently known and that a
    short pure-component simulation can measure — density for a molecular
    liquid, a lattice constant for a crystal, a characteristic bond length for a
    small molecule, and so on. This is what keeps the validation general: the
    downstream measurement and the judging critic never hardcode a property;
    this step decides, per component, what is worth checking.
    """

    SKILL_SECTION = "validation"
    BASELINE_PROMPT_TEMPLATE = _REFERENCE_SELECTOR_PROMPT

    def select(
        self,
        components: List[Dict[str, Any]],
        system_description: str = "",
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Choose a reference property per component.

        Args:
            components: ``[{"name", "smiles", "role"?}, ...]``. ``role`` is
                optional free text (e.g. "solvent", "cation").
            system_description: What is being simulated (context for the choice).
            skill, domain: Optional skill bundle whose ``validation`` section
                supplies domain guidance on what to check.

        Returns:
            ``{"status", "selections": [{"component", "property", "measurable",
            "rationale"}, ...]}``. Empty selections (no LLM call) when no
            components are supplied.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        if not components:
            return {"status": "success", "selections": []}

        lines = []
        for c in components:
            smi = f" [{c['smiles']}]" if c.get("smiles") else ""
            role = f" — {c['role']}" if c.get("role") else ""
            lines.append(f"- {c.get('name') or c.get('smiles')}{smi}{role}")
        components_block = "\n".join(lines)

        skill_context = self._load_skill_section(skill, domain or "")
        prompt = self.BASELINE_PROMPT_TEMPLATE.format(
            skill_context=skill_context or "(no engine skill loaded)",
            system_description=system_description or "(not provided)",
            components_block=components_block,
        )
        report = self._generate_json(prompt)
        report.setdefault("status", "success")
        report.setdefault("selections", [])
        return report


# ──────────────────────────────────────────────────────────────────────────
# ReparameterizationAdvisor
# ──────────────────────────────────────────────────────────────────────────

_REPARAM_ADVISOR_PROMPT = """\
A pre-run validation flagged the force field: one or more pure-component
reference properties contradict known behaviour, so the model is untrustworthy
for production. Recommend a concrete corrective action so it can be fixed and
re-validated before the run — no production compute is spent until it passes.

{skill_context}

=== System ===
{system_description}

=== Current force-field backend ===
{backend}

=== Flagged reference properties ===
{flagged_block}

Reason about the LIKELY cause of each flagged value — partial charges, van der
Waals / Lennard-Jones terms, bonded/torsion terms, or a chemistry the base force
field does not really cover — and recommend ONE concrete corrective action:
- "add_force_field": supplement or replace the offending component's parameters
  with a validated set (e.g. a literature model for that chemistry), supplied
  through the backend's extra-force-field channel.
- "adjust_parameters": change specific named terms — for a targeted issue.
- "switch_backend": use a backend that covers this chemistry better.
- "escalate": no confident automatic fix — hand to the human with the diagnosis.

Recommend the human confirm or supply the concrete parameters unless the fix is
unambiguous; a wrong "fix" that still fails wastes another validation cycle.

Return a JSON object:
  status              "success"
  diagnosis           which component/property, and the likely force-field cause
  recommended_action  one of "add_force_field" | "adjust_parameters" |
                      "switch_backend" | "escalate"
  detail              concrete specifics (what to add / change / switch to)
  requires_human      true|false — whether a human must supply or approve the fix
  rationale           short prose
"""


class ReparameterizationAdvisor(_CriticBase):
    """Recommends how to fix a force field that the pre-run check flagged.

    Given the flagged pure-component reference properties (the inconsistent
    entries from :class:`ReferencePropertyCritic`), it reasons about the likely
    parameter-level cause and proposes a concrete corrective action —
    supplementing the component's parameters, adjusting named terms, switching
    backend, or escalating to a human when there is no confident automatic fix.
    It advises; it does not apply the fix, and (matching the human-in-the-loop
    use-case contract) flags when a human must supply or approve it.
    """

    SKILL_SECTION = "validation"
    BASELINE_PROMPT_TEMPLATE = _REPARAM_ADVISOR_PROMPT

    def advise(
        self,
        flagged: List[Dict[str, Any]],
        system_description: str = "",
        backend: str = "",
        skill: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recommend a corrective action for the flagged reference properties.

        Args:
            flagged: The inconsistent measurements — each
                ``{"component", "property", "reasoning", ...}`` (the ``poor``
                entries the critic returned).
            system_description: What is being simulated.
            backend: The force-field backend in use (e.g. ``"openff"``).
            skill, domain: Optional skill bundle whose ``validation`` section
                supplies domain guidance on likely causes and fixes.

        Returns:
            ``{"status", "diagnosis", "recommended_action", "detail",
            "requires_human", "rationale"}``. When ``flagged`` is empty, returns
            a no-op ``escalate`` with no LLM call.
        """
        if skill and not domain:
            raise ValueError(
                "domain is required when skill is provided (e.g. 'lammps' with "
                "'molecular_dynamics')."
            )
        if not flagged:
            return {
                "status": "success",
                "diagnosis": "No flagged properties supplied.",
                "recommended_action": "escalate",
                "detail": "Nothing to fix.",
                "requires_human": True,
                "rationale": "Advisor called with no flagged measurements.",
            }

        lines = []
        for m in flagged:
            prop = f" — {m['property']}" if m.get("property") else ""
            why = f": {m['reasoning']}" if m.get("reasoning") else ""
            lines.append(f"- {m.get('component')}{prop}{why}")
        flagged_block = "\n".join(lines)

        skill_context = self._load_skill_section(skill, domain or "")
        prompt = self.BASELINE_PROMPT_TEMPLATE.format(
            skill_context=skill_context or "(no engine skill loaded)",
            system_description=system_description or "(not provided)",
            backend=backend or "(not specified)",
            flagged_block=flagged_block,
        )
        report = self._generate_json(prompt)
        report.setdefault("status", "success")
        report.setdefault("recommended_action", "escalate")
        report.setdefault("requires_human", True)
        return report
