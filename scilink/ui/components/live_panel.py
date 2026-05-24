"""Streamlit panel for the live-monitoring mode.

Renders one of two views depending on session state:

  Setup view  — when no live session is active. The user enters a
                data path and describes the experiment in natural
                language; an LLM parse extracts the structured config
                (skill, source kind, reading interval, triggers) for
                the user to review before starting the session. The
                model + API key + consent come from the standard
                sidebar fields (no panel-level prereqs banner — that
                matches how analyze / plan / simulate work).

  Dashboard   — when a live session is running. Layout: latest data
                plot + metric time-series chart on the left, decision
                feed on the right, controls (Interpret Now, Stop) at
                the bottom. Auto-refreshes via
                ``st.fragment(run_every="2s")`` — same primitive
                the existing chat-task polling uses
                (``app.py:539``).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_live_panel() -> None:
    """Top-level renderer; called from ``app.py`` when ``app_mode == 'live'``."""
    st.header("Live Monitoring")
    st.caption(
        "Real-time agent interpretation of an in-progress measurement. "
        "The fast reading loop runs every couple of seconds; the LLM is "
        "invoked only when an interesting event fires."
    )

    if st.session_state.get("live_session") is None:
        _render_setup()
    else:
        _render_dashboard()


# ---------------------------------------------------------------------------
# Setup view
# ---------------------------------------------------------------------------


def _compose_guidance(skill_guidance: Optional[str],
                       additional_guidance: Optional[str]) -> str:
    """Concatenate the skill's baseline qualitative-check guidance with the
    operator's session-specific watch-for instructions. Either part may
    be empty; returns "" only when both are."""
    parts: list[str] = []
    if skill_guidance and str(skill_guidance).strip():
        parts.append(str(skill_guidance).strip())
    if additional_guidance and str(additional_guidance).strip():
        parts.append(
            "Session-specific guidance from the operator:\n"
            + str(additional_guidance).strip()
        )
    return "\n\n".join(parts)


def _infer_light_model(main_model: str) -> str:
    """Default cheap/fast companion for the qualitative-check trigger.

    Gemini 2.5 Flash regardless of which provider the main model uses —
    cheapest production-grade option across the board, and the user
    can override in the panel's text field. Cross-provider usage
    (e.g. Claude main + Gemini light) requires GEMINI_API_KEY in the
    environment in addition to the main provider's key.
    """
    return "gemini-2.5-flash"


def _resolve_sidebar_config() -> tuple[str, str, bool]:
    """Read model / API key / consent from the standard sidebar fields.

    Matches the keys the sidebar writes: cfg_model_preset / cfg_model_custom
    (sidebar.py:213-222) and cfg_api_key / cfg_consent (lines 223, 267).
    """
    preset = st.session_state.get("cfg_model_preset", "")
    if preset == "Custom":
        model = st.session_state.get("cfg_model_custom", "") or ""
    else:
        model = preset or "claude-opus-4-6"
    api_key = st.session_state.get("cfg_api_key") or ""
    consent = bool(st.session_state.get("cfg_consent"))
    return model, api_key, consent


_PARSE_SYSTEM_PROMPT = """You extract structured live-monitoring configuration from a scientist's plain-language description of an experiment.

You will be given a short description of an in-progress measurement
(XRD, Raman, EELS, generic spectroscopy, etc.). Return ONE JSON object
matching this schema — no prose, no markdown fences:

{{
  "skill_label": "<domain/name>",            // see AVAILABLE_SKILLS below
  "source_kind": "mtime_poll" | "append_only" | "directory_watch",
  "source_pattern": "*.csv",                  // glob, only used when source_kind == directory_watch
  "source_strategy": "newest" | "unseen",     // only used when source_kind == directory_watch
  "source_sort_by": "mtime" | "name",         // only used when source_kind == directory_watch
  "reading_interval_sec": 2.0,                // float, seconds between readings
  "threshold": null | <float 0..1>,           // optional per-session ThresholdCrossTrigger
  "chemistry_hint": null | ["Si","O"] | [["Si"],["Ge"]],   // optional; for structure_matching skills
  "additional_guidance": null | "<text>",     // session-specific watch-for guidance — see below
  "summary": "one sentence describing what you extracted"
}}

Source-kind disambiguation rules:
  - The instrument **rewrites a single file** as the scan progresses  → mtime_poll
  - The instrument **appends to a single file** (one new row per step) → append_only
  - The instrument writes **one file per measurement into a folder**   → directory_watch
  If the user says "a folder of files" or names a pattern like
  "scan_XXXX.csv", choose directory_watch.

Reading-interval defaults to 2.0 s unless the user specifies otherwise.

Triggers: each skill ships its own deterministic trigger taxonomy
(declared in skill frontmatter). The user can't toggle the skill's
triggers from prose — they fire automatically. The only ADDITIVE
per-session trigger you can extract here is a ThresholdCrossTrigger:
if the user says "alert me when FOM > 0.7" or "fire when confidence
crosses 0.85", set ``threshold`` to that float. Otherwise leave null.

additional_guidance: extract any session-specific qualitative
watch-for instructions the user mentions — "watch for the 47° peak
because this sample undergoes amorphization above 700 K", "alert me
if intensity ratio between (111) and (220) drifts more than 30%",
"keep an eye on noise floor in the high-2θ region", etc. These get
concatenated with the skill's baseline qualitative guidance so the
Stage-1 cheap LLM sees both. Leave as null when the user gives no
domain-specific cues. Strip out source-kind / interval / skill-pick
language — those go in their own fields, not here.

Pick the skill_label whose description best matches the experiment.
If nothing matches, use "diagnostics/live_passthrough".

AVAILABLE_SKILLS:
{available_skills}
"""


def _parse_description(
    description: str, model: str, api_key: str,
    skill_options: list[dict],
) -> Optional[dict]:
    """Call the LLM to convert a natural-language description into a
    structured config dict. Returns None on failure (caller surfaces error)."""
    try:
        import litellm
    except ImportError:
        st.error("LiteLLM not available; cannot parse description.")
        return None

    from scilink.wrappers.litellm_wrapper import _normalize_model_name

    skills_blob = "\n".join(
        f"  - {s['label']}: {s['description'][:200]}"
        for s in skill_options
    ) or "  (no live-reading-enabled skills found)"
    system = _PARSE_SYSTEM_PROMPT.format(available_skills=skills_blob)

    try:
        resp = litellm.completion(
            model=_normalize_model_name(model),
            api_key=api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": description.strip()},
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None

    try:
        text = resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Unexpected LLM response shape: {e}")
        return None

    return _extract_json(text)


def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of the LLM response (tolerates surrounding text)."""
    # Strip common code-fence wrappers
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last <= first:
            st.error("No JSON object found in LLM response.")
            with st.expander("Raw LLM response"):
                st.code(text[:2000])
            return None
        candidate = text[first : last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM JSON: {e}")
        with st.expander("Raw LLM response"):
            st.code(candidate[:2000])
        return None


def _render_setup() -> None:
    # Sidebar handles model / API key / consent — same as analyze / plan /
    # simulate. No prereqs banner here; the sidebar IS the visible status,
    # and the Start button gates on consent (sidebar.py:343 pattern).
    model, api_key, consent = _resolve_sidebar_config()
    env_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_key = api_key or env_key

    # Skills are now OPTIONAL context for codegen — no longer required.
    # When present, their description + analysis section get folded into
    # the script-generation prompt. When absent, the LLM generates from
    # the operator's description alone.
    skill_options = _discover_live_enabled_skills()

    st.subheader("Start a live session")

    data_path = st.text_input(
        "Data file or directory path",
        key="live_data_path_input",
        placeholder="/path/to/in-progress-scan.csv   OR   /path/to/scan_directory/",
        help="Single file (instrument rewrites or appends as it scans) "
             "OR directory (instrument writes one file per measurement step).",
    )

    reference_path = st.text_input(
        "Reference spectrum (optional, recommended)",
        key="live_reference_data_path",
        placeholder="/path/to/representative_spectrum.csv",
        help="A single file containing a representative reading of your "
             "experiment — e.g. the first spectrum captured manually or a "
             "stored example from a similar past run.",
    )

    description = st.text_area(
        "Describe your experiment",
        key="live_description",
        height=180,
        placeholder=(
            "Examples:\n"
            "• I'm running an XRD scan on a Si sample at 800 K. The "
            "instrument writes scan_XXXX.csv files into a folder, one per "
            "measurement. Watch specifically for the (111) peak losing "
            "intensity — this sample undergoes amorphization above ~700 K.\n\n"
            "• Single CSV file gets rewritten as the scan progresses. "
            "Check every 3 seconds. Alert me when confidence crosses 0.85.\n\n"
            "• Time-resolved Raman series, files accumulating in /data/run42/. "
            "Keep an eye on the D-band intensity ratio."
        ),
        help="Plain language. The LLM extracts the source kind, reading "
             "interval, optional threshold, chemistry hint, and any "
             "domain-specific watch-for instructions.",
    )

    # Light-model picker + optional API key for the qualitative-progress
    # trigger (Stage 1 LLM). Side-by-side: model name on the left, API
    # key on the right. Both fields are optional. The main sidebar
    # model still does the full-quality interpretations when any trigger
    # fires; the qualitative model is the cheap supervisor.
    light_default = _infer_light_model(model)
    qm_col, qk_col = st.columns(2)
    with qm_col:
        light_model = st.text_input(
            "Qualitative-check model (optional)",
            value=st.session_state.get("live_light_model", light_default),
            key="live_light_model",
            help="Cheap/fast model that periodically (~45 s) checks whether "
                 "the recent reading history shows qualitative patterns the "
                 "deterministic triggers miss.",
        )
    with qk_col:
        light_api_key = st.text_input(
            "Qualitative-check API key (optional)",
            value=st.session_state.get("live_light_api_key", ""),
            key="live_light_api_key",
            type="password",
            help="Only needed if the qualitative-check model uses a "
                 "different provider than the sidebar's main model.",
        )

    # Single-click flow: parse the description and start in one action.
    # Disable until consent + non-empty data path + non-empty description.
    start_disabled = (
        not consent or not data_path
        or not (description and description.strip())
        or not resolved_key
    )
    start_help = None
    if not consent:
        start_help = "Check the code-execution consent box in the sidebar first."
    elif not resolved_key:
        start_help = "Set an API key in the sidebar (or LLM env var)."
    elif not data_path:
        start_help = "Enter a data file or directory path above."
    elif not (description and description.strip()):
        start_help = "Describe the experiment above."

    # Optional skill picker for codegen CONTEXT only (not required)
    skill_choice = None
    selected_skill_for_codegen = None
    if skill_options:
        choices = ["(none — use description only)"] + [s["label"] for s in skill_options]
        skill_choice = st.selectbox(
            "Optional skill for codegen context", options=choices,
            help="If selected, the skill's description + analysis section "
                 "are passed as additional context to the LLM that "
                 "generates the per-reading analysis script.",
        )
        if skill_choice and skill_choice != "(none — use description only)":
            selected_skill_for_codegen = next(
                (s for s in skill_options if s["label"] == skill_choice), None
            )

    if st.button(
        "Start Live Session", type="primary",
        disabled=start_disabled, help=start_help,
    ):
        with st.spinner("Parsing description, generating analysis script…"):
            parsed = _parse_description(
                description.strip(), model, resolved_key, skill_options,
            )
        if parsed is None:
            return  # _parse_description already surfaced the error
        # Normalize light model: blank or 'none' → disabled
        lm = (light_model or "").strip()
        if lm.lower() in ("", "none"):
            lm = None
        # Operator-supplied API key for the light model (optional)
        lak = (light_api_key or "").strip() or None
        # Force the selected skill (if any) — overrides the LLM's auto-pick
        if selected_skill_for_codegen is not None:
            parsed["skill_label"] = selected_skill_for_codegen["label"]
        _start_from_parsed(
            parsed=parsed, data_path=data_path,
            model=model, api_key=resolved_key, skill_options=skill_options,
            light_model=lm, light_api_key=lak,
        )


def _start_from_parsed(*, parsed: dict, data_path: str, model: str,
                        api_key: str, skill_options: list[dict],
                        light_model: Optional[str] = None,
                        light_api_key: Optional[str] = None) -> None:
    # Skill is OPTIONAL — used only for codegen context. If the parser
    # picked a real skill, fold it in; otherwise proceed without.
    skill_label = parsed.get("skill_label", "") or ""
    selected = next((s for s in skill_options if s["label"] == skill_label), None)
    skill_name = selected["name"] if selected else None
    skill_domain = selected["domain"] if selected else None

    if not Path(data_path).exists():
        st.warning(
            f"Path does not exist yet: {data_path}. The session will start "
            "and the reading loop will pick up the file once your instrument "
            "creates it."
        )

    source_kind = parsed.get("source_kind", "mtime_poll")
    source_kwargs = (
        {
            "pattern": parsed.get("source_pattern", "*.txt"),
            "strategy": parsed.get("source_strategy", "newest"),
            "sort_by": parsed.get("source_sort_by", "mtime"),
        }
        if source_kind == "directory_watch" else {}
    )
    triggers_in = parsed.get("triggers", {}) or {}
    threshold_val = parsed.get("threshold")
    if threshold_val is None:
        threshold_val = triggers_in.get("threshold")
    triggers = {
        "threshold": float(threshold_val) if threshold_val is not None else None,
    }

    st.session_state.live_chemistry_hint = parsed.get("chemistry_hint")
    st.session_state.live_parsed_config = parsed

    additional_guidance = (parsed.get("additional_guidance") or "").strip() or None
    # Pass the operator's full description through to codegen so the
    # generated reading_fn knows the experiment shape.
    description = (parsed.get("summary") or "").strip()
    if not description:
        description = st.session_state.get("live_description", "")

    _spin_up_live_session(
        model=model,
        api_key=api_key,
        data_path=data_path,
        source_kind=source_kind,
        source_kwargs=source_kwargs,
        skill_name=skill_name,
        skill_domain=skill_domain,
        reading_interval_sec=float(parsed.get("reading_interval_sec", 2.0)),
        triggers=triggers,
        light_model=light_model,
        light_api_key=light_api_key,
        additional_guidance=additional_guidance,
        description=description,
    )


def _discover_live_enabled_skills() -> list[dict]:
    """Walk all available skills; return those whose frontmatter declares
    a non-empty `live_reading.reading_fn`. Caller still resolves the dotted
    path at start time so we fail fast on bad paths."""
    from scilink.skills.loader import list_all_skills, load_skill

    out: list[dict] = []
    for domain, names in list_all_skills().items():
        for name in names:
            try:
                parsed = load_skill(name, domain=domain)
            except Exception:
                continue
            block = parsed.get("meta", {}).get("live_reading") or {}
            if not isinstance(block, dict):
                continue
            if not block.get("enabled", True):
                continue
            if not block.get("reading_fn"):
                continue
            out.append({
                "name": name,
                "domain": domain,
                "label": f"{domain}/{name}",
                "description": parsed.get("meta", {}).get("description", ""),
            })
    out.sort(key=lambda s: s["label"])
    return out


def _spin_up_live_session(
    *,
    model: str,
    api_key: str,
    data_path: str,
    source_kind: str,
    source_kwargs: Optional[dict] = None,
    skill_name: Optional[str] = None,
    skill_domain: Optional[str] = None,
    reading_interval_sec: float,
    triggers: dict,
    light_model: Optional[str] = None,
    light_api_key: Optional[str] = None,
    additional_guidance: Optional[str] = None,
    description: str = "",
) -> None:
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )
    from scilink.agents.exp_agents.live_codegen import (
        generate_and_validate_with_retry,
        latest_data_from_file,
    )
    from scilink.agents.exp_agents.live_data_sources import (
        AppendOnlyFileSource,
        DirectoryWatchSource,
        MtimePollFileSource,
    )
    from scilink.agents.exp_agents.live_session import LiveSession
    from scilink.agents.exp_agents.live_triggers import (
        DEFAULT_QUALITATIVE_GUIDANCE,
        HeartbeatTrigger,
        ManualTrigger,
        QualitativeProgressTrigger,
        ThresholdCrossTrigger,
        TriggerPolicy,
    )
    from scilink.skills.loader import load_skill
    from scilink.ui.config import SESSION_DIR_PREFIXES

    # Session dir parallels the analyze/plan/simulate pattern.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"{SESSION_DIR_PREFIXES['live']}_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Live mode is fixed at AUTONOMOUS (sidebar autonomy picker ignored).
    try:
        orch = AnalysisOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=api_key,
            model_name=model,
            analysis_mode=AnalysisMode.AUTONOMOUS,
        )
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return

    # Optional skill — context-only for codegen
    skill = {}
    skill_context = None
    if skill_name and skill_domain:
        try:
            skill = load_skill(skill_name, domain=skill_domain)
            skill_context = (
                f"Skill: {skill_domain}/{skill_name}\n"
                f"Description: {skill.get('meta', {}).get('description', '')}\n\n"
                f"Analysis section:\n{skill.get('analysis', '') or '(none)'}"
            )
        except Exception as e:
            st.warning(
                f"Could not load skill {skill_domain}/{skill_name}: {e} — "
                "proceeding without skill context."
            )

    # Generate + validate the initial reading_fn. If the operator
    # supplied a reference spectrum file, validate the script against
    # it (analyze-mode parity: must successfully fit one example before
    # going live). If validation fails, regenerate with the failure
    # context up to 3 times — same iterative-retry pattern analyze mode
    # uses for its fit scripts.
    reference_path = st.session_state.get("live_reference_data_path")
    reference_payloads = []
    if reference_path:
        ref = latest_data_from_file(reference_path)
        if ref is None:
            st.warning(
                f"Reference spectrum path not readable: {reference_path}. "
                "Proceeding with conservative validation (script will be "
                "imported but not exercised on real data)."
            )
        else:
            reference_payloads.append(ref)

    with st.spinner(
        f"Generating initial analysis script (validating against "
        f"{'reference spectrum' if reference_payloads else 'imports only — no reference data provided'})…"
    ):
        reading_fn, final_version, attempts = generate_and_validate_with_retry(
            description=description or "(no description provided)",
            additional_guidance=additional_guidance,
            skill_context=skill_context,
            model=model,
            api_key=api_key,
            session_dir=session_dir,
            reference_data=reference_payloads,
            max_attempts=3,
        )
    if reading_fn is None:
        st.error(
            "All 3 attempts at generating a working analysis script failed:\n\n"
            + "\n".join(f"  • {a}" for a in attempts)
        )
        return
    if attempts:
        st.info(
            f"Initial script generated and validated on attempt v{final_version} "
            f"after {len(attempts)} prior attempt(s):"
            + "\n" + "\n".join(f"  • {a}" for a in attempts)
        )

    # Data source
    if source_kind == "directory_watch":
        source = DirectoryWatchSource(
            data_path,
            pattern=(source_kwargs or {}).get("pattern", "*.txt"),
            strategy=(source_kwargs or {}).get("strategy", "newest"),
            sort_by=(source_kwargs or {}).get("sort_by", "mtime"),
        )
    elif source_kind == "append_only":
        source = AppendOnlyFileSource(data_path)
    else:
        source = MtimePollFileSource(data_path)

    # Trigger policy: adaptive mode only ships framework triggers —
    # ManualTrigger, optional ThresholdCrossTrigger, and the supervisor
    # (QualitativeProgressTrigger with enable_adaptation=True).
    trigger_list: list = []
    if triggers["threshold"] is not None:
        trigger_list.append(
            ThresholdCrossTrigger(threshold=triggers["threshold"], direction="above")
        )

    qcheck = skill.get("meta", {}).get("live_reading", {}).get("qualitative_check") or {}
    guidance = qcheck.get("guidance")
    qcheck_enabled = qcheck.get("enabled", True)
    if (light_model and api_key
            and (not guidance or not qcheck_enabled)
            and not additional_guidance):
        guidance = DEFAULT_QUALITATIVE_GUIDANCE
        qcheck_enabled = True
    composed_guidance = _compose_guidance(guidance, additional_guidance)
    if light_model and qcheck_enabled and composed_guidance:
        # Operator-supplied light-model API key wins; else fall back to
        # the main API key (which _resolve_provider_api_key in the trigger
        # will further fall through to env vars matching the model's provider).
        supervisor_api_key = light_api_key or api_key
        trigger_list.append(QualitativeProgressTrigger(
            guidance=composed_guidance,
            model=light_model,
            api_key=supervisor_api_key,
            interval_sec=float(qcheck.get("interval_sec", 45.0)),
            history_n=int(qcheck.get("history_n", 10)),
            enable_adaptation=True,
        ))
    trigger_list.append(ManualTrigger())  # "Interpret Now" button
    policy = TriggerPolicy(triggers=trigger_list)

    session = LiveSession(
        orchestrator=orch,
        data_source=source,
        reading_fn=reading_fn,
        reading_interval_sec=reading_interval_sec,
        trigger_policy=policy,
        history_path=session_dir / "live_readings.jsonl",
        skill_state=skill,
        on_llm_response=lambda ev, result: _record_llm_response(ev, result),
        # Adaptive script regeneration kwargs
        session_dir=session_dir,
        adapt_model=model,
        adapt_api_key=api_key,
        adapt_description=description,
        adapt_additional_guidance=additional_guidance,
        adapt_skill_context=skill_context,
    )
    # Sync the session's adapt-version cursor with whatever version
    # validation activated (may be > 1 if retries were needed).
    session._adapt_version = final_version
    chem_hint = st.session_state.get("live_chemistry_hint")
    if chem_hint is not None:
        session._session_state["chemistry_hint"] = chem_hint
    session._session_state["candidates_dir"] = str(session_dir / "candidates")
    session.start()

    st.session_state.live_orch = orch
    st.session_state.live_session = session
    st.session_state.live_session_dir = str(session_dir)
    st.session_state.live_data_path = data_path
    st.session_state.live_skill_label = (
        f"{skill_domain}/{skill_name}" if (skill_domain and skill_name)
        else "(codegen — no skill)"
    )
    st.session_state.live_session_feed = []
    st.rerun()


def _record_llm_response(event, result: dict) -> None:
    """Callback called from the LiveSession dispatcher thread when run_task returns.

    Pushes a structured entry into st.session_state.live_session_feed for the
    dashboard fragment to render. The fragment owns rendering; this thread
    only mutates the list.
    """
    feed = st.session_state.get("live_session_feed")
    if feed is None:
        return  # session not fully wired yet
    feed.append({
        "timestamp": time.time(),
        "event_name": event.name,
        "event_details": event.details,
        "status": result.get("status", "unknown"),
        "summary": result.get("summary", ""),
    })


# ---------------------------------------------------------------------------
# Dashboard view
# ---------------------------------------------------------------------------


def _render_dashboard() -> None:
    session = st.session_state.live_session
    session_dir = st.session_state.get("live_session_dir", "?")
    skill_label = st.session_state.get("live_skill_label", "?")
    data_path = st.session_state.get("live_data_path", "?")

    # Header bar — shows the parsed config so the user can verify the
    # LLM understood the experiment correctly. If it's wrong, click Stop,
    # refine the description, and start a new session.
    parsed = st.session_state.get("live_parsed_config", {}) or {}
    source_kind = parsed.get("source_kind", "?")
    source_detail = ""
    if source_kind == "directory_watch":
        source_detail = (
            f" (pattern `{parsed.get('source_pattern', '*.txt')}`, "
            f"sort `{parsed.get('source_sort_by', 'mtime')}`)"
        )

    # Trigger summary in the header — count of active triggers in the
    # session's policy (skill-declared + framework-injected). Avoids
    # reaching into the parsed-config triggers dict (which is gone now
    # that skills own the deterministic list).
    n_triggers = len(getattr(session.policy, "triggers", []))
    extra_threshold = parsed.get("threshold")
    if extra_threshold is None:
        # Backwards compat with older parse shape
        extra_threshold = (parsed.get("triggers") or {}).get("threshold")
    threshold_blob = (
        f" · threshold>{extra_threshold:.2f}" if extra_threshold is not None else ""
    )
    chem = parsed.get("chemistry_hint")
    chem_blob = f" · chemistry `{chem}`" if chem else ""
    extra_guidance = (parsed.get("additional_guidance") or "").strip()

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        script_version = getattr(session, "_adapt_version", 1)
        adapt_count = getattr(session, "_adapt_count", 0)
        adapt_blob = (
            f" · script `v{script_version}` ({adapt_count} adaptation{'s' if adapt_count != 1 else ''})"
        )
        st.markdown(
            f"**Session active** · skill `{skill_label}` · "
            f"data `{data_path}` · interval `{session.reading_interval_sec:.1f}s`"
            f"{adapt_blob}"
        )
        st.caption(
            f"source `{source_kind}`{source_detail} · "
            f"{n_triggers} active triggers{threshold_blob}{chem_blob}"
        )
        if extra_guidance:
            with st.expander("Session-specific guidance"):
                st.write(extra_guidance)
        # Show the active analysis script source for transparency
        script_path = (
            Path(st.session_state.get("live_session_dir", ""))
            / f"reading_script_v{script_version}.py"
        )
        if script_path.is_file():
            with st.expander(f"Active analysis script (v{script_version})"):
                st.code(script_path.read_text(), language="python")
    with col_h2:
        if st.button("Stop Session", type="primary"):
            session.stop(timeout=3.0)
            st.session_state.live_session = None
            st.session_state.live_orch = None
            st.session_state.live_parsed_config = None
            st.rerun()

    @st.fragment(run_every="2s")
    def _live_view() -> None:
        latest = session.latest()
        hist = session.history()

        # ── Top-level status pills ────────────────────────────────
        verdict = latest.verdict if latest else "unknown"
        metric_value = latest.primary_metric if latest else 0.0
        metric_name = latest.metric_name if latest else "metric"
        verdict_color = {
            "accept": "🟢", "marginal": "🟡", "reject": "🔴",
        }.get(verdict, "⚪")
        st.markdown(
            f"### {verdict_color} **{verdict.upper()}**  ·  "
            f"{metric_name}: **{metric_value:.3f}**  ·  "
            f"LLM: {'⏳ busy' if session.llm_busy else '✅ idle'}  ·  "
            f"readings: {len(hist)}"
        )

        col_left, col_right = st.columns([3, 2])

        # ── Left: data + metric chart ─────────────────────────────
        with col_left:
            if hist:
                metrics_series = [t.primary_metric for t in hist]
                st.line_chart(metrics_series, height=200, use_container_width=True)
                st.caption(f"{metric_name} over time ({len(hist)} readings)")
            else:
                st.info("Waiting for first reading — data source has not produced new data yet.")

            # Show detected features for the latest reading, if any
            if latest and latest.detected_features:
                with st.expander(f"Detected features ({len(latest.detected_features)})"):
                    st.json(latest.detected_features[:30])

        # ── Right: decision feed ──────────────────────────────────
        with col_right:
            feed = st.session_state.get("live_session_feed", []) or []
            st.markdown("**Decision feed**")
            if not feed:
                st.caption("No LLM interpretations yet. Triggers will fire as events occur, "
                           "or click _Interpret Now_ below to request one manually.")
            else:
                # Most recent first
                for entry in list(reversed(feed))[:10]:
                    ts = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
                    name = entry["event_name"]
                    status = entry["status"]
                    summary = entry["summary"]
                    icon = "✅" if status == "success" else "⚠️"
                    with st.container(border=True):
                        st.markdown(f"{icon} **{ts}** · `{name}`")
                        if summary:
                            st.write(summary[:1500])
                            if len(summary) > 1500:
                                st.caption(f"({len(summary) - 1500} more characters truncated)")

        # ── Controls ──────────────────────────────────────────────
        if st.button("Interpret Now", help="Manually request an LLM interpretation"):
            session.force_interpretation()
            st.toast("Manual interpretation requested.")

        st.caption(f"Session dir: `{session_dir}` · JSONL: `live_readings.jsonl`")

    _live_view()


__all__ = ["render_live_panel"]
