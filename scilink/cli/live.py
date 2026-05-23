#!/usr/bin/env python3
"""
scilink live — real-time monitoring of an in-progress measurement.

Polls a file the instrument is writing to, runs a skill-registered
reading function every couple of seconds, and invokes the LLM only when
an "interesting event" fires (verdict change, new feature, confidence
reversal, optional periodic status updates). Mirrors the UI's Live Mode but for
headless / SSH / cron use cases.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scilink live",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Watch an XRD scan as it writes to a CSV; identify Si vs Ge
  scilink live ~/diffractometer/scan.csv --skill xrd

  # Append-only data source (instrument writes incrementally) + a
  # tighter accept threshold + periodic status update every 30 s
  scilink live /tmp/raman_live.txt --skill xrd \\
      --source-kind append_only --threshold 0.8 --heartbeat-sec 30

  # Custom output directory and reading cadence
  scilink live ~/scan.csv --skill xrd \\
      --reading-interval 1.0 --output-dir ./my_live_session

  # Replay a previously-recorded JSONL against a (possibly different)
  # trigger policy — useful for tuning thresholds
  scilink live --replay ./live_session_20260521_180448/live_readings.jsonl \\
      --threshold 0.9

Environment variables (any of these will be picked up):
  ANTHROPIC_API_KEY / CLAUDE_API_KEY
  GEMINI_API_KEY / GOOGLE_API_KEY
  OPENAI_API_KEY
  SCILINK_LOCAL_CIF_DIR     (for structure_matching skills)
  MP_API_KEY                (Materials Project, optional)

Ctrl-C to stop the session cleanly (flushes the JSONL stream and
runs any pending LLM call to completion).
""",
    )
    parser.add_argument(
        "data_path", nargs="?",
        help="File OR directory the instrument is writing to (see "
             "--source-kind). Polled every --reading-interval seconds. "
             "Omit with --replay.",
    )
    parser.add_argument(
        "--description", default=None,
        help="Free-text description of the experiment. The LLM uses this "
             "to generate the per-reading analysis script at session "
             "start (and to regenerate it mid-session if the supervisor "
             "detects a deviation that requires a different analysis "
             "approach). Required for non-replay runs.",
    )
    parser.add_argument(
        "--reference-data", default=None,
        help="Path to a representative reading (e.g. the first spectrum "
             "captured manually, or a stored example from a similar past "
             "run). The generated analysis script is validated against "
             "this file before the session starts; on failure, the script "
             "is regenerated with the failure context, up to 3 attempts. "
             "Without this, validation is conservative ('does the "
             "script import cleanly?').",
    )
    parser.add_argument(
        "--skill", default=None,
        help="Optional skill name (or domain/name) to provide context "
             "for the script generation. If omitted, the LLM generates "
             "the script from the description alone.",
    )
    parser.add_argument(
        "--source-kind",
        choices=("mtime_poll", "append_only", "directory_watch"),
        default="mtime_poll",
        help=(
            "How to detect new data:\n"
            "  mtime_poll      — single file, re-read on mtime change "
            "(default; instruments that rewrite the file).\n"
            "  append_only     — single file, return only newly-appended "
            "bytes.\n"
            "  directory_watch — folder of files, return the newest each "
            "reading (time-resolved experiments where each datapoint is a "
            "new file). Combine with --pattern to filter (e.g. '*.csv')."
        ),
    )
    parser.add_argument(
        "--pattern", default="*.txt",
        help="Glob pattern for --source-kind directory_watch. Default: '*.txt'.",
    )
    parser.add_argument(
        "--directory-strategy",
        choices=("newest", "unseen"),
        default="newest",
        help=(
            "directory_watch only. 'newest' (default) returns just the "
            "latest file each reading; 'unseen' walks each new file once "
            "in sort order, so a burst of N files produces N readings."
        ),
    )
    parser.add_argument(
        "--directory-sort",
        choices=("mtime", "name"),
        default="mtime",
        help=(
            "directory_watch only. Sort files by 'mtime' (default) or "
            "'name' (alphabetical — useful for zero-padded sequential "
            "filenames where mtime is unreliable)."
        ),
    )
    parser.add_argument(
        "--reading-interval", type=float, default=2.0,
        help="Seconds between readings. Default: 2.0.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Optional cap on total session duration in seconds. "
             "Default: run until Ctrl-C.",
    )
    parser.add_argument(
        "--heartbeat-sec", type=float, default=0.0,
        help="Periodic status update interval — the LLM produces a "
             "routine narrative every N seconds even when nothing has "
             "changed. Off by default (most scans don't want noise on "
             "quiet stretches). Set to a positive number (e.g. 60) to "
             "enable periodic updates.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Optional confidence threshold (above this → trigger "
             "fires). When unset, no threshold trigger is added.",
    )
    parser.add_argument(
        "--threshold-direction", choices=("above", "below"),
        default="above",
        help="Crossing direction for the threshold trigger. Default: above.",
    )
    # Note: skills now own the deterministic-trigger list (declared via
    # live_reading.triggers in skill frontmatter). To disable a trigger
    # the skill ships, edit the skill (or fork it via SCILINK_SKILLS_PATH).
    # Per-session additions (threshold, heartbeat, qualitative) remain
    # available via the flags below.
    parser.add_argument(
        "--model", default="claude-opus-4-6",
        help="LLM model to use for the slow loop. Default: claude-opus-4-6.",
    )
    parser.add_argument(
        "--qualitative-model", default=None,
        help="Cheap/fast LLM that periodically (~45s by default) "
             "checks whether the recent readings show patterns the "
             "deterministic triggers miss. Defaults to a small "
             "companion of --model (e.g. claude-haiku-4-5 for claude-opus). "
             "Pass 'none' to disable.",
    )
    parser.add_argument(
        "--qualitative-interval", type=float, default=45.0,
        help="Seconds between qualitative-check LLM calls. Default: 45.",
    )
    parser.add_argument(
        "--additional-guidance", default=None,
        help="Session-specific watch-for instructions appended to the "
             "skill's baseline qualitative-check guidance. E.g. \"alert "
             "me if the (111) peak loses intensity by more than 30% — "
             "sample undergoes amorphization above 700 K\". The Stage-1 "
             "cheap LLM sees the skill baseline + this additional text.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="LLM API key. Falls back to ANTHROPIC_API_KEY / "
             "CLAUDE_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Session directory (live_readings.jsonl lives here). "
             "Default: ./live_session_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--chemistry-hint", default=None,
        help="Comma-separated chemistry hint passed to the skill via "
             "session_state['chemistry_hint']. For structure_matching/xrd: "
             "'Si,O' for single-phase TiO2; pipe-separated groups "
             "'Si|Ge|C,O' for multi-hypothesis search.",
    )
    parser.add_argument(
        "--replay",
        help="Path to a previous session's live_readings.jsonl. When set, "
             "the data_path argument is ignored and the recorded readings "
             "are re-emitted against the configured trigger policy "
             "(no LLM calls by default; use --replay-llm-redo to "
             "re-fire run_task per trigger).",
    )
    parser.add_argument(
        "--replay-speed", type=float, default=None,
        help="Replay-mode speed multiplier. None (default) = instant; "
             "1.0 = real time; 10.0 = 10x faster.",
    )
    parser.add_argument(
        "--replay-llm-redo", action="store_true",
        help="During --replay, actually call orchestrator.run_task for "
             "each trigger (useful for debugging a wrong LLM call).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging (DEBUG level).",
    )
    return parser


def _resolve_api_key(args) -> Optional[str]:
    if args.api_key:
        return args.api_key
    for env in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY",
                 "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    return None


def _parse_chemistry_hint(raw: Optional[str]):
    """Parse the --chemistry-hint string into a list or list-of-lists.

    'Si,O'         → ['Si', 'O']                (single hypothesis)
    'Si|Ge|C,O'    → [['Si'], ['Ge'], ['C', 'O']]  (multi-hypothesis)
    """
    if not raw:
        return None
    if "|" in raw:
        groups = [g.strip() for g in raw.split("|")]
        return [[e.strip() for e in g.split(",") if e.strip()] for g in groups if g]
    return [e.strip() for e in raw.split(",") if e.strip()]


def _infer_light_model(main_model: str) -> str:
    """Default cheap/fast companion for the qualitative-check trigger.

    Gemini 2.5 Flash regardless of main-model provider — cheapest
    production-grade option, overridable with --qualitative-model.
    Cross-provider usage needs GEMINI_API_KEY in addition to whatever
    key the main model uses.
    """
    return "gemini-2.5-flash"


def _compose_guidance(skill_guidance, additional_guidance) -> str:
    """Concatenate skill baseline + operator session-specific guidance.
    Mirrors the helper of the same name in ui/components/live_panel.py."""
    parts: list[str] = []
    if skill_guidance and str(skill_guidance).strip():
        parts.append(str(skill_guidance).strip())
    if additional_guidance and str(additional_guidance).strip():
        parts.append(
            "Session-specific guidance from the operator:\n"
            + str(additional_guidance).strip()
        )
    return "\n\n".join(parts)


def _build_policy(args, *, skill_meta: dict | None = None,
                   resolved_api_key: str | None = None):
    """Build the trigger policy for a live session.

    Adaptive-script live mode pivots away from skill-declared
    deterministic triggers. The policy is now:

      1. :class:`QualitativeProgressTrigger` (the SUPERVISOR) — runs
         every ~45 s on the cheap model; can decide ok / flag /
         adapt. Adapt requests trigger script regeneration; flag
         requests trigger full-model interpretation.
      2. :class:`ThresholdCrossTrigger` if ``--threshold`` is passed
         (per-session additive).
      3. :class:`HeartbeatTrigger` if ``--heartbeat-sec > 0``
         (per-session additive; off by default).
      4. :class:`ManualTrigger` always — the "Interpret Now" button.

    The supervisor's guidance is the concatenation of the skill's
    qualitative_check guidance (if any) + the operator's
    --additional-guidance. When both are empty, falls back to
    DEFAULT_QUALITATIVE_GUIDANCE so the supervisor still has a job.
    """
    from scilink.agents.exp_agents.live_triggers import (
        DEFAULT_QUALITATIVE_GUIDANCE, HeartbeatTrigger, ManualTrigger,
        QualitativeProgressTrigger, ThresholdCrossTrigger, TriggerPolicy,
    )

    triggers: list = []

    # Per-session additions
    if args.threshold is not None:
        triggers.append(ThresholdCrossTrigger(
            threshold=args.threshold, direction=args.threshold_direction,
        ))
    if args.heartbeat_sec and args.heartbeat_sec > 0:
        triggers.append(HeartbeatTrigger(interval_sec=args.heartbeat_sec))

    # Supervisor (qualitative trigger with adapt enabled by default)
    light_arg = (args.qualitative_model or "").strip().lower()
    if light_arg == "none":
        light_model = None
    elif light_arg:
        light_model = args.qualitative_model
    else:
        light_model = _infer_light_model(args.model)

    qcheck = ((skill_meta or {}).get("live_reading") or {}).get("qualitative_check") or {}
    guidance = qcheck.get("guidance")
    qcheck_enabled = qcheck.get("enabled", True)
    additional_guidance = (getattr(args, "additional_guidance", None) or "").strip() or None
    if (light_model and resolved_api_key
            and (not guidance or not qcheck_enabled)
            and not additional_guidance):
        guidance = DEFAULT_QUALITATIVE_GUIDANCE
        qcheck_enabled = True
    composed_guidance = _compose_guidance(guidance, additional_guidance)
    if (light_model and qcheck_enabled and composed_guidance and resolved_api_key):
        triggers.append(QualitativeProgressTrigger(
            guidance=composed_guidance,
            model=light_model,
            api_key=resolved_api_key,
            interval_sec=float(args.qualitative_interval or qcheck.get("interval_sec", 45.0)),
            history_n=int(qcheck.get("history_n", 10)),
            enable_adaptation=True,
        ))

    triggers.append(ManualTrigger())
    return TriggerPolicy(triggers=triggers)


def _format_tick(result) -> str:
    return (f"  metric={result.primary_metric:.3f}  verdict={result.verdict:<8s}"
            f"  features={len(result.detected_features):2d}"
            f"  {result.notes[:80]}")


def _format_llm(event, result: dict) -> str:
    s = (result.get("summary") or "").strip().splitlines()
    head = s[0] if s else "(empty)"
    return (f"  trigger={event.name}  status={result.get('status', '?')}"
            f"  {head[:200]}")


def _run_live(args, log) -> int:
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, AnalysisOrchestratorAgent,
    )
    from scilink.agents.exp_agents.live_codegen import (
        generate_and_validate_with_retry, latest_data_from_file,
    )
    from scilink.agents.exp_agents.live_data_sources import (
        AppendOnlyFileSource, DirectoryWatchSource, MtimePollFileSource,
    )
    from scilink.agents.exp_agents.live_session import LiveSession
    from scilink.skills.loader import load_skill

    if not args.data_path:
        log.error("data_path is required when not using --replay")
        return 2
    if not args.description:
        log.error("--description is required (a free-text experiment summary)")
        return 2

    api_key = _resolve_api_key(args)
    if not api_key:
        log.error("No API key — pass --api-key or set ANTHROPIC_API_KEY / "
                  "GEMINI_API_KEY / OPENAI_API_KEY in the environment.")
        return 2

    # Output dir
    if args.output_dir:
        session_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"live_session_{ts}").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    log.info("Session dir: %s", session_dir)

    # Optional skill — provides context for codegen but isn't strictly
    # required in adaptive-script mode. When --skill is omitted, the
    # LLM generates the reading_fn from the description alone.
    skill = {}
    skill_context = None
    if args.skill:
        if "/" in args.skill and not args.skill.startswith("/"):
            domain, name = args.skill.split("/", 1)
        else:
            name = args.skill
            domain = "structure_matching" if name == "xrd" else "curve_fitting"
        try:
            skill = load_skill(name, domain=domain)
            skill_context = (
                f"Skill: {domain}/{name}\n"
                f"Description: {skill.get('meta', {}).get('description', '')}\n\n"
                f"Analysis section:\n{skill.get('analysis', '') or '(none)'}"
            )
            log.info("Skill loaded: %s/%s (context-only)", domain, name)
        except Exception as e:
            log.warning("Could not load skill %s: %s — proceeding without skill context",
                        args.skill, e)

    # Generate + validate the initial reading_fn (analyze-mode parity).
    # If --reference-data is provided, the script must process it
    # successfully; on failure, regenerate with the failure context.
    reference_payloads = []
    if args.reference_data:
        ref = latest_data_from_file(args.reference_data)
        if ref is None:
            log.warning(
                "Reference spectrum path not readable: %s. "
                "Proceeding with conservative validation only.",
                args.reference_data,
            )
        else:
            reference_payloads.append(ref)
            log.info("Reference spectrum loaded: %s (%d bytes)",
                     args.reference_data,
                     len(ref.text) if ref.text else 0)
    else:
        log.info("No --reference-data given; validation is import-only.")

    additional_guidance = (args.additional_guidance or "").strip() or None
    log.info("Generating initial reading_fn via %s (max 3 attempts)…",
             args.model)
    reading_fn, final_version, attempts = generate_and_validate_with_retry(
        description=args.description,
        additional_guidance=additional_guidance,
        skill_context=skill_context,
        model=args.model,
        api_key=api_key,
        session_dir=session_dir,
        reference_data=reference_payloads,
        max_attempts=3,
    )
    if reading_fn is None:
        log.error(
            "All 3 attempts at generating a working analysis script failed:"
        )
        for a in attempts:
            log.error("  %s", a)
        return 3
    if attempts:
        log.info("Script v%d validated after %d prior attempt(s):",
                 final_version, len(attempts))
        for a in attempts:
            log.info("  %s", a)
    log.info("Active reading_fn: %s",
             session_dir / f"reading_script_v{final_version}.py")

    # Data source
    if args.source_kind == "directory_watch":
        source = DirectoryWatchSource(
            args.data_path,
            pattern=args.pattern,
            strategy=args.directory_strategy,
            sort_by=args.directory_sort,
        )
        log.info(
            "Data source: directory_watch (%s, pattern=%r, strategy=%s, sort_by=%s)",
            args.data_path, args.pattern, args.directory_strategy, args.directory_sort,
        )
    elif args.source_kind == "append_only":
        source = AppendOnlyFileSource(args.data_path)
        log.info("Data source: append_only (%s)", args.data_path)
    else:
        source = MtimePollFileSource(args.data_path)
        log.info("Data source: mtime_poll (%s)", args.data_path)

    # Orchestrator (AUTONOMOUS — no human-feedback prompts during live readings)
    try:
        orch = AnalysisOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=api_key,
            model_name=args.model,
            analysis_mode=AnalysisMode.AUTONOMOUS,
        )
    except Exception as e:
        log.error("Failed to initialize agent: %s", e)
        return 4

    policy = _build_policy(args, skill_meta=skill.get("meta", {}),
                            resolved_api_key=api_key)

    def on_reading(result):
        log.info("[TICK] %s", _format_tick(result))

    def on_llm(event, result):
        log.info("[LLM ] %s", _format_llm(event, result))

    session = LiveSession(
        orchestrator=orch, data_source=source, reading_fn=reading_fn,
        reading_interval_sec=args.reading_interval,
        trigger_policy=policy,
        history_path=session_dir / "live_readings.jsonl",
        skill_state=skill,
        on_reading=on_reading,
        on_llm_response=on_llm,
        # Adaptive-script regeneration kwargs
        session_dir=session_dir,
        adapt_model=args.model,
        adapt_api_key=api_key,
        adapt_description=args.description,
        adapt_additional_guidance=additional_guidance,
        adapt_skill_context=skill_context,
    )
    session._adapt_version = final_version

    chemistry = _parse_chemistry_hint(args.chemistry_hint)
    if chemistry is not None:
        session._session_state["chemistry_hint"] = chemistry
    session._session_state["candidates_dir"] = str(session_dir / "candidates")

    # Ctrl-C → graceful stop
    def _on_signal(signum, _frame):
        log.info("Signal %d received — stopping session...", signum)
        session.stop(timeout=10.0)
        log.info("Session stopped cleanly.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Starting LiveSession (reading interval %.1fs)", args.reading_interval)
    session.start()

    started_at = time.monotonic()
    try:
        while True:
            time.sleep(0.5)
            if args.duration is not None and (time.monotonic() - started_at) >= args.duration:
                log.info("Duration cap reached (%.1fs) — stopping.", args.duration)
                break
    finally:
        session.stop(timeout=15.0)
    log.info("Session ended.")
    log.info("JSONL: %s", session_dir / "live_readings.jsonl")
    return 0


def _run_replay(args, log) -> int:
    from scilink.agents.exp_agents.live_replay import replay_jsonl

    policy = _build_policy(args)

    orch = None
    if args.replay_llm_redo:
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisMode, AnalysisOrchestratorAgent,
        )
        api_key = _resolve_api_key(args)
        if not api_key:
            log.error("--replay-llm-redo requires an API key.")
            return 2
        # Use a throwaway dir; replay doesn't need a persistent session
        tmp_dir = Path(args.output_dir or f"replay_{datetime.now():%Y%m%d_%H%M%S}").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        orch = AnalysisOrchestratorAgent(
            base_dir=str(tmp_dir), api_key=api_key,
            model_name=args.model, analysis_mode=AnalysisMode.AUTONOMOUS,
        )

    def on_reading(result):
        log.info("[TICK] %s", _format_tick(result))

    def on_event(ev):
        log.info("[TRIG] %s  %s", ev.name,
                  {k: v for k, v in (ev.details or {}).items() if k != "reading_ref"})

    log.info("Replaying %s ...", args.replay)
    report = replay_jsonl(
        args.replay,
        trigger_policy=policy,
        speed=args.replay_speed,
        on_reading=on_reading,
        on_event=on_event,
        orchestrator=orch,
        llm_mode="redo" if args.replay_llm_redo else "skip",
    )
    log.info("Replay complete.")
    log.info("  readings:           %d", report.reading_count)
    log.info("  triggers fired:  %d (by name: %s)",
             report.trigger_count, dict(report.trigger_event_counts))
    log.info("  LLM responses:   %d", len(report.llm_responses))
    log.info("  duration:        %.1fs", report.duration_sec)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s.%(msecs)03d %(levelname)5s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("scilink-live")

    if args.replay:
        return _run_replay(args, log)
    return _run_live(args, log)


if __name__ == "__main__":
    sys.exit(main())
