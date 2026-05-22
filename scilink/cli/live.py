#!/usr/bin/env python3
"""
scilink live — real-time monitoring of an in-progress measurement.

Polls a file the instrument is writing to, runs a skill-registered
tick function every couple of seconds, and invokes the LLM only when
an "interesting event" fires (verdict change, new feature, confidence
reversal, periodic heartbeat). Mirrors the UI's Live Mode but for
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
  # tighter accept threshold + heartbeat every 30 s
  scilink live /tmp/raman_live.txt --skill xrd \\
      --source-kind append_only --threshold 0.8 --heartbeat-sec 30

  # Custom output directory and tick cadence
  scilink live ~/scan.csv --skill xrd \\
      --tick-interval 1.0 --output-dir ./my_live_session

  # Replay a previously-recorded JSONL against a (possibly different)
  # trigger policy — useful for tuning thresholds
  scilink live --replay ./live_session_20260521_180448/live_ticks.jsonl \\
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
        help="File the instrument is writing to. Polled every "
             "--tick-interval seconds. Omit with --replay.",
    )
    parser.add_argument(
        "--skill", default="xrd",
        help="Skill name (or domain/name) whose live_tick.tick_fn "
             "drives the loop. Default: xrd.",
    )
    parser.add_argument(
        "--source-kind", choices=("mtime_poll", "append_only"),
        default="mtime_poll",
        help="How to detect new data. 'mtime_poll' (default) re-reads "
             "the whole file when its mtime advances; 'append_only' "
             "tracks a byte offset and returns only the newly-"
             "appended chunk.",
    )
    parser.add_argument(
        "--tick-interval", type=float, default=2.0,
        help="Seconds between ticks. Default: 2.0.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Optional cap on total session duration in seconds. "
             "Default: run until Ctrl-C.",
    )
    parser.add_argument(
        "--heartbeat-sec", type=float, default=60.0,
        help="Heartbeat trigger interval (LLM produces a routine "
             "narrative every N seconds even when nothing changed). "
             "Set to 0 to disable. Default: 60.",
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
    parser.add_argument(
        "--no-verdict-trigger", action="store_true",
        help="Disable the verdict-change trigger (rare — kept for replay tuning).",
    )
    parser.add_argument(
        "--no-new-feature-trigger", action="store_true",
        help="Disable the new-feature trigger.",
    )
    parser.add_argument(
        "--no-reversal-trigger", action="store_true",
        help="Disable the confidence-reversal trigger.",
    )
    parser.add_argument(
        "--model", default="claude-opus-4-6",
        help="LLM model to use for the slow loop. Default: claude-opus-4-6.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="LLM API key. Falls back to ANTHROPIC_API_KEY / "
             "CLAUDE_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Session directory (live_ticks.jsonl lives here). "
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
        help="Path to a previous session's live_ticks.jsonl. When set, "
             "the data_path argument is ignored and the recorded ticks "
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


def _build_policy(args):
    from scilink.agents.exp_agents.live_triggers import (
        ConfidenceReversalTrigger, HeartbeatTrigger, ManualTrigger,
        NewFeatureTrigger, ThresholdCrossTrigger, TriggerPolicy,
        VerdictChangeTrigger,
    )
    triggers = []
    if not args.no_verdict_trigger:
        triggers.append(VerdictChangeTrigger())
    if not args.no_new_feature_trigger:
        triggers.append(NewFeatureTrigger())
    if not args.no_reversal_trigger:
        triggers.append(ConfidenceReversalTrigger())
    if args.threshold is not None:
        triggers.append(ThresholdCrossTrigger(
            threshold=args.threshold, direction=args.threshold_direction,
        ))
    if args.heartbeat_sec and args.heartbeat_sec > 0:
        triggers.append(HeartbeatTrigger(interval_sec=args.heartbeat_sec))
    triggers.append(ManualTrigger())  # always available; we don't expose "interpret now" in CLI v1
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
    from scilink.agents.exp_agents.live_data_sources import (
        AppendOnlyFileSource, MtimePollFileSource,
    )
    from scilink.agents.exp_agents.live_session import LiveSession
    from scilink.skills.loader import load_skill, resolve_tick_fn

    if not args.data_path:
        log.error("data_path is required when not using --replay")
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

    # Skill resolution — accept either bare name or "domain/name"
    if "/" in args.skill and not args.skill.startswith("/"):
        domain, name = args.skill.split("/", 1)
    else:
        # Default domain for live tick currently is structure_matching for xrd;
        # the loader's cross-domain fallback will find it from curve_fitting too.
        name = args.skill
        domain = "structure_matching" if name == "xrd" else "curve_fitting"
    try:
        skill = load_skill(name, domain=domain)
    except Exception as e:
        log.error("Failed to load skill %s/%s: %s", domain, name, e)
        return 3
    tick_fn = resolve_tick_fn(skill.get("meta", {}))
    if tick_fn is None:
        log.error("Skill %s does not declare a live_tick.tick_fn. "
                  "Check the skill's frontmatter.", args.skill)
        return 3
    log.info("Skill loaded: %s/%s", domain, name)

    # Data source
    source_cls = (AppendOnlyFileSource if args.source_kind == "append_only"
                  else MtimePollFileSource)
    source = source_cls(args.data_path)
    log.info("Data source: %s (%s)", args.source_kind, args.data_path)

    # Orchestrator (AUTONOMOUS — no human-feedback prompts during live ticks)
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

    policy = _build_policy(args)

    def on_tick(result):
        log.info("[TICK] %s", _format_tick(result))

    def on_llm(event, result):
        log.info("[LLM ] %s", _format_llm(event, result))

    session = LiveSession(
        orchestrator=orch, data_source=source, tick_fn=tick_fn,
        tick_interval_sec=args.tick_interval,
        trigger_policy=policy,
        history_path=session_dir / "live_ticks.jsonl",
        skill_state=skill,
        on_tick=on_tick,
        on_llm_response=on_llm,
    )

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

    log.info("Starting LiveSession (tick interval %.1fs)", args.tick_interval)
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
    log.info("JSONL: %s", session_dir / "live_ticks.jsonl")
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

    def on_tick(result):
        log.info("[TICK] %s", _format_tick(result))

    def on_event(ev):
        log.info("[TRIG] %s  %s", ev.name,
                  {k: v for k, v in (ev.details or {}).items() if k != "tick_ref"})

    log.info("Replaying %s ...", args.replay)
    report = replay_jsonl(
        args.replay,
        trigger_policy=policy,
        speed=args.replay_speed,
        on_tick=on_tick,
        on_event=on_event,
        orchestrator=orch,
        llm_mode="redo" if args.replay_llm_redo else "skip",
    )
    log.info("Replay complete.")
    log.info("  ticks:           %d", report.tick_count)
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
