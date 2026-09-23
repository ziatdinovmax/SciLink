"""Fakes for the terminal-shell tests: an orchestrator that narrates like
the real ones (tool calls, a thought, a HITL question, an answer) and a
mode adapter that builds it without any LLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from scilink import hitl
from scilink.cli.shell.modes import ModeAdapter


class FakeOrchestrator:
    """``chat()`` prints the narration the real orchestrators print, asks
    one question through the HITL chokepoint, and returns markdown.

    ``self_prints_answer`` mimics the analyze/plan orchestrators, which
    print a ``🤖 Agent:`` block themselves before returning the text.
    ``block_until_stopped`` keeps printing until the capture's stop lands
    (the Ctrl+C path). ``raise_error`` raises inside the turn.
    """

    def __init__(self, base_dir: str = ".", *, self_prints_answer: bool = False,
                 block_until_stopped: bool = False, raise_error: bool = False,
                 question_kind: str = "confirm", ask: bool = True) -> None:
        self.base_dir = base_dir
        self.self_prints_answer = self_prints_answer
        self.block_until_stopped = block_until_stopped
        self.raise_error = raise_error
        self.question_kind = question_kind
        self.ask = ask
        self.message_count = 0
        self.mode = "autopilot"
        self.answers: List[str] = []
        self.checkpoints = 0
        self.last_task = None

    def chat(self, user_input: str) -> str:
        self.message_count += 1
        print("  🔧 Calling tool: delegate_to_analysis")
        print("  ⏳ Waiting for meta-orchestrator response ...")
        print("  💭 Deciding which specialist fits")
        print("     the data looks like a spectrum")
        print("Executing generated code")
        if self.block_until_stopped:
            while True:
                time.sleep(0.02)
                print("still working")
        if self.raise_error:
            raise RuntimeError("boom")
        if self.ask:
            print("=" * 40)
            print("🙋 REQUESTING FEEDBACK")   # the banner the presenter keys on
            answer = hitl.request_human_feedback(
                "Review the plan and press Enter to approve:", kind=self.question_kind,
                default="")
            self.answers.append(answer)
        text = f"## Result\n\nThe answer to *{user_input}* is **42**."
        if self.self_prints_answer:
            print("\n🤖 Agent:")
            print(text)
        return text

    def run_task(self, task: str, context=None, autonomy=None, **kw) -> Dict[str, Any]:
        self.last_task = task
        print("  🔧 Calling tool: run_analysis")
        return {"status": "success", "task": task, "summary": "done: " + task,
                "files_produced": ["a.png"], "key_findings": ["finding one"],
                "suggested_followups": [], "warnings": []}

    def save_checkpoint(self) -> str:
        self.checkpoints += 1
        print("💾 Checkpoint saved")
        path = Path(self.base_dir) / "checkpoint.json"
        path.write_text(json.dumps({"analysis_results": []}))
        return str(path)

    def get_human_feedback_setting(self) -> bool:
        return self.mode != "autonomous"


class FakeAdapter(ModeAdapter):
    key = "meta"
    description = "fake"
    epilog = ""

    def __init__(self, **orchestrator_kwargs) -> None:
        self.orchestrator_kwargs = orchestrator_kwargs
        self.built: List[FakeOrchestrator] = []

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="test-model")
        p.add_argument("--base-url", dest="base_url", default=None)
        p.add_argument("--api-key", dest="api_key", default="k")
        p.add_argument("--mode", default="autopilot")
        p.add_argument("--session-dir", dest="session_dir", default=None)
        p.add_argument("--restore", action="store_true")
        p.add_argument("--message", dest="initial_message", default=None)

    def build(self, args, creds, session_dir, *, restore, extras):
        agent = FakeOrchestrator(str(session_dir), **self.orchestrator_kwargs)
        agent.restored = restore
        self.built.append(agent)
        return agent

    def get_autonomy(self, agent) -> str:
        return agent.mode

    def set_autonomy(self, agent, level: str) -> None:
        agent.mode = level

    def status_fields(self, agent):
        return [("Fake field", "fake value")]

    def initial_turns(self, args, agent):
        return [args.initial_message] if getattr(args, "initial_message", None) else []

    def headless_run(self, agent, task):
        return agent.run_task(task)


def make_args(adapter, argv=()) -> argparse.Namespace:
    from scilink.cli.shell.app import build_parser
    return build_parser(adapter).parse_args(list(argv))
