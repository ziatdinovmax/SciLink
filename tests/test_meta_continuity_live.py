#!/usr/bin/env python3
"""Live test — specialist continuity across meta delegations.

Does TWO delegations to each specialist (planning, analysis) through a
persistent meta child and confirms the second delegation sees the first:

  delegation 1  →  plant a reference token
  delegation 2  →  ask the specialist to recall the token

If the specialist is one persistent agent reused across delegations (the
design), delegation 2 recalls the token — which is what makes a refinement
delegation possible. Delegations are issued via MetaOrchestratorAgent._delegate
directly (the method behind the delegate_to_* tools) so the test is
deterministic and cheap — it does not spin the meta LLM.

Ad-hoc live test — NOT committed. Needs an LLM API key in the environment
(ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY, per --model).

    python tests/test_meta_continuity_live.py [--model NAME] [--base-url URL]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from scilink import auth
from scilink.agents.meta_agent.meta_orchestrator import (
    MetaOrchestratorAgent, MetaMode,
)


def _parse(result):
    try:
        return json.loads(result) if isinstance(result, str) else (result or {})
    except (ValueError, TypeError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    api_key = auth.get_api_key_for_model(args.model)
    if not api_key and not args.base_url:
        print(f"ERROR: no API key in environment for model '{args.model}'.")
        print("Set ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY.")
        return 2

    base = Path(tempfile.mkdtemp(prefix="meta_continuity_"))
    print(f"session dir : {base}")
    print(f"model       : {args.model}\n")

    meta = MetaOrchestratorAgent(
        base_dir=str(base / "meta_session"),
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
        meta_mode=MetaMode.AUTONOMOUS,   # children run end-to-end, no pauses
    )

    overall_ok = True
    for mode, token in (("planning", "PLN-TOKEN-7Q2X"),
                        ("analysis", "ANL-TOKEN-4K9M")):
        print(f"=== {mode.upper()} continuity ===")

        t1 = (
            "This is an internal bookkeeping step — no analysis or planning "
            f"work is needed. Note this reference token for later: {token}. "
            "Reply with just the word 'noted'. Do NOT call any tools."
        )
        r1 = _parse(meta._delegate(mode, t1, label=f"{mode} token plant"))
        child_after_1 = meta._children.get(mode)
        print(f"  delegation 1 status : {r1.get('status')}")

        t2 = (
            "Earlier in this session I gave you a reference token for "
            "bookkeeping. Reply with ONLY that exact token, nothing else. "
            "Do NOT call any tools."
        )
        r2 = _parse(meta._delegate(mode, t2, label=f"{mode} token recall"))
        child_after_2 = meta._children.get(mode)
        summary2 = r2.get("summary") or ""
        print(f"  delegation 2 status : {r2.get('status')}")
        print(f"  delegation 2 reply  : {summary2[:200]!r}")

        same_child = child_after_1 is child_after_2 and child_after_1 is not None
        recalled = token in summary2
        print(f"  same persistent child object : {same_child}")
        print(f"  delegation 2 recalled token  : {recalled}")
        if not (same_child and recalled):
            overall_ok = False
        print()

    print(f"delegation ledger entries: {len(meta._delegation_ledger)}")
    print()
    if overall_ok:
        print("RESULT: PASS — each specialist persists across delegations; "
              "a follow-up/refinement delegation sees the prior work.")
        return 0
    print("RESULT: FAIL — a specialist did not retain prior-delegation context.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
