#!/usr/bin/env python3
"""Live test — specialists build on / refine prior delegations.

ANALYSIS: delegation 1 fits the peaks of a synthetic two-peak spectrum;
delegation 2 asks a question answerable ONLY from those fit results (which
peak is larger, the amplitude ratio) and is told not to re-run — testing
that an analysis delegation builds on a previous analysis's results.

PLANNING: delegation 1 produces an initial experimental plan; delegation 2
feeds in NEW results for that plan and asks to refine it — testing that a
planning delegation refines a plan based on new results.

Delegations go through MetaOrchestratorAgent._delegate (the method behind
the delegate_to_* tools). Automated checks are soft (free-form output) — the
printed delegation-2 summaries are the real evidence.

Ad-hoc live test — NOT committed. Needs an LLM API key in the environment;
the planning half also needs a Gemini embedding key (GEMINI_API_KEY /
GOOGLE_API_KEY) for the planning knowledge base.

    python tests/test_meta_buildupon_live.py [--model NAME] [--base-url URL]
                                             [--only analysis|planning|both]
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from scilink import auth
from scilink.agents.meta_agent.meta_orchestrator import (
    MetaOrchestratorAgent, MetaMode,
)


def _parse(result):
    try:
        return json.loads(result) if isinstance(result, str) else (result or {})
    except (ValueError, TypeError):
        return {}


def _indent(text, n=6):
    pad = " " * n
    return "\n".join(pad + ln for ln in (text or "").splitlines())


def _show(label, res):
    print(f"\n  --- {label} ---")
    print(f"  status : {res.get('status')}")
    print(f"  summary:\n{_indent((res.get('summary') or '').strip())}")
    kf = res.get("key_findings") or []
    if kf:
        print(f"  key_findings: {kf}")
    fp = res.get("files_produced") or []
    if fp:
        print(f"  files_produced ({len(fp)}):")
        for f in fp:
            print(f"      {f}")
    if res.get("error"):
        print(f"  error: {res.get('error')}")


def _make_spectrum(path):
    """Synthetic intensity-vs-position curve: a large peak (amp 1.0 at
    position 30) and a small peak (amp 0.5 at position 70) — ratio 2:1."""
    x = np.linspace(0.0, 100.0, 400)
    rng = np.random.RandomState(0)

    def g(amp, center, width):
        return amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    y = g(1.0, 30.0, 4.0) + g(0.5, 70.0, 5.0) + 0.01 * rng.randn(x.size)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["position", "intensity"])
        for xi, yi in zip(x, y):
            w.writerow([f"{xi:.4f}", f"{yi:.6f}"])


def _analysis_test(meta, spectrum):
    """delegation 1 fits the peaks; delegation 2 must answer from that fit."""
    print("=" * 68)
    print("ANALYSIS — does delegation 2 build on delegation 1's results?")
    print("=" * 68)
    a1 = _parse(meta._delegate(
        "analysis",
        f"Analyze the 1-D spectrum at {spectrum}. It is a measured "
        f"intensity-vs-position curve with two peaks. Fit the peaks and "
        f"report each peak's fitted position and amplitude.",
        label="spectrum peak fit",
    ))
    _show("analysis delegation 1 (fit the peaks)", a1)

    analysis_child = meta._children.get("analysis")
    n_before = len(getattr(analysis_child, "analysis_results", []))

    a2 = _parse(meta._delegate(
        "analysis",
        "In your previous analysis you fitted the peaks of that spectrum. "
        "Using ONLY those existing fit results — do NOT run the analysis "
        "again — answer: which fitted peak has the LARGER amplitude (give "
        "its fitted position), and what is the ratio of the larger "
        "amplitude to the smaller amplitude?",
        label="compare fitted peaks",
    ))
    _show("analysis delegation 2 (build on the fit)", a2)

    n_after = len(getattr(analysis_child, "analysis_results", []))
    reran = n_after > n_before
    s2 = (a2.get("summary") or "").lower()
    print(f"\n  delegation 2 ran a NEW analysis: {reran} "
          f"({'re-ran' if reran else 'used prior results without re-running'})")
    return {
        "analysis builds on previous": (
            a1.get("status") == "success"
            and a2.get("status") == "success"
            and "peak 1" in s2          # identifies delegation 1's larger peak
            and not reran               # answered purely from prior results
        )
    }


def _planning_test(meta):
    """delegation 1 makes an initial plan; delegation 2 refines it on new data."""
    print("\n" + "=" * 68)
    print("PLANNING — does delegation 2 refine the plan from new results?")
    print("=" * 68)
    p1 = _parse(meta._delegate(
        "planning",
        "Create a brief initial experimental plan to optimize the yield of "
        "a generic chemical reaction. Propose exactly THREE candidate "
        "experiments, numbered 1-3, one or two sentences each. Keep it "
        "concise — no economic analysis and no code.",
        label="initial yield-optimization plan",
    ))
    _show("planning delegation 1 (initial plan)", p1)

    p2 = _parse(meta._delegate(
        "planning",
        "New experimental results have come in for the plan you just "
        "produced: candidate experiment #1 gave a very low yield (a failed "
        "direction); candidate experiment #2 gave the best yield so far (a "
        "promising direction). Refine your previous plan based on these new "
        "results: drop or rework experiment #1's direction, and build on "
        "experiment #2. Present the refined plan.",
        label="refine plan from new results",
    ))
    _show("planning delegation 2 (refined plan)", p2)

    s2 = (p2.get("summary") or "").lower()
    # The refine tool writes plan.json/.html into the per-delegation subdir
    # (Part 0 fix) — its presence in delegation 2's output proves a
    # plan-writing tool ran for this refinement, not just a chat reply.
    plan_artifact = any(
        Path(str(f)).name in ("plan.json", "plan.html")
        for f in (p2.get("files_produced") or [])
    )
    print(f"\n  delegation 2 wrote a plan artifact (plan.json/.html): {plan_artifact}")
    return {
        "planning refines on new results": (
            p1.get("status") == "success"
            and p2.get("status") == "success"
            and any(w in s2 for w in ("refin", "updat", "revis"))
        ),
        "planning wrote a refined plan artifact": plan_artifact,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--only", choices=["analysis", "planning", "both"],
                    default="both")
    args = ap.parse_args()

    api_key = auth.get_api_key_for_model(args.model)
    if not api_key and not args.base_url:
        print(f"ERROR: no API key in environment for model '{args.model}'.")
        return 2

    # Planning's knowledge base embeds with a Gemini model — needs its own key.
    emb_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    base = Path(tempfile.mkdtemp(prefix="meta_buildupon_"))
    spectrum = base / "spectrum.csv"
    _make_spectrum(spectrum)
    print(f"session dir   : {base}")
    print(f"model         : {args.model}")
    print(f"embedding key : {'found' if emb_key else 'NOT found (planning KB will fail)'}")
    print(f"running       : {args.only}\n")

    meta = MetaOrchestratorAgent(
        base_dir=str(base / "meta_session"),
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
        embedding_api_key=emb_key,
        meta_mode=MetaMode.AUTONOMOUS,
    )

    verdict = {}
    if args.only in ("analysis", "both"):
        verdict.update(_analysis_test(meta, spectrum))
    if args.only in ("planning", "both"):
        verdict.update(_planning_test(meta))

    print("\n" + "=" * 68)
    print(f"delegation ledger entries: {len(meta._delegation_ledger)}")
    for k, v in verdict.items():
        print(f"  {k:38s}: {'PASS' if v else 'CHECK'}")
    print("=" * 68)
    print("\nNOTE: checks are soft — read the delegation-2 summaries above to "
          "confirm each genuinely built on / refined delegation 1.")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
