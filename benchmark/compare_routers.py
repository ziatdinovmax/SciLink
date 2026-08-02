"""Side-by-side comparison of test_router runs across models.

Reads every ``benchmark/outputs/test_router/<model-slug>/manifest.json``
and prints (and optionally writes) a markdown table with one row per
query and one column per model.  Highlights queries where the models
disagree.

Examples
--------
    python -m benchmark.compare_routers                # stdout only
    python -m benchmark.compare_routers --write        # also writes
                                                       # comparison.md alongside
    python -m benchmark.compare_routers --models sonnet,opus-4-6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


_DEFAULT_ROOT = Path(__file__).resolve().parent / "outputs" / "test_router"


# ──────────────────────────────────────────────────────────────────
#  Manifest discovery
# ──────────────────────────────────────────────────────────────────

def _discover_runs(root: Path,
                   model_filter: List[str] | None) -> Dict[str, dict]:
    """Walk root for ``<slug>/manifest.json``; return {slug → manifest}.
    Also accepts a flat ``root/manifest.json`` (legacy unnameespaced
    layout) and tags it as 'unknown'."""
    out: Dict[str, dict] = {}

    # legacy: a manifest sitting directly in root
    legacy = root / "manifest.json"
    if legacy.is_file():
        with open(legacy) as f:
            data = json.load(f)
        model = data.get("metrics", {}).get("model") or "unknown"
        out[model] = data

    # namespaced layout
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = child / "manifest.json"
        if not m.is_file():
            continue
        with open(m) as f:
            data = json.load(f)
        # prefer the model name stamped into metrics; fall back to dir name
        slug = data.get("metrics", {}).get("model") or child.name
        out[slug] = data

    if model_filter:
        out = {k: v for k, v in out.items()
               if any(s in k for s in model_filter)}
    return out


# ──────────────────────────────────────────────────────────────────
#  Per-case rendering
# ──────────────────────────────────────────────────────────────────

def _case_outcome(case: dict) -> Tuple[str, str]:
    """Return (label, marker) for one case.  Marker is ✓ / ✗ / · / err."""
    actual_scale  = case.get("actual", {}).get("scale")
    actual_engine = case.get("actual", {}).get("engine")
    if "capability gap" in (case.get("notes") or "").lower():
        body = (f"{actual_scale}/{actual_engine}"
                if actual_scale else "—")
        return body, "·"
    if case.get("passed"):
        return f"{actual_scale}/{actual_engine}", "✓"
    if actual_scale is None and actual_engine is None:
        return "None/None", "✗"
    return f"{actual_scale}/{actual_engine}", "✗"


def _render(runs: Dict[str, dict]) -> str:
    if not runs:
        return "(no manifests found)\n"

    models = list(runs.keys())
    # Build a stable case-id ordering from the first run
    first_cases = runs[models[0]]["cases"]
    case_ids: List[str] = [c["id"] for c in first_cases]

    lines: List[str] = []
    lines.append("# test_router :: cross-model comparison\n")

    # headline metrics row
    lines.append("## Headline metrics\n")
    lines.append(f"| metric | {' | '.join(models)} |")
    lines.append(f"|{'|'.join(['---'] * (1 + len(models)))}|")
    for metric in ("joint_accuracy", "scale_accuracy", "engine_accuracy",
                   "n_scorable", "n_capability_gaps"):
        row = [metric]
        for m in models:
            val = runs[m]["metrics"].get(metric)
            if isinstance(val, float):
                row.append(f"{val:.2f}")
            elif val is None:
                row.append("—")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # per-query table
    lines.append("## Per-query  ( ✓ pass · capability gap ✗ miss )\n")
    header = ["query (id)"] + models + ["agree?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cid in case_ids:
        row = [cid]
        outcomes: List[str] = []
        for m in models:
            case = next((c for c in runs[m]["cases"] if c["id"] == cid),
                        None)
            if case is None:
                row.append("—")
                outcomes.append("missing")
            else:
                body, mark = _case_outcome(case)
                row.append(f"{mark} {body}")
                outcomes.append(body)
        agree = "✓" if len(set(outcomes)) == 1 else "✗"
        row.append(agree)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # divergences (only)
    lines.append("## Where the models diverged\n")
    any_div = False
    for cid in case_ids:
        outcomes = []
        for m in models:
            case = next((c for c in runs[m]["cases"] if c["id"] == cid),
                        None)
            outcomes.append(_case_outcome(case)[0] if case else "missing")
        if len(set(outcomes)) > 1:
            any_div = True
            lines.append(f"- **{cid}**")
            for m, o in zip(models, outcomes):
                lines.append(f"  - `{m}` → {o}")
    if not any_div:
        lines.append("_(none — every model picked the same scale/engine "
                     "for every query)_")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_routers", description=__doc__)
    parser.add_argument("--root", default=str(_DEFAULT_ROOT),
                        help="directory holding <model-slug>/manifest.json files")
    parser.add_argument("--models", default=None,
                        help="optional comma-separated filter, e.g. "
                             "'sonnet,opus-4-6' — substring match against slug")
    parser.add_argument("--write", action="store_true",
                        help="also write the report to comparison.md "
                             "inside the root directory")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"!! no such directory: {root}", file=sys.stderr)
        return 2

    model_filter = (
        [m.strip() for m in args.models.split(",")] if args.models else None
    )
    runs = _discover_runs(root, model_filter)
    if not runs:
        print(f"!! no manifests under {root}", file=sys.stderr)
        return 2

    report = _render(runs)
    print(report)

    if args.write:
        out = root / "comparison.md"
        out.write_text(report)
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
