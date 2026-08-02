"""Router stability test — pick a small set of representative prompts
and fire each at the router N times to measure how variable the
agent's (scale, engine) choice is.

Defaults to one easy + one medium + one hard prompt:

  * ``pd_01_lattice_cu``        (easy)    — clean periodic_dft case
  * ``mlip_02_battery_cathode`` (medium)  — CHGNet vs MACE pick is the
                                            qualitative signal we're
                                            watching across runs
  * ``amb_03_melting_point_cu`` (hard)    — genuinely open-ended;
                                            three valid scales

Use ``--queries`` to override the picks.

For each prompt, reports the distribution of (scale/engine) picks, the
% landed on the most common choice (a "stability score"), and the
number of distinct picks.

Same --model namespacing as test_router so outputs land at
``benchmark/outputs/test_router_variability/<model-slug>/`` and the
runner composes with the rest of the suite.

Examples
--------
    python -m benchmark.runner test_router_variability                # 8 trials × 3 prompts, default model
    python -m benchmark.runner test_router_variability --n-trials 12  # more shots
    python -m benchmark.runner test_router_variability \\
        --queries pd_01_lattice_cu,amb_05_water_at_interface --n-trials 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ._score import RunnerManifest, write_manifest, write_summary_md
from .queries import ALL_QUERIES, RouterQuery
from .test_router import (
    _build_model, _build_mock_software, _model_slug,
)


_DEFAULT_PICKS = (
    "pd_01_lattice_cu",          # easy
    "mlip_02_battery_cathode",   # medium
    "amb_03_melting_point_cu",   # hard
)


# ──────────────────────────────────────────────────────────────────
#  One prompt × N trials
# ──────────────────────────────────────────────────────────────────

def _run_trials(router, query: RouterQuery, n: int) -> List[Dict]:
    """Fire ``n`` router calls on the same prompt; return the decisions."""
    trials: List[Dict] = []
    for i in range(n):
        try:
            d = router.route(query.prompt)
        except Exception as exc:
            d = {"error": f"router raised: {exc!r}"}
        trials.append({
            "trial":  i + 1,
            "scale":  d.get("scale"),
            "engine": d.get("engine"),
            "reason": d.get("reason", ""),
            "error":  d.get("error"),
        })
    return trials


def _summarise_trials(trials: List[Dict]) -> Dict:
    """Distribution + stability stats for one prompt's trials."""
    picks: Counter = Counter()
    for t in trials:
        s, e = t.get("scale"), t.get("engine")
        picks[f"{s}/{e}"] += 1
    n = sum(picks.values()) or 1
    most_common_pick, most_common_n = picks.most_common(1)[0]
    return {
        "n_trials":           n,
        "picks":              dict(picks.most_common()),
        "unique_picks":       len(picks),
        "most_common_pick":   most_common_pick,
        "stability":          round(most_common_n / n, 3),  # 1.0 = perfectly stable
    }


# ──────────────────────────────────────────────────────────────────
#  Markdown summary  (richer than the default _score writer)
# ──────────────────────────────────────────────────────────────────

def _render_summary_md(model: str,
                       n_trials: int,
                       per_prompt: Dict[str, Dict]) -> str:
    lines: List[str] = []
    lines.append(f"# test_router_variability :: {model}")
    lines.append("")
    lines.append(f"- trials per prompt: **{n_trials}**")
    lines.append(f"- prompts: {len(per_prompt)}")
    lines.append("")
    # one-line stability summary
    avg_stab = (sum(p["summary"]["stability"] for p in per_prompt.values())
                / max(1, len(per_prompt)))
    lines.append(f"- mean stability across prompts: **{avg_stab:.2f}** "
                 f"(1.00 = identical pick every trial)")
    lines.append("")

    for qid, body in per_prompt.items():
        q = body["query"]
        s = body["summary"]
        lines.append(f"## {qid}  ({q['difficulty']})\n")
        lines.append(f"> {q['prompt']}")
        lines.append("")
        lines.append(f"- stability: **{s['stability']:.2f}**   "
                     f"({s['unique_picks']} distinct pick"
                     f"{'s' if s['unique_picks'] > 1 else ''} across "
                     f"{s['n_trials']} trials)")
        lines.append(f"- most common: `{s['most_common_pick']}`")
        lines.append("")
        lines.append("| pick | count | fraction |")
        lines.append("|---|---|---|")
        for pick, count in s["picks"].items():
            frac = count / s["n_trials"]
            lines.append(f"| `{pick}` | {count} | {frac:.2f} |")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def _pick_queries(ids: List[str]) -> List[RouterQuery]:
    by_id = {q.id: q for q in ALL_QUERIES}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"!! unknown query id(s): {missing}\n"
                         f"   known: {sorted(by_id)}")
    return [by_id[i] for i in ids]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="test_router_variability", description=__doc__)
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--n-trials", type=int, default=8,
                        help="number of router calls per prompt (default 8)")
    parser.add_argument("--queries",  default=",".join(_DEFAULT_PICKS),
                        help="comma-separated query ids to test")
    parser.add_argument("--out-dir",  default=None,
                        help="output directory.  Default auto-namespaces "
                             "under benchmark/outputs/test_router_variability/"
                             "<model-slug>/")
    parser.add_argument("--use-real-software", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pick_ids = [q.strip() for q in args.queries.split(",") if q.strip()]
    queries = _pick_queries(pick_ids)

    print(f"test_router_variability :: {len(queries)} prompts × "
          f"{args.n_trials} trials  model={args.model}")
    for q in queries:
        print(f"  [{q.difficulty:<6}] {q.id:<26}  expect={q.expected_scale}")

    if args.dry_run:
        return 0

    # Build the router (lazy import so dry-run works without API key)
    from scilink.agents.sim_agents.simulation_router import (
        SimulationRouter, discover_scale_agents,
    )
    model = _build_model(args.model, args.api_key, args.base_url)
    if args.use_real_software:
        print("router :: using AvailableSoftware.auto() (real installed set)")
        router = SimulationRouter(model=model)
    else:
        mock = _build_mock_software()
        scales = sorted(mock._data.keys())
        print(f"router :: using mock software (every engine available, "
              f"scales={scales})")
        router = SimulationRouter(model=model, available_software=mock)

    per_prompt: Dict[str, Dict] = {}
    for q in queries:
        print(f"\n— {q.id} ({q.difficulty}) —")
        trials = _run_trials(router, q, args.n_trials)
        summary = _summarise_trials(trials)
        per_prompt[q.id] = {
            "query": {"id": q.id, "prompt": q.prompt,
                      "difficulty": q.difficulty,
                      "expected_scale": q.expected_scale},
            "trials":  trials,
            "summary": summary,
        }
        # printout: one line per trial + stability line
        for t in trials:
            mark = "✓" if t.get("scale") and not t.get("error") else "✗"
            print(f"  {mark} trial {t['trial']:2d}: "
                  f"{t.get('scale')}/{t.get('engine')}")
        print(f"  → stability {summary['stability']:.2f}   "
              f"({summary['unique_picks']} distinct pick"
              f"{'s' if summary['unique_picks'] > 1 else ''})")
        print(f"  → most common: {summary['most_common_pick']}  "
              f"({summary['picks'][summary['most_common_pick']]}"
              f"/{summary['n_trials']})")

    # ── Persist ─────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(
        "benchmark/outputs/test_router_variability", _model_slug(args.model))
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "runner":         "test_router_variability",
        "mode":           "local",
        "model":          args.model,
        "n_trials":       args.n_trials,
        "prompts":        per_prompt,
        "metrics": {
            "model":         args.model,
            "n_trials":      args.n_trials,
            "n_prompts":     len(per_prompt),
            "mean_stability": round(
                sum(p["summary"]["stability"] for p in per_prompt.values())
                / max(1, len(per_prompt)),
                3),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(payload, f, indent=2)

    md = _render_summary_md(args.model, args.n_trials, per_prompt)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)

    print(f"\nwrote {out_dir}/manifest.json + summary.md")
    print(f"mean stability across {len(per_prompt)} prompts: "
          f"{payload['metrics']['mean_stability']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
