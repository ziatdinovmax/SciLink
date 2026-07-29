"""Offline tests for the resource guards + figure-integrity audit rule
(post-rerun hardening items: memory-aware fan-out scheduling, RAM-aware
fit_per_pixel worker cap, fusion-audit figure integrity).

  conda run -n scilink python tests/test_resource_guards.py
"""
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    import scilink.agents.meta_agent.fanout as fanout

    # ------------------------------------------------------------------
    print("1) branch memory estimator:")
    d = tempfile.mkdtemp()
    big = Path(d) / "cube.npy"
    big.write_bytes(b"\0" * 200_000_000)          # 200 MB input
    est_big = fanout._branch_mem_estimate({"data_path": str(big)})
    check("large input -> data multiple + pool overhead",
          est_big > 200e6 * fanout._BRANCH_MEM_FACTOR
          and est_big >= fanout._BRANCH_POOL_OVERHEAD)
    small = Path(d) / "spec.txt"
    small.write_bytes(b"\0" * 10_000)
    est_small = fanout._branch_mem_estimate({"data_path": str(small)})
    check("small input -> floor, no pool overhead",
          est_small == fanout._BRANCH_MEM_FLOOR)
    check("missing path -> floor (never raises)",
          fanout._branch_mem_estimate({"data_path": "/nope"}) ==
          fanout._BRANCH_MEM_FLOOR)
    for f in Path(d).glob("*"):
        f.write_bytes(b"\0" * 1000)
    est_dir = fanout._branch_mem_estimate({"data_path": d, "pattern": "*.txt"})
    check("directory + pattern sums matching files only",
          est_dir == fanout._BRANCH_MEM_FLOOR)

    # ------------------------------------------------------------------
    print("2) admission guard:")
    fanout._mem_running.clear()
    fanout._admit_branch("a", 1e12, "first")      # huge, but nothing running
    check("first branch always admitted (progress guarantee)",
          "a" in fanout._mem_running)

    admitted = threading.Event()

    def _second():
        fanout._admit_branch("b", 1e15, "second")  # cannot fit while a runs
        admitted.set()

    with mock.patch.object(fanout, "_available_memory",
                           return_value=2e9):
        t = threading.Thread(target=_second, daemon=True)
        t.start()
        time.sleep(0.5)
        check("oversized branch HELD while another runs",
              not admitted.is_set())
        fanout._release_branch("a")
        t.join(timeout=5)
        check("held branch admitted when the runner releases "
              "(progress guarantee again)", admitted.is_set())
    fanout._mem_running.clear()

    with mock.patch.object(fanout, "_available_memory", return_value=None):
        fanout._admit_branch("c", 1e15, "no-psutil")
        check("unknown available memory -> guard disables (admits)",
              "c" in fanout._mem_running)
    fanout._mem_running.clear()

    check("env override plumbed",
          isinstance(fanout.FANOUT_MAX_WORKERS, int))

    # ------------------------------------------------------------------
    print("3) RAM-aware fit_per_pixel worker cap:")
    from scilink.skills._shared import parallel_pixel_fit as ppf
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 900, 60)
    cube = (5 + 3 * np.exp(-0.5 * ((wl - 600) / 20) ** 2)[None, None, :]
            + rng.normal(0, 0.3, (16, 16, wl.size)))
    model = [{"type": "gaussian", "window": (550, 650)}, "constant"]

    class _VM:
        available = 2.2e9        # room for ~3 workers after the 1 GB margin

    with mock.patch("psutil.virtual_memory", return_value=_VM()):
        r = ppf.fit_per_pixel(cube, wl, model, n_jobs=-1, chunk_size=64)
    check("default worker count capped by available RAM",
          r["stats"]["n_jobs"] <= 3
          and any("capped by available memory" in n for n in r["notes"]))
    check("capped run still fits correctly", r["coverage"].mean() > 0.95)
    with mock.patch("psutil.virtual_memory", return_value=_VM()):
        r2 = ppf.fit_per_pixel(cube, wl, model, n_jobs=4, chunk_size=64)
    check("explicit n_jobs honored (no silent cap)",
          not any("capped" in n for n in r2["notes"]))

    # ------------------------------------------------------------------
    print("4) fusion audit figure-integrity rule:")
    t = fanout.FUSION_VERIFICATION_INSTRUCTIONS
    check("audit checklist carries FIGURE INTEGRITY",
          "FIGURE INTEGRITY" in t and "EMPTY" in t and "refine" in t)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"RESOURCE GUARDS: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
