#!/usr/bin/env python3
"""Build the full COD fingerprint library for ``search_match_pattern``.

Shard-at-a-time, resumable, disk-bounded pipeline over the canonical COD rsync
mirror (rsync://www.crystallography.net/cif/ — 9 top-level shards, ~500k CIFs,
~100+ GB of CIF text total):

    for each shard 1..9:
        rsync the shard              (resumable; ~10-25 GB on disk at once)
        fingerprint it in parallel   (N worker processes, chunked file lists)
        write shard parquet, DELETE the CIFs
    concatenate shard parquets, dedup across shards -> cod_fingerprints.parquet

Only the parquet survives (~100s of MB for the full COD), so peak disk usage is
a single shard. A state file records completed shards — rerun the script after
any interruption and it continues where it stopped. Expect hours of download +
tens of CPU-hours of pattern computation for the full mirror; run it on a
machine you can leave alone, or pass --shards to build a subset first.

Usage:
    python scripts/build_cod_fingerprints.py --work ~/cod_fp_build \
        --out ~/cod_fingerprints.parquet [--shards 1,2] [--workers 8]

Then:  export SCILINK_XRD_FINGERPRINT_DB=~/cod_fingerprints.parquet
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

RSYNC_ROOT = "rsync://www.crystallography.net/cif/"
ALL_SHARDS = [str(i) for i in range(1, 10)]


def _fingerprint_chunk(args):
    """Worker: build a partial parquet from a chunk of CIFs (symlink dir)."""
    chunk_dir, out_path = args
    from scilink.skills.structure_matching.xrd.fingerprint import (
        build_fingerprint_library)
    try:
        summary = build_fingerprint_library(chunk_dir, out_path)
        return {"out": out_path, **summary}
    except Exception as exc:  # a poisoned chunk must not sink the shard
        return {"out": out_path, "error": str(exc)[:300]}


def build_shard(shard: str, work: str, workers: int) -> str:
    import pandas as pd

    tag = shard.replace("/", "_")   # nested sub-shards (e.g. '1/00') allowed
    shard_parquet = os.path.join(work, f"shard_{tag}.parquet")
    if os.path.exists(shard_parquet):
        print(f"[shard {shard}] parquet exists — skipping")
        return shard_parquet

    cif_dir = os.path.join(work, f"cif_{tag}")
    os.makedirs(cif_dir, exist_ok=True)
    print(f"[shard {shard}] rsync -> {cif_dir}", flush=True)
    subprocess.run(["rsync", "-a", "--delete", f"{RSYNC_ROOT}{shard}/", cif_dir],
                   check=True)

    files = []
    for root, _d, fns in os.walk(cif_dir):
        files += [os.path.join(root, f) for f in fns if f.endswith(".cif")]
    print(f"[shard {shard}] {len(files)} CIFs; fingerprinting on {workers} workers",
          flush=True)

    # chunk via symlink dirs (build_fingerprint_library walks a directory).
    # MANY more chunks than workers: COD entry cost varies by orders of
    # magnitude (large organic cells), so coarse chunks leave one straggler
    # holding the whole shard (observed: 7/8 chunks done, 1 at 11+ CPU-hours).
    chunks_root = os.path.join(work, f"chunks_{tag}")
    shutil.rmtree(chunks_root, ignore_errors=True)
    n_chunks = workers * 8
    tasks = []
    for w in range(n_chunks):
        chunk_files = files[w::n_chunks]
        if not chunk_files:
            continue
        cdir = os.path.join(chunks_root, str(w))
        os.makedirs(cdir, exist_ok=True)
        for f in chunk_files:
            os.symlink(f, os.path.join(cdir, os.path.basename(f)))
        tasks.append((cdir, os.path.join(chunks_root, f"part_{w}.parquet")))

    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_fingerprint_chunk, tasks, chunksize=1)
    n_ok = sum(r.get("n_indexed", 0) for r in results)
    errs = [r for r in results if "error" in r]
    print(f"[shard {shard}] indexed {n_ok} in {time.time()-t0:.0f}s; "
          f"{len(errs)} chunk errors", flush=True)

    parts = [r["out"] for r in results if os.path.exists(r.get("out", ""))]
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(shard_parquet, index=False)

    shutil.rmtree(cif_dir, ignore_errors=True)     # reclaim the shard's disk
    shutil.rmtree(chunks_root, ignore_errors=True)
    return shard_parquet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="working directory")
    ap.add_argument("--out", required=True, help="final parquet path")
    ap.add_argument("--shards", default=",".join(ALL_SHARDS),
                    help="comma list of COD top-level shards (default: all)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    import pandas as pd
    os.makedirs(args.work, exist_ok=True)
    state_path = os.path.join(args.work, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": []}

    shard_parquets = []
    for shard in args.shards.split(","):
        shard = shard.strip()
        sp = build_shard(shard, args.work, args.workers)
        shard_parquets.append(sp)
        if shard not in state["done"]:
            state["done"].append(shard)
            json.dump(state, open(state_path, "w"))

    print("concatenating + cross-shard dedup", flush=True)
    df = pd.concat([pd.read_parquet(p) for p in shard_parquets], ignore_index=True)
    # same dedup key as the builder, applied across shards
    key = (df["formula"] + "|"
           + df[["a", "b", "c"]].round(2).astype(str).agg("-".join, axis=1)
           + "|" + df["volume"].round(1).astype(str))
    before = len(df)
    df = df[~key.duplicated()].reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"DONE: {len(df)} entries ({before - len(df)} cross-shard duplicates "
          f"dropped) -> {args.out}")
    print(f"export SCILINK_XRD_FINGERPRINT_DB={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
