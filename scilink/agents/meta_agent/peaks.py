"""What an item class was measured to cost: peak memory and tokens.

Fan-out's admission estimates memory from input bytes
(``fanout._branch_mem_estimate``). The 1024 x 1024 image that froze an 8 GB
laptop is about 8 MB; the scripts that ran on it peaked at 6 to 8 GB, loading
a DCNN ensemble, not the input. No estimate from the input can see that. So
every item that ran as a process (``placements.LocalProcess``) records what
its tree held at its peak, and what it spent in tokens, against its item
CLASS, and the next item of that class is sized from the record.

The class is computable from the spec alone — it has to be known before the
item runs — so it is the mode, the kind of data (by the largest unit's file
type), that unit's in-memory size bucket (powers of two of a megabyte) and
how many units there are. Two cubes of the same size and kind are one
class whatever the agent then chooses to do with them, which is the
granularity the measurement can honestly support. The table keeps the MAX
seen per class: a class is sized by its worst run, never by its average.
Recorded: a run that did its class's work (peak and tokens), and a run that
ended on memory — killed by the system, or cancelled by the guard — whose
peak is a fact about this machine and a lower bound of what the class
needs (peak only; it can only raise the max). A run that failed before it
started would measure its imports, and is not.

One table per SciLink home (``measured_items.json``), written under the
same file lock as the other shared stores; a thread item, whose peak cannot
be split from the process's, records nothing.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

TABLE_NAME = "measured_items.json"
#: What admission plans for over a class's measured peak: the next run of a
#: class is not the one that was measured.
PEAK_HEADROOM = 1.2
_UNITS_BUCKETS = ((1, "1"), (4, "few"), (32, "many"))
_KINDS = {
    "image": {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".dm3", ".dm4", ".mrc", ".ser"},
    "curve": {".csv", ".txt", ".tsv", ".dat", ".xy", ".xye", ".spc", ".spe", ".chi", ".xrdml"},
    "array": {".npy", ".npz", ".mat", ".raw"},
    "hdf5": {".h5", ".hdf5", ".nxs", ".emd"},
}


def table_path() -> Path:
    from ...skills.loader import scilink_home
    return scilink_home() / TABLE_NAME


def _kind(suffix: str) -> str:
    for kind, suffixes in _KINDS.items():
        if suffix in suffixes:
            return kind
    return "other"


def _bucket_mb(nbytes: float) -> str:
    mb = max(nbytes, 1.0) / 1e6
    return f"{int(2 ** math.ceil(math.log2(mb))) if mb > 1 else 1}MB"


def _units_bucket(n: int) -> str:
    for bound, name in _UNITS_BUCKETS:
        if n <= bound:
            return name
    return "many"


def item_class(item: dict) -> str:
    """``mode[:kind:largest-unit-bucket:units[:wN]]`` — see the module
    docstring. ``:wN`` is the series' replay workers when more than one
    (#750's own resolver and cap: a series' peak is its units in flight at
    once, so a four-worker run and a one-worker run of the same cubes are
    two classes). An analysis item with data it cannot read is
    ``analysis:unreadable``."""
    mode = str(item.get("mode") or "")
    if mode != "analysis" or not item.get("data_path"):
        return mode
    from ...utils.workers import resolve_workers
    from . import fanout as fo
    try:
        files = fo._data_files(Path(str(item["data_path"])).expanduser(), item.get("pattern"))
        if not files:
            return f"{mode}:nodata"
        largest = max(files, key=fo._in_memory_bytes)
        workers = min(resolve_workers(item.get("series_workers"), "SCILINK_HS_SERIES_WORKERS", 1),
                      max(len(files) - 1, 1))
        return (f"{mode}:{_kind(largest.suffix.lower())}:{_bucket_mb(fo._in_memory_bytes(largest))}"
                f":{_units_bucket(len(files))}" + (f":w{workers}" if workers > 1 else ""))
    except Exception:  # noqa: BLE001 - a class must never break an item
        return f"{mode}:unreadable"


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def measured(cls: str, path: Optional[Path] = None) -> Optional[dict]:
    """The class's record — ``{peak_rss_bytes, tokens, runs, last}`` — or
    ``None`` when nothing of the class was measured yet."""
    row = _load(path or table_path()).get(cls)
    return dict(row) if isinstance(row, dict) and row.get("runs") else None


def record(cls: str, *, peak_rss_bytes: Optional[float] = None, tokens: Optional[int] = None,
           path: Optional[Path] = None) -> dict:
    """Fold one run into the class's record (max of each figure, one more
    run) and return the record. Nothing to record is a no-op."""
    if not cls or (not peak_rss_bytes and not tokens):
        return measured(cls, path) or {}
    from ...utils.file_lock import path_lock
    path = path or table_path()
    with path_lock(path, label="the measured-items table"):
        table = _load(path)
        row = table.get(cls) if isinstance(table.get(cls), dict) else {}
        row = {"peak_rss_bytes": max(float(row.get("peak_rss_bytes") or 0.0), float(peak_rss_bytes or 0.0)) or None,
               "tokens": max(int(row.get("tokens") or 0), int(tokens or 0)) or None,
               "runs": int(row.get("runs") or 0) + 1, "last": round(time.time())}
        table[cls] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    return row


def forget(cls: str, path: Optional[Path] = None) -> bool:
    """Drop a class's record (a fixed leak, a new version, a home shared
    across machines): the next run of the class is sized from the input
    again and measured afresh. Returns whether there was one."""
    from ...utils.file_lock import path_lock
    path = path or table_path()
    with path_lock(path, label="the measured-items table"):
        table = _load(path)
        if cls not in table:
            return False
        del table[cls]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    return True


def measured_peak(cls: str, path: Optional[Path] = None) -> Optional[float]:
    """The raw peak a class was measured at (no headroom): what a refusal
    is judged on — a class that ran here must not be refused here."""
    row = measured(cls, path)
    return float(row["peak_rss_bytes"]) if row and row.get("peak_rss_bytes") else None


def peak_estimate(cls: str, path: Optional[Path] = None) -> Optional[float]:
    """What admission plans for a class that was measured: its peak with
    ``PEAK_HEADROOM``; ``None`` for an unmeasured class (the caller falls
    back to the input-based estimate)."""
    row = measured(cls, path)
    if not row or not row.get("peak_rss_bytes"):
        return None
    return float(row["peak_rss_bytes"]) * PEAK_HEADROOM


def token_estimate(cls: str, path: Optional[Path] = None) -> Optional[int]:
    row = measured(cls, path)
    if not row or not row.get("tokens"):
        return None
    return int(row["tokens"])
