"""LLM usage per workspace: a ledger of every call, a budget, a period.

One server hosts one workspace, so the ledger is per process and exact for
the process; sessions are best-effort (see ``tracing.bind_session``). Records
go to a JSONL file (``SCILINK_USAGE_FILE``, else ``<session root>/usage.jsonl``)
so a redeploy keeps the count, and a ``period`` record marks where the
current billing period starts — the control plane opens a new period after
it has read the old one. The budget is a soft cap on tokens in the current
period (``SCILINK_TOKEN_BUDGET``): once spent, new turns are refused with a
message that says so, and the work already running finishes.

Imports nothing from the server package; the terminal shell can use it too.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


class UsageLedger:
    def __init__(self, path, budget_tokens: Optional[int] = None) -> None:
        self.path = Path(path)
        self.budget = int(budget_tokens) if budget_tokens else None
        self._lock = threading.Lock()
        self._reset_totals()
        self._load()

    # -- state ------------------------------------------------------------
    def _reset_totals(self) -> None:
        self.period_start = time.time()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.by_model: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        self.by_session: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def _apply(self, rec: Dict[str, Any]) -> None:
        if rec.get("kind") == "period":
            self._reset_totals()
            self.period_start = float(rec.get("ts") or time.time())
            return
        p, c = int(rec.get("prompt_tokens") or 0), int(rec.get("completion_tokens") or 0)
        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c
        self.seconds += float(rec.get("latency_s") or 0.0)
        for table, key in ((self.by_model, rec.get("model") or "unknown"),
                           (self.by_session, rec.get("session") or "unattributed")):
            row = table[str(key)]
            row["calls"] += 1
            row["prompt_tokens"] += p
            row["completion_tokens"] += c

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        self._apply(rec)
        except OSError:
            pass

    def _append(self, rec: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass                       # the count in memory still holds

    # -- the sink ---------------------------------------------------------
    def record(self, model, prompt_tokens, completion_tokens, latency_s, session) -> None:
        rec = {"ts": round(time.time(), 3), "model": model,
               "prompt_tokens": int(prompt_tokens or 0),
               "completion_tokens": int(completion_tokens or 0),
               "latency_s": round(float(latency_s or 0.0), 3), "session": session}
        with self._lock:
            self._apply(rec)
            self._append(rec)

    # -- answers ----------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def over_budget(self) -> bool:
        return self.budget is not None and self.total_tokens >= self.budget

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            out = {
                "period_start": self.period_start,
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_seconds": round(self.seconds, 1),
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
                "by_session": {k: dict(v) for k, v in self.by_session.items()},
                "budget_tokens": self.budget,
                "remaining_tokens": (None if self.budget is None
                                     else max(0, self.budget - self.total_tokens)),
                "over_budget": self.over_budget(),
                "file": str(self.path),
            }
        return out

    def new_period(self) -> Dict[str, Any]:
        """Close the current period: the totals start again from zero, the
        records stay in the file under the period marker."""
        with self._lock:
            self._reset_totals()
            self._append({"kind": "period", "ts": round(self.period_start, 3)})
        return self.summary()


def ledger_for(session_root, budget_env: str = "SCILINK_TOKEN_BUDGET") -> UsageLedger:
    path = os.environ.get("SCILINK_USAGE_FILE") or (Path(session_root) / "usage.jsonl")
    raw = os.environ.get(budget_env)
    budget = None
    if raw:
        try:
            budget = int(float(raw))
        except ValueError:
            budget = None
    return UsageLedger(path, budget)
