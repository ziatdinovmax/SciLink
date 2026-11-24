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
        # What each session has cost over its whole life: NOT reset by a new
        # period, since "what this session has cost" is asked of the session.
        self.session_lifetime: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "cost_usd": 0.0, "priced_calls": 0})
        self._reset_totals()
        self._load()

    # -- state ------------------------------------------------------------
    def _reset_totals(self) -> None:
        self.period_start = time.time()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        # The dollar estimate of the calls recorded WITH one (``pricing``:
        # a provider's own API) and how many calls that is. A record written
        # before the estimate existed, or for a route with no known price,
        # adds to neither: the two are shown together, never the sum alone.
        self.cost_usd = 0.0
        self.priced_calls = 0
        row = lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,   # noqa: E731
                       "cost_usd": 0.0, "priced_calls": 0}
        self.by_model: Dict[str, Dict[str, float]] = defaultdict(row)
        self.by_session: Dict[str, Dict[str, float]] = defaultdict(row)
        # Only calls made for a worker (a fan-out branch) land
        # here; a session's own calls stay in by_session alone.
        self.by_worker: Dict[str, Dict[str, float]] = defaultdict(row)

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
        cost = rec.get("cost_usd")
        cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        if cost is not None:
            self.cost_usd += cost
            self.priced_calls += 1
        life = self.session_lifetime[str(rec.get("session") or "unattributed")]
        life["calls"] += 1
        life["cost_usd"] += cost or 0.0
        life["priced_calls"] += 0 if cost is None else 1
        tables = [(self.by_model, rec.get("model") or "unknown"),
                  (self.by_session, rec.get("session") or "unattributed")]
        if rec.get("worker"):
            # Keyed within the session: two sessions' workers may share a label.
            tables.append((self.by_worker, f"{rec.get('session') or 'unattributed'}/{rec['worker']}"))
        for table, key in tables:
            row = table[str(key)]
            row["calls"] += 1
            row["prompt_tokens"] += p
            row["completion_tokens"] += c
            if cost is not None:
                row["cost_usd"] += cost
                row["priced_calls"] += 1

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
    def record(self, model, prompt_tokens, completion_tokens, latency_s, session,
               worker=None, cost_usd=None) -> None:
        rec = {"ts": round(time.time(), 3), "model": model,
               "prompt_tokens": int(prompt_tokens or 0),
               "completion_tokens": int(completion_tokens or 0),
               "latency_s": round(float(latency_s or 0.0), 3), "session": session}
        if worker:
            rec["worker"] = worker
        if cost_usd is not None:             # the call's dollar estimate, when its route has a price
            rec["cost_usd"] = round(float(cost_usd), 6)
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
        from . import pricing

        def rows(table):
            # ``cost_text`` is the figure as a surface shows it ("" when no
            # call of the row had a price): one formatter, here, for all.
            return {k: {**v, "cost_usd": round(v["cost_usd"], 6),
                        "cost_text": pricing.cost_text(v["cost_usd"], v["calls"], v["priced_calls"])}
                    for k, v in table.items()}

        with self._lock:
            out = {
                "period_start": self.period_start,
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_seconds": round(self.seconds, 1),
                "cost_usd": round(self.cost_usd, 6),
                "priced_calls": self.priced_calls,
                "cost_text": pricing.cost_text(self.cost_usd, self.calls, self.priced_calls),
                "by_model": rows(self.by_model),
                "by_session": rows(self.by_session),
                "by_worker": rows(self.by_worker),
                "budget_tokens": self.budget,
                "remaining_tokens": (None if self.budget is None
                                     else max(0, self.budget - self.total_tokens)),
                "over_budget": self.over_budget(),
                "file": str(self.path),
            }
        return out

    def session_cost(self, session_id: str) -> Dict[str, Any]:
        """What one session's calls have cost over its whole life, across
        billing periods. Best-effort like every per-session figure: a call
        the process could not attribute to a session is in none."""
        from . import pricing
        with self._lock:
            row = dict(self.session_lifetime.get(str(session_id))
                       or {"calls": 0, "cost_usd": 0.0, "priced_calls": 0})
        return {"session": str(session_id), "calls": row["calls"], "priced_calls": row["priced_calls"],
                "cost_usd": round(row["cost_usd"], 6),
                "cost_text": pricing.cost_text(row["cost_usd"], row["calls"], row["priced_calls"])}

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
