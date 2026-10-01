"""Run several delegations of any mode at once: ``run_swarm``.

A swarm here is a list of work items, each ``{mode, task, context?, subject?,
label?, data_path?, reads_board?, check?}``, run concurrently on ephemeral
workers (``workers.build_child``) in ``<meta_session>/swarm/<NN>_<slug>/``.
Each item is an ordinary delegation on the ledger (``swarm`` /
``parallel_group`` name the run), so its result is read, threaded and fused
like any other, and its findings are posted on the board when it finishes
(``board.py``, through ``_close_delegation``). An item that opts in with
``reads_board`` reads the board ONCE, when it starts (after admission), and
gets the verified findings on its subject rendered into its task as hints;
the ids it saw are stamped on its ledger entry as ``reads``. An item tagged
``check`` is refused a read by the board itself. No item starts another.

The coordinator is deterministic, with no model call:

- a **capacity plan** before anything starts: an item whose memory estimate
  exceeds what this machine can ever give is not started (its result says
  why); the plan is shown at the autopilot gate with the items;
- **admission** by memory (the fan-out's ``_admit_branch``): items wait for
  headroom instead of overcommitting;
- a **memory guard** while running: when free memory falls below the hard
  floor, the running item expected to hold the most memory (the newest of
  equals) is cancelled and queued to run again, alone, once;
- a wall-clock **budget** per item, as in the fan-out;
- one **question queue**: in autopilot each worker's questions are tagged with
  who asks and about what, served to the person one at a time from a thread
  of their own (``hitl.QuestionServer``, so the coordinator keeps polling
  while a question is on screen), and time out to the gate's default
  (``hitl.question_timeout_s``) — a timeout the gate can tell from an answer.

**Reactions** (stage 3, ``reactions.py``): the meta may declare
``subscriptions`` with the swarm — *when a record of kind K on subject S with
status V is posted, enqueue this item* — and the coordinator applies them,
deterministically, each time an item closes: the fired item is an ordinary
item with its cause and causal ``chain`` stamped on its ledger entry at that
moment; a cycle (the same ``(mode, subject, kind)`` hop twice in one chain),
a subject re-triggered past its cap, a subscription past ``max_fires``, the
item limit, or a record at the end of a long supersede chain is refused and
the refusal recorded on the triggering entry and in the result. A worker's
``suggested_followups`` are posted as ``task_request`` records (provisional,
never read by default) and become items only through a subscription on that
kind: workers ask, the coordinator decides, and no worker starts a worker.

There is no swarm resume: after a Stop, items still queued are left
``running`` on the ledger until the next turn's sweep marks them interrupted.
"""

from __future__ import annotations

import contextlib
import json
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...hitl import (QueueChannel, QuestionServer, WorkerChannel, question_timeout_s,
                     request_human_feedback, set_thread_channel, subject_block, make_subject)
from ...utils.workers import resolve_workers
from . import fanout as fo
from . import reactions
from .workers import MODES, build_child, release_child

SWARM_MAX_ITEMS = 8
SWARM_MAX_WORKERS = resolve_workers(None, "SCILINK_SWARM_MAX_WORKERS", 3)
SWARM_ITEM_TIME_BUDGET_S = 3600.0
#: Below this much free memory the guard cancels the newest running item.
SWARM_MEMORY_FLOOR_BYTES = 7.5e8
#: How long the coordinator waits for a cancelled worker to end before it
#: gives up that item's rerun (a hung call never prints, so the cancel may
#: never land).
SWARM_DRAIN_TIMEOUT_S = 600.0
_POLL_S = 5
_HEARTBEAT_S = 60
#: An item's context is kept on its ledger entry (so a re-run can start from
#: it) when it is small; the ledger is checkpointed, so a large one is not.
_CONTEXT_KEEP_CHARS = 8000

# A planning or simulation item's own working set, beside the imports every
# worker thread already shares with the process.
_MODE_MEM_FLOOR = {"analysis": fo._BRANCH_MEM_FLOOR, "planning": 5e8, "simulation": 5e8}


class _Unattended:
    """Workers of an autonomous swarm: every question gets its default."""

    def ask(self, req) -> str:
        return req.default or ""


def _memory() -> Dict[str, Optional[float]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": float(vm.total), "available": float(vm.available)}
    except Exception:  # noqa: BLE001
        return {"total": None, "available": None}


def _estimate(item: dict) -> float:
    if item["mode"] == "analysis" and item.get("data_path"):
        return fo._branch_mem_estimate({"data_path": item["data_path"],
                                        "pattern": item.get("pattern")})
    return _MODE_MEM_FLOOR[item["mode"]]


#: Stamped by the coordinator when it fires a reaction; never taken from a
#: caller's item (a chain nobody enqueued would defeat the cycle check).
_COORDINATOR_FIELDS = ("caused_by", "chain", "subscription")


def normalize_items(items: Any, *, coordinator: bool = False) -> Tuple[List[dict], List[dict]]:
    """Valid items (with a label and a slug) and the refused ones, each with
    the reason."""
    ok, refused = [], []
    for i, raw in enumerate(items or []):
        if not isinstance(raw, dict):
            refused.append({"item": i + 1, "reason": "not an object"})
            continue
        mode = str(raw.get("mode") or "").strip().lower()
        task = str(raw.get("task") or "").strip()
        label = str(raw.get("label") or "").strip() or f"{mode or 'item'} {i + 1}"
        if mode not in MODES:
            refused.append({"item": i + 1, "label": label,
                            "reason": f"mode must be one of {list(MODES)}"})
            continue
        if not task:
            refused.append({"item": i + 1, "label": label, "reason": "empty task"})
            continue
        ok.append({**{k: v for k, v in raw.items() if coordinator or k not in _COORDINATOR_FIELDS},
                   "mode": mode, "task": task, "label": label,
                   "subject": (str(raw.get("subject")).strip() if raw.get("subject") else None),
                   "reads_board": _read_spec(raw.get("reads_board")),
                   "check": bool(raw.get("check")),
                   "slug": fo._slug(label)})
    if len(ok) > SWARM_MAX_ITEMS:
        for it in ok[SWARM_MAX_ITEMS:]:
            refused.append({"label": it["label"],
                            "reason": f"over the limit of {SWARM_MAX_ITEMS} items per swarm"})
        ok = ok[:SWARM_MAX_ITEMS]
    return ok, refused


def _read_spec(raw: Any) -> Optional[dict]:
    """What an item asked to read: ``True`` / ``{}`` (verified records on the
    item's subject — every subject when the item has none) or
    ``{subject?, kinds?, include_provisional?}``; ``None`` reads nothing."""
    if raw is None or raw is False or raw == "":
        return None
    if not isinstance(raw, dict):
        return {}                     # true, "yes", 1: the item's subject, verified records
    # An empty object is the tool schema's plain opt-in; it is falsy in
    # Python, which is why this does not test ``not raw``.
    spec: dict = {}
    if raw.get("subject"):
        spec["subject"] = str(raw["subject"]).strip()
    kinds = raw.get("kinds") or raw.get("kind")
    if kinds:
        spec["kinds"] = [str(k).strip().lower() for k in
                         (kinds if isinstance(kinds, (list, tuple)) else [kinds])]
    if raw.get("include_provisional"):
        spec["include_provisional"] = True
    return spec


def _read_board(orch, item: dict, entry: dict) -> str:
    """The item's one board read, at its start. Returns the block to append
    to the task (empty when nothing was read); stamps ``reads`` and
    ``board_version`` on the entry, or ``board_read_refused`` for a check."""
    from . import board as board_mod
    spec = item.get("reads_board")
    board = getattr(orch, "board", None)
    if spec is None or board is None:
        return ""
    try:
        # A hazard on the subject reaches every reader, whatever its kinds
        # filter (robustness item 4): the filter narrows findings, never
        # warnings.
        view = board.snapshot(subject=spec.get("subject") or item.get("subject"),
                              kind=spec.get("kinds"),
                              include_provisional=bool(spec.get("include_provisional")),
                              check=bool(item.get("check")), with_hazards=True)
    except board_mod.BoardReadRefused as exc:
        with orch._fanout_lock:
            entry["board_read_refused"] = str(exc)
        print(f"  🙈 '{item['label']}' is a check: its board read was refused.")
        return ""
    except ValueError as exc:
        with orch._fanout_lock:
            entry["board_read_refused"] = str(exc)
        print(f"  ⚠️  '{item['label']}': board read not possible ({exc}).")
        return ""
    total = len(view)
    # What is rendered is what is stamped as read; a hazard is never cut.
    view = view.newest(board_mod.READ_MAX_RECORDS, pin=("hazard",))
    with orch._fanout_lock:
        # A reaction already "read" the finding that caused it (its task
        # quotes it): the cause stays in reads beside what the board showed.
        entry["reads"] = sorted(set(entry.get("reads") or []) | set(view.ids))
        entry["board_version"] = view.version
        if view.include_provisional or any(r["status"] == "provisional" for r in view.records):
            # A provisional record was shown — asked for, or a standing
            # hazard delivered whatever the filter: marked either way.
            entry["reads_provisional"] = True
    print(f"  📋 '{item['label']}' read {len(view)} board finding(s)"
          + (f" (the newest of {total})" if total > len(view) else "")
          + (" (provisional included)" if view.include_provisional else "") + ".")
    return "\n".join(board_mod.render(view))


def capacity_plan(items: List[dict], memory: Optional[Dict[str, Optional[float]]] = None) -> dict:
    """Which items this machine can run, and whether together or in turn.

    An item that needs more than the machine has in total, less a margin for
    the system, can never run here and is not started. The rest are admitted
    by free memory as they go, so items that do not fit together wait for
    each other instead of overcommitting.
    """
    mem = memory or _memory()
    total, avail = mem.get("total"), mem.get("available")
    margin = fo._BRANCH_MEM_MARGIN
    ceiling = (total - margin) if total else None
    run, refused = [], []
    for it in items:
        est = _estimate(it)
        it["_mem_est"] = est
        if ceiling is not None and est > ceiling:
            refused.append({"label": it["label"], "reason": (
                f"needs about {est / 1e9:.1f} GB; this machine has {total / 1e9:.1f} GB in all")})
        else:
            run.append(it)
    need = sum(it["_mem_est"] for it in run)
    together = avail is None or need + margin <= avail
    return {"run": run, "refused": refused, "estimated_bytes": need,
            "available_bytes": avail, "total_bytes": total, "together": together,
            "workers": min(len(run), SWARM_MAX_WORKERS) if run else 0}


def _subscription_line(sub: dict) -> str:
    on, enq = sub["on"], sub["enqueue"]
    status = {"verified": "a verified", "provisional": "a provisional", "any": "any"}[on["status"]]
    return (f"- on {status} **{on['kind']}**"
            + (f" on '{on['subject']}'" if on.get("subject") else " on any subject")
            + f" → {enq['mode']} '{enq['label']}'"
            + (f" (up to {sub['max_fires']} times)" if sub.get("max_fires", 1) != 1 else " (once)"))


def swarm_plan_subject(plan: dict, attended: bool, subscriptions: Optional[List[dict]] = None,
                       budget: Optional[dict] = None) -> dict:
    """What the swarm gate shows, as subject blocks (scilink.hitl): the
    items, and — because a reaction starts work nobody listed — every
    subscription and the most items the swarm may run in all."""
    rows = [f"- **{it['label']}** ({it['mode']}{', ' + it['subject'] if it.get('subject') else ''}; "
            f"~{it['_mem_est'] / 1e9:.1f} GB) — {it['task'][:160]}"
            for it in plan["run"]]
    blocks = [subject_block("text", label=f"🐝 Items ({len(plan['run'])})", markdown="\n".join(rows))]
    fields = [{"label": "Runs at once", "value": str(plan["workers"])},
              {"label": "Memory", "value": _memory_line(plan)}]
    if subscriptions:
        bounds = budget or {}
        blocks.append(subject_block("text", label=f"🔔 Reactions ({len(subscriptions)})",
                                    markdown="\n".join(_subscription_line(s) for s in subscriptions)))
        fields.append({"label": "Items in all", "value": (
            f"up to {bounds.get('max_items', SWARM_MAX_ITEMS)} (at most {bounds.get('max_reactions', SWARM_MAX_ITEMS)} "
            f"fired; a subject re-triggered at most {bounds.get('max_triggers_per_subject', reactions.MAX_TRIGGERS_PER_SUBJECT)} times)")})
    blocks.append(subject_block("fields", items=fields))
    if plan["refused"]:
        blocks.append(subject_block("notice", title="⛔ Not started", tone="warn",
                                    lines=[f"{r['label']}: {r['reason']}" for r in plan["refused"]]))
    blocks.append(subject_block("text", label="Questions", markdown=(
        "Items pause for approvals; questions come one at a time, labelled by item."
        if attended else "Items run autonomously.")))
    return make_subject("Launch this swarm?", blocks)


def _memory_line(plan: dict) -> str:
    need = plan["estimated_bytes"] / 1e9
    if plan["available_bytes"] is None:
        return f"about {need:.1f} GB estimated"
    free = plan["available_bytes"] / 1e9
    return (f"about {need:.1f} GB estimated, {free:.1f} GB free: "
            + ("they fit together" if plan["together"] else "some will wait for others to finish"))


def _confirm(orch, plan: dict, attended: bool, subscriptions: Optional[List[dict]] = None,
             budget: Optional[dict] = None) -> bool:
    print("\n" + "=" * 78)
    print(f"🐝 SWARM — {len(plan['run'])} item(s), {plan['workers']} at a time")
    for it in plan["run"]:
        print(f"    • {it['label']}  [{it['mode']}]  ~{it['_mem_est'] / 1e9:.1f} GB")
    print(f"  Memory: {_memory_line(plan)}")
    if subscriptions:
        print(f"  🔔 Reactions ({len(subscriptions)}), up to {(budget or {}).get('max_items', SWARM_MAX_ITEMS)} items in all:")
        for sub in subscriptions:
            print(f"    {_subscription_line(sub)[2:]}")
    for r in plan["refused"]:
        print(f"  ⛔ not started: {r['label']} — {r['reason']}")
    print("=" * 78)
    try:
        ans = request_human_feedback(
            "\n🤔 Launch this swarm? [y/N]: ", kind="confirm", options=["y", "n"], default="n",
            origin={"stage": "swarm_confirm"},
            subject=swarm_plan_subject(plan, attended, subscriptions, budget),
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def _autonomy_enum(mode: str):
    if mode == "analysis":
        from ..exp_agents.analysis_orchestrator import AnalysisMode
        return AnalysisMode
    if mode == "planning":
        from ..planning_agents.planning_orchestrator import AutonomyLevel
        return AutonomyLevel
    from ..sim_agents.simulation_orchestrator import SimulationMode
    return SimulationMode


def _error_result(message: str, status: str = "error") -> dict:
    return {"status": status, "error": message, "summary": "", "key_findings": [],
            "files_produced": [], "suggested_followups": [], "warnings": []}


def _run_item(orch, item: dict, entry: dict, channel, autonomy: str, stop_event) -> None:
    """One item on its own worker, into its own ledger entry. Never raises
    except for a user's Stop, which must reach the coordinator."""
    from ... import tracing
    from ...session_events import append_event, set_thread_event_log
    index = entry["index"]
    base_dir = Path(orch.base_dir) / "swarm" / f"{index:02d}_{item['slug']}"
    mem_key = f"swarm:{index}"
    fo._admit_branch(mem_key, item["_mem_est"], item["label"])
    entry["_started_at"] = time.monotonic()
    entry.pop("_human_wait_s", None)          # a restored entry may carry stale waits
    entry.pop("_waiting_since", None)
    entry["_branch_tid"] = threading.get_ident()
    fo._register_branch_stop(stop_event)
    set_thread_channel(channel)
    set_thread_event_log(Path(orch.base_dir) / "events.jsonl")
    tag = tracing.attributed(worker=f"swarm:{index:02d}_{item['slug']}")
    tag.__enter__()
    result = _error_result("item aborted before completion")
    child = None
    from ...hitl import unattended_questions
    unattended_before = unattended_questions()
    try:
        try:
            # The read comes after admission, so a later-admitted item sees
            # what earlier items of the same swarm have already posted. The
            # ledger keeps the task as actually sent, block included.
            task = item["task"] + _read_board(orch, item, entry)
            if task != item["task"]:
                with orch._fanout_lock:
                    entry["task"] = task
            child = build_child(orch, item["mode"], base_dir, label=f"Swarm: {item['label']}")
            result = child.run_task(task, context=item.get("context"),
                                    autonomy=_autonomy_enum(item["mode"])[autonomy])
        except Exception as exc:  # noqa: BLE001
            fo.logger.exception(f"swarm item {index} failed: {exc}")
            result = _error_result(str(exc))
        except BaseException as exc:
            from ...ui.output_capture import AgentStoppedError
            if not (isinstance(exc, AgentStoppedError) and stop_event.is_set()):
                raise
            result = _error_result(entry.get("_cancel_reason") or "cancelled", status="cancelled")
    finally:
        n_unattended = unattended_questions() - unattended_before
        if n_unattended and isinstance(result, dict):
            result.setdefault("warnings", []).append(
                f"{n_unattended} question(s) got no answer in time and took their defaults "
                "(unattended; nothing here counts as a human decision)")
        try:
            if child is not None:
                release_child(child)
        finally:
            # Released last and unconditionally: on a Stop, anything above may
            # raise, and a held memory reservation would outlive the item.
            fo._unregister_branch_stop()
            tag.__exit__(None, None, None)
            try:
                append_event("swarm_item", {"label": item["label"], "mode": item["mode"],
                                            "session_dir": str(base_dir)},
                             json.dumps({"status": result.get("status")}, default=str),
                             branch=item["label"])
            finally:
                set_thread_event_log(None)
                set_thread_channel(None)
                fo._release_branch(mem_key)
    if entry.get("timed_out") or entry.get("_cancelled"):
        entry["late_result"] = {"status": result.get("status")}
        return
    orch._close_delegation(entry, result)


def _guard_memory(orch, running: Dict[Any, dict], fut_item: Dict[Any, dict], fut_stop,
                  requeue: List[dict], floor: float) -> Optional[Any]:
    """Cancel the running item expected to hold the most memory (the newest
    of equals) when free memory is below ``floor``: it frees the most. It is
    queued to run again alone, once. Returns the cancelled item's future,
    which the caller watches until that worker has let go of its memory."""
    avail = _memory().get("available")
    if avail is None or avail >= floor:
        return None
    live = [(f, e) for f, e in running.items()
            if e.get("status") == "running" and e.get("_started_at") and not e.get("_cancelled")]
    if len(live) < 2:
        # The guard is against overcommitting by several items at once. One
        # item alone is an ordinary delegation's risk, and cancelling it to
        # run it alone again would change nothing.
        return None
    fut, entry = max(live, key=lambda fe: (fut_item[fe[0]]["_mem_est"], fe[1]["_started_at"]))
    item = fut_item[fut]
    entry["_cancelled"] = True
    reason = (f"memory_pressure: {avail / 1e9:.2f} GB free, below the "
              f"{floor / 1e9:.2f} GB floor")
    entry["_cancel_reason"] = reason
    print(f"  🧯 free memory is low ({avail / 1e9:.2f} GB) — cancelling '{item['label']}' "
          + ("and running it again alone afterwards" if not item.get("_retried") else "(already run again once)"))
    fut_stop[fut].set()
    tid = entry.get("_branch_tid")
    if tid:
        try:
            from ...executors import kill_subprocesses_for_thread
            kill_subprocesses_for_thread(tid)
        except Exception:  # noqa: BLE001
            pass
    orch._close_delegation(entry, _error_result(reason, status="cancelled"))
    if not item.get("_retried"):
        requeue.append({**item, "_retried": True})
    return fut


def swarm_budget(raw: Any) -> dict:
    """The swarm's bounds: ``max_items`` (initial and fired together, at most
    ``SWARM_MAX_ITEMS``), ``max_reactions`` (fired items), and
    ``max_triggers_per_subject``. The budget is the termination proof."""
    raw = raw if isinstance(raw, dict) else {}

    def bounded(key, default, cap=None):
        try:
            v = int(raw.get(key, default))
        except (TypeError, ValueError):
            v = default
        v = max(0, v)
        return min(v, cap) if cap is not None else v
    max_items = max(2, bounded("max_items", SWARM_MAX_ITEMS, SWARM_MAX_ITEMS))
    return {"max_items": max_items,
            "max_reactions": bounded("max_reactions", max_items, max_items),
            "max_triggers_per_subject": bounded("max_triggers_per_subject",
                                                reactions.MAX_TRIGGERS_PER_SUBJECT, max_items)}


def run_swarm(orch, items: Any, item_time_budget_s: Optional[float] = None,
              subscriptions: Any = None, budget: Any = None) -> str:
    """Plan → confirm → run the items concurrently, reacting to what they
    post under the declared ``subscriptions`` and ``budget``. Returns JSON."""
    valid, refused = normalize_items(items)
    subs, subs_refused = reactions.normalize_subscriptions(subscriptions)
    bounds = swarm_budget(budget)
    if not valid:
        return json.dumps({"status": "error", "message": "No runnable items.",
                           "not_started": refused})
    if len(valid) == 1:
        return json.dumps({"status": "error", "not_started": refused, "message": (
            "A swarm needs at least two items; one item is an ordinary delegation "
            "(delegate_to_analysis / delegate_to_planning / delegate_to_simulation).")})
    if len(valid) > bounds["max_items"]:
        refused += [{"label": it["label"], "reason": f"over the swarm budget of {bounds['max_items']} items"}
                    for it in valid[bounds["max_items"]:]]
        valid = valid[:bounds["max_items"]]
    plan = capacity_plan(valid)
    refused += plan["refused"]
    if not plan["run"]:
        return json.dumps({"status": "error", "message": "No item fits this machine.",
                           "not_started": refused})
    attended = bool(getattr(orch, "_enable_human_feedback", False))
    if attended and not _confirm(orch, plan, attended, subs, bounds):
        return json.dumps({"status": "declined", "message": "The user declined the swarm.",
                           "not_started": refused})

    budget = (float(item_time_budget_s) if item_time_budget_s is not None
              else SWARM_ITEM_TIME_BUDGET_S)                 # <= 0 disables it
    swarm_id = f"swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    autonomy = orch.meta_mode.name
    queue = QueueChannel(timeout_s=question_timeout_s()) if attended else None
    fo._ensure_stop_guard_installed()
    t0 = time.monotonic()
    entries: List[dict] = []

    def launch(pool, item) -> Any:
        entry = orch._open_delegation(item["mode"], item["task"], item.get("context"), None,
                                      item["label"])
        with orch._fanout_lock:          # new keys on a live entry: see _ledger_snapshot
            entry["swarm"] = swarm_id
            entry["parallel_group"] = swarm_id
            if item.get("subject"):
                entry["subject"] = item["subject"]
            if item["mode"] == "analysis" and item.get("data_path"):
                # As _delegate stamps it: a later fuse_delegations re-runs
                # its complementarity gate from the entries' data paths.
                entry["data_path"] = str(item["data_path"])
            entry["_budget_s"] = budget
            # The item's own inputs, so a re-run (retract_finding's
            # rerun_items) starts from what this one had.
            if item.get("reads_board") is not None:
                entry["reads_board"] = dict(item["reads_board"])
            if item.get("check"):
                entry["check"] = True
            if isinstance(item.get("context"), dict) and item["context"]:
                if len(json.dumps(item["context"], default=str)) <= _CONTEXT_KEEP_CHARS:
                    entry["context"] = item["context"]
                else:
                    entry["context_omitted"] = "too large to keep on the ledger"
            if item.get("rests_on"):
                # What the caller says this item's work rests on (a re-run's
                # original cause): reads it did not make on the board, which
                # can only add couplings, never remove one.
                entry["reads"] = sorted(set(entry.get("reads") or []) | {str(f) for f in item["rests_on"] if f})
            if item.get("caused_by"):
                # A reaction: its cause and chain, stamped as it is enqueued.
                # The cause is a READ: the task quotes the finding, so what
                # rests on the finding rests on this item's records too
                # (taint, independence), whether or not it reads the board.
                entry["caused_by"] = list(item["caused_by"])
                entry["chain"] = list(item.get("chain") or [])
                entry["subscription"] = item.get("subscription")
                entry["reads"] = sorted(set(entry.get("reads") or []) | set(item["caused_by"]))
                entry["board_version"] = len(orch.board) if getattr(orch, "board", None) is not None else None
        entries.append(entry)
        channel = (WorkerChannel(queue, item["label"], subject=item.get("subject"), kind="worker",
                                 on_wait=fo.note_human_wait(entry))
                   if queue is not None else _Unattended())
        stop_ev = threading.Event()
        fut = pool.submit(fo._attributed_branch(_run_item), orch, item, entry, channel,
                          autonomy, stop_ev)
        return fut, entry, stop_ev

    print(f"  🐝 {swarm_id}: {len(plan['run'])} item(s), up to {plan['workers']} at a time"
          + (f", {len(subs)} subscription(s)" if subs else ""))
    orch._auto_checkpoint(verbose=False)
    requeue: List[dict] = []
    fired: List[dict] = []
    refused_reactions: List[dict] = []
    fired_by_sub: Dict[int, int] = {}
    fired_pairs: set = set()
    triggers_by_subject: Dict[Optional[str], int] = {}
    launched = {"n": len(plan["run"])}
    ceiling = (plan["total_bytes"] - fo._BRANCH_MEM_MARGIN) if plan.get("total_bytes") else None

    def react(pool, entry: dict, item: dict) -> List[Any]:
        """Apply every subscription to what this item just posted; launch
        what fires. Returns the new futures."""
        board = getattr(orch, "board", None)
        if not subs or board is None or entry.get("status") != "success":
            return []
        new_futs = []
        for fid in list(entry.get("posted") or []):
            try:
                rec = board.get(fid)
            except KeyError:
                continue
            for sub in subs:
                if len(fired) >= bounds["max_reactions"]:
                    why = f"max_reactions ({bounds['max_reactions']}) reached"
                    new_item = None
                    if not reactions.matches(sub, rec):
                        continue
                else:
                    new_item, why = reactions.decide(
                        sub, rec, entry, board=board, fired_by_sub=fired_by_sub,
                        fired_pairs=fired_pairs, triggers_by_subject=triggers_by_subject,
                        items_so_far=launched["n"], max_items=bounds["max_items"],
                        max_per_subject=bounds["max_triggers_per_subject"])
                if new_item is None:
                    if why is not None and "already fired" not in why:
                        note = {"subscription": sub["index"], "finding_id": fid, "kind": rec.get("kind"),
                                "subject": rec.get("subject"), "from_index": entry["index"], "reason": why,
                                "chain": list(entry.get("chain") or [])}
                        refused_reactions.append(note)
                        with orch._fanout_lock:
                            entry.setdefault("refused_reactions", []).append(
                                {k: v for k, v in note.items() if k != "chain"})
                        print(f"  ⛔ reaction refused ({sub['enqueue']['label']} on {fid}): {why}")
                    continue
                ok, bad = normalize_items([new_item], coordinator=True)
                if not ok:
                    refused_reactions.append({"subscription": sub["index"], "finding_id": fid,
                                              "from_index": entry["index"], "reason": bad[0]["reason"]})
                    continue
                new_item = ok[0]
                new_item["_mem_est"] = _estimate(new_item)
                if ceiling is not None and new_item["_mem_est"] > ceiling:
                    refused_reactions.append({"subscription": sub["index"], "finding_id": fid,
                                              "from_index": entry["index"], "reason": (
                                                  f"needs about {new_item['_mem_est'] / 1e9:.1f} GB; this "
                                                  f"machine has {plan['total_bytes'] / 1e9:.1f} GB in all")})
                    continue
                fired_pairs.add((sub["index"], fid))
                fired_by_sub[sub["index"]] = fired_by_sub.get(sub["index"], 0) + 1
                subj = reactions._norm_subject(new_item.get("subject"))
                triggers_by_subject[subj] = triggers_by_subject.get(subj, 0) + 1
                launched["n"] += 1
                fut, new_entry, stop_ev = launch(pool, new_item)
                fut_entry[fut], fut_stop[fut], fut_label[fut] = new_entry, stop_ev, new_item["label"]
                fut_item[fut] = new_item
                fired.append({"delegation_index": new_entry["index"], "label": new_item["label"],
                              "mode": new_item["mode"], "subject": new_item.get("subject"),
                              "subscription": sub["index"], "caused_by": [fid],
                              "from_index": entry["index"], "chain": list(new_item["chain"])})
                print(f"  🔔 '{entry.get('label')}' posted a {rec.get('kind')} ({fid}) → "
                      f"starting '{new_item['label']}' [{new_item['mode']}]")
                new_futs.append(fut)
        return new_futs

    seen_errors: set = set()
    channel_warnings: List[str] = []
    pool = ThreadPoolExecutor(max_workers=max(1, plan["workers"]))
    server = QuestionServer(queue) if queue is not None else contextlib.nullcontext()
    fut_entry, fut_stop, fut_label, fut_item = {}, {}, {}, {}
    try:
        server.__enter__()
        for item in plan["run"]:
            fut, entry, stop_ev = launch(pool, item)
            fut_entry[fut], fut_stop[fut], fut_label[fut] = entry, stop_ev, item["label"]
            fut_item[fut] = item
        pending, since_tick = set(fut_entry), 0.0
        draining: set = set()     # cancelled for memory, not yet ended
        drain_since: Optional[float] = None
        while pending or requeue:
            t_poll = time.monotonic()
            if pending:
                done, pending = wait(pending, timeout=_POLL_S)
            else:                 # only a rerun is left, waiting for the cancelled worker to end
                done = set()
                wait(draining, timeout=_POLL_S)
                # This print is also where a user's Stop lands on this thread.
                print(f"  ⏳ waiting for the cancelled worker to end before the rerun "
                      f"({int(time.monotonic() - (drain_since or t_poll))} s) ...")
            since_tick += time.monotonic() - t_poll
            if server is not None and getattr(server, "error", None) is not None:
                from ...ui.output_capture import AgentStoppedError
                if isinstance(server.error, AgentStoppedError):
                    raise server.error          # the person's Stop reaches the swarm
                if server.error not in seen_errors:
                    seen_errors.add(server.error)
                    channel_warnings.append(
                        f"the person's channel raised {type(server.error).__name__}: "
                        "remaining questions took their defaults, unattended")
                    print(f"  ⚠️  {channel_warnings[-1]}.")
            for f in done:
                f.result()
                print(f"  ✅ swarm item finished: {fut_label[f]} ({fut_entry[f].get('status')})")
                since_tick = 0.0
                for nf in react(pool, fut_entry[f], fut_item[f]):
                    pending.add(nf)
            # One cancellation at a time: memory is read again only once the
            # cancelled worker has actually ended and let go of what it held.
            draining = {f for f in draining if not f.done()}
            if draining and drain_since is None:
                drain_since = time.monotonic()
            elif not draining:
                drain_since = None
            if draining and time.monotonic() - drain_since > SWARM_DRAIN_TIMEOUT_S:
                # A hung worker (a call that never returns, compute that never
                # prints) never lets go: give the rerun up rather than wait forever.
                for item in requeue:
                    refused.append({"label": item["label"], "reason": (
                        f"rerun abandoned: the cancelled worker had not ended after "
                        f"{int(SWARM_DRAIN_TIMEOUT_S)} s")})
                    print(f"  ⚠️  giving up the rerun of '{item['label']}': its cancelled "
                          "worker has not ended.")
                for f in draining:
                    # Its reservation would otherwise hold later swarms and
                    # fan-outs for as long as the hung thread lives.
                    fo._release_branch(f"swarm:{fut_entry[f]['index']}")
                requeue.clear()
                draining.clear()
                drain_since = None
            if not draining:
                cancelled = _guard_memory(orch, {f: fut_entry[f] for f in pending}, fut_item,
                                          fut_stop, requeue, SWARM_MEMORY_FLOOR_BYTES)
                if cancelled is not None:
                    draining.add(cancelled)
            pending = {f for f in pending if not fut_entry[f].get("_cancelled")}
            if budget > 0:
                fo._cancel_overdue_branches(orch, pending, fut_entry, fut_stop, fut_label,
                                            budget, noun="swarm item")
            if not pending and requeue and not draining:
                # A cancelled item runs again once the others are done AND the
                # cancelled worker has ended: alone.
                item = requeue.pop(0)
                print(f"  🔁 running '{item['label']}' again, alone")
                fut, entry, stop_ev = launch(pool, item)
                fut_entry[fut], fut_stop[fut], fut_label[fut] = entry, stop_ev, item["label"]
                fut_item[fut] = item
                pending = {fut}
            if pending and since_tick >= _HEARTBEAT_S:
                since_tick = 0.0
                print(f"  ⏳ {len(pending)} swarm item(s) still running ...")
    except BaseException:
        # A Stop (or any failure of the coordinator): the items still running
        # are cancelled the way a budget cancels them, so they wind down and
        # let go of their memory instead of running on unattended.
        for f, ev in list(fut_stop.items()):
            if not f.done():
                fut_entry[f]["_cancelled"] = True    # a late end must not overwrite the verdict
                ev.set()
                tid = fut_entry[f].get("_branch_tid")
                if tid:
                    try:
                        from ...executors import kill_subprocesses_for_thread
                        kill_subprocesses_for_thread(tid)
                    except Exception:  # noqa: BLE001
                        pass
        raise
    finally:
        server.__exit__(None, None, None)
        pool.shutdown(wait=False, cancel_futures=True)
        orch._auto_checkpoint(verbose=False)

    results = [{"delegation_index": e["index"], "label": e.get("label"), "mode": e.get("mode"),
                "subject": e.get("subject"), "status": e.get("status"),
                "summary": (e.get("summary") or "")[:600],
                "key_findings": (e.get("key_findings") or [])[:6],
                "files_produced": len(e.get("files_produced") or []),
                "warnings": list(e.get("warnings") or []),
                "reads": list(e.get("reads") or []),
                "board_read_refused": e.get("board_read_refused"),
                "posted": list(e.get("posted") or []),
                **({"caused_by": list(e["caused_by"]), "chain": list(e.get("chain") or []),
                    "subscription": e.get("subscription")} if e.get("caused_by") else {}),
                "error": e.get("error")} for e in entries]
    ok = [r for r in results if r["status"] == "success"]
    return json.dumps({
        "status": "success" if ok else "error",
        "swarm_id": swarm_id,
        "seconds": round(time.monotonic() - t0),
        "results": results,
        "not_started": refused,
        "subscriptions": {"accepted": len(subs), "refused": subs_refused},
        "fired": fired,
        "refused_reactions": _collapse_refusals(refused_reactions),
        "task_requests": _task_requests(orch, entries),
        "warnings": channel_warnings,
        "board_version": len(orch.board) if getattr(orch, "board", None) is not None else None,
        "message": (f"{len(ok)} of {len(results)} item(s) succeeded"
                    + (f", {len(fired)} of them reactions" if fired else "")
                    + (f"; {len(refused_reactions)} reaction(s) refused (see refused_reactions)"
                       if refused_reactions else "")
                    + ". Each is a delegation on the "
                    "ledger: read one with get_delegation_history, thread its findings into a next "
                    "delegation's context, or fuse analysis items with fuse_delegations. Their "
                    "verified findings are on the board (get_board); a next swarm's items read "
                    "them with reads_board."),
    }, default=str)


def _task_requests(orch, entries: List[dict]) -> List[dict]:
    """What the workers asked for (``task_request`` records they posted) and
    whether a subscription made an item of it: asked and not done is listed,
    never silently dropped, and never done on a worker's say-so."""
    board = getattr(orch, "board", None)
    if board is None:
        return []
    became: Dict[str, int] = {}
    for e in entries:
        for fid in e.get("caused_by") or []:
            became[fid] = e["index"]
    out = []
    for e in entries:
        for fid in e.get("posted") or []:
            try:
                rec = board.get(fid)
            except KeyError:
                continue
            if rec.get("kind") == "task_request":
                out.append({"finding_id": fid, "from": e.get("label"), "from_index": e["index"],
                            "text": (rec.get("payload") or {}).get("text"),
                            "item": became.get(fid)})
    return out


def _collapse_refusals(refusals: List[dict]) -> List[dict]:
    """One line per (subscription, reason) with the first refusal's detail and
    the other findings it also refused, instead of a full chain per refusal
    (an adversarial rule set produces dozens)."""
    out: List[dict] = []
    seen: Dict[Tuple[Any, str], dict] = {}
    for r in refusals:
        key = (r.get("subscription"), str(r.get("reason")))
        if key in seen:
            seen[key].setdefault("also", []).append(r.get("finding_id"))
            seen[key]["count"] = seen[key].get("count", 1) + 1
            continue
        seen[key] = dict(r)
        out.append(seen[key])
    return out
