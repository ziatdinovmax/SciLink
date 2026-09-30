"""Run several delegations of any mode at once: ``run_swarm``.

A swarm here is a list of work items, each ``{mode, task, context?, subject?,
label?, data_path?}``, run concurrently on ephemeral workers
(``workers.build_child``) in ``<meta_session>/swarm/<NN>_<slug>/``. Each item
is an ordinary delegation on the ledger (``swarm`` / ``parallel_group`` name
the run), so its result is read, threaded and fused like any other. Items do
not see each other's results: there is no shared board yet (stage 2 of
docs/proposals/agent-swarms.md), and no item starts another.

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


def normalize_items(items: Any) -> Tuple[List[dict], List[dict]]:
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
        ok.append({**raw, "mode": mode, "task": task, "label": label,
                   "subject": (str(raw.get("subject")).strip() if raw.get("subject") else None),
                   "slug": fo._slug(label)})
    if len(ok) > SWARM_MAX_ITEMS:
        for it in ok[SWARM_MAX_ITEMS:]:
            refused.append({"label": it["label"],
                            "reason": f"over the limit of {SWARM_MAX_ITEMS} items per swarm"})
        ok = ok[:SWARM_MAX_ITEMS]
    return ok, refused


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


def swarm_plan_subject(plan: dict, attended: bool) -> dict:
    """What the swarm gate shows, as subject blocks (scilink.hitl)."""
    rows = [f"- **{it['label']}** ({it['mode']}{', ' + it['subject'] if it.get('subject') else ''}; "
            f"~{it['_mem_est'] / 1e9:.1f} GB) — {it['task'][:160]}"
            for it in plan["run"]]
    blocks = [subject_block("text", label=f"🐝 Items ({len(plan['run'])})", markdown="\n".join(rows))]
    fields = [{"label": "Runs at once", "value": str(plan["workers"])},
              {"label": "Memory", "value": _memory_line(plan)}]
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


def _confirm(orch, plan: dict, attended: bool) -> bool:
    print("\n" + "=" * 78)
    print(f"🐝 SWARM — {len(plan['run'])} item(s), {plan['workers']} at a time")
    for it in plan["run"]:
        print(f"    • {it['label']}  [{it['mode']}]  ~{it['_mem_est'] / 1e9:.1f} GB")
    print(f"  Memory: {_memory_line(plan)}")
    for r in plan["refused"]:
        print(f"  ⛔ not started: {r['label']} — {r['reason']}")
    print("=" * 78)
    try:
        ans = request_human_feedback(
            "\n🤔 Launch this swarm? [y/N]: ", kind="confirm", options=["y", "n"], default="n",
            origin={"stage": "swarm_confirm"}, subject=swarm_plan_subject(plan, attended),
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
            child = build_child(orch, item["mode"], base_dir, label=f"Swarm: {item['label']}")
            result = child.run_task(item["task"], context=item.get("context"),
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


def run_swarm(orch, items: Any, item_time_budget_s: Optional[float] = None) -> str:
    """Plan → confirm → run the items concurrently. Returns JSON."""
    valid, refused = normalize_items(items)
    if not valid:
        return json.dumps({"status": "error", "message": "No runnable items.",
                           "not_started": refused})
    if len(valid) == 1:
        return json.dumps({"status": "error", "not_started": refused, "message": (
            "A swarm needs at least two items; one item is an ordinary delegation "
            "(delegate_to_analysis / delegate_to_planning / delegate_to_simulation).")})
    plan = capacity_plan(valid)
    refused += plan["refused"]
    if not plan["run"]:
        return json.dumps({"status": "error", "message": "No item fits this machine.",
                           "not_started": refused})
    attended = bool(getattr(orch, "_enable_human_feedback", False))
    if attended and not _confirm(orch, plan, attended):
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
            entry["_budget_s"] = budget
        entries.append(entry)
        channel = (WorkerChannel(queue, item["label"], subject=item.get("subject"), kind="worker",
                                 on_wait=fo.note_human_wait(entry))
                   if queue is not None else _Unattended())
        stop_ev = threading.Event()
        fut = pool.submit(fo._attributed_branch(_run_item), orch, item, entry, channel,
                          autonomy, stop_ev)
        return fut, entry, stop_ev

    print(f"  🐝 {swarm_id}: {len(plan['run'])} item(s), up to {plan['workers']} at a time")
    orch._auto_checkpoint(verbose=False)
    requeue: List[dict] = []
    seen_errors: set = set()
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
                    print(f"  ⚠️  the person's channel raised {type(server.error).__name__}: "
                          "remaining questions take their defaults, unattended.")
            for f in done:
                f.result()
                print(f"  ✅ swarm item finished: {fut_label[f]} ({fut_entry[f].get('status')})")
                since_tick = 0.0
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
                "error": e.get("error")} for e in entries]
    ok = [r for r in results if r["status"] == "success"]
    return json.dumps({
        "status": "success" if ok else "error",
        "swarm_id": swarm_id,
        "seconds": round(time.monotonic() - t0),
        "results": results,
        "not_started": refused,
        "message": (f"{len(ok)} of {len(results)} item(s) succeeded. Each is a delegation on the "
                    "ledger: read one with get_delegation_history, thread its findings into a next "
                    "delegation's context, or fuse analysis items with fuse_delegations."),
    }, default=str)
