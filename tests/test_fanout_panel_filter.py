"""UI verbose-panel trimming for best-of-N fan-out workers.

`keep_in_fanout_panel` trims a CONCURRENT candidate worker's per-attempt
narration to milestones (executing / verification / outcome) plus any
warning/error. A lone probe attempt and non-worker threads keep full detail.
The CLI/log files are unaffected (this is consulted only by the UI handler).
"""
import logging
import threading

from scilink.utils import log_context as lc

# Messages as they reach the UI handler — i.e. after the record factory has
# prepended the `[cand_NN]` prefix. (message, levelno).
_MILESTONES = {
    "executing": ("[cand_01]    Executing Python script...", logging.INFO),
    "verification": ("[cand_01]    Verification 2/7 (annealing level 1)...",
                     logging.INFO),
    "approved": ("[cand_01]    Analysis approved (score = 0.83)", logging.INFO),
    "quality": ("[cand_01] Quality score = 0.83 (meets threshold 0.7)",
                logging.INFO),
}
_DETAIL = {
    "deviations": ("[cand_01]    Justified plan deviations: min_area filter "
                   "and gradient weight ...", logging.INFO),
    "subscores": ("[cand_01]    Sub-scores: completeness=0.75, correctness="
                  "0.85, relevance=0.95", logging.INFO),
    "issues": ("[cand_01]    Found 3 issue(s)", logging.INFO),
    "reanalyzing": ("[cand_01]    Re-analyzing with verification feedback...",
                    logging.INFO),
    "pipeline": ("[cand_01]    Attempt 1: Resolve pixel size via ...",
                 logging.INFO),
    # WARNING-level per-attempt churn — must trim for concise workers too
    # (this is the line the user flagged). Candidate OUTCOMES are logged on
    # the parent thread, so trimming worker WARNINGs hides nothing important.
    "conformance": ("[cand_01]    Plan conformance issue: threshold_rel is "
                    "hardcoded to 0.1 and never tuned ...", logging.WARNING),
    "retry_failed": ("[cand_01]    Attempt 1 failed: missing outputs",
                     logging.WARNING),
}
# A hard candidate crash (ERROR) still surfaces even when trimmed.
_ERROR = ("[cand_01]    Attempt 1 error: boom", logging.ERROR)


def _classify(prefix: bool) -> dict:
    """Register the worker thread with the given prefix flag and classify."""
    out = {}

    def worker():
        lc.register_worker(parent_thread_id=987654, tag="cand_01",
                           prefix=prefix)
        tid = threading.get_ident()
        try:
            for label, (msg, lvl) in {**_MILESTONES, **_DETAIL}.items():
                out[label] = lc.keep_in_fanout_panel(tid, msg, lvl)
            out["error"] = lc.keep_in_fanout_panel(tid, *_ERROR)
        finally:
            lc.unregister_worker()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return out


def test_concurrent_fanout_worker_trims_detail_keeps_milestones():
    res = _classify(prefix=True)
    assert all(res[k] for k in _MILESTONES), "milestones must survive"
    assert not any(res[k] for k in _DETAIL), "per-attempt detail must be dropped"
    assert res["error"], "errors always pass"


def test_lone_probe_worker_keeps_full_detail():
    # prefix=False == displays like a single run (escalation probe attempt 0):
    # only the concurrent [cand_NN] workers are trimmed, so nothing here is.
    res = _classify(prefix=False)
    assert all(res.values())


def test_non_worker_thread_keeps_full_detail():
    # The chat thread itself (orchestration milestones, never registered).
    tid = threading.get_ident()
    assert lc.is_concise_fanout_worker(tid) is False
    for msg, lvl in {**_MILESTONES, **_DETAIL}.values():
        assert lc.keep_in_fanout_panel(tid, msg, lvl) is True


def test_ui_handler_path_trims_concurrent_fanout():
    # Replicate the app.py handler wiring end-to-end: record-factory prefix +
    # effective_thread routing (worker emits from another thread, mapped back
    # to the chat thread) + the composed filter. Validates the real path, not
    # just the classifier.
    import io

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    chat_thread = threading.get_ident()
    handler.addFilter(
        lambda r: (
            lc.effective_thread(r.thread) == chat_thread
            and lc.keep_in_fanout_panel(r.thread, r.getMessage(), r.levelno)
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.INFO)
    logger = logging.getLogger("test.fanout.panel")
    try:
        def worker():
            lc.register_worker(chat_thread, "cand_01", prefix=True)
            try:
                logger.info("   Executing Python script...")
                logger.info("   Sub-scores: completeness=0.75, correctness=0.85")
                logger.info("   Verification 2/7 (annealing level 1)...")
                logger.info("   Justified plan deviations: min_area filter ...")
                logger.warning("   Plan conformance issue: threshold hardcoded")
                logger.info("   Analysis approved (score = 0.83)")
            finally:
                lc.unregister_worker()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    out = buf.getvalue()
    assert "[cand_01]" in out                      # factory prefix applied
    assert "Executing Python script" in out        # milestone kept
    assert "Verification 2/7" in out                # milestone kept
    assert "Analysis approved" in out               # milestone kept
    assert "Sub-scores" not in out                  # detail trimmed
    assert "Justified plan deviations" not in out   # detail trimmed
    assert "Plan conformance issue" not in out      # WARNING churn trimmed
