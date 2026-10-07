"""A swarm item's narration is tagged with its label — on the thread path
by the item's own stream wrapper, on the process path by the relay — and the
two surfaces read the tag back (tests/test_narration.py). Fan-out branches
are untouched."""

import json
import sys
import threading
from pathlib import Path

from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent import swarm
from scilink.agents.meta_agent.placements import LocalProcess


def test_tag_lines_prefixes_every_line_that_begins_in_the_write():
    text, at_start = fo.tag_lines("A", "one\ntwo\n", True)
    assert (text, at_start) == ("[A] one\n[A] two\n", True)
    text, at_start = fo.tag_lines("A", "partial", True)
    assert (text, at_start) == ("[A] partial", False)
    text, at_start = fo.tag_lines("A", " more\nnext", False)      # continues the partial line
    assert (text, at_start) == (" more\n[A] next", False)
    assert fo.tag_lines("A", "\n", True) == ("[A] \n", True)
    assert fo.tag_lines("A", "", False) == ("", False)


def test_a_labelled_threads_prints_are_tagged_and_other_threads_are_not():
    class Sink:
        def __init__(self):
            self.chunks = []

        def write(self, data):
            self.chunks.append(data)
            return len(data)

        def flush(self):
            pass
    sink = Sink()
    stream = fo._ThreadStopStream(sink)
    ev = threading.Event()
    done = threading.Event()

    def item():
        fo._register_branch_stop(ev, label="XRD 300 K")
        try:
            stream.write("💭 first\n")
            stream.write("     cont")
            stream.write("inued\n")
        finally:
            fo._unregister_branch_stop()
            done.set()
    t = threading.Thread(target=item)
    t.start()
    done.wait(5)
    stream.write("the meta's own line\n")                 # this thread carries no label
    assert "".join(sink.chunks) == "[XRD 300 K] 💭 first\n[XRD 300 K]      continued\nthe meta's own line\n"
    # a fan-out branch registers no label: its output is as it was
    fo._register_branch_stop(threading.Event())
    try:
        stream.write("branch line\n")
    finally:
        fo._unregister_branch_stop()
    assert sink.chunks[-1] == "branch line\n"


def test_a_swarm_items_prints_carry_its_label_in_the_turn(tmp_path, monkeypatch, capsys):
    from test_run_swarm import FakeWorker
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SCILINK_SWARM_PLACEMENT", "thread")
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})

    class Talker(FakeWorker):
        def run_task(self, task, context=None, autonomy=None):
            print(f"💭 working on {task}")
            return super().run_task(task, context, autonomy)

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return Talker(mode, base_dir, lambda task: {})
    monkeypatch.setattr(swarm, "build_child", build)
    meta = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                                 meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    meta._enable_human_feedback = False
    res = json.loads(swarm.run_swarm(meta, [{"mode": "planning", "task": "a", "label": "plan A"},
                                            {"mode": "planning", "task": "b", "label": "plan B"}]))
    assert all(r["status"] == "success" for r in res["results"])
    out = capsys.readouterr().out
    assert "[plan A] 💭 working on a" in out and "[plan B] 💭 working on b" in out
    assert "[plan A] [plan" not in out                      # never tagged twice
    from scilink.ui.vocabulary import COORDINATOR_MARK
    assert COORDINATOR_MARK + "🐝" in out                     # the coordinator's own lines are marked, never tagged
    assert "[plan A]   🐝" not in out and "[plan A] " + COORDINATOR_MARK not in out


def test_the_relay_tags_a_process_workers_lines(capsys):
    import tempfile
    from test_swarm_scheduler import TARGET
    pl = LocalProcess(target=TARGET)
    h = pl.submit({"mode": "analysis", "task": "plain", "label": "XRD 300 K", "autonomy": "AUTONOMOUS",
                   "host": {}, "base_dir": tempfile.mkdtemp(prefix="swarm-item-")})
    pl.wait(h, poll_s=0.05)
    out = capsys.readouterr().out
    assert "[XRD 300 K] worker line 0" in out and "[XRD 300 K] worker line 2" in out


def test_threads_an_item_starts_inherit_its_label(monkeypatch):
    """Item 2 of the review: a best-of-N candidate thread started with
    ``attributed_to_current`` or ``inherited_context`` prints under the
    item's label, so two items' candidates can be told apart."""
    from scilink.utils.log_context import attributed_to_current, inherited_context

    class Sink:
        chunks = []

        def write(self, data):
            Sink.chunks.append(data)
            return len(data)

        def flush(self):
            pass
    Sink.chunks = []
    stream = fo._ThreadStopStream(Sink())
    done = threading.Event()

    def candidate(n):
        stream.write(f"[cand_{n:02d}] Verification 1/7\n")

    def item():
        fo._register_branch_stop(threading.Event(), label="fit A")
        try:
            t1 = threading.Thread(target=attributed_to_current(candidate), args=(1,))
            ctx = inherited_context()                      # made on the item's thread

            def with_ctx():
                with ctx.applied():                        # applied on the child
                    candidate(2)
            t2 = threading.Thread(target=with_ctx)
            t1.start(); t2.start(); t1.join(5); t2.join(5)
            stream.write("💭 the item itself\n")
        finally:
            fo._unregister_branch_stop()
            done.set()
    threading.Thread(target=item).start()
    done.wait(10)
    out = "".join(Sink.chunks)
    assert "[fit A] [cand_01] Verification 1/7\n" in out and "[fit A] [cand_02] Verification 1/7\n" in out
    assert "[fit A] 💭 the item itself\n" in out
    from scilink.ui.narration import LineClassifier
    ln = LineClassifier().push("[fit A] [cand_01] Verification 1/7")
    assert (ln.kind, ln.worker) == ("candidate", "fit A")


def test_a_long_or_bracketed_label_is_tagged_in_a_form_the_reader_accepts():
    from scilink.ui.narration import LineClassifier
    from scilink.utils.log_context import current_label

    class Sink:
        chunks = []

        def write(self, data):
            Sink.chunks.append(data)
            return len(data)

        def flush(self):
            pass
    Sink.chunks = []
    stream = fo._ThreadStopStream(Sink())
    done = threading.Event()
    label = "Raman A7 anatase fraction vs annealing temperature series [second pass]"

    def item():
        fo._register_branch_stop(threading.Event(), label=label)
        try:
            Sink.tag = current_label()
            stream.write("💭 a thought\n")
        finally:
            fo._unregister_branch_stop()
            done.set()
    threading.Thread(target=item).start()
    done.wait(5)
    ln = LineClassifier().push("".join(Sink.chunks).rstrip("\n"))
    assert (ln.kind, ln.verbose, ln.worker) == ("thought", False, Sink.tag)
    assert len(Sink.tag) <= 48 and "[" not in Sink.tag


def test_the_relay_inside_a_labelled_item_does_not_tag_twice(capsys, monkeypatch):
    import tempfile
    from scilink.utils.log_context import register_label, unregister_label
    from test_swarm_scheduler import TARGET
    monkeypatch.setattr(sys, "stdout", fo._ThreadStopStream(sys.stdout))   # as a swarm installs it
    register_label("XRD 300 K")
    try:
        pl = LocalProcess(target=TARGET)
        h = pl.submit({"mode": "analysis", "task": "plain", "label": "XRD 300 K", "autonomy": "AUTONOMOUS",
                       "host": {}, "base_dir": tempfile.mkdtemp(prefix="swarm-item-")})
        pl.wait(h, poll_s=0.05)
    finally:
        unregister_label()
    out = capsys.readouterr().out
    assert "[XRD 300 K] worker line 1" in out and "[XRD 300 K] [XRD 300 K]" not in out


def test_the_line_start_state_goes_with_the_label():
    """The nit: a child thread that ended mid-line must not leave its state
    for a later thread with a reused id."""
    import threading as _t
    from scilink.utils.log_context import line_start, register_label, unregister_label
    tid = _t.get_ident()
    register_label("X")
    try:
        text, at_start = fo.tag_lines("X", "partial", line_start(tid))
        from scilink.utils.log_context import set_line_start
        set_line_start(tid, at_start)
        assert line_start(tid) is False
    finally:
        unregister_label()
    assert line_start(tid) is True                 # cleared with the label
