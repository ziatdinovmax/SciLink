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
    assert "\n🐝" in out or "  🐝" in out                    # the coordinator's own lines carry no tag
    assert "[plan A]   🐝" not in out


def test_the_relay_tags_a_process_workers_lines(capsys):
    import tempfile
    from test_swarm_scheduler import TARGET
    pl = LocalProcess(target=TARGET)
    h = pl.submit({"mode": "analysis", "task": "plain", "label": "XRD 300 K", "autonomy": "AUTONOMOUS",
                   "host": {}, "base_dir": tempfile.mkdtemp(prefix="swarm-item-")})
    pl.wait(h, poll_s=0.05)
    out = capsys.readouterr().out
    assert "[XRD 300 K] worker line 0" in out and "[XRD 300 K] worker line 2" in out
