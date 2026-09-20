"""``-p`` mode: result on stdout in the requested format, narration on
stderr, exit code from the task status, every question auto-accepted."""

import json

import pytest

from scilink.cli.shell import headless

from shell_fakes import FakeAdapter, make_args


def test_json_output_and_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = FakeAdapter()
    args = make_args(adapter, ["-p", "analyze it", "--output-format", "json", "--yes"])
    code = headless.run(adapter, args, args.print_task)
    out, err = capsys.readouterr()
    assert code == 0
    payload = json.loads(out)
    assert payload["status"] == "success" and payload["summary"] == "done: analyze it"
    assert "Calling tool" in err and "Calling tool" not in out
    assert adapter.built[0].mode == "autonomous"


def test_text_and_markdown_output(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = FakeAdapter()
    args = make_args(adapter, ["-p", "task", "--yes"])
    headless.run(adapter, args, "task")
    out, _ = capsys.readouterr()
    assert out.startswith("done: task") and "Key findings:" in out and "- finding one" in out
    md = headless.render_result({"summary": "s", "key_findings": ["f"], "files_produced": ["a"]},
                                "markdown")
    assert "## Key findings" in md and "## Files" in md


def test_failure_status_exit_code():
    assert headless.exit_code({"status": "error"}) == 1
    assert headless.exit_code({"status": "success"}) == 0


def test_questions_are_auto_accepted(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class AskingAdapter(FakeAdapter):
        def headless_run(self, agent, task):
            agent.chat(task)   # asks through the chokepoint
            return {"status": "success", "summary": "answered " + ",".join(agent.answers)}

    adapter = AskingAdapter()
    args = make_args(adapter, ["-p", "t", "--yes"])
    code = headless.run(adapter, args, "t")
    out, _ = capsys.readouterr()
    assert code == 0 and out.strip() == "answered"
    assert adapter.built[0].answers == [""]
