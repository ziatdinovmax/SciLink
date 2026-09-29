"""``scilink analyze -p TASK --data X --metadata Y`` hands X and Y to the task.

The interactive shell seeds its first turn from --data / --metadata
(``initial_turns``); headless runs one task through ``run_task`` and never
read those turns, so both flags were silently dropped: live, the agent
answered that no data path had been given and asked for one.
"""

import json
from pathlib import Path

from scilink.cli.shell import headless
from scilink.cli.shell.modes import AnalyzeAdapter

from shell_fakes import make_args


class _Agent:
    def __init__(self):
        self.calls = []
        self.mode = "co-pilot"

    def run_task(self, task, context=None, autonomy=None):
        self.calls.append({"task": task, "context": context, "autonomy": autonomy})
        return {"status": "success", "summary": "ok", "key_findings": [], "files_produced": []}


class _Adapter(AnalyzeAdapter):
    """The real analyze adapter (flags, headless_context, headless_run) with
    the orchestrator swapped for a recorder."""

    def __init__(self):
        self.agent = _Agent()

    def build(self, args, creds, session_dir, *, restore, extras):
        return self.agent

    def get_autonomy(self, agent):
        return agent.mode

    def set_autonomy(self, agent, level):
        agent.mode = level


def _files(tmp_path):
    data = tmp_path / "spectrum.txt"
    data.write_text("1 2\n3 4\n")
    meta = tmp_path / "spectrum.json"
    meta.write_text(json.dumps({"technique": "Raman"}))
    return data, meta


def test_data_and_metadata_reach_the_headless_task(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    data, meta = _files(tmp_path)
    adapter = _Adapter()
    args = make_args(adapter, ["-p", "fit the peaks", "--data", str(data), "--metadata", str(meta),
                               "--yes", "--api-key", "k", "--session-dir", str(tmp_path / "s")])
    assert headless.run(adapter, args, args.print_task) == 0
    call = adapter.agent.calls[0]
    assert call["task"] == "fit the peaks"
    assert call["context"]["data_path"] == str(data.absolute())
    assert call["context"]["metadata_path"] == str(meta.absolute())
    assert call["context"]["metadata_format"].startswith("JSON")


def test_a_text_description_is_labelled_for_conversion(tmp_path):
    desc = tmp_path / "notes.txt"
    desc.write_text("Raman, 532 nm")
    ctx = AnalyzeAdapter().headless_context(make_args(AnalyzeAdapter(), ["--metadata", str(desc)]))
    assert ctx == {"metadata_path": str(desc.absolute()),
                   "metadata_format": "text description: convert it to metadata"}


def test_no_inputs_means_no_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    adapter = _Adapter()
    args = make_args(adapter, ["-p", "t", "--yes", "--api-key", "k", "--session-dir", str(tmp_path / "s")])
    headless.run(adapter, args, "t")
    assert adapter.agent.calls[0]["context"] is None


def test_every_mode_accepts_a_headless_context():
    import inspect
    from scilink.cli.shell.modes import ADAPTERS
    for key, cls in ADAPTERS.items():
        params = inspect.signature(cls.headless_run).parameters
        assert "context" in params, key
