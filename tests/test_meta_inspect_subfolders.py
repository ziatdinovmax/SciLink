"""``inspect_uploads`` sees subfolders.

Its listing was one level deep AND silent about directories: a nested drop
(one folder per sample / condition) came back as "empty" or as the few loose
top-level files, so the meta had no evidence anything deeper existed and
could not route it. Now every listing names its subfolders (file counts,
extension histogram, whether they nest further), and ``recursive=True``
probes the nested files within a depth / count cap.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.meta_agent.meta_orchestrator_tools import (
    MetaOrchestratorTools, _walk_files)


def _tools(base_dir: Path):
    t = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
    t.orch = SimpleNamespace(model=None, base_dir=base_dir)
    t.logger = None
    cap = {}
    t._register_tool = (
        lambda func, name, description, parameters, required=None:
        cap.update({name: (func, parameters)}))
    MetaOrchestratorTools._register_all_tools(t)
    return cap


@pytest.fixture
def nested(tmp_path):
    up = tmp_path / "uploads"
    (up / "sampleA").mkdir(parents=True)
    (up / "sampleB" / "deeper").mkdir(parents=True)
    (up / ".hidden").mkdir()
    (up / "README.md").write_text("top-level note")
    np.save(up / "sampleA" / "scan1.npy", np.zeros((4, 4)))
    np.save(up / "sampleA" / "scan2.npy", np.zeros((4, 4)))
    (up / "sampleA" / "scan1.json").write_text('{"temperature": 5}')
    (up / "sampleB" / "spec.csv").write_text("x,y\n1,2\n")
    (up / "sampleB" / "deeper" / "log.txt").write_text("deep")
    (up / ".hidden" / "secret.csv").write_text("no")
    (up / "sampleA" / ".DS_Store").write_bytes(b"\x00")
    return tmp_path


def test_listing_names_subfolders_without_recursing(nested):
    cap = _tools(nested)
    inspect, params = cap["inspect_uploads"]
    out = json.loads(inspect())
    assert out["status"] == "success"
    # one-level listing unchanged: only the loose top-level file is probed
    assert [Path(f["file"]).name for f in out["files"]] == ["README.md"]
    assert out["n_files"] == 1
    subs = {Path(d["path"]).name: d for d in out["subdirectories"]}
    assert set(subs) == {"sampleA", "sampleB"}          # hidden dir skipped
    assert subs["sampleA"]["n_files"] == 3              # .DS_Store skipped
    assert subs["sampleA"]["extensions"] == {".json": 1, ".npy": 2}
    assert subs["sampleA"]["n_subdirs"] == 0
    assert subs["sampleB"]["n_subdirs"] == 1
    # the model is told what it is not seeing, and how to see it
    assert "recursive" in out["hint"]
    # the schema exposes the knobs
    assert "recursive" in params and "max_depth" in params


def test_recursive_probes_nested_files_top_down(nested):
    cap = _tools(nested)
    inspect, _ = cap["inspect_uploads"]
    out = json.loads(inspect(recursive=True))
    names = [Path(f["file"]).name for f in out["files"]]
    assert names[0] == "README.md"                      # top level first
    assert {"scan1.npy", "scan2.npy", "scan1.json", "spec.csv", "log.txt"} <= set(names)
    assert "secret.csv" not in names and ".DS_Store" not in names
    assert out["recursive"] is True and out["n_files"] == 6
    assert "hint" not in out
    # probes still carry content evidence for nested files
    npy = next(f for f in out["files"] if f["file"].endswith("scan1.npy"))
    assert npy["shape"] == [4, 4]
    # the subdirectory summary is present in both modes
    assert len(out["subdirectories"]) == 2


def test_max_depth_caps_the_walk(nested):
    cap = _tools(nested)
    inspect, _ = cap["inspect_uploads"]
    out = json.loads(inspect(recursive=True, max_depth=1))
    names = [Path(f["file"]).name for f in out["files"]]
    assert "spec.csv" in names and "log.txt" not in names
    assert out["depth_truncated"] is True


def test_explicit_subfolder_and_single_file_paths_still_work(nested):
    cap = _tools(nested)
    inspect, _ = cap["inspect_uploads"]
    sub = json.loads(inspect(path=str(nested / "uploads" / "sampleA")))
    assert sub["n_files"] == 3 and sub["subdirectories"] == []
    one = json.loads(inspect(path=str(nested / "uploads" / "sampleB" / "spec.csv")))
    assert one["n_files"] == 1 and one["files"][0]["kind"] == "table"
    missing = json.loads(inspect(path=str(nested / "nope")))
    assert missing["status"] == "error"


def test_walk_file_cap(tmp_path, monkeypatch):
    import scilink.agents.meta_agent.meta_orchestrator_tools as m
    monkeypatch.setattr(m, "_WALK_FILES_MAX", 3)
    d = tmp_path / "many"
    (d / "s").mkdir(parents=True)
    for i in range(5):
        (d / "s" / f"f{i}.txt").write_text("x")
    walked = _walk_files(d, max_depth=3)
    assert len(walked["files"]) == 3 and walked["depth_truncated"] is True
