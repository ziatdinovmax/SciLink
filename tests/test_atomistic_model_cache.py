"""The DCNN ensemble is downloaded once per machine, not once per image.

Generated analysis scripts run in a fresh per-item working directory. The model
manager's default directory was relative (``dcnn_trained``), so every image
analysis that used the atom finder downloaded ~770 MB into its own folder.
Found live: 11 frames of a live image loop took 36 s each and filled a disk.
"""

import logging
import os

from scilink.skills._shared import atomistic_model_manager as mm


def _fake_download(calls):
    def download(gdrive_id, output_dir, logger):
        calls.append(output_dir)
        os.makedirs(os.path.join(output_dir, "atomnet_ensemble"), exist_ok=True)
        open(os.path.join(output_dir, "atomnet_ensemble", "atomnet3_0.tar"), "w").close()
        return True
    return download


def test_the_default_is_a_persistent_cache_shared_by_every_working_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_MODELS", raising=False)
    monkeypatch.setattr(mm, "_download_and_extract_model", _fake_download(calls))
    paths = []
    for item in ("image_0000", "image_0001", "frame_000007"):       # three scripts, three working directories
        (tmp_path / item).mkdir()
        monkeypatch.chdir(tmp_path / item)
        paths.append(mm.get_or_download_atomistic_model({}, logging.getLogger("t")))
        assert not (tmp_path / item / "dcnn_trained").exists()      # nothing lands next to the data
    assert len(calls) == 1 and len(set(paths)) == 1
    assert paths[0].startswith(str(tmp_path / "home" / "models" / "dcnn_trained"))


def test_a_folder_already_in_the_working_directory_still_wins(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mm, "_download_and_extract_model", _fake_download(calls))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dcnn_trained").mkdir()
    (tmp_path / "dcnn_trained" / "atomnet3_a.tar").write_text("")
    assert mm.get_or_download_atomistic_model({}, logging.getLogger("t")) == "dcnn_trained"
    assert calls == []


def test_the_cache_can_be_placed_elsewhere(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SCILINK_MODELS", str(tmp_path / "big_disk"))
    monkeypatch.setattr(mm, "_download_and_extract_model", _fake_download(calls))
    monkeypatch.chdir(tmp_path)
    path = mm.get_or_download_atomistic_model({}, logging.getLogger("t"))
    assert path.startswith(str(tmp_path / "big_disk" / "dcnn_trained"))
