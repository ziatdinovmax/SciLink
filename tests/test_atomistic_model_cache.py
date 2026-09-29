"""The DCNN ensemble is downloaded once per machine, not once per image.

Generated analysis scripts run in a fresh per-item working directory. The model
manager's default directory was relative (``dcnn_trained``), so every image
analysis that used the atom finder downloaded ~770 MB into its own folder.
Found live: 11 frames of a live image loop took 36 s each and filled a disk.
"""

import logging
import os
from pathlib import Path

from scilink.skills._shared import atomistic_model_manager as mm


def _fake_download(calls):
    def download(gdrive_id, output_dir, logger, **kwargs):
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


def test_release_asset_is_tried_first_and_drive_is_the_fallback(tmp_path, monkeypatch):
    """The weights come from the SciLink release asset over plain HTTPS; the
    Google Drive copy is only the fallback, so a CI build or a container
    never depends on Drive when the asset is reachable."""
    import logging
    from scilink.skills._shared import atomistic_model_manager as m
    from scilink.skills.image_analysis.atomic_stem import atomic_stem as tools
    calls = []
    def fake_url(url, dest, logger):
        calls.append(("url", url))
        Path(dest).write_bytes(b"zip")
        return dest
    def fake_gdown(gid, dest, logger):
        calls.append(("gdown", gid))
        Path(dest).write_bytes(b"zip")
        return dest
    monkeypatch.setattr(m, "_download_url", fake_url)
    monkeypatch.setattr(tools, "download_file_with_gdown", fake_gdown)
    monkeypatch.setattr(tools, "unzip_file", lambda z, out, logger: True)
    out = str(tmp_path / "dcnn_trained")
    assert m._download_and_extract_model("gid", out, logging.getLogger("t"), url="https://x/y.zip")
    assert calls == [("url", "https://x/y.zip")]
    calls.clear()
    monkeypatch.setattr(m, "_download_url", lambda url, dest, logger: calls.append(("url", url)) or None)
    assert m._download_and_extract_model("gid", out, logging.getLogger("t"), url="https://x/y.zip")
    assert calls == [("url", "https://x/y.zip"), ("gdown", "gid")]
    calls.clear()
    assert m._download_and_extract_model("gid", out, logging.getLogger("t"))     # no url: Drive only
    assert calls == [("gdown", "gid")]


def test_url_download_cleans_up_a_failed_partial(tmp_path):
    import logging
    from scilink.skills._shared import atomistic_model_manager as m
    dest = str(tmp_path / "w.zip")
    assert m._download_url("https://127.0.0.1:9/nothing.zip", dest, logging.getLogger("t")) is None
    assert not (tmp_path / "w.zip").exists() and not (tmp_path / "w.zip.part").exists()
    assert m.DEFAULT_DCNN_MODEL_URL.startswith("https://github.com/ziatdinovmax/SciLink/releases/download/")
