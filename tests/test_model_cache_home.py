"""Downloaded model weights live under the relocatable model cache.

``SCILINK_HOME`` (or ``SCILINK_MODELS``) relocates every persistent store; the
SAM checkpoint dir used to be pinned to ``~/.cache/scilink/checkpoints`` at
three sites, so a relocated home (a container, a per-campaign volume) lost
its weights and re-downloaded them. All three now share one rule, and the old
location is honoured only when it already holds weights the cache lacks.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _rule_via_probe(env):
    from scilink.agents.exp_agents._exec_env import PROBE_SCRIPT
    script = PROBE_SCRIPT % {"packages": "[]", "marker": "@@PROBE@@"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("@@PROBE@@"))
    return json.loads(line[len("@@PROBE@@"):])["checkpoint_dir"]


def test_sam_checkpoint_dir_follows_the_model_cache(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from scilink.skills._shared.particle_analyzer import sam_checkpoint_dir
    monkeypatch.setenv("HOME", str(tmp_path / "h"))     # no real legacy cache
    monkeypatch.delenv("SCILINK_MODELS", raising=False)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    assert Path(sam_checkpoint_dir()) == tmp_path / "home" / "models" / "sam"
    monkeypatch.setenv("SCILINK_MODELS", str(tmp_path / "weights"))
    assert Path(sam_checkpoint_dir()) == tmp_path / "weights" / "sam"


def test_legacy_cache_kept_only_while_it_holds_the_weights(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from scilink.skills._shared.particle_analyzer import sam_checkpoint_dir
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.delenv("SCILINK_MODELS", raising=False)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    legacy = tmp_path / "h" / ".cache" / "scilink" / "checkpoints"
    legacy.mkdir(parents=True)
    (legacy / "sam_vit_h.pth").write_bytes(b"x")
    assert Path(sam_checkpoint_dir()) == legacy          # nothing re-downloaded
    current = tmp_path / "home" / "models" / "sam"
    current.mkdir(parents=True)
    (current / "sam_vit_h.pth").write_bytes(b"x")
    assert Path(sam_checkpoint_dir()) == current         # the cache wins once filled


def test_sandbox_probe_reports_the_same_dir(tmp_path, monkeypatch):
    import os
    env = {**os.environ, "HOME": str(tmp_path / "h"),
           "SCILINK_HOME": str(tmp_path / "home")}
    env.pop("SCILINK_MODELS", None)
    assert Path(_rule_via_probe(env)) == tmp_path / "home" / "models" / "sam"
    env["SCILINK_MODELS"] = str(tmp_path / "weights")
    assert Path(_rule_via_probe(env)) == tmp_path / "weights" / "sam"
