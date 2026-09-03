"""ArtifactTracker re-surfaces a figure rewritten in place.

A curve-fit / hyperspectral refinement loop overwrites its
``visualization.png`` at the same path across turns. The image sweep used to
key its known-set on the bare path, so the rewritten figure was judged
"already seen" and never re-emitted — the web UI kept showing the first
render. Image identity is now ``path + mtime_ns`` (matching the report/doc
sweeps), so a rewrite re-surfaces while a byte-for-byte unchanged file does
not. Deterministic, offline.
"""
import os

from scilink.server.artifacts import ArtifactTracker


def _write(p, text):
    p.write_text(text)
    return p


def _bump_mtime(p, delta_ns=5_000_000_000):
    """Force a strictly newer mtime (avoids coarse-filesystem-resolution
    flakiness where two quick writes share a timestamp)."""
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns + delta_ns, st.st_mtime_ns + delta_ns))


def test_rewritten_image_is_resurfaced(tmp_path):
    png = _write(tmp_path / "visualization.png", "v1")
    t = ArtifactTracker(str(tmp_path))

    first = t.find_new_images()
    assert str(png) in first

    # Same path, unchanged content/mtime -> not re-surfaced.
    assert t.find_new_images() == []

    # Overwrite in place with a newer mtime -> surfaced again.
    png.write_text("v2")
    _bump_mtime(png)
    again = t.find_new_images()
    assert str(png) in again, "a rewritten figure must re-surface"


def test_resume_marks_current_version_but_not_a_later_rewrite(tmp_path):
    png = _write(tmp_path / "visualization.png", "v1")
    t = ArtifactTracker(str(tmp_path))
    t.mark_all_existing()  # resume: everything on disk is already surfaced

    assert t.find_new_images() == []  # the current version stays suppressed

    png.write_text("v2")
    _bump_mtime(png)
    assert str(png) in t.find_new_images()  # but a later rewrite comes through


def test_unchanged_file_across_sweeps_never_repeats(tmp_path):
    _write(tmp_path / "a.png", "x")
    t = ArtifactTracker(str(tmp_path))
    assert len(t.find_new_images()) == 1
    for _ in range(3):
        assert t.find_new_images() == []


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
