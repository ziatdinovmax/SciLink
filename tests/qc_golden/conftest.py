import logging

import pytest

logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def _golden_env(monkeypatch):
    """Sandbox approval + memory staging gated off for deterministic offline runs.

    Set via monkeypatch (per-test, auto-restored) rather than module-level
    ``os.environ`` writes: the latter leaked ``SCILINK_T2_AUTODISTILL=0``
    process-wide and broke ``test_persistent_memory``'s staging tests whenever
    the two files shared a pytest process.
    """
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    monkeypatch.setenv("SCILINK_T2_AUTODISTILL", "0")
    monkeypatch.setenv("SCILINK_FEEDBACK_AUTODISTILL", "0")
