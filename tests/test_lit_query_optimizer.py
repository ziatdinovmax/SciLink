"""Offline tests for `optimize_query_for_analysis`.

Mocks the LLM client; no API calls. Renders real previews (matplotlib +
PIL) so the dispatch paths are exercised end-to-end.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.lit_agents.optimize_query_for_analysis import (
    DEFAULT_SCOUT_N,
    _build_preview,
    optimize_query_for_analysis,
)


class _FakeModel:
    """Captures the last `generate_content` call and returns a canned response."""

    def __init__(self, refined: str = "REFINED QUERY", raise_exc: Exception | None = None):
        self.refined = refined
        self.raise_exc = raise_exc
        self.last_parts: list | None = None
        self.call_count = 0

    def generate_content(self, parts):
        self.call_count += 1
        self.last_parts = parts
        if self.raise_exc:
            raise self.raise_exc
        return SimpleNamespace(text=json.dumps({"refined_query": self.refined}))


def _has_image_part(parts: list) -> bool:
    return any(isinstance(p, dict) and "mime_type" in p and "data" in p for p in parts)


# --- Single image -------------------------------------------------------------

def test_single_image_npy_builds_preview_and_calls_llm(tmp_path: Path):
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    path = tmp_path / "img.npy"
    np.save(path, arr)

    model = _FakeModel(refined="refined Q for single image")
    out = optimize_query_for_analysis(
        raw_query="generic grain query",
        data_type="image",
        data_path=str(path),
        metadata={"material": "steel"},
        model=model,
    )

    assert model.call_count == 1
    assert _has_image_part(model.last_parts)
    assert out == "refined Q for single image"


# --- Image series via directory ----------------------------------------------

def test_image_series_directory_builds_montage(tmp_path: Path):
    for i in range(7):
        arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
        np.save(tmp_path / f"frame_{i:02d}.npy", arr)

    model = _FakeModel(refined="series-refined")
    preview = _build_preview("image", str(tmp_path), {"technique": "AFM"})
    assert preview is not None
    assert preview[1] == "image/jpeg"
    assert isinstance(preview[0], (bytes, bytearray)) and len(preview[0]) > 100

    out = optimize_query_for_analysis(
        raw_query="texture analysis",
        data_type="image",
        data_path=str(tmp_path),
        metadata={"technique": "AFM"},
        model=model,
    )
    assert model.call_count == 1
    assert _has_image_part(model.last_parts)
    assert out == "series-refined"


# --- Image series stored as 3D .npy stack ------------------------------------

def test_image_stack_npy_builds_montage(tmp_path: Path):
    stack = (np.random.rand(8, 32, 32) * 255).astype(np.uint8)
    path = tmp_path / "stack.npy"
    np.save(path, stack)

    preview = _build_preview("image", str(path), None)
    assert preview is not None and preview[1] == "image/jpeg"
    # Montage should be larger than a single-frame thumbnail
    assert len(preview[0]) > 500


# --- Single curve -------------------------------------------------------------

def test_single_curve_npy_builds_plot(tmp_path: Path):
    x = np.linspace(0, 10, 200)
    y = np.exp(-((x - 5) ** 2) / 2)
    path = tmp_path / "spec.npy"
    np.save(path, np.stack([x, y]))  # shape (2, N)

    preview = _build_preview("curve", str(path), {"xlabel": "x", "ylabel": "I"})
    assert preview is not None and preview[1] == "image/png"


# --- Curve series via 2D .npy stack ------------------------------------------

def test_curve_stack_npy_builds_overlay(tmp_path: Path):
    spectra = np.random.rand(6, 200)  # N=6 spectra of length 200
    path = tmp_path / "spectra.npy"
    np.save(path, spectra)

    preview = _build_preview("curve", str(path), None)
    assert preview is not None and preview[1] == "image/png"


# --- Fallbacks ----------------------------------------------------------------

def test_no_data_no_metadata_skips_llm():
    """No data and no metadata: refinement is pointless — skip the LLM call."""
    model = _FakeModel()
    out = optimize_query_for_analysis(
        raw_query="raw",
        data_type=None,
        data_path=None,
        metadata=None,
        model=model,
    )
    assert out == "raw"
    assert model.call_count == 0  # no point calling LLM with nothing to add


def test_metadata_only_still_refines():
    """No data path, but metadata is present — still worth a refinement pass."""
    model = _FakeModel(refined="metadata-grounded query")
    out = optimize_query_for_analysis(
        raw_query="raw",
        data_type=None,
        data_path=None,
        metadata={"material": "MoS2", "technique": "Raman"},
        model=model,
    )
    assert out == "metadata-grounded query"
    assert model.call_count == 1
    # no image part — metadata-only path
    assert not _has_image_part(model.last_parts)


def test_llm_error_returns_raw(tmp_path: Path):
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    path = tmp_path / "img.npy"
    np.save(path, arr)

    model = _FakeModel(raise_exc=RuntimeError("rate limited"))
    out = optimize_query_for_analysis(
        raw_query="raw fallback query",
        data_type="image",
        data_path=str(path),
        metadata={"material": "steel"},
        model=model,
    )
    assert out == "raw fallback query"


def test_unparseable_llm_response_returns_raw(tmp_path: Path):
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    path = tmp_path / "img.npy"
    np.save(path, arr)

    class _NoiseModel:
        def generate_content(self, parts):
            return SimpleNamespace(text="not json\nmulti\nline gibberish")

    out = optimize_query_for_analysis(
        raw_query="raw",
        data_type="image",
        data_path=str(path),
        metadata={"material": "steel"},
        model=_NoiseModel(),
    )
    assert out == "raw"


def test_unknown_data_type_metadata_only(tmp_path: Path):
    """Unknown/unsupported data_type: fall through to metadata-only refinement."""
    model = _FakeModel(refined="meta-only refined")
    out = optimize_query_for_analysis(
        raw_query="raw",
        data_type="hyperspectral",  # no preview builder in v1
        data_path=str(tmp_path),
        metadata={"technique": "EELS"},
        model=model,
    )
    assert out == "meta-only refined"
    assert model.call_count == 1
    assert not _has_image_part(model.last_parts)


# --- Sampling stride ----------------------------------------------------------

def test_image_series_samples_at_most_N(tmp_path: Path):
    """Directory with >>N frames → exactly N images in the montage."""
    for i in range(20):
        np.save(tmp_path / f"f_{i:03d}.npy", (np.random.rand(16, 16) * 255).astype(np.uint8))

    # We can't easily inspect the montage layout, but we can verify the
    # sampling code path didn't crash and produced a non-trivial JPEG.
    preview = _build_preview("image", str(tmp_path), None)
    assert preview is not None
    # Larger than any single-frame thumbnail
    assert len(preview[0]) > 1000
    # Sanity: DEFAULT_SCOUT_N is a small fixed number
    assert DEFAULT_SCOUT_N == 5
