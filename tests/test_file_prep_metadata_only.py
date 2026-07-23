"""Deterministic metadata-only split for pickled .npy containers (issue #380).

Covers the three layers of the fix:
  1. the member-scan gate (extraction-biased: any ndarray → NOT metadata-only);
  2. the no-LLM metadata-only split (real header-shaped dict → metadata JSON,
     no data leg, model/executor never touched);
  3. the unblinded meta probe (pickled .npy → structural description instead
     of "unreadable").
"""

import io
import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from scilink.utils.file_prep import (
    _is_data_member,
    _find_data_members,
    _pyobj_equal,
    prepare_inputs,
    prepare_inputs_batch,
    probe_pickled_npy,
    split_pickled_metadata_only,
    stage_pairs_flat,
    _file_paths,
)


# Mirrors the real-world instrument-header shape that motivated the fix:
# scalars, strings, and SMALL numeric lists only — no data member anywhere.
HEADER = {
    "dim_px": [168, 168],
    "pos_xy": [-8.826198e-09, 1.862286e-07],
    "size_xy": [1.5e-08, 1.5e-08],
    "angle": -70.0,
    "sweep_signal": "Bias (V)",
    "fixed_parameters": ["Sweep Start", "Sweep End"],
    "num_sweep_signal": 151,
    "channels": ["Current (A)", "LIX 1 omega (A)", "LIY 1 omega (A)"],
    "measure_delay": 0.01,
    "comment": "1V 200pA, 1 to -1, 977hz 20mV",
}


def _save_pickled(tmp_path: Path, obj, name="header.npy") -> Path:
    p = tmp_path / name
    np.save(p, np.array(obj, dtype=object) if isinstance(obj, dict) else obj,
            allow_pickle=True)
    return p


# ---------------------------------------------------------------------------
# 1. Member-scan gate
# ---------------------------------------------------------------------------

def test_gate_any_ndarray_is_data_regardless_of_size():
    # Extraction bias: even a 2-element ndarray disqualifies metadata-only.
    assert _is_data_member(np.array([1.5, 2.5]))
    assert _is_data_member(np.zeros((168, 168, 151)))


def test_gate_small_numeric_lists_are_metadata():
    assert not _is_data_member([168, 168])
    assert not _is_data_member((-8.8e-09, 1.9e-07))


def test_gate_large_numeric_list_is_data():
    assert _is_data_member(list(np.linspace(-1, 1, 151)))
    # ... including a nested numeric block whose OUTER length is small
    assert _is_data_member([[float(i) for i in range(100)] for _ in range(2)])


def test_gate_non_numeric_members_are_metadata():
    assert not _is_data_member("Bias (V)")
    assert not _is_data_member(151)
    assert not _is_data_member(["Sweep Start", "Sweep End"])
    assert not _is_data_member(np.array([True, False]))  # bool ≠ numeric data


def test_scan_finds_nested_data():
    obj = {"meta": {"a": 1}, "inner": {"spectrum": np.arange(10.0)}}
    assert _find_data_members(obj) == ["inner.spectrum"]
    assert _find_data_members(HEADER) == []


def test_scan_finds_ndarrays_nested_in_lists():
    # The any-size ndarray rule must hold THROUGH a plain list: a list judged
    # non-data as a whole (ragged, or coerced block under the list threshold)
    # can still hold real ndarray members.
    ragged = {"spectra": [np.arange(20.0), np.arange(31.0)], "note": "hdr"}
    assert _find_data_members(ragged) == ["spectra[0]", "spectra[1]"]
    small = {"spectra": [np.arange(10.0), np.arange(10.0)]}
    assert _find_data_members(small) == ["spectra[0]", "spectra[1]"]
    # small scalar lists (and nested ones) are still metadata
    assert _find_data_members({"dim_px": [168, 168]}) == []
    assert _find_data_members({"roi": [[1.0, 2.0], [3.0, 4.0]]}) == []


def test_list_nested_ndarray_forces_codegen_path(tmp_path):
    p = _save_pickled(
        tmp_path,
        dict(HEADER, spectra=[np.arange(20.0), np.arange(31.0)]),
        "h3.npy",
    )
    assert split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out")) is None


# ---------------------------------------------------------------------------
# 2. Metadata-only split
# ---------------------------------------------------------------------------

def test_metadata_only_split_on_header(tmp_path):
    p = _save_pickled(tmp_path, HEADER)
    res = split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out"))
    assert res is not None and res["status"] == "success"
    assert res["data_path"] is None and res["metadata_only"] is True
    meta = json.loads(Path(res["metadata_path"]).read_text())
    assert meta["dim_px"] == [168, 168]
    assert meta["comment"] == HEADER["comment"]
    assert set(meta) == set(HEADER)


def test_combined_container_is_not_eligible(tmp_path):
    combined = dict(HEADER, spectrum=np.linspace(0, 1, 151))
    p = _save_pickled(tmp_path, combined, "combined.npy")
    assert split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out")) is None


def test_small_ndarray_forces_codegen_path(tmp_path):
    # The bias rule: an ndarray member of ANY size routes through codegen.
    p = _save_pickled(tmp_path, dict(HEADER, pos=np.array([1.0, 2.0])), "h2.npy")
    assert split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out")) is None


def test_plain_numeric_npy_is_not_eligible(tmp_path):
    p = tmp_path / "cube.npy"
    np.save(p, np.zeros((4, 4, 5)))
    assert split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out")) is None


def test_non_dict_container_is_wrapped(tmp_path):
    p = tmp_path / "strings.npy"
    np.save(p, np.array(["a", "b", "c"], dtype=object), allow_pickle=True)
    res = split_pickled_metadata_only(p, _file_paths(p, tmp_path / "out"))
    assert res is not None and res["status"] == "success"
    meta = json.loads(Path(res["metadata_path"]).read_text())
    assert meta == {"value": ["a", "b", "c"]}


def test_prepare_inputs_short_circuits_without_llm(tmp_path):
    # model/executor are never touched on the deterministic path — passing
    # None for both proves it.
    p = _save_pickled(tmp_path, HEADER)
    res = prepare_inputs(p, model=None, executor=None, output_dir=tmp_path / "out")
    assert res["status"] == "success" and res.get("metadata_only") is True
    assert res["attempts"] == 0


class _FailingModel:
    """A model whose use proves the codegen path was entered."""
    def generate_content(self, prompt):
        raise AssertionError("codegen path entered")


def test_prepare_inputs_combined_still_uses_codegen(tmp_path):
    # _generate_split_script folds model exceptions into a normal error
    # result, so the sentinel model's message surfacing there proves the
    # combined container took the codegen path (not the metadata-only one).
    combined = dict(HEADER, spectrum=np.linspace(0, 1, 151))
    p = _save_pickled(tmp_path, combined, "combined.npy")
    res = prepare_inputs(p, model=_FailingModel(), executor=None,
                         output_dir=tmp_path / "out", max_retries=0)
    assert res["status"] == "error"
    assert "codegen path entered" in res["message"]


def test_batch_mixes_metadata_only_and_arrays(tmp_path):
    h = _save_pickled(tmp_path, HEADER)
    res = prepare_inputs_batch([h], model=None, executor=None,
                               output_dir=tmp_path / "out")
    assert res["status"] == "success"
    assert res["results"][0]["metadata_only"] is True
    # metadata-only entries never join flat series staging (no data leg)
    staged = stage_pairs_flat(res["results"], tmp_path / "flat")
    assert staged["n"] == 0


# ---------------------------------------------------------------------------
# 3. Probe + comparator
# ---------------------------------------------------------------------------

def test_probe_pickled_npy_describes_structure(tmp_path):
    p = _save_pickled(tmp_path, HEADER)
    info = probe_pickled_npy(p)
    assert info["kind"] == "object_array" and info["container"] == "dict"
    assert "dim_px" in info["top_level_keys"]
    assert info["metadata_only"] is True

    combined = dict(HEADER, spectrum=np.linspace(0, 1, 151))
    p2 = _save_pickled(tmp_path, combined, "c.npy")
    info2 = probe_pickled_npy(p2)
    assert info2["metadata_only"] is False and "spectrum" in info2["data_members"]


def test_meta_probe_file_unblinded(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator_tools import _probe_file
    p = _save_pickled(tmp_path, HEADER)
    info = _probe_file(p)
    assert info["kind"] == "object_array"
    assert info.get("metadata_only") is True
    # plain arrays are untouched by the change
    q = tmp_path / "arr.npy"
    np.save(q, np.zeros((3, 4)))
    assert _probe_file(q)["kind"] == "array"


def test_pyobj_equal_handles_dicts_with_arrays():
    a = {"x": np.arange(5.0), "meta": {"k": [1, 2]}}
    b = {"x": np.arange(5.0) + 1e-12, "meta": {"k": [1, 2]}}
    c = {"x": np.arange(5.0) * 2, "meta": {"k": [1, 2]}}
    assert _pyobj_equal(a, b)
    assert not _pyobj_equal(a, c)
