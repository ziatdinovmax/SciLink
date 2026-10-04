"""The admission estimate follows the largest unit, and sees nested data (#724).

``_branch_mem_estimate`` summed a NON-recursive glob of a folder x 6. A folder
of six hyperspectral cubes (1.55 GB) was estimated at 13.3 GB though an item
peaks at 4-8 GB however many cubes there are (units run one at a time), and a
raw-instrument folder whose ~10 GB of stacks sit in a subfolder was estimated
at the 0.5 GB floor. The estimate now walks recursively over data files,
takes the LARGEST unit (in memory: an .npy from its header, an HDF5 from its
largest dataset, uncompressed), multiplies by ``series_workers`` when replays
fan out, and estimates a raw container by its preparation.

  conda run -n scilink python -m pytest tests/test_mem_estimate.py -q
"""
import json

import numpy as np
import pytest

from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent import swarm


def _cubes(d, n, shape=(64, 64, 128)):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.save(d / f"cube_{i}.npy", np.zeros(shape, dtype=np.float32))
        (d / f"cube_{i}.json").write_text(json.dumps({"index": i}))   # a sidecar is not data
    return np.zeros(shape, dtype=np.float32).nbytes


def test_a_series_folder_is_estimated_by_its_largest_unit(tmp_path):
    one = _cubes(tmp_path / "six", 6, shape=(256, 256, 400))           # 105 MB each, 629 MB in all
    est = fo._branch_mem_estimate({"data_path": str(tmp_path / "six")})
    assert est == pytest.approx(one * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD)
    single = tmp_path / "one"
    _cubes(single, 1, shape=(256, 256, 400))
    assert fo._branch_mem_estimate({"data_path": str(single)}) == pytest.approx(est)   # one cube or six: the same
    # replays fanned out to two workers hold two units at once
    est2 = fo._branch_mem_estimate({"data_path": str(tmp_path / "six"), "series_workers": 2})
    assert est2 == pytest.approx(2 * one * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD)


def test_nested_data_is_counted_and_a_pattern_still_restricts(tmp_path):
    root = tmp_path / "bundle"
    one = _cubes(root / "hsi" / "deep", 2, shape=(256, 256, 400))
    (root / "README.md").write_text("x" * 10)
    est = fo._branch_mem_estimate({"data_path": str(root)})
    assert est == pytest.approx(one * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD)   # the old glob saw 10 bytes
    assert fo._branch_mem_estimate({"data_path": str(root), "pattern": "*.txt"}) == fo._BRANCH_MEM_FLOOR


def test_hdf5_is_estimated_uncompressed(tmp_path):
    h5py = pytest.importorskip("h5py")
    f = tmp_path / "stack.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("data/frames", data=np.zeros((100, 512, 512), dtype=np.float32),
                         compression="gzip")                            # compresses to almost nothing
    uncompressed = 100 * 512 * 512 * 4
    assert f.stat().st_size < uncompressed / 50
    assert fo._in_memory_bytes(f) == uncompressed
    est = fo._branch_mem_estimate({"data_path": str(f)})
    assert est == pytest.approx(uncompressed * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD)


def test_a_raw_instrument_folder_is_estimated_by_its_preparation(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "raw"
    (root / "stacks").mkdir(parents=True)
    (root / "reconstruction_manifest.json").write_text(json.dumps(
        {"generic_image_routing_permitted": False, "analysis_status": "ready_for_reconstruction"}))
    for i in range(3):        # declared, never written: the file holds only the header
        with h5py.File(root / "stacks" / f"run_{i}.h5", "w") as h:
            h.create_dataset("holograms", shape=(1000, 512, 512), dtype=np.uint16, compression="gzip")
    stack = 1000 * 512 * 512 * 2                                         # 524 MB per stack, uncompressed
    est = fo._branch_mem_estimate({"data_path": str(root)})
    assert est == pytest.approx(stack * fo._BRANCH_PREP_FACTOR)            # one stack at a time, not three
    assert est > fo._BRANCH_MEM_FLOOR                                       # the old top-level glob gave the floor


def test_the_capacity_plan_admits_together_what_fits(tmp_path):
    """Two six-cube series items: the old sum-based estimate put each at six
    cubes' worth; by their largest unit both fit on the machine together."""
    for name in ("a", "b"):
        _cubes(tmp_path / name, 6, shape=(256, 256, 400))
    items = [{"label": n, "mode": "analysis", "data_path": str(tmp_path / n), "task": "t"} for n in ("a", "b")]
    one = fo._branch_mem_estimate({"data_path": str(tmp_path / "a")})
    mem = {"total": 2 * one + fo._BRANCH_MEM_MARGIN + 1e9, "available": 2 * one + fo._BRANCH_MEM_MARGIN + 1e8}
    plan = swarm.capacity_plan(items, memory=mem)
    assert not plan["refused"] and plan["together"] is True
    assert plan["estimated_bytes"] == pytest.approx(2 * one)


def test_a_bundle_with_a_nested_raw_folder_is_estimated_by_unit_kind(tmp_path):
    """A study bundle (seen on a real one): analysis cubes beside a raw
    container in its own subfolder. The bundle root is not itself raw, so a
    rule read only at the root gave the raw stack the analysis factor (16.7
    GB on the real bundle); each file is now estimated by what will load it."""
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "bundle"
    one = _cubes(root / "hsi", 3, shape=(256, 256, 400))
    raw = root / "mmzi"
    (raw / "raw_hdf5").mkdir(parents=True)
    (raw / "reconstruction_manifest.json").write_text(json.dumps({"generic_image_routing_permitted": False}))
    with h5py.File(raw / "raw_hdf5" / "run.h5", "w") as h:
        h.create_dataset("holograms", shape=(1000, 512, 512), dtype=np.uint16, compression="gzip")
    stack = 1000 * 512 * 512 * 2
    est = fo._branch_mem_estimate({"data_path": str(root)})
    assert est == pytest.approx(max(one * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD,
                                    stack * fo._BRANCH_PREP_FACTOR))
    # the stack is not an analysis unit: the old root-only rule estimated it as one
    assert est < stack * fo._BRANCH_MEM_FACTOR + fo._BRANCH_POOL_OVERHEAD
