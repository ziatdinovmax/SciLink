"""Tests for the script bank (issue #346, step 1: write hook + fingerprints).

Covers:
  - the deterministic data fingerprints (curve peak census, image
    texture/periodicity, hyperspectral band summary) on synthetic data;
  - bank CRUD with script-hash dedup and usage-stat accumulation;
  - the enable gate (follows the persistent-memory master switch;
    ``SCILINK_SCRIPT_BANK`` overrides in both directions);
  - the three agents' write hooks (approved-only, once-per-run dedup,
    failure isolation, inert when disabled).

No LLM calls anywhere — the write path is deterministic by design.
"""

import logging
import types
from pathlib import Path

import numpy as np
import pytest

from scilink.skills._shared import _script_bank as sb


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    """Isolate the store and enable the master switch (bank follows it)."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    monkeypatch.delenv("SCILINK_SCRIPT_BANK", raising=False)


# ──────────────────────────────────────────────────────────────
# Fingerprints
# ──────────────────────────────────────────────────────────────

class TestCurveFingerprint:
    def _three_gaussians(self, noise=0.05, seed=0):
        rs = np.random.RandomState(seed)
        x = np.linspace(0, 100, 2000)
        y = (0.02 * x
             + 5.0 * np.exp(-(x - 20) ** 2 / 4)
             + 3.0 * np.exp(-(x - 50) ** 2 / 9)
             + 1.5 * np.exp(-(x - 80) ** 2 / 2)
             + rs.normal(0, noise, x.size))
        return x, y

    def test_peak_census_exact(self):
        x, y = self._three_gaussians()
        fp = sb.curve_fingerprint(x, y, x_units="eV")
        assert fp["kind"] == "curve"
        assert fp["n_points"] == 2000
        assert fp["x_units"] == "eV"
        assert fp["x_range"] == [0.0, 100.0]
        assert fp["peaks"]["count"] == 3
        positions = sorted(p["position"] for p in fp["peaks"]["top"])
        for found, true in zip(positions, [20, 50, 80]):
            assert abs(found - true) < 1.0
        assert fp["snr"] > 20
        assert fp["baseline"]["drift"] > 0.1  # the ramp

    def test_peak_census_noisy(self):
        x, y = self._three_gaussians(noise=0.3)
        assert sb.curve_fingerprint(x, y)["peaks"]["count"] == 3

    def test_weak_shoulder_recovered(self):
        rs = np.random.RandomState(0)
        x = np.linspace(0, 100, 2000)
        y = (5.0 * np.exp(-(x - 50) ** 2 / 9)
             + 0.6 * np.exp(-(x - 58) ** 2 / 4)
             + rs.normal(0, 0.05, x.size))
        assert sb.curve_fingerprint(x, y)["peaks"]["count"] == 2

    def test_degenerate_inputs(self):
        assert sb.curve_fingerprint([], [])["n_points"] == 0
        flat = sb.curve_fingerprint(np.arange(100), np.ones(100))
        assert flat["peaks"]["count"] == 0
        with_nans = sb.curve_fingerprint(
            np.arange(50, dtype=float), np.full(50, np.nan))
        assert with_nans["peaks"]["count"] == 0  # no crash


class TestImageFingerprint:
    def test_lattice_vs_noise_periodicity(self):
        rs = np.random.RandomState(0)
        xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        lattice = np.sin(xx * 0.5) * np.sin(yy * 0.5) + rs.normal(0, 0.1, (256, 256))
        noise = rs.normal(0, 1, (256, 256))
        fp_lat = sb.image_fingerprint(lattice, pixel_size_nm=0.1)
        fp_noise = sb.image_fingerprint(noise)
        assert fp_lat["shape"] == [256, 256]
        assert fp_lat["pixel_size_nm"] == 0.1
        assert fp_lat["fft_periodicity"] > 50
        assert fp_noise["fft_periodicity"] < 5

    def test_rgb_collapsed_and_degenerate(self):
        rgb = np.zeros((32, 32, 3))
        assert sb.image_fingerprint(rgb)["shape"] == [32, 32]
        assert sb.image_fingerprint(np.zeros((0, 0)))["shape"] == [0, 0]


class TestHyperspectralFingerprint:
    def test_edge_cube(self):
        rs = np.random.RandomState(0)
        axis = np.linspace(400, 600, 300)
        spec = (1 / (1 + np.exp(-(axis - 500) / 3))
                + 0.3 * np.exp(-(axis - 550) ** 2 / 20))
        cube = spec[None, None, :] * (1 + rs.normal(0, 0.05, (8, 8, 300)))
        fp = sb.hyperspectral_fingerprint(cube, axis, "eV")
        assert fp["shape"] == [8, 8, 300]
        assert fp["axis"] == {"units": "eV", "start": 400.0, "end": 600.0,
                              "n_channels": 300}
        assert len(fp["band_means"]) == 16
        assert fp["band_means"][0] < 0.1      # pre-edge ~0
        assert fp["band_means"][-1] > 0.5     # post-edge high
        assert fp["peaks"]["count"] >= 1

    def test_channels_fallback(self):
        cube = np.ones((2, 2, 64))
        fp = sb.hyperspectral_fingerprint(cube)
        assert fp["axis"]["units"] == "channels"
        assert fp["axis"]["end"] == 63.0


# ──────────────────────────────────────────────────────────────
# CRUD + dedup
# ──────────────────────────────────────────────────────────────

def _record(script="import numpy as np\nprint(1)\n", session="run_A", r2=0.98):
    return {
        "working_script": script,
        "measurement_context": {"technique": "XRD"},
        "data_fingerprint": {"kind": "curve", "n_points": 100},
        "outcome": {"model_type": "pseudo_voigt",
                    "metric": {"name": "r_squared", "value": r2}},
        "provenance": {"session": session},
    }


class TestBankCRUD:
    def test_create_then_dedup_update(self):
        r1 = sb.add_record("curve_fitting", _record(session="run_A", r2=0.98))
        assert r1["action"] == "created"
        # Same script up to trailing whitespace → update, not duplicate.
        r2 = sb.add_record("curve_fitting", _record(
            script="import numpy as np  \nprint(1)\n", session="run_B", r2=0.99))
        assert r2 == {"id": r1["id"], "action": "updated"}

        recs = sb.list_records("curve_fitting")
        assert len(recs) == 1
        rec = recs[0]
        assert rec["stats"]["n_successes"] == 2
        assert rec["sessions"] == ["run_A", "run_B"]
        assert rec["outcome"]["best_metric"]["value"] == 0.99
        assert rec["outcome"]["metric"]["value"] == 0.98  # original preserved

    def test_distinct_script_new_record(self):
        sb.add_record("curve_fitting", _record())
        r = sb.add_record("curve_fitting", _record(script="totally different"))
        assert r["action"] == "created"
        assert len(sb.list_records("curve_fitting")) == 2

    def test_get_remove_and_domain_filter(self):
        rid = sb.add_record("curve_fitting", _record())["id"]
        sb.add_record("image_analysis", _record(script="img script"))
        assert sb.get_record("curve_fitting", rid)["id"] == rid
        assert len(sb.list_records()) == 2
        assert len(sb.list_records("image_analysis")) == 1
        assert sb.remove_records("curve_fitting", [rid]) == 1
        assert sb.list_records("curve_fitting") == []

    def test_no_script_skipped(self):
        rec = _record()
        rec["working_script"] = "   "
        assert sb.add_record("curve_fitting", rec)["action"] == "skipped_no_script"
        assert sb.list_records("curve_fitting") == []


class TestBankEnabled:
    def test_follows_memory_switch(self, monkeypatch):
        assert sb.bank_enabled() is True          # SCILINK_MEMORY=1 (fixture)
        monkeypatch.setenv("SCILINK_MEMORY", "0")
        assert sb.bank_enabled() is False

    def test_flag_overrides_both_ways(self, monkeypatch):
        monkeypatch.setenv("SCILINK_SCRIPT_BANK", "0")
        assert sb.bank_enabled() is False         # off even with memory on
        monkeypatch.setenv("SCILINK_MEMORY", "0")
        monkeypatch.setenv("SCILINK_SCRIPT_BANK", "1")
        assert sb.bank_enabled() is True          # on even with memory off


# ──────────────────────────────────────────────────────────────
# Agent write hooks (unbound calls on fake agents — no LLM anywhere)
# ──────────────────────────────────────────────────────────────

def _fake_agent(home):
    a = types.SimpleNamespace()
    a.logger = logging.getLogger("banktest")
    a.output_dir = Path(home) / "sess"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    return a


def _curve_state(n_results=2, shared_script=True):
    rs = np.random.RandomState(0)
    x = np.linspace(0, 100, 500)
    y = 5 * np.exp(-(x - 50) ** 2 / 9) + rs.normal(0, 0.05, x.size)
    stack = np.stack([np.vstack([x, y])] * max(1, n_results))
    results = []
    for i in range(n_results):
        script = ("import numpy as np\n# SHARED\n" if shared_script
                  else f"import numpy as np\n# S{i}\n")
        results.append({
            "index": i, "name": f"s{i}", "success": True,
            "model_type": "1 Gaussian",
            "fit_quality": {"r_squared": 0.99},
            "script": script,
            "quality_history": {"approved": True, "verification_iterations": []},
        })
    return {
        "locked_fitting_config": {"physical_model": "1 Gaussian",
                                  "analysis_approach": "single peak fit"},
        "skills_loaded": [],
        "system_info": {"technique": "PL", "x_units": "nm"},
        "data_path": "/data/spectrum.csv",
        "spectrum_stack": stack,
        "series_results": results,
    }


class TestCurveBankHook:
    def _run(self, tmp_path, state):
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        return CurveFittingAgent._maybe_bank_scripts(_fake_agent(tmp_path), state)

    def test_banks_approved_scripts_once_per_run(self, tmp_path):
        banked = self._run(tmp_path, _curve_state(n_results=2, shared_script=True))
        assert len(banked) == 1  # locked-recipe series: one distinct script
        rec = sb.list_records("curve_fitting")[0]
        assert rec["stats"]["n_successes"] == 1
        assert rec["data_fingerprint"]["peaks"]["count"] == 1
        assert rec["data_fingerprint"]["x_units"] == "nm"
        assert rec["measurement_context"]["technique"] == "PL"
        assert rec["outcome"]["metric"]["value"] == 0.99
        assert rec["provenance"]["data_file"] == "spectrum.csv"
        assert "# SHARED" in rec["working_script"]

    def test_distinct_scripts_bank_separately(self, tmp_path):
        banked = self._run(tmp_path, _curve_state(n_results=2, shared_script=False))
        assert len(banked) == 2

    def test_second_run_updates_stats(self, tmp_path):
        self._run(tmp_path, _curve_state())
        agent2 = _fake_agent(tmp_path)
        agent2.output_dir = Path(tmp_path) / "sess2"
        agent2.output_dir.mkdir(exist_ok=True)
        from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        CurveFittingAgent._maybe_bank_scripts(agent2, _curve_state())
        rec = sb.list_records("curve_fitting")[0]
        assert rec["stats"]["n_successes"] == 2
        assert len(rec["sessions"]) == 2

    def test_unapproved_and_failed_skipped(self, tmp_path):
        state = _curve_state(n_results=2, shared_script=False)
        state["series_results"][0]["quality_history"]["approved"] = False
        state["series_results"][1]["success"] = False
        assert self._run(tmp_path, state) == []
        assert sb.list_records("curve_fitting") == []

    def test_inert_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_MEMORY", "0")
        assert self._run(tmp_path, _curve_state()) == []
        assert not sb.bank_dir().exists()

    def test_failure_isolated(self, tmp_path):
        state = _curve_state()
        state["series_results"] = "not a list"  # forces an internal error
        assert self._run(tmp_path, state) == []


class TestImageBankHook:
    def test_banks_with_image_fingerprint(self, tmp_path):
        from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
        rs = np.random.RandomState(0)
        state = {
            "analysis_approach": "grain segmentation",
            "skills_loaded": [],
            "system_info": {"instrument": "SEM"},
            "image_paths": ["/data/frame.tif"],
            "image_stack": [rs.normal(0, 1, (64, 64))],
            "series_results": [{
                "index": 0, "name": "frame", "success": True,
                "analysis_type": "segmentation",
                "script": "import numpy as np\n# IMG\n",
                "quality_history": {"approved": True, "final_score": 0.9},
            }],
        }
        banked = ImageAnalysisAgent._maybe_bank_scripts(_fake_agent(tmp_path), state)
        assert len(banked) == 1
        rec = sb.list_records("image_analysis")[0]
        assert rec["data_fingerprint"]["kind"] == "image"
        assert rec["data_fingerprint"]["shape"] == [64, 64]
        assert rec["outcome"]["metric"] == {"name": "quality_score", "value": 0.9}
        assert rec["provenance"]["data_file"] == "frame.tif"


class TestHyperspectralBankHook:
    def test_banks_with_cube_fingerprint(self, tmp_path):
        from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
            HyperspectralAnalysisAgent,
        )
        rs = np.random.RandomState(0)
        cube = rs.random((4, 4, 128)) + np.linspace(0, 1, 128)
        agent = _fake_agent(tmp_path)
        agent._handle_system_info = lambda si: si or {}
        agent._load_hyperspectral_data = lambda p: cube
        records = [{
            "target": "thickness map",
            "required_outputs": ["map"],
            "script": "import numpy as np\n# HS\n",
            "quality_history": {"approved": True, "final_passed_fraction": 1.0},
        }, {
            "target": "rejected map",
            "script": "import numpy as np\n# BAD\n",
            "quality_history": {"approved": False},
        }]
        banked = HyperspectralAnalysisAgent._maybe_bank_scripts(
            agent, records, {"skills_loaded": []}, "/data/cube.npy",
            {"technique": "EELS",
             "energy_range": {"start": 400, "end": 600, "units": "eV"}},
        )
        assert len(banked) == 1
        rec = sb.list_records("hyperspectral")[0]
        assert rec["data_fingerprint"]["kind"] == "hyperspectral"
        assert rec["data_fingerprint"]["shape"] == [4, 4, 128]
        assert rec["outcome"]["metric"] == {"name": "passed_fraction", "value": 1.0}
        assert rec["technique_signals"]["analysis_target"] == "thickness map"
        assert rec["provenance"]["data_file"] == "cube.npy"

    def test_cube_load_failure_still_banks(self, tmp_path):
        from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
            HyperspectralAnalysisAgent,
        )
        agent = _fake_agent(tmp_path)
        agent._handle_system_info = lambda si: si or {}

        def _boom(_):
            raise RuntimeError("no such file")

        agent._load_hyperspectral_data = _boom
        records = [{
            "target": "map", "script": "import numpy as np\n# HS2\n",
            "quality_history": {"approved": True},
        }]
        banked = HyperspectralAnalysisAgent._maybe_bank_scripts(
            agent, records, {}, "/gone.npy", {})
        assert len(banked) == 1
        assert sb.list_records("hyperspectral")[0]["data_fingerprint"] is None
