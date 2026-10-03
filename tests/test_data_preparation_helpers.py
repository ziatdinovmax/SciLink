"""Deterministic helpers of the mmzi_hologram_reconstruction skill on a
synthetic off-axis hologram stack with a known phase object."""
import json
import csv
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("skimage")

from scilink.skills.data_preparation.mmzi_hologram_reconstruction.reconstruct import (  # noqa: E402
    reconstruct_offaxis_hologram_stack, auto_pick_carrier, OffAxisReconstructor, producer_qc)
from scilink.skills.data_preparation.mmzi_hologram_reconstruction.derive import (  # noqa: E402
    derive_phase_products, trace_consistency_checks)

H, W = 400, 600
FX, FY = 0.08, -0.05          # carrier: +48 columns, -20 rows in the FFT
SCALE = 0.35
PH, PW = round(H * SCALE), round(W * SCALE)
CARRIER = (PH // 2 + FY * H, PW // 2 + FX * W)   # (50, 153)


def _bump(shape, amp):
    y, x = np.mgrid[: shape[0], : shape[1]]
    cy, cx = shape[0] * 0.5, shape[1] * 0.3
    return amp * np.exp(-((y - cy) ** 2 / (2 * (0.15 * shape[0]) ** 2) + (x - cx) ** 2 / (2 * (0.12 * shape[1]) ** 2)))


def _hologram(phase):
    y, x = np.mgrid[:H, :W]
    return (1000 + 800 * np.cos(2 * np.pi * (FX * x + FY * y) + phase)).astype(np.uint16)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    d = tmp_path_factory.mktemp("holo")
    amps = [0.0, 0.0, 1.5, 3.0, 3.0, 3.0]
    frames = np.stack([_hologram(_bump((H, W), a)) for a in amps])
    with h5py.File(d / "run.h5", "w") as f:
        f.create_dataset("data/raw", data=frames)
        f.create_dataset("data/reference", data=_hologram(np.zeros((H, W))))
    return d, amps


def test_auto_carrier_finds_sideband(stack):
    d, _ = stack
    with h5py.File(d / "run.h5") as f:
        ref = f["data/reference"][()]
    r, c = auto_pick_carrier(ref, (PH, PW), dc_exclusion_radius=20)
    assert abs(r - CARRIER[0]) <= 1 and abs(c - CARRIER[1]) <= 1


def test_reconstruct_recovers_phase_object(stack, tmp_path):
    d, amps = stack
    out = tmp_path / "out"
    contract = {"reconstruction_profile": {"name": "generic_offaxis", "apodization_window": "separable_hann",
                                           "mask_taper_fraction": 0.2, "intensity_offset_percentile": 0.5},
                "effective_reconstruction": {"processed_fft_shape_yx": [PH, PW],
                                             "carrier_fftshift_row_col": list(CARRIER),
                                             "mask_radius_fft_pixels": 30.0}, "contract_sha256": "abc"}
    res = reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(out), raw_dataset="data/raw",
                                             reference_dataset="data/reference", contract=contract,
                                             expected_contract_sha256="abc")
    assert res["status"] == "success" and res["n_frames"] == 6 and res["carrier_source"] == "contract"
    assert res["reference_sideband_contrast"] > 5
    ph = np.load(res["phase_output"])
    assert ph.shape == (6, PH, PW) and ph.dtype == np.float32
    truth = _bump((PH, PW), 1.0)
    core = truth > 0.5
    for t, a in enumerate(amps):
        rel = ph[t][core].mean() - ph[t][truth < 0.02].mean()      # object minus background
        assert abs(rel - a * truth[core].mean()) < 0.15, (t, rel)
    receipt = json.loads(Path(res["receipt"]).read_text())
    assert receipt["absolute_quantities_inferred"] is False and receipt["contract_sha256"] == "abc"
    with pytest.raises(ValueError):
        reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(out), raw_dataset="data/raw",
                                           reference_dataset="data/reference", contract=contract,
                                           expected_contract_sha256="wrong")


def test_qc_gate_passes_on_targets_and_fails_on_bad_targets(stack, tmp_path):
    d, _ = stack
    out = tmp_path / "out"
    kw = dict(raw_dataset="data/raw", reference_dataset="data/reference", contract=None,
              carrier=CARRIER, processed_scale=SCALE, mask_radius_px=30.0)
    first = reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(out / "a"), **kw)
    ph = np.load(first["phase_output"])
    np.save(out / "targets.npy", ph[[0, 3]])
    good = reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(out / "b"), validation_frames_npy=str(out / "targets.npy"),
                                              validation_frame_indices=[0, 3], **kw)
    assert good["producer_validation_passed"] is True and good["min_circular_coherence"] > 0.999
    np.save(out / "bad.npy", np.random.default_rng(0).uniform(-np.pi, np.pi, ph[[0, 3]].shape).astype(np.float32))
    bad = reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(out / "c"), validation_frames_npy=str(out / "bad.npy"),
                                             validation_frame_indices=[0, 3], **kw)
    assert bad["status"] == "qc_failed" and bad["producer_validation_passed"] is False


def test_producer_qc_is_offset_invariant():
    rng = np.random.default_rng(1)
    a = rng.uniform(-3, 3, (50, 60)).astype(np.float32)
    m = producer_qc(a, np.angle(np.exp(1j * (a + 1.7))))
    assert m["circular_coherence"] > 0.999 and abs(m["global_phase_offset_rad"] - 1.7) < 1e-3


def test_derive_products_recover_step_and_map(tmp_path):
    n, h, w = 60, 120, 200
    rng = np.random.default_rng(2)
    y, x = np.mgrid[:h, :w]
    shape_left = 6.0 * np.exp(-((x - 40) ** 2) / (2 * 25 ** 2))        # localized on the left
    piston = rng.uniform(-np.pi, np.pi, n)                                # random per-frame piston
    amp = np.array([0.0] * 28 + [0.5, 1.0] + [1.0] * 30)
    stack = np.stack([np.angle(np.exp(1j * (piston[t] + amp[t] * shape_left + 0.02 * rng.standard_normal((h, w)))))
                      for t in range(n)]).astype(np.float32)
    np.save(tmp_path / "run_wrapped_phase.npy", stack)
    with (tmp_path / "timeline.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["frame_index", "capture_elapsed_s", "magnet_state", "magnet_position_mm"])
        for t in range(n):
            st = "retracted" if t < 28 else ("moving" if t < 30 else "at_cuvette")
            wr.writerow([t, t * 1.0, st, "" if st == "moving" else (0 if st == "retracted" else 20)])
    res = derive_phase_products(str(tmp_path / "run_wrapped_phase.npy"), str(tmp_path / "out"),
                                timeline_csv=str(tmp_path / "timeline.csv"), transition_states=["moving"],
                                steady_window_frames=20, bin_factor=2, smoothing_sigma=1.0,
                                band_rows=[40, 80], roi_x_ranges={"left": [30, 50], "mid": [100, 120], "right": [160, 180]},
                                reference_roi="right")
    assert res["status"] == "success" and res["conditions"] == ["retracted", "moving", "at_cuvette"]
    steps = res["steady_state_steps_rad"]
    assert abs(steps["left"] - 6.0 * np.exp(0)) < 0.6 and abs(steps["mid"]) < 0.5, steps
    dmap = np.load(res["diff_map"])
    assert dmap.shape == (60, 100)
    assert abs(np.nanmean(dmap[20:40, 15:25]) - np.nanmean(dmap[20:40, 80:90]) - 6.0) < 0.6
    rows = list(csv.DictReader(open(res["roi_curve"])))
    assert len(rows) == n and "phase_left_minus_right_rad" in rows[0] and "template_amplitude_fraction" in rows[0]
    amp_tr = np.array([float(r["template_amplitude_fraction"]) for r in rows])
    assert abs(amp_tr[:20].mean()) < 0.1 and abs(amp_tr[-20:].mean() - 1.0) < 0.1
    assert rows[29]["transition"] == "1" and rows[0]["state_code"] == "1.0" and rows[-1]["state_code"] == "0.0"   # sorted: at_cuvette=0, retracted=1
    assert "state" not in rows[0] and all(float(v) == float(v) or True for v in rows[0].values())   # all numeric
    for r in rows:
        for v in r.values():
            float(v)
    side = json.loads(Path(res["roi_curve_sidecar"]).read_text())
    assert side["primary_columns"]["y"] == "phase_left_minus_right_rad" and "interpretation_limits" in side
    assert side["state_code_map"]["0.0"] == "at_cuvette" and side["state_code_map"]["1.0"] == "retracted"
    # top-level numeric condition fields for the sidecar-series extraction
    assert side["has_transition"] == 1 and side["first_state_code"] == 1.0 and side["last_state_code"] == 0.0
    assert side["n_conditions"] == 2 and abs(side["transition_start_s"] - 28.0) < 1e-6
    assert side["condition_sequence"] == "retracted -> moving -> at_cuvette"
    mside = json.loads(Path(res["diff_map_sidecar"]).read_text()); assert mside["has_transition"] == 1
    assert Path(res["diff_map_sidecar"]).is_file() and Path(res["quicklook"]).is_file()
    assert res["cross_checks"]["steady_map_discontinuity_fraction"] < 0.02


def test_generic_path_default_mask_is_capped_and_recovers_object(stack, tmp_path):
    d, amps = stack
    res = reconstruct_offaxis_hologram_stack(str(d / "run.h5"), str(tmp_path / "g"), raw_dataset="data/raw",
                                             reference_dataset="data/reference", contract=None, carrier="auto",
                                             processed_scale=SCALE)      # default mask 63 > 0.6 * 52
    receipt = json.loads(Path(res["receipt"]).read_text())
    assert receipt["mask_radius_capped_to"] is not None and receipt["mask_radius_capped_to"] < 40
    ph = np.load(res["phase_output"]); truth = _bump((PH, PW), 1.0); core = truth > 0.5
    rel = ph[-1][core].mean() - ph[-1][truth < 0.02].mean()
    assert abs(rel - amps[-1] * truth[core].mean()) < 0.2, rel
    assert res["reference_sideband_contrast"] > 5


def test_auto_dense_fringe_exclusion_and_guard(tmp_path):
    n, h, w = 30, 80, 160
    y, x = np.mgrid[:h, :w]
    steep = np.where(x < 30, 2.0 * (30 - x), 0.0)          # 2 rad/px near the left edge: aliased
    smooth = 0.02 * x
    for name, field in (("steep", steep), ("smooth", smooth)):
        stack = np.stack([np.angle(np.exp(1j * (field * (t >= 15)))) for t in range(n)]).astype(np.float32)
        np.save(tmp_path / f"{name}_wrapped_phase.npy", stack)
        res = derive_phase_products(str(tmp_path / f"{name}_wrapped_phase.npy"), str(tmp_path / name),
                                    steady_window_frames=10, bin_factor=1, smoothing_sigma=0,
                                    auto_exclude_dense_fringes=True)
        side = json.loads(Path(res["diff_map_sidecar"]).read_text())
        limits = " ".join(side["interpretation_limits"])
        if name == "steep":
            assert "excluded" in limits and np.isnan(np.load(res["diff_map"].replace(".npy", "_nan_outside_valid.npy"))[:, :25]).all()
        else:
            assert "excluded" not in limits
    with pytest.raises(ValueError):
        derive_phase_products(str(tmp_path / "smooth_wrapped_phase.npy"), str(tmp_path / "bad"),
                              steady_window_frames=10, bin_factor=1, smoothing_sigma=0, exclude_columns=[[0, 150]])


def test_state_codes_override_is_shared_across_runs(tmp_path):
    n, h, w = 30, 40, 60
    stack = np.zeros((n, h, w), dtype=np.float32); np.save(tmp_path / "ctl_wrapped_phase.npy", stack)
    with (tmp_path / "tl.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["frame_index", "capture_elapsed_s", "magnet_state"])
        for t in range(n): wr.writerow([t, t, "retracted"])
    res = derive_phase_products(str(tmp_path / "ctl_wrapped_phase.npy"), str(tmp_path / "o"), timeline_csv=str(tmp_path / "tl.csv"),
                                steady_window_frames=10, bin_factor=1, smoothing_sigma=0, state_codes={"retracted": 0, "at_cuvette": 1})
    side = json.loads(Path(res["roi_curve_sidecar"]).read_text())
    assert side["first_state_code"] == 0.0 and side["state_code_map"]["1.0"] == "at_cuvette"


def _transition_stack(kind, seed, n=40, h=120, w=200, noise=0.25):
    """Piston-randomised wrapped stack: a +8 rad rise across x = 100 switched on
    halfway. ``null``: a y-gradient localized at that rise spans 3pi across the
    trace band, so the band-mean phasor of those columns passes through zero;
    ``gap``: a decorrelated blob covers the band rows at the rise (columns gated
    out of the band, valid above and below), with > pi of change across it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:h, :w].astype(float)
    shape = 8.0 / (1 + np.exp(-(x - 100) / 6.0))
    if kind == "null":
        shape = shape + (3 * np.pi / 40) * np.exp(-((x - 100) ** 2) / (2 * 8 ** 2)) * (y - 60)
    piston = rng.uniform(-np.pi, np.pi, n)
    frames = []
    for t in range(n):
        ph = piston[t] + (t >= n // 2) * shape + noise * rng.standard_normal((h, w))
        if kind == "gap":
            blob = (np.abs(x - 100) <= 12) & (np.abs(y - 60) <= 25)
            ph = np.where(blob, rng.uniform(-np.pi, np.pi, (h, w)), ph)
        frames.append(np.angle(np.exp(1j * ph)))
    return np.stack(frames).astype(np.float32)


def _derive_transition(tmp_path, kind, seed, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / f"{kind}{seed}_wrapped_phase.npy", _transition_stack(kind, seed))
    return derive_phase_products(str(tmp_path / f"{kind}{seed}_wrapped_phase.npy"), str(tmp_path / f"{kind}{seed}"),
                                 steady_window_frames=15, bin_factor=1, smoothing_sigma=1.0, band_rows=[40, 80],
                                 roi_x_ranges={"left": [30, 50], "mid": [60, 75], "right": [150, 170]},
                                 reference_roi="right", **kw)


@pytest.mark.parametrize("kind,seed", [("null", 0), ("null", 1), ("gap", 0)])
def test_trace_survives_column_null_and_gated_gap(tmp_path, kind, seed):
    """The 1D column-profile unwrap slips by 2pi through a vanishing column or
    across a gated gap with > pi of change (on every frame, or only on some);
    the reported per-frame 2D-unwrap trace matches the map, and the QC flags
    fire on the column-profile trace only."""
    res = _derive_transition(tmp_path, kind, seed)
    x = res["cross_checks"]
    map_step = x["steady_map_roi_differences_rad"]["left"]
    assert abs(map_step + 8.0) < 0.3
    assert abs(res["steady_state_steps_rad"]["left"] - map_step) < 0.3
    rows = list(csv.DictReader(open(res["roi_curve"])))
    late = np.array([float(r["phase_left_minus_right_rad"]) for r in rows[20:]])
    assert np.all(np.abs(late + 8.0) < 1.0), late                      # no frame slipped
    assert res["trace_qc_passed"] is True and x["trace_qc_passed"] is True
    assert all(v["frame_consistency"]["passed"] and v["step_agreement"]["passed"] for v in x["trace_qc"].values())
    diag = x["diagnostic_column_profile_trace"]
    assert diag["qc_passed"] is False and diag["n_slipped_frames"]["left"] >= 1
    side = json.loads(Path(res["roi_curve_sidecar"]).read_text())
    assert side["cross_checks"]["trace_qc_passed"] is True


def test_trace_qc_passes_clean_transition(tmp_path):
    res = _derive_transition(tmp_path, "clean", 0)
    x = res["cross_checks"]
    assert res["trace_qc_passed"] is True
    assert x["diagnostic_column_profile_trace"]["qc_passed"] is True
    assert x["trace_qc"]["left"]["frame_consistency"]["n_frames_checked"] >= 35
    # the coherence bar reaches the per-frame check: an unreachable bar judges no frame
    res2 = _derive_transition(tmp_path / "bar", "clean", 0, trace_slip_min_template_coherence=1.01)
    assert res2["cross_checks"]["trace_qc"]["left"]["frame_consistency"]["n_frames_checked"] == 0


def test_trace_consistency_checks_flags_slips_and_step_disagreement():
    n = 30
    amp = np.r_[np.zeros(10), np.ones(20)]
    coh = np.full(n, 0.95)
    map_step = -21.0
    trace = amp * map_step + 0.1 * np.cos(np.arange(n))
    clean = trace_consistency_checks(trace, amp, coh, map_step, trace[10:].mean() - trace[:10].mean())
    assert clean["passed"] and clean["frame_consistency"]["n_slipped_frames"] == 0
    slipped = trace.copy(); slipped[[14, 22]] += 2 * np.pi
    q = trace_consistency_checks(slipped, amp, coh, map_step, -21.0, frame_labels=list(range(100, 100 + n)))
    assert not q["passed"] and not q["frame_consistency"]["passed"] and q["step_agreement"]["passed"]
    assert [s["frame"] for s in q["frame_consistency"]["slipped_frames"]] == [114, 122]
    assert all(s["cycles"] == 1 for s in q["frame_consistency"]["slipped_frames"])
    # a slip on an incoherent template frame is not judged
    coh2 = coh.copy(); coh2[14] = 0.2
    q2 = trace_consistency_checks(slipped, amp, coh2, map_step, -21.0)
    assert [s["frame"] for s in q2["frame_consistency"]["slipped_frames"]] == [22]
    # step agreement: -14 vs -21 fails (15 % of 21 = 3.15 rad); near-zero control steps pass on the floor
    s = trace_consistency_checks(trace, amp, coh, map_step, -14.0)
    assert not s["step_agreement"]["passed"] and abs(s["step_agreement"]["allowed_rad"] - 3.15) < 1e-9
    assert trace_consistency_checks(np.zeros(n), np.zeros(n), coh, 0.4, -0.5)["step_agreement"]["passed"]


def test_trace_qc_tolerance_knobs_change_the_verdict():
    n = 20
    amp = np.ones(n); coh = np.full(n, 0.95); map_step = 10.0
    trace = np.full(n, map_step); trace[5] += 2 * np.pi - 1.3                  # 1.3 rad short of a full cycle
    loose = trace_consistency_checks(trace, amp, coh, map_step, map_step, slip_tolerance_rad=1.5)
    strict = trace_consistency_checks(trace, amp, coh, map_step, map_step, slip_tolerance_rad=1.0)
    assert loose["frame_consistency"]["n_slipped_frames"] == 1 and strict["frame_consistency"]["n_slipped_frames"] == 0
    a = trace_consistency_checks(trace, amp, coh, -21.0, -18.5)                # diff 2.5 <= 0.15 x 21
    b = trace_consistency_checks(trace, amp, coh, -21.0, -18.5, step_rel_tol=0.1)
    c = trace_consistency_checks(trace, amp, coh, -21.0, -18.5, step_rel_tol=0.1, step_abs_floor_rad=3.0)
    assert a["step_agreement"]["passed"] and not b["step_agreement"]["passed"] and c["step_agreement"]["passed"]
