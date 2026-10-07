"""An identification overlay is drawn where its score was computed (#775).

- Every scorer returns the registration it scored under; ``register_overlay``
  moves the FULL simulated pattern by it, so the overlay lands on the data
  peaks, and its legend label states the registration.
- A fitted scale beyond the band for the reference cell's kind (1 %
  experimental or unknown, 3 % computed) is a warning, printed as a marker the
  curve agent lifts into the run's caveats with the script's own warnings.
- A measured (COD) cell wins the dedup over a DFT-relaxed (MP) one and keeps
  the other source's id and stability.
- ``plot_match_overlay`` draws labelled axes, one line for one phase.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd.overlay import (
    apply_registration, lattice_scale_warnings, plot_match_overlay, register_overlay, registration_record)
from scilink.skills.structure_matching.xrd.score_match_fast import _broaden_peaks, score_xrd_match_fast
from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust

# a generic stick pattern, simulated from a reference cell 2.2 % larger than
# the sample's: every simulated peak sits at a lower angle than measured
SIM = np.array([22.10, 31.40, 38.70, 44.90, 50.30, 55.80, 61.20, 66.70, 73.40])
SIM_I = np.array([100, 20, 35, 22, 21, 14, 6, 7, 10.0])
TRUE = {"lattice_scale": 1.022, "two_theta_scale": 1.0, "zero_shift": 0.05}
X = np.arange(20.0, 80.0, 0.02)
EXP_POS = apply_registration(SIM, TRUE)
Y = _broaden_peaks(X, EXP_POS, SIM_I, 0.15) + 1.0
SIMULATED = {"two_theta": SIM.tolist(), "intensities": SIM_I.tolist()}


def _lands(overlay, tol=0.2):
    return float(np.max(np.abs(np.asarray(overlay["two_theta"]) - EXP_POS))) <= tol


@pytest.mark.parametrize("algorithm", ["hanawalt", "mip"])
def test_the_robust_scorers_overlay_lands_on_the_data(algorithm):
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y, algorithm=algorithm)
    assert res["verdict"] == "accept"
    assert float(np.max(np.abs(SIM - EXP_POS))) > 1.0                       # the raw overlay does not
    ov = register_overlay(SIMULATED, res, formula="AB2", emit_warnings=False)
    assert _lands(ov) and ov["intensities"] == SIM_I.tolist()
    # exactly where the scorer put every peak it matched
    drawn = np.asarray(ov["two_theta"])
    assert res["matched_peaks"] and all(np.min(np.abs(drawn - m["sim_pos"])) < 1e-6 for m in res["matched_peaks"])
    assert ov["label"].startswith("Simulated AB2 (match overlay; ") and "scale ×1.0" in ov["label"]


def test_the_fast_scorers_overlay_lands_on_the_data_and_its_shift_has_the_right_sign():
    res = score_xrd_match_fast(X, Y, SIM, SIM_I, scale_search=(0.96, 1.04, 0.002))
    assert _lands(register_overlay(SIMULATED, res, emit_warnings=False))
    for d in (0.3, -0.3):
        y = _broaden_peaks(X, SIM + d, SIM_I, 0.15) + 1.0
        ov = register_overlay(SIMULATED, score_xrd_match_fast(X, y, SIM, SIM_I, scale_search=None),
                              emit_warnings=False)
        assert np.allclose(ov["two_theta"], SIM + d, atol=0.03)


def test_the_multiphase_scorers_overlay_lands_per_phase():
    pytest.importorskip("pulp")
    from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_multiphase
    from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks
    pk = extract_peaks(X, Y, max_peaks=20)
    res = score_xrd_match_multiphase(
        {"positions": pk["positions"], "intensities": pk["intensities"]},
        [{"id": "mp-1", "formula": "AB2", "source": "mp", "sim_two_theta": SIM.tolist(),
          "sim_intensity": SIM_I.tolist()}])
    (phase,) = res["active_phases"]
    assert phase["registration"]["reference_cell"] == "computed"
    ov = register_overlay(SIMULATED, res, phase="mp-1", emit_warnings=False)
    assert _lands(ov) and ov["matched"] and ov["label"].startswith("Simulated AB2 (match overlay")
    # by its candidate id only, even with one active phase: an index or no
    # phase raises on EVERY frame, so the anchor's ladder fixes it before a lock
    for bad in (None, 0):
        with pytest.raises(ValueError, match="candidate id"):
            register_overlay(SIMULATED, res, phase=bad)


def test_polymorphs_sharing_a_formula_are_named_by_id_never_the_first_hit():
    """Two active phases with one formula, fitted at different scales: each
    id draws its own registration; the shared formula is refused."""
    pytest.importorskip("pulp")
    from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_multiphase
    from scilink.skills.structure_matching.xrd.extract_peaks import extract_peaks
    sim_b = np.array([27.30, 35.90, 43.10, 57.40, 64.80, 70.90])
    sim_b_i = np.array([100, 40, 30, 25, 15, 12.0])
    pos_a, pos_b = apply_registration(SIM, {"lattice_scale": 1.02}), apply_registration(sim_b, {"lattice_scale": 0.99})
    y = _broaden_peaks(X, pos_a, SIM_I, 0.15) + _broaden_peaks(X, pos_b, sim_b_i, 0.15) + 1.0
    pk = extract_peaks(X, y, max_peaks=30)
    res = score_xrd_match_multiphase(
        {"positions": pk["positions"], "intensities": pk["intensities"]},
        [{"id": "a", "formula": "AB2", "sim_two_theta": SIM.tolist(), "sim_intensity": SIM_I.tolist()},
         {"id": "b", "formula": "AB2", "sim_two_theta": sim_b.tolist(), "sim_intensity": sim_b_i.tolist()}])
    assert {p["id"] for p in res["active_phases"]} == {"a", "b"}
    ov_b = register_overlay({"two_theta": sim_b.tolist(), "intensities": sim_b_i.tolist()}, res, phase="b",
                            emit_warnings=False)
    assert np.max(np.abs(np.asarray(ov_b["two_theta"]) - pos_b)) < 0.12 and "×0.99" in ov_b["label"]
    # a formula two polymorphs share names neither of them
    assert register_overlay(SIMULATED, res, phase="AB2", emit_warnings=False)["matched"] is False
    with pytest.raises(ValueError):
        register_overlay(SIMULATED, res, phase=None, emit_warnings=False)


def test_a_locked_scripts_phase_means_the_same_on_every_frame(capsys):
    """A locked recipe meets frames its anchor did not have: the phase it
    draws by id is drawn on every frame, matched or not, and nothing raises
    because the frame's phases changed."""
    reg_a = registration_record(lattice_scale=1.03, two_theta_scale=1.001, zero_shift=0.04, reference_cell="cod")
    reg_b = registration_record(lattice_scale=0.994, two_theta_scale=1.001, zero_shift=0.04)
    shared = registration_record(two_theta_scale=1.001, zero_shift=0.04)
    pa = {"id": "a", "formula": "AB2", "registration": reg_a, "warnings": lattice_scale_warnings(reg_a, "AB2")}
    pb = {"id": "b", "formula": "CD", "registration": reg_b, "warnings": []}
    frames = {"a and b": {"active_phases": [pa, pb], "registration": shared},
              "a gone": {"active_phases": [pb], "registration": shared},
              "nothing matched": {"active_phases": [], "algorithm": "mip_multiphase"}}
    for name, res in frames.items():
        for pid, phase in (("a", pa), ("b", pb)):
            ov = register_overlay(SIMULATED, res, phase=pid, formula=phase["formula"])
            if phase in res["active_phases"]:
                assert ov["matched"] and ov["registration"] == phase["registration"], name
                assert f"Simulated {phase['formula']} (match overlay" in ov["label"]
            else:
                # on the shared terms, never another phase's scale or name
                assert ov["matched"] is False and ov["warnings"] == [], name
                assert ov["label"] == f"Simulated {phase['formula']} (not matched in this frame)"
                assert ov["registration"]["lattice_scale"] == 1.0
                assert np.allclose(ov["two_theta"], apply_registration(SIM, res.get("registration")))
    # only the matched, out-of-band phase printed a caveat, once per frame it was matched
    markers = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("TOOL_WARNINGS_JSON:")]
    assert len(markers) == 1 and "×1.030" in json.loads(markers[0].split(":", 1)[1])[0]
    # what does not mean the same on every frame raises on every frame
    for res in frames.values():
        for bad in (None, 0, 1.0):
            with pytest.raises(ValueError, match="candidate id"):
                register_overlay(SIMULATED, res, phase=bad, emit_warnings=False)


def test_a_term_that_rounds_to_nothing_is_not_in_the_label():
    reg = registration_record(lattice_scale=1.0002, zero_shift=-0.003)
    ov = register_overlay(SIMULATED, {"registration": reg}, formula="AB2", emit_warnings=False)
    assert ov["label"] == "Simulated AB2 (match overlay)"


def test_a_large_scale_is_a_caveat_by_the_kind_of_reference_cell():
    exp = registration_record(lattice_scale=1.022, reference_cell="cod")
    assert lattice_scale_warnings(exp) and "experimental reference cell" in lattice_scale_warnings(exp)[0]
    assert not lattice_scale_warnings(registration_record(lattice_scale=1.003, reference_cell="cod"))
    assert not lattice_scale_warnings(registration_record(lattice_scale=1.022, reference_cell="mp"))
    # a user's local CIF can be measured or relaxed: unknown, and said so
    local = lattice_scale_warnings(registration_record(lattice_scale=1.022, reference_cell="local"))
    assert local and "unknown origin" in local[0]
    assert lattice_scale_warnings(registration_record(lattice_scale=1.035, reference_cell="computed"))
    unknown = lattice_scale_warnings(registration_record(lattice_scale=0.978))
    assert unknown and "-2.2 %" in unknown[0] and "computed cell is typically" in unknown[0]
    assert lattice_scale_warnings(registration_record(two_theta_scale=1.02, zero_shift=0.1))   # a linear stand-in
    # the scorers carry it, and the reference kind decides it
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y, reference_cell="cod")
    assert res["warnings"] and res["registration"]["reference_cell"] == "experimental"
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y, reference_cell="mp")
    assert res["warnings"] == []
    near = apply_registration(SIM, {"lattice_scale": 1.003})
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X,
                                 exp_intensity=_broaden_peaks(X, near, SIM_I, 0.15) + 1.0, reference_cell="cod")
    assert res["warnings"] == []


def test_the_plotted_matchs_caveat_reaches_the_runs_result(tmp_path, monkeypatch, capsys):
    """register_overlay prints the marker; the curve agent lifts it, with the
    script's own ``warnings``, into the unit's caveats. A script that reports
    none gets no caveats, as before."""
    from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y, reference_cell="cod")
    register_overlay(SIMULATED, res, formula="AB2")
    marker = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("TOOL_WARNINGS_JSON:")]
    assert len(marker) == 1 and json.loads(marker[0].split(":", 1)[1]) == res["warnings"]
    assert "AB2" not in marker[0]                       # worded as the scorer worded it

    def run(stdout_extra, fit_extra=None):
        out = {"model_type": "match", "parameters": {"AB2": {"figure_of_merit": 0.9}},
               "fit_quality": {"figure_of_merit": 0.9}, **(fit_extra or {})}
        run = {"status": "success", "visualization_path": "viz.png", "visualization_bytes": b"", "exec": {},
               "stdout": "\n".join(stdout_extra + ["FIT_RESULTS_JSON:" + json.dumps(out)])}
        monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: run)
        c = cc.UnifiedSeriesProcessingController.__new__(cc.UnifiedSeriesProcessingController)
        c.logger = logging.getLogger("t775"); c.output_dir = Path(tmp_path); c.executor = object()
        c._extract_extra_operands = lambda state, p: None
        c._extra_operand_block = lambda state: ""
        c._compute_statistics = lambda cd: {"n_points": X.size, "x_range": [20.0, 80.0]}
        c._should_escalate_timeout_model = lambda *a, **k: False
        c._correct_script = lambda state, script, err: (script, "x")
        return c._fit_single_spectrum({}, np.column_stack([X, Y]), "s.csv", "s", 0, base_script="script")

    # the script also copies the scorer's warning: the run holds it once
    scored = res["warnings"]
    res = run(marker, {"warnings": ["The background was fitted on a narrow window."] + scored})
    assert res["success"]
    assert res["caveats"] == ["The background was fitted on a narrow window."] + scored
    assert "match needed a fitted lattice scale of ×1.0" in res["caveats"][1]
    assert "caveats" not in run([])


def test_a_measured_cell_wins_the_dedup_whatever_each_backend_writes():
    """Each backend's own notation: COD a spaced Hill formula, a spaced symbol
    with its setting and a string number; MP the compact forms and an int.
    One structure, one entry: the measured cell, carrying the computed one's
    id, stability and rank, at the computed one's place in the list."""
    from scilink.skills.structure_matching._backends import StructureCandidate
    from scilink.skills.structure_matching.xrd.search_structures import _dedupe
    for order in ((0, 1, 2), (1, 0, 2)):
        cands = [StructureCandidate(id="mp-1", source="mp", formula="ZrO2", space_group="P2_1/c",
                                    metadata={"energy_above_hull": 0.0, "spacegroup_number": 14}, rank_score=1.0),
                 StructureCandidate(id="cod-1", source="cod", formula="O2 Zr", space_group="P 1 21/c 1",
                                    metadata={"spacegroup_number": "14"}, rank_score=0.4),
                 StructureCandidate(id="mp-2", source="mp", formula="ZrO2", space_group="Fm-3m",
                                    metadata={"spacegroup_number": 225}, rank_score=0.9)]
        kept = _dedupe([cands[i] for i in order])
        assert [c.id for c in kept] == ["cod-1", "mp-2"]
        assert kept[0].rank_score == 1.0 and kept[0].metadata["energy_above_hull"] == 0.0
        assert kept[0].metadata["also_in"] == [{"source": "mp", "id": "mp-1"}]


def test_the_overlay_figure_has_labelled_axes_and_one_line_per_phase():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y)
    ov = register_overlay(SIMULATED, res, formula="AB2", emit_warnings=False)
    fig, (ax, dax) = plt.subplots(2, 1, sharex=True)
    out = plot_match_overlay(ax, X, Y, [ov], difference_ax=dax)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Data", ov["label"]]                                       # no sum for one phase
    assert ax.get_xlabel() == "2θ (°)" and dax.get_xlabel() == "2θ (°)"
    assert any(t.get_text() for t in ax.get_xticklabels()) and any(t.get_text() for t in dax.get_xticklabels())
    # the drawn overlay peaks where the data does
    i = int(np.argmax(out["overlay_sum"]))
    assert abs(X[i] - X[int(np.argmax(Y))]) < 0.1
    # drawn on the data's baseline, not on zero: between peaks the difference is ~0
    resid = Y - np.asarray(out["background"]) - np.asarray(out["overlay_sum"])
    assert abs(float(np.median(resid))) < 0.05 and abs(float(np.median(out["background"])) - 1.0) < 0.05
    # the y axis is log only when the PEAKS span more than ~1.5 decades, not a
    # low background floor (a background-subtracted pattern sits near zero)
    assert out["log_scale"] is False and out["y_scale"] == "linear"
    wide_i = np.array([100, 0.5, 35, 22, 21, 14, 6, 7, 10.0])
    wide = _broaden_peaks(X, EXP_POS, wide_i, 0.15) + 1.0
    fig2, ax2 = plt.subplots()
    assert plot_match_overlay(ax2, X, wide, [ov])["y_scale"] == "log"
    plt.close(fig2)
    # background-subtracted (reaching zero and below): symlog, not a minority
    # phase hidden on a linear axis
    rng = np.random.default_rng(0)
    sub = _broaden_peaks(X, EXP_POS, wide_i, 0.15) + rng.normal(0, 0.02, X.size)
    fig2, ax2 = plt.subplots()
    assert plot_match_overlay(ax2, X, sub, [ov], background=None)["y_scale"] == "symlog"
    assert ax2.get_yscale() == "symlog"
    plt.close(fig2)
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_match_overlay(ax, X, Y, [{"two_theta": SIM.tolist(), "label": "x"}])


@pytest.mark.parametrize("step", [0.02, 0.005])
@pytest.mark.parametrize("sigma", [0.02, 0.2])
def test_noise_is_not_read_as_the_weakest_peak(step, sigma):
    """Peaks spanning under a decade stay linear on noisy data (a noise bump,
    or one on a peak's flank, is not a peak); two decades above the noise go log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(20.0, 80.0, step)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        narrow = _broaden_peaks(x, EXP_POS, SIM_I.clip(min=14), 0.15) + 1.0 + rng.normal(0, sigma, x.size)
        broad = _broaden_peaks(x, EXP_POS, np.array([100, 1.5, 35, 22, 21, 14, 6, 7, 10.0]), 0.15) + 20.0 \
            + rng.normal(0, sigma, x.size)
        fig, ax = plt.subplots()
        assert plot_match_overlay(ax, x, narrow, [])["y_scale"] == "linear"
        assert plot_match_overlay(ax, x, broad, [])["y_scale"] == "log"
        plt.close(fig)


def test_a_phase_not_matched_in_this_frame_is_drawn_apart_from_the_sum():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y)
    ov = register_overlay(SIMULATED, res, formula="AB2", emit_warnings=False)
    gone = register_overlay(SIMULATED, {"active_phases": [], "registration": registration_record()},
                            phase="b", formula="CD")
    fig, ax = plt.subplots()
    alone = plot_match_overlay(ax, X, Y, [ov])
    fig2, ax2 = plt.subplots()
    both = plot_match_overlay(ax2, X, Y, [ov, gone])
    labels = [t.get_text() for t in ax2.get_legend().get_texts()]
    assert labels == ["Data", ov["label"], "Simulated CD (not matched in this frame)"]     # no sum
    assert np.allclose(both["overlay_sum"], alone["overlay_sum"])
    plt.close(fig); plt.close(fig2)


def test_the_tools_are_in_the_skills_inventory():
    from scilink.skills._shared._registry import _collect_specs_from_module
    from scilink.skills.structure_matching.xrd import overlay
    assert {s.name for s in _collect_specs_from_module(overlay)} == {"register_overlay", "plot_match_overlay"}
