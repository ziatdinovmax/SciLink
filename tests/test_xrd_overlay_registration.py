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

# an anatase-like stick pattern, simulated from a reference cell 2.2 % larger
# than the sample's: every simulated peak sits at a lower angle than measured
SIM = np.array([25.28, 37.80, 48.05, 53.89, 55.06, 62.69, 68.76, 70.31, 75.03])
SIM_I = np.array([100, 20, 35, 22, 21, 14, 6, 7, 10.0])
TRUE = {"lattice_scale": 1.022, "two_theta_scale": 1.0, "zero_shift": 0.05}
X = np.arange(20.0, 80.0, 0.02)
EXP_POS = apply_registration(SIM, TRUE)
Y = _broaden_peaks(X, EXP_POS, SIM_I, 0.15) + 1.0
SIMULATED = {"two_theta": SIM.tolist(), "intensities": SIM_I.tolist()}


def _lands(overlay, tol=0.12):
    return float(np.max(np.abs(np.asarray(overlay["two_theta"]) - EXP_POS))) <= tol


@pytest.mark.parametrize("algorithm", ["hanawalt", "mip"])
def test_the_robust_scorers_overlay_lands_on_the_data(algorithm):
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y, algorithm=algorithm)
    assert res["verdict"] == "accept"
    assert float(np.max(np.abs(SIM - EXP_POS))) > 1.0                       # the raw overlay does not
    ov = register_overlay(SIMULATED, res, formula="TiO2", emit_warnings=False)
    assert _lands(ov) and ov["intensities"] == SIM_I.tolist()
    assert ov["label"].startswith("Simulated TiO2 (match overlay; ") and "scale ×1.0" in ov["label"]


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
        [{"id": "mp-390", "formula": "TiO2", "source": "mp", "sim_two_theta": SIM.tolist(),
          "sim_intensity": SIM_I.tolist()}])
    (phase,) = res["active_phases"]
    assert phase["registration"]["reference_cell"] == "computed"
    assert _lands(register_overlay(SIMULATED, res, phase="TiO2", emit_warnings=False))
    assert register_overlay(SIMULATED, res, emit_warnings=False)["label"].startswith("Simulated TiO2")
    with pytest.raises(ValueError):
        register_overlay(SIMULATED, res, phase="ZnO")


def test_a_large_scale_is_a_caveat_by_the_kind_of_reference_cell():
    exp = registration_record(lattice_scale=1.022, reference_cell="cod")
    assert lattice_scale_warnings(exp) and "experimental reference cell" in lattice_scale_warnings(exp)[0]
    assert not lattice_scale_warnings(registration_record(lattice_scale=1.003, reference_cell="cod"))
    assert not lattice_scale_warnings(registration_record(lattice_scale=1.022, reference_cell="mp"))
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
    register_overlay(SIMULATED, res, formula="TiO2")
    marker = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("TOOL_WARNINGS_JSON:")]
    assert len(marker) == 1 and json.loads(marker[0].split(":", 1)[1]) == res["warnings"]

    def run(stdout_extra, fit_extra=None):
        out = {"model_type": "match", "parameters": {"TiO2": {"figure_of_merit": 0.9}},
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


def test_a_measured_cell_wins_the_dedup_and_keeps_the_computed_ones_stability():
    from scilink.skills.structure_matching._backends import StructureCandidate
    from scilink.skills.structure_matching.xrd.search_structures import _dedupe
    for order in ((0, 1), (1, 0)):
        cands = [StructureCandidate(id="mp-390", source="mp", formula="TiO2", space_group="I4_1/amd",
                                    metadata={"energy_above_hull": 0.0}, rank_score=1.0),
                 StructureCandidate(id="9015929", source="cod", formula="TiO2", space_group="I4_1/amd",
                                    rank_score=0.4)]
        (kept,) = _dedupe([cands[i] for i in order])
        assert kept.source == "cod" and kept.rank_score == 1.0
        assert kept.metadata["also_in"] == [{"source": "mp", "id": "mp-390"}]
        assert kept.metadata["energy_above_hull"] == 0.0


def test_the_overlay_figure_has_labelled_axes_and_one_line_per_phase():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = score_xrd_match_robust(SIM, SIM_I, exp_two_theta=X, exp_intensity=Y)
    ov = register_overlay(SIMULATED, res, formula="TiO2", emit_warnings=False)
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
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_match_overlay(ax, X, Y, [{"two_theta": SIM.tolist(), "label": "x"}])


def test_the_tools_are_in_the_skills_inventory():
    from scilink.skills._shared._registry import _collect_specs_from_module
    from scilink.skills.structure_matching.xrd import overlay
    assert {s.name for s in _collect_specs_from_module(overlay)} == {"register_overlay", "plot_match_overlay"}
