"""Tests for the technique-agnostic reconcile core (skills/_shared/_reconcile).

Uses NON-XRD vocabulary (NMR chemical shifts as feature positions, species as
labels) to prove the core is not XRD-bound — it operates on generic
(position, weight) features and (value, label) identifications, so any
spectroscopy with a profile-fitting and an identification pass reuses it.
"""

from __future__ import annotations

import pytest

from scilink.skills._shared._reconcile import (
    reconcile_series, render_reconcile_report)


def _nmr_titration(n=21):
    """Reactant peaks (1.2, 3.4 ppm) convert to product (1.5, 2.9 ppm) across
    a titration; identification names 'reactant' early, 'product' late with a
    null gap through the mid-point."""
    ffr, lfr = [], []
    for i in range(n):
        x = i / (n - 1)                          # extent of reaction
        pH = 3.0 + 8.0 * x                        # the series variable
        feats = [{"position": 1.2, "weight": 100 * (1 - x)},
                 {"position": 3.4, "weight": 60 * (1 - x)},
                 {"position": 1.5, "weight": 100 * x},
                 {"position": 2.9, "weight": 60 * x}]
        ffr.append({"value": pH, "features": feats})
        lab = None if 9 <= i <= 11 else ("reactant" if i < 10 else "product")
        lfr.append({"value": pH, "label": lab})
    return ffr, lfr


def test_generic_reconcile_labels_and_transitions():
    ffr, lfr = _nmr_titration()
    r = reconcile_series(ffr, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] == "product"
    # both transitions near the reaction midpoint (pH 7)
    assert abs(r["transition_profile"] - 7.0) < 1.5
    assert abs(r["transition_identification"] - 7.0) < 2.0
    assert r["agreement"]["verdict"] == "consistent"
    by_pos = {t["position"]: t for t in r["tracked_features"]}
    assert by_pos[1.2]["regime"] == "low" and by_pos[1.2]["label"] == "reactant"
    assert by_pos[1.5]["regime"] == "high" and by_pos[1.5]["label"] == "product"


def test_center_area_aliases_accepted():
    # a caller may pass profile-fit peaks as center/area instead of
    # position/weight — the core accepts the aliases.
    ffr, lfr = _nmr_titration()
    ffr2 = [{"value": f["value"],
             "peaks": [{"center": p["position"], "area": p["weight"]}
                       for p in f["features"]]} for f in ffr]
    r = reconcile_series(ffr2, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] == "product"


def test_unidentified_regime_stays_none():
    ffr, lfr = _nmr_titration()
    for i in range(10, len(lfr)):
        lfr[i]["label"] = None                   # product never named
    r = reconcile_series(ffr, lfr)
    assert r["low_regime_label"] == "reactant"
    assert r["high_regime_label"] is None
    assert r["transition_profile"] is not None
    assert r["agreement"]["verdict"] == "one_sided"


def test_auto_tol_is_not_technique_bound():
    # The default tolerance must NOT be a fixed technique-specific value (0.25
    # is fine for 2θ° but would merge NMR peaks ~0.15 ppm apart). Auto-scaling
    # from the peak spacing keeps close features separate regardless of x-units.
    n = 15
    ff = [{"value": i, "features": [
        {"position": 1.00, "weight": 100 * (1 - i / (n - 1))},
        {"position": 1.15, "weight": 80 * (1 - i / (n - 1))},   # 0.15 apart
        {"position": 2.00, "weight": 100 * i / (n - 1)}]} for i in range(n)]
    lf = [{"value": i, "label": "A" if i < 8 else "B"} for i in range(n)]
    r = reconcile_series(ff, lf)                     # tol=None -> auto
    pos = sorted(round(t["position"], 2) for t in r["tracked_features"])
    assert 1.0 in pos and 1.15 in pos               # kept separate
    assert r["tolerance_used"] < 0.15               # auto-scaled below the gap
    # the old fixed default WOULD have merged them
    r_fixed = reconcile_series(ff, lf, tol=0.25)
    assert len({round(t["position"], 2) for t in r_fixed["tracked_features"]}) < 3


def test_regime_and_crossover_knobs_exposed():
    ffr, lfr = _nmr_titration()
    # knobs must be accepted and affect the split/threshold without error
    r = reconcile_series(ffr, lfr, regime_window_frac=0.15, crossover_threshold=0.4)
    assert r["low_regime_label"] == "reactant"
    assert r["transition_profile"] is not None


def test_multiregime_model_is_a_named_seam():
    ffr, lfr = _nmr_titration()
    with pytest.raises(ValueError, match="single_crossover"):
        reconcile_series(ffr, lfr, transition_model="multi_regime")


def test_report_renders_with_honest_unidentified(tmp_path):
    ffr, lfr = _nmr_titration()
    # end-phase never named -> report must render and flag it honestly
    for i in range(10, len(lfr)):
        lfr[i]["label"] = None
    r = reconcile_series(ffr, lfr)
    out = tmp_path / "report.html"
    render_reconcile_report(r, str(out), series_variable="pH")
    html = out.read_text()
    assert "reactant" in html                       # named start phase
    assert "UNIDENTIFIED" in html                   # honest end-phase flag
    assert "one-sided" in html                       # agreement verdict surfaced
    assert "pH" in html                              # series-variable label used
    assert "Tracked features" in html


def test_report_renders_interpretation_section(tmp_path):
    # The reconcile report must be able to carry an LLM-authored synthesis, so
    # it is not the only one of the three reports without a narrative. When an
    # interpretation is supplied it appears as its own section, attributed; the
    # computed numbers stay present alongside it.
    ffr, lfr = _nmr_titration()
    r = reconcile_series(ffr, lfr)
    out = tmp_path / "report.html"
    synthesis = ("The reactant converts cleanly to the product near pH 7.\n\n"
                 "Both passes place the crossover within one unit, corroborating it.")
    render_reconcile_report(r, str(out), series_variable="pH",
                            interpretation=synthesis)
    html = out.read_text()
    assert "Interpretation" in html
    assert "converts cleanly to the product" in html          # narrative rendered
    assert "corroborating it" in html                          # second paragraph too
    assert "scientific synthesis by the analysis orchestrator" in html  # attributed
    assert "reactant" in html and "product" in html            # computed labels stay


def test_report_interpretation_optional(tmp_path):
    # Absent interpretation, the report still renders (the deterministic note is
    # the always-present fallback) and shows no Interpretation section.
    ffr, lfr = _nmr_titration()
    r = reconcile_series(ffr, lfr)
    out = tmp_path / "report.html"
    render_reconcile_report(r, str(out), series_variable="pH")
    html = out.read_text()
    assert "How to read this" in html                          # methodological note present
    assert "scientific synthesis by the analysis orchestrator" not in html


def test_report_embeds_figure(tmp_path):
    ffr, lfr = _nmr_titration()
    fig = tmp_path / "f.png"
    from scilink.skills._shared._reconcile import _plot_generic
    r = reconcile_series(ffr, lfr)
    _plot_generic(r, str(fig))
    r["figure"] = str(fig)
    out = tmp_path / "report.html"
    render_reconcile_report(r, str(out))
    assert "data:image/png;base64" in out.read_text()   # figure embedded, self-contained


def test_guards():
    with pytest.raises(ValueError):
        reconcile_series([{"value": 0, "features": []}], [{"value": 0, "label": None}])
