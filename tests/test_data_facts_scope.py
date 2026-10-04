"""The data-facts list is the strongest few field-mean features, not every feature (#739).

`_data_facts` lists at most ``max_peaks`` features per direction, found in the
field MEAN: a weaker band, one confined to part of the field, or a step such
as an absorption edge can be real and absent from the list (on this machine's
runs: Raman G/D/diamond bands 12-28 channels from any listed feature, and
X-ray edges never listed). The guidance said "if no feature is listed where
the requested one should be ... declare not_measurable", reading the list as
complete. It now states the list's scope, and a declaration for an unlisted
real feature is never auto-repaired: it goes to the judge.

  conda run -n scilink python -m pytest tests/test_data_facts_scope.py -q
"""
import numpy as np

from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc


def _five_band_cube():
    """Five emission bands of falling height: the fifth, real and well above
    the noise, is beyond the four the facts list."""
    rng = np.random.default_rng(0)
    E = np.linspace(400.0, 900.0, 500)
    heights = {450: 1.0, 530: 0.8, 610: 0.6, 690: 0.45, 820: 0.3}
    y = 0.05 + sum(h * np.exp(-0.5 * ((E - c) / 8.0) ** 2) for c, h in heights.items())
    return (y[None, None, :] * np.ones((10, 10, 1)) + rng.normal(0, 0.005, (10, 10, E.size))), E


def test_the_guidance_states_the_lists_scope():
    cube, E = _five_band_cube()
    facts = hc._data_facts(cube, E, "nm")
    assert len(facts["peaks"]) == 4 and not any(abs(p - 820) < 5 for p in facts["peaks"])   # the fifth is unlisted
    text = facts["text"]
    assert "holds only the field mean's peaks and dips, at most 4 per direction" in text
    assert "can be real and unlisted" in text
    assert "If no feature is listed where the requested one should be, test it there and declare" not in text


def test_a_declaration_for_an_unlisted_real_band_is_judged_not_repaired():
    cube, E = _five_band_cube()
    facts = hc._data_facts(cube, E, "nm")
    # a real band (the fifth) the facts do not list: no listed feature inside the window,
    # so the in-place repair does not fire — the judge decides, as for any declaration
    unlisted = {"feature": "band near 820 nm", "window": [800, 840], "evidence": "x", "description": "y"}
    assert hc._contradicting_feature(unlisted, facts) is None
    listed = {**unlisted, "window": [600, 620]}                   # the 610 nm band is listed
    hit = hc._contradicting_feature(listed, facts)
    assert hit is not None and abs(hit["position"] - 610) < 5


def test_an_edge_is_never_called_featureless():
    """An absorption edge (a step) is never a peak or a dip, so the facts list
    nothing for it however strong it is. The no-feature line once said "the
    field mean looks featureless" right before the guidance that a step can be
    real and unlisted; it now says what was tested and what was not."""
    E = np.linspace(400.0, 900.0, 300)
    y = 0.5 + np.arctan((E - 650.0) / 5.0) / np.pi                      # a jump of ~1, ~2000 sigma of the mean
    cube = y[None, None, :] * np.ones((20, 20, 1)) + np.random.default_rng(0).normal(0, 0.01, (20, 20, E.size))
    facts = hc._data_facts(cube, E, "nm")
    assert facts["features"] == [] and facts["measurable"] is False
    assert "featureless" not in facts["text"]
    assert "a step such as an edge" in facts["text"] and "is not tested here" in facts["text"]


def test_only_a_listed_feature_rejects_a_declaration():
    """The guidance names weak, local and step features as possibly real and
    unlisted, then says which declarations are rejected: those whose window
    holds a LISTED feature (what `_contradicting_feature` enforces) — never an
    honest null over an unlisted one after the test the sentence asks for."""
    cube, E = _five_band_cube()
    text = hc._data_facts(cube, E, "nm")["text"]
    assert "a declaration whose window holds a LISTED feature is rejected" in text
    assert "holds one of these features is rejected" not in text
