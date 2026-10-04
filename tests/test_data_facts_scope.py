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
    assert "holds only the 4 strongest field-mean features per direction" in text
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
