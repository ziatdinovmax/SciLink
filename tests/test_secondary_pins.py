"""A pin on a component the fit does not target is a caveat, not a degenerate fit (#742).

Re-applying the pinned-at-bound rule to every curve series on disk: 4 of the
12 with recorded bounds had a pinned anchor, every real multi-component
dataset among them, and in every case the pinned parameter belonged to a
secondary component (a broad water hump, a weak side band, a C-H stretch the
objective did not ask for) while the study's band fitted well. One such pin
withheld the whole regime. A fitting script now declares its ``targets`` (the
components whose parameters answer the plan's ``parameters_to_extract``). A
pin on a target is degenerate exactly as before; a pin on any other component
is recorded as a caveat, its value reported as no value (a bound is not a
measurement), and the unit — and through it the regime — is judged on its
targets. A fit that declares no targets (or names none of its components) is
judged as today.

  conda run -n scilink python -m pytest tests/test_secondary_pins.py -q
"""
import json

from scilink.agents.exp_agents._verification_record import analysis_verdict
from scilink.skills._shared.curve_fitting_tools import split_pins_by_targets

import test_series_verdict_path as sv

PIN_BG = {"component": "background", "parameter": "center", "value": 700.0, "bound": 700.0, "side": "upper"}
PIN_PEAK = {"component": "peak_1", "parameter": "fwhm", "value": 12.0, "bound": 12.0, "side": "upper"}


def test_split_pins_by_targets():
    params = {"peak_1": {}, "background": {}}
    assert split_pins_by_targets([PIN_BG, PIN_PEAK], ["peak_1"], params) == ([PIN_PEAK], [PIN_BG])
    # undeclared, empty, or naming no fitted component: every component is a target
    for targets in (None, [], ["peak_9"], "peak_1"):
        assert split_pins_by_targets([PIN_BG, PIN_PEAK], targets, params) == ([PIN_BG, PIN_PEAK], [])


class SecondaryPinExecutor(sv.FakeExecutor):
    """As FakeExecutor, with a broad background component in the fit. For the
    named units its centre ends AT its bound (the bounds printed as a real
    script prints them); ``targets`` declares what the fit is for."""

    def __init__(self, r2_by_name, pinned_names, targets=("peak_1",), pin_component="background"):
        super().__init__(r2_by_name)
        self.pinned_names, self.targets, self.pin_component = set(pinned_names), targets, pin_component

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        res = super().execute_script(script, working_dir=working_dir, timeout=timeout, **kw)
        name = self.calls[-1][0] if self.calls else None
        if res.get("status") != "success":
            return res
        out = json.loads(res["stdout"].split("FIT_RESULTS_JSON:", 1)[1])
        out["parameters"]["background"] = {"center": 650.0, "amplitude": 0.3, "fwhm": 300.0}
        out["bounds"] = {"background": {"center": [500.0, 700.0]}, "peak_1": {"fwhm": [1.0, 40.0]}}
        if self.targets is not None:
            out["targets"] = list(self.targets)
        if name in self.pinned_names:
            comp = out["parameters"][self.pin_component]
            if self.pin_component == "background":
                comp["center"] = 700.0
            else:
                comp["fwhm"] = 40.0
        res["stdout"] = "FIT_RESULTS_JSON:" + json.dumps(out)
        return res


def _run(tmp_path, monkeypatch, ex, anchor=None):
    follower_r2 = {"spectrum_0001": 0.97, "spectrum_0002": 0.96}
    ex.r2_by_name.update(follower_r2)
    state, _ = sv.run_series(tmp_path, monkeypatch, names=sv.NAMES,
                             anchors={"spectrum_0000": anchor or sv.OK}, follower_r2=follower_r2, executor=ex)
    raw = {u["name"]: u for u in state["series_results"]}
    results = sv.compile_results(tmp_path, state)
    by_name = {u["name"]: u for u in results["individual_results"]}
    flags = {f["name"]: f["reason"] for f in (state.get("flagged_spectra") or [])}
    return raw, by_name, flags, results


def test_a_follower_pinned_on_a_secondary_component_verifies_with_a_caveat(tmp_path, monkeypatch):
    ex = SecondaryPinExecutor({}, {"spectrum_0001"})
    raw, by_name, flags, results = _run(tmp_path, monkeypatch, ex)
    row = raw["spectrum_0001"]
    assert [n for n, _ in ex.calls].count("spectrum_0001") == 1           # no relax-and-refit for a secondary pin
    assert row["replay_verbatim"] is True and not row.get("pinned_at_bound")
    assert row["secondary_pins"][0]["component"] == "background"
    assert row["parameters"]["background"]["center"] is None                # a bound is not a measurement
    assert row["parameters"]["peak_1"]["center"] == 144.0                   # the target's values stand
    assert any("Secondary component pinned at bound" in c for c in row["caveats"])
    assert "spectrum_0001" not in flags and "quality_warning" not in row
    assert by_name["spectrum_0001"]["unit_verdict"]["verified"] is True
    assert analysis_verdict(results)["verified"] is True


def test_a_pin_on_a_target_still_withholds(tmp_path, monkeypatch):
    ex = SecondaryPinExecutor({}, {"spectrum_0001"}, pin_component="peak_1")
    raw, by_name, flags, results = _run(tmp_path, monkeypatch, ex)
    assert raw["spectrum_0001"]["pinned_at_bound"][0]["component"] == "peak_1"
    assert flags.get("spectrum_0001") == "pinned_at_bound"
    assert by_name["spectrum_0001"]["unit_verdict"]["verified"] is False
    assert not analysis_verdict(results)["verified"]


def test_a_fit_that_declares_no_targets_is_judged_as_before(tmp_path, monkeypatch):
    ex = SecondaryPinExecutor({}, {"spectrum_0001"}, targets=None)
    raw, by_name, flags, results = _run(tmp_path, monkeypatch, ex)
    assert raw["spectrum_0001"]["pinned_at_bound"][0]["component"] == "background"
    assert "secondary_pins" not in raw["spectrum_0001"]
    assert flags.get("spectrum_0001") == "pinned_at_bound"
    assert by_name["spectrum_0001"]["unit_verdict"]["verified"] is False


def test_an_approved_fit_with_only_secondary_pins_is_verified(tmp_path, monkeypatch):
    """What the controller now writes for an anchor (or a single run) pinned
    only on a secondary component — ``secondary_pins`` and a caveat, no
    ``pinned_at_bound`` — is an approved fit: verified, so a regime anchored
    on it locks and its followers verify on their own fits. The cascade #742
    measured on real spectra (one secondary pin on the anchor withholding
    every unit) does not happen; a TARGET pin on the anchor still withholds
    the regime, as before (#726)."""
    rec = sv._canned_anchor("spectrum_0000", 0, **sv.OK)
    rec["secondary_pins"] = [PIN_BG]
    rec["fit_quality"] = {**rec["fit_quality"], "secondary_pins": [PIN_BG]}
    rec["caveats"] = ["Secondary component pinned at bound — background.center = 700 at its upper bound 700"]
    v = analysis_verdict({**rec, "status": "success"})
    assert v["verified"] is True, v
    raw, by_name, flags, results = _run(tmp_path / "t", monkeypatch, SecondaryPinExecutor({}, set()),
                                        anchor={**sv.OK, "pinned": [PIN_PEAK]})
    assert not analysis_verdict(results)["verified"]
    assert "degenerate fit" in by_name["spectrum_0001"]["unit_verdict"]["reason"]


def test_the_targets_contract_is_in_the_generation_and_the_correction_prompt(tmp_path):
    """The declared ``targets`` are part of the results contract the fitting
    script is written to (the generation prompt) and must survive a repair
    (the correction prompt, which the pinned-at-bound retry also uses)."""
    import logging
    from types import SimpleNamespace
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    from scilink.agents.exp_agents.instruct import (FITTING_SCRIPT_INSTRUCTIONS,
                                                    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS)
    prompts = []

    class _Model:
        def generate_content(self, contents, **kw):
            prompts.append(contents if isinstance(contents, str) else json.dumps(contents, default=str))
            return SimpleNamespace(text=json.dumps({"script": "print(1)", "diagnosis": "fixed"}))
    ctrl = UnifiedSeriesProcessingController(
        model=_Model(), logger=logging.getLogger("t"), generation_config=None, safety_settings=None,
        parse_fn=lambda r: (json.loads(r.text), None), executor=sv.FakeExecutor({}),
        script_instructions=FITTING_SCRIPT_INSTRUCTIONS, correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir=str(tmp_path), plot_fn=lambda d, i: b"p", r2_threshold=0.95,
        parallel_workers=1)
    state = {"locked_fitting_config": {"physical_model": "two Gaussians and a broad background",
                                       "parameters_to_extract": ["peak_1 center", "peak_1 area"]}}
    stats = {"n_points": 100, "x_range": (0.0, 1.0), "y_range": (0.0, 1.0)}
    helper = getattr(ctrl, "_fitting_helper", ctrl)          # the fitting helper builds both prompts
    helper._generate_fitting_script(state, "data.npy", stats)
    helper._correct_script(state, "print(0)", "DEGENERATE FIT — peak_1.fwhm pinned at a bound")
    assert len(prompts) == 2
    assert '"targets": ["peak_1", ...]' in prompts[0] and "parameters_to_extract" in prompts[0]
    assert "its `targets` list unchanged" in prompts[1]
