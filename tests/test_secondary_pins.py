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
    assert flags.get("spectrum_0001") == "secondary_pin" and "quality_warning" not in row   # a non-refit caveat flag
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
            return SimpleNamespace(text=json.dumps({"script": S1, "diagnosis": "fixed"}))
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



S1 = "x = np.load('data.npy')  # first script"
S2 = "x = np.load('data.npy')  # corrected script"


class _ScriptedExecutor:
    """The fitting script's output keyed by the script text: lets a correction
    change what the fit declares, as a real corrected script would."""
    timeout = 30

    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        from pathlib import Path
        self.calls.append(script)
        if self.outputs.get(script) is None:                  # this script fails to run
            return {"status": "error", "stdout": "", "stderr": "Traceback: boom", "message": "script failed: boom"}
        (Path(working_dir) / "visualization.png").write_bytes(b"png")
        return {"status": "success", "stdout": "FIT_RESULTS_JSON:" + json.dumps(self.outputs[script]),
                "stderr": "", "message": ""}


def _fit_out(targets, pin_peak):
    out = {"model_type": "two peaks", "fit_quality": {"r_squared": 0.99, "rmse": 0.01},
           "parameters": {"peak_1": {"center": 144.0, "amplitude": 1.0, "fwhm": 40.0 if pin_peak else 12.0},
                          "background": {"center": 700.0, "amplitude": 0.3, "fwhm": 300.0}},
           "bounds": {"peak_1": {"fwhm": [1.0, 40.0]}, "background": {"center": [500.0, 700.0]}}}
    if targets is not None:
        out["targets"] = targets
    return out


def _fresh_fit(tmp_path, monkeypatch, outputs, corrected=S2, **fit_kwargs):
    """The NON-held path (an anchor, a refit, fresh code): the real
    _fit_single_spectrum, the model writing S1 and correcting to ``corrected``."""
    import numpy as np
    ex = _ScriptedExecutor(outputs)
    ctrl = sv._controller(tmp_path, ex)
    monkeypatch.setattr(ctrl, "_generate_fitting_script", lambda *a, **k: S1)
    monkeypatch.setattr(ctrl, "_correct_script", lambda state, script, err: (corrected, "fixed"))
    monkeypatch.setattr(ctrl, "_check_plan_conformance", lambda state, script: None)
    state = {"locked_fitting_config": {"physical_model": "two peaks",
                                       "parameters_to_extract": ["peak_1 fwhm"]}, "system_info": {}}
    res = ctrl._fit_single_spectrum(state=state, curve_data=sv._spectrum(0), data_path=str(tmp_path / "d.npy"),
                                    spectrum_name="spectrum_0000", spectrum_idx=0, **fit_kwargs)
    return res, ex


def test_a_correction_cannot_shrink_targets_to_clear_a_target_pin(tmp_path, monkeypatch):
    """The pin retry sends "DEGENERATE FIT — peak_1.fwhm pinned" to the
    corrector; a corrected script that keeps the pin but drops peak_1 from its
    targets would turn the target pin "secondary" and the unit verified. The
    first declaration is frozen: the fit stays degenerate, peak_1.fwhm keeps its
    value (it is the plan's quantity), and nothing is nulled."""
    res, ex = _fresh_fit(tmp_path, monkeypatch, {S1: _fit_out(["peak_1"], True),
                                                  S2: _fit_out(["background"], True)})
    assert ex.calls[:2] == [S1, S2]                                   # the pin retry ran
    assert res["pinned_at_bound"][0]["component"] == "peak_1"
    assert res["parameters"]["peak_1"]["fwhm"] == 40.0 and "secondary_pins" not in res
    assert res["targets"] == ["peak_1", "background"]                     # frozen, only ever added to
    assert res["quality_warning"].startswith("Degenerate fit")


def test_a_secondary_pin_on_the_non_held_path_skips_the_relax_retry(tmp_path, monkeypatch):
    """On an anchor or refit (not a held follower), a pin only on a component
    the fit does not target is not sent through the relax-and-refit retry."""
    res, ex = _fresh_fit(tmp_path, monkeypatch, {S1: {**_fit_out(["peak_1"], False),
                                                        "parameters": {**_fit_out(["peak_1"], False)["parameters"],
                                                                       "background": {"center": 700.0, "center_err": 3.0}}}})
    assert ex.calls == [S1]                                             # one run: no retry
    assert res["secondary_pins"][0]["component"] == "background"
    assert res["parameters"]["background"]["center"] is None and res["parameters"]["background"]["center_err"] is None
    assert "pinned_at_bound" not in res



S0 = "x = np.load('data.npy')  # the locked script, which fails here"


def test_a_follower_starts_from_its_anchors_targets(tmp_path, monkeypatch):
    """A held follower whose locked script fails to RUN on attempt 1 declares
    nothing then; its repair declares ["background"] with the target peak_1
    pinned. Seeded with its regime anchor's targets, the pin stays a target pin
    and the value is kept; unseeded, the repair's narrow declaration would make
    it "secondary" and null the plan's quantity — the hole the seed closes."""
    outputs = {S0: None, S2: _fit_out(["background"], True)}
    seeded, ex = _fresh_fit(tmp_path / "seeded", monkeypatch, outputs,
                            base_script=S0, hold_recipe=True, seed_targets=["peak_1"])
    assert ex.calls[:2] == [S0, S2]                                       # failed to run, then repaired
    assert seeded["pinned_at_bound"][0]["component"] == "peak_1" and seeded["parameters"]["peak_1"]["fwhm"] == 40.0
    unseeded, _ = _fresh_fit(tmp_path / "unseeded", monkeypatch, outputs, base_script=S0, hold_recipe=True)
    assert unseeded["secondary_pins"][0]["component"] == "peak_1" and unseeded["parameters"]["peak_1"]["fwhm"] is None


def test_secondary_pin_caveats_never_make_a_series_read_as_a_mismatch():
    """The report calls a MAJORITY of flagged frames a series-wide mismatch
    ("below the acceptance threshold"). A secondary-pin caveat is not a failing
    frame: a verified series with pinned backgrounds reads as flagged for
    review, while failing flags still make the mismatch."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedCurveReportController
    rep = UnifiedCurveReportController.__new__(UnifiedCurveReportController)
    rows = [{"index": i, "name": f"s{i}", "success": True, "fit_quality": {"r_squared": 0.99}} for i in range(3)]

    def flag(i, reason):
        return {"index": i, "name": f"s{i}", "reason": reason, "r_squared": None, "series_mean": None,
                "series_std": None, "deviation_sigma": None, "recommendation": "x"}
    caveats = rep._generate_flagged_spectra_section(
        [flag(0, "below_threshold"), flag(1, "secondary_pin"), flag(2, "secondary_pin")], rows, {})
    assert "Series-Wide Mismatch" not in caveats
    failing = rep._generate_flagged_spectra_section([flag(i, "below_threshold") for i in range(3)], rows, {})
    assert "Series-Wide Mismatch" in failing
