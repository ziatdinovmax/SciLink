"""Pinned outputs: names that mean the same thing under every recipe.

Seen live: ``peak_1_amplitude`` was the peak HEIGHT in one generated reference
script and the AREA in the next, so an objective keyed on a script's own
parameter name can silently change meaning when a live loop re-anchors. The
caller declares the outputs it needs; every recipe the loop locks is extended to
report exactly those names, and accepted only if — on the reference data — the
fit is untouched and every output is present and finite.

No LLM calls anywhere — a scripted model stands in.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.live import MeasurementLoop
from scilink.live.pinning import (check_pinned, lift_outputs, pin_outputs,
                                  splice_edits)

SCRIPT = '''import json
area, width = 4.0, 0.5
results = {"parameters": {"peak_1": {"amplitude": area, "fwhm": width}},
           "fit_quality": {"r_squared": 0.99}}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
print("CUSTOM_SCRIPT_SUCCESS")
'''

BRANCHY = '''import json
def main(ok):
    results = {"parameters": {}}
    if ok:
        print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
    else:
        print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
'''


class TestSplice:
    def test_the_block_lands_before_the_print_at_its_indentation(self):
        [edit] = splice_edits(SCRIPT, 'results["parameters"]["outputs"] = {"h": area / width}')
        assert edit["old_text"] == 'print(f"FIT_RESULTS_JSON:{json.dumps(results)}")'
        lines = edit["new_text"].splitlines()
        assert lines[0].startswith("# --- pinned outputs")
        assert lines[1] == 'results["parameters"]["outputs"] = {"h": area / width}'
        assert lines[-1] == edit["old_text"]

    def test_the_models_own_indentation_is_irrelevant(self):
        # The failure mode that discarded a live edit-adaptation: stray indent.
        messy = '\n        h = area / width\n        results["parameters"]["outputs"] = {"h": h}\n'
        [edit] = splice_edits(SCRIPT, messy)
        body = edit["new_text"].splitlines()[1:3]
        assert body == ['h = area / width', 'results["parameters"]["outputs"] = {"h": h}']

    def test_nested_blocks_keep_their_relative_indent(self):
        code = "if width > 0:\n    h = area / width\nelse:\n    h = 0.0"
        [edit] = splice_edits(SCRIPT, code)
        assert "if width > 0:\n    h = area / width" in edit["new_text"]

    def test_an_indented_print_indents_the_block(self):
        edits = splice_edits(BRANCHY, 'results["parameters"]["outputs"] = {"h": 1.0}')
        assert len(edits) == 1 and edits[0]["replace_all"] is True     # same line twice
        assert '        results["parameters"]["outputs"]' in edits[0]["new_text"]

    def test_the_spliced_script_really_runs(self, tmp_path):
        from scilink.utils.file_edit import apply_snippet_edits
        edits = splice_edits(SCRIPT, 'results["parameters"]["outputs"] = {"h": area / width}')
        text = apply_snippet_edits(SCRIPT, edits)["text"]
        import subprocess, sys
        out = subprocess.run([sys.executable, "-c", text], capture_output=True, text=True).stdout
        payload = json.loads(out.split("FIT_RESULTS_JSON:")[1].splitlines()[0])
        assert payload["parameters"]["outputs"] == {"h": 8.0}

    def test_a_script_with_no_results_print_is_refused(self):
        with pytest.raises(ValueError, match="no FIT_RESULTS_JSON print"):
            splice_edits("x = 1\n", "y = 2")
        with pytest.raises(ValueError, match="no code"):
            splice_edits(SCRIPT, "   \n  ")


class TestEquivalenceGate:
    BEFORE = {"peak_1_amplitude": 4.0, "peak_1_fwhm": 0.5, "fit_r_squared": 0.99}

    def test_outputs_added_and_the_fit_untouched(self):
        after = {**self.BEFORE, "outputs_h": 8.0}
        assert check_pinned(self.BEFORE, after, ["h"]) == []

    def test_a_missing_or_non_finite_output(self):
        assert "missing" in check_pinned(self.BEFORE, dict(self.BEFORE), ["h"])[0]
        after = {**self.BEFORE, "outputs_h": float("nan")}
        assert "not a finite number" in check_pinned(self.BEFORE, after, ["h"])[0]

    def test_the_added_lines_may_not_move_the_fit(self):
        after = {**self.BEFORE, "peak_1_fwhm": 0.51, "outputs_h": 8.0}
        [p] = check_pinned(self.BEFORE, after, ["h"])
        assert "peak_1_fwhm" in p and "must not alter the fit" in p
        gone = {"peak_1_amplitude": 4.0, "fit_r_squared": 0.99, "outputs_h": 8.0}
        assert "disappeared" in check_pinned(self.BEFORE, gone, ["h"])[0]

    def test_lift(self):
        assert lift_outputs({"outputs_h": 8.0, "peak_1_fwhm": 0.5}) == {
            "outputs_h": 8.0, "h": 8.0, "peak_1_fwhm": 0.5}


class Scripted:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate_content(self, prompt, **kw):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.replies.pop(0))


def _reply(code, why="height = area / width"):
    return json.dumps({"code": code, "rationale": why})


def _runner(script):
    """replay(edits): actually run the (edited) script and flatten its output."""
    import subprocess, sys
    from scilink.agents.exp_agents.feature_table import _flatten_scalars
    from scilink.utils.file_edit import apply_snippet_edits

    def replay(edits):
        text = apply_snippet_edits(script, edits)["text"] if edits else script
        r = subprocess.run([sys.executable, "-c", text], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1])
        payload = json.loads(r.stdout.split("FIT_RESULTS_JSON:")[1].splitlines()[0])
        flat = {k: float(v) for k, v in _flatten_scalars(payload["parameters"]).items()}
        flat.update({f"fit_{k}": float(v) for k, v in payload["fit_quality"].items()})
        return flat
    return replay


class TestPinOutputs:
    OUT = {"peak1_height": "height of the first peak above the baseline"}

    def test_accepted_first_time(self):
        m = Scripted([_reply('results["parameters"]["outputs"] = {"peak1_height": area / width}')])
        out = pin_outputs(script=SCRIPT, outputs=self.OUT, model=m, replay=_runner(SCRIPT))
        assert out["attempts"] == 1 and out["features"]["peak1_height"] == 8.0
        assert out["features"]["peak_1_amplitude"] == 4.0          # the area is still the area
        assert "height of the first peak above the baseline" in m.prompts[0]

    def test_a_crash_is_fed_back_and_retried(self):
        m = Scripted([_reply('results["parameters"]["outputs"] = {"peak1_height": hieght}'),
                      _reply('results["parameters"]["outputs"] = {"peak1_height": area / width}')])
        out = pin_outputs(script=SCRIPT, outputs=self.OUT, model=m, replay=_runner(SCRIPT))
        assert out["attempts"] == 2
        assert "previous attempt was rejected" in m.prompts[1] and "hieght" in m.prompts[1]

    def test_code_that_moves_the_fit_is_rejected(self):
        bad = 'width = 0.7\nresults["parameters"]["peak_1"]["fwhm"] = width\n' \
              'results["parameters"]["outputs"] = {"peak1_height": area / width}'
        m = Scripted([_reply(bad)] * 3)
        with pytest.raises(RuntimeError, match="must not alter the fit"):
            pin_outputs(script=SCRIPT, outputs=self.OUT, model=m, replay=_runner(SCRIPT))
        assert len(m.prompts) == 3

    def test_names_must_be_identifiers(self):
        with pytest.raises(ValueError, match="identifiers"):
            pin_outputs(script=SCRIPT, outputs={"peak height": "x"}, model=None,
                        replay=lambda e: {})


# ──────────────────────────────────────────────────────────────
# in the loop
# ──────────────────────────────────────────────────────────────

class PinningAgent:
    """Replays for real: applies the script_edits and runs the script."""
    script = SCRIPT
    model = None
    calls = []

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def analyze(self, data, **kw):
        PinningAgent.calls.append(kw)
        import subprocess, sys
        from scilink.utils.file_edit import apply_snippet_edits
        text = PinningAgent.script
        if kw.get("script_edits"):
            text = apply_snippet_edits(text, kw["script_edits"])["text"]
        r = subprocess.run([sys.executable, "-c", text], capture_output=True, text=True)
        if r.returncode != 0:
            return {"status": "error", "error": {"error": r.stderr[-200:]}}
        payload = json.loads(r.stdout.split("FIT_RESULTS_JSON:")[1].splitlines()[0])
        return {"status": "success", "output_directory": self.output_dir,
                "fitting_parameters": payload["parameters"],
                "fit_quality": payload["fit_quality"],
                "reuse_validity": {"verdict": "good", "drift": "none"},
                "stage_timings": {"llm_calls": 0}}


def _anchor(root):
    d = root / "reference_run"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "fitting_script.py").write_text(SCRIPT)
    (d / "series_fit_results.json").write_text(json.dumps({"results": []}))
    (d / "analysis_results.json").write_text(json.dumps({
        "status": "success", "fit_quality": {"r_squared": 0.99},
        "fitting_parameters": {"peak_1": {"amplitude": 4.0, "fwhm": 0.5}}}))
    return d


class TestLoopWithPinnedOutputs:
    def _loop(self, tmp_path, replies, **kw):
        PinningAgent.calls = []
        PinningAgent.model = Scripted(replies)
        return MeasurementLoop(str(tmp_path / "loop"), agent_factory=PinningAgent,
                               outputs={"peak1_height": "height of the first peak"}, **kw)

    GOOD = _reply('results["parameters"]["outputs"] = {"peak1_height": area / width}')

    def test_frames_report_the_pinned_name(self, tmp_path):
        loop = self._loop(tmp_path, [self.GOOD], objective_key="peak1_height")
        rec = loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")
        assert rec["pinned_outputs"]["attempts"] == 1
        assert rec["recipe"]["pinned_outputs"] == ["peak1_height"]
        assert rec["reference_features"]["peak1_height"] == 8.0
        frame = loop.step("f1.csv")
        assert frame["features"]["peak1_height"] == 8.0 and frame["objective"] == 8.0
        assert frame["flags"] == [] and frame["llm_calls"] == 0
        assert PinningAgent.calls[-1]["script_edits"] == loop._edits        # every frame carries them

    def test_the_objective_may_only_be_a_name_the_recipe_reports(self, tmp_path):
        loop = self._loop(tmp_path, [self.GOOD], objective_key="peak1_area")
        with pytest.raises(ValueError, match="peak1_height"):
            loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")

    def test_an_adopted_anchor_needs_its_reference_data(self, tmp_path):
        loop = self._loop(tmp_path, [self.GOOD])
        with pytest.raises(ValueError, match="reference_data"):
            loop.setup(anchor=str(_anchor(tmp_path)))

    def test_a_recipe_that_cannot_be_pinned_does_not_arm_the_loop(self, tmp_path):
        loop = self._loop(tmp_path, [_reply("x = undefined_name")] * 3)
        with pytest.raises(RuntimeError, match="could not pin outputs"):
            loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")

    def test_amendments_stack_on_top_of_the_pin(self, tmp_path):
        loop = self._loop(tmp_path, [self.GOOD])
        loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")
        loop.amend([{"old_text": "area, width = 4.0, 0.5", "new_text": "area, width = 4.0, 0.25"}])
        assert loop.step("f.csv")["features"]["peak1_height"] == 16.0       # pinned value follows

    def test_a_reanchor_adopts_the_workers_pin_for_the_NEW_script(self, tmp_path):
        loop = self._loop(tmp_path, [self.GOOD], breach_patience=1,
                          escalation_runner=lambda spec: SimpleNamespace(
                              spec=spec, poll=lambda: result))
        loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")
        new = tmp_path / "regime2"
        (new / "scripts").mkdir(parents=True)
        new_script = SCRIPT.replace("area, width = 4.0, 0.5", "area, width = 9.0, 0.5")
        (new / "scripts" / "fitting_script.py").write_text(new_script)
        (new / "series_fit_results.json").write_text("{}")
        pin = splice_edits(new_script, 'results["parameters"]["outputs"] = {"peak1_height": area / width}')
        result = {"status": "success", "output_directory": str(new), "seconds": 60, "llm_calls": 5,
                  "pin_edits": pin, "pin_features": {"peak1_height": 18.0}}
        started = loop.escalate("frame.csv")
        assert started["event"] == "escalation_started"
        PinningAgent.script = new_script
        try:
            frame = loop.step("after.csv")
        finally:
            PinningAgent.script = SCRIPT
        assert frame["features"]["peak1_height"] == 18.0                 # same NAME, new recipe
        assert loop._edits == pin and loop.recipe["pinned_outputs"] == ["peak1_height"]

    def test_a_reanchor_that_cannot_be_pinned_keeps_the_old_recipe(self, tmp_path):
        result = {"status": "success", "output_directory": None, "pin_error": "RuntimeError: nope"}
        loop = self._loop(tmp_path, [self.GOOD], escalation_runner=lambda spec: SimpleNamespace(
            poll=lambda: result))
        loop.setup(anchor=str(_anchor(tmp_path)), reference_data="ref.csv")
        old = loop.recipe["id"]
        new = tmp_path / "regime2"
        (new / "scripts").mkdir(parents=True)
        (new / "scripts" / "fitting_script.py").write_text(SCRIPT)
        (new / "series_fit_results.json").write_text("{}")
        result["output_directory"] = str(new)
        loop.escalate("frame.csv")
        loop.step("after.csv")
        assert loop.recipe["id"] == old
        [failed] = [e for e in loop.read_log() if e["event"] == "escalation_failed"]
        assert "pinned outputs" in failed["error"]


def test_targets_scope_the_plan_as_well_as_the_verdict():
    from scilink.agents.exp_agents._qc_profile import planning_addendum
    a = planning_addendum({"analysis_targets": ["position of the first peak"]})
    assert "position of the first peak" in a and "leave out structure" in a
    assert planning_addendum({"analysis_targets": []}) is None
    assert planning_addendum(None) is None
