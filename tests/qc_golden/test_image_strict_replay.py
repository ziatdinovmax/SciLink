"""A strict image replay: a whole image analysis with ZERO model calls, judged on
evidence alone. The prerequisite for an image frame in a live loop.

An ordinary image reuse still makes about six calls (skill suggestion, planning,
plan validation, one vision review, the tier 2 decision, synthesis). Under
``strict_replay`` none of them runs: the approved script is executed verbatim,
the verdict comes from ``_replay_feature_gate`` (what the script reports against
what it reported on its reference), a script that raises fails the frame, and no
HTML report is written. ``ScriptedModel([])`` fails loudly on ANY call.
"""

import json

import numpy as np
import pytest

from .fixtures import write_blob_image
from .harness import ScriptedModel
from .scenarios import image_rules


def _prior(tmp_path):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    data = tmp_path / "data"
    data.mkdir()
    image = write_blob_image(data / "blobs.npy")
    agent = ImageAnalysisAgent(output_dir=str(tmp_path / "prior"), enable_human_feedback=False,
                               use_literature=False)
    agent.model = ScriptedModel(image_rules("happy"))
    assert agent.analyze(str(image))["status"] == "success"
    return image, tmp_path / "prior"


def _strict(tmp_path, name, image, prior, **kw):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(output_dir=str(tmp_path / name), enable_human_feedback=False,
                               use_literature=False)
    agent.model = ScriptedModel([])                       # any call is a failure
    return agent.analyze(str(image), prior_analysis_paths=[str(prior)], reuse_locked_script=True,
                         strict_replay=True, **kw), agent.model


def test_a_strict_replay_is_a_whole_image_analysis_with_no_model_call(tmp_path):
    image, prior = _prior(tmp_path)
    result, model = _strict(tmp_path, "frame", image, prior)
    assert result["status"] == "success", result.get("error")
    assert model.calls == []
    rv = result["reuse_validity"]
    assert rv["verdict"] == "good" and rv["gate"] == "deterministic"
    reference = json.loads((prior / "analysis_results.json").read_text())["extracted_features"]
    assert result["extracted_features"] == reference      # same image, same script, same numbers
    assert (result.get("stage_timings") or {}).get("llm_calls", 0) == 0
    assert not list((tmp_path / "frame").glob("*.html"))   # no per-frame report
    assert list((tmp_path / "frame").rglob("visualization.png"))   # the overlay is kept


def test_an_image_with_nothing_in_it_is_judged_poor_on_evidence(tmp_path):
    image, prior = _prior(tmp_path)
    script = next((prior / "scripts").glob("*.py"))       # the canned script hardcodes its count: make it measure
    script.write_text(script.read_text().replace(
        '"blob_count": 3,', '"blob_count": int(__import__("scipy.ndimage").ndimage.label(mask)[1]),'))
    empty = tmp_path / "data" / "empty.npy"
    np.save(empty, np.full((128, 128), 0.02, dtype=np.float32))
    result, model = _strict(tmp_path, "empty", empty, prior)
    assert model.calls == []
    rv = result["reuse_validity"]
    assert rv["verdict"] == "poor" and "nothing was found" in rv["message"]
    assert result["extracted_features"]["blob_count"] == 0   # the numbers are still reported, flagged


def test_a_script_that_raises_fails_the_frame_and_is_not_repaired(tmp_path):
    image, prior = _prior(tmp_path)
    script = next((prior / "scripts").glob("*.py"))
    script.write_text("raise RuntimeError('the locked script does not run on this image')\n" + script.read_text())
    result, model = _strict(tmp_path, "broken", image, prior)
    assert model.calls == []                              # no correction call, no re-derivation
    failed = result["status"] != "success" or (result.get("reuse_validity") or {}).get("verdict") == "failed"
    assert failed


def test_strict_needs_a_script_to_replay(tmp_path):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    data = tmp_path / "data"
    data.mkdir()
    image = write_blob_image(data / "blobs.npy")
    agent = ImageAnalysisAgent(output_dir=str(tmp_path / "out"), enable_human_feedback=False, use_literature=False)
    agent.model = ScriptedModel([])
    res = agent.analyze(str(image), strict_replay=True)
    assert res["status"] == "error" and "strict_replay requires" in res["error"]["error"]


class TestTheGate:
    REF = {"blob_count": 3, "mask_fraction": 0.12, "mean_diameter_px": 14.0, "label": "x"}

    def _gate(self, features, reference=None):
        from scilink.agents.exp_agents.controllers.image_analysis_controllers import _replay_feature_gate
        return _replay_feature_gate(features, self.REF if reference is None else reference)

    def test_values_may_move(self):
        assert self._gate({"blob_count": 41, "mask_fraction": 0.6, "mean_diameter_px": 3.0}) == (True, "")
        assert self._gate({"blob_count": 0, "mask_fraction": 0.11, "mean_diameter_px": 14.2})[0]   # one zero is data

    def test_a_quantity_that_stops_being_reported_or_is_not_finite(self):
        ok, why = self._gate({"blob_count": 3, "mask_fraction": 0.1})
        assert not ok and "mean_diameter_px" in why
        ok, why = self._gate({"blob_count": 3, "mask_fraction": float("nan"), "mean_diameter_px": 1.0})
        assert not ok and "not finite" in why

    def test_nothing_found_at_all(self):
        ok, why = self._gate({"blob_count": 0, "mask_fraction": 0.0, "mean_diameter_px": 0.0})
        assert not ok and "nothing was found" in why

    def test_without_a_reference_it_asks_only_for_finite_numbers(self):
        assert self._gate({"a": 1.0}, reference={})[0] and not self._gate({"a": float("inf")}, reference={})[0]
        assert not self._gate({}, reference={})[0]
