"""From "the data changed" to "is this new?": the slow half of a discovery
(scilink/live/discovery.py), the chain of the first SciLink paper run on the
frame a live loop flagged. No model, no network: everything is injected."""

import json

from scilink.live.discovery import assess_change, describe


class Agent:
    seen = []

    def __init__(self, out):
        self.out = out

    def analyze(self, data, **kw):
        Agent.seen.append(kw)
        return {"status": "success", "output_directory": self.out, "detailed_analysis": "A second population.",
                "stage_timings": {"llm_calls": 5},
                "scientific_claims": [
                    {"claim": "A second population of 2 nm particles nucleates at 450 C.",
                     "has_anyone_question": "Has anyone seen secondary nucleation of 2 nm particles at 450 C?"},
                    {"claim": "The large particles keep coarsening.", "has_anyone_question": "Has anyone ...?"},
                    {"claim": "no question here"}]}


class Lit:
    def query_literature(self, q):
        return {"status": "success", "formatted_answer": "two reports" if "coarsening" in q or "..." in q else "none found"}


class Scorer:
    def score_novelty(self, q, answer):
        return {"novelty_score": 5 if answer == "none found" else 2, "explanation": answer}


def test_claims_about_the_changed_frame_and_how_new_each_is(tmp_path):
    Agent.seen = []
    res = assess_change(str(tmp_path / "frame.npy"), modality="image", system_info={"technique": "TEM"},
                        what_changed="new structure appeared at length scales of 3 to 5 nm",
                        agent_kwargs={}, out_dir=str(tmp_path / "d"), agent_factory=Agent,
                        literature=Lit(), scorer=Scorer())
    assert res["status"] == "success" and res["highest_novelty"] == 5 and res["literature"] == "asked for 2 claim(s)"
    assert [c.get("novelty_score") for c in res["claims"]] == [5, 2, None]
    assert Agent.seen[0]["hints"].startswith("new structure appeared") and Agent.seen[0]["profile"] == "quick"
    assert json.loads((tmp_path / "d" / "discovery.json").read_text())["claims"][0]["novelty_score"] == 5
    assert describe(res)[0].startswith("[novelty 5/5] A second population")


def test_without_a_literature_key_the_claims_stand_and_it_says_so(tmp_path):
    res = assess_change(str(tmp_path / "f.npy"), modality="image", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "d"), agent_factory=Agent)
    assert res["status"] == "success" and len(res["claims"]) == 3
    assert "no literature key" in res["literature"] and "highest_novelty" not in res


def test_it_never_raises(tmp_path):
    class Broken:
        def __init__(self, out): pass
        def analyze(self, *a, **k): raise RuntimeError("no model")
    res = assess_change(str(tmp_path / "f.npy"), modality="curve", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "d"), agent_factory=Broken)
    assert res["status"] == "error" and "no model" in res["error"] and res["claims"] == []


def test_the_loop_runs_it_for_the_latest_change_and_puts_it_on_the_log(tmp_path):
    from tests.test_measurement_loop import FakeAgent, make_anchor
    from scilink.live import MeasurementLoop
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=FakeAgent)
    loop.setup(anchor=str(make_anchor(tmp_path)))
    loop._last_novelty = {"step": 7, "since_step": 6, "data": str(tmp_path / "f7.csv"),
                          "where": [{"kind": "new", "x_from": 16.5, "x_to": 17.5, "x_peak": 17.0, "share": 1.0}]}
    Agent.seen = []
    out = loop.assess_change(agent_factory=Agent, literature=Lit(), scorer=Scorer())
    assert out["highest_novelty"] == 5 and "new structure appeared from 16.5 to 17.5" in Agent.seen[0]["hints"]
    event = next(e for e in loop.read_log() if e["event"] == "discovery")
    assert event["about_step"] == 7 and event["claims"][0]["novelty_score"] == 5


def test_the_assessments_numbers_are_put_beside_the_recipes_for_the_same_frame(tmp_path):
    """What a person at a pause decides on: the recipe says 52, a fresh look says
    115. The analysis is asked for the tracked names, like a rebuild is."""
    from scilink.live import MeasurementLoop
    from scilink.live.simulators import get_simulator
    from tests.test_image_measurement_loop import _anchor, _factory

    class Counts(Agent):
        def analyze(self, data, **kw):
            res = super().analyze(data, **kw)
            return {**res, "extracted_features": {"particle_count": 115, "Mean_Diameter_nm_all": 3.9, "other": 1.0}}

    sim = get_simulator("particle_coarsening_images", seed=2)
    loop = MeasurementLoop(str(tmp_path / "loop"), system_info=sim.system_info, instrument=sim,
                           outputs=sim.outputs, agent_factory=_factory, check_portability=False)
    ref = sim.acquire({}).save(str(tmp_path / "ref"), 0, stem="reference")
    loop.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})), reference_data=ref)
    frame = sim.acquire({})
    record = loop.step(frame.save(str(tmp_path / "frames"), 1))
    loop._last_novelty = {"step": record["step"], "since_step": 1, "data": record["data"], "where": []}
    Agent.seen = []
    out = loop.assess_change(agent_factory=Counts)
    assert "particle_count" in Agent.seen[0]["objective"]                  # asked for by name
    assert out["compared"]["particle_count"] == {"recipe": record["features"]["particle_count"], "analysis": 115.0}
    assert out["compared"]["mean_diameter_nm"]["analysis"] == 3.9         # matched by name, not by luck
    assert "other" not in out["compared"]
    event = next(e for e in loop.read_log() if e["event"] == "discovery")
    assert event["compared"] == out["compared"]
    loop.close()


def test_an_assessment_whose_analysis_failed_is_tried_once_deeper_and_never_called_measured(tmp_path):
    """Seen live: a quick image script raised, and the write-up still carried
    claims made from looking at the frame. That is not a measurement."""
    class Flaky(Agent):
        def analyze(self, data, **kw):
            res = super().analyze(data, **kw)
            ok = kw["profile"] == "thorough" and Flaky.recovers
            return {**res, "status": "success" if ok else "error", "error": {"error": "All 1 image analysis(es) failed"},
                    "extracted_features": {"particle_count": 115} if ok else {}}

    Agent.seen, Flaky.recovers = [], True
    out = assess_change(str(tmp_path / "f.npy"), modality="image", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "a"), agent_factory=Flaky)
    assert [k["profile"] for k in Agent.seen] == ["quick", "thorough"]
    assert out["status"] == "success" and out["retried"] and out["features"] == {"particle_count": 115.0}

    Agent.seen, Flaky.recovers = [], False
    out = assess_change(str(tmp_path / "f.npy"), modality="image", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "b"), agent_factory=Flaky)
    assert out["status"] == "unmeasured" and out["claims"] and out["features"] == {}
    assert "failed" in out["analysis_error"] and len(Agent.seen) == 2


def test_the_literature_key_can_come_from_the_environment(tmp_path, monkeypatch):
    """The literature agents read FUTUREHOUSE_API_KEY themselves; the chain must
    not skip the literature just because no key was passed explicitly."""
    made = []

    class FakeLit(Lit):
        def __init__(self, api_key=None, max_wait_time=0):
            made.append(api_key)

    monkeypatch.setenv("FUTUREHOUSE_API_KEY", "fh-from-env")
    monkeypatch.setattr("scilink.agents.lit_agents.OwlLiteratureAgent", FakeLit, raising=False)
    out = assess_change(str(tmp_path / "f.npy"), modality="image", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "a"), agent_factory=Agent, scorer=Scorer())
    assert made == ["fh-from-env"] and out["literature"].startswith("asked for")
    monkeypatch.delenv("FUTUREHOUSE_API_KEY")
    out = assess_change(str(tmp_path / "f.npy"), modality="image", system_info={}, what_changed=None,
                        agent_kwargs={}, out_dir=str(tmp_path / "b"), agent_factory=Agent, scorer=Scorer())
    assert out["literature"].startswith("not asked") and "FUTUREHOUSE_API_KEY" in out["literature"]
