"""The whole pre-run reference check composed in one call.

select -> measure the chosen property -> judge. All three steps injected, so
this runs with no model, no sim, no force field.
"""

from scilink.agents.sim_agents.reference_validation import run_reference_check

COMPONENTS = [
    {"name": "water", "smiles": "O"},
    {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C"},
]


def _select(components, sysdesc):
    return {"selections": [
        {"component": "water", "property": "density", "measurable": True},
        {"component": "EIS", "property": "density", "measurable": True},
    ]}


def _judge_poor(measurements, sysdesc):
    # Flags whichever measured value is below 1.0 (stand-in for the critic).
    bad = [m for m in measurements if m.get("value", 9) < 1.0]
    return {"verdict": "poor" if bad else "good",
            "failure_class": "force_field" if bad else None,
            "per_measurement": measurements}


def test_full_chain_catches_the_bad_component():
    def measure(component, prop):
        return {"property": prop, "units": "g/cm^3",
                "value": 1.00 if component["smiles"] == "O" else 0.90}

    out = run_reference_check(
        COMPONENTS, "aqueous sulfone electrolyte",
        select_fn=_select, measure_fn=measure, judge_fn=_judge_poor)

    assert out["verdict"]["verdict"] == "poor"           # pre-run catch fires
    assert out["verdict"]["failure_class"] == "force_field"
    assert len(out["selections"]) == 2
    vals = {m["component"]: m for m in out["measurements"]}
    assert vals["EIS"]["value"] == 0.90 and vals["EIS"]["property"] == "density"


def test_good_system_passes():
    def measure(component, prop):
        return {"property": prop, "value": 1.00, "units": "g/cm^3"}

    out = run_reference_check(
        COMPONENTS, "aqueous electrolyte",
        select_fn=_select, measure_fn=measure, judge_fn=_judge_poor)
    assert out["verdict"]["verdict"] == "good"
    assert out["verdict"]["failure_class"] is None


def test_selected_property_is_passed_to_the_measurer():
    seen = {}

    def select(components, sysdesc):
        return {"selections": [
            {"component": "silicon", "property": "lattice constant",
             "measurable": True}]}

    def measure(component, prop):
        seen["property"] = prop
        return {"property": prop, "value": 5.43, "units": "angstrom"}

    run_reference_check(
        [{"name": "silicon"}], "bulk Si",
        select_fn=select, measure_fn=measure,
        judge_fn=lambda m, s: {"verdict": "good"})
    assert seen["property"] == "lattice constant"       # selection drives measurement


def test_non_measurable_component_is_skipped_not_measured():
    called = {"n": 0}

    def select(components, sysdesc):
        return {"selections": [
            {"component": "novelX", "property": None, "measurable": False,
             "rationale": "no known reference"}]}

    def measure(component, prop):
        called["n"] += 1
        return {"property": prop, "value": 1.0, "units": "g/cm^3"}

    out = run_reference_check(
        [{"name": "novelX", "smiles": "C#N"}], "unknown compound",
        select_fn=select, measure_fn=measure,
        judge_fn=lambda m, s: {"verdict": "good"})
    assert called["n"] == 0                              # never measured
    assert out["measurements"][0]["status"] == "unmeasured"
