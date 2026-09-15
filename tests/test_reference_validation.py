"""The engine-neutral reference-property validation stage.

Dedupes components, collects each distinct one's measured property, and stays
alive when an individual measurement fails. The heavy backend-specific
measurement is injected (``measure_fn``), so these run with no sim, no force
field, and no engine.
"""

from scilink.agents.sim_agents.reference_validation import (
    validate_component_properties,
)


def test_dedupes_and_collects_measurements():
    comps = [
        {"name": "water", "smiles": "O", "count": 1000},
        {"name": "water", "smiles": "O", "count": 500},   # dup → measured once
        {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C", "count": 100},
    ]
    seen_smiles = []

    def fake_measure(c):
        seen_smiles.append(c["smiles"])
        return {"value": 1.00 if c["smiles"] == "O" else 1.03, "units": "g/cm^3"}

    report = validate_component_properties(comps, fake_measure)

    assert seen_smiles == ["O", "CCS(=O)(=O)C(C)C"]        # water measured once
    assert report["status"] == "success"
    assert report["reference_property"] == "density"
    by = {m["component"]: m for m in report["measurements"]}
    assert by["water"]["status"] == "measured" and by["water"]["value"] == 1.00
    assert by["EIS"]["value"] == 1.03 and by["EIS"]["units"] == "g/cm^3"


def test_extra_evidence_fields_carry_through():
    def fake_measure(c):
        return {"value": 1.03, "units": "g/cm^3", "n_molecules": 200,
                "equilibrated_ps": 20.0}

    report = validate_component_properties(
        [{"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C"}], fake_measure)
    m = report["measurements"][0]
    assert m["n_molecules"] == 200 and m["equilibrated_ps"] == 20.0


def test_component_failure_is_not_fatal():
    def fake_measure(c):
        if c["smiles"] == "X":
            raise RuntimeError("packmol failed")
        return {"value": 1.0, "units": "g/cm^3"}

    report = validate_component_properties(
        [{"name": "good", "smiles": "O"}, {"name": "bad", "smiles": "X"}],
        fake_measure)
    by = {m["component"]: m for m in report["measurements"]}
    assert by["good"]["status"] == "measured"
    assert by["bad"]["status"] == "unmeasured"
    assert "packmol failed" in by["bad"]["error"]
    assert report["status"] == "success"          # partial evidence still counts


def test_no_measurements_status():
    report = validate_component_properties(
        [{"name": "a", "smiles": "O"}], lambda c: None)
    assert report["status"] == "no_measurements"
    assert report["measurements"][0]["status"] == "unmeasured"
    assert report["measurements"][0]["error"] == "no value returned"


def test_reference_property_is_recorded():
    report = validate_component_properties(
        [{"name": "a", "smiles": "O"}],
        lambda c: {"value": 900.0, "units": "K"},
        reference_property="melting_point")
    assert report["reference_property"] == "melting_point"
    assert report["measurements"][0]["value"] == 900.0
