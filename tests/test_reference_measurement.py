"""Pure-component property measurement — orchestration only.

Property-general: the property is passed in, and the measuring run is injected,
so these run with no simulation, no packmol, no force field, and for any
property (density here, a lattice constant in one case).
"""

from scilink.agents.sim_agents.reference_measurement import (
    measure_pure_component_property,
)

EIS = {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C"}


def _fake_build(smiles, n, rho, wd):
    return "/tmp/fake_ref.extxyz"


def _fake_parameterize(components, coords, wd):
    return {"backend": "fake", "components": components}


def test_successful_measurement_returns_property_value():
    result = measure_pure_component_property(
        EIS, "density", "/tmp/wd",
        parameterize_fn=_fake_parameterize,
        run_measure_fn=lambda ps, prop, wd: {"value": 1.03, "units": "g/cm^3"},
        build_fn=_fake_build, n_molecules=150)
    assert result == {"property": "density", "value": 1.03,
                      "units": "g/cm^3", "n_molecules": 150}


def test_property_is_passed_through_to_the_run():
    seen = {}

    def run_measure(ps, prop, wd):
        seen["property"] = prop
        return {"value": 5.43, "units": "angstrom"}

    result = measure_pure_component_property(
        {"name": "Si", "smiles": "[Si]"}, "lattice constant", "/tmp/wd",
        parameterize_fn=_fake_parameterize, run_measure_fn=run_measure,
        build_fn=_fake_build)
    assert seen["property"] == "lattice constant"
    assert result["property"] == "lattice constant" and result["value"] == 5.43
    assert result["units"] == "angstrom"


def test_parameterize_receives_single_pure_component():
    seen = {}

    def spy_parameterize(components, coords, wd):
        seen["components"] = components
        seen["coords"] = coords
        return {"ok": True}

    measure_pure_component_property(
        EIS, "density", "/tmp/wd",
        parameterize_fn=spy_parameterize,
        run_measure_fn=lambda ps, p, wd: {"value": 1.1, "units": "g/cm^3"},
        build_fn=_fake_build, n_molecules=200)
    assert seen["components"] == [
        {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C", "count": 200}]
    assert seen["coords"] == "/tmp/fake_ref.extxyz"


def test_missing_smiles_is_an_error_not_a_crash():
    result = measure_pure_component_property(
        {"name": "mystery"}, "density", "/tmp/wd",
        parameterize_fn=_fake_parameterize,
        run_measure_fn=lambda ps, p, wd: {"value": 1.0}, build_fn=_fake_build)
    assert "error" in result and "SMILES" in result["error"]


def test_build_failure_is_an_error():
    result = measure_pure_component_property(
        EIS, "density", "/tmp/wd",
        parameterize_fn=_fake_parameterize,
        run_measure_fn=lambda ps, p, wd: {"value": 1.0},
        build_fn=lambda *a, **k: None)
    assert "error" in result and "build" in result["error"].lower()


def test_parameterize_exception_is_caught():
    def boom(*a, **k):
        raise RuntimeError("no vdW for the metal")

    result = measure_pure_component_property(
        EIS, "density", "/tmp/wd", parameterize_fn=boom,
        run_measure_fn=lambda ps, p, wd: {"value": 1.0}, build_fn=_fake_build)
    assert "error" in result and "no vdW" in result["error"]


def test_run_returning_none_is_an_error():
    result = measure_pure_component_property(
        EIS, "density", "/tmp/wd", parameterize_fn=_fake_parameterize,
        run_measure_fn=lambda ps, p, wd: None, build_fn=_fake_build)
    assert "error" in result and "density" in result["error"]
