"""Pure-component density measurement — orchestration only.

The backend/engine-touching steps (pack, parameterize, run) are injected, so
these run with no simulation, no packmol, no force field.
"""

from scilink.agents.sim_agents.reference_measurement import (
    measure_pure_component_density,
)

EIS = {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C"}


def _fake_pack(smiles, n, rho, wd):
    return "/tmp/fake_box.extxyz"


def _fake_parameterize(components, coords, wd):
    # Stand-in ParameterizedSystem; only identity matters to the orchestration.
    return {"backend": "fake", "components": components}


def test_successful_measurement_returns_density():
    result = measure_pure_component_density(
        EIS, "/tmp/wd",
        parameterize_fn=_fake_parameterize,
        run_npt_fn=lambda ps, wd: 1.03,
        pack_fn=_fake_pack,
        n_molecules=150,
    )
    assert result == {"property": "density", "value": 1.03,
                      "units": "g/cm^3", "n_molecules": 150}


def test_parameterize_receives_single_pure_component():
    seen = {}

    def spy_parameterize(components, coords, wd):
        seen["components"] = components
        seen["coords"] = coords
        return {"ok": True}

    measure_pure_component_density(
        EIS, "/tmp/wd",
        parameterize_fn=spy_parameterize,
        run_npt_fn=lambda ps, wd: 1.1,
        pack_fn=_fake_pack, n_molecules=200,
    )
    assert seen["components"] == [
        {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C", "count": 200}]
    assert seen["coords"] == "/tmp/fake_box.extxyz"


def test_missing_smiles_is_an_error_not_a_crash():
    result = measure_pure_component_density(
        {"name": "mystery"}, "/tmp/wd",
        parameterize_fn=_fake_parameterize, run_npt_fn=lambda ps, wd: 1.0,
        pack_fn=_fake_pack)
    assert "error" in result and "SMILES" in result["error"]


def test_packing_failure_is_an_error():
    result = measure_pure_component_density(
        EIS, "/tmp/wd",
        parameterize_fn=_fake_parameterize, run_npt_fn=lambda ps, wd: 1.0,
        pack_fn=lambda *a, **k: None)
    assert "error" in result and "pack" in result["error"].lower()


def test_parameterize_exception_is_caught():
    def boom(*a, **k):
        raise RuntimeError("no vdW for the metal")

    result = measure_pure_component_density(
        EIS, "/tmp/wd", parameterize_fn=boom,
        run_npt_fn=lambda ps, wd: 1.0, pack_fn=_fake_pack)
    assert "error" in result and "no vdW" in result["error"]


def test_run_returning_none_is_an_error():
    result = measure_pure_component_density(
        EIS, "/tmp/wd", parameterize_fn=_fake_parameterize,
        run_npt_fn=lambda ps, wd: None, pack_fn=_fake_pack)
    assert "error" in result and "no density" in result["error"]
