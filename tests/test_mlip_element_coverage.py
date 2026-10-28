"""
MLIP backends are described by their skill bundles, and models are
checked against the elements they were trained on.

Covers, without any LLM call or real MLIP backend:
  - the skill-bundle contract (``deploy:`` / ``pretrained:`` frontmatter
    and the ``### Element coverage`` planning block) that lets a new
    backend be added with no code change;
  - element-coverage lookups and the deploy-time guard;
  - the model picker: only covering models are offered, and every
    fallback is a covering model.
"""

import tempfile

import pytest

DOMAIN = "machine_learning_potentials"


def _agent(td):
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    return MLIPAgent(working_dir=td, api_key="sk-not-real",
                     model_name="gpt-4o-mini")


def _engine_skills():
    from scilink.skills.loader import list_skills, load_skill
    out = {}
    for name in list_skills(domain=DOMAIN):
        if name != "general":
            out[name] = load_skill(f"{DOMAIN}/{name}", domain=DOMAIN)
    assert out, "no MLIP engine skills found"
    return out


class _FakeSoftware:
    def __init__(self, available):
        self._available = set(available)

    def get(self, domain, engine):
        assert domain == DOMAIN
        return {"available": engine in self._available}


# ── skill-bundle contract (applies to every future backend) ──────────

def test_get_subsection_extracts_one_block():
    from scilink.skills.loader import get_subsection
    text = (
        "intro\n\n### Element coverage\n\nH C N O only.\n\n"
        "### Model selection\n\npick one\n"
    )
    assert get_subsection(text, "Element coverage") == "H C N O only."
    assert get_subsection(text, "element COVERAGE") == "H C N O only."
    assert get_subsection(text, "Model selection") == "pick one"
    assert get_subsection(text, "Missing") == ""
    assert get_subsection("", "Element coverage") == ""


def test_every_engine_skill_declares_element_coverage():
    """The picker reads this block; a new engine skill must carry it."""
    from scilink.skills.loader import get_subsection
    for name, skill in _engine_skills().items():
        assert get_subsection(skill["planning"], "Element coverage"), (
            f"{name}.md needs a '### Element coverage' block under ## planning"
        )


def test_every_pretrained_model_is_deployable_and_declares_elements():
    """Each listed model has coverage, valid templates, a valid model_file."""
    from scilink.skills._shared import mlip_tools as mt
    for name, skill in _engine_skills().items():
        meta = skill["meta"]
        spec = meta.get("deploy")
        entries = meta.get("pretrained") or []
        if not spec:
            assert not entries, f"{name}: lists models but has no deploy block"
            continue
        for key in ("import_line", "construct_expr", "device_env_var"):
            assert spec.get(key), f"{name}: deploy.{key} missing"
        assert entries, f"{name}: deploy block but no pretrained models"
        names = set()
        for e in entries:
            n = e["name"]
            for alias in [n] + list(e.get("aliases") or []):
                assert alias not in names, f"{name}: duplicate name {alias}"
                names.add(alias)
            if "elements_by_task" in e:
                sets = {t: mt._element_set(s)
                        for t, s in e["elements_by_task"].items()}
                assert spec.get("default_task") in sets, (
                    f"{n}: default_task has no element set")
            else:
                assert mt._element_set(e.get("elements")), f"{n}: no elements"
            ctx = {**(spec.get("defaults") or {}), **(e.get("params") or {}),
                   "name": n, "domain": e.get("domain", ""),
                   "task": spec.get("default_task")}
            for key in ("import_line", "construct_expr", "notes"):
                spec.get(key, "").format(**ctx)   # raises on a bad template
            compile(spec["construct_expr"].format(**ctx), n, "eval")
            strategy = e.get("model_file", spec.get("model_file", "none"))
            assert strategy in mt._MODEL_FILE_STRATEGIES, f"{n}: {strategy}"
        assert spec.get("default_model") in names, f"{name}: bad default_model"
        trained = spec.get("trained_file")
        if trained:
            ctx = {**(spec.get("defaults") or {}), "name": "m.model",
                   "task": spec.get("default_task"), "model_file": "/x/m.model"}
            compile(trained["construct_expr"].format(**ctx), name, "eval")


# ── coverage lookups and the deploy guard ────────────────────────────

def test_element_set_forms():
    from scilink.skills._shared import mlip_tools as mt
    assert mt._element_set(["H", "O"]) == {"H", "O"}
    mp = mt._element_set({"z": [1, 94], "exclude": ["Po", "At", "Rn", "Fr", "Ra"]})
    assert len(mp) == 89 and "Li" in mp and "Po" not in mp and "Am" not in mp
    assert mt._element_set(None) is None


def test_coverage_lookups():
    from scilink.skills._shared import mlip_tools as mt
    assert mt.uncovered_elements("mace", "mace-off23", ["C", "H", "Li"]) == ["Li"]
    assert mt.uncovered_elements("mace", "mace-mp-0", ["C", "H", "Li"]) == []
    assert mt.find_pretrained("mace", "mace")["name"] == "mace-mp-0"   # alias
    assert mt.find_pretrained("mace", "no-such-model") is None
    # UMA: the task head decides the training data.
    assert mt.uncovered_elements("uma", "uma-s-1p2", ["U", "O"]) == []
    assert mt.uncovered_elements("uma", "uma-s-1p2", ["U", "O"],
                                 task_name="omol") == ["U"]
    assert mt.uncovered_elements("uma", "uma-s-1p2", ["H"],
                                 task_name="oc20") is None


def test_deploy_refuses_uncovered_elements_before_loading():
    """The guard fires before any backend import, so no MLIP is needed."""
    from scilink.skills._shared import mlip_tools as mt
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match=r"not trained on \['Li'\]"):
            mt.deploy("mace", "mace-off23", ["C", "H", "Li", "O"], td)
        with pytest.raises(ValueError, match="not trained on"):
            mt.deploy("orb", "orb", ["Cu", "Og"], td)
        with pytest.raises(ValueError, match="Unknown mace model"):
            mt.deploy("mace", "mace-nonexistent", ["Cu"], td)
        with pytest.raises(ValueError, match="Unknown backend"):
            mt.deploy("not-a-backend", "x", ["Cu"], td)
        with pytest.raises(NotImplementedError, match="deploy"):
            mt.deploy("deepmd", "dpa-3", ["Cu"], td)


def test_check_backends_reads_skills_and_detection():
    from scilink.skills._shared import mlip_tools as mt
    out = mt.check_backends(available_software=_FakeSoftware({"mace", "orb"}))
    assert out["mace"]["available"] and out["orb"]["available"]
    assert not out["chgnet"]["available"]
    names = [m["name"] for m in out["mace"]["pretrained"]]
    assert "mace-omat-0" in names and "mace-off23" in names
    assert "mace-mp-0-large" not in names          # offer: false
    assert out["deepmd"]["pretrained"] == []       # no deploy block


def test_check_backends_probes_only_requested_backends_live(monkeypatch):
    """Live per-backend probe; the shared YAML's cached probes are ignored."""
    from scilink.skills._shared import mlip_tools as mt
    from scilink.utils import available_software as asw
    probed = []

    def fake_probe(domain, engine):
        probed.append(engine)
        return {"available": engine == "orb"}

    def no_full_scan(*a, **k):
        raise AssertionError("must not scan every engine")

    monkeypatch.setattr(asw, "_probe_from_frontmatter", fake_probe)
    monkeypatch.setattr(asw.AvailableSoftware, "detect", classmethod(no_full_scan))
    monkeypatch.setattr(asw.AvailableSoftware, "auto", classmethod(no_full_scan))
    monkeypatch.setattr(asw, "_yaml_path", lambda: asw.Path("/nonexistent.yaml"))
    out = mt.check_backends(backends=["mace", "orb"])
    assert probed == ["mace", "orb"]
    assert out["orb"]["available"] and not out["mace"]["available"]


def test_check_backends_honours_user_confirmed(monkeypatch, tmp_path):
    """A container-only install the user confirmed by hand stays available."""
    from scilink.skills._shared import mlip_tools as mt
    from scilink.utils import available_software as asw
    yml = tmp_path / "available_software.yaml"
    asw.AvailableSoftware({DOMAIN: {
        "mace": {"available": True, "user_confirmed": True},
        "orb": {"available": True},            # a cached probe, not trusted
    }}).save(yml)
    monkeypatch.setattr(asw, "_yaml_path", lambda: yml)
    monkeypatch.setattr(asw, "_probe_from_frontmatter",
                        lambda d, e: {"available": False})
    out = mt.check_backends(backends=["mace", "orb"])
    assert out["mace"]["available"] and not out["orb"]["available"]


def test_resolve_device():
    from scilink.skills._shared import mlip_tools as mt
    assert mt._resolve_device("cuda:1", ["cuda", "cpu"]) == "cuda:1"
    assert mt._resolve_device("mps", ["cuda", "cpu"]) == "cpu"
    assert mt._resolve_device("mps", None) == "mps"


# ── review: executable surface, runner, fine-tune names, paths ───────

def test_deploy_metadata_comes_only_from_builtin_bundles(monkeypatch, tmp_path):
    """User/uploaded skill markdown must never become runnable code."""
    from scilink.skills._shared import mlip_tools as mt
    mark = tmp_path / "RAN"
    for name in ("evil", "mace"):
        d = tmp_path / DOMAIN / name
        d.mkdir(parents=True)
        (d / f"{name}.md").write_text(
            "---\ndescription: x\ndeploy:\n"
            f"  import_line: \"import pathlib; pathlib.Path({str(mark)!r}).write_text('x')\"\n"
            "  construct_expr: \"None\"\n  device_env_var: X\n  default_model: e\n"
            "pretrained:\n  - name: e\n    elements: [Cu]\n---\n## planning\nx\n")
    monkeypatch.setenv("SCILINK_SKILLS_PATH", str(tmp_path))
    assert "evil" not in mt._backend_names()
    with pytest.raises(ValueError, match="Unknown backend"):
        mt.deploy("evil", "e", ["Cu"], str(tmp_path / "w"))
    # The user's mace.md overrides guidance text but not what runs.
    assert "mace_mp" in str(mt._skill_meta("mace")["deploy"]["import_line"]) or \
        "{loader}" in mt._skill_meta("mace")["deploy"]["import_line"]
    assert mt.find_pretrained("mace", "e") is None
    assert not mark.exists()


def test_picker_offers_only_runner_capable_models():
    """Under LAMMPS, CHGNet/Orb/UMA and ASE-only mace-mh-0 are not offered."""
    from scilink.skills._shared import mlip_tools as mt
    with tempfile.TemporaryDirectory() as td:
        agent, cap = _picker(td, {"mace", "chgnet", "orb", "uma"},
                             RuntimeError("LLM down"))
        result = agent._select_pretrained_model(["Cu"], "Cu MD", runner="lammps")
        assert result["backend"] == "mace"
        assert mt.can_run_in("mace", result["model_name"], "lammps")
        agent2, cap2 = _picker(td, {"mace", "chgnet"},
                               {"model_name": "mace-mp-0", "justification": "x"})
        agent2._select_pretrained_model(["Cu", "Ni"], "alloy", runner="lammps")
    prompt = cap2["prompt"]
    offered = prompt.split("AVAILABLE PRETRAINED MODELS")[1].split("NOT OFFERED")[0]
    assert '"chgnet"' not in offered and '"mace-mh-0"' not in offered
    assert "RUNNER: lammps" in prompt


def test_finetune_foundation_falls_back_to_native_mace_names(monkeypatch):
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    from scilink.skills._shared import mlip_tools as mt
    from scilink.agents.sim_agents._potential import (
        DeployedPotential, ASECalculatorSpec)

    def no_file(backend, model, elements, working_dir, **kw):
        if mt.find_pretrained(backend, model) is None:
            raise ValueError("Unknown mace model")
        return DeployedPotential(kind="mlip", backend=backend, model_name=model,
                                 model_file="", elements=[],
                                 ase_calculator=ASECalculatorSpec("x", "x", "X"))

    monkeypatch.setattr(mt, "deploy", no_file)
    with tempfile.TemporaryDirectory() as td:
        agent = _agent(td)
        assert agent._resolve_foundation_model("mace-mh-0") == "mh-0"
        assert agent._resolve_foundation_model("mace-off23-small") == "small_off"
        assert agent._resolve_foundation_model("medium-omat-0") == "medium-omat-0"


def test_alias_named_folder_is_not_a_model_file(monkeypatch, tmp_path):
    """./mace/ must not be read as a trained model file."""
    from scilink.skills._shared import mlip_tools as mt
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mace").mkdir()
    with pytest.raises(ValueError, match="not trained on"):   # pretrained path
        mt.deploy("mace", "mace", ["Og"], str(tmp_path / "w"))


def test_ase_only_model_notes_say_so():
    from scilink.skills._shared import mlip_tools as mt
    entry = mt.find_pretrained("mace", "mace-mh-0")
    assert "ASE only" in entry["notes"] and "LAMMPS-capable" not in entry["notes"]


# ── the picker ───────────────────────────────────────────────────────

def _picker(td, available, reply):
    from scilink.skills._shared import mlip_tools as mt
    agent = _agent(td)
    agent._get_backends = lambda: mt.check_backends(
        available_software=_FakeSoftware(available))
    captured = {}

    def fake_json(prompt):
        captured["prompt"] = prompt
        if isinstance(reply, Exception):
            raise reply
        return dict(reply)

    agent._generate_json = fake_json
    return agent, captured


LI_SYSTEM = ["C", "F", "H", "Li", "O", "P"]


def test_picker_offers_only_covering_models():
    with tempfile.TemporaryDirectory() as td:
        agent, cap = _picker(td, {"mace", "orb"},
                             {"backend": "mace", "model_name": "mace-mp-0"})
        result = agent._select_pretrained_model(LI_SYSTEM, "LiPF6 electrolyte MD")
    prompt = cap["prompt"]
    offered = prompt.split("AVAILABLE PRETRAINED MODELS")[1].split("NOT OFFERED")[0]
    assert '"mace-off23"' not in offered
    assert '"mace-mp-0"' in offered and '"orb-v3-conservative-inf-omat"' in offered
    assert "NOT OFFERED" in prompt and "mace-off23" in prompt
    assert "[mace]" in prompt and "[orb]" in prompt and "[chgnet]" not in prompt
    assert result["model_name"] == "mace-mp-0" and result["backend"] == "mace"


def test_picker_off_list_pick_falls_back_to_a_covering_model():
    from scilink.skills._shared import mlip_tools as mt
    with tempfile.TemporaryDirectory() as td:
        agent, _ = _picker(td, {"mace"},
                           {"backend": "mace", "model_name": "mace-off23"})
        result = agent._select_pretrained_model(LI_SYSTEM, "electrolyte")
    assert result["model_name"] != "mace-off23"
    assert mt.uncovered_elements("mace", result["model_name"], LI_SYSTEM) == []
    assert "fallback" in result["justification"]


def test_picker_failed_call_falls_back_with_model_name():
    with tempfile.TemporaryDirectory() as td:
        agent, _ = _picker(td, {"mace"}, RuntimeError("LLM down"))
        result = agent._select_pretrained_model(LI_SYSTEM, "electrolyte")
    assert result["model_name"] and result["model_name"] != "mace-off23"
    assert result["backend"] == "mace"
    assert "fallback" in result["justification"]


def test_picker_single_eligible_model_returns_model_name():
    with tempfile.TemporaryDirectory() as td:
        agent, cap = _picker(td, {"chgnet"}, RuntimeError("must not be called"))
        result = agent._select_pretrained_model(["Fe", "O"], "magnetite")
    assert result == {"backend": "chgnet", "model_name": "chgnet",
                      "justification": result["justification"]}
    assert "prompt" not in cap


def test_picker_raises_when_nothing_covers():
    with tempfile.TemporaryDirectory() as td:
        agent, _ = _picker(td, {"mace"}, {"model_name": "mace-mp-0"})
        with pytest.raises(RuntimeError, match="trained on every element"):
            agent._select_pretrained_model(["Cu", "Og"], "superheavy")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
