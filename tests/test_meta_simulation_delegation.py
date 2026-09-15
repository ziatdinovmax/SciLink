"""Meta-agent simulation delegation: registration + ase-free lazy import.

Both checks run in a fresh subprocess so the ``ase``-blocking test and the
"sim orchestrator not yet imported" assertion are not polluted by other tests
importing ``scilink.agents.sim_agents`` first.
"""

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True,
    )


def test_delegate_to_simulation_registered():
    """The meta registers delegate_to_simulation alongside analysis/planning."""
    r = _run(
        """
        import tempfile
        from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
        m = MetaOrchestratorAgent(base_dir=tempfile.mkdtemp(),
                                  api_key='dummy', base_url='http://x')
        names = [s['function']['name'] for s in m.tools.openai_schemas]
        assert 'delegate_to_simulation' in names, names
        assert 'delegate_to_analysis' in names and 'delegate_to_planning' in names
        spec = next(s for s in m.tools.openai_schemas
                    if s['function']['name'] == 'delegate_to_simulation')
        props = spec['function']['parameters']['properties']
        assert {'task', 'context', 'context_from', 'label'} <= set(props), props
        assert spec['function']['parameters']['required'] == ['task', 'label']
        print('OK')
        """
    )
    assert "OK" in r.stdout, r.stderr


def test_delegate_routes_to_sim_child_with_meta_autonomy():
    """The delegation path creates/uses the sim child, calls run_task with the
    meta's own autonomy, records a ledger entry, and returns a JSON summary.

    Runs inline with a stubbed sim child (pre-seeded), so no engine stack or LLM
    is needed — it exercises the meta-side plumbing, not the sim orchestrator's
    science.
    """
    import json
    import tempfile
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    from scilink.agents.sim_agents.simulation_orchestrator import SimulationMode

    m = MetaOrchestratorAgent(base_dir=tempfile.mkdtemp(),
                              api_key="dummy", base_url="http://x")

    calls = {}

    class _StubSim:
        _agent_label = "Simulation specialist"

        def run_task(self, task, context=None, autonomy=None, **kw):
            calls.update(task=task, context=context, autonomy=autonomy)
            return {
                "status": "success",
                "summary": "built and briefly ran an MD of bulk water",
                "key_findings": ["equilibrated density ~1.0 g/cc"],
                "files_produced": ["/tmp/run.lammps"],
                "structures": [{"path": "/tmp/system.data"}],
                "suggested_followups": [], "warnings": [],
            }

    m._children["simulation"] = _StubSim()   # pre-seed: no real sim stack built

    out = m._delegate("simulation",
                      "Build and briefly run MD of bulk water at 298 K",
                      label="water MD")

    assert calls["task"].startswith("Build and briefly run MD")
    # the meta's autonomy mode is propagated to the specialist's run_task
    assert calls["autonomy"] == SimulationMode[m.meta_mode.name]
    # the delegation is recorded in the ledger under the simulation mode
    assert any(e.get("mode") == "simulation" for e in m._delegation_ledger)
    # a JSON summary flows back to the meta LLM
    assert json.loads(out)["mode"] == "simulation"


def test_meta_imports_without_ase():
    """The meta module stays importable when the optional ``ase`` dep is absent,
    and constructing/importing it does not eagerly load the sim orchestrator."""
    r = _run(
        """
        import builtins, sys
        _orig = builtins.__import__
        def block(name, *a, **k):
            if name == 'ase' or name.startswith('ase.'):
                raise ImportError('ase blocked for test')
            return _orig(name, *a, **k)
        builtins.__import__ = block
        import scilink.agents.meta_agent.meta_orchestrator  # noqa: F401
        assert 'scilink.agents.sim_agents.simulation_orchestrator' not in sys.modules
        print('OK')
        """
    )
    assert "OK" in r.stdout, r.stderr
