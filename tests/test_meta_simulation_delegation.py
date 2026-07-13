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
