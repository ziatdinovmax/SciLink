"""
The DeployedPotential descriptor — the contract between the agent
that *produces* an interatomic potential and the agent that *runs a
simulation* with it.

Architectural rationale (see project_mlip_md_delegation memory):
an MLIP is just a potential. Running MD with MACE is the same
orchestration problem as running MD with EAM — and MDSimulationAgent
already solves that, request-driven and skill-backed. So MLIPAgent's
job ends at "here is a deployed potential"; MDSimulationAgent takes
that descriptor and generates the actual run (relax / MD / whatever
the research goal asks for).

Extensibility design — N + M, not N × M:

  DeployedPotential is **engine-neutral**. It carries only facts about
  the potential itself: which backend family it is, where the model
  file lives, what elements it covers, and how to construct it as an
  ASE calculator. ASE is the universal runner — every MLIP backend
  definitionally has an ASE calculator — so ``ase_calculator`` is the
  one runner-specific thing that genuinely belongs here.

  Engine-specific integration (LAMMPS pair_style, a future GROMACS
  mechanism, OpenMM's TorchForce, ...) does NOT live on this
  descriptor. It lives with the engine — each MD engine's tools
  module answers "given a potential of family X, here's how *I* run
  it" at runtime. That keeps the cost of a new MD engine to "add one
  tools module" and the cost of a new MLIP backend to "add one
  ASECalculatorSpec", with neither side enumerating the other.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ASECalculatorSpec:
    """How an ASE script should construct this potential's calculator.

    The MD agent's ASE-runner template fills the calculator
    construction from these three fields and never imports any MLIP
    backend itself — which is exactly why one generic ASE-script
    template covers every backend. Adding a backend is declaring one
    of these (three strings), not writing a new script generator.

    Fields
    ------
    import_line:
        The exact import statement, e.g.
        ``"from mace.calculators import mace_mp"``.
    construct_expr:
        A Python expression that builds the calculator, referencing
        the module-level ``DEVICE`` name the template defines, e.g.
        ``"mace_mp(model='medium', device=DEVICE, default_dtype='float64')"``.
    device_env_var:
        Env var the generated script reads to override the device at
        run time, e.g. ``"MACE_DEVICE"``.
    """
    import_line: str
    construct_expr: str
    device_env_var: str


@dataclass
class DeployedPotential:
    """A ready-to-use interatomic potential, handed from a
    potential-producing agent (today: MLIPAgent) to MDSimulationAgent.

    Engine-neutral by construction — see the module docstring. The MD
    agent reads ``ase_calculator`` directly for the ASE runner and
    passes the whole descriptor to an engine's tools module for any
    engine-specific runner.

    Attributes
    ----------
    kind:
        "mlip" today; "classical" is a future possibility if classical
        force fields ever get the same producer/runner split.
    backend:
        Engine-family keyword the producing agent used — "mace",
        "chgnet", "nequip", ... An MD engine's tools module uses this
        to decide whether and how it can integrate the potential.
        The MD agent itself does not branch on it.
    model_name:
        Human-readable model identifier (e.g. "mace-mp-0", "chgnet").
    model_file:
        Path to the on-disk model artifact, or "" for backends that
        bundle their weights (CHGNet). Passed through to engine tools
        modules that need it (e.g. LAMMPS pair_coeff).
    elements:
        Chemical elements the structure is expected to contain, in the
        order an engine's pair_coeff-style line would want them.
    ase_calculator:
        How to construct the calculator inside an ASE script. Always
        present — this is the universal runner.
    notes:
        Free-text provenance from the producing agent (model-selection
        rationale, caveats) — surfaced in the run README.
    """
    kind: str
    backend: str
    model_name: str
    model_file: str
    elements: List[str]
    ase_calculator: ASECalculatorSpec
    notes: str = ""
