"""How SciLink deploys Orb (see ``mlip_tools.deploy`` and mace_deploy.py).

API as of orb-models 0.7.0: ``pretrained.<model>()`` returns
``(model, atoms_adapter)``.
"""

DEPLOY = {
    "import_line": "from orb_models.forcefield import pretrained; from orb_models.forcefield.inference.calculator import ORBCalculator",
    "construct_expr": "ORBCalculator(*pretrained.{loader}(device=DEVICE, precision='float32-high'), device=DEVICE)",
    "device_env_var": "ORB_DEVICE",
    "devices": ["cuda", "cpu"],
    "default_model": "orb-v3-conservative-inf-omat",
    "fallback_rank": 30,
    "model_file": "none",
    "notes": "Orb {name}. ASE-only: the MD agent must use the ASE runner.",
}
