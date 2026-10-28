"""How SciLink deploys CHGNet (see ``mlip_tools.deploy`` and mace_deploy.py)."""

DEPLOY = {
    "import_line": "from chgnet.model.dynamics import CHGNetCalculator",
    "construct_expr": "CHGNetCalculator(use_device=DEVICE)",
    "device_env_var": "CHGNET_DEVICE",
    "devices": ["cuda", "cpu", "mps"],
    "default_model": "chgnet",
    "fallback_rank": 40,
    "model_file": "none",
    "notes": "CHGNet universal MPtrj-pretrained model. ASE-only: no LAMMPS pair_style; the MD agent must use the ASE runner.",
    "trained_file": {
        "construct_expr": "CHGNetCalculator(model=__import__('chgnet.model', fromlist=['CHGNet']).CHGNet.from_file({model_file!r}), use_device=DEVICE)",
        "notes": "Trained CHGNet model ({name}). ASE-only: no LAMMPS pair_style.",
    },
}
