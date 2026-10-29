"""How SciLink deploys UMA (see ``mlip_tools.deploy`` and mace_deploy.py).

fairchem's ``get_predict_unit`` accepts only "cuda" or "cpu", so an indexed
device ("cuda:1") is passed as "cuda"; pick the GPU with
CUDA_VISIBLE_DEVICES.
"""

DEPLOY = {
    "import_line": "from fairchem.core import FAIRChemCalculator",
    "construct_expr": "FAIRChemCalculator(__import__('fairchem.core', fromlist=['pretrained_mlip']).pretrained_mlip.get_predict_unit({name!r}, device=('cuda' if str(DEVICE).startswith('cuda') else 'cpu')), task_name={task!r})",
    "device_env_var": "UMA_DEVICE",
    "devices": ["cuda", "cpu"],
    "default_model": "uma-s-1p2",
    # UMA is multi-task: the head sets which training data (and elements) apply.
    "default_task": "omat",
    "fallback_rank": 20,
    "model_file": "none",
    "notes": "UMA {name} (task head '{task}'). ASE-only: no LAMMPS pair_style in the base fairchem package; the MD agent must use the ASE runner. Pass task_name='omol' for molecular systems.",
}
