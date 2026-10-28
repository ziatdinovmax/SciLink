"""How SciLink deploys MACE models (see ``mlip_tools.deploy``).

Runnable deploy recipes live in package ``.py`` files, never in skill
markdown, so markdown stays non-executable. The skill's ``pretrained:``
frontmatter lists the models (with their ``params``, which fill the
``{loader}`` / ``{size}`` / ``{kwargs}`` fields below) and the elements
each was trained on. Templates are ``str.format`` strings; the calculator
expression may use ``DEVICE``, which the generated run script defines.
Importing this module has no side effects (no MACE import).
"""

DEPLOY = {
    "import_line": "from mace.calculators import {loader}",
    "construct_expr": "{loader}(model={size!r}, device=DEVICE, default_dtype='float64', **{kwargs!r})",
    "device_env_var": "MACE_DEVICE",
    "devices": ["cuda", "cpu", "mps"],
    "defaults": {"kwargs": {}},
    "default_model": "mace-mp-0",
    # Lower = tried first when the picker must fall back without an LLM pick.
    "fallback_rank": 10,
    # LAMMPS needs an on-disk model: save the loaded torch model to working_dir.
    "model_file": "torch_save",
    "notes": "MACE {name} ({size}, {domain}). LAMMPS-capable via pair_style mace (single node).",
    "trained_file": {
        "import_line": "from mace.calculators import MACECalculator",
        "construct_expr": "MACECalculator(model_paths=[{model_file!r}], device=DEVICE, default_dtype='float64')",
        "notes": "Trained MACE model ({name}). LAMMPS-capable via pair_style mace (single node).",
    },
}
