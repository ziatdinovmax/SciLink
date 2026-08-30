"""Cross-restart persistence for the LAMMPS / VASP wizards' infrastructure
fields (HPC env, SLURM settings, paths, etc.).

Scientific fields (structure description, research goal) are deliberately
not persisted -- they change per simulation. Secrets (FutureHouse key,
LLM API key) live in the sidebar config, not in wizard state, and are
also not saved here. The state file is a plain JSON cache; corruption,
schema drift, or missing files all fall back to widget defaults.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st


_STATE_PATH = Path.home() / ".scilink" / "wizard_state.json"

# Widget keys persisted per wizard. Anything not listed here is treated
# as scientific / one-off / secret and intentionally not saved.
_LAMMPS_KEYS: tuple[str, ...] = (
    "hpc_exec_mode",
    "hpc_module_cmds",
    "hpc_lammps_cmd",
    "hpc_python_cmd",
    "hpc_container_runtime",
    "hpc_container_image",
    "hpc_image_tar",
    "hpc_extra_runtime_flags",
    "_hpc_mounts",
    "hpc_struct_source",
    "hpc_ff_source",
    "hpc_work_dir",
    "hpc_slurm_account",
    "hpc_slurm_time",
    "hpc_slurm_nodes",
    "hpc_slurm_ntasks",
    "hpc_slurm_jname",
    "hpc_slurm_partition",
    "hpc_extra_sbatch",
    "hpc_api_key_env",
    "hpc_max_attempts",
    "hpc_stage_timeout",
)

_VASP_KEYS: tuple[str, ...] = (
    "vasp_incar_method",
    "vasp_command",
    "vasp_module_commands",
    "vasp_pseudo_dir",
    "vasp_remote_work_dir",
    "vasp_slurm_account",
    "vasp_slurm_time",
    "vasp_slurm_nodes",
    "vasp_slurm_ntasks",
    "vasp_slurm_jname",
    "vasp_slurm_partition",
    "vasp_extra_sbatch",
)


def _load() -> Dict[str, Dict[str, Any]]:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception as exc:
        logging.warning("Could not read %s: %s", _STATE_PATH, exc)
        return {}


def _save(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        from ...utils.text_io import atomic_write_text
        atomic_write_text(_STATE_PATH, json.dumps(data, indent=2))
    except Exception as exc:
        logging.warning("Could not write %s: %s", _STATE_PATH, exc)


def _apply(scope: str) -> None:
    """Pre-fill st.session_state with saved values for `scope`. Safe to call
    on every Configure render; only sets keys that aren't already in
    session state, so it never clobbers a value the user has typed during
    the current session."""
    saved = _load().get(scope, {})
    for key, value in saved.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_current(scope: str, keys: Iterable[str]) -> None:
    snap = {key: st.session_state[key] for key in keys if key in st.session_state}
    if not snap:
        return
    data = _load()
    data[scope] = snap
    _save(data)


def apply_lammps_defaults() -> None:
    _apply("lammps")


def apply_vasp_defaults() -> None:
    _apply("vasp")


def save_lammps() -> None:
    _save_current("lammps", _LAMMPS_KEYS)


def save_vasp() -> None:
    _save_current("vasp", _VASP_KEYS)
