"""Capability availability of the script-execution environment (#569).

The planner and the verifier can prescribe — and even hard-mandate — fixes
that call for an optional capability (a package, a model checkpoint, a
GPU) without knowing whether the environment the generated script runs in
actually has it. The load-bearing subtlety: scripts run in the
``sys.executable`` subprocess (``executors.ScriptExecutor``), which can
differ from the process that imported scilink — so the probe below is a
script run *through the executor*, never an import in the parent.

``probe_execution_environment`` runs it once per process (cached by the
interpreter path; a session shares one executor) and returns a plain
dict; ``exec_env_block`` renders that dict as a prompt section the
planner / codegen / verifier read as a deterministic signal, with the
rule that an optional capability may only be prescribed when it is
listed available, advisory-with-fallback, and never hard-mandated when
absent. ``capability_available`` is the programmatic check.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional capabilities worth knowing about before prescribing them. Keyed
# by the import name; the label is what prompts and tools call it.
OPTIONAL_PACKAGES = {
    "segment_anything": "SAM (segment_anything)",
    "torch": "PyTorch",
    "cv2": "OpenCV",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "scipy": "SciPy",
    "lmfit": "lmfit",
    "hyperspy": "HyperSpy",
    "atomai": "AtomAI",
    "pymatgen": "pymatgen",
    "ase": "ASE",
    "mp_api": "Materials Project API client",
    "networkx": "networkx",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PIL": "Pillow",
}

_MARKER = "EXEC_ENV_JSON:"

# Runs in the execution subprocess. Import failures are answers, not errors.
PROBE_SCRIPT = r'''
import importlib, json, os, sys, glob
packages = {}
for name in %(packages)s:
    try:
        mod = importlib.import_module(name)
        packages[name] = getattr(mod, "__version__", None) or "present"
    except Exception as exc:  # noqa: BLE001 - absence is the answer
        packages[name] = None
devices = {"cuda": False, "mps": False}
try:
    import torch
    devices["cuda"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    devices["mps"] = bool(mps is not None and mps.is_available())
except Exception:  # noqa: BLE001
    pass
ckpt_dir = os.path.join(os.path.expanduser("~"), ".cache", "scilink", "checkpoints")
checkpoints = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.pth")))
print("%(marker)s" + json.dumps({
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "packages": packages,
    "devices": devices,
    "sam_checkpoints": checkpoints,
    "checkpoint_dir": ckpt_dir,
}))
'''

_cache: Dict[str, Dict[str, Any]] = {}


def _cache_key(executor: Any) -> str:
    # The executor runs scripts under sys.executable; a different executor
    # object in the same process sees the same environment.
    return str(getattr(executor, "python_executable", None) or sys.executable)


def probe_execution_environment(executor: Any = None, *, force: bool = False,
                                timeout: int = 120) -> Dict[str, Any]:
    """Probe the script-execution environment once (cached per interpreter).

    Never raises: a failed probe yields ``{"status": "unknown", ...}`` so a
    caller degrades to "unknown, do not hard-mandate" rather than to a
    crash. Pass the agent's ``ScriptExecutor`` so the probe runs exactly
    where the generated scripts run; without one a default executor is
    built (same interpreter).
    """
    key = _cache_key(executor)
    if not force and key in _cache:
        return _cache[key]
    if executor is None:
        try:
            from ...executors import ScriptExecutor
            executor = ScriptExecutor(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            env = {"status": "unknown", "error": f"no executor: {exc}"}
            _cache[key] = env
            return env
    script = PROBE_SCRIPT % {"packages": json.dumps(sorted(OPTIONAL_PACKAGES)),
                             "marker": _MARKER}
    env: Dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(prefix="scilink_env_probe_") as wd:
            out = executor.execute_script(script, working_dir=wd, timeout=timeout)
        env = _parse_probe_output(out)
    except Exception as exc:  # noqa: BLE001
        env = {"status": "unknown", "error": str(exc)[:300]}
    _cache[key] = env
    if env.get("status") == "ok":
        missing = [n for n, v in env["packages"].items() if v is None]
        logger.info(f"Execution environment probed: {env['python']} — "
                    f"{len(env['packages']) - len(missing)} optional packages present, "
                    f"missing: {', '.join(missing) or 'none'}; devices "
                    f"cuda={env['devices'].get('cuda')} mps={env['devices'].get('mps')}")
    else:
        logger.warning(f"Execution environment probe failed: {env.get('error')}")
    return env


def _parse_probe_output(out: Any) -> Dict[str, Any]:
    text = ""
    if isinstance(out, dict):
        if out.get("status") == "error":
            return {"status": "unknown",
                    "error": str(out.get("message") or out.get("error") or "probe failed")[:300]}
        text = str(out.get("stdout") or out.get("output") or "")
    else:
        text = str(out or "")
    m = re.search(re.escape(_MARKER) + r"(\{.*\})", text)
    if not m:
        return {"status": "unknown", "error": "probe printed no result"}
    data = json.loads(m.group(1))
    data["status"] = "ok"
    return data


def capability_available(env: Optional[Dict[str, Any]], name: str) -> Optional[bool]:
    """True / False for a probed package or device (``"cuda"``, ``"mps"``,
    ``"gpu"``, ``"sam_checkpoint"``); None when the environment is unknown."""
    if not env or env.get("status") != "ok":
        return None
    if name in ("cuda", "mps"):
        return bool(env.get("devices", {}).get(name))
    if name == "gpu":
        d = env.get("devices", {})
        return bool(d.get("cuda") or d.get("mps"))
    if name == "sam_checkpoint":
        return bool(env.get("sam_checkpoints"))
    return env.get("packages", {}).get(name) is not None


def exec_env_block(env: Optional[Dict[str, Any]] = None, *, executor: Any = None,
                   for_verifier: bool = False) -> str:
    """The prompt section. ``env`` defaults to the cached probe (run now if
    needed). Renders what is present and what is absent, then the rule."""
    if env is None:
        env = probe_execution_environment(executor)
    lines = ["\n## Execution environment (probed in the script-execution interpreter)"]
    if env.get("status") != "ok":
        lines.append(
            "Unknown — the probe did not run. Treat every optional capability "
            "(SAM, GPU, optional packages) as UNVERIFIED: do not hard-mandate any of "
            "them; if you prescribe one, keep a sanctioned fallback path.")
        return "\n".join(lines) + "\n"
    pk = env.get("packages", {})
    present = [f"{OPTIONAL_PACKAGES.get(n, n)} {v}" if v not in (None, "present")
               else OPTIONAL_PACKAGES.get(n, n)
               for n, v in sorted(pk.items()) if v is not None]
    absent = [OPTIONAL_PACKAGES.get(n, n) for n, v in sorted(pk.items()) if v is None]
    d = env.get("devices", {})
    device = "CUDA GPU" if d.get("cuda") else "Apple MPS GPU" if d.get("mps") else "CPU only"
    lines.append(f"- Python: {env.get('python')} ({env.get('python_version')})")
    lines.append(f"- Compute: {device}")
    lines.append(f"- Available optional packages: {', '.join(present) or 'none'}")
    lines.append(f"- ABSENT optional packages: {', '.join(absent) or 'none'}")
    ck = env.get("sam_checkpoints") or []
    sam_ok = pk.get("segment_anything") is not None
    if sam_ok:
        lines.append(f"- SAM checkpoints on disk: {', '.join(ck) if ck else 'none (first use downloads ~2.5 GB for vit_h)'}")
    lines.append(
        "RULE: only prescribe or mandate a capability listed as available. An "
        "absent package, checkpoint or GPU cannot be fixed by the code generator — "
        "do not prescribe it, and never make it the sole method. An optional "
        "capability that IS available is still advisory: prescribe it WITH a "
        "sanctioned fallback (the pipeline must still complete if the tool "
        "returns status='unavailable' or fails)."
        + (" A tool that reports status='unavailable' has said all it can — do not "
           "re-prescribe it; change method." if for_verifier else ""))
    return "\n".join(lines) + "\n"


def reset_cache() -> None:
    _cache.clear()
