"""Feature-flag probes for the Streamlit UI.

Today there is one flag: whether simulate mode (Simulate mode button,
HPC sidebar, Simulations tab) is exposed. The flag resolves in this
order:

  1. ``SCILINK_SIMULATE`` env var, if set — ``0/false/no/off`` forces
     off, ``1/true/yes/on`` forces on. Other values fall through.
  2. Otherwise, probe whether ``paramiko`` (the [sim]-exclusive SSH
     dependency) can be imported.

The probe used to also require ``ase``, but ase can ride along via
unrelated libraries (e.g. ``sidpy`` from the pycroscopy ecosystem), so
``ase`` being importable is not a reliable signal that the user opted
into ``scilink[sim]``. ``paramiko`` has no such cross-ecosystem
overlap in our user base.

Note: the result is cached for the process lifetime (``@lru_cache``).
Changing ``SCILINK_SIMULATE`` requires restarting Streamlit.
"""
from __future__ import annotations

import os
from functools import lru_cache


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def simulate_enabled() -> bool:
    override = os.environ.get("SCILINK_SIMULATE", "").strip().lower()
    if override in _TRUE:
        return True
    if override in _FALSE:
        return False
    try:
        import paramiko  # noqa: F401
    except ImportError:
        return False
    return True
