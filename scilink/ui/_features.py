"""Feature-flag probes for the Streamlit UI.

Today there is one flag: whether simulate mode (Simulate mode button,
HPC sidebar, Simulations tab) is exposed. The flag resolves in this
order:

  1. ``SCILINK_SIMULATE`` env var, if set — ``0/false/no/off`` forces
     off, ``1/true/yes/on`` forces on. Other values fall through.
  2. Otherwise, probe whether ``ase`` can be imported. The simulation
     agents need it, and it is a core dependency, so this is on unless
     someone removed it.

The probe used to check ``paramiko`` as a sign that the user installed
``scilink[sim]``. Both are core dependencies now, and simulate mode
needs nothing from that extra.

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
        import ase  # noqa: F401
    except ImportError:
        return False
    return True
