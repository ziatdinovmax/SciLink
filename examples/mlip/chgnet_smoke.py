"""
Minimal CHGNet end-to-end smoke (no SciLink agent, no LLM).

Confirms on a fresh node that:
  1. chgnet imports cleanly
  2. CHGNetCalculator constructs (downloads bundled weights on first
     run, cached afterwards)
  3. The calculator returns sensible energies and forces on a small
     periodic structure

Run:
    python examples/mlip/chgnet_smoke.py
    CHGNET_DEVICE=cpu python examples/mlip/chgnet_smoke.py  # force CPU

Exits 0 on success, non-zero with a one-line reason on failure.
"""

import os
import sys


def main() -> int:
    try:
        import chgnet
    except ImportError as exc:
        print(f"FAIL: chgnet not installed ({exc}). "
              "pip install chgnet")
        return 1
    print(f"chgnet version : {getattr(chgnet, '__version__', 'unknown')}")

    try:
        import torch
        from ase.build import bulk
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        print(f"FAIL: missing dep ({exc})")
        return 2

    device = os.environ.get("CHGNET_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device         : {device}")

    # Tiny FCC Cu (4 atoms, periodic) — universal test case
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    print(f"system         : Cu fcc bulk, {len(atoms)} atoms, cubic={atoms.cell.cellpar()[:3]}")

    try:
        calc = CHGNetCalculator(use_device=device)
    except Exception as exc:
        print(f"FAIL: CHGNetCalculator construction error: {exc!r}")
        return 3

    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception as exc:
        print(f"FAIL: forward pass error: {exc!r}")
        return 4

    print(f"energy         : {energy:.6f} eV")
    print(f"forces shape   : {forces.shape}")
    print(f"forces max|.|  : {abs(forces).max():.4e} eV/Å")

    if abs(forces).max() > 1.0:
        # equilibrium fcc Cu should have ~0 forces; high values
        # indicate the calculator misbehaved or wrong elements were
        # assigned
        print("WARN: max force > 1 eV/Å on equilibrium-looking Cu; "
              "check calculator state.")

    print()
    print("OK -- chgnet end-to-end smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
