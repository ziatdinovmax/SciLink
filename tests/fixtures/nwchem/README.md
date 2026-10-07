# NWChem output fixtures

Real NWChem output captured so parsers are tested against actual module wording,
not a guessed format.

- `h2o_b3lyp_smoketest.out` — DFT single point (activates the `snapshot_run`
  layer-3 parse test in `tests/test_molecular_qc_output.py`).
- `h2o_tce_ccsdt_7.2.3.out` — excerpt, TCE CCSD(T) (`CCSD(T) total energy /
  hartree = ...`).
- `h2o_direct_mp2_ccsdt_7.2.3.out` — excerpt, direct MP2 + classic CCSD(T)
  (`Total MP2 energy ...`, `Total CCSD(T) energy: ...`).

The two correlated excerpts were captured from NWChem 7.2.3 (water / 3-21G) on
Deception 2026-10-07 and drive the `read_convergence_observable` correlated-
energy tests in `tests/test_nwchem_convergence_hooks.py` — the two module
families write the total-energy line differently, and the sweep must read the
correlated total (MP2/CCSD(T)), not the SCF reference.

Drop any additional real NWChem `*.out` here (e.g. `dipropylamine_opt.out`) to
exercise more of the parser; the `cclib`-based tests also need `cclib` installed.

Why this is empty: the project archive's `3_DFT/nwchem_jobs/` contains only
`_rdkit.xyz` **inputs** — NWChem was never run — so no real output was
available to capture as a fixture. One short NWChem run (geometry opt or even a
single point on a small amine) produces a suitable `.out`. The test also needs
`cclib` installed in the test environment.
"""
