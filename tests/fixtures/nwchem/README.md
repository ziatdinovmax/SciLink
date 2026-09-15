# NWChem output fixtures (for `snapshot_run` tests)

Drop a real NWChem output log here (e.g. `dipropylamine_opt.out`) to activate
the layer-3 parse test in `tests/test_molecular_qc_output.py`
(`test_snapshot_real_output`), which is skipped until a `*.out` file exists in
this directory.

Why this is empty: the project archive's `3_DFT/nwchem_jobs/` contains only
`_rdkit.xyz` **inputs** — NWChem was never run — so no real output was
available to capture as a fixture. One short NWChem run (geometry opt or even a
single point on a small amine) produces a suitable `.out`. The test also needs
`cclib` installed in the test environment.
"""
