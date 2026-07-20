# XRD identification — dependencies & installation

The powder-XRD identification stack is layered so that each capability tier
pulls only what it needs. Most users stop at Tier 2 and never compile
anything.

```
Tier 1  pip install scilink[structure-matching]     search/simulate/score + fingerprint tools
Tier 2  scilink fetch-xrd-library                   one-time reference-library download
Tier 3  pip install "GSAS-II @ git+..."  (needs Fortran)  GSAS-II: profiles, indexing, Le Bail, Rietveld
```

## Tool → dependency matrix

| Tool | Needs | Tier |
|---|---|---|
| `search_structures` (COD / MP / local CIF) | pymatgen (+ `MP_API_KEY` for MP) | 1 |
| `simulate_xrd_pattern` (engine=`pymatgen`) | pymatgen | 1 |
| `score_xrd_match_fast` / `_robust` / `_multiphase` | pymatgen, pulp | 1 |
| `extract_peaks`, `resolve_wavelength` | scipy/numpy | 1 |
| `calibrate_zero` (internal standard) | numpy only | 1 |
| `determine_space_group` (absence analysis) | numpy only | 1 |
| `search_match_pattern` (fingerprint ID) | pymatgen, pyarrow **+ the library (Tier 2)** | 1+2 |
| `build_fingerprint_library` | pymatgen, pyarrow | 1 |
| `simulate_xrd_pattern` (engine=`gsas`) | GSAS-II | 3 |
| `index_pattern` (autoindexing) | GSAS-II | 3 |
| `validate_cell_lebail` (cell arbiter) | GSAS-II | 3 |
| `refine_rietveld` | GSAS-II (+ pymatgen for CIF canonicalization) | 3 |

The two identification workflows map onto the tiers:

- **Known-phase (fingerprint) route** — `calibrate_zero → extract_peaks →
  search_match_pattern → simulate+score confirm → refine_rietveld`. Everything
  up to the final Rietveld runs on Tiers 1–2 (pure Python, works offline).
- **New-phase (discovery) route** — `calibrate_zero → extract_peaks →
  index_pattern → validate_cell_lebail → determine_space_group →
  refine_rietveld`. The indexing / Le Bail / Rietveld steps are Tier 3.

## Tier 1 — structure-matching extra

```bash
pip install "scilink[structure-matching]"
```

Pulls `pymatgen` (pinned `<2026` — the 2026 line breaks the mp-api import
chain), `mp-api`, `pulp`, `pyarrow`. Pure Python; no compilers.

Optional environment:

| Variable | Purpose |
|---|---|
| `MP_API_KEY` / `MATERIALS_PROJECT_API_KEY` | enables the Materials Project backend |
| `SCILINK_LOCAL_CIF_DIR` | a local CIF directory as a search backend |

## Tier 2 — the fingerprint reference library

`search_match_pattern` matches measured peaks against a **precomputed**
library of reference patterns (COD-derived, d-spacings + relative
intensities). Three ways to get one, in order of typical preference:

1. **Fetch the prebuilt artifact** (recommended; one-time, a few hundred MB):

   ```bash
   scilink fetch-xrd-library
   ```

   Installs to `~/.scilink/xrd_fingerprints/cod_fingerprints.parquet`
   (the persistent per-user store — survives pip upgrades), checksum-verified.
   The download is deliberately explicit, never an implicit side effect of an
   analysis run.

2. **Point at a shared copy** (HPC / group installs):

   ```bash
   export SCILINK_XRD_FINGERPRINT_DB=/shared/data/cod_fingerprints.parquet
   ```

3. **Build your own** — from the COD mirror or ANY CIF collection (private
   structures, licensed ICSD dumps, a lab library):

   ```bash
   # full COD (~500k CIFs, ~100 GB downloaded shard-by-shard and deleted
   # after fingerprinting; hours of wall time; resumable):
   python scripts/build_cod_fingerprints.py \
       --work ~/cod_fp_build --out ~/cod_fingerprints.parquet

   # any local CIF directory (minutes for a few thousand CIFs):
   python -c "from scilink.skills.structure_matching.xrd.fingerprint import \
       build_fingerprint_library as b; print(b('/my/cifs', 'my_lib.parquet'))"
   ```

   Builder curation knobs (all documented in the function): `max_sites`
   (default 500 — larger cells are skipped and counted; kinematic pattern
   cost scales ~ sites × reflections and macromolecular fingerprints are
   useless for powder ID), `min_lines`, `two_theta_max`, duplicate collapse
   by formula+cell, disorder flagging.

Resolution order at query time: explicit `library_path` argument →
`SCILINK_XRD_FINGERPRINT_DB` → the per-user store. A missing library raises
an error naming all three routes.

Licensing: COD data are open (cite COD); redistributing derived fingerprints
with attribution follows the same practice as Match!/QualX COD reference DBs.
ICDD-derived data must never be redistributed.

## Tier 3 — GSAS-II extra (profiles, indexing, Le Bail, Rietveld)

GSAS-II is built **from source** at install time, so a Fortran toolchain must
exist first. The verified recipe (conda supplies the compilers):

```bash
conda create -n scilink python=3.12
conda activate scilink
conda install -c conda-forge fortran-compiler meson ninja
pip install "scilink[structure-matching]"   # BEFORE GSAS-II: settles the numpy ABI
pip install "GSAS-II @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git"
```

Verify:

```bash
python -c "from GSASII import GSASIIscriptable; print('GSAS-II OK')"
```

### Known pitfalls (each observed and diagnosed in practice)

| Symptom | Cause / fix |
|---|---|
| `ERROR: Compiler gfortran cannot compile programs` during pip install | The conda compiler activation scripts did not run (`CONDA_BUILD_SYSROOT` unset). Happens with bare `conda run -n env pip install ...`. Fix: `conda activate` the env in a shell (or `source .../etc/profile.d/conda.sh && conda activate env`), then pip install. |
| `Unknown compiler(s): gfortran` | No Fortran toolchain at all — run the `conda install -c conda-forge fortran-compiler meson ninja` step. |
| numpy ABI errors importing GSAS-II | GSAS-II was compiled against a different numpy than the one now installed. Fix: install scilink (which pins the scientific stack) **before** GSAS-II; reinstall GSAS-II after major numpy changes. |
| `ModuleNotFoundError: GSASII` but tools "registered" | Expected degradation: the Tier-3 tools stay importable and registry-visible without GSAS-II and raise an actionable error naming this recipe only when actually called. |
| GSAS-II console noise (`config.ini does not exist`, refinement logs) | Harmless; the tools capture or tolerate it. |

GSAS-II is deliberately **not a scilink extra**: it is not on PyPI, and PyPI
rejects package metadata that carries a direct URL dependency (so a `[gsas]`
extra would block publishing scilink itself). It must be installed manually
with the command above.

## Maintainers — refreshing the prebuilt artifact

COD grows continuously; refresh the artifact periodically:

1. Run `scripts/build_cod_fingerprints.py` (shard-at-a-time, resumable,
   deletes CIFs after fingerprinting — peak disk ≈ one shard ≈ 20 GB).
2. Upload the parquet to a GitHub Release tagged `xrd-fplib-vN` (or Zenodo for
   a DOI), record its `sha256`.
3. Update `DEFAULT_LIBRARY_URL` / `DEFAULT_LIBRARY_SHA256` in
   `scilink/skills/structure_matching/xrd/fingerprint.py`.
