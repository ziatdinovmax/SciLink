# EELS — plasmonic hyperspectral demo

A low-loss EELS hyperspectral datacube of self-assembled monolayers of
fluorine and tin doped indium oxide (FT:IO) nanocrystal arrays. The energy window is the localized
surface-plasmon-resonance (LSPR) region of the doped In₂O₃ system.

The example exercises the hyperspectral agent.

## Run

**Streamlit UI** (recommended):

```bash
scilink ui
```

— upload `datacube.npy` and `datacube.json`.

**CLI**:

```bash
scilink analyze --data examples/eels_plasmons_demo/datacube.npy \
                --metadata examples/eels_plasmons_demo/datacube.json
```

## What to expect

The agent should recognize the hyperspectral structure, identify the
LSPR feature in the low-loss window, and produce spatial maps of the
plasmon intensity / peak position across the nanocrystal array.
