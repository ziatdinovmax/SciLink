Each example lives in its own subfolder, with a short README, a data
file, and a metadata sidecar:

- `polycrystalline_grains_demo/` — optical microscopy, grain analysis
- `ferroelectric_polarization_demo/` — atomic-resolution HAADF-STEM, per-unit-cell polarization + domain walls
- `eels_plasmons_demo/` — hyperspectral EELS, plasmon mapping
- `eels_identification_demo/` — synthetic core-loss EELS, curve-fitting agent in identification mode
- `planning_agent_demo/` — planning orchestrator, critical-material recovery from produced water

To launch a chat session and pick the data interactively, run

```bash
scilink analyze
```

To generate an example AMBER force field, run

```bash
scilink prepare-ff --test --solvate --goal "Test solvated system"
```
