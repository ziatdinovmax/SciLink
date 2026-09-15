# Polycrystalline microstructure — grain analysis demo

A bright-field optical micrograph of an etched cross-section of
polycrystalline stainless steel 304 (100 × 100 µm field of view).
A general-purpose microstructure example for the image-analysis agent.
No specific goal required, jsut let the agent figure out the analysis goals on itw own.

## Run

**Streamlit UI** (recommended):

```bash
scilink ui
```

— upload `image.npy` and `image.json`.

**CLI**:

```bash
scilink analyze --data examples/polycrystalline_grains_demo/image.npy \
                --metadata examples/polycrystalline_grains_demo/image.json
```

## What to expect

The agent should recognize the polycrystalline morphology, segment the
grain-boundary network, and report on grain-size statistics and grain
shape.
