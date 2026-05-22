---
description: Diagnostics live-mode skill — counts peaks in 2-column CSV / whitespace data and emits verdict by peak count. For testing live-mode plumbing (UI, CLI, simulator) without depending on any scientific skill.
live_tick:
  enabled: true
  tick_fn: scilink.skills.diagnostics.live_passthrough.live_tick:passthrough_tick
  data_type: spectrum_1d
  trigger_overrides:
    heartbeat_sec: 30
---
# Diagnostics — Live Passthrough Skill

## overview

A **plumbing test** for SciLink's live-monitoring infrastructure. Not a
scientific skill — does not query databases, does not simulate
patterns, makes no claims about phase identification. It exists so
that:

- The live mode UI / CLI / `LiveSession` / replay can be exercised
  end-to-end without depending on any specific scientific skill.
- Skill authors implementing their own `live_tick` have a minimal
  reference for the contract.
- Reviewers can verify the live-mode infrastructure on a clean branch
  without setting up a structure-matching fixture / API key chain.

## analysis

The tick function:

1. Parses the latest data text as a 2-column CSV (or whitespace-
   delimited) `(x, y)` table.
2. Runs `scipy.signal.find_peaks` on `y` (or a trivial threshold count
   if scipy isn't available) to count peaks.
3. Maps the count to a verdict:

   ```
   0 peaks   → unknown    (no data or instrument just warming up)
   1 peak    → reject     (one peak isn't usually identifiable)
   2-3 peaks → marginal
   ≥4 peaks  → accept
   ```

4. Emits `LiveTickResult` with `primary_metric = peak count`,
   `detected_features = [{position, intensity}, ...]`, and a one-line
   `notes` description.

Pair with `scilink live-simulator <out> --mode directory` (or
`--mode rewrite`) to see verdict transitions as Si Bragg peaks emerge
over time.

## implementation

See `live_tick.py` in this folder. ~50 lines; no external dependencies
beyond numpy/scipy (already required by SciLink).
