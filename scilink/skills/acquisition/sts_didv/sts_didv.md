---
description: Steering scanning tunnelling spectroscopy (dI/dV point spectra with a lock-in) — bias modulation against energy resolution, averaging against drift, tip changes, setpoint effects.
technique: ["scanning tunnelling spectroscopy", "scanning tunneling spectroscopy", STS, "dI/dV", "STM spectroscopy", "tunnelling spectroscopy"]
---

# STS dI/dV acquisition

## Overview

A frame is one dI/dV spectrum against sample bias, measured with a lock-in: a
small AC modulation is added to the bias and the current response at that
frequency is recorded. The measured curve is the sample's density of states
convolved with two broadenings — thermal, about 3.5 k_B T (≈ 0.3 meV per kelvin),
and instrumental, set by the modulation amplitude (≈ 2.5 × the rms amplitude
for a sine). Features narrower than their quadrature sum cannot be resolved,
and sharp features such as superconducting coherence peaks are lowered and
pushed outward by it.

## Tradeoffs

- **Modulation amplitude** buys signal linearly — the lock-in output is
  proportional to it, so noise relative to signal falls as 1/amplitude — and
  costs energy resolution. Once the instrumental broadening exceeds the thermal
  one, it biases everything read from sharp features: gaps come out too large,
  coherence peaks too low, in-gap states too broad. A fit that includes the
  broadening explicitly recovers some of this, but with growing parameter
  correlation between gap, lifetime broadening and instrumental width.
- **Averages** (sweeps, or lock-in time constant) buy signal-to-noise as the
  square root of their number, with no resolution cost. They cost time, and time
  brings drift: the tip–sample distance creeps, the tip position wanders off the
  intended site, and the chance of a tip change during the spectrum grows.
- The rule that follows: keep the modulation at or below the thermal broadening
  and buy signal-to-noise with averages, until drift or the time budget stops
  you.

## Limits

- Do not recommend a modulation whose broadening (≈ 2.5 × rms amplitude) exceeds
  the narrowest feature of interest or the thermal width, unless the goal
  explicitly accepts lost resolution.
- Respect any stated cap on averages or time per spectrum; long spectra lose
  registry with the intended site.
- Setpoint (stabilisation bias and current) changes the normalisation and the
  tip height. It is not an acquisition-quality knob; do not recommend changing
  it to gain signal unless it is offered as a parameter and the goal asks.

## Quality

- A sudden, persistent change of spectral SHAPE — a sloping or asymmetric
  background, a new broad feature, changed peak asymmetry — with no change of
  site is a tip change. No acquisition setting repairs it: the fit recipe must
  be rebuilt (and the tip reconditioned), so say so rather than turning knobs.
- A smooth evolution of gap or peak height from frame to frame while moving
  along a line or map is spatial variation of the sample — the signal, not drift.
- Noise that is periodic in bias (mains pickup) or correlated between adjacent
  points does not average down as the square root; more averages will
  underdeliver.
- If the fitted gap changes when ONLY the modulation is changed, the modulation
  is too large — the smaller-modulation value is the better one.

## Strategy

1. First set the modulation to the largest value that still respects the
   resolution limit above, and hold it there for the whole line or map, so every
   spectrum carries the same instrumental broadening and can be compared.
2. Then raise averages until the uncertainty on the target quantity meets the
   goal, in factors of about two to four, up to the stated cap.
3. After a tip change, keep the acquisition settings; recommend nothing but the
   recipe rebuild. Re-tune averages only once fits are good again, since the new
   tip may have a different signal level.
4. When the goal is met, stop changing parameters: a mid-line change of
   modulation puts a step into the very quantity being mapped.
