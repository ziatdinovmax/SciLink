---
description: Steering a powder X-ray diffraction series (beamline or laboratory, in situ) — exposure time against counting statistics, time resolution through a transformation, detector saturation and beam damage.
technique: ["X-ray diffraction", XRD, "powder diffraction", "powder X-ray diffraction", "synchrotron powder X-ray diffraction", "in-situ XRD"]
---

# Powder XRD acquisition

## Overview

A frame is one diffraction pattern: counts against 2θ (or Q). The noise is
Poisson, so the relative precision of a peak's intensity goes as one over the
square root of its counts, and the precision of its position goes as its width
divided by the square root of its counts. In an in-situ series (temperature
ramp, reaction, cycling) each pattern also averages over whatever the sample did
during the exposure.

## Tradeoffs

- **Exposure time** is usually the only free knob. Counts scale linearly with
  it; precision improves as its square root; time (or temperature) resolution
  degrades linearly. During a ramp, an exposure spans `ramp rate × exposure` of
  the control variable — a pattern taken across a transition is a mixture of
  both sides and looks like anomalous broadening or a split peak.
- Position precision is cheap: a strong reflection's position is known to a
  small fraction of its width after modest counts. Weak reflections, minority
  phase fractions and peak-shape parameters are what need long exposures.
- On area detectors the strongest reflection saturates first; saturation flattens
  the peak top and biases height and width while leaving the position roughly
  right.

## Limits

- Do not exceed the exposure at which the strongest reflection approaches the
  detector's linear range.
- At a synchrotron, dose matters for beam-sensitive samples (hydrates, organics,
  battery electrolytes, some halides): peaks that weaken or shift at constant
  conditions, only where the beam sits, are beam damage — reduce exposure or
  ask for attenuation or a fresh spot, do not add dose.
- Never trade away the time resolution needed to resolve a transformation in
  progress: keep several frames across the transition's width.

## Quality

- A smooth, monotonic shift of all reflections with the control variable is
  thermal expansion or a solid-solution change — expected, not a fault.
- New reflections growing while others fade is a phase transformation: the
  analysis recipe needs rebuilding, and the acquisition should get FASTER, not
  slower, while it lasts.
- A falling height at constant width and position, with total counts falling
  too, is a beam or sample-position problem (beam decay, sample sintering away
  from the beam), not a structural change.
- Spotty or wildly varying relative intensities between frames mean poor powder
  statistics (few, large grains); more exposure will not fix it.

## Strategy

1. Size the exposure to the weakest quantity the goal actually names. If only
   the main reflection's position and width are wanted, exposures can be short.
2. Away from any transformation, lengthen exposures to the precision needed and
   no further. When the stream shows a transformation starting (a second set of
   reflections, fit quality falling, fractions changing), shorten the exposure
   so that at least five to ten frames span it, and restore it afterwards.
3. Change exposure by factors of about two; smaller changes are invisible in the
   statistics.
