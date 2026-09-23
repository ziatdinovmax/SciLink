---
description: Steering electron-microscope image acquisition (TEM, STEM, SEM frames of a changing sample) — dose against noise and damage, focus, sampling against field of view, frame time against drift and time resolution.
technique: ["TEM", "STEM", "HAADF", "HAADF-STEM", "bright-field", "electron microscopy", "in-situ TEM", "SEM", "transmission electron microscopy", "scanning transmission electron microscopy"]
---

# Electron-microscope image acquisition

## Overview

A frame is one image of a sample that may be changing: particles that grow or
dissolve, a lattice that transforms, a front that moves. What is tracked is read
off the image by segmentation or by locating features, so the acquisition has to
keep those features detectable and measurable at the same standard from frame to
frame. Four things set that standard: the dose per frame (noise), the focus
(contrast transfer and edge sharpness), the sampling (pixels per feature against
field of view), and the frame time (drift within a frame, and how fast a change
can be followed).

## Tradeoffs

- **Dose** (beam current x dwell or exposure time) buys signal-to-noise as its
  square root. It costs damage, and damage is cumulative: knock-on displacement,
  radiolysis, beam-induced heating, carbon contamination and, for in-situ work,
  beam-driven reactions that are then measured as if they were the sample's own.
  A measurement that needs a higher dose per frame usually affords fewer frames.
- **Focus**. Out of focus, small features lose contrast first and edges soften,
  so a size measured by threshold drifts and the smallest objects drop out of a
  count long before the image looks bad. In phase-contrast TEM a deliberate
  defocus buys contrast for weak objects and costs resolution and delocalised
  edges; in STEM-HAADF best focus is simply sharpest.
- **Sampling and field of view**. More pixels per feature measures each one
  better; a larger field measures more of them. Counting statistics improve as
  the square root of the number of objects in the field, size statistics with
  pixels across each object. Below about five pixels across, a diameter is
  mostly the threshold's choice.
- **Frame time**. A longer frame (slower scan, longer exposure, more averaged
  frames) lowers noise, smears anything that moves during it, and bends scanned
  images by drift. The time between frames is the time resolution of everything
  that is tracked.

## Limits

- Never exceed a stated dose, dose-rate or total-dose limit; if the sample's
  damage threshold is known, keep the accumulated dose of the WHOLE remaining run
  under it, not just the next frame's.
- Do not change magnification, field of view or binning in the middle of a run
  unless the goal asks: sizes and counts before and after are no longer the same
  measurement, and the locked analysis was verified at one sampling.
- Do not recommend a frame time longer than the interval in which the sample is
  expected to change, nor one at which drift exceeds about a pixel.
- Focus, stigmation and alignment belong to the operator or the microscope's own
  routines unless they are offered as acquisition parameters.

## Quality

- A uniform loss of sharpness across the whole field, with the high spatial
  frequencies of the power spectrum falling, is defocus or drift, not the
  sample: nothing about the sample should be concluded from those frames, and no
  analysis setting repairs them. Refocus; do not re-tune dose.
- A count that falls while the mean size rises can be coarsening, or the
  smallest objects sinking under the noise or the blur. If it follows a change
  of dose or focus, it is the acquisition.
- A change confined to the irradiated area, growing with accumulated dose and
  not with the experimental variable, is beam damage or contamination. Lower the
  dose rate or blank between frames; more dose makes it worse.
- Features that change only at the edges of the field, or a field that slides
  steadily, are drift of the stage or the image, not of the sample.
- New small objects appearing everywhere at once deserve a look before they are
  believed: nucleation is real, and so are noise specks that a threshold starts
  to pick up when the contrast drops.

## Strategy

1. Fix sampling and field of view first, from the smallest feature that must be
   measured (at least five to ten pixels across) and the number of objects
   needed for the statistics, and hold them for the run.
2. Set the dose per frame to the lowest value at which the tracked quantities
   are stable from frame to frame on an unchanging sample; spend the remaining
   dose budget on more frames, not on prettier ones.
3. While nothing changes, lengthen the interval between frames or blank the
   beam: dose spent on identical frames is damage without information.
4. When a change is announced, shorten the interval before anything else, and
   raise the dose per frame only if the new features are at the noise limit. A
   pause, where the experiment can be held, is the moment to refocus and to
   decide the sampling for what is now there.
5. When the goal is met, stop changing parameters: a change of dose or focus in
   mid-run puts a step into the quantity being followed.
