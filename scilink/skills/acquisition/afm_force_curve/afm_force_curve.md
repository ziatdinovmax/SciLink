---
description: Steering AFM force spectroscopy — trigger force and approach speed against indentation depth, contact-region sampling, sample and tip damage, hydrodynamic and viscoelastic artefacts.
technique: ["AFM force spectroscopy", "force spectroscopy", "force-distance curve", "force curve", "AFM nanoindentation", "atomic force microscopy"]
---

# AFM force-curve acquisition

## Overview

A frame is one approach (or approach–retract) curve: force against piezo
displacement. It has a flat non-contact baseline, a contact point, and a contact
region whose slope or curvature carries the mechanics. What is steered by is
usually the contact point and a stiffness or modulus; both come from the contact
region only, so their precision is set by how many points lie in it and how far
it extends above the noise.

## Tradeoffs

- **Trigger (maximum) force** sets how deep the tip goes. A higher trigger gives
  a longer contact region — more points, better-conditioned slope — but larger
  indentation: on soft or thin samples the substrate starts to be felt (apparent
  stiffening) once indentation exceeds roughly 10 % of the thickness, and
  plastic deformation, tip blunting or sample damage set in. A lower trigger
  protects tip and sample and keeps the response elastic, at the price of a
  short, noisy contact region.
- On a STIFF sample the contact region is steep: for the same trigger force it
  spans few nanometres and few points, so stiffness precision falls exactly
  where stiffness rises. That, not noise, is why stiffness estimates get noisy
  after moving onto a stiffer region; the remedy is a higher trigger (or denser
  sampling), within the limits below.
- **Approach speed** buys throughput. In liquid, hydrodynamic drag on the
  cantilever adds a speed-proportional force offset and tilts the baseline;
  viscoelastic samples look stiffer when probed faster. Too slow and thermal
  drift and piezo creep distort the curve. Speed also sets the points per
  nanometre at a fixed sampling rate.

## Limits

- Never recommend a trigger force above what the instrument description gives as
  safe for the tip and sample; on soft, thin or biological samples stay in the
  low range unless the goal says otherwise.
- Do not change trigger force and speed together: their effects on apparent
  stiffness cannot then be separated.
- Absolute stiffness rests on the cantilever calibration (deflection sensitivity
  and spring constant). No acquisition parameter fixes a calibration error;
  do not try.

## Quality

- A baseline that is tilted or oscillates (optical interference) biases the
  contact point; it is an alignment or speed issue, not a sample property.
- A jump-to-contact before the contact region is adhesion or a capillary neck:
  the contact point is then defined differently and the fit recipe may need
  rebuilding.
- An abrupt, persistent change of slope AND contact point between consecutive
  curves means the tip moved onto different material or height — a real change.
  A slow drift of the contact point alone is thermal or piezo drift.
- A contact region with fewer than roughly ten points cannot support a slope
  estimate, whatever the fit quality says.

## Strategy

1. After a change to a stiffer region, check the number of points in contact.
   If it has collapsed, raise the trigger force stepwise (about 1.5× per step)
   until the contact region is again well sampled, staying inside the limits.
2. After a change to a softer region, do the opposite: lower the trigger force
   to keep indentation shallow.
3. Leave the approach speed alone unless the baseline shows drag or drift; then
   change only the speed, and compare stiffness before and after to see whether
   the sample is rate-dependent.
4. When precision already meets the goal, do not change anything: every change
   of trigger force changes the indentation depth and therefore what is being
   measured.
