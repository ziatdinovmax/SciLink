---
description: Segmentation of touching or overlapping objects (grains, particles, droplets, cells, bubbles) — separates detection from instance partitioning when binary masks merge neighbors.
---
# Overlapping / Touching Object Segmentation Skill

## overview

Segmentation of objects that touch, overlap, or are spatially connected.
Applies to any image where individual objects must be counted and measured
but appear merged in a binary mask — grains, particles, droplets, domains,
cells, bubbles, etc. The key principle is to separate detection (finding
where individual objects are) from assignment (labeling which pixels belong
to which object).

## planning

### foundational
**Check first whether the problem actually needs instance segmentation.**
Several common cases resolve with simple classical methods before
reaching for a heavy model like SAM:

- **Well-separated objects** → Otsu + connected components is enough.
- **Objects separated by a visible boundary feature** (dark grain
  boundaries in etched metal, stained cell walls, bright domain
  edges) → use the boundary itself to separate the objects. Extract
  the boundary first — the method depends on how the boundary looks
  in the pixel data:
    - **Default: global Otsu thresholding** (`skimage.filters.threshold_otsu`)
      for clean dark/bright lines on a uniformly illuminated image.
      Simpler and usually sharper than adaptive variants; tune nothing.
    - **Adaptive thresholding** (`cv2.adaptiveThreshold` or
      `skimage.filters.threshold_local`) only when illumination varies
      noticeably across the field of view — block size and offset add
      tuning surface area that doesn't pay rent on evenly lit images.
    - **Edge detection** (Canny/Sobel) for gradient edges where there's
      no clean intensity step.
    - **Gradient magnitude or texture filters** for softer / textured
      boundaries.
    - Add **morphological closing** afterward if the boundary is broken.
  Once you have the boundary map, the two classical ways to turn it
  into labeled objects are (a) invert and run connected components on
  the interiors, or (b) use the boundary map as a watershed landscape.
  Usually sharper and faster than SAM for these images — SAM does not
  know to treat a thin boundary feature as an object separator.
- **Clean foreground-background intensity separation** → Otsu +
  connected components + morphological cleanup.

Only reach for SAM when objects genuinely touch or overlap AND no
visible boundary feature delineates them — the case where classical
approaches would merge adjacent objects into single blobs.

Connected component labeling on a binary mask merges all touching pixels
into one object. If the image shows touching or overlapping objects, the
pipeline must include a splitting step between mask creation and
measurement. The main approaches, in order of preference:

1. **SAM instance segmentation** (preferred): Use `run_sam_analysis`
   from `scilink.skills._shared.sam`. SAM detects individual object instances
   directly, even when they overlap, without requiring thresholding
   or binary masks. Works for any object shape. Tune via
   `sam_parameters` preset and `min_area`/`pruning_iou_threshold`.
   Avoid Gaussian blur before SAM unless noise is very high.

2. **Watershed splitting**: Create binary mask (any method) → distance
   transform → find markers (local maxima of distance transform) →
   watershed on inverted distance transform. Best for roughly convex
   objects when SAM is unavailable or produces poor results. Key
   parameter: `min_distance` in `peak_local_max` should approximate
   the object radius.

3. **Instance detection**: Detect individual objects directly from
   the image without relying on a binary mask. For elliptical objects:
   Hough ellipse detection (`skimage.transform.hough_ellipse`) on an
   edge map (e.g., Canny). For circular objects: Hough circles
   (`cv2.HoughCircles`). Note: `cv2.fitEllipse` fits an ellipse to
   an existing contour — it cannot separate overlapping objects and
   should only be used after splitting.

4. **Contour decomposition**: Find contours of merged blobs → detect
   concavity points where objects touch (using convex hull defects) →
   split along concavity lines. Works when contact regions create
   visible indentations in the merged contour.

### advanced
When objects have different intensities (e.g., multi-phase domains),
cluster by intensity first (k-means, GMM), then apply splitting within
each cluster. Watershed markers can be improved by weighting the distance
transform with edge gradients (Sobel magnitude) so that watershed
boundaries follow real inter-object edges.

## analysis

### foundational
**SAM implementation (preferred):**
Pass a 2D grayscale array or an HxWx3 RGB uint8 array. For multi-channel
images that are not RGB (e.g., 2-channel or 4-channel), pass a single
channel (e.g., `image[:,:,0]`).
**Tune all numeric parameters to the image** — the values below are
syntax examples only, not recommended defaults.
```
from scilink.skills._shared.sam import run_sam_analysis
result = run_sam_analysis(image_array, params={
    "sam_parameters": "default",     # always start from here; try "sensitive" if objects are missed
    "min_area": <tune>,              # minimum object area in pixels — set from image
    "max_area": <tune>,              # maximum object area in pixels — set from image
    "pruning_iou_threshold": <tune>  # masks with IoU above this are removed; lower = stricter, higher = keeps more overlapping objects
})
# Build labeled mask from SAM particles
labeled = np.zeros(image_array.shape[:2], dtype=np.int32)
for i, p in enumerate(result["particles"]):
    labeled[np.array(p["mask"], dtype=bool)] = i + 1
props = skimage.measure.regionprops(labeled, intensity_image=image_array)
```

**Watershed implementation (fallback):**
```
binary_mask = threshold(image)
distance = scipy.ndimage.distance_transform_edt(binary_mask)
markers = skimage.feature.peak_local_max(distance, min_distance=estimated_radius)
labeled_markers = scipy.ndimage.label(markers)[0]
labels = skimage.segmentation.watershed(-distance, labeled_markers, mask=binary_mask)
```

**Ellipse detection implementation:**
```
contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        # ellipse = ((cx, cy), (major, minor), angle)
```

After segmentation, filter out small fragments (area < threshold) and
border-touching objects if needed, then extract region properties.

### advanced
For highly elongated objects, distance-transform watershed tends to
over-segment because the distance ridge is flat. Use the Sobel gradient
as watershed landscape instead of inverted distance transform, with
markers still from distance transform peaks. This makes watershed
boundaries follow actual edges rather than geometric centers.

## interpretation

### foundational
After splitting, verify that the object count matches visual inspection.
Compare the size distribution to expected physical sizes — a bimodal
distribution with many small fragments suggests over-segmentation, while
too few large objects suggests under-segmentation.

## validation

### foundational
**Object count**: Should match visual estimate within ±20%.

**Size distribution**: Should be unimodal or match expected physics.
Many fragments below 1/4 of the typical object area indicate
over-segmentation artifacts.

**Shape metrics**: Circularity and solidity should be physically
reasonable for the object type (e.g., >0.7 for droplets/bubbles,
variable for grains).
