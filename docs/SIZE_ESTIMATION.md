# Approximate Fruit Size Estimation

## What Phase 3 Measures

FruitVision can estimate an **approximate physical horizontal bounding-box width**
using a known reference object visible in the same image. It uses the existing
frozen YOLO11n baseline; no model training or new dependency is needed.

This is a simple reference scale, **not a true fruit diameter measurement** or a
full camera calibration. A banana's horizontal box width, for example, changes
with its orientation and is not necessarily its length or thickness. A grape
bunch detection measures the bunch's box, not individual grape diameters.

The user measures the reference. The ten-class fruit detector does not detect
cards, rulers, or other reference objects automatically. Supplying a reference
does not add, remove, or reclassify any detections or change fruit counts.

## Method

```text
reference_width_pixels = reference_x2 − reference_x1  # if a box is provided
cm_per_pixel = known_reference_width_cm / reference_width_pixels
fruit_bbox_width_pixels = fruit_x2 − fruit_x1
approximate_width_cm = fruit_bbox_width_pixels × cm_per_pixel
```

The scale applies to horizontal widths at approximately the reference's depth.
Reference and fruit coordinates must come from the **same image and pixel scale**.
All boxes use `x1 y1 x2 y2`, meaning left, top, right, bottom in pixels, not the
normalized YOLO training-label format.

### Example calculation — arithmetic illustration only

Assume a reference width of **8.56 cm**, measured as **250 pixels** in an image,
and a detected fruit box spanning **200 pixels** horizontally:

```text
cm_per_pixel = 8.56 / 250 = 0.03424 cm/pixel
approximate bounding-box width = 200 × 0.03424 = 6.848 cm ≈ 6.85 cm
```

These values illustrate the formula; they are **not a real dataset measurement,
model prediction, or accuracy result**. Displaying two decimal places is a formatting
choice, not a claim of measurement precision.

## Preparing a Reference Image

1. Place a reference of known width next to the fruit, as close as practical to
   the same depth and measurement plane. A ruler behind a fruit's front surface
   can still introduce depth error.
2. Keep the known reference width horizontal in the image, with the reference
   facing the camera. A rotated or tilted rectangle's enclosing box does not
   necessarily represent the known physical side length.
3. Use a clear image with fully visible reference edges and fruit. Avoid clipped
   or heavily occluded objects.
4. Measure the reference's horizontal pixel width, or its tight bounding box, in
   the **EXIF-oriented original image**. FruitVision corrects phone-photo orientation
   before inference. Do not use coordinates from a scaled preview, the 640-pixel
   model input, or the output image's added count footer.
5. Re-measure the reference for every new image. Moving the camera, changing zoom,
   cropping/resizing, or moving objects can invalidate a previously measured scale.

The software can validate numeric inputs and image boundaries; it cannot verify
that the supplied physical width is true, that the reference is actually present,
or that every fruit lies at the same depth.

## Command-Line Usage

Run from the repository root with the existing environment activated. Replace
`examples/fruit_with_reference.jpg` with your own image. The numbers below are
illustrative and must be replaced by measurements of that image.

### Supply reference width in pixels

```bash
python src/predict.py \
  --image examples/fruit_with_reference.jpg \
  --reference-width-cm 8.56 \
  --reference-width-pixels 250 \
  --output-dir outputs/predictions/with_size
```

### Supply a reference bounding box

```bash
python src/predict.py \
  --image examples/fruit_with_reference.jpg \
  --reference-width-cm 8.56 \
  --reference-bbox 20 30 270 110 \
  --output-dir outputs/predictions/with_size
```

The example box has width `270 − 20 = 250` pixels. Supply **one** pixel input mode,
not both. Both commands default to the existing frozen checkpoint:
`outputs/training/fruitvision_baseline/weights/best.pt`.

### Ordinary prediction without size estimation

```bash
python src/predict.py --image examples/fruits.jpg
```

With no reference options, the detection JSON schema, console summary, counts,
confidence, inference timing, and annotation behavior remain unchanged. The pure
`predict_image()` API and its return signature are unchanged as well.

The annotated image still shows the existing detection boxes, class names,
confidence, and count footer. Approximate sizes appear in the **console and JSON**;
this phase adds no size overlay or UI.

## Prediction JSON

When a valid reference is supplied, two optional additions are written:

| Location | Fields and meaning |
| --- | --- |
| Top-level `size_estimation` | `approximate: true`, `method`, `reference_source`, supplied reference width/box, derived pixel width, `cm_per_pixel`, image dimensions, coordinate convention, measurement description, and limitations |
| Each `detections[i].size_estimation` | `approximate: true`, `bbox_width_pixels`, `approximate_width_cm`, and a description stating that the quantity is horizontal box width, not diameter |

Existing per-detection boxes, IDs, names, confidence, and summary fields are
preserved. The reference's `reference_bbox_xyxy` is `null` when a pixel width was
provided directly. The `summary.inference_time_ms` definition remains unchanged:
it measures detector prediction/extraction, not this optional conversion.

No reference produces **no additional size fields**. Partial reference information
produces an actionable error rather than invented values. With a valid reference
but no detections, calibration metadata is retained and `detections` is empty;
no fruit measurements are fabricated.

## Python Module

`src/size_estimation.py` contains two public functions independent of YOLO:

```python
from src.size_estimation import build_reference, estimate_fruit_widths

# Use measurements from the same oriented image as these existing detections.
reference = build_reference(
    known_reference_width_cm=known_width_cm,
    reference_bbox_xyxy=measured_reference_box,
    image_size=(image.width, image.height),
)
detections_with_sizes = estimate_fruit_widths(detections, reference)
```

`build_reference()` returns `None` when no reference measurements are supplied.
Otherwise it validates the measurements and returns scale/provenance metadata.
`estimate_fruit_widths()` returns new detection dictionaries, leaving the input
records unchanged. Pass the metadata returned by `build_reference()`; do not invent
a scale from a fruit's assumed typical size. The CLI always supplies image dimensions
for bounds checking; direct callers should do the same.

## Invalid Inputs and Tests

The code rejects incomplete/conflicting reference inputs, zero/negative/nonfinite
widths, invalid or out-of-image boxes, pixel widths exceeding the image width,
missing fruit boxes, and nonfinite calculated results. It does not clip or repair
invalid measurements silently. Reference validation happens before model loading
and inference in the CLI.

```bash
python -m unittest discover -s tests -v
```

**61 tests passed:** 46 existing tests plus 15 size-estimation tests. New tests cover
conversion arithmetic, box/pixel reference equivalence, per-detection widths,
preserved counts/inputs, absent and partial references, invalid numbers/boxes,
overflow, no detections, CLI JSON integration, and EXIF-oriented bounds.
The test log is saved at `outputs/size_estimation_tests.txt`. Temporary fixtures
verify software behavior; they do not validate physical measurement accuracy.

## Limitations

- **Perspective and depth:** a single scale cannot describe objects at different
  camera distances. Even fruit on a table has three-dimensional depth; a planar
  reference nearby only approximates the relevant scale.
- **Camera distance and zoom:** changing either changes pixels per centimeter.
  Recalibrate in the same image instead of carrying a scale between photographs.
- **Orientation:** fruit orientation changes horizontal projected width. Reference
  rotation/tilt can make its box width inconsistent with its known physical width.
- **Lens distortion:** scale can vary across the image, particularly near edges.
  This module performs no distortion correction or perspective rectification.
- **Detection and visibility:** loose, clipped, missed, overlapping, or occluded
  boxes affect estimates. Detector confidence is not confidence in size accuracy.
- **Reference measurement:** inaccurate physical dimensions or manual pixel edges
  bias every estimate. The model cannot confirm the user's calibration assumptions.
- **Box width versus anatomy:** axis-aligned enclosing rectangles can include
  background, leaves, or stems and do not establish true diameter, length, volume,
  or weight. No uncertainty interval or physical accuracy score has been measured.
- **Existing dataset:** its source images and labels have no verified physical
  references or measurement ground truth. Their dimensions alone cannot establish
  fruit size in centimeters. No real size-results claim is made from that dataset.

## Recommended Next Step

Capture a small set of your own fruit images with a measured reference at similar
depth, record independent ruler/caliper measurements and the intended horizontal
dimension, then compare estimates with those measurements. Report absolute errors
and repeatability before claiming accuracy. Include changes in depth/orientation
to reveal where the single-reference assumption fails. Keep the detector frozen;
size validation does not require retraining. UI work remains a separate phase.
