"""Approximate physical bounding-box widths from a user-measured reference.

This module does not detect the reference, infer depth, or measure true diameter.
Reference and fruit must be at approximately the same depth in the same image.
"""

import math
from collections.abc import Sequence


def _positive(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number.") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _box(box: Sequence[float], image_size: Sequence[float] | None = None) -> list[float]:
    """Check xyxy pixel coordinates without clipping or silently repairing boxes."""
    try:
        if len(box) != 4 or any(isinstance(value, bool) for value in box):
            raise ValueError
        values = [float(value) for value in box]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Bounding box must contain four finite pixel coordinates: x1 y1 x2 y2.") from error
    x1, y1, x2, y2 = values
    if not all(math.isfinite(value) for value in values) or not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("Bounding box needs finite, nonnegative coordinates with x1 < x2 and y1 < y2.")
    if image_size is not None and (x2 > image_size[0] or y2 > image_size[1]):
        raise ValueError("Bounding box extends outside the EXIF-oriented original image.")
    return values


def build_reference(
    known_reference_width_cm: float | None = None,
    reference_width_pixels: float | None = None,
    reference_bbox_xyxy: Sequence[float] | None = None,
    image_size: Sequence[float] | None = None,
) -> dict | None:
    """Validate manual calibration; return None only when no reference is supplied.

    Provide a known width in cm plus exactly one of a pixel width or an xyxy box.
    Coordinates refer to the same EXIF-oriented image as the fruit predictions,
    before YOLO's internal resizing. image_size enables image-boundary checks.
    """
    if all(value is None for value in (known_reference_width_cm, reference_width_pixels, reference_bbox_xyxy)):
        return None
    if known_reference_width_cm is None or (reference_width_pixels is None and reference_bbox_xyxy is None):
        raise ValueError("Reference needs --reference-width-cm and either --reference-width-pixels or --reference-bbox.")
    if reference_width_pixels is not None and reference_bbox_xyxy is not None:
        raise ValueError("Supply either reference pixel width or reference bounding box, not both.")
    width_cm = _positive(known_reference_width_cm, "Known reference width in cm")
    size = None
    if image_size is not None:
        if len(image_size) != 2:
            raise ValueError("image_size must contain width and height.")
        size = [_positive(value, "Image dimension") for value in image_size]
    reference_box = _box(reference_bbox_xyxy, size) if reference_bbox_xyxy is not None else None
    width_pixels = (reference_box[2] - reference_box[0] if reference_box is not None
                    else _positive(reference_width_pixels, "Reference width in pixels"))
    if size is not None and width_pixels > size[0]:
        raise ValueError("Reference pixel width exceeds the EXIF-oriented original image width.")
    # This is a local image scale, not a full camera calibration or depth estimate.
    scale = _positive(width_cm / width_pixels, "Calculated cm_per_pixel")
    return {
        "approximate": True,
        "method": "same_image_reference_width",
        "reference_source": "user_supplied",
        "known_reference_width_cm": width_cm,
        "reference_width_pixels": width_pixels,
        "reference_bbox_xyxy": reference_box,
        "cm_per_pixel": scale,
        "image_size_pixels": size,
        "coordinate_system": "xyxy pixels in EXIF-oriented original image",
        "measurement": "Approximate horizontal bounding-box width, not true fruit diameter.",
        "limitations": (
            "Assumes reference and fruit are at approximately the same depth, with the known reference "
            "width aligned to the image horizontal axis. Perspective, orientation, camera distance, "
            "lens distortion, occlusion and box error can bias estimates. Accuracy is not validated."
        ),
    }


def estimate_fruit_widths(detections: list[dict], reference: dict | None = None) -> list[dict]:
    """Return detection copies with optional approximate width; never change counts.

    A detector box encloses the visible object. Its horizontal extent is neither
    a segmentation measurement nor a guaranteed anatomical diameter.
    """
    if reference is None:
        return [dict(detection) for detection in detections]
    scale = _positive(reference.get("cm_per_pixel"), "Reference cm_per_pixel")
    results = []
    for index, detection in enumerate(detections):
        if "bbox_xyxy" not in detection:
            raise ValueError(f"Detection {index} has no bbox_xyxy for size estimation.")
        x1, _, x2, _ = _box(detection["bbox_xyxy"], reference.get("image_size_pixels"))
        pixel_width = x2 - x1
        estimated_cm = _positive(pixel_width * scale, "Estimated bounding-box width in cm")
        results.append({**detection, "size_estimation": {
            "approximate": True,
            "bbox_width_pixels": pixel_width,
            "approximate_width_cm": estimated_cm,
            "measurement": "horizontal bounding-box width; not true diameter",
        }})
    return results
