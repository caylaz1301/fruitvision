"""Reduce available image detail, keeping the detector input setting fixed."""

import math

from PIL import Image


def reduce_resolution(image: Image.Image, scale: float = 0.25) -> Image.Image:
    """Downsample with area averaging, then bilinearly restore the source canvas.

    No crop or coordinate shift occurs, so normalized YOLO boxes stay unchanged.
    Upsampling restores dimensions, not the detail discarded by downsampling.
    """
    if not math.isfinite(scale) or not 0 < scale < 1:
        raise ValueError("Resolution scale must be finite and between 0 and 1.")
    image = image.convert("RGB")
    reduced = tuple(max(1, round(dimension * scale)) for dimension in image.size)
    return image.resize(reduced, Image.Resampling.BOX).resize(image.size, Image.Resampling.BILINEAR)
