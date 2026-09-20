"""Change brightness without moving pixels or bounding boxes."""

import math

from PIL import Image, ImageEnhance


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Multiply RGB intensities; bright values clip at 255 (not gamma correction)."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Brightness factor must be finite and positive.")
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)
