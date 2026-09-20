"""Simulate lost image detail while preserving annotation geometry."""

import math

from PIL import Image, ImageFilter


def gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Pillow approximates Gaussian filtering; radius is its standard deviation."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Blur radius must be finite and positive.")
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
