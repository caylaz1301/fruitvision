"""Draw predictions in original-image pixel coordinates and plot fruit counts."""

from pathlib import Path
from colorsys import hsv_to_rgb

from PIL import Image, ImageDraw, ImageFont

def class_color(class_id: int) -> tuple[int, int, int]:
    """Stable, generic colors derived from IDs, independent of fruit names."""
    return tuple(round(channel * 255) for channel in hsv_to_rgb((class_id * 0.618034) % 1, 0.75, 0.65))


def place_label(left: float, preferred_top: float, width: float, height: float,
                image_height: int, occupied: list[tuple]) -> tuple:
    """Move neighboring labels vertically so their text remains readable."""
    top = max(0, min(preferred_top, image_height - height))
    candidates = [top]
    for step in range(1, max(1, int(image_height // (height + 2)))):
        candidates.extend((top - step * (height + 2), top + step * (height + 2)))
    for candidate in candidates:
        if not 0 <= candidate <= image_height - height:
            continue
        rectangle = (left, candidate, left + width, candidate + height)
        if not any(rectangle[0] < other[2] and rectangle[2] > other[0]
                   and rectangle[1] < other[3] and rectangle[3] > other[1] for other in occupied):
            return rectangle
    # Extremely crowded/tiny images may not have enough room for every label.
    return left, top, left + width, top + height


def annotate_image(image: Image.Image, detections: list[dict], summary: dict) -> Image.Image:
    """Return an RGB image with boxes, labels, confidence, and a count footer."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    # Phone images are downscaled in the UI. Scale text with the image so labels
    # remain legible after display resizing; this never changes detection boxes.
    font = ImageFont.load_default(size=max(12, round(max(image.size) / 40)))
    line_width = max(2, round(max(image.size) / 300))
    # Draw all outlines first so later boxes cannot cross previously drawn text.
    for detection in detections:
        draw.rectangle(detection["bbox_xyxy"], outline=class_color(detection["class_id"]), width=line_width)
    occupied = []
    for detection in detections:
        # xyxy means left, top, right, bottom in ORIGINAL image pixels.
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        text = f"{detection['class_name']} {detection['confidence']:.1%}"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bounds[2], bounds[3] + 4
        left = max(0, min(x1, annotated.width - text_width - 4))
        rectangle = place_label(left, y1 - text_height, text_width + 4, text_height, annotated.height, occupied)
        occupied.append(rectangle)
        top = rectangle[1]
        # A dark label background keeps white text readable for every class hue.
        draw.rectangle(rectangle, fill=(21, 38, 53))
        draw.text((left + 2, top), text, fill="white", font=font)
    lines = ["Detected fruits"] + [f"{name.title()}: {count}" for name, count in summary["counts"].items()]
    average = summary["average_confidence"]
    lines += [f"Total Objects: {summary['total_objects']}",
              f"Average Confidence: {average:.1%}" if average is not None else "Average Confidence: N/A"]
    canvas = Image.new("RGB", (max(annotated.width, 280), annotated.height + 20 * len(lines) + 12), "white")
    canvas.paste(annotated, (0, 0))
    footer = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        footer.text((8, annotated.height + 6 + index * 20), line, fill="black")
    return canvas


def plot_counts(counts: dict[str, int], output: Path, title: str = "Fruit counts",
                ylabel: str = "Detected objects") -> None:
    """Save a simple bar chart; counts describe detections after thresholding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axis = plt.subplots(figsize=(max(7, len(counts) * 0.9), 4.5))
    colors = [tuple(channel / 255 for channel in class_color(index)) for index in range(len(counts))]
    bars = axis.bar([name.title() for name in counts], list(counts.values()), color=colors)
    axis.bar_label(bars)
    maximum = max(counts.values(), default=0)
    axis.set(ylabel=ylabel, title=title, ylim=(0, max(1, maximum * 1.15)))
    axis.tick_params(axis="x", labelrotation=35)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
