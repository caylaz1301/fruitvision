"""Count plain detection dictionaries without importing YOLO or torch."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def summarize_detections(
    detections: Iterable[Mapping[str, Any]], class_names: Sequence[str]
) -> dict:
    """Return zero-inclusive counts and mean confidence (None for no detections)."""
    counts = dict.fromkeys(class_names, 0)
    confidences = []
    for detection in detections:
        name = detection["class_name"]
        confidence = float(detection["confidence"])
        if name not in counts:
            raise ValueError(f"Unknown class: {name}")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        counts[name] += 1
        confidences.append(confidence)
    return {
        "counts": counts,
        "total_objects": len(confidences),
        # Mean confidence describes retained detections; it is NOT accuracy.
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def format_summary(summary: dict) -> str:
    average = summary["average_confidence"]
    confidence = f"{average:.1%}" if average is not None else "N/A (no detections)"
    lines = ["Detection Summary", ""]
    lines.extend(f"{name.title()}: {count}" for name, count in summary["counts"].items())
    lines.extend(["", f"Total Objects: {summary['total_objects']}", f"Average Confidence: {confidence}"])
    return "\n".join(lines)
