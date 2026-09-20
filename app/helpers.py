"""Small UI adapters; model inference and measurements stay in src/."""

import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from src.common import BEST_MODEL, CLASS_NAMES, ROOT
from src.predict import predict_image
from src.size_estimation import estimate_fruit_widths
from src.visualization import annotate_image

METRICS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "map50": "mAP50", "map50_95": "mAP50–95"}
CONDITION_LABELS = {"normal": "Normal", "dark": "Dark", "bright": "Bright", "blur": "Blur", "low_resolution": "Low Resolution"}
SIZE_NOTICE = "Approximate measurement only. Accuracy depends on perspective, object depth, orientation, camera distance, calibration, and detection quality."
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_upload(content: bytes) -> Image.Image:
    """Validate actual image format and orient pixels before reference measurements."""
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a JPG or PNG file no larger than 10 MB.")
    try:
        with Image.open(BytesIO(content)) as original:
            if original.format not in ("JPEG", "PNG"):
                raise ValueError("Unsupported image. Please upload a JPG or PNG.")
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError("This image is too large. Use an image with at most 20 megapixels.")
            original.load()
            return ImageOps.exif_transpose(original).convert("RGB")
    # Ultralytics' PIL patch may attempt optional HEIF imports on invalid bytes.
    # HEIF is unsupported here; a missing decoder should still be a friendly error.
    except (UnidentifiedImageError, OSError, ImportError, Image.DecompressionBombError) as error:
        raise ValueError("This image could not be read. Try another JPG or PNG.") from error


def prediction_signature(content: bytes, confidence: float, reference: dict | None) -> str:
    """Identify image + settings so edited inputs never display stale predictions."""
    settings = json.dumps([confidence, reference], sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(content + settings).hexdigest()


def run_prediction(model, image: Image.Image, filename: str, confidence: float,
                   reference: dict | None, device: str) -> tuple[dict, Image.Image]:
    """Call the existing inference/counting pipeline and optional size conversion."""
    detections, summary = predict_image(model, image, confidence=confidence, img_size=640, device=device)
    if reference is not None:
        detections = estimate_fruit_widths(detections, reference)
    payload = {"image": filename, "weights": str(BEST_MODEL), "confidence_threshold": confidence,
               "img_size": 640, "device": device, "image_width": image.width, "image_height": image.height,
               "coordinate_system": "xyxy pixels in EXIF-oriented original image",
               "summary": summary, "detections": detections}
    if reference is not None:
        payload["size_estimation"] = reference
    # Reuse the existing renderer; the dashboard shows counts in its own summary.
    annotated = annotate_image(image, detections, summary).crop((0, 0, image.width, image.height))
    return payload, annotated


def detection_rows(detections: list[dict]) -> list[dict]:
    rows = []
    include_sizes = any("size_estimation" in detection for detection in detections)
    for index, detection in enumerate(detections, 1):
        row = {"Object": index, "Fruit": detection["class_name"].title(),
               "Confidence (%)": round(detection["confidence"] * 100, 2)}
        row.update({name: round(value, 2) for name, value in zip(("X1", "Y1", "X2", "Y2"), detection["bbox_xyxy"])})
        if include_sizes:
            value = detection.get("size_estimation", {}).get("approximate_width_cm")
            row["Approx. width (cm)"] = round(value, 2) if value is not None else None
        rows.append(row)
    return rows


def image_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def read_report(paths: list[Path], required_keys: tuple[str, ...]) -> tuple[dict, Path]:
    """Prefer local run files, then curated exports; never replace corrupt data silently."""
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("Saved results are unavailable. Restore the existing result exports to view this page.")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read saved results: {path.name}.") from error
    if not isinstance(report, dict) or any(key not in report for key in required_keys):
        raise ValueError(f"Saved results have an unexpected format: {path.name}.")
    return report, path


def validate_metrics(rows: list[dict]) -> None:
    for row in rows:
        for name in METRICS:
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Saved metrics contain missing or invalid values. No substitute scores are shown.")


def baseline_report(split: str) -> tuple[dict, Path]:
    if split not in ("val", "test"):
        raise ValueError("Choose validation or test results.")
    name = "validation" if split == "val" else "test"
    report, path = read_report([ROOT / f"outputs/metrics/baseline_{name}/metrics.json",
                               ROOT / f"docs/results/{name}_metrics.json"], ("overall", "per_class"))
    validate_metrics([report["overall"], *report["per_class"]])
    return report, path


def robustness_report() -> tuple[dict, Path]:
    report, path = read_report([ROOT / "outputs/experiments/robustness/results.json",
                               ROOT / "docs/results/robustness/results.json"], ("overall", "per_class"))
    if {row["condition"] for row in report["overall"]} != set(CONDITION_LABELS):
        raise ValueError("Saved robustness results do not include all five conditions.")
    validate_metrics([*report["overall"], *report["per_class"]])
    return report, path


def metric_rows(rows: list[dict]) -> list[dict]:
    return [{"Class": row["class_name"].title(), "Ground-truth objects": row.get("ground_truth_objects"),
             **{label + " (%)": round(row[key] * 100, 2) for key, label in METRICS.items()}}
            for row in rows]
