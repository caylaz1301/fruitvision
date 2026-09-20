"""Run fruit detection on one image and save an annotation plus JSON details."""

import argparse
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import BEST_MODEL, CLASS_NAMES, ROOT, load_detector, positive_int, probability, save_json, select_device
from src.counting import format_summary, summarize_detections
from src.size_estimation import build_reference, estimate_fruit_widths
from src.visualization import annotate_image, plot_counts


def extract_detections(result) -> list[dict]:
    """Convert Ultralytics tensors to ordinary, JSON-compatible Python values."""
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    return [
        {"bbox_xyxy": box, "class_id": int(class_id),
         "class_name": result.names[int(class_id)], "confidence": float(confidence)}
        for box, class_id, confidence in zip(boxes, class_ids, confidences)
    ]


def predict_image(model, image: Image.Image, confidence: float = 0.25,
                  img_size: int = 640, device: str = "cpu") -> tuple[list[dict], dict]:
    """Infer once; keep counting independent of model implementation."""
    # Confidence filters weak predictions. Raising it can reduce false positives
    # but can also miss real fruit (false negatives).
    start = perf_counter()
    result = model.predict(source=image, conf=confidence, imgsz=img_size, device=device, verbose=False)[0]
    detections = extract_detections(result)
    summary = summarize_detections(detections, CLASS_NAMES)
    # Wall time includes prediction setup/preprocessing and result transfer, not model loading.
    summary["inference_time_ms"] = (perf_counter() - start) * 1000
    return detections, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=BEST_MODEL)
    parser.add_argument("--confidence", type=probability, default=0.25)
    parser.add_argument("--img-size", type=positive_int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/predictions")
    parser.add_argument("--plot-counts", action="store_true")
    parser.add_argument("--reference-width-cm", type=float, help="Known physical reference width in this image (cm).")
    reference_pixels = parser.add_mutually_exclusive_group()
    reference_pixels.add_argument("--reference-width-pixels", type=float,
                                  help="Reference horizontal width in EXIF-oriented original image pixels.")
    reference_pixels.add_argument("--reference-bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                                  help="Reference xyxy box in EXIF-oriented original image pixels.")
    args = parser.parse_args()
    try:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}. Pass --image PATH to a real image.")
        with Image.open(args.image) as original:
            # Normalize phone-photo orientation and color channels before inference.
            image = ImageOps.exif_transpose(original).convert("RGB")
        # Reject partial/invalid reference data before loading or running the model.
        reference = build_reference(args.reference_width_cm, args.reference_width_pixels,
                                    args.reference_bbox, image.size)
        model = load_detector(args.weights)
        device = select_device(args.device)
        detections, summary = predict_image(model, image, args.confidence, args.img_size, device)
        if reference is not None:
            detections = estimate_fruit_widths(detections, reference)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = args.output_dir / f"{args.image.stem}_annotated.jpg"
        annotate_image(image, detections, summary).save(annotated_path, quality=95)
        payload = {
            "image": str(args.image.resolve()), "weights": str(args.weights.resolve()),
            "confidence_threshold": args.confidence, "img_size": args.img_size,
            "coordinate_system": "xyxy pixels in EXIF-oriented original image",
            "image_width": image.width, "image_height": image.height,
            "summary": summary, "detections": detections,
        }
        if reference is not None:
            payload["size_estimation"] = reference
        save_json(args.output_dir / f"{args.image.stem}_detections.json", payload)
        if args.plot_counts:
            plot_counts(summary["counts"], args.output_dir / f"{args.image.stem}_counts.png")
        print(format_summary(summary))
        if reference is not None:
            print("\nApproximate horizontal bounding-box widths (not true fruit diameters)")
            for index, detection in enumerate(detections, 1):
                width = detection["size_estimation"]["approximate_width_cm"]
                print(f"{index}. {detection['class_name'].title()}: approximately {width:.2f} cm")
            if not detections:
                print("No detections; no fruit widths estimated.")
            print("Assumes the reference and fruit are at approximately the same depth in this image.")
        print(f"\nAnnotated image: {annotated_path}")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Prediction error: {error}\n")


if __name__ == "__main__":
    main()
