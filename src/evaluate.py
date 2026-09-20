"""Evaluate real labeled images and export overall and per-class metrics."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import BEST_MODEL, CLASS_NAMES, DATASET, ROOT, load_detector, package_versions, positive_int, save_json, select_device
from src.preprocessing import save_dataset_config, validate_dataset


def collect_metrics(metrics) -> dict:
    """Use Ultralytics' per-class F1 at its selected operating point."""
    box = metrics.box
    per_class = []
    class_indices = {int(class_id): index for index, class_id in enumerate(box.ap_class_index)}
    for class_id, name in enumerate(CLASS_NAMES):
        row = {"class_id": class_id, "class_name": name}
        counts = getattr(metrics, "nt_per_class", None)
        row["ground_truth_objects"] = int(counts[class_id]) if counts is not None else None
        if class_id not in class_indices:
            # A class without ground truth has no measurable AP on this split.
            row.update(dict.fromkeys(("precision", "recall", "f1", "map50", "map50_95")))
        else:
            index = class_indices[class_id]
            row.update(precision=float(box.p[index]), recall=float(box.r[index]),
                       f1=float(box.f1[index]), map50=float(box.ap50[index]), map50_95=float(box.ap[index]))
        per_class.append(row)
    f1_values = [row["f1"] for row in per_class if row["f1"] is not None]
    if not f1_values:
        raise ValueError("Evaluation returned no per-class metrics. Inspect labels and validation logs.")
    return {
        "overall": {"precision": float(box.mp), "recall": float(box.mr),
                    "f1": sum(f1_values) / len(f1_values),
                    "map50": float(box.map50), "map50_95": float(box.map)},
        "per_class": per_class,
        "metric_notes": (
            "Macro averages over classes with ground truth. Precision, recall and F1 use "
            "Ultralytics' confidence operating point selected for best mean F1 at IoU 0.5. "
            "F1 is the mean of per-class F1, not the harmonic mean of overall precision and recall. "
            "AP integrates the precision-recall curve; null means no ground truth for that class."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--weights", type=Path, default=BEST_MODEL)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--img-size", type=positive_int, default=640)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="fruitvision_test", help="Unique evaluation run folder.")
    args = parser.parse_args()
    try:
        if Path(args.name).name != args.name or args.name in (".", ".."):
            raise ValueError("--name must be a single folder name.")
        run_dir = ROOT / "outputs/metrics" / args.name
        if run_dir.exists():
            raise ValueError(f"Evaluation already exists: {run_dir}. Choose a new --name.")
        # Inspect all splits to catch exact train/test leakage before reporting results.
        config, dataset_report = validate_dataset(args.data)
        model = load_detector(args.weights)
        device = select_device(args.device)
        resolved_path = save_dataset_config(config, run_dir)
        # IoU = overlap area / union area for predicted and ground-truth boxes.
        # mAP50 uses IoU >= .50; mAP50-95 averages .50, .55, ..., .95.
        # A low confidence floor preserves the precision-recall curve for AP.
        metrics = model.val(
            data=str(resolved_path), split=args.split, imgsz=args.img_size,
            batch=args.batch_size, device=device, conf=0.001, augment=False,
            project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            plots=True, workers=0, seed=42, deterministic=True,
            # Save the same evaluation predictions for error analysis without
            # rerunning evaluation or changing its confidence operating point.
            save_txt=True, save_conf=True,
        )
        report = collect_metrics(metrics)
        matrix = metrics.confusion_matrix.matrix
        save_json(run_dir / "confusion_matrix.json", {
            "matrix": matrix.tolist(), "labels": [*CLASS_NAMES, "background"],
            "rows": "predicted class", "columns": "ground-truth class",
            "confidence_threshold": 0.001, "iou_threshold": 0.45,
            "note": "Ultralytics confusion-matrix thresholds differ from AP and the diagnostic analysis at confidence 0.25 / IoU 0.5.",
        })
        save_json(run_dir / "image_metrics.json", metrics.box.image_metrics)
        report["run"] = {"weights": str(args.weights.resolve()), "split": args.split,
                         "img_size": args.img_size, "batch_size": args.batch_size,
                         "confidence_floor": 0.001, "seed": 42, "device": device,
                         "packages": package_versions(), "dataset_summary": dataset_report,
                         "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        save_json(run_dir / "metrics.json", report)
        for filename, rows in (("summary.csv", [report["overall"]]), ("per_class.csv", report["per_class"])):
            with (run_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for name, value in report["overall"].items():
            print(f"{name}: {value:.4f}")
        print(f"Metrics and evaluation plots: {run_dir}")
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as error:
        parser.exit(2, f"Evaluation error: {error}\n")


if __name__ == "__main__":
    main()
