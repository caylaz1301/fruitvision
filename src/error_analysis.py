"""Describe real saved evaluation predictions at a fixed diagnostic threshold."""

import argparse
from pathlib import Path

import yaml
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CLASS_NAMES, DATASET, ROOT, save_json
from src.counting import summarize_detections
from src.preprocessing import read_labels
from src.visualization import annotate_image


def box_iou(first: list[float], second: list[float]) -> float:
    """IoU measures overlap relative to the union of two axis-aligned boxes."""
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def xywh_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = values
    return [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]


def match_detections(truth: list[dict], predictions: list[dict], iou_threshold: float = 0.5) -> dict:
    """Greedy confidence-ordered, class-aware matching for descriptive analysis.

    This is a fixed-threshold diagnostic, not a replacement AP calculation.
    Wrong-class overlaps remain both FP and FN; each box is matched at most once.
    """
    available = set(range(len(truth)))
    matches, unmatched = [], []
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True):
        prediction = predictions[index]
        candidates = [(box_iou(prediction["bbox_xyxy"], truth[j]["bbox_xyxy"]), j) for j in available
                      if truth[j]["class_id"] == prediction["class_id"]]
        overlap, target = max(candidates, default=(0, -1))
        if overlap >= iou_threshold and target >= 0:
            matches.append({"prediction": index, "ground_truth": target, "iou": overlap})
            available.remove(target)
        else:
            unmatched.append(index)
    confusions, used_targets = [], set()
    for index in unmatched:
        prediction = predictions[index]
        candidates = [(box_iou(prediction["bbox_xyxy"], truth[j]["bbox_xyxy"]), j) for j in available - used_targets
                      if truth[j]["class_id"] != prediction["class_id"]]
        overlap, target = max(candidates, default=(0, -1))
        if overlap >= iou_threshold and target >= 0:
            confusions.append({"prediction": index, "ground_truth": target, "iou": overlap,
                               "true_class": truth[target]["class_name"], "predicted_class": prediction["class_name"]})
            used_targets.add(target)
    return {"tp": len(matches), "fp": len(unmatched), "fn": len(available),
            "matches": matches, "false_positive_indices": unmatched,
            "false_negative_indices": sorted(available), "class_confusions": confusions}


def analyze_saved_predictions(data: Path, evaluation: Path, output: Path) -> dict:
    """Read evaluation exports once and save a reproducible per-image error index."""
    import json

    metrics = json.loads((evaluation / "metrics.json").read_text())
    split = metrics["run"]["split"]
    config = yaml.safe_load(data.read_text())
    root = (data.resolve().parent / config["path"]).resolve()
    results = []
    for image_path in sorted((root / "images" / split).rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        truth = [{"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                  "bbox_xyxy": xywh_to_xyxy([x, y, w, h], width, height)}
                 for class_id, x, y, w, h in read_labels(root / "labels" / split / image_path.relative_to(root / "images" / split).with_suffix('.txt'))]
        prediction_path = evaluation / "labels" / f"{image_path.stem}.txt"
        predictions, low_confidence = [], []
        for line in prediction_path.read_text().splitlines() if prediction_path.exists() else []:
            class_text, x, y, w, h, confidence_text = line.split()
            class_id, confidence = int(class_text), float(confidence_text)
            prediction = {"class_id": class_id, "class_name": CLASS_NAMES[class_id], "confidence": confidence,
                          "bbox_xyxy": xywh_to_xyxy(list(map(float, [x, y, w, h])), width, height)}
            if confidence >= 0.25:
                predictions.append(prediction)
            elif confidence >= 0.05:
                low_confidence.append(prediction)
        diagnostic = match_detections(truth, predictions)
        small = [index for index, box in enumerate(truth) if
                 (box["bbox_xyxy"][2]-box["bbox_xyxy"][0]) * (box["bbox_xyxy"][3]-box["bbox_xyxy"][1]) / (width*height) < 0.01]
        results.append({"image": str(image_path), "ground_truth": truth, "predictions": predictions,
                        "low_confidence_predictions": low_confidence, "small_ground_truth_indices": small,
                        "small_false_negatives": len(set(small) & set(diagnostic["false_negative_indices"])),
                        "crowded": len(truth) >= 5, "multiple_classes": len({box["class_id"] for box in truth}) > 1,
                        **diagnostic})
    report = {"split": split, "evaluation": str(evaluation.resolve()),
              "confidence_threshold": 0.25, "iou_threshold": 0.5,
              "small_definition": "ground-truth box area < 1% of image area; not the COCO size definition",
              "crowded_definition": "at least five annotated objects",
              "notes": "Fixed thresholds chosen before evaluation; greedy diagnostic matching can differ from Ultralytics AP matching. No hyperparameters were tuned.",
              "totals": {key: sum(item[key] for item in results) for key in ("tp", "fp", "fn", "small_false_negatives")},
              "images": results}
    save_json(output / "error_analysis.json", report)
    return report


def save_comparison(record: dict, output: Path) -> None:
    """Save labeled ground truth beside actual predictions, with distinct headings."""
    from PIL import ImageDraw

    with Image.open(record["image"]) as source:
        original = source.convert("RGB")
    truth = original.copy()
    draw = ImageDraw.Draw(truth)
    for box in record["ground_truth"]:
        draw.rectangle(box["bbox_xyxy"], outline="lime", width=2)
        draw.text(tuple(box["bbox_xyxy"][:2]), box["class_name"], fill="black", stroke_width=1, stroke_fill="white")
    predicted = annotate_image(original, record["predictions"], summarize_detections(record["predictions"], CLASS_NAMES))
    canvas = Image.new("RGB", (original.width * 2, predicted.height + 28), "white")
    canvas.paste(truth, (0, 28))
    canvas.paste(predicted, (original.width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), "Ground truth (source labels)", fill="black")
    draw.text((original.width + 5, 6), f"Predictions: TP {record['tp']} / FP {record['fp']} / FN {record['fn']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def save_review_examples(report: dict, output: Path) -> list[dict]:
    """Select condition-based and error-ranked cases, including imperfect results."""
    records = report["images"]
    selected = {}
    for category, predicate in (
        ("single_object", lambda r: len(r["ground_truth"]) == 1),
        ("multiple_objects", lambda r: len(r["ground_truth"]) >= 2),
        ("multiple_classes", lambda r: r["multiple_classes"]),
        ("true_positive", lambda r: r["tp"] > 0 and r["fp"] == r["fn"] == 0),
    ):
        match = next((record for record in records if predicate(record)), None)
        if match:
            selected[category] = match
    selected["crowded"] = max(records, key=lambda r: len(r["ground_truth"]))
    selected["small_objects"] = max(records, key=lambda r: len(r["small_ground_truth_indices"]))
    for category, key in (("false_positives", "fp"), ("false_negatives", "fn")):
        record = max(records, key=lambda r: r[key])
        if record[key]:
            selected[category] = record
    confusions = [record for record in records if record["class_confusions"]]
    if confusions:
        selected["class_confusion"] = max(confusions, key=lambda r: max(r["predictions"][c["prediction"]]["confidence"] for c in r["class_confusions"]))
    for record in records:
        if any(p["class_id"] == t["class_id"] and box_iou(p["bbox_xyxy"], t["bbox_xyxy"]) >= 0.5
               for p in record["low_confidence_predictions"]
               for t in (record["ground_truth"][i] for i in record["false_negative_indices"])):
            selected["low_confidence_match"] = record
            break
    index = []
    for category, record in selected.items():
        path = output / "examples" / f"{category}.jpg"
        save_comparison(record, path)
        index.append({"category": category, "image": record["image"], "comparison": str(path),
                      "tp": record["tp"], "fp": record["fp"], "fn": record["fn"],
                      "class_confusions": record["class_confusions"]})
    save_json(output / "review_index.json", {"examples": index,
              "selection": "Condition-based first filename, highest-error cases, and highest-confidence observed confusion; no tuning."})
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/baseline_validation")
    args = parser.parse_args()
    report = analyze_saved_predictions(args.data, args.evaluation, args.output)
    save_review_examples(report, args.output)
    print(f"Analyzed {len(report['images'])} {report['split']} images: {report['totals']}")


if __name__ == "__main__":
    main()
