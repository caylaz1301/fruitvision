"""Evaluate five fixed image conditions on validation, without model training."""

import argparse
import csv
import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.brightness_test import adjust_brightness
from experiments.blur_test import gaussian_blur
from experiments.resolution_test import reduce_resolution
from src.audit_dataset import audit_processed
from src.common import BEST_MODEL, CLASS_NAMES, DATASET, ROOT, load_detector, package_versions, save_json, select_device
from src.evaluate import collect_metrics
from src.preprocessing import inspect_pairs

# Prespecified severities, not selected after inspecting evaluation results.
CONDITIONS = {
    "normal": {"transform": "identity"},
    "dark": {"transform": "brightness", "factor": 0.5},
    "bright": {"transform": "brightness", "factor": 1.5},
    "blur": {"transform": "gaussian_blur", "radius": 2.0},
    "low_resolution": {"transform": "resolution", "scale": 0.25,
                       "downsample": "BOX", "upsample": "BILINEAR"},
}
METRICS = ("precision", "recall", "f1", "map50", "map50_95")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_image(image: Image.Image, condition: str) -> Image.Image:
    """Each condition starts from the same decoded original, never another variant."""
    settings = CONDITIONS[condition]
    if condition == "normal":
        return image.convert("RGB").copy()
    if condition in ("dark", "bright"):
        return adjust_brightness(image, settings["factor"])
    if condition == "blur":
        return gaussian_blur(image, settings["radius"])
    return reduce_resolution(image, settings["scale"])


def prepare_condition(records: list[dict], directory: Path, condition: str) -> Path:
    """Create a separate validation copy and verify dimensions and exact label bytes."""
    if directory.exists():
        raise ValueError(f"Refusing to overwrite existing condition: {directory}")
    # Reject an overlapping output path before making any directory or writing files.
    for record in records:
        for key in ("image", "label"):
            source = record[key].resolve()
            if directory.resolve() in source.parents or source.parent in directory.resolve().parents:
                raise ValueError("Condition output must be separate from source images and labels.")
    manifest = []
    for record in records:
        relative = record["relative"].with_suffix(".png")
        destination = directory / "images/val" / relative
        label = directory / "labels/val" / relative.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record["image"]) as original:
            changed = transform_image(original, condition)
            if changed.size != original.size:
                raise ValueError("Transformation changed image dimensions; boxes would need adjustment.")
            # Lossless PNG for EVERY condition avoids condition-specific JPEG artifacts.
            changed.save(destination, format="PNG")
        shutil.copyfile(record["label"], label)
        if label.read_bytes() != record["label"].read_bytes():
            raise ValueError(f"Label copy differs: {label}")
        manifest.append({"source_image": str(record["image"]), "image": relative.as_posix(),
                         "image_sha256": file_hash(destination), "label_sha256": file_hash(label),
                         "width": record["width"], "height": record["height"]})
    checked = inspect_pairs(directory / "images/val", directory / "labels/val")
    if len(checked) != len(records):
        raise ValueError("Transformed image count differs from original validation split.")
    config = {"path": str(directory.resolve()), "names": dict(enumerate(CLASS_NAMES)),
              # Ultralytics requires a train key even for val-only YAML. This dataset
              # is never passed to model.train(); both aliases refer only to validation.
              "train": "images/val", "val": "images/val"}
    config_path = directory / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    save_json(directory / "manifest.json", {"condition": condition, "settings": CONDITIONS[condition],
              "images": len(checked), "objects": sum(len(r["boxes"]) for r in checked),
              "labels_byte_identical": True, "files": manifest})
    return config_path


def degradation(normal: float | None, changed: float | None) -> dict:
    """Positive means worse; negative means improvement. Zero reference is undefined."""
    if normal is None or changed is None:
        return {"relative_percent": None, "percentage_points": None}
    return {"relative_percent": 100 * (normal - changed) / normal if normal != 0 else None,
            "percentage_points": 100 * (normal - changed)}


def compare_results(reports: dict) -> tuple[list[dict], list[dict]]:
    overall, per_class = [], []
    normal = reports["normal"]
    normal_classes = {row["class_id"]: row for row in normal["per_class"]}
    for condition, report in reports.items():
        row = {"condition": condition, **report["overall"]}
        for metric in METRICS:
            delta = degradation(normal["overall"][metric], row[metric])
            row[f"{metric}_degradation_percent"] = delta["relative_percent"]
            row[f"{metric}_drop_pp"] = delta["percentage_points"]
        overall.append(row)
        for result in report["per_class"]:
            reference = normal_classes[result["class_id"]]
            if result["ground_truth_objects"] != reference["ground_truth_objects"]:
                raise ValueError("Ground-truth counts changed between conditions.")
            row = {"condition": condition, **result}
            for metric in METRICS:
                delta = degradation(reference[metric], result[metric])
                row[f"{metric}_degradation_percent"] = delta["relative_percent"]
                row[f"{metric}_drop_pp"] = delta["percentage_points"]
            per_class.append(row)
    return overall, per_class


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(overall: list[dict], per_class: list[dict], output: Path) -> None:
    """Plot real aggregate scores, relative degradation, and class AP changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    labels = [row["condition"] for row in overall]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, metric in enumerate(METRICS):
        ax.bar([j + (i - 2) * 0.15 for j in range(len(labels))],
               [row[metric] * 100 for row in overall], width=0.15, label=metric)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Score (%)", ylim=(0, 100),
           title="Frozen baseline: validation robustness")
    ax.legend(ncol=5, fontsize=9)
    fig.savefig(output / "overall_metrics.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(labels, [row["map50_95_degradation_percent"] for row in overall])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(ylabel="Relative mAP50–95 degradation (%)", title="Positive = worse than normal")
    fig.savefig(output / "degradation.png", dpi=150)
    plt.close(fig)
    rows = {(row["condition"], row["class_id"]): row for row in per_class}
    values = [[rows[(condition, i)]["map50_95_drop_pp"] for condition in labels[1:]]
              for i in range(len(CLASS_NAMES))]
    bound = max(abs(v) for row in values for v in row) or 1
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    heat = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set(xticks=range(len(labels)-1), xticklabels=labels[1:], yticks=range(len(CLASS_NAMES)),
           yticklabels=CLASS_NAMES, title="Per-class AP50–95 drop (percentage points)")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if abs(value) > bound * .55 else "black")
    fig.colorbar(heat, ax=ax, label="Positive = worse")
    fig.savefig(output / "per_class_degradation.png", dpi=150)
    plt.close(fig)


def run_experiments(data: Path, weights: Path, output: Path, device: str) -> dict:
    """Run each validation condition once using identical evaluation settings."""
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Preserve it and choose a new --output.")
    audit = audit_processed(data)
    source_root = Path(audit["dataset_directory"])
    output = output.resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Experiment output must be separate from the source dataset.")
    records = inspect_pairs(source_root / "images/val", source_root / "labels/val")
    checkpoint_hash = file_hash(weights)
    device = select_device(device)
    output.mkdir(parents=True)
    metadata = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "weights": str(weights.resolve()), "checkpoint_sha256": checkpoint_hash,
                "source_dataset_sha256": audit["dataset_sha256"], "split": "val",
                "validation_counts": audit["splits"]["val"], "class_mapping": audit["class_mapping"],
                "conditions": CONDITIONS, "python": sys.version, "platform": platform.platform(),
                "packages": package_versions(), "device": device,
                "primary_comparison": "relative mAP50_95 degradation versus same-run normal",
                "note": "Diagnostic results: annotations are imperfect. MPS may be nondeterministic."}
    evaluation = dict(split="val", imgsz=640, batch=8, conf=0.001, iou=0.7, max_det=300,
                      augment=False, workers=0, seed=42, deterministic=True, device=device,
                      plots=True, save_txt=True, save_conf=True, verbose=False)
    metadata["evaluation_settings"] = evaluation
    save_json(output / "run_metadata.json", metadata)
    save_json(output / "source_audit.json", audit)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    reports = {}
    try:
        for condition in CONDITIONS:
            if file_hash(weights) != checkpoint_hash:
                raise ValueError("Frozen checkpoint changed during experiments.")
            print(f"Preparing and evaluating {condition} on {len(records)} validation images.", flush=True)
            config = prepare_condition(records, output / "datasets" / condition, condition)
            # Reload the SAME file for each condition; no training API is called.
            model = load_detector(weights)
            run_dir = output / "evaluations" / condition
            metrics = model.val(data=str(config), project=str(run_dir.parent), name=condition,
                                exist_ok=False, **evaluation)
            report = collect_metrics(metrics)
            expected = audit["splits"]["val"]["objects_per_class"]
            if any(row["ground_truth_objects"] != expected[row["class_name"]] for row in report["per_class"]):
                raise ValueError("Evaluated ground-truth counts do not match source labels.")
            report.update(condition=condition, transformation=CONDITIONS[condition],
                          checkpoint_sha256=checkpoint_hash,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
            save_json(run_dir / "metrics.json", report)
            write_csv(run_dir / "per_class.csv", report["per_class"])
            reports[condition] = report
            del model, metrics
        final_audit = audit_processed(data)
        if final_audit["dataset_sha256"] != audit["dataset_sha256"] or file_hash(weights) != checkpoint_hash:
            raise ValueError("Original dataset or frozen checkpoint changed during experiments.")
        overall, per_class = compare_results(reports)
        worst = min(overall, key=lambda row: row["map50_95"])["condition"]
        results = {"overall": overall, "per_class": per_class, "worst_condition_by_map50_95": worst,
                   "degradation_definition": "100 * (normal - condition) / normal; negative means improvement; null if normal is zero",
                   "metric_notes": reports["normal"]["metric_notes"]}
        save_json(output / "results.json", results)
        write_csv(output / "overall.csv", overall)
        write_csv(output / "per_class.csv", per_class)
        plot_comparison(overall, per_class, output / "figures")
        metadata.update(status="completed", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_dataset_unchanged=True, checkpoint_unchanged=True)
        save_json(output / "run_metadata.json", metadata)
        return results
    except Exception as error:
        metadata.update(status="failed", error=str(error), ended_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / "run_metadata.json", metadata)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--weights", type=Path, default=BEST_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/robustness")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or CUDA index")
    args = parser.parse_args()
    try:
        results = run_experiments(args.data, args.weights, args.output, args.device)
        print(f"Completed. Worst mAP50–95 condition: {results['worst_condition_by_map50_95']}")
        print(f"Results: {args.output}")
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as error:
        parser.exit(2, f"Robustness error: {error}\n")


if __name__ == "__main__":
    main()
