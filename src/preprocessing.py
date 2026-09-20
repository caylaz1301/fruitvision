"""Validate YOLO annotations and optionally split already-annotated images."""

import argparse
import hashlib
import math
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CLASS_NAMES, DATASET, ROOT, save_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_exact_duplicate_rows(path: Path) -> dict:
    """Remove only repeated, identical nonempty rows and return an audit record."""
    before = path.read_text(encoding="utf-8")
    read_labels(path)  # Never repair an invalid annotation by silently dropping it.
    seen, retained, removed = set(), [], []
    for number, line in enumerate(before.splitlines(), 1):
        if line.strip() and line in seen:
            removed.append(number)
        else:
            retained.append(line)
            seen.add(line)
    after = "\n".join(retained) + ("\n" if before.endswith("\n") else "")
    if removed:
        path.write_text(after, encoding="utf-8")
    return {"file": str(path), "before": before, "after": after if removed else before,
            "removed_line_numbers": removed, "issue": "identical duplicate annotation rows",
            "action": "removed redundant exact rows" if removed else "unchanged",
            "reason": "Identical rows encode the same class and bounding box; distinct rows are retained."}


def parse_detection(line: str) -> tuple[int, float, float, float, float]:
    """Validate one detection box; polygon conversion belongs to the importer."""
    class_text, *coordinates = line.split()
    class_id = int(class_text)
    x, y, width, height = map(float, coordinates)
    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError(f"class ID must be 0–{len(CLASS_NAMES) - 1}")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)):
        raise ValueError("coordinates must be finite and normalized to [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    # Centers alone are insufficient: the full box must fit inside the image.
    if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
        raise ValueError("box extends outside image boundaries")
    return class_id, x, y, width, height


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read normalized center/width/height boxes, rejecting malformed labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {path}. Use an empty .txt only for a verified background image.")
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            box = parse_detection(line)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected class_id x_center y_center width height; {error}") from error
        boxes.append(box)
    return boxes


def inspect_pairs(images_dir: Path, labels_dir: Path) -> list[dict]:
    """Check image readability, matching labels, and duplicate filenames."""
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}. See data/README.md for dataset placement.")
    records, expected_labels = [], set()
    for image_path in images:
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")
        if label_path in expected_labels:
            raise ValueError(f"Two images share a label filename: {label_path}")
        expected_labels.add(label_path)
        with Image.open(image_path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"{image_path}: apply EXIF orientation before annotating/exporting.")
            image.load()
            width, height = image.size
        boxes = read_labels(label_path)
        records.append({"image": image_path, "label": label_path, "relative": relative,
                        "width": width, "height": height, "boxes": boxes})
    orphan_labels = set(labels_dir.rglob("*.txt")) - expected_labels
    if orphan_labels:
        raise ValueError(f"Label without a supported image: {sorted(orphan_labels)[0]}")
    return records


def validate_dataset(config_path: Path = DATASET, splits: tuple[str, ...] = ("train", "val", "test")) -> tuple[dict, dict]:
    """Resolve paths explicitly and reject exact image leakage between splits."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("names") != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"Dataset names must use this exact mapping: {dict(enumerate(CLASS_NAMES))}")
    if "download" in config:
        raise ValueError("Remove the YAML download field: this project uses local datasets only.")
    root = (config_path.resolve().parent / config.get("path", "../data/processed")).resolve()
    resolved = {"path": str(root), "names": config["names"]}
    if "roboflow" in config:
        resolved["roboflow"] = config["roboflow"]
    for split in ("train", "val", "test"):
        # A single directory per split keeps the images/labels relationship explicit.
        expected = f"images/{split}"
        if config.get(split) != expected:
            raise ValueError(f"Set {split}: {expected} in {config_path}; see data/README.md.")
        resolved[split] = expected
    report, seen = {}, {}
    for split in splits:
        records = inspect_pairs(root / "images" / split, root / "labels" / split)
        counts = dict.fromkeys(CLASS_NAMES, 0)
        for record in records:
            fingerprint = hashlib.sha256(record["image"].read_bytes()).hexdigest()
            if fingerprint in seen and seen[fingerprint] != split:
                raise ValueError(f"Data leakage: {record['image']} duplicates an image in {seen[fingerprint]}.")
            seen[fingerprint] = split
            for class_id, *_ in record["boxes"]:
                counts[CLASS_NAMES[class_id]] += 1
        if not sum(counts.values()):
            raise ValueError(f"{split} has no annotated fruit objects. Add real bounding-box labels.")
        report[split] = {"images": len(records), "objects_per_class": counts}
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            print(f"Warning: {split} has no annotations for: {', '.join(missing)}.")
    return resolved, report


def save_dataset_config(config: dict, directory: Path) -> Path:
    """An absolute dataset root avoids Ultralytics' global dataset-directory setting."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset.resolved.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def split_dataset(raw: Path, output: Path, train_ratio: float, val_ratio: float, seed: int) -> dict:
    """Copy paired files into seeded splits without changing pixels or annotations."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Ratios must be positive and leave a positive test fraction.")
    if output.exists() and any(p.is_file() and p.name != ".gitkeep" for p in output.rglob("*")):
        raise ValueError(f"Output is not empty: {output}. Choose a new --output directory to avoid overwriting data.")
    records = inspect_pairs(raw / "images", raw / "labels")
    fingerprints = [hashlib.sha256(record["image"].read_bytes()).hexdigest() for record in records]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Duplicate image files found. Deduplicate the raw dataset before splitting.")
    random.Random(seed).shuffle(records)
    train_end = int(len(records) * train_ratio)
    val_end = train_end + int(len(records) * val_ratio)
    partitions = {"train": records[:train_end], "val": records[train_end:val_end], "test": records[val_end:]}
    if any(not items for items in partitions.values()):
        raise ValueError("Too few images for these ratios: each split must contain at least one image.")
    manifest = {"seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio,
                "source": str(raw.resolve()), "splits": {}}
    for split, items in partitions.items():
        manifest["splits"][split] = []
        for record in items:
            for kind, key in (("images", "image"), ("labels", "label")):
                relative = record["relative"] if kind == "images" else record["relative"].with_suffix(".txt")
                destination = output / kind / split / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record[key], destination)
            manifest["splits"][split].append({"image": record["relative"].as_posix(),
                "sha256": hashlib.sha256(record["image"].read_bytes()).hexdigest()})
    save_json(output / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Validate all three processed splits.")
    check.add_argument("--data", type=Path, default=DATASET)
    split = commands.add_parser("split", help="Split raw/images and raw/labels (already annotated).")
    split.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    split.add_argument("--output", type=Path, default=ROOT / "data/processed")
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--val-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            _, report = validate_dataset(args.data)
            for name, values in report.items():
                print(f"{name}: {values}")
        else:
            manifest = split_dataset(args.raw, args.output, args.train_ratio, args.val_ratio, args.seed)
            print({name: len(items) for name, items in manifest["splits"].items()})
            print("Split complete. Validate the processed dataset before training.")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"Dataset error: {error}\n")


if __name__ == "__main__":
    main()
