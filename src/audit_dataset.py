"""Audit the current processed dataset, including corrections since ZIP import."""

import argparse
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CLASS_NAMES, DATASET, ROOT, save_json
from src.preprocessing import inspect_pairs, validate_dataset


def audit_processed(config_path: Path = DATASET) -> dict:
    """Validate all splits and fingerprint exact data used for an experiment."""
    config, splits = validate_dataset(config_path)
    root = Path(config["path"])
    image_hashes, pixels = defaultdict(list), defaultdict(list)
    files, empty, repeated = [], [], []
    for split in ("train", "val", "test"):
        records = inspect_pairs(root / "images" / split, root / "labels" / split)
        splits[split]["labels"] = len(records)
        splits[split]["total_objects"] = sum(splits[split]["objects_per_class"].values())
        for record in records:
            for kind in ("image", "label"):
                path = record[kind]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                relative = path.relative_to(root).as_posix()
                files.append({"path": relative, "sha256": digest})
                if kind == "image":
                    image_hashes[digest].append(relative)
            if not record["boxes"]:
                empty.append(record["label"].relative_to(root).as_posix())
            seen = set()
            for box in record["boxes"]:
                if box in seen:
                    repeated.append({"file": record["label"].relative_to(root).as_posix(), "box": box})
                seen.add(box)
            with Image.open(record["image"]) as image:
                digest = hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()
                pixels[digest].append(record["image"].relative_to(root).as_posix())
    pixel_duplicates = [names for names in pixels.values() if len(names) > 1]
    leakage = [group for group in pixel_duplicates if len({name.split('/')[1] for name in group}) > 1]
    if leakage:
        raise ValueError(f"Decoded-pixel cross-split leakage: {leakage}")
    files.sort(key=lambda entry: entry["path"])
    fingerprint = hashlib.sha256("\n".join(f"{entry['path']} {entry['sha256']}" for entry in files).encode()).hexdigest()
    return {"status": "valid", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_directory": str(root), "class_mapping": dict(enumerate(CLASS_NAMES)),
            "source": config.get("roboflow"), "splits": splits,
            "total_images": sum(split["images"] for split in splits.values()),
            "total_labels": sum(split["labels"] for split in splits.values()),
            "total_objects": sum(split["total_objects"] for split in splits.values()),
            "objects_per_class": {name: sum(split["objects_per_class"][name] for split in splits.values()) for name in CLASS_NAMES},
            "empty_label_files": empty, "duplicate_annotation_rows": repeated,
            "exact_duplicate_images": [names for names in image_hashes.values() if len(names) > 1],
            "identical_pixel_images": pixel_duplicates, "cross_split_duplicates": leakage,
            "missing_pairs": [], "invalid_labels": [], "dataset_sha256": fingerprint, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dataset_audit_baseline.json")
    args = parser.parse_args()
    try:
        report = audit_processed(args.data)
        save_json(args.output, report)
        print(f"Valid: {report['total_images']} images, {report['total_objects']} objects; audit: {args.output}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        save_json(args.output, {"status": "failed", "error": str(error)})
        parser.exit(2, f"Audit error: {error}\n")


if __name__ == "__main__":
    main()
