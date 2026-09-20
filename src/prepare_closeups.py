"""Build a separate training copy with label-preserving close-ups of real fruit.

Each crop encloses ALL annotated objects with 25% padding on each side.
Validation/test pixels and labels are copied unchanged; no source file is edited.
"""

import argparse
import math
import shutil
from pathlib import Path

from PIL import Image
import yaml

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATASET, ROOT, save_json
from src.preprocessing import inspect_pairs, parse_detection, validate_dataset


def closeup_geometry(boxes: list[tuple], size: tuple[int, int], padding: float = 0.25):
    """Return integer crop bounds and normalized boxes without dropping objects.

    The union of existing boxes avoids inventing annotations or cutting off a
    labeled fruit. Pixel coordinates change, so centers and sizes are normalized
    again against the crop dimensions rather than the original image dimensions.
    """
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("Padding must be finite and nonnegative.")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not boxes:
        return None
    pixels = []
    for box in boxes:
        class_id, x, y, w, h = parse_detection(" ".join(map(str, box)))
        pixels.append((class_id, (x-w/2)*width, (y-h/2)*height,
                       (x+w/2)*width, (y+h/2)*height))
    x1, y1 = min(b[1] for b in pixels), min(b[2] for b in pixels)
    x2, y2 = max(b[3] for b in pixels), max(b[4] for b in pixels)
    pad_x, pad_y = (x2-x1)*padding, (y2-y1)*padding
    left, top = max(0, math.floor(x1-pad_x)), max(0, math.floor(y1-pad_y))
    right, bottom = min(width, math.ceil(x2+pad_x)), min(height, math.ceil(y2+pad_y))
    crop_w, crop_h = right-left, bottom-top
    if crop_w < 2 or crop_h < 2:
        return None
    normalized = [(c, ((a+b)/2-left)/crop_w, ((d+e)/2-top)/crop_h,
                   (b-a)/crop_w, (e-d)/crop_h) for c, a, d, b, e in pixels]
    return (left, top, right, bottom), normalized


def prepare_closeups(source: Path, destination: Path, padding: float = 0.25) -> dict:
    config, before = validate_dataset(source)
    root = Path(config["path"])
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Destination already exists; choose a new directory to preserve prior data.")
    if root == destination or root in destination.parents:
        raise ValueError("Use a destination outside the original dataset.")
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("Padding must be finite and nonnegative.")
    records = inspect_pairs(root / "images/train", root / "labels/train")
    # Copies, not symlinks: future operations on this experiment cannot alter sources.
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            shutil.copytree(root / kind / split, destination / kind / split)
    crops, skipped = [], []
    for record in records:
        geometry = closeup_geometry(record["boxes"], (record["width"], record["height"]), padding)
        if geometry is None:
            skipped.append({"image": str(record["relative"]), "reason": "empty label or crop smaller than two pixels"})
            continue
        bounds, boxes = geometry
        l, t, r, b = bounds
        # Avoid near-identical full-image copies and report every skipped crop.
        if (r-l)*(b-t) >= 0.95 * record["width"] * record["height"]:
            skipped.append({"image": str(record["relative"]), "reason": "crop covers at least 95% of original"})
            continue
        relative = record["relative"].with_name("closeup__" + record["relative"].stem + ".png")
        image_path = destination / "images/train" / relative
        label_path = destination / "labels/train" / relative.with_suffix(".txt")
        if image_path.exists() or label_path.exists():
            raise ValueError(f"Crop filename collision: {relative}")
        with Image.open(record["image"]) as image:
            image.convert("RGB").crop(bounds).save(image_path)
        lines = [" ".join([str(box[0]), *(f"{v:.10f}" for v in box[1:])]) for box in boxes]
        for line in lines:
            parse_detection(line)
        label_path.write_text("\n".join(lines) + "\n")
        crops.append({"source": str(record["relative"]), "crop": str(relative),
                      "bounds_xyxy": list(bounds), "objects_retained": len(boxes)})
    derived = {**config, "path": str(destination)}
    config_path = destination / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(derived, sort_keys=False))
    _, after = validate_dataset(config_path)
    report = {"source_config": str(source.resolve()), "configuration": str(config_path),
              "padding": padding, "before": before, "after": after,
              "crops_added": len(crops), "crops": crops, "skipped": skipped,
              "note": "Only training gets extra crops. All original boxes retained; no new annotations inferred. Source data untouched."}
    save_json(destination / "closeup_manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "data/closeups_v1")
    args = parser.parse_args()
    try:
        report = prepare_closeups(args.data, args.output)
        print(f"Added {report['crops_added']} training close-ups. Config: {report['configuration']}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"Close-up preparation error: {error}\n")
