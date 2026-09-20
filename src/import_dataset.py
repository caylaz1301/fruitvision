"""Import the local Roboflow Fruit ZIP, preserving splits and converting polygons."""

import argparse
import hashlib
import math
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import yaml
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CLASS_NAMES, ROOT, save_json
from src.preprocessing import IMAGE_SUFFIXES, parse_detection, validate_dataset
from src.visualization import plot_counts

SPLIT_MAPPING = {"train": "train", "valid": "val", "test": "test"}
SOURCE = {
    "workspace": "simons-workspace-l89qb", "project": "fruit-smrhb-tp0nb",
    "version": 1, "license": "CC BY 4.0",
    "url": "https://universe.roboflow.com/simons-workspace-l89qb/fruit-smrhb-tp0nb/dataset/1",
}
METADATA_FILES = {"data.yaml", "README.roboflow.txt", "README.dataset.txt"}


def convert_annotation(line: str) -> tuple[str, bool]:
    """Return a validated detection line and whether a polygon was converted.

    Polygon coordinates are normalized. Their extrema form an axis-aligned box;
    the class ID stays unchanged. No clipping or silent label removal is allowed.
    """
    parts = line.split()
    if not parts:
        raise ValueError("empty annotation line")
    class_id = int(parts[0])
    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError("class ID must be 0–9")
    if len(parts) == 5:
        parse_detection(line)
        return line.strip(), False
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError("polygon needs at least three complete x/y pairs")
    coordinates = [float(value) for value in parts[1:]]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
        raise ValueError("polygon coordinates must be finite and within [0, 1]")
    xs, ys = coordinates[::2], coordinates[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    # 17 significant digits preserve Python float precision deterministically.
    converted = f"{class_id} " + " ".join(format(value, ".17g") for value in box)
    parse_detection(converted)
    return converted, True


def inspect_archive(archive: ZipFile, report: dict) -> dict:
    """Index safe members without extractall, rejecting ambiguous destinations."""
    members = {}
    case_names = {}
    total_size = 0
    if len(archive.infolist()) > 50000:
        raise ValueError("Archive has too many entries (limit: 50,000).")
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                or not path.parts or str(path) != name.rstrip("/")):
            raise ValueError(f"Unsafe archive path: {name}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError(f"Unsupported archive entry type: {name}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted archive entry: {name}")
        total_size += info.file_size
        if info.file_size > 100 * 1024**2 or total_size > 5 * 1024**3:
            raise ValueError("Archive exceeds safe uncompressed size limits (100 MiB/file, 5 GiB total).")
        if info.is_dir():
            continue
        if name.casefold() in case_names:
            previous = case_names[name.casefold()]
            identical_metadata = name == previous and name in METADATA_FILES and archive.read(info) == archive.read(members[previous])
            report["duplicate_filenames"].append({"name": name, "identical_metadata": identical_metadata})
            if not identical_metadata:
                raise ValueError(f"Ambiguous duplicate archive filename: {name}")
            continue
        case_names[name.casefold()] = name
        members[name] = info
    report["archive_entries"] = len(archive.infolist())
    report["uncompressed_bytes"] = total_size
    return members


def verify_source(archive: ZipFile, members: dict) -> dict:
    """Verify the source class order and retain Roboflow attribution verbatim."""
    if "data.yaml" not in members:
        raise ValueError("Archive is missing root data.yaml.")
    config = yaml.safe_load(archive.read(members["data.yaml"]))
    if not isinstance(config, dict):
        raise ValueError("Source data.yaml must be a mapping.")
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != dict(enumerate(CLASS_NAMES)) or config.get("nc") != len(CLASS_NAMES):
        raise ValueError(f"Source class mapping must be exactly {dict(enumerate(CLASS_NAMES))}")
    metadata = config.get("roboflow", {})
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in SOURCE.items()):
        raise ValueError("Roboflow metadata does not match the requested Fruit project/version/license.")
    for source_split, target_split in SPLIT_MAPPING.items():
        key = "val" if source_split == "valid" else source_split
        # Exported YAML's ../ paths are metadata; do not follow them on disk.
        if config.get(key) not in (f"../{source_split}/images", f"{source_split}/images"):
            raise ValueError(f"Unexpected source split path: {key}: {config.get(key)}")
        for kind in ("images", "labels"):
            if not any(name.startswith(f"{source_split}/{kind}/") for name in members):
                raise ValueError(f"Missing source folder: {source_split}/{kind}")
    return config


def duplicate_groups(hashes: dict[str, list[str]]) -> list[list[str]]:
    return [sorted(names) for _, names in sorted(hashes.items()) if len(names) > 1]


def populate_staging(archive: ZipFile, members: dict, staging: Path, report: dict) -> None:
    """Copy original image bytes, validate labels, and collect measured statistics."""
    file_hashes = {kind: defaultdict(list) for kind in ("images", "labels")}
    pixel_hashes = defaultdict(list)
    image_names = defaultdict(list)
    converted_labels = {}
    for source_split, split in SPLIT_MAPPING.items():
        images, labels = {}, {}
        for name, info in members.items():
            path = PurePosixPath(name)
            if path.parts[0] != source_split:
                continue
            if len(path.parts) < 3 or path.parts[1] not in ("images", "labels"):
                raise ValueError(f"Unexpected dataset entry: {name}")
            relative = Path(*path.parts[2:])
            if path.parts[1] == "images" and relative.suffix.lower() in IMAGE_SUFFIXES:
                images[relative] = info
            elif path.parts[1] == "labels" and relative.suffix == ".txt":
                labels[relative] = info
            else:
                raise ValueError(f"Unsupported dataset file: {name}")
        counts = dict.fromkeys(CLASS_NAMES, 0)
        split_report = {"images": len(images), "labels": len(labels), "standard_annotations": 0,
                        "polygon_annotations_converted": 0, "total_objects": 0,
                        "objects_per_class": counts, "image_dimensions": {}}
        report["splits"][split] = split_report
        expected_labels = Counter(relative.with_suffix(".txt") for relative in images)
        for relative, count in expected_labels.items():
            if count > 1:
                report["errors"].append(f"Multiple images share label {source_split}/{relative}")
        for relative in sorted(set(expected_labels) - labels.keys()):
            report["missing_image_label_pairs"].append(f"Missing label: {source_split}/labels/{relative}")
        for relative in sorted(labels.keys() - set(expected_labels)):
            report["missing_image_label_pairs"].append(f"Missing image: {source_split}/labels/{relative}")
        dimensions = Counter()
        for relative, info in sorted(images.items()):
            data = archive.read(info)
            file_hashes["images"][hashlib.sha256(data).hexdigest()].append(info.filename)
            image_names[relative.name.casefold()].append(info.filename)
            try:
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation must be normalized before annotation")
                    dimensions[f"{image.width}x{image.height}"] += 1
                    # Also find identical decoded pixels, even if JPEG metadata differs.
                    fingerprint = hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()
                    pixel_hashes[fingerprint].append(info.filename)
            except (OSError, ValueError) as error:
                report["unreadable_images"].append({"file": info.filename, "error": str(error)})
            destination = staging / "images" / split / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        split_report["image_dimensions"] = dict(dimensions)
        for relative, info in sorted(labels.items()):
            data = archive.read(info)
            file_hashes["labels"][hashlib.sha256(data).hexdigest()].append(info.filename)
            lines = data.decode("utf-8-sig").splitlines()
            output_lines = []
            seen_boxes = {}
            if not any(line.strip() for line in lines):
                report["empty_label_files"].append(info.filename)
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    class_id = int(line.split()[0])
                    if class_id not in range(len(CLASS_NAMES)):
                        report["out_of_range_class_ids"].append({"file": info.filename, "line": line_number, "class_id": class_id})
                        continue
                    converted, is_polygon = convert_annotation(line)
                    output_lines.append(converted)
                    box = parse_detection(converted)
                    if box in seen_boxes:
                        report["duplicate_annotation_rows"].append({"file": info.filename,
                            "line": line_number, "first_line": seen_boxes[box], "class_id": class_id})
                    else:
                        seen_boxes[box] = line_number
                    counts[CLASS_NAMES[class_id]] += 1
                    split_report["total_objects"] += 1
                    key = "polygon_annotations_converted" if is_polygon else "standard_annotations"
                    split_report[key] += 1
                except ValueError as error:
                    report["invalid_annotations"].append({"file": info.filename, "line": line_number, "error": str(error)})
            converted_labels[info.filename] = [parse_detection(line) for line in output_lines]
            destination = staging / "labels" / split / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    report["exact_duplicated_files"] = {kind: duplicate_groups(hashes) for kind, hashes in file_hashes.items()}
    report["identical_pixel_images"] = duplicate_groups(pixel_hashes)
    report["repeated_image_basenames"] = duplicate_groups(image_names)
    for group in report["identical_pixel_images"]:
        if len({name.split("/")[0] for name in group}) > 1:
            report["errors"].append(f"Cross-split image leakage: {group}")
        labels_for_group = [converted_labels.get(str(PurePosixPath(name.replace('/images/', '/labels/')).with_suffix('.txt'))) for name in group]
        if any(boxes != labels_for_group[0] for boxes in labels_for_group[1:]):
            report["duplicate_images_with_different_labels"].append(group)
    report["total_images"] = sum(split["images"] for split in report["splits"].values())
    report["total_labels"] = sum(split["labels"] for split in report["splits"].values())
    for key in ("total_objects", "standard_annotations", "polygon_annotations_converted"):
        report[key] = sum(split[key] for split in report["splits"].values())
    report["objects_per_class"] = {name: sum(split["objects_per_class"][name] for split in report["splits"].values()) for name in CLASS_NAMES}
    report["invalid_coordinates"] = [issue for issue in report["invalid_annotations"]
                                     if any(term in issue["error"] for term in ("coordinates", "width", "height", "boundaries"))]


def import_dataset(zip_path: Path, output: Path, audit_path: Path) -> dict:
    """Stage and validate the full import before publishing into an empty target.

    Audit errors prevent publishing. Within-split duplicates are retained and
    reported; cross-split duplicate images prevent training on leaked data.
    """
    if output.expanduser().is_symlink():
        raise ValueError("Import destination must not be a symlink.")
    zip_path, output = zip_path.expanduser().resolve(), output.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    if audit_path == zip_path:
        raise ValueError("The audit path must not overwrite the source ZIP.")
    if output.exists() and any(p.is_file() and p.name != ".gitkeep" for p in output.rglob("*")):
        raise ValueError(f"Import destination is not empty: {output}. Existing datasets are never overwritten.")
    if output.exists() and any(p.is_symlink() for p in [output, *output.rglob("*")]):
        raise ValueError("Import destination must not contain symlinks.")
    report = {"status": "inspecting", "source_zip": str(zip_path),
              "source_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
              "dataset_source": SOURCE, "class_mapping": dict(enumerate(CLASS_NAMES)),
              "split_mapping": SPLIT_MAPPING, "processed_directory": str(output), "splits": {},
              "duplicate_filenames": [], "missing_image_label_pairs": [], "invalid_annotations": [],
              "invalid_coordinates": [], "out_of_range_class_ids": [], "empty_label_files": [],
              "unreadable_images": [], "duplicate_images_with_different_labels": [],
              "duplicate_annotation_rows": [], "errors": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(prefix=".fruit-import-", dir=output.parent) as temporary:
            members = inspect_archive(archive, report)
            config = verify_source(archive, members)
            unknown = [name for name in members if name not in METADATA_FILES and name.split('/')[0] not in SPLIT_MAPPING]
            if unknown:
                raise ValueError(f"Unexpected archive files: {unknown}")
            report["source_data_yaml"] = config
            staging = Path(temporary) / "processed"
            populate_staging(archive, members, staging, report)
            if any(report[key] for key in ("errors", "missing_image_label_pairs", "invalid_annotations", "out_of_range_class_ids", "unreadable_images")):
                raise ValueError("Critical dataset inconsistencies found; inspect the audit report. No dataset was published.")
            resolved = {"path": str(staging), "names": dict(enumerate(CLASS_NAMES)), "roboflow": config["roboflow"],
                        **{split: f"images/{split}" for split in SPLIT_MAPPING.values()}}
            config_path = Path(temporary) / "dataset.yaml"
            config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
            _, report["validation"] = validate_dataset(config_path)
            metadata_dir = staging / "source_metadata"
            metadata_dir.mkdir()
            for name in sorted(METADATA_FILES & members.keys()):
                (metadata_dir / name).write_bytes(archive.read(members[name]))
            if hashlib.sha256(zip_path.read_bytes()).hexdigest() != report["source_zip_sha256"]:
                raise ValueError("Source ZIP changed during import; refusing to publish.")
            # The destination precheck protects existing data; only validated files reach it.
            shutil.copytree(staging, output, dirs_exist_ok=True)
            report["status"] = "valid"
            save_json(output / "import_manifest.json", report)
    except (OSError, ValueError, BadZipFile, yaml.YAMLError) as error:
        report["status"] = "failed"
        report["errors"].append(str(error))
        save_json(audit_path, report)
        raise ValueError(f"{error}\nAudit: {audit_path}") from error
    save_json(audit_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, dest="zip_path")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/dataset_audit.json")
    args = parser.parse_args()
    try:
        report = import_dataset(args.zip_path, args.output, args.audit)
        plot_counts(report["objects_per_class"], ROOT / "outputs/figures/dataset_class_distribution.png",
                    title="Fruit v1 — object annotations across all splits", ylabel="Annotated objects")
        print(f"Imported {report['total_images']} images, {report['total_objects']} objects; "
              f"converted {report['polygon_annotations_converted']} polygons.")
        for split, values in report["splits"].items():
            print(f"{split}: {values['images']} images, {values['total_objects']} objects")
        print(f"Audit: {args.audit}")
        if report["identical_pixel_images"]:
            print(f"Warning: {len(report['identical_pixel_images'])} within-split duplicate image groups retained. See audit.")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Import error: {error}\n")


if __name__ == "__main__":
    main()
