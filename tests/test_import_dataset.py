"""Small in-memory archive fixtures exercise import rules, never model quality."""

import hashlib
import json
import tempfile
import unittest
import warnings
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import yaml
from PIL import Image

from src.common import CLASS_NAMES, DATASET
from src.counting import summarize_detections
from src.import_dataset import SOURCE, convert_annotation, import_dataset
from src.preprocessing import read_labels


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "processed"
        self.audit = self.root / "audit.json"

    def tearDown(self):
        self.temporary.cleanup()

    def make_archive(self, labels=None, names=None, duplicate_metadata=False, extra=None,
                     leak=False, missing_label=False):
        archive_path = self.root / "fruit.zip"
        config = {"nc": 10, "names": list(CLASS_NAMES) if names is None else names,
                  "roboflow": SOURCE, "train": "../train/images", "val": "../valid/images", "test": "../test/images"}
        with warnings.catch_warnings(), ZipFile(archive_path, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("data.yaml", yaml.safe_dump(config))
            if duplicate_metadata:
                archive.writestr("data.yaml", yaml.safe_dump(config))
            for color, split in enumerate(("train", "valid", "test"), 1):
                buffer = BytesIO()
                Image.new("RGB", (20, 20), (1 if leak else color, 0, 0)).save(buffer, format="PNG")
                archive.writestr(f"{split}/images/fruit.png", buffer.getvalue())
                if not (missing_label and split == "valid"):
                    text = labels if labels is not None else "\n".join(f"{i} 0.5 0.5 0.25 0.25" for i in range(10))
                    archive.writestr(f"{split}/labels/fruit.txt", text)
            if extra:
                for name, contents in extra.items():
                    archive.writestr(name, contents)
        return archive_path

    def test_ten_class_configuration_exact_order(self):
        expected = ["apple", "banana", "grape", "kiwi", "lemon", "orange", "peach", "pineapple", "strawberry", "watermelon"]
        self.assertEqual(list(CLASS_NAMES), expected)
        config = yaml.safe_load(DATASET.read_text())
        self.assertEqual(config["names"], dict(enumerate(expected)))
        self.assertEqual(config["val"], "images/val")

    def test_known_polygon_and_id_preservation(self):
        converted, polygon = convert_annotation("9 0.25 0.125 0.75 0.125 0.75 0.875 0.25 0.875")
        self.assertTrue(polygon)
        self.assertEqual(converted, "9 0.5 0.5 0.5 0.75")
        for class_id in range(10):
            line = f"{class_id} 0.5 0.5 0.25 0.25"
            self.assertEqual(convert_annotation(line), (line, False))

    def test_polygon_rejects_invalid_coordinates_and_shape(self):
        for line in ("9 0 0 1 1 0", "9 0 0 1 1", "10 0 0 1 0 1 1",
                     "0 nan 0 1 0 1 1", "0 -0.1 0 1 0 1 1", "0 0.5 0 0.5 0.5 0.5 1",
                     "0 0 0 1.1 0 1 1", "0.0 0 0 1 0 1 1"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                convert_annotation(line)

    def test_import_preserves_splits_images_and_metadata(self):
        archive = self.make_archive(duplicate_metadata=True)
        original_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        report = import_dataset(archive, self.output, self.audit)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["total_images"], 3)
        self.assertEqual(report["total_objects"], 30)
        self.assertEqual(report["objects_per_class"], dict.fromkeys(CLASS_NAMES, 3))
        self.assertEqual(report["standard_annotations"], 30)
        self.assertEqual(report["polygon_annotations_converted"], 0)
        self.assertEqual(len(report["duplicate_filenames"]), 1)
        self.assertTrue(report["duplicate_filenames"][0]["identical_metadata"])
        self.assertFalse((self.output / "images/valid").exists())
        with ZipFile(archive) as source:
            self.assertEqual((self.output / "images/val/fruit.png").read_bytes(), source.read("valid/images/fruit.png"))
            self.assertEqual((self.output / "source_metadata/data.yaml").read_bytes(), source.read("data.yaml"))
        self.assertEqual([box[0] for box in read_labels(self.output / "labels/val/fruit.txt")], list(range(10)))
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), original_hash)
        self.assertEqual(json.loads(self.audit.read_text())["total_labels"], 3)
        with self.assertRaisesRegex(ValueError, "not empty"):
            import_dataset(archive, self.output, self.audit)

    def test_mixed_annotations_preserve_every_object(self):
        archive = self.make_archive(labels="5 0.5 0.5 0.5 0.5\n8 0.25 0.25 0.75 0.25 0.75 0.75")
        report = import_dataset(archive, self.output, self.audit)
        self.assertEqual(report["standard_annotations"], 3)
        self.assertEqual(report["polygon_annotations_converted"], 3)
        self.assertEqual(report["total_objects"], 6)
        self.assertEqual([box[0] for box in read_labels(self.output / "labels/test/fruit.txt")], [5, 8])

    def test_wrong_mapping_blocks_import(self):
        names = list(CLASS_NAMES)
        names[2], names[5] = names[5], names[2]
        archive = self.make_archive(names=names)
        with self.assertRaisesRegex(ValueError, "class mapping"):
            import_dataset(archive, self.output, self.audit)
        self.assertFalse(self.output.exists())

    def test_invalid_annotations_produce_failed_audit_without_publishing(self):
        archive = self.make_archive(labels="11 0.5 0.5 0.2 0.2\n0 -0.1 0 1 0 1 1")
        with self.assertRaisesRegex(ValueError, "Critical"):
            import_dataset(archive, self.output, self.audit)
        report = json.loads(self.audit.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["out_of_range_class_ids"]), 3)
        self.assertEqual(len(report["invalid_coordinates"]), 3)
        self.assertFalse(self.output.exists())

    def test_unsafe_paths_and_duplicate_data_filenames_block_import(self):
        for entry in ("../escaped.txt", "/absolute.txt", "train/images/fruit.png"):
            archive = self.make_archive(extra={entry: "unsafe"})
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                import_dataset(archive, self.output, self.audit)
            self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_cross_split_duplicates_block_import(self):
        archive = self.make_archive(leak=True)
        with self.assertRaisesRegex(ValueError, "Critical"):
            import_dataset(archive, self.output, self.audit)
        self.assertIn("Cross-split", json.loads(self.audit.read_text())["errors"][0])
        self.assertFalse(self.output.exists())

    def test_missing_label_blocks_import(self):
        archive = self.make_archive(extra={"valid/labels/unpaired.txt": "0 0.5 0.5 0.2 0.2"}, missing_label=True)
        with self.assertRaisesRegex(ValueError, "Critical"):
            import_dataset(archive, self.output, self.audit)
        report = json.loads(self.audit.read_text())
        self.assertEqual(len(report["missing_image_label_pairs"]), 2)
        self.assertFalse(self.output.exists())

    def test_counting_all_ten_classes(self):
        detections = [{"class_name": name, "confidence": 0.5} for name in CLASS_NAMES]
        summary = summarize_detections(detections, CLASS_NAMES)
        self.assertEqual(summary["counts"], dict.fromkeys(CLASS_NAMES, 1))
        self.assertEqual(summary["total_objects"], 10)
        self.assertEqual(summary["average_confidence"], 0.5)

    def test_source_zip_cannot_be_used_as_audit_output(self):
        archive = self.make_archive()
        original = archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "source ZIP"):
            import_dataset(archive, self.output, archive)
        self.assertEqual(archive.read_bytes(), original)

    def test_repeated_annotation_rows_are_reported_and_preserved(self):
        archive = self.make_archive(labels="0 0.5 0.5 1 1\n0 0.5 0.5 1 1")
        report = import_dataset(archive, self.output, self.audit)
        self.assertEqual(report["total_objects"], 6)
        self.assertEqual(len(report["duplicate_annotation_rows"]), 3)
        self.assertEqual(len(read_labels(self.output / "labels/train/fruit.txt")), 2)

    def test_symlink_destination_is_refused(self):
        archive = self.make_archive()
        real_directory = self.root / "original"
        real_directory.mkdir()
        self.output.symlink_to(real_directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            import_dataset(archive, self.output, self.audit)
        self.assertEqual(list(real_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
