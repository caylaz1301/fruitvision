"""Temporary fixtures test plumbing only; they are not a fruit training dataset."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

from src.common import CLASS_NAMES, ROOT, load_detector
from src.counting import format_summary, summarize_detections
from src.evaluate import collect_metrics
from src.predict import extract_detections
from src.preprocessing import read_labels, remove_exact_duplicate_rows, split_dataset, validate_dataset
from src.visualization import annotate_image, place_label, plot_counts


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_pair(self, images: Path, labels: Path, name: str, color: int = 1):
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (color, 0, 0)).save(images / f"{name}.png")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    def processed_config(self) -> Path:
        for color, split in enumerate(("train", "val", "test"), 1):
            self.write_pair(self.root / "processed/images" / split,
                            self.root / "processed/labels" / split, split, color)
        config = {"path": "processed", "names": dict(enumerate(CLASS_NAMES)),
                  **{split: f"images/{split}" for split in ("train", "val", "test")}}
        path = self.root / "dataset.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_counts_and_empty_predictions(self):
        summary = summarize_detections([
            {"class_name": "apple", "confidence": 0.8},
            {"class_name": "apple", "confidence": 0.6},
            {"class_name": "banana", "confidence": 1.0},
        ], CLASS_NAMES)
        self.assertEqual(summary["counts"], {name: (2 if name == "apple" else 1 if name == "banana" else 0) for name in CLASS_NAMES})
        self.assertEqual(summary["total_objects"], 3)
        self.assertAlmostEqual(summary["average_confidence"], 0.8)
        empty = summarize_detections([], CLASS_NAMES)
        self.assertEqual(empty["total_objects"], 0)
        self.assertIsNone(empty["average_confidence"])
        self.assertIn("N/A", format_summary(empty))

    def test_invalid_detection_scores_and_classes(self):
        for item in ({"class_name": "pear", "confidence": 0.8},
                     {"class_name": "apple", "confidence": float("nan")},
                     {"class_name": "apple", "confidence": 2}):
            with self.assertRaises(ValueError):
                summarize_detections([item], CLASS_NAMES)

    def test_annotation_validation(self):
        label = self.root / "label.txt"
        label.write_text("3 0.5 0.5 1 1\n")
        self.assertEqual(read_labels(label)[0], (3, 0.5, 0.5, 1, 1))
        for bad in ("0 0.5 0.5", "10 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2",
                    "0 0.5 0.5 0 0.2", "0 0.9 0.5 0.4 0.4", "0.0 0.5 0.5 0.2 0.2"):
            label.write_text(bad)
            with self.subTest(label=bad), self.assertRaises(ValueError):
                read_labels(label)
        label.write_text("")
        self.assertEqual(read_labels(label), [])

    def test_missing_label_is_not_silent_background(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing label"):
            read_labels(self.root / "missing.txt")

    def test_remove_only_exact_duplicate_rows(self):
        label = self.root / "duplicates.txt"
        original = "0 0.5 0.5 1 1\n0 0.5 0.5 1 1\n0 0.5 0.5 0.9 0.9\n"
        label.write_text(original)
        record = remove_exact_duplicate_rows(label)
        self.assertEqual(record["before"], original)
        self.assertEqual(record["removed_line_numbers"], [2])
        self.assertEqual(len(read_labels(label)), 2)
        self.assertEqual(remove_exact_duplicate_rows(label)["action"], "unchanged")

    def test_config_paths_and_leakage(self):
        path = self.processed_config()
        config, report = validate_dataset(path)
        self.assertEqual(config["path"], str((self.root / "processed").resolve()))
        self.assertEqual(report["test"]["objects_per_class"]["apple"], 1)
        shutil.copy2(self.root / "processed/images/train/train.png", self.root / "processed/images/test/test.png")
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            validate_dataset(path)

    def test_processed_audit_tracks_annotation_changes(self):
        from src.audit_dataset import audit_processed

        config = self.processed_config()
        first = audit_processed(config)
        self.assertEqual((first["total_images"], first["total_labels"], first["total_objects"]), (3, 3, 3))
        self.assertEqual(first["dataset_sha256"], audit_processed(config)["dataset_sha256"])
        label = self.root / "processed/labels/train/train.txt"
        label.write_text(label.read_text() + "1 0.5 0.5 0.2 0.2\n")
        changed = audit_processed(config)
        self.assertEqual(changed["total_objects"], 4)
        self.assertNotEqual(first["dataset_sha256"], changed["dataset_sha256"])

    def test_seeded_split_preserves_pairs_and_refuses_overwrite(self):
        raw = self.root / "raw"
        for index in range(20):
            self.write_pair(raw / "images", raw / "labels", str(index), index)
        first = split_dataset(raw, self.root / "first", 0.7, 0.15, 42)
        second = split_dataset(raw, self.root / "second", 0.7, 0.15, 42)
        self.assertEqual(first, second)
        self.assertEqual([len(items) for items in first["splits"].values()], [14, 3, 3])
        all_names = [item["image"] for items in first["splits"].values() for item in items]
        self.assertEqual(len(set(all_names)), 20)
        for split, items in first["splits"].items():
            for item in items:
                label = self.root / "first/labels" / split / Path(item["image"]).with_suffix(".txt")
                self.assertEqual(read_labels(label), [(0, 0.5, 0.5, 0.4, 0.4)])
        with self.assertRaisesRegex(ValueError, "not empty"):
            split_dataset(raw, self.root / "first", 0.7, 0.15, 42)

    def test_split_rejects_duplicate_images(self):
        raw = self.root / "raw"
        for index in range(10):
            self.write_pair(raw / "images", raw / "labels", str(index))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_dataset(raw, self.root / "split", 0.7, 0.15, 42)

    def test_metric_averaging_and_absent_classes(self):
        # Arithmetic fixture, never saved as a model evaluation result.
        box = SimpleNamespace(ap_class_index=[0, 2], p=[0.8, 0.5], r=[0.4, 1.0],
                              f1=[8 / 15, 2 / 3], ap50=[0.6, 0.7], ap=[0.3, 0.4],
                              mp=0.65, mr=0.7, map50=0.65, map=0.35)
        report = collect_metrics(SimpleNamespace(box=box))
        self.assertAlmostEqual(report["overall"]["f1"], 0.6)
        self.assertIsNone(report["per_class"][1]["map50"])
        self.assertEqual(report["per_class"][2]["recall"], 1)
        json.dumps(report, allow_nan=False)

    def test_ultralytics_result_extraction(self):
        # Exercise the real Results/tensor API without running or scoring a model.
        import torch
        from ultralytics.engine.results import Results

        pixels = torch.zeros((30, 40, 3), dtype=torch.uint8).numpy()
        names = dict(enumerate(CLASS_NAMES))
        for rows in ([], [[2, 3, 20, 25, 0.8, 0], [21, 2, 35, 24, 0.6, 8]]):
            boxes = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
            result = Results(orig_img=pixels, path="unit-test-only.png", names=names, boxes=boxes)
            detections = extract_detections(result)
            self.assertEqual(len(detections), len(rows))
            if rows:
                self.assertEqual(detections[1]["class_name"], "strawberry")
                self.assertEqual(detections[0]["bbox_xyxy"], [2, 3, 20, 25])
                self.assertAlmostEqual(detections[0]["confidence"], 0.8)
            json.dumps(detections, allow_nan=False)

    def test_visualization_preserves_source_and_handles_empty(self):
        original = Image.new("RGB", (100, 100), "white")
        detections = [{"class_id": 0, "class_name": "apple", "confidence": 0.9, "bbox_xyxy": [20, 20, 80, 80]}]
        annotated = annotate_image(original, detections, summarize_detections(detections, CLASS_NAMES))
        self.assertEqual(original.getpixel((20, 20)), (255, 255, 255))
        self.assertNotEqual(annotated.getpixel((20, 30)), (255, 255, 255))
        self.assertGreater(annotated.height, original.height)
        empty = summarize_detections([], CLASS_NAMES)
        annotate_image(original, [], empty).save(self.root / "empty.png")
        plot_counts(empty["counts"], self.root / "counts.png")
        self.assertTrue((self.root / "counts.png").is_file())

    def test_ultralytics_metrics_with_no_predictions(self):
        import torch
        from ultralytics.utils.metrics import DetMetrics

        metrics = DetMetrics(names=dict(enumerate(CLASS_NAMES)))
        empty = torch.empty(0).numpy()
        targets = torch.tensor([0, 3]).numpy()
        # Test the evaluator's zero-detection edge case; no result files are saved.
        metrics.update_stats({"tp": torch.zeros((0, 10), dtype=torch.bool).numpy(),
                              "conf": empty, "pred_cls": empty, "target_cls": targets,
                              "target_img": targets, "im_name": "unit-test-only"})
        metrics.process(plot=False)
        report = collect_metrics(metrics)
        self.assertEqual(report["per_class"][0]["ground_truth_objects"], 1)
        self.assertEqual(report["per_class"][1]["ground_truth_objects"], 0)
        self.assertEqual(set(report["overall"].values()), {0.0})
        self.assertEqual(report["per_class"][3]["f1"], 0.0)
        self.assertIsNone(report["per_class"][1]["f1"])

    def test_neighboring_labels_do_not_overlap(self):
        first = place_label(250, 135, 110, 15, 384, [])
        second = place_label(290, 135, 90, 15, 384, [first])
        self.assertTrue(second[3] <= first[1] or second[1] >= first[3])
        self.assertGreaterEqual(second[1], 0)
        self.assertLessEqual(second[3], 384)

    def test_missing_weights_are_clear(self):
        with self.assertRaisesRegex(FileNotFoundError, "Run python src/train.py"):
            load_detector(self.root / "missing.pt")

    def test_cli_help_and_missing_data(self):
        for script in ("train", "predict", "evaluate", "preprocessing", "import_dataset"):
            result = subprocess.run([sys.executable, str(ROOT / f"src/{script}.py"), "--help"],
                                    capture_output=True, text=True, cwd=self.root)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(ROOT / "src/train.py"), "--data", str(self.root / "absent.yaml")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Dataset configuration not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
