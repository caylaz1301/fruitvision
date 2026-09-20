"""Small temporary fixtures verify experiment plumbing, never model performance."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from PIL import Image

from experiments.brightness_test import adjust_brightness
from experiments.blur_test import gaussian_blur
from experiments.resolution_test import reduce_resolution
from experiments.experiment_runner import (
    CONDITIONS, METRICS, compare_results, degradation, prepare_condition, run_experiments, transform_image,
)
from src.common import CLASS_NAMES, ROOT
from src.preprocessing import inspect_pairs


class RobustnessTests(unittest.TestCase):
    def test_brightness_scaling_clipping_and_source_preservation(self):
        original = Image.new("RGB", (8, 6), (100, 200, 10))
        self.assertEqual(adjust_brightness(original, .5).getpixel((0, 0)), (50, 100, 5))
        self.assertEqual(adjust_brightness(original, 1.5).getpixel((0, 0)), (150, 255, 15))
        self.assertEqual(original.getpixel((0, 0)), (100, 200, 10))

    def test_blur_spreads_detail_without_moving_canvas(self):
        original = Image.new("RGB", (31, 31))
        original.putpixel((15, 15), (255, 255, 255))
        changed = gaussian_blur(original, 2)
        self.assertEqual(changed.size, original.size)
        self.assertLess(changed.getpixel((15, 15))[0], 255)
        self.assertGreater(changed.getpixel((16, 15))[0], 0)
        self.assertEqual(changed.getpixel((14, 15)), changed.getpixel((16, 15)))
        self.assertEqual(original.getpixel((15, 15)), (255, 255, 255))

    def test_resolution_removes_detail_restores_dimensions(self):
        original = Image.new("RGB", (32, 24))
        original.putdata([(255, 255, 255) if (i % 32 + i // 32) % 2 else (0, 0, 0) for i in range(32 * 24)])
        before = original.tobytes()
        changed = reduce_resolution(original, .25)
        self.assertEqual(changed.size, original.size)
        self.assertNotEqual(changed.tobytes(), before)
        self.assertLess(max(changed.getdata())[0] - min(changed.getdata())[0], 255)
        self.assertEqual(original.tobytes(), before)

    def test_invalid_transformation_parameters(self):
        image = Image.new("RGB", (8, 8))
        for function in (adjust_brightness, gaussian_blur, reduce_resolution):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(function=function.__name__, value=value), self.assertRaises(ValueError):
                    function(image, value)
        with self.assertRaises(ValueError):
            reduce_resolution(image, 1)

    def test_normal_is_an_independent_pixel_identical_copy(self):
        image = Image.new("RGB", (8, 6), (50, 90, 200))
        result = transform_image(image, "normal")
        self.assertIsNot(result, image)
        self.assertEqual(result.tobytes(), image.tobytes())
        with self.assertRaises(KeyError):
            transform_image(image, "unknown")

    def make_source(self, root):
        for i, split in enumerate(("train", "val", "test")):
            images, labels = root / "images" / split, root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (32, 24), (40 + i, 100, 180)).save(images / "sample.png")
            (labels / "sample.txt").write_text("".join(f"{c} 0.5 0.5 0.5 0.5\n" for c in range(10)))
        config = root.parent / "config.yaml"
        config.write_text(yaml.safe_dump({"path": str(root), "names": dict(enumerate(CLASS_NAMES)),
                                         **{s: f"images/{s}" for s in ("train", "val", "test")}}))
        return config

    def test_every_condition_preserves_label_bytes_and_normalized_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_source(root / "source")
            records = inspect_pairs(root / "source/images/val", root / "source/labels/val")
            originals = {r[k]: r[k].read_bytes() for r in records for k in ("image", "label")}
            for condition in CONDITIONS:
                directory = root / condition
                path = prepare_condition(records, directory, condition)
                output = inspect_pairs(directory / "images/val", directory / "labels/val")
                self.assertEqual(output[0]["boxes"], records[0]["boxes"])
                self.assertEqual((output[0]["width"], output[0]["height"]), (32, 24))
                self.assertEqual(output[0]["label"].read_bytes(), originals[records[0]["label"]])
                self.assertNotIn("test", yaml.safe_load(path.read_text()))
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    prepare_condition(records, directory, condition)
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)

    def test_relative_degradation_is_signed_and_handles_zero_reference(self):
        self.assertAlmostEqual(degradation(.8, .6)["relative_percent"], 25)
        self.assertAlmostEqual(degradation(.8, .6)["percentage_points"], 20)
        self.assertAlmostEqual(degradation(.5, .6)["relative_percent"], -20)
        self.assertEqual(degradation(.8, .8)["relative_percent"], 0)
        self.assertIsNone(degradation(0, .5)["relative_percent"])
        self.assertIsNone(degradation(None, .5)["percentage_points"])

    def fixture_report(self):
        return {"overall": dict.fromkeys(METRICS, .5),
                "per_class": [{"class_id": i, "class_name": name, "ground_truth_objects": 1,
                               **dict.fromkeys(METRICS, .5)} for i, name in enumerate(CLASS_NAMES)],
                "metric_notes": "Temporary arithmetic fixture, not measured detector performance."}

    def test_comparison_rejects_changed_ground_truth_counts(self):
        normal, altered = self.fixture_report(), self.fixture_report()
        altered["per_class"][0]["ground_truth_objects"] = 2
        with self.assertRaisesRegex(ValueError, "counts changed"):
            compare_results({"normal": normal, "dark": altered})

    def test_runner_uses_same_checkpoint_validation_only_and_never_trains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_source(root / "source")
            weights = root / "fixture.pt"
            weights.write_bytes(b"unit-test-only: model loading is mocked")
            model = Mock()
            with patch("experiments.experiment_runner.load_detector", return_value=model) as loader, \
                 patch("experiments.experiment_runner.collect_metrics", side_effect=lambda _: self.fixture_report()), \
                 patch("experiments.experiment_runner.plot_comparison"):
                output = root / "output"
                result = run_experiments(config, weights, output, "cpu")
                self.assertEqual(loader.call_count, 5)
                self.assertTrue(all(call.args == (weights,) for call in loader.call_args_list))
                model.train.assert_not_called()
                self.assertEqual(model.val.call_count, 5)
                for call in model.val.call_args_list:
                    self.assertEqual(call.kwargs["split"], "val")
                    self.assertFalse(call.kwargs["augment"])
                    self.assertEqual(call.kwargs["imgsz"], 640)
                self.assertEqual(len(result["per_class"]), 50)
                metadata = json.loads((output / "run_metadata.json").read_text())
                self.assertTrue(metadata["source_dataset_unchanged"])
                self.assertTrue(metadata["checkpoint_unchanged"])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run_experiments(config, weights, output, "cpu")

    def test_cli_help_is_available_without_running_evaluation(self):
        result = subprocess.run([sys.executable, str(ROOT / "experiments/experiment_runner.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
