"""Calibration arithmetic and prediction integration; fixtures are not real measurements."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.counting import summarize_detections
from src.common import CLASS_NAMES
from src.predict import main
from src.size_estimation import build_reference, estimate_fruit_widths


class SizeEstimationTests(unittest.TestCase):
    def test_known_reference_conversion_math(self):
        reference = build_reference(8.56, reference_width_pixels=250)
        result = estimate_fruit_widths([{"bbox_xyxy": [10, 20, 210, 100]}], reference)[0]
        self.assertAlmostEqual(reference["cm_per_pixel"], .03424)
        self.assertAlmostEqual(result["size_estimation"]["approximate_width_cm"], 6.848)
        self.assertTrue(result["size_estimation"]["approximate"])

    def test_reference_box_and_pixel_width_are_equivalent(self):
        pixels = build_reference(8.56, reference_width_pixels=250, image_size=(500, 300))
        box = build_reference(8.56, reference_bbox_xyxy=[20, 30, 270, 90], image_size=(500, 300))
        self.assertEqual(box["reference_width_pixels"], 250)
        self.assertEqual(box["cm_per_pixel"], pixels["cm_per_pixel"])

    def test_each_detection_gets_its_own_width_without_mutating_inputs(self):
        detections = [{"bbox_xyxy": [0, 0, 100, 50], "class_name": "apple", "confidence": .8},
                      {"bbox_xyxy": [20, 0, 220, 60], "class_name": "banana", "confidence": .9}]
        before = copy.deepcopy(detections)
        result = estimate_fruit_widths(detections, build_reference(10, 100))
        self.assertEqual([x["size_estimation"]["approximate_width_cm"] for x in result], [10, 20])
        self.assertEqual(detections, before)
        self.assertEqual(summarize_detections(result, CLASS_NAMES), summarize_detections(before, CLASS_NAMES))

    def test_no_reference_leaves_detection_schema_unchanged(self):
        self.assertIsNone(build_reference())
        detections = [{"bbox_xyxy": [0, 0, 100, 50], "class_name": "apple"}]
        self.assertEqual(estimate_fruit_widths(detections), detections)
        self.assertNotIn("size_estimation", estimate_fruit_widths(detections)[0])

    def test_incomplete_reference_is_an_error(self):
        for args in ({"known_reference_width_cm": 8.56}, {"reference_width_pixels": 250},
                     {"reference_bbox_xyxy": [0, 0, 250, 30]}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "Reference needs"):
                build_reference(**args)

    def test_ambiguous_reference_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            build_reference(8.56, 250, [0, 0, 250, 30])

    def test_invalid_reference_lengths(self):
        for value in (0, -1, float("nan"), float("inf"), True, "not a number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_reference(value, 250)
                with self.assertRaises(ValueError):
                    build_reference(8.56, value)

    def test_invalid_reference_boxes_and_image_boundaries(self):
        for box in ([0, 0, 0, 10], [20, 0, 10, 10], [0, 10, 10, 0],
                    [-1, 0, 10, 10], [0, 0, float("nan"), 10], [0, 0, 10],
                    [0, 0, 101, 30], [0, 0, 80, 51], [False, 0, 20, 20]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                build_reference(8.56, reference_bbox_xyxy=box, image_size=(100, 50))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_reference(8.56, 101, image_size=(100, 50))

    def test_invalid_detection_boxes_are_not_silently_measured(self):
        reference = build_reference(8.56, 50, image_size=(100, 50))
        for detection in ({}, {"bbox_xyxy": [0, 0, 0, 10]}, {"bbox_xyxy": [0, 0, 101, 10]}):
            with self.subTest(detection=detection), self.assertRaises(ValueError):
                estimate_fruit_widths([detection], reference)

    def test_nonfinite_derived_results_are_rejected(self):
        with self.assertRaises(ValueError):
            build_reference(1e308, 1e-308)
        with self.assertRaises(ValueError):
            estimate_fruit_widths([{"bbox_xyxy": [0, 0, 100, 10]}], build_reference(1e308, 1))
        with self.assertRaises(ValueError):
            estimate_fruit_widths([], {})

    def test_empty_detections_have_no_invented_measurements(self):
        self.assertEqual(estimate_fruit_widths([], build_reference(8.56, 250)), [])

    def run_cli(self, reference_args, orientation=None):
        """Run the real CLI/JSON path with mocked detector output in a temp folder."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            original = Image.new("RGB", (200, 120), "white")
            exif = original.getexif()
            if orientation is not None:
                exif[274] = orientation
            original.save(image, exif=exif)
            detections = [{"bbox_xyxy": [20, 10, 120, 90], "class_id": 0,
                           "class_name": "apple", "confidence": .8}]
            summary = {**summarize_detections(detections, CLASS_NAMES), "inference_time_ms": 1.0}
            argv = ["predict.py", "--image", str(image), "--output-dir", str(root / "out"),
                    "--device", "cpu", *reference_args]
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.argv", argv), patch("src.predict.load_detector") as loader, \
                 patch("src.predict.predict_image", return_value=(detections, summary)), \
                 redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    main()
                except SystemExit as error:
                    self.assertEqual(error.code, 2)
                    loader.assert_not_called()
                    self.assertFalse((root / "out").exists())
                    return None, stderr.getvalue()
            payload = json.loads((root / "out/fixture_detections.json").read_text())
            self.assertTrue((root / "out/fixture_annotated.jpg").is_file())
            self.assertEqual(payload["summary"], summary)
            return payload, stdout.getvalue()

    def test_prediction_json_without_reference_is_backward_compatible(self):
        result, console = self.run_cli([])
        self.assertEqual(set(result), {"image", "weights", "confidence_threshold", "img_size",
                                      "coordinate_system", "image_width", "image_height", "summary", "detections"})
        self.assertNotIn("size_estimation", result["detections"][0])
        self.assertNotIn("Approximate", console)

    def test_prediction_json_with_both_reference_input_modes(self):
        for args in (["--reference-width-pixels", "50"], ["--reference-bbox", "0", "0", "50", "30"]):
            with self.subTest(args=args):
                result, console = self.run_cli(["--reference-width-cm", "8.56", *args])
                self.assertTrue(result["size_estimation"]["approximate"])
                self.assertAlmostEqual(result["detections"][0]["size_estimation"]["approximate_width_cm"], 17.12)
                self.assertIn("not true fruit diameters", console)
                json.dumps(result, allow_nan=False)

    def test_partial_reference_rejected_before_inference(self):
        result, message = self.run_cli(["--reference-width-cm", "8.56"])
        self.assertIsNone(result)
        self.assertIn("Reference needs", message)

    def test_reference_bounds_use_exif_oriented_image(self):
        # Orientation 6 turns 200x120 into 120x200; width 150 must be rejected.
        result, message = self.run_cli(["--reference-width-cm", "8.56", "--reference-width-pixels", "150"], orientation=6)
        self.assertIsNone(result)
        self.assertIn("exceeds", message)


if __name__ == "__main__":
    unittest.main()
