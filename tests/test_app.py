"""UI adapters and page flows. Temporary fixtures are not detector benchmarks."""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.helpers import (decode_upload, detection_rows, prediction_signature, read_report,
                         run_prediction, validate_metrics)
from src.common import CLASS_NAMES, ROOT
from src.counting import summarize_detections
from src.size_estimation import build_reference


def fixture_image(format="PNG", orientation=None):
    image = Image.new("RGB", (200, 120), "white")
    stream = BytesIO()
    exif = image.getexif()
    if orientation:
        exif[274] = orientation
    image.save(stream, format=format, exif=exif)
    return stream.getvalue()


class AppHelpersTests(unittest.TestCase):
    def test_decoding_rgb_and_exif_orientation(self):
        image = decode_upload(fixture_image("JPEG", 6))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (120, 200))

    def test_invalid_and_unsupported_images(self):
        for content in (b"", b"not an image", fixture_image("GIF")):
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                decode_upload(content)
        with patch("app.helpers.MAX_UPLOAD_BYTES", 1), self.assertRaises(ValueError):
            decode_upload(fixture_image())
        with patch("app.helpers.MAX_IMAGE_PIXELS", 1), self.assertRaises(ValueError):
            decode_upload(fixture_image())

    def test_signature_changes_with_every_input(self):
        signature = prediction_signature(b"image", .25, None)
        self.assertEqual(signature, prediction_signature(b"image", .25, None))
        self.assertNotEqual(signature, prediction_signature(b"other", .25, None))
        self.assertNotEqual(signature, prediction_signature(b"image", .5, None))
        self.assertNotEqual(signature, prediction_signature(b"image", .25, build_reference(8.56, 250)))

    def test_table_sizes_are_optional_and_confidence_is_percent(self):
        detection = {"class_name": "apple", "confidence": .8, "bbox_xyxy": [1, 2, 10, 20]}
        row = detection_rows([detection])[0]
        self.assertEqual(row["Confidence (%)"], 80)
        self.assertEqual(row["X2"], 10)
        self.assertNotIn("Approx. width (cm)", row)
        detection["size_estimation"] = {"approximate_width_cm": 1.234}
        self.assertEqual(detection_rows([detection])[0]["Approx. width (cm)"], 1.23)

    def test_reports_missing_corrupt_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, backup = root / "primary.json", root / "backup.json"
            with self.assertRaises(FileNotFoundError):
                read_report([primary, backup], ("overall",))
            backup.write_text(json.dumps({"overall": {}}))
            self.assertEqual(read_report([primary, backup], ("overall",))[1], backup)
            primary.write_text("not json")
            with self.assertRaises(ValueError):
                read_report([primary, backup], ("overall",))

    def test_invalid_metrics_are_not_replaced_with_scores(self):
        for value in (None, float("nan"), float("inf"), -1, 2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_metrics([{"precision": value}])

    def test_inference_adapter_reuses_existing_pipeline(self):
        detections = [{"class_id": 0, "class_name": "apple", "confidence": .8, "bbox_xyxy": [10, 10, 110, 90]}]
        summary = {**summarize_detections(detections, CLASS_NAMES), "inference_time_ms": 10}
        model = Mock()
        with patch("app.helpers.predict_image", return_value=(detections, summary)) as predict:
            payload, annotated = run_prediction(model, decode_upload(fixture_image()), "fixture.png", .25,
                                                build_reference(10, 100, image_size=(200, 120)), "cpu")
        predict.assert_called_once()
        self.assertEqual(payload["summary"], summary)
        self.assertEqual(payload["detections"][0]["size_estimation"]["approximate_width_cm"], 10)
        self.assertEqual(annotated.size, (200, 120))
        self.assertNotIn("size_estimation", detections[0])
        model.train.assert_not_called()


class AppPageTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def app(self):
        return AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=20).run()

    def test_all_five_pages_render_and_empty_analytics_links_to_detect(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(next(button for button in app.button if button.label == "Run Detection").disabled)
        for page in ("Analytics", "Model Performance", "Robustness", "About"):
            app.radio(key="page").set_value(page).run()
            self.assertFalse(app.exception, page)
        app.radio(key="page").set_value("Analytics").run()
        next(button for button in app.button if button.label == "Go to Detect").click().run()
        self.assertEqual(app.radio(key="page").value, "Detect")
        self.assertFalse(app.exception)

    def test_invalid_reference_disables_inference(self):
        app = self.app()
        app.session_state["upload_data"] = ("fixture.png", fixture_image())
        app.run()
        app.toggle(key="use_reference").set_value(True).run()
        self.assertTrue(app.error)
        self.assertTrue(next(button for button in app.button if button.label == "Run Detection").disabled)

    def test_empty_detect_is_compact_and_settings_are_collapsed(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertFalse(any('class="stats"' in item.value for item in app.markdown))
        settings = next(item for item in app.expander if item.label == 'Detection settings')
        self.assertFalse(settings.proto.expanded)
        self.assertTrue(app.slider(key='confidence'))

    def test_clearing_image_discards_its_reference(self):
        app = self.app()
        app.session_state["upload_data"] = ("fixture.png", fixture_image())
        app.run()
        app.toggle(key="use_reference").set_value(True).run()
        app.number_input(key="reference_cm").set_value(8.56)
        app.number_input(key="reference_pixels").set_value(50).run()
        next(button for button in app.button if button.label == "Clear image").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.toggle(key="use_reference").value)
        self.assertIsNone(app.session_state["upload_data"])
        self.assertNotIn("reference_cm", app.session_state)

    def test_result_survives_navigation_and_clears_on_threshold_change(self):
        detections = [{"class_id": 0, "class_name": "apple", "confidence": .8, "bbox_xyxy": [10, 10, 110, 90]}]
        summary = {**summarize_detections(detections, CLASS_NAMES), "inference_time_ms": 10}
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "fixture.pt"
            weights.touch()
            with patch("src.common.BEST_MODEL", weights), patch("src.common.load_detector", return_value=Mock()) as loader, \
                 patch("app.helpers.predict_image", return_value=(detections, summary)):
                app = self.app()
                app.session_state["upload_data"] = ("fixture.png", fixture_image())
                app.run()
                app.slider(key="confidence").set_value(.5).run()
                next(button for button in app.button if button.label == "Run Detection").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["prediction"]["payload"]["summary"]["total_objects"], 1)
                strip = next(item.value for item in app.markdown if 'class="count-strip"' in item.value)
                self.assertIn('Apple', strip)
                self.assertNotIn('Banana', strip)
                self.assertFalse(next(item for item in app.expander if item.label == 'Detection details & JSON').proto.expanded)
                app.radio(key="page").set_value("Analytics").run()
                self.assertFalse(app.exception)
                app.radio(key="page").set_value("Detect").run()
                self.assertEqual(app.slider(key="confidence").value, .5)
                self.assertIsNotNone(app.session_state["prediction"])
                next(button for button in app.button if button.label == "Run Detection").click().run()
                self.assertEqual(loader.call_count, 1)  # The cached model survives reruns.
                app.slider(key="confidence").set_value(.6).run()
                self.assertNotIn("prediction", app.session_state)

    def test_model_failure_is_visible_without_exception_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "fixture.pt"
            weights.touch()
            with patch("src.common.BEST_MODEL", weights), patch("src.common.load_detector", side_effect=RuntimeError("Unreadable model")):
                app = self.app()
                app.session_state["upload_data"] = ("fixture.png", fixture_image())
                app.run()
                next(button for button in app.button if button.label == "Run Detection").click().run()
                self.assertFalse(app.exception)
                self.assertIn("Unreadable model", app.error[0].value)

    def test_no_detections_with_size_reference_has_no_invented_measurements(self):
        summary = {**summarize_detections([], CLASS_NAMES), "inference_time_ms": 10}
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "fixture.pt"
            weights.touch()
            with patch("src.common.BEST_MODEL", weights), patch("src.common.load_detector", return_value=Mock()), \
                 patch("app.helpers.predict_image", return_value=([], summary)):
                app = self.app()
                app.session_state["upload_data"] = ("fixture.png", fixture_image())
                app.run()
                app.toggle(key="use_reference").set_value(True).run()
                app.number_input(key="reference_cm").set_value(8.56)
                app.number_input(key="reference_pixels").set_value(50).run()
                next(button for button in app.button if button.label == "Run Detection").click().run()
                self.assertFalse(app.exception)
                result = app.session_state["prediction"]["payload"]
                self.assertEqual(result["detections"], [])
                self.assertTrue(result["size_estimation"]["approximate"])
                self.assertIsNone(result["summary"]["average_confidence"])


if __name__ == "__main__":
    unittest.main()
