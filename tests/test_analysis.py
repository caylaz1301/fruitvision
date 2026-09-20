"""Regression checks for diagnostic matching and evaluated class counts."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.error_analysis import box_iou, match_detections
from src.train import resume_training


def box(class_id=0, confidence=0.9):
    return {"class_id": class_id, "class_name": ("apple", "banana")[class_id],
            "bbox_xyxy": [0, 0, 10, 10], "confidence": confidence}


class AnalysisTests(unittest.TestCase):
    def test_iou_overlap_and_disjoint_boxes(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_duplicate_prediction_does_not_match_twice(self):
        result = match_detections([box()], [box(confidence=0.5), box(confidence=0.9)])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["matches"][0]["prediction"], 1)

    def test_wrong_class_is_both_false_positive_and_false_negative(self):
        result = match_detections([box(0)], [box(1)])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["class_confusions"][0]["true_class"], "apple")
        self.assertEqual(result["class_confusions"][0]["predicted_class"], "banana")

    def test_no_predictions_retains_all_misses(self):
        result = match_detections([box(0), box(1)], [])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 2))

    def test_completed_run_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights").mkdir()
            checkpoint = root / "weights/last.pt"
            checkpoint.touch()  # Must never reach model loading.
            (root / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
            with patch("ultralytics.YOLO") as loader:
                with self.assertRaisesRegex(ValueError, "already complete"):
                    resume_training(checkpoint, "cpu")
                loader.assert_not_called()

    def test_resume_refuses_changed_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights").mkdir()
            checkpoint = root / "weights/last.pt"
            checkpoint.touch()
            (root / "run_metadata.json").write_text(json.dumps({"status": "running", "arguments": {"data": "dataset.yaml"}}))
            (root / "dataset_audit_baseline.json").write_text(json.dumps({"dataset_sha256": "original"}))
            with patch("src.audit_dataset.audit_processed", return_value={"dataset_sha256": "changed"}), patch("ultralytics.YOLO") as loader:
                with self.assertRaisesRegex(ValueError, "Dataset changed"):
                    resume_training(checkpoint, "cpu")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
