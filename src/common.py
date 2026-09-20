"""Shared paths, CLI validation, and model loading."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "configs/dataset.yaml"
BEST_MODEL = ROOT / "outputs/training/fruitvision_baseline/weights/best.pt"
CLASS_NAMES = (
    "apple", "banana", "grape", "kiwi", "lemon", "orange", "peach",
    "pineapple", "strawberry", "watermelon",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Must be between 0 and 1.")
    return number


def select_device(requested: str = "auto") -> str:
    """Prefer CUDA, then Apple Silicon's MPS GPU, then CPU."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_detector(weights: Path):
    """Require local fruit-trained detection weights; never silently use COCO."""
    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {weights}\n"
            "Run python src/train.py after preparing the dataset, or pass --weights PATH."
        )
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "detect" or model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"Expected a detection model with this exact class mapping: {dict(enumerate(CLASS_NAMES))}")
    return model


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("ultralytics", "torch", "Pillow", "PyYAML", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
