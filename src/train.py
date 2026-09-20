"""Fine-tune a pretrained YOLO detector on ten fruit classes."""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import yaml

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATASET, ROOT, package_versions, positive_int, probability, save_json, select_device
from src.preprocessing import save_dataset_config, validate_dataset


def fine_tuning_settings(learning_rate: float | None, epochs: int) -> dict:
    """Keep baseline defaults, or use gentle AdamW updates for fine-tuning.

    AdamW must not inherit SGD's 0.1 bias warm-up rate. Mosaic is disabled from
    the first epoch so real close-up crops retain their intended image context.
    """
    if learning_rate is None:
        return {}
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1 or epochs <= 0:
        raise ValueError("Fine-tuning requires a positive learning rate <= 1 and epoch count.")
    return {"optimizer": "AdamW", "lr0": learning_rate, "warmup_bias_lr": 0.0,
            "close_mosaic": epochs}


def resume_training(checkpoint: Path, requested_device: str) -> None:
    """Continue an interrupted run using its saved optimizer and configuration."""
    from ultralytics import YOLO
    from src.audit_dataset import audit_processed

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    run_dir = checkpoint.parents[1]
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") == "completed":
        raise ValueError("This run is already complete; no further training is needed.")
    # A resume must use the same images/labels, not silently adopt dataset edits.
    saved_audit = json.loads((run_dir / "dataset_audit_baseline.json").read_text())
    current_audit = audit_processed(Path(metadata["arguments"]["data"]))
    if current_audit["dataset_sha256"] != saved_audit["dataset_sha256"]:
        raise ValueError("Dataset changed since training started; refusing to resume this baseline.")
    with (run_dir / "results.csv").open() as stream:
        previous = list(csv.DictReader(stream))
    event = {"checkpoint": str(checkpoint), "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
             "epochs_completed_before_resume": len(previous),
             "completed_epoch_seconds_before_resume": float(previous[-1]["time"]),
             "reason": "Continuation after the previous training process stopped; saved configuration and optimizer restored."}
    metadata.setdefault("resumptions", []).append(event)
    save_json(metadata_path, metadata)
    start = perf_counter()
    model = YOLO(str(checkpoint))
    model.train(resume=True, device=select_device(requested_device))
    with (run_dir / "results.csv").open() as stream:
        completed = len(list(csv.DictReader(stream)))
    event["duration_seconds"] = perf_counter() - start
    metadata.update(status="completed", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                    epochs_completed=completed, early_stopped=completed < metadata["arguments"]["epochs"])
    save_json(metadata_path, metadata)
    print(f"Resumed run completed. Best model: {run_dir / 'weights/best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained detection checkpoint name or local path.")
    # An epoch is one pass through the training images.
    parser.add_argument("--epochs", type=positive_int, default=50)
    # imgsz controls the model input size: larger images cost more memory.
    parser.add_argument("--img-size", type=positive_int, default=640)
    # A batch is the group of images used for one gradient update.
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index (e.g. 0).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=probability, help="Optional fine-tuning rate; explicitly uses AdamW instead of auto optimizer.")
    parser.add_argument("--scale", type=probability, default=0.5, help="Training-only scale variation; default preserves the baseline.")
    parser.add_argument("--rotation", type=float, default=0.0, help="Training-only random rotation in degrees, from 0 to 180.")
    parser.add_argument("--name", default="fruitvision", help="Unique folder name below outputs/training.")
    parser.add_argument("--resume", type=Path, help="Continue an interrupted run from last.pt using its saved settings.")
    args = parser.parse_args()
    try:
        if not 0 <= args.rotation <= 180:
            raise ValueError("--rotation must be between 0 and 180 degrees.")
        if args.learning_rate is not None and args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive.")
        if args.resume:
            resume_training(args.resume, args.device)
            return
        if Path(args.name).name != args.name or args.name in (".", ".."):
            raise ValueError("--name must be a single folder name.")
        run_dir = ROOT / "outputs/training" / args.name
        if run_dir.exists():
            raise ValueError(f"Run already exists: {run_dir}. Use a new --name to preserve it.")
        # Check data before model loading so missing data never triggers a download.
        config, report = validate_dataset(args.data, ("train", "val"))
        if any(count == 0 for count in report["train"]["objects_per_class"].values()):
            raise ValueError("Training needs annotations for all ten classes.")
        if not args.model.endswith(".pt"):
            raise ValueError("Use a pretrained detection .pt checkpoint, such as yolo11n.pt.")
        from src.audit_dataset import audit_processed

        # Persist the exact data fingerprint so an interrupted run can resume safely.
        dataset_audit = audit_processed(args.data)
        device = select_device(args.device)
        from ultralytics import YOLO

        # Pretrained weights contain visual features learned on another dataset.
        # Transfer learning fine-tunes these features using our fruit labels.
        model = YOLO(args.model)
        if model.task != "detect":
            raise ValueError("Use a pretrained detection .pt checkpoint, such as yolo11n.pt.")
        resolved_path = save_dataset_config(config, run_dir)
        save_json(run_dir / "dataset_audit_baseline.json", dataset_audit)
        metadata = {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": device, "packages": package_versions(), "dataset_summary": report,
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
            "reproducibility_note": "Seeds are fixed; MPS operations may remain nondeterministic.",
        }
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run_dir / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
        save_json(run_dir / "run_metadata.json", metadata)
        start = perf_counter()
        print(f"Training on {device}. Outputs: {run_dir}")
        # A small explicit learning rate adapts existing features gently. Auto
        # optimizer selection can override lr0, so specify AdamW for this option.
        optimizer_settings = fine_tuning_settings(args.learning_rate, args.epochs)
        model.train(
            data=str(resolved_path), epochs=args.epochs, imgsz=args.img_size,
            batch=args.batch_size, device=device, seed=args.seed, deterministic=True,
            project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            workers=0, save=True, plots=True,
            # Augmentation changes training examples and transforms their boxes together.
            # Validation/test images are not randomly augmented.
            fliplr=0.5, mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            scale=args.scale, degrees=args.rotation, **optimizer_settings,
        )
        best = run_dir / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError(f"Training ended without best.pt; inspect {run_dir}.")
        with (run_dir / "results.csv").open() as stream:
            completed_epochs = len(list(csv.DictReader(stream)))
        metadata.update(status="completed", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=perf_counter() - start, epochs_completed=completed_epochs,
                        early_stopped=completed_epochs < args.epochs)
        save_json(run_dir / "run_metadata.json", metadata)
        print(f"Best model: {best}")
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as error:
        parser.exit(2, f"Training error: {error}\n")


if __name__ == "__main__":
    main()
