# Real-world banana/lemon review — September 2026

## Outcome

**The banana error remains unresolved. The application retains the original baseline.**
The simpler UI is implemented; unsuccessful model experiments are recorded rather
than substituted for the baseline or described as improvements.

## Evidence from the three supplied photos

All checks used the same confidence threshold (0.25), image size (640), original
EXIF-oriented RGB image, and CPU inference. Neither filenames nor expected fruit
classes influence inference. User-provided images were not put into training.

| Photo | Baseline output | Final candidate output |
| --- | --- | --- |
| `(1).jpeg`: bananas | Lemon 42.59% | Lemon 30.99%; Banana 26.88% |
| `(2).jpeg`: grapes | Grape 70.82% | Grape 76.24% |
| `.jpeg`: grapes | Grape 77.58% | Grape 77.88% |

These are prediction confidences, not accuracy. The candidate's banana detection
coexists with an incorrect lemon prediction; it is **not a successful correction**.
The photos have no audited object-level boxes, so no AP or object-count accuracy is
calculated for them. A bunch can be represented by one box in this dataset; the UI
now calls these “Detected objects” and explains the counting limitation.

## Dataset investigation

The original training split contains 157 images with 200 banana annotations.
Of these filenames, 104 start with `banana4_` and 34 with `banana5_`. Visual review
of a systematic 24-image sample shows many similar metallic backgrounds and small
fruit groups. Names alone do not prove duplication; this visual similarity is a
possible source of limited domain coverage, not a demonstrated causal explanation.
The original median banana box width is 19.73% of the image; median area is 5.40%.
The supplied phone photo is a close-up with overlapping, plump bananas.

A separate copy at `data/closeups_v1/` adds 1,021 training crops to 1,550 originals
(2,571 training images). These are **derived views, not new independent photos**.
Each crop encloses the union of all existing annotation boxes with 25% padding on
each side, clipped to the image. Centers and dimensions are renormalized to that
crop. No labeled object is discarded, no box is invented, and IDs are unchanged.
529 potential crops were skipped for empty/tiny labels or >=95% image coverage;
each skip and crop is listed in `data/closeups_v1/closeup_manifest.json`.
Original validation (430) and test (237) images and labels are byte-identical copies.
The original processed dataset and source ZIP are untouched. Existing annotation
imperfections remain; crop augmentation cannot repair missing labels or add new
fruit varieties, lighting, and backgrounds.

## Training attempts

1. **Adaptation v1:** baseline initialization, requested 12 epochs, MPS, batch 8,
   image size 640, AdamW lr=0.0001, scale=0.8, rotation ±15°, seed 42.
   Seven epochs completed; MPS failed during epoch 8 with
   `Invalid buffer size: 17179869184.00 GiB`. The temporary best checkpoint still
   called the banana lemon (70.5%). Inspection also found that explicit AdamW had
   inherited a 0.1 bias warm-up rate. The failed run is preserved and rejected.
2. **Close-ups v2:** fresh initialization from the unchanged baseline, six epochs
   on the derived dataset, MPS, batch 8, image size 640, AdamW lr=0.0001,
   scale=0.5, rotation ±15°, seed 42. Bias warm-up starts at zero; mosaic is disabled
   from the first epoch. All six epochs completed in 1218.94 seconds of measured
   training time (about 20 minutes 19 seconds). `best.pt` and `last.pt` exist.
   Best checkpoint selection uses validation performance, not the phone photos.

Each run records its resolved YAML, environment, seed, data fingerprint, and
metadata in `outputs/training/<run>/`. MPS is not perfectly deterministic.

## Comparable validation evaluation

Both checkpoints were evaluated on the **original** 430 validation images with
`src/evaluate.py`, CPU, batch 8, size 640, confidence floor 0.001, seed 42, and no
inference augmentation. The held-out test set was not reevaluated or used to tune.
All values below are percentages; F1 is the macro per-class F1 from the existing
evaluation module.

| Model | Precision | Recall | F1 | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (retained) | 89.36 | 85.77 | 86.78 | 88.55 | 71.08 |
| Close-ups v2 (rejected) | 86.30 | 87.94 | 86.71 | 88.21 | 59.99 |

**Deployment decision:** retain baseline. The candidate loses about 11.09
percentage points of mAP50–95 and still produces a lemon false positive on bananas.
Higher recall alone does not justify switching. This is evidence against deploying
this candidate, not proof that further training can never help.

Machine-readable aggregate/per-class metrics and photo predictions:
[real_world_review.json](results/real_world_review.json).
Full logs, source-scale audit, predictions, and integrity checks remain under
`outputs/improvement_20260919/` (the task continued on September 20).

## Reproduce the recorded workflow

Use new output/run names when repeating an experiment; existing runs are protected.

```bash
python src/prepare_closeups.py --output data/closeups_v1
python src/train.py --data data/closeups_v1/dataset.yaml \
  --model outputs/training/fruitvision_baseline/weights/best.pt \
  --epochs 6 --img-size 640 --batch-size 8 --device mps \
  --learning-rate 0.0001 --scale 0.5 --rotation 15 \
  --name fruitvision_closeups_v2
python src/evaluate.py --split val --device cpu --batch-size 8 \
  --weights outputs/training/fruitvision_closeups_v2/weights/best.pt \
  --name improvement_closeups_validation
```

These commands document the rejected experiment; they are not a recommendation to
keep repeating it or replace the production checkpoint.

## UI changes and verification

- Detection output appears directly; original photo remains in an expander.
- Show only detected class counts, object count, and average confidence.
- Hide threshold controls and table/JSON details until requested; retain sizing.
- Remove placeholder metrics, sidebar slogans, and repeated technical copy.
- Scale annotation text and line widths to photo dimensions for readable labels.
- Preserve all five pages, caching, CLI behavior, and JSON fields.
- **82 tests pass**, including crop-coordinate/source-preservation checks, a
  warm-up regression test, and compact UI checks. Browser verification covered
  real inference, JSON downloads, all pages, mobile navigation, and no overflow.

## Next step

Collect and annotate varied real photos of the intended banana varieties alongside
lemon and other confusable fruit: close-up and distant views, different backgrounds,
lighting, and occlusion. Group photos from the same capture session before splitting
to avoid similar scenes leaking across splits. Keep a separate, unseen phone-photo
validation set. Review whether annotations represent individual fruit or bunches
before interpreting counts. Decide promotion criteria before the next run.

This recommendation follows the observed coverage gap and unsuccessful trials,
and agrees with Ultralytics' guidance on
[image variety for real-world use](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/).
Simply training until these three familiar photos look correct would not establish
that the detector generalizes to new images.
