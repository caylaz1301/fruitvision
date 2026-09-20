# Preparing a real fruit detection dataset

The current baseline is the locally imported **Roboflow Fruit v1** dataset with
2,217 images and ten classes. See the [main README](../README.md#dataset-attribution)
for CC BY 4.0 attribution. For a future dataset, use **object detection** annotations
with bounding boxes, or annotate your own photos. A classification dataset with
one folder per fruit is not sufficient: the detector needs each fruit's location.
Record the source URL, license, download date, class mapping, and any filtering in
your experiment notes. Check that the license permits your intended use.

## Import the supplied Roboflow ZIP

```bash
python src/import_dataset.py --zip ~/Downloads/Fruit.v1i.yolov11.zip
python src/preprocessing.py validate
```

The importer preserves 1,550 train, 430 validation, and 237 test images. It maps
`valid` to `val`; it never follows the source YAML's `../` paths on your filesystem.
Original image bytes are copied unchanged, and `data.yaml`, `README.roboflow.txt`,
and `README.dataset.txt` are retained in `processed/source_metadata/`.
No dataset is downloaded. The original ZIP is never written to.

The archive contains mixed detection and segmentation annotations. A polygon line
has at least three normalized x/y pairs. Conversion uses:

```text
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
width = x_max - x_min
height = y_max - y_min
```

Class IDs stay unchanged. Every converted detection is validated. Conversion is
deterministic (17 significant digits); standard box lines retain their numeric
text. No annotations are clipped or silently discarded. Invalid objects block
publication of the whole staged import. The measured conversion count is 162,
leaving 4,724 total objects across all splits.

`outputs/dataset_audit.json` records per-split/per-class counts, conversion counts,
source metadata/hash, missing pairs, invalid labels, empty labels, duplicate names,
byte-identical files, and decoded-pixel-identical images. The real class chart is
`outputs/figures/dataset_class_distribution.png`. Identical repeated metadata is
recorded and retained once; ambiguous duplicate data paths and cross-split image
leakage are fatal. Within-split duplicates are reported and retained.

### Source-quality findings

- Two duplicate image pairs within splits have differing annotations; one training
  apple pair differs substantially. No cross-split exact/pixel duplicates were found.
- `train/labels/img_411_jpeg.rf.594874dd52557bd505fa98f2fa77b9a3.txt` is empty, but
  its image visibly contains orange fruit. It remains unchanged, so the model sees
  this mislabeled sample as background. It is not a verified negative example.
- One training file (`-00098_jpeg_jpg.rf.aec4825c02cae8aecba83b0e528e6a0a.txt`)
  repeats the same apple box twice in the source archive. Before baseline training,
  only the redundant processed row was removed. The original import audit retains
  source counts; `outputs/dataset_audit_baseline.json` records 3,404 current training
  annotations. The before/after text is in `outputs/dataset_corrections.json`.
- All source images were stretched to 384×384 by Roboflow before export.
- The most frequent class (watermelon, 999 objects) has about 4.46 times the
  examples of the least frequent class (lemon, 224). No balancing was applied.

These issues should be reviewed before a final portfolio performance claim.
Record any future corrections as a new dataset version rather than silently
changing this baseline. Automated checks establish structural validity; they
cannot guarantee that a human drew the correct box or class label.

## Already split YOLO data

Place your real images and corresponding label files here:

```text
data/processed/
├── images/
│   ├── train/  # training photos
│   ├── val/    # validation photos
│   └── test/   # held-out test photos
└── labels/
    ├── train/  # matching .txt files
    ├── val/
    └── test/
```

For example, `images/train/basket.jpg` pairs with `labels/train/basket.txt`.
Nested directories are supported if the same relative layout exists in labels.
Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
Do not use two images with the same relative stem (such as `basket.jpg` and
`basket.png`); they would share one ambiguous label file.

`configs/dataset.yaml` uses this mapping:

| ID | Class |
| --- | --- |
| 0 | apple |
| 1 | banana |
| 2 | grape |
| 3 | kiwi |
| 4 | lemon |
| 5 | orange |
| 6 | peach |
| 7 | pineapple |
| 8 | strawberry |
| 9 | watermelon |

Remap a public dataset's IDs before use; IDs often differ between datasets.
Annotate every visible target fruit consistently. For non-target objects, omit
their annotations after remapping; do not accidentally delete target labels.
An empty `.txt` means a verified image containing **no target fruits**. FruitVision
requires explicit empty files so an accidentally missing label cannot silently
become a background example.

## YOLO annotation format

One line per object, with **five values**:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to `[0, 1]`. Width and height must be greater than zero.
For an image of width `W`, height `H`, and pixel corners `(x_min, y_min, x_max, y_max)`:

```text
x_center = (x_min + x_max) / (2 * W)
y_center = (y_min + y_max) / (2 * H)
width    = (x_max - x_min) / W
height   = (y_max - y_min) / H
```

The origin is the top left: x grows rightward and y grows downward. The class ID
is an integer, not a normalized coordinate. Width and height are required;
center coordinates alone cannot describe a box. Coordinates refer to the original
image before YOLO's internal resizing. Visually inspect labels after conversion.

Normalize phone-photo EXIF orientation **before annotation/export**. The checker
rejects stored rotations to prevent inconsistent image/box alignment. Do not rotate
or crop already annotated images without also transforming their boxes.

## Other datasets only: unsplit, already annotated data

**Do not re-split the current Roboflow dataset.** For a different, unsplit dataset,
put matching files in `data/raw/images/` and `data/raw/labels/`, then run:

```bash
python src/preprocessing.py split --seed 42
python src/preprocessing.py validate
```

The utility copies files into 70% train, 15% validation, and the remainder test
(rounding down train/validation counts). Originals are preserved. Set
`--train-ratio` and `--val-ratio` to change this. The destination must be empty
apart from `.gitkeep` files. Use `--output` for a new destination, and update
`path` in `configs/dataset.yaml` accordingly.

Splitting saves filenames, image SHA-256 hashes, ratios, and seed in
`data/processed/split_manifest.json`. A fixed seed reproduces the split for the
same input files. This is a simple image-level random split, **not** a stratified
or group-aware split. Check class coverage with `validate` and the exploration
notebook; the training split must include all ten classes. Aim to include all
ten in validation and test too.

Keep photos of the same fruit arrangement, video sequence, or photo session in
one split. For such data, prepare grouped splits manually instead of using this
random splitter. Exact file duplicates are detected, but near-duplicates and
re-encoded copies require your review. Preserve an official public dataset split
when appropriate; do not repeatedly change your test split to improve scores.

## What preprocessing does

The utility checks images and annotations and copies paired files. It does not
invent labels or resize your source images. Ultralytics handles model-input
resizing with padding (letterboxing), pixel scaling, and box transformations.
Training augmentations, such as flips, color changes, and mosaic combinations,
operate on images and labels together. Validation and test do not use random
training augmentation.

Training learns weights; validation helps select settings and the best checkpoint;
test estimates performance on unseen data after you finalize those decisions.
All three splits need annotated objects for a meaningful full evaluation.

## Paths and checks

Run `python src/preprocessing.py validate` before training. `path` in the config
is resolved relative to the config file by FruitVision. An absolute copy is saved
inside each run folder for Ultralytics; use the supplied scripts with this YAML.
The checker expects the fixed `images/train`, `images/val`, `images/test` layout
and matching labels, rather than image-list text files or download instructions.

Datasets and generated split manifests are ignored by Git. Share provenance and
split decisions in documentation without committing the image collection.

## Optional close-up training copy

`python src/prepare_closeups.py --output data/closeups_v1` builds an isolated,
ignored experiment dataset. It retains original training examples and adds crops
around the union of their existing boxes, with 25% padding. Every labeled object
is retained and coordinates are renormalized. Validation/test files are copied
byte-for-byte; original files and the Roboflow ZIP are not edited.

These are extra views of existing photographs, not independent new data. The
manifest records every crop and skip. Existing destinations are protected from
overwrite. The recorded close-up model did not solve the reported banana error;
see [the experiment review](../docs/REAL_WORLD_REVIEW.md) before repeating it.
