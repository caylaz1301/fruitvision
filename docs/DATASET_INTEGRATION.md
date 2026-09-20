# FruitVision — ten-class dataset integration report

## Outcome

**Dataset integration and the five-epoch sanity run succeeded.** All 28 tests passed. The original archive, images, class IDs, and split membership were preserved. No final baseline, robustness experiments, size estimation, or UI were run or added.

## Dataset provenance

- ZIP: `/Users/vdr/Downloads/Fruit.v1i.yolov11.zip`
- SHA-256: `678be4a962f09ed357b67ec0344bf07002f9f465632ae3b66d068f523ad46013`
- Source: [Roboflow Fruit v1](https://universe.roboflow.com/simons-workspace-l89qb/fruit-smrhb-tp0nb/dataset/1), workspace `simons-workspace-l89qb`, project `fruit-smrhb-tp0nb`, version 1.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The export identifies the provider as a Roboflow user.
- Changes: map the `valid` directory to `val`; convert normalized polygons to enclosing detection boxes. Original metadata is retained in `data/processed/source_metadata/`.

## Measured dataset counts

| Split | Images | Labels | Standard boxes | Converted polygons | Annotation rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,550 | 1,550 | 3,271 | 134 | 3,405 |
| val | 430 | 430 | 865 | 15 | 880 |
| test | 237 | 237 | 426 | 13 | 439 |
| **Total** | **2,217** | **2,217** | **4,562** | **162** | **4,724** |

All class totals exactly match the user-supplied sanity reference. These are recalculated annotation-row counts, including the one duplicate row described below.

| ID | Class | Objects |
| --- | --- | ---: |
| 0 | apple | 273 |
| 1 | banana | 269 |
| 2 | grape | 431 |
| 3 | kiwi | 593 |
| 4 | lemon | 224 |
| 5 | orange | 424 |
| 6 | peach | 235 |
| 7 | pineapple | 321 |
| 8 | strawberry | 955 |
| 9 | watermelon | 999 |

## Quality findings

- Zero missing image-label pairs, invalid annotation coordinates, out-of-range class IDs, or unreadable images.
- No byte-identical or decoded-pixel-identical images across splits. This does not rule out near-duplicate scenes.
- Three identical `data.yaml` entries were verified; one copy was preserved and the two repetitions were recorded.
- Two within-split duplicate image pairs (one train, one validation) have different labels. The training apple pair has a full-image box in one label and a much smaller box in the other. All are retained.
- One training label is empty despite visible orange fruit: `img_411_jpeg.rf.594874dd52557bd505fa98f2fa77b9a3.txt`. This likely missing annotation remains unchanged; it is not a verified background image.
- One training file repeats an identical apple box: `-00098_jpeg_jpg.rf.aec4825c02cae8aecba83b0e528e6a0a.txt`. The import preserves both rows. Ultralytics logs removal of one duplicate when loading, so training uses 3,404 effective objects versus 3,405 source rows.
- All 2,217 images are 384×384; the source metadata records stretch resizing before export. A 640-pixel training input upsamples them.
- Class imbalance: watermelon has 999 annotations versus lemon’s 224 (about 4.46×). No resampling or balancing was applied. Review per-class evaluation after baseline training.
- Fixed seeds are recorded, but PyTorch warns that some MPS scatter/index operations are nondeterministic.

## Checks and sanity training

- `python -m unittest discover -s tests -v`: **28 passed**, including all 13 original applicable tests updated to the ten-class mapping.
- `python src/preprocessing.py validate`: **passed** on all supplied splits.
- Source comparison: all 2,217 image files are byte-identical; all 4,724 annotation rows retained; exactly 162 polygons converted. Original ZIP hash rechecked after training.
- Notebook Python cells parse and both notebooks remain unexecuted.
- Sanity command: `python src/train.py --epochs 5 --img-size 640 --batch-size 8 --name sanity_10class`.
- Training: **5 epochs completed**, Apple M2 **MPS**, 8 GiB shared memory; about 13 minutes. Validation used `val`, and final checkpoint validation completed. Recorded training values are finite.
- Checkpoints: `outputs/training/sanity_10class/weights/best.pt` and `last.pt`.
- Training artifacts: `results.csv`, `args.yaml`, resolved dataset YAML, `run_metadata.json`, `dataset_provenance.json`, `environment.txt`, plots, and `outputs/training/sanity_10class.log`.

### Real-image inference verification

Three filenames were chosen before prediction: first, middle, and last of the sorted test split. Inference used the unchanged confidence threshold 0.25 on CPU. No test metrics or hyperparameter tuning were performed.

| Image | Returned detections | Annotated output |
| --- | ---: | --- |
| `-00064_jpeg_jpg.rf.b4594ae38c4757c9f75dcdafa410ad2d.jpg` | 1 | [View](../outputs/predictions/sanity_10class/-00064_jpeg_jpg.rf.b4594ae38c4757c9f75dcdafa410ad2d_annotated.jpg) |
| `banana5_043_jpg.rf.a80e07d9d10bbdb4e9539086fe57174d.jpg` | 2 | [View](../outputs/predictions/sanity_10class/banana5_043_jpg.rf.a80e07d9d10bbdb4e9539086fe57174d_annotated.jpg) |
| `watermalon85_jpg.rf.23a36897aac5660fb469c4f2b04f7f93.jpg` | 4 | [View](../outputs/predictions/sanity_10class/watermalon85_jpg.rf.23a36897aac5660fb469c4f2b04f7f93_annotated.jpg) |

Bounding boxes, class names, confidence labels, ten-class summaries, count arithmetic, JSON coordinates, count charts, and inference timing were checked. All three annotations were visually reviewed. Neighboring text labels were adjusted to avoid overlap. Counts describe returned detections, not guaranteed ground-truth accuracy. Five-epoch metrics are **not final model performance**.

## Files changed

- `src/common.py`: explicit ten-class mapping and checkpoint class validation.
- `src/preprocessing.py`: shared box parser, ten-class checks, preserved Roboflow metadata.
- `src/train.py`: ten-class training checks and comments.
- `src/predict.py`: inference timing added while retaining the existing detection/summary API.
- `src/visualization.py`: generic class colors, readable ten-class charts, non-overlapping label placement.
- `configs/dataset.yaml`: exact source class IDs and Roboflow attribution.
- `tests/test_pipeline.py`: preserved existing checks, updated mapping, added neighboring-label regression coverage.
- `README.md`, `data/README.md`, and both notebooks: actual dataset, conversion, imbalance, quality findings, and commands.
- `.gitignore`: keeps datasets, checkpoints, generated outputs, and temporary import directories excluded.
- `src/counting.py` and `src/evaluate.py` were reviewed and already operate generically over the shared class mapping; no rewrite was needed. No dependencies were added.

## Files added

- `src/import_dataset.py`: safe staged local ZIP import and deterministic polygon conversion.
- `tests/test_import_dataset.py`: importer, mapping, conversion, preservation, and rejection tests.
- `docs/DATASET_INTEGRATION.md`: this report.
- Generated local artifacts: `outputs/dataset_audit.json`, `outputs/dataset_audit.csv`, `outputs/dataset_quality_review.json`, `outputs/figures/dataset_class_distribution.png`, `outputs/test_results.txt`, `outputs/sanity_inference_verification.json`, and `outputs/integration_summary.json`.
- Imported dataset, source metadata, manifest, sanity checkpoints, and prediction images/JSON/count charts remain excluded from Git.

## Next baseline command

```bash
python src/train.py --epochs 50 --img-size 640 --batch-size 8 --name fruitvision_baseline
```

Batch size 8 completed the sanity run with roughly 3.25 GB reported GPU memory usage and is a conservative choice on 8 GiB shared-memory hardware. It leaves more headroom than 16. A fifty-epoch run will take substantially longer; it was **not launched**. Preserve this baseline’s disclosed label issues or explicitly version future corrections. Do not tune using held-out test results.
