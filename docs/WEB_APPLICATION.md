# FruitVision Web Application

**FruitVision — Multi-Fruit Detection & Visual Analytics** is a local Streamlit
dashboard around the existing frozen baseline and recorded experiments.

## Launch

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

Open `[http://127.0.0.1:8501]`. `.streamlit/config.toml` supplies the theme, disables
usage statistics, limits upload bytes, and binds the server to localhost. A missing
checkpoint does not prevent browsing saved results; Detect explains which local
checkpoint needs to be restored.

## Pages

- **Detect:** upload JPG/PNG and explicitly run inference. The annotated result
  appears directly; the original photo stays available in an expander. Object
  count, average confidence, and counts for detected classes are shown first.
  Threshold controls and table/JSON details are collapsed until requested. The
  JSON still contains counts for all ten classes. No reference is needed.
- **Approximate Size Estimation** inside Detect: enable only when a real reference
  is present. Provide its known width in centimeters and pixel width or xyxy box.
  Coordinates refer to the original EXIF-oriented image, not the scaled preview.
  Partial, nonpositive, or out-of-bounds inputs prevent inference with a clear
  message. Size fields and the warning are shown only when calibration is enabled.
- **Analytics:** count chart, class shares, confidence histogram, and details from
  the current session prediction. Empty sessions explain how to create a result;
  zero detections give zero counts and unavailable average confidence.
- **Model Performance:** reads saved validation/test metric JSON and figures.
  Reports precision, recall, F1, mAP50, mAP50–95, per-class metrics, confusion matrix,
  and training curves. It does not rerun evaluation.
- **Robustness:** reads the five saved validation conditions and displays measured
  metrics, relative degradation, class sensitivity, and fixed settings. Results
  are diagnostic because source annotations are imperfect.
- **About:** purpose, supported classes, limitations, dataset attribution, and
  session/privacy behavior.

## Architecture

| File/module | Responsibility |
| --- | --- |
| `app/app.py` | Navigation, Streamlit widgets, session state, cached model, display |
| `app/helpers.py` | Image validation, result-to-table conversion, saved-report readers, inference adapter |
| `app/styles.css` | Responsive custom layout, readable colors, focus and reduced-motion styles |
| `src/predict.py` | Existing `predict_image()` inference and detection extraction |
| `src/counting.py` | Existing summaries, called by the prediction pipeline |
| `src/visualization.py` | Existing box/class/confidence rendering; the UI crops only its added count footer because cards show those counts |
| `src/size_estimation.py` | Existing reference validation and approximate widths |
| `src/evaluate.py` exports | Source of baseline metric JSON, tables, and figures |
| `experiments/experiment_runner.py` | Shared degradation calculation; recorded outputs provide robustness scores |

No model training or experiment execution path is exposed by the dashboard. The
CLI remains unchanged. `best.pt` is loaded from the existing baseline directory,
with automatic CUDA/MPS/CPU selection inherited from `src/common.py`.

Model loading uses `st.cache_resource`, keyed by checkpoint path, modification time,
and file size. A lock serializes inference through its shared mutable YOLO predictor.
Uploaded photos and results stay in per-session state, never in the global cache.
This follows Streamlit's [shared-resource caching guidance](https://docs.streamlit.io/develop/concepts/architecture/caching).

Image bytes and detection controls persist when navigating between sections. The
image/settings signature invalidates stale results when the user changes the
upload, threshold, or reference. Clear image removes the active upload and result.
Changing or clearing the image also resets the size reference: calibration belongs
to that particular photo and must be entered again for a different image.
Downloads are created in memory; the app does not save images into the dataset.

## Result Sources

Baseline JSON is loaded from `outputs/metrics/baseline_validation/metrics.json` or
`outputs/metrics/baseline_test/metrics.json`, falling back to the corresponding
`docs/results/*_metrics.json` only if the primary file is missing. Robustness reads
`outputs/experiments/robustness/results.json`, with the curated
`docs/results/robustness/results.json` as a fallback. Figures have matching local
run/curated fallback locations.

Malformed files produce errors; they are not replaced with sample scores or silently
substituted by a fallback. Missing figures show a notice. The displayed file path
identifies provenance. Performance pages describe historical experiments, while
Analytics describes the current uploaded image.

## Verification

The full suite passed **82 tests**, including the existing pipeline, UI, close-up
preparation, and fine-tuning regression tests. Coverage includes supported/corrupt uploads, orientation, image limits,
input signatures, table formatting, corrupt/missing metrics, reuse of prediction
logic, all five page renders, incomplete references, model caching/failure, no
detections, retained navigation state, stale-result clearing, and reference reset.

```bash
python -m unittest discover -s tests -v
```

Log: `outputs/web_app_tests.txt`. App tests use Streamlit's
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
and temporary fixtures; their mock predictions are not model benchmarks.

Browser checks on a 1440px desktop viewport and a 390px mobile viewport covered
real validation-image upload/inference, downloaded JSON count consistency, page
navigation, and responsive layout. The test image was
`-00027_jpeg_jpg.rf.a58b422bf217ce644f295e060f540d5a.jpg`; no physical reference was
invented for it. The baseline checkpoint SHA-256 remains
`52a50db1f0c3e7384040359887683bfdaa764ae996c72e87fd7f194064430d3c`.

## Known Limitations and Troubleshooting

- Local use only: no authentication, durable upload storage, shared history, or
  production deployment controls. A reload/server restart can clear session data.
- Uploads are JPG/PNG, at most 10 MB and 20 megapixels. Very large photos should be
  resized before measuring a reference, using measurements from that final image.
- Shared-model inference is serialized. First inference includes initialization
  and can be slower; the displayed timing is not a hardware throughput benchmark.
- Size is an approximate horizontal box width, not diameter. Perspective, depth,
  orientation, distance, lens distortion, calibration, and detection errors remain.
- Dataset annotation defects and limited domain coverage still apply. Confidence
  is not measured accuracy. See the baseline and robustness reports for evidence.
- Styling uses Streamlit test identifiers and is checked with pinned version
  1.64.0. Upgrades can require a style/accessibility regression check.
- This machine's pre-existing virtual environment inherits system packages.
  Installing Streamlit reported a dependency conflict between its Starlette version
  and an inherited, unused FastAPI 0.115.6. FruitVision does not use FastAPI, and its
  full tests and browser flows passed. For a clean standalone environment, use the
  README's standard `python3 -m venv .venv` installation without system-site packages;
  do not reuse this environment for that unrelated FastAPI application.

Dataset source and CC BY 4.0 attribution remain in README and the About page.

## Real-world review and simpler Detect layout

The September 2026 update removes empty statistic cards, zero-count fruit chips,
sidebar slogans, and repeated explanations. Result images are shown immediately;
secondary information stays accessible through expanders. Box-label text scales
with the original photo dimensions. Counts describe detected boxes; a bunch may
be one object, depending on learned annotation conventions.

The additional fine-tuning candidate was rejected; the app's baseline path and
historical performance/robustness reports remain unchanged. The reported banana
photo still needs a model/data improvement. See [Real-world review](REAL_WORLD_REVIEW.md)
for actual predictions, comparable validation metrics, and the next data requirement.
