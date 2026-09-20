"""FruitVision dashboard. Launch from the repository root: streamlit run app/app.py."""

import html
import json
import sys
from pathlib import Path
from threading import Lock

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.helpers import (CONDITION_LABELS, METRICS, SIZE_NOTICE, baseline_report, decode_upload,
                         detection_rows, image_bytes, metric_rows, prediction_signature,
                         read_report, robustness_report, run_prediction)
from experiments.experiment_runner import degradation
from src.common import BEST_MODEL, CLASS_NAMES, ROOT, load_detector, select_device
from src.size_estimation import build_reference

st.set_page_config(page_title="FruitVision · Visual Analytics", page_icon="◈", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'styles.css').read_text()}</style>", unsafe_allow_html=True)

# Keep detection controls when another section temporarily omits their widgets.
for setting in ("confidence", "use_reference", "reference_cm", "reference_mode", "reference_pixels",
                "reference_X1", "reference_Y1", "reference_X2", "reference_Y2"):
    if setting in st.session_state:
        st.session_state[setting] = st.session_state[setting]


@st.cache_resource(show_spinner=False)
def cached_detector(path: str, modified_ns: int, file_size: int):
    """YOLO's mutable predictor is shared, so serialize inference across sessions.

    File metadata invalidates the cache if the checkpoint is deliberately replaced.
    Uploaded images and predictions are never stored in this global cache.
    """
    return load_detector(Path(path)), Lock(), select_device()


def heading(title: str, description: str) -> None:
    st.markdown(f'<div class="page-heading"><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def cards(items: list[tuple[str, str]]) -> None:
    markup = ''.join(f'<div class="stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>' for label, value in items)
    st.markdown(f'<div class="stats">{markup}</div>', unsafe_allow_html=True)


def empty_panel(title: str, description: str) -> None:
    st.markdown(f'<div class="empty-panel"><div class="viewfinder" aria-hidden="true">⌖</div><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def upload_changed(key: str) -> None:
    upload = st.session_state.get(key)
    st.session_state["upload_data"] = (upload.name, upload.getvalue()) if upload is not None else None
    st.session_state.pop("prediction", None)
    clear_reference()


def clear_reference() -> None:
    """A new image needs a new physical reference, not the last photo's scale."""
    st.session_state["use_reference"] = False
    for key in ("reference_cm", "reference_pixels", "reference_mode", "reference_X1", "reference_Y1", "reference_X2", "reference_Y2"):
        st.session_state.pop(key, None)


def clear_image() -> None:
    st.session_state["upload_data"] = None
    st.session_state.pop("prediction", None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
    clear_reference()


def go_to_detect() -> None:
    st.session_state["page"] = "Detect"


def reference_controls(image) -> tuple[dict | None, str | None]:
    reference, error = None, None
    with st.expander("Approximate Size Estimation"):
        enabled = st.toggle("Use a reference object", key="use_reference")
        st.caption("Optional. Measure a known object in the same image, at approximately the fruit's depth.")
        if enabled:
            st.warning(SIZE_NOTICE)
            width_cm = st.number_input("Known reference width (cm)", min_value=0.0, value=None,
                                       placeholder="Enter measured width", key="reference_cm")
            mode = st.radio("Reference measurement", ["Pixel width", "Bounding box"], horizontal=True, key="reference_mode")
            width_pixels, box = None, None
            if mode == "Pixel width":
                width_pixels = st.number_input("Reference width (pixels)", min_value=0.0, value=None,
                                               placeholder="Width in original image", key="reference_pixels")
            else:
                values = []
                for column, name in zip(st.columns(4), ("X1", "Y1", "X2", "Y2")):
                    with column:
                        values.append(st.number_input(name, min_value=0.0, value=None, key=f"reference_{name}"))
                box = values
            st.caption("Use original image pixels after orientation correction, not coordinates from this scaled preview.")
            try:
                if width_cm is None or (mode == "Pixel width" and width_pixels is None) or (box is not None and None in box):
                    raise ValueError("Enter the known width and a complete pixel measurement to enable detection.")
                reference = build_reference(width_cm, width_pixels, box, image.size if image is not None else None)
            except ValueError as exception:
                error = str(exception)
                st.error(error)
    return reference, error


def result_cards(payload: dict) -> None:
    summary = payload["summary"]
    average = summary["average_confidence"]
    cards([("Total fruits", str(summary["total_objects"])),
           ("Detected classes", str(sum(count > 0 for count in summary["counts"].values()))),
           ("Average confidence", f"{average:.1%}" if average is not None else "N/A"),
           ("Inference time", f"{summary['inference_time_ms']:.0f} ms")])


def details(payload: dict) -> None:
    if "size_estimation" in payload:
        st.warning(SIZE_NOTICE)
        st.caption("Widths describe horizontal detection boxes, not true fruit diameters.")
    rows = detection_rows(payload["detections"])
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Box coordinates are x1, y1, x2, y2 in original, orientation-corrected image pixels.")
    else:
        st.info("No fruit detections at this threshold. Try another photo or a lower confidence threshold.")


def detect_page() -> None:
    heading("Detect fruit", "Upload a photo to identify and count fruit.")
    controls, preview = st.columns([0.9, 1.7], gap="large")
    with controls:
        upload_key = f"image_upload_{st.session_state.get('upload_version', 0)}"
        st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], max_upload_size=10,
                         key=upload_key, on_change=upload_changed, args=(upload_key,),
                         help="JPG or PNG · up to 10 MB and 20 megapixels. Photos stay in this session.")
        upload = st.session_state.get("upload_data")
        image, input_error = None, None
        if upload:
            try:
                image = decode_upload(upload[1])
                st.caption(f"{image.width} × {image.height} pixels")
                st.button("Clear image", on_click=clear_image, type="tertiary")
            except ValueError as error:
                input_error = str(error)
                st.error(input_error)
        with st.expander("Detection settings"):
            confidence = st.slider("Confidence threshold", 0.05, 0.95,
                                   None if "confidence" in st.session_state else 0.25, 0.05, key="confidence")
            st.caption("Higher values hide weaker predictions. They do not correct a wrong label.")
        reference, reference_error = reference_controls(image)
        signature = prediction_signature(upload[1], confidence, reference) if image is not None and not reference_error else None
        existing = st.session_state.get("prediction")
        if existing is not None and existing["signature"] != signature:
            st.session_state.pop("prediction", None)
        if st.button("Run Detection", type="primary", width="stretch",
                     disabled=image is None or bool(input_error or reference_error)):
            st.session_state.pop("prediction", None)
            try:
                with st.spinner("Finding fruit in your image…"):
                    if not BEST_MODEL.is_file():
                        raise FileNotFoundError("The frozen baseline checkpoint is missing. Restore outputs/training/fruitvision_baseline/weights/best.pt.")
                    info = BEST_MODEL.stat()
                    model, lock, device = cached_detector(str(BEST_MODEL), info.st_mtime_ns, info.st_size)
                    with lock:
                        payload, annotated = run_prediction(model, image, upload[0], confidence, reference, device)
                    st.session_state["prediction"] = {"signature": signature, "payload": payload,
                                                       "annotated": image_bytes(annotated)}
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                st.error(f"Detection could not complete. {error}")
            except Exception:
                st.error("The model could not be loaded or run. Check the local checkpoint and installed dependencies, then try again.")
    prediction = st.session_state.get("prediction")
    with preview:
        if prediction:
            st.image(prediction["annotated"], width="stretch")
            with st.expander("View original photo"):
                st.image(image, width="stretch")
        elif image is not None:
            st.image(image, width="stretch")
        else:
            empty_panel("Choose a photo", "JPG or PNG · up to 10 MB")
    if not prediction:
        return
    payload = prediction["payload"]
    average = payload["summary"]["average_confidence"]
    cards([("Detected objects", str(payload["summary"]["total_objects"])),
           ("Average confidence", f"{average:.1%}" if average is not None else "N/A")])
    counts = payload["summary"]["counts"]
    chips = ''.join(f'<span class="count-chip">{html.escape(name.title())}<b>{count}</b></span>' for name, count in counts.items() if count)
    st.markdown(f'<div class="count-strip">{chips}</div>', unsafe_allow_html=True)
    st.caption("Check the labels: confidence is not a guarantee. A fruit bunch may count as one object.")
    if not payload["detections"]:
        st.info("No fruit detected. Try a clearer photo or adjust Detection settings.")
    if "size_estimation" in payload:
        st.warning(SIZE_NOTICE)
        st.dataframe([{ "Fruit": row["Fruit"], "Approx. width (cm)": row["Approx. width (cm)"]}
                      for row in detection_rows(payload["detections"])], hide_index=True, width="stretch")
    st.download_button("Download annotated image", prediction["annotated"], file_name="fruitvision_detection.png", mime="image/png")
    with st.expander("Detection details & JSON"):
        st.caption(f"Inference: {payload['summary']['inference_time_ms']:.0f} ms · confidence threshold: {payload['confidence_threshold']:.0%}")
        details(payload)
        st.download_button("Download predictions · JSON", json.dumps(payload, indent=2, allow_nan=False),
                           file_name="fruitvision_predictions.json", mime="application/json")


def analytics_page() -> None:
    heading("Every detection, in context.", "Explore the current image's class counts, confidence scores, and object-level details.")
    prediction = st.session_state.get("prediction")
    if prediction is None:
        empty_panel("Your analysis starts with a detection", "Open Detect, upload a photo, and run the model. Charts here will use only your current prediction.")
        st.button("Go to Detect", type="primary", on_click=go_to_detect)
        return
    payload = prediction["payload"]
    st.caption(f"Current image: {payload['image']} · threshold {payload['confidence_threshold']:.0%}")
    result_cards(payload)
    counts = payload["summary"]["counts"]
    total = payload["summary"]["total_objects"]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Fruit counts")
        st.bar_chart({"Fruit": [name.title() for name in counts], "Objects": list(counts.values())},
                     x="Fruit", y="Objects", horizontal=True, height=350)
    with right:
        st.subheader("Confidence distribution")
        if total:
            # This is a display histogram, not another inference or accuracy estimate.
            from matplotlib.figure import Figure
            figure = Figure(figsize=(6, 3.8), layout="constrained")
            axis = figure.subplots()
            axis.hist([row["confidence"] * 100 for row in payload["detections"]], bins=list(range(0, 101, 10)))
            axis.set(xlabel="Confidence (%)", ylabel="Detected objects", xlim=(0, 100))
            from matplotlib.ticker import MaxNLocator
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            st.pyplot(figure, width="stretch")
        else:
            st.info("No detections to plot. Counts remain zero; confidence is unavailable.")
        st.caption("Confidence describes model scores, not measured accuracy.")
    with st.expander("Class distribution", expanded=True):
        st.dataframe([{"Fruit": name.title(), "Count": count, "Share (%)": round(count / total * 100, 1) if total else None}
                      for name, count in counts.items()], hide_index=True, width="stretch")
    st.subheader("Detection details")
    details(payload)


def show_figure(paths: list[Path], caption: str) -> None:
    path = next((path for path in paths if path.is_file()), None)
    if path:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info("This figure is not available in the saved run.")


def performance_page() -> None:
    heading("Understand the baseline.", "Recorded baseline performance. Explore validation and held-out test results without running a new evaluation.")
    split_label = st.radio("Evaluation split", ["Validation", "Held-out test"], horizontal=True)
    split = "val" if split_label == "Validation" else "test"
    try:
        report, path = baseline_report(split)
        cards([(label, f"{report['overall'][key]:.2%}") for key, label in METRICS.items()])
        st.caption(f"Source: {path.relative_to(ROOT)} · frozen baseline checkpoint")
        st.subheader("Per-class performance")
        st.dataframe(metric_rows(report["per_class"]), hide_index=True, width="stretch")
        st.bar_chart({"Fruit": [row["class_name"].title() for row in report["per_class"]],
                      "AP50–95 (%)": [row["map50_95"] * 100 for row in report["per_class"]]},
                     x="Fruit", y="AP50–95 (%)", horizontal=True, height=350)
        with st.expander("How to read these metrics"):
            st.write(report.get("metric_notes", "Metrics are loaded directly from the recorded evaluation."))
            st.write("mAP50–95 requires progressively tighter box overlap. Confidence is not accuracy; these scores describe this dataset and split.")
        matrix, curves = st.tabs(["Confusion matrix", "Training curves"])
        name = "validation" if split == "val" else "test"
        with matrix:
            show_figure([ROOT / f"outputs/metrics/baseline_{name}/confusion_matrix_normalized.png",
                         ROOT / f"docs/assets/{name}_confusion_matrix.png"], "Recorded normalized confusion matrix · predicted rows, true columns")
            st.caption("The saved baseline matrix uses confidence 0.001 and IoU 0.45; its low-score assignments differ from standard inference and AP matching.")
        with curves:
            show_figure([ROOT / "outputs/training/fruitvision_baseline/results.png", ROOT / "docs/assets/training_curves.png"], "Original Ultralytics training curves")
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(str(error))


def robustness_page() -> None:
    heading("What happens when the image changes?", "Five controlled validation conditions. One frozen model. No retraining.")
    st.info("Robustness results are diagnostic because source dataset annotations are imperfect. Rankings apply to the tested transformation settings.")
    try:
        report, path = robustness_report()
        rows = report["overall"]
        normal = next(row for row in rows if row["condition"] == "normal")
        changes = [{"Condition": CONDITION_LABELS[row["condition"]],
                    "Relative degradation (%)": degradation(normal["map50_95"], row["map50_95"])["relative_percent"]} for row in rows]
        worst = min(rows, key=lambda row: row["map50_95"])
        change = degradation(normal["map50_95"], worst["map50_95"])
        cards([("Normal mAP50–95", f"{normal['map50_95']:.2%}"),
               ("Lowest mAP50–95", f"{worst['map50_95']:.2%}"),
               ("Largest relative drop", f"{change['relative_percent']:.2f}%")])
        st.caption(f"Lowest mAP50–95: {CONDITION_LABELS[worst['condition']]} · source: {path.relative_to(ROOT)}")
        st.subheader("Degradation from normal")
        st.bar_chart(changes, x="Condition", y="Relative degradation (%)", sort=False, height=310)
        st.caption("100 × (normal − condition) / normal. Positive means worse; negative means improvement.")
        st.dataframe([{"Condition": CONDITION_LABELS[row["condition"]],
                       **{label + " (%)": round(row[key] * 100, 2) for key, label in METRICS.items()}}
                      for row in rows], hide_index=True, width="stretch")
        st.subheader("Per-class sensitivity")
        show_figure([ROOT / "outputs/experiments/robustness/figures/per_class_degradation.png",
                     ROOT / "docs/assets/robustness/per_class_degradation.png"], "AP50–95 loss in percentage points versus normal")
        condition = st.selectbox("Inspect a condition", list(CONDITION_LABELS), format_func=CONDITION_LABELS.get)
        st.dataframe(metric_rows([row for row in report["per_class"] if row["condition"] == condition]), hide_index=True, width="stretch")
        with st.expander("Fixed transformation settings"):
            metadata, _ = read_report([ROOT / "outputs/experiments/robustness/run_metadata.json",
                                       ROOT / "docs/results/robustness/run_metadata.json"], ("conditions",))
            st.json(metadata["conditions"])
            st.write("Original dimensions and labels are preserved. Conditions use lossless image copies. These are saved validation results, not experiments on your uploaded photo.")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        st.error(f"Robustness results are unavailable: {error}")


def about_page() -> None:
    heading("A closer look at computer vision.", "FruitVision · Multi-Fruit Detection & Visual Analytics")
    st.markdown("""FruitVision is a learning and portfolio project that connects object detection to evidence:
    detect fruit, explore predictions, inspect a measured baseline, and understand its limitations.

    The dashboard reuses the existing Python pipeline. Inference runs locally with the frozen
    **YOLO11n** baseline. Model Performance and Robustness read saved results; visiting them
    never trains or evaluates a model.
    """)
    st.subheader("Ten classes, one detector")
    st.write(" · ".join(name.title() for name in CLASS_NAMES))
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Interpreting predictions")
        st.write("Counts come from retained detection boxes. Occlusion, small objects, difficult backgrounds, and incomplete annotations can affect results. Confidence is not a guarantee of correctness.")
        st.write(SIZE_NOTICE)
        st.caption("Size is an approximate horizontal box width, not a true diameter. Physical accuracy has not been validated.")
    with right:
        st.subheader("Dataset & attribution")
        st.markdown("""**Fruit, version 1** · Roboflow workspace `simons-workspace-l89qb`,
        project `fruit-smrhb-tp0nb`.

        [Dataset source](https://universe.roboflow.com/simons-workspace-l89qb/fruit-smrhb-tp0nb/dataset/1)
        · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

        Original splits are preserved. Polygon labels were converted to enclosing boxes;
        one identical duplicate training row was removed. Source annotations remain imperfect.
        """)
    st.divider()
    st.subheader("Your session")
    st.write("Uploaded photos and predictions stay in server memory for this session. Download the image or JSON to keep a result. The app does not save uploads to the dataset. A browser reload or server restart can clear your session.")
    st.caption("Designed for local use. No authentication or shared result history is included.")


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">Fv</span><div>FruitVision</div></div>', unsafe_allow_html=True)
    st.radio("Workspace", ["Detect", "Analytics", "Model Performance", "Robustness", "About"], key="page", label_visibility="collapsed")

st.markdown('<div class="workspace-bar"><span>FruitVision</span></div>', unsafe_allow_html=True)
{"Detect": detect_page, "Analytics": analytics_page, "Model Performance": performance_page,
 "Robustness": robustness_page, "About": about_page}[st.session_state["page"]]()
