"""Offline-safe Streamlit demo for geospatial VQA evidence and robustness."""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The gate demo must never depend on Wi-Fi. A cached local model may still run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sih26167-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "sih26167-cache"))

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.registry import get  # noqa: E402
from orchestrator.router import route  # noqa: E402
from orchestrator.trace import append_record, verify_chain  # noqa: E402

MODEL_NAME = "qwen2.5vl-3b"
RESULTS_PATH = ROOT / "results" / "qwen2.5vl-3b__ladder__rescored__20260904.json"
SAR_IMAGE_PATH = ROOT / "data" / "sar_gate" / "rendered" / "mumbai_coastal.png"
SAR_ANNOTATION_PATH = ROOT / "data" / "sar_gate" / "annotation_template.md"
GOLDEN_QUESTION = "Is there a building in this image?"


@st.cache_data
def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_mumbai_interpretation(path: Path) -> str:
    document = path.read_text(encoding="utf-8")
    section = document.split("## Mumbai coastal", 1)[1].split(
        "## Maharashtra farmland", 1
    )[0]
    lines = [line for line in section.strip().splitlines() if not line.startswith("![")]
    return "\n".join(lines).strip()


def golden_result(report: dict[str, Any]) -> dict[str, Any]:
    for row in report["results"]:
        if (
            float(row["gsd"]) == 0.3
            and row["question"] == GOLDEN_QUESTION
            and bool(row["correct"])
        ):
            return row
    raise ValueError("The committed ladder artifact has no correct 0.3 m golden query")


def normalize_scene_id(scene_id: str) -> str:
    if Path(scene_id).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        scene_id = str(Path(scene_id).with_suffix(""))
    return scene_id.replace(
        "loveda_Train_Rural_images_png_", "loveda_LoveDA_images_png_"
    )


def find_cached_result(
    report: dict[str, Any], scene_id: str, question: str
) -> dict[str, Any] | None:
    normalized = normalize_scene_id(scene_id)
    return next(
        (
            row
            for row in report["results"]
            if row["tile_id"] == normalized and row["question"] == question
        ),
        None,
    )


def local_golden_image(scene_id: str) -> Path | None:
    local_id = scene_id.replace(
        "loveda_LoveDA_images_png_", "loveda_Train_Rural_images_png_"
    )
    gsd = scene_id.rsplit("_gsd", 1)[-1]
    candidate = ROOT / "data" / "ladder" / gsd / f"{local_id}.png"
    return candidate if candidate.is_file() else None


def cached_response(
    cached: dict[str, Any], sensor: str, failure_reason: str
) -> tuple[dict[str, Any], str]:
    model = get(MODEL_NAME)
    trace = append_record(
        {
            "model_name": MODEL_NAME,
            "model_version": model.version,
            "params": {
                "execution_mode": "cached_result",
                "results_artifact": str(RESULTS_PATH.relative_to(ROOT)),
                "scene_id": cached["tile_id"],
                "sensor": sensor,
            },
            "input_summary": {
                "image_paths": cached.get("image_paths", []),
                "question": cached["question"],
                "n_images": len(cached.get("image_paths", [])),
            },
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        }
    )
    response = {
        "answer": cached["prediction"]["answer"],
        "evidence": cached["prediction"].get("evidence", []),
        "trace": trace,
    }
    notice = (
        f"Live Qwen inference unavailable ({failure_reason}) — showing cached result "
        "for this scene."
    )
    return response, notice


def execute_query(
    image_path: Path | None,
    scene_id: str,
    sensor: str,
    question: str,
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, bool]:
    cached = find_cached_result(report, scene_id, question)
    if image_path is None:
        if cached is None:
            return None, "No local image or verified cached result matches this query.", True
        response, notice = cached_response(cached, sensor, "local scene pixels unavailable")
        return response, notice, False
    try:
        response = route(
            model_name=MODEL_NAME,
            image_paths=[str(image_path)],
            question=question,
            params={"scene_id": scene_id, "sensor": sensor, "execution_mode": "live"},
        )
        return response, "Live Qwen2.5-VL-3B inference completed.", False
    except (RuntimeError, OSError) as exc:
        if cached is None:
            return (
                None,
                f"Live Qwen inference unavailable ({exc}). No verified cached result "
                "matches this scene and question.",
                True,
            )
        reason = "no GPU" if "CUDA GPU" in str(exc) else str(exc)
        response, notice = cached_response(cached, sensor, reason)
        return response, notice, False


def evidence_fields(trace: dict[str, Any]) -> dict[str, Any]:
    params = trace.get("params", {})
    input_summary = trace.get("input_summary", {})
    fields = {
        "question": input_summary.get("question"),
        "scene_id": params.get("scene_id"),
        "sensor": params.get("sensor"),
        "model_name": trace.get("model_name"),
        "model_version": trace.get("model_version"),
        "timestamp": trace.get("timestamp_iso"),
        "record_hash": trace.get("record_hash"),
        "prev_hash": trace.get("prev_hash"),
    }
    return {
        key: value
        for key, value in fields.items()
        if value is not None and (value != "" or key == "prev_hash")
    }


st.set_page_config(page_title="Geospatial VQA", page_icon="🛰️", layout="wide")
st.title("Multi-sensor Geospatial VQA")
st.caption("Real Qwen inference when locally available; verified cached evidence otherwise.")

try:
    ladder_report = load_results(RESULTS_PATH)
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Required ladder artifact could not be loaded: {exc}")
    st.stop()

ask_tab, robustness_tab, sar_tab = st.tabs(
    ["Ask Qwen", "Resolution robustness", "SAR interpretation"]
)

with ask_tab:
    source_mode = st.radio(
        "Query source",
        ["Verified golden scene", "Upload a scene"],
        horizontal=True,
    )
    temporary_path: Path | None = None
    if source_mode == "Verified golden scene":
        cached_golden = golden_result(ladder_report)
        scene_id = cached_golden["tile_id"]
        sensor = "LoveDA"
        question = st.text_input("Question", value=cached_golden["question"], disabled=True)
        image_path = local_golden_image(scene_id)
        st.code(scene_id, language=None)
        if image_path is not None:
            st.image(str(image_path), caption=f"Golden scene: {scene_id}", width=520)
        else:
            st.info("Local golden pixels are absent; the committed cached result remains available.")
    else:
        uploaded = st.file_uploader(
            "Upload a tile image", type=["png", "jpg", "jpeg", "tif", "tiff"]
        )
        question = st.text_input("Question")
        sensor_choice = st.selectbox("Sensor/source", ["Unspecified", "LoveDA", "SAR"])
        sensor = "" if sensor_choice == "Unspecified" else sensor_choice
        scene_id = uploaded.name if uploaded is not None else ""
        image_path = None

    if st.button("Ask", type="primary"):
        st.session_state.pop("last_response", None)
        st.session_state.pop("last_notice", None)
        st.session_state.pop("last_notice_error", None)
        if not question.strip():
            st.error("Enter a question.")
        elif source_mode == "Upload a scene" and uploaded is None:
            st.error("Upload an image.")
        else:
            try:
                if source_mode == "Upload a scene" and uploaded is not None:
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                        handle.write(uploaded.getbuffer())
                        temporary_path = Path(handle.name)
                    image_path = temporary_path
                response, notice, is_error = execute_query(
                    image_path, scene_id, sensor, question, ladder_report
                )
                st.session_state["last_response"] = response
                st.session_state["last_notice"] = notice
                st.session_state["last_notice_error"] = is_error
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    response = st.session_state.get("last_response")
    notice = st.session_state.get("last_notice")
    if notice:
        if st.session_state.get("last_notice_error"):
            st.error(notice)
        elif notice.startswith("Live Qwen inference unavailable"):
            st.warning(notice)
        else:
            st.success(notice)
    if response:
        st.subheader("Answer")
        st.write(response["answer"])
        st.caption("Confidence calibration pending")
        with st.expander("Evidence and execution trace"):
            st.json(evidence_fields(response["trace"]))

    if st.button("Verify trace"):
        verified, message = verify_chain()
        if verified:
            st.success("✓ Chain verified")
            st.caption(message)
        else:
            st.error(f"Trace verification failed: {message}")

with robustness_tab:
    st.header("Resolution robustness")
    st.caption(
        f"Model: {ladder_report['model']} · Samples: {ladder_report['n_samples']} · "
        f"Run: {ladder_report['timestamp']}"
    )
    rung_items = sorted(
        ((float(gsd), metrics) for gsd, metrics in ladder_report["per_rung"].items()),
        key=lambda item: item[0],
    )
    gsds = [gsd for gsd, _ in rung_items]
    accuracies = [metrics["accuracy"] for _, metrics in rung_items]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(gsds, accuracies, marker="o", linewidth=2)
    axis.set_xlabel("Ground sample distance (m)")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(gsds)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    st.pyplot(figure)
    plt.close(figure)
    st.dataframe(
        [
            {
                "GSD (m)": gsd,
                "Accuracy": metrics["accuracy"],
                "Binary predicted-yes rate": metrics["pred_yes_rate_on_binary"],
            }
            for gsd, metrics in rung_items
        ],
        hide_index=True,
        width="stretch",
    )
    for rung in ladder_report.get("degenerate_rungs", []):
        warning = ladder_report["per_rung"].get(rung, {}).get("warning")
        if warning:
            st.warning(warning)

with sar_tab:
    st.header("Mumbai coastal SAR")
    if SAR_IMAGE_PATH.is_file():
        st.image(
            str(SAR_IMAGE_PATH),
            caption="Processed Sentinel-1 SAR: Mumbai coastal",
            width="stretch",
        )
    else:
        st.info("The local processed Mumbai SAR render is unavailable on this machine.")
    st.subheader("Analyst interpretation")
    try:
        st.markdown(load_mumbai_interpretation(SAR_ANNOTATION_PATH))
    except (OSError, IndexError) as exc:
        st.error(f"Analyst interpretation could not be loaded: {exc}")
