"""Offline-safe Streamlit demo for geospatial VQA evidence and robustness."""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The gate demo must never depend on Wi-Fi. A cached local model may still run.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sih26167-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "sih26167-cache"))

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_gui import golden_assets  # noqa: E402
from orchestrator.registry import get  # noqa: E402
from orchestrator.router import route  # noqa: E402
from orchestrator.trace import append_record, verify_chain  # noqa: E402

MODEL_NAME = "qwen2.5vl-3b"
RESULTS_PATH = ROOT / "results" / "qwen2.5vl-3b__ladder__rescored__20260904.json"
SAR_IMAGE_PATH = ROOT / "data" / "sar_gate" / "rendered" / "mumbai_coastal.png"
SAR_ANNOTATION_PATH = ROOT / "data" / "sar_gate" / "annotation_template.md"
AVAILABLE_CAPABILITIES = (
    "Single-image VQA",
    "Resolution robustness evaluation",
    "Execution audit trace",
    "SAR preprocessing / analyst validation",
)
IN_DEVELOPMENT_CAPABILITIES = (
    "Grounding",
    "Bi-temporal Change-VQA",
    "Optical-SAR fusion",
    "RS fine-tuning",
)


@st.cache_data
def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_mumbai_interpretation(path: Path) -> str:
    document = path.read_text(encoding="utf-8")
    try:
        section = document.split("## Mumbai coastal", 1)[1].split(
            "## Maharashtra farmland", 1
        )[0]
    except IndexError as exc:
        raise ValueError(
            "The Mumbai coastal annotation section is missing or incomplete."
        ) from exc
    lines = [line for line in section.strip().splitlines() if not line.startswith("![")]
    return "\n".join(lines).strip()


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
        "execution_mode": params.get("execution_mode"),
        "results_artifact": params.get("results_artifact"),
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


def clear_last_response() -> None:
    """Discard answer/provenance state when the displayed query source changes."""
    st.session_state.pop("last_response", None)
    st.session_state.pop("last_notice", None)
    st.session_state.pop("last_notice_error", None)


def render_trace_verification() -> None:
    """Render the unchanged audit-chain action in its current visual context."""
    if st.button("Verify trace"):
        verified, message = verify_chain()
        if verified:
            st.success("✓ Chain verified")
            st.caption(message)
        else:
            st.error(f"Trace verification failed: {message}")


st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    :root { --sat-accent: #35c6d0; --sat-surface: #0e1b28; --sat-border: #23384a; }
    .stApp { background-color: #07111b; }
    [data-testid="stHeader"] { background-color: rgba(7, 17, 27, 0.92); }
    [data-testid="stMainBlockContainer"] { padding-top: 2.25rem; padding-bottom: 3rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--sat-border);
        border-radius: 9px;
        background-color: rgba(14, 27, 40, 0.58);
    }
    .st-key-capability-status { border-left: 3px solid var(--sat-accent); }
    .st-key-answer-value p {
        color: #f4fbfc;
        font-size: clamp(2.25rem, 5vw, 4rem);
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.05;
        margin: 0.15rem 0 0.45rem;
    }
    .sat-status-row { display: flex; justify-content: flex-end; gap: 0.45rem; flex-wrap: wrap; }
    .sat-chip {
        border: 1px solid #2b8790;
        border-radius: 999px;
        color: #a8f0f3;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.65rem;
        white-space: nowrap;
    }
    [data-testid="stCode"] { font-size: 0.78rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--sat-border); }
    .stTabs [data-baseweb="tab"] { font-weight: 650; letter-spacing: 0.015em; }
    .stButton > button[kind="primary"] { font-weight: 750; letter-spacing: 0.035em; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
        .sat-status-row { justify-content: flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

brand_column, system_column = st.columns([4, 2], vertical_alignment="center")
with brand_column:
    st.title("SATQUERY AI")
    st.caption("Evidence-backed geospatial intelligence")
with system_column:
    st.markdown(
        '<div class="sat-status-row"><span class="sat-chip">OFFLINE READY</span>'
        '<span class="sat-chip">AUDITABLE</span></div>',
        unsafe_allow_html=True,
    )

with st.container(border=True, key="capability-status"):
    st.caption("CAPABILITY STATUS")
    available_column, development_column = st.columns(2)
    with available_column:
        st.markdown("**AVAILABLE NOW**")
        st.caption(" · ".join(AVAILABLE_CAPABILITIES))
    with development_column:
        st.markdown("**IN DEVELOPMENT**")
        st.caption(" · ".join(IN_DEVELOPMENT_CAPABILITIES))

try:
    ladder_report = load_results(RESULTS_PATH)
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Required ladder artifact could not be loaded: {exc}")
    st.stop()

ask_tab, robustness_tab, sar_tab = st.tabs(
    ["Ask SatQuery", "Resolution Robustness", "SAR Validation"]
)

with ask_tab:
    source_mode = st.radio(
        "Query source",
        ["Verified golden scene", "Upload a scene"],
        horizontal=True,
        on_change=clear_last_response,
    )
    input_column, answer_column = st.columns([3, 2], gap="large")
    with input_column:
        temporary_path: Path | None = None
        if source_mode == "Verified golden scene":
            cached_golden = golden_assets.golden_result(ladder_report)
            scene_id = cached_golden["tile_id"]
            sensor = "LoveDA"
            question = answer_column.text_input("Question", value=cached_golden["question"], disabled=True)
            gsd_label = f"{cached_golden['gsd']} m"
            image_path = golden_assets.local_golden_image(scene_id, ROOT)
            if image_path is not None:
                st.image(str(image_path), caption=f"Golden scene: {scene_id}", width="stretch")
            else:
                st.info("Local golden pixels are absent; the committed cached result remains available.")
        else:
            uploaded = st.file_uploader(
                "Upload a tile image", type=["png", "jpg", "jpeg", "tif", "tiff"]
            )
            if uploaded is not None:
                st.image(uploaded.getvalue(), caption=f"Uploaded scene: {uploaded.name}", width="stretch")
            question = answer_column.text_input("Question")
            sensor_choice = st.selectbox("Sensor/source", ["Unspecified", "LoveDA", "SAR"])
            sensor = "" if sensor_choice == "Unspecified" else sensor_choice
            scene_id = uploaded.name if uploaded is not None else ""
            image_path = None
            gsd_label = "unknown"
        st.caption("Scene ID")
        st.code(scene_id or "No scene uploaded", language=None)
        st.caption(f"Sensor: {sensor or 'Unspecified'}")
        st.caption(f"GSD: {gsd_label}")

    with answer_column:
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
        params = response.get("trace", {}).get("params", {}) if response else {}
        execution_mode = params.get("execution_mode")
        if notice:
            if st.session_state.get("last_notice_error"):
                st.error(notice)
            elif execution_mode == "cached_result":
                st.warning(notice)
            elif execution_mode == "live":
                st.success(notice)
            else:
                st.error(notice)
        if response:
            st.subheader("Answer")
            if execution_mode == "live":
                st.success("LIVE INFERENCE")
            elif execution_mode == "cached_result":
                st.info("VERIFIED CACHED RESULT")
                if params.get("results_artifact"):
                    st.caption("Committed results artifact")
                    st.code(params["results_artifact"], language=None)
                else:
                    st.error("Cached result is missing its results_artifact provenance.")
            else:
                st.error(f"Missing or unrecognized execution_mode in response trace: {execution_mode!r}")
            st.caption(f"Model: {MODEL_NAME}")
            with st.container(key="answer-value"):
                st.write(response["answer"])
            st.caption("Confidence calibration pending")

    if response:
        st.subheader("Evidence and execution trace")
        evidence = evidence_fields(response["trace"])

        record_hash = evidence.get("record_hash")
        prev_hash = evidence.get("prev_hash")

        def hash_presentation(full_hash: str | None) -> tuple[str, str]:
            if full_hash is None:
                return "Not recorded", "No hash value was recorded."
            if full_hash == "":
                return "First record", 'Full value: "" (empty string; no previous record).'
            display = (
                f"{full_hash[:8]}...{full_hash[-8:]}"
                if len(full_hash) > 16 else full_hash
            )
            return display, f"Full value: {full_hash}"

        record_display, record_help = hash_presentation(record_hash)
        prev_display, prev_help = hash_presentation(prev_hash)
        identity_card, execution_card, question_card, integrity_card = st.columns(
            4, gap="medium"
        )
        with identity_card.container(border=True):
            st.caption("Identity")
            st.caption("Scene ID")
            st.code(evidence.get("scene_id", "Not recorded"), language=None)
            st.metric("Sensor", evidence.get("sensor", "Not recorded"))
        with execution_card.container(border=True):
            st.caption("Execution")
            st.metric("Model", evidence.get("model_name", "Not recorded"))
            st.caption("Model version")
            st.code(evidence.get("model_version", "Not recorded"), language=None)
            st.metric("Execution mode", evidence.get("execution_mode", "Not recorded"))
            st.caption("Timestamp")
            st.code(evidence.get("timestamp", "Not recorded"), language=None)
            if "results_artifact" in evidence:
                st.caption("Results artifact")
                st.code(evidence["results_artifact"], language=None)
        with question_card.container(border=True):
            st.caption("Question")
            st.text(evidence.get("question", "Not recorded"))
        with integrity_card.container(border=True):
            st.caption("Integrity — full hashes available from help icons")
            st.metric("Record hash", record_display, help=record_help)
            st.metric("Previous hash", prev_display, help=prev_help)
            render_trace_verification()

        with st.expander("Raw evidence JSON", expanded=False):
            st.json(evidence)
    else:
        render_trace_verification()

    st.caption(
        "Every model/tool invocation is chained to the previous execution record. "
        "Altering an earlier record invalidates verification."
    )

with robustness_tab:
    st.header("Resolution robustness")
    st.caption(
        f"Model: {ladder_report['model']} · Samples: {ladder_report['n_samples']} · "
        f"Run: {ladder_report['timestamp']}"
    )
    rung_items = sorted(
        ladder_report["per_rung"].items(),
        key=lambda item: float(item[0]),
    )
    degenerate_rungs = set(ladder_report["degenerate_rungs"])
    gsds = [float(gsd) for gsd, _ in rung_items]
    open_accuracies = [metrics["open_accuracy"] for _, metrics in rung_items]
    accuracies = [metrics["accuracy"] for _, metrics in rung_items]
    st.caption(
        "Aggregate accuracy can look healthy even when the model collapses to one binary "
        "answer class. SatQuery explicitly detects and flags this failure mode."
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    figure.patch.set_facecolor("#07111b")
    axis.set_facecolor("#0e1b28")
    axis.plot(
        gsds, open_accuracies, marker="o", linewidth=3, color="#35c6d0",
        label="Open-question accuracy",
    )
    axis.plot(
        gsds, accuracies, linestyle="--", linewidth=1.2, color="#8da2b5",
        label="Aggregate accuracy (answer-collapse sensitive)",
    )
    flagged = [(float(gsd), metrics) for gsd, metrics in rung_items if gsd in degenerate_rungs]
    if flagged:
        axis.scatter(
            [gsd for gsd, _ in flagged],
            [metrics["open_accuracy"] for _, metrics in flagged],
            marker="X", s=90, color="#f0aa3c", zorder=3,
            label="Flagged degenerate rung",
        )
        axis.scatter(
            [gsd for gsd, _ in flagged],
            [metrics["accuracy"] for _, metrics in flagged],
            marker="X", s=90, color="#f0aa3c", zorder=3,
        )
    axis.set_xlabel("Ground sample distance (m)")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(
        gsds,
        [f"{float(gsd):g}{'*' if gsd in degenerate_rungs else ''}" for gsd, _ in rung_items],
    )
    axis.set_ylim(0, 1)
    axis.tick_params(colors="#c8d5df")
    axis.xaxis.label.set_color("#c8d5df")
    axis.yaxis.label.set_color("#c8d5df")
    for spine in axis.spines.values():
        spine.set_color("#2a4052")
    axis.grid(color="#2a4052", alpha=0.55)
    legend = axis.legend(loc="upper right", fontsize=8)
    legend.get_frame().set_facecolor("#0e1b28")
    legend.get_frame().set_edgecolor("#2a4052")
    for label in legend.get_texts():
        label.set_color("#dce7ee")
    st.pyplot(figure)
    plt.close(figure)
    if flagged:
        st.caption(
            "* and orange X markers identify degenerate rungs flagged in the committed artifact."
        )
    st.dataframe(
        [
            {
                "GSD (m)": float(gsd),
                "Status": "⚠ Degenerate" if gsd in degenerate_rungs else "Not flagged",
                "Open-question accuracy": metrics["open_accuracy"],
                "Accuracy": metrics["accuracy"],
                "Binary predicted-yes rate": metrics["pred_yes_rate_on_binary"],
            }
            for gsd, metrics in rung_items
        ],
        column_config={
            "Open-question accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn("Aggregate accuracy", format="%.4f"),
            "Binary predicted-yes rate": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        width="stretch",
    )
    for rung in ladder_report.get("degenerate_rungs", []):
        warning = ladder_report["per_rung"].get(rung, {}).get("warning")
        if warning:
            st.warning(warning)

with sar_tab:
    st.header("Mumbai coastal SAR")
    st.info("HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT")
    st.caption("Optical–SAR fusion is in development; this tab shows analyst validation only.")
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
        annotation = load_mumbai_interpretation(SAR_ANNOTATION_PATH)
    except (OSError, ValueError) as exc:
        st.error(f"Analyst interpretation could not be loaded: {exc}")
    else:
        categories = (
            ("Water areas:", "Water"),
            ("Urban/built-up:", "Built-up"),
            ("Vegetation:", "Vegetation"),
            ("Terrain artifacts (layover/foreshortening/shadow):", "Terrain"),
        )
        try:
            lines = annotation.splitlines()
            headings = [heading for heading, _ in categories] + ["Why it looks this way:"]
            for heading in headings:
                if lines.count(heading) != 1:
                    raise ValueError(f"Expected one '{heading}' heading.")
            positions = [lines.index(heading) for heading in headings]
            if positions != sorted(positions):
                raise ValueError("Category headings are out of order.")
            summaries = []
            for index, (_, label) in enumerate(categories):
                bucket = lines[positions[index] + 1:positions[index + 1]]
                identity_lines = [
                    line for line in bucket
                    if line.lstrip().startswith(("- Region:", "Class:"))
                ]
                if identity_lines:
                    excerpt = "\n".join(identity_lines)
                else:
                    first_sentence, period, _ = "\n".join(bucket).strip().partition(".")
                    excerpt = first_sentence + period
                if not excerpt.strip():
                    raise ValueError(f"The {label} category is empty.")
                summaries.append((label, excerpt))
        except ValueError as exc:
            st.error(f"Category summaries unavailable: {exc} View the full annotation below.")
        else:
            summary_columns = st.columns(4, gap="medium")
            for column, (label, excerpt) in zip(summary_columns, summaries):
                with column.container(border=True):
                    st.subheader(label)
                    st.markdown(excerpt)
        with st.expander("View full analyst annotation"):
            st.markdown(annotation)
