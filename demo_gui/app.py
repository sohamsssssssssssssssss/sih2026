"""Phase 0 Streamlit shell for geospatial VQA."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.router import route  # noqa: E402

st.set_page_config(page_title="Geospatial VQA", page_icon="🛰️")
st.title("Multi-agent Geospatial VQA")
uploaded = st.file_uploader("Upload a tile image", type=["png", "jpg", "jpeg", "tif", "tiff"])
question = st.text_input("Question")

if st.button("Ask"):
    if uploaded is None or not question.strip():
        st.error("Upload an image and enter a question.")
    else:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(uploaded.getbuffer())
            image_path = handle.name
        result = route(model_name="mock", image_paths=[image_path], question=question)
        st.subheader("Answer")
        st.write(result["answer"])
        st.metric("Confidence", f"{result['confidence']:.2f}")
        trace = result["trace"]
        with st.expander("Execution trace"):
            st.json(
                {
                    "model_name": trace["model_name"],
                    "timestamp_iso": trace["timestamp_iso"],
                    "record_hash": trace["record_hash"],
                }
            )
