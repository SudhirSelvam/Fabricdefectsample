"""Real-time fabric defect detection with Streamlit and streamlit-webrtc."""

from __future__ import annotations

import csv
import io
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


CLASS_NAMES = [
    "DropStitch",
    "Hole",
    "HorizontalMark",
    "Knot",
    "OilSpot",
    "Setup",
    "Slub",
    "VerticalLineMark",
    "YarnDust",
]
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
MAX_HISTORY_ROWS = 500
INFERENCE_SIZE = 416
INFERENCE_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[YOLO, str]:
    """Load the trained model once per Streamlit process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. Upload best.pt next to app.py."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))
    model.to(device)
    return model, device


@dataclass
class DetectionState:
    """Thread-safe state shared by the WebRTC callback and the UI."""

    counts: Counter[str] = field(default_factory=Counter)
    total: int = 0
    fps: float = 0.0
    history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_HISTORY_ROWS)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_times: deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "counts": {name: self.counts.get(name, 0) for name in CLASS_NAMES},
                "total": self.total,
                "fps": self.fps,
                "history": list(self.history),
            }

    def reset(self) -> None:
        with self.lock:
            self.counts.clear()
            self.total = 0
            self.fps = 0.0
            self.history.clear()
            self.frame_times.clear()


def update_statistics(
    state: DetectionState, detections: list[dict[str, Any]], timestamp: float
) -> None:
    """Record detections and calculate a rolling frame rate."""
    with state.lock:
        for detection in detections:
            class_name = detection["class_name"]
            state.counts[class_name] += 1
            state.total += 1
            state.history.append(
                {
                    "Time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
                    "Defect": class_name,
                    "Confidence": round(float(detection["confidence"]), 3),
                    "X": detection["x"],
                    "Y": detection["y"],
                    "Width": detection["width"],
                    "Height": detection["height"],
                }
            )

        state.frame_times.append(timestamp)
        if len(state.frame_times) >= 2:
            elapsed = state.frame_times[-1] - state.frame_times[0]
            if elapsed > 0:
                state.fps = (len(state.frame_times) - 1) / elapsed


def draw_detections(
    frame: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    """Draw readable bounding boxes and confidence labels on a BGR frame."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        color = (0, 210, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_height - baseline - 4),
            (x1 + text_width + 8, label_y),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (8, 15, 24),
            2,
            cv2.LINE_AA,
        )
    return annotated


def process_frame(
    frame: av.VideoFrame,
    model: YOLO,
    state: DetectionState,
    confidence_threshold: float,
    device: str,
) -> av.VideoFrame:
    """Run one YOLO inference and return the annotated WebRTC frame."""
    image = frame.to_ndarray(format="bgr24")
    detections: list[dict[str, Any]] = []

    # Ultralytics model inference is serialized because the cached model can be
    # shared by multiple browser sessions in the same Streamlit process.
    with INFERENCE_LOCK:
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            imgsz=INFERENCE_SIZE,
            max_det=30,
            half=device.startswith("cuda"),
            device=device,
            verbose=False,
        )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        coordinates = boxes.xyxy.int().cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.int().cpu().numpy()
        for box, confidence, class_id in zip(
            coordinates, confidences, class_ids
        ):
            class_index = int(class_id)
            class_name = (
                CLASS_NAMES[class_index]
                if 0 <= class_index < len(CLASS_NAMES)
                else f"Class {class_index}"
            )
            x1, y1, x2, y2 = [int(value) for value in box]
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "box": (x1, y1, x2, y2),
                    "x": x1,
                    "y": y1,
                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1),
                }
            )

    timestamp = time.time()
    update_statistics(state, detections, timestamp)
    return av.VideoFrame.from_ndarray(draw_detections(image, detections), format="bgr24")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stMetric"] {
            background: #172033; border: 1px solid #263653;
            padding: 12px; border-radius: 10px;
        }
        .block-container { max-width: 1400px; padding-top: 1.2rem; }
        @media (max-width: 640px) {
            .block-container { padding: .75rem .65rem 2rem; }
            h1 { font-size: 1.65rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(state: DetectionState) -> None:
    snapshot = state.snapshot()
    counts = snapshot["counts"]
    history = snapshot["history"]
    st.subheader("Live statistics")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total defects", snapshot["total"])
    metric_columns[1].metric(
        "Most frequent",
        max(counts, key=counts.get) if snapshot["total"] else "—",
    )
    metric_columns[2].metric("Current FPS", f"{snapshot['fps']:.1f}")
    metric_columns[3].metric(
        "Defect types", sum(value > 0 for value in counts.values())
    )

    left, right = st.columns((1, 1), gap="medium")
    with left:
        chart_data = pd.DataFrame(
            {"Defect": list(counts.keys()), "Detections": list(counts.values())}
        )
        chart = px.bar(
            chart_data,
            x="Detections",
            y="Defect",
            orientation="h",
            template="plotly_dark",
            color="Detections",
            color_continuous_scale="Turbo",
        )
        chart.update_layout(
            height=390, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False
        )
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Detection history")
        history_df = pd.DataFrame(history)
        if history_df.empty:
            st.info("Detections will appear here as the camera finds defects.")
        else:
            st.dataframe(history_df.iloc[::-1], use_container_width=True, height=320)
            csv_buffer = io.StringIO()
            history_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_MINIMAL)
            st.download_button(
                "Download detection history (CSV)",
                csv_buffer.getvalue(),
                file_name="fabric_defect_history.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="Fabric Defect Detection",
        page_icon="🧵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    st.title("🧵 Fabric Defect Detection")
    st.caption("Real-time YOLO inspection from your device camera")

    try:
        model, device = load_model()
        model_status = "Ready"
    except (FileNotFoundError, RuntimeError, OSError) as error:
        st.error(f"Unable to load the detection model: {error}")
        st.stop()

    if "detection_state" not in st.session_state:
        st.session_state.detection_state = DetectionState()
    state: DetectionState = st.session_state.detection_state

    with st.sidebar:
        st.header("Controls")
        st.success(f"Model: {model_status}")
        st.info(f"Device: {'GPU (CUDA)' if device.startswith('cuda') else 'CPU'}")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            help="Only predictions at or above this confidence are shown.",
        )
        if st.button("Reset statistics", use_container_width=True):
            state.reset()
            st.rerun()

        st.markdown("---")
        st.markdown(
            "Allow camera access when your browser asks. Camera access requires "
            "HTTPS (provided automatically by Streamlit Community Cloud)."
        )

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        return process_frame(
            frame, model, state, confidence_threshold, device
        )

    try:
        stream_context = webrtc_streamer(
            key="fabric-defect-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=video_callback,
            async_processing=True,
            video_receiver_size=1,
            media_stream_constraints={
                # Keep constraints intentionally broad for desktop, Android,
                # and iOS browser compatibility.
                "video": True,
                "audio": False,
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "playsInline": True,
                "muted": True,
            },
        )
        if not stream_context.state.playing:
            st.info(
                "Click **Start** in the camera panel, then allow camera access "
                "in your browser."
            )
    except Exception as error:
        st.error(
            "The camera stream could not start. Check browser camera permissions, "
            f"HTTPS, and firewall settings. Details: {error}"
        )

    # Fragments keep metrics current without interrupting the WebRTC component.
    if hasattr(st, "fragment"):
        @st.fragment(run_every="1s")
        def live_dashboard() -> None:
            render_dashboard(state)

        live_dashboard()
    else:
        render_dashboard(state)


if __name__ == "__main__":
    main()
