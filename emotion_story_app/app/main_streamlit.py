"""Primary Streamlit entrypoint using WebRTC capture and shared worker threads."""

from __future__ import annotations

import atexit
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from app.config import DEFAULT_STORY_GRAPH_PATH
from app.state import SharedState
from app.workers.story_worker import StoryWorker
from app.workers.vision_worker import VisionWorker


def _safe_start_workers() -> None:
    if "shared_state" not in st.session_state:
        st.session_state.shared_state = SharedState()
    if "stop_event" not in st.session_state:
        import threading

        st.session_state.stop_event = threading.Event()
    if "vision_worker" not in st.session_state:
        st.session_state.vision_worker = VisionWorker(st.session_state.shared_state, st.session_state.stop_event)
        st.session_state.vision_worker.start()
    if "story_worker" not in st.session_state:
        st.session_state.story_worker = StoryWorker(
            st.session_state.shared_state,
            st.session_state.stop_event,
            str(DEFAULT_STORY_GRAPH_PATH),
        )
        st.session_state.story_worker.start()


def _stop_workers() -> None:
    stop_event = st.session_state.get("stop_event")
    if stop_event is not None:
        stop_event.set()


atexit.register(_stop_workers)

st.set_page_config(page_title="Emotion-Aware Storytelling", layout="wide")
st.title("🎭 Emotion-Aware Storytelling (CPU MVP)")
_safe_start_workers()
state: SharedState = st.session_state.shared_state


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    ts = time.time()

    with state.lock:
        state.latest_frame = img
        state.latest_frame_ts = ts
        box = state.face_box
        emotion = state.stable_emotion
        conf = state.emotion_conf

    overlay = img.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(
            overlay,
            f"{emotion} ({conf:.2f})",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 220, 50),
            2,
        )

    return av.VideoFrame.from_ndarray(overlay, format="bgr24")


webrtc_streamer(
    key="emotion-story-webrtc",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Live Emotion Status")
    with state.lock:
        face_present = state.face_present
        emotion = state.stable_emotion
        conf = state.emotion_conf
        fps = state.vision_fps_est
        infer_ms = state.last_infer_ms
        node_id = state.story_node_id
        errors = list(state.errors)

    st.metric("Face detected", "Yes" if face_present else "No")
    st.metric("Stable emotion", emotion)
    st.metric("Confidence", f"{conf:.2f}")
    st.metric("Vision FPS (est)", f"{fps:.1f}")
    st.metric("Last inference", f"{infer_ms:.1f} ms")
    st.metric("Current node", node_id)
    if errors:
        st.warning("\n".join(errors[-3:]))

with col2:
    st.subheader("Story Panel")
    with state.lock:
        story_text = state.story_text
    st.write(story_text if story_text else "Waiting for camera frames and story worker...")

st.caption(f"Graph: {Path(DEFAULT_STORY_GRAPH_PATH).name}")
