"""Fallback runner using OpenCV webcam window (no streamlit-webrtc dependency at runtime)."""

from __future__ import annotations

import threading
import time

import cv2

from app.config import DEFAULT_STORY_GRAPH_PATH
from app.state import SharedState
from app.workers.story_worker import StoryWorker
from app.workers.vision_worker import VisionWorker


def run() -> None:
    state = SharedState()
    stop_event = threading.Event()

    vision = VisionWorker(state, stop_event)
    story = StoryWorker(state, stop_event, str(DEFAULT_STORY_GRAPH_PATH))
    vision.start()
    story.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            with state.lock:
                state.latest_frame = frame.copy()
                state.latest_frame_ts = time.time()
                box = state.face_box
                emotion = state.stable_emotion
                conf = state.emotion_conf
                node_id = state.story_node_id
                story_text = state.story_text

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} {conf:.2f}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Node: {node_id}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
            preview_text = (story_text[:70] + "...") if len(story_text) > 70 else story_text
            cv2.putText(frame, preview_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

            cv2.imshow("Emotion-Aware Storytelling", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
