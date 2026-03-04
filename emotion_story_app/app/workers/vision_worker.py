"""Vision worker thread: face detect + emotion inference + smoothing."""

from __future__ import annotations

import threading
import time

import numpy as np

from app.config import (
    EMOTION_FPS,
    FACE_DETECT_EVERY_N,
    HOLD_LAST_EMOTION_ON_NO_FACE,
    IOU_KEEP_THRESHOLD,
    MIN_CONF,
    MODEL_INPUT_SIZE,
    NO_FACE_TIMEOUT,
    STABLE_FRAMES,
    WINDOW_SECONDS,
)
from app.state import SharedState
from app.utils.logger import configure_logger
from app.utils.timing import PeriodicTimer
from app.vision.emotion_model import EmotionModel
from app.vision.face_detector import FaceDetector
from app.vision.primary_face import choose_primary_face
from app.vision.roi import crop_face_roi
from app.vision.smoother import EmotionSmoother


class VisionWorker(threading.Thread):
    """Background worker that updates face box and stable emotion."""

    def __init__(self, state: SharedState, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = stop_event
        self.logger = configure_logger("vision_worker")

        self.detector = FaceDetector()
        self.model = EmotionModel()
        self.smoother = EmotionSmoother(WINDOW_SECONDS, STABLE_FRAMES, MIN_CONF)

        self._last_face_seen_ts = 0.0
        self._prev_box: tuple[int, int, int, int] | None = None
        self._detect_tick_count = 0
        self._latest_detections: list[tuple[int, int, int, int]] = []

    def run(self) -> None:
        timer = PeriodicTimer(1.0 / EMOTION_FPS)
        last_tick_ts = time.monotonic()
        while not self.stop_event.is_set():
            start = time.monotonic()

            with self.state.lock:
                frame = None if self.state.latest_frame is None else self.state.latest_frame.copy()
                frame_ts = self.state.latest_frame_ts

            if frame is not None:
                self._process_frame(frame, frame_ts)

            infer_ms = (time.monotonic() - start) * 1000.0
            now = time.monotonic()
            fps = 1.0 / max(1e-6, now - last_tick_ts)
            last_tick_ts = now
            with self.state.lock:
                self.state.last_infer_ms = infer_ms
                self.state.vision_fps_est = fps

            timer.wait_next()

        self.detector.close()

    def _process_frame(self, frame: np.ndarray, frame_ts: float) -> None:
        h, w = frame.shape[:2]
        self._detect_tick_count += 1

        run_detect = (self._detect_tick_count % FACE_DETECT_EVERY_N) == 0 or self._prev_box is None
        if run_detect:
            self._latest_detections = self.detector.detect_faces(frame)

        primary = choose_primary_face(
            self._latest_detections,
            self._prev_box,
            w,
            h,
            IOU_KEEP_THRESHOLD,
        )

        if primary is None:
            no_face_duration = time.time() - self._last_face_seen_ts
            if no_face_duration >= NO_FACE_TIMEOUT:
                with self.state.lock:
                    was_present = self.state.face_present
                    self.state.face_box = None
                    self.state.face_present = False
                    if not HOLD_LAST_EMOTION_ON_NO_FACE:
                        self.state.stable_emotion = "neutral"
                        self.state.emotion_conf = 0.0
                if was_present:
                    self.logger.info("Face lost")
            return

        with self.state.lock:
            was_present = self.state.face_present
        if not was_present:
            self.logger.info("Face detected")

        self._last_face_seen_ts = time.time()
        self._prev_box = primary
        roi = crop_face_roi(frame, primary, MODEL_INPUT_SIZE)
        probs = self.model.predict_proba(roi) if roi is not None else {"neutral": 1.0}
        prev_emotion = self.smoother.current_emotion
        emotion, conf, _ = self.smoother.update(probs, frame_ts)

        if emotion != prev_emotion:
            self.logger.info("Emotion switched: %s -> %s (%.2f)", prev_emotion, emotion, conf)

        with self.state.lock:
            self.state.face_box = primary
            self.state.face_present = True
            self.state.stable_emotion = emotion
            self.state.emotion_conf = conf
