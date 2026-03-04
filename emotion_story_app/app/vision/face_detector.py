"""MediaPipe face detection wrapper."""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Fast CPU face detector built on MediaPipe."""

    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.5) -> None:
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_faces(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return detected face boxes in (x1, y1, x2, y2) pixel coordinates."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)
        if not results.detections:
            return []

        h, w = frame_bgr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x2 = min(w - 1, x1 + max(bw, 1))
            y2 = min(h - 1, y1 + max(bh, 1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def close(self) -> None:
        self._detector.close()
