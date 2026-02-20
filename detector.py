"""Face detection module using MediaPipe with OpenCV fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    score: float


class FaceDetector:
    """Detect faces and return cropped ROIs from frames."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._backend = "opencv"
        self._mp_detector = None
        self._cascade = None

        try:
            import mediapipe as mp

            self._backend = "mediapipe"
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        except Exception:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        detections: List[FaceDetection] = []

        if self._backend == "mediapipe" and self._mp_detector is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_detector.process(frame_rgb)
            if results.detections:
                for det in results.detections:
                    score = float(det.score[0])
                    if score < self.min_confidence:
                        continue
                    box = det.location_data.relative_bounding_box
                    x = max(0, int(box.xmin * w))
                    y = max(0, int(box.ymin * h))
                    bw = int(box.width * w)
                    bh = int(box.height * h)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    detections.append(FaceDetection((x, y, bw, bh), score))
            return detections

        if self._cascade is None:
            return detections

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        for (x, y, bw, bh) in faces:
            detections.append(FaceDetection((int(x), int(y), int(bw), int(bh)), 1.0))
        return detections

    @staticmethod
    def crop_face(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 8) -> np.ndarray:
        x, y, w, h = bbox
        height, width = frame_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        return frame_bgr[y1:y2, x1:x2]
