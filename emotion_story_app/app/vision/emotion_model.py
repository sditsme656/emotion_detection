"""Emotion model wrapper (FER preferred, DeepFace optional fallback)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.config import CANONICAL_EMOTIONS, USE_DEEPFACE

FER_TO_CANONICAL = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

DEEPFACE_TO_CANONICAL = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


class EmotionModel:
    """Wrapper exposing predict_proba(face_roi) -> canonical emotion probabilities."""

    def __init__(self, use_deepface: bool = USE_DEEPFACE) -> None:
        self.use_deepface = use_deepface
        self._backend: str = "none"
        self._model: Any = None

        if not use_deepface:
            try:
                from fer import FER

                self._model = FER(mtcnn=False)
                self._backend = "fer"
                return
            except Exception:
                self._backend = "none"

        if use_deepface:
            try:
                from deepface import DeepFace  # type: ignore

                self._model = DeepFace
                self._backend = "deepface"
                return
            except Exception:
                self._backend = "none"

    @property
    def backend(self) -> str:
        return self._backend

    def _blank_probs(self) -> dict[str, float]:
        base = {label: 0.0 for label in CANONICAL_EMOTIONS}
        base["neutral"] = 1.0
        return base

    def _normalize(self, probs: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
        canonical = {label: 0.0 for label in CANONICAL_EMOTIONS}
        for src, value in probs.items():
            dst = mapping.get(src.lower())
            if dst in canonical:
                canonical[dst] += float(value)

        total = sum(canonical.values())
        if total <= 1e-8:
            return self._blank_probs()
        return {k: v / total for k, v in canonical.items()}

    def predict_proba(self, face_roi_bgr: np.ndarray) -> dict[str, float]:
        """Predict canonical emotion probabilities from a cropped face ROI."""
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return self._blank_probs()

        if self._backend == "fer":
            rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
            result = self._model.detect_emotions(rgb)
            if not result:
                return self._blank_probs()
            return self._normalize(result[0].get("emotions", {}), FER_TO_CANONICAL)

        if self._backend == "deepface":
            res = self._model.analyze(
                img_path=face_roi_bgr,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            if isinstance(res, list):
                res = res[0]
            emotions = res.get("emotion", {})
            percent = {k: float(v) / 100.0 for k, v in emotions.items()}
            return self._normalize(percent, DEEPFACE_TO_CANONICAL)

        return self._blank_probs()
