"""Utility functions for image preprocessing and temporal smoothing."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Tuple

import cv2
import numpy as np
import torch


def preprocess_face(face_bgr: np.ndarray, image_size: int = 48) -> torch.Tensor:
    """Preprocess BGR face ROI to normalized tensor shape [1, 1, H, W]."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


class RollingEmotionSmoother:
    """Smooth noisy per-frame predictions with rolling averaging."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.history: Deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, probs: Iterable[float]) -> Tuple[int, float, np.ndarray]:
        probs_arr = np.asarray(list(probs), dtype=np.float32)
        self.history.append(probs_arr)
        avg_probs = np.mean(np.stack(self.history, axis=0), axis=0)
        idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[idx])
        return idx, conf, avg_probs

    def reset(self) -> None:
        self.history.clear()
