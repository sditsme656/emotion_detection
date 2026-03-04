"""Face ROI extraction utilities."""

from __future__ import annotations

import cv2
import numpy as np


def crop_face_roi(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    target_size: tuple[int, int],
    margin_ratio: float = 0.15,
) -> np.ndarray | None:
    """Crop and resize face ROI with bounds checking and configurable margin."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    rx1 = max(0, x1 - mx)
    ry1 = max(0, y1 - my)
    rx2 = min(w, x2 + mx)
    ry2 = min(h, y2 + my)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
