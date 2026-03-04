"""Primary face selection helpers."""

from __future__ import annotations

import math


def compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / max(union, 1)


def choose_primary_face(
    detections: list[tuple[int, int, int, int]],
    prev_box: tuple[int, int, int, int] | None,
    frame_w: int,
    frame_h: int,
    iou_keep_threshold: float,
) -> tuple[int, int, int, int] | None:
    """Pick one primary face using IoU lock and area-center fallback scoring."""
    if not detections:
        return None

    if prev_box is not None:
        best_iou = -1.0
        best_det = None
        for det in detections:
            iou = compute_iou(prev_box, det)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        if best_det is not None and best_iou > iou_keep_threshold:
            return best_det

    frame_area = max(1, frame_w * frame_h)
    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0
    max_center_dist = math.hypot(cx_frame, cy_frame)

    best_score = -1.0
    best_box = detections[0]
    for x1, y1, x2, y2 in detections:
        area = max(1, (x2 - x1) * (y2 - y1))
        norm_area = min(1.0, area / frame_area)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_dist = math.hypot(cx - cx_frame, cy - cy_frame)
        norm_center = min(1.0, center_dist / max_center_dist)

        score = 0.6 * norm_area + 0.4 * (1.0 - norm_center)
        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    return best_box
