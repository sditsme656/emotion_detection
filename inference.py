"""Real-time multi-stage emotion detection inference pipeline."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import torch

from detector import FaceDetector
from model import EMOTION_CLASSES, load_model
from utils import RollingEmotionSmoother, preprocess_face


@dataclass
class FramePacket:
    frame: any
    timestamp: float


@dataclass
class TrackState:
    """Stores smoothing state for an individual face track."""

    bbox: Tuple[int, int, int, int]
    last_seen: int
    smoother: RollingEmotionSmoother


class ThreadedCamera:
    """Asynchronous camera reader that always keeps the latest frame."""

    def __init__(self, src: int = 0, max_buffer: int = 2) -> None:
        self.capture = cv2.VideoCapture(src)
        self.buffer: queue.Queue[FramePacket] = queue.Queue(maxsize=max_buffer)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCamera":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            packet = FramePacket(frame=frame, timestamp=time.time())
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
            try:
                self.buffer.put_nowait(packet)
            except queue.Full:
                pass

    def read(self, timeout: float = 0.1) -> Optional[FramePacket]:
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU for (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _assign_track_id(
    bbox: Tuple[int, int, int, int],
    tracks: Dict[int, TrackState],
    frame_idx: int,
    fallback_id: int,
    min_iou: float = 0.3,
) -> int:
    """Assign an existing track ID by IoU matching, else return fallback ID."""
    best_track_id = fallback_id
    best_iou = 0.0

    for track_id, track in tracks.items():
        if frame_idx - track.last_seen > 1:
            continue
        iou = _bbox_iou(bbox, track.bbox)
        if iou > best_iou:
            best_iou = iou
            best_track_id = track_id

    if best_iou < min_iou:
        return fallback_id
    return best_track_id


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        backbone=args.backbone,
        num_classes=len(EMOTION_CLASSES),
    )
    detector = FaceDetector(min_confidence=args.face_conf)

    tracks: Dict[int, TrackState] = {}
    next_track_id = 0
    frame_idx = 0
    max_track_age = 30

    camera = ThreadedCamera(src=args.camera).start()

    try:
        while True:
            packet = camera.read(timeout=0.5)
            if packet is None:
                continue

            frame = packet.frame
            frame_idx += 1
            detections = detector.detect(frame)

            for det in detections:
                x, y, w, h = det.bbox
                face_roi = detector.crop_face(frame, det.bbox)
                if face_roi.size == 0:
                    continue

                track_id = _assign_track_id(det.bbox, tracks, frame_idx, next_track_id)
                if track_id == next_track_id:
                    tracks[track_id] = TrackState(
                        bbox=det.bbox,
                        last_seen=frame_idx,
                        smoother=RollingEmotionSmoother(window_size=args.window),
                    )
                    next_track_id += 1

                track = tracks[track_id]
                track.bbox = det.bbox
                track.last_seen = frame_idx

                face_tensor = preprocess_face(face_roi).to(device)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                label_idx, confidence, _ = track.smoother.update(probs)
                label = EMOTION_CLASSES[label_idx]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Emotion: {label}",
                    (x, max(20, y - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}",
                    (x, max(40, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

            stale_track_ids = [
                track_id
                for track_id, track in tracks.items()
                if frame_idx - track.last_seen > max_track_age
            ]
            for track_id in stale_track_ids:
                del tracks[track_id]

            cv2.imshow("Student Emotion Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time student emotion detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="Emotion classifier backbone",
    )
    parser.add_argument("--window", type=int, default=30, help="Temporal smoothing window")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face detector confidence")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
