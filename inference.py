"""Real-time multi-stage emotion detection inference pipeline."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import torch

from detector import FaceDetector
from model import EMOTION_CLASSES, load_model
from utils import RollingEmotionSmoother, preprocess_face


@dataclass
class FramePacket:
    frame: any
    timestamp: float


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
    smoother = RollingEmotionSmoother(window_size=args.window)

    camera = ThreadedCamera(src=args.camera).start()

    try:
        while True:
            packet = camera.read(timeout=0.5)
            if packet is None:
                continue

            frame = packet.frame
            detections = detector.detect(frame)

            for det in detections:
                x, y, w, h = det.bbox
                face_roi = detector.crop_face(frame, det.bbox)
                if face_roi.size == 0:
                    continue

                face_tensor = preprocess_face(face_roi).to(device)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                label_idx, confidence, _ = smoother.update(probs)
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
