"""Temporal smoothing with hysteresis for emotion predictions."""

from __future__ import annotations

import time
from collections import deque

from app.config import CANONICAL_EMOTIONS


class EmotionSmoother:
    """Maintains a sliding time window of probabilities and a hysteresis state."""

    def __init__(self, window_seconds: float, stable_frames: int, min_conf: float) -> None:
        self.window_seconds = window_seconds
        self.stable_frames = stable_frames
        self.min_conf = min_conf

        self._samples: deque[tuple[float, dict[str, float]]] = deque()
        self.current_emotion: str = "neutral"
        self.current_confidence: float = 0.0

        self._pending_emotion: str | None = None
        self._pending_count: int = 0

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def update(self, probs: dict[str, float], ts: float | None = None) -> tuple[str, float, dict[str, float]]:
        """Update smoother and return (stable_emotion, confidence, mean_probs)."""
        now = ts if ts is not None else time.time()
        normalized = {label: float(probs.get(label, 0.0)) for label in CANONICAL_EMOTIONS}
        total = sum(normalized.values())
        if total > 1e-8:
            normalized = {k: v / total for k, v in normalized.items()}

        self._samples.append((now, normalized))
        self._trim(now)

        if not self._samples:
            return self.current_emotion, self.current_confidence, normalized

        mean_probs = {label: 0.0 for label in CANONICAL_EMOTIONS}
        for _, sample in self._samples:
            for label, val in sample.items():
                mean_probs[label] += val
        n = float(len(self._samples))
        mean_probs = {k: v / n for k, v in mean_probs.items()}

        candidate = max(mean_probs, key=mean_probs.get)
        candidate_conf = float(mean_probs[candidate])

        if candidate_conf < self.min_conf:
            self._pending_emotion = None
            self._pending_count = 0
            return self.current_emotion, self.current_confidence, mean_probs

        if candidate == self.current_emotion:
            self.current_confidence = candidate_conf
            self._pending_emotion = None
            self._pending_count = 0
            return self.current_emotion, self.current_confidence, mean_probs

        if self._pending_emotion == candidate:
            self._pending_count += 1
        else:
            self._pending_emotion = candidate
            self._pending_count = 1

        if self._pending_count >= self.stable_frames:
            self.current_emotion = candidate
            self.current_confidence = candidate_conf
            self._pending_emotion = None
            self._pending_count = 0

        return self.current_emotion, self.current_confidence, mean_probs
