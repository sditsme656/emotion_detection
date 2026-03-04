"""Thread-safe shared state for UI, vision worker, and story worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np


@dataclass
class SharedState:
    """Data exchanged between workers and UI thread."""

    latest_frame: Optional[np.ndarray] = None
    latest_frame_ts: float = 0.0

    face_box: Optional[tuple[int, int, int, int]] = None
    face_present: bool = False

    stable_emotion: str = "neutral"
    emotion_conf: float = 0.0

    story_node_id: str = ""
    story_text: str = ""

    last_infer_ms: float = 0.0
    vision_fps_est: float = 0.0
    errors: list[str] = field(default_factory=list)

    lock: Lock = field(default_factory=Lock)
