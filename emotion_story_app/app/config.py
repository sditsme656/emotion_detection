"""Central configuration values for the Emotion-Aware Storytelling MVP."""

from __future__ import annotations

from pathlib import Path

# Vision / inference loop
EMOTION_FPS: int = 8
FACE_DETECT_EVERY_N: int = 6
WINDOW_SECONDS: float = 1.5
STABLE_FRAMES: int = 8
MIN_CONF: float = 0.55
IOU_KEEP_THRESHOLD: float = 0.30
NO_FACE_TIMEOUT: float = 1.0

# Story loop
STORY_TICK_SECONDS: float = 3.0
STORY_MIN_TRANSITION_GAP: float = 2.5

# Behavior toggles
USE_DEEPFACE: bool = False
HOLD_LAST_EMOTION_ON_NO_FACE: bool = True

# Rendering / model input
MODEL_INPUT_SIZE: tuple[int, int] = (64, 64)
CANONICAL_EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STORY_GRAPH_PATH = ROOT_DIR / "story_graphs" / "story_graph.json"
