"""Story worker thread handling low-frequency story progression."""

from __future__ import annotations

import threading

from app.config import STORY_MIN_TRANSITION_GAP, STORY_TICK_SECONDS
from app.state import SharedState
from app.story.graph_engine import StoryEngine
from app.story.graph_schema import load_story_graph
from app.utils.logger import configure_logger
from app.utils.timing import PeriodicTimer


class StoryWorker(threading.Thread):
    """Background worker for story transitions and text rendering."""

    def __init__(self, state: SharedState, stop_event: threading.Event, graph_path: str) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = stop_event
        self.logger = configure_logger("story_worker")

        graph = load_story_graph(graph_path)
        self.engine = StoryEngine(graph=graph, min_transition_gap=STORY_MIN_TRANSITION_GAP)

    def run(self) -> None:
        timer = PeriodicTimer(STORY_TICK_SECONDS)
        while not self.stop_event.is_set():
            with self.state.lock:
                emotion = self.state.stable_emotion
                face_present = self.state.face_present

            prev_node = self.engine.current_node_id
            node_id, text = self.engine.step(emotion=emotion, face_present=face_present)
            if node_id != prev_node:
                self.logger.info("Story transition: %s -> %s (emotion=%s)", prev_node, node_id, emotion)

            with self.state.lock:
                self.state.story_node_id = node_id
                self.state.story_text = text

            timer.wait_next()
