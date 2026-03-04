"""Story graph progression engine."""

from __future__ import annotations

import time

from app.story.graph_schema import StoryGraph, StoryNode
from app.story.renderer import render_text


class StoryEngine:
    """Controls node progression and branching on decision nodes."""

    def __init__(self, graph: StoryGraph, min_transition_gap: float) -> None:
        self.graph = graph
        self.current_node_id: str = graph.start_node
        self.min_transition_gap = min_transition_gap
        self._last_transition_ts: float = 0.0

    def current_node(self) -> StoryNode:
        return self.graph.nodes[self.current_node_id]

    def _can_transition(self, now: float) -> bool:
        return (now - self._last_transition_ts) >= self.min_transition_gap

    def _transition_to(self, node_id: str, now: float) -> None:
        self.current_node_id = node_id
        self._last_transition_ts = now

    def step(self, emotion: str, face_present: bool, now: float | None = None) -> tuple[str, str]:
        """Advance story respecting transition cooldown and face presence requirements."""
        ts = now if now is not None else time.time()
        node = self.current_node()
        text = render_text(node, emotion)

        if not face_present:
            return node.id, text
        if not self._can_transition(ts):
            return node.id, text

        if node.type == "narration" and node.next:
            self._transition_to(node.next, ts)
        elif node.type == "decision":
            next_id = node.fallback_to
            for choice in node.choices:
                if emotion in choice.if_emotion_in:
                    next_id = choice.to
                    break
            if next_id:
                self._transition_to(next_id, ts)

        node = self.current_node()
        return node.id, render_text(node, emotion)
