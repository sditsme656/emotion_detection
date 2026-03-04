"""Story text renderer."""

from __future__ import annotations

from app.story.graph_schema import StoryNode


def render_text(node: StoryNode, emotion: str) -> str:
    """Return emotion variant if present, else base text."""
    return node.variants.get(emotion, node.base_text)
