"""Story graph schema validation and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Choice:
    label: str
    to: str
    if_emotion_in: list[str]


@dataclass
class StoryNode:
    id: str
    type: str
    base_text: str
    variants: dict[str, str] = field(default_factory=dict)
    next: str | None = None
    choices: list[Choice] = field(default_factory=list)
    fallback_to: str | None = None


@dataclass
class StoryGraph:
    title: str
    start_node: str
    nodes: dict[str, StoryNode]


def load_story_graph(path: str | Path) -> StoryGraph:
    """Load and validate story graph JSON file."""
    graph_path = Path(path)
    raw = json.loads(graph_path.read_text(encoding="utf-8"))

    if "meta" not in raw or "nodes" not in raw:
        raise ValueError("Graph must contain meta and nodes")

    meta = raw["meta"]
    title = meta.get("title", "Untitled")
    start_node = meta.get("start_node")
    if not isinstance(start_node, str) or not start_node:
        raise ValueError("meta.start_node is required")

    nodes_raw = raw["nodes"]
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("nodes must be a non-empty list")

    nodes: dict[str, StoryNode] = {}
    for node_raw in nodes_raw:
        node_id = node_raw.get("id")
        node_type = node_raw.get("type")
        base_text = node_raw.get("base_text")
        if not all(isinstance(v, str) and v for v in (node_id, node_type, base_text)):
            raise ValueError(f"Node missing required fields: {node_raw}")

        if node_type not in {"narration", "decision"}:
            raise ValueError(f"Invalid node type '{node_type}' for node '{node_id}'")

        variants = node_raw.get("variants", {})
        if variants is None:
            variants = {}
        if not isinstance(variants, dict):
            raise ValueError(f"variants must be a dict for node '{node_id}'")

        node = StoryNode(
            id=node_id,
            type=node_type,
            base_text=base_text,
            variants=variants,
            next=node_raw.get("next"),
            fallback_to=node_raw.get("fallback_to"),
        )

        if node_type == "narration":
            if not isinstance(node.next, str) or not node.next:
                raise ValueError(f"Narration node '{node_id}' must define non-empty next")

        if node_type == "decision":
            if not isinstance(node.fallback_to, str) or not node.fallback_to:
                raise ValueError(f"Decision node '{node_id}' must define fallback_to")
            choices_raw = node_raw.get("choices", [])
            if not isinstance(choices_raw, list) or not choices_raw:
                raise ValueError(f"Decision node '{node_id}' must include non-empty choices")
            for c in choices_raw:
                if not isinstance(c.get("label"), str) or not isinstance(c.get("to"), str):
                    raise ValueError(f"Invalid choice in node '{node_id}': {c}")
                emotions = c.get("if_emotion_in", [])
                if not isinstance(emotions, list) or not all(isinstance(e, str) for e in emotions):
                    raise ValueError(f"choice.if_emotion_in must be a list[str] in node '{node_id}'")
                node.choices.append(Choice(label=c["label"], to=c["to"], if_emotion_in=emotions))

        if node_id in nodes:
            raise ValueError(f"Duplicate node id '{node_id}'")
        nodes[node_id] = node

    if start_node not in nodes:
        raise ValueError(f"start_node '{start_node}' not found in nodes")

    for node in nodes.values():
        refs: list[str] = []
        if node.type == "narration" and node.next:
            refs.append(node.next)
        if node.type == "decision":
            if node.fallback_to:
                refs.append(node.fallback_to)
            refs.extend([c.to for c in node.choices])
        for ref in refs:
            if ref not in nodes:
                raise ValueError(f"Node '{node.id}' references missing node '{ref}'")

    return StoryGraph(title=title, start_node=start_node, nodes=nodes)
