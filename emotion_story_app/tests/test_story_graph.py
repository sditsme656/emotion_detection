from pathlib import Path

from app.story.graph_schema import load_story_graph


def test_story_graph_loads_and_references_are_valid() -> None:
    graph_path = Path(__file__).resolve().parents[1] / "story_graphs" / "story_graph.json"
    graph = load_story_graph(graph_path)

    assert graph.start_node in graph.nodes

    for node in graph.nodes.values():
        if node.type == "narration":
            assert node.next in graph.nodes
        else:
            assert node.fallback_to in graph.nodes
            for choice in node.choices:
                assert choice.to in graph.nodes
