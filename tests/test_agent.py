from src.agent import build_self_improving_graph


def test_graph_compilation():
    graph = build_self_improving_graph()
    assert graph is not None
