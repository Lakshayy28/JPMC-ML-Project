from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.analytics import compute_graph_summary, compute_node_features
from fri.graph.builder import build_archive_transaction_graph
from fri.graph.io import load_archive_graph_data


def test_archive_graph_can_be_loaded_and_analyzed() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    archive_data = load_archive_graph_data(settings.graph.archive_sample)
    graph = build_archive_transaction_graph(archive_data.nodes, archive_data.transactions)
    summary = compute_graph_summary(graph)
    features = compute_node_features(graph, include_communities=False)

    assert summary["node_count"] == archive_data.metadata["reported_node_count"]
    assert summary["edge_count"] > 0
    assert not features.empty
    assert "pagerank" in features.columns
    assert "is_fraud" in features.columns
