from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.builder import build_archive_transaction_graph
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_graph_feature_bundle


def test_graph_feature_bundle_contains_embeddings() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    archive_data = load_archive_graph_data(settings.graph.archive_sample)
    graph = build_archive_transaction_graph(archive_data.nodes, archive_data.transactions)
    bundle = build_graph_feature_bundle(graph, include_communities=False, embedding_dimensions=8)

    assert not bundle["embeddings"].empty
    assert bundle["embeddings"].shape[0] == archive_data.metadata["reported_node_count"]
    assert "embedding_0" in bundle["embeddings"].columns
    assert "pagerank" in bundle["combined"].columns
