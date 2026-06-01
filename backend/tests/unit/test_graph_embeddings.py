from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle


def test_graph_feature_bundle_contains_embeddings() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    archive_data = load_archive_graph_data(settings.graph.archive_sample)
    bundle = build_archive_feature_bundle(
        archive_data.nodes,
        archive_data.transactions,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=False,
        embedding_dimensions=8,
        include_embeddings=True,
    )

    assert not bundle["embeddings"].empty
    assert bundle["embeddings"].shape[0] == archive_data.metadata["reported_node_count"]
    assert "embedding_0" in bundle["embeddings"].columns
    assert "pagerank" in bundle["combined"].columns
    assert "outgoing_tx_velocity_7d" in bundle["combined"].columns
    assert not bundle["merchant_features"].empty
