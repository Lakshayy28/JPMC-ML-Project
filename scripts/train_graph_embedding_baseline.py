from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.builder import build_archive_transaction_graph
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_graph_feature_bundle
from fri.models.graph_embedding import train_embedding_and_combined_baselines


def main() -> None:
    settings = load_settings()
    archive_data = load_archive_graph_data(settings.graph.archive_sample)
    graph = build_archive_transaction_graph(archive_data.nodes, archive_data.transactions)
    feature_bundle = build_graph_feature_bundle(
        graph,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
        embedding_dimensions=settings.graph.embedding_dimensions,
        embedding_random_state=settings.models.random_state,
    )
    metrics = train_embedding_and_combined_baselines(
        feature_bundle,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "graph_embedding_metrics.json"
    embeddings_path = output_dir / "archive_node_embeddings.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_bundle["embeddings"].to_csv(embeddings_path, index=False)

    print(f"Wrote graph embedding metrics to: {metrics_path}")
    print(f"Wrote graph node embeddings to: {embeddings_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
