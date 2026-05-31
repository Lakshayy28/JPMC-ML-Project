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
from fri.models.pytorch_gnn import train_pytorch_gcn


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
    metrics = train_pytorch_gcn(
        graph,
        feature_bundle,
        feature_source=settings.gnn.feature_source,
        hidden_dim=settings.gnn.hidden_dim,
        dropout=settings.gnn.dropout,
        learning_rate=settings.gnn.learning_rate,
        weight_decay=settings.gnn.weight_decay,
        epochs=settings.gnn.epochs,
        patience=settings.gnn.patience,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pytorch_gcn_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote PyTorch GCN metrics to: {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
