from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.models.pytorch_gnn import train_pytorch_gcn


def test_pytorch_gcn_trains_on_small_graph() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, edge_count=1.0, total_amount=5.0)
    graph.add_edge(1, 2, edge_count=1.0, total_amount=3.0)
    graph.add_edge(2, 3, edge_count=1.0, total_amount=4.0)
    graph.add_edge(3, 4, edge_count=1.0, total_amount=2.0)
    graph.add_edge(4, 5, edge_count=1.0, total_amount=1.0)
    for node_id in range(6):
        graph.add_node(node_id, is_fraud=int(node_id in {2, 4}), initial_balance=float(node_id + 1), fraud_step=-1)

    node_features = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4, 5],
            "in_degree": [0, 1, 1, 1, 1, 1],
            "out_degree": [1, 1, 1, 1, 1, 0],
            "weighted_in_degree": [0.0, 5.0, 3.0, 4.0, 2.0, 1.0],
            "weighted_out_degree": [5.0, 3.0, 4.0, 2.0, 1.0, 0.0],
            "pagerank": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
            "clustering_coefficient": [0.0] * 6,
            "weak_component_id": [0] * 6,
            "weak_component_size": [6] * 6,
            "community_id": [0, 0, 0, 1, 1, 1],
            "community_size": [3] * 6,
            "is_fraud": [0, 0, 1, 0, 1, 0],
            "initial_balance": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "fraud_step": [-1, -1, 2, -1, 3, -1],
        }
    )
    embeddings = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4, 5],
            "embedding_0": [0.0, 0.1, 0.4, 0.2, 0.5, 0.3],
            "embedding_1": [0.0, 0.2, 0.5, 0.3, 0.6, 0.4],
        }
    )
    feature_bundle = {
        "node_features": node_features,
        "embeddings": embeddings,
        "combined": node_features.merge(embeddings, on="node_id", how="left"),
    }

    metrics = train_pytorch_gcn(
        graph,
        feature_bundle,
        feature_source="combined",
        hidden_dim=8,
        epochs=10,
        patience=5,
        test_size=0.33,
        random_state=7,
    )

    assert metrics["model_name"] == "pytorch_gcn"
    assert metrics["feature_dimension"] > 0
    assert metrics["test_rows"] > 0
