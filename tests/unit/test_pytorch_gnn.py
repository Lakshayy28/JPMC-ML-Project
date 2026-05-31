from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("torch_geometric")

from fri.models.pytorch_gnn import build_pyg_graph_data_from_tables, resolve_training_device, train_pyg_minibatch


def test_pytorch_gcn_trains_on_small_graph() -> None:
    nodes = pd.DataFrame(
        {
            "nodeid": list(range(12)),
            "isFraud": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "init_balance": [float(index + 1) * 100.0 for index in range(12)],
            "fraudStep": [-1, -1, 5, -1, -1, 7, -1, -1, 9, -1, -1, 11],
        }
    )
    transactions = pd.DataFrame(
        {
            "sourceNodeId": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 5, 8, 11],
            "targetNodeId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 5, 8, 11, 2],
            "value": [10.0, 11.0, 40.0, 8.0, 12.0, 45.0, 9.0, 10.0, 50.0, 11.0, 9.0, 55.0, 42.0, 48.0, 53.0, 46.0],
            "time": list(range(1, 17)),
        }
    )
    graph_bundle = build_pyg_graph_data_from_tables(nodes, transactions)

    metrics = train_pyg_minibatch(
        graph_bundle.data,
        feature_columns=graph_bundle.feature_columns,
        hidden_dim=8,
        epochs=10,
        patience=5,
        batch_size=4,
        fan_out=(2, 2),
        num_workers=0,
        device=resolve_training_device(use_cuda=False, requested_device="cpu"),
        pin_memory=False,
        random_state=7,
        test_size=0.25,
    )

    assert metrics["model_name"] == "pytorch_graphsage"
    assert metrics["feature_dimension"] > 0
    assert metrics["test_rows"] > 0
