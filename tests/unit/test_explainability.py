from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("torch_geometric")

from fri.explainability.service import HeteroGraphExplainerService
from fri.models.pytorch_gnn import build_hetero_gat_model, build_pyg_graph_data_from_tables, prepare_hetero_inference_data


def test_hetero_explainer_service_returns_structured_report() -> None:
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
    inference_data = prepare_hetero_inference_data(
        graph_bundle.data,
        random_state=7,
        test_size=0.25,
        pin_memory=False,
    )
    model = build_hetero_gat_model(inference_data, hidden_dim=8, dropout=0.1).eval()
    merchant_node_ids = [f"merchant_{index:03d}" for index in range(int(inference_data["merchant"].x.shape[0]))]
    service = HeteroGraphExplainerService(
        model,
        feature_columns=list(graph_bundle.feature_columns),
        account_node_ids=[int(value) for value in inference_data["account"].node_id.detach().cpu().tolist()],
        merchant_node_ids=merchant_node_ids,
        top_k_features=3,
        top_k_edges=3,
        explainer_epochs=2,
    )

    report = service.explain_account(inference_data, 0)

    assert report.node_id == 0
    assert 0.0 <= report.risk_score <= 1.0
    assert report.top_node_features
    assert isinstance(report.critical_edges, list)