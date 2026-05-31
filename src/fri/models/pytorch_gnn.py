from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GNNResult:
    average_precision: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    train_rows: int
    validation_rows: int
    test_rows: int
    best_epoch: int
    best_validation_average_precision: float | None


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        return torch.sparse.mm(adjacency, support)


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, adjacency)


def _prepare_feature_frame(feature_bundle: dict[str, pd.DataFrame], *, feature_source: str, label_column: str) -> pd.DataFrame:
    if feature_source not in feature_bundle:
        raise KeyError(f"Unknown feature source '{feature_source}'. Expected one of {sorted(feature_bundle)}")

    frame = feature_bundle[feature_source].copy()
    if label_column not in frame.columns:
        source_frame = feature_bundle["node_features"]
        frame = frame.merge(source_frame[["node_id", label_column]], on="node_id", how="left")

    drop_columns = [column for column in ["fraud_step"] if column in frame.columns]
    return frame.drop(columns=drop_columns)


def _standardize_features(frame: pd.DataFrame, *, label_column: str) -> tuple[list[Any], np.ndarray, np.ndarray, list[str]]:
    numeric = frame.select_dtypes(include=["number", "bool"]).copy()
    if label_column not in numeric.columns:
        raise KeyError(f"Expected label column '{label_column}' in numeric feature frame")

    y = numeric.pop(label_column).astype(int).to_numpy()
    if "node_id" in numeric.columns:
        numeric = numeric.drop(columns=["node_id"])

    x = numeric.astype(float).to_numpy()
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    x = (x - means) / stds
    return frame["node_id"].tolist(), x.astype(np.float32), y.astype(np.int64), list(numeric.columns)


def _build_normalized_adjacency(graph: nx.DiGraph, ordered_nodes: list[Any], *, weight: str = "edge_count") -> torch.Tensor:
    sparse_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
    torch.sparse.check_sparse_tensor_invariants.disable()
    undirected = graph.to_undirected()
    node_index = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for source, destination, attrs in undirected.edges(data=True):
        source_idx = node_index[source]
        destination_idx = node_index[destination]
        edge_weight = float(attrs.get(weight, 1.0))
        rows.extend([source_idx, destination_idx])
        cols.extend([destination_idx, source_idx])
        values.extend([edge_weight, edge_weight])

    for index in range(len(ordered_nodes)):
        rows.append(index)
        cols.append(index)
        values.append(1.0)

    try:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        sparse_values = torch.tensor(values, dtype=torch.float32)
        adjacency = torch.sparse_coo_tensor(
            indices,
            sparse_values,
            (len(ordered_nodes), len(ordered_nodes)),
            check_invariants=False,
        ).coalesce()
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        degree_inverse_sqrt = torch.pow(degree, -0.5)
        degree_inverse_sqrt[torch.isinf(degree_inverse_sqrt)] = 0.0

        normalized_values = (
            degree_inverse_sqrt[adjacency.indices()[0]] * adjacency.values() * degree_inverse_sqrt[adjacency.indices()[1]]
        )
        return torch.sparse_coo_tensor(
            adjacency.indices(),
            normalized_values,
            adjacency.size(),
            check_invariants=False,
        ).coalesce()
    finally:
        if sparse_checks_enabled:
            torch.sparse.check_sparse_tensor_invariants.enable()


def _can_stratify(labels: np.ndarray, *, test_size: float) -> bool:
    if len(labels) < 4:
        return False

    counts = np.bincount(labels)
    present_counts = counts[counts > 0]
    if len(present_counts) < 2 or int(present_counts.min()) < 2:
        return False

    test_rows = int(np.ceil(len(labels) * test_size))
    train_rows = len(labels) - test_rows
    class_count = int(len(present_counts))
    return test_rows >= class_count and train_rows >= class_count


def _split_indices(labels: np.ndarray, *, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if _can_stratify(labels, test_size=test_size) else None,
    )
    validation_fraction = test_size / (1.0 - test_size)
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=(
            labels[train_indices]
            if _can_stratify(labels[train_indices], test_size=validation_fraction)
            else None
        ),
    )
    return train_indices, validation_indices, test_indices


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    def metric_or_none(metric, *args, **kwargs):
        try:
            return float(metric(*args, **kwargs))
        except ValueError:
            return None

    unique_labels = np.unique(y_true)
    return {
        "average_precision": metric_or_none(average_precision_score, y_true, probabilities)
        if 1 in unique_labels
        else None,
        "precision": metric_or_none(precision_score, y_true, predictions, zero_division=0),
        "recall": metric_or_none(recall_score, y_true, predictions, zero_division=0),
        "f1": metric_or_none(f1_score, y_true, predictions, zero_division=0),
        "roc_auc": metric_or_none(roc_auc_score, y_true, probabilities) if len(unique_labels) > 1 else None,
    }


def _selection_score(labels: np.ndarray, probabilities: np.ndarray, loss_value: float) -> tuple[float, float | None]:
    if 1 not in np.unique(labels):
        return -loss_value, None
    try:
        average_precision = float(average_precision_score(labels, probabilities))
    except ValueError:
        average_precision = None
    return (average_precision if average_precision is not None else -loss_value), average_precision


def train_pytorch_gcn(
    graph: nx.DiGraph,
    feature_bundle: dict[str, pd.DataFrame],
    *,
    feature_source: str = "combined",
    label_column: str = "is_fraud",
    hidden_dim: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 120,
    patience: int = 20,
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, float | int | None | str]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    prepared = _prepare_feature_frame(feature_bundle, feature_source=feature_source, label_column=label_column)
    node_ids, x_array, y_array, feature_columns = _standardize_features(prepared, label_column=label_column)
    adjacency = _build_normalized_adjacency(graph, node_ids)

    x = torch.tensor(x_array, dtype=torch.float32)
    y = torch.tensor(y_array, dtype=torch.long)
    train_idx, val_idx, test_idx = _split_indices(y_array, test_size=test_size, random_state=random_state)
    train_index_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_index_tensor = torch.tensor(val_idx, dtype=torch.long)
    test_index_tensor = torch.tensor(test_idx, dtype=torch.long)

    class_counts = np.bincount(y_array)
    negative_count = max(int(class_counts[0]), 1)
    positive_count = max(int(class_counts[1]), 1)
    class_weights = torch.tensor([1.0, negative_count / positive_count], dtype=torch.float32)

    model = GCN(input_dim=x.shape[1], hidden_dim=hidden_dim, output_dim=2, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_ap: float | None = None
    best_selection_score = float("-inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjacency)
        loss = criterion(logits[train_index_tensor], y[train_index_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, adjacency)[val_index_tensor]
            val_loss = float(criterion(val_logits, y[val_index_tensor]).item())
            val_probabilities = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_labels = y[val_index_tensor].cpu().numpy()
            selection_score, val_average_precision = _selection_score(val_labels, val_probabilities, val_loss)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_ap = val_average_precision
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(x, adjacency)[test_index_tensor]
        test_probabilities = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_predictions = torch.argmax(test_logits, dim=1).cpu().numpy()
        test_labels = y[test_index_tensor].cpu().numpy()

    metrics = _classification_metrics(test_labels, test_probabilities, test_predictions)
    result = GNNResult(
        average_precision=metrics["average_precision"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        train_rows=int(len(train_idx)),
        validation_rows=int(len(val_idx)),
        test_rows=int(len(test_idx)),
        best_epoch=int(best_epoch),
        best_validation_average_precision=best_val_ap,
    )

    payload = asdict(result)
    payload.update(
        {
            "model_name": "pytorch_gcn",
            "feature_source": feature_source,
            "feature_dimension": int(x.shape[1]),
            "feature_columns": feature_columns,
        }
    )
    return payload
