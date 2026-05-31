from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle


@dataclass(frozen=True)
class PYGGraphBundle:
    data: HeteroData
    feature_columns: tuple[str, ...]
    merchant_feature_columns: tuple[str, ...]
    edge_feature_columns: tuple[str, ...]


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
    best_validation_f1: float | None
    optimal_threshold: float


class SpatialTemporalHeteroGAT(nn.Module):
    def __init__(
        self,
        account_input_dim: int,
        merchant_input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        *,
        output_dim: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.account_encoder = nn.Linear(account_input_dim, hidden_dim)
        self.merchant_encoder = nn.Linear(merchant_input_dim, hidden_dim)
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                self._build_conv(hidden_dim, edge_dim=edge_dim, heads=heads),
                self._build_conv(hidden_dim, edge_dim=edge_dim, heads=heads),
            ]
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _build_conv(hidden_dim: int, *, edge_dim: int, heads: int) -> HeteroConv:
        return HeteroConv(
            {
                ("account", "transfers", "account"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                ),
                ("account", "buys_from", "merchant"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                ),
                ("merchant", "rev_buys_from", "account"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "account": self.account_encoder(data["account"].x),
            "merchant": self.merchant_encoder(data["merchant"].x),
        }
        edge_attr_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {node_type: F.elu(features) for node_type, features in x_dict.items()}
            x_dict = {
                node_type: F.dropout(features, p=self.dropout, training=self.training)
                for node_type, features in x_dict.items()
            }

        return self.classifier(x_dict["account"])


def resolve_training_device(*, use_cuda: bool = True, requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    requested = torch.device(requested_device)
    if requested.type == "cuda" and (not use_cuda or not torch.cuda.is_available()):
        return torch.device("cpu")
    return requested


def _build_hetero_graph_bundle(feature_bundle: dict[str, object]) -> PYGGraphBundle:
    account_frame = (
        feature_bundle["node_features"]
        if isinstance(feature_bundle["node_features"], pd.DataFrame)
        else pd.DataFrame()
    )
    merchant_frame = (
        feature_bundle["merchant_features"]
        if isinstance(feature_bundle["merchant_features"], pd.DataFrame)
        else pd.DataFrame()
    )
    transfer_frame = (
        feature_bundle["normalized_transactions"]
        if isinstance(feature_bundle["normalized_transactions"], pd.DataFrame)
        else pd.DataFrame()
    )
    merchant_links = (
        feature_bundle["merchant_links"]
        if isinstance(feature_bundle["merchant_links"], pd.DataFrame)
        else pd.DataFrame()
    )

    if account_frame.empty:
        raise ValueError("Account feature frame is empty; cannot build a hetero graph bundle")
    if merchant_frame.empty:
        raise ValueError("Merchant feature frame is empty; cannot build a hetero graph bundle")
    if transfer_frame.empty:
        raise ValueError("Transfer edge frame is empty; cannot build a hetero graph bundle")
    if merchant_links.empty:
        raise ValueError("Merchant interaction frame is empty; cannot build a hetero graph bundle")

    account_frame = account_frame.sort_values("node_id").reset_index(drop=True)
    merchant_frame = merchant_frame.sort_values("merchant_id").reset_index(drop=True)

    account_feature_columns = tuple(
        column for column in account_frame.columns if column not in {"node_id", "is_fraud", "fraud_step"}
    )
    merchant_feature_columns = tuple(column for column in merchant_frame.columns if column != "merchant_id")
    edge_feature_columns = ("amount", "event_time", "transaction_type_code")

    account_ids = account_frame["node_id"].astype(int).to_numpy(copy=True)
    account_index = {node_id: index for index, node_id in enumerate(account_ids)}
    merchant_ids = merchant_frame["merchant_id"].astype(str).tolist()
    merchant_index = {merchant_id: index for index, merchant_id in enumerate(merchant_ids)}

    transfer_sources = transfer_frame["source_node_id"].map(account_index)
    transfer_targets = transfer_frame["target_node_id"].map(account_index)
    transfer_mask = transfer_sources.notna() & transfer_targets.notna()
    transfer_edge_index = torch.tensor(
        np.vstack(
            [
                transfer_sources.loc[transfer_mask].astype(np.int64).to_numpy(),
                transfer_targets.loc[transfer_mask].astype(np.int64).to_numpy(),
            ]
        ),
        dtype=torch.long,
    )
    transfer_edge_attr = torch.from_numpy(
        transfer_frame.loc[transfer_mask, list(edge_feature_columns)].astype(np.float32).to_numpy(copy=True)
    )

    merchant_sources = merchant_links["source_node_id"].map(account_index)
    merchant_targets = merchant_links["merchant_id"].map(merchant_index)
    merchant_mask = merchant_sources.notna() & merchant_targets.notna()
    merchant_edge_index = torch.tensor(
        np.vstack(
            [
                merchant_sources.loc[merchant_mask].astype(np.int64).to_numpy(),
                merchant_targets.loc[merchant_mask].astype(np.int64).to_numpy(),
            ]
        ),
        dtype=torch.long,
    )
    merchant_edge_attr = torch.from_numpy(
        merchant_links.loc[merchant_mask, list(edge_feature_columns)].astype(np.float32).to_numpy(copy=True)
    )

    data = HeteroData()
    data["account"].x = torch.from_numpy(
        account_frame.loc[:, account_feature_columns].astype(np.float32).to_numpy(copy=True)
    )
    data["account"].y = torch.from_numpy(account_frame["is_fraud"].astype(np.int64).to_numpy(copy=True))
    data["account"].node_id = torch.from_numpy(account_ids.astype(np.int64))
    data["account"].fraud_step = torch.from_numpy(account_frame["fraud_step"].astype(np.int64).to_numpy(copy=True))

    data["merchant"].x = torch.from_numpy(
        merchant_frame.loc[:, merchant_feature_columns].astype(np.float32).to_numpy(copy=True)
    )

    data["account", "transfers", "account"].edge_index = transfer_edge_index
    data["account", "transfers", "account"].edge_attr = transfer_edge_attr
    data["account", "buys_from", "merchant"].edge_index = merchant_edge_index
    data["account", "buys_from", "merchant"].edge_attr = merchant_edge_attr
    data["merchant", "rev_buys_from", "account"].edge_index = merchant_edge_index.flip(0)
    data["merchant", "rev_buys_from", "account"].edge_attr = merchant_edge_attr.clone()

    return PYGGraphBundle(
        data=data,
        feature_columns=account_feature_columns,
        merchant_feature_columns=merchant_feature_columns,
        edge_feature_columns=edge_feature_columns,
    )


def build_pyg_graph_data_from_tables(
    nodes: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    temporal_windows: Sequence[int] = (1, 7, 30),
    merchant_seed: int = 17,
    merchant_pool_size: int = 24,
    include_communities: bool = True,
    community_seed: int = 42,
) -> PYGGraphBundle:
    feature_bundle = build_archive_feature_bundle(
        nodes,
        transactions,
        temporal_windows=temporal_windows,
        merchant_seed=merchant_seed,
        merchant_pool_size=merchant_pool_size,
        include_communities=include_communities,
        community_seed=community_seed,
        include_embeddings=False,
    )
    return _build_hetero_graph_bundle(feature_bundle)


def build_pyg_graph_data_from_archive(
    archive_path: str | Path,
    *,
    chunksize: int = 100_000,
    temporal_windows: Sequence[int] = (1, 7, 30),
    merchant_seed: int = 17,
    merchant_pool_size: int = 24,
    include_communities: bool = True,
    community_seed: int = 42,
) -> PYGGraphBundle:
    del chunksize
    archive_data = load_archive_graph_data(archive_path)
    return build_pyg_graph_data_from_tables(
        archive_data.nodes,
        archive_data.transactions,
        temporal_windows=temporal_windows,
        merchant_seed=merchant_seed,
        merchant_pool_size=merchant_pool_size,
        include_communities=include_communities,
        community_seed=community_seed,
    )


def _normalize_tensor(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    means = matrix.mean(dim=0)
    stds = matrix.std(dim=0, unbiased=False)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def _normalize_hetero_data(data: HeteroData, train_indices: torch.Tensor) -> HeteroData:
    normalized = copy.deepcopy(data)
    account_train_x = normalized["account"].x[train_indices]
    train_means = account_train_x.mean(dim=0)
    train_stds = account_train_x.std(dim=0, unbiased=False)
    train_stds[train_stds == 0] = 1.0
    normalized["account"].x = (normalized["account"].x - train_means) / train_stds
    normalized["merchant"].x = _normalize_tensor(normalized["merchant"].x)

    for edge_type in normalized.edge_types:
        normalized[edge_type].edge_attr = _normalize_tensor(normalized[edge_type].edge_attr)

    return normalized


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


def _metric_or_none(metric, *args, **kwargs) -> float | None:
    try:
        return float(metric(*args, **kwargs))
    except ValueError:
        return None


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    unique_labels = np.unique(y_true)
    return {
        "average_precision": _metric_or_none(average_precision_score, y_true, probabilities)
        if 1 in unique_labels
        else None,
        "precision": _metric_or_none(precision_score, y_true, predictions, zero_division=0),
        "recall": _metric_or_none(recall_score, y_true, predictions, zero_division=0),
        "f1": _metric_or_none(f1_score, y_true, predictions, zero_division=0),
        "roc_auc": _metric_or_none(roc_auc_score, y_true, probabilities) if len(unique_labels) > 1 else None,
    }


def _average_precision_or_none(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if labels.size == 0 or 1 not in np.unique(labels):
        return None
    return _metric_or_none(average_precision_score, labels, probabilities)


def _optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    default_threshold: float,
) -> tuple[float, float | None]:
    if labels.size == 0:
        return float(default_threshold), None

    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = -1.0
    optimal_threshold = float(default_threshold)

    for threshold in thresholds:
        current_predictions = (probabilities >= threshold).astype(np.int64)
        current_f1 = float(f1_score(labels, current_predictions, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = float(threshold)

    return optimal_threshold, (best_f1 if best_f1 >= 0.0 else None)


def _evaluate_full_batch(
    model: SpatialTemporalHeteroGAT,
    data: HeteroData,
    criterion: nn.Module,
    indices: torch.Tensor,
    *,
    decision_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data)[indices]
        labels = data["account"].y[indices]
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        predictions = (probabilities >= decision_threshold).astype(np.int64)
        label_array = labels.detach().cpu().numpy()
    return label_array, probabilities, predictions, float(loss.item())


def train_pyg_minibatch(
    data: HeteroData,
    *,
    feature_columns: Sequence[str] | None = None,
    merchant_feature_columns: Sequence[str] | None = None,
    edge_feature_columns: Sequence[str] | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 1024,
    fan_out: Sequence[int] = (25, 10),
    num_workers: int = 0,
    device: torch.device | None = None,
    pin_memory: bool = True,
    checkpoint_path: str | Path | None = None,
    pos_weight_multiplier: float = 1.0,
    decision_threshold: float = 0.5,
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, float | int | None | str | list[str]]:
    del batch_size
    del fan_out
    del num_workers

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    resolved_device = device or torch.device("cpu")
    labels_array = data["account"].y.cpu().numpy()
    train_idx, val_idx, test_idx = _split_indices(labels_array, test_size=test_size, random_state=random_state)
    normalized_data = _normalize_hetero_data(data, torch.tensor(train_idx, dtype=torch.long))
    if pin_memory and resolved_device.type == "cuda":
        normalized_data = normalized_data.pin_memory()
    normalized_data = normalized_data.to(resolved_device, non_blocking=True)

    train_index_tensor = torch.tensor(train_idx, dtype=torch.long, device=resolved_device)
    val_index_tensor = torch.tensor(val_idx, dtype=torch.long, device=resolved_device)
    test_index_tensor = torch.tensor(test_idx, dtype=torch.long, device=resolved_device)

    class_counts = np.bincount(labels_array[train_idx], minlength=2)
    negative_count = max(int(class_counts[0]), 1)
    positive_count = max(int(class_counts[1]), 1)
    class_weights = torch.tensor(
        [1.0, (negative_count / positive_count) * pos_weight_multiplier],
        dtype=torch.float32,
        device=resolved_device,
    )

    model = SpatialTemporalHeteroGAT(
        account_input_dim=int(normalized_data["account"].x.shape[1]),
        merchant_input_dim=int(normalized_data["merchant"].x.shape[1]),
        edge_dim=int(normalized_data["account", "transfers", "account"].edge_attr.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_file is not None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_ap: float | None = None
    best_val_f1: float | None = None
    best_selection_score = float("-inf")
    best_threshold = float(decision_threshold)
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized_data)
        train_logits = logits[train_index_tensor]
        train_labels = normalized_data["account"].y[train_index_tensor]
        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        val_labels, val_probabilities, _, val_loss = _evaluate_full_batch(
            model,
            normalized_data,
            criterion,
            val_index_tensor,
            decision_threshold=decision_threshold,
        )
        optimal_threshold, val_f1 = _optimal_threshold(
            val_labels,
            val_probabilities,
            default_threshold=decision_threshold,
        )
        selection_score = float(val_f1) if val_f1 is not None else -val_loss
        val_average_precision = _average_precision_or_none(val_labels, val_probabilities)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_ap = val_average_precision
            best_val_f1 = val_f1
            best_threshold = optimal_threshold
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_file is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "optimal_threshold": best_threshold,
                    },
                    checkpoint_file,
                )
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[GNN Epoch {epoch:03d}/{epochs}] "
                f"Loss: {loss.item():.4f} | "
                f"Val Selection Score: {selection_score:.4f} | "
                f"Patience Counter: {patience_counter}/{patience}",
                flush=True,
            )

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint_file is not None and checkpoint_file.exists():
        checkpoint_payload = torch.load(checkpoint_file, map_location=resolved_device)
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        best_threshold = float(checkpoint_payload.get("optimal_threshold", decision_threshold))

    test_labels, test_probabilities, test_predictions, _ = _evaluate_full_batch(
        model,
        normalized_data,
        criterion,
        test_index_tensor,
        decision_threshold=best_threshold,
    )
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
        best_validation_f1=best_val_f1,
        optimal_threshold=float(best_threshold),
    )

    payload = asdict(result)
    payload.update(
        {
            "model_name": "pytorch_hetero_gat",
            "loader_backend": "full_batch_hetero",
            "device": str(resolved_device),
            "feature_dimension": int(normalized_data["account"].x.shape[1]),
            "merchant_feature_dimension": int(normalized_data["merchant"].x.shape[1]),
            "edge_feature_dimension": int(normalized_data["account", "transfers", "account"].edge_attr.shape[1]),
            "feature_columns": list(
                feature_columns
                or [f"account_x_{index}" for index in range(int(normalized_data["account"].x.shape[1]))]
            ),
            "merchant_feature_columns": list(
                merchant_feature_columns
                or [f"merchant_x_{index}" for index in range(int(normalized_data["merchant"].x.shape[1]))]
            ),
            "edge_feature_columns": list(edge_feature_columns or ["amount", "event_time", "transaction_type_code"]),
            "checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
            "batch_size": 0,
            "fan_out": [],
            "pos_weight_multiplier": float(pos_weight_multiplier),
            "decision_threshold": float(best_threshold),
            "configured_decision_threshold": float(decision_threshold),
        }
    )
    return payload


def train_pytorch_gcn(
    data: HeteroData,
    *,
    feature_columns: Sequence[str] | None = None,
    merchant_feature_columns: Sequence[str] | None = None,
    edge_feature_columns: Sequence[str] | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 1024,
    fan_out: Sequence[int] = (25, 10),
    num_workers: int = 0,
    device: torch.device | None = None,
    pin_memory: bool = True,
    checkpoint_path: str | Path | None = None,
    pos_weight_multiplier: float = 1.0,
    decision_threshold: float = 0.5,
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, float | int | None | str | list[str]]:
    return train_pyg_minibatch(
        data,
        feature_columns=feature_columns,
        merchant_feature_columns=merchant_feature_columns,
        edge_feature_columns=edge_feature_columns,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        fan_out=fan_out,
        num_workers=num_workers,
        device=device,
        pin_memory=pin_memory,
        checkpoint_path=checkpoint_path,
        pos_weight_multiplier=pos_weight_multiplier,
        decision_threshold=decision_threshold,
        random_state=random_state,
        test_size=test_size,
    )
