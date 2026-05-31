from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
from torch_geometric.utils import k_hop_subgraph, to_undirected

from fri.data.loaders import stream_archive_graph_nodes, stream_archive_graph_transactions


@dataclass(frozen=True)
class PYGGraphBundle:
    data: Data
    feature_columns: tuple[str, ...]


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


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class _FallbackNeighborLoader:
    def __init__(
        self,
        data: Data,
        input_nodes: torch.Tensor,
        *,
        batch_size: int,
        fan_out: Sequence[int],
        shuffle: bool,
    ) -> None:
        self.data = data
        self.input_nodes = input_nodes.cpu()
        self.batch_size = batch_size
        self.num_hops = len(fan_out)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Data]:
        if self.shuffle:
            order = torch.randperm(self.input_nodes.numel())
            nodes = self.input_nodes[order]
        else:
            nodes = self.input_nodes

        for seed_nodes in torch.split(nodes, self.batch_size):
            subset, edge_index, mapping, _ = k_hop_subgraph(
                seed_nodes,
                self.num_hops,
                self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes,
            )
            batch = Data(
                x=self.data.x[subset],
                edge_index=edge_index,
                y=self.data.y[subset],
                n_id=subset,
                node_id=self.data.node_id[subset],
            )
            batch.batch_size = int(seed_nodes.numel())
            batch.seed_node_index = mapping
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(self.input_nodes.numel() / self.batch_size))


def resolve_training_device(*, use_cuda: bool = True, requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    requested = torch.device(requested_device)
    if requested.type == "cuda" and (not use_cuda or not torch.cuda.is_available()):
        return torch.device("cpu")
    return requested


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], *, label: str) -> None:
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"{label} is missing required columns: {missing_display}")


def _collect_node_frame(node_chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in node_chunks:
        _validate_columns(chunk, {"nodeid", "isFraud", "init_balance", "fraudStep"}, label="Archive node chunk")
        frames.append(
            chunk[["nodeid", "isFraud", "init_balance", "fraudStep"]].rename(
                columns={
                    "nodeid": "node_id",
                    "isFraud": "is_fraud",
                    "init_balance": "initial_balance",
                    "fraudStep": "fraud_step",
                }
            )
        )

    if not frames:
        raise ValueError("No node chunks were available to build the PyG graph bundle")

    node_frame = pd.concat(frames, ignore_index=True)
    if node_frame["node_id"].duplicated().any():
        raise ValueError("Archive nodes contain duplicate node identifiers")

    return node_frame.sort_values("node_id").reset_index(drop=True)


def _build_pyg_graph_bundle(node_frame: pd.DataFrame, transaction_chunks: Iterable[pd.DataFrame]) -> PYGGraphBundle:
    node_ids = node_frame["node_id"].astype(int).to_numpy(copy=True)
    num_nodes = len(node_ids)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}

    in_counts = np.zeros(num_nodes, dtype=np.float32)
    out_counts = np.zeros(num_nodes, dtype=np.float32)
    in_amounts = np.zeros(num_nodes, dtype=np.float32)
    out_amounts = np.zeros(num_nodes, dtype=np.float32)
    first_in_time = np.full(num_nodes, np.inf, dtype=np.float32)
    last_in_time = np.full(num_nodes, -np.inf, dtype=np.float32)
    first_out_time = np.full(num_nodes, np.inf, dtype=np.float32)
    last_out_time = np.full(num_nodes, -np.inf, dtype=np.float32)
    edge_sources: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []

    for chunk in transaction_chunks:
        _validate_columns(chunk, {"sourceNodeId", "targetNodeId", "value", "time"}, label="Archive transaction chunk")
        work = chunk[["sourceNodeId", "targetNodeId", "value", "time"]].copy()
        work = work.dropna(subset=["sourceNodeId", "targetNodeId", "value", "time"])
        if work.empty:
            continue

        source_index = work["sourceNodeId"].map(node_index)
        target_index = work["targetNodeId"].map(node_index)
        valid_mask = source_index.notna() & target_index.notna()
        if not valid_mask.any():
            continue

        source_nodes = source_index[valid_mask].astype(np.int64).to_numpy()
        target_nodes = target_index[valid_mask].astype(np.int64).to_numpy()
        values = pd.to_numeric(work.loc[valid_mask, "value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        times = pd.to_numeric(work.loc[valid_mask, "time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        edge_sources.append(source_nodes)
        edge_targets.append(target_nodes)
        np.add.at(out_counts, source_nodes, 1.0)
        np.add.at(in_counts, target_nodes, 1.0)
        np.add.at(out_amounts, source_nodes, values)
        np.add.at(in_amounts, target_nodes, values)
        np.minimum.at(first_out_time, source_nodes, times)
        np.maximum.at(last_out_time, source_nodes, times)
        np.minimum.at(first_in_time, target_nodes, times)
        np.maximum.at(last_in_time, target_nodes, times)

    if not edge_sources:
        raise ValueError("No valid archive transactions were available to build the PyG graph bundle")

    directed_edge_index = torch.tensor(
        np.vstack([np.concatenate(edge_sources), np.concatenate(edge_targets)]),
        dtype=torch.long,
    )
    edge_index = to_undirected(directed_edge_index, num_nodes=num_nodes)

    in_amount_mean = np.divide(in_amounts, in_counts, out=np.zeros_like(in_amounts), where=in_counts > 0)
    out_amount_mean = np.divide(out_amounts, out_counts, out=np.zeros_like(out_amounts), where=out_counts > 0)
    in_time_span = np.where(np.isfinite(first_in_time), np.maximum(last_in_time - first_in_time, 0.0), 0.0)
    out_time_span = np.where(np.isfinite(first_out_time), np.maximum(last_out_time - first_out_time, 0.0), 0.0)
    total_counts = in_counts + out_counts
    total_amounts = in_amounts + out_amounts
    net_out_amount = out_amounts - in_amounts

    feature_columns = (
        "initial_balance",
        "in_transaction_count",
        "out_transaction_count",
        "in_amount_total",
        "out_amount_total",
        "in_amount_mean",
        "out_amount_mean",
        "in_time_span",
        "out_time_span",
        "total_transaction_count",
        "total_amount",
        "net_out_amount",
    )
    x = np.column_stack(
        [
            node_frame["initial_balance"].astype(np.float32).to_numpy(),
            in_counts,
            out_counts,
            in_amounts,
            out_amounts,
            in_amount_mean,
            out_amount_mean,
            in_time_span.astype(np.float32),
            out_time_span.astype(np.float32),
            total_counts,
            total_amounts,
            net_out_amount,
        ]
    ).astype(np.float32)
    y = node_frame["is_fraud"].astype(np.int64).to_numpy(copy=True)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        node_id=torch.from_numpy(node_ids.astype(np.int64)),
        fraud_step=torch.from_numpy(node_frame["fraud_step"].astype(np.int64).to_numpy(copy=True)),
        num_nodes=num_nodes,
    )
    return PYGGraphBundle(data=data, feature_columns=feature_columns)


def build_pyg_graph_data_from_tables(nodes: pd.DataFrame, transactions: pd.DataFrame) -> PYGGraphBundle:
    node_frame = _collect_node_frame([nodes])
    return _build_pyg_graph_bundle(node_frame, [transactions])


def build_pyg_graph_data_from_archive(archive_path: str | Path, *, chunksize: int = 100_000) -> PYGGraphBundle:
    node_frame = _collect_node_frame(stream_archive_graph_nodes(archive_path, chunksize=chunksize))
    return _build_pyg_graph_bundle(
        node_frame,
        stream_archive_graph_transactions(archive_path, chunksize=chunksize),
    )


def _make_neighbor_loader(
    data: Data,
    *,
    input_nodes: torch.Tensor,
    batch_size: int,
    fan_out: Sequence[int],
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> tuple[object, str]:
    if WITH_PYG_LIB or WITH_TORCH_SPARSE:
        return (
            NeighborLoader(
                data,
                num_neighbors=list(fan_out),
                input_nodes=input_nodes,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
            ),
            "neighbor_loader",
        )
    return (
        _FallbackNeighborLoader(
            data,
            input_nodes,
            batch_size=batch_size,
            fan_out=fan_out,
            shuffle=shuffle,
        ),
        "k_hop_fallback",
    )


def _normalize_node_features(data: Data, train_indices: np.ndarray) -> Data:
    normalized = copy.copy(data)
    train_x = normalized.x[train_indices]
    means = train_x.mean(dim=0)
    stds = train_x.std(dim=0, unbiased=False)
    stds[stds == 0] = 1.0
    normalized.x = (normalized.x - means) / stds
    return normalized


def _seed_node_index(batch: Data) -> torch.Tensor:
    if hasattr(batch, "seed_node_index"):
        return batch.seed_node_index
    return torch.arange(batch.batch_size, device=batch.x.device)


def _evaluate_minibatch(
    model: GraphSAGE,
    loader: object,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    weights: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            seed_index = _seed_node_index(batch)
            logits = model(batch.x, batch.edge_index)[seed_index]
            seed_labels = batch.y[seed_index]
            loss = criterion(logits, seed_labels)
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = seed_labels.cpu().numpy()

            labels.append(batch_labels)
            probabilities.append(batch_probabilities)
            predictions.append(batch_predictions)
            losses.append(float(loss.item()))
            weights.append(int(seed_labels.numel()))

    loss_value = float(np.average(losses, weights=weights)) if losses else 0.0
    return (
        np.concatenate(labels) if labels else np.array([], dtype=np.int64),
        np.concatenate(probabilities) if probabilities else np.array([], dtype=np.float32),
        np.concatenate(predictions) if predictions else np.array([], dtype=np.int64),
        loss_value,
    )


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


def train_pyg_minibatch(
    data: Data,
    *,
    feature_columns: Sequence[str] | None = None,
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
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, float | int | None | str | list[str]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    resolved_device = device or torch.device("cpu")
    labels_array = data.y.cpu().numpy()
    train_idx, val_idx, test_idx = _split_indices(labels_array, test_size=test_size, random_state=random_state)
    normalized_data = _normalize_node_features(data, train_idx)
    if pin_memory and resolved_device.type == "cuda":
        normalized_data = normalized_data.pin_memory()

    train_index_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_index_tensor = torch.tensor(val_idx, dtype=torch.long)
    test_index_tensor = torch.tensor(test_idx, dtype=torch.long)
    train_loader, loader_backend = _make_neighbor_loader(
        normalized_data,
        input_nodes=train_index_tensor,
        batch_size=batch_size,
        fan_out=fan_out,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory and resolved_device.type == "cuda",
    )
    val_loader, _ = _make_neighbor_loader(
        normalized_data,
        input_nodes=val_index_tensor,
        batch_size=batch_size,
        fan_out=fan_out,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory and resolved_device.type == "cuda",
    )
    test_loader, _ = _make_neighbor_loader(
        normalized_data,
        input_nodes=test_index_tensor,
        batch_size=batch_size,
        fan_out=fan_out,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory and resolved_device.type == "cuda",
    )

    class_counts = np.bincount(labels_array[train_idx], minlength=2)
    negative_count = max(int(class_counts[0]), 1)
    positive_count = max(int(class_counts[1]), 1)
    class_weights = torch.tensor([1.0, negative_count / positive_count], dtype=torch.float32, device=resolved_device)

    model = GraphSAGE(
        input_dim=int(normalized_data.num_node_features),
        hidden_dim=hidden_dim,
        output_dim=2,
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
    best_selection_score = float("-inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(resolved_device, non_blocking=True)
            seed_index = _seed_node_index(batch)
            logits = model(batch.x, batch.edge_index)[seed_index]
            labels = batch.y[seed_index]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_labels, val_probabilities, _, val_loss = _evaluate_minibatch(
            model,
            val_loader,
            criterion,
            device=resolved_device,
        )
        selection_score, val_average_precision = _selection_score(val_labels, val_probabilities, val_loss)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_ap = val_average_precision
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_file is not None:
                torch.save({"model_state_dict": best_state, "best_epoch": best_epoch}, checkpoint_file)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint_file is not None and checkpoint_file.exists():
        checkpoint_payload = torch.load(checkpoint_file, map_location=resolved_device)
        model.load_state_dict(checkpoint_payload["model_state_dict"])

    test_labels, test_probabilities, test_predictions, _ = _evaluate_minibatch(
        model,
        test_loader,
        criterion,
        device=resolved_device,
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
    )

    payload = asdict(result)
    payload.update(
        {
            "model_name": "pytorch_graphsage",
            "loader_backend": loader_backend,
            "device": str(resolved_device),
            "feature_dimension": int(normalized_data.num_node_features),
            "feature_columns": list(feature_columns or [f"x_{index}" for index in range(int(normalized_data.num_node_features))]),
            "checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
            "batch_size": int(batch_size),
            "fan_out": [int(value) for value in fan_out],
        }
    )
    return payload


def train_pytorch_gcn(
    data: Data,
    *,
    feature_columns: Sequence[str] | None = None,
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
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, float | int | None | str | list[str]]:
    return train_pyg_minibatch(
        data,
        feature_columns=feature_columns,
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
        random_state=random_state,
        test_size=test_size,
    )
