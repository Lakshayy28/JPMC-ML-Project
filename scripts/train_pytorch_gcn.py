from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.models.pytorch_gnn import build_pyg_graph_data_from_archive, resolve_training_device, train_pyg_minibatch


def main() -> None:
    settings = load_settings()
    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample
    graph_bundle = build_pyg_graph_data_from_archive(
        archive_path,
        chunksize=settings.dataset.archive_chunksize,
    )
    device = resolve_training_device(
        use_cuda=settings.hardware.use_cuda,
        requested_device=settings.hardware.device,
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / settings.gnn.checkpoint_name
    metrics = train_pyg_minibatch(
        graph_bundle.data,
        feature_columns=graph_bundle.feature_columns,
        hidden_dim=settings.gnn.hidden_dim,
        dropout=settings.gnn.dropout,
        learning_rate=settings.gnn.learning_rate,
        weight_decay=settings.gnn.weight_decay,
        epochs=settings.gnn.epochs,
        patience=settings.gnn.patience,
        batch_size=settings.gnn.loader_batch_size,
        fan_out=settings.gnn.loader_fan_out,
        num_workers=settings.gnn.loader_num_workers,
        device=device,
        pin_memory=settings.hardware.pin_memory,
        checkpoint_path=checkpoint_path,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
    )

    output_path = output_dir / "pytorch_gcn_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Wrote PyTorch GNN metrics to: {output_path}")
    print(f"Wrote checkpoint to: {checkpoint_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
