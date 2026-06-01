"""
FRI Applied Scientist Audit: Standalone GNN Training Script
============================================================
This script replicates the exact backend training pipeline using
the real AMLSim 20K_fanin200cycle200 archive data.

It exercises:
1. Archive data loading (io.load_archive_graph_data)
2. Feature engineering (service.build_archive_feature_bundle)
3. PyG hetero graph construction (pytorch_gnn.build_pyg_graph_data_from_archive)
4. SpatialTemporalHeteroGAT training (pytorch_gnn.train_pytorch_gcn)
5. Tabular baseline training (baseline.train_binary_models)

The results should match the existing backend metrics at:
  artifacts/graph/pytorch_gcn_metrics.json
  artifacts/baseline_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle
from fri.models.baseline import train_binary_models
from fri.models.pytorch_gnn import (
    build_pyg_graph_data_from_archive,
    resolve_training_device,
    train_pytorch_gcn,
)


def print_section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_metrics_table(metrics: dict, title: str) -> None:
    print(f"\n  --- {title} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>40s}: {value:.4f}")
        elif isinstance(value, (int, str)):
            print(f"  {key:>40s}: {value}")


def main() -> None:
    start_time = time.time()

    # ── Load Settings ────────────────────────────────────────────────
    print_section("1. LOADING SETTINGS")
    settings = load_settings()
    print(f"  Archive path   : {settings.dataset.graph_archive}")
    print(f"  Temporal windows: {settings.temporal.windows}")
    print(f"  GNN hidden_dim : {settings.gnn.hidden_dim}")
    print(f"  GNN epochs     : {settings.gnn.epochs}")
    print(f"  GNN patience   : {settings.gnn.patience}")
    print(f"  GNN LR         : {settings.gnn.learning_rate}")
    print(f"  GNN dropout    : {settings.gnn.dropout}")
    print(f"  pos_weight_mult: {settings.gnn.pos_weight_multiplier}")
    print(f"  Test size      : {settings.models.test_size}")
    print(f"  Random state   : {settings.models.random_state}")

    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample

    # ── Load Archive Data ────────────────────────────────────────────
    print_section("2. LOADING ARCHIVE DATA")
    archive_data = load_archive_graph_data(archive_path)
    print(f"  Archive        : {archive_data.sample_name}")
    print(f"  Nodes          : {len(archive_data.nodes)}")
    print(f"  Transactions   : {len(archive_data.transactions)}")
    print(f"  Metadata       : {archive_data.metadata}")

    # ── Feature Engineering ──────────────────────────────────────────
    print_section("3. FEATURE ENGINEERING (build_archive_feature_bundle)")
    feature_bundle = build_archive_feature_bundle(
        archive_data.nodes,
        archive_data.transactions,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
        include_embeddings=False,
    )
    node_features = feature_bundle["node_features"]
    merchant_features = feature_bundle["merchant_features"]
    normalized_transactions = feature_bundle["normalized_transactions"]
    merchant_links = feature_bundle["merchant_links"]
    tabular_features = feature_bundle["tabular_account_features"]

    print(f"  Account node features shape : {node_features.shape}")
    print(f"  Merchant features shape     : {merchant_features.shape}")
    print(f"  Normalized txns shape       : {normalized_transactions.shape}")
    print(f"  Merchant links shape        : {merchant_links.shape}")

    account_feature_cols = [
        c for c in node_features.columns
        if c not in {"node_id", "is_fraud", "fraud_step"}
    ]
    merchant_feature_cols = [
        c for c in merchant_features.columns
        if c != "merchant_id"
    ]
    print(f"  Account feature count       : {len(account_feature_cols)}")
    print(f"  Merchant feature count      : {len(merchant_feature_cols)}")
    print(f"  Account feature columns     :")
    for col in account_feature_cols:
        print(f"    - {col}")

    # ── Tabular Baselines ────────────────────────────────────────────
    print_section("4. TABULAR BASELINE TRAINING")
    tabular_frame = tabular_features.copy()
    tabular_frame["label"] = tabular_frame["is_fraud"].astype(int)

    baseline_metrics = train_binary_models(
        tabular_frame,
        target_column="label",
        id_columns=("node_id", "is_fraud", "fraud_step"),
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        verbose=True,
        run_label="archive_account_tabular",
    )
    for model_name, result in baseline_metrics.items():
        print_metrics_table(result, f"Tabular Baseline: {model_name}")

    # ── Build PyG HeteroData ─────────────────────────────────────────
    print_section("5. BUILDING PyG HETEROGENEOUS GRAPH")
    graph_bundle = build_pyg_graph_data_from_archive(
        archive_path,
        chunksize=settings.dataset.archive_chunksize,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
    )
    pyg_data = graph_bundle.data
    print(f"  HeteroData: {pyg_data}")
    print(f"  Node types: {pyg_data.node_types}")
    print(f"  Edge types: {pyg_data.edge_types}")
    print(f"  Account features shape : {pyg_data['account'].x.shape}")
    print(f"  Merchant features shape: {pyg_data['merchant'].x.shape}")
    print(f"  Account labels shape   : {pyg_data['account'].y.shape}")
    print(f"  Feature columns        : {len(graph_bundle.feature_columns)} dims")
    print(f"  Merchant feature cols  : {len(graph_bundle.merchant_feature_columns)} dims")
    print(f"  Edge feature cols      : {graph_bundle.edge_feature_columns}")

    for et in pyg_data.edge_types:
        ei = pyg_data[et].edge_index
        ea = pyg_data[et].edge_attr
        print(f"  Edge type {et}: {ei.shape[1]} edges, attr shape {ea.shape}")

    # Label distribution
    labels = pyg_data["account"].y.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} ({c / len(labels) * 100:.1f}%)")

    # ── Train Hetero GAT ─────────────────────────────────────────────
    print_section("6. TRAINING SpatialTemporalHeteroGAT (FULL-BATCH)")
    device = resolve_training_device(
        use_cuda=settings.hardware.use_cuda,
        requested_device=settings.hardware.device,
    )
    print(f"  Training device: {device}")

    output_dir = REPO_ROOT / "artifacts" / "graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "audit_hetero_gat_model.pt"

    gnn_metrics = train_pytorch_gcn(
        graph_bundle.data,
        feature_columns=graph_bundle.feature_columns,
        merchant_feature_columns=graph_bundle.merchant_feature_columns,
        edge_feature_columns=graph_bundle.edge_feature_columns,
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
        pos_weight_multiplier=settings.gnn.pos_weight_multiplier,
        decision_threshold=settings.gnn.decision_threshold,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
    )

    # ── Results ──────────────────────────────────────────────────────
    print_section("7. FINAL RESULTS")
    print_metrics_table(
        {
            "Precision": gnn_metrics.get("precision"),
            "Recall": gnn_metrics.get("recall"),
            "F1-Score": gnn_metrics.get("f1"),
            "PR-AUC": gnn_metrics.get("average_precision"),
            "ROC-AUC": gnn_metrics.get("roc_auc"),
            "Best Epoch": gnn_metrics.get("best_epoch"),
            "Best Val F1": gnn_metrics.get("best_validation_f1"),
            "Best Val PR-AUC": gnn_metrics.get("best_validation_average_precision"),
            "Optimal Threshold": gnn_metrics.get("optimal_threshold"),
            "Train Rows": gnn_metrics.get("train_rows"),
            "Val Rows": gnn_metrics.get("validation_rows"),
            "Test Rows": gnn_metrics.get("test_rows"),
            "Feature Dim": gnn_metrics.get("feature_dimension"),
            "Merchant Feature Dim": gnn_metrics.get("merchant_feature_dimension"),
            "Edge Feature Dim": gnn_metrics.get("edge_feature_dimension"),
            "Device": gnn_metrics.get("device"),
        },
        "SpatialTemporalHeteroGAT Test Metrics",
    )

    # Save audit results
    audit_output_path = output_dir / "audit_gnn_metrics.json"
    audit_output_path.write_text(json.dumps(gnn_metrics, indent=2), encoding="utf-8")
    print(f"\n  Wrote audit GNN metrics to: {audit_output_path}")
    print(f"  Wrote checkpoint to: {checkpoint_path}")

    # ── Compare with existing backend results ────────────────────────
    existing_metrics_path = output_dir / "pytorch_gcn_metrics.json"
    if existing_metrics_path.exists():
        print_section("8. COMPARISON WITH EXISTING BACKEND RESULTS")
        existing = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
        compare_keys = ["average_precision", "precision", "recall", "f1", "roc_auc"]
        print(f"  {'Metric':<25s} {'Existing':>12s} {'Audit':>12s} {'Delta':>12s}")
        print(f"  {'-' * 61}")
        for key in compare_keys:
            ev = existing.get(key)
            av = gnn_metrics.get(key)
            if ev is not None and av is not None:
                delta = av - ev
                print(f"  {key:<25s} {ev:>12.4f} {av:>12.4f} {delta:>+12.4f}")
            else:
                print(f"  {key:<25s} {'n/a':>12s} {'n/a':>12s} {'n/a':>12s}")

    elapsed = time.time() - start_time
    print(f"\n  Total elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
