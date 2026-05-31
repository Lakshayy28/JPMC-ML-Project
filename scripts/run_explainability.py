from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.explainability.service import HeteroGraphExplainerService
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_graph_feature_bundle
from fri.models.pytorch_gnn import (
    build_pyg_graph_data_from_feature_bundle,
    load_trained_hetero_gat_model,
    prepare_hetero_inference_data,
    resolve_training_device,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explainability for the highest-risk account in the trained hetero GAT model.")
    parser.add_argument("--explainer-epochs", type=int, default=200, help="Number of GNNExplainer optimization epochs.")
    parser.add_argument("--top-k-features", type=int, default=5, help="Number of top feature attributions to retain.")
    parser.add_argument("--top-k-edges", type=int, default=5, help="Number of top structural edge attributions to retain.")
    return parser.parse_args()


def _load_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file does not exist: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _merchant_node_ids(feature_bundle: dict[str, object]) -> list[str]:
    merchant_features = feature_bundle.get("merchant_features")
    if not isinstance(merchant_features, pd.DataFrame) or "merchant_id" not in merchant_features.columns:
        return []
    return merchant_features["merchant_id"].astype(str).tolist()


def _raw_edge_records(feature_bundle: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}

    transfer_frame = feature_bundle.get("normalized_transactions")
    if isinstance(transfer_frame, pd.DataFrame) and not transfer_frame.empty:
        records["account->transfers->account"] = transfer_frame[
            ["transaction_id", "source_node_id", "target_node_id", "amount", "event_time", "transaction_type_code"]
        ].to_dict("records")

    merchant_links = feature_bundle.get("merchant_links")
    if isinstance(merchant_links, pd.DataFrame) and not merchant_links.empty:
        merchant_records = merchant_links[
            ["transaction_id", "source_node_id", "merchant_id", "amount", "event_time", "transaction_type_code"]
        ].to_dict("records")
        records["account->buys_from->merchant"] = merchant_records
        records["merchant->rev_buys_from->account"] = merchant_records

    return records


def _print_summary(report: dict[str, object]) -> None:
    print(f"Explained account node id: {report['node_id']}")
    print(f"Predicted fraud probability: {float(report['risk_score']):.4f}")
    print("Top 3 contributing features:")
    top_node_features = report.get("top_node_features", {})
    if isinstance(top_node_features, dict):
        for feature_name, score in list(top_node_features.items())[:3]:
            print(f"  - {feature_name}: {float(score):.4f}")

    print("Top contributing structural edges:")
    critical_edges = report.get("critical_edges", [])
    if isinstance(critical_edges, list) and critical_edges:
        for edge in critical_edges[:3]:
            if not isinstance(edge, dict):
                continue
            print(
                "  - "
                f"{edge.get('relation', 'unknown')} | "
                f"{edge.get('source_node_id', edge.get('source_index'))} -> "
                f"{edge.get('target_node_id', edge.get('target_index'))} | "
                f"importance={float(edge.get('importance', 0.0)):.4f}"
            )
    else:
        print("  - No critical structural edges were returned by the explainer")


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample
    archive_data = load_archive_graph_data(archive_path)

    print(f"--> Loading unified 20K archive graph bundle from: {archive_path}", flush=True)
    feature_bundle = build_graph_feature_bundle(
        nodes=archive_data.nodes,
        transactions=archive_data.transactions,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
        embedding_dimensions=settings.graph.embedding_dimensions,
        embedding_random_state=settings.models.random_state,
        include_embeddings=True,
    )
    graph_bundle = build_pyg_graph_data_from_feature_bundle(feature_bundle)

    output_dir = settings.graph.output_root
    metrics_path = output_dir / "pytorch_gcn_metrics.json"
    checkpoint_path = output_dir / settings.gnn.checkpoint_name
    metrics = _load_metrics(metrics_path)

    device = resolve_training_device(
        use_cuda=settings.hardware.use_cuda,
        requested_device=settings.hardware.device,
    )
    normalized_data = prepare_hetero_inference_data(
        graph_bundle.data,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        device=device,
        pin_memory=settings.hardware.pin_memory,
    )
    model, _ = load_trained_hetero_gat_model(
        normalized_data,
        checkpoint_path,
        hidden_dim=settings.gnn.hidden_dim,
        dropout=settings.gnn.dropout,
        device=device,
    )

    print(f"--> Running forward pass on device: {device}", flush=True)
    with torch.no_grad():
        logits = model(normalized_data)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    account_index = int(torch.argmax(probabilities).detach().cpu().item())
    highest_risk_score = float(probabilities[account_index].detach().cpu().item())
    print(
        f"--> Highest predicted fraud probability is {highest_risk_score:.4f} for account index {account_index}",
        flush=True,
    )

    account_node_ids = [int(value) for value in normalized_data["account"].node_id.detach().cpu().tolist()]
    merchant_node_ids = _merchant_node_ids(feature_bundle)
    service = HeteroGraphExplainerService(
        model,
        feature_columns=list(metrics.get("feature_columns", [])),
        account_node_ids=account_node_ids,
        merchant_node_ids=merchant_node_ids,
        raw_edge_records=_raw_edge_records(feature_bundle),
        top_k_features=args.top_k_features,
        top_k_edges=args.top_k_edges,
        explainer_epochs=args.explainer_epochs,
    )

    print("--> Generating explanation report with PyG Explainer...", flush=True)
    report = service.explain_account(normalized_data, account_index, epochs=args.explainer_epochs)
    output_path = output_dir / "account_explanation_sample.json"
    output_payload = asdict(report)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Wrote account explanation sample to: {output_path}", flush=True)
    _print_summary(output_payload)


if __name__ == "__main__":
    main()