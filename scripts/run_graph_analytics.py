from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.analytics import compute_graph_summary, compute_node_features, top_nodes_by_metric
from fri.graph.builder import build_archive_transaction_graph
from fri.graph.io import load_archive_graph_data


def main() -> None:
    settings = load_settings()
    archive_data = load_archive_graph_data(settings.graph.archive_sample)
    graph = build_archive_transaction_graph(archive_data.nodes, archive_data.transactions)
    node_features = compute_node_features(
        graph,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
    )

    summary = compute_graph_summary(graph)
    summary.update(
        {
            "archive_sample": str(settings.graph.archive_sample),
            "sample_name": archive_data.sample_name,
            "metadata": archive_data.metadata,
            "top_pagerank_nodes": top_nodes_by_metric(node_features, "pagerank", top_n=10),
            "top_out_degree_nodes": top_nodes_by_metric(node_features, "out_degree", top_n=10),
        }
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "archive_graph_summary.json"
    node_features_path = output_dir / "archive_node_features.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    node_features.to_csv(node_features_path, index=False)

    print(f"Wrote graph summary to: {summary_path}")
    print(f"Wrote node features to: {node_features_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
