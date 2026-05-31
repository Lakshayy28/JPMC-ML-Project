from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle
from fri.models.graph_baseline import train_graph_node_baselines


def main() -> None:
    print("--> Loading graph baseline settings...", flush=True)
    settings = load_settings()

    print(f"--> Loading archive graph data from: {settings.graph.archive_sample}", flush=True)
    archive_data = load_archive_graph_data(settings.graph.archive_sample)

    print("--> Building shared 20K archive node feature bundle...", flush=True)
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

    print("--> Training graph classical baseline estimators...", flush=True)
    metrics = train_graph_node_baselines(
        node_features,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        verbose=True,
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph_baseline_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Wrote execution metrics output to: {output_path}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
