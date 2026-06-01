from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle
from fri.models.graph_embedding import train_embedding_and_combined_baselines


def main() -> None:
    print("--> Loading graph embedding baseline settings...", flush=True)
    settings = load_settings()

    print(f"--> Loading archive graph data from: {settings.graph.archive_sample}", flush=True)
    archive_data = load_archive_graph_data(settings.graph.archive_sample)

    print("--> Generating unified archive graph embeddings and temporal feature bundle...", flush=True)
    feature_bundle = build_archive_feature_bundle(
        archive_data.nodes,
        archive_data.transactions,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
        embedding_dimensions=settings.graph.embedding_dimensions,
        embedding_random_state=settings.models.random_state,
        include_embeddings=True,
    )

    print("--> Training graph embedding baseline estimators...", flush=True)
    metrics = train_embedding_and_combined_baselines(
        feature_bundle,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        verbose=True,
    )

    output_dir = settings.graph.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "graph_embedding_metrics.json"
    embeddings_path = output_dir / "archive_node_embeddings.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_bundle["embeddings"].to_csv(embeddings_path, index=False)

    print(f"[SUCCESS] Wrote execution metrics output to: {metrics_path}", flush=True)
    print(f"[SUCCESS] Wrote graph node embeddings to: {embeddings_path}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
