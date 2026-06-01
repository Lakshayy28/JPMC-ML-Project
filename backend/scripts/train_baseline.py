from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_archive_feature_bundle
from fri.models.baseline import train_binary_models


def main() -> None:
    print("--> Loading tabular baseline settings...", flush=True)
    settings = load_settings()

    print(f"--> Loading unified archive graph data from: {settings.graph.archive_sample}", flush=True)
    archive_data = load_archive_graph_data(settings.graph.archive_sample)

    print("--> Flattening 20K archive transaction attributes into account-level tabular features...", flush=True)
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
    tabular_features = feature_bundle["tabular_account_features"].copy()
    tabular_features["label"] = tabular_features["is_fraud"].astype(int)

    print("--> Training tabular baseline estimators...", flush=True)
    metrics = train_binary_models(
        tabular_features,
        target_column="label",
        id_columns=("node_id", "is_fraud", "fraud_step"),
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        verbose=True,
        run_label="archive_account_tabular",
    )

    output_dir = REPO_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Wrote execution metrics output to: {output_file}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
