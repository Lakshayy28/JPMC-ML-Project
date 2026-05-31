from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_graph_feature_bundle
from fri.temporal.drift import select_distribution_feature_columns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate feature drift against the running FRI API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for the running API service.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="Number of recent feature rows to send to the drift endpoint.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=5.0,
        help="Multiplicative surge factor applied to the drifted behavioral features.",
    )
    return parser.parse_args()


def _load_recent_feature_payload(sample_size: int, multiplier: float) -> list[dict[str, Any]]:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample
    archive_data = load_archive_graph_data(archive_path)
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

    feature_frame = feature_bundle.get("tabular_account_features")
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        raise RuntimeError("Expected non-empty tabular_account_features from the archive feature bundle")

    feature_columns = select_distribution_feature_columns(feature_frame)
    payload_frame = feature_frame[feature_columns].head(sample_size).copy()
    if payload_frame.empty:
        raise RuntimeError("No recent feature rows were available for drift simulation")

    for column in ("outgoing_tx_velocity_30d", "total_amount"):
        if column not in payload_frame.columns:
            raise KeyError(f"Expected drift simulation column to exist: {column}")
        payload_frame.loc[:, column] = pd.to_numeric(payload_frame[column], errors="coerce").fillna(0.0) * multiplier

    return payload_frame.to_dict(orient="records")


def main() -> int:
    args = _parse_args()
    payload = _load_recent_feature_payload(sample_size=args.sample_size, multiplier=args.multiplier)
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/analyze-drift",
        json=payload,
        timeout=None,
    )

    print(f"POST /analyze-drift -> {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    response.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())