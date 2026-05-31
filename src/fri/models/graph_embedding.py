from __future__ import annotations

import pandas as pd

from fri.models.baseline import train_binary_models


def train_embedding_and_combined_baselines(
    feature_bundle: dict[str, pd.DataFrame],
    *,
    label_column: str = "is_fraud",
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    node_features = feature_bundle["node_features"]
    embeddings = feature_bundle["embeddings"]
    combined = feature_bundle["combined"]

    if label_column not in node_features.columns:
        raise KeyError(f"Expected label column '{label_column}' in node features")

    embedding_frame = embeddings.merge(node_features[["node_id", label_column]], on="node_id", how="left")
    embedding_frame["label"] = embedding_frame[label_column].astype(int)
    combined_frame = combined.copy()
    combined_frame["label"] = combined_frame[label_column].astype(int)

    return {
        "embedding_only": train_binary_models(
            embedding_frame,
            target_column="label",
            id_columns=("node_id", label_column),
            random_state=random_state,
            test_size=test_size,
        ),
        "combined_graph_features_and_embeddings": train_binary_models(
            combined_frame,
            target_column="label",
            id_columns=("node_id", label_column, "fraud_step"),
            random_state=random_state,
            test_size=test_size,
        ),
    }
