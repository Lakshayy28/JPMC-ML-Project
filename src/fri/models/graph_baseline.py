from __future__ import annotations

import pandas as pd

from fri.models.baseline import train_binary_models


def train_graph_node_baselines(
    node_features: pd.DataFrame,
    *,
    label_column: str = "is_fraud",
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, dict[str, float | int | None]]:
    if label_column not in node_features.columns:
        raise KeyError(f"Expected label column '{label_column}' in node feature frame")

    frame = node_features.copy()
    frame["label"] = frame[label_column].astype(int)
    drop_columns = [column for column in ["node_id", label_column, "fraud_step"] if column in frame.columns]

    return train_binary_models(
        frame,
        target_column="label",
        id_columns=tuple(drop_columns),
        random_state=random_state,
        test_size=test_size,
    )
