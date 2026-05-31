from __future__ import annotations

import hashlib
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from fri.graph.analytics import compute_node_features
from fri.graph.builder import build_archive_transaction_graph
from fri.graph.embeddings import compute_spectral_node_embeddings


def _stable_bucket(value: str, seed: int, pool_size: int, prefix: str) -> str:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    digest = hashlib.sha256(f"{seed}:{prefix}:{value}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % pool_size
    return f"{prefix}_{bucket:03d}"


def _fill_numeric_na(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].fillna(0.0)
    return frame


def normalize_archive_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"nodeid", "isFraud", "init_balance", "fraudStep"}
    missing_columns = required_columns.difference(nodes.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise KeyError(f"Archive nodes are missing required columns: {missing_display}")

    frame = nodes[["nodeid", "isFraud", "init_balance", "fraudStep"]].rename(
        columns={
            "nodeid": "node_id",
            "isFraud": "is_fraud",
            "init_balance": "initial_balance",
            "fraudStep": "fraud_step",
        }
    ).copy()
    frame["node_id"] = pd.to_numeric(frame["node_id"], errors="coerce").fillna(-1).astype(int)
    frame["is_fraud"] = pd.to_numeric(frame["is_fraud"], errors="coerce").fillna(0).astype(int)
    frame["initial_balance"] = pd.to_numeric(frame["initial_balance"], errors="coerce").fillna(0.0)
    frame["fraud_step"] = pd.to_numeric(frame["fraud_step"], errors="coerce").fillna(-1).astype(int)
    frame = frame.drop_duplicates(subset=["node_id"]).sort_values("node_id").reset_index(drop=True)
    return frame


def normalize_archive_transactions(transactions: pd.DataFrame, node_ids: set[int]) -> pd.DataFrame:
    required_columns = {"sourceNodeId", "targetNodeId", "value", "time"}
    missing_columns = required_columns.difference(transactions.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise KeyError(f"Archive transactions are missing required columns: {missing_display}")

    frame = transactions[["sourceNodeId", "targetNodeId", "value", "time"]].rename(
        columns={
            "sourceNodeId": "source_node_id",
            "targetNodeId": "target_node_id",
            "value": "amount",
            "time": "event_time",
        }
    ).copy()
    frame = frame.dropna(subset=["source_node_id", "target_node_id", "amount", "event_time"])
    frame["source_node_id"] = pd.to_numeric(frame["source_node_id"], errors="coerce").fillna(-1).astype(int)
    frame["target_node_id"] = pd.to_numeric(frame["target_node_id"], errors="coerce").fillna(-1).astype(int)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    frame["event_time"] = pd.to_numeric(frame["event_time"], errors="coerce").fillna(0).astype(int)
    frame = frame[
        frame["source_node_id"].isin(node_ids) & frame["target_node_id"].isin(node_ids)
    ].reset_index(drop=True)
    frame["transaction_id"] = np.arange(len(frame), dtype=np.int64)
    frame["transaction_type_code"] = 0.0
    return frame


def _merge_on_node_id(base_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        return base_frame
    return base_frame.merge(feature_frame, on="node_id", how="left")


def _account_flow_features(node_frame: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    features = node_frame[["node_id", "is_fraud", "initial_balance", "fraud_step"]].copy()

    outgoing = transactions.groupby("source_node_id").agg(
        outgoing_transaction_count=("transaction_id", "count"),
        outgoing_amount_total=("amount", "sum"),
        outgoing_amount_mean=("amount", "mean"),
        outgoing_counterparty_count=("target_node_id", "nunique"),
        outgoing_first_time=("event_time", "min"),
        outgoing_last_time=("event_time", "max"),
    ).reset_index().rename(columns={"source_node_id": "node_id"})
    incoming = transactions.groupby("target_node_id").agg(
        incoming_transaction_count=("transaction_id", "count"),
        incoming_amount_total=("amount", "sum"),
        incoming_amount_mean=("amount", "mean"),
        incoming_counterparty_count=("source_node_id", "nunique"),
        incoming_first_time=("event_time", "min"),
        incoming_last_time=("event_time", "max"),
    ).reset_index().rename(columns={"target_node_id": "node_id"})

    features = _merge_on_node_id(features, outgoing)
    features = _merge_on_node_id(features, incoming)
    features = _fill_numeric_na(features)
    features["outgoing_time_span"] = (features["outgoing_last_time"] - features["outgoing_first_time"]).clip(lower=0)
    features["incoming_time_span"] = (features["incoming_last_time"] - features["incoming_first_time"]).clip(lower=0)
    features["total_transaction_count"] = (
        features["outgoing_transaction_count"] + features["incoming_transaction_count"]
    )
    features["total_amount"] = features["outgoing_amount_total"] + features["incoming_amount_total"]
    features["net_outgoing_amount"] = features["outgoing_amount_total"] - features["incoming_amount_total"]
    return features


def _window_account_features(features: pd.DataFrame, transactions: pd.DataFrame, temporal_windows: Sequence[int]) -> pd.DataFrame:
    enriched = features.copy()
    if transactions.empty:
        return enriched

    latest_event_time = int(transactions["event_time"].max())
    for window in sorted({int(value) for value in temporal_windows}):
        cutoff = latest_event_time - window + 1
        window_transactions = transactions.loc[transactions["event_time"] >= cutoff]

        outgoing = window_transactions.groupby("source_node_id").agg(
            outgoing_transaction_count=("transaction_id", "count"),
            outgoing_amount_total=("amount", "sum"),
            outgoing_counterparty_count=("target_node_id", "nunique"),
        ).reset_index().rename(columns={"source_node_id": "node_id"})
        incoming = window_transactions.groupby("target_node_id").agg(
            incoming_transaction_count=("transaction_id", "count"),
            incoming_amount_total=("amount", "sum"),
            incoming_counterparty_count=("source_node_id", "nunique"),
        ).reset_index().rename(columns={"target_node_id": "node_id"})

        outgoing = outgoing.rename(
            columns={
                "outgoing_transaction_count": f"_outgoing_transaction_count_{window}d",
                "outgoing_amount_total": f"_outgoing_amount_total_{window}d",
                "outgoing_counterparty_count": f"_outgoing_counterparty_count_{window}d",
            }
        )
        incoming = incoming.rename(
            columns={
                "incoming_transaction_count": f"_incoming_transaction_count_{window}d",
                "incoming_amount_total": f"_incoming_amount_total_{window}d",
                "incoming_counterparty_count": f"_incoming_counterparty_count_{window}d",
            }
        )

        enriched = _merge_on_node_id(enriched, outgoing)
        enriched = _merge_on_node_id(enriched, incoming)
        enriched = _fill_numeric_na(enriched)
        denominator = float(max(window, 1))
        enriched[f"outgoing_tx_velocity_{window}d"] = enriched[f"_outgoing_transaction_count_{window}d"] / denominator
        enriched[f"incoming_tx_velocity_{window}d"] = enriched[f"_incoming_transaction_count_{window}d"] / denominator
        enriched[f"outgoing_amount_velocity_{window}d"] = enriched[f"_outgoing_amount_total_{window}d"] / denominator
        enriched[f"incoming_amount_velocity_{window}d"] = enriched[f"_incoming_amount_total_{window}d"] / denominator
        enriched[f"outgoing_counterparty_velocity_{window}d"] = (
            enriched[f"_outgoing_counterparty_count_{window}d"] / denominator
        )
        enriched[f"incoming_counterparty_velocity_{window}d"] = (
            enriched[f"_incoming_counterparty_count_{window}d"] / denominator
        )

    temp_columns = [column for column in enriched.columns if column.startswith("_")]
    if temp_columns:
        enriched = enriched.drop(columns=temp_columns)
    return enriched


def _derive_archive_merchants(
    transactions: pd.DataFrame,
    *,
    merchant_seed: int,
    merchant_pool_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merchant_links = transactions[
        ["transaction_id", "source_node_id", "target_node_id", "amount", "event_time"]
    ].copy()
    merchant_links["merchant_anchor"] = merchant_links["target_node_id"].astype(str)
    merchant_links["merchant_id"] = merchant_links["merchant_anchor"].map(
        lambda value: _stable_bucket(value, merchant_seed + 300, merchant_pool_size, "merchant")
    )
    merchant_links["merchant_segment"] = merchant_links["merchant_id"].str.extract(r"(\d+)$")[0].fillna("0").astype(int)
    max_segment = float(max(merchant_pool_size - 1, 1))
    merchant_links["transaction_type_code"] = 1.0 + (merchant_links["merchant_segment"] / max_segment)
    merchants = merchant_links.groupby("merchant_id").agg(
        transaction_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
        unique_buyer_count=("source_node_id", "nunique"),
        unique_anchor_account_count=("target_node_id", "nunique"),
        first_event_time=("event_time", "min"),
        last_event_time=("event_time", "max"),
        merchant_segment=("merchant_segment", "max"),
    ).reset_index()
    merchants["active_time_span"] = (merchants["last_event_time"] - merchants["first_event_time"]).clip(lower=0)
    return merchants, merchant_links


def _merchant_temporal_features(
    merchant_frame: pd.DataFrame,
    merchant_links: pd.DataFrame,
    temporal_windows: Sequence[int],
) -> pd.DataFrame:
    enriched = merchant_frame.copy()
    if merchant_links.empty:
        return enriched

    latest_event_time = int(merchant_links["event_time"].max())
    for window in sorted({int(value) for value in temporal_windows}):
        cutoff = latest_event_time - window + 1
        window_links = merchant_links.loc[merchant_links["event_time"] >= cutoff]
        aggregates = window_links.groupby("merchant_id").agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            unique_buyer_count=("source_node_id", "nunique"),
        ).reset_index().rename(
            columns={
                "transaction_count": f"_merchant_transaction_count_{window}d",
                "total_amount": f"_merchant_total_amount_{window}d",
                "unique_buyer_count": f"_merchant_unique_buyer_count_{window}d",
            }
        )
        enriched = enriched.merge(aggregates, on="merchant_id", how="left")
        enriched = _fill_numeric_na(enriched)
        denominator = float(max(window, 1))
        enriched[f"merchant_tx_velocity_{window}d"] = enriched[f"_merchant_transaction_count_{window}d"] / denominator
        enriched[f"merchant_amount_velocity_{window}d"] = enriched[f"_merchant_total_amount_{window}d"] / denominator
        enriched[f"merchant_buyer_velocity_{window}d"] = (
            enriched[f"_merchant_unique_buyer_count_{window}d"] / denominator
        )

    temp_columns = [column for column in enriched.columns if column.startswith("_")]
    if temp_columns:
        enriched = enriched.drop(columns=temp_columns)
    return enriched


def _account_merchant_interactions(account_features: pd.DataFrame, merchant_links: pd.DataFrame) -> pd.DataFrame:
    interactions = merchant_links.groupby("source_node_id").agg(
        merchant_transaction_count=("transaction_id", "count"),
        merchant_total_amount=("amount", "sum"),
        unique_merchant_count=("merchant_id", "nunique"),
    ).reset_index().rename(columns={"source_node_id": "node_id"})
    enriched = account_features.merge(interactions, on="node_id", how="left")
    return _fill_numeric_na(enriched)


def build_archive_feature_bundle(
    nodes: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    temporal_windows: Sequence[int] = (1, 7, 30),
    merchant_seed: int = 17,
    merchant_pool_size: int = 24,
    include_communities: bool = True,
    community_seed: int = 42,
    embedding_dimensions: int = 16,
    embedding_random_state: int = 42,
    include_embeddings: bool = True,
) -> dict[str, pd.DataFrame | nx.DiGraph]:
    node_frame = normalize_archive_nodes(nodes)
    transaction_frame = normalize_archive_transactions(transactions, set(node_frame["node_id"].tolist()))
    account_tabular_features = _account_flow_features(node_frame, transaction_frame)
    account_tabular_features = _window_account_features(
        account_tabular_features,
        transaction_frame,
        temporal_windows,
    )

    merchant_features, merchant_links = _derive_archive_merchants(
        transaction_frame,
        merchant_seed=merchant_seed,
        merchant_pool_size=merchant_pool_size,
    )
    merchant_features = _merchant_temporal_features(merchant_features, merchant_links, temporal_windows)
    account_tabular_features = _account_merchant_interactions(account_tabular_features, merchant_links)

    graph = build_archive_transaction_graph(nodes, transactions)
    graph_node_features = compute_node_features(
        graph,
        include_communities=include_communities,
        community_seed=community_seed,
    )
    temporal_columns = [
        column
        for column in account_tabular_features.columns
        if column not in {"node_id", "is_fraud", "initial_balance", "fraud_step"}
    ]
    graph_node_features = graph_node_features.merge(
        account_tabular_features[["node_id", *temporal_columns]],
        on="node_id",
        how="left",
    )
    graph_node_features = _fill_numeric_na(graph_node_features)

    embeddings = pd.DataFrame(columns=["node_id"])
    if include_embeddings:
        embeddings = compute_spectral_node_embeddings(
            graph,
            dimensions=embedding_dimensions,
            random_state=embedding_random_state,
        )
    combined = (
        graph_node_features.merge(embeddings, on="node_id", how="left") if not embeddings.empty else graph_node_features.copy()
    )

    return {
        "graph": graph,
        "normalized_nodes": node_frame,
        "normalized_transactions": transaction_frame,
        "tabular_account_features": account_tabular_features,
        "node_features": graph_node_features,
        "embeddings": embeddings,
        "combined": combined,
        "merchant_features": merchant_features,
        "merchant_links": merchant_links,
    }


def build_graph_feature_bundle(
    graph: nx.DiGraph,
    *,
    include_communities: bool = True,
    community_seed: int = 42,
    embedding_dimensions: int = 16,
    embedding_random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    node_features = compute_node_features(
        graph,
        include_communities=include_communities,
        community_seed=community_seed,
    )
    embeddings = compute_spectral_node_embeddings(
        graph,
        dimensions=embedding_dimensions,
        random_state=embedding_random_state,
    )
    combined = node_features.merge(embeddings, on="node_id", how="left") if not node_features.empty else embeddings
    return {
        "node_features": node_features,
        "embeddings": embeddings,
        "combined": combined,
    }
