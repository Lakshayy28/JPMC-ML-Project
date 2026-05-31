from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from fri.features.temporal import build_party_temporal_features, build_transaction_temporal_features


def _shared_count(links: pd.DataFrame, entity_column: str, id_column: str) -> pd.DataFrame:
    shared = links.groupby(entity_column)[id_column].nunique().rename("shared_count").reset_index()
    return links.merge(shared, on=entity_column, how="left")[[id_column, "shared_count"]].drop_duplicates(id_column)


def build_transaction_features(
    tables: dict[str, pd.DataFrame],
    *,
    temporal_windows: Sequence[int],
) -> pd.DataFrame:
    transactions = tables["transactions"].copy()
    accounts = tables["accounts"].copy()
    account_device_links = tables["account_device_links"].copy()
    account_ip_links = tables["account_ip_links"].copy()
    temporal_features = build_transaction_temporal_features(transactions, windows=temporal_windows)

    outgoing = transactions.groupby("source_account_id").agg(
        outgoing_transaction_count=("transaction_id", "size"),
        outgoing_amount_total=("amount", "sum"),
        unique_destinations=("destination_account_id", "nunique"),
    )
    incoming = transactions.groupby("destination_account_id").agg(
        incoming_transaction_count=("transaction_id", "size"),
        incoming_amount_total=("amount", "sum"),
        unique_sources=("source_account_id", "nunique"),
    )

    device_share = _shared_count(account_device_links, "device_id", "account_id").rename(
        columns={"shared_count": "shared_device_count"}
    )
    ip_share = _shared_count(account_ip_links, "ip_id", "account_id").rename(
        columns={"shared_count": "shared_ip_count"}
    )

    account_projection = accounts[["account_id", "initial_balance", "is_alerted"]].copy()
    account_projection = account_projection.merge(device_share, on="account_id", how="left")
    account_projection = account_projection.merge(ip_share, on="account_id", how="left")
    account_projection = account_projection.fillna({"shared_device_count": 0, "shared_ip_count": 0})

    features = transactions.merge(
        outgoing, left_on="source_account_id", right_index=True, how="left"
    ).merge(
        incoming, left_on="destination_account_id", right_index=True, how="left"
    )
    features = features.merge(
        account_projection.add_prefix("source_"),
        left_on="source_account_id",
        right_on="source_account_id",
        how="left",
    )
    features = features.merge(
        account_projection.add_prefix("destination_"),
        left_on="destination_account_id",
        right_on="destination_account_id",
        how="left",
    )
    features = features.merge(temporal_features, on="transaction_id", how="left")
    features = features.fillna(0)
    features["label"] = features["is_alert_related"].astype(int)
    return features[
        [
            "transaction_id",
            "source_account_id",
            "destination_account_id",
            "transaction_type",
            "amount",
            "event_step",
            "is_cash",
            "outgoing_transaction_count",
            "outgoing_amount_total",
            "unique_destinations",
            "incoming_transaction_count",
            "incoming_amount_total",
            "unique_sources",
            "source_initial_balance",
            "source_is_alerted",
            "source_shared_device_count",
            "source_shared_ip_count",
            "destination_initial_balance",
            "destination_is_alerted",
            "destination_shared_device_count",
            "destination_shared_ip_count",
            "source_account_previous_gap",
            "source_account_count_prev_1",
            "source_account_amount_prev_1",
            "source_account_count_prev_7",
            "source_account_amount_prev_7",
            "source_account_count_prev_30",
            "source_account_amount_prev_30",
            "source_account_velocity_ratio_1_to_30",
            "source_account_amount_ratio_1_to_30",
            "destination_account_previous_gap",
            "destination_account_count_prev_1",
            "destination_account_amount_prev_1",
            "destination_account_count_prev_7",
            "destination_account_amount_prev_7",
            "destination_account_count_prev_30",
            "destination_account_amount_prev_30",
            "destination_account_velocity_ratio_1_to_30",
            "destination_account_amount_ratio_1_to_30",
            "source_party_previous_gap",
            "source_party_count_prev_1",
            "source_party_amount_prev_1",
            "source_party_count_prev_7",
            "source_party_amount_prev_7",
            "source_party_count_prev_30",
            "source_party_amount_prev_30",
            "source_party_velocity_ratio_1_to_30",
            "source_party_amount_ratio_1_to_30",
            "label",
            "label_source",
        ]
    ]


def build_party_features(
    tables: dict[str, pd.DataFrame],
    *,
    temporal_windows: Sequence[int],
) -> pd.DataFrame:
    parties = tables["parties"].copy()
    accounts = tables["accounts"].copy()
    transactions = tables["transactions"].copy()
    account_device_links = tables["account_device_links"].copy()
    account_ip_links = tables["account_ip_links"].copy()
    temporal_features = build_party_temporal_features(transactions, parties, windows=temporal_windows)

    outgoing = transactions.groupby("source_party_id").agg(
        outgoing_transaction_count=("transaction_id", "size"),
        outgoing_amount_total=("amount", "sum"),
        unique_destination_accounts=("destination_account_id", "nunique"),
    )
    incoming = transactions.groupby("destination_party_id").agg(
        incoming_transaction_count=("transaction_id", "size"),
        incoming_amount_total=("amount", "sum"),
        unique_source_accounts=("source_account_id", "nunique"),
    )

    device_count = account_device_links.merge(accounts[["account_id", "party_id"]], on="account_id", how="left")
    device_count = device_count.groupby("party_id")["device_id"].nunique().rename("device_count")

    ip_count = account_ip_links.merge(accounts[["account_id", "party_id"]], on="account_id", how="left")
    ip_count = ip_count.groupby("party_id")["ip_id"].nunique().rename("ip_count")

    features = parties.merge(outgoing, left_on="party_id", right_index=True, how="left")
    features = features.merge(incoming, left_on="party_id", right_index=True, how="left")
    features = features.merge(device_count, left_on="party_id", right_index=True, how="left")
    features = features.merge(ip_count, left_on="party_id", right_index=True, how="left")
    features = features.merge(temporal_features, on="party_id", how="left")
    features = features.fillna(0)
    features["label"] = features["is_alerted"].astype(int)
    return features[
        [
            "party_id",
            "party_type",
            "country_code",
            "account_count",
            "alert_count",
            "outgoing_transaction_count",
            "outgoing_amount_total",
            "unique_destination_accounts",
            "incoming_transaction_count",
            "incoming_amount_total",
            "unique_source_accounts",
            "device_count",
            "ip_count",
            "first_activity_step",
            "last_activity_step",
            "activity_span_steps",
            "average_outgoing_gap",
            "max_outgoing_gap",
            "outgoing_count_recent_1",
            "outgoing_amount_recent_1",
            "outgoing_count_recent_7",
            "outgoing_amount_recent_7",
            "outgoing_count_recent_30",
            "outgoing_amount_recent_30",
            "outgoing_velocity_ratio_1_to_30",
            "average_incoming_gap",
            "max_incoming_gap",
            "incoming_count_recent_1",
            "incoming_amount_recent_1",
            "incoming_count_recent_7",
            "incoming_amount_recent_7",
            "incoming_count_recent_30",
            "incoming_amount_recent_30",
            "label",
        ]
    ]


def build_feature_sets(
    tables: dict[str, pd.DataFrame],
    *,
    temporal_windows: Sequence[int] = (1, 7, 30),
) -> dict[str, pd.DataFrame]:
    return {
        "transaction": build_transaction_features(tables, temporal_windows=temporal_windows),
        "party": build_party_features(tables, temporal_windows=temporal_windows),
    }
