from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def build_temporal_activity_summary(
    tables: dict[str, pd.DataFrame],
    feature_sets: dict[str, pd.DataFrame],
    *,
    windows: Sequence[int],
) -> dict[str, object]:
    transactions = tables["transactions"].copy()
    parties = tables["parties"].copy()
    transaction_features = feature_sets["transaction"]
    party_features = feature_sets["party"]

    min_step = int(transactions["event_step"].min()) if not transactions.empty else 0
    max_step = int(transactions["event_step"].max()) if not transactions.empty else 0
    by_step = (
        transactions.groupby("event_step")
        .agg(
            transaction_count=("transaction_id", "size"),
            alert_related_count=("is_alert_related", "sum"),
            cash_count=("is_cash", "sum"),
            amount_total=("amount", "sum"),
        )
        .reset_index()
    )
    by_step["alert_related_rate"] = by_step["alert_related_count"] / by_step["transaction_count"]
    by_step["cash_rate"] = by_step["cash_count"] / by_step["transaction_count"]

    recent_window_stats: dict[str, object] = {}
    for window in windows:
        recent = transactions[transactions["event_step"] >= max_step - window + 1]
        recent_window_stats[f"transactions_recent_{window}"] = int(len(recent))
        recent_window_stats[f"amount_recent_{window}"] = float(recent["amount"].sum())
        recent_window_stats[f"alert_related_recent_{window}"] = int(recent["is_alert_related"].sum())

    top_party_activity = party_features.sort_values("outgoing_count_recent_30", ascending=False).head(10)
    top_transaction_velocity = transaction_features.sort_values(
        "source_account_velocity_ratio_1_to_30", ascending=False
    ).head(10)

    return {
        "step_range": {"min": min_step, "max": max_step},
        "transaction_count": int(len(transactions)),
        "party_count": int(len(parties)),
        "activity_by_step": by_step.to_dict("records"),
        "recent_window_stats": recent_window_stats,
        "top_parties_by_recent_outgoing_activity": top_party_activity[
            ["party_id", "outgoing_count_recent_30", "outgoing_amount_recent_30", "label"]
        ].to_dict("records"),
        "top_transactions_by_velocity": top_transaction_velocity[
            [
                "transaction_id",
                "source_account_id",
                "source_account_velocity_ratio_1_to_30",
                "source_account_count_prev_7",
                "label",
            ]
        ].to_dict("records"),
    }
