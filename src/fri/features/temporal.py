from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_WINDOWS: tuple[int, ...] = (1, 7, 30)


def _prior_window_stats(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    step_column: str,
    amount_column: str,
    transaction_id_column: str,
    prefix: str,
    windows: Sequence[int],
) -> pd.DataFrame:
    required = [transaction_id_column, entity_column, step_column, amount_column]
    working = frame[required].copy()
    working = working[working[entity_column].fillna("") != ""]
    if working.empty:
        return pd.DataFrame(columns=[transaction_id_column])

    working = working.sort_values([entity_column, step_column, transaction_id_column]).reset_index(drop=True)
    result = working[[transaction_id_column]].copy()

    previous_gap = np.full(len(working), -1, dtype=float)
    for _, group in working.groupby(entity_column, sort=False):
        indices = group.index.to_numpy()
        steps = group[step_column].to_numpy(dtype=float)
        if len(steps) > 1:
            previous_gap[indices[1:]] = steps[1:] - steps[:-1]
    result[f"{prefix}_previous_gap"] = previous_gap

    for window in windows:
        counts = np.zeros(len(working), dtype=float)
        amounts = np.zeros(len(working), dtype=float)
        alert_related = np.zeros(len(working), dtype=float)
        for _, group in working.groupby(entity_column, sort=False):
            indices = group.index.to_numpy()
            steps = group[step_column].to_numpy(dtype=float)
            values = group[amount_column].to_numpy(dtype=float)
            prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
            left = 0
            for position, current_step in enumerate(steps):
                while left < position and steps[left] < current_step - window:
                    left += 1
                counts[indices[position]] = float(position - left)
                amounts[indices[position]] = float(prefix_sum[position] - prefix_sum[left])
        result[f"{prefix}_count_prev_{window}"] = counts
        result[f"{prefix}_amount_prev_{window}"] = amounts

    longest_window = max(windows)
    result[f"{prefix}_velocity_ratio_1_to_{longest_window}"] = np.divide(
        result[f"{prefix}_count_prev_1"],
        result[f"{prefix}_count_prev_{longest_window}"].replace(0, np.nan),
    ).fillna(0.0)
    result[f"{prefix}_amount_ratio_1_to_{longest_window}"] = np.divide(
        result[f"{prefix}_amount_prev_1"],
        result[f"{prefix}_amount_prev_{longest_window}"].replace(0, np.nan),
    ).fillna(0.0)

    return result


def build_transaction_temporal_features(
    transactions: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    base = transactions[["transaction_id", "source_account_id", "destination_account_id", "source_party_id", "destination_party_id", "event_step", "amount", "is_alert_related"]].copy()
    source_account = _prior_window_stats(
        base,
        entity_column="source_account_id",
        step_column="event_step",
        amount_column="amount",
        transaction_id_column="transaction_id",
        prefix="source_account",
        windows=windows,
    )
    destination_account = _prior_window_stats(
        base,
        entity_column="destination_account_id",
        step_column="event_step",
        amount_column="amount",
        transaction_id_column="transaction_id",
        prefix="destination_account",
        windows=windows,
    )
    source_party = _prior_window_stats(
        base,
        entity_column="source_party_id",
        step_column="event_step",
        amount_column="amount",
        transaction_id_column="transaction_id",
        prefix="source_party",
        windows=windows,
    )

    merged = transactions[["transaction_id"]].copy()
    for frame in (source_account, destination_account, source_party):
        merged = merged.merge(frame, on="transaction_id", how="left")
    return merged.fillna(0.0)


def build_party_temporal_features(
    transactions: pd.DataFrame,
    parties: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    if transactions.empty:
        return parties[["party_id"]].copy()

    max_step = int(transactions["event_step"].max()) if not transactions.empty else 0
    records: list[dict[str, float | int | str]] = []
    for party_id, group in transactions[transactions["source_party_id"].fillna("") != ""].groupby("source_party_id"):
        group = group.sort_values(["event_step", "transaction_id"])
        steps = group["event_step"].to_numpy(dtype=float)
        outgoing_amounts = group["amount"].to_numpy(dtype=float)
        inter_event_gaps = np.diff(steps) if len(steps) > 1 else np.array([], dtype=float)

        record: dict[str, float | int | str] = {
            "party_id": party_id,
            "first_activity_step": int(steps.min()) if len(steps) else -1,
            "last_activity_step": int(steps.max()) if len(steps) else -1,
            "activity_span_steps": int(steps.max() - steps.min()) if len(steps) else 0,
            "average_outgoing_gap": float(inter_event_gaps.mean()) if len(inter_event_gaps) else 0.0,
            "max_outgoing_gap": float(inter_event_gaps.max()) if len(inter_event_gaps) else 0.0,
        }
        for window in windows:
            recent = group[group["event_step"] >= max_step - window + 1]
            record[f"outgoing_count_recent_{window}"] = int(len(recent))
            record[f"outgoing_amount_recent_{window}"] = float(recent["amount"].sum())
        longest_window = max(windows)
        record[f"outgoing_velocity_ratio_1_to_{longest_window}"] = (
            float(record["outgoing_count_recent_1"]) / float(record[f"outgoing_count_recent_{longest_window}"])
            if record[f"outgoing_count_recent_{longest_window}"]
            else 0.0
        )
        records.append(record)

    outgoing_summary = pd.DataFrame.from_records(records)
    if outgoing_summary.empty:
        return parties[["party_id"]].copy()

    incoming_records: list[dict[str, float | int | str]] = []
    for party_id, group in transactions[transactions["destination_party_id"].fillna("") != ""].groupby("destination_party_id"):
        group = group.sort_values(["event_step", "transaction_id"])
        steps = group["event_step"].to_numpy(dtype=float)
        inter_event_gaps = np.diff(steps) if len(steps) > 1 else np.array([], dtype=float)

        record = {
            "party_id": party_id,
            "average_incoming_gap": float(inter_event_gaps.mean()) if len(inter_event_gaps) else 0.0,
            "max_incoming_gap": float(inter_event_gaps.max()) if len(inter_event_gaps) else 0.0,
        }
        for window in windows:
            recent = group[group["event_step"] >= max_step - window + 1]
            record[f"incoming_count_recent_{window}"] = int(len(recent))
            record[f"incoming_amount_recent_{window}"] = float(recent["amount"].sum())
        incoming_records.append(record)

    incoming_summary = pd.DataFrame.from_records(incoming_records)
    merged = parties[["party_id"]].merge(outgoing_summary, on="party_id", how="left")
    merged = merged.merge(incoming_summary, on="party_id", how="left")
    return merged.fillna(0.0)
