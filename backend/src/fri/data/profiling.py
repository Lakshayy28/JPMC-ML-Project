from __future__ import annotations

import math

import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _describe_numeric(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    description = series.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    clean_description: dict[str, float] = {}
    for key, value in description.items():
        if pd.isna(value):
            continue
        clean_description[str(key)] = float(value)
    return clean_description


def profile_canonical_tables(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    accounts = tables["accounts"]
    parties = tables["parties"]
    transactions = tables["transactions"]
    devices = tables["devices"]
    ip_addresses = tables["ip_addresses"]
    merchants = tables["merchants"]

    return {
        "table_row_counts": {name: int(len(table)) for name, table in tables.items()},
        "account_alert_rate": _safe_ratio(accounts["is_alerted"].sum(), len(accounts)),
        "party_alert_rate": _safe_ratio(parties["is_alerted"].sum(), len(parties)),
        "transaction_alert_rate": _safe_ratio(transactions["is_alert_related"].sum(), len(transactions)),
        "cash_transaction_rate": _safe_ratio(transactions["is_cash"].sum(), len(transactions)),
        "transaction_type_distribution": {
            key: int(value) for key, value in transactions["transaction_type"].value_counts().to_dict().items()
        },
        "label_source_distribution": {
            key: int(value) for key, value in transactions["label_source"].value_counts().to_dict().items()
        },
        "amount_summary": _describe_numeric(transactions["amount"]),
        "initial_balance_summary": _describe_numeric(accounts["initial_balance"]),
        "top_shared_devices": devices.sort_values("shared_account_count", ascending=False).head(10).to_dict("records"),
        "top_shared_ip_addresses": ip_addresses.sort_values("shared_account_count", ascending=False).head(10).to_dict("records"),
        "top_merchants": merchants.sort_values("transaction_count", ascending=False).head(10).to_dict("records"),
    }
