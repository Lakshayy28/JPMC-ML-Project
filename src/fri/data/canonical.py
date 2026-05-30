from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fri.config import EnrichmentSettings
from fri.data.loaders import RawAmlsimData


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def _text_series(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    column = _find_column(df, *candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype="string")
    return df[column].fillna(default).astype("string").str.strip()


def _numeric_series(df: pd.DataFrame, *candidates: str, default: float = 0.0) -> pd.Series:
    column = _find_column(df, *candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _boolean_series(df: pd.DataFrame, *candidates: str, default: bool = False) -> pd.Series:
    column = _find_column(df, *candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype="bool")
    raw = df[column].fillna(default)
    if pd.api.types.is_bool_dtype(raw):
        return raw.astype(bool)
    normalized = raw.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _stable_bucket(value: str, seed: int, pool_size: int, prefix: str) -> str:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    digest = hashlib.sha256(f"{seed}:{prefix}:{value}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % pool_size
    return f"{prefix}_{bucket:03d}"


def _normalize_alerts(raw_alerts: pd.DataFrame, accounts: pd.DataFrame, layout_name: str) -> pd.DataFrame:
    if raw_alerts.empty:
        return pd.DataFrame(
            columns=["alert_id", "alert_type", "account_id", "party_id", "event_value", "is_sar", "layout_name"]
        )

    alerts = pd.DataFrame(
        {
            "alert_id": _text_series(raw_alerts, "ALERT_KEY", "alert_id"),
            "alert_type": _text_series(raw_alerts, "CHECK_NAME", "alert_type", "ALERT_TEXT", default="unknown"),
            "account_id": _text_series(raw_alerts, "ACCOUNT_ID", "acct_id"),
            "party_id": _text_series(raw_alerts, "CUSTOMER_ID", "cust_id", "partyId"),
            "event_value": _text_series(raw_alerts, "EVENT_DATE", "start", "tran_timestamp"),
            "is_sar": _boolean_series(raw_alerts, "is_sar", "Escalated_To_Case_Investigation", default=True),
            "layout_name": layout_name,
        }
    )
    account_lookup = accounts.set_index("account_id")["party_id"] if not accounts.empty else pd.Series(dtype="string")
    known_party_ids = set(accounts["party_id"]) if not accounts.empty else set()
    unresolved_party_mask = (alerts["party_id"] == "") | (~alerts["party_id"].isin(known_party_ids))
    if unresolved_party_mask.any():
        alerts.loc[unresolved_party_mask, "party_id"] = (
            alerts.loc[unresolved_party_mask, "account_id"].map(account_lookup).fillna("")
        )
    return alerts.drop_duplicates().reset_index(drop=True)


def _normalize_accounts(raw_accounts: pd.DataFrame, raw_alerts: pd.DataFrame, layout_name: str) -> pd.DataFrame:
    accounts = pd.DataFrame(
        {
            "account_id": _text_series(raw_accounts, "ACCOUNT_ID", "acct_id"),
            "party_id": _text_series(raw_accounts, "PRIMARY_CUSTOMER_ID", "cust_id", "partyId"),
            "bank_id": _text_series(raw_accounts, "bank_id", "BANK_ID", default="bank_default"),
            "country_code": _text_series(raw_accounts, "country", default="US"),
            "business_type": _text_series(raw_accounts, "business", "business_type", "type", default="unknown"),
            "initial_balance": _numeric_series(raw_accounts, "init_balance", "initial_deposit", default=0.0),
            "start_step": _numeric_series(raw_accounts, "start", "start_day", "open_dt", default=0.0).astype(int),
            "end_step": _numeric_series(raw_accounts, "end", "end_day", "close_dt", default=0.0).astype(int),
            "source_suspicious_flag": _boolean_series(raw_accounts, "suspicious", "prior_sar_count", "isFraud", default=False),
            "model_id": _numeric_series(raw_accounts, "modelID", "tx_behavior_id", default=0).astype(int),
            "layout_name": layout_name,
            "source_type": "source_native",
        }
    )
    accounts.loc[accounts["party_id"] == "", "party_id"] = "party_" + accounts.loc[accounts["party_id"] == "", "account_id"]

    alert_accounts = set(_text_series(raw_alerts, "ACCOUNT_ID", "acct_id")) if not raw_alerts.empty else set()
    alert_counts = _text_series(raw_alerts, "ACCOUNT_ID", "acct_id").value_counts() if not raw_alerts.empty else pd.Series(dtype="int64")
    accounts["is_alerted"] = accounts["account_id"].isin(alert_accounts)
    accounts["alert_count"] = accounts["account_id"].map(alert_counts).fillna(0).astype(int)
    return accounts.drop_duplicates(subset=["account_id"]).reset_index(drop=True)


def _normalize_parties(accounts: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    parties = accounts[["party_id", "country_code", "layout_name"]].drop_duplicates().copy()
    parties["party_type"] = "individual"
    parties["source_type"] = "source_native_or_sample_fallback"
    party_alert_counts = alerts["party_id"].value_counts() if not alerts.empty else pd.Series(dtype="int64")
    party_account_counts = accounts.groupby("party_id")["account_id"].nunique()
    parties["account_count"] = parties["party_id"].map(party_account_counts).fillna(0).astype(int)
    parties["is_alerted"] = parties["party_id"].isin(set(alerts["party_id"])) if not alerts.empty else False
    parties["alert_count"] = parties["party_id"].map(party_alert_counts).fillna(0).astype(int)
    return parties.reset_index(drop=True)


def _normalize_banks(accounts: pd.DataFrame) -> pd.DataFrame:
    banks = accounts[["bank_id", "country_code"]].drop_duplicates().copy()
    banks["source_type"] = "source_native"
    return banks.rename(columns={"country_code": "country_hint"}).reset_index(drop=True)


def _normalize_transactions(
    raw_tables: dict[str, pd.DataFrame],
    accounts: pd.DataFrame,
    alerts: pd.DataFrame,
    layout_name: str,
) -> pd.DataFrame:
    raw_transactions = raw_tables.get("transactions", pd.DataFrame())
    raw_cash_transactions = raw_tables.get("cash_transactions", pd.DataFrame())
    raw_alert_transactions = raw_tables.get("alert_transactions", pd.DataFrame())

    transactions = pd.DataFrame(
        {
            "transaction_id": _text_series(raw_transactions, "TXN_ID", "tran_id"),
            "source_account_id": _text_series(raw_transactions, "ACCOUNT_ID", "orig_acct"),
            "destination_account_id": _text_series(raw_transactions, "COUNTER_PARTY_ACCOUNT_NUM", "bene_acct"),
            "transaction_type": _text_series(raw_transactions, "TXN_SOURCE_TYPE_CODE", "tx_type", default="TRANSFER"),
            "amount": _numeric_series(raw_transactions, "TXN_AMOUNT_ORIG", "base_amt", default=0.0),
            "event_step": _numeric_series(raw_transactions, "start", "RUN_DATE", default=0.0).astype(int),
            "event_timestamp": _text_series(raw_transactions, "tran_timestamp"),
            "is_cash": False,
            "layout_name": layout_name,
        }
    )

    if not raw_cash_transactions.empty:
        cash_transactions = pd.DataFrame(
            {
                "transaction_id": _text_series(raw_cash_transactions, "TXN_ID", "tran_id"),
                "source_account_id": _text_series(raw_cash_transactions, "ACCOUNT_ID", "orig_acct"),
                "destination_account_id": "",
                "transaction_type": _text_series(raw_cash_transactions, "TXN_SOURCE_TYPE_CODE", "tx_type", default="CASH"),
                "amount": _numeric_series(raw_cash_transactions, "TXN_AMOUNT_ORIG", "base_amt", default=0.0),
                "event_step": _numeric_series(raw_cash_transactions, "RUN_DATE", "start", default=0.0).astype(int),
                "event_timestamp": _text_series(raw_cash_transactions, "tran_timestamp"),
                "is_cash": True,
                "layout_name": layout_name,
            }
        )
        transactions = pd.concat([transactions, cash_transactions], ignore_index=True)

    account_to_party = accounts.set_index("account_id")["party_id"] if not accounts.empty else pd.Series(dtype="string")
    alerted_accounts = set(alerts["account_id"]) if not alerts.empty else set()
    direct_alert_transactions = set(_text_series(raw_alert_transactions, "tran_id", "TXN_ID")) if not raw_alert_transactions.empty else set()

    transactions["source_party_id"] = transactions["source_account_id"].map(account_to_party).fillna("")
    transactions["destination_party_id"] = transactions["destination_account_id"].map(account_to_party).fillna("")
    transactions["is_alert_related"] = (
        transactions["transaction_id"].isin(direct_alert_transactions)
        | transactions["source_account_id"].isin(alerted_accounts)
        | transactions["destination_account_id"].isin(alerted_accounts)
    )
    transactions["label_source"] = "none"
    transactions.loc[transactions["source_account_id"].isin(alerted_accounts) | transactions["destination_account_id"].isin(alerted_accounts), "label_source"] = "alert_adjacent_account"
    transactions.loc[transactions["transaction_id"].isin(direct_alert_transactions), "label_source"] = "direct_alert_transaction"
    transactions["source_type"] = "source_native"
    return transactions.drop_duplicates(subset=["transaction_id", "is_cash"]).reset_index(drop=True)


def _derive_devices(accounts: pd.DataFrame, enrichment: EnrichmentSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_device_links = accounts[["account_id", "is_alerted"]].copy()
    account_device_links["device_id"] = account_device_links.apply(
        lambda row: _stable_bucket(
            row["account_id"],
            enrichment.seed + (100 if row["is_alerted"] else 0),
            enrichment.suspicious_device_pool_size if row["is_alerted"] else enrichment.device_pool_size,
            "device",
        ),
        axis=1,
    )
    shared_counts = account_device_links.groupby("device_id")["account_id"].nunique()
    devices = pd.DataFrame({"device_id": shared_counts.index, "shared_account_count": shared_counts.values})
    devices["source_type"] = "derived"
    devices["device_family"] = devices["device_id"].str.extract(r"(\d+)$").fillna("0")
    return devices.reset_index(drop=True), account_device_links.drop(columns=["is_alerted"]).reset_index(drop=True)


def _derive_ip_addresses(accounts: pd.DataFrame, enrichment: EnrichmentSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_ip_links = accounts[["account_id", "is_alerted", "country_code"]].copy()
    account_ip_links["ip_id"] = account_ip_links.apply(
        lambda row: _stable_bucket(
            row["account_id"],
            enrichment.seed + (200 if row["is_alerted"] else 0),
            enrichment.suspicious_ip_pool_size if row["is_alerted"] else enrichment.ip_pool_size,
            "ip",
        ),
        axis=1,
    )
    shared_counts = account_ip_links.groupby("ip_id")["account_id"].nunique()
    ip_addresses = pd.DataFrame({"ip_id": shared_counts.index, "shared_account_count": shared_counts.values})
    ip_addresses["country_hint"] = "US"
    ip_addresses["source_type"] = "derived"
    return ip_addresses.reset_index(drop=True), account_ip_links.drop(columns=["is_alerted"]).reset_index(drop=True)


def _derive_merchants(transactions: pd.DataFrame, enrichment: EnrichmentSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    merchant_links = transactions[["transaction_id", "destination_account_id", "source_account_id", "transaction_type"]].copy()
    merchant_links["merchant_anchor"] = merchant_links["destination_account_id"].where(
        merchant_links["destination_account_id"] != "", merchant_links["source_account_id"]
    )
    merchant_links["merchant_id"] = merchant_links.apply(
        lambda row: _stable_bucket(
            f"{row['merchant_anchor']}:{row['transaction_type']}",
            enrichment.seed + 300,
            enrichment.merchant_pool_size,
            "merchant",
        ),
        axis=1,
    )
    merchant_counts = merchant_links.groupby("merchant_id")["transaction_id"].nunique()
    merchants = pd.DataFrame({"merchant_id": merchant_counts.index, "transaction_count": merchant_counts.values})
    merchants["source_type"] = "derived"
    merchants["merchant_segment"] = merchants["merchant_id"].str.extract(r"(\d+)$").fillna("0")
    return merchants.reset_index(drop=True), merchant_links[["transaction_id", "merchant_id"]].reset_index(drop=True)


def build_canonical_tables(raw_data: RawAmlsimData, enrichment: EnrichmentSettings) -> dict[str, pd.DataFrame]:
    layout_name = raw_data.layout.name
    raw_tables = raw_data.tables
    accounts = _normalize_accounts(raw_tables["accounts"], raw_tables["alerts"], layout_name)
    alerts = _normalize_alerts(raw_tables["alerts"], accounts, layout_name)
    parties = _normalize_parties(accounts, alerts)
    banks = _normalize_banks(accounts)
    transactions = _normalize_transactions(raw_tables, accounts, alerts, layout_name)
    devices, account_device_links = _derive_devices(accounts, enrichment)
    ip_addresses, account_ip_links = _derive_ip_addresses(accounts, enrichment)
    merchants, transaction_merchant_links = _derive_merchants(transactions, enrichment)

    return {
        "parties": parties,
        "accounts": accounts,
        "transactions": transactions,
        "alerts": alerts,
        "banks": banks,
        "devices": devices,
        "ip_addresses": ip_addresses,
        "merchants": merchants,
        "account_device_links": account_device_links,
        "account_ip_links": account_ip_links,
        "transaction_merchant_links": transaction_merchant_links,
    }


def write_canonical_tables(tables: dict[str, pd.DataFrame], output_root: str | Path) -> None:
    root_path = Path(output_root)
    root_path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(root_path / f"{name}.csv", index=False)
