from __future__ import annotations

import pandas as pd


def _null_summary(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(value) for column, value in df.isna().sum().to_dict().items() if int(value) > 0}


def _duplicate_count(df: pd.DataFrame, subset: list[str]) -> int:
    if not subset:
        return 0
    return int(df.duplicated(subset=subset).sum())


def _foreign_key_check(
    source: pd.DataFrame,
    source_column: str,
    target: pd.DataFrame,
    target_column: str,
    *,
    allow_blank: bool = False,
) -> dict[str, int | bool]:
    values = source[source_column].fillna("")
    if allow_blank:
        values = values[values != ""]
    missing_mask = ~values.isin(set(target[target_column]))
    return {
        "passed": bool((~missing_mask).all()),
        "missing_references": int(missing_mask.sum()),
        "checked_rows": int(len(values)),
    }


def validate_canonical_tables(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    accounts = tables["accounts"]
    parties = tables["parties"]
    transactions = tables["transactions"]
    alerts = tables["alerts"]
    banks = tables["banks"]
    devices = tables["devices"]
    ip_addresses = tables["ip_addresses"]
    merchants = tables["merchants"]
    account_device_links = tables["account_device_links"]
    account_ip_links = tables["account_ip_links"]
    transaction_merchant_links = tables["transaction_merchant_links"]

    uniqueness_checks = {
        "accounts.account_id": _duplicate_count(accounts, ["account_id"]),
        "parties.party_id": _duplicate_count(parties, ["party_id"]),
        "transactions.transaction_id_is_cash": _duplicate_count(transactions, ["transaction_id", "is_cash"]),
        "alerts.alert_account": _duplicate_count(alerts, ["alert_id", "account_id"]),
        "banks.bank_id": _duplicate_count(banks, ["bank_id"]),
        "devices.device_id": _duplicate_count(devices, ["device_id"]),
        "ip_addresses.ip_id": _duplicate_count(ip_addresses, ["ip_id"]),
        "merchants.merchant_id": _duplicate_count(merchants, ["merchant_id"]),
    }

    referential_integrity = {
        "accounts.party_id -> parties.party_id": _foreign_key_check(accounts, "party_id", parties, "party_id"),
        "accounts.bank_id -> banks.bank_id": _foreign_key_check(accounts, "bank_id", banks, "bank_id"),
        "alerts.account_id -> accounts.account_id": _foreign_key_check(alerts, "account_id", accounts, "account_id"),
        "alerts.party_id -> parties.party_id": _foreign_key_check(alerts, "party_id", parties, "party_id"),
        "transactions.source_account_id -> accounts.account_id": _foreign_key_check(
            transactions, "source_account_id", accounts, "account_id"
        ),
        "transactions.destination_account_id -> accounts.account_id": _foreign_key_check(
            transactions, "destination_account_id", accounts, "account_id", allow_blank=True
        ),
        "account_device_links.account_id -> accounts.account_id": _foreign_key_check(
            account_device_links, "account_id", accounts, "account_id"
        ),
        "account_device_links.device_id -> devices.device_id": _foreign_key_check(
            account_device_links, "device_id", devices, "device_id"
        ),
        "account_ip_links.account_id -> accounts.account_id": _foreign_key_check(
            account_ip_links, "account_id", accounts, "account_id"
        ),
        "account_ip_links.ip_id -> ip_addresses.ip_id": _foreign_key_check(
            account_ip_links, "ip_id", ip_addresses, "ip_id"
        ),
        "transaction_merchant_links.transaction_id -> transactions.transaction_id": _foreign_key_check(
            transaction_merchant_links, "transaction_id", transactions, "transaction_id"
        ),
        "transaction_merchant_links.merchant_id -> merchants.merchant_id": _foreign_key_check(
            transaction_merchant_links, "merchant_id", merchants, "merchant_id"
        ),
    }

    required_columns = {
        "accounts": ["account_id", "party_id", "bank_id", "initial_balance", "is_alerted"],
        "parties": ["party_id", "party_type", "is_alerted"],
        "transactions": ["transaction_id", "source_account_id", "amount", "is_alert_related"],
        "alerts": ["alert_id", "account_id", "party_id"],
    }

    required_column_report = {
        table_name: {
            "passed": all(column in tables[table_name].columns for column in columns),
            "missing_columns": [column for column in columns if column not in tables[table_name].columns],
        }
        for table_name, columns in required_columns.items()
    }

    return {
        "row_counts": {name: int(len(table)) for name, table in tables.items()},
        "null_counts": {name: _null_summary(table) for name, table in tables.items()},
        "uniqueness_checks": uniqueness_checks,
        "referential_integrity": referential_integrity,
        "required_columns": required_column_report,
        "all_checks_passed": bool(
            all(value == 0 for value in uniqueness_checks.values())
            and all(item["passed"] for item in referential_integrity.values())
            and all(item["passed"] for item in required_column_report.values())
        ),
    }
