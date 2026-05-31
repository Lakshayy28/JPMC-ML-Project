# Data Dictionary

## Purpose

Define the current canonical processed tables produced from IBM AMLSim sample-mode ingestion and deterministic enrichment.

## Scope

This dictionary reflects the outputs currently written under `data/processed/amlsim`.

## Core Tables

### `parties.csv`

| Column | Meaning |
| --- | --- |
| `party_id` | canonical party identifier; in sample mode sourced from `PRIMARY_CUSTOMER_ID` |
| `country_code` | country hint inherited from account data |
| `layout_name` | raw AMLSim layout used to produce the row |
| `party_type` | current coarse party type; currently `individual` in sample mode |
| `source_type` | lineage marker showing source-native or fallback derivation |
| `account_count` | number of accounts linked to the party |
| `is_alerted` | whether the party is linked to at least one normalized alert |
| `alert_count` | number of normalized alert memberships linked to the party |

### `accounts.csv`

| Column | Meaning |
| --- | --- |
| `account_id` | canonical account identifier |
| `party_id` | linked owner party identifier |
| `bank_id` | hosting institution boundary |
| `country_code` | account geography hint |
| `business_type` | source business label or account category |
| `initial_balance` | initial account balance from AMLSim source |
| `start_step` | account start step or open date proxy |
| `end_step` | account end step or close date proxy |
| `source_suspicious_flag` | suspicious flag from raw source if present |
| `model_id` | AMLSim transaction behavior model identifier |
| `layout_name` | discovered raw AMLSim layout name |
| `source_type` | lineage marker |
| `is_alerted` | whether the account is referenced by a normalized alert |
| `alert_count` | count of alert memberships tied to the account |

### `transactions.csv`

| Column | Meaning |
| --- | --- |
| `transaction_id` | canonical transaction identifier |
| `source_account_id` | source account for the transfer or cash event |
| `destination_account_id` | destination account; blank for cash-only rows |
| `transaction_type` | source transaction event type |
| `amount` | transaction amount |
| `event_step` | step-level time value from AMLSim |
| `event_timestamp` | richer timestamp if present in full converted layouts |
| `is_cash` | whether the row came from the cash transaction surface |
| `layout_name` | raw AMLSim layout name |
| `source_party_id` | source party resolved from source account |
| `destination_party_id` | destination party resolved from destination account |
| `is_alert_related` | label flag showing alert linkage or alert adjacency |
| `label_source` | provenance of transaction label: direct or proxy |
| `source_type` | lineage marker |

### `alerts.csv`

| Column | Meaning |
| --- | --- |
| `alert_id` | canonical alert identifier |
| `alert_type` | typology or check name |
| `account_id` | linked alerted account |
| `party_id` | linked alerted party |
| `event_value` | event date or step captured from the raw alert file |
| `is_sar` | whether the alert is treated as suspicious/SAR-like |
| `layout_name` | raw AMLSim layout name |

### `banks.csv`

| Column | Meaning |
| --- | --- |
| `bank_id` | canonical bank identifier |
| `country_hint` | country inferred from linked accounts |
| `source_type` | lineage marker |

## Derived Infrastructure Tables

### `devices.csv`

| Column | Meaning |
| --- | --- |
| `device_id` | deterministic synthetic device identifier |
| `shared_account_count` | number of accounts mapped to the device |
| `source_type` | always `derived` in the current MVP |
| `device_family` | bucket identifier derived from deterministic hashing |

### `ip_addresses.csv`

| Column | Meaning |
| --- | --- |
| `ip_id` | deterministic synthetic IP identifier |
| `shared_account_count` | number of accounts mapped to the IP |
| `country_hint` | current geography hint for the synthetic IP |
| `source_type` | always `derived` in the current MVP |

### `merchants.csv`

| Column | Meaning |
| --- | --- |
| `merchant_id` | deterministic synthetic merchant identifier |
| `transaction_count` | number of transactions linked to the merchant |
| `source_type` | always `derived` in the current MVP |
| `merchant_segment` | hash-bucket segment for the synthetic merchant |

## Link Tables

### `account_device_links.csv`

| Column | Meaning |
| --- | --- |
| `account_id` | linked account |
| `device_id` | deterministic synthetic device |

### `account_ip_links.csv`

| Column | Meaning |
| --- | --- |
| `account_id` | linked account |
| `country_code` | geography carried forward from the source account |
| `ip_id` | deterministic synthetic IP |

### `transaction_merchant_links.csv`

| Column | Meaning |
| --- | --- |
| `transaction_id` | linked transaction |
| `merchant_id` | deterministic synthetic merchant |

## Lineage Notes

1. `parties`, `accounts`, `transactions`, `alerts` and `banks` are source-native or sample-layout normalized tables.
2. `devices`, `ip_addresses`, `merchants` and all link tables are deterministic derived MVP entities.
3. `destination_account_id` and `destination_party_id` can be blank for cash-only rows.
4. `event_timestamp` is blank in the current sample layout and becomes populated only when richer AMLSim converted outputs are used.