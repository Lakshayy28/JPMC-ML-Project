# Data Validation Specification

## Purpose

Define the current validation rules for the canonical processed data layer and record the latest validation outcome.

## Validation Artifact

- current artifact: `artifacts/data_validation_report.json`
- current status: all implemented checks pass

## Validation Categories

### 1. Required Columns

Current required-column checks ensure the minimum schema exists for:

- `accounts`
- `parties`
- `transactions`
- `alerts`

Required columns include identifiers, ownership fields and label-driving fields such as `is_alerted` and `is_alert_related`.

### 2. Uniqueness Rules

| Rule | Status |
| --- | --- |
| `accounts.account_id` unique | Pass |
| `parties.party_id` unique | Pass |
| `transactions.transaction_id + is_cash` unique | Pass |
| `alerts.alert_id + account_id` unique | Pass |
| `banks.bank_id` unique | Pass |
| `devices.device_id` unique | Pass |
| `ip_addresses.ip_id` unique | Pass |
| `merchants.merchant_id` unique | Pass |

### 3. Referential Integrity

| Relationship | Status |
| --- | --- |
| `accounts.party_id -> parties.party_id` | Pass |
| `accounts.bank_id -> banks.bank_id` | Pass |
| `alerts.account_id -> accounts.account_id` | Pass |
| `alerts.party_id -> parties.party_id` | Pass |
| `transactions.source_account_id -> accounts.account_id` | Pass |
| `transactions.destination_account_id -> accounts.account_id` with blanks allowed | Pass |
| `account_device_links.account_id -> accounts.account_id` | Pass |
| `account_device_links.device_id -> devices.device_id` | Pass |
| `account_ip_links.account_id -> accounts.account_id` | Pass |
| `account_ip_links.ip_id -> ip_addresses.ip_id` | Pass |
| `transaction_merchant_links.transaction_id -> transactions.transaction_id` | Pass |
| `transaction_merchant_links.merchant_id -> merchants.merchant_id` | Pass |

### 4. Null Monitoring

Current null checks are informational. The current expected sparse fields are:

- `transactions.destination_account_id`
- `transactions.destination_party_id`
- `transactions.event_timestamp`

Nulls in these fields are acceptable in sample mode and should only fail validation when running converted AMLSim layouts that are expected to populate them.

## Current Validation Outcome

The latest run reports:

- all checks passed
- no uniqueness violations
- no referential integrity failures
- nulls only in expected sparse transaction columns

## Next Validation Enhancements

1. add threshold checks for alert-rate drift and cash-transaction concentration
2. validate enrichment determinism by checking repeat runs produce stable Device/IP/Merchant assignments
3. add converted-layout rules requiring direct alert transaction exports when those files exist
4. add time-range checks once full AMLSim timestamps are available