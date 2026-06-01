# EDA Report

## Purpose

Summarize the current exploratory findings from the canonical processed AMLSim sample outputs.

## Data Source

- canonical processed tables in `data/processed/amlsim`
- generated from the bundled AMLSim sample outputs using the current normalization and enrichment pipeline
- profiling artifact: `artifacts/eda_summary.json`

## Table Volumes

| Table | Rows |
| --- | --- |
| Parties | 30 |
| Accounts | 30 |
| Transactions | 133 |
| Alerts | 2 |
| Banks | 1 |
| Devices | 14 |
| IP Addresses | 24 |
| Merchants | 19 |

## Label Rates

| Metric | Value |
| --- | --- |
| Account alert rate | 0.0667 |
| Party alert rate | 0.0667 |
| Transaction alert-related rate | 0.1880 |
| Cash transaction rate | 0.6617 |

## Transaction Mix

Current sample transaction type counts:

| Type | Count |
| --- | --- |
| CASH-OUT | 80 |
| CHECK | 13 |
| CREDIT | 12 |
| DEPOSIT | 12 |
| WIRE | 8 |
| CASH-IN | 8 |

## Amount Distribution

| Statistic | Value |
| --- | --- |
| Mean | 88.97 |
| Std | 54.05 |
| Min | 10.23 |
| 25th percentile | 44.99 |
| Median | 77.58 |
| 75th percentile | 131.00 |
| Max | 198.27 |

## Initial Balance Distribution

| Statistic | Value |
| --- | --- |
| Mean | 161.30 |
| Std | 25.35 |
| Min | 110.07 |
| 25th percentile | 142.40 |
| Median | 161.47 |
| 75th percentile | 183.59 |
| Max | 198.28 |

## Derived Infrastructure Concentration

### Devices

- the most reused device currently links 4 accounts
- several devices link 3 accounts, which is useful for graph linkage despite being synthetic

### IP Addresses

- the current deterministic enrichment spreads 30 accounts across 24 synthetic IPs
- the most reused IPs link 2 accounts each

### Merchants

- one synthetic merchant cluster (`merchant_006`) absorbs 80 transactions, driven by the current enrichment rules around cash-heavy activity

## Nulls And Expected Sparsity

The validation report shows three null-bearing transaction fields:

- `destination_account_id`: 88 blank values
- `destination_party_id`: 88 blank values
- `event_timestamp`: 133 blank values

These are expected for the current sample layout because:

1. cash transaction rows do not have destination accounts
2. destination party resolution depends on destination account presence
3. the bundled sample layout uses step-like timing instead of full converted timestamps

## Leakage And Interpretation Risks

1. sample-size risk: the current processed dataset is too small for reliable portfolio-grade performance claims
2. label-proxy risk: transaction labels currently rely on alert-adjacent account membership rather than direct alert transaction exports
3. enrichment risk: Device, IP Address and Merchant patterns are deterministic synthetic linkages and must not be described as observed production behavior

## Conclusions

The sample-backed processed layer is good enough for validating the canonical schema, the enrichment logic and the first feature pipelines. It is not yet sufficient for serious model selection or final evaluation. The next EDA milestone should be based on a larger AMLSim converted output.