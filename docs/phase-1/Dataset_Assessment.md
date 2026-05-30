# Dataset Assessment

## Purpose

Assess IBM AMLSim as the canonical MVP dataset for the Financial Risk Intelligence Platform and document the exact artifact pinned for implementation.

## Dataset Decision

IBM AMLSim is the canonical MVP dataset.

Reasons:

1. It provides an account-to-account transaction network suitable for graph analytics.
2. It exposes known suspicious patterns through typology parameters and alert outputs.
3. It supports both transaction-centric and account-centric modelling better than PaySim for this project's graph-heavy scope.
4. It can be extended with deterministic Device, IP Address and Merchant enrichment without changing the source-native backbone.

## Pinned Artifact

| Item | Value |
| --- | --- |
| Source | `https://github.com/IBM/AMLSim` |
| Local Artifact Type | shallow clone of repository |
| Branch Guidance | `master` per repository README |
| Pinned Commit | `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca` |
| Releases | none published at time of assessment |
| Workspace Location | `data/external/AMLSim` |

## Observed Artifact Inventory

### Bundled Sample Outputs

Observed directly from the cloned repository:

| File | Records | Notes |
| --- | --- | --- |
| `sample/outputs/accounts.csv` | 30 | sample account export with `PRIMARY_CUSTOMER_ID` |
| `sample/outputs/alerts.csv` | 2 | account/customer alert rows from cycle pattern sample |
| `sample/outputs/cash_tx.csv` | 88 | cash in/out events |
| `sample/outputs/tx.csv` | 45 | account-to-account transaction events |

### Bundled Sample Parameter Files

| File | Notes |
| --- | --- |
| `sample/paramFiles/accounts.csv` | aggregated account generation parameters |
| `sample/paramFiles/alertPatterns.csv` | sample cycle pattern specification |
| `sample/paramFiles/degree.csv` | sample in/out degree distribution |
| `sample/paramFiles/transactionType.csv` | sample transaction type frequencies |

### Full Parameterized 1K Configuration

Observed directly from `paramFiles/1K`:

| File | Notes |
| --- | --- |
| `accounts.csv` | aggregated account generation input with bank assignment |
| `alertPatterns.csv` | `fan_in`, `fan_out`, and `cycle` typologies |
| `degree.csv` | degree distribution for generated graph |
| `normalModels.csv` | normal transaction behaviors |
| `transactionType.csv` | transfer type frequencies |
| `schema.json` | output schema sections for accounts, transactions, alert members, parties and mappings |
| `conf.json` | simulation seed, total steps, base date and output file names |

## Schema Surface Summary

### What The Sample Outputs Prove

The bundled sample outputs confirm a lightweight legacy-style export layout with these headers:

- accounts: `ACCOUNT_ID`, `PRIMARY_CUSTOMER_ID`, `init_balance`, `country`, `business`, `suspicious`, `isFraud`, `modelID`
- alerts: `ALERT_KEY`, `ALERT_TEXT`, `ACCOUNT_ID`, `CUSTOMER_ID`, `EVENT_DATE`, `CHECK_NAME`
- cash transactions: `TXN_ID`, `ACCOUNT_ID`, `BRANCH_ID`, `TXN_SOURCE_TYPE_CODE`, `TXN_AMOUNT_ORIG`, `RUN_DATE`
- transactions: `TXN_ID`, `ACCOUNT_ID`, `COUNTER_PARTY_ACCOUNT_NUM`, `TXN_SOURCE_TYPE_CODE`, `TXN_AMOUNT_ORIG`, `start`, `end`

### What The Converted Full Schema Supports

The `paramFiles/1K/schema.json` and `paramFiles/1K/conf.json` define a richer converted output surface including:

- `accounts.csv`
- `transactions.csv`
- `cash_tx.csv`
- `alert_accounts.csv`
- `alert_transactions.csv`
- `sar_accounts.csv`
- `individuals-bulkload.csv`
- `organizations-bulkload.csv`
- `accountMapping.csv`
- `resolvedentities.csv`

This richer surface is what the canonical schema should target even if the first implementation slice uses the smaller bundled sample outputs for immediate execution.

## Strengths

| Area | Assessment |
| --- | --- |
| Graph Readiness | Strong. Transactions are naturally directed edges between accounts. |
| Known Suspicious Patterns | Strong. Alert pattern inputs and alert outputs give controlled suspicious structures. |
| Account And Party Modelling | Strong in converted schema; moderate in bundled sample outputs. |
| Temporal Modelling | Moderate to strong. `total_steps`, `base_date`, and event-step fields support rolling windows. |
| Account-Level Risk Labels | Stronger than transaction-only simulators because alert membership is explicit. |
| Portfolio Suitability | Strong. Open, reproducible and aligned with graph-based AML work. |

## Limitations

| Area | Impact On Project |
| --- | --- |
| No source-native Device or IP data | Requires deterministic enrichment from the first implementation slice |
| No source-native Merchant dimension in observed sample outputs | Merchant must be derived or introduced as synthetic counterparty clusters |
| Sample outputs are small | Immediate implementation is easy, but evaluation is too small for serious benchmarking |
| Full-scale data generation requires Java and simulator setup | Canonical schema can be designed now, but larger data runs are a later execution step |
| Synthetic identity and ATO behavior are weakly represented | These use cases need cautious documentation and enrichment-based approximation |
| Sample transaction labels are indirect | Transaction-risk baseline must distinguish direct alert labels from alert-adjacent proxies |

## Suitability By Use Case

| Use Case | Suitability | Notes |
| --- | --- | --- |
| Fraud Rings | High | Best-supported area due to native ring-like typologies |
| Mule Accounts | Medium to High | Supported through flow behavior and alert membership |
| Synthetic Identity Fraud | Low to Medium | Needs lifecycle and infrastructure enrichment |
| Account Takeover | Low to Medium | Needs derived Device/IP behavior and temporal drift logic |
| Anti-Money Laundering | High | Core design intent of AMLSim |

## Data Quality And Engineering Notes

1. The sample outputs use older file names such as `tx.csv` and `alerts.csv`, while the richer converted schema uses `transactions.csv` and `alert_accounts.csv`.
2. The sample outputs include both account and cash-transaction surfaces, which is enough to start ingestion and canonical normalization.
3. The `1K` configuration sets `random_seed = 0`, `total_steps = 720`, and `base_date = 2017-01-01`, which should be preserved in project metadata for reproducibility.
4. The richer schema includes party and account mapping files, but those are not bundled in the simple sample output directory. The first implementation slice therefore needs a fallback rule for building parties from sample account ownership columns.

## Decision Rationale

IBM AMLSim remains the correct canonical dataset despite its limitations because the project is centered on graph-based financial risk intelligence, not only on point fraud classification. The dataset gives a transaction graph, alert typologies and a richer target schema for party and account mapping. Those benefits outweigh the need to synthesize Device, IP Address and Merchant entities for the MVP.

## Immediate Implementation Consequences

1. Loaders must support both the bundled sample layout and the richer converted output layout.
2. Canonical processed tables must distinguish source-native entities from derived entities.
3. The first baseline should use account/customer and transaction views at the same time, but label provenance must remain explicit.
4. Larger AMLSim runs should be treated as a controlled next step after the scaffold is stable.

## Acceptance Criteria

- the pinned AMLSim artifact is recorded by source and commit
- sample outputs and richer converted outputs are both documented
- the dataset decision is justified in terms of graph, temporal and alert coverage
- data gaps that require synthetic enrichment are explicit rather than hidden