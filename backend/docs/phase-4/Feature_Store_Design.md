# Feature Store Design

## Purpose

Describe the current feature storage and computation pattern for the Financial Risk Intelligence Platform.

## Current Design

The current feature layer is code-first and file-backed rather than an external feature store service.

### Source Layers

1. raw AMLSim layouts under `data/external/AMLSim`
2. canonical processed tables under `data/processed/amlsim`
3. derived feature frames built in memory from the canonical tables

### Current Feature Domains

| Domain | Granularity | Current Source |
| --- | --- | --- |
| Transaction features | one row per transaction | `src/fri/features/baseline.py` |
| Party features | one row per party | `src/fri/features/baseline.py` |
| Graph node features | one row per graph node | `src/fri/graph/analytics.py` |

## Offline Feature Contracts

### Transaction Contract

- entity key: `transaction_id`
- source join keys: `source_account_id`, `destination_account_id`
- label field: `label`
- label lineage: `label_source`

### Party Contract

- entity key: `party_id`
- label field: `label`
- aggregated from account ownership and transaction behavior

### Graph Node Contract

- entity key: `node_id`
- label field: `is_fraud` in graph archive mode
- derived from graph topology and node attributes

## Storage Approach

Current storage is artifact-backed:

- canonical processed tables are persisted to CSV
- graph node features are persisted to `artifacts/graph/archive_node_features.csv`
- model metrics are persisted as JSON artifacts

This is sufficient for MVP iteration because:

1. reproducibility matters more than serving latency right now
2. the project is still stabilizing the canonical schema
3. feature definitions are changing as phases progress

## Future Feature Store Evolution

1. move canonical processed tables to parquet
2. version feature sets by dataset layout and enrichment seed
3. separate offline training features from online scoring features
4. add a registry for feature names, formulas, windows and leakage notes